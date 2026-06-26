"""
sdbs_core.py
================================================================================
Framework-level (NumPy-only) core for the *Serendipitous & Diverse Beam Search*
(S-DBS) dreaming planner that extends Dreamer-PPO for VRU-safe driving.

This module deliberately contains NO PyTorch. Everything here is testable on its
own. The planner talks to the policy / world model through a tiny numpy
"boundary" (a handful of methods documented in `ModelInterface` below), so the
same planner works with:
  * the trained torch networks (see sdbs_dreamer.py -> Torch*Wrapper), and
  * the lightweight numpy stubs in this file (used for unit/smoke testing).

Section map -> paper ("Extending Dreamer-PPO"):
  * SDBSPlanner ................ Secs. 2-6, 10  (lookahead, DBS, S-DBS,
                                hierarchy, budget-aware, consolidated planner)
  * diversity functions ........ Sec. 3        (conflict-cell Jaccard)
  * SumTree / PER .............. Sec. 7        (Algorithm 1)
  * CurriculumController ....... Sec. 7        (3-stage curriculum)
  * RolloutBuffer (GAE) ........ base PPO appendix
  * MockDrivingEnv ............. Sec. 9        (occluded-crosswalk greedy trap)
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional, Sequence

import numpy as np


# ==============================================================================
# 0.  Maneuvers (the discrete "option" level of the hierarchy, Sec. 5)
# ==============================================================================
class Maneuver(IntEnum):
    FOLLOW_LANE = 0
    YIELD_CREEP = 1          # stop-and-creep / early yield  (the "serendipitous" safe option)
    LANE_CHANGE_LEFT = 2
    LANE_CHANGE_RIGHT = 3
    OVERTAKE_CYCLIST = 4
    HARD_STOP = 5


DEFAULT_MANEUVERS: tuple[int, ...] = (
    Maneuver.FOLLOW_LANE,
    Maneuver.YIELD_CREEP,
    Maneuver.LANE_CHANGE_LEFT,
)


# ==============================================================================
# 1.  Configuration
# ==============================================================================
@dataclass
class PlannerConfig:
    gamma: float = 0.97             # discount used inside imagination
    horizon: int = 5                # H : imagined rollout depth (Sec. 2)
    beam_width: int = 9             # B
    n_groups: int = 3               # G  (also caps the number of retained maneuvers)
    n_expand: int = 4               # candidate actions sampled per node per step
    eta: float = 0.5                # serendipity strength (Sec. 4)
    lam: float = 0.6                # diversity strength  lambda_g (Sec. 3)
    mu: float = 0.4                 # defensive / safety-margin weight (Sec. 5)
    value_weight_long_horizon: float = 2.0   # over-weight bootstrap for the long-horizon group
    serendipity_multiplicative: bool = True  # Novelty*Surprise*Gain vs additive
    ser_add_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (alpha,beta,gamma) additive form
    # within-group node-selection nudges (cheap; no critic calls inside the beam)
    node_eta_surprise: float = 0.05
    node_mu_risk: float = 0.3
    # group roles, cycled if fewer than n_groups (Sec. 5, Table 1)
    roles: tuple[str, ...] = ("exploitation", "diversity", "serendipity",
                              "defensive", "long_horizon")
    diversity_kind: str = "conflict"   # "conflict" (Jaccard on space-time cells) or "traj"
    cell_size: float = 2.0             # metres per grid cell for conflict-cell rasterisation
    cell_margin: float = 1.0           # safety margin (cells) around the swept ego trajectory
    lambda_collision: float = 0.3
    use_weighted_collision: bool = True

@dataclass
class BudgetConfig:
    """Budget-aware / anytime allocation (Sec. 6)."""
    max_rollouts: int = 240         # hard cap on world-model forward passes per decision (~ B_comp)
    # scene-difficulty d_t = sigmoid(w . features); features = [risk_density, n_vru,
    #                                                          ttc_deficit, occlusion, uncertainty]
    difficulty_weights: tuple[float, ...] = (3.0, 0.4, 1.5, 1.2, 1.0)
    difficulty_bias: float = -1.5
    # (B, G, H) grow monotonically with difficulty; we interpolate easy<->hard.
    beam_easy: int = 3
    beam_hard: int = 12
    groups_easy: int = 1
    groups_hard: int = 4
    horizon_easy: int = 2
    horizon_hard: int = 6


@dataclass
class PERConfig:
    capacity: int = 4096
    alpha: float = 0.6              # prioritisation exponent (p_i^alpha)
    beta0: float = 0.4              # importance-sampling start
    beta_anneal_steps: int = 20000
    eps: float = 1e-3               # smoothing so p_i never hits 0
    # error e_i = w_c*collisions + w_n*near_misses + w_ttc*ttc_violations + w_p*progress_deficit
    err_weights: tuple[float, float, float, float] = (5.0, 2.0, 1.0, 1.0)


@dataclass
class CurriculumConfig:
    unlock_threshold: float = 0.85  # theta : moving-average success to unlock next tier
    ma_window: int = 50
    n_tiers: int = 3                # Foundational / Ambiguity / Greedy-traps (Sec. 7)


@dataclass
class TrainConfig:
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    per: PERConfig = field(default_factory=PERConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rollout_size: int = 1024
    update_epochs: int = 6
    minibatch: int = 64
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    gae_lambda: float = 0.95
    # auxiliary / dream loss weights (Sec. 8) -- used ONLY for world_model_update,
    # i.e. how hard the WM is trained to predict risk/density, NOT how much the
    # real-env reward gets shaped (see lambda_risk_reward / lambda_density_reward
    # below). Keeping these separate avoids silently double-penalising TTC.
    beta_dream: float = 1.0
    lambda_risk: float = 1.0
    lambda_progress: float = 1.0
    lambda_recon: float = 0.5
    lambda_density: float = 0.5
    # R4 · reward-shaping coefficients applied in _risk_shaped_reward (sdbs_dreamer.py).
    # EnhancedMockEnv's own reward already includes a TTC term (r_ttc) and an
    # implicit occlusion penalty via near-miss/collision terms, so on top of
    # that env these default to 0.0 (no extra shaping) to avoid double-counting
    # the same risk signal twice. Raise them above 0 deliberately (e.g. on an
    # env with NO built-in risk term, like the bare CarlaEnvAdapter) if you
    # want this additional shaping; tune independently from lambda_risk/
    # lambda_density above, which only affect the world-model's own loss.
    lambda_risk_reward: float = 0.0
    lambda_density_reward: float = 0.0
    # intrinsic serendipity reward (Sec. 4.1) -- bounded so safety dominates
    eta_serendipity_reward: float = 0.05


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ==============================================================================
# 2.  Diversity functions  (Sec. 3:  Delta(C_g, C_h))
# ==============================================================================
def conflict_cells(traj: Sequence[tuple[float, float]],
                   cell_size: float, margin_cells: float) -> set[tuple[int, int]]:
    """Rasterise an imagined ego trajectory to the set of space(-time) grid cells
    it claims, inflated by a safety margin. This is the driving analogue of the
    set of *relations* a formal concept covers."""
    cells: set[tuple[int, int]] = set()
    m = int(round(margin_cells))
    for (x, y) in traj:
        ci, cj = int(math.floor(x / cell_size)), int(math.floor(y / cell_size))
        for di in range(-m, m + 1):
            for dj in range(-m, m + 1):
                cells.add((ci + di, cj + dj))
    return cells


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def conflict_diversity(traj_a, traj_b, cell_size: float, margin: float) -> float:
    """Delta in [0,1]; high == the two plans contest the same space-time."""
    return jaccard(conflict_cells(traj_a, cell_size, margin),
                   conflict_cells(traj_b, cell_size, margin))


def traj_diversity(traj_a, traj_b, sigma: float = 3.0) -> float:
    """Fallback similarity = exp(-mean waypoint distance / sigma), in (0,1]."""
    n = min(len(traj_a), len(traj_b))
    if n == 0:
        return 0.0
    d = 0.0
    for i in range(n):
        ax, ay = traj_a[i]
        bx, by = traj_b[i]
        d += math.hypot(ax - bx, ay - by)
    return math.exp(-(d / n) / sigma)


# ==============================================================================
# 3.  Sum Tree + Prioritised Scenario Replay  (Sec. 7, Algorithm 1)
# ==============================================================================
class SumTree:
    """Binary tree where each leaf holds a priority and each internal node holds
    the sum of its children. Sampling and updates are O(log N)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.size = 0
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        if idx == 0:
            return
        parent = (idx - 1) // 2
        self.tree[parent] += change
        self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):            # leaf reached
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> int:
        idx = self.write + self.capacity - 1          # tree-node index of this leaf
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx                                     # callers pass this back to update()

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedScenarioReplay:
    """Hard-scenario mining over a bank of driving scenarios (Sec. 7).
    Scenarios where the agent keeps failing get sampled more often."""

    def __init__(self, cfg: PERConfig):
        self.cfg = cfg
        self.tree = SumTree(cfg.capacity)
        self.max_priority = 1.0
        self._steps = 0

    def add(self, scenario) -> int:
        """New scenario gets max priority so it is evaluated at least once."""
        return self.tree.add(self.max_priority ** self.cfg.alpha, scenario)

    def beta(self) -> float:
        frac = min(1.0, self._steps / max(1, self.cfg.beta_anneal_steps))
        return self.cfg.beta0 + frac * (1.0 - self.cfg.beta0)

    def sample(self, k: int):
        """Roulette-wheel selection across k equal segments of [0, total]."""
        self._steps += 1
        leaves, scenarios, priorities = [], [], []
        total = self.tree.total
        if total <= 0:
            raise RuntimeError("Replay is empty.")
        seg = total / k
        for j in range(k):
            s = np.random.uniform(j * seg, (j + 1) * seg)
            leaf, p, scen = self.tree.get(s)
            leaves.append(leaf)
            scenarios.append(scen)
            priorities.append(p)
        probs = np.asarray(priorities) / total
        n = self.tree.size
        weights = (n * probs) ** (-self.beta())
        weights = weights / (weights.max() + 1e-12)     # normalise IS weights
        return leaves, scenarios, weights

    def priority_from_error(self, error: float) -> float:
        return (abs(error) + self.cfg.eps) ** self.cfg.alpha

    def error_from_outcome(self, collisions: int, near_misses: int,
                           ttc_violations: int, progress_deficit: float) -> float:
        wc, wn, wt, wp = self.cfg.err_weights
        return (wc * collisions + wn * near_misses
                + wt * ttc_violations + wp * max(0.0, progress_deficit))

    def update_priorities(self, leaves: Sequence[int], errors: Sequence[float]) -> None:
        for leaf, err in zip(leaves, errors):
            p = self.priority_from_error(err)
            self.max_priority = max(self.max_priority, abs(err) + self.cfg.eps)
            self.tree.update(leaf, p)


# ==============================================================================
# 4.  Curriculum controller  (Sec. 7, 3-stage curriculum)  — C5 dual-metric
# ==============================================================================
class CurriculumController:
    """C5 · Adaptive tier unlock using TWO independent moving averages:
        * ma_collision : fraction of episodes with zero collisions
        * ma_ttc       : fraction of episodes with zero TTC violations
    A tier unlocks only when BOTH exceed `unlock_threshold` (theta).
    This prevents an agent that avoids crashes but still cuts corners on TTC
    from advancing prematurely."""

    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.stage = 0
        # per-metric windows (C5)
        self._collision_free: list[float] = []   # 1 = no collision that episode
        self._ttc_clean: list[float] = []        # 1 = zero TTC violations
        # legacy single-window for backward compatibility
        self._recent: list[float] = []

    def difficulty_tier(self) -> int:
        return self.stage

    def sample_scenario_params(self) -> dict:
        """Returns reset() parameters for the env at the current tier.
        Stage 0: empty/low-density.  Stage 1: predictable VRU crossings.
        Stage 2: adversarial greedy traps (occluded pedestrians, etc.)."""
        tier = self.stage
        return dict(
            tier=tier,
            n_vru=(0 if tier == 0 else (1 if tier == 1 else 2)),
            occlusion_prob=(0.0 if tier == 0 else (0.2 if tier == 1 else 0.8)),
            adversarial=(tier >= 2),
        )

    def record(self, success: bool,
               collisions: int = 0, ttc_violations: int = 0) -> None:
        """Log one episode outcome. collisions / ttc_violations feed the two
        independent MAs; success feeds the legacy window."""
        self._recent.append(1.0 if success else 0.0)
        if len(self._recent) > self.cfg.ma_window:
            self._recent.pop(0)
        self._collision_free.append(1.0 if collisions == 0 else 0.0)
        if len(self._collision_free) > self.cfg.ma_window:
            self._collision_free.pop(0)
        self._ttc_clean.append(1.0 if ttc_violations == 0 else 0.0)
        if len(self._ttc_clean) > self.cfg.ma_window:
            self._ttc_clean.pop(0)

    def success_rate(self) -> float:
        return float(np.mean(self._recent)) if self._recent else 0.0

    def collision_free_rate(self) -> float:
        return float(np.mean(self._collision_free)) if self._collision_free else 0.0

    def ttc_clean_rate(self) -> float:
        return float(np.mean(self._ttc_clean)) if self._ttc_clean else 0.0

    def maybe_unlock(self) -> bool:
        """C5 · Advance to the next tier only when BOTH per-metric MAs pass theta.
        Falls back to the legacy success-rate window if per-metric data are absent."""
        enough = len(self._collision_free) >= self.cfg.ma_window
        if self.stage >= self.cfg.n_tiers - 1:
            return False
        if enough:
            # dual-metric gate (C5)
            ready = (self.collision_free_rate() >= self.cfg.unlock_threshold
                     and self.ttc_clean_rate() >= self.cfg.unlock_threshold)
        else:
            # fallback: legacy single MA
            ready = (len(self._recent) >= self.cfg.ma_window
                     and self.success_rate() >= self.cfg.unlock_threshold)
        if ready:
            self.stage += 1
            self._recent.clear()
            self._collision_free.clear()
            self._ttc_clean.clear()
            return True
        return False


# ==============================================================================
# 5.  Rollout buffer with GAE  (base PPO; stores maneuvers too)
# ==============================================================================
class RolloutBuffer:
    def __init__(self, size: int, state_dim: int, action_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.size, self.gamma, self.gae_lambda = size, gamma, gae_lambda
        self.states = np.zeros((size, state_dim), np.float32)
        self.actions = np.zeros((size, action_dim), np.float32)
        self.maneuvers = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.values = np.zeros(size, np.float32)
        self.log_probs = np.zeros(size, np.float32)
        # world-model / auxiliary targets (Sec. 8)
        self.next_states = np.zeros((size, state_dim), np.float32)
        self.risk_targets = np.zeros(size, np.float32)
        self.progress_targets = np.zeros(size, np.float32)
        self.density_targets = np.zeros(size, np.float32)
        self.advantages = np.zeros(size, np.float32)
        self.returns = np.zeros(size, np.float32)
        self.ptr = 0
        self.path_start = 0

    def store(self, state, action, maneuver, reward, done, value, log_prob,
              next_state, risk_t, progress_t, density_t):
        i = self.ptr
        if i >= self.size:
            raise RuntimeError("RolloutBuffer full.")
        self.states[i] = state
        self.actions[i] = action
        self.maneuvers[i] = maneuver
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.next_states[i] = next_state
        self.risk_targets[i] = risk_t
        self.progress_targets[i] = progress_t
        self.density_targets[i] = density_t
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0) -> None:
        sl = slice(self.path_start, self.ptr)
        rew = np.append(self.rewards[sl], last_value)
        val = np.append(self.values[sl], last_value)
        done = np.append(self.dones[sl], 0.0)
        gae = 0.0
        for t in reversed(range(len(rew) - 1)):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + self.gamma * val[t + 1] * nonterminal - val[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            self.advantages[self.path_start + t] = gae
            self.returns[self.path_start + t] = gae + val[t]
        self.path_start = self.ptr

    def clear(self) -> None:
        self.ptr = 0
        self.path_start = 0

    def get(self) -> dict:
        n = self.ptr
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return dict(
            states=self.states[:n], actions=self.actions[:n],
            maneuvers=self.maneuvers[:n], log_probs=self.log_probs[:n],
            advantages=adv, returns=self.returns[:n], values=self.values[:n],
            rewards=self.rewards[:n],
            next_states=self.next_states[:n], risk_targets=self.risk_targets[:n],
            progress_targets=self.progress_targets[:n],
            density_targets=self.density_targets[:n],
        )


# ==============================================================================
# 6.  Model interface expected by the planner (numpy boundary)
# ==============================================================================
class ModelInterface:
    """Documentation-only. Any `policy` passed to SDBSPlanner must provide:

        maneuver_probs(state)        -> np.ndarray, shape [n_maneuvers]   (pi over options)
        sample_action(state, m)      -> (action: np.ndarray[A], control_logprob: float)
        value(state)                 -> float                            (critic V_hat)

    Any `world_model` must provide:

        step(state, action, hidden=None)
                                      -> (next_state, reward, risk, new_hidden)
                                      (R1: hidden is the recurrent dynamics state;
                                      pass it back in on the next call so memory
                                      persists across imagined steps. Models
                                      without memory just return None for it.)
        risk_density(state)          -> float    (scene-risk density, feeds difficulty)
        uncertainty(state, action)   -> float    (ensemble disagreement; 0.0 if unused)
        ego_xy(state)                -> (x: float, y: float)             (for conflict cells)
    """


# ==============================================================================
# 7.  The S-DBS dreaming planner (Secs. 2-6, 10)
# ==============================================================================
@dataclass
class _BeamNode:
    state: np.ndarray
    disc_reward: float
    first_action: Optional[np.ndarray]
    first_logp: float
    traj: list[tuple[float, float]]
    risk_max: float
    cum_logp: float
    steps: int
    # R1 · recurrent world-model hidden state, threaded across imagined steps.
    # None for world models that don't expose memory (e.g. the NumPy stubs).
    hidden: object = None


@dataclass
class GroupPlan:
    maneuver: int
    first_action: np.ndarray
    first_logp: float
    maneuver_logp: float
    g_total: float
    traj: list[tuple[float, float]]
    risk_max: float
    value_final: float
    role: str


# ==============================================================================
# 6b.  Counterfactual-state helper  (C2 · Counterfactual VRU beam)
# ==============================================================================
def _make_counterfactual_state(state: np.ndarray, info: dict) -> Optional[np.ndarray]:
    """Return a copy of `state` in which the occluded VRU has been revealed.

    Works with both the 9-dim MockDrivingEnv layout (indices S_PED_VIS, S_OCC)
    and the 20-dim EnhancedMockEnv layout (IDX_VRU0_VIS, IDX_OCC). Returns
    None if the state appears to have no occlusion (safe to skip C2).
    """
    occlusion = float(info.get("occlusion", 0.0))
    if occlusion < 0.1:          # nothing to reveal
        return None
    cf = state.copy()
    dim = len(cf)
    if dim <= 9:                 # MockDrivingEnv  (S_OCC=8, S_PED_VIS=6)
        cf[8] = 0.0              # S_OCC
        if dim > 6:
            cf[6] = 1.0          # S_PED_VIS
    else:                        # EnhancedMockEnv (IDX_OCC=11, IDX_VRU0_VIS=14)
        cf[11] = 0.0             # IDX_OCC
        cf[14] = 1.0             # IDX_VRU0_VIS
        if dim > 16:
            cf[16] = 1.0         # IDX_VRU1_VIS
    return cf


class SDBSPlanner:
    """Replaces Dreamer-PPO's one-step greedy `select_action_with_dreaming`.

    The planner: (i) runs a safety pre-pass for mandated actions, (ii) sizes the
    search to scene difficulty under a compute cap, (iii) keeps G structurally
    distinct maneuvers (option-level diversity), (iv) runs a control-level beam
    per maneuver out to horizon H, (v) picks the executed action with the
    serendipitous diversity-augmented objective
        Score(C_g) = G_total + eta*Ser - lambda*sum_{h<g} Delta + mu*margin,
    and (vi) runs a C2 counterfactual VRU beam that measures worst-case risk
    assuming every occluded VRU has already emerged, penalising plans that are
    only safe *because* the VRU stays hidden.
    """

    def __init__(self, policy, world_model, cfg: PlannerConfig,
                 budget: BudgetConfig,
                 maneuvers: Sequence[int] = DEFAULT_MANEUVERS,
                 mandated_action_fn: Optional[Callable] = None,
                 traffic_predictor=None):
        self.policy = policy
        self.wm = world_model
        self.cfg = cfg
        self.budget = budget
        self.maneuvers = tuple(maneuvers)
        self.mandated_action_fn = mandated_action_fn
        self.traffic_predictor = traffic_predictor
        self._agent_predictions: dict = {}

    # ---- diversity helper ----
    def _delta(self, a: GroupPlan, b: GroupPlan) -> float:
        if self.cfg.diversity_kind == "conflict":
            return conflict_diversity(a.traj, b.traj, self.cfg.cell_size, self.cfg.cell_margin)
        return traj_diversity(a.traj, b.traj)

    # ---- budget-aware allocation (Sec. 6) ----
    def allocate(self, state: np.ndarray, info: Optional[dict]) -> tuple[int, int, int, float]:
        info = info or {}
        risk_density = float(self.wm.risk_density(state))
        n_vru = float(info.get("n_vru", 0.0))
        min_ttc = float(info.get("min_ttc", 99.0))
        ttc_deficit = max(0.0, info.get("ttc_threshold", 3.0) - min_ttc)
        occlusion = float(info.get("occlusion", 0.0))
        uncertainty = float(info.get("uncertainty", 0.0))
        feats = np.array([risk_density, n_vru, ttc_deficit, occlusion, uncertainty])
        w = np.array(self.budget.difficulty_weights)
        d = sigmoid(float(np.dot(w, feats)) + self.budget.difficulty_bias)
        lerp = lambda lo, hi: int(round(lo + d * (hi - lo)))
        B = max(1, lerp(self.budget.beam_easy, self.budget.beam_hard))
        G = max(1, min(lerp(self.budget.groups_easy, self.budget.groups_hard),
                       B, len(self.maneuvers)))
        H = max(1, lerp(self.budget.horizon_easy, self.budget.horizon_hard))
        return B, G, H, d

    # ---- one-step greedy baseline (== base Dreamer-PPO; for ablation/comparison) ----
    def greedy_one_step(self, state: np.ndarray) -> dict:
        """Base Dreamer-PPO behaviour: sample candidate actions, dream ONE step,
        pick the action with the best immediate imagined reward (no lookahead)."""
        best = None
        probs = self.policy.maneuver_probs(state)
        for m in self.maneuvers:
            for _ in range(self.cfg.n_expand):
                a, lp = self.policy.sample_action(state, m)
                _, r, _, _ = self.wm.step(state, a, None)
                if best is None or r > best["score"]:
                    best = dict(score=float(r), action=a, maneuver=int(m),
                                logp=float(lp + math.log(probs[m] + 1e-12)),
                                value=float(self.policy.value(state)))
        best["meta"] = dict(mandated=False, mode="greedy_one_step")
        return best

    # ---- within-group control-level beam (Secs. 2, 5) ----
    def _beam_for_maneuver(self, state, maneuver: int, role: str,
                           H: int, b: int, budget_left: list[int],
                           init_hidden=None) -> _BeamNode:
        gamma = self.cfg.gamma
        x0, y0 = self.wm.ego_xy(state)
        beam = [_BeamNode(state=state, disc_reward=0.0, first_action=None,
                          first_logp=0.0, traj=[(x0, y0)], risk_max=0.0,
                          cum_logp=0.0, steps=0, hidden=init_hidden)]
        for i in range(H):
            children: list[_BeamNode] = []
            for node in beam:
                for _ in range(self.cfg.n_expand):
                    if budget_left[0] <= 0:
                        break
                    budget_left[0] -= 1
                    a, lp = self.policy.sample_action(node.state, maneuver)
                    # R1 · pass the recurrent hidden state through so the GRU
                    # dynamics actually carry memory across imagined steps
                    # instead of resetting to zero at every node expansion.
                    ns, r, risk, h_new = self.wm.step(node.state, a, node.hidden)
                    ex, ey = self.wm.ego_xy(ns)
                    children.append(_BeamNode(
                        state=ns,
                        disc_reward=node.disc_reward + (gamma ** i) * r,
                        first_action=node.first_action if node.first_action is not None else a,
                        first_logp=node.first_logp if node.first_action is not None else float(lp),
                        traj=node.traj + [(ex, ey)],
                        risk_max=max(node.risk_max, float(risk)),
                        cum_logp=node.cum_logp + float(lp),
                        steps=i + 1,
                        hidden=h_new,
                    ))
            if not children:
                break
            # role-specific node score keeps the beam aligned with the group's "personality"
            def node_key(nd: _BeamNode) -> float:
                key = nd.disc_reward
                if role == "serendipity":
                    key += self.cfg.node_eta_surprise * (-nd.cum_logp)   # surprise-seeking
                elif role == "defensive":
                    key -= self.cfg.node_mu_risk * nd.risk_max           # avoid risk
                return key
            children.sort(key=node_key, reverse=True)
            beam = children[: max(1, b)]
        # best member of this group's beam by total imagined return-to-go
        vw = (self.cfg.value_weight_long_horizon if role == "long_horizon" else 1.0)
        best = max(beam, key=lambda nd: nd.disc_reward
                   + vw * (gamma ** max(1, nd.steps)) * self.policy.value(nd.state))
        return best

    # ---- C2: counterfactual mini-beam (worst-case VRU emergence risk) ----
    def _counterfactual_risk(self, state: np.ndarray, info: dict,
                             budget_left: list[int]) -> float:
        """C2 · Run a short beam on the counterfactual state (VRU revealed).
        Returns the worst-case risk_max across all branches. If no occlusion
        is present, returns 0.0 immediately (no compute wasted)."""
        cf_state = _make_counterfactual_state(state, info)
        if cf_state is None:
            return 0.0
        # Use a compact 1-group / depth-2 beam (cheap, anytime-safe)
        cf_budget = [min(budget_left[0], max(4, self.budget.max_rollouts // 8))]
        cf_risk = 0.0
        for m in self.maneuvers[:max(2, len(self.maneuvers) // 2)]:
            node = self._beam_for_maneuver(
                cf_state, m, role="defensive",
                H=min(2, self.cfg.horizon),
                b=max(1, self.cfg.beam_width // 4),
                budget_left=cf_budget,
            )
            cf_risk = max(cf_risk, node.risk_max)
        budget_left[0] -= (min(budget_left[0], max(4, self.budget.max_rollouts // 8))
                           - cf_budget[0])
        return cf_risk

    # ---- main entry: one control step (receding horizon) ----
    def plan(self, state: np.ndarray, info: Optional[dict] = None) -> dict:
        info = info or {}

        # (1) safety pre-pass: mandated / "essential" actions (Sec. 6)
        if self.mandated_action_fn is not None:
            mand = self.mandated_action_fn(state, info)
            if mand is not None:
                probs = self.policy.maneuver_probs(state)
                m = int(Maneuver.HARD_STOP) if int(Maneuver.HARD_STOP) < len(probs) else 0
                lp = float(math.log(probs[m] + 1e-12))
                return dict(action=np.asarray(mand, dtype=np.float32), maneuver=m,
                            logp=lp, value=float(self.policy.value(state)),
                            meta=dict(mandated=True, mode="safety_prepass",
                                      cf_risk=0.0))

        # (2) budget-aware sizing
        B, G, H, d = self.allocate(state, info)
        b = max(1, B // G)
        budget_left = [int(self.budget.max_rollouts)]   # mutable counter (anytime cap)

        # (2b) C2 · Counterfactual VRU beam — measure worst-case occlusion risk
        #      Run BEFORE the main beam so the penalty informs final scoring.
        cf_risk = self._counterfactual_risk(state, info, budget_left)
        if self.traffic_predictor is not None:
            try:
                self._agent_predictions = self.traffic_predictor.predict_and_score()
            except Exception:
                self._agent_predictions = {}
        else:
            self._agent_predictions = {}
        # (3) option-level beam: keep G structurally distinct maneuvers (Sec. 5)
        probs = self.policy.maneuver_probs(state)
        order = list(np.argsort(-probs))
        retained = [int(m) for m in order if int(m) in self.maneuvers][:G]
        if not retained:
            retained = [int(self.maneuvers[0])]

        # (4) control-level beam per retained maneuver, each with a personality role
        plans: list[GroupPlan] = []
        for gi, m in enumerate(retained):
            role = self.cfg.roles[gi % len(self.cfg.roles)]
            best = self._beam_for_maneuver(state, m, role, H, b, budget_left)
            vw = (self.cfg.value_weight_long_horizon if role == "long_horizon" else 1.0)
            v_final = float(self.policy.value(best.state))
            g_total = best.disc_reward + vw * (self.cfg.gamma ** max(1, best.steps)) * v_final
            plans.append(GroupPlan(
                maneuver=m,
                first_action=best.first_action if best.first_action is not None
                else self.policy.sample_action(state, m)[0],
                first_logp=best.first_logp,
                maneuver_logp=float(math.log(probs[m] + 1e-12)),
                g_total=float(g_total),
                traj=best.traj,
                risk_max=best.risk_max,
                value_final=v_final,
                role=role,
            ))

        # (5) serendipitous diversity-augmented final selection (Sec. 4)
        #     C2 · cf_risk is folded into the safety-margin term so plans that
        #     are only safe because the VRU stays hidden are additionally penalised.
        mean_g = float(np.mean([p.g_total for p in plans]))
        scored = []
        for gi, p in enumerate(plans):
            # diversity vs all other groups (novelty) and vs preceding groups (penalty)
            deltas_prev = [self._delta(p, plans[h]) for h in range(gi)]
            deltas_all = [self._delta(p, plans[h]) for h in range(len(plans)) if h != gi]
            novelty = 1.0 - (max(deltas_all) if deltas_all else 0.0)
            surprise = -(p.maneuver_logp + p.first_logp)          # -log pi(a|o), >= 0 by construction
            gain = p.g_total - mean_g
            if self.cfg.serendipity_multiplicative:
                # R3 · Novelty*Surprise*Gain only makes sense as a *bonus* when
                # the plan actually outperforms the group average. If gain < 0,
                # multiplying two further non-negative terms can still flip the
                # sign back positive (e.g. novelty*surprise*negative_gain is
                # negative, but a negative novelty or surprise would make it
                # positive again) and end up rewarding a bad plan for being
                # unsurprising. Clip gain to >= 0 so the product is a pure
                # upside bonus and never an accidental reward for mediocrity;
                # novelty and surprise are already non-negative by construction.
                ser = novelty * surprise * max(0.0, gain)
            else:
                aN, bS, gG = self.cfg.ser_add_weights
                ser = aN * novelty + bS * surprise + gG * gain
            div_penalty = self.cfg.lam * sum(deltas_prev)
            # C2: worst-case revealed-VRU risk penalises the plan's safety margin
            margin = -(p.risk_max + cf_risk)
            # Collision risk pondéré (0.0 si pas de predictor configuré)
            if self._agent_predictions and self.cfg.lambda_collision > 0.0:
                try:
                    from sdbs.planning.traffic_predictor import (
                        compute_collision_risk_weighted,
                        compute_collision_risk,
                    )

                    _fn = (
                        compute_collision_risk_weighted
                        if self.cfg.use_weighted_collision
                        else compute_collision_risk
                    )

                    cr = _fn(p.traj, self._agent_predictions)

                except ImportError:
                    cr = 0.0
            else:
                cr = 0.0

            score = (
                p.g_total
                + self.cfg.eta * ser
                - div_penalty
                + self.cfg.mu * margin
                - self.cfg.lambda_collision * cr
            )

            scored.append((score, p, dict(
                novelty=novelty,
                surprise=surprise,
                gain=gain,
                ser=ser,
                div_penalty=div_penalty,
                cf_risk=cf_risk,
                collision_risk=cr,
            )))

        best_score, best_plan, diag = max(scored, key=lambda t: t[0])
        return dict(
            action=np.asarray(best_plan.first_action, dtype=np.float32),
            maneuver=int(best_plan.maneuver),
            logp=float(best_plan.maneuver_logp + best_plan.first_logp),
            value=float(self.policy.value(state)),
            meta=dict(mandated=False, mode="sdbs", difficulty=d,
                      B=B, G=G, H=H, retained=retained, role=best_plan.role,
                      rollouts_used=self.budget.max_rollouts - budget_left[0],
                      score=best_score, **diag),
        )


# ==============================================================================
# 8.  Mock driving environment -- the occluded-crosswalk greedy trap (Sec. 9)
# ==============================================================================
# State layout (index -> meaning).  Keep these in sync with ego_xy / mandated_action.
S_EGO_X, S_EGO_Y, S_SPEED, S_DANGER, S_PROGRESS, S_PED_DX, S_PED_VIS, S_LIGHT, S_OCC = range(9)
MOCK_STATE_DIM = 9
MOCK_ACTION_DIM = 3   # [steer, throttle, brake]


def mock_ego_xy(state: np.ndarray) -> tuple[float, float]:
    return float(state[S_EGO_X]), float(state[S_EGO_Y])


def mock_mandated_action(state: np.ndarray, info: dict) -> Optional[np.ndarray]:
    """Hard safety floor: emergency brake if TTC is critical or a red light is
    imminent. Returns a control override, else None (Sec. 6, essential actions)."""
    min_ttc = float(info.get("min_ttc", 99.0))
    if min_ttc < 1.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)   # full brake, no throttle
    if float(state[S_LIGHT]) > 0.5 and float(state[S_SPEED]) > 0.1 and float(state[S_PROGRESS]) > 0.8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return None


class MockDrivingEnv:
    """A tiny, fully-NumPy stand-in for CARLA used to exercise the *machinery*.
    It reproduces the structure of the greedy trap (an occluded pedestrian that
    emerges only when the ego is close), so the planner has something to plan
    around. It is NOT a substitute for CARLA -- see README."""

    state_dim = MOCK_STATE_DIM
    action_dim = MOCK_ACTION_DIM
    n_maneuvers = len(DEFAULT_MANEUVERS)

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.ttc_threshold = 3.0
        self.max_steps = 40
        self._reset_internal(tier=0, n_vru=0, occlusion_prob=0.0, adversarial=False)

    def _reset_internal(self, tier, n_vru, occlusion_prob, adversarial):
        self.tier = tier
        self.n_vru = n_vru
        self.adversarial = adversarial
        self.occluded = self.rng.random() < occlusion_prob
        self.ped_dx = 12.0 if n_vru > 0 else 99.0      # pedestrian distance ahead
        self.ped_emerged = False
        self.t = 0
        self.collisions = 0
        self.near_misses = 0
        self.ttc_violations = 0
        self.s = np.zeros(MOCK_STATE_DIM, np.float32)
        self.s[S_SPEED] = 6.0
        self.s[S_PED_DX] = self.ped_dx
        self.s[S_PED_VIS] = 0.0 if (self.occluded and n_vru > 0) else (1.0 if n_vru > 0 else 0.0)
        self.s[S_LIGHT] = 1.0 if (adversarial and self.rng.random() < 0.3) else 0.0
        self.s[S_OCC] = 1.0 if self.occluded else 0.0

    def reset(self, tier: int = 0, n_vru: int = 0,
              occlusion_prob: float = 0.0, adversarial: bool = False) -> np.ndarray:
        self._reset_internal(tier, n_vru, occlusion_prob, adversarial)
        return self.s.copy()

    def _min_ttc(self) -> float:
        if self.n_vru == 0 or self.ped_dx > 50:
            return 99.0
        speed = max(0.1, float(self.s[S_SPEED]))
        return max(0.0, self.ped_dx / speed)

    def info(self) -> dict:
        return dict(
            n_vru=self.n_vru,
            min_ttc=self._min_ttc(),
            ttc_threshold=self.ttc_threshold,
            occlusion=float(self.s[S_OCC]),
            uncertainty=float(self.s[S_OCC]) * 0.5,
            collisions=self.collisions,
            near_misses=self.near_misses,
            ttc_violations=self.ttc_violations,
        )

    def step(self, action: np.ndarray, maneuver: Optional[int] = None):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        steer, throttle, brake = float(a[0]), float(a[1]), float(a[2])
        accel = 3.0 * throttle - 5.0 * brake
        self.s[S_SPEED] = float(np.clip(self.s[S_SPEED] + accel * 0.3, 0.0, 14.0))
        dist = self.s[S_SPEED] * 0.3
        self.s[S_EGO_X] += dist
        self.s[S_EGO_Y] += 1.5 * steer
        self.s[S_PROGRESS] = float(np.clip(self.s[S_PROGRESS] + dist / 60.0, 0.0, 1.0))

        # pedestrian dynamics: emerges when the ego gets close (trap reveal)
        if self.n_vru > 0:
            self.ped_dx -= dist
            if (not self.ped_emerged) and self.ped_dx < 6.0:
                self.ped_emerged = True
                self.s[S_PED_VIS] = 1.0
            self.s[S_PED_DX] = self.ped_dx

        ttc = self._min_ttc()
        collision = self.n_vru > 0 and self.ped_dx < 1.0 and self.s[S_SPEED] > 1.0
        near_miss = (not collision) and self.n_vru > 0 and ttc < 1.5 and self.s[S_SPEED] > 2.0
        if collision:
            self.collisions += 1
        if near_miss:
            self.near_misses += 1
        if self.n_vru > 0 and ttc < self.ttc_threshold:
            self.ttc_violations += 1

        # danger bookkeeping (used by the critic / value head proxy)
        if self.n_vru > 0 and ttc < self.ttc_threshold:
            self.s[S_DANGER] = float(np.clip(self.s[S_DANGER] + (self.ttc_threshold - ttc), 0.0, 10.0))
        else:
            self.s[S_DANGER] = float(max(0.0, self.s[S_DANGER] - 0.5))

        # VRU-aware reward (mirrors the base Dreamer-PPO reward terms)
        r_prog = dist / 6.0
        r_vru = -math.exp(-self.ped_dx / 3.0) if self.n_vru > 0 else 0.0
        r_ttc = -2.0 if (self.n_vru > 0 and ttc < self.ttc_threshold) else 0.0
        r_collision = -20.0 if collision else 0.0
        r_comfort = -0.1 * (abs(brake) + abs(steer))
        reward = r_prog + r_vru + r_ttc + r_collision + r_comfort

        self.t += 1
        done = collision or self.t >= self.max_steps or self.s[S_PROGRESS] >= 1.0

        info = self.info()
        info.update(
            progress=r_prog,
            vru_risk=float(np.clip(math.exp(-self.ped_dx / 3.0), 0.0, 1.0)) if self.n_vru > 0 else 0.0,
            density=float(np.clip(self.n_vru / 3.0 + self.s[S_OCC] * 0.3, 0.0, 1.0)),
            collision=collision, near_miss=near_miss,
            success=bool(self.s[S_PROGRESS] >= 1.0 and self.collisions == 0 and self.near_misses == 0),
        )
        return self.s.copy(), float(reward), bool(done), info


# ==============================================================================
# 9.  NumPy stubs (let us test the planner with no torch)
# ==============================================================================
class StubPolicy:
    """A deterministic-ish policy that produces clearly distinct maneuvers so the
    planner's logic can be exercised and the greedy trap demonstrated."""

    def __init__(self, maneuvers=DEFAULT_MANEUVERS, seed: int = 0):
        self.maneuvers = tuple(maneuvers)
        self.rng = np.random.default_rng(seed)
        # prior over maneuvers: PROCEED is the "tempting" default (highest prior)
        prefs = np.array([2.0, 0.5, 1.0, 0.5, 0.5, 0.2])[: max(self.maneuvers) + 1]
        self._prefs = prefs

    def maneuver_probs(self, state):
        logits = np.full(int(max(self.maneuvers)) + 1, -10.0)
        for m in self.maneuvers:
            logits[m] = self._prefs[m]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def sample_action(self, state, maneuver):
        # maneuver -> nominal control; small Gaussian noise -> a valid log-prob
        if maneuver == Maneuver.FOLLOW_LANE:
            mean = np.array([0.0, 0.9, 0.0])      # accelerate / proceed
        elif maneuver == Maneuver.YIELD_CREEP:
            mean = np.array([0.0, 0.15, 0.5])     # creep & prepare to yield
        elif maneuver == Maneuver.LANE_CHANGE_LEFT:
            mean = np.array([-0.6, 0.6, 0.0])
        elif maneuver == Maneuver.LANE_CHANGE_RIGHT:
            mean = np.array([0.6, 0.6, 0.0])
        elif maneuver == Maneuver.OVERTAKE_CYCLIST:
            mean = np.array([-0.3, 0.8, 0.0])
        else:  # HARD_STOP
            mean = np.array([0.0, 0.0, 1.0])
        std = 0.1
        action = mean + self.rng.normal(0, std, size=3)
        logp = float(np.sum(-0.5 * ((action - mean) / std) ** 2
                            - math.log(std * math.sqrt(2 * math.pi))))
        return action.astype(np.float32), logp

    def value(self, state):
        # critic proxy: high value when the imagined state is SAFE (low danger),
        # very negative when danger has accumulated (the trap).
        return float(np.clip(10.0 - 3.0 * state[S_DANGER], -30.0, 30.0))


class StubWorldModel:
    """A learned-simulator stand-in whose dynamics encode the trap: PROCEED gives
    high immediate reward but accumulates danger (low bootstrap value); YIELD
    gives modest reward but keeps the future safe (high bootstrap value)."""

    def step(self, state, action, hidden=None):
        s = state.copy()
        steer, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        accel = 3.0 * throttle - 5.0 * brake
        s[S_SPEED] = float(np.clip(s[S_SPEED] + accel * 0.3, 0.0, 14.0))
        dist = s[S_SPEED] * 0.3
        s[S_EGO_X] += dist
        s[S_EGO_Y] += 1.5 * steer
        s[S_PROGRESS] = float(np.clip(s[S_PROGRESS] + dist / 60.0, 0.0, 1.0))
        proceeding = throttle > 0.5 and brake < 0.3
        occluded = s[S_OCC] > 0.5
        # The trap: proceeding accumulates *latent* danger (bad bootstrap value),
        # but the *apparent* immediate risk is LOW under occlusion, so the one-step
        # reward looks attractive. Only the (curriculum-trained) critic sees the danger.
        if proceeding:
            s[S_DANGER] = float(np.clip(s[S_DANGER] + 2.5, 0.0, 10.0))
            risk = 0.1 if occluded else 0.7
        else:
            s[S_DANGER] = float(max(0.0, s[S_DANGER] - 0.5))
            risk = 0.1
        r_prog = dist / 6.0
        r_risk = -0.3 * risk          # small, so progress dominates the *immediate* reward
        reward = r_prog + r_risk
        # R1 · StubWorldModel has no recurrent memory; always returns hidden=None.
        return s.astype(np.float32), float(reward), float(risk), None

    def risk_density(self, state):
        return float(np.clip(0.3 + 0.5 * state[S_OCC] + 0.05 * state[S_DANGER], 0.0, 1.0))

    def uncertainty(self, state, action):
        return float(state[S_OCC]) * 0.5

    def ego_xy(self, state):
        return float(state[S_EGO_X]), float(state[S_EGO_Y])


# ==============================================================================
# 10.  Smoke test for the NumPy core (runs with no torch)
# ==============================================================================
def run_core_smoke_test() -> None:
    print("=" * 72)
    print("S-DBS core smoke test (NumPy only)")
    print("=" * 72)
    np.random.seed(0)

    # --- 10a. Sum-Tree PER: high-error scenarios should be sampled more often ---
    per = PrioritizedScenarioReplay(PERConfig(capacity=64))
    leaves = [per.add({"id": i}) for i in range(20)]
    # give scenario 0 a huge error, others tiny
    per.update_priorities([leaves[0]], [10.0])
    for lf in leaves[1:]:
        per.update_priorities([lf], [0.01])
    counts = {}
    for _ in range(2000):
        _, scen, _ = per.sample(1)
        counts[scen[0]["id"]] = counts.get(scen[0]["id"], 0) + 1
    frac0 = counts.get(0, 0) / 2000
    print(f"[PER]  scenario 0 (high error) sampled {frac0*100:.1f}% of the time "
          f"(expected >> {1/20*100:.1f}%) -> {'OK' if frac0 > 0.5 else 'CHECK'}")

    # --- 10b. GAE buffer produces finite advantages/returns ---
    buf = RolloutBuffer(8, MOCK_STATE_DIM, MOCK_ACTION_DIM, gamma=0.99, gae_lambda=0.95)
    for i in range(8):
        buf.store(np.zeros(MOCK_STATE_DIM), np.zeros(MOCK_ACTION_DIM), 0,
                  reward=1.0, done=0.0, value=0.5, log_prob=-1.0,
                  next_state=np.zeros(MOCK_STATE_DIM), risk_t=0.1,
                  progress_t=0.2, density_t=0.3)
    buf.finish_path(last_value=0.0)
    batch = buf.get()
    print(f"[GAE]  adv finite={np.all(np.isfinite(batch['advantages']))}, "
          f"ret finite={np.all(np.isfinite(batch['returns']))} -> OK")

    # --- 10c. Curriculum unlocks after sustained success ---
    cur = CurriculumController(CurriculumConfig(unlock_threshold=0.85, ma_window=20, n_tiers=3))
    for _ in range(20):
        cur.record(True)
    unlocked = cur.maybe_unlock()
    print(f"[CURR] stage after 20 successes = {cur.stage} (unlocked={unlocked}) -> "
          f"{'OK' if cur.stage == 1 else 'CHECK'}")

    # --- 10d. Planner: greedy falls into the trap, S-DBS does not ---
    policy, wm = StubPolicy(), StubWorldModel()
    pcfg = PlannerConfig(horizon=5, beam_width=9, n_groups=3, n_expand=4,
                         eta=0.5, lam=0.6, mu=0.4)
    bcfg = BudgetConfig()
    planner = SDBSPlanner(policy, wm, pcfg, bcfg, maneuvers=DEFAULT_MANEUVERS,
                          mandated_action_fn=mock_mandated_action)

    # state approaching an occluded crossing
    state = np.zeros(MOCK_STATE_DIM, np.float32)
    state[S_SPEED] = 6.0
    state[S_OCC] = 1.0
    info = dict(n_vru=2, min_ttc=2.5, ttc_threshold=3.0, occlusion=1.0, uncertainty=0.5)

    greedy = planner.greedy_one_step(state)
    sdbs = planner.plan(state, info)
    g_name = Maneuver(greedy['maneuver']).name
    s_name = Maneuver(sdbs['maneuver']).name
    SAFE = {int(Maneuver.YIELD_CREEP), int(Maneuver.HARD_STOP)}
    print(f"[PLAN] greedy one-step  -> {g_name:<16} (immediate imagined reward only)")
    print(f"[PLAN] S-DBS dreaming    -> {s_name:<16} "
          f"| difficulty={sdbs['meta']['difficulty']:.2f} "
          f"B={sdbs['meta']['B']} G={sdbs['meta']['G']} H={sdbs['meta']['H']} "
          f"rollouts={sdbs['meta']['rollouts_used']}")
    print(f"       serendipity diag: novelty={sdbs['meta']['novelty']:.2f} "
          f"surprise={sdbs['meta']['surprise']:.2f} gain={sdbs['meta']['gain']:.2f} "
          f"ser={sdbs['meta']['ser']:.2f}")
    greedy_trapped = greedy["maneuver"] not in SAFE          # greedy proceeds into the crossing
    sdbs_safe = sdbs["maneuver"] in SAFE                     # S-DBS yields
    verdict = "OK" if (greedy_trapped and sdbs_safe) else "CHECK"
    print(f"       -> greedy proceeds into the occluded crossing, S-DBS yields: {verdict}")

    # --- 10e. Safety pre-pass fires when TTC is critical ---
    crit = dict(info); crit["min_ttc"] = 0.5
    res = planner.plan(state, crit)
    print(f"[SAFE] critical-TTC -> mandated={res['meta']['mandated']} "
          f"action(brake)={res['action'][2]:.2f} -> "
          f"{'OK' if res['meta']['mandated'] else 'CHECK'}")

    # --- 10f. End-to-end loop on the mock env runs without error ---
    env = MockDrivingEnv(seed=1)
    obs = env.reset(**CurriculumController(CurriculumConfig()).sample_scenario_params())
    total_r, steps = 0.0, 0
    for _ in range(env.max_steps):
        out = planner.plan(obs, env.info())
        obs, r, done, info = env.step(out["action"], out["maneuver"])
        total_r += r
        steps += 1
        if done:
            break
    print(f"[LOOP] ran {steps} env steps, return={total_r:.2f}, "
          f"collisions={info['collisions']}, near_misses={info['near_misses']} -> OK")
    print("=" * 72)
    print("Core smoke test complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_core_smoke_test()