"""
sdbs_dreamer.py
================================================================================
PyTorch networks + training loop for the S-DBS dreaming planner, built on the
NumPy core in `sdbs_core.py`.

This file is import-safe WITHOUT PyTorch: if torch is missing, importing the
module still works and `main()` falls back to the NumPy core smoke test. The
torch-dependent classes are only defined when torch is present.

Section map -> paper ("Extending Dreamer-PPO"):
  * ActorCritic ............... Sec. 5      (option/maneuver head + control head)
  * WorldModel ................ Secs. 2, 8  (latent dynamics + reward/risk/progress
                                            + reconstruction + risk-density heads)
  * WorldModelEnsemble ........ Sec. 8      (epistemic uncertainty / disagreement)
  * Torch*Wrapper ............. numpy boundary the planner consumes
  * ppo_update ................ base PPO + aux heads
  * world_model_update ........ Sec. 8      (dream + grounding losses)
  * intrinsic serendipity ..... Sec. 4.1    (bounded reward shaping)
  * CarlaEnvAdapter ........... Sec. 9 / eval protocol (stub to implement)
  * train() ................... Sec. 10     (curriculum + PER + dreaming planner)

Contributions supplémentaires (au-delà du papier de référence) :
  * MultiAgentPredictor ........ prédiction multi-VRU avec intention sémantique
                                 (traverse / longe / s'arrête) + incertitude
                                 MC-dropout (use_traffic_predictor=True)
  * collect_rollout ............ mise à jour du tracker multi-agent à chaque step
                                 via step_info["agent_observations"]
  * build_agent ................ instancie et câble le MultiAgentPredictor au
                                 SDBSPlanner (lambda_collision dans PlannerConfig)

R8 · Squashed Gaussian action bound fix
  * ActorCritic.control_mean previously fed straight into Normal(mean, std)
    with NO final activation, so sampled actions were unbounded on R (e.g.
    [3.7, 0.6], [-1.7, -4.7] observed in logs) even though every consumer
    (EnhancedMockEnv, highway-env) expects actions in [-1, 1]. EnhancedMockEnv
    silently clips on receipt, which masked the bug but desynchronised the
    PPO importance ratio: log_prob was computed under the *unbounded* Normal
    for an action that was NOT the one actually executed after clipping.
    Fixed here with a squashed Gaussian: sample u ~ Normal(mean, std) on R,
    execute a = tanh(u) in (-1,1), and correct the log-prob by the tanh
    Jacobian: log pi(a) = log N(u; mean, std) - sum log(1 - tanh(u)^2).
    This makes log_prob consistent with the action actually sent to the env,
    which PPO's ratio = exp(logp_new - logp_old) requires to be unbiased.
================================================================================
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from sdbs.core.sdbs_core import (
    Maneuver, DEFAULT_MANEUVERS,
    PlannerConfig, BudgetConfig, PERConfig, CurriculumConfig, TrainConfig,
    SDBSPlanner, PrioritizedScenarioReplay, CurriculumController, RolloutBuffer,
    MockDrivingEnv, mock_ego_xy, mock_mandated_action, run_core_smoke_test,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    HAS_TORCH = True
except Exception:                      # torch not installed -> core-only mode
    HAS_TORCH = False


if HAS_TORCH:
    # ==========================================================================
    # 0.  Running observation normaliser  (R5 fix)
    # ==========================================================================
    class RunningNorm(nn.Module):
        """Welford running mean/std for observation normalisation.

        EnhancedMockEnv's 20-dim state mixes wildly different scales (speed in
        [0,14], dx in [-5,99], visibility/occlusion flags in [0,1], ...).
        Feeding that straight into a Tanh-activated MLP/GRU makes the
        large-scale dims dominate gradients and saturate the small-scale ones.
        This module tracks per-dimension running statistics (updated only
        during rollout collection, via `update()`) and normalises every input
        to the network with them. Registered as buffers so they're saved /
        loaded with the rest of the model's state_dict (and therefore survive
        checkpointing and the CARLA hand-off).
        """

        def __init__(self, dim: int, eps: float = 1e-4, clip: float = 10.0):
            super().__init__()
            self.eps = eps
            self.clip = clip
            self.register_buffer("count", torch.tensor(eps))
            self.register_buffer("mean", torch.zeros(dim))
            self.register_buffer("var", torch.ones(dim))

        @torch.no_grad()
        def update(self, x: torch.Tensor) -> None:
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch_count = x.shape[0]
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            total = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * (batch_count / total)
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            new_var = (m_a + m_b + delta.pow(2) * self.count * batch_count / total) / total
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(total)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            std = torch.sqrt(self.var + self.eps)
            return torch.clamp((x - self.mean) / std, -self.clip, self.clip)

    # ==========================================================================
    # 0b.  Squashed Gaussian helper  (R8 fix)
    # ==========================================================================
    # Numerical safety margin: tanh saturates to exactly +/-1.0 in float32 well
    # before the underlying pre-squash value is actually infinite, which would
    # make log(1 - tanh(u)^2) -> log(0) -> -inf and poison the loss. Clamping
    # the squashed action away from the boundary keeps the correction term
    # finite without perceptibly biasing the action itself.
    _TANH_EPS = 1e-6

    def _squash(u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(u)

    def _squash_log_prob(u: torch.Tensor, normal: "Normal") -> torch.Tensor:
        """log pi(a=tanh(u)) where u ~ normal, summed over the action dims.

        Standard SAC-style correction:
            log pi(a) = log N(u) - sum_i log(1 - tanh(u_i)^2)
        computed in a numerically stable form via softplus rather than naively
        evaluating 1 - tanh(u)^2 (which underflows to 0 for |u| >~ 4, well
        within the range an untrained policy's mean/std can produce).
        """
        log_prob_u = normal.log_prob(u).sum(-1)
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u)), the stable form
        correction = (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return log_prob_u - correction

    # ==========================================================================
    # 1.  Actor-Critic with a maneuver (option) head  (Sec. 5)
    # ==========================================================================
    class ActorCritic(nn.Module):
        """Hierarchical policy:
            * a discrete *maneuver* head pi(m | o)         (the option level), and
            * a continuous *control* head pi(a | o, m)     conditioned on the maneuver,
            * a shared critic V(o).

        R8 · Control actions are bounded to (-1, 1) via a squashed Gaussian:
        the network parametrises Normal(mean, std) over an UNBOUNDED pre-squash
        variable u; the executed action is a = tanh(u). log_prob is corrected
        by the tanh Jacobian everywhere an action is sampled or evaluated, so
        PPO's importance ratio stays consistent with what was actually sent to
        the environment. Callers that need the pre-squash mean/std directly
        (e.g. for diagnostics) can still get them from `_control_dist`."""

        def __init__(self, state_dim: int, action_dim: int, n_maneuvers: int,
                     hidden: int = 256):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.n_maneuvers = n_maneuvers
            self.obs_norm = RunningNorm(state_dim)
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            self.maneuver_head = nn.Linear(hidden, n_maneuvers)
            self.control_mean = nn.Sequential(
                nn.Linear(hidden + n_maneuvers, hidden), nn.Tanh(),
                nn.Linear(hidden, action_dim),
            )
            self.control_logstd = nn.Parameter(torch.full((action_dim,), -0.5))
            self.critic = nn.Linear(hidden, 1)

        # --- primitives ---
        def features(self, s):
            return self.shared(self.obs_norm(s))

        def maneuver_logits(self, s):
            return self.maneuver_head(self.features(s))

        def value(self, s):
            return self.critic(self.features(s)).squeeze(-1)

        def _control_dist(self, h, maneuver_onehot):
            """Returns the PRE-squash Normal(mean, std) over u in R^action_dim.
            Callers must apply tanh (and the matching log-prob correction) to
            get the actual bounded action -- see `act`, `act_with_maneuver`,
            `evaluate` below, which all do this consistently."""
            x = torch.cat([h, maneuver_onehot], dim=-1)
            mean = self.control_mean(x)
            std = self.control_logstd.exp().expand_as(mean)
            return Normal(mean, std)

        # --- sampling ---
        def act(self, s):
            """Sample maneuver then control. Returns (m, a, logp, value), where
            `a` is already squashed to (-1, 1) and `logp` is the corresponding
            corrected log-probability (maneuver logp + squashed control logp)."""
            h = self.features(s)
            m_dist = Categorical(logits=self.maneuver_head(h))
            m = m_dist.sample()
            onehot = F.one_hot(m, self.n_maneuvers).float()
            c_dist = self._control_dist(h, onehot)
            u = c_dist.rsample()
            a = _squash(u)
            logp = m_dist.log_prob(m) + _squash_log_prob(u, c_dist)
            return m, a, logp, self.critic(h).squeeze(-1)

        def act_with_maneuver(self, s, maneuver_idx):
            """Sample a control conditioned on a *given* maneuver (used by the
            planner's per-group control beam). Returns (a, control_logp), `a`
            already squashed to (-1, 1) and `control_logp` corrected for it."""
            h = self.features(s)
            m = torch.as_tensor(maneuver_idx, dtype=torch.long, device=h.device)
            if m.dim() == 0:
                m = m.unsqueeze(0)
            onehot = F.one_hot(m, self.n_maneuvers).float()
            c_dist = self._control_dist(h, onehot)
            u = c_dist.rsample()
            a = _squash(u)
            logp = _squash_log_prob(u, c_dist)
            return a, logp

        # --- evaluation for PPO ---
        def evaluate(self, s, m, a):
            """Batched. `a` is the SQUASHED action (in (-1,1)) as stored in the
            rollout buffer. Returns (logp, entropy, value) for the joint
            (maneuver, control) policy, with logp computed consistently under
            the same squashed-Gaussian correction used at sampling time.

            R8 · To evaluate log_prob(a) under the squashed distribution we
            need the pre-squash u = atanh(a). `a` is clamped away from +/-1
            first since PPO's training actions were generated by tanh and are
            therefore strictly inside (-1,1) up to float precision, but stored/
            reloaded values could in principle sit exactly on the boundary and
            send atanh to +/-inf.
            """
            h = self.features(s)
            m_dist = Categorical(logits=self.maneuver_head(h))
            onehot = F.one_hot(m.long(), self.n_maneuvers).float()
            c_dist = self._control_dist(h, onehot)
            a_clamped = a.clamp(-1.0 + _TANH_EPS, 1.0 - _TANH_EPS)
            u = torch.atanh(a_clamped)
            logp = m_dist.log_prob(m.long()) + _squash_log_prob(u, c_dist)
            # Entropy has no closed form for a squashed Gaussian; we use the
            # pre-squash Normal entropy as in the standard SAC approximation
            # (exact for the maneuver Categorical, approximate for control).
            entropy = m_dist.entropy() + c_dist.entropy().sum(-1)
            return logp, entropy, self.critic(h).squeeze(-1)

    # ==========================================================================
    # 2.  World model with grounding heads  (Secs. 2, 8)
    # ==========================================================================
    class WorldModel(nn.Module):
        """Encodes o_t -> latent z, predicts a one-step imagined transition plus
        auxiliary targets. Uses a GRU for the dynamics core so temporal patterns
        (e.g. a VRU decelerating across steps) are captured. Heads:
            next_state  -- imagined o_{t+1}              (dream loss, Sec. 2/8)
            reward      -- imagined scalar reward
            risk        -- VRU-risk in [0,1]             (aux head)
            progress    -- route-progress scalar         (aux head)
            recon       -- reconstruct o_t from z         (scene-reconstruction, Sec. 8)
            density     -- scene risk-density in [0,1]    (risk-density head, Sec. 8)
        """

        def __init__(self, state_dim: int, action_dim: int,
                     hidden: int = 256, latent: int = 128):
            super().__init__()
            self.latent = latent
            # R5 · normalise the encoder's input only; output heads (next_state,
            # recon) stay in raw observation scale since their targets/losses
            # are computed against raw env states.
            self.obs_norm = RunningNorm(state_dim)
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, latent), nn.ReLU(),
            )
            # GRU dynamics: captures temporal structure across imagined steps
            self.gru = nn.GRUCell(latent + action_dim, hidden)
            self.next_state_head = nn.Linear(hidden, state_dim)
            self.reward_head = nn.Linear(hidden, 1)
            self.risk_head = nn.Linear(hidden, 1)
            self.progress_head = nn.Linear(hidden, 1)
            self.recon_head = nn.Linear(latent, state_dim)
            self.density_head = nn.Linear(latent, 1)

        def forward(self, s, a, h: Optional[torch.Tensor] = None):
            """Single-step imagination. h is the optional GRU hidden state
            (None = zeros). Returns the outputs dict plus the new hidden state."""
            z = self.encoder(self.obs_norm(s))
            batch = z.shape[0] if z.dim() > 1 else 1
            if h is None:
                h = torch.zeros(batch, self.gru.hidden_size,
                                dtype=z.dtype, device=z.device)
                if z.dim() == 1:
                    h = h.squeeze(0)
            gru_in = torch.cat([z, a], dim=-1)
            h_new = self.gru(gru_in, h)
            return dict(
                next_state=self.next_state_head(h_new),
                reward=self.reward_head(h_new).squeeze(-1),
                risk=torch.sigmoid(self.risk_head(h_new)).squeeze(-1),
                progress=self.progress_head(h_new).squeeze(-1),
                recon=self.recon_head(z),
                density=torch.sigmoid(self.density_head(z)).squeeze(-1),
                latent=z,
                hidden=h_new,
            )

        def scene(self, s):
            """Scene heads that do NOT need an action (used for risk-density and
            reconstruction grounding, and by the planner's difficulty estimate)."""
            z = self.encoder(self.obs_norm(s))
            return dict(
                recon=self.recon_head(z),
                density=torch.sigmoid(self.density_head(z)).squeeze(-1),
                latent=z,
            )

    class WorldModelEnsemble(nn.Module):
        """Small ensemble for epistemic uncertainty (Sec. 8). Disagreement on the
        predicted next state is high under occlusion / novel configurations and
        feeds both the serendipity bonus and the difficulty-aware budget."""

        def __init__(self, n_models: int, state_dim: int, action_dim: int,
                     hidden: int = 256, latent: int = 128):
            super().__init__()
            self.models = nn.ModuleList([
                WorldModel(state_dim, action_dim, hidden, latent)
                for _ in range(n_models)
            ])

        def forward(self, s, a):
            return [m(s, a) for m in self.models]

        def predict(self, s, a):
            outs = self.forward(s, a)
            ns = torch.stack([o["next_state"] for o in outs], dim=0)      # [n,B,D]
            return dict(
                next_state=ns.mean(0),
                reward=torch.stack([o["reward"] for o in outs], 0).mean(0),
                risk=torch.stack([o["risk"] for o in outs], 0).mean(0),
                density=torch.stack([o["density"] for o in outs], 0).mean(0),
                uncertainty=ns.var(0).mean(-1),                           # [B]
            )

    # ==========================================================================
    # 3.  Wrappers: torch networks -> the NumPy interface the planner expects
    # ==========================================================================
    class TorchPolicyWrapper:
        def __init__(self, policy: "ActorCritic",
                     maneuvers=DEFAULT_MANEUVERS, device: str = "cpu"):
            self.policy = policy
            self.maneuvers = tuple(maneuvers)
            self.device = device

        def _t(self, state):
            return torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                   device=self.device).unsqueeze(0)

        @torch.no_grad()
        def maneuver_probs(self, state):
            logits = self.policy.maneuver_logits(self._t(state))
            return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        @torch.no_grad()
        def sample_action(self, state, maneuver):
            # R8 · already squashed to (-1, 1) inside act_with_maneuver.
            a, logp = self.policy.act_with_maneuver(self._t(state), [int(maneuver)])
            return a.squeeze(0).cpu().numpy(), float(logp.item())

        @torch.no_grad()
        def value(self, state):
            return float(self.policy.value(self._t(state)).item())

    class TorchWorldModelWrapper:
        def __init__(self, wm: "WorldModel",
                     ensemble: Optional["WorldModelEnsemble"] = None,
                     ego_xy_fn: Callable = mock_ego_xy, device: str = "cpu"):
            self.wm = wm
            self.ensemble = ensemble
            self.ego_xy_fn = ego_xy_fn
            self.device = device

        def _t(self, x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                   device=self.device).unsqueeze(0)

        @torch.no_grad()
        def step(self, state, action, hidden=None):
            # R1 · accept and propagate the GRU hidden state across imagined
            # steps. Without this, every imagined transition resets the
            # dynamics core to zero, defeating the point of using a GRU.
            h_in = None
            if hidden is not None:
                h_in = torch.as_tensor(np.asarray(hidden), dtype=torch.float32,
                                       device=self.device)
                if h_in.dim() == 1:
                    h_in = h_in.unsqueeze(0)
            out = self.wm(self._t(state), self._t(action), h_in)
            h_out = out["hidden"].squeeze(0).cpu().numpy()
            return (out["next_state"].squeeze(0).cpu().numpy(),
                    float(out["reward"].item()), float(out["risk"].item()),
                    h_out)

        @torch.no_grad()
        def risk_density(self, state):
            return float(self.wm.scene(self._t(state))["density"].item())

        @torch.no_grad()
        def uncertainty(self, state, action):
            if self.ensemble is None:
                return 0.0
            u = self.ensemble.predict(self._t(state), self._t(action))["uncertainty"]
            return float(u.item())

        def ego_xy(self, state):
            return self.ego_xy_fn(state)

    # ==========================================================================
    # 4.  Losses  (base PPO + auxiliary/dream losses, Sec. 8)
    # ==========================================================================
    def _risk_shaped_reward(reward: float, info: dict, cfg: TrainConfig) -> float:
        """C1/R4 · Optional risk-aware reward shaping applied to the real
        environment step, on top of whatever risk term the env's own reward
        function already includes.

        r_safe = r  - lambda_risk_reward * TTC^{-1}  - lambda_density_reward * occ

        IMPORTANT: this uses cfg.lambda_risk_reward / cfg.lambda_density_reward,
        which default to 0.0 and are intentionally SEPARATE from
        cfg.lambda_risk / cfg.lambda_density (those only weight the world
        model's auxiliary prediction losses in world_model_update, not the
        reward the policy is optimised against). EnhancedMockEnv already
        bakes a TTC penalty (r_ttc) and collision/near-miss terms into its
        reward, so stacking another TTC^{-1} penalty here at full strength
        on top of that would silently double-count the same risk signal and
        push the agent toward over-braking. Raise lambda_risk_reward /
        lambda_density_reward above 0 only for envs (e.g. a bare
        CarlaEnvAdapter) whose own reward has no risk term of its own.
        """
        min_ttc = float(info.get("min_ttc", 99.0))
        occ     = float(info.get("occlusion", 0.0))
        ttc_pen = cfg.lambda_risk_reward * (1.0 / max(min_ttc, 0.5))  # TTC^{-1}, clipped
        occ_pen = cfg.lambda_density_reward * occ
        return float(reward) - ttc_pen - occ_pen

    def ppo_update(policy, optimizer, batch, cfg: TrainConfig):
        logp, entropy, values = policy.evaluate(
            batch["states"], batch["maneuvers"], batch["actions"])
        ratio = torch.exp(logp - batch["log_probs"])
        adv = batch["advantages"]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        loss_pi = -torch.min(unclipped, clipped).mean()
        loss_v = F.mse_loss(values, batch["returns"])
        loss_ent = -entropy.mean()
        loss = loss_pi + cfg.vf_coef * loss_v + cfg.ent_coef * loss_ent
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        return dict(loss_pi=float(loss_pi.item()), loss_v=float(loss_v.item()),
                    entropy=float(entropy.mean().item()))

    def _whiten(wm, x):
        """Rescale an observation-scale tensor with the world model's own running
        stats (detached, so it only reshapes the loss geometry and does not
        backprop into the normaliser). Without this, next_state/recon MSE is
        dominated by the large-magnitude dims (vru_dx, min_ttc reach ~99) and the
        safety-critical [0,1] dims (visibility, occlusion, small-TTC) receive
        ~10^4x weaker gradients -- the WM learns geometry but ignores VRU danger.

        The std is floored at 1.0 so this can only ever *down-weight* wide dims,
        never *up-weight* narrow/near-constant ones. That floor is essential:
        dims that stay constant in a tier (e.g. unused VRU slots pinned at 99)
        have running variance -> 0, and naively dividing by sqrt(var) blows their
        contribution up ~100x and makes the loss diverge.
        """
        mean = wm.obs_norm.mean.detach()
        std = torch.sqrt(wm.obs_norm.var).detach().clamp(min=1.0)
        return (x - mean) / std

    def world_model_update(wm, optimizer, batch, cfg: TrainConfig):
        out = wm(batch["states"], batch["actions"])
        scene = wm.scene(batch["states"])
        # next_state / recon losses computed in whitened space so every
        # observation dimension contributes comparable gradient (see _whiten).
        loss_state = F.mse_loss(_whiten(wm, out["next_state"]),
                                _whiten(wm, batch["next_states"]))
        loss_reward = F.mse_loss(out["reward"], batch["rewards"])
        loss_risk = F.mse_loss(out["risk"], batch["risk_targets"])
        loss_progress = F.mse_loss(out["progress"], batch["progress_targets"])
        loss_recon = F.mse_loss(_whiten(wm, scene["recon"]),
                                _whiten(wm, batch["states"]))      # autoencoder grounding
        loss_density = F.mse_loss(scene["density"], batch["density_targets"])
        loss = (cfg.beta_dream * loss_state
                + loss_reward
                + cfg.lambda_risk * loss_risk
                + cfg.lambda_progress * loss_progress
                + cfg.lambda_recon * loss_recon
                + cfg.lambda_density * loss_density)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 0.5)
        optimizer.step()
        return dict(loss_state=float(loss_state.item()),
                    loss_risk=float(loss_risk.item()),
                    loss_recon=float(loss_recon.item()))

    def world_model_ensemble_update(ensemble, optimizer, batch, cfg: TrainConfig):
        """Train every ensemble member on a bootstrap resample of the batch so
        their disagreement is a meaningful epistemic-uncertainty signal."""
        n = batch["states"].shape[0]
        total = 0.0
        optimizer.zero_grad()
        for m in ensemble.models:
            idx = torch.randint(0, n, (n,), device=batch["states"].device)
            out = m(batch["states"][idx], batch["actions"][idx])
            loss = (F.mse_loss(_whiten(m, out["next_state"]),
                               _whiten(m, batch["next_states"][idx]))
                    + F.mse_loss(out["risk"], batch["risk_targets"][idx]))
            loss.backward()
            total += float(loss.item())
        nn.utils.clip_grad_norm_(ensemble.parameters(), 0.5)
        optimizer.step()
        return dict(ensemble_loss=total / max(1, len(ensemble.models)))

    # ==========================================================================
    # 5.  Agent assembly + training loop  (Sec. 10)
    # ==========================================================================
    def build_agent(env, cfg: TrainConfig, maneuvers=DEFAULT_MANEUVERS,
                    use_ensemble: bool = False, ego_xy_fn: Callable = mock_ego_xy,
                    mandated_action_fn: Callable = mock_mandated_action,
                    device: str = "cpu",
                    policy_state: Optional[dict] = None,
                    wm_state: Optional[dict] = None,
                    ensemble_state: Optional[dict] = None,
                    use_traffic_predictor: bool = False):
        """Build the full agent. If `wm_state` / `policy_state` / `ensemble_state`
        are provided (e.g. loaded from a checkpoint via load_world_model_checkpoint
        in run_training.py), the corresponding networks are initialised from
        them with strict=False, instead of always starting from scratch.

        use_traffic_predictor : si True et que traffic_predictor.py est disponible,
            instancie un MultiAgentPredictor (prédiction de trajectoire + intention
            sémantique + incertitude MC-dropout) et le câble dans le SDBSPlanner.
            Le scoring du beam est alors pondéré par P(INTENT_CROSS) de chaque VRU,
            évitant de pénaliser les plans qui croisent des piétons qui longent
            le trottoir sans intention de traverser.
        """
        n_man = int(max(maneuvers)) + 1
        policy = ActorCritic(env.state_dim, env.action_dim, n_man).to(device)
        wm = WorldModel(env.state_dim, env.action_dim).to(device)
        ensemble = (WorldModelEnsemble(3, env.state_dim, env.action_dim).to(device)
                    if use_ensemble else None)

        if policy_state is not None:
            missing, unexpected = policy.load_state_dict(policy_state, strict=False)
            print(f"[build_agent] policy loaded from checkpoint "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if wm_state is not None:
            missing, unexpected = wm.load_state_dict(wm_state, strict=False)
            print(f"[build_agent] world model loaded from checkpoint "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if ensemble is not None and ensemble_state is not None:
            missing, unexpected = ensemble.load_state_dict(ensemble_state, strict=False)
            print(f"[build_agent] ensemble loaded from checkpoint "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        elif ensemble is not None and wm_state is not None:
            # No dedicated ensemble checkpoint but we do have a WM checkpoint:
            # seed every ensemble member from it (anchor + noised siblings) so
            # the ensemble starts with informative, non-random dynamics.
            ensemble.models[0].load_state_dict(wm_state, strict=False)
            for i in range(1, len(ensemble.models)):
                ensemble.models[i].load_state_dict(wm_state, strict=False)
                with torch.no_grad():
                    for p in ensemble.models[i].parameters():
                        p.add_(torch.randn_like(p) * 0.01)
            print("[build_agent] ensemble seeded from world-model checkpoint "
                  f"({len(ensemble.models)} members)")

        pwrap  = TorchPolicyWrapper(policy, maneuvers, device)
        wmwrap = TorchWorldModelWrapper(wm, ensemble, ego_xy_fn, device)

        # ------------------------------------------------------------------
        # MultiAgentPredictor (contribution supplémentaire au papier)
        # Prédit trajectoires + intention sémantique (traverse/longe/s'arrête)
        # + incertitude MC-dropout pour chaque VRU tracké.
        # Câblé dans SDBSPlanner via traffic_predictor= ; le scoring du beam
        # est pondéré par P(INTENT_CROSS) grâce à lambda_collision dans cfg.planner.
        # ------------------------------------------------------------------
        traffic_pred = None
        if use_traffic_predictor:
            try:
                from sdbs.planning.traffic_predictor import MultiAgentPredictor
                traffic_pred = MultiAgentPredictor(
                    state_dim  = env.state_dim,
                    max_agents = 10,
                    horizon    = cfg.planner.horizon,
                    hidden_dim = 128,
                    seq_len    = 5,
                    mc_passes  = 5,
                    device     = device,
                )
                print("[build_agent] MultiAgentPredictor activé "
                      f"(horizon={cfg.planner.horizon}, mc_passes=5).")
            except ImportError:
                print("[build_agent] traffic_predictor.py introuvable — "
                      "MultiAgentPredictor désactivé.")

        planner = SDBSPlanner(pwrap, wmwrap, cfg.planner, cfg.budget,
                              maneuvers=maneuvers,
                              mandated_action_fn=mandated_action_fn,
                              traffic_predictor=traffic_pred)

        opt_pi  = torch.optim.Adam(policy.parameters(), lr=3e-4)
        opt_wm  = torch.optim.Adam(wm.parameters(), lr=3e-4)
        opt_ens = (torch.optim.Adam(ensemble.parameters(), lr=3e-4)
                   if ensemble is not None else None)
        return dict(policy=policy, wm=wm, ensemble=ensemble, planner=planner,
                    pwrap=pwrap, wmwrap=wmwrap,
                    opt_pi=opt_pi, opt_wm=opt_wm, opt_ens=opt_ens)

    def collect_rollout(env, agent, cfg: TrainConfig, per: PrioritizedScenarioReplay,
                        curriculum: CurriculumController, rollout_size: int,
                        device: str = "cpu") -> RolloutBuffer:
        planner = agent["planner"]
        buf = RolloutBuffer(rollout_size, env.state_dim, env.action_dim,
                            gamma=cfg.planner.gamma, gae_lambda=cfg.gae_lambda)
        # sample a starting scenario from the prioritised bank (hard-scenario mining)
        leaves, scenarios, _ = per.sample(1)
        active_leaf, active_scn = leaves[0], scenarios[0]
        obs = env.reset(**{k: active_scn[k] for k in
                           ("tier", "n_vru", "occlusion_prob", "adversarial")
                           if k in active_scn})
        progress0 = 0.0
        while buf.ptr < rollout_size:
            # R5 · feed the real observed state into the running normalisers
            # so obs_norm actually tracks the true observation distribution
            # (only ever updated on real env states, never on imagined ones).
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
            agent["policy"].obs_norm.update(obs_t)
            agent["wm"].obs_norm.update(obs_t)
            if agent["ensemble"] is not None:
                for m in agent["ensemble"].models:
                    m.obs_norm.update(obs_t)

            info = env.info()
            out = planner.plan(obs, info)
            next_obs, reward, done, step_info = env.step(out["action"], out["maneuver"])

            # ------------------------------------------------------------------
            # Mise à jour du tracker multi-agent (contribution supplémentaire)
            # Lit les observations VRU depuis step_info["agent_observations"]
            # si le CARLA adapter les exporte (clé optionnelle — ignorée en mode
            # mock où la clé est absente). Format :
            #   dict agent_id -> {"position": [x,y], "velocity": [vx,vy]}
            # Le MultiAgentPredictor accumule l'historique (seq_len derniers pas)
            # et prédit trajectoire + intention à chaque plan() suivant.
            # ------------------------------------------------------------------
            _tp = agent["planner"].traffic_predictor
            if _tp is not None:
                for a_id, a_data in step_info.get("agent_observations", {}).items():
                    _tp.observe_agent(
                        agent_id = int(a_id),
                        position = a_data.get("position", [0.0, 0.0]),
                        velocity = a_data.get("velocity", [0.0, 0.0]),
                    )

            # C1 · Risk-aware reward shaping on the real env reward
            shaped_reward = _risk_shaped_reward(reward, step_info, cfg)

            # C4 · intrinsic serendipity reward (Sec. 4.1): bounded so safety dominates
            # and never applied when a hard safety response was mandated.
            if not out["meta"].get("mandated", False):
                ser = float(out["meta"].get("ser", 0.0))
                shaped_reward = shaped_reward + cfg.eta_serendipity_reward * math.tanh(ser)

            buf.store(
                state=obs, action=out["action"], maneuver=out["maneuver"],
                reward=shaped_reward, done=float(done), value=out["value"],
                log_prob=out["logp"], next_state=next_obs,
                risk_t=float(step_info.get("vru_risk", 0.0)),
                progress_t=float(step_info.get("progress", 0.0)),
                density_t=float(step_info.get("density", 0.0)),
            )
            obs = next_obs
            if done:
                buf.finish_path(last_value=0.0)
                success = bool(step_info.get("success", False))
                curriculum.record(
                    success,
                    collisions=int(step_info.get("collisions", 0)),
                    ttc_violations=int(step_info.get("ttc_violations", 0)),
                )
                # update this scenario's priority from how badly the agent did
                err = per.error_from_outcome(
                    collisions=int(step_info.get("collisions", 0)),
                    near_misses=int(step_info.get("near_misses", 0)),
                    ttc_violations=int(step_info.get("ttc_violations", 0)),
                    progress_deficit=1.0 - float(step_info.get("progress", progress0)),
                )
                per.update_priorities([active_leaf], [err])
                # next scenario
                leaves, scenarios, _ = per.sample(1)
                active_leaf, active_scn = leaves[0], scenarios[0]
                obs = env.reset(**{k: active_scn[k] for k in
                                   ("tier", "n_vru", "occlusion_prob", "adversarial")
                                   if k in active_scn})
        # bootstrap the final (possibly unfinished) path
        if buf.path_start < buf.ptr:
            last_v = agent["pwrap"].value(obs)
            buf.finish_path(last_value=last_v)
        return buf

    def _to_tensors(batch: dict, device: str) -> dict:
        out = {}
        for k, v in batch.items():
            dtype = torch.long if k == "maneuvers" else torch.float32
            out[k] = torch.as_tensor(v, dtype=dtype, device=device)
        return out

    def _default_eval_scenarios() -> list[dict]:
        """A small fixed battery spanning all 3 tiers, held out of the PER bank,
        used purely to measure generalisation instead of self-grading on the
        same distribution the agent is being trained/curriculum-mined on."""
        return [
            dict(tier=0, n_vru=0, occlusion_prob=0.0, adversarial=False),
            dict(tier=1, n_vru=1, occlusion_prob=0.0, adversarial=False),
            dict(tier=1, n_vru=2, occlusion_prob=0.1, adversarial=False),
            dict(tier=2, n_vru=1, occlusion_prob=0.8, adversarial=False),
            dict(tier=2, n_vru=2, occlusion_prob=1.0, adversarial=True),
            dict(tier=2, n_vru=3, occlusion_prob=1.0, adversarial=True),
        ]

    def evaluate_agent(env, agent, eval_scenarios: list[dict],
                       episodes_per_scenario: int = 2) -> dict:
        """C3 · Held-out evaluation, decoupled from the PER training bank.
        Runs the *current* planner deterministically-ish on a fixed scenario
        battery and reports aggregate metrics. This is what you should actually
        trust when judging whether the agent generalises, since PER success/
        collision-free rates are measured on the same distribution being mined
        for training and will look better than true performance."""
        planner = agent["planner"]
        n_ep = 0
        successes = collisions = near_misses = ttc_viol = 0
        total_return = 0.0
        for scn in eval_scenarios:
            for _ in range(episodes_per_scenario):
                obs = env.reset(**{k: scn[k] for k in
                                   ("tier", "n_vru", "occlusion_prob", "adversarial")
                                   if k in scn})
                done = False
                ep_return = 0.0
                step_info = {}
                steps = 0
                while not done and steps < getattr(env, "max_steps", 200):
                    info = env.info()
                    out = planner.plan(obs, info)
                    obs, r, done, step_info = env.step(out["action"], out["maneuver"])
                    ep_return += r
                    steps += 1
                n_ep += 1
                total_return += ep_return
                successes += int(bool(step_info.get("success", False)))
                collisions += int(step_info.get("collisions", 0) > 0)
                near_misses += int(step_info.get("near_misses", 0) > 0)
                ttc_viol += int(step_info.get("ttc_violations", 0) > 0)
        n_ep = max(1, n_ep)
        return dict(
            episodes=n_ep,
            avg_return=total_return / n_ep,
            success_rate=successes / n_ep,
            collision_free_rate=1.0 - collisions / n_ep,
            near_miss_free_rate=1.0 - near_misses / n_ep,
            ttc_clean_rate=1.0 - ttc_viol / n_ep,
        )

    def train(env, cfg: Optional[TrainConfig] = None, n_iterations: int = 100,
              rollout_size: Optional[int] = None, use_ensemble: bool = False,
              ego_xy_fn: Callable = mock_ego_xy,
              mandated_action_fn: Callable = mock_mandated_action,
              device: str = "cpu", seed: int = 0, verbose: bool = True,
              save_every: int = 0, save_path: str = "checkpoint.pt",
              policy_state: Optional[dict] = None,
              wm_state: Optional[dict] = None,
              ensemble_state: Optional[dict] = None,
              resume_path: Optional[str] = None,
              eval_every: int = 0,
              eval_scenarios: Optional[list[dict]] = None,
              eval_episodes_per_scenario: int = 2,
              use_traffic_predictor: bool = False):
        """Main training loop.

        Set save_every > 0 to checkpoint everything (networks + optimizers +
        curriculum + PER state + RNG) every that many iterations, to save_path.
        Pass resume_path to a previous full checkpoint to continue training
        without losing optimizer momentum / curriculum progress (R2 fix).
        Pass policy_state / wm_state / ensemble_state (state_dicts) to seed
        the networks from an externally pre-trained checkpoint (R2/C of the
        --wm_checkpoint flow in run_training.py) -- this actually wires the
        loaded weights into the agent that train() builds, instead of
        discarding them.
        Set eval_every > 0 to run a held-out evaluation battery (decoupled
        from the PER training bank) every that many iterations (C3 fix).
        Set use_traffic_predictor=True pour activer le MultiAgentPredictor :
        prédiction trajectoire + intention sémantique + incertitude MC-dropout
        pour chaque VRU tracké. Nécessite traffic_predictor.py dans le path et
        que l'env exporte step_info["agent_observations"] (CarlaEnvAdapter le
        fait automatiquement ; MockEnv l'ignore silencieusement).
        """
        cfg = cfg or TrainConfig()
        rollout_size = rollout_size or cfg.rollout_size
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = build_agent(env, cfg, use_ensemble=use_ensemble,
                             ego_xy_fn=ego_xy_fn, mandated_action_fn=mandated_action_fn,
                             device=device, policy_state=policy_state,
                             wm_state=wm_state, ensemble_state=ensemble_state,
                             use_traffic_predictor=use_traffic_predictor)
        per = PrioritizedScenarioReplay(cfg.per)
        curriculum = CurriculumController(cfg.curriculum)
        # seed the scenario bank at the current tier
        for _ in range(64):
            per.add(curriculum.sample_scenario_params())

        start_it = 0
        if resume_path is not None:
            print(f"[resume] Chargement du checkpoint complet depuis {resume_path}...")
            # R7 · PyTorch >= 2.6 defaults torch.load(weights_only=True), which
            # rejects checkpoints containing numpy RNG state (saved above for
            # exact resume). These checkpoints are produced by this same code
            # (trusted, not arbitrary downloads), so weights_only=False is safe
            # here.
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if ckpt.get("policy") is not None:
                agent["policy"].load_state_dict(ckpt["policy"], strict=False)
            if ckpt.get("wm") is not None:
                agent["wm"].load_state_dict(ckpt["wm"], strict=False)
            if ckpt.get("ensemble") is not None and agent["ensemble"] is not None:
                agent["ensemble"].load_state_dict(ckpt["ensemble"], strict=False)
            if ckpt.get("opt_pi") is not None:
                agent["opt_pi"].load_state_dict(ckpt["opt_pi"])
            if ckpt.get("opt_wm") is not None:
                agent["opt_wm"].load_state_dict(ckpt["opt_wm"])
            if ckpt.get("opt_ens") is not None and agent["opt_ens"] is not None:
                agent["opt_ens"].load_state_dict(ckpt["opt_ens"])
            if ckpt.get("curriculum_state") is not None:
                cs = ckpt["curriculum_state"]
                curriculum.stage = cs.get("stage", curriculum.stage)
                curriculum._recent = list(cs.get("recent", []))
                curriculum._collision_free = list(cs.get("collision_free", []))
                curriculum._ttc_clean = list(cs.get("ttc_clean", []))
            if ckpt.get("torch_rng_state") is not None:
                torch.set_rng_state(ckpt["torch_rng_state"])
            if ckpt.get("numpy_rng_state") is not None:
                np.random.set_state(ckpt["numpy_rng_state"])
            start_it = int(ckpt.get("iteration", 0))
            print(f"[resume] Reprise a l'iteration {start_it}, tier={curriculum.stage}")

        eval_scenarios = eval_scenarios or _default_eval_scenarios()

        for it in range(start_it, n_iterations):
            buf = collect_rollout(env, agent, cfg, per, curriculum,
                                  rollout_size, device)
            batch = _to_tensors(buf.get(), device)
            n = batch["states"].shape[0]

            pi_stats = wm_stats = {}
            for _ in range(cfg.update_epochs):
                perm = torch.randperm(n, device=device)
                for start in range(0, n, cfg.minibatch):
                    mb_idx = perm[start:start + cfg.minibatch]
                    mb = {k: v[mb_idx] for k, v in batch.items()}
                    pi_stats = ppo_update(agent["policy"], agent["opt_pi"], mb, cfg)
                    wm_stats = world_model_update(agent["wm"], agent["opt_wm"], mb, cfg)
                    if agent["ensemble"] is not None:
                        world_model_ensemble_update(agent["ensemble"], agent["opt_ens"], mb, cfg)

            unlocked = curriculum.maybe_unlock()
            # top up the scenario bank when a new tier unlocks
            if unlocked:
                for _ in range(64):
                    per.add(curriculum.sample_scenario_params())

            # C3 · held-out evaluation, decoupled from the PER bank
            eval_stats = None
            if eval_every > 0 and (it + 1) % eval_every == 0:
                eval_stats = evaluate_agent(env, agent, eval_scenarios,
                                            eval_episodes_per_scenario)
                if verbose:
                    print(f"  [eval] return={eval_stats['avg_return']:.2f} "
                          f"succ={eval_stats['success_rate']:.2f} "
                          f"col_free={eval_stats['collision_free_rate']:.2f} "
                          f"ttc_ok={eval_stats['ttc_clean_rate']:.2f} "
                          f"(n={eval_stats['episodes']})")

            # periodic checkpoint -- R2: now saves optimizers + curriculum + PER
            # state + RNG state too, so resume_path can pick up training without
            # losing Adam momentum or curriculum progress.
            if save_every > 0 and (it + 1) % save_every == 0:
                torch.save({
                    "iteration": it + 1,
                    "policy": agent["policy"].state_dict(),
                    "wm": agent["wm"].state_dict(),
                    "ensemble": agent["ensemble"].state_dict()
                              if agent["ensemble"] is not None else None,
                    "opt_pi": agent["opt_pi"].state_dict(),
                    "opt_wm": agent["opt_wm"].state_dict(),
                    "opt_ens": agent["opt_ens"].state_dict()
                              if agent["opt_ens"] is not None else None,
                    "tier": curriculum.stage,
                    "curriculum_state": dict(
                        stage=curriculum.stage,
                        recent=list(curriculum._recent),
                        collision_free=list(curriculum._collision_free),
                        ttc_clean=list(curriculum._ttc_clean),
                    ),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                }, save_path)
                if verbose:
                    print(f"  [ckpt] saved to {save_path}")

            if verbose:
                col_rate = curriculum.collision_free_rate()
                ttc_rate = curriculum.ttc_clean_rate()
                print(f"iter {it:04d} | tier {curriculum.stage} "
                      f"succ {curriculum.success_rate():.2f} "
                      f"col_free {col_rate:.2f} ttc_ok {ttc_rate:.2f}"
                      f"| pi {pi_stats.get('loss_pi', 0):+.3f} "
                      f"v {pi_stats.get('loss_v', 0):.3f} "
                      f"ent {pi_stats.get('entropy', 0):.2f} "
                      f"| wm_state {wm_stats.get('loss_state', 0):.3f} "
                      f"wm_recon {wm_stats.get('loss_recon', 0):.3f}"
                      + (" | UNLOCKED" if unlocked else ""))
        return agent


# ==============================================================================
# 6.  CARLA adapter (stub to implement -- mirror MockDrivingEnv's interface)
# ==============================================================================
class CarlaEnvAdapter:
    """Plug the planner into CARLA by implementing this class against your CARLA
    / gym wrapper. It MUST expose exactly the interface the training loop and the
    planner rely on (the same one `MockDrivingEnv` implements):

      Attributes
      ----------
      state_dim   : int   length of the structured observation o_t
      action_dim  : int   continuous control dim (>= 3: steer, throttle, brake[, stop])
      n_maneuvers : int   number of discrete maneuvers (== max(Maneuver)+1)

      Methods
      -------
      reset(tier, n_vru, occlusion_prob, adversarial) -> obs : np.ndarray[state_dim]
      step(action, maneuver=None) -> (obs, reward, done, info)
      info() -> dict with at least:
          min_ttc, ttc_threshold, n_vru, occlusion, uncertainty,
          collisions, near_misses, ttc_violations,
          progress, vru_risk, density, success

    You must also provide, matching your state layout:
      * ego_xy_fn(state) -> (x, y)            for conflict-cell diversity (Sec. 3)
      * mandated_action_fn(state, info)       the hard safety floor (Sec. 6):
            emergency brake when min_ttc < tau_safe, stop at red, etc.

    Note on action range (R8): the policy now emits actions already squashed
    to (-1, 1) via tanh, so a downstream adapter does NOT need to clip/squash
    again -- but it DOES need to map that (-1,1) range to whatever physical
    units the simulator expects (e.g. steer in radians, throttle/brake in
    [0,1] if your sim doesn't use a symmetric range for those two channels).

    Mapping CARLA -> observation o_t = [e_t, l_t, m_t, r_t, v_t, p_t]:
      * e_t : ego kinematics (speed, accel, heading) from the CARLA ego sensors
      * l_t : lane / route features (waypoint API, lane invasion)
      * m_t : map / right-of-way (traffic-light state, stop signs)
      * r_t : risk features (TTC to nearest VRU, occlusion flags from semantic LiDAR)
      * v_t : VRU tracks (pedestrian/cyclist positions & velocities)
      * p_t : progress along the planned route

    Targets for the world-model heads come from the simulator at each step:
      * next_state target  = the actual next observation
      * risk target        = ground-truth VRU proximity/TTC risk
      * progress target     = per-step route progress
      * density target      = scene risk-density (e.g. contested-cell fraction)

    Scenario tiers (curriculum, Sec. 7):
      * tier 0: empty/low-density routes
      * tier 1: predictable VRU crossings at marked intersections
      * tier 2: adversarial greedy traps (occluded pedestrians, jaywalkers,
                cyclists overtaking on the right, amber-light dilemmas)
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CarlaEnvAdapter is a stub. Implement reset/step/info against your "
            "CARLA wrapper and supply ego_xy_fn + mandated_action_fn. See the "
            "docstring and README for the required interface."
        )


# ==============================================================================
# 7.  Entry point
# ==============================================================================
def main():
    if not HAS_TORCH:
        print("PyTorch is not available in this environment.")
        print("Running the NumPy core smoke test instead "
              "(install torch to run the full training demo).\n")
        run_core_smoke_test()
        return

    # Tiny end-to-end demo on the mock env to verify everything wires together.
    # (This trains nothing meaningful -- it just exercises the full loop.)
    print("PyTorch found -- running a short training demo on MockDrivingEnv.\n")
    env = MockDrivingEnv(seed=0)
    cfg = TrainConfig()
    train(env, cfg, n_iterations=3, rollout_size=256, use_ensemble=False,
          ego_xy_fn=mock_ego_xy, mandated_action_fn=mock_mandated_action,
          device="cpu", seed=0, verbose=True)
    print("\nDemo finished. Swap MockDrivingEnv for CarlaEnvAdapter to train for real.")


if __name__ == "__main__":
    main()