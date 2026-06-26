"""
enhanced_mock_env.py
================================================================================
An enriched drop-in replacement for MockDrivingEnv that simulates more
realistic CARLA-like scenarios while keeping everything in pure NumPy.

Why this instead of MockDrivingEnv?
  * state_dim = 20  (vs 9) — matches the CarlaEnvAdapter observation layout
    described in the README: [ego_kinematics, lane/route, map/right-of-way,
    risk, VRU_tracks, progress].  When you swap in the real CARLA adapter later
    you change NOTHING in the planner or training loop.
  * Multiple simultaneous VRUs (up to 3) with independent trajectories.
  * Five scenario archetypes across the three curriculum tiers:
      tier 0 — clear straight road
      tier 1 — visible pedestrian at marked crosswalk
             — cyclist in bike lane, predictable
      tier 2 — occluded pedestrian (the classic greedy trap)
             — jaywalker with random crossing time
             — adversarial: red light + cyclist + occlusion together
  * Observation fields mirror the 6-group structure [e,l,m,r,v,p] so your
    ego_xy_fn and mandated_action_fn transfer directly to CARLA.
  * render_ascii() gives a tiny top-down text view useful for debugging.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


# ==============================================================================
# 0.  State layout  (mirrors the CarlaEnvAdapter docstring observation groups)
# ==============================================================================
#  [0]  ego_x          — longitudinal position (m)
#  [1]  ego_y          — lateral position (m)
#  [2]  speed          — m/s
#  [3]  accel          — m/s²  (last applied)
#  [4]  heading_err    — radians from lane centre-line
#  [5]  lane_offset    — m from lane centre
#  [6]  dist_to_junction — m
#  [7]  tl_state       — 0 green / 0.5 amber / 1 red
#  [8]  stop_sign      — 0 / 1
#  [9]  right_of_way   — 1 = ego has ROW, 0 = must yield
#  [10] min_ttc        — seconds to nearest VRU (99 if none)
#  [11] occlusion_flag — fraction of VRU zone occluded [0,1]
#  [12] vru0_dx        — longitudinal distance to VRU-0 (m); 99 if absent
#  [13] vru0_dy        — lateral distance to VRU-0 (m)
#  [14] vru0_vis       — visibility [0,1]
#  [15] vru1_dx        — VRU-1 longitudinal distance
#  [16] vru1_vis       — VRU-1 visibility
#  [17] vru2_dx        — VRU-2 longitudinal distance
#  [18] vru2_vis       — VRU-2 visibility
#  [19] progress       — route completion [0,1]

(IDX_EGO_X, IDX_EGO_Y, IDX_SPEED, IDX_ACCEL, IDX_HEAD_ERR,
 IDX_LANE_OFF, IDX_DIST_JCT, IDX_TL, IDX_STOP, IDX_ROW,
 IDX_MIN_TTC, IDX_OCC,
 IDX_VRU0_DX, IDX_VRU0_DY, IDX_VRU0_VIS,
 IDX_VRU1_DX, IDX_VRU1_VIS,
 IDX_VRU2_DX, IDX_VRU2_VIS,
 IDX_PROGRESS) = range(20)

ENHANCED_STATE_DIM = 20
ENHANCED_ACTION_DIM = 3   # [steer ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]]


# ==============================================================================
# 1.  Scenario archetypes
# ==============================================================================
class ScenarioType(IntEnum):
    CLEAR            = 0   # tier 0
    VISIBLE_PED      = 1   # tier 1
    VISIBLE_CYCLIST  = 2   # tier 1
    OCCLUDED_PED     = 3   # tier 2  ← the greedy trap
    JAYWALKER        = 4   # tier 2
    ADVERSARIAL      = 5   # tier 2


# Maps curriculum tier → which scenarios can appear
TIER_SCENARIOS: dict[int, list[ScenarioType]] = {
    0: [ScenarioType.CLEAR],
    1: [ScenarioType.VISIBLE_PED, ScenarioType.VISIBLE_CYCLIST],
    2: [ScenarioType.OCCLUDED_PED, ScenarioType.JAYWALKER, ScenarioType.ADVERSARIAL],
}


@dataclass
class VRU:
    """A single Vulnerable Road User."""
    dx: float          # longitudinal distance ahead of ego (m)
    dy: float          # lateral offset from ego lane centre (m)
    speed_x: float     # VRU longitudinal speed (m/s)  — usually 0 for pedestrians
    speed_y: float     # VRU lateral/crossing speed (m/s)
    visible: float     # 1 = fully visible, 0 = occluded
    width: float = 0.5 # collision radius (m)
    emerged: bool = False

    def ttc(self, ego_speed: float) -> float:
        """Approximate TTC. Returns 99 if not on a collision course."""
        if self.dx > 40 or self.dx < 0:
            return 99.0
        if abs(self.dy) > 2.5:  # pedestrian has crossed the lane safely
            return 99.0
        closing = max(ego_speed - self.speed_x, 0.1)
        return self.dx / closing


# ==============================================================================
# 2.  Enhanced environment
# ==============================================================================
class EnhancedMockEnv:
    """
    Drop-in replacement for MockDrivingEnv with richer scenarios.

    Interface is identical to MockDrivingEnv (and CarlaEnvAdapter):
      reset(tier, n_vru, occlusion_prob, adversarial) -> obs
      step(action, maneuver=None)                    -> (obs, reward, done, info)
      info()                                         -> dict
    """

    state_dim  = ENHANCED_STATE_DIM
    action_dim = ENHANCED_ACTION_DIM
    n_maneuvers = 6   # must match len(Maneuver) in sdbs_core

    # physics
    DT          = 0.3   # seconds per step
    SPEED_MAX   = 14.0  # m/s (~50 km/h urban)
    ROUTE_LEN   = 80.0  # metres
    JUNCTION_POS = 30.0 # metres along route where crosswalk/junction sits

    def __init__(self, seed: int = 0, domain_randomization: bool = False,
                 sensor_noise_std: float = 0.15, action_latency_steps: int = 0,
                 accel_scale_range: tuple[float, float] = (0.85, 1.15),
                 vis_flicker_prob: float = 0.05):
        """
        domain_randomization : master switch (off by default so existing
            scenario banks / smoke tests stay exactly reproducible).
        sensor_noise_std      : Gaussian noise (m) added to VRU dx/dy before
            they're written into the observation, mimicking LiDAR/camera
            tracking jitter -- the policy must learn to stay robust to noisy
            range estimates, which CARLA's sensors will also produce.
        action_latency_steps  : delays applied actions by N steps (actuator/
            communication lag), so the policy doesn't learn to depend on
            instantaneous response that won't be available on a real stack.
        accel_scale_range     : per-episode random multiplier on the
            throttle/brake -> accel mapping (friction / vehicle-mass
            variation), so the policy doesn't overfit to one fixed dynamic.
        vis_flicker_prob      : per-step probability a visible VRU is
            mis-reported as occluded for one frame (sensor dropout), on top
            of the scripted occlusion scenarios.
        """
        self.rng = np.random.default_rng(seed)
        self.ttc_threshold = 3.0
        self.max_steps = 60
        self._vrus: list[VRU] = []
        self.s = np.zeros(ENHANCED_STATE_DIM, np.float32)
        self._scenario = ScenarioType.CLEAR
        # R6 · domain-randomization config (sim2real prep ahead of CARLA)
        self.domain_randomization = domain_randomization
        self.sensor_noise_std = sensor_noise_std
        self.action_latency_steps = action_latency_steps
        self.accel_scale_range = accel_scale_range
        self.vis_flicker_prob = vis_flicker_prob
        self._accel_scale = 1.0
        self._action_queue: list[np.ndarray] = []
        self._reset_counters()

    # ------------------------------------------------------------------ #
    # reset / scenario builder
    # ------------------------------------------------------------------ #
    def reset(self, tier: int = 0, n_vru: int = 0,
              occlusion_prob: float = 0.0, adversarial: bool = False) -> np.ndarray:
        self._reset_counters()
        self._vrus = []
        self.s = np.zeros(ENHANCED_STATE_DIM, np.float32)
        self._action_queue = []

        # R6 · per-episode dynamics randomization (friction / mass proxy)
        if self.domain_randomization:
            lo, hi = self.accel_scale_range
            self._accel_scale = float(self.rng.uniform(lo, hi))
        else:
            self._accel_scale = 1.0

        # pick scenario from the tier
        candidates = TIER_SCENARIOS.get(tier, TIER_SCENARIOS[0])
        self._scenario = ScenarioType(self.rng.choice([int(c) for c in candidates]))

        # initialise ego
        self.s[IDX_SPEED] = 6.0
        self.s[IDX_DIST_JCT] = self.JUNCTION_POS
        self.s[IDX_ROW] = 1.0
        self.s[IDX_MIN_TTC] = 99.0

        self._build_scenario(occlusion_prob, adversarial)
        self._update_obs()
        return self.s.copy()

    def _reset_counters(self):
        self.t = 0
        self.collisions = 0
        self.near_misses = 0
        self.ttc_violations = 0
        self._done = False

    def _build_scenario(self, occlusion_prob: float, adversarial: bool):
        sc = self._scenario

        if sc == ScenarioType.CLEAR:
            pass  # no VRUs, no traffic lights

        elif sc == ScenarioType.VISIBLE_PED:
            # pedestrian crossing ahead at junction, fully visible
            self._vrus.append(VRU(
                dx=self.JUNCTION_POS, dy=0.0,
                speed_x=0.0, speed_y=1.2,
                visible=1.0,
            ))

        elif sc == ScenarioType.VISIBLE_CYCLIST:
            # cyclist in right bike lane, moving in same direction but slower
            self._vrus.append(VRU(
                dx=8.0, dy=2.0,
                speed_x=4.0, speed_y=0.0,
                visible=1.0,
            ))

        elif sc == ScenarioType.OCCLUDED_PED:
            # THE GREEDY TRAP — pedestrian hidden behind parked car
            occ = 1.0 if self.rng.random() < max(occlusion_prob, 0.9) else 0.0
            self._vrus.append(VRU(
                dx=self.JUNCTION_POS, dy=0.0,
                speed_x=0.0, speed_y=1.0,
                visible=1.0 - occ,
                emerged=False,
            ))
            self.s[IDX_OCC] = occ

        elif sc == ScenarioType.JAYWALKER:
            # pedestrian that steps out at a random time
            self._vrus.append(VRU(
                dx=self.JUNCTION_POS - float(self.rng.uniform(0, 8)),
                dy=3.5,                     # starts on footpath
                speed_x=0.0, speed_y=1.4,
                visible=1.0,
            ))

        elif sc == ScenarioType.ADVERSARIAL:
            # red light + occluded pedestrian + cyclist — maximum challenge
            self.s[IDX_TL] = 1.0
            self.s[IDX_ROW] = 0.0
            occ = float(self.rng.random() < 0.85)
            self._vrus.append(VRU(
                dx=self.JUNCTION_POS, dy=0.0,
                speed_x=0.0, speed_y=0.9,
                visible=1.0 - occ,
            ))
            self._vrus.append(VRU(
                dx=6.0, dy=2.0,
                speed_x=3.5, speed_y=0.0,
                visible=1.0,
            ))
            self.s[IDX_OCC] = occ

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray, maneuver: Optional[int] = None):
        if self._done:
            raise RuntimeError("Call reset() before step() after an episode ends.")

        a = np.asarray(action, np.float32).clip(-1, 1)

        # R6 · actuator/communication latency: queue the commanded action and
        # actually apply one issued `action_latency_steps` steps ago, so the
        # policy can't rely on instantaneous actuation (CARLA + a real control
        # stack will never give you that).
        if self.domain_randomization and self.action_latency_steps > 0:
            self._action_queue.append(a)
            if len(self._action_queue) > self.action_latency_steps:
                a = self._action_queue.pop(0)
            else:
                a = np.zeros_like(a)   # no command has "arrived" yet

        steer    = float(np.clip(a[0], -1, 1))
        throttle = float(np.clip(a[1],  0, 1))
        brake    = float(np.clip(a[2],  0, 1))

        # ego physics (R6: per-episode friction/mass scale on the accel response)
        accel = self._accel_scale * (3.0 * throttle - 6.0 * brake)
        new_speed = float(np.clip(self.s[IDX_SPEED] + accel * self.DT, 0.0, self.SPEED_MAX))
        dist = ((self.s[IDX_SPEED] + new_speed) / 2.0) * self.DT

        self.s[IDX_SPEED]    = new_speed
        self.s[IDX_ACCEL]    = accel
        self.s[IDX_EGO_X]   += dist
        self.s[IDX_EGO_Y]   += 1.5 * steer * self.DT
        self.s[IDX_HEAD_ERR] = float(np.clip(steer * 0.1, -0.5, 0.5))
        self.s[IDX_LANE_OFF] = self.s[IDX_EGO_Y]
        self.s[IDX_PROGRESS] = float(np.clip(self.s[IDX_EGO_X] / self.ROUTE_LEN, 0.0, 1.0))
        self.s[IDX_DIST_JCT] = max(0.0, self.JUNCTION_POS - self.s[IDX_EGO_X])

        # Traffic light turns green after 4.5 seconds (15 steps)
        if self.s[IDX_TL] > 0.5 and self.t > 15:
            self.s[IDX_TL] = 0.0

        # VRU dynamics
        collision = False
        near_miss = False
        for vru in self._vrus:
            vru.dx -= dist
            vru.dy += vru.speed_y * self.DT

            # occlusion reveal: when ego is close, the hidden VRU emerges
            if not vru.emerged and vru.dx < 7.0 and vru.visible < 0.5:
                vru.emerged = True
                vru.visible = 1.0

            ttc = vru.ttc(new_speed)
            if vru.dx < vru.width and abs(vru.dy) < 1.5 and new_speed > 0.5:
                collision = True
            elif ttc < 1.5 and new_speed > 2.0 and abs(vru.dy) < 2.0:
                near_miss = True
            if ttc < self.ttc_threshold:
                self.ttc_violations += 1

        if collision:
            self.collisions += 1
        if near_miss:
            self.near_misses += 1

        # compute reward
        reward = self._reward(dist, steer, brake, collision, near_miss)

        self.t += 1
        done = collision or self.t >= self.max_steps or self.s[IDX_PROGRESS] >= 1.0
        self._done = done

        self._update_obs()
        info = self._build_info(collision, near_miss)
        return self.s.copy(), float(reward), bool(done), info

    # ------------------------------------------------------------------ #
    # reward function  (mirrors the paper's multi-term reward)
    # ------------------------------------------------------------------ #
    def _reward(self, dist, steer, brake, collision, near_miss) -> float:
        r_prog     =  dist / 8.0
        r_collision = -20.0 if collision else 0.0
        r_near_miss =  -4.0 if near_miss  else 0.0
        r_ttc = 0.0
        for vru in self._vrus:
            ttc = vru.ttc(self.s[IDX_SPEED])
            if ttc < self.ttc_threshold:
                r_ttc -= 2.0 * (1.0 - ttc / self.ttc_threshold)
        r_comfort  = -0.05 * (abs(steer) + abs(brake))
        r_tl       = -5.0 if (self.s[IDX_TL] > 0.5 and self.s[IDX_SPEED] > 1.0) else 0.0
        r_lane     = -0.2 * abs(self.s[IDX_LANE_OFF])
        return r_prog + r_collision + r_near_miss + r_ttc + r_comfort + r_tl + r_lane

    # ------------------------------------------------------------------ #
    # obs update  (fills the 20-dim state vector from VRU list)
    # ------------------------------------------------------------------ #
    def _update_obs(self):
        # risk summary -- always computed from ground truth (self._vrus), never
        # from the noised observation slots below. The safety floor
        # (enhanced_mandated_action) and info()/_build_info both read min_ttc
        # from this, and a hard safety system relying on noisy range estimates
        # would defeat the point of having one.
        min_ttc = 99.0
        for vru in self._vrus:
            min_ttc = min(min_ttc, vru.ttc(self.s[IDX_SPEED]))
        self.s[IDX_MIN_TTC] = float(min_ttc)

        # per-VRU slots (up to 3) -- this is what the learned policy/world model
        # actually see, so R6 sensor noise/flicker is injected only here.
        dx_slots  = [IDX_VRU0_DX,  IDX_VRU1_DX,  IDX_VRU2_DX]
        vis_slots = [IDX_VRU0_VIS, IDX_VRU1_VIS, IDX_VRU2_VIS]
        for i in range(3):
            if i < len(self._vrus):
                dx = self._vrus[i].dx
                vis = self._vrus[i].visible
                if self.domain_randomization:
                    # R6 · range-sensor jitter (LiDAR/camera tracking noise)
                    dx = dx + float(self.rng.normal(0.0, self.sensor_noise_std))
                    # R6 · occasional dropout: a visible VRU briefly mis-reported
                    # as occluded, simulating sensor flicker/false negatives
                    if vis > 0.5 and self.rng.random() < self.vis_flicker_prob:
                        vis = 0.0
                self.s[dx_slots[i]]  = float(np.clip(dx,  -5, 99))
                self.s[vis_slots[i]] = float(vis)
            else:
                self.s[dx_slots[i]]  = 99.0
                self.s[vis_slots[i]] = 0.0

        # VRU-0 lateral distance (also jittered, same sensor model)
        if self._vrus:
            dy = self._vrus[0].dy
            if self.domain_randomization:
                dy = dy + float(self.rng.normal(0.0, self.sensor_noise_std))
            self.s[IDX_VRU0_DY] = float(dy)

    # ------------------------------------------------------------------ #
    # info  (matches the dict the planner and training loop expect)
    # ------------------------------------------------------------------ #
    def info(self) -> dict:
        return self._build_info(False, False)

    def _build_info(self, collision: bool, near_miss: bool) -> dict:
        min_ttc = float(self.s[IDX_MIN_TTC])
        n_vru   = len(self._vrus)
        density = float(np.clip(n_vru / 3.0 + self.s[IDX_OCC] * 0.3, 0.0, 1.0))
        vru_risk = 0.0
        for vru in self._vrus:
            vru_risk = max(vru_risk, float(np.clip(math.exp(-vru.dx / 4.0), 0.0, 1.0)))

        return dict(
            n_vru          = n_vru,
            min_ttc        = min_ttc,
            ttc_threshold  = self.ttc_threshold,
            occlusion      = float(self.s[IDX_OCC]),
            uncertainty    = float(self.s[IDX_OCC]) * 0.6,
            collisions     = self.collisions,
            near_misses    = self.near_misses,
            ttc_violations = self.ttc_violations,
            progress       = float(self.s[IDX_PROGRESS]),
            vru_risk       = vru_risk,
            density        = density,
            collision      = collision,
            near_miss      = near_miss,
            success        = bool(
                self.s[IDX_PROGRESS] >= 1.0
                and self.collisions == 0
                and self.near_misses == 0
            ),
            scenario       = self._scenario.name,
        )

    # ------------------------------------------------------------------ #
    # debug visualisation (top-down ASCII)
    # ------------------------------------------------------------------ #
    def render_ascii(self, width: int = 60) -> str:
        """Tiny top-down text render for debugging in logs / notebooks."""
        road = [" "] * width
        ego_col = min(int(self.s[IDX_EGO_X] / self.ROUTE_LEN * width), width - 1)
        road[ego_col] = "E"
        for vru in self._vrus:
            col = max(0, min(int((self.s[IDX_EGO_X] + vru.dx) / self.ROUTE_LEN * width), width - 1))
            road[col] = "P" if vru.visible > 0.5 else "?"
        lane = "|" + "".join(road) + "|"
        info = (f"  spd={self.s[IDX_SPEED]:.1f} ttc={self.s[IDX_MIN_TTC]:.1f}"
                f" prog={self.s[IDX_PROGRESS]:.2f} scn={self._scenario.name}")
        return lane + info


# ==============================================================================
# 3.  ego_xy_fn and mandated_action_fn for the enhanced env
#     (pass these to build_agent / train in sdbs_dreamer.py)
# ==============================================================================
def enhanced_ego_xy(state: np.ndarray):
    """Returns (x, y) for conflict-cell diversity (Sec. 3)."""
    return float(state[IDX_EGO_X]), float(state[IDX_EGO_Y])


def enhanced_mandated_action(state: np.ndarray, info: dict) -> Optional[np.ndarray]:
    """
    Hard safety floor (Sec. 6 essential actions).
    Priority order: collision-imminent brake > red light stop > yield ROW.
    Returns np.ndarray([steer, throttle, brake]) or None.
    """
    FULL_BRAKE = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    SOFT_BRAKE = np.array([0.0, 0.0, 0.4], dtype=np.float32)

    min_ttc = float(info.get("min_ttc", 99.0))

    # Emergency: imminent collision
    if min_ttc < 1.0:
        return FULL_BRAKE

    # Red light or stop sign — stop if still moving
    if float(state[IDX_TL]) > 0.5 and float(state[IDX_SPEED]) > 0.5:
        return FULL_BRAKE if float(state[IDX_SPEED]) > 4.0 else SOFT_BRAKE

    if float(state[IDX_STOP]) > 0.5 and float(state[IDX_SPEED]) > 0.5:
        return SOFT_BRAKE

    # Yield when ROW is not ours and a VRU is close
    if float(state[IDX_ROW]) < 0.5 and min_ttc < 3.0:
        return SOFT_BRAKE

    return None


# ==============================================================================
# 4.  Scenario bank helper
#     Generates the initial PER scenario bank across all three tiers.
# ==============================================================================
def make_scenario_bank() -> list[dict]:
    """
    Returns a list of scenario dicts compatible with env.reset(**scn).
    The PrioritizedScenarioReplay in sdbs_core.py stores and replays these.
    """
    bank: list[dict] = []

    # Tier 0: simple empty roads
    for _ in range(8):
        bank.append(dict(tier=0, n_vru=0, occlusion_prob=0.0, adversarial=False))

    # Tier 1: visible VRUs — predictable interactions
    for n in [1, 2]:
        bank.append(dict(tier=1, n_vru=n, occlusion_prob=0.0,  adversarial=False))
        bank.append(dict(tier=1, n_vru=n, occlusion_prob=0.1,  adversarial=False))

    # Tier 2: hard greedy-trap scenarios
    for occ in [0.7, 0.9, 1.0]:
        bank.append(dict(tier=2, n_vru=1, occlusion_prob=occ, adversarial=False))
        bank.append(dict(tier=2, n_vru=2, occlusion_prob=occ, adversarial=True))
    bank.append(dict(tier=2, n_vru=3, occlusion_prob=1.0, adversarial=True))

    return bank


# ==============================================================================
# 5.  Quick smoke test
# ==============================================================================
def run_enhanced_smoke_test():
    print("=" * 60)
    print("EnhancedMockEnv — smoke test")
    print("=" * 60)

    env = EnhancedMockEnv(seed=42)

    scenarios_to_test = [
        dict(tier=0, n_vru=0, occlusion_prob=0.0, adversarial=False),
        dict(tier=1, n_vru=1, occlusion_prob=0.0, adversarial=False),
        dict(tier=2, n_vru=1, occlusion_prob=1.0, adversarial=False),
        dict(tier=2, n_vru=2, occlusion_prob=0.9, adversarial=True),
    ]

    for scn in scenarios_to_test:
        obs = env.reset(**scn)
        assert obs.shape == (ENHANCED_STATE_DIM,), f"Wrong obs shape: {obs.shape}"
        print(f"\nScenario tier={scn['tier']} occ={scn['occlusion_prob']:.1f} "
              f"adv={scn['adversarial']}  →  {env._scenario.name}")
        print(env.render_ascii())

        total_r = 0.0
        for step_i in range(15):
            # mild throttle, no steer
            action = np.array([0.0, 0.4, 0.0], dtype=np.float32)
            obs, r, done, info = env.step(action)
            total_r += r
            if step_i % 5 == 0:
                print(f"  step {step_i:2d}: {env.render_ascii(40)}")
                print(f"           ttc={info['min_ttc']:.2f}  "
                      f"near_miss={info['near_miss']}  col={info['collision']}")
            if done:
                break

        print(f"  Episode ended: success={info['success']}  "
              f"total_reward={total_r:.2f}")

    # test mandated_action_fn
    print("\n--- mandated_action_fn ---")
    env.reset(tier=2, n_vru=1, occlusion_prob=1.0, adversarial=False)
    fake_info = dict(min_ttc=0.5, ttc_threshold=3.0, n_vru=1)
    action = enhanced_mandated_action(env.s, fake_info)
    assert action is not None and action[2] == 1.0, "Emergency brake not triggered!"
    print("  Emergency brake at TTC<1.0: OK")
    fake_info["min_ttc"] = 5.0
    env.s[IDX_TL] = 1.0
    env.s[IDX_SPEED] = 8.0
    action = enhanced_mandated_action(env.s, fake_info)
    assert action is not None and action[2] > 0.0, "Red light brake not triggered!"
    print("  Red light brake: OK")

    # scenario bank
    bank = make_scenario_bank()
    print(f"\nScenario bank: {len(bank)} scenarios across 3 tiers")
    for t in [0, 1, 2]:
        count = sum(1 for s in bank if s["tier"] == t)
        print(f"  Tier {t}: {count} scenarios")

    print("\n✓  EnhancedMockEnv smoke test passed.\n")
    print("Next step: pass this env to sdbs_dreamer.train() like this:\n")
    print("  from enhanced_mock_env import EnhancedMockEnv, enhanced_ego_xy,")
    print("                                enhanced_mandated_action, make_scenario_bank")
    print("  from sdbs_dreamer import train")
    print("  from sdbs_core import TrainConfig, PrioritizedScenarioReplay\n")
    print("  env = EnhancedMockEnv(seed=0)")
    print("  train(env, TrainConfig(), n_iterations=50,")
    print("        ego_xy_fn=enhanced_ego_xy,")
    print("        mandated_action_fn=enhanced_mandated_action,")
    print("        use_ensemble=True, device='cpu')")


if __name__ == "__main__":
    run_enhanced_smoke_test()