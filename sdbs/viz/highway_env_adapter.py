"""
highway_env_adapter.py
================================================================================
Drop-in replacement pour EnhancedMockEnv / CarlaEnvAdapter, basé sur la
bibliothèque `highway-env` (Gymnasium). Contrairement au mode replay de
visualize_pg.py, ici l'agent PILOTE RÉELLEMENT le véhicule à chaque step --
c'est un vrai test en ligne du planner S-DBS, pas un rejeu de frames
pré-calculées.

Pourquoi highway-env plutôt que pygame seul :
  - L'environnement gère lui-même route, trafic, collisions, physique du
    véhicule (kinematic bicycle model) -- vous n'avez plus à le coder à la main
    comme dans EnhancedMockEnv.
  - Rendu optionnel en rgb_array (pas besoin de display système qui plantait
    avec pygame direct) : SDL_VIDEODRIVER=dummy + render_mode="rgb_array".
  - Plusieurs scénarios prêts à l'emploi : highway-v0, intersection-v0,
    roundabout-v0, merge-v0, racetrack-v0, parking-v0 -- bons proxies pour vos
    tiers de curriculum (route simple / intersection / trafic dense).

Installation :
    pip install highway-env gymnasium

Usage :
    from highway_env_adapter import HighwayEnvAdapter, highway_ego_xy, highway_mandated_action
    env = HighwayEnvAdapter(scenario="highway-v0", seed=0)
    obs = env.reset(tier=2, n_vru=0, occlusion_prob=0.0, adversarial=False)
    out = planner.plan(obs, env.info())
    obs, r, done, info = env.step(out["action"], out["maneuver"])

Fix R1 (ContinuousAction + discrete _rewards bug) :
    Plusieurs envs highway-env (merge-v0, roundabout-v0, …) implémentent
    _rewards(action) avec `action in [0, 2]`, ce qui suppose une action
    DISCRÈTE (entier). Quand on passe ContinuousAction, highway-env appelle
    _info(obs, action=action_space.sample()) au reset() -- action est alors
    un array numpy, et `array in [0, 2]` lève ValueError.
    Solution : patch universel sur AbstractEnv._info() qui normalise l'action
    en int scalaire avant de l'envoyer à _rewards(), couvrant tous les envs
    concernés en une seule interception.
================================================================================
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

# IMPORTANT : ne PAS positionner SDL_VIDEODRIVER=dummy ici. highway-env
# desactive completement son rendu (EnvViewer.enabled=False -> frames noires)
# des qu'il detecte ce flag, peu importe le mode demande. Le vrai mecanisme
# headless de highway-env passe par config["offscreen_rendering"]=True
# (pygame.Surface en memoire, sans display reel ni driver dummy requis) --
# c'est ce que fait HighwayEnvAdapter._make_config() ci-dessous.

import gymnasium as gym
import highway_env  # noqa: F401  -- l'import enregistre les envs Gymnasium

# ---------------------------------------------------------------------------
# Fix R1 – patch DÉFINITIF du bug "action in [0, 2]" dans highway-env.
#
# Racine du problème : merge_env, roundabout_env, intersection_env (et
# potentiellement d'autres) définissent _rewards(action) en supposant une
# action DISCRÈTE (int). Ces classes redéfinissent aussi _reward (singulier),
# donc patcher AbstractEnv._reward ne suffit pas -- Python résout _reward via
# MRO et appelle directement la version de la sous-classe.
#
# Solution définitive : on itère sur TOUS les sous-modules highway_env.envs,
# on trouve chaque classe qui hérite d'AbstractEnv et qui possède sa propre
# méthode _rewards, et on wrappe _rewards sur place pour normaliser l'action
# en int si c'est un array. Un seul bloc, zéro maintenance future.
# ---------------------------------------------------------------------------
def _to_discrete(action) -> int:
    """np.ndarray → int scalaire pour satisfaire `action in [0, 2]`."""
    if isinstance(action, np.ndarray):
        return int(action.flat[0]) if action.size == 1 else int(np.argmax(np.abs(action)))
    if isinstance(action, (float, np.floating)):
        return int(round(action))
    return int(action) if action is not None else 0

def _make_rewards_wrapper(original_rewards):
    def _wrapped(self, action):
        if isinstance(action, np.ndarray):
            action = _to_discrete(action)
        return original_rewards(self, action)
    return _wrapped

try:
    import inspect
    import importlib
    import pkgutil
    import highway_env.envs as _hwe

    from highway_env.envs.common.abstract import AbstractEnv as _AbstractEnv

    # Charger tous les sous-modules de highway_env.envs pour que les classes
    # soient définies et visibles avant qu'on les inspecte.
    for _info_mod in pkgutil.walk_packages(
        path=_hwe.__path__, prefix=_hwe.__name__ + "."
    ):
        try:
            importlib.import_module(_info_mod.name)
        except Exception:
            pass

    # Parcourir toutes les sous-classes connues d'AbstractEnv et wrapper
    # _rewards si la classe la redéfinit elle-même (pas héritée).
    _patched_classes: list[str] = []
    for _cls in _AbstractEnv.__subclasses__():
        if "_rewards" in _cls.__dict__:   # redéfinie dans cette classe précisément
            _cls._rewards = _make_rewards_wrapper(_cls._rewards)
            _patched_classes.append(_cls.__name__)

    if _patched_classes:
        print(f"[highway_env_adapter] Fix R1 appliqué sur : {', '.join(_patched_classes)}")

except Exception as _e:
    print(f"[highway_env_adapter] Fix R1 partiel ({_e}) -- certains envs peuvent crasher")


# ------------------------------------------------------------------------
# Layout d'observation : on aplatit les N véhicules les plus proches
# (presence, x, y, vx, vy) -- même esprit que IDX_* dans enhanced_mock_env.py
# ------------------------------------------------------------------------
N_VEHICLES_OBS = 6          # ego + 5 voisins -> 6*5=30.
                             # IMPORTANT : doit matcher le state_dim utilise au
                             # MOMENT DE L'ENTRAINEMENT. Si vous chargez un
                             # checkpoint entraine sur EnhancedMockEnv (20 dims),
                             # mettez N_VEHICLES_OBS=4 a la place -- mais le
                             # comportement ne sera pas significatif (layout de
                             # features totalement different). Le mieux est de
                             # reentrainer directement sur highway-env (voir
                             # run_training.py --mode highway).
FEATURES = ["presence", "x", "y", "vx", "vy"]
HIGHWAY_STATE_DIM = N_VEHICLES_OBS * len(FEATURES)   # 6*5=30
HIGHWAY_ACTION_DIM = 2       # [steer-ish (lateral), accel (longitudinal)] dans [-1,1]

# Scénarios disponibles dans highway-env, mappés sur vos tiers de curriculum.
# merge-v0 est retiré du mapping automatique par tier (bug _rewards + action
# discrète) -- il reste utilisable via HighwayEnvAdapter(scenario="merge-v0")
# si le monkey-patch ci-dessus est actif.
SCENARIO_BY_TIER = {
    0: "highway-v0",          # route libre, peu de trafic -- tier 0
    1: "roundabout-v0",       # rond-point -- interactions prévisibles -- tier 1
                               # (remplace merge-v0 qui attend des actions discrètes)
    2: "intersection-v0",     # intersection avec priorité/feux -- tier 2 (greedy trap)
}


def highway_ego_xy(state: np.ndarray) -> tuple:
    """Position (x, y) de l'ego pour les cellules de conflit S-DBS.
    L'ego est toujours la première ligne du bloc observation."""
    return float(state[1]), float(state[2])   # x, y de la ligne ego (presence à l'indice 0)


def highway_mandated_action(state: np.ndarray, info: dict) -> Optional[np.ndarray]:
    """Plancher de sécurité (Sec. 6) : freinage d'urgence si collision déjà
    détectée par l'env, ou si un véhicule est anormalement proche devant."""
    if info.get("crashed", False):
        return np.array([0.0, -1.0], dtype=np.float32)   # [steer=0, accel=-1 -> frein]
    min_ttc = float(info.get("min_ttc", 99.0))
    if min_ttc < 1.0:
        return np.array([0.0, -1.0], dtype=np.float32)
    return None


class HighwayEnvAdapter:
    """Wrap un environnement highway-env Gymnasium pour exposer EXACTEMENT
    l'interface attendue par SDBSPlanner / train() :
        reset(tier, n_vru, occlusion_prob, adversarial) -> obs (np.ndarray)
        step(action, maneuver=None)                     -> (obs, reward, done, info)
        info()                                           -> dict
        state_dim, action_dim
    """

    state_dim = HIGHWAY_STATE_DIM
    action_dim = HIGHWAY_ACTION_DIM
    n_maneuvers = 6   # doit matcher len(Maneuver) dans sdbs_core -- voir step()

    def __init__(self, scenario: Optional[str] = None, seed: int = 0,
                 duration_s: int = 40, render: bool = False):
        """scenario : force un scénario highway-env précis (ex: "highway-v0").
        Si None, reset() choisit le scénario selon `tier` via SCENARIO_BY_TIER.
        render : True -> render_mode="human" (nécessite un display fonctionnel) ;
                 False (défaut) -> render_mode="rgb_array" (headless, sûr)."""
        self.seed = seed
        self.duration_s = duration_s
        self.render_mode = "human" if render else "rgb_array"
        self._forced_scenario = scenario
        self.env: Optional[gym.Env] = None
        self.collisions = 0
        self.near_misses = 0
        self.ttc_violations = 0
        self._last_obs_raw = None
        self._last_info: dict = {}
        self._t = 0
        self.max_steps = duration_s * 5   # approx, ajusté après reset selon policy_frequency

    # ------------------------------------------------------------------
    def _make_config(self, tier: int, n_vru: int, adversarial: bool) -> dict:
        density = {0: 10, 1: 15, 2: 25}.get(tier, 15) + (10 if adversarial else 0)
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": N_VEHICLES_OBS - 1,
                "features": FEATURES,
                "absolute": False,
                "normalize": True,
                "see_behind": True,
            },
            "action": {
                # Fix R1 : on force toujours ContinuousAction ici pour que
                # action_space.sample() retourne un array float32 cohérent avec
                # ce que step() envoie. merge-v0 est géré par le monkey-patch.
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
            "lanes_count": 4 if tier >= 2 else 3,
            "vehicles_count": int(density),
            "duration": self.duration_s,
            "collision_reward": -8.0,
            "right_lane_reward": 0.05,
            "high_speed_reward": 0.3,
            "reward_speed_range": [18, 28],
            "normalize_reward": False,
            "offscreen_rendering": self.render_mode != "human",
        }

    # ------------------------------------------------------------------
    def reset(self, tier: int = 0, n_vru: int = 0,
              occlusion_prob: float = 0.0, adversarial: bool = False) -> np.ndarray:
        scenario = self._forced_scenario or SCENARIO_BY_TIER.get(tier, "highway-v0")
        cfg = self._make_config(tier, n_vru, adversarial)

        if self.env is not None:
            self.env.close()
            self.env = None

        # Fix R1 – gym.make() appelle env.__init__() qui lui-même appelle
        # self.reset() AVANT que notre config ne soit prise en compte pour
        # action_space, d'où le crash dans _rewards(action_space.sample()).
        # On passe la config au constructeur via le paramètre `config` de
        # highway-env (supporté depuis highway-env 1.x) pour qu'elle soit
        # appliquée avant le premier reset interne.
        self.env = gym.make(scenario, render_mode=self.render_mode, config=cfg)

        obs_raw, info = self.env.reset(seed=self.seed)
        self.seed += 1   # nouvelle seed à chaque épisode pour varier le trafic

        self.collisions = 0
        self.near_misses = 0
        self.ttc_violations = 0
        self._t = 0
        self._last_obs_raw = obs_raw
        self._last_info = self._build_info(info, reward=0.0)
        return self._flatten_obs(obs_raw)

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray, maneuver: Optional[int] = None):
        a = np.tanh(np.asarray(action, dtype=np.float32))
        a = np.clip(a, -1.0, 1.0)
        # highway-env ContinuousAction attend [acceleration, steering]
        # a[0] est le steer (selon notre HIGHWAY_ACTION_DIM), a[1] est l'accel
        gym_action = np.array([a[1], a[0]], dtype=np.float32)
        obs_raw, reward, terminated, truncated, info = self.env.step(gym_action)
        done = bool(terminated or truncated)
        self._t += 1

        crashed = bool(info.get("crashed", False))
        if crashed:
            self.collisions += 1

        min_ttc = self._estimate_min_ttc(obs_raw)
        near_miss = (not crashed) and min_ttc < 1.5
        if near_miss:
            self.near_misses += 1
        if min_ttc < 2.0:
            self.ttc_violations += 1

        self._last_obs_raw = obs_raw
        self._last_info = self._build_info(info, reward, crashed=crashed,
                                           near_miss=near_miss, min_ttc=min_ttc,
                                           done=done)
        return self._flatten_obs(obs_raw), float(reward), done, self._last_info

    def info(self) -> dict:
        return dict(self._last_info)

    # ------------------------------------------------------------------
    def _flatten_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        out = np.zeros((N_VEHICLES_OBS, len(FEATURES)), dtype=np.float32)
        n = min(obs_raw.shape[0], N_VEHICLES_OBS)
        out[:n] = obs_raw[:n]
        return out.reshape(-1)

    def _estimate_min_ttc(self, obs_raw: np.ndarray) -> float:
        """TTC approx : pour chaque véhicule présent devant l'ego (x>0, |y|
        faible), distance / vitesse de rapprochement. obs_raw est en repère
        relatif à l'ego (absolute=False), donc la ligne 0 = ego (x=y=vx=vy=0
        par convention highway-env), les lignes suivantes = voisins relatifs."""
        min_ttc = 99.0
        for row in obs_raw[1:]:
            presence, x, y, vx, vy = row[:5]
            if presence < 0.5:
                continue
            if x <= 0 or abs(y) > 0.5:   # derrière, ou pas dans la voie ego
                continue
            closing = max(-vx, 0.05)     # vx relatif négatif = se rapproche
            min_ttc = min(min_ttc, float(x) / closing if closing > 0 else 99.0)
        return float(np.clip(min_ttc, 0.0, 99.0))

    def _build_info(self, gym_info: dict, reward: float, crashed: bool = False,
                    near_miss: bool = False, min_ttc: float = 99.0,
                    done: bool = False) -> dict:
        progress = min(1.0, self._t / max(1, self.max_steps))
        return dict(
            n_vru=0,                       # highway-env n'a pas de piétons par défaut
            min_ttc=min_ttc,
            ttc_threshold=3.0,
            occlusion=0.0,
            uncertainty=0.0,
            collisions=self.collisions,
            near_misses=self.near_misses,
            ttc_violations=self.ttc_violations,
            collision=crashed,
            near_miss=near_miss,
            progress=progress,
            vru_risk=0.0,
            density=0.0,
            reward=reward,
            speed=float(gym_info.get("speed", 0.0)),
            crashed=bool(gym_info.get("crashed", crashed)),
            success=bool(done and not crashed and self._t >= self.max_steps * 0.9),
        )

    # ------------------------------------------------------------------
    def render_frame(self) -> Optional[np.ndarray]:
        """Retourne la frame courante en rgb_array (np.ndarray HxWx3), utile
        pour sauvegarder un GIF/vidéo sans display système."""
        if self.env is None:
            return None
        return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None