"""
replay_data.py
================================================================================
Collecte les donnees d'un episode (greedy vs S-DBS) en utilisant le VRAI
policy + world model + planner charges depuis un checkpoint entraine.

Separe deliberement de la logique de rendu (visualize_pygame.py) : ce module
ne sait rien de pygame, il produit juste une liste de Frame + un summary dict.
Ca permet de tester la collecte independamment de l'affichage, et de reutiliser
les memes donnees pour un futur export web / logging / replay differe.
================================================================================
"""
from __future__ import annotations
from torch import device

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sdbs.core.sdbs_core import (
    SDBSPlanner, PlannerConfig, BudgetConfig, DEFAULT_MANEUVERS, Maneuver,
)
from sdbs.envs.enhanced_mock_env import (
    EnhancedMockEnv, enhanced_ego_xy, enhanced_mandated_action,
    IDX_EGO_X, IDX_EGO_Y, IDX_SPEED, IDX_VRU0_DX, IDX_VRU0_VIS, IDX_TL, IDX_OCC,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


MANEUVER_NAMES = {int(m): m.name for m in Maneuver}


# ==============================================================================
# 1.  Frame — un instant t de l'episode, suffisant pour le rendu
# ==============================================================================
@dataclass
class Frame:
    ego_x: float
    ego_y: float
    speed: float
    ped_dx: float
    ped_visible: bool
    light_red: bool
    min_ttc: float
    reward: float
    cum_reward: float
    collision: bool
    near_miss: bool
    maneuver: int
    meta: dict = field(default_factory=dict)


@dataclass
class EpisodeResult:
    frames: list[Frame]
    success: bool
    collisions: int
    near_misses: int
    total_reward: float
    steps: int
    scenario: str
    mode: str          # "greedy" ou "sdbs"


# ==============================================================================
# 2.  Chargement de l'agent entraine
# ==============================================================================
def load_trained_agent(
    checkpoint_path,
    device="cpu",
    env=None,
    ego_xy_fn=None,
    mandated_action_fn=None,
):
    """Construit un agent (policy+WM+planner) et y charge un checkpoint produit
    par sdbs_dreamer.train() / collect_rollout(). Le checkpoint attendu est un
    dict avec des cles 'policy' et 'wm' (state_dicts), 'ensemble' optionnel.

    Si le fichier ne contient qu'un state_dict de WorldModel seul (sortie de
    pretrain_world_model_offline.py via --wm_checkpoint), seul le WM est charge
    et la policy reste a son init -- utile pour valider le WM avant le RL complet.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour charger un checkpoint entraine.")

    from sdbs.model.sdbs_dreamer import build_agent, TrainConfig

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and ("wm" in raw or "policy" in raw):
        wm_state = raw.get("wm")
        policy_state = raw.get("policy")
        ensemble_state = raw.get("ensemble")
    else:
        wm_state = raw if not hasattr(raw, "state_dict") else raw.state_dict()
        policy_state = None
        ensemble_state = None

    dummy_env = EnhancedMockEnv(seed=0)
    cfg = TrainConfig()
    if env is None:
        env = EnhancedMockEnv(seed=0)
        ego_xy_fn = enhanced_ego_xy
        mandated_action_fn = enhanced_mandated_action

    agent = build_agent(
        env,
        cfg,
        use_ensemble=(ensemble_state is not None),
        ego_xy_fn=ego_xy_fn,
        mandated_action_fn=mandated_action_fn,
        device=device,
    )

    if policy_state is not None:
        agent["policy"].load_state_dict(policy_state)
        print(f"[replay_data] policy chargee depuis {checkpoint_path}")
    else:
        print("[replay_data] aucun state_dict policy trouve -- poids aleatoires "
              "(le WM seul a ete charge si present).")

    if wm_state is not None:
        agent["wm"].load_state_dict(wm_state)
        print(f"[replay_data] world model charge depuis {checkpoint_path}")

    if ensemble_state is not None and agent["ensemble"] is not None:
        agent["ensemble"].load_state_dict(ensemble_state)
        print(f"[replay_data] ensemble charge depuis {checkpoint_path}")

    agent["policy"].eval()
    agent["wm"].eval()
    if agent["ensemble"] is not None:
        agent["ensemble"].eval()

    return agent


# ==============================================================================
# 3.  Collecte d'un episode (greedy ou S-DBS), modele entraine
# ==============================================================================
def run_episode(agent: dict, seed: int, use_sdbs: bool,
                tier: int = 2, n_vru: int = 1, occlusion_prob: float = 1.0,
                adversarial: bool = False) -> EpisodeResult:
    """Joue un episode complet avec le planner de `agent` (greedy_one_step ou
    plan() selon use_sdbs), sur EnhancedMockEnv, et enregistre chaque pas dans
    une liste de Frame consommable directement par le renderer pygame."""
    env = EnhancedMockEnv(seed=seed)
    planner: SDBSPlanner = agent["planner"]

    obs = env.reset(tier=tier, n_vru=n_vru, occlusion_prob=occlusion_prob,
                    adversarial=adversarial)
    frames: list[Frame] = []
    cum_r = 0.0
    step_info: dict = {}
    stuck_steps = 0

    for _ in range(env.max_steps):
        info = env.info()
        out = planner.plan(obs, info) if use_sdbs else planner.greedy_one_step(obs)
        action = out["action"]
        maneuver = out.get("maneuver", -1)
        meta = out.get("meta", {})

        # anti-deadlock : si bloqué trop longtemps malgré un planner actif,
        # on force un proceed doux pour casser la boucle freinage<->ttc figé
        if float(obs[IDX_SPEED]) < 0.3:
            stuck_steps += 1
        else:
            stuck_steps = 0
        if stuck_steps > 6:
            action = np.array([0.0, 0.4, 0.0], dtype=np.float32)
            stuck_steps = 0

        next_obs, r, done, step_info = env.step(action, maneuver)
        print(action, step_info.get("min_ttc"), obs[IDX_SPEED])
        cum_r += r

        if len(env._vrus) > 0:
            ped_dx = env._vrus[0].dx
        else:
            ped_dx = 99.0

        frames.append(Frame(
            ego_x=float(obs[IDX_EGO_X]), ego_y=float(obs[IDX_EGO_Y]),
            speed=float(obs[IDX_SPEED]),
            ped_dx=float(ped_dx),
            ped_visible=float(obs[IDX_VRU0_VIS]) > 0.5,
            light_red=float(obs[IDX_TL]) > 0.5,
            min_ttc=float(step_info.get("min_ttc", 99.0)),
            reward=float(r), cum_reward=cum_r,
            collision=bool(step_info.get("collision", False)),
            near_miss=bool(step_info.get("near_miss", False)),
            maneuver=int(maneuver), meta=meta,
        ))
        obs = next_obs
        if done:
            break

    return EpisodeResult(
        frames=frames,
        success=bool(step_info.get("success", False)),
        collisions=env.collisions,
        near_misses=env.near_misses,
        total_reward=cum_r,
        steps=len(frames),
        scenario=f"tier={tier} n_vru={n_vru} occlusion={occlusion_prob:.1f}"
                 + (" adversarial" if adversarial else ""),
        mode="sdbs" if use_sdbs else "greedy",
    )


def run_comparison(agent: dict, seed: int, **scenario_kwargs
                   ) -> tuple[EpisodeResult, EpisodeResult]:
    """Joue le MEME seed/scenario en greedy puis en S-DBS, pour comparaison
    cote-a-cote directe (meme depart, seule la strategie de decision change)."""
    greedy = run_episode(agent, seed, use_sdbs=False, **scenario_kwargs)
    sdbs = run_episode(agent, seed, use_sdbs=True, **scenario_kwargs)
    return greedy, sdbs