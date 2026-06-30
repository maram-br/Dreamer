"""
run_training.py
================================================================================
Entry point for the S-DBS Dreamer-PPO project.

Deux modes :
  1.  Mock (dÃ©faut) â€” EnhancedMockEnv, tout tourne sur CPU sans CARLA.
  2.  CARLA â€” remplace EnhancedMockEnv par CarlaEnvAdapter.

Usage:
  # Mode mock (dÃ©veloppement local)
  python run_training.py --mode mock --iters 100 --device cpu

  # Charger un checkpoint de world model prÃ©-entraÃ®nÃ©
  python run_training.py --mode mock --wm_checkpoint ton_checkpoint.pt

  # Smoke test rapide (3 itÃ©rations)
  python run_training.py --smoke

  # Plus tard, sur le serveur CARLA
  python run_training.py --mode carla --device cuda --iters 1000
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# VÃ©rifier que sdbs_core et sdbs_dreamer sont accessibles
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sdbs.envs.enhanced_mock_env import (
    EnhancedMockEnv,
    enhanced_ego_xy,
    enhanced_mandated_action,
    make_scenario_bank,
    ENHANCED_STATE_DIM,
    ENHANCED_ACTION_DIM,
)
from sdbs.core.sdbs_core import (
    TrainConfig, PlannerConfig, BudgetConfig, PERConfig, CurriculumConfig,
    PrioritizedScenarioReplay, DEFAULT_MANEUVERS,
    run_core_smoke_test,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1.  Chargement de checkpoint
# ==============================================================================
def load_world_model_checkpoint(path: str, device: str = "cpu") -> dict:
    """
    Charge un checkpoint et retourne un dict de state_dicts bruts
    {"wm": ..., "policy": ..., "ensemble": ...} (clÃ©s absentes si non prÃ©sentes
    dans le fichier).

    R2 fix: cette fonction ne construit plus elle-mÃªme de WorldModel/Ensemble
    (l'ancienne version le faisait puis ces objets n'Ã©taient jamais rÃ©utilisÃ©s
    par train(), qui repartait toujours de zÃ©ro). Elle se limite maintenant Ã 
    extraire les state_dicts ; c'est train()/build_agent() (sdbs_dreamer.py)
    qui construisent les rÃ©seaux et y chargent ces poids, donc le checkpoint
    chargÃ© ici est rÃ©ellement utilisÃ© pendant l'entraÃ®nement.

    Le fichier peut Ãªtre :
      - Un state_dict de WorldModel seul   : torch.save(model.state_dict(), path)
      - Un dict avec 'model'               : torch.save({'model': sd, ...}, path)
      - Un checkpoint complet de train()   : contient 'policy', 'wm', 'ensemble'
      - Un WorldModel complet (objet)      : torch.save(model, path)  (dÃ©conseillÃ©)
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour charger un checkpoint.")

    print(f"[checkpoint] Chargement depuis {path} â€¦")
    # R7 Â· weights_only=False: this checkpoint is your own training output
    # (trusted), and PyTorch >= 2.6 defaults to weights_only=True which
    # rejects plain state_dicts saved alongside numpy/curriculum metadata.
    raw = torch.load(path, map_location=device, weights_only=False)

    out: dict = {}
    if isinstance(raw, dict) and ("wm" in raw or "policy" in raw):
        # checkpoint complet produit par train() (save_every / resume_path)
        if raw.get("wm") is not None:
            out["wm"] = raw["wm"]
        if raw.get("policy") is not None:
            out["policy"] = raw["policy"]
        if raw.get("ensemble") is not None:
            out["ensemble"] = raw["ensemble"]
        print(f"[checkpoint] Checkpoint complet dÃ©tectÃ© (iteration="
              f"{raw.get('iteration', '?')}, tier={raw.get('tier', '?')})")
    elif isinstance(raw, dict) and "model" in raw:
        out["wm"] = raw["model"]
        print(f"[checkpoint] TrouvÃ© 'model' dans le dict, step={raw.get('step', '?')}")
    elif isinstance(raw, dict) and all(isinstance(k, str) for k in raw):
        # c'est directement un state_dict de WorldModel
        out["wm"] = raw
    else:
        # objet WorldModel complet
        out["wm"] = raw.state_dict()

    for k, sd in out.items():
        print(f"[checkpoint] '{k}' : {len(sd)} tenseurs trouvÃ©s")
    return out


# ==============================================================================
# 2.  Configuration recommandÃ©e pour EnhancedMockEnv
# ==============================================================================
def make_config(smoke: bool = False) -> TrainConfig:
    cfg = TrainConfig()

    # Planner : beam modÃ©rÃ© pour le mock (rapide sur CPU)
    cfg.planner = PlannerConfig(
        gamma=0.97,
        horizon=4,
        beam_width=6,
        n_groups=3,
        n_expand=4,
        eta=0.4,        # serendipity
        lam=0.5,        # diversity
        mu=0.3,         # dÃ©fensif
    )
    # Budget : rÃ©duit pour le mock (pas de GPU)
    cfg.budget = BudgetConfig(
        max_rollouts=120,
        beam_easy=2, beam_hard=6,
        groups_easy=1, groups_hard=3,
        horizon_easy=2, horizon_hard=5,
    )
    cfg.per = PERConfig(capacity=512 if smoke else 2048)
    cfg.curriculum = CurriculumConfig(unlock_threshold=0.80, ma_window=30)

    if smoke:
        cfg.rollout_size = 64
        cfg.update_epochs = 2
        cfg.minibatch = 16
    else:
        cfg.rollout_size = 512
        cfg.update_epochs = 6
        cfg.minibatch = 64

    return cfg


# ==============================================================================
# 3.  Initialisation du bank PER
# ==============================================================================
def init_per(cfg: TrainConfig) -> PrioritizedScenarioReplay:
    """Remplit le PER avec le scenario bank initial."""
    per = PrioritizedScenarioReplay(cfg.per)
    bank = make_scenario_bank()
    for scn in bank:
        per.add(scn)
    print(f"[PER] Bank initialisÃ© : {len(bank)} scÃ©narios")
    return per


# ==============================================================================
# 4.  Training loop (wrapper autour de sdbs_dreamer.train)
# ==============================================================================
def run_mock_training(args):
    if not HAS_TORCH:
        print("PyTorch non disponible â€” smoke test NumPy seulement.")
        run_core_smoke_test()
        return

    from sdbs.model.sdbs_dreamer import train

    print("\n" + "=" * 60)
    print("S-DBS Dreamer-PPO â€” Mode Mock (EnhancedMockEnv)")
    print("=" * 60)

    env = EnhancedMockEnv(
        seed=args.seed,
        domain_randomization=args.domain_randomization,
    )
    cfg = make_config(smoke=args.smoke)

    print(f"Env       : EnhancedMockEnv  state_dim={env.state_dim}  "
          f"action_dim={env.action_dim}")
    print(f"Device    : {args.device}")
    print(f"ItÃ©rations: {args.iters}")
    print(f"Rollout   : {cfg.rollout_size} steps")
    if args.domain_randomization:
        print("Domain randomization : ON (bruit capteur + latence + friction)")

    # R2 Â· Charger un checkpoint WorldModel prÃ©-entraÃ®nÃ© et l'injecter
    # rÃ©ellement dans l'agent via build_agent (au lieu de le charger puis de
    # ne jamais s'en servir, comme dans la version prÃ©cÃ©dente de ce script).
    wm_state = policy_state = ensemble_state = None
    if args.wm_checkpoint:
        state_dicts = load_world_model_checkpoint(args.wm_checkpoint, args.device)
        wm_state = state_dicts.get("wm")
        policy_state = state_dicts.get("policy")
        ensemble_state = state_dicts.get("ensemble")
        print("[INFO] Poids chargÃ©s et transmis Ã  build_agent() : "
              f"wm={'oui' if wm_state else 'non'} "
              f"policy={'oui' if policy_state else 'non'} "
              f"ensemble={'oui' if ensemble_state else 'non'}")

    t0 = time.time()
    train(
        env          = env,
        cfg          = cfg,
        n_iterations = args.iters,
        rollout_size = cfg.rollout_size,
        use_ensemble = not args.no_ensemble,
        ego_xy_fn    = enhanced_ego_xy,
        mandated_action_fn = enhanced_mandated_action,
        device       = args.device,
        seed         = args.seed,
        verbose      = True,
        save_every   = args.save_every,
        save_path    = args.save_path,
        wm_state       = wm_state,
        policy_state   = policy_state,
        ensemble_state = ensemble_state,
        resume_path    = args.resume,
        eval_every     = args.eval_every,
        use_traffic_predictor = args.traffic_predictor,
    )
    elapsed = time.time() - t0
    print(f"\nOK Training termine en {elapsed:.1f}s")
    print("Pour passer sur CARLA : remplace EnhancedMockEnv par CarlaEnvAdapter")
    print("et relance avec --mode carla --device cuda")
def run_highway_training(args):
    if not HAS_TORCH:
        print("PyTorch requis pour le mode highway.")
        return
 
    from sdbs.model.sdbs_dreamer import train
    from sdbs.viz.highway_env_adapter import HighwayEnvAdapter, highway_ego_xy, highway_mandated_action
 
    print("\n" + "=" * 60)
    print("S-DBS Dreamer-PPO — Mode highway-env")
    print("=" * 60)
 
    env = HighwayEnvAdapter(seed=args.seed)
    cfg = make_config(smoke=args.smoke)
 
    print(f"Env       : HighwayEnvAdapter  state_dim={env.state_dim}  "
          f"action_dim={env.action_dim}")
    print(f"Device    : {args.device}")
    print(f"Iterations: {args.iters}")
 
    wm_state = policy_state = ensemble_state = None
    if args.wm_checkpoint:
        state_dicts = load_world_model_checkpoint(args.wm_checkpoint, args.device)
        wm_state = state_dicts.get("wm")
        policy_state = state_dicts.get("policy")
        ensemble_state = state_dicts.get("ensemble")
 
    import time
    t0 = time.time()
    train(
        env=env,
        cfg=cfg,
        n_iterations=args.iters,
        rollout_size=cfg.rollout_size,
        use_ensemble=not args.no_ensemble,
        ego_xy_fn=highway_ego_xy,
        mandated_action_fn=highway_mandated_action,
        device=args.device,
        seed=args.seed,
        verbose=True,
        save_every=args.save_every,
        save_path=args.save_path,
        wm_state=wm_state,
        policy_state=policy_state,
        ensemble_state=ensemble_state,
        resume_path=args.resume,
        eval_every=args.eval_every,
    )
    elapsed = time.time() - t0
    print(f"\nOK Training highway-env termine en {elapsed:.1f}s")
 
def run_carla_training(args):
    if not HAS_TORCH:
        print("PyTorch requis pour le mode carla.")
        return

    from sdbs.model.sdbs_dreamer import train
    from sdbs.envs.carla_env_adapter import (
        CarlaEnvAdapter,
        carla_ego_xy,
        carla_mandated_action,
    )

    print("\n" + "=" * 60)
    print("S-DBS Dreamer-PPO — Mode CARLA")
    print("=" * 60)

    env = CarlaEnvAdapter(
        host   = args.carla_host,
        port   = args.carla_port,
        seed   = args.seed,
        domain_randomization = args.domain_randomization,
        town   = args.town,
    )

    cfg = make_config(smoke=args.smoke)

    print(f"Env       : CarlaEnvAdapter  state_dim={env.state_dim}  "
        f"action_dim={env.action_dim}")
    print(f"Serveur   : {args.carla_host}:{args.carla_port}  carte={args.town}")
    print(f"Device    : {args.device}")
    print(f"Itérations: {args.iters}")

    wm_state = policy_state = ensemble_state = None
    if args.wm_checkpoint:
        state_dicts  = load_world_model_checkpoint(args.wm_checkpoint, args.device)
        wm_state     = state_dicts.get("wm")
        policy_state = state_dicts.get("policy")
        ensemble_state = state_dicts.get("ensemble")

    t0 = time.time()
    try:
        train(
            env          = env,
            cfg          = cfg,
            n_iterations = args.iters,
            rollout_size = cfg.rollout_size,
            use_ensemble = not args.no_ensemble,
            ego_xy_fn    = carla_ego_xy,
            mandated_action_fn = carla_mandated_action,
            device       = args.device,
            seed         = args.seed,
            verbose      = True,
            save_every   = args.save_every,
            save_path    = args.save_path,
            wm_state       = wm_state,
            policy_state   = policy_state,
            ensemble_state = ensemble_state,
            resume_path    = args.resume,
            eval_every     = args.eval_every,
            use_traffic_predictor = args.traffic_predictor,
        )
    finally:
        env.close()   # remettre CARLA en mode asynchrone proprement

    elapsed = time.time() - t0
    print(f"\nOK Training CARLA terminé en {elapsed:.1f}s")
# ==============================================================================
# 5.  Greedy vs S-DBS ablation rapide (sans training, dÃ©mo du greedy trap)
# ==============================================================================
def run_ablation_demo():
    """
    Compare greedy_one_step vs SDBSPlanner.plan sur le scÃ©nario OCCLUDED_PED.
    Ne nÃ©cessite pas PyTorch â€” utilise les stubs NumPy de sdbs_core.
    """
    from sdbs.core.sdbs_core import (
        SDBSPlanner, PlannerConfig, BudgetConfig,
        StubPolicy, StubWorldModel,
        MockDrivingEnv, mock_ego_xy, mock_mandated_action,
    )

    print("\n" + "=" * 60)
    print("Ablation : Greedy one-step vs S-DBS  (occluded-crosswalk)")
    print("=" * 60)

    # MockDrivingEnv (9-dim, NumPy pur) â€” le stub est cÃ¢blÃ© sur cet espace
    env         = MockDrivingEnv(seed=0)
    stub_policy = StubPolicy(seed=0)       # pas d'argument seed dans __init__
    stub_wm     = StubWorldModel()         # pas d'argument

    cfg_plan   = PlannerConfig(horizon=5, beam_width=6, n_groups=3,
                                eta=0.4, lam=0.5, mu=0.3)
    cfg_budget = BudgetConfig(max_rollouts=120)
    planner = SDBSPlanner(stub_policy, stub_wm, cfg_plan, cfg_budget,
                          mandated_action_fn=mock_mandated_action)

    results = {}
    for use_sdbs in [False, True]:
        name = "S-DBS" if use_sdbs else "Greedy"
        obs = env.reset(tier=2, n_vru=1, occlusion_prob=1.0, adversarial=False)
        total_r = 0.0
        min_ttcs = []
        for _ in range(env.max_steps):
            info = env.info()
            if use_sdbs:
                out = planner.plan(obs, info)
            else:
                out = planner.greedy_one_step(obs)   # greedy_one_step prend state, pas info
            action = out["action"]
            obs, r, done, step_info = env.step(action)
            total_r += r
            min_ttcs.append(step_info["min_ttc"])
            if done:
                break
        results[name] = dict(
            reward      = round(total_r, 2),
            min_ttc     = round(min(min_ttcs), 2),
            collisions  = env.collisions,
            near_misses = env.near_misses,
            success     = step_info.get("success", False),
        )

    print(f"\n{'MÃ©trique':<20} {'Greedy':>12} {'S-DBS':>12}")
    print("-" * 46)
    for k in ["reward", "min_ttc", "collisions", "near_misses", "success"]:
        print(f"{k:<20} {str(results['Greedy'][k]):>12} {str(results['S-DBS'][k]):>12}")
    print()


# ==============================================================================
# 6.  Point d'entrÃ©e
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="S-DBS Dreamer-PPO training")
    p.add_argument("--mode", default="mock", choices=["mock", "carla", "highway"])
    p.add_argument("--smoke",         action="store_true",
                   help="3 itÃ©rations, rollout court â€” vÃ©rifie que tout se branche")
    p.add_argument("--ablation",      action="store_true",
                   help="DÃ©mo greedy vs S-DBS sans training (NumPy pur)")
    p.add_argument("--iters",         type=int, default=50)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--wm_checkpoint", default=None,
                   help="Chemin vers un checkpoint WorldModel prÃ©-entraÃ®nÃ© (.pt)")
    p.add_argument("--no_ensemble",   action="store_true",
                   help="DÃ©sactive WorldModelEnsemble (plus rapide, moins d'incertitude)")
    p.add_argument("--save_every",    type=int, default=0,
                   help="Sauvegarder un checkpoint tous les N itÃ©rations (0 = jamais)")
    p.add_argument("--save_path",     default="checkpoint.pt",
                   help="Chemin du fichier checkpoint (dÃ©faut: checkpoint.pt)")
    p.add_argument("--resume",        default=None,
                   help="Reprendre depuis un checkpoint COMPLET sauvegardÃ© par "
                        "--save_every (rÃ©seaux + optimizers + curriculum + RNG), "
                        "contrairement Ã  --wm_checkpoint qui ne charge que des poids")
    p.add_argument("--eval_every",    type=int, default=0,
                   help="Lancer une Ã©valuation hold-out (hors PER) tous les N "
                        "itÃ©rations (0 = jamais). Donne une mesure de "
                        "gÃ©nÃ©ralisation non biaisÃ©e par le curriculum/PER")
    p.add_argument("--domain_randomization", action="store_true",
                   help="Active le bruit capteur, la latence d'action et la "
                        "variation de friction dans EnhancedMockEnv (prÃ©pa sim2real "
                        "avant de passer sur CARLA)")
    p.add_argument("--carla_host", default="localhost",
                   help="Adresse IP du serveur CARLA (défaut: localhost)")
    p.add_argument("--carla_port", type=int, default=2000,
                   help="Port du serveur CARLA (défaut: 2000)")
    p.add_argument("--town",       default="Town05",
                   help="Carte CARLA à charger (défaut: Town05)")
    p.add_argument("--traffic_predictor", action="store_true",
                   help="Active le traffic predictor (par défaut false)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ablation:
        run_ablation_demo()
        return

    if args.smoke:
        args.iters = 3
        print("[smoke] Mode smoke activÃ© : 3 itÃ©rations.")

    if args.mode == "mock":
        run_mock_training(args)
    elif args.mode == "carla":
        run_carla_training(args)
    elif args.mode == "highway":
        run_highway_training(args)


if __name__ == "__main__":
    main()
