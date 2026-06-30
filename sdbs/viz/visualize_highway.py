"""
visualize_highway.py
================================================================================
Visualise un AGENT ENTRAÎNÉ (checkpoint .pt) en train de piloter réellement
dans highway-env -- pas un replay, le planner S-DBS prend les décisions à
chaque step, en live.

Deux modes de sortie :
  --out video.mp4 / .gif   : enregistre l'épisode, AUCUN display requis
                              (fonctionne sur n'importe quel serveur headless)
  --live                   : ouvre une vraie fenêtre pygame highway-env
                              (nécessite un display fonctionnel)

Usage :
    python visualize_highway.py --checkpoint mon_checkpoint.pt --out episode.mp4
    python visualize_highway.py --checkpoint mon_checkpoint.pt --tier 2 --adversarial --out hard.gif
    python visualize_highway.py --checkpoint mon_checkpoint.pt --live
    python visualize_highway.py --checkpoint mon_checkpoint.pt --scenario intersection-v0 --out inter.mp4

Sans --checkpoint : agent à poids aléatoires (vérifie juste que le pipeline
tourne, comportement non représentatif).
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from highway_env_adapter import HighwayEnvAdapter, highway_ego_xy, highway_mandated_action


def run_episode(planner, env: HighwayEnvAdapter, max_steps: int = 300,
                collect_frames: bool = True, verbose: bool = True):
    obs = env.reset(tier=env._tier_for_reset, n_vru=0,
                    occlusion_prob=0.0, adversarial=env._adversarial_for_reset)
    frames = []
    total_r = 0.0
    if collect_frames:
        f = env.render_frame()
        if f is not None:
            frames.append(f)

    for step in range(max_steps):
        info = env.info()
        out = planner.plan(obs, info)
        obs, r, done, info = env.step(out["action"], out["maneuver"])
        total_r += r

        if collect_frames:
            f = env.render_frame()
            if f is not None:
                frames.append(f)

        if verbose and step % 10 == 0:
            mode = out["meta"].get("mode", "?")
            print(f"  step {step:3d}  speed={info['speed']:5.1f}  "
                  f"ttc={info['min_ttc']:5.2f}  mode={mode:>14}  "
                  f"reward={r:+.2f}")

        if done:
            print(f"  -> episode termine au step {step} "
                  f"(crashed={info['crashed']}, success={info['success']})")
            break

    print(f"  reward total = {total_r:+.2f}")
    return frames


def main():
    p = argparse.ArgumentParser(description="Visualiser un agent S-DBS entraine dans highway-env")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint .pt entraine. Sans cet argument: poids aleatoires.")
    p.add_argument("--scenario", default=None,
                   help="Force un scenario highway-env (highway-v0, intersection-v0, "
                        "merge-v0, roundabout-v0, racetrack-v0). Sinon derive de --tier.")
    p.add_argument("--tier", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None,
                   help="Chemin de sortie video/gif (ex: episode.mp4 ou episode.gif). "
                        "Fonctionne sans display (headless).")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--live", action="store_true",
                   help="Ouvre une fenetre pygame en direct (necessite un display "
                        "fonctionnel -- a eviter si pygame plante chez vous).")
    args = p.parse_args()

    if not args.out and not args.live:
        args.out = "episode.mp4"
        print(f"[info] Ni --out ni --live precise -> sauvegarde par defaut: {args.out}")

    # --------------------------------------------------------------------
    # Construction de l'agent (planner reel si checkpoint fourni, sinon stub)
    # --------------------------------------------------------------------
    if args.checkpoint:
        from sdbs.viz.replay_data import load_trained_agent
        print(f"[load] Chargement du checkpoint {args.checkpoint} ...")
        dummy_env = HighwayEnvAdapter()
        agent = load_trained_agent(
            args.checkpoint,
            device=args.device,
            env=dummy_env,
            ego_xy_fn=highway_ego_xy,
            mandated_action_fn=highway_mandated_action,
        )
        planner = agent["planner"]
    else:
        print("[!] Aucun --checkpoint -- agent a poids aleatoires "
              "(comportement non representatif).")
        from sdbs.core.sdbs_core import (
            SDBSPlanner, PlannerConfig, BudgetConfig, StubPolicy, StubWorldModel,
        )
        policy = StubPolicy(seed=args.seed)
        wm = StubWorldModel()
        cfg_plan = PlannerConfig(horizon=5, beam_width=6, n_groups=3, eta=0.4, lam=0.5, mu=0.3)
        cfg_budget = BudgetConfig(max_rollouts=120)
        planner = SDBSPlanner(policy, wm, cfg_plan, cfg_budget,
                              mandated_action_fn=highway_mandated_action)

    # --------------------------------------------------------------------
    # Environnement
    # --------------------------------------------------------------------
    env = HighwayEnvAdapter(
        scenario=args.scenario, seed=args.seed,
        render=args.live,            # "human" si --live, sinon rgb_array headless
    )
    env._tier_for_reset = args.tier
    env._adversarial_for_reset = args.adversarial

    print(f"\n[run] scenario={args.scenario or f'(derive de tier={args.tier})'}  "
          f"adversarial={args.adversarial}  seed={args.seed}\n")

    frames = run_episode(planner, env, max_steps=args.max_steps,
                         collect_frames=bool(args.out), verbose=True)
    env.close()

    # --------------------------------------------------------------------
    # Sauvegarde video/gif (aucun display requis)
    # --------------------------------------------------------------------
    if args.out and frames:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".gif":
            import imageio
            imageio.mimsave(str(out_path), frames, fps=args.fps)
        else:
            import imageio
            writer = imageio.get_writer(str(out_path), fps=args.fps)
            for f in frames:
                writer.append_data(f)
            writer.close()
        print(f"\n[saved] {len(frames)} frames -> {out_path.resolve()}")


if __name__ == "__main__":
    main()