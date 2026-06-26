"""
collect_carla_driving_data.py
================================================================================
Collecte des transitions (state, action, next_state, risk_target,
progress_target) en pilotant l'ego avec le Traffic Manager de CARLA
(autopilote) -- aucun agent entraÃ®nÃ© requis. C'est la conduite "normale" dont
on parlait : utile pour prÃ©-entraÃ®ner le WorldModel (encoder+GRU+next_state)
avant de l'utiliser dans le pipeline S-DBS / PPO.

Deux modes :

  1.  RÃ‰EL (nÃ©cessite un serveur CARLA lancÃ© + le module `carla`) :
        python collect_carla_driving_data.py --host localhost --port 2000 \
            --episodes 20 --steps_per_episode 600 --out data/carla_normal.npz

  2.  --mock (AUCUNE dÃ©pendance CARLA, tourne ici/en local) :
        python collect_carla_driving_data.py --mock --episodes 20 \
            --steps_per_episode 200 --out data/mock_normal.npz

      Pilote EnhancedMockEnv avec un contrÃ´leur heuristique simple (pas un
      agent entraÃ®nÃ©) pour produire des donnÃ©es structurellement identiques
      (mÃªmes colonnes, mÃªmes dtypes) Ã  ce que produirait le mode rÃ©el. Sert Ã 
      valider tout le pipeline (collecte -> pretrain_world_model_offline.py
      -> build_agent(wm_state=...)) avant d'avoir accÃ¨s Ã  un serveur CARLA.

Format de sortie (.npz) : states, actions, next_states, risk_targets,
progress_targets -- exactement les clÃ©s que consomme WorldModelBuffer /
pretrain_world_model_offline.py.
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


# ==============================================================================
# 1.  Mode --mock : pas de CARLA, pour tester tout le pipeline localement
# ==============================================================================
def _heuristic_action(obs: np.ndarray) -> np.ndarray:
    """ContrÃ´leur heuristique simple (PAS un agent entraÃ®nÃ©) : accÃ©lÃ¨re sur
    route dÃ©gagÃ©e, freine progressivement si un VRU est visible et proche ou
    si un feu est rouge. Sert uniquement Ã  gÃ©nÃ©rer une conduite "normale"
    plausible pour les donnÃ©es de prÃ©-entraÃ®nement -- pas Ã  Ãªtre performant
    ni sÃ»r ; c'est `mandated_action_fn` / S-DBS qui gÃ¨rent la sÃ©curitÃ©
    pendant l'entraÃ®nement RL, pas ce script de collecte."""
    from sdbs.envs.enhanced_mock_env import (
        IDX_SPEED, IDX_VRU0_DX, IDX_VRU0_VIS, IDX_TL, IDX_MIN_TTC,
    )
    speed = float(obs[IDX_SPEED])
    min_ttc = float(obs[IDX_MIN_TTC])
    light_red = float(obs[IDX_TL]) > 0.5

    steer = float(np.random.normal(0.0, 0.05))   # lÃ©ger bruit pour varier les trajectoires
    if light_red or min_ttc < 4.0:
        throttle, brake = 0.0, float(np.clip(1.5 / max(min_ttc, 0.5), 0.0, 1.0))
    elif speed < 8.0:
        throttle, brake = 0.7, 0.0
    else:
        throttle, brake = 0.3, 0.0
    return np.array([steer, throttle, brake], dtype=np.float32)


def collect_mock(episodes: int, steps_per_episode: int, seed: int = 0) -> dict:
    from sdbs.envs.enhanced_mock_env import EnhancedMockEnv

    env = EnhancedMockEnv(seed=seed, domain_randomization=True)
    states, actions, next_states = [], [], []
    risk_targets, progress_targets = [], []

    tiers = [0, 0, 1, 1, 2]   # majoritairement de la conduite "normale" (tiers 0-1)
    for ep in range(episodes):
        tier = tiers[ep % len(tiers)]
        obs = env.reset(tier=tier, n_vru=(0 if tier == 0 else 1),
                        occlusion_prob=0.3, adversarial=False)
        for _ in range(steps_per_episode):
            action = _heuristic_action(obs)
            next_obs, reward, done, info = env.step(action)
            states.append(obs.copy())
            actions.append(action.copy())
            next_states.append(next_obs.copy())
            risk_targets.append(float(info.get("vru_risk", 0.0)))
            progress_targets.append(float(info.get("progress", 0.0)))
            obs = next_obs
            if done:
                obs = env.reset(tier=tier, n_vru=(0 if tier == 0 else 1),
                               occlusion_prob=0.3, adversarial=False)
        print(f"[mock] Ã©pisode {ep+1}/{episodes} (tier={tier}) -> "
              f"{len(states)} transitions cumulÃ©es")

    return dict(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        risk_targets=np.asarray(risk_targets, dtype=np.float32),
        progress_targets=np.asarray(progress_targets, dtype=np.float32),
    )


# ==============================================================================
# 2.  Mode rÃ©el : pilotage par le Traffic Manager de CARLA
# ==============================================================================
def collect_carla(host: str, port: int, episodes: int, steps_per_episode: int,
                  n_vehicles: int, n_walkers: int, town: str,
                  seed: int = 0) -> dict:
    try:
        import carla 
    except ImportError as e:
        raise RuntimeError(
            "Le module `carla` (PythonAPI) n'est pas installable depuis ce "
            "rÃ©seau restreint -- ce mode doit Ãªtre lancÃ© sur la machine oÃ¹ "
            "tourne rÃ©ellement ton serveur CARLA, avec son PythonAPI sur le "
            "PYTHONPATH. Utilise --mock pour tester le pipeline ici."
        ) from e

    from sdbs.envs.carla_obs_utils import (
        RouteTracker, extract_observation, get_applied_action,
    )

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    world = client.load_world(town) if town else client.get_world()
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    rng = np.random.default_rng(seed)
    blueprint_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    traffic_lights = list(world.get_actors().filter("traffic.traffic_light"))

    states, actions, next_states = [], [], []
    risk_targets, progress_targets = [], []

    for ep in range(episodes):
        actors_to_destroy = []
        try:
            # ---- ego (autopilote via Traffic Manager) ----
            ego_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
            spawn_tf = spawn_points[int(rng.integers(0, len(spawn_points)))]
            ego = world.spawn_actor(ego_bp, spawn_tf)
            ego.set_autopilot(True, tm.get_port())
            actors_to_destroy.append(ego)

            # ---- trafic NPC (pour avoir des occluders rÃ©alistes) ----
            vehicles = []
            v_bps = blueprint_lib.filter("vehicle.*")
            for _ in range(n_vehicles):
                bp = v_bps[int(rng.integers(0, len(v_bps)))]
                sp = spawn_points[int(rng.integers(0, len(spawn_points)))]
                v = world.try_spawn_actor(bp, sp)
                if v is not None:
                    v.set_autopilot(True, tm.get_port())
                    vehicles.append(v)
                    actors_to_destroy.append(v)

            # ---- piÃ©tons (VRU) ----
            walkers = []
            w_bps = blueprint_lib.filter("walker.pedestrian.*")
            for _ in range(n_walkers):
                bp = w_bps[int(rng.integers(0, len(w_bps)))]
                loc = world.get_random_location_from_navigation()
                if loc is None:
                    continue
                w = world.try_spawn_actor(bp, carla.Transform(loc))
                if w is not None:
                    walkers.append(w)
                    actors_to_destroy.append(w)
                    controller_bp = blueprint_lib.find("controller.ai.walker")
                    ctrl = world.spawn_actor(controller_bp, carla.Transform(), w)
                    ctrl.start()
                    ctrl.go_to_location(world.get_random_location_from_navigation())
                    actors_to_destroy.append(ctrl)

            world.tick()

            # ---- route (pour la progression / IDX_PROGRESS) ----
            map_ = world.get_map()
            start_wp = map_.get_waypoint(ego.get_location())
            wps = [start_wp]
            cur = start_wp
            for _ in range(200):
                nxt = cur.next(2.0)
                if not nxt:
                    break
                cur = nxt[0]
                wps.append(cur)
            route = RouteTracker(waypoints=wps, route_len_m=2.0 * len(wps))

            obs = extract_observation(world, ego, route, walkers, vehicles,
                                      traffic_lights)
            for _ in range(steps_per_episode):
                action = get_applied_action(ego)
                world.tick()
                next_obs = extract_observation(world, ego, route, walkers,
                                               vehicles, traffic_lights)

                min_ttc = float(next_obs[11])  # IDX_MIN_TTC (cf. carla_obs_utils)
                vru_risk = float(np.clip(np.exp(-max(min_ttc, 0.0) / 3.0), 0.0, 1.0))
                progress = float(next_obs[19])  # IDX_PROGRESS

                states.append(obs.copy())
                actions.append(action.copy())
                next_states.append(next_obs.copy())
                risk_targets.append(vru_risk)
                progress_targets.append(progress)
                obs = next_obs

            print(f"[carla] Ã©pisode {ep+1}/{episodes} -> "
                  f"{len(states)} transitions cumulÃ©es")
        finally:
            for a in actors_to_destroy:
                try:
                    a.destroy()
                except Exception:
                    pass

    settings.synchronous_mode = False
    world.apply_settings(settings)

    return dict(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        risk_targets=np.asarray(risk_targets, dtype=np.float32),
        progress_targets=np.asarray(progress_targets, dtype=np.float32),
    )


# ==============================================================================
# 3.  EntrÃ©e
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description="Collecte de conduite normale "
                                            "(CARLA Traffic Manager ou --mock)")
    p.add_argument("--mock", action="store_true",
                   help="Pas de CARLA : utilise EnhancedMockEnv + heuristique "
                        "pour tester tout le pipeline localement")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="", help="ex: Town03 ; vide = carte dÃ©jÃ  chargÃ©e")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps_per_episode", type=int, default=400)
    p.add_argument("--n_vehicles", type=int, default=15)
    p.add_argument("--n_walkers", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="carla_normal_driving.npz")
    args = p.parse_args()

    t0 = time.time()
    if args.mock:
        print("[collect] Mode --mock (EnhancedMockEnv, aucune dÃ©pendance CARLA)")
        data = collect_mock(args.episodes, args.steps_per_episode, args.seed)
    else:
        print(f"[collect] Mode CARLA rÃ©el -> {args.host}:{args.port}")
        data = collect_carla(args.host, args.port, args.episodes,
                             args.steps_per_episode, args.n_vehicles,
                             args.n_walkers, args.town, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    elapsed = time.time() - t0
    print(f"\n[collect] {data['states'].shape[0]} transitions sauvegardÃ©es "
          f"dans {args.out} en {elapsed:.1f}s")
    print(f"          state_dim={data['states'].shape[1]} "
          f"action_dim={data['actions'].shape[1]}")


if __name__ == "__main__":
    main()
