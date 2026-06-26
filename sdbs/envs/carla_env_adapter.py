"""
carla_env_adapter.py
================================================================================
Drop-in replacement for EnhancedMockEnv that wraps a live CARLA server.

Interface identique à EnhancedMockEnv :
  - reset(tier, n_vru, occlusion_prob, adversarial) -> np.ndarray
  - step(action)  -> (obs, reward, done, info)
  - info()        -> dict
  - state_dim, action_dim

Dépendances :
  pip install carla numpy
  (carla wheel disponible sur le serveur CARLA, ex: carla-0.9.15-cp310-...)

Usage depuis run_training.py :
  python run_training.py --mode carla --device cuda --iters 1000
================================================================================
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Optional

import numpy as np

try:
    import carla
    HAS_CARLA = True
except ImportError:
    HAS_CARLA = False
    # Permet d'importer le module sans CARLA installé (ex: CI)
    # Une erreur claire sera levée à l'instanciation.

# ---------------------------------------------------------------------------
# Dimensions — doivent correspondre à celles d'EnhancedMockEnv
# ---------------------------------------------------------------------------
# obs = [ego_x, ego_y, ego_vx, ego_vy, ego_yaw,        (5)
#         nearest_vru_rx, nearest_vru_ry, nearest_vru_v, (3)
#         ttc, occlusion_flag,                           (2)
#         road_curvature, speed_limit_norm,              (2)
#         vru1_rx, vru1_ry, vru2_rx, vru2_ry,           (4)
#         lateral_offset, heading_error]                 (2)
#                                                  total = 18
CARLA_STATE_DIM  = 18
CARLA_ACTION_DIM = 2   # [steering (-1..1), accel (-1..1) : <0 = freinage]

# Constantes physiques / capteurs
MAX_SPEED_MS      = 14.0   # ~50 km/h
COLLISION_RADIUS  = 2.5    # mètres  — détection proximité VRU
TTC_INF           = 10.0   # valeur sentinelle si pas de conflit
MAX_EPISODE_STEPS = 300
SPAWN_RADIUS      = 40.0   # rayon autour de l'ego pour spawner les VRU


# ==============================================================================
# Fonctions compagnes (même rôle que enhanced_ego_xy / enhanced_mandated_action)
# ==============================================================================

def carla_ego_xy(obs: np.ndarray):
    """Extrait (x, y) de l'observation vectorielle."""
    return float(obs[0]), float(obs[1])


def carla_mandated_action(obs: np.ndarray, info: dict) -> Optional[np.ndarray]:
    """
    Retourne une action imposée si la sécurité l'exige, sinon None.
    Le planner S-DBS appellera cette fonction à chaque pas.
    """
    ttc = float(obs[8])
    if ttc < 1.5:
        # Freinage d'urgence
        return np.array([0.0, -1.0], dtype=np.float32)
    if info.get("collision", False):
        return np.array([0.0, -1.0], dtype=np.float32)
    return None


# ==============================================================================
# Adapter principal
# ==============================================================================

class CarlaEnvAdapter:
    """
    Wraps a CARLA server as a gym-like environment compatible avec
    EnhancedMockEnv (même state_dim, action_dim, reset/step/info).
    """

    state_dim  = CARLA_STATE_DIM
    action_dim = CARLA_ACTION_DIM

    # ------------------------------------------------------------------
    def __init__(
        self,
        host: str  = "localhost",
        port: int  = 2000,
        seed: int  = 0,
        domain_randomization: bool = False,
        timeout: float = 20.0,
        town: str = "Town05",        # Town05 = intersections urbaines
        render: bool = False,        # True = spectator actif (debug)
        fixed_delta_seconds: float = 0.05,   # 20 Hz
    ):
        if not HAS_CARLA:
            raise RuntimeError(
                "Le module 'carla' n'est pas installé.\n"
                "Installez le wheel fourni avec votre serveur CARLA :\n"
                "  pip install <chemin>/carla-*.whl"
            )

        self.host   = host
        self.port   = port
        self.seed   = seed
        self.dr     = domain_randomization
        self.render = render
        self.fixed_dt = fixed_delta_seconds
        self.town   = town

        self._rng = np.random.default_rng(seed)
        random.seed(seed)

        # Connexion CARLA
        print(f"[CARLA] Connexion à {host}:{port} …")
        self._client  = carla.Client(host, port)
        self._client.set_timeout(timeout)
        self._world   = self._client.load_world(town)
        print(f"[CARLA] Monde '{town}' chargé.")

        # Mode synchrone
        settings = self._world.get_settings()
        settings.synchronous_mode      = True
        settings.fixed_delta_seconds   = fixed_delta_seconds
        settings.no_rendering_mode     = not render
        self._world.apply_settings(settings)

        self._tm = self._client.get_trafficmanager()
        self._tm.set_synchronous_mode(True)
        self._tm.set_random_device_seed(seed)

        self._bp_lib    = self._world.get_blueprint_library()
        self._spawn_pts = self._world.get_map().get_spawn_points()

        # Acteurs gérés par l'épisode
        self._ego       : Optional[carla.Vehicle]    = None
        self._vrus      : list[carla.Walker]         = []
        self._vru_ctrls : list[carla.WalkerAIController] = []
        self._sensors   : list                       = []

        # État interne
        self._step_count  = 0
        self._collision   = False
        self._lane_inv    = False
        self._last_obs    = np.zeros(CARLA_STATE_DIM, dtype=np.float32)
        self._last_info   : dict = {}

        # Callbacks capteurs
        self._col_hist : list  = []

        # Tier / paramètres épisode courant
        self._tier          = 1
        self._n_vru         = 0
        self._occlusion_p   = 0.0
        self._adversarial   = False

        # Stats épisode
        self.collisions  = 0
        self.near_misses = 0

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------

    def reset(
        self,
        tier: int   = 1,
        n_vru: int  = 0,
        occlusion_prob: float = 0.0,
        adversarial: bool = False,
    ) -> np.ndarray:
        """Réinitialise l'épisode et retourne l'obs initiale."""
        self._tier        = tier
        self._n_vru       = n_vru
        self._occlusion_p = occlusion_prob
        self._adversarial = adversarial

        self._destroy_actors()

        # Spawn ego
        spawn_pt = random.choice(self._spawn_pts)
        ego_bp   = self._bp_lib.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "hero")
        self._ego = self._world.try_spawn_actor(ego_bp, spawn_pt)
        if self._ego is None:
            # Fallback : essaie d'autres points
            for sp in self._spawn_pts:
                self._ego = self._world.try_spawn_actor(ego_bp, sp)
                if self._ego is not None:
                    break
        if self._ego is None:
            raise RuntimeError("[CARLA] Impossible de spawner le véhicule ego.")

        # Capteurs
        self._attach_sensors()

        # Spawn VRU
        self._spawn_vrus(n_vru)

        # Domain randomisation
        if self.dr:
            self._apply_domain_randomization()

        # Avancer quelques ticks pour stabiliser la physique
        for _ in range(10):
            self._world.tick()

        self._step_count  = 0
        self._collision   = False
        self._lane_inv    = False
        self._col_hist    = []
        self.collisions   = 0
        self.near_misses  = 0

        self._last_obs  = self._build_obs()
        self._last_info = self._build_info(reward=0.0, done=False)
        return self._last_obs.copy()

    def step(self, action: np.ndarray):
        """
        action : np.ndarray shape (2,)
          action[0] = steering  ∈ [-1, 1]
          action[1] = accel     ∈ [-1, 1]  (<0 = frein)
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], -1.0, 1.0))

        if accel >= 0.0:
            ctrl = carla.VehicleControl(
                throttle = accel,
                steer    = steer,
                brake    = 0.0,
            )
        else:
            ctrl = carla.VehicleControl(
                throttle = 0.0,
                steer    = steer,
                brake    = -accel,
            )

        self._ego.apply_control(ctrl)
        self._world.tick()
        self._step_count += 1

        obs    = self._build_obs()
        reward = self._compute_reward(obs, action)
        done   = self._is_done(obs)
        iinfo  = self._build_info(reward, done)

        self._last_obs  = obs
        self._last_info = iinfo
        return obs.copy(), reward, done, iinfo

    def info(self) -> dict:
        return self._last_info.copy()

    # ------------------------------------------------------------------
    # Construction de l'observation
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(CARLA_STATE_DIM, dtype=np.float32)
        if self._ego is None:
            return obs

        t   = self._ego.get_transform()
        v   = self._ego.get_velocity()
        yaw = math.radians(t.rotation.yaw)

        ego_x  = t.location.x
        ego_y  = t.location.y
        ego_vx = v.x
        ego_vy = v.y

        obs[0] = ego_x  / 100.0   # normalisation grossière
        obs[1] = ego_y  / 100.0
        obs[2] = ego_vx / MAX_SPEED_MS
        obs[3] = ego_vy / MAX_SPEED_MS
        obs[4] = yaw    / math.pi

        # VRU le plus proche
        min_dist = float("inf")
        nearest_rv = np.zeros(3)
        vru_slots  = np.zeros(4)   # 2 VRU × (rx, ry)

        for i, vru in enumerate(self._vrus[:2]):
            vt  = vru.get_transform()
            vv  = vru.get_velocity()
            rx  = (vt.location.x - ego_x) / 50.0
            ry  = (vt.location.y - ego_y) / 50.0
            spd = math.hypot(vv.x, vv.y) / 2.0   # piéton ~1-2 m/s

            d = math.hypot(rx, ry) * 50.0
            if d < min_dist:
                min_dist   = d
                nearest_rv = np.array([rx, ry, spd])

            vru_slots[2*i]   = rx
            vru_slots[2*i+1] = ry

        obs[5] = nearest_rv[0]
        obs[6] = nearest_rv[1]
        obs[7] = nearest_rv[2]

        # TTC simplifié : distance / vitesse relative radiale
        ego_spd = math.hypot(ego_vx, ego_vy)
        if min_dist < 50.0 and ego_spd > 0.5:
            ttc = min(min_dist / max(ego_spd, 0.1), TTC_INF)
        else:
            ttc = TTC_INF
        obs[8] = ttc / TTC_INF

        # Occlusion (stochastique selon le tier)
        obs[9] = float(self._rng.random() < self._occlusion_p)

        # Road curvature (approximation via waypoint)
        wp = self._world.get_map().get_waypoint(t.location)
        next_wps = wp.next(5.0)
        if next_wps:
            nwp   = next_wps[0]
            dyaw  = math.radians(nwp.transform.rotation.yaw - t.rotation.yaw)
            dyaw  = (dyaw + math.pi) % (2 * math.pi) - math.pi
            obs[10] = dyaw / math.pi
        else:
            obs[10] = 0.0

        # Speed limit
        spd_lim = wp.speed_limit if hasattr(wp, "speed_limit") else 50.0
        obs[11] = min(spd_lim / 3.6, MAX_SPEED_MS) / MAX_SPEED_MS

        # VRU slots
        obs[12] = vru_slots[0]
        obs[13] = vru_slots[1]
        obs[14] = vru_slots[2]
        obs[15] = vru_slots[3]

        # Lateral offset & heading error
        lane_center = wp.transform.location
        lane_yaw    = math.radians(wp.transform.rotation.yaw)
        dx = ego_x - lane_center.x
        dy = ego_y - lane_center.y
        lat_off = -dx * math.sin(lane_yaw) + dy * math.cos(lane_yaw)
        head_err = yaw - lane_yaw
        head_err = (head_err + math.pi) % (2 * math.pi) - math.pi
        obs[16] = lat_off   / 3.0
        obs[17] = head_err  / math.pi

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        r = 0.0

        # 1. Vitesse : récompense de progresser
        ego_spd = math.hypot(
            self._ego.get_velocity().x,
            self._ego.get_velocity().y,
        )
        spd_norm = ego_spd / MAX_SPEED_MS
        r += 0.3 * spd_norm

        # 2. Sécurité VRU
        ttc = float(obs[8]) * TTC_INF
        if ttc < 2.0:
            r -= 0.5 * (2.0 - ttc) / 2.0   # pénalité progressive

        # 3. Collision détectée par le capteur
        if self._collision:
            r -= 5.0
            self.collisions += 1
            self._collision = False

        # 4. Near-miss
        if ttc < 1.0:
            self.near_misses += 1

        # 5. Tenue de voie
        lat_off = float(obs[16]) * 3.0
        r -= 0.1 * abs(lat_off)

        # 6. Confort : pénalité sur les actions brusques
        r -= 0.05 * (abs(action[0]) + max(0.0, -action[1]))

        return float(r)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _is_done(self, obs: np.ndarray) -> bool:
        if self._step_count >= MAX_EPISODE_STEPS:
            return True
        # Collision grave
        if len(self._col_hist) > 0:
            return True
        # Hors route (lateral offset trop grand)
        if abs(float(obs[16]) * 3.0) > 3.0:
            return True
        return False

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _build_info(self, reward: float, done: bool) -> dict:
        obs = self._last_obs
        ttc = float(obs[8]) * TTC_INF
        return {
            "min_ttc"    : ttc,
            "collision"  : len(self._col_hist) > 0,
            "near_misses": self.near_misses,
            "collisions" : self.collisions,
            "step"       : self._step_count,
            "success"    : done and len(self._col_hist) == 0
                           and self._step_count >= MAX_EPISODE_STEPS * 0.9,
            "reward"     : reward,
        }

    # ------------------------------------------------------------------
    # Capteurs
    # ------------------------------------------------------------------

    def _attach_sensors(self):
        # Collision sensor
        col_bp = self._bp_lib.find("sensor.other.collision")
        col_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego
        )
        col_sensor.listen(self._on_collision)
        self._sensors.append(col_sensor)

        # Lane invasion sensor
        li_bp = self._bp_lib.find("sensor.other.lane_invasion")
        li_sensor = self._world.spawn_actor(
            li_bp, carla.Transform(), attach_to=self._ego
        )
        li_sensor.listen(self._on_lane_invasion)
        self._sensors.append(li_sensor)

    def _on_collision(self, event):
        self._collision = True
        self._col_hist.append(event)

    def _on_lane_invasion(self, event):
        self._lane_inv = True

    # ------------------------------------------------------------------
    # Spawn VRU (piétons)
    # ------------------------------------------------------------------

    def _spawn_vrus(self, n_vru: int):
        if n_vru == 0:
            return
        if self._ego is None:
            return

        walker_bp_list = self._bp_lib.filter("walker.pedestrian.*")
        ctrl_bp = self._bp_lib.find("controller.ai.walker")
        ego_loc = self._ego.get_transform().location

        spawned = 0
        attempts = 0
        while spawned < n_vru and attempts < n_vru * 10:
            attempts += 1
            angle = random.uniform(0, 2 * math.pi)
            dist  = random.uniform(10.0, SPAWN_RADIUS)
            loc   = carla.Location(
                x = ego_loc.x + dist * math.cos(angle),
                y = ego_loc.y + dist * math.sin(angle),
                z = ego_loc.z + 1.0,
            )
            # Trouver un waypoint navigable proche
            wp = self._world.get_map().get_waypoint(
                loc, project_to_road=False,
                lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder
            )
            if wp is None:
                # Fallback : trottoir ou bord de route
                wp = self._world.get_map().get_waypoint(loc, project_to_road=True)
            spawn_t = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z),
                carla.Rotation(yaw=random.uniform(0, 360)),
            )
            bp = random.choice(walker_bp_list)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self._world.try_spawn_actor(bp, spawn_t)
            if walker is None:
                continue

            ctrl = self._world.spawn_actor(ctrl_bp, carla.Transform(),
                                           attach_to=walker)
            self._world.tick()
            ctrl.start()
            # Destination aléatoire pour que le piéton se déplace
            dest = ego_loc  # marche vers l'ego — crée un conflit
            ctrl.go_to_location(dest)
            ctrl.set_max_speed(random.uniform(0.8, 1.5))

            self._vrus.append(walker)
            self._vru_ctrls.append(ctrl)
            spawned += 1

        if spawned < n_vru:
            print(f"[CARLA] Attention : seulement {spawned}/{n_vru} VRU spawnés.")

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------

    def _apply_domain_randomization(self):
        """
        Varie : météo, friction des roues, paramètres Traffic Manager.
        """
        # Météo aléatoire
        weather = carla.WeatherParameters(
            cloudiness        = self._rng.uniform(0, 80),
            precipitation     = self._rng.uniform(0, 40),
            sun_altitude_angle= self._rng.uniform(10, 90),
            fog_density       = self._rng.uniform(0, 20),
            wind_intensity    = self._rng.uniform(0, 50),
        )
        self._world.set_weather(weather)

        # Friction des roues
        friction_val = self._rng.uniform(0.6, 1.0)
        friction_bp  = self._bp_lib.find("static.trigger.friction")
        extent = carla.Location(5.0, 5.0, 5.0)
        transform = self._ego.get_transform()
        friction_bp.set_attribute("friction",    str(friction_val))
        friction_bp.set_attribute("extent_x",    str(extent.x))
        friction_bp.set_attribute("extent_y",    str(extent.y))
        friction_bp.set_attribute("extent_z",    str(extent.z))
        # (trigger de friction — attach au sol ; simplifié ici)

        # Traffic Manager : variation latence / comportement
        self._tm.global_percentage_speed_difference(
            float(self._rng.uniform(-10, 20))
        )

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    def _destroy_actors(self):
        # Arrêter les AI walkers d'abord
        for ctrl in self._vru_ctrls:
            try:
                ctrl.stop()
                ctrl.destroy()
            except Exception:
                pass
        self._vru_ctrls = []

        for w in self._vrus:
            try:
                w.destroy()
            except Exception:
                pass
        self._vrus = []

        for s in self._sensors:
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self._sensors = []

        if self._ego is not None:
            try:
                self._ego.destroy()
            except Exception:
                pass
            self._ego = None

        # Un tick pour que CARLA purge
        try:
            self._world.tick()
        except Exception:
            pass

    def close(self):
        """Appeler en fin de session pour rendre le serveur en mode async."""
        self._destroy_actors()
        settings = self._world.get_settings()
        settings.synchronous_mode    = False
        settings.no_rendering_mode   = False
        self._world.apply_settings(settings)
        self._tm.set_synchronous_mode(False)
        print("[CARLA] Environnement fermé.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass