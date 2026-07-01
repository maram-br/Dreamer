"""
carla_env_adapter.py
================================================================================
Drop-in replacement for EnhancedMockEnv that wraps a live CARLA server.

Interface identique à EnhancedMockEnv :
  - reset(tier, n_vru, occlusion_prob, adversarial) -> np.ndarray
  - step(action, maneuver=None)  -> (obs, reward, done, info)
  - info()        -> dict
  - state_dim, action_dim

⚠️ Layout d'observation : 20-dim / action 3-dim [steer, throttle, brake],
EXACTEMENT le même que EnhancedMockEnv et carla_obs_utils.extract_observation.
C'est ce qui rend le hand-off mock -> CARLA réellement transparent : un
WorldModel pré-entraîné sur des logs CARLA (collect_carla_driving_data.py ->
pretrain_world_model_offline.py) se charge sans mismatch de dimension via
build_agent(wm_state=...), et enhanced_ego_xy / enhanced_mandated_action
transfèrent tels quels (réutilisés ci-dessous).

Dépendances :
  pip install carla numpy
  (carla wheel disponible sur le serveur CARLA, ex: carla-0.9.15-cp310-...)

Usage depuis run_training.py :
  python run_training.py --mode carla --device cuda --iters 1000
================================================================================
"""

from __future__ import annotations

import math
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

# L'extraction d'observation ET le layout 20-dim sont partagés avec la collecte
# de données (carla_obs_utils) : une seule source de vérité pour le layout.
from sdbs.envs.carla_obs_utils import (
    RouteTracker, extract_observation,
    CARLA_STATE_DIM, CARLA_ACTION_DIM,
    IDX_EGO_X, IDX_EGO_Y, IDX_SPEED, IDX_LANE_OFF, IDX_TL,
    IDX_MIN_TTC, IDX_OCC, IDX_VRU0_DX, IDX_VRU0_VIS, IDX_PROGRESS,
)
# Le layout étant identique à EnhancedMockEnv, on réutilise directement sa
# fonction ego_xy et son safety-floor mandaté (pas de duplication).
from sdbs.envs.enhanced_mock_env import (
    enhanced_ego_xy, enhanced_mandated_action,
)

# Constantes physiques / capteurs
MAX_SPEED_MS      = 14.0   # ~50 km/h
TTC_THRESHOLD     = 3.0    # seuil de violation TTC (aligné sur EnhancedMockEnv)
MAX_EPISODE_STEPS = 300
SPAWN_RADIUS      = 40.0   # rayon autour de l'ego pour spawner les VRU
OCCLUSION_REVEAL_DX = 7.0  # m : un VRU occlus scénarisé se révèle sous cette distance


# ==============================================================================
# Fonctions compagnes (même rôle que enhanced_ego_xy / enhanced_mandated_action)
# Le layout étant identique, on délègue aux versions EnhancedMockEnv pour éviter
# toute divergence de comportement entre mock et CARLA.
# ==============================================================================

def carla_ego_xy(obs: np.ndarray):
    """(x, y) route-relatifs pour la diversité conflict-cell (Sec. 3)."""
    return enhanced_ego_xy(obs)


def carla_mandated_action(obs: np.ndarray, info: dict) -> Optional[np.ndarray]:
    """Safety-floor imposé (freinage d'urgence / feu rouge / priorité).
    Retourne une action 3-dim [steer, throttle, brake] ou None."""
    return enhanced_mandated_action(obs, info)


# ==============================================================================
# Adapter principal
# ==============================================================================

class CarlaEnvAdapter:
    """
    Wraps a CARLA server as a gym-like environment compatible avec
    EnhancedMockEnv (même state_dim=20, action_dim=3, reset/step/info).
    """

    state_dim  = CARLA_STATE_DIM     # 20
    action_dim = CARLA_ACTION_DIM    # 3  -> [steer, throttle, brake]
    n_maneuvers = 6                  # doit matcher len(Maneuver) dans sdbs_core

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
        self.ttc_threshold = TTC_THRESHOLD

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
        self._vehicles  : list[carla.Vehicle]        = []   # occluders (pour extract_observation)
        self._sensors   : list                       = []
        self._route     : Optional[RouteTracker]     = None
        self._traffic_lights : list                  = []

        # État interne
        self._step_count  = 0
        self._collision   = False
        self._lane_inv    = False
        self._occluded    = False   # occlusion scénarisée (greedy trap), tirée par épisode
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
        self.collisions     = 0
        self.near_misses    = 0
        self.ttc_violations = 0

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
        """Réinitialise l'épisode et retourne l'obs initiale (20-dim)."""
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

        # Route (pour progress / lane-offset via carla_obs_utils) + feux de la carte
        self._build_route()
        self._traffic_lights = list(
            self._world.get_actors().filter("traffic.traffic_light"))

        # Occlusion scénarisée tirée une fois par épisode (greedy trap)
        self._occluded = bool(self._rng.random() < occlusion_prob)

        self._step_count  = 0
        self._collision   = False
        self._lane_inv    = False
        self._col_hist    = []
        self.collisions   = 0
        self.near_misses  = 0
        self.ttc_violations = 0

        self._last_obs  = self._build_obs()
        self._last_info = self._build_info(reward=0.0, done=False)
        return self._last_obs.copy()

    def step(self, action: np.ndarray, maneuver: Optional[int] = None):
        """
        action : np.ndarray shape (3,)  -> [steer, throttle, brake]
          steer    ∈ [-1, 1]
          throttle ∈ [0, 1]   (valeurs négatives clampées à 0)
          brake    ∈ [0, 1]   (valeurs négatives clampées à 0)
        maneuver : option discrète du planner (ignorée ici — la dynamique CARLA
          n'est pilotée que par le contrôle continu). Présent uniquement pour
          rester signature-compatible avec EnhancedMockEnv/MockDrivingEnv, que
          la boucle d'entraînement appelle via env.step(action, maneuver).
        """
        a = np.asarray(action, np.float32).reshape(-1)
        steer    = float(np.clip(a[0], -1.0, 1.0))
        throttle = float(np.clip(a[1],  0.0, 1.0))
        brake    = float(np.clip(a[2],  0.0, 1.0))

        ctrl = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self._ego.apply_control(ctrl)
        self._world.tick()
        self._step_count += 1

        obs = self._build_obs()

        # comptage sécurité (alimente reward, curriculum ttc_clean, et PER)
        ttc   = float(obs[IDX_MIN_TTC])
        speed = float(obs[IDX_SPEED])
        near_miss = (len(self._vrus) > 0 and ttc < 1.5 and speed > 2.0)
        if len(self._vrus) > 0 and ttc < self.ttc_threshold:
            self.ttc_violations += 1
        if near_miss:
            self.near_misses += 1

        reward = self._compute_reward(obs, np.array([steer, throttle, brake]),
                                      near_miss)
        done   = self._is_done(obs)

        self._last_obs  = obs
        self._last_info = self._build_info(reward, done)
        return obs.copy(), float(reward), bool(done), self._last_info

    def info(self) -> dict:
        return self._last_info.copy()

    # ------------------------------------------------------------------
    # Construction de l'observation  (délègue au layout 20-dim partagé)
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        if self._ego is None or self._route is None:
            return np.zeros(CARLA_STATE_DIM, dtype=np.float32)

        obs = extract_observation(
            self._world, self._ego, self._route,
            self._vrus, self._vehicles, self._traffic_lights,
        )

        # Occlusion scénarisée (greedy trap) : tant que le VRU le plus proche
        # reste loin, on le masque ; il se révèle quand l'ego s'approche
        # (< OCCLUSION_REVEAL_DX), comme dans EnhancedMockEnv.
        if self._occluded and len(self._vrus) > 0:
            if float(obs[IDX_VRU0_DX]) > OCCLUSION_REVEAL_DX:
                obs[IDX_OCC]      = 1.0
                obs[IDX_VRU0_VIS] = 0.0
        return obs

    # ------------------------------------------------------------------
    # Route : polyligne de waypoints devant l'ego (progress + lane offset)
    # ------------------------------------------------------------------

    def _build_route(self):
        map_ = self._world.get_map()
        start_wp = map_.get_waypoint(self._ego.get_location())
        wps = [start_wp]
        cur = start_wp
        for _ in range(200):
            nxt = cur.next(2.0)
            if not nxt:
                break
            cur = nxt[0]
            wps.append(cur)
        self._route = RouteTracker(waypoints=wps, route_len_m=2.0 * len(wps))

    # ------------------------------------------------------------------
    # Reward  (mêmes termes/échelle que EnhancedMockEnv._reward, pour que le
    # reward-shaping et le curriculum se transfèrent du mock à CARLA)
    # ------------------------------------------------------------------

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray,
                        near_miss: bool) -> float:
        steer, _throttle, brake = float(action[0]), float(action[1]), float(action[2])
        speed = float(obs[IDX_SPEED])
        dist  = speed * self.fixed_dt   # progression approx. sur ce pas (m)

        r_prog      =  dist / 8.0
        collision   = len(self._col_hist) > 0
        r_collision = -20.0 if collision else 0.0
        r_near_miss =  -4.0 if near_miss else 0.0

        ttc = float(obs[IDX_MIN_TTC])
        r_ttc = 0.0
        if len(self._vrus) > 0 and ttc < self.ttc_threshold:
            r_ttc = -2.0 * (1.0 - ttc / self.ttc_threshold)

        r_comfort = -0.05 * (abs(steer) + abs(brake))
        r_tl      = -5.0 if (float(obs[IDX_TL]) > 0.5 and speed > 1.0) else 0.0
        r_lane    = -0.2 * abs(float(obs[IDX_LANE_OFF]))

        if collision:
            self.collisions += 1
            self._col_hist = []   # consommé (une collision comptée une fois)

        return float(r_prog + r_collision + r_near_miss + r_ttc
                     + r_comfort + r_tl + r_lane)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _is_done(self, obs: np.ndarray) -> bool:
        if self._step_count >= MAX_EPISODE_STEPS:
            return True
        if len(self._col_hist) > 0:          # collision non encore consommée
            return True
        if float(obs[IDX_PROGRESS]) >= 1.0:  # route terminée
            return True
        if abs(float(obs[IDX_LANE_OFF])) > 3.0:  # sortie de voie
            return True
        return False

    # ------------------------------------------------------------------
    # Info dict  (mêmes clés que EnhancedMockEnv._build_info)
    # ------------------------------------------------------------------

    def _build_info(self, reward: float, done: bool) -> dict:
        obs   = self._last_obs
        ttc   = float(obs[IDX_MIN_TTC])
        occ   = float(obs[IDX_OCC])
        n_vru = len(self._vrus)

        nearest_dx = float(obs[IDX_VRU0_DX]) if n_vru else 99.0
        vru_risk   = float(np.clip(math.exp(-nearest_dx / 4.0), 0.0, 1.0)) if n_vru else 0.0
        density    = float(np.clip(n_vru / 3.0 + occ * 0.3, 0.0, 1.0))
        progress   = float(obs[IDX_PROGRESS])

        return {
            "min_ttc"       : ttc,
            "ttc_threshold" : self.ttc_threshold,
            "n_vru"         : n_vru,
            "occlusion"     : occ,
            "uncertainty"   : occ * 0.6,
            "collision"     : len(self._col_hist) > 0,
            "near_misses"   : self.near_misses,
            "collisions"    : self.collisions,
            "ttc_violations": self.ttc_violations,
            # cibles des têtes du world model (sinon entraînées sur des zéros)
            "vru_risk"      : vru_risk,
            "density"       : density,
            "progress"      : progress,
            "step"          : self._step_count,
            "success"       : bool(progress >= 1.0 and self.collisions == 0
                                   and self.near_misses == 0),
            "scenario"      : ("ADVERSARIAL" if self._adversarial
                               else ("OCCLUDED_PED" if self._occluded else "CLEAR")),
            "reward"        : reward,
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

        for v in self._vehicles:
            try:
                v.destroy()
            except Exception:
                pass
        self._vehicles = []

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
