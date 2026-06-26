"""
carla_obs_utils.py
================================================================================
Extraction d'observation depuis un monde CARLA réel, dans EXACTEMENT le même
layout 20-dim que EnhancedMockEnv (cf. enhanced_mock_env.py, section 0). C'est
ce qui permet de pré-entraîner le WorldModel sur des logs CARLA réels puis de
charger ces poids via build_agent(wm_state=...) sans rien changer côté
sdbs_core.py / sdbs_dreamer.py.

Ce module ne contient AUCUNE logique de policy/reward/RL : c'est uniquement de
l'extraction de features depuis l'API CARLA, utilisé par
collect_carla_driving_data.py pendant une conduite pilotée par le Traffic
Manager (autopilote), donc sans aucun besoin d'agent entraîné.

⚠️ Nécessite le module `carla` (PythonAPI du serveur CARLA) et un serveur
CARLA lancé. Ce fichier n'est pas exécutable seul -- il est importé par
collect_carla_driving_data.py en mode réel (pas en --mock).

Si tu changes le layout dans enhanced_mock_env.py, mets à jour ce fichier en
même temps : c'est la même structure, dupliquée volontairement pour ne pas
faire dépendre EnhancedMockEnv (NumPy pur, pas de CARLA) de ce module.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Réplique du layout de enhanced_mock_env.py -- garder synchronisé.
(IDX_EGO_X, IDX_EGO_Y, IDX_SPEED, IDX_ACCEL, IDX_HEAD_ERR,
 IDX_LANE_OFF, IDX_DIST_JCT, IDX_TL, IDX_STOP, IDX_ROW,
 IDX_MIN_TTC, IDX_OCC,
 IDX_VRU0_DX, IDX_VRU0_DY, IDX_VRU0_VIS,
 IDX_VRU1_DX, IDX_VRU1_VIS,
 IDX_VRU2_DX, IDX_VRU2_VIS,
 IDX_PROGRESS) = range(20)

CARLA_STATE_DIM = 20
CARLA_ACTION_DIM = 3   # [steer, throttle, brake] -- même squelette que EnhancedMockEnv
VRU_DETECTION_RADIUS = 40.0   # m, ignore les piétons plus loin que ça
OCCLUSION_RAY_STEP = 1.0      # m, résolution du raycast simplifié d'occlusion


@dataclass
class RouteTracker:
    """Suit la progression de l'ego le long d'une liste de waypoints CARLA.
    Très simple par design : on ne fait QUE de la collecte de données, pas du
    contrôle, donc pas besoin d'un planner de route sophistiqué ici."""
    waypoints: list   # liste de carla.Waypoint, dans l'ordre de la route
    route_len_m: float

    def progress_and_pos(self, ego_location) -> tuple[float, int]:
        """Retourne (fraction [0,1] parcourue, index du waypoint le plus
        proche) en projetant la position de l'ego sur la polyligne de route."""
        best_d, best_i = float("inf"), 0
        for i, wp in enumerate(self.waypoints):
            d = ego_location.distance(wp.transform.location)
            if d < best_d:
                best_d, best_i = d, i
        frac = best_i / max(1, len(self.waypoints) - 1)
        return float(np.clip(frac, 0.0, 1.0)), best_i

    def dist_to_next_junction(self, idx: int) -> float:
        """Distance approx. (en nb de waypoints * pas moyen) jusqu'au premier
        waypoint marqué `is_junction` après l'index donné."""
        step = 2.0  # m, pas typique entre waypoints générés à intervalle fixe
        for j in range(idx, len(self.waypoints)):
            if self.waypoints[j].is_junction:
                return (j - idx) * step
        return 999.0


def _relative_xy(ego_transform, target_location) -> tuple[float, float]:
    """Position de `target_location` dans le repère local de l'ego
    (x = longitudinal devant l'ego, y = latéral, + = à droite)."""
    dx = target_location.x - ego_transform.location.x
    dy = target_location.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    local_x = dx * cos_y - dy * sin_y
    local_y = dx * sin_y + dy * cos_y
    return local_x, local_y


def _is_occluded(world, ego_transform, walker_location, vehicles) -> bool:
    """Test d'occlusion simplifié : on tire un rayon discret entre l'ego et le
    piéton, et on regarde si un véhicule (autre que l'ego) se trouve à moins
    de ~2.2 m (demi-largeur de voiture + marge) de la ligne de visée à un
    point intermédiaire. C'est une approximation volontairement simple --
    suffisante pour générer un signal d'occlusion réaliste pour le
    pré-entraînement, sans dépendre d'un capteur semantic-LiDAR spécifique.

    Si ton serveur CARLA expose `world.cast_ray` (>= 0.9.14) ou un capteur
    semantic LiDAR, remplace cette fonction par une requête réelle pour un
    signal plus fidèle -- l'interface (retourne un bool) reste identique.
    """
    ex, ey = ego_transform.location.x, ego_transform.location.y
    wx, wy = walker_location.x, walker_location.y
    dist = math.hypot(wx - ex, wy - ey)
    if dist < 1e-3:
        return False
    n_steps = max(1, int(dist / OCCLUSION_RAY_STEP))
    for v in vehicles:
        vloc = v.get_location()
        for k in range(1, n_steps):
            t = k / n_steps
            px = ex + t * (wx - ex)
            py = ey + t * (wy - ey)
            if math.hypot(vloc.x - px, vloc.y - py) < 2.2:
                # le véhicule v est assis à peu près sur la ligne de visée
                # entre l'ego et le piéton -> on considère le piéton occulté
                return True
    return False


def extract_observation(world, ego_vehicle, route: RouteTracker,
                        walkers: list, vehicles: list,
                        traffic_lights: list) -> np.ndarray:
    """Construit le vecteur d'observation 20-dim pour l'état CARLA courant.

    Paramètres
    ----------
    world          : carla.World
    ego_vehicle    : carla.Vehicle (l'acteur piloté par le Traffic Manager)
    route          : RouteTracker construit une fois au début de l'épisode
    walkers        : liste de carla.Walker vivants dans la scène
    vehicles       : liste de carla.Vehicle AUTRES que l'ego (pour l'occlusion
                     et la détection de feu/priorité)
    traffic_lights : liste de carla.TrafficLight de la carte

    Retourne np.float32[20], dans le même ordre que ENHANCED_STATE_DIM.
    """
    s = np.zeros(CARLA_STATE_DIM, dtype=np.float32)

    tf = ego_vehicle.get_transform()
    vel = ego_vehicle.get_velocity()
    ctrl = ego_vehicle.get_control()
    speed = math.hypot(vel.x, vel.y)

    frac, wp_idx = route.progress_and_pos(tf.location)
    dist_jct = route.dist_to_next_junction(wp_idx)

    s[IDX_EGO_X] = frac * route.route_len_m       # position longitudinale "route-relative"
    s[IDX_EGO_Y] = 0.0                            # offset latéral calculé séparément ci-dessous
    s[IDX_SPEED] = speed
    s[IDX_ACCEL] = float(ctrl.throttle) * 3.0 - float(ctrl.brake) * 6.0  # proxy, cohérent avec EnhancedMockEnv
    s[IDX_DIST_JCT] = dist_jct
    s[IDX_PROGRESS] = frac

    # lane offset / heading error via le waypoint le plus proche
    wp = route.waypoints[wp_idx]
    lane_yaw = math.radians(wp.transform.rotation.yaw)
    heading_err = math.radians(tf.rotation.yaw) - lane_yaw
    heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))
    lx, ly = _relative_xy(wp.transform, tf.location)
    s[IDX_HEAD_ERR] = float(np.clip(heading_err, -0.5, 0.5))
    s[IDX_LANE_OFF] = float(ly)

    # feu de signalisation le plus proche devant l'ego
    s[IDX_TL] = 0.0
    s[IDX_STOP] = 0.0
    s[IDX_ROW] = 1.0
    best_tl_d = float("inf")
    for tl in traffic_lights:
        d = tf.location.distance(tl.get_location())
        if d < best_tl_d and d < 50.0:
            best_tl_d = d
            state = tl.get_state()
            # carla.TrafficLightState: Red=0, Yellow=1, Green=2, Off=3, Unknown=4
            if str(state).lower().endswith("red"):
                s[IDX_TL] = 1.0
                s[IDX_ROW] = 0.0
            elif str(state).lower().endswith("yellow"):
                s[IDX_TL] = 0.5

    # piétons (VRU) : on garde les 3 plus proches dans un cône avant l'ego
    ped_candidates = []
    for w in walkers:
        wloc = w.get_location()
        lx_w, ly_w = _relative_xy(tf, wloc)
        if lx_w < -5.0 or lx_w > VRU_DETECTION_RADIUS:
            continue  # derrière l'ego ou trop loin
        dist = math.hypot(lx_w, ly_w)
        occluded = _is_occluded(world, tf, wloc, vehicles)
        ped_candidates.append((dist, lx_w, ly_w, occluded))
    ped_candidates.sort(key=lambda c: c[0])

    dx_slots = [IDX_VRU0_DX, IDX_VRU1_DX, IDX_VRU2_DX]
    vis_slots = [IDX_VRU0_VIS, IDX_VRU1_VIS, IDX_VRU2_VIS]
    min_ttc = 99.0
    for i in range(3):
        if i < len(ped_candidates):
            dist, lx_w, ly_w, occluded = ped_candidates[i]
            s[dx_slots[i]] = float(np.clip(lx_w, -5, 99))
            s[vis_slots[i]] = 0.0 if occluded else 1.0
            if i == 0:
                s[IDX_VRU0_DY] = float(ly_w)
            if not occluded and abs(ly_w) < 2.5 and lx_w > 0:
                closing = max(speed, 0.1)
                min_ttc = min(min_ttc, lx_w / closing)
        else:
            s[dx_slots[i]] = 99.0
            s[vis_slots[i]] = 0.0
    s[IDX_MIN_TTC] = float(min_ttc)
    s[IDX_OCC] = float(any(c[3] for c in ped_candidates[:3])) if ped_candidates else 0.0

    return s


def get_applied_action(ego_vehicle) -> np.ndarray:
    """Action [steer, throttle, brake] EFFECTIVEMENT appliquée par le Traffic
    Manager à ce step -- c'est ce qu'on enregistre comme `action` dans le
    buffer, puisque pendant la collecte personne n'exécute la politique:
    le WM apprend "à quelle transition correspond cette commande", peu
    importe qui l'a émise."""
    ctrl = ego_vehicle.get_control()
    return np.array([float(ctrl.steer), float(ctrl.throttle), float(ctrl.brake)],
                    dtype=np.float32)