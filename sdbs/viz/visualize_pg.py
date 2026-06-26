"""
visualize_pg.py
================================================================================
Replay visuel pygame : Greedy one-step vs S-DBS sur le "greedy trap"
(piÃ©ton occultÃ© qui surgit derriÃ¨re une voiture garÃ©e / un angle mort).

Deux modes :

  1.  DÃ‰MO (sans checkpoint) â€” utilise les stubs NumPy de sdbs_core.py
      (StubPolicy / StubWorldModel) sur MockDrivingEnv (9-dim). Aucune
      dÃ©pendance Ã  torch, tourne instantanÃ©ment, sert Ã  illustrer le
      mÃ©canisme du greedy trap.

        python visualize_pg.py

  2.  MODÃˆLE ENTRAÃŽNÃ‰ â€” charge un checkpoint produit par run_training.py
      (--save_every) et compare, sur EnhancedMockEnv (20-dim), le
      greedy_one_step et le SDBSPlanner.plan utilisant TON policy/world
      model entraÃ®nÃ©s (mÃªme rÃ©seau des deux cÃ´tÃ©s ; seule la stratÃ©gie de
      dÃ©cision change). C'est la vraie comparaison "mon modÃ¨le".

        python visualize_pg.py --checkpoint checkpoint.pt

ContrÃ´les :
  ESPACE      pause / lecture
  -> / Right  avancer d'un step (en pause)
  R           recommencer le mÃªme scÃ©nario (mÃªme seed)
  N           nouveau scÃ©nario (seed suivant)
  +/-         vitesse de lecture
  ESC / Q     quitter
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from sdbs.core.sdbs_core import (
    SDBSPlanner, PlannerConfig, BudgetConfig, DEFAULT_MANEUVERS, Maneuver,
    StubPolicy, StubWorldModel, MockDrivingEnv, mock_ego_xy, mock_mandated_action,
    TrainConfig,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1.  Ã‰pisode enregistrÃ© (une trajectoire complÃ¨te, image par image)
# ==============================================================================
class Frame:
    __slots__ = ("ego_x", "ego_y", "speed", "vru_dx", "vru_dy", "vru_visible",
                 "min_ttc", "reward", "cum_reward", "collision", "near_miss",
                 "light_red", "maneuver", "meta")

    def __init__(self, ego_x, ego_y, speed, vru_dx, vru_dy, vru_visible,
                 min_ttc, reward, cum_reward, collision, near_miss,
                 light_red, maneuver, meta):
        self.ego_x, self.ego_y, self.speed = ego_x, ego_y, speed
        self.vru_dx, self.vru_dy, self.vru_visible = vru_dx, vru_dy, vru_visible
        self.min_ttc = min_ttc
        self.reward, self.cum_reward = reward, cum_reward
        self.collision, self.near_miss = collision, near_miss
        self.light_red = light_red
        self.maneuver = maneuver
        self.meta = meta or {}


def run_episode_stub(seed: int, use_sdbs: bool) -> tuple[list[Frame], dict]:
    """Mode dÃ©mo : MockDrivingEnv (9-dim) + stubs NumPy. Pas besoin de torch."""
    env = MockDrivingEnv(seed=seed)
    policy = StubPolicy(seed=seed)
    wm = StubWorldModel()
    cfg_plan = PlannerConfig(horizon=5, beam_width=6, n_groups=3,
                              eta=0.4, lam=0.5, mu=0.3)
    cfg_budget = BudgetConfig(max_rollouts=120)
    planner = SDBSPlanner(policy, wm, cfg_plan, cfg_budget,
                          mandated_action_fn=mock_mandated_action)

    obs = env.reset(tier=2, n_vru=1, occlusion_prob=1.0, adversarial=False)
    frames: list[Frame] = []
    cum_r = 0.0
    for _ in range(env.max_steps):
        info = env.info()
        if use_sdbs:
            out = planner.plan(obs, info)
        else:
            out = planner.greedy_one_step(obs)
        action = out["action"]
        man = out.get("maneuver", -1)
        meta = out.get("meta", {})
        next_obs, r, done, step_info = env.step(action)
        cum_r += r
        frames.append(Frame(
            ego_x=float(obs[0]), ego_y=float(obs[1]), speed=float(obs[2]),
            vru_dx=float(env.ped_dx), vru_dy=0.0,
            vru_visible=float(obs[6]) > 0.5,
            min_ttc=float(step_info.get("min_ttc", 99.0)),
            reward=float(r), cum_reward=cum_r,
            collision=bool(step_info.get("collision", False)),
            near_miss=bool(step_info.get("near_miss", False)),
            light_red=float(obs[7]) > 0.5,
            maneuver=man, meta=meta,
        ))
        obs = next_obs
        if done:
            break
    summary = dict(
        success=bool(step_info.get("success", False)),
        collisions=env.collisions, near_misses=env.near_misses,
        total_reward=cum_r, steps=len(frames),
        scenario="OCCLUDED_PED (mock, n_vru=1, occ=1.0)",
    )
    return frames, summary


def run_episode_trained(seed: int, use_sdbs: bool, agent: dict,
                        device: str) -> tuple[list[Frame], dict]:
    """Mode modÃ¨le entraÃ®nÃ© : EnhancedMockEnv (20-dim) + ton policy/WM rÃ©els."""
    from sdbs.envs.enhanced_mock_env import (
        EnhancedMockEnv, enhanced_mandated_action, IDX_EGO_X, IDX_EGO_Y,
        IDX_SPEED, IDX_VRU0_DX, IDX_VRU0_DY, IDX_VRU0_VIS, IDX_TL,
    )
    env = EnhancedMockEnv(seed=seed)
    planner = agent["planner"]
    obs = env.reset(tier=2, n_vru=1, occlusion_prob=1.0, adversarial=False)
    frames: list[Frame] = []
    cum_r = 0.0
    step_info = {}
    for _ in range(env.max_steps):
        info = env.info()
        if use_sdbs:
            out = planner.plan(obs, info)
        else:
            out = planner.greedy_one_step(obs)
        action = out["action"]
        man = out.get("maneuver", -1)
        meta = out.get("meta", {})
        next_obs, r, done, step_info = env.step(action, man)
        cum_r += r
        frames.append(Frame(
            ego_x=float(obs[IDX_EGO_X]), ego_y=float(obs[IDX_EGO_Y]),
            speed=float(obs[IDX_SPEED]),
            vru_dx=float(obs[IDX_VRU0_DX]), vru_dy=float(obs[IDX_VRU0_DY]),
            vru_visible=float(obs[IDX_VRU0_VIS]) > 0.5,
            min_ttc=float(step_info.get("min_ttc", 99.0)),
            reward=float(r), cum_reward=cum_r,
            collision=bool(step_info.get("collision", False)),
            near_miss=bool(step_info.get("near_miss", False)),
            light_red=float(obs[IDX_TL]) > 0.5,
            maneuver=man, meta=meta,
        ))
        obs = next_obs
        if done:
            break
    summary = dict(
        success=bool(step_info.get("success", False)),
        collisions=env.collisions, near_misses=env.near_misses,
        total_reward=cum_r, steps=len(frames),
        scenario="OCCLUDED_PED (EnhancedMockEnv, n_vru=1, occ=1.0)",
    )
    return frames, summary


def load_trained_agent(checkpoint_path: str, device: str = "cpu") -> dict:
    """Construit un agent (policy+WM+ensemble+planner) et y charge un
    checkpoint produit par run_training.py / sdbs_dreamer.train()."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour charger un checkpoint entraÃ®nÃ©.")
    from sdbs.model.sdbs_dreamer import build_agent
    from sdbs.envs.enhanced_mock_env import (
        EnhancedMockEnv, enhanced_ego_xy, enhanced_mandated_action,
    )

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and ("wm" in raw or "policy" in raw):
        wm_state = raw.get("wm")
        policy_state = raw.get("policy")
        ensemble_state = raw.get("ensemble")
    else:
        # state_dict de WorldModel seul (--wm_checkpoint), pas de policy
        wm_state = raw if isinstance(raw, dict) else raw.state_dict()
        policy_state = ensemble_state = None

    dummy_env = EnhancedMockEnv(seed=0)
    cfg = TrainConfig()
    agent = build_agent(
        dummy_env, cfg, use_ensemble=(ensemble_state is not None),
        ego_xy_fn=enhanced_ego_xy, mandated_action_fn=enhanced_mandated_action,
        device=device, policy_state=policy_state, wm_state=wm_state,
        ensemble_state=ensemble_state,
    )
    return agent


# ==============================================================================
# 2.  Rendu pygame
# ==============================================================================
WIDTH, HEIGHT = 1280, 760
PANEL_W = WIDTH // 2
ROAD_TOP = 140
ROAD_H = 260
ROUTE_LEN_DEFAULT = 80.0  # m, mappÃ© sur la largeur du panneau

COL_BG = (18, 20, 26)
COL_ROAD = (45, 47, 55)
COL_LANE = (210, 210, 90)
COL_EGO = (70, 170, 255)
COL_EGO_TRAIL = (40, 90, 140)
COL_VRU_VISIBLE = (235, 70, 70)
COL_VRU_HIDDEN = (120, 120, 120)
COL_TEXT = (235, 235, 240)
COL_TEXT_DIM = (150, 150, 160)
COL_OK = (90, 220, 120)
COL_BAD = (235, 70, 70)
COL_WARN = (235, 190, 60)
COL_PANEL_SDBS = (30, 60, 50)
COL_PANEL_GREEDY = (60, 35, 35)

MANEUVER_NAMES = {int(m): m.name for m in Maneuver}


def x_to_px(x: float, route_len: float) -> int:
    margin = 60
    usable = PANEL_W - 2 * margin
    return margin + int(np.clip(x / route_len, 0.0, 1.0) * usable)


def draw_panel(surf, font, font_big, x0: int, title: str, title_color,
               frames: list[Frame], idx: int, route_len: float, flash_t: int):
    panel_rect = pygame.Rect(x0, 0, PANEL_W, HEIGHT)
    pygame.draw.rect(surf, COL_BG, panel_rect)

    # title bar
    pygame.draw.rect(surf, title_color, (x0, 0, PANEL_W, 44))
    label = font_big.render(title, True, (15, 15, 18))
    surf.blit(label, (x0 + 16, 8))

    # road
    road_rect = pygame.Rect(x0 + 20, ROAD_TOP, PANEL_W - 40, ROAD_H)
    pygame.draw.rect(surf, COL_ROAD, road_rect)
    pygame.draw.line(surf, COL_LANE, (x0 + 20, ROAD_TOP + ROAD_H // 2),
                     (x0 + PANEL_W - 20, ROAD_TOP + ROAD_H // 2), 2)
    # crosswalk marker (junction) at fixed fraction of route
    jx = x0 + 20 + x_to_px(30.0, route_len) - 20
    pygame.draw.rect(surf, (200, 200, 200), (jx, ROAD_TOP, 6, ROAD_H))

    if not frames:
        surf.blit(font.render("(aucune donnÃ©e)", True, COL_TEXT_DIM),
                  (x0 + 30, ROAD_TOP + ROAD_H + 20))
        return

    idx = min(idx, len(frames) - 1)
    f = frames[idx]
    cy = ROAD_TOP + ROAD_H // 2

    # trail (path already driven)
    pts = []
    for j in range(0, idx + 1):
        px = x0 + 20 + x_to_px(frames[j].ego_x, route_len)
        pts.append((px, cy))
    if len(pts) > 1:
        pygame.draw.lines(surf, COL_EGO_TRAIL, False, pts, 3)

    # VRU
    vru_x = f.ego_x + f.vru_dx
    vpx = x0 + 20 + x_to_px(vru_x, route_len)
    vpy = cy - int(np.clip(f.vru_dy, -2.5, 2.5) * 18)
    vcol = COL_VRU_VISIBLE if f.vru_visible else COL_VRU_HIDDEN
    pygame.draw.circle(surf, vcol, (vpx, vpy), 9)
    if not f.vru_visible:
        # occlusion halo to make "hidden" legible at a glance
        pygame.draw.circle(surf, (80, 80, 80), (vpx, vpy), 14, width=2)

    # red light indicator
    if f.light_red:
        pygame.draw.circle(surf, (235, 60, 60), (x0 + PANEL_W - 40, ROAD_TOP - 20), 8)

    # ego
    epx = x0 + 20 + x_to_px(f.ego_x, route_len)
    flash = (flash_t % 10) < 5
    ego_color = (COL_BAD if (f.collision and flash) else
                (COL_WARN if (f.near_miss and flash) else COL_EGO))
    pygame.draw.circle(surf, ego_color, (epx, cy), 11)
    pygame.draw.polygon(surf, ego_color, [(epx, cy - 16), (epx - 6, cy - 6), (epx + 6, cy - 6)])

    # ---- HUD ----
    hud_y = ROAD_TOP + ROAD_H + 18
    lines = [
        (f"step {idx+1}/{len(frames)}   speed={f.speed:.1f} m/s", COL_TEXT),
        (f"min_ttc={f.min_ttc:5.2f}s   reward={f.reward:+.2f}   "
         f"cum_reward={f.cum_reward:+.2f}", COL_TEXT),
    ]
    man_name = MANEUVER_NAMES.get(int(f.maneuver), "?")
    mode = f.meta.get("mode", "")
    mandated = f.meta.get("mandated", False)
    lines.append((f"maneuver={man_name}" + (f"  [{mode}]" if mode else "")
                 + ("  âš  SAFETY OVERRIDE" if mandated else ""),
                 COL_BAD if mandated else COL_TEXT_DIM))
    if "difficulty" in f.meta:
        lines.append((f"difficulty={f.meta['difficulty']:.2f}  "
                      f"B={f.meta.get('B')} G={f.meta.get('G')} H={f.meta.get('H')}  "
                      f"rollouts={f.meta.get('rollouts_used')}", COL_TEXT_DIM))
    status_bits = []
    if f.collision:
        status_bits.append(("COLLISION", COL_BAD))
    if f.near_miss:
        status_bits.append(("NEAR-MISS", COL_WARN))
    if not f.vru_visible:
        status_bits.append(("VRU OCCLUDED", COL_TEXT_DIM))

    for txt, col in lines:
        surf.blit(font.render(txt, True, col), (x0 + 30, hud_y))
        hud_y += 24
    hud_y += 6
    sx = x0 + 30
    for txt, col in status_bits:
        chip = font.render(f" {txt} ", True, (15, 15, 18))
        rect = chip.get_rect(topleft=(sx, hud_y))
        pygame.draw.rect(surf, col, rect.inflate(8, 4), border_radius=4)
        surf.blit(chip, rect)
        sx += rect.width + 18


def draw_summary_bar(surf, font, font_big, summaries: dict):
    y = HEIGHT - 70
    pygame.draw.rect(surf, (10, 10, 14), (0, y, WIDTH, 70))
    pygame.draw.line(surf, (60, 60, 70), (0, y), (WIDTH, y), 1)
    for i, (name, s) in enumerate(summaries.items()):
        x0 = i * PANEL_W + 20
        col = COL_OK if s["success"] else COL_TEXT
        txt = (f"{name}:  reward={s['total_reward']:+.2f}   "
               f"collisions={s['collisions']}   near_misses={s['near_misses']}   "
               f"success={s['success']}   steps={s['steps']}")
        surf.blit(font.render(txt, True, col), (x0, y + 12))
    hint = "ESPACE pause/lecture   â†/â†’ step   R rejouer   N nouveau seed   +/- vitesse   ESC quitter"
    surf.blit(font.render(hint, True, COL_TEXT_DIM), (20, y + 40))


# ==============================================================================
# 3.  Boucle principale
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description="Replay visuel Greedy vs S-DBS")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint entraÃ®nÃ© (.pt) Ã  comparer ; sans cet "
                        "argument, dÃ©mo instantanÃ©e avec les stubs NumPy.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=8,
                   help="Images par seconde en lecture automatique")
    args = p.parse_args()

    pygame.init()
    pygame.display.set_caption("S-DBS vs Greedy â€” occluded-crosswalk greedy trap")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)
    font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)

    agent = None
    use_trained = args.checkpoint is not None
    if use_trained:
        print(f"[viz] Chargement du checkpoint entraÃ®nÃ© : {args.checkpoint}")
        agent = load_trained_agent(args.checkpoint, args.device)
        route_len = 80.0
        run_fn = lambda seed, use_sdbs: run_episode_trained(seed, use_sdbs, agent, args.device)
    else:
        print("[viz] Aucun --checkpoint fourni -> mode dÃ©mo (stubs NumPy, "
              "MockDrivingEnv, greedy trap illustratif).")
        route_len = 60.0
        run_fn = lambda seed, use_sdbs: run_episode_stub(seed, use_sdbs)

    def load_seed(seed: int):
        g_frames, g_summary = run_fn(seed, False)
        s_frames, s_summary = run_fn(seed, True)
        return g_frames, g_summary, s_frames, s_summary

    seed = args.seed
    g_frames, g_summary, s_frames, s_summary = load_seed(seed)

    idx = 0
    playing = True
    fps_play = args.fps
    flash_t = 0
    max_len = max(len(g_frames), len(s_frames), 1)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    playing = False
                    idx = min(idx + 1, max_len - 1)
                elif event.key == pygame.K_LEFT:
                    playing = False
                    idx = max(idx - 1, 0)
                elif event.key == pygame.K_r:
                    g_frames, g_summary, s_frames, s_summary = load_seed(seed)
                    idx = 0
                    max_len = max(len(g_frames), len(s_frames), 1)
                elif event.key == pygame.K_n:
                    seed += 1
                    g_frames, g_summary, s_frames, s_summary = load_seed(seed)
                    idx = 0
                    max_len = max(len(g_frames), len(s_frames), 1)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps_play = min(30, fps_play + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps_play = max(1, fps_play - 1)

        if playing:
            idx += 1
            if idx >= max_len:
                idx = max_len - 1
                playing = False
        flash_t += 1

        screen.fill(COL_BG)
        draw_panel(screen, font, font_big, 0, "GREEDY (one-step)", COL_PANEL_GREEDY[::-1]
                  if False else (220, 120, 110), g_frames, idx, route_len, flash_t)
        draw_panel(screen, font, font_big, PANEL_W, "S-DBS (dreaming planner)",
                  (120, 220, 160), s_frames, idx, route_len, flash_t)
        pygame.draw.line(screen, (60, 60, 70), (PANEL_W, 0), (PANEL_W, HEIGHT - 70), 1)
        draw_summary_bar(screen, font, font_big,
                         {"GREEDY": g_summary, "S-DBS": s_summary})

        title_extra = f"  seed={seed}" + ("  [modÃ¨le entraÃ®nÃ©]" if use_trained else "  [dÃ©mo stub]")
        pygame.display.set_caption(f"S-DBS vs Greedy{title_extra}  â€”  "
                                   f"{'lecture' if playing else 'pause'} @ {fps_play} fps")

        pygame.display.flip()
        clock.tick(fps_play if playing else 30)

    pygame.quit()


if __name__ == "__main__":
    main()
