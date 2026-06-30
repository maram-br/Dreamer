# -*- coding: utf-8 -*-
"""
visualize_pg.py
================================================================================
Replay visuel pygame du planificateur S-DBS, avec un VRAI modele entraine
(policy + world model + planner charges depuis un checkpoint .pt).

Deux modes :

  --mode demo   Deux panneaux cote a cote, GREEDY vs S-DBS, meme seed,
                meme depart. Design epure pour montrer la difference de
                comportement (le greedy trap : pieton occlus qui surgit).

  --mode debug  Un seul panneau pleine largeur, avec en plus le detail des
                5 groupes S-DBS (manoeuvre, score, role) au pas courant --
                utile pour comprendre POURQUOI le planner a choisi telle
                manoeuvre, pas seulement CE qu'il a choisi.

Usage :
    python visualize_pygame.py --checkpoint mon_checkpoint.pt --mode demo
    python visualize_pygame.py --checkpoint mon_checkpoint.pt --mode debug --seed 12

Sans --checkpoint, charge un agent a poids aleatoires (utile pour verifier
que le pipeline tourne, mais le comportement ne reflete rien d'entraine).

Controles (les deux modes) :
  ESPACE       pause / lecture
  -> / <-      avancer / reculer d'un pas (en pause)
  R            rejouer le meme scenario (meme seed)
  N            nouveau scenario (seed suivant)
  +/-          vitesse de lecture
  ESC / Q      quitter
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

from sdbs.core.sdbs_core import Maneuver
from sdbs.viz.replay_data import (
    Frame, EpisodeResult, MANEUVER_NAMES,
    load_trained_agent, run_comparison, run_episode,
)

try:
    from sdbs.model.sdbs_dreamer import build_agent, TrainConfig
    from sdbs.envs.enhanced_mock_env import EnhancedMockEnv, enhanced_ego_xy, enhanced_mandated_action
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ==============================================================================
# 1.  Palette -- design system coherent (gris neutre + accents semantiques)
# ==============================================================================
# Surfaces
COL_BG          = (15,  17,  21)
COL_SURFACE     = (23,  26,  31)
COL_SURFACE_HI  = (30,  34,  41)
COL_BORDER      = (42,  46,  54)

# Texte
COL_TEXT        = (224, 226, 230)
COL_TEXT_DIM    = (138, 143, 153)
COL_TEXT_FAINT  = (90,  94,  102)

# Route
COL_ROAD        = (33,  36,  42)
COL_LANE        = (70,  76,  88)
COL_CROSSWALK   = (180, 184, 192)

# Semantique (memes accents que les diagrammes : teal=ok, coral=danger, amber=warning)
COL_TEAL        = (29,  158, 117)
COL_TEAL_LIGHT  = (93,  202, 165)
COL_CORAL       = (216, 90,  48)
COL_CORAL_LIGHT = (240, 153, 123)
COL_AMBER       = (186, 117, 23)
COL_AMBER_LIGHT = (250, 199, 117)
COL_BLUE        = (55,  138, 221)
COL_BLUE_LIGHT  = (133, 183, 235)
COL_PURPLE      = (127, 119, 221)

COL_EGO         = COL_BLUE
COL_EGO_TRAIL   = (40,  60,  90)
COL_PED_VISIBLE = COL_CORAL
COL_PED_HIDDEN  = (80,  84,  92)

MANEUVER_COLOR = {
    int(Maneuver.FOLLOW_LANE):       COL_BLUE,
    int(Maneuver.YIELD_CREEP):       COL_TEAL,
    int(Maneuver.LANE_CHANGE_LEFT):  COL_AMBER,
    int(Maneuver.LANE_CHANGE_RIGHT): COL_PURPLE,
    int(Maneuver.OVERTAKE_CYCLIST):  COL_PURPLE,
    int(Maneuver.HARD_STOP):         COL_CORAL,
}

GROUP_COLORS = [COL_BLUE, COL_TEAL, COL_PURPLE, COL_CORAL, COL_AMBER]


def maneuver_color(m: int) -> tuple:
    return MANEUVER_COLOR.get(int(m), COL_TEXT_DIM)


# ==============================================================================
# 2.  Layout
# ==============================================================================
DEMO_W, DEMO_H   = 1280, 720
DEBUG_W, DEBUG_H = 1280, 820

ROAD_H_FRAC = 0.34   # fraction de hauteur de panneau occupee par la route


# ==============================================================================
# 3.  Helpers de dessin
# ==============================================================================
def rounded_panel(surf, rect: pygame.Rect, color=COL_SURFACE,
                  border=COL_BORDER, radius=10):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, width=1, border_radius=radius)


def text(surf, font, s: str, pos, color=COL_TEXT):
    surf.blit(font.render(s, True, color), pos)


def x_to_px(x: float, route_len: float, x0: int, width: int, margin: int = 50) -> int:
    usable = width - 2 * margin
    return x0 + margin + int(np.clip(x / route_len, 0.0, 1.0) * usable)


def draw_chip(surf, font, label: str, pos, fg, bg):
    pad_x, pad_y = 8, 4
    txt_surf = font.render(label, True, fg)
    rect = txt_surf.get_rect(topleft=pos).inflate(pad_x * 2, pad_y * 2)
    pygame.draw.rect(surf, bg, rect, border_radius=5)
    surf.blit(txt_surf, (rect.x + pad_x, rect.y + pad_y))
    return rect.width


# ==============================================================================
# 4.  Dessin de la route + acteurs (partage entre les deux modes)
# ==============================================================================
def draw_road(surf, font_sm, rect: pygame.Rect, frame: Frame, route_len: float,
             flash_on: bool, label: str = ""):
    """Dessine la route, le pieton, l'ego, dans `rect`. Retourne le y du centre
    de voie (utile pour aligner d'autres elements)."""
    road_h = int(rect.height * ROAD_H_FRAC)
    road_rect = pygame.Rect(rect.x, rect.y, rect.width, road_h)
    pygame.draw.rect(surf, COL_ROAD, road_rect, border_radius=8)

    cy = road_rect.centery
    pygame.draw.line(surf, COL_LANE, (road_rect.x + 16, cy),
                     (road_rect.right - 16, cy), 2)

    # passage pieton (position fixe a 30% de la route)
    jx = x_to_px(route_len * 0.5, route_len, road_rect.x, road_rect.width)
    for i in range(5):
        sy = road_rect.y + 10 + i * (road_h - 20) // 4
        pygame.draw.line(surf, COL_CROSSWALK, (jx - 12, sy), (jx + 12, sy), 3)

    # feu rouge (indicateur compact en coin)
    if frame.light_red:
        pygame.draw.circle(surf, COL_CORAL, (road_rect.right - 18, road_rect.y + 14), 6)

    # trajectoire (ego_x est deja absolu ; on centre la fenetre sur l'ego)
    epx = x_to_px(frame.ego_x, route_len, road_rect.x, road_rect.width)

    # pieton -- position relative a l'ego (ped_dx = distance devant)
    ped_x_world = frame.ego_x + frame.ped_dx
    ppx = x_to_px(ped_x_world, route_len, road_rect.x, road_rect.width)
    ppy = cy - 16
    if road_rect.x < ppx < road_rect.right:
        pcol = COL_PED_VISIBLE if frame.ped_visible else COL_PED_HIDDEN
        pygame.draw.circle(surf, pcol, (ppx, ppy), 7)
        pygame.draw.line(surf, pcol, (ppx, ppy + 5), (ppx, ppy + 16), 2)
        if not frame.ped_visible:
            pygame.draw.circle(surf, COL_TEXT_FAINT, (ppx, ppy), 11, width=1)

    # ego (triangle pointant vers la droite = sens de marche)
    flash = flash_on
    ego_col = (COL_CORAL if (frame.collision and flash) else
              (COL_AMBER if (frame.near_miss and flash) else COL_EGO))
    pygame.draw.circle(surf, ego_col, (epx, cy), 9)
    pygame.draw.polygon(surf, ego_col, [
        (epx, cy - 13), (epx - 5, cy - 5), (epx + 5, cy - 5)])

    if label:
        text(surf, font_sm, label, (rect.x + 12, rect.y + 8), COL_TEXT_DIM)

    return road_rect


def draw_status_chips(surf, font_sm, x0: int, y0: int, frame: Frame) -> int:
    """Chips d'etat (collision / near-miss / occlusion). Retourne le y suivant."""
    sx = x0
    if frame.collision:
        sx += draw_chip(surf, font_sm, "COLLISION", (sx, y0),
                        (20, 8, 6), COL_CORAL) + 8
    if frame.near_miss:
        sx += draw_chip(surf, font_sm, "NEAR-MISS", (sx, y0),
                        (24, 16, 4), COL_AMBER) + 8
    if not frame.ped_visible:
        draw_chip(surf, font_sm, "VRU OCCLUDED", (sx, y0),
                  COL_TEXT, COL_SURFACE_HI)
    return y0 + 26


def draw_hud(surf, font, font_sm, x0: int, y0: int, idx: int, n: int, frame: Frame):
    """Bloc texte : step, vitesse, TTC, reward, manoeuvre, [safety override]."""
    text(surf, font, f"pas {idx + 1}/{n}   vitesse {frame.speed:.1f} m/s", (x0, y0), COL_TEXT)
    y0 += 22
    ttc_col = COL_CORAL if frame.min_ttc < 1.5 else (COL_AMBER if frame.min_ttc < 3.0 else COL_TEXT_DIM)
    text(surf, font_sm, f"TTC min {frame.min_ttc:5.2f}s", (x0, y0), ttc_col)
    text(surf, font_sm, f"reward {frame.reward:+.2f}   cumul {frame.cum_reward:+.2f}",
        (x0 + 130, y0), COL_TEXT_DIM)
    y0 += 22

    man_name = MANEUVER_NAMES.get(int(frame.maneuver), "?")
    mcol = maneuver_color(frame.maneuver)
    mandated = frame.meta.get("mandated", False)
    mode = frame.meta.get("mode", "")
    suffix = "  SAFETY OVERRIDE" if mandated else ""
    text(surf, font, f"-> {man_name}" + (f"  [{mode}]" if mode else "") + suffix,
        (x0, y0), COL_CORAL if mandated else mcol)
    y0 += 24

    if "difficulty" in frame.meta:
        text(surf, font_sm,
            f"difficulte {frame.meta['difficulty']:.2f}   "
            f"B={frame.meta.get('B')} G={frame.meta.get('G')} H={frame.meta.get('H')}   "
            f"rollouts={frame.meta.get('rollouts_used')}",
            (x0, y0), COL_TEXT_FAINT)
        y0 += 20

    y0 = draw_status_chips(surf, font_sm, x0, y0 + 4, frame)
    return y0


# ==============================================================================
# 5.  MODE DEMO -- deux panneaux cote a cote
# ==============================================================================
def draw_summary_bar(surf, font, font_sm, w: int, h: int,
                     g_res: EpisodeResult, s_res: EpisodeResult,
                     fps_play: int, playing: bool, seed: int):
    bar_h = 64
    rect = pygame.Rect(0, h - bar_h, w, bar_h)
    rounded_panel(surf, rect, color=COL_SURFACE, radius=0)
    pygame.draw.line(surf, COL_BORDER, (0, h - bar_h), (w, h - bar_h), 1)

    half = w // 2
    for i, (name, res, col) in enumerate([("GREEDY", g_res, COL_CORAL_LIGHT),
                                          ("S-DBS", s_res, COL_TEAL_LIGHT)]):
        x0 = i * half + 20
        ok_col = COL_TEAL_LIGHT if res.success else COL_TEXT_DIM
        text(surf, font, name, (x0, h - bar_h + 8), col)
        text(surf, font_sm,
            f"reward {res.total_reward:+.2f}   collisions {res.collisions}   "
            f"near-miss {res.near_misses}   succes {res.success}   pas {res.steps}",
            (x0 + 70, h - bar_h + 10), ok_col)

    hint = (f"seed={seed}   ESPACE pause/lecture   fleches: pas a pas   "
           f"R rejouer   N nouveau   +/- vitesse ({fps_play}fps)   ESC quitter")
    text(surf, font_sm, hint, (20, h - bar_h + 36), COL_TEXT_FAINT)


def run_demo_mode(agent: dict, args):
    pygame.init()
    pygame.display.set_caption("S-DBS vs Greedy -- demo")
    screen = pygame.display.set_mode((DEMO_W, DEMO_H))
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("Helvetica,Arial", 13)
    font    = pygame.font.SysFont("Helvetica,Arial", 15)
    font_lg = pygame.font.SysFont("Helvetica,Arial", 18, bold=True)

    route_len = 60.0
    panel_w = DEMO_W // 2

    def load(seed):
        return run_comparison(agent, seed, tier=2, n_vru=1, occlusion_prob=1.0)

    seed = args.seed
    g_res, s_res = load(seed)
    idx, playing, fps_play, flash_t = 0, True, args.fps, 0
    max_len = max(len(g_res.frames), len(s_res.frames), 1)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_RIGHT:
                    playing = False; idx = min(idx + 1, max_len - 1)
                elif ev.key == pygame.K_LEFT:
                    playing = False; idx = max(idx - 1, 0)
                elif ev.key == pygame.K_r:
                    g_res, s_res = load(seed); idx = 0
                    max_len = max(len(g_res.frames), len(s_res.frames), 1)
                elif ev.key == pygame.K_n:
                    seed += 1; g_res, s_res = load(seed); idx = 0
                    max_len = max(len(g_res.frames), len(s_res.frames), 1)
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps_play = min(30, fps_play + 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps_play = max(1, fps_play - 1)

        if playing:
            idx = min(idx + 1, max_len - 1)
            if idx == max_len - 1:
                playing = False
        flash_t += 1
        flash_on = (flash_t % 10) < 5

        screen.fill(COL_BG)

        for i, (label, res, title_col) in enumerate(
            [("GREEDY  (one-step)", g_res, COL_CORAL_LIGHT),
             ("S-DBS  (dreaming planner)", s_res, COL_TEAL_LIGHT)]):
            x0 = i * panel_w
            panel_rect = pygame.Rect(x0 + 10, 10, panel_w - 20, DEMO_H - 96)
            rounded_panel(screen, panel_rect)
            text(screen, font_lg, label, (panel_rect.x + 16, panel_rect.y + 12), title_col)

            road_area = pygame.Rect(panel_rect.x + 14, panel_rect.y + 46,
                                    panel_rect.width - 28, int(panel_rect.height * ROAD_H_FRAC))
            if res.frames:
                fi = min(idx, len(res.frames) - 1)
                frame = res.frames[fi]
                # trail
                pts = []
                for j in range(fi + 1):
                    px = x_to_px(res.frames[j].ego_x, route_len, road_area.x, road_area.width)
                    pts.append((px, road_area.centery))
                if len(pts) > 1:
                    pygame.draw.lines(screen, COL_EGO_TRAIL, False, pts, 3)
                draw_road(screen, font_sm, road_area, frame, route_len, flash_on)
                draw_hud(screen, font, font_sm, panel_rect.x + 16,
                        road_area.bottom + 18, fi, len(res.frames), frame)
            else:
                text(screen, font_sm, "(aucune donnee)", (panel_rect.x + 16, road_area.bottom + 18),
                    COL_TEXT_FAINT)

        pygame.draw.line(screen, COL_BORDER, (panel_w, 10), (panel_w, DEMO_H - 96), 1)
        draw_summary_bar(screen, font, font_sm, DEMO_W, DEMO_H, g_res, s_res, fps_play, playing, seed)

        pygame.display.set_caption(
            f"S-DBS vs Greedy -- seed={seed}  {'lecture' if playing else 'pause'} @ {fps_play}fps")
        pygame.display.flip()
        clock.tick(fps_play if playing else 30)

    pygame.quit()


# ==============================================================================
# 6.  MODE DEBUG -- un panneau, plus le detail des 5 groupes S-DBS
# ==============================================================================
def draw_groups_panel(surf, font, font_sm, rect: pygame.Rect, frame: Frame):
    rounded_panel(surf, rect)
    text(surf, font, "S-DBS -- groupes retenus a ce pas", (rect.x + 14, rect.y + 12), COL_TEXT)

    retained = frame.meta.get("retained", [])
    role = frame.meta.get("role", "")
    score = frame.meta.get("score", None)

    y = rect.y + 42
    if not retained:
        text(surf, font_sm, "(safety pre-pass actif -- pas de beam ce pas)",
            (rect.x + 14, y), COL_TEXT_FAINT)
        return

    header = f"{'manoeuvre':<22}{'role gagnant':<16}"
    text(surf, font_sm, header, (rect.x + 14, y), COL_TEXT_FAINT)
    y += 20
    pygame.draw.line(surf, COL_BORDER, (rect.x + 12, y), (rect.right - 12, y), 1)
    y += 8

    chosen_m = int(frame.maneuver)
    for i, m in enumerate(retained):
        m = int(m)
        is_chosen = (m == chosen_m)
        col = GROUP_COLORS[i % len(GROUP_COLORS)]
        row_h = 26
        if is_chosen:
            hl = pygame.Rect(rect.x + 8, y - 3, rect.width - 16, row_h)
            pygame.draw.rect(surf, COL_SURFACE_HI, hl, border_radius=5)
        name = MANEUVER_NAMES.get(m, "?")
        text(surf, font_sm, name, (rect.x + 16, y), col)
        marker = " <- choisie" if is_chosen else ""
        text(surf, font_sm, marker, (rect.x + 220, y), COL_TEAL_LIGHT if is_chosen else COL_TEXT_FAINT)
        y += row_h

    y += 6
    pygame.draw.line(surf, COL_BORDER, (rect.x + 12, y), (rect.right - 12, y), 1)
    y += 12

    text(surf, font_sm, f"role gagnant: {role}", (rect.x + 14, y), COL_TEXT_DIM)
    y += 20
    if score is not None:
        text(surf, font_sm, f"score final: {score:+.3f}", (rect.x + 14, y), COL_TEXT_DIM)
        y += 20
    for k in ("novelty", "surprise", "gain", "ser", "div_penalty"):
        if k in frame.meta:
            text(surf, font_sm, f"{k}: {frame.meta[k]:+.3f}", (rect.x + 14, y), COL_TEXT_FAINT)
            y += 18


def draw_risk_strip(surf, font_sm, rect: pygame.Rect, frames: list[Frame], idx: int):
    """Mini-graphique TTC au fil du temps -- repere visuel de la tension."""
    rounded_panel(surf, rect)
    text(surf, font_sm, "TTC min au fil du temps", (rect.x + 12, rect.y + 8), COL_TEXT_DIM)
    if len(frames) < 2:
        return
    plot = pygame.Rect(rect.x + 12, rect.y + 28, rect.width - 24, rect.height - 40)
    pygame.draw.line(surf, COL_BORDER, (plot.x, plot.bottom), (plot.right, plot.bottom), 1)

    cap = 6.0
    pts = []
    for j, f in enumerate(frames):
        px = plot.x + int(j / max(1, len(frames) - 1) * plot.width)
        v = min(f.min_ttc, cap) / cap
        py = plot.bottom - int(v * plot.height)
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(surf, COL_BLUE_LIGHT, False, pts, 2)
    # seuil de danger (TTC=1.5s)
    danger_y = plot.bottom - int(min(1.5, cap) / cap * plot.height)
    pygame.draw.line(surf, COL_CORAL, (plot.x, danger_y), (plot.right, danger_y), 1)
    # curseur position actuelle
    if idx < len(pts):
        pygame.draw.circle(surf, COL_TEXT, pts[min(idx, len(pts) - 1)], 4)


def run_debug_mode(agent: dict, args):
    pygame.init()
    pygame.display.set_caption("S-DBS -- mode debug")
    screen = pygame.display.set_mode((DEBUG_W, DEBUG_H))
    clock = pygame.time.Clock()
    font_sm = pygame.font.SysFont("Helvetica,Arial", 13)
    font    = pygame.font.SysFont("Helvetica,Arial", 15)
    font_lg = pygame.font.SysFont("Helvetica,Arial", 18, bold=True)

    route_len = 60.0

    def load(seed):
        return run_episode(agent, seed, use_sdbs=True, tier=2, n_vru=1, occlusion_prob=1.0)

    seed = args.seed
    res = load(seed)
    idx, playing, fps_play, flash_t = 0, True, args.fps, 0
    max_len = max(len(res.frames), 1)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_RIGHT:
                    playing = False; idx = min(idx + 1, max_len - 1)
                elif ev.key == pygame.K_LEFT:
                    playing = False; idx = max(idx - 1, 0)
                elif ev.key == pygame.K_r:
                    res = load(seed); idx = 0; max_len = max(len(res.frames), 1)
                elif ev.key == pygame.K_n:
                    seed += 1; res = load(seed); idx = 0; max_len = max(len(res.frames), 1)
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps_play = min(30, fps_play + 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps_play = max(1, fps_play - 1)

        if playing:
            idx = min(idx + 1, max_len - 1)
            if idx == max_len - 1:
                playing = False
        flash_t += 1
        flash_on = (flash_t % 10) < 5

        screen.fill(COL_BG)

        # bandeau titre
        text(screen, font_lg, f"S-DBS dreaming planner -- seed={seed}", (16, 14), COL_TEXT)
        text(screen, font_sm,
            f"{res.scenario}   succes={res.success}   "
            f"collisions={res.collisions}   near-miss={res.near_misses}",
            (16, 40), COL_TEXT_DIM)

        # route pleine largeur
        road_panel = pygame.Rect(12, 64, DEBUG_W - 24, 220)
        rounded_panel(screen, road_panel)
        road_area = pygame.Rect(road_panel.x + 14, road_panel.y + 14,
                                road_panel.width - 28, road_panel.height - 28)
        if res.frames:
            fi = min(idx, len(res.frames) - 1)
            frame = res.frames[fi]
            pts = []
            for j in range(fi + 1):
                px = x_to_px(res.frames[j].ego_x, route_len, road_area.x, road_area.width)
                pts.append((px, road_area.centery))
            if len(pts) > 1:
                pygame.draw.lines(screen, COL_EGO_TRAIL, False, pts, 3)
            draw_road(screen, font_sm, road_area, frame, route_len, flash_on)

            # bandeau bas gauche : HUD compact
            hud_rect = pygame.Rect(12, 296, 620, 200)
            rounded_panel(screen, hud_rect)
            text(screen, font, "Etat ego", (hud_rect.x + 14, hud_rect.y + 12), COL_TEXT)
            draw_hud(screen, font, font_sm, hud_rect.x + 14, hud_rect.y + 42,
                    fi, len(res.frames), frame)

            # groupes S-DBS a droite
            groups_rect = pygame.Rect(644, 296, DEBUG_W - 656, 280)
            draw_groups_panel(screen, font, font_sm, groups_rect, frame)

            # mini-graphe TTC sous le HUD
            risk_rect = pygame.Rect(12, 504, 620, 120)
            draw_risk_strip(screen, font_sm, risk_rect, res.frames, fi)

        # bandeau controles
        hint = (f"pas {idx+1}/{max_len}   ESPACE pause/lecture   fleches: pas a pas   "
               f"R rejouer   N nouveau   +/- vitesse ({fps_play}fps)   ESC quitter")
        text(screen, font_sm, hint, (16, DEBUG_H - 26), COL_TEXT_FAINT)

        pygame.display.set_caption(
            f"S-DBS debug -- seed={seed}  {'lecture' if playing else 'pause'} @ {fps_play}fps")
        pygame.display.flip()
        clock.tick(fps_play if playing else 30)

    pygame.quit()


# ==============================================================================
# 7.  Point d'entree
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description="Replay visuel S-DBS (checkpoint entraine)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint .pt produit par l'entrainement. Sans cet "
                        "argument, agent a poids aleatoires (verif pipeline seulement).")
    p.add_argument("--mode", choices=["demo", "debug"], default="demo")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    if not HAS_TORCH:
        raise RuntimeError("PyTorch requis pour ce script.")

    if args.checkpoint:
        agent = load_trained_agent(args.checkpoint, device=args.device)
    else:
        print("[viz] Aucun --checkpoint -- agent a poids aleatoires "
              "(comportement non representatif d'un entrainement).")
        env = EnhancedMockEnv(seed=0)
        cfg = TrainConfig()
        agent = build_agent(env, cfg, use_ensemble=False, ego_xy_fn=enhanced_ego_xy,
                            mandated_action_fn=enhanced_mandated_action, device=args.device)

    if args.mode == "demo":
        run_demo_mode(agent, args)
    else:
        run_debug_mode(agent, args)


if __name__ == "__main__":
    main()