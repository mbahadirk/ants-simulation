"""
Cizim katmani: dunya, karincalar, debug overlay, HUD.
Tum cizimler Camera uzerinden olceklenir (zoom + pan).
"""

import math

import numpy as np
import pygame

import config as C

# ant.png yukari (north) bakiyor varsayilir; heading=0 (dogu) icin -90 ofset.
ANT_IMG_OFFSET = -90

# debug isin renkleri (one-hot index -> renk)
RAY_HIT_COLORS = {
    0: (90, 230, 90),    # besin
    1: (160, 160, 175),  # tas
    2: (150, 110, 80),   # engel
    3: (240, 120, 200),  # karinca
    4: (240, 200, 80),   # yuva
}


def _hsv_color(h, s=0.7, v=1.0):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class Renderer:
    def __init__(self):
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        # ant.png yukle
        try:
            img = pygame.image.load(C.ANT_IMAGE).convert_alpha()
        except Exception:
            img = pygame.Surface((32, 32), pygame.SRCALPHA)
            pygame.draw.circle(img, (40, 30, 20), (16, 16), 14)
        self.ant_base = img
        self._rot_cache = {}  # (deg_bucket, size) -> surface

    # --------------------------------------------------------------- dunya
    def draw_world(self, surf, world, camera, show_pheromones=True):
        surf.fill(C.BG_COLOR)
        if show_pheromones:
            self._draw_pheromones(surf, world, camera)
        cs = C.CELL_SIZE

        # gorunur hucre araligi
        wx0, wy0 = camera.x, camera.y
        wx1, wy1 = camera.screen_to_world(C.SCREEN_W, C.SCREEN_H)
        c0 = max(0, int(wx0 // cs))
        r0 = max(0, int(wy0 // cs))
        c1 = min(C.GRID_W, int(wx1 // cs) + 2)
        r1 = min(C.GRID_H, int(wy1 // cs) + 2)

        z = camera.zoom
        scaled_cs = cs * z

        for row in range(r0, r1):
            for col in range(c0, c1):
                t = int(world.grid[row, col])
                if t == C.EMPTY or t == C.NEST:
                    continue
                sx, sy = camera.world_to_screen(col * cs, row * cs)
                rect = pygame.Rect(sx, sy, scaled_cs + 1, scaled_cs + 1)
                if t == C.FOOD:
                    pygame.draw.circle(surf, C.COLORS[C.FOOD],
                                       (int(sx + scaled_cs / 2), int(sy + scaled_cs / 2)),
                                       max(2, int(scaled_cs * 0.32)))
                elif t == C.STONE:
                    pygame.draw.rect(surf, C.COLORS[C.STONE], rect, border_radius=max(2, int(scaled_cs * 0.25)))
                elif t == C.OBSTACLE:
                    pygame.draw.rect(surf, C.COLORS[C.OBSTACLE], rect)

        # yuva (daire)
        nx, ny = camera.world_to_screen(*world.nest_pos)
        nr = camera.scale(world.nest_radius)
        pygame.draw.circle(surf, (120, 85, 30), (int(nx), int(ny)), int(nr))
        pygame.draw.circle(surf, C.COLORS[C.NEST], (int(nx), int(ny)), int(nr), max(2, int(nr * 0.15)))

    def _draw_pheromones(self, surf, world, camera):
        # (GRID_H, GRID_W) -> (GRID_W, GRID_H) make_surface icin
        home = np.clip(world.ph_home / C.PH_MAX, 0, 1).T
        food = np.clip(world.ph_food / C.PH_MAX, 0, 1).T
        if home.max() <= 0 and food.max() <= 0:
            return
        arr = np.zeros((C.GRID_W, C.GRID_H, 3), dtype=np.uint8)
        arr[..., 0] = np.clip(food * 235, 0, 255).astype(np.uint8)          # R: besin izi
        arr[..., 1] = np.clip(food * 90 + home * 35, 0, 255).astype(np.uint8)
        arr[..., 2] = np.clip(home * 215, 0, 255).astype(np.uint8)          # B: home izi
        small = pygame.surfarray.make_surface(arr)
        tw = max(1, int(C.WORLD_W * camera.zoom))
        th = max(1, int(C.WORLD_H * camera.zoom))
        scaled = pygame.transform.smoothscale(small, (tw, th))
        sx, sy = camera.world_to_screen(0, 0)
        surf.blit(scaled, (sx, sy), special_flags=pygame.BLEND_RGB_ADD)

    # ------------------------------------------------------------ karincalar
    def _ant_surface(self, heading, size):
        deg = int((-math.degrees(heading) + ANT_IMG_OFFSET) // 6 * 6)  # 6 derecelik kovalar
        key = (deg, size)
        surf = self._rot_cache.get(key)
        if surf is None:
            base = pygame.transform.smoothscale(self.ant_base, (size, size))
            surf = pygame.transform.rotate(base, deg)
            self._rot_cache[key] = surf
            if len(self._rot_cache) > 4000:
                self._rot_cache.clear()
        return surf

    def draw_ants(self, surf, sim, camera, debug=False):
        size = max(6, int(C.ANT_SIZE * camera.zoom))
        for ant in sim.ants:
            sx, sy = camera.world_to_screen(ant.x, ant.y)
            if sx < -40 or sy < -40 or sx > C.SCREEN_W + 40 or sy > C.SCREEN_H + 40:
                continue

            if debug:
                self._draw_vision(surf, ant, camera)

            img = self._ant_surface(ant.heading, size)
            rect = img.get_rect(center=(sx, sy))
            surf.blit(img, rect)

            # besin tasiyorsa kucuk yesil nokta
            if ant.carrying:
                pygame.draw.circle(surf, (90, 230, 90), (int(sx), int(sy - size * 0.4)),
                                   max(2, int(size * 0.18)))

            # secili karinca -> halka
            if sim.selected is ant:
                pygame.draw.circle(surf, (255, 255, 120), (int(sx), int(sy)),
                                   int(size * 0.9), 2)

    def _draw_vision(self, surf, ant, camera):
        sx, sy = camera.world_to_screen(ant.x, ant.y)
        for (ang, dist, idx) in ant.last_rays:
            ex = ant.x + math.cos(ang) * dist
            ey = ant.y + math.sin(ang) * dist
            esx, esy = camera.world_to_screen(ex, ey)
            if idx is None:
                color = (60, 60, 70)
            else:
                color = RAY_HIT_COLORS.get(idx, (200, 200, 200))
            pygame.draw.line(surf, color, (sx, sy), (esx, esy), 1)
            if idx is not None:
                pygame.draw.circle(surf, color, (int(esx), int(esy)), 3)

    # ----------------------------------------------------------------- HUD
    def draw_hud(self, surf, sim, camera, debug, recorder, paused=False):
        st = sim.stats()
        lines = [
            f"Karinca: {st['pop']}   Tasiyan: {st['carrying']}",
            f"Teslim: {st['delivered']}   Dogum: {st['births']}   Olum: {st['deaths']}",
            f"Nesil: {st['generation']}   Besin: {st['food_left']}   Sure: {int(st['time'])}s",
            f"Zoom: x{camera.zoom:.1f}",
        ]
        y = 8
        for ln in lines:
            self._text(surf, ln, 10, y)
            y += 20

        # mod gostergeleri
        if debug:
            self._text(surf, "DEBUG", C.SCREEN_W - 90, 10, (120, 220, 255))
        if recorder and recorder.recording:
            pygame.draw.circle(surf, (230, 50, 50), (C.SCREEN_W - 80, 36), 7)
            self._text(surf, f"REC ({recorder.backend})", C.SCREEN_W - 65, 28, (230, 80, 80))
        if paused:
            self._text(surf, "DURAKLATILDI (space)", C.SCREEN_W // 2 - 110, 10, (255, 220, 120))

        # secili karinca paneli
        if debug and sim.selected is not None:
            self._draw_ant_panel(surf, sim.selected)

        # yardim
        help_txt = "D:debug  Z:zoom  S:kayit  Space:duraklat  Ok:pan  Tik:secim  R:reset-kamera  ESC:menu"
        self._text(surf, help_txt, 10, C.SCREEN_H - 22, (150, 150, 150))

    def _draw_ant_panel(self, surf, ant):
        x0, y0, w, h = C.SCREEN_W - 230, 60, 220, 150
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 210))
        surf.blit(panel, (x0, y0))
        info = [
            f"ID: {ant.id}  Nesil: {ant.generation}",
            f"Yas: {ant.age:.1f}/{ant.lifespan:.0f}s",
            f"Enerji: {max(0,ant.energy):.2f}",
            f"Tasiyor: {'evet' if ant.carrying else 'hayir'}",
            f"Teslim: {ant.food_delivered}",
            f"Aksiyon: {C.ACTION_NAMES[ant.last_action]}",
        ]
        yy = y0 + 8
        for ln in info:
            self._text(surf, ln, x0 + 10, yy, (220, 220, 230))
            yy += 22

    def _text(self, surf, txt, x, y, color=(230, 230, 230)):
        surf.blit(self.font.render(txt, True, color), (x, y))
