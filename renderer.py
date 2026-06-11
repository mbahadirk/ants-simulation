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

# debug gorus renkleri (one-hot index -> renk): 0besin 1tas 2engel 3karinca 4yuva
VIS_OBJ_COLORS = [
    (230, 50, 50),     # besin
    (160, 160, 175),   # tas
    (150, 110, 80),    # engel
    (240, 120, 200),   # karinca
    (240, 200, 80),    # yuva
]


def _hsv_color(h, s=0.7, v=1.0):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class Renderer:
    def __init__(self):
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 12, bold=True)
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
    def draw_world(self, surf, world, camera, show_pheromones=True, debug=False):
        surf.fill(C.BG_COLOR)
        if show_pheromones:
            self._draw_fields(surf, world, camera, debug=debug)
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
                    ccx = int(sx + scaled_cs / 2)
                    ccy = int(sy + scaled_cs / 2)
                    pygame.draw.circle(surf, C.COLORS[C.FOOD], (ccx, ccy),
                                       max(2, int(scaled_cs * 0.32)))
                    # yeterince yakinsa kalan miktari yaz
                    if scaled_cs >= 22:
                        amt = int(world.food_amount[row, col])
                        t_img = self.small.render(str(amt), True, (255, 255, 255))
                        surf.blit(t_img, t_img.get_rect(center=(ccx, ccy)))
                elif t == C.STONE:
                    pygame.draw.rect(surf, C.COLORS[C.STONE], rect, border_radius=max(2, int(scaled_cs * 0.25)))
                elif t == C.OBSTACLE:
                    pygame.draw.rect(surf, C.COLORS[C.OBSTACLE], rect)

        # yuva (daire)
        nx, ny = camera.world_to_screen(*world.nest_pos)
        nr = camera.scale(world.nest_radius)
        pygame.draw.circle(surf, (120, 85, 30), (int(nx), int(ny)), int(nr))
        pygame.draw.circle(surf, C.COLORS[C.NEST], (int(nx), int(ny)), int(nr), max(2, int(nr * 0.15)))

    def _draw_fields(self, surf, world, camera, debug=False):
        """Feromon izleri (kirmizi=besin izi, mavi=home izi) + besin kokusu (sari)."""
        # (GRID_H, GRID_W) -> (GRID_W, GRID_H) make_surface icin
        home = np.clip(world.ph_home / C.PH_MAX, 0, 1).T
        food = np.clip(world.ph_food / C.PH_MAX, 0, 1).T
        odor = np.clip(world.food_odor, 0, 1).T
        # debug'da koku daha belirgin gorunsun
        odor_gain = 170 if debug else 105
        if home.max() <= 0 and food.max() <= 0 and odor.max() <= 0:
            return
        arr = np.zeros((C.GRID_W, C.GRID_H, 3), dtype=np.uint8)
        arr[..., 0] = np.clip(food * 235 + odor * odor_gain * 0.9, 0, 255).astype(np.uint8)      # R: besin izi + koku
        arr[..., 1] = np.clip(food * 70 + home * 30 + odor * odor_gain * 0.8, 0, 255).astype(np.uint8)  # G: koku + izler
        arr[..., 2] = np.clip(home * 215, 0, 255).astype(np.uint8)                              # B: home izi
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

            # besin tasiyorsa kucuk kirmizi nokta
            if ant.carrying:
                pygame.draw.circle(surf, (230, 50, 50), (int(sx), int(sy - size * 0.4)),
                                   max(2, int(size * 0.18)))

            # secili karinca -> halka
            if sim.selected is ant:
                pygame.draw.circle(surf, (255, 255, 120), (int(sx), int(sy)),
                                   int(size * 0.9), 2)

    def _draw_vision(self, surf, ant, camera):
        sx, sy = camera.world_to_screen(ant.x, ant.y)
        rad = camera.scale(C.VISION_RANGE)
        # 180 derece gorus yelpazesi (heading +- 90 derece yay)
        if rad > 2:
            half = C.VISION_FOV / 2.0
            pts = [(sx, sy)]
            steps = 18
            for i in range(steps + 1):
                a = ant.heading - half + (C.VISION_FOV * i / steps)
                pts.append((sx + math.cos(a) * rad, sy + math.sin(a) * rad))
            pygame.draw.polygon(surf, (70, 70, 85), pts, 1)
        # gorulen nesnelere cizgi + nokta (one-hot indekse gore renk)
        for (oi, ox, oy, dist) in ant.last_seen:
            osx, osy = camera.world_to_screen(ox, oy)
            color = VIS_OBJ_COLORS[oi] if 0 <= oi < len(VIS_OBJ_COLORS) else (200, 200, 200)
            pygame.draw.line(surf, color, (sx, sy), (osx, osy), 1)
            pygame.draw.circle(surf, color, (int(osx), int(osy)), 3)

    # ----------------------------------------------------------------- HUD
    def draw_hud(self, surf, sim, camera, debug, recorder, paused=False, speed=1.0):
        st = sim.stats()
        lines = [
            f"Karinca: {st['pop']}   Tasiyan: {st['carrying']}",
            f"Teslim: {st['delivered']}   Dogum: {st['births']}   Olum: {st['deaths']}",
            f"Nesil: {st['generation']}   Besin: {st['food_left']}   Sure: {int(st['time'])}s",
            f"Onur listesi: {st['hof_size']}   En iyi fitness: {st['hof_best']:.1f}",
            f"Zoom: x{camera.zoom:.1f}   Speed: x{speed:g}   Omur: {int(C.LIFESPAN_MIN)}-{int(C.LIFESPAN_MAX)}s",
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
        help_txt = ("D:debug  Z:zoom  O/P:hiz-/+  K/L:omur-/+  S:kayit  Space:duraklat  "
                    "Ok:pan  Tik:secim  R:reset  ESC:menu")
        self._text(surf, help_txt, 10, C.SCREEN_H - 22, (150, 150, 150))

    def _draw_ant_panel(self, surf, ant):
        x0, y0, w, h = C.SCREEN_W - 240, 60, 230, 224
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 210))
        surf.blit(panel, (x0, y0))
        info = [
            (f"ID: {ant.id}  Nesil: {ant.generation}", (220, 220, 230)),
            (f"Yas: {ant.age:.1f}/{ant.lifespan:.0f}s", (220, 220, 230)),
            (f"Enerji: {max(0,ant.energy):.2f}", (220, 220, 230)),
            (f"Tasiyor: {'evet' if ant.carrying else 'hayir'}", (220, 220, 230)),
            (f"Bulunan: {ant.food_found}  Teslim: {ant.food_delivered}", (220, 220, 230)),
            (f"Aksiyon: {C.ACTION_NAMES[ant.last_action]}", (220, 220, 230)),
            (f"Koku algi: {ant.last_odor:.2f}", (120, 230, 120)),
            (f"Food feromon: {ant.last_food_ph:.2f}", (230, 130, 130)),
            (f"Duvar carpma: {ant.wall_hits}  Bekle: {ant.idle_steps}", (230, 200, 130)),
        ]
        yy = y0 + 8
        for ln, col in info:
            self._text(surf, ln, x0 + 10, yy, col)
            yy += 22

    def draw_flash(self, surf, text):
        """Ekranin altinda gecici bildirim kutusu (kayit/devam)."""
        if not text:
            return
        img = self.big.render(text, True, (255, 240, 160))
        w, h = img.get_size()
        x = (C.SCREEN_W - w) // 2
        y = C.SCREEN_H - 70
        bg = pygame.Surface((w + 24, h + 16), pygame.SRCALPHA)
        bg.fill((20, 30, 20, 220))
        surf.blit(bg, (x - 12, y - 8))
        surf.blit(img, (x, y))

    def _text(self, surf, txt, x, y, color=(230, 230, 230)):
        surf.blit(self.font.render(txt, True, color), (x, y))
