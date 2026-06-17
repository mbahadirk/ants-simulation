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

# debug gorus renkleri (one-hot index -> renk): 0:besin 1:engelli(tas|engel) 2:karinca
VIS_OBJ_COLORS = [
    (230, 50, 50),     # besin
    (160, 160, 175),   # engelli (tas veya engel)
    (240, 120, 200),   # karinca
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
        """TEK feromon: guclendikce parlak CYAN (cok guclu iz -> beyazimsi). Sari=koku."""
        # (GRID_H, GRID_W) -> (GRID_W, GRID_H) make_surface icin
        ph = np.clip(world.ph / C.PH_MAX, 0, 1).T
        odor = np.clip(world.food_odor, 0, 1).T
        # guclu iz (esik ustu sik gecilen yol) -> parlatma icin
        denom = max(1.0, C.PH_MAX - C.PH_STRONG_THRESH)
        strong = np.clip((world.ph - C.PH_STRONG_THRESH) / denom, 0, 1).T
        odor_gain = 170 if debug else 105
        if ph.max() <= 0 and odor.max() <= 0:
            return
        arr = np.zeros((C.GRID_W, C.GRID_H, 3), dtype=np.uint8)
        arr[..., 0] = np.clip(odor * odor_gain * 0.9 + ph * 160 + strong * 220, 0, 255).astype(np.uint8)  # R: koku + mor
        arr[..., 1] = np.clip(odor * odor_gain * 0.8 + ph * 20  + strong * 30, 0, 255).astype(np.uint8)  # G: sadece koku
        arr[..., 2] = np.clip(ph * 230 + strong * 130, 0, 255).astype(np.uint8)                          # B: dominant (violet)
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
                if getattr(sim.world, "homing_enabled", True):
                    self._draw_homing_arrow(surf, ant, camera, size)

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

    def _draw_homing_arrow(self, surf, ant, camera, size):
        """Debug: yuvaya dogru homing oku. Mavi=bos geziyor, sari=besin tasiyor."""
        sx, sy = camera.world_to_screen(ant.x, ant.y)
        # yuva yonu (dunya koordinatlarinda)
        nx, ny = ant.x, ant.y  # hesap icin ant konumunu kullan
        # world'e erisim yok ama ant.last_seen yok, dogrudan heading+homing hesaplayabiliriz
        # Ancak ant nesnesi world.nest_pos'u saklamiyor; renderer world'e erisemiyor.
        # Cozum: ant.sense() zaten homing vektorunu hesapliyor, onu saklayalim.
        # Burada ant._homing_angle kullaniyoruz (sense()'te set edilir).
        hangle = getattr(ant, "_homing_world_angle", None)
        if hangle is None:
            return
        color = (240, 200, 60) if ant.carrying else (80, 160, 255)
        length = max(18, size * 1.4)
        ex = sx + math.cos(hangle) * length
        ey = sy + math.sin(hangle) * length
        pygame.draw.line(surf, color, (int(sx), int(sy)), (int(ex), int(ey)), 2)
        # ok ucu (kucuk ucgen)
        tip_angle = math.atan2(ey - sy, ex - sx)
        spread = 0.45
        tip_len = max(6, size * 0.45)
        for sign in (+1, -1):
            bx = ex - math.cos(tip_angle + sign * spread) * tip_len
            by = ey - math.sin(tip_angle + sign * spread) * tip_len
            pygame.draw.line(surf, color, (int(ex), int(ey)), (int(bx), int(by)), 2)

    # ----------------------------------------------------------------- HUD
    def draw_hud(self, surf, sim, camera, debug, recorder, paused=False, speed=1.0):
        st = sim.stats()
        lines = [
            f"Ants: {st['pop']}   Carrying: {st['carrying']}",
            f"Delivered: {st['delivered']}   Births: {st['births']}   Deaths: {st['deaths']}",
            f"Generation: {st['generation']}   Food: {st['food_left']}   Time: {int(st['time'])}s",
            f"Hall of fame: {st['hof_size']}   Best fitness: {st['hof_best']:.1f}",
            f"Zoom: x{camera.zoom:.1f}   Speed: x{speed:g}   Lifespan: {int(C.LIFESPAN_MIN)}-{int(C.LIFESPAN_MAX)}s",
        ]
        bank = getattr(sim, "bank", None)
        map_name = getattr(sim, "map_name", None)
        if map_name:
            if bank is not None and len(bank) > 0:
                lines.append(f"Map: {map_name}   Bank: {len(bank)} models "
                             f"(best {bank.best_fitness():.0f})")
            else:
                lines.append(f"Map: {map_name}   Bank: empty")
        y = 8
        for ln in lines:
            self._text(surf, ln, 10, y)
            y += 20

        # mod gostergeleri
        if debug:
            self._text(surf, "DEBUG", C.SCREEN_W - 90, 10, (120, 220, 255))
        # otomatik spawn / besin gostergeleri (kapali oldugunda uyar)
        auto_spawn = getattr(sim, "auto_spawn", True)
        auto_food  = getattr(sim, "auto_food",  True)
        ind_x = C.SCREEN_W - 270   # settings butonu sag ustte (36px), burasi biraz solda
        ind_y = 10
        if not auto_spawn:
            self._text(surf, "SPAWN:OFF [F]", ind_x, ind_y, (255, 120, 60))
            ind_y += 20
        if not auto_food:
            self._text(surf, "FOOD-AUTO:OFF [G]", ind_x, ind_y, (255, 200, 60))
        if recorder and recorder.recording:
            pygame.draw.circle(surf, (230, 50, 50), (C.SCREEN_W - 80, 36), 7)
            self._text(surf, f"REC ({recorder.backend})", C.SCREEN_W - 65, 28, (230, 80, 80))
        if paused:
            self._text(surf, "PAUSED (space)", C.SCREEN_W // 2 - 90, 10, (255, 220, 120))

        # selected ant panel
        if debug and sim.selected is not None:
            self._draw_ant_panel(surf, sim.selected)

        # help
        help_txt = ("D:debug  Z:zoom  O/P:speed  K/L:lifespan  T:stats  H:save-demo  "
                    "M:next-map  S:record  F:spawn-toggle  G:food-toggle  MidClick:food  "
                    "Space:pause  Arrows:pan  Click:select  R:reset  ESC:menu")
        self._text(surf, help_txt, 10, C.SCREEN_H - 22, (150, 150, 150))

    def _draw_ant_panel(self, surf, ant):
        x0, y0, w, h = C.SCREEN_W - 240, 60, 230, 224
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((20, 20, 30, 210))
        surf.blit(panel, (x0, y0))
        info = [
            (f"ID: {ant.id}  Gen: {ant.generation}", (220, 220, 230)),
            (f"Age: {ant.age:.1f}/{ant.lifespan:.0f}s", (220, 220, 230)),
            (f"Energy: {max(0,ant.energy):.2f}", (220, 220, 230)),
            (f"Carrying: {'yes' if ant.carrying else 'no'}", (220, 220, 230)),
            (f"Found: {ant.food_found}  Delivered: {ant.food_delivered}", (220, 220, 230)),
            (f"Action: {C.ACTION_NAMES[ant.last_action]}", (220, 220, 230)),
            (f"Odor sense: {ant.last_odor:.2f}", (120, 230, 120)),
            (f"Food pheromone: {ant.last_food_ph:.2f}", (230, 130, 130)),
            (f"Wall hits: {ant.wall_hits}  Idle: {ant.idle_steps}", (230, 200, 130)),
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


# ===========================================================================
# Settings Panel  (sag ust kose dugmesi + acilir panel)
# ===========================================================================
class SettingsPanel:
    """Gear dugmesine tiklaninca acilan ayar paneli.

    Kullanim (main.py):
        panel = SettingsPanel()
        # event dongusu icinde:
        speed = panel.handle_event(event, sim, speed)
        # cizim:
        panel.draw(surf, sim, speed)
    """

    BTN = 36          # gear buton kenari
    PW  = 290         # panel genisligi
    PH  = 386         # panel yuksekligi (3 toggle + 3 slider + tool picker)
    PAD = 12

    # (slider_id, etiket, birim, min, max, log_scale)
    _SLIDERS = [
        ("speed",      "Speed",      "x",  C.SIM_SPEED_MIN, C.SIM_SPEED_MAX, True),
        ("food_val",   "Food Value", " ",  1,               500,             False),
        ("food_rate",  "Spawn Rate", "s",  5,               300,             False),
    ]

    # (tile_id, etiket, renk)
    _TOOLS = [
        (C.FOOD,     "Food",   (200, 60,  60)),
        (C.STONE,    "Stone",  (110, 110, 120)),
        (C.EMPTY,    "Erase",  (50,  70,  50)),
    ]

    def __init__(self):
        self.open       = False
        self.brush_tool = C.FOOD   # orta tik ile yerlestirilecek oge
        self.btn    = pygame.Rect(C.SCREEN_W - self.BTN - 8, 8, self.BTN, self.BTN)
        self._drag  = None
        self._font  = None
        self._sfont = None

    # ---------------------------------------------------------------- fontlar
    def _init_fonts(self):
        if self._font is None:
            self._font  = pygame.font.SysFont("consolas", 15, bold=True)
            self._sfont = pygame.font.SysFont("consolas", 13)

    # ---------------------------------------------------------------- panel konumu
    def _panel_rect(self):
        return pygame.Rect(C.SCREEN_W - self.PW - 8,
                           8 + self.BTN + 4,
                           self.PW, self.PH)

    # ---------------------------------------------------------------- slider hesaplari
    @staticmethod
    def _val_to_t(val, lo, hi, log):
        import math
        if log:
            return math.log(val / lo) / math.log(hi / lo)
        return (val - lo) / (hi - lo)

    @staticmethod
    def _t_to_val(t, lo, hi, log):
        import math
        t = max(0.0, min(1.0, t))
        if log:
            return lo * (hi / lo) ** t
        return lo + t * (hi - lo)

    def _track_rect(self, panel_x, row_y):
        lw = 96    # etiket genisligi
        vw = 42    # deger yazi genisligi
        tw = self.PW - self.PAD * 2 - lw - vw - 8
        return pygame.Rect(panel_x + self.PAD + lw, row_y + 12, tw, 8)

    # ---------------------------------------------------------------- event
    def handle_event(self, event, sim, speed):
        """Olaylari isle; guncellenmis speed degerini doner."""
        self._init_fonts()
        pr = self._panel_rect()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos

            # gear buton
            if self.btn.collidepoint(pos):
                self.open = not self.open
                self._drag = None
                return speed

            if not self.open:
                return speed

            # panel disina tiklandiysa kapat
            if not pr.collidepoint(pos):
                self.open = False
                self._drag = None
                return speed

            # toggle: Ant Spawn
            tr1 = self._toggle_rect(pr, 0)
            if tr1.collidepoint(pos):
                sim.auto_spawn = not getattr(sim, "auto_spawn", True)
                return speed

            # toggle: Auto Food
            tr2 = self._toggle_rect(pr, 1)
            if tr2.collidepoint(pos):
                sim.auto_food = not getattr(sim, "auto_food", True)
                return speed

            # toggle: Homing
            tr3 = self._toggle_rect(pr, 2)
            if tr3.collidepoint(pos):
                sim.world.homing_enabled = not sim.world.homing_enabled
                return speed

            # tool butonlari (paint tool secimi)
            for i, (tile_id, _, _col) in enumerate(self._TOOLS):
                if self._tool_rects(pr)[i].collidepoint(pos):
                    self.brush_tool = tile_id
                    return speed

            # slider tiklamasi baslat
            for i, (sid, lbl, unit, lo, hi, log) in enumerate(self._SLIDERS):
                row_y = pr.y + self.PAD + 132 + i * 52
                track = self._track_rect(pr.x, row_y)
                hit   = pygame.Rect(track.x - 8, track.y - 8,
                                    track.w + 16, track.h + 16)
                if hit.collidepoint(pos):
                    self._drag = (sid, track, lo, hi, log)
                    speed = self._apply_drag(pos[0], track, sid, lo, hi, log, sim, speed)
                    return speed


        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag = None

        elif event.type == pygame.MOUSEMOTION:
            if self._drag:
                sid, track, lo, hi, log = self._drag
                speed = self._apply_drag(event.pos[0], track, sid, lo, hi, log, sim, speed)

        return speed

    def _apply_drag(self, mx, track, sid, lo, hi, log, sim, speed):
        t   = (mx - track.x) / max(1, track.w)
        val = self._t_to_val(t, lo, hi, log)
        if sid == "speed":
            speed = max(lo, min(hi, val))
        elif sid == "food_val":
            C.FOOD_SPAWN_AMOUNT = max(int(lo), min(int(hi), round(val)))
        elif sid == "food_rate":
            C.FOOD_SPAWN_INTERVAL = max(lo, min(hi, val))
        return speed

    # ---------------------------------------------------------------- toggle / tool yardimcilari
    def _toggle_rect(self, pr, idx):
        y = pr.y + self.PAD + 36 + idx * 36
        return pygame.Rect(pr.x + self.PW - self.PAD - 64, y + 2, 62, 24)

    def _tool_rects(self, pr):
        """Her paint-tool butonu icin Rect listesi doner."""
        n   = len(self._TOOLS)
        bw  = (self.PW - self.PAD * 2 - (n - 1) * 6) // n
        y   = pr.y + self.PH - self.PAD - 30
        return [pygame.Rect(pr.x + self.PAD + i * (bw + 6), y, bw, 28)
                for i in range(n)]

    # ---------------------------------------------------------------- cizim
    def draw(self, surf, sim, speed):
        self._init_fonts()
        fn, fs = self._font, self._sfont

        # --- gear butonu ---
        col_btn = (70, 90, 100) if self.open else (45, 50, 60)
        pygame.draw.rect(surf, col_btn, self.btn, border_radius=8)
        pygame.draw.rect(surf, (120, 140, 160), self.btn, 2, border_radius=8)
        gear = fn.render("⚙", True, (220, 230, 240))
        surf.blit(gear, gear.get_rect(center=self.btn.center))

        if not self.open:
            return

        pr = self._panel_rect()

        # --- panel arka plani ---
        bg = pygame.Surface((pr.w, pr.h), pygame.SRCALPHA)
        bg.fill((18, 20, 28, 230))
        surf.blit(bg, pr.topleft)
        pygame.draw.rect(surf, (80, 90, 110), pr, 2, border_radius=8)

        # baslik
        t = fn.render("Settings", True, (210, 220, 240))
        surf.blit(t, (pr.x + self.PAD, pr.y + self.PAD))
        pygame.draw.line(surf, (60, 70, 90),
                         (pr.x + self.PAD, pr.y + 30),
                         (pr.x + pr.w - self.PAD, pr.y + 30))

        # --- toggle satirlari ---
        toggles = [
            ("Ant Spawn", getattr(sim, "auto_spawn", True)),
            ("Auto Food", getattr(sim, "auto_food",  True)),
            ("Homing",    getattr(sim.world, "homing_enabled", True)),
        ]
        for i, (lbl, state) in enumerate(toggles):
            y = pr.y + self.PAD + 36 + i * 36
            surf.blit(fs.render(lbl, True, (200, 210, 220)), (pr.x + self.PAD, y + 4))
            tr = self._toggle_rect(pr, i)
            tc = (40, 160, 80) if state else (140, 50, 50)
            pygame.draw.rect(surf, tc, tr, border_radius=5)
            pygame.draw.rect(surf, (90, 110, 130), tr, 1, border_radius=5)
            on_off = fn.render("ON" if state else "OFF", True, (240, 240, 240))
            surf.blit(on_off, on_off.get_rect(center=tr.center))

        # ayirici
        sep_y = pr.y + self.PAD + 36 + 3 * 36 + 6
        pygame.draw.line(surf, (60, 70, 90),
                         (pr.x + self.PAD, sep_y),
                         (pr.x + pr.w - self.PAD, sep_y))

        # --- slider satirlari ---
        vals = [speed, float(C.FOOD_SPAWN_AMOUNT), float(C.FOOD_SPAWN_INTERVAL)]
        for i, ((sid, lbl, unit, lo, hi, log), val) in enumerate(
                zip(self._SLIDERS, vals)):
            row_y = pr.y + self.PAD + 132 + i * 52
            # etiket
            surf.blit(fs.render(lbl, True, (190, 205, 220)),
                      (pr.x + self.PAD, row_y))
            # deger metni
            if sid == "speed":
                vtxt = f"x{val:.2g}"
            elif sid == "food_rate":
                vtxt = f"{val:.0f}s"
            else:
                vtxt = str(int(val))
            surf.blit(fn.render(vtxt, True, (240, 240, 160)),
                      (pr.x + pr.w - self.PAD - 42, row_y))
            # track
            track = self._track_rect(pr.x, row_y)
            pygame.draw.rect(surf, (50, 55, 70), track, border_radius=4)
            t = self._val_to_t(val, lo, hi, log)
            t = max(0.0, min(1.0, t))
            fx = int(track.x + t * track.w)
            filled = pygame.Rect(track.x, track.y, fx - track.x, track.h)
            if filled.w > 0:
                pygame.draw.rect(surf, (70, 140, 200), filled, border_radius=4)
            pygame.draw.circle(surf, (210, 230, 255), (fx, track.centery), 7)
            pygame.draw.circle(surf, (100, 130, 170), (fx, track.centery), 7, 2)

        # --- paint tool secici ---
        sep2_y = pr.y + self.PH - self.PAD - 30 - 10
        pygame.draw.line(surf, (60, 70, 90),
                         (pr.x + self.PAD, sep2_y),
                         (pr.x + pr.w - self.PAD, sep2_y))
        surf.blit(fs.render("Mid-click tool:", True, (180, 190, 200)),
                  (pr.x + self.PAD, sep2_y + 4))
        tool_rects = self._tool_rects(pr)
        for i, ((tile_id, lbl, col), rect) in enumerate(zip(self._TOOLS, tool_rects)):
            active = (self.brush_tool == tile_id)
            bg_col = tuple(min(255, c + 40) for c in col) if active else col
            pygame.draw.rect(surf, bg_col, rect, border_radius=5)
            brd = (255, 240, 100) if active else (70, 80, 95)
            pygame.draw.rect(surf, brd, rect, 2 if active else 1, border_radius=5)
            txt = fn.render(lbl, True, (240, 240, 240))
            surf.blit(txt, txt.get_rect(center=rect.center))
