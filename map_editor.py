"""
UI tabanli harita editoru (tamamen pygame).

- Sol panelde arac butonlari: Besin, Tas, Engel, Yuva, Sil.
- Sol tik (basili tutarak suru) ile secili nesneyi yerlestirir.
- Sag tik siler.
- Firca boyutu [ ve ] tuslari ile ayarlanir.
- Harita adi girisi -> maps/<ad>.json olarak kaydeder.
- Load butonu mevcut haritalar arasindan secim yapar.
- New butonu bos harita baslatir.
- Ctrl+S ile hizli kaydet.
"""

import os
import glob

import numpy as np
import pygame

import config as C
from world import World

PANEL_W = 190
MAPS_DIR = getattr(C, "MAPS_DIR", "maps")


class Button:
    def __init__(self, label, x, y, w, h, value=None, color=(60, 60, 70)):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        self.value = value
        self.color = color

    def draw(self, surf, font, active=False):
        col = self.color
        if active:
            col = tuple(min(255, c + 70) for c in col)
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, (20, 20, 24), self.rect, 2, border_radius=6)
        t = font.render(self.label, True, (235, 235, 235))
        surf.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, pos):
        return self.rect.collidepoint(pos)


class MapEditor:
    def __init__(self, screen, grid=None, food_amount=None, map_name=None):
        self.screen = screen
        self.font  = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 13)
        self.tiny  = pygame.font.SysFont("consolas", 11, bold=True)

        if grid is None:
            grid = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int16)
        self.grid = grid.copy()

        self.food_amount = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int32)
        if food_amount is not None:
            fa = np.array(food_amount, dtype=np.int32)
            if fa.shape == self.food_amount.shape:
                self.food_amount = fa
        miss = (self.grid == C.FOOD) & (self.food_amount <= 0)
        self.food_amount[miss] = C.FOOD_DEFAULT_AMOUNT

        # harita adi
        self.map_name      = map_name or "default_map"
        self.map_name_text = self.map_name
        self.editing_name  = False

        # cizim alani olcegi
        area_w = C.SCREEN_W - PANEL_W
        self.scale = min(area_w / C.WORLD_W, C.SCREEN_H / C.WORLD_H)
        self.ox = PANEL_W
        self.oy = 0

        self.current = C.FOOD
        self.brush   = 1
        self.message = ""
        self.msg_timer = 0.0

        self.amount      = C.FOOD_DEFAULT_AMOUNT
        self.amount_text = str(self.amount)
        self.editing_amount = False

        self._build_buttons()

    # ------------------------------------------------------------------
    def _build_buttons(self):
        x = 12
        w = PANEL_W - 24
        y = 16

        # baslik icin yer birak
        y += 22   # "MAP EDITOR"
        y += 18   # bosluk

        # --- harita adi kutusu ---
        self.name_label_y = y
        y += 18
        self.name_box = pygame.Rect(x, y, w, 28)
        y += 36

        # --- arac butonlari ---
        lbl_y = y
        y += 18   # "Brush / Selected" satirlari icin yer
        y += 16
        self.tool_buttons = []
        tools = [
            ("1 Food",     C.FOOD,     C.COLORS[C.FOOD]),
            ("2 Stone",    C.STONE,    C.COLORS[C.STONE]),
            ("3 Obstacle", C.OBSTACLE, C.COLORS[C.OBSTACLE]),
            ("4 Nest",     C.NEST,     C.COLORS[C.NEST]),
            ("0 Erase",    C.EMPTY,    (50, 50, 55)),
        ]
        for label, val, col in tools:
            self.tool_buttons.append(Button(label, x, y, w, 36, value=val, color=col))
            y += 43

        # --- besin miktari ---
        y += 4
        self.amount_label_y = y
        y += 18
        self.amount_box = pygame.Rect(x, y, w, 28)
        y += 38

        # --- aksiyon butonlari ---
        self.btn_save   = Button("Save  (Ctrl+S)", x, y, w, 34, color=(40, 110, 60));  y += 42
        self.btn_new    = Button("New Map",         x, y, w, 34, color=(50, 80, 120));  y += 42
        self.btn_load   = Button("Load Map",        x, y, w, 34, color=(70, 60, 100));  y += 42
        self.btn_clear  = Button("Clear",           x, y, w, 34, color=(120, 60, 40));  y += 42
        self.btn_border = Button("Border Wall",     x, y, w, 34, color=(70, 60, 50));   y += 42
        self.btn_menu   = Button("Menu (ESC)",      x, y, w, 34, color=(60, 60, 80))

        self._lbl_y = lbl_y   # brush/selected satirlari

    # ----------------------------------------------------------- koordinatlar
    def _screen_to_cell(self, mx, my):
        wx = (mx - self.ox) / self.scale
        wy = (my - self.oy) / self.scale
        col = int(wx // C.CELL_SIZE)
        row = int(wy // C.CELL_SIZE)
        return col, row

    def _paint(self, mx, my, value):
        col, row = self._screen_to_cell(mx, my)
        b = self.brush
        for r in range(row - b + 1, row + b):
            for c in range(col - b + 1, col + b):
                if 0 <= r < C.GRID_H and 0 <= c < C.GRID_W:
                    self.grid[r, c] = value
                    self.food_amount[r, c] = self.amount if value == C.FOOD else 0

    def _flash(self, msg):
        self.message   = msg
        self.msg_timer = 2.5

    # ------------------------------------------------------------------- loop
    def run(self):
        clock   = pygame.time.Clock()
        painting = None
        while True:
            dt = clock.tick(C.FPS) / 1000.0
            self.msg_timer = max(0.0, self.msg_timer - dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit

                elif event.type == pygame.KEYDOWN:
                    if self.editing_name:
                        self._handle_name_key(event)
                    elif self.editing_amount:
                        self._handle_amount_key(event)
                    elif event.key == pygame.K_ESCAPE:
                        return "menu"
                    elif event.key == pygame.K_1:
                        self.current = C.FOOD
                    elif event.key == pygame.K_2:
                        self.current = C.STONE
                    elif event.key == pygame.K_3:
                        self.current = C.OBSTACLE
                    elif event.key == pygame.K_4:
                        self.current = C.NEST
                    elif event.key == pygame.K_0:
                        self.current = C.EMPTY
                    elif event.key == pygame.K_LEFTBRACKET:
                        self.brush = max(1, self.brush - 1)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self.brush = min(6, self.brush + 1)
                    elif event.key == pygame.K_s and (event.mod & pygame.KMOD_CTRL):
                        self._save()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[0] < PANEL_W:
                        self._handle_panel_click(event.pos)
                    else:
                        if self.editing_name:   self._commit_name()
                        if self.editing_amount: self._commit_amount()
                        if event.button == 1:
                            painting = "paint"
                            self._paint(*event.pos, self.current)
                        elif event.button == 3:
                            painting = "erase"
                            self._paint(*event.pos, C.EMPTY)

                elif event.type == pygame.MOUSEBUTTONUP:
                    painting = None

                elif event.type == pygame.MOUSEMOTION and painting:
                    if event.pos[0] >= PANEL_W:
                        val = self.current if painting == "paint" else C.EMPTY
                        self._paint(*event.pos, val)

            self._draw()
            pygame.display.flip()

    # ----------------------------------------- klavye girdileri
    def _handle_name_key(self, event):
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self._commit_name()
        elif event.key == pygame.K_BACKSPACE:
            self.map_name_text = self.map_name_text[:-1]
        elif event.unicode and len(self.map_name_text) < 28:
            ch = event.unicode
            # dosya adi icin guvenli karakterler
            if ch.isalnum() or ch in "-_":
                self.map_name_text += ch

    def _commit_name(self):
        name = self.map_name_text.strip() or self.map_name
        # dosya adi temizle
        name = "".join(c for c in name if c.isalnum() or c in "-_")
        self.map_name      = name or "default_map"
        self.map_name_text = self.map_name
        self.editing_name  = False

    def _handle_amount_key(self, event):
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self._commit_amount()
        elif event.key == pygame.K_BACKSPACE:
            self.amount_text = self.amount_text[:-1]
        elif event.unicode.isdigit() and len(self.amount_text) < 4:
            self.amount_text += event.unicode

    def _commit_amount(self):
        try:
            v = int(self.amount_text) if self.amount_text else C.FOOD_DEFAULT_AMOUNT
        except ValueError:
            v = C.FOOD_DEFAULT_AMOUNT
        self.amount      = max(1, min(C.FOOD_MAX_AMOUNT, v))
        self.amount_text = str(self.amount)
        self.editing_amount = False

    # ----------------------------------------- panel tiklari
    def _handle_panel_click(self, pos):
        # ad kutusu
        if self.name_box.collidepoint(pos):
            if self.editing_amount: self._commit_amount()
            self.editing_name  = True
            self.map_name_text = ""
            return
        # miktar kutusu
        if self.amount_box.collidepoint(pos):
            if self.editing_name: self._commit_name()
            self.editing_amount = True
            self.amount_text    = ""
            return
        # diger tiklarda aktif girisleri kapat
        if self.editing_name:   self._commit_name()
        if self.editing_amount: self._commit_amount()

        for b in self.tool_buttons:
            if b.hit(pos):
                self.current = b.value
                return
        if self.btn_save.hit(pos):
            self._save()
        elif self.btn_new.hit(pos):
            self._new_map()
        elif self.btn_load.hit(pos):
            self._open_load_overlay()
        elif self.btn_clear.hit(pos):
            self.grid[:]        = C.EMPTY
            self.food_amount[:] = 0
            self._flash("Map cleared")
        elif self.btn_border.hit(pos):
            for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
                self.grid[sl]        = C.OBSTACLE
                self.food_amount[sl] = 0
            self._flash("Border added")
        elif self.btn_menu.hit(pos):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

    # ----------------------------------------- kaydet / yeni / yukle
    def _save_path(self):
        os.makedirs(MAPS_DIR, exist_ok=True)
        return os.path.join(MAPS_DIR, self.map_name + ".json")

    def _save(self):
        if not np.any(self.grid == C.NEST):
            self._flash("WARNING: No nest! Place a nest first.")
            return
        self.food_amount[self.grid != C.FOOD] = 0
        path = self._save_path()
        World(grid=self.grid.copy(), food_amount=self.food_amount.copy()).save(path)
        self._flash(f"Saved -> {self.map_name}.json")

    def _new_map(self):
        self.grid[:]        = C.EMPTY
        self.food_amount[:] = 0
        self.map_name       = "new_map"
        self.map_name_text  = self.map_name
        self._flash("New map — edit name, then Save.")

    def _open_load_overlay(self):
        """Mevcut haritalar arasindan secim yapilan basit bir overlay."""
        maps = sorted(glob.glob(os.path.join(MAPS_DIR, "*.json")))
        if not maps:
            self._flash("No maps found in maps/ folder.")
            return

        font_big = pygame.font.SysFont("consolas", 20, bold=True)
        font_row = pygame.font.SysFont("consolas", 16)
        clock    = pygame.time.Clock()

        ROW_H  = 38
        PAD    = 16
        OW     = min(560, C.SCREEN_W - 60)
        OH     = min(len(maps) * ROW_H + PAD * 3 + 40, C.SCREEN_H - 80)
        ox     = (C.SCREEN_W - OW) // 2
        oy     = (C.SCREEN_H - OH) // 2

        scroll = 0
        visible = (OH - PAD * 2 - 40) // ROW_H

        while True:
            clock.tick(C.FPS)
            mouse = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return  # overlay kapat, harita degismez
                    elif event.key == pygame.K_UP:
                        scroll = max(0, scroll - 1)
                    elif event.key == pygame.K_DOWN:
                        scroll = max(0, min(len(maps) - visible, scroll + 1))
                elif event.type == pygame.MOUSEWHEEL:
                    scroll = max(0, min(len(maps) - visible, scroll - event.y))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # kapat alani disinda herhangi bir tik
                    if not pygame.Rect(ox, oy, OW, OH).collidepoint(mouse):
                        return
                    # satir secimi
                    for i, idx in enumerate(range(scroll, min(scroll + visible, len(maps)))):
                        ry = oy + PAD + 40 + i * ROW_H
                        row_rect = pygame.Rect(ox + PAD, ry, OW - PAD * 2, ROW_H - 4)
                        if row_rect.collidepoint(mouse):
                            self._load_map(maps[idx])
                            return

            # --- ciz ---
            # arkaplan karartma
            dim = pygame.Surface((C.SCREEN_W, C.SCREEN_H), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 160))
            self.screen.blit(dim, (0, 0))

            # overlay pencere
            pygame.draw.rect(self.screen, (28, 28, 36),
                             pygame.Rect(ox, oy, OW, OH), border_radius=10)
            pygame.draw.rect(self.screen, (80, 80, 100),
                             pygame.Rect(ox, oy, OW, OH), 2, border_radius=10)

            title_img = font_big.render("Load Map  (ESC=cancel)", True, (220, 220, 240))
            self.screen.blit(title_img, (ox + PAD, oy + PAD))

            for i, idx in enumerate(range(scroll, min(scroll + visible, len(maps)))):
                ry = oy + PAD + 40 + i * ROW_H
                row_rect = pygame.Rect(ox + PAD, ry, OW - PAD * 2, ROW_H - 4)
                hot = row_rect.collidepoint(mouse)
                pygame.draw.rect(self.screen, (60, 70, 90) if hot else (40, 42, 52),
                                 row_rect, border_radius=5)
                name = os.path.splitext(os.path.basename(maps[idx]))[0]
                row_img = font_row.render(name, True, (230, 230, 230))
                self.screen.blit(row_img, row_img.get_rect(midleft=(row_rect.x + 10,
                                                                      row_rect.centery)))

            # scrollbar ipucu
            if len(maps) > visible:
                hint = font_row.render(f"{scroll+1}-{min(scroll+visible,len(maps))}/{len(maps)}  ↑↓",
                                       True, (140, 140, 160))
                self.screen.blit(hint, (ox + OW - hint.get_width() - PAD,
                                        oy + PAD + 8))

            pygame.display.flip()

    def _load_map(self, path):
        try:
            w = World.load(path)
            self.grid       = w.grid.copy()
            self.food_amount = w.food_amount.copy()
            name = os.path.splitext(os.path.basename(path))[0]
            self.map_name      = name
            self.map_name_text = name
            self._flash(f"Loaded: {name}")
        except Exception as e:
            self._flash(f"Load error: {e}")

    # ------------------------------------------------------------------- draw
    def _draw(self):
        self.screen.fill((30, 28, 26))
        cs = C.CELL_SIZE * self.scale

        for row in range(C.GRID_H):
            for col in range(C.GRID_W):
                t = int(self.grid[row, col])
                x = self.ox + col * cs
                y = self.oy + row * cs
                rect = pygame.Rect(x, y, cs + 1, cs + 1)
                base = C.COLORS.get(t, C.COLORS[C.EMPTY])
                pygame.draw.rect(self.screen, base, rect)
                if t == C.FOOD and cs >= 13:
                    n = self.tiny.render(str(int(self.food_amount[row, col])), True, (255, 255, 255))
                    self.screen.blit(n, n.get_rect(center=(x + cs / 2, y + cs / 2)))

        for col in range(C.GRID_W + 1):
            x = self.ox + col * cs
            pygame.draw.line(self.screen, (45, 43, 40), (x, 0), (x, C.GRID_H * cs))
        for row in range(C.GRID_H + 1):
            y = self.oy + row * cs
            pygame.draw.line(self.screen, (45, 43, 40), (self.ox, y), (self.ox + C.GRID_W * cs, y))

        mx, my = pygame.mouse.get_pos()
        if mx >= PANEL_W:
            col, row = self._screen_to_cell(mx, my)
            b = self.brush
            px = self.ox + (col - b + 1) * cs
            py = self.oy + (row - b + 1) * cs
            pygame.draw.rect(self.screen, (255, 255, 255),
                             pygame.Rect(px, py, cs * (2 * b - 1), cs * (2 * b - 1)), 1)

        self._draw_panel()

    def _draw_panel(self):
        pygame.draw.rect(self.screen, (22, 22, 28), pygame.Rect(0, 0, PANEL_W, C.SCREEN_H))

        y = 16
        title = self.font.render("MAP EDITOR", True, (240, 240, 240))
        self.screen.blit(title, (12, y)); y += 22

        # --- harita adi ---
        lbl = self.small.render("Map name:", True, (180, 200, 180))
        self.screen.blit(lbl, (12, self.name_label_y))
        box_col  = (40, 70, 50) if self.editing_name else (45, 45, 52)
        brd_col  = (120, 200, 140) if self.editing_name else (20, 20, 24)
        pygame.draw.rect(self.screen, box_col,  self.name_box, border_radius=5)
        pygame.draw.rect(self.screen, brd_col,  self.name_box, 2, border_radius=5)
        shown_name = (self.map_name_text + "|") if self.editing_name else self.map_name
        n_img = self.small.render(shown_name[:26], True, (235, 235, 235))
        self.screen.blit(n_img, n_img.get_rect(midleft=(self.name_box.x + 6,
                                                          self.name_box.centery)))

        # --- firca / secili ---
        sub = self.small.render(f"Brush: {self.brush}  ([ ])", True, (180, 180, 180))
        self.screen.blit(sub, (12, self._lbl_y))
        cur = self.small.render(f"Tool: {C.TILE_NAMES[self.current]}", True, (200, 220, 200))
        self.screen.blit(cur, (12, self._lbl_y + 16))

        for b in self.tool_buttons:
            b.draw(self.screen, self.small, active=(b.value == self.current))

        # --- besin miktari ---
        lbl2 = self.small.render("Food amount:", True, (200, 220, 200))
        self.screen.blit(lbl2, (12, self.amount_label_y))
        box_col2 = (40, 80, 50) if self.editing_amount else (45, 45, 52)
        brd_col2 = (120, 200, 140) if self.editing_amount else (20, 20, 24)
        pygame.draw.rect(self.screen, box_col2, self.amount_box, border_radius=5)
        pygame.draw.rect(self.screen, brd_col2, self.amount_box, 2, border_radius=5)
        shown_amt = (self.amount_text + "|") if self.editing_amount else str(self.amount)
        a_img = self.font.render(shown_amt, True, (235, 235, 235))
        self.screen.blit(a_img, a_img.get_rect(center=self.amount_box.center))

        self.btn_save.draw(self.screen,   self.small)
        self.btn_new.draw(self.screen,    self.small)
        self.btn_load.draw(self.screen,   self.small)
        self.btn_clear.draw(self.screen,  self.small)
        self.btn_border.draw(self.screen, self.small)
        self.btn_menu.draw(self.screen,   self.small)

        if self.msg_timer > 0:
            m = self.small.render(self.message, True, (255, 230, 140))
            self.screen.blit(m, (12, C.SCREEN_H - 28))
