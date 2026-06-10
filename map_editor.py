"""
UI tabanli harita editoru (tamamen pygame).

- Sol panelde arac butonlari: Besin, Tas, Engel, Yuva, Sil.
- Sol tik (basili tutarak suru) ile secili nesneyi yerlestirir.
- Sag tik siler.
- Firca boyutu [ ve ] tuslari ile ayarlanir.
- Kaydet butonu / Ctrl+S -> maps/default_map.json
- Temizle butonu haritayi bosaltir, Menu butonu ana menuye doner.
"""

import numpy as np
import pygame

import config as C
from world import World

PANEL_W = 190


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
    def __init__(self, screen, grid=None):
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 13)
        if grid is None:
            grid = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int16)
        self.grid = grid.copy()

        # cizim alani olcegi
        area_w = C.SCREEN_W - PANEL_W
        self.scale = min(area_w / C.WORLD_W, C.SCREEN_H / C.WORLD_H)
        self.ox = PANEL_W
        self.oy = 0

        self.current = C.FOOD
        self.brush = 1
        self.message = ""
        self.msg_timer = 0.0

        self._build_buttons()

    def _build_buttons(self):
        x = 12
        w = PANEL_W - 24
        y = 90
        self.tool_buttons = []
        tools = [
            ("1 Besin", C.FOOD, C.COLORS[C.FOOD]),
            ("2 Tas", C.STONE, C.COLORS[C.STONE]),
            ("3 Engel", C.OBSTACLE, C.COLORS[C.OBSTACLE]),
            ("4 Yuva", C.NEST, C.COLORS[C.NEST]),
            ("0 Sil", C.EMPTY, (50, 50, 55)),
        ]
        for label, val, col in tools:
            self.tool_buttons.append(Button(label, x, y, w, 38, value=val, color=col))
            y += 46

        y += 14
        self.btn_save = Button("Kaydet (Ctrl+S)", x, y, w, 36, color=(40, 110, 60)); y += 44
        self.btn_clear = Button("Temizle", x, y, w, 36, color=(120, 60, 40)); y += 44
        self.btn_border = Button("Cerceve Duvar", x, y, w, 36, color=(70, 60, 50)); y += 44
        self.btn_menu = Button("Menu (ESC)", x, y, w, 36, color=(60, 60, 80))

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

    def _flash(self, msg):
        self.message = msg
        self.msg_timer = 2.0

    # ------------------------------------------------------------------- loop
    def run(self):
        """Editoru calistirir; cikista 'menu' doner."""
        clock = pygame.time.Clock()
        painting = None  # 'paint' / 'erase' / None
        running = True
        while running:
            dt = clock.tick(C.FPS) / 1000.0
            self.msg_timer = max(0.0, self.msg_timer - dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
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

        return "menu"

    def _handle_panel_click(self, pos):
        for b in self.tool_buttons:
            if b.hit(pos):
                self.current = b.value
                return
        if self.btn_save.hit(pos):
            self._save()
        elif self.btn_clear.hit(pos):
            self.grid[:] = C.EMPTY
            self._flash("Harita temizlendi")
        elif self.btn_border.hit(pos):
            self.grid[0, :] = C.OBSTACLE
            self.grid[-1, :] = C.OBSTACLE
            self.grid[:, 0] = C.OBSTACLE
            self.grid[:, -1] = C.OBSTACLE
            self._flash("Cerceve eklendi")
        elif self.btn_menu.hit(pos):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

    def _save(self):
        if not np.any(self.grid == C.NEST):
            self._flash("UYARI: Yuva yok! Once yuva koy.")
            return
        World(grid=self.grid.copy()).save(C.MAP_FILE)
        self._flash(f"Kaydedildi -> {C.MAP_FILE}")

    # ------------------------------------------------------------------- draw
    def _draw(self):
        self.screen.fill((30, 28, 26))
        cs = C.CELL_SIZE * self.scale

        # izgara hucreleri
        for row in range(C.GRID_H):
            for col in range(C.GRID_W):
                t = int(self.grid[row, col])
                x = self.ox + col * cs
                y = self.oy + row * cs
                rect = pygame.Rect(x, y, cs + 1, cs + 1)
                base = C.COLORS.get(t, C.COLORS[C.EMPTY])
                pygame.draw.rect(self.screen, base, rect)
        # izgara cizgileri
        for col in range(C.GRID_W + 1):
            x = self.ox + col * cs
            pygame.draw.line(self.screen, (45, 43, 40), (x, 0), (x, C.GRID_H * cs))
        for row in range(C.GRID_H + 1):
            y = self.oy + row * cs
            pygame.draw.line(self.screen, (45, 43, 40), (self.ox, y), (self.ox + C.GRID_W * cs, y))

        # firca onizleme
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
        title = self.font.render("HARITA EDITORU", True, (240, 240, 240))
        self.screen.blit(title, (12, 18))
        sub = self.small.render(f"Firca: {self.brush}  ([ ])", True, (180, 180, 180))
        self.screen.blit(sub, (12, 44))
        cur = self.small.render(f"Secili: {C.TILE_NAMES[self.current]}", True, (200, 220, 200))
        self.screen.blit(cur, (12, 62))

        for b in self.tool_buttons:
            b.draw(self.screen, self.small, active=(b.value == self.current))
        self.btn_save.draw(self.screen, self.small)
        self.btn_clear.draw(self.screen, self.small)
        self.btn_border.draw(self.screen, self.small)
        self.btn_menu.draw(self.screen, self.small)

        if self.msg_timer > 0:
            m = self.small.render(self.message, True, (255, 230, 140))
            self.screen.blit(m, (12, C.SCREEN_H - 28))
