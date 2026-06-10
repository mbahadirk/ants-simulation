"""
Kamera: dunya <-> ekran donusumu, zoom ve pan (kaydirma).

z tusu zoom seviyelerini dolasir; ok tuslari / mouse ile pan yapilir.
Bir karinca takip ediliyorsa kamera onu merkezler.
"""

import config as C

ZOOM_LEVELS = [1.0, 1.6, 2.5, 4.0]


class Camera:
    def __init__(self):
        self.zoom_idx = 0
        self.zoom = ZOOM_LEVELS[0]
        # dunya koordinatinda kameranin sol-ust kosesi
        self.x = 0.0
        self.y = 0.0
        self.follow = None  # takip edilen Ant ya da None

    def cycle_zoom(self):
        self.zoom_idx = (self.zoom_idx + 1) % len(ZOOM_LEVELS)
        self.zoom = ZOOM_LEVELS[self.zoom_idx]
        self._clamp()

    def reset(self):
        self.zoom_idx = 0
        self.zoom = ZOOM_LEVELS[0]
        self.x = 0.0
        self.y = 0.0
        self.follow = None

    def pan(self, dx, dy):
        self.follow = None
        self.x += dx / self.zoom
        self.y += dy / self.zoom
        self._clamp()

    def center_on(self, wx, wy):
        view_w = C.SCREEN_W / self.zoom
        view_h = C.SCREEN_H / self.zoom
        self.x = wx - view_w / 2
        self.y = wy - view_h / 2
        self._clamp()

    def update(self):
        if self.follow is not None and self.follow.alive:
            self.center_on(self.follow.x, self.follow.y)

    def _clamp(self):
        view_w = C.SCREEN_W / self.zoom
        view_h = C.SCREEN_H / self.zoom
        max_x = max(0.0, C.WORLD_W - view_w)
        max_y = max(0.0, C.WORLD_H - view_h)
        self.x = min(max(self.x, 0.0), max_x)
        self.y = min(max(self.y, 0.0), max_y)

    # ---------------------------------------------------- donusum yardimcilari
    def world_to_screen(self, wx, wy):
        return (wx - self.x) * self.zoom, (wy - self.y) * self.zoom

    def screen_to_world(self, sx, sy):
        return sx / self.zoom + self.x, sy / self.zoom + self.y

    def scale(self, v):
        return v * self.zoom
