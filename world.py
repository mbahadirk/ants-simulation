"""
Dunya / harita: izgara tabanli tile haritasi.

Karincalar surekli (float) koordinatlarda hareket eder, ama harita
hucrelerden olusur. Cesitli sorgular (gecilebilir mi, hucre tipi nedir,
isin nereye carpar) burada saglanir.
"""

import json
import os

import numpy as np

import config as C


class World:
    def __init__(self, grid=None):
        if grid is None:
            grid = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int16)
        self.grid = grid
        self.nest_cell = None          # (col, row)
        self.nest_pos = None           # (x, y) piksel merkez
        self._locate_nest()
        self.delivered_food = 0        # yuvaya teslim edilen toplam besin

        # feromon alanlari (home izi, food izi)
        self.ph_home = np.zeros((C.GRID_H, C.GRID_W), dtype=np.float32)
        self.ph_food = np.zeros((C.GRID_H, C.GRID_W), dtype=np.float32)
        self._diffuse_acc = 0.0

    # ------------------------------------------------------------------ nest
    def _locate_nest(self):
        ys, xs = np.where(self.grid == C.NEST)
        if len(xs) == 0:
            # yuva yoksa ortaya bir tane koy
            cx, cy = C.GRID_W // 2, C.GRID_H // 2
            self.grid[cy, cx] = C.NEST
            ys, xs = np.array([cy]), np.array([cx])
        # yuva merkezini ortalama hucreden al
        cx = float(xs.mean())
        cy = float(ys.mean())
        self.nest_cell = (int(round(cx)), int(round(cy)))
        self.nest_pos = ((cx + 0.5) * C.CELL_SIZE, (cy + 0.5) * C.CELL_SIZE)
        self.nest_radius = 1.6 * C.CELL_SIZE

    # -------------------------------------------------------------- sorgular
    def in_bounds(self, x, y):
        return 0 <= x < C.WORLD_W and 0 <= y < C.WORLD_H

    def cell_at(self, x, y):
        if not self.in_bounds(x, y):
            return C.OBSTACLE  # disari = duvar gibi
        col = int(x // C.CELL_SIZE)
        row = int(y // C.CELL_SIZE)
        return int(self.grid[row, col])

    def is_blocked(self, x, y):
        t = self.cell_at(x, y)
        return t == C.STONE or t == C.OBSTACLE

    def cell_index(self, x, y):
        return int(x // C.CELL_SIZE), int(y // C.CELL_SIZE)

    # ----------------------------------------------------------------- besin
    def take_food(self, x, y):
        """Verilen konumdaki hucrede besin varsa alir (True doner)."""
        if not self.in_bounds(x, y):
            return False
        col = int(x // C.CELL_SIZE)
        row = int(y // C.CELL_SIZE)
        if self.grid[row, col] == C.FOOD:
            if not C.NEST_REGROW_FOOD:
                self.grid[row, col] = C.EMPTY
            return True
        return False

    def at_nest(self, x, y):
        dx = x - self.nest_pos[0]
        dy = y - self.nest_pos[1]
        return (dx * dx + dy * dy) <= (self.nest_radius * self.nest_radius)

    def food_count(self):
        return int(np.count_nonzero(self.grid == C.FOOD))

    # ------------------------------------------------------------- feromon
    def deposit(self, field, x, y, amount):
        if not self.in_bounds(x, y):
            return
        col = int(x // C.CELL_SIZE)
        row = int(y // C.CELL_SIZE)
        arr = self.ph_home if field == C.PH_HOME else self.ph_food
        arr[row, col] = min(C.PH_MAX, arr[row, col] + amount)

    def sample(self, field, x, y):
        """Verilen konumdaki feromon yogunlugu (0..1 normalize)."""
        if not self.in_bounds(x, y):
            return 0.0
        col = int(x // C.CELL_SIZE)
        row = int(y // C.CELL_SIZE)
        arr = self.ph_home if field == C.PH_HOME else self.ph_food
        return float(arr[row, col]) / C.PH_MAX

    def update_pheromones(self, dt):
        # buharlasma
        decay = max(0.0, 1.0 - C.PH_EVAPORATION * dt)
        self.ph_home *= decay
        self.ph_food *= decay

        # arada bir hafif yayilim (komsu ortalamasiyla)
        self._diffuse_acc += dt
        if self._diffuse_acc >= C.PH_DIFFUSE_EVERY:
            self._diffuse_acc = 0.0
            self.ph_home = self._diffuse(self.ph_home)
            self.ph_food = self._diffuse(self.ph_food)

    @staticmethod
    def _diffuse(a):
        # 4-komsu hafif bulaniklastirma (kenarlar korunur)
        out = a.copy()
        out[1:-1, 1:-1] = (
            a[1:-1, 1:-1] * 0.6
            + (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) * 0.1
        )
        return out

    # ----------------------------------------------------------- raycasting
    def cast_ray(self, x, y, angle, max_range, ignore_first=True):
        """
        (x,y)'den 'angle' yonunde isin atar.
        Ilk carptigi nesne tipini ve mesafesini doner.
        Donus: (obj_type, distance). Carpma yoksa (EMPTY, max_range).
        Not: karincalar bu fonksiyonda gorunmez; onlari sim ayrica ekler.
        """
        dx = np.cos(angle)
        dy = np.sin(angle)
        d = C.RAY_STEP if ignore_first else 0.0
        while d <= max_range:
            px = x + dx * d
            py = y + dy * d
            if not self.in_bounds(px, py):
                return C.OBSTACLE, d
            t = self.cell_at(px, py)
            if t == C.FOOD or t == C.STONE or t == C.OBSTACLE:
                return t, d
            if t == C.NEST:
                return C.NEST, d
            d += C.RAY_STEP
        return C.EMPTY, max_range

    # ----------------------------------------------------------------- kayit
    def to_dict(self):
        return {
            "grid_w": C.GRID_W,
            "grid_h": C.GRID_H,
            "cell_size": C.CELL_SIZE,
            "grid": self.grid.tolist(),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        grid = np.array(data["grid"], dtype=np.int16)
        return cls(grid=grid)


def make_default_world():
    """Basit ornek bir harita uretir (yuva ortada, kenarlarda besin, biraz tas)."""
    grid = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int16)

    # cevre duvari
    grid[0, :] = C.OBSTACLE
    grid[-1, :] = C.OBSTACLE
    grid[:, 0] = C.OBSTACLE
    grid[:, -1] = C.OBSTACLE

    # yuva (ortada 2x2)
    cx, cy = C.GRID_W // 2, C.GRID_H // 2
    for ddy in range(2):
        for ddx in range(2):
            grid[cy + ddy, cx + ddx] = C.NEST

    rng = np.random.default_rng(7)

    # besin kumeleri (koselerde) - uzak hedefler
    clusters = [(6, 6), (C.GRID_W - 8, 6), (6, C.GRID_H - 8), (C.GRID_W - 8, C.GRID_H - 8)]
    for bx, by in clusters:
        for _ in range(16):
            ox = bx + int(rng.integers(-3, 4))
            oy = by + int(rng.integers(-3, 4))
            if 1 <= ox < C.GRID_W - 1 and 1 <= oy < C.GRID_H - 1 and grid[oy, ox] == C.EMPTY:
                grid[oy, ox] = C.FOOD

    # yuva cevresine yakin besin halkasi - kesfin baslamasi icin
    # (keaif/evrim bootstrap'i: yakin besin -> iz olusur -> uzaga yayilir)
    for _ in range(70):
        ang = rng.uniform(0, 2 * np.pi)
        rad = int(rng.integers(4, 15))
        ox = int(cx + np.cos(ang) * rad)
        oy = int(cy + np.sin(ang) * rad)
        if 1 <= ox < C.GRID_W - 1 and 1 <= oy < C.GRID_H - 1 and grid[oy, ox] == C.EMPTY:
            grid[oy, ox] = C.FOOD

    # dagilmis taslar (yuvaya cok yakin olmasin)
    for _ in range(30):
        ox = int(rng.integers(2, C.GRID_W - 2))
        oy = int(rng.integers(2, C.GRID_H - 2))
        if grid[oy, ox] == C.EMPTY and abs(ox - cx) > 4 and abs(oy - cy) > 4:
            grid[oy, ox] = C.STONE

    return World(grid=grid)
