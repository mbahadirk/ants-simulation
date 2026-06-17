"""
Dunya / harita: izgara tabanli tile haritasi.

Karincalar surekli (float) koordinatlarda hareket eder, ama harita
hucrelerden olusur. Cesitli sorgular (gecilebilir mi, hucre tipi nedir,
isin nereye carpar) burada saglanir.
"""

import json
import os
from collections import deque

import numpy as np

import config as C

_NEIGHBORS8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]


class World:
    def __init__(self, grid=None, food_amount=None):
        if grid is None:
            grid = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int16)
        self.grid = grid
        self.nest_cell = None          # (col, row)
        self.nest_pos = None           # (x, y) piksel merkez
        self._locate_nest()
        self.delivered_food = 0        # yuvaya teslim edilen toplam besin

        # besin miktarlari: her FOOD hucresinde kalan birim sayisi (0=yok)
        self.food_amount = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int32)
        if food_amount is not None:
            fa = np.array(food_amount, dtype=np.int32)
            if fa.shape == self.food_amount.shape:
                self.food_amount = fa
        # FOOD hucresi olup miktari 0 olanlara varsayilan ata (eski/eksik veri)
        missing = (self.grid == C.FOOD) & (self.food_amount <= 0)
        self.food_amount[missing] = C.FOOD_DEFAULT_AMOUNT
        # FOOD olmayan yerlerde miktar 0
        self.food_amount[self.grid != C.FOOD] = 0

        # feromon alanlari (home izi, food izi)
        self.ph_home = np.zeros((C.GRID_H, C.GRID_W), dtype=np.float32)
        self.ph_food = np.zeros((C.GRID_H, C.GRID_W), dtype=np.float32)
        self._diffuse_acc = 0.0

        # besin kokusu (statik gradyan) - kaynaklardan yayilir
        self.food_odor = np.zeros((C.GRID_H, C.GRID_W), dtype=np.float32)
        self.odor_dirty = False        # besin bitince yeniden hesap gerekir
        self._odor_acc = 0.0
        self.compute_food_odor()

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
        self.nest_radius = 2.5 * C.CELL_SIZE

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

    def _type_rc(self, row, col):
        if 0 <= row < C.GRID_H and 0 <= col < C.GRID_W:
            return int(self.grid[row, col])
        return C.OBSTACLE

    def _blocked_rc(self, row, col):
        t = self._type_rc(row, col)
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
            self.food_amount[row, col] -= 1
            if self.food_amount[row, col] <= 0:
                # kaynak bitti -> hucre bosalir, koku yeniden hesaplanmali
                self.grid[row, col] = C.EMPTY
                self.food_amount[row, col] = 0
                self.odor_dirty = True
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

    # ------------------------------------------------ bilineer ornek + gradyan
    @staticmethod
    def _bilinear(arr, x, y):
        """arr izgarasini surekli (x,y) piksel konumunda bilineer interpolasyonla
        orntekler. Hucre kuantizasyonunu yumusatir -> puruzsuz gradyan."""
        cs = C.CELL_SIZE
        H, W = arr.shape
        fx = x / cs - 0.5
        fy = y / cs - 0.5
        c0 = int(np.floor(fx)); r0 = int(np.floor(fy))
        tx = fx - c0; ty = fy - r0
        c0c = min(max(c0, 0), W - 1); c1c = min(max(c0 + 1, 0), W - 1)
        r0c = min(max(r0, 0), H - 1); r1c = min(max(r0 + 1, 0), H - 1)
        v00 = arr[r0c, c0c]; v01 = arr[r0c, c1c]
        v10 = arr[r1c, c0c]; v11 = arr[r1c, c1c]
        top = v00 * (1 - tx) + v01 * tx
        bot = v10 * (1 - tx) + v11 * tx
        return float(top * (1 - ty) + bot * ty)

    def field_gradient(self, arr, x, y, step=None):
        """arr alaninin (x,y)'deki yerel gradyanini (gx, gy) merkezi farkla doner.
        Birim: alan-degeri / piksel. Yokus-yukari yon = (gx, gy)."""
        h = step or C.CHEM_GRAD_STEP
        gx = (self._bilinear(arr, x + h, y) - self._bilinear(arr, x - h, y)) / (2 * h)
        gy = (self._bilinear(arr, x, y + h) - self._bilinear(arr, x, y - h)) / (2 * h)
        return gx, gy

    def update_pheromones(self, dt):
        # buharlasma
        decay_home = max(0.0, 1.0 - C.PH_HOME_EVAPORATION * dt)
        self.ph_home *= decay_home
        # besin feromonu: "super iz" (esik ustu yogun yollar) cok daha yavas
        # buharlasir -> sik kullanilan besin yollari kalici olur.
        decay_food = max(0.0, 1.0 - C.PH_FOOD_EVAPORATION * dt)
        decay_food_strong = max(0.0, 1.0 - C.PH_FOOD_EVAPORATION
                                * C.PH_FOOD_STRONG_EVAP_MULT * dt)
        strong = self.ph_food > C.PH_FOOD_STRONG_THRESH
        self.ph_food = np.where(strong, self.ph_food * decay_food_strong,
                                self.ph_food * decay_food)

        # arada bir hafif yayilim (komsu ortalamasiyla)
        self._diffuse_acc += dt
        if self._diffuse_acc >= C.PH_DIFFUSE_EVERY:
            self._diffuse_acc = 0.0
            self.ph_home = self._diffuse(self.ph_home)
            self.ph_food = self._diffuse(self.ph_food)

        # besin bittiyse koku gradyanini yeniden hesapla (kisilmis)
        if self.odor_dirty:
            self._odor_acc += dt
            if self._odor_acc >= 0.75:
                self._odor_acc = 0.0
                self.odor_dirty = False
                self.compute_food_odor()

    @staticmethod
    def _diffuse(a):
        # 4-komsu hafif bulaniklastirma (kenarlar korunur)
        out = a.copy()
        out[1:-1, 1:-1] = (
            a[1:-1, 1:-1] * 0.6
            + (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) * 0.1
        )
        return out

    # ------------------------------------------------------------- besin koku
    def compute_food_odor(self):
        """
        Besin kokusu gradyanini cok-kaynakli BFS ile hesaplar.
        Her hucre icin en yakin besine olan (duvarlardan DOLANARAK) mesafe
        bulunur; koku = 1 - mesafe/menzil. Boylece koku harita boyunca genis
        bir gradyan olusturur ve karincalar yogunlugu tirmanip besine ulasir.
        Duvarlar (tas/engel) kokuyu hem engeller hem de etrafindan dolastirir.
        """
        H, W = C.GRID_H, C.GRID_W
        blocked = (self.grid == C.OBSTACLE) | (self.grid == C.STONE)
        INF = 1 << 30
        dist = np.full((H, W), INF, dtype=np.int32)

        q = deque()
        for r, c in np.argwhere(self.grid == C.FOOD):
            dist[r, c] = 0
            q.append((int(r), int(c)))

        while q:
            r, c = q.popleft()
            d1 = dist[r, c] + 1
            for dr, dc in _NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not blocked[nr, nc] and dist[nr, nc] > d1:
                    dist[nr, nc] = d1
                    q.append((nr, nc))

        odor = np.clip(1.0 - dist / float(C.ODOR_RANGE_CELLS), 0.0, 1.0).astype(np.float32)
        odor[blocked] = 0.0
        self.food_odor = odor

    def sample_odor(self, x, y):
        """Verilen konumdaki besin kokusu yogunlugu (0..1)."""
        if not self.in_bounds(x, y):
            return 0.0
        col = int(x // C.CELL_SIZE)
        row = int(y // C.CELL_SIZE)
        return float(self.food_odor[row, col])

    # ----------------------------------------------------------- raycasting
    def cast_ray(self, x, y, angle, max_range, ignore_first=True):
        """
        (x,y)'den 'angle' yonunde isin atar.
        Ilk carptigi nesne tipini ve mesafesini doner.
        Donus: (obj_type, distance). Carpma yoksa (EMPTY, max_range).

        KOSE-KESME ENGELI: isin iki capraz blok arasindaki bosluktan
        gecemez. Hucre hem satir hem sutun degistirdiginde (capraz gecis),
        aradaki iki dik hucre de bloke ise duvar olarak algilanir; boylece
        karincalar capraz bloklar arasindan "engel yok" sanmaz.
        Not: karincalar bu fonksiyonda gorunmez; onlari sim ayrica ekler.
        """
        cs = C.CELL_SIZE
        dx = np.cos(angle)
        dy = np.sin(angle)
        pcol = int(x // cs)
        prow = int(y // cs)
        d = C.RAY_STEP if ignore_first else 0.0
        while d <= max_range:
            px = x + dx * d
            py = y + dy * d
            if not self.in_bounds(px, py):
                return C.OBSTACLE, d
            col = int(px // cs)
            row = int(py // cs)
            # capraz gecis: aradaki iki dik hucre de bloke ise duvar say
            if col != pcol and row != prow:
                if self._blocked_rc(prow, col) and self._blocked_rc(row, pcol):
                    return self._type_rc(prow, col), d
            t = int(self.grid[row, col])
            if t == C.FOOD or t == C.STONE or t == C.OBSTACLE:
                return t, d
            if t == C.NEST:
                return C.NEST, d
            pcol, prow = col, row
            d += C.RAY_STEP
        return C.EMPTY, max_range

    def line_of_sight(self, x1, y1, x2, y2):
        """(x1,y1) ile (x2,y2) arasinda DUVAR (tas/engel) yoksa True.
        Baslangic ve hedef hucreleri haric. Duvar arkasini gormeyi engeller."""
        cs = C.CELL_SIZE
        scol, srow = int(x1 // cs), int(y1 // cs)
        tcol, trow = int(x2 // cs), int(y2 // cs)
        dx, dy = x2 - x1, y2 - y1
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return True
        ux, uy = dx / dist, dy / dist
        step = cs * 0.5            # yarim hucre adimi (performans icin yeterli)
        d = step
        while d < dist:
            col = int((x1 + ux * d) // cs)
            row = int((y1 + uy * d) // cs)
            if (col, row) != (scol, srow) and (col, row) != (tcol, trow):
                if self._blocked_rc(row, col):
                    return False
            d += step
        return True

    # ------------------------------------------------------- periyodik besin
    def spawn_random_food(self, rng, amount):
        """Bos bir hucrede besin olusturur. Yuvadan UZAK hucreler tercih edilir
        (gercek forage hedefi olussun, yuva-cevresi topla-getir ozendirilmesin)."""
        empties = np.argwhere(self.grid == C.EMPTY)
        if len(empties) == 0:
            return None
        ncol, nrow = self.nest_cell
        # her bos hucrenin yuvaya hucre-mesafesi
        d = np.hypot(empties[:, 1] - ncol, empties[:, 0] - nrow)
        far = empties[d >= C.FOOD_SPAWN_MIN_NEST_CELLS]
        pool = far if len(far) > 0 else empties
        r, c = pool[int(rng.integers(0, len(pool)))]
        self.grid[r, c] = C.FOOD
        self.food_amount[r, c] = int(amount)
        self.odor_dirty = True
        return (int(r), int(c))

    def food_near_nest_count(self, max_cells):
        """Yuvadan max_cells hucre icindeki FOOD hucre sayisi (RL bootstrap kontrolu)."""
        ncol, nrow = self.nest_cell
        ys, xs = np.where(self.grid == C.FOOD)
        if len(xs) == 0:
            return 0
        d = np.hypot(xs - ncol, ys - nrow)
        return int(np.count_nonzero(d <= max_cells))

    def spawn_food_near_nest(self, rng, amount, max_cells):
        """Yuvanin YAKININDA (max_cells icinde, ama yuvanin uzerinde degil) bos bir
        hucrede besin olusturur. RL curriculum bootstrap'i icin (kisa tur)."""
        ncol, nrow = self.nest_cell
        empties = np.argwhere(self.grid == C.EMPTY)
        if len(empties) == 0:
            return None
        d = np.hypot(empties[:, 1] - ncol, empties[:, 0] - nrow)
        near = empties[(d <= max_cells) & (d >= 3.0)]
        if len(near) == 0:
            return None
        r, c = near[int(rng.integers(0, len(near)))]
        self.grid[r, c] = C.FOOD
        self.food_amount[r, c] = int(amount)
        self.odor_dirty = True
        return (int(r), int(c))

    # ----------------------------------------------------------------- kayit
    def to_dict(self):
        return {
            "grid_w": C.GRID_W,
            "grid_h": C.GRID_H,
            "cell_size": C.CELL_SIZE,
            "grid": self.grid.tolist(),
            "food_amount": self.food_amount.tolist(),
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
        if grid.shape != (C.GRID_H, C.GRID_W):
            raise ValueError(
                f"Harita boyutu uyumsuz: {grid.shape} != {(C.GRID_H, C.GRID_W)}. "
                "config.py GRID_W/GRID_H ile uyusmuyor."
            )
        fa = data.get("food_amount")
        return cls(grid=grid, food_amount=fa)


def make_random_world(seed=None):
    """Rastgele EGITIM haritasi: rastgele yuva konumu, duvar segmentleri ve
    yuvadan uzak besin kumeleri. Coklu-harita egitiminde modellerin tek bir
    haritayi ezberlemesini onlemek, genellesmis forage davranisi evrimlestirmek
    icin kullanilir ('New Random Map' butonu)."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((C.GRID_H, C.GRID_W), dtype=np.int16)

    # cevre duvari
    grid[0, :] = C.OBSTACLE
    grid[-1, :] = C.OBSTACLE
    grid[:, 0] = C.OBSTACLE
    grid[:, -1] = C.OBSTACLE

    # yuva: kenarlardan uzak rastgele konum (2x2)
    cx = int(rng.integers(12, C.GRID_W - 13))
    cy = int(rng.integers(8, C.GRID_H - 9))
    grid[cy:cy + 2, cx:cx + 2] = C.NEST

    # rastgele duvar segmentleri (yatay/dikey tas-engel cizgileri)
    n_walls = int(rng.integers(8, 16))
    for _ in range(n_walls):
        tile = C.STONE if rng.random() < 0.5 else C.OBSTACLE
        length = int(rng.integers(3, 11))
        horizontal = rng.random() < 0.5
        r = int(rng.integers(2, C.GRID_H - 2))
        c = int(rng.integers(2, C.GRID_W - 2))
        for i in range(length):
            rr = r + (0 if horizontal else i)
            cc = c + (i if horizontal else 0)
            if not (1 <= rr < C.GRID_H - 1 and 1 <= cc < C.GRID_W - 1):
                break
            # yuvanin yakinini kapatma (cikis yolu kalsin)
            if abs(cc - cx) <= 4 and abs(rr - cy) <= 4:
                continue
            if grid[rr, cc] == C.EMPTY:
                grid[rr, cc] = tile

    # besin kumeleri: yuvadan uzakta (gercek forage hedefleri)
    n_clusters = int(rng.integers(5, 9))
    placed = 0
    attempts = 0
    while placed < n_clusters and attempts < 200:
        attempts += 1
        bc = int(rng.integers(3, C.GRID_W - 3))
        br = int(rng.integers(3, C.GRID_H - 3))
        if np.hypot(bc - cx, br - cy) < C.FOOD_SPAWN_MIN_NEST_CELLS:
            continue
        n_food = int(rng.integers(6, 13))
        for _ in range(n_food):
            oc = bc + int(rng.integers(-3, 4))
            orr = br + int(rng.integers(-3, 4))
            if (1 <= oc < C.GRID_W - 1 and 1 <= orr < C.GRID_H - 1
                    and grid[orr, oc] == C.EMPTY):
                grid[orr, oc] = C.FOOD
        placed += 1

    return World(grid=grid)


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
