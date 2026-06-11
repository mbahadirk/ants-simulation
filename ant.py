"""
Karinca varligi.

Her karincanin: konumu, yon acisi (heading), enerjisi, yasi, yasam suresi,
bir beyni (LSTMPolicy) ve onundeki 180 derecelik sektor tabanli gorusu vardir.

Sensor -> beyin (forward) -> aksiyon (ileri/geri/sol/sag/bekle) -> hareket.
Hareket sirasinda tas/engel gecilemez; besin alinir, yuvaya teslim edilir.
"""

import numpy as np

import config as C
from neural_network import LSTMPolicy

# Nesne tipi -> sektor one-hot indeksi (0:besin 1:tas 2:engel 3:karinca 4:yuva)
_VIS_OBJ_INDEX = {C.FOOD: 0, C.STONE: 1, C.OBSTACLE: 2, C.NEST: 4}
_ANT_OBJ_INDEX = 3


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Ant:
    _next_id = 0

    def __init__(self, x, y, genome=None, rng=None, generation=0):
        self.id = Ant._next_id
        Ant._next_id += 1

        self.rng = rng or np.random.default_rng()
        self.x = float(x)
        self.y = float(y)
        self.heading = float(self.rng.uniform(0, 2 * np.pi))

        self.brain = LSTMPolicy(genome=genome, rng=self.rng)

        self.energy = C.ENERGY_MAX
        self.age = 0.0
        self.lifespan = float(self.rng.uniform(C.LIFESPAN_MIN, C.LIFESPAN_MAX))
        self.base_lifespan = self.lifespan   # besin bulunca buna gore uzatilir
        self.carrying = False
        self.alive = True
        self.generation = generation

        # istatistik / ureme icin
        self.food_delivered = 0
        self.food_found = 0
        self.wall_hits = 0
        self.idle_steps = 0

        # odul sekillendirme (reward shaping)
        self.fitness_bonus = 0.0       # ilerleme odullerinin toplami
        self.min_home_dist = None      # tasirken yuvaya ulasilan en kucuk mesafe
        self.prev_home_dist = None     # tasirken bir onceki adimin yuva mesafesi
        self.max_odor_seen = 0.0       # bos gezerken tirmanilan en yuksek koku
        self.carry_distance = 0.0      # besin aldiktan sonra katedilen mesafe (kisa iz icin)

        # debug icin son gorulen nesneler: (tip, x, y, dist)
        self.last_seen = []
        self.last_action = C.ACTION_NONE
        self.last_odor = 0.0       # son algilanan besin kokusu (max anten)
        self.last_food_ph = 0.0    # son algilanan food feromonu (max anten)

        # benzersiz bir renk (debug'da ayirt etmek icin) -> genoma bagli
        g = self.brain.get_genome()
        h = int(abs(np.sum(g[:8]) * 1000)) % 360
        self.hue = h

    # ---------------------------------------------------------------- genom
    def genome(self):
        return self.brain.get_genome()

    def fitness(self):
        # Not: besin bulma odulu (mesafeye gore) fitness_bonus icine eklenir.
        return (self.food_delivered * C.FITNESS_DELIVER_W
                + self.fitness_bonus)

    # -------------------------------------------------------------- sensorler
    def sense(self, world, neighbors):
        """
        DAIRESEL gorus: karincanin etrafindaki VISION_RANGE yaricapli daire
        icindeki nesneleri gorur. Her nesne tipi (besin/tas/engel/karinca/yuva)
        icin gorus hatti ACIK (onunde duvar olmayan) EN YAKIN ornek bulunur.
        Duvar arkasindaki nesne GORUNMEZ. Her tip icin girdi:
        [var_mi, yakinlik(0..1), sin(rel_aci), cos(rel_aci)].
        """
        cs = C.CELL_SIZE
        rng = C.VISION_RANGE
        half_fov = C.VISION_FOV / 2.0
        bin_w = C.VISION_FOV / C.N_SECTORS
        ns = C.N_SECTORS

        # her sektorde: en yakin nesnenin mesafesi, one-hot indeksi, konumu
        sect_dist = [rng] * ns
        sect_idx = [None] * ns
        sect_pos = [None] * ns

        r0 = max(0, int((self.y - rng) // cs))
        r1 = min(C.GRID_H - 1, int((self.y + rng) // cs))
        c0 = max(0, int((self.x - rng) // cs))
        c1 = min(C.GRID_W - 1, int((self.x + rng) // cs))

        grid = world.grid
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                oi = _VIS_OBJ_INDEX.get(int(grid[r, c]))
                if oi is None:        # EMPTY ya da gorulmeyen tip
                    continue
                # hucrenin karincaya en yakin noktasi -> mesafe
                px = max(c * cs, min(self.x, (c + 1) * cs))
                py = max(r * cs, min(self.y, (r + 1) * cs))
                dist = np.hypot(px - self.x, py - self.y)
                if dist > rng:
                    continue
                # hucre merkezine gore yon (heading'e gore)
                cx = (c + 0.5) * cs
                cy = (r + 0.5) * cs
                rel = _wrap_angle(np.arctan2(cy - self.y, cx - self.x) - self.heading)
                if abs(rel) > half_fov:        # 180 derece disinda
                    continue
                k = int((rel + half_fov) / bin_w)
                if k >= ns:
                    k = ns - 1
                # bu yonde daha yakin nesne varsa onu gormeye devam (okluzyon)
                if dist < sect_dist[k]:
                    sect_dist[k] = dist
                    sect_idx[k] = oi
                    sect_pos[k] = (cx, cy)

        # diger karincalar
        for other in neighbors:
            if other is self or not other.alive:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist = np.hypot(dx, dy)
            if dist > rng or dist < 1e-3:
                continue
            rel = _wrap_angle(np.arctan2(dy, dx) - self.heading)
            if abs(rel) > half_fov:
                continue
            k = int((rel + half_fov) / bin_w)
            if k >= ns:
                k = ns - 1
            if dist < sect_dist[k]:
                sect_dist[k] = dist
                sect_idx[k] = _ANT_OBJ_INDEX
                sect_pos[k] = (other.x, other.y)

        # gorus vektoru + debug bilgisi
        vis = np.zeros(C.VISION_INPUTS, dtype=np.float32)
        self.last_seen = []
        for k in range(ns):
            if sect_idx[k] is None:
                continue
            base = k * C.SECTOR_FEATURES
            vis[base] = 1.0 - sect_dist[k] / rng      # yakinlik (1=cok yakin)
            vis[base + 1 + sect_idx[k]] = 1.0
            self.last_seen.append((sect_idx[k], sect_pos[k][0], sect_pos[k][1], sect_dist[k]))

        feats = vis

        # --- homing (yuva yonu, path integration) ---
        ndx = world.nest_pos[0] - self.x
        ndy = world.nest_pos[1] - self.y
        ndist = np.hypot(ndx, ndy)
        nrel = _wrap_angle(np.arctan2(ndy, ndx) - self.heading)
        max_d = np.hypot(C.WORLD_W, C.WORLD_H)
        homing = np.array([
            np.sin(nrel),
            np.cos(nrel),
            min(1.0, ndist / max_d),
        ], dtype=np.float32)

        # --- feromon antenleri (sol/orta/sag x home/food) ---
        ph = np.zeros(C.PHEROMONE_INPUTS, dtype=np.float32)
        i = 0
        for off in C.PH_SAMPLE_ANGLES:
            ang = self.heading + off
            px = self.x + np.cos(ang) * C.PH_SAMPLE_DIST
            py = self.y + np.sin(ang) * C.PH_SAMPLE_DIST
            ph[i] = world.sample(C.PH_HOME, px, py)
            ph[i + 1] = world.sample(C.PH_FOOD, px, py)
            i += 2

        # --- besin kokusu antenleri (sol/orta/sag) ---
        odor = np.zeros(C.ODOR_INPUTS, dtype=np.float32)
        for j, off in enumerate(C.PH_SAMPLE_ANGLES):
            ang = self.heading + off
            px = self.x + np.cos(ang) * C.ODOR_SAMPLE_DIST
            py = self.y + np.sin(ang) * C.ODOR_SAMPLE_DIST
            odor[j] = world.sample_odor(px, py)
        # debug paneli icin: karincanin gercekten algiladigi degerler
        self.last_odor = float(odor.max())
        self.last_food_ph = float(ph[1::2].max()) if C.PHEROMONE_INPUTS else 0.0

        extra = np.array([
            1.0 if self.carrying else 0.0,
            self.energy,
        ], dtype=np.float32)
        return np.concatenate([feats, homing, ph, odor, extra])

    # --------------------------------------------------------------- guncelle
    def update(self, dt, world, neighbors):
        """
        Bir adim: algila -> dusun -> hareket et -> besin/aclik/yas islemleri.
        Donus: olay sozlugu, orn. {'delivered': True} ureme icin sim tarafindan kullanilir.
        """
        events = {"delivered": False, "picked": False}
        if not self.alive:
            return events

        obs = self.sense(world, neighbors)
        action = self.brain.forward(obs)
        self.last_action = action

        # --- hareket ---
        ox, oy = self.x, self.y
        moved = True
        if action == C.ACTION_LEFT:
            self.heading -= C.TURN_SPEED * dt
        elif action == C.ACTION_RIGHT:
            self.heading += C.TURN_SPEED * dt
        elif action == C.ACTION_FORWARD:
            moved = self._try_move(C.MOVE_SPEED * dt, world)
        elif action == C.ACTION_BACK:
            moved = self._try_move(-C.BACK_SPEED * dt, world)
        # ACTION_NONE -> hareket yok

        self.heading = _wrap_angle(self.heading)
        disp = np.hypot(self.x - ox, self.y - oy)  # bu adimda katedilen mesafe

        # --- cezalar (enerji + fitness) ---
        # duvara/tasa carpip ilerleyemediyse ceza
        if action in (C.ACTION_FORWARD, C.ACTION_BACK) and not moved:
            self.energy -= C.WALL_PENALTY_RATE * dt
            self.fitness_bonus -= C.WALL_FIT_PENALTY
            self.wall_hits += 1
            # en dis cerceveye (harita kenari) carptiysa EK ceza
            cs = C.CELL_SIZE
            if (self.x < cs or self.x > C.WORLD_W - cs
                    or self.y < cs or self.y > C.WORLD_H - cs):
                self.energy -= C.BORDER_PENALTY_RATE * dt
                self.fitness_bonus -= C.BORDER_FIT_PENALTY
        # 'bekle' (sabit durma) cezasi
        if action == C.ACTION_NONE:
            self.energy -= C.IDLE_PENALTY_RATE * dt
            self.fitness_bonus -= C.IDLE_FIT_PENALTY
            self.idle_steps += 1

        # --- besin alma ---
        if not self.carrying and world.take_food(self.x, self.y):
            self.carrying = True
            self.energy = C.ENERGY_MAX  # besin buldu -> aclik sifirlanir
            self.food_found += 1
            events["picked"] = True
            self.carry_distance = 0.0   # kisa feromon izi icin sifirla
            # besin bulundugu yer yuvadan ne kadar uzaksa o kadar cok odul
            dnest = np.hypot(self.x - world.nest_pos[0], self.y - world.nest_pos[1])
            self.fitness_bonus += C.FITNESS_FIND_BASE + dnest * C.FITNESS_FIND_DIST_W
            # geri donus odulu icin baslangic mesafelerini kaydet
            self.min_home_dist = dnest
            self.prev_home_dist = dnest
            # besin bulan karincaya 1 omur (taban) kadar ek sure
            if C.LIFESPAN_FOOD_BONUS:
                self.lifespan = min(self.lifespan + self.base_lifespan,
                                    self.base_lifespan * C.LIFESPAN_MAX_MULT)

        # --- yuvaya teslim ---
        if self.carrying and world.at_nest(self.x, self.y):
            self.carrying = False
            self.energy = C.ENERGY_MAX
            self.food_delivered += 1
            events["delivered"] = True
            self.min_home_dist = None
            self.prev_home_dist = None
            self.max_odor_seen = 0.0    # yeni arayis basliyor

        # --- odul sekillendirme: dogru yonde ilerlemeyi odullendir ---
        if self.carrying:
            home_dist = np.hypot(self.x - world.nest_pos[0], self.y - world.nest_pos[1])
            # yuvaya YENI en yakin mesafeye ulastiysa odul
            if self.min_home_dist is not None and home_dist < self.min_home_dist:
                self.fitness_bonus += (self.min_home_dist - home_dist) * C.RETURN_REWARD_W
                self.min_home_dist = home_dist
            # besin tasirken yuvadan UZAKLASTIYSA ceza (yemi alip donmeyenler elenir)
            if self.prev_home_dist is not None and home_dist > self.prev_home_dist:
                self.fitness_bonus -= (home_dist - self.prev_home_dist) * C.CARRY_AWAY_PENALTY_W
            self.prev_home_dist = home_dist
        else:
            # bos gezerken besin kokusunu YENI en yuksek seviyeye tirmandiysa odul
            if self.last_odor > self.max_odor_seen:
                self.fitness_bonus += (self.last_odor - self.max_odor_seen) * C.FORAGE_REWARD_W
                self.max_odor_seen = self.last_odor

        # --- feromon birak (antenle koklananin kaynagi) ---
        if self.carrying:
            # besin feromonu besin aldiktan sonraki belli mesafe boyunca birakilir.
            # COK besin teslim etmis (basarili) karincalar DAHA GUCLU iz birakir ->
            # sik kullanilan besin yollari "super iz"e (mor) doner.
            self.carry_distance += disp
            if self.carry_distance <= C.PH_FOOD_TRAIL_DIST:
                strength = 1.0 + self.food_delivered * C.PH_SUCCESS_FACTOR
                world.deposit(C.PH_FOOD, self.x, self.y,
                              C.PH_DEPOSIT_FOOD * strength * dt * 60)
        else:
            world.deposit(C.PH_HOME, self.x, self.y, C.PH_DEPOSIT_HOME * dt * 60)

        # --- enerji / yas ---
        self.energy -= dt / C.STARVE_TIME
        self.age += dt
        if self.energy <= 0.0 or self.age >= self.lifespan:
            self.alive = False

        return events

    def _try_move(self, dist, world):
        """Hareketi dener; en az bir eksende ilerlediyse True, tamamen blokeyse False."""
        ox, oy = self.x, self.y
        nx = self.x + np.cos(self.heading) * dist
        ny = self.y + np.sin(self.heading) * dist
        # eksen bazli kayma: bir eksen bloke olsa bile digerinde kayabilsin
        if not world.is_blocked(nx, self.y) and world.in_bounds(nx, self.y):
            self.x = nx
        if not world.is_blocked(self.x, ny) and world.in_bounds(self.x, ny):
            self.y = ny
        # konum belirgin sekilde degistiyse hareket etti say
        return abs(self.x - ox) > 1e-6 or abs(self.y - oy) > 1e-6
