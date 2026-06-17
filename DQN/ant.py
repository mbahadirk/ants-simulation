"""
Karinca varligi.

Her karincanin: konumu, yon acisi (heading), enerjisi, yasi, yasam suresi,
bir beyni (MLP veya LSTM; config.BRAIN_ARCH) ve 180 derecelik sektor gorusu vardir.

Sensor -> beyin (forward) -> aksiyon (ileri/geri/sol/sag/bekle) -> hareket.
Hareket sirasinda tas/engel gecilemez; besin alinir, yuvaya teslim edilir.
"""

import numpy as np

import config as C
from neural_network import make_brain

# Nesne tipi -> sektor one-hot indeksi (0:besin 1:engelli[tas|engel] 2:karinca)
# Yuva goruten cikarildi (homing girdisi zaten karsilar); tas+engel "engelli"de birlesti.
_VIS_OBJ_INDEX = {C.FOOD: 0, C.STONE: 1, C.OBSTACLE: 1}
_ANT_OBJ_INDEX = 2


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

        self.brain = make_brain(genome=genome, rng=self.rng)

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

        # odul (DQN: adim-basina; neuroevo: omur boyu toplam fitness_bonus)
        self.fitness_bonus = 0.0       # tum adim-odullerinin toplami
        self.last_reward = 0.0         # son adimin odulu (DQN gecisi icin)
        self.epsilon = 0.0             # per-ant kesif orani (DQN'de doumda atanir)
        self._max_d = float(np.hypot(C.WORLD_W, C.WORLD_H))  # potansiyel normalizasyonu
        self.min_home_dist = None      # (eski neuroevo alanlari; save/load uyumu)
        self.prev_home_dist = None
        self.max_odor_seen = 0.0
        self.max_food_ph_seen = 0.0
        self.carry_distance = 0.0      # besin aldiktan sonra katedilen mesafe (kisa iz icin)
        self.last_find_dist = 0.0      # son bulunan besinin yuvadan uzakligi (teslim odulu icin)

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
        # Tum oduller (teslim mesafeye gore, bulma, iz/koku takibi, geri donus)
        # ve cezalar fitness_bonus icinde toplanir -> tek kaynak, cift sayim yok.
        return self.fitness_bonus

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
        # debug oku icin dunya koordinatlarinda yuva yonu (renderer kullanir)
        self._homing_world_angle = float(np.arctan2(ndy, ndx))

        # --- kimyasal alanlar: GRADYAN-YON algilama (koku + feromon) ---
        # Eski 3-nokta yontemi alan doygunken duzlestir -> yon yok. Yeni yontem:
        # alanin yerel gradyanini (yokus-yukari yonu) hesaplar ve heading'e gore
        # (sin, cos) + buyukluk olarak verir -> alan seviyesinden bagimsiz NET yon.
        cosh = np.cos(self.heading)
        sinh = np.sin(self.heading)

        def _grad_rel(arr):
            gx, gy = world.field_gradient(arr, self.x, self.y)
            mag = np.hypot(gx, gy)
            if mag < 1e-12:
                return 0.0, 0.0, 0.0
            ux, uy = gx / mag, gy / mag
            rel_cos = ux * cosh + uy * sinh       # heading dogrultusu bileseni
            rel_sin = -ux * sinh + uy * cosh      # heading'e dik (sol/sag) bilesen
            mnorm = min(1.0, mag * C.CHEM_GRAD_NORM)
            return float(rel_sin), float(rel_cos), float(mnorm)

        # besin kokusu: yokus-yukari yon (sin,cos) + buyukluk + yerel deger.
        # TASIRKEN koku girdisi SIFIRLANIR -> karinca besin kokusunu kovalamak
        # yerine yuvaya (homing) doner; donus ogrenmesi cok kolaylasir.
        if self.carrying:
            odor = np.zeros(4, dtype=np.float32)
            o_local = 0.0
        else:
            o_sin, o_cos, o_mag = _grad_rel(world.food_odor)
            o_local = world._bilinear(world.food_odor, self.x, self.y)
            odor = np.array([o_sin, o_cos, o_mag, o_local], dtype=np.float32)

        # food feromonu (iz): yokus-yukari yon (sin,cos) + yerel deger (0..1)
        f_sin, f_cos, _f_mag = _grad_rel(world.ph_food)
        f_local = min(1.0, world._bilinear(world.ph_food, self.x, self.y) / C.PH_MAX)
        ph_food_in = np.array([f_sin, f_cos, f_local], dtype=np.float32)

        # home feromonu: sadece yerel deger (homing zaten yon veriyor)
        h_local = min(1.0, world._bilinear(world.ph_home, self.x, self.y) / C.PH_MAX)
        ph_home_in = np.array([h_local], dtype=np.float32)

        # debug paneli icin: karincanin gercekten algiladigi degerler
        self.last_odor = float(o_local)
        self.last_food_ph = float(f_local)

        extra = np.array([
            1.0 if self.carrying else 0.0,
            self.energy,
        ], dtype=np.float32)
        return np.concatenate([feats, homing, ph_food_in, ph_home_in, odor, extra])

    # --------------------------------------------------------------- algila
    def observe(self, world, neighbors):
        """Durum vektoru (state) - DQN ajani ve neuroevo beyni ayni girdiyi alir."""
        return self.sense(world, neighbors)

    def _potential(self, world):
        """Potansiyel-tabanli sekillendirme icin phi(s). Teleskopik (farm'a kapali):
        bos gezerken 'isindikca' (koku+iz) artar; tasirken yuvaya yaklastikca artar.
        Tasima potansiyeli DAHA DIK olceklenir (RL_RETURN_SCALE) -> adim-basina
        donus gradyani foraging gradyaniyla kiyaslanabilir olur (yoksa ~8x zayif)."""
        if self.carrying:
            nd = np.hypot(self.x - world.nest_pos[0], self.y - world.nest_pos[1])
            return -min(1.0, nd / (self._max_d * C.RL_RETURN_SCALE))
        # YALNIZCA besin kokusu tirmanmasi odullenir; feromon izi ODULLENMEZ
        # (feromona yapisip titreme davranisini onlemek icin). Feromon hala
        # algilanir (ag girdisi) ve birakilir, ama odule katkisi yoktur.
        return float(world._bilinear(world.food_odor, self.x, self.y))

    # --------------------------------------------------------------- guncelle
    def apply_action(self, action, dt, world, neighbors):
        """Verilen aksiyonu uygula; ADIM-BASINA odulu hesaplayip self.last_reward'a
        yazar (DQN gecisi icin) ve fitness_bonus'a ekler. Olay sozlugu doner."""
        events = {"delivered": False, "picked": False}
        if not self.alive:
            self.last_reward = 0.0
            return events

        self.last_action = action
        phi_old = self._potential(world)
        carrying_before = self.carrying

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
        self.heading = _wrap_angle(self.heading)
        disp = np.hypot(self.x - ox, self.y - oy)

        r = -C.RL_STEP_COST   # her adim kucuk maliyet (verimlilik tesviki)

        # --- cezalar ---
        if action in (C.ACTION_FORWARD, C.ACTION_BACK) and not moved:
            self.energy -= C.WALL_PENALTY_RATE * dt
            self.wall_hits += 1
            r -= C.RL_WALL_PEN
            cs = C.CELL_SIZE
            if (self.x < cs or self.x > C.WORLD_W - cs
                    or self.y < cs or self.y > C.WORLD_H - cs):
                self.energy -= C.BORDER_PENALTY_RATE * dt
                r -= C.RL_BORDER_PEN
        if action == C.ACTION_NONE:
            self.energy -= C.IDLE_PENALTY_RATE * dt
            self.idle_steps += 1
            r -= C.RL_IDLE_PEN

        # --- besin alma ---
        if not self.carrying and world.take_food(self.x, self.y):
            self.carrying = True
            self.energy = C.ENERGY_MAX
            self.food_found += 1
            events["picked"] = True
            self.carry_distance = 0.0
            dnest = np.hypot(self.x - world.nest_pos[0], self.y - world.nest_pos[1])
            self.last_find_dist = dnest
            r += C.RL_FIND_W * dnest          # uzaktan bulmak daha degerli
            if C.LIFESPAN_FOOD_BONUS:
                self.lifespan = min(self.lifespan + self.base_lifespan,
                                    self.base_lifespan * C.LIFESPAN_MAX_MULT)

        # --- yuvaya teslim ---
        if self.carrying and world.at_nest(self.x, self.y):
            self.carrying = False
            self.energy = C.ENERGY_MAX
            self.food_delivered += 1
            events["delivered"] = True
            r += C.RL_DELIVER_R + self.last_find_dist * C.RL_DELIVER_DIST_W

        # --- potansiyel-tabanli sekillendirme (teleskopik, farm'a kapali) ---
        # carrying durumu DEGISTIYSE potansiyel tabani degisir -> sahte sicrama
        # olusur; o adimda sekillendirme atlanir (olay odulu zaten kredilendirdi).
        phi_new = self._potential(world)
        if self.carrying == carrying_before:
            r += C.RL_SHAPE_W * (C.RL_GAMMA * phi_new - phi_old)

        # --- feromon birak ---
        if self.carrying:
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

        self.last_reward = float(r)
        self.fitness_bonus += r
        return events

    def update(self, dt, world, neighbors):
        """Neuroevo yolu: algila -> beyin karar verir -> uygula. (DQN'de sim
        aksiyonu paylasilan ajandan secip apply_action'i dogrudan cagirir.)"""
        if not self.alive:
            return {"delivered": False, "picked": False}
        obs = self.observe(world, neighbors)
        action = self.brain.forward(obs)
        return self.apply_action(action, dt, world, neighbors)

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
