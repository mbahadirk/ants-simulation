"""
Karinca varligi.

Her karincanin: konumu, yon acisi (heading), enerjisi, yasi, yasam suresi,
bir beyni (LSTMPolicy) ve kisa menzilli isin sensorleri vardir.

Sensor -> beyin (forward) -> aksiyon (ileri/geri/sol/sag/bekle) -> hareket.
Hareket sirasinda tas/engel gecilemez; besin alinir, yuvaya teslim edilir.
"""

import numpy as np

import config as C
from neural_network import LSTMPolicy

# Sensorde nesne tipi -> one-hot index eslemesi
# 0:food 1:stone 2:obstacle 3:ant 4:nest
_OBJ_INDEX = {
    C.FOOD: 0,
    C.STONE: 1,
    C.OBSTACLE: 2,
    C.NEST: 4,
}
ANT_HIT = 3  # baska karinca algilandiginda kullanilan one-hot index

# Isin acilarinin (heading'e gore) onceden hesaplanmis ofsetleri
_RAY_OFFSETS = np.linspace(-C.VISION_FOV / 2.0, C.VISION_FOV / 2.0, C.N_RAYS)


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
        self.carrying = False
        self.alive = True
        self.generation = generation

        # istatistik / ureme icin
        self.food_delivered = 0
        self.food_found = 0

        # debug icin son sensor okumasi: (angle, dist, obj_index_or_None)
        self.last_rays = []
        self.last_action = C.ACTION_NONE

        # benzersiz bir renk (debug'da ayirt etmek icin) -> genoma bagli
        g = self.brain.get_genome()
        h = int(abs(np.sum(g[:8]) * 1000)) % 360
        self.hue = h

    # ---------------------------------------------------------------- genom
    def genome(self):
        return self.brain.get_genome()

    def fitness(self):
        return (self.food_delivered * C.FITNESS_DELIVER_W
                + self.food_found * C.FITNESS_FIND_W)

    # -------------------------------------------------------------- sensorler
    def sense(self, world, neighbors):
        """
        Kisa menzilli isinlarla cevreyi algilar.
        world: tas/engel/besin/yuva icin raycast.
        neighbors: yakindaki diger karincalar (isin uzerine duserse 'ant').
        Donus: INPUT_SIZE uzunlugunda numpy vektor.
        """
        feats = np.zeros(C.N_RAYS * C.RAY_FEATURES, dtype=np.float32)
        self.last_rays = []

        ray_dist = np.empty(C.N_RAYS, dtype=np.float32)
        ray_idx = [None] * C.N_RAYS  # one-hot index ya da None

        for k, off in enumerate(_RAY_OFFSETS):
            ang = self.heading + off
            obj_type, dist = world.cast_ray(self.x, self.y, ang, C.VISION_RANGE)
            ray_dist[k] = dist
            if obj_type in _OBJ_INDEX and dist < C.VISION_RANGE:
                ray_idx[k] = _OBJ_INDEX[obj_type]

        # diger karincalari isinlara yerlestir
        half_fov = C.VISION_FOV / 2.0
        bin_w = C.VISION_FOV / max(1, (C.N_RAYS - 1))
        for other in neighbors:
            if other is self or not other.alive:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            dist = np.hypot(dx, dy)
            if dist > C.VISION_RANGE or dist < 1e-3:
                continue
            rel = _wrap_angle(np.arctan2(dy, dx) - self.heading)
            if abs(rel) > half_fov:
                continue
            k = int(round((rel + half_fov) / bin_w))
            k = max(0, min(C.N_RAYS - 1, k))
            if dist < ray_dist[k]:
                ray_dist[k] = dist
                ray_idx[k] = ANT_HIT

        # ozellik vektorunu doldur + debug isinlarini kaydet
        for k in range(C.N_RAYS):
            base = k * C.RAY_FEATURES
            feats[base] = ray_dist[k] / C.VISION_RANGE  # mesafe (0..1)
            if ray_idx[k] is not None:
                feats[base + 1 + ray_idx[k]] = 1.0
            ang = self.heading + _RAY_OFFSETS[k]
            self.last_rays.append((ang, float(ray_dist[k]), ray_idx[k]))

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

        extra = np.array([
            1.0 if self.carrying else 0.0,
            self.energy,
        ], dtype=np.float32)
        return np.concatenate([feats, homing, ph, extra])

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
        if action == C.ACTION_LEFT:
            self.heading -= C.TURN_SPEED * dt
        elif action == C.ACTION_RIGHT:
            self.heading += C.TURN_SPEED * dt
        elif action == C.ACTION_FORWARD:
            self._try_move(C.MOVE_SPEED * dt, world)
        elif action == C.ACTION_BACK:
            self._try_move(-C.BACK_SPEED * dt, world)
        # ACTION_NONE -> hareket yok

        self.heading = _wrap_angle(self.heading)

        # --- besin alma ---
        if not self.carrying and world.take_food(self.x, self.y):
            self.carrying = True
            self.energy = C.ENERGY_MAX  # besin buldu -> aclik sifirlanir
            self.food_found += 1
            events["picked"] = True

        # --- yuvaya teslim ---
        if self.carrying and world.at_nest(self.x, self.y):
            self.carrying = False
            self.energy = C.ENERGY_MAX
            self.food_delivered += 1
            events["delivered"] = True

        # --- feromon birak (antenle koklananin kaynagi) ---
        if self.carrying:
            world.deposit(C.PH_FOOD, self.x, self.y, C.PH_DEPOSIT_FOOD * dt * 60)
        else:
            world.deposit(C.PH_HOME, self.x, self.y, C.PH_DEPOSIT_HOME * dt * 60)

        # --- enerji / yas ---
        self.energy -= dt / C.STARVE_TIME
        self.age += dt
        if self.energy <= 0.0 or self.age >= self.lifespan:
            self.alive = False

        return events

    def _try_move(self, dist, world):
        nx = self.x + np.cos(self.heading) * dist
        ny = self.y + np.sin(self.heading) * dist
        # eksen bazli kayma: bir eksen bloke olsa bile digerinde kayabilsin
        if not world.is_blocked(nx, self.y) and world.in_bounds(nx, self.y):
            self.x = nx
        if not world.is_blocked(self.x, ny) and world.in_bounds(self.x, ny):
            self.y = ny
