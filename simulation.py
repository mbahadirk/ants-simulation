"""
Simulasyon: populasyon yonetimi + genetik ureme.

- Baslangicta INITIAL_POP karinca yuvada rastgele genomlarla dogar.
- Her karinca algilar, dusunur, hareket eder (ant.update).
- Yuvaya her FOOD_PER_BIRTH (3) besin teslim edildiginde, o besinleri getiren
  son N_PARENTS (3) karincanin genomundan crossover+mutasyon ile 1 yavru uretilir.
- Yas / aclik nedeniyle olen karincalar kaldirilir.
- Populasyon MIN_POP altina duserse rastgele takviye yapilir (nesil tukenmesin).
"""

import numpy as np

import config as C
from world import make_default_world
from ant import Ant
from neural_network import breed


class Simulation:
    def __init__(self, world=None, seed=None):
        self.rng = np.random.default_rng(seed)
        self.world = world or make_default_world()
        self.ants = []

        # ureme tamponu: teslim eden karincalarin (genom, nesil) bilgisi
        self._parent_buffer = []
        # fitness havuzu: en iyi olen karincalarin (fitness, genom, nesil)
        self._fitness_pool = []

        # istatistikler
        self.total_delivered = 0
        self.births = 0
        self.deaths = 0
        self.generation = 0
        self.sim_time = 0.0
        self.selected = None  # debug'da secili karinca

        self._spawn_initial()

    # ------------------------------------------------------------- baslangic
    def _spawn_initial(self):
        for _ in range(C.INITIAL_POP):
            self._spawn_ant(genome=None, generation=0)

    def _nest_spawn_pos(self):
        nx, ny = self.world.nest_pos
        r = self.world.nest_radius * 0.6
        ang = self.rng.uniform(0, 2 * np.pi)
        rad = self.rng.uniform(0, r)
        return nx + np.cos(ang) * rad, ny + np.sin(ang) * rad

    def _spawn_ant(self, genome=None, generation=0):
        if len(self.ants) >= C.MAX_POP:
            return None
        x, y = self._nest_spawn_pos()
        ant = Ant(x, y, genome=genome, rng=self.rng, generation=generation)
        self.ants.append(ant)
        return ant

    # ------------------------------------------------------------- guncelleme
    def update(self, dt):
        self.sim_time += dt
        for ant in self.ants:
            events = ant.update(dt, self.world, self.ants)
            if events["delivered"]:
                self._on_delivery(ant)

        self.world.update_pheromones(dt)

        # olenleri kaldir (olmeden onceki en iyileri fitness havuzuna al)
        before = len(self.ants)
        survivors = []
        for a in self.ants:
            if a.alive:
                survivors.append(a)
            else:
                self._record_fitness(a)
        self.ants = survivors
        self.deaths += before - len(self.ants)
        if self.selected is not None and not self.selected.alive:
            self.selected = None

        # populasyon dususte takviye
        while len(self.ants) < C.MIN_POP:
            if self._reinforce() is None:
                break

    def _record_fitness(self, ant):
        """Olen karincanin genomunu fitness havuzunda tut (en iyi N saklanir)."""
        f = ant.fitness()
        if f <= 0:
            return
        self._fitness_pool.append((f, ant.genome(), ant.generation))
        self._fitness_pool.sort(key=lambda t: t[0], reverse=True)
        del self._fitness_pool[C.TOP_SURVIVORS:]

    def _reinforce(self):
        """Populasyonu rastgele yerine en iyi genomlardan tamamlar."""
        pool = []
        # once yasayan en iyiler, sonra fitness havuzu
        living = sorted(self.ants, key=lambda a: a.fitness(), reverse=True)
        pool += [(a.fitness(), a.genome(), a.generation) for a in living if a.fitness() > 0]
        pool += self._fitness_pool
        pool.sort(key=lambda t: t[0], reverse=True)
        pool = pool[:C.TOP_SURVIVORS]

        if C.FITNESS_REINFORCE and len(pool) >= 2:
            k = min(C.N_PARENTS, len(pool))
            idx = self.rng.choice(len(pool), size=k, replace=False)
            genomes = [pool[i][1] for i in idx]
            gen = max(pool[i][2] for i in idx) + 1
            child = breed(genomes, self.rng, C.MUTATION_RATE, C.MUTATION_SCALE)
            return self._spawn_ant(genome=child, generation=gen)
        # havuz yoksa rastgele
        return self._spawn_ant(genome=None, generation=self.generation)

    def _on_delivery(self, ant):
        self.total_delivered += 1
        self.world.delivered_food += 1
        self._parent_buffer.append((ant.genome(), ant.generation))

        if len(self._parent_buffer) >= C.FOOD_PER_BIRTH:
            self._reproduce()
            self._parent_buffer = []

    def _reproduce(self):
        parents = self._parent_buffer[-C.N_PARENTS:]
        genomes = [g for (g, _) in parents]
        # ebeveyn sayisi azsa (teorik olarak olmaz) mevcut olanlarla devam
        if len(genomes) < 2:
            return
        child_genome = breed(genomes, self.rng, C.MUTATION_RATE, C.MUTATION_SCALE)
        child_gen = max(g for (_, g) in parents) + 1
        self.generation = max(self.generation, child_gen)
        if self._spawn_ant(genome=child_genome, generation=child_gen) is not None:
            self.births += 1

    # ------------------------------------------------------------------ debug
    def select_at(self, wx, wy, radius=18.0):
        """Verilen dunya konumuna en yakin karincayi secer (debug takip)."""
        best = None
        best_d = radius * radius
        for a in self.ants:
            d = (a.x - wx) ** 2 + (a.y - wy) ** 2
            if d < best_d:
                best_d = d
                best = a
        self.selected = best
        return best

    # ------------------------------------------------------------------ stats
    def stats(self):
        carrying = sum(1 for a in self.ants if a.carrying)
        return {
            "pop": len(self.ants),
            "carrying": carrying,
            "delivered": self.total_delivered,
            "births": self.births,
            "deaths": self.deaths,
            "generation": self.generation,
            "food_left": self.world.food_count(),
            "time": self.sim_time,
        }
