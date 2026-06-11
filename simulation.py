"""
Simulasyon: populasyon yonetimi + genetik ureme.

- Baslangicta INITIAL_POP karinca yuvada rastgele genomlarla dogar.
- Her karinca algilar, dusunur, hareket eder (ant.update).
- Yuvaya besin getiren HER karincadan, kendi genomundan (mutasyonla)
  OFFSPRING_PER_DELIVERY (3) adet yavru dogar.
- Yas / aclik nedeniyle olen karincalar kaldirilir.
- Populasyon MIN_POP altina duserse rastgele takviye yapilir (nesil tukenmesin).
"""

import os
import pickle

import numpy as np

import config as C
from world import make_default_world, World
from ant import Ant
from neural_network import breed, mutate


class Simulation:
    def __init__(self, world=None, seed=None):
        self.rng = np.random.default_rng(seed)
        self.world = world or make_default_world()
        self.ants = []

        # ONUR LISTESI (hall of fame): tum zamanlarin en iyi karincalari.
        # ant_id -> (fitness, genom, nesil). Olen karincalarin yerine gelenler
        # buradan uretilir. Yasayan elitler de periyodik olarak buraya sunulur.
        self.hall = {}
        self._hof_acc = 0.0

        # istatistikler
        self.total_delivered = 0
        self.births = 0
        self.deaths = 0
        self.generation = 0
        self.sim_time = 0.0
        self._food_spawn_acc = 0.0   # periyodik besin zamanlayicisi
        self.selected = None  # debug'da secili karinca

        # zaman serisi istatistik (T tusu ile gorsellestirilir)
        self.history = []            # her ornek: dict(t,pop,births,deaths,delivered,hof_best,gen)
        self._stats_acc = 0.0

        self._spawn_initial()

    def _record_history(self):
        s = self.stats()
        self.history.append({
            "t": round(self.sim_time, 1),
            "pop": s["pop"],
            "births": s["births"],
            "deaths": s["deaths"],
            "delivered": s["delivered"],
            "hof_best": round(s["hof_best"], 2),
            "gen": s["generation"],
        })

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

        # periyodik besin: her FOOD_SPAWN_INTERVAL saniyede bir rastgele bos hucre
        self._food_spawn_acc += dt
        if self._food_spawn_acc >= C.FOOD_SPAWN_INTERVAL:
            self._food_spawn_acc -= C.FOOD_SPAWN_INTERVAL
            self.world.spawn_random_food(self.rng, C.FOOD_SPAWN_AMOUNT)

        # zaman serisi istatistik ornegi
        self._stats_acc += dt
        if self._stats_acc >= C.STATS_SAMPLE_INTERVAL:
            self._stats_acc -= C.STATS_SAMPLE_INTERVAL
            self._record_history()

        for ant in self.ants:
            events = ant.update(dt, self.world, self.ants)
            if events["delivered"]:
                self._on_delivery(ant)

        self.world.update_pheromones(dt)

        # yasayan en iyi karincayi periyodik olarak onur listesine sun
        self._hof_acc += dt
        if self._hof_acc >= C.HOF_OFFER_EVERY and self.ants:
            self._hof_acc = 0.0
            best = max(self.ants, key=lambda a: a.fitness())
            self._offer_hall(best)

        # olenleri kaldir (olmeden once onur listesine sun)
        before = len(self.ants)
        survivors = []
        for a in self.ants:
            if a.alive:
                survivors.append(a)
            else:
                self._offer_hall(a)
        self.ants = survivors
        self.deaths += before - len(self.ants)
        if self.selected is not None and not self.selected.alive:
            self.selected = None

        # populasyon dususte: olenlerin yerine TUM ZAMANLARIN EN IYILERINDEN uret
        while len(self.ants) < C.MIN_POP:
            if self._reinforce() is None:
                break

    def _offer_hall(self, ant):
        """Karincayi onur listesine sunar (tum zamanlarin en iyileri saklanir)."""
        f = ant.fitness()
        if f <= 0:
            return
        prev = self.hall.get(ant.id)
        # ayni karinca daha once eklendiyse, fitness'i arttiysa guncelle
        if prev is None or f > prev[0]:
            self.hall[ant.id] = (f, ant.genome(), ant.generation)
        # listeyi en iyi HALL_OF_FAME_SIZE ile sinirla (en dusukleri at)
        if len(self.hall) > C.HALL_OF_FAME_SIZE:
            best_ids = sorted(self.hall, key=lambda k: self.hall[k][0],
                              reverse=True)[:C.HALL_OF_FAME_SIZE]
            self.hall = {k: self.hall[k] for k in best_ids}

    def hall_best(self):
        """Onur listesini fitness'a gore azalan sirada (fitness, genom, nesil) doner."""
        return sorted(self.hall.values(), key=lambda t: t[0], reverse=True)

    def _reinforce(self):
        """Olen karincanin yerine TUM ZAMANLARIN EN IYI karincalarindan biri gelir."""
        elite = self.hall_best()

        if C.FITNESS_REINFORCE and elite:
            if len(elite) >= 2:
                # en basarili N ebeveynin genetigini birlestir
                k = min(C.N_PARENTS, len(elite))
                parents = elite[:k]
                genomes = [p[1] for p in parents]
                gen = max(p[2] for p in parents) + 1
                child = breed(genomes, self.rng, C.MUTATION_RATE, C.MUTATION_SCALE)
            else:
                # tek elit varsa onu mutasyonla klonla
                child = mutate(elite[0][1], self.rng, C.MUTATION_RATE, C.MUTATION_SCALE)
                gen = elite[0][2] + 1
            self.generation = max(self.generation, gen)
            return self._spawn_ant(genome=child, generation=gen)
        # onur listesi henuz bossa (baslangic) rastgele
        return self._spawn_ant(genome=None, generation=self.generation)

    def _on_delivery(self, ant):
        self.total_delivered += 1
        self.world.delivered_food += 1
        # Teslim eden HER karincadan, kendi genomundan (mutasyonla) birden cok yavru
        parent_genome = ant.genome()
        child_gen = ant.generation + 1
        self.generation = max(self.generation, child_gen)
        for _ in range(C.OFFSPRING_PER_DELIVERY):
            child = mutate(parent_genome, self.rng, C.MUTATION_RATE, C.MUTATION_SCALE)
            if self._spawn_ant(genome=child, generation=child_gen) is not None:
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
        hof_best = max((t[0] for t in self.hall.values()), default=0.0)
        return {
            "pop": len(self.ants),
            "carrying": carrying,
            "delivered": self.total_delivered,
            "births": self.births,
            "deaths": self.deaths,
            "generation": self.generation,
            "food_left": self.world.food_count(),
            "time": self.sim_time,
            "hof_size": len(self.hall),
            "hof_best": hof_best,
        }

    # ------------------------------------------------------------------ kayit
    SAVE_VERSION = 1

    def _ant_state(self, a):
        return {
            "x": a.x, "y": a.y, "heading": a.heading, "energy": a.energy,
            "age": a.age, "lifespan": a.lifespan, "alive": a.alive,
            "carrying": a.carrying, "generation": a.generation,
            "food_found": a.food_found, "food_delivered": a.food_delivered,
            "wall_hits": a.wall_hits, "idle_steps": a.idle_steps,
            "fitness_bonus": a.fitness_bonus, "min_home_dist": a.min_home_dist,
            "max_odor_seen": a.max_odor_seen, "carry_distance": a.carry_distance,
            "genome": a.brain.get_genome(),
            "h": a.brain.h.copy(), "c": a.brain.c.copy(),
        }

    def save(self, path):
        """Simulasyon state'ini kaydet (kaldigi yerden devam etmek icin)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "version": self.SAVE_VERSION,
            "ants": [self._ant_state(a) for a in self.ants],
            "world_grid": self.world.grid.copy(),
            "world_food_amount": self.world.food_amount.copy(),
            "world_ph_home": self.world.ph_home.copy(),
            "world_ph_food": self.world.ph_food.copy(),
            "world_food_odor": self.world.food_odor.copy(),
            "world_delivered_food": self.world.delivered_food,
            "total_delivered": self.total_delivered,
            "births": self.births,
            "deaths": self.deaths,
            "generation": self.generation,
            "sim_time": self.sim_time,
            "food_spawn_acc": self._food_spawn_acc,
            "next_ant_id": Ant._next_id,
            "history": list(self.history),
            # onur listesi (tum zamanlarin en iyileri)
            "hall": {aid: (f, g.copy(), gen) for aid, (f, g, gen) in self.hall.items()},
            # ayarlar (devamda tutarlilik)
            "lifespan_min": C.LIFESPAN_MIN, "lifespan_max": C.LIFESPAN_MAX,
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
        os.replace(tmp, path)  # atomik yazim (yarim kayit olmasin)

    @classmethod
    def load(cls, path):
        """Kaydedilmis simulasyonu yukle ve devam ettir."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        # ag mimarisi degistiyse (genom boyutu) eski checkpoint uyumsuzdur
        from neural_network import LSTMPolicy
        expected = LSTMPolicy().genome_size
        ants_state = state.get("ants", [])
        if ants_state and len(ants_state[0]["genome"]) != expected:
            raise ValueError(
                f"Checkpoint sinir agi boyutu uyumsuz "
                f"({len(ants_state[0]['genome'])} != {expected}). "
                "Gorus/ag mimarisi degistigi icin eski kayit yuklenemez."
            )

        # dunyayi kaydedilen grid'ten yeniden olustur (yuva + koku otomatik)
        world = World(grid=state["world_grid"].copy(),
                      food_amount=state["world_food_amount"].copy())
        world.ph_home = state["world_ph_home"].copy()
        world.ph_food = state["world_ph_food"].copy()
        world.food_odor = state["world_food_odor"].copy()
        world.delivered_food = state.get("world_delivered_food", 0)

        sim = cls(world=world, seed=None)
        sim.ants.clear()

        for s in state["ants"]:
            a = Ant(s["x"], s["y"], genome=s["genome"], rng=sim.rng,
                    generation=s["generation"])
            a.heading = float(s["heading"])
            a.energy = float(s["energy"])
            a.age = float(s["age"])
            a.lifespan = float(s["lifespan"])
            a.alive = bool(s["alive"])
            a.carrying = bool(s["carrying"])
            a.food_found = s["food_found"]
            a.food_delivered = s["food_delivered"]
            a.wall_hits = s["wall_hits"]
            a.idle_steps = s["idle_steps"]
            a.fitness_bonus = s["fitness_bonus"]
            a.min_home_dist = s["min_home_dist"]
            a.max_odor_seen = s["max_odor_seen"]
            a.carry_distance = s.get("carry_distance", 0.0)
            a.brain.h = s["h"].copy()      # LSTM gizli durumu
            a.brain.c = s["c"].copy()
            sim.ants.append(a)

        Ant._next_id = state.get("next_ant_id", len(sim.ants))
        sim.total_delivered = state["total_delivered"]
        sim.births = state["births"]
        sim.deaths = state["deaths"]
        sim.generation = state["generation"]
        sim.sim_time = state["sim_time"]
        sim._food_spawn_acc = state.get("food_spawn_acc", 0.0)
        sim.history = list(state.get("history", []))
        # onur listesini geri yukle (eski kayitlarda olmayabilir)
        hall = state.get("hall", {})
        sim.hall = {aid: (f, g.copy(), gen) for aid, (f, g, gen) in hall.items()}

        # ayarlari geri yukle
        C.LIFESPAN_MIN = state.get("lifespan_min", C.LIFESPAN_MIN)
        C.LIFESPAN_MAX = state.get("lifespan_max", C.LIFESPAN_MAX)
        return sim
