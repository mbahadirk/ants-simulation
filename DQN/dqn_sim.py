"""
DQN simulasyonu: tek PAYLASILAN Q-agi tum karincalardan toplanan adim-basina
gecislerle (s, a, r, s', done) egitilir.

Neuroevrimden farki: GENOM/UREME/HOF YOKTUR. Sabit sayida (RL_POP) karinca
paylasilan politikayla hareket eder, deneyim uretir; olen karincanin yerine
ayni politikayla yenisi gelir. Ag, deneyim tekrar bellekistinden minibatch'lerle
surekli ogrenir.

Simulation ile AYNI genel arayuzu sunar (world, ants, selected, update(dt),
stats(), select_at, history) -> renderer/main/stats_view degismeden calisir.
"""

from collections import deque

import numpy as np

import config as C
from world import make_default_world
from ant import Ant
from dqn import DQNAgent


class DQNSimulation:
    is_dqn = True

    def __init__(self, world=None, seed=None, map_name="default", agent=None):
        self.rng = np.random.default_rng(seed)
        self.world = world or make_default_world()
        self.ants = []
        self.map_name = map_name
        self.selected = None

        self.agent = agent or DQNAgent(C.INPUT_SIZE, C.OUTPUT_SIZE)

        # istatistikler (Simulation ile ayni anahtarlar)
        self.total_delivered = 0
        self.total_found = 0
        self.births = 0
        self.deaths = 0
        self.generation = 0          # DQN'de nesil yok; arayuz uyumu icin 0
        self.sim_time = 0.0
        self._food_spawn_acc = 0.0
        self._near_food_acc = 0.0
        self._stats_acc = 0.0
        self._save_acc = 0.0
        self.history = []
        self.avg_return = 0.0        # olen karincalarin omur-getirisi (moving avg)
        self.recent_rewards = deque(maxlen=3000)

        for _ in range(C.RL_POP):
            self._spawn_ant()

    # ------------------------------------------------------------- baslangic
    def _nest_spawn_pos(self):
        nx, ny = self.world.nest_pos
        r = self.world.nest_radius * 0.6
        ang = self.rng.uniform(0, 2 * np.pi)
        rad = self.rng.uniform(0, r)
        return nx + np.cos(ang) * rad, ny + np.sin(ang) * rad

    def _spawn_ant(self, is_birth=False):
        """is_birth=True yalnizca TESLIMAT-kaynakli (koloni buyumesi) dogumlar icin.
        Baslangic ve olum-yerine-koyma dogumlari births sayacina EKLENMEZ."""
        x, y = self._nest_spawn_pos()
        a = Ant(x, y, genome=None, rng=self.rng, generation=0)
        a.last_state = None      # ilk adimda gozlemlenecek
        a.ep_return = 0.0        # omur boyu toplam odul (avg_return icin)
        # PER-ANT epsilon (Ape-X): log-uniform [RL_EPS_MIN, RL_EPS_MAX] -> bazi
        # karincalar kasifci, bazilari somurucu; kesif asla olmez.
        u = self.rng.random()
        a.epsilon = float(C.RL_EPS_MIN * (C.RL_EPS_MAX / C.RL_EPS_MIN) ** u)
        self.ants.append(a)
        if is_birth:
            self.births += 1
        return a

    # ------------------------------------------------------------- guncelleme
    def update(self, dt):
        self.sim_time += dt
        ants = self.ants

        # periyodik besin (uzak forage hedefleri)
        self._food_spawn_acc += dt
        if self._food_spawn_acc >= C.FOOD_SPAWN_INTERVAL:
            self._food_spawn_acc -= C.FOOD_SPAWN_INTERVAL
            self.world.spawn_random_food(self.rng, C.FOOD_SPAWN_AMOUNT)

        # RL curriculum: yuvaya yakin besin (kisa tur -> ilk teslimatlar orneklensin)
        self._near_food_acc += dt
        if self._near_food_acc >= C.RL_NEAR_FOOD_EVERY:
            self._near_food_acc -= C.RL_NEAR_FOOD_EVERY
            if self.world.food_near_nest_count(C.RL_NEAR_FOOD_CELLS) < C.RL_NEAR_FOOD_MAX:
                self.world.spawn_food_near_nest(self.rng, C.RL_NEAR_FOOD_AMOUNT,
                                                C.RL_NEAR_FOOD_CELLS)

        # 1) durumlar (yeni/ilk adim karincalari icin gozlemle)
        for a in ants:
            if a.last_state is None:
                a.last_state = a.observe(self.world, ants)
        S = np.stack([a.last_state for a in ants]).astype(np.float32)

        # 2) aksiyonlar (PER-ANT epsilon-greedy, paylasilan ag)
        eps = np.array([a.epsilon for a in ants], dtype=np.float32)
        A = self.agent.act(S, eps)

        # 3) uygula + adim odulleri
        rewards = np.empty(len(ants), dtype=np.float32)
        deliveries_this_step = 0
        for i, a in enumerate(ants):
            ev = a.apply_action(int(A[i]), dt, self.world, ants)
            if ev["picked"]:
                self.total_found += 1
            if ev["delivered"]:
                self.total_delivered += 1
                self.world.delivered_food += 1
                deliveries_this_step += 1
            rewards[i] = a.last_reward
            a.ep_return += a.last_reward

        # feromon alanlarini guncelle (sonraki durumu etkiler)
        self.world.update_pheromones(dt)

        # 4) sonraki durumlar + gecisleri belleğe yaz
        S2 = np.zeros_like(S)
        D = np.zeros(len(ants), dtype=np.float32)
        for i, a in enumerate(ants):
            if a.alive:
                ns = a.observe(self.world, ants)
                a.last_state = ns
            else:
                ns = np.zeros(C.INPUT_SIZE, dtype=np.float32)
                D[i] = 1.0
                a.last_state = None
            S2[i] = ns
        self.agent.remember_batch(S, A, rewards, S2, D)
        self.recent_rewards.extend(rewards.tolist())

        # 5) ogren (minibatch gradyan adimi)
        self.agent.train_step()

        # 6) olenleri kaldir, yerine ayni politikayla yenisini koy
        survivors = []
        for a in ants:
            if a.alive:
                survivors.append(a)
            else:
                self.deaths += 1
                self.avg_return = 0.99 * self.avg_return + 0.01 * a.ep_return
                if self.selected is a:
                    self.selected = None
        self.ants = survivors
        # olum-yerine-koyma: tabani (RL_POP) koru -> births'e EKLENMEZ
        while len(self.ants) < C.RL_POP:
            self._spawn_ant(is_birth=False)
        # TESLIMAT-kaynakli dogumlar: basarili forage koloniyi buyutur (MAX_POP'a kadar)
        for _ in range(deliveries_this_step):
            if len(self.ants) >= C.MAX_POP:
                break
            self._spawn_ant(is_birth=True)

        # istatistik ornegi
        self._stats_acc += dt
        if self._stats_acc >= C.STATS_SAMPLE_INTERVAL:
            self._stats_acc -= C.STATS_SAMPLE_INTERVAL
            self._record_history()

        # periyodik ajan kaydi (kosular/haritalar arasi tasinir)
        self._save_acc += dt
        if self._save_acc >= C.BANK_MERGE_EVERY:
            self._save_acc = 0.0
            self.agent.save()

    # ------------------------------------------------------------------ debug
    def select_at(self, wx, wy, radius=18.0):
        best, best_d = None, radius * radius
        for a in self.ants:
            d = (a.x - wx) ** 2 + (a.y - wy) ** 2
            if d < best_d:
                best_d, best = d, a
        self.selected = best
        return best

    # ------------------------------------------------------------------ stats
    def avg_reward(self):
        return float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0

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
            "hof_size": len(self.agent.buffer),     # buffer doluluk (HUD'da gosterilir)
            "hof_best": self.avg_return,            # ogrenme gostergesi (omur-getirisi)
        }

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

    # ------------------------------------------------------------------ kayit
    def save_bank(self):
        """main.py uyumu: ESC/cikis/harita-degisiminde ajan agirliklarini kaydeder."""
        self.agent.save()
        return True

    def save(self, path):
        """Demo (H) icin hafif kayit: istatistik gecmisi + ajan agirliklari.
        (DQN demolari 'resume' edilmez; sadece Stats grafikleri icindir.)"""
        import pickle, os
        self.agent.save()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"dqn": True, "map_name": self.map_name,
                         "sim_time": self.sim_time, "history": list(self.history)}, f)
