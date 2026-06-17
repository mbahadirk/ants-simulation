"""
Deep Q-Network (saf NumPy, elle backprop + Adam).

Neuroevrimin aksine: tek bir PAYLASILAN Q-agi tum karincalarin ADIM-BASINA
deneyiminden ogrenir. Kredi atamasi TD-hedefiyle (bootstrap) algoritma
tarafindan yapilir; kesif epsilon-greedy ile acikca yonetilir.

- QNetwork:  girdi(45) -> Dense(gizli, tanh) -> Dense(5 aksiyon-degeri).
             forward batched; train_on_batch elle backprop + Adam.
- ReplayBuffer: (s, a, r, s', done) gecislerini saklar, minibatch orneklar.
- DQNAgent: online + hedef ag, epsilon zamanlamasi, Double-DQN hedefi,
            save/load (.npz) -> politika kosular/haritalar arasi tasinir.
"""

import os

import numpy as np

import config as C


class _Adam:
    """Tek bir parametre dizisi icin Adam optimizer durumu."""

    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        param -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class QNetwork:
    """45 -> tanh(H) -> 5 aksiyon-degeri. Elle backprop, Adam."""

    def __init__(self, in_size, hidden, out_size, lr, rng):
        self.rng = rng
        # He/Xavier benzeri baslatma
        self.W1 = (rng.standard_normal((hidden, in_size)) / np.sqrt(in_size)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((out_size, hidden)) / np.sqrt(hidden)).astype(np.float32)
        self.b2 = np.zeros(out_size, dtype=np.float32)
        self._opt = {
            "W1": _Adam(self.W1.shape, lr), "b1": _Adam(self.b1.shape, lr),
            "W2": _Adam(self.W2.shape, lr), "b2": _Adam(self.b2.shape, lr),
        }

    def forward(self, X):
        """X: (B, in) -> Q: (B, out). Ara katmani onbellege alir (backward icin)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        self._X = X
        self._h = np.tanh(X @ self.W1.T + self.b1)     # (B, H)
        return self._h @ self.W2.T + self.b2           # (B, O)

    def train_on_batch(self, X, actions, targets):
        """Q(X)[actions] -> targets icin bir gradyan adimi. Huber (clipped) kayip."""
        B = X.shape[0]
        Q = self.forward(X)                            # (B, O), onbellek doldu
        idx = np.arange(B)
        q_sa = Q[idx, actions]                          # (B,)
        td = q_sa - targets                            # hata
        # Huber: gradyani [-1,1] kirp (DQN kararliligi)
        g = np.clip(td, -1.0, 1.0) / B
        dQ = np.zeros_like(Q)
        dQ[idx, actions] = g                           # sadece secilen aksiyon
        # backward
        dW2 = dQ.T @ self._h                           # (O, H)
        db2 = dQ.sum(axis=0)                           # (O,)
        dh = (dQ @ self.W2) * (1.0 - self._h ** 2)     # (B, H), tanh turevi
        dW1 = dh.T @ self._X                           # (H, in)
        db1 = dh.sum(axis=0)                           # (H,)
        self._opt["W1"].step(self.W1, dW1)
        self._opt["b1"].step(self.b1, db1)
        self._opt["W2"].step(self.W2, dW2)
        self._opt["b2"].step(self.b2, db2)
        return float(np.mean(np.minimum(np.abs(td), 1.0) ** 2))  # kaba kayip

    def get_params(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_params(self, p):
        self.W1 = p["W1"].copy(); self.b1 = p["b1"].copy()
        self.W2 = p["W2"].copy(); self.b2 = p["b2"].copy()


class ReplayBuffer:
    """Sabit boyutlu dairesel (s, a, r, s', done) deposu."""

    def __init__(self, capacity, state_dim, rng):
        self.cap = capacity
        self.rng = rng
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, s, a, r, s2, done):
        i = self.idx
        self.s[i] = s; self.a[i] = a; self.r[i] = r
        self.s2[i] = s2; self.d[i] = 1.0 if done else 0.0
        self.idx = (i + 1) % self.cap
        if self.idx == 0:
            self.full = True

    def add_batch(self, S, A, R, S2, D):
        for k in range(len(A)):
            self.add(S[k], A[k], R[k], S2[k], D[k])

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, batch):
        n = len(self)
        ix = self.rng.integers(0, n, size=batch)
        return self.s[ix], self.a[ix], self.r[ix], self.s2[ix], self.d[ix]


class DQNAgent:
    """Paylasilan DQN ajani: online + hedef ag, Double-DQN, epsilon-greedy."""

    def __init__(self, state_dim, n_actions, seed=None, path=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.path = path or C.RL_AGENT_FILE

        self.online = QNetwork(state_dim, C.RL_HIDDEN, n_actions, C.RL_LR, self.rng)
        self.target = QNetwork(state_dim, C.RL_HIDDEN, n_actions, C.RL_LR, self.rng)
        self.target.set_params(self.online.get_params())

        self.buffer = ReplayBuffer(C.RL_BUFFER, state_dim, self.rng)
        # TASIMA (carrying) gecisleri icin ayri tampon -> dengeli ornekleme.
        # Donus gecisleri tum deneyimin <%1'i; bunlar olmadan donus politikasi
        # neredeyse hic egitilmez (sinif dengesizligi).
        self.cbuffer = ReplayBuffer(C.RL_BUFFER // 2, state_dim, self.rng)
        self.env_steps = 0       # toplam yasanan gecis (epsilon zamanlamasi)
        self.learn_steps = 0     # toplam gradyan adimi (hedef senk.)
        self.last_loss = 0.0

        if os.path.exists(self.path):
            try:
                self.load()
            except Exception as e:
                print(f"[DQN] ajan yuklenemedi ({e}); sifirdan baslaniyor")

    # ------------------------------------------------------------- aksiyon
    def act(self, states, epsilons=None, greedy=False):
        """states: (N, state_dim) -> aksiyonlar (N,). PER-ANT epsilon-greedy:
        epsilons (N,) her karincanin kendi kesif orani. greedy=True -> saf argmax."""
        Q = self.online.forward(states)
        best = np.argmax(Q, axis=1)
        if greedy or epsilons is None:
            return best.astype(np.int64)
        N = len(best)
        explore = self.rng.random(N) < np.asarray(epsilons)
        rand = self.rng.integers(0, self.n_actions, size=N)
        return np.where(explore, rand, best).astype(np.int64)

    def greedy_values(self, state):
        """Tek bir durum icin Q-degerleri (debug paneli icin)."""
        return self.online.forward(state[None, :])[0]

    # ------------------------------------------------------------- ogrenme
    def remember_batch(self, S, A, R, S2, D):
        self.buffer.add_batch(S, A, R, S2, D)
        # tasima gecislerini (state'te carrying biti, index -2) ayrica dengeli tampona
        carry = S[:, -2] > 0.5
        if carry.any():
            self.cbuffer.add_batch(S[carry], A[carry], R[carry], S2[carry], D[carry])
        self.env_steps += len(A)

    def _sample_balanced(self):
        """Minibatch'in RL_CARRY_FRAC orani tasima tamponundan, kalani anadan."""
        b = C.RL_BATCH
        if len(self.cbuffer) >= max(16, int(b * C.RL_CARRY_FRAC)):
            bc = int(b * C.RL_CARRY_FRAC)
            bm = b - bc
            s1, a1, r1, n1, d1 = self.buffer.sample(bm)
            s2, a2, r2, n2, d2 = self.cbuffer.sample(bc)
            return (np.concatenate([s1, s2]), np.concatenate([a1, a2]),
                    np.concatenate([r1, r2]), np.concatenate([n1, n2]),
                    np.concatenate([d1, d2]))
        return self.buffer.sample(b)

    def train_step(self):
        if len(self.buffer) < C.RL_MIN_BUFFER:
            return
        for _ in range(C.RL_TRAIN_PER_STEP):
            S, A, R, S2, D = self._sample_balanced()
            # Double DQN: aksiyon online ag ile secilir, deger hedef agdan alinir
            a_star = np.argmax(self.online.forward(S2), axis=1)
            q2 = self.target.forward(S2)[np.arange(len(A)), a_star]
            y = R + C.RL_GAMMA * (1.0 - D) * q2
            self.last_loss = self.online.train_on_batch(S, A, y)
            self.learn_steps += 1
            if self.learn_steps % C.RL_TARGET_SYNC == 0:
                self.target.set_params(self.online.get_params())

    # ------------------------------------------------------------- kalicilik
    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        p = self.online.get_params()
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, W1=p["W1"], b1=p["b1"], W2=p["W2"], b2=p["b2"],
                 env_steps=np.int64(self.env_steps),
                 learn_steps=np.int64(self.learn_steps),
                 state_dim=np.int64(self.state_dim),
                 hidden=np.int64(C.RL_HIDDEN))
        os.replace(tmp, self.path)

    def load(self):
        d = np.load(self.path)
        if int(d["state_dim"]) != self.state_dim or int(d["hidden"]) != C.RL_HIDDEN:
            raise ValueError(
                f"Ajan boyutu uyumsuz (state {int(d['state_dim'])}!={self.state_dim} "
                f"veya hidden {int(d['hidden'])}!={C.RL_HIDDEN})")
        p = {"W1": d["W1"], "b1": d["b1"], "W2": d["W2"], "b2": d["b2"]}
        self.online.set_params(p)
        self.target.set_params(p)
        self.env_steps = int(d["env_steps"])
        self.learn_steps = int(d["learn_steps"])
        print(f"[DQN] ajan yuklendi (env_steps={self.env_steps}, "
              f"learn_steps={self.learn_steps})")
