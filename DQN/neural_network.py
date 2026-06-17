"""
Sinir agi (saf NumPy). Backprop YOKTUR: tum agirliklar dogrudan "genom" olarak
ele alinir; genetik algoritma bunlari crossover + mutasyon ile evrimlestirir.

Iki mimari (config.BRAIN_ARCH ile secilir):
- "mlp"  : girdi -> Dense(gizli, tanh) -> Dense(cikis) -> argmax  (ONERILEN)
           Saf feedforward; kucuk genom, puruzsuz fitness manzarasi, hizli evrim.
- "lstm" : girdi -> Dense(encoder) -> LSTM -> Dense(cikis) -> argmax
           Recurrent (zamansal hafiza h,c); daha buyuk genom, daha zor evrim.

Her iki policy ayni arayuzu sunar: forward(x)->aksiyon, get_genome/set_genome,
genome_size, reset_state(), ve h/c durum dizileri (MLP'de bos -> save/load uyumu).
"""

import numpy as np

import config as C
from config import (
    INPUT_SIZE, ENCODER_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    ACTION_FORWARD, ACTION_NONE, ACTION_BACK,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LSTMPolicy:
    """Tek bir karincanin beyni: Dense(encoder) -> LSTM -> Dense(cikis)."""

    def __init__(self, genome=None, rng=None):
        self.input_size = INPUT_SIZE
        self.encoder_size = ENCODER_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE
        self.rng = rng or np.random.default_rng()

        I, E, H, O = INPUT_SIZE, ENCODER_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
        z = E + H  # LSTM girisinde [encoder_cikti ; h_prev] birlestirilir

        # Parametre sekilleri (sirayla genoma serilesir)
        self._shapes = [
            ("W_e", (E, I)), ("b_e", (E,)),   # Dense kodlayici (LSTM oncesi)
            ("W_i", (H, z)), ("b_i", (H,)),   # input gate
            ("W_f", (H, z)), ("b_f", (H,)),   # forget gate
            ("W_o", (H, z)), ("b_o", (H,)),   # output gate
            ("W_g", (H, z)), ("b_g", (H,)),   # cell candidate
            ("W_y", (O, H)), ("b_y", (O,)),   # Dense cikis
        ]
        self.params = {}
        if genome is not None:
            self.set_genome(genome)
        else:
            self._init_random()

        self.reset_state()

    # ------------------------------------------------------------------ init
    def _init_random(self):
        for name, shape in self._shapes:
            if name.startswith("b"):
                self.params[name] = np.zeros(shape, dtype=np.float32)
            else:
                # Xavier benzeri kucuk baslatma
                fan = shape[1]
                std = 1.0 / np.sqrt(fan)
                self.params[name] = (self.rng.standard_normal(shape) * std).astype(np.float32)
        # forget gate biasini hafif pozitif baslatmak hafizayi kolaylastirir
        self.params["b_f"] += 1.0

        # Kesif onyargisi: rastgele beyinler de kesfetsin diye baslangicta
        # "ileri" cikisina hafif pozitif bias verilir. Cok guclu olursa
        # karincalar surekli ileri gider; bu yuzden olculu tutulur.
        self.params["b_y"][ACTION_FORWARD] = 0.4
        self.params["b_y"][ACTION_NONE] = -0.3
        self.params["b_y"][ACTION_BACK] = -0.3

    def reset_state(self):
        self.h = np.zeros(self.hidden_size, dtype=np.float32)
        self.c = np.zeros(self.hidden_size, dtype=np.float32)

    # ------------------------------------------------------------- genom I/O
    @property
    def genome_size(self):
        return sum(int(np.prod(shape)) for _, shape in self._shapes)

    def get_genome(self):
        return np.concatenate([self.params[name].ravel() for name, _ in self._shapes])

    def set_genome(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        idx = 0
        for name, shape in self._shapes:
            n = int(np.prod(shape))
            self.params[name] = vec[idx:idx + n].reshape(shape).astype(np.float32)
            idx += n

    # -------------------------------------------------------------- forward
    def forward(self, x):
        """x: (INPUT_SIZE,) -> aksiyon indeksi (int)."""
        x = np.asarray(x, dtype=np.float32)
        p = self.params

        # Dense kodlayici (LSTM oncesi)
        e = np.tanh(p["W_e"] @ x + p["b_e"])

        # LSTM
        z = np.concatenate([e, self.h])
        i = _sigmoid(p["W_i"] @ z + p["b_i"])
        f = _sigmoid(p["W_f"] @ z + p["b_f"])
        o = _sigmoid(p["W_o"] @ z + p["b_o"])
        g = np.tanh(p["W_g"] @ z + p["b_g"])
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)

        # Dense cikis
        y = p["W_y"] @ self.h + p["b_y"]
        return int(np.argmax(y))


class MLPPolicy:
    """Saf feedforward beyin:  girdi -> Dense(gizli, tanh) -> Dense(cikis) -> argmax.

    Recurrence yok -> her aksiyon yalnizca o anki girdiden hesaplanir. Genom kucuk,
    mutasyonun etkisi yerel/anlik -> GA icin daha puruzsuz, hizli tirmanilan manzara.
    """

    def __init__(self, genome=None, rng=None):
        self.input_size = INPUT_SIZE
        self.hidden_size = C.MLP_HIDDEN
        self.output_size = OUTPUT_SIZE
        self.rng = rng or np.random.default_rng()

        I, Hh, O = INPUT_SIZE, self.hidden_size, OUTPUT_SIZE
        self._shapes = [
            ("W1", (Hh, I)), ("b1", (Hh,)),   # gizli katman
            ("W2", (O, Hh)), ("b2", (O,)),    # cikis katmani
        ]
        self.params = {}
        if genome is not None:
            self.set_genome(genome)
        else:
            self._init_random()
        # MLP'de hafiza yok; LSTM ile ayni save/load arayuzu icin bos durum
        self.h = np.zeros(0, dtype=np.float32)
        self.c = np.zeros(0, dtype=np.float32)

    def _init_random(self):
        for name, shape in self._shapes:
            if name.startswith("b"):
                self.params[name] = np.zeros(shape, dtype=np.float32)
            else:
                std = 1.0 / np.sqrt(shape[1])
                self.params[name] = (self.rng.standard_normal(shape) * std).astype(np.float32)
        # Kesif onyargisi (LSTM ile ayni): baslangicta hafif "ileri" egilimi.
        self.params["b2"][ACTION_FORWARD] = 0.4
        self.params["b2"][ACTION_NONE] = -0.3
        self.params["b2"][ACTION_BACK] = -0.3

    def reset_state(self):
        pass  # hafiza yok

    @property
    def genome_size(self):
        return sum(int(np.prod(shape)) for _, shape in self._shapes)

    def get_genome(self):
        return np.concatenate([self.params[name].ravel() for name, _ in self._shapes])

    def set_genome(self, vec):
        vec = np.asarray(vec, dtype=np.float32)
        idx = 0
        for name, shape in self._shapes:
            n = int(np.prod(shape))
            self.params[name] = vec[idx:idx + n].reshape(shape).astype(np.float32)
            idx += n

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        p = self.params
        hcur = np.tanh(p["W1"] @ x + p["b1"])
        y = p["W2"] @ hcur + p["b2"]
        return int(np.argmax(y))


# --------------------------------------------------------------------------
# Beyin fabrikasi: config.BRAIN_ARCH'a gore dogru policy'yi uretir
# --------------------------------------------------------------------------
def make_brain(genome=None, rng=None):
    arch = str(getattr(C, "BRAIN_ARCH", "mlp")).lower()
    if arch == "lstm":
        return LSTMPolicy(genome=genome, rng=rng)
    return MLPPolicy(genome=genome, rng=rng)


def brain_genome_size():
    """Aktif mimarinin genom boyutu (checkpoint uyumluluk kontrolu icin)."""
    return make_brain().genome_size


# --------------------------------------------------------------------------
# Genetik operatorler (genom = duz agirlik vektoru)
# --------------------------------------------------------------------------
def crossover(parent_genomes, rng):
    """
    N ebeveynin genomundan tek bir cocuk genomu uretir.
    Her gen icin ebeveynlerden biri rastgele secilir (uniform crossover).
    """
    parents = [np.asarray(g, dtype=np.float32) for g in parent_genomes]
    n_genes = parents[0].size
    n_parents = len(parents)
    # her gen icin hangi ebeveynden alinacagi
    choice = rng.integers(0, n_parents, size=n_genes)
    stacked = np.stack(parents, axis=0)            # (P, G)
    child = stacked[choice, np.arange(n_genes)]    # (G,)
    return child.astype(np.float32)


def mutate(genome, rng, rate, scale):
    """Genoma gauss mutasyonu uygular (yerinde degil, kopya doner)."""
    genome = np.array(genome, dtype=np.float32)
    mask = rng.random(genome.size) < rate
    noise = rng.standard_normal(genome.size).astype(np.float32) * scale
    genome[mask] += noise[mask]
    return genome


def breed(parent_genomes, rng, rate, scale):
    """crossover + mutate kisayolu -> yeni yavru genomu."""
    child = crossover(parent_genomes, rng)
    return mutate(child, rng, rate, scale)
