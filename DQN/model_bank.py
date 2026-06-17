"""
Model bankasi: hall-of-fame genomlarinin KALICI (diskte) arsivi.

Amac: bir kosuda evrimlesen en iyi beyinler kaybolmasin; sonraki kosular
(ayni veya FARKLI haritalarda) bu genomlardan tohumlanarak baslasin.
Boylece modeller harita degistikce genellesir ve kosudan kosuya guclenir.

- Girdi: sim.hall (ant_id -> (fitness, genom, nesil)) periyodik birlestirilir.
- Saklanan: en iyi MODEL_BANK_SIZE genom; her girdide hangi haritalarda
  egitildigi (maps listesi) tutulur.
- Mimari korumasi: beyin mimarisi/genom boyutu degisirse eski girdiler
  sessizce atlanir (banka bozulmaz, uyumlu girdiler korunur).
"""

import os
import pickle
import time

import numpy as np

import config as C
from neural_network import brain_genome_size


class ModelBank:
    def __init__(self, path=None):
        self.path = path or C.MODEL_BANK_FILE
        self.entries = []                    # her biri: dict (asagida)
        self.gsize = brain_genome_size()
        self.arch = str(C.BRAIN_ARCH).lower()
        self._load()

    # ------------------------------------------------------------------ I/O
    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[BANK] okunamadi ({e}); bos banka ile devam")
            return
        skipped = 0
        for e in data.get("entries", []):
            g = np.asarray(e.get("genome"), dtype=np.float32)
            if g.size != self.gsize or str(e.get("arch", "")).lower() != self.arch:
                skipped += 1
                continue
            e["genome"] = g
            e.setdefault("maps", [])
            self.entries.append(e)
        self._trim()
        if skipped:
            print(f"[BANK] {skipped} girdi atlandi (mimari/genom boyutu uyumsuz)")
        if self.entries:
            print(f"[BANK] {len(self.entries)} model yuklendi "
                  f"(en iyi fitness {self.entries[0]['fitness']:.1f})")

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({
                "version": 1,
                "arch": self.arch,
                "gsize": self.gsize,
                "entries": self.entries,
            }, f)
        os.replace(tmp, self.path)  # atomik yazim

    # ------------------------------------------------------------- birlestir
    @staticmethod
    def _key(genome):
        return hash(genome.tobytes())

    def merge_hall(self, hall, map_name):
        """sim.hall girdilerini bankayla birlestirir. Degisiklik olduysa True.

        Ayni genom zaten bankadaysa: fitness'i yukseldiyse guncellenir ve
        egitildigi haritalar listesine map_name eklenir. Yeni genomlar
        eklenir; banka en iyi MODEL_BANK_SIZE girdiyle sinirli tutulur.
        """
        changed = False
        index = {self._key(e["genome"]): e for e in self.entries}
        for (fit, genome, gen) in hall.values():
            g = np.asarray(genome, dtype=np.float32)
            if g.size != self.gsize:
                continue
            k = self._key(g)
            prev = index.get(k)
            if prev is not None:
                if fit > prev["fitness"]:
                    prev["fitness"] = float(fit)
                    changed = True
                if map_name not in prev["maps"]:
                    prev["maps"].append(map_name)
                    changed = True
            else:
                entry = {
                    "genome": g.copy(),
                    "fitness": float(fit),
                    "generation": int(gen),
                    "arch": self.arch,
                    "maps": [map_name],
                    "created": time.strftime("%Y%m%d_%H%M%S"),
                }
                self.entries.append(entry)
                index[k] = entry
                changed = True
        if changed:
            self._trim()
        return changed

    def _trim(self):
        self.entries.sort(key=lambda e: e["fitness"], reverse=True)
        del self.entries[C.MODEL_BANK_SIZE:]

    # -------------------------------------------------------------- sorgular
    def top_genomes(self, n=None):
        n = len(self.entries) if n is None else n
        return [e["genome"] for e in self.entries[:n]]

    def best_fitness(self):
        return self.entries[0]["fitness"] if self.entries else 0.0

    def __len__(self):
        return len(self.entries)
