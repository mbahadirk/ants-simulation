# DQN Surumu (Reinforcement Learning) — Anlik Yedek

Bu klasor, projenin **Deep Q-Network (DQN)** ile calisan tam surumunun bir
anlik kopyasidir. Ana proje genetik algoritmaya (GA) geri dondurulmustur; bu
klasor DQN calismasini ileride geri getirebilmek icin saklanmistir.

## Calistirma
Bu klasoru bagimsiz bir proje gibi calistirabilirsiniz:
```bash
cd DQN
python main.py
```
(`config.py` icinde `TRAIN_MODE = "dqn"`.)

## DQN'e ozgu dosyalar/ozellikler
- `dqn.py` — NumPy Q-agi (elle backprop + Adam), Double-DQN, dengeli replay
  (tasima gecisleri icin ayri tampon), per-ant (Ape-X) epsilon, .npz kalicilik
- `dqn_sim.py` — paylasilan politika simulasyonu (Simulation ile ayni arayuz),
  yuvaya yakin besin curriculum bootstrap'i, teslimat-kaynakli koloni buyumesi
- `ant.py` — `observe()` / `apply_action()` ayrimi, potansiyel-tabanli adim-basina
  odul, tasirken koku girdisi kapali
- `config.py` — `TRAIN_MODE`, tum `RL_*` hiperparametreleri
- Paylasilan tek Q-agi; per-ant epsilon ile dagitik kesif; global koku gradyani

## Onemli kararlar (teshis gunlugu)
- Fitness gurultusu kaniti: ayni genom -91..+6 fitness -> neuroevrim sinyali zayif
- Epsilon pacing, adim maliyeti, global koku, dengeli replay, tasirken koku kapatma
  adim adim duzeltildi; greedy degerlendirmede gercek teslimat dogrulandi.
