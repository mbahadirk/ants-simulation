# Sinir Ağı Mimarisi (MLP)

`config.BRAIN_ARCH = "mlp"` aktif mimari. Saf feedforward (recurrence yok):
girdi → Dense(gizli, tanh) → Dense(çıkış) → argmax. Ağırlıklar backprop ile
değil, **genetik algoritma** (crossover + Gaussian mutasyon) ile evrimleşir.

```
INPUT (36)                    HIDDEN (32)              OUTPUT (5)
─────────────                 ────────────             ──────────

Görüş (24)  ──┐
  8 sektör ×   │
  [yakınlık,   │
   besin,      │
   engelli]    │
               │
Homing (3)  ───┤
  sin, cos,    │
  mesafe       │
               ├──►  Dense(32, tanh)  ──►  Dense(5)  ──►  argmax
Feromon (3) ───┤        W1: 32×36           W2: 5×32        │
  sin, cos,    │        b1: 32              b2: 5           ▼
  yerel        │                                        [aksiyon]
               │                                       wait/forward/
Koku (4)    ───┤                                       back/left/right
  sin, cos,    │
  büyüklük,    │
  yerel        │
               │
Ek (2)      ───┘
  taşıyor_mu,
  enerji

Genom boyutu = W1 + b1 + W2 + b2
            = (32×36) + 32 + (5×32) + 5
            = 1152 + 32 + 160 + 5 = 1349 parametre
```

---

## INPUT — 36 nöron

Tüm yön bilgileri **egosentrik** (karıncanın kendi `heading`'ine göre bağıl)
verilir — karınca mutlak açısını bilmek zorunda değildir, sadece "hedef
benim önümde mi, solumda mı, sağımda mı" bilgisini sin/cos olarak alır.

### 1) Görüş — 24 nöron (`VISION_INPUTS = N_SECTORS(8) × SECTOR_FEATURES(3)`)
180° görüş açısı 8 açısal dilime (sektör) bölünür. Her sektör için 3 değer:

| # | Anlamı |
|---|---|
| 1 | **Yakınlık** (0..1, 1=çok yakın) — bu sektördeki en yakın nesnenin mesafesi |
| 2 | **Besin var mı** (one-hot, 0/1) |
| 3 | **Engelli var mı** (one-hot, 0/1) — taş + duvar birleşik |

- Okluzyon var: bir sektördeki en yakın nesne arkasındakini gizler (taşın
  arkasındaki besin görünmez).
- Yuva görüşten çıkarıldı (homing girdisi zaten karşılıyor).
- Diğer karıncalar görüşten çıkarıldı (gereksiz konum gürültüsü yaratıyordu;
  karınca-karınca etkileşimi feromon/koku üzerinden zaten dolaylı var).

### 2) Homing — 3 nöron (`HOMING_INPUTS`)
Yuvanın yönü, path-integration ile (GPS değil, tahmini konum bilgisi):

| # | Anlamı |
|---|---|
| 1 | `sin(yuva_açısı - heading)` — yuva sola mı sağa mı |
| 2 | `cos(yuva_açısı - heading)` — yuva öne mi arkaya mı |
| 3 | Yuvaya normalize mesafe (0..1) |

### 3) Feromon (tek alan) — 3 nöron (`PHEROMONE_INPUTS`)
Tüm karıncaların bıraktığı tek bir iz alanının yerel **gradyanı**:

| # | Anlamı |
|---|---|
| 1 | `sin` — izin güçlendiği yön (heading'e göre bağıl, sol/sağ) |
| 2 | `cos` — izin güçlendiği yön (heading'e göre bağıl, ileri/geri) |
| 3 | Yerel iz yoğunluğu (0..1, `PH_MAX`'a normalize) |

### 4) Besin kokusu — 4 nöron (`ODOR_INPUTS`)
Besin kaynaklarından BFS ile yayılan statik koku alanının gradyanı:

| # | Anlamı |
|---|---|
| 1 | `sin` — kokunun arttığı yön (bağıl) |
| 2 | `cos` — kokunun arttığı yön (bağıl) |
| 3 | Gradyan büyüklüğü (yön sinyalinin gücü, 0..1) |
| 4 | Yerel koku yoğunluğu (0..1) |

> **Not:** Besin taşırken (`carrying=True`) bu 4 değer tamamen **sıfırlanır**
> — karınca koku tarafından dikkati dağılmasın, sadece homing ile yuvaya
> dönsün.

### 5) Ek girdiler — 2 nöron (`EXTRA_INPUTS`)

| # | Anlamı |
|---|---|
| 1 | Besin taşıyor mu (0/1) |
| 2 | Enerji seviyesi (0..1, `ENERGY_MAX`'a normalize) |

---

## HIDDEN — 32 nöron

- Tek gizli katman, **tanh** aktivasyon.
- `W1` (32×36) + `b1` (32) — girişten gizli katmana.
- Genomun en büyük parçası (1184 / 1349 parametre, ~%88).

---

## OUTPUT — 5 nöron (`OUTPUT_SIZE`)

`W2` (5×32) + `b2` (5) ile hesaplanan ham skorlardan **argmax** alınarak tek
bir aksiyon seçilir (softmax/olasılık yok — doğrudan en yüksek skor kazanır):

| İndeks | Aksiyon | Anlamı |
|---|---|---|
| 0 | `wait` | Bekle (hareket etme, dönme) |
| 1 | `forward` | İleri hareket (`MOVE_SPEED`) |
| 2 | `back` | Geri hareket (`BACK_SPEED`) |
| 3 | `left` | Sola dön (`TURN_SPEED`) |
| 4 | `right` | Sağa dön (`TURN_SPEED`) |

Başlangıç ağırlıklarında küçük bir **keşif önyargısı** var:
`b2[forward] = +0.4`, `b2[wait] = b2[back] = -0.3` — rastgele doğan
beyinler de baştan biraz ileri gitmeye eğilimli olsun diye.

---

## Genetik operatörler (öğrenme mekanizması)

Backprop yok. Genom = düz ağırlık vektörü (1349 sayı):
- **Crossover:** Her gen için iki ebeveynden biri rastgele seçilir (uniform crossover).
- **Mutasyon:** `MUTATION_RATE` olasılıkla her gene Gaussian gürültü (`MUTATION_SCALE`) eklenir.
- **Seçilim:** Fitness'a göre (bkz. `penalties and rewards.md`) — başarılı genomlar
  Hall of Fame ve Model Bank'a girer, yeni doğumlar bunlardan türetilir.
