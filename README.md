# Karınca Nöroevrim Simülasyonu 🐜

LSTM + feedforward sinir ağı taşıyan karıncaların **genetik algoritma (nöroevrim)** ile
karınca-benzeri davranış geliştirdiği bir pygame simülasyonu. Karıncalar kısa menzilli
"anten" sensörleriyle çevrelerini algılar, feromon izleri bırakır, besin bulup yuvaya
taşır ve başarılı olanların genetiğinden yeni yavrular doğar.

---

## Kurulum

```bash
pip install -r requirements.txt
python main.py
```

Zorunlu: `pygame`, `numpy`. Ekran kaydı için `opencv-python` önerilir (yoksa kayıt
otomatik olarak PNG kare dizisi olarak alınır).

---

## Kontroller (Simülasyon)

| Tuş | İşlev |
|-----|-------|
| **D** | Debug modu: görüş ışınları + seçili karınca paneli |
| **Z** | Zoom seviyesi (x1 → x1.6 → x2.5 → x4) |
| **S** | Ekran kaydını başlat / durdur (`recordings/` klasörüne) |
| **Space** | Duraklat / devam |
| **Ok tuşları** | Haritada gez (pan) |
| **Sol tık** | (Debug) bir karınca seç ve kamerayla takip et |
| **R** | Kamerayı sıfırla / takibi bırak |
| **ESC** | Menüye dön |

## Kontroller (Harita Editörü)

- **Sol panel butonları** veya **1-4 / 0** tuşları: Besin, Taş, Engel, Yuva, Sil
- **Sol tık (sürükle)**: yerleştir · **Sağ tık**: sil
- **[ ] tuşları**: fırça boyutu
- **Kaydet** (veya Ctrl+S): `maps/default_map.json`
- **Çerçeve Duvar**: kenarlara hızlı duvar · **Temizle**: haritayı boşalt
- **ESC / Menü**: ana menüye dön

> Editörde kaydettiğin harita, simülasyon başlarken otomatik yüklenir.

---

## Nasıl çalışır?

### Sinir ağı (saf NumPy, backprop yok)
`girdi → LSTM(16) → Dense(5) → argmax → aksiyon`

- **Aksiyonlar:** `bekle, ileri, geri, sol, sağ` (yalnızca bu 5'i).
- Her karıncanın kendi LSTM gizli durumu vardır → zamansal hafıza.
- Ağın tüm ağırlıkları bir **genom** (vektör) olarak ele alınır.

### Girdiler (53 değer)
- **7 anten ışını** × `[mesafe, besin, taş, engel, karınca, yuva]` = 42
- **Homing** (yön bütünleme): yuva yönü `sin/cos` + mesafe = 3
- **Feromon antenleri**: sol/orta/sağ × (home izi, food izi) = 6
- **Durum**: taşıyor mu, enerji = 2

### Feromon
Gerçek karıncalar gibi: besin taşımayanlar **home izi**, besin taşıyanlar **food izi**
bırakır. İzler buharlaşır ve hafifçe yayılır; antenlerle koklanıp ağa girdi olur.
Bu, "ilk bulan yol açar, diğerleri takip eder" pozitif geri beslemesini doğurur.
Ekranda izler parlayarak görünür (belgesel için).

### Genetik algoritma
- Başlangıçta `INITIAL_POP` karınca **rastgele** genomlarla doğar.
- Yuvaya her **3 besin** teslim edildiğinde, o besinleri getiren **3 karıncanın**
  genomundan **uniform crossover + gauss mutasyon** ile **1 yavru** üretilir.
- **Fitness takviyesi:** popülasyon `MIN_POP` altına düşerse rastgele yerine en çok
  besin bulan/teslim eden hayatta kalanların genomundan takviye yapılır.

### Yaşam döngüsü
- Her karıncanın **45–60 sn** ömrü vardır (yaşlılıktan ölür).
- **Açlık:** enerji zamanla azalır; besin bulununca sıfırlanır. Yeterince uzun süre
  (`STARVE_TIME`) besine ulaşamayan karınca ölür.

---

## Ayar düğmeleri (`config.py`)

Tüm denge buradan ayarlanır. Sık kullanılanlar:

| Sabit | Anlamı |
|-------|--------|
| `VISION_RANGE`, `N_RAYS`, `VISION_FOV` | Görüş menzili / ışın sayısı / açısı |
| `LIFESPAN_MIN/MAX`, `STARVE_TIME` | Ömür ve açlık süreleri |
| `MOVE_SPEED`, `TURN_SPEED` | Hareket / dönüş hızı |
| `INITIAL_POP`, `MIN_POP`, `MAX_POP` | Popülasyon sınırları |
| `FOOD_PER_BIRTH`, `N_PARENTS` | Üreme eşiği / ebeveyn sayısı |
| `MUTATION_RATE`, `MUTATION_SCALE` | Mutasyon yoğunluğu |
| `PH_DEPOSIT_*`, `PH_EVAPORATION` | Feromon bırakma / buharlaşma |
| `HIDDEN_SIZE` | LSTM gizli katman boyutu |

> **Not — evrim bootstrap'i:** Sıfırdan rastgele beyinlerle, seyrek/uzak besinde koloninin
> kendiliğinden gelişmesi zordur. Bunu kolaylaştırmak için (a) başlangıç ağlarına hafif
> "ileri" keşif eğilimi verildi, (b) varsayılan haritaya yuva çevresine yakın besin halkası
> eklendi. Kendi haritanı yaparken keşfin mümkün olması için yuvaya makul yakınlıkta da
> besin bırakman koloninin daha hızlı gelişmesini sağlar.

---

## Dosya yapısı

```
main.py            # menü + simülasyon döngüsü + tuş yönetimi
config.py          # tüm sabitler / ayarlar
neural_network.py  # NumPy LSTM+FF, crossover/mutate/breed
ant.py             # karınca: sensör, homing, feromon, hareket, fitness
world.py           # ızgara harita, raycast, feromon alanları, kayıt/yükleme
simulation.py      # popülasyon, üreme, fitness takviyesi
camera.py          # zoom + pan + takip
renderer.py        # dünya/feromon/karınca/debug/HUD çizimi
map_editor.py      # UI tabanlı harita editörü
recorder.py        # ekran kaydı (cv2 / imageio / PNG)
ant.png            # karınca görseli
maps/              # kaydedilen haritalar
recordings/        # kayıtlar
```
