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
| **D** | Debug modu: görüş ışınları + koku (belirgin) + seçili karınca paneli |
| **Z** | Zoom seviyesi (x1 → x1.6 → x2.5 → x4) — **fare imlecine doğru** |
| **Fare tekeri** | Kademe kademe yakınlaş/uzaklaş (imlece doğru) |
| **O / P** | Simülasyon hızı: **O** yarıya indir, **P** iki katına çıkar (x0.25–x16). HUD'da "Speed" |
| **K / L** | Karınca ömrü: **K** 15 sn azalt, **L** 15 sn artır (varsayılan 150–180 sn). HUD'da "Omur" |
| **S** | Ekran kaydını başlat / durdur (`recordings/` klasörüne) |
| **Space** | Duraklat / devam |
| **Ok tuşları** | Haritada gez (pan) |
| **Sol tık** | (Debug) bir karınca seç ve kamerayla takip et |
| **R** | Kamerayı sıfırla / takibi bırak |
| **ESC** | Menüye dön |

## Kontroller (Harita Editörü)

- **Sol panel butonları** veya **1-4 / 0** tuşları: Besin, Taş, Engel, Yuva, Sil
- **Sol tık (sürükle)**: yerleştir · **Sağ tık**: sil
- **Besin miktarı kutusu**: tıkla, bir sayı yaz (ör. `10`), Enter. Bundan sonra
  koyduğun her besin o kadar kez alınınca biter. Hücrelerin üstünde kalan miktar yazar.
- **[ ] tuşları**: fırça boyutu
- **Kaydet** (veya Ctrl+S): `maps/default_map.json` (miktarlar dahil kaydedilir)
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

### Girdiler (86 değer)
- **180° sektör görüş**: `N_SECTORS=12` dilim × `[yakınlık, besin, taş, engel, karınca, yuva]` = 72
- **Homing** (yön bütünleme): yuva yönü `sin/cos` + mesafe = 3
- **Feromon antenleri**: sol/orta/sağ × (home izi, food izi) = 6
- **Besin kokusu antenleri**: sol/orta/sağ × koku gradyanı = 3
- **Durum**: taşıyor mu, enerji = 2

### 180° sektör görüş (oklüzyonlu)
Karıncanın önündeki **180°** görüş açısı `N_SECTORS` açısal **dilime** bölünür.
Her dilimde o yöndeki **en yakın** nesne görünür; **arkasındaki nesneler bu yakın
nesne tarafından gizlenir** (oklüzyon). Böylece:
- Bir **taşın arkasındaki besin görünmez** (ajanları yanlış yöne sürüklemez).
- Ama 180° içindeki **farklı yönlerdeki tüm nesneler** görünür (her yön = bir dilim).
- Arkadaki (±90° dışı) nesneler görünmez.

Her dilim için ağa: yakınlık + nesne tipi (one-hot) verilir. Debug modunda (D) **180°
görüş yelpazesi** ve görülen nesnelere giden renkli çizgiler çizilir.

### Besin kokusu (geniş, harita çapında gradyan)
Besin kokusu **çok-kaynaklı BFS** ile hesaplanır: her hücre için en yakın besine
olan (duvarlardan **dolanarak**) mesafe bulunur, koku = `1 - mesafe/menzil`. Böylece
koku **tüm haritaya** yayılır (varsayılan menzil 45 hücre → haritanın ~%93'ü) ve
karıncalar yoğunluğu **tırmanarak** besini uzaktan bulur. Duvar arkasındaki besinin
kokusu, etrafından dolaşma mesafesi kadar zayıftır (gerçekçi). `ODOR_RANGE_CELLS`
ile menzili büyütüp küçültebilirsin. Besin bitince gradyan otomatik yeniden hesaplanır.

### Besin (miktarlı kaynak + periyodik)
Her besin kaynağının bir **miktar değeri** vardır (editörde girilir, varsayılan
`FOOD_DEFAULT_AMOUNT=10`). Bir karınca bir birim alıp taşır; miktar **0 olunca
kaynak biter** (hücre boşalır ve koku gradyanı yeniden hesaplanır). Aynı kaynaktan
miktar bitene kadar farklı karıncalar tekrar tekrar alabilir.

**Periyodik besin:** her `FOOD_SPAWN_INTERVAL` (60) saniyede bir, taş/engel olmayan
rastgele boş bir hücrede `FOOD_SPAWN_AMOUNT` (5) değerinde yeni bir besin kaynağı oluşur.

### Feromon
Gerçek karıncalar gibi: besin taşımayanlar **home izi**, besin taşıyanlar **food izi**
bırakır. İzler buharlaşır ve hafifçe yayılır; antenlerle koklanıp ağa girdi olur.
Bu, "ilk bulan yol açar, diğerleri takip eder" pozitif geri beslemesini doğurur.
Ekranda izler parlayarak görünür (belgesel için).

> **Koku vs feromon farkı:** *koku* besinin kendisinden yayılır (sabit), *feromon*
> karıncaların bıraktığı izlerdir (dinamik). İkisi birlikte besin bulmayı ve
> yuvaya dönüşü çok güçlü kılar.

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

### Ödül ve ceza + reward shaping (evrimin ANAHTARI)
Sıfırdan rastgele beyinlerle, üreme yalnızca *teslimat*a bağlı olsaydı evrim hiç
başlamazdı (teslimat çok nadir → seçilim yok → öğrenme yok = "seyrek ödül tuzağı").
Bunu **ödül şekillendirme (reward shaping)** ile kırıyoruz — karıncalar tam
sonuç olmadan da **doğru yönde ilerledikçe** puan alır:

- **Geri dönüş ödülü** (`RETURN_REWARD_W`): besin taşırken yuvaya her *yeni en yakın*
  mesafeye ulaştığında puan. → "besini yuvaya geri götürme" davranışı kademeli evrimleşir.
  Ağ zaten yuva yönünü (homing girdisi) görüyor; eksik olan seçilim baskısıydı, o da bu.
- **Arama ödülü** (`FORAGE_REWARD_W`): boş gezerken besin kokusunu *yeni en yüksek*
  seviyeye tırmandığında puan. → "kokuyu takip edip besine git" davranışı evrimleşir.
- (Farming'i önlemek için sadece **yeni en iyi ilerleme** ödüllenir; ileri-geri
  gidip puan toplanamaz.)

Klasik ödül/ceza:
- **Ödül** ✅: besin bulma (`FITNESS_FIND_W`), yuvaya teslim (`FITNESS_DELIVER_W`, en ağır).
  Teslim üremeyi tetikler; bulma/teslim enerjiyi sıfırlar (hayatta kalma).
- **Ceza** ❌: duvara/taşa çarpıp ilerleyememe (`WALL_PENALTY_RATE`) ve "bekle"
  (sabit durma, `IDLE_PENALTY_RATE`) **ekstra enerji** tüketir → daha çabuk ölüm →
  evrimde elenir. Açlık ve yaşlılık da doğal cezalardır.

Bu shaping'li fitness, popülasyon takviyesinde de kullanılır: ilk teslimattan önce
bile "eve doğru ilerleyen" karıncalar seçilir, böylece koloni **hızlanarak** gelişir.
Debug panelinde (D + karıncaya tıkla) anlık koku algısı, food feromonu, duvar çarpma
ve bekleme sayaçları görünür — karıncanın kokuyu aldığını buradan doğrularsın.

### Hız kontrolü
**O** hızı yarıya, **P** iki katına çıkarır (x0.25–x16). Sabit zaman adımlı fizik
sayesinde hız değişse de davranış kararlıdır. HUD'da `Speed: x..` görünür.

### Ömür kontrolü
**K** karınca ömrünü 15 sn azaltır, **L** 15 sn artırır (varsayılan 150–180 sn). Ömür
kısaldığında popülasyon turnover hızlanır → seçilim daha güçlü olur → evrim hızlanır
(ama koloni çöp riskine girer). Uzattığında evrim yavaşlar ama karıncalara daha çok
deneme süresi tanıyabilir. HUD'da mevcut aralık `Omur: 150-180s` olarak görünür.

---

## Ayar düğmeleri (`config.py`)

Tüm denge buradan ayarlanır. Sık kullanılanlar:

| Sabit | Anlamı |
|-------|--------|
| `VISION_RANGE`, `N_SECTORS`, `VISION_FOV` | Görüş menzili / dilim sayısı / açısı (180°) |
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
