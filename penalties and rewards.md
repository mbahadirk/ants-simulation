# Ödül ve Ceza Mekanizmaları

Bu doküman, `ant.py` ve `config.py` içinde tanımlı tüm fitness (ödül/ceza) ve
enerji mekanizmalarını listeler. Fitness, genetik algoritmanın seçilim
baskısını oluşturan **tek kaynak**tır (`Ant.fitness()` sadece
`self.fitness_bonus` döner — çift sayım yok).

İki ayrı katman vardır:
1. **Fitness cezası/ödülü** — doğrudan üreme şansını etkiler (GA seçilimi).
2. **Enerji cezası** — `self.energy`'i düşürür, enerji 0'a inerse karınca ölür
   (hayatta kalma baskısı, fitness'tan ayrı).

---

## 1. Hareket / Engel Cezaları

### Duvara çarpma (WALL)
- **Tetik:** `ACTION_FORWARD` veya `ACTION_BACK` seçildi ama hareket
  edemedi (taş/engel/harita kenarı).
- **Enerji cezası:** `WALL_PENALTY_RATE = 0.08` /sn
- **Fitness cezası:** `WALL_FIT_PENALTY = 0.10` (adım başına sabit)
- **Sayaç:** `self.wall_hits += 1`

### Dış çerçeve (BORDER) — ek ceza
- **Tetik:** Duvar çarpması SIRASINDA karınca haritanın en dış hücresinde ise
  (kenara yapışmayı caydırmak için).
- **Enerji cezası:** `BORDER_PENALTY_RATE = 0.10` /sn (WALL cezasına ek)
- **Fitness cezası:** `BORDER_FIT_PENALTY = 0.22` (WALL cezasına ek)

### Sabit durma (IDLE)
- **Tetik:** `ACTION_NONE` seçildi (bilinçli bekleme).
- **Enerji cezası:** `IDLE_PENALTY_RATE = 0.0` → **enerji tüketmez** (bekleme
  hayatta kalmayı tehlikeye atmaz).
- **Fitness cezası:** `IDLE_FIT_PENALTY = 0.06` (adım başına; sürekli durmayı
  GA seviyesinde caydırır ama can almaz)
- **Sayaç:** `self.idle_steps += 1`

### Sağ-sol salınımı (JITTER) — **YENİ (düzeltildi)**
- **Tetik:** **Ardışık 2 veya daha fazla** ters dönüş (örn. SAĞ→SOL→SAĞ) —
  yerinde ileri/geri ilerleme olmadan tekrar tekrar yön değiştirmek.
- **Fitness cezası:** `JITTER_FIT_PENALTY = 0.025` (eşik aşıldığında adım
  başına)
- **Neden eklendi:** Karıncalar sabit durma cezasından kaçınmak için
  `ACTION_NONE` yerine sağ-sol dönüşü ardışık tekrarlayarak aynı yerde
  titreyebiliyordu.
- **Önemli düzeltme:** İlk versiyonda **tek bir** ters dönüş bile
  cezalandırılıyordu. Bu, dar bir boşluktan geçerken veya besine hassas
  hizalanırken yapılan **normal düzeltme** hareketini de cezalandırıyordu —
  sonucunda karıncalar dönüşten tamamen kaçınıp duvara/taşa yapışmayı veya
  besin üzerinde donup kalmayı (dönmemeyi) öğrendi. Düzeltme: sayaç
  (`self.turn_reversal_count`) sadece **ardışık 2. ters dönüşten** itibaren
  ceza uygular; tek seferlik düzeltme artık ücretsizdir.
- **Not:** `FORWARD`/`BACK`/`NONE` aksiyonu görülünce zincir ve sayaç
  sıfırlanır (gerçek bir ilerleme sonrası yön değiştirmek titreme sayılmaz).

---

## 2. Besin Bulma Ödülü

- **Tetik:** Karınca boşken bir besin hücresine ulaştı (`world.take_food`
  başarılı).
- **Ödül formülü:**
  ```
  fitness_bonus += FITNESS_FIND_BASE + dnest * FITNESS_FIND_DIST_W
  ```
  - `FITNESS_FIND_BASE = 0.0` — sabit taban **kaldırıldı**: yuvaya çok yakın
    besin bulmak ~0 puan verir, "yuva etrafında dönüp besin toplama"
    stratejisi ödülsüz kalır.
  - `FITNESS_FIND_DIST_W = 0.06` — besin yuvadan ne kadar uzaktaysa o kadar
    fazla ödül (piksel başına).
  - `dnest` = besinin bulunduğu noktanın yuvaya mesafesi.
- **Yan etkiler:**
  - `self.last_find_dist = dnest` kaydedilir (teslim ödülünü ölçeklemek için).
  - `self.min_home_dist` / `self.prev_home_dist` bu mesafeyle başlatılır
    (geri dönüş ödülü için referans).
  - `self.max_explore_dist = 0.0` (keşif ratchet'i sıfırlanır — yeni arayış
    başlıyor).
  - **Ömür bonusu:** `LIFESPAN_FOOD_BONUS=True` ise ömre bir taban ömür kadar
    süre eklenir (en fazla `LIFESPAN_MAX_MULT=4.0` kat tabana kadar).
  - Enerji tam dolar (`energy = ENERGY_MAX`) — açlıktan ölme riski sıfırlanır.

---

## 3. Teslimat Ödülü

- **Tetik:** Karınca besin taşırken yuvaya ulaştı (`world.at_nest`).
- **Ödül formülü:**
  ```
  fitness_bonus += FITNESS_DELIVER_W + last_find_dist * FITNESS_DELIVER_DIST_W
  ```
  - `FITNESS_DELIVER_W = 4.0` — teslim başına sabit ödül (önceden 8.0 idi;
    ağırlık mesafe bileşenine kaydırıldı).
  - `FITNESS_DELIVER_DIST_W = 0.06` — besin ne kadar uzaktan getirildiyse o
    kadar ek ödül (önceden 0.02 idi, 3× artırıldı).
  - Sonuç: yuva yanında topla-getir turu çok az puan verir; uzun mesafe
    forage'lar belirgin şekilde daha değerli olur → gerçek forager'lar Hall
    of Fame'e girer.

### Yol verimliliği bonusu — **YENİ**
- **Tetik:** Teslimat anında, taşıma boyunca kat edilen gerçek mesafe
  (`self.carry_distance` — her adımda `disp` ile biriktirilir) düz hat
  mesafesiyle (`last_find_dist`) karşılaştırılır.
- **Ödül formülü:**
  ```
  efficiency = min(1.0, last_find_dist / max(carry_distance, 1.0))
  fitness_bonus += efficiency * PATH_EFFICIENCY_W
  ```
  - `PATH_EFFICIENCY_W = 3.0` — teslimat başına, verimlilik oranıyla ölçekli
    bonus (1.0 = mükemmel düz yol, daha dolambaçlı yollar daha az bonus alır).
  - **Mevcut teslimat ödülünü DEĞİŞTİRMEZ, ona EKTİR** — sadece kısa/dogrudan
    yol bulan karıncalara fazladan ödül verir.
- **Neden eklendi:** Besin bulma ve yuvaya dönme ödülleri zaten vardı ama
  hiçbiri "ne kadar VERİMLİ" gittiğini ölçmüyordu — uzun dolambaçlı bir yolla
  da gelen karınca aynı teslimat ödülünü alıyordu. Bu bonus, daha kısa/düz
  yol bulan genomları doğrudan ödüllendirerek seçilim baskısı oluşturur.
- **Sıfırlamalar:** `min_home_dist`, `prev_home_dist` → `None`;
  `max_odor_seen`, `max_food_ph_seen` → `0.0` (yeni arayış döngüsü başlar).
  `carry_distance` bir sonraki besin alımında `0.0`'a sıfırlanır.

---

## 4. Geri Dönüş Ödül Şekillendirmesi (taşırken)

Sadece `self.carrying == True` iken çalışır.

### Yuvaya yaklaşma ödülü (RATCHET)
- **Tetik:** Bu adımda yuvaya, o taşıma turunda ulaşılan **en yakın**
  mesafeden daha yakına gelindi.
- **Ödül formülü:**
  ```
  fitness_bonus += (min_home_dist - home_dist) * RETURN_REWARD_W
  ```
  - `RETURN_REWARD_W = 0.040` (piksel başına).
  - Ratchet mantığı: sadece yeni bir minimum mesafeye ulaşılırsa ödül verilir
    → yerinde salınarak (ileri-geri) farm edilemez.

### Yuvadan uzaklaşma cezası — **KAPATILDI**
- **Eski davranış:** Taşırken yuvadan uzaklaşınca
  `fitness_bonus -= (home_dist - prev_home_dist) * CARRY_AWAY_PENALTY_W`
  cezası uygulanırdı.
- **Neden kapatıldı:** Taş/engel etrafından dolanmak için geçici olarak
  yuvadan uzaklaşmak ZORUNLU. Bu ceza, taşıyan karıncaların engelin
  çevresinden dolanmaya çalışırken (anlık uzaklaşma gerekiyor) titreyip
  kilitlenmesine sebep oluyordu.
- **Güncel değer:** `CARRY_AWAY_PENALTY_W = 0.0` (kod hâlâ mevcut ama etkisiz;
  istenirse `config.py`'den tekrar açılabilir).

### Yuvaya bakma ödülü — **YENİ**
- **Tetik:** Taşırken homing açısı (`nrel` — yuvanın heading'e göre bağıl
  açısı) ne kadar küçükse (yuva tam önde) ve karınca **gerçekten hareket
  ediyorsa** (`disp > 1e-3`) o kadar ödül.
- **Ödül formülü:**
  ```
  align = max(0.0, cos(nrel))           # 1.0 = yuva tam onunde, 0 = yan/arka
  fitness_bonus += align * FACE_NEST_REWARD_W * disp
  ```
  - `FACE_NEST_REWARD_W = 0.05` (piksel başına, hizalanmayla ölçekli).
  - `disp` çarpanı kritik: **sabit durup sadece yuvaya bakarak ödül
    farmlanamaz** — ödül ancak kat edilen mesafeyle birlikte gelir. Aynı
    mantık feromon bırakma ve diğer hareket-bağımlı ödüllerde de kullanılıyor.
  - `align` negatife düşmez (`max(0, cos(nrel))`) — yuvaya sırtını dönmüş
    hareket ekstra ceza almaz, sadece bu ödülden pay alamaz.
- **Neden eklendi:** `RETURN_REWARD_W` sadece mesafenin azalmasını ödüllendiriyordu;
  bu ödül ayrıca **doğru yöne bakma/yönelme** davranışını doğrudan teşvik
  ediyor — homing açısının küçük tutulması (yuvaya dönük gitme) öğrenilir.

---

## 5. Keşif / Takip Ödül Şekillendirmesi (boş gezerken)

Sadece `self.carrying == False` iken çalışır.

### Koku tırmanma ödülü (RATCHET)
- **Tetik:** Algılanan besin kokusu (`last_odor`) o arayışta ulaşılan en
  yüksek seviyeyi geçti.
- **Ödül formülü:**
  ```
  fitness_bonus += (last_odor - max_odor_seen) * FORAGE_REWARD_W
  ```
  - `FORAGE_REWARD_W = 6.0` (koku 0..1 aralığında, artış başına).
  - Ratchet: sadece yeni en yüksek seviyeye ulaşılırsa ödül.

### İz takip ödülü (RATCHET)
- **Tetik:** Algılanan food-feromon yoğunluğu (`last_food_ph`) o arayışta
  ulaşılan en yüksek seviyeyi geçti.
- **Ödül formülü:**
  ```
  fitness_bonus += (last_food_ph - max_food_ph_seen) * TRAIL_FOLLOW_W
  ```
  - `TRAIL_FOLLOW_W = 9.0`.
  - Başarılı karıncaların bıraktığı izi takip etmeyi doğrudan ödüllendirir
    (gerçek karınca kolonisi pozitif geri beslemesinin taklidi).

### Keşif ödülü (RATCHET) — **YENİ**
- **Tetik:** Karınca yuvadan o arayışta hiç gidilmemiş en uzak mesafeye
  ulaştı.
- **Ödül formülü:**
  ```
  fitness_bonus += (cur_dist - max_explore_dist) * EXPLORE_REWARD_W
  ```
  - `EXPLORE_REWARD_W = 0.012` (piksel başına).
  - Ratchet: yerinde dönerek farm edilemez; sadece gerçekten daha uzağa
    giden karıncalar ödüllenir. Amaç: uzak besin kaynaklarına ulaşmak için
    aktif keşfi teşvik etmek.

### Görülen besine yönelme ödülü (hareket-bağımlı) — **YENİ (v2)**
- **Tetik:** Görüş alanında (180°, `last_seen`) besin görünüyor, karınca
  **gerçekten hareket ediyor** (`disp > 1e-3`) ve hareketi besine doğru
  **açısal hizalı**.
- **Ödül formülü:**
  ```
  fx, fy = en_yakin_gorunen_besinin_konumu
  frel = wrap_angle(atan2(fy-y, fx-x) - heading)
  align = max(0.0, cos(frel))         # 1.0 = besin tam onunde
  fitness_bonus += align * FOOD_FACE_REWARD_W * disp
  ```
  - `FOOD_FACE_REWARD_W = 0.05` (piksel başına, hizalanmayla ölçekli).
  - `FACE_NEST_REWARD_W` ile birebir aynı desen: ödül hem hizalanmaya HEM
    hareket miktarına (`disp`) bağlı — sabit durup besine bakarak farm
    edilemez.
- **İlk versiyondan (v1) farkı:** Önceki sürüm sadece "besine olan mesafe
  azaldı mı" (ratchet) bakıyordu. Bu, dolaşarak da zaman zaman mesafe
  azaltılabildiği için zayıf bir sinyaldi — karıncalar besini görse de
  etrafında dolanmayı tercih edip doğrudan üzerine gitmiyordu. v2, "besine
  DOĞRU yürüyor musun" (yön + hareket) ölçer; yan/geri hareket veya dolanma
  artık ödül vermez.
- **Neden eklendi:** Koku tırmanma (`FORAGE_REWARD_W`) genel yön sağlar ama
  besinin TAM ÜSTÜNE gitmeyi garanti etmez. Bu ödül, gözle görülen besine
  doğrudan yönelmeyi/yaklaşmayı ayrıca ödüllendirir.

---

## 6. Feromon Bırakma (ödül değil, davranışsal mekanizma)

Fitness'a doğrudan girmez ama dolaylı olarak iz-takip ödülünü (`TRAIL_FOLLOW_W`)
besler.

- **Tetik:** Bu adımda kat edilen mesafe (`disp`) > `1e-3` (sabit durma /
  dönme / titremede birikim olmaz).
- **Formül:**
  ```
  amount = PH_DEPOSIT_BASE * (PH_CARRY_MULT if carrying else 1.0)
  world.deposit(x, y, amount * disp)
  ```
  - `PH_DEPOSIT_BASE = 0.04` (piksel başına, boş gezen karınca).
  - `PH_CARRY_MULT = 25.0` — besin taşıyan karınca 25× daha güçlü iz bırakır.
  - Birikim `PH_MAX = 200.0`'a kadar sınırlı; `PH_EVAPORATION = 0.02`/sn
    hızında buharlaşır.

---

## 7. Enerji / Yaşam Süresi (ölüm koşulları)

Fitness'tan ayrı; karınca yaşıyor mu, ölüyor mu belirler.

- **Pasif açlık tüketimi:** Her adımda `energy -= dt / STARVE_TIME`
  (`STARVE_TIME = 40.0` sn → ~40 saniyede besin bulunamazsa ölür).
- **Ölüm koşulu:** `energy <= 0.0` **veya** `age >= lifespan`.
- **Enerji sıfırlanma noktaları:**
  - Besin bulununca (`energy = ENERGY_MAX`)
  - Yuvaya teslim edince (`energy = ENERGY_MAX`)
- **Ömür:** `lifespan` başlangıçta `[LIFESPAN_MIN, LIFESPAN_MAX] = [150, 180]`
  sn arasında rastgele atanır; besin bulununca `LIFESPAN_FOOD_BONUS` aktifse
  taban ömür kadar uzar (en fazla `LIFESPAN_MAX_MULT=4.0` kat tabana kadar).

---

## Özet Tablo

| Mekanizma | Tip | Tetik | Ağırlık | Katman |
|---|---|---|---|---|
| Wall hit | Ceza | Duvara çarpma | `WALL_PENALTY_RATE=0.08`/sn, `WALL_FIT_PENALTY=0.10` | Enerji + Fitness |
| Border hit | Ceza (ek) | Dış çerçevede çarpma | `BORDER_PENALTY_RATE=0.10`/sn, `BORDER_FIT_PENALTY=0.22` | Enerji + Fitness |
| Idle | Ceza | Bekleme aksiyonu | `IDLE_PENALTY_RATE=0.0`, `IDLE_FIT_PENALTY=0.06` | Fitness only |
| Jitter (salınım) | Ceza | Ardışık ters dönüş (sağ↔sol) | `JITTER_FIT_PENALTY=0.04` | Fitness only |
| Besin bulma | Ödül | Besin alındı | `FITNESS_FIND_DIST_W=0.06` × mesafe | Fitness |
| Teslimat | Ödül | Yuvaya teslim | `FITNESS_DELIVER_W=4.0` + `0.06`×mesafe | Fitness |
| Yol verimliliği | Ödül | Teslimde düz/kat-edilen mesafe oranı | `PATH_EFFICIENCY_W=3.0` × oran | Fitness |
| Yuvaya yaklaşma | Ödül (ratchet) | Taşırken yeni min mesafe | `RETURN_REWARD_W=0.040` | Fitness |
| Yuvaya bakma | Ödül (hareket-bağımlı) | Taşırken hizalanma × disp | `FACE_NEST_REWARD_W=0.05` | Fitness |
| Yuvadan uzaklaşma | **Kapalı** | — | `CARRY_AWAY_PENALTY_W=0.0` | — |
| Koku tırmanma | Ödül (ratchet) | Boşken yeni max koku | `FORAGE_REWARD_W=6.0` | Fitness |
| İz takip | Ödül (ratchet) | Boşken yeni max feromon | `TRAIL_FOLLOW_W=9.0` | Fitness |
| Keşif | Ödül (ratchet) | Boşken yeni max uzaklık | `EXPLORE_REWARD_W=0.012` | Fitness |
| Besine yönelme | Ödül (hareket-bağımlı) | Boşken görülen besine hizalanma × disp | `FOOD_FACE_REWARD_W=0.05` | Fitness |
| Açlık | Ölüm koşulu | `STARVE_TIME=40` sn besin yok | — | Enerji |
| Yaşlanma | Ölüm koşulu | `age >= lifespan` | — | Yaşam süresi |

Tüm sayısal değerler `config.py` içinde tanımlıdır ve simülasyon çalışırken
ayarlar paneli veya doğrudan dosya düzenlemesiyle değiştirilebilir.
