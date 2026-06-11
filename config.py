"""
Genel yapilandirma / sabitler.
Tum moduller buradan beslenir. Degerleri buradan ayarlayabilirsin.
"""

# ---------------------------------------------------------------------------
# Ekran / dunya
# ---------------------------------------------------------------------------
CELL_SIZE = 20                  # bir izgara hucresinin piksel boyutu
GRID_W = 80                     # yatay hucre sayisi
GRID_H = 50                     # dikey hucre sayisi

WORLD_W = GRID_W * CELL_SIZE     # dunya genisligi (piksel) = 1600
WORLD_H = GRID_H * CELL_SIZE     # dunya yuksekligi (piksel) = 1000

# Pencere/render cozunurlugu (dunya ile birebir -> net goruntu, net kayit).
# Kucuk ekranda calisiyorsan bu degerleri dusur (dunya da otomatik kuculur).
SCREEN_W = 1600
SCREEN_H = 1000
FPS = 60

# Simulasyon hizi (O: yariya indir, P: iki katina cikar)
SIM_SPEED_DEFAULT = 1.0
SIM_SPEED_MIN = 0.25
SIM_SPEED_MAX = 16.0
FIXED_DT = 1.0 / 60.0           # sabit fizik adimi (hizdan bagimsiz kararlilik)
MAX_SUBSTEPS = 40               # bir karede en fazla alt-adim (spiral'i onler)

# ---------------------------------------------------------------------------
# Hucre tipleri (map editor + dunya bunlari kullanir)
# ---------------------------------------------------------------------------
EMPTY = 0
FOOD = 1
STONE = 2        # tas: gecilmez, "tas" olarak algilanir
OBSTACLE = 3     # engel/duvar: gecilmez, "engel" olarak algilanir
NEST = 4         # yuva

TILE_NAMES = {
    EMPTY: "Empty",
    FOOD: "Food",
    STONE: "Stone",
    OBSTACLE: "Obstacle",
    NEST: "Nest",
}

# Hucre renkleri
COLORS = {
    EMPTY: (20, 45, 20),
    FOOD: (230, 50, 50),
    STONE: (130, 130, 140),
    OBSTACLE: (70, 55, 45),
    NEST: (200, 150, 60),
}
GRID_LINE_COLOR = (30, 60, 30)
BG_COLOR = (15, 38, 15)

# ---------------------------------------------------------------------------
# Karinca / hareket
# ---------------------------------------------------------------------------
ANT_SIZE = 18                   # cizim boyutu (piksel)
MOVE_SPEED = 180.0              # ileri hiz (piksel / saniye)
BACK_SPEED = 90.0               # geri hiz
TURN_SPEED = 14.0               # donus hizi (radyan / saniye) - iki katina cikarildi

# Yasam suresi (saniye) -> her karinca icin bu araliktan rastgele secilir
# K/L tuslariyla dinamik olarak ayarlanabilir (+/- 15sn)
LIFESPAN_MIN = 150.0
LIFESPAN_MAX = 180.0
# Besin BULUNCA omre 1 omur kadar (taban omur) sure eklenir -> forager'lara
# daha fazla zaman taninir. Toplam omur tabanin bu kadar katiyla sinirli.
LIFESPAN_FOOD_BONUS = True
LIFESPAN_MAX_MULT = 4.0        # omur, taban omrun en fazla bu kati olabilir

# Aclik / enerji: besine yeterince uzun sure ulasamayan olur.
STARVE_TIME = 30.0             # bu kadar saniye besin bulamazsa olur (geri donus icin pay)
ENERGY_MAX = 1.0

# Cezalar - iki katmanli:
#  1) Enerji cezasi: ekstra enerji tuketir -> daha cabuk olur (hayatta kalma baskisi)
WALL_PENALTY_RATE = 0.28       # duvara/tasa carpip ilerleyemeyince saniyelik enerji cezasi
IDLE_PENALTY_RATE = 0.15       # 'bekle' (hicbir sey yapma) aksiyonu saniyelik enerji cezasi
#  2) Fitness cezasi: dogrudan basari puanini dusurur (secilim baskisi).
WALL_FIT_PENALTY = 0.10        # adim basina duvara carpma fitness cezasi (arttirildi)
IDLE_FIT_PENALTY = 0.06        # adim basina sabit durma fitness cezasi
# En dis cerceveye (harita kenari) carpma EK cezasi (kenara sikismayi onler)
BORDER_FIT_PENALTY = 0.22      # adim basina ek fitness cezasi (sadece dis cerceve)
BORDER_PENALTY_RATE = 0.35     # saniyelik ek enerji cezasi (sadece dis cerceve)
# Besin TASIRKEN yuvadan UZAKLASMA cezasi (piksel basina) -> yemi alip donmeyenler elenir
CARRY_AWAY_PENALTY_W = 0.05

# ---------------------------------------------------------------------------
# Gorus (180 derece SEKTOR tabanli, okluzyonlu)
# ---------------------------------------------------------------------------
# Karincanin onunde 180 derecelik gorus acisi acisal DILIMLERE (sektor) bolunur.
# Her dilimde o yondeki EN YAKIN nesne gorunur; arkasindaki nesneler bu yakin
# nesne tarafindan GIZLENIR (okluzyon). Boylece bir tasin arkasindaki besin
# gorunmez, ama 180 derece icindeki tum (farkli yonlerdeki) nesneler gorunur.
VISION_RANGE = 130.0            # gorus yaricapi (piksel) ~6.5 hucre
VISION_FOV = 3.141592653589793  # 180 derece (on yari daire)
N_SECTORS = 12                  # gorus acisini kac dilime bolelim
N_VIS_OBJ = 5                   # one-hot tipler: besin, tas, engel, karinca, yuva
# Her sektor: [yakinlik(0..1), besin, tas, engel, karinca, yuva] = 6
SECTOR_FEATURES = 1 + N_VIS_OBJ
VISION_INPUTS = N_SECTORS * SECTOR_FEATURES
RAY_STEP = 6.0                  # (eski raycast yardimcisi icin)

# Homing (yon butunleme): yuva yonu (sin, cos) + normalize mesafe
HOMING_INPUTS = 3

# Feromon antenleri: 3 yon (sol/orta/sag) x 2 alan (home/food) = 6
PH_SAMPLE_ANGLES = (-0.55, 0.0, 0.55)   # heading'e gore (radyan)
PH_SAMPLE_DIST = 1.5 * CELL_SIZE        # antenin onunden ornek alma mesafesi
PHEROMONE_INPUTS = len(PH_SAMPLE_ANGLES) * 2

# Besin kokusu antenleri: ayni 3 yonde besin koku gradyani = 3
ODOR_INPUTS = len(PH_SAMPLE_ANGLES)

# Ek girisler: tasiyor_mu, enerji
EXTRA_INPUTS = 2

INPUT_SIZE = (VISION_INPUTS + HOMING_INPUTS
              + PHEROMONE_INPUTS + ODOR_INPUTS + EXTRA_INPUTS)

# ---------------------------------------------------------------------------
# Feromon (karincalar antenle koklar; gercek karinca davranisinin temeli)
# ---------------------------------------------------------------------------
# Iki alan: HOME izi (besin tasimayan/aranan karincalar birakir) ve
# FOOD izi (besin tasiyan karincalar birakir -> digerleri besine ulasir).
PH_HOME = 0
PH_FOOD = 1
PH_DEPOSIT_HOME = 3.0       # adim basina home feromon birakimi
PH_DEPOSIT_FOOD = 14.0      # adim basina food feromon birakimi
# Cok besin teslim etmis (basarili) karincalar DAHA GUCLU iz birakir:
# birakim = PH_DEPOSIT_FOOD * (1 + food_delivered * PH_SUCCESS_FACTOR)
PH_SUCCESS_FACTOR = 0.5
# Besin feromonu IZ UZUNLUGU: besin aldiktan sonra bu mesafe boyunca birakilir.
PH_FOOD_TRAIL_DIST = 1500.0  # piksel (~75 hucre)
PH_HOME_EVAPORATION = 0.06   # home feromonu saniyelik buharlasma orani
PH_FOOD_EVAPORATION = 0.005  # besin feromonu saniyelik buharlasma orani (yavas)
PH_MAX = 200.0              # feromon tavani (normalize icin)
# "Super iz" (yogun besin yolu): esigin ustundeki guclu izler COK daha yavas
# buharlasir ve MOR renge doner -> sik kullanilan besin yollari kalici olur.
PH_FOOD_STRONG_THRESH = 110.0   # bu yogunlugun ustu "super iz" sayilir
PH_FOOD_STRONG_EVAP_MULT = 0.12 # super izde buharlasma carpani (cok yavas)
# Yayilim (difuzyon) izleri genisletir -> yon algisini bozar. Cok seyrek tutulur
# ki izler KESKIN kalsin ve karincalar yonu (sol/orta/sag anten) ayirt edebilsin.
PH_DIFFUSE_EVERY = 8.0      # neredeyse kapali (sadece cok nadir hafif yumusatma)

# ---------------------------------------------------------------------------
# Besin kokusu (statik koku alani)
# ---------------------------------------------------------------------------
# Feromondan farkli: koku, besin KAYNAKLARININ kendisinden yayilir ve genis
# bir gradyan olusturur. Cok-kaynakli BFS ile besine olan (duvarlardan
# dolanarak) mesafe hesaplanir; koku = 1 - mesafe/menzil. Boylece koku
# HARITA BOYUNCA yayilir ve karincalar yogunlugu tirmanip besini bulur.
ODOR_RANGE_CELLS = 25       # koku menzili (hucre). Buyut -> besin bulmak kolaylasir
                            # (uzun menzilli arama sinyali), kucult -> daha yerel.
ODOR_SAMPLE_DIST = 2.0 * CELL_SIZE  # koku ornekleme mesafesi (anten araligi)

# ---------------------------------------------------------------------------
# Besin davranisi
# ---------------------------------------------------------------------------
# Her besin kaynaginin bir MIKTAR degeri vardir (editorde girilir). Besin
# her alindiginda miktar 1 azalir; 0 olunca kaynak biter (hucre bosalir).
FOOD_DEFAULT_AMOUNT = 10        # editorde yeni besin icin varsayilan miktar
FOOD_MAX_AMOUNT = 999

# Periyodik besin: her FOOD_SPAWN_INTERVAL saniyede bir, tas/engel olmayan
# rastgele bos bir hucrede FOOD_SPAWN_AMOUNT degerinde besin olusur.
FOOD_SPAWN_INTERVAL = 60.0
FOOD_SPAWN_AMOUNT = 5

# ---------------------------------------------------------------------------
# Sinir agi:  girdi -> Dense(encoder) -> LSTM -> Dense(cikis) -> argmax
# ---------------------------------------------------------------------------
# LSTM'den ONCE bir Dense kodlayici: cok sayidaki girdiyi (86) daha kucuk,
# anlamli bir temsile sikistirir -> LSTM girdisi kuculur, genom kuculur,
# GA icin arama uzayi daralir (daha hizli evrim).
ENCODER_SIZE = 24               # LSTM oncesi Dense kodlayici boyutu
HIDDEN_SIZE = 16                # LSTM gizli katman boyutu
OUTPUT_SIZE = 5                  # 0:hicbiri 1:ileri 2:geri 3:sol 4:sag

ACTION_NONE = 0
ACTION_FORWARD = 1
ACTION_BACK = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_NAMES = {
    ACTION_NONE: "wait",
    ACTION_FORWARD: "forward",
    ACTION_BACK: "back",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}

# ---------------------------------------------------------------------------
# Genetik algoritma
# ---------------------------------------------------------------------------
INITIAL_POP = 40                # baslangic karinca sayisi
MIN_POP = 15                    # bu sayinin altina dusulurse takviye
MAX_POP = 80                    # ust sinir
# Ureme: yuvaya besin getiren HER karincadan, o karincanin genomundan
# (mutasyonla) OFFSPRING_PER_DELIVERY adet yavru dogar.
OFFSPRING_PER_DELIVERY = 5      # her teslim eden karincadan kac yavru
N_PARENTS = 3                   # (onur listesi takviyesinde) kac ebeveyn birlestirilir
MUTATION_RATE = 0.06            # gen basina mutasyon olasiligi (dusuk -> kararli rafine)
MUTATION_SCALE = 0.14           # mutasyon gauss siddeti

# Fitness takviyesi: populasyon dususte rastgele yerine TUM ZAMANLARIN EN IYI
# karincalarindan (hall of fame) ureme yapilir. Olen karincalarin yerine
# gelenler bu onur listesindeki en basarili genomlardan uretilir.
FITNESS_REINFORCE = True
HALL_OF_FAME_SIZE = 12         # tum zamanlarin en iyi N karincasi saklanir
HOF_OFFER_EVERY = 0.5          # yasayan en iyi karinca bu sikligla onur listesine sunulur
# fitness = teslim*DELIVER_W + (mesafeye gore besin bulma) + odul_sekillendirme
FITNESS_DELIVER_W = 10.0
# Besin bulma odulu YUVADAN UZAKLIGA gore olceklenir: yuvaya yakin besin az,
# uzaktaki besin cok odul verir. Bu, yuva etrafinda donen karincalar yerine
# gercek FORAGER'lari (uzaga gidip besin toplayan) secer.
FITNESS_FIND_BASE = 1.0        # her bulmada sabit kucuk odul
FITNESS_FIND_DIST_W = 0.03     # piksel basina ek odul (yuvadan uzaklik)

# Odul sekillendirme (reward shaping) - SEYREK ODUL TUZAGINI kirar:
# Karincalar tam teslimat/bulma olmadan da DOGRU YONDE ILERLEDIKCE puan alir,
# boylece evrim kademeli olarak "besine git" ve "yuvaya geri don" davranisini
# secebilir. Farming'i onlemek icin sadece YENI en iyi ilerleme odullenir.
# Yuvaya yaklasma odulu artirildi ama adim basina katkisi (~0.047) iki fitness
# cezasindan (0.06) BIRAZ DAHA AZ kalacak sekilde ayarlandi.
RETURN_REWARD_W = 0.040        # besin tasirken yuvaya yaklasma odulu (piksel basina)
# Koku takibini OGRETEN odul: bos gezerken besin kokusunu tirmandikca puan.
# Guclendirildi ki karincalar kokuyu takip etmeyi ogrensin.
FORAGE_REWARD_W = 6.0          # bos gezerken besin kokusunu tirmanma odulu (0..1 artis)

# ---------------------------------------------------------------------------
# Kayit (recording)
# ---------------------------------------------------------------------------
RECORD_DIR = "recordings"
RECORD_FPS = 30

# ---------------------------------------------------------------------------
# Dosyalar
# ---------------------------------------------------------------------------
MAP_FILE = "maps/default_map.json"
ANT_IMAGE = "ant.png"
CHECKPOINT_FILE = "sim_checkpoint.pkl"   # (eski) tekil kayit
DEMO_DIR = "demos"                       # H ile kaydedilen demolar (gecmis tutulur)

# ---------------------------------------------------------------------------
# Istatistik (zaman serisi - T tusu ile gorsellestirilir)
# ---------------------------------------------------------------------------
STATS_SAMPLE_INTERVAL = 2.0     # bu kadar sim-saniyesinde bir anlik veri kaydet
STATS_RATE_WINDOW = 10.0        # oran grafikleri icin pencere (saniye) - "her 10 sn"
STATS_EXPORT_DIR = "stats_exports"   # E tusu ile disa aktarilan grafik/CSV klasoru
