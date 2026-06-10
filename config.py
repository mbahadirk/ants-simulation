"""
Genel yapilandirma / sabitler.
Tum moduller buradan beslenir. Degerleri buradan ayarlayabilirsin.
"""

# ---------------------------------------------------------------------------
# Ekran / dunya
# ---------------------------------------------------------------------------
CELL_SIZE = 20                  # bir izgara hucresinin piksel boyutu
GRID_W = 64                     # yatay hucre sayisi
GRID_H = 40                     # dikey hucre sayisi

WORLD_W = GRID_W * CELL_SIZE     # dunya genisligi (piksel)
WORLD_H = GRID_H * CELL_SIZE     # dunya yuksekligi (piksel)

SCREEN_W = 1280
SCREEN_H = 800
FPS = 60

# ---------------------------------------------------------------------------
# Hucre tipleri (map editor + dunya bunlari kullanir)
# ---------------------------------------------------------------------------
EMPTY = 0
FOOD = 1
STONE = 2        # tas: gecilmez, "tas" olarak algilanir
OBSTACLE = 3     # engel/duvar: gecilmez, "engel" olarak algilanir
NEST = 4         # yuva

TILE_NAMES = {
    EMPTY: "Bos",
    FOOD: "Besin",
    STONE: "Tas",
    OBSTACLE: "Engel",
    NEST: "Yuva",
}

# Hucre renkleri
COLORS = {
    EMPTY: (24, 22, 20),
    FOOD: (90, 200, 90),
    STONE: (130, 130, 140),
    OBSTACLE: (70, 55, 45),
    NEST: (200, 150, 60),
}
GRID_LINE_COLOR = (40, 38, 34)
BG_COLOR = (18, 16, 14)

# ---------------------------------------------------------------------------
# Karinca / hareket
# ---------------------------------------------------------------------------
ANT_SIZE = 14                   # cizim boyutu (piksel)
MOVE_SPEED = 70.0               # ileri hiz (piksel / saniye)
BACK_SPEED = 35.0               # geri hiz
TURN_SPEED = 3.4                # donus hizi (radyan / saniye)

# Yasam suresi (saniye) -> her karinca icin bu araliktan rastgele secilir
LIFESPAN_MIN = 45.0
LIFESPAN_MAX = 60.0

# Aclik / enerji: besine yeterince uzun sure ulasamayan olur.
STARVE_TIME = 22.0              # bu kadar saniye besin bulamazsa olur
ENERGY_MAX = 1.0

# ---------------------------------------------------------------------------
# Gorus (anten / isin sensorleri)
# ---------------------------------------------------------------------------
N_RAYS = 7                      # anten isin sayisi
VISION_RANGE = 70.0             # kisa gorus mesafesi (piksel) ~3.5 hucre
VISION_FOV = 2.4                # gorus acisi (radyan) ~ 137 derece
RAY_STEP = 6.0                  # isin ornekleme adimi (piksel)

# Sensorun algiladigi nesne tipleri (one-hot) -> food, stone, obstacle, ant, nest
N_OBJ_TYPES = 5
# Her isin: [mesafe_normalize, food, stone, obstacle, ant, nest] = 6 deger
RAY_FEATURES = 1 + N_OBJ_TYPES

# Homing (yon butunleme): yuva yonu (sin, cos) + normalize mesafe
HOMING_INPUTS = 3

# Feromon antenleri: 3 yon (sol/orta/sag) x 2 alan (home/food) = 6
PH_SAMPLE_ANGLES = (-0.55, 0.0, 0.55)   # heading'e gore (radyan)
PH_SAMPLE_DIST = 1.5 * CELL_SIZE        # antenin onunden ornek alma mesafesi
PHEROMONE_INPUTS = len(PH_SAMPLE_ANGLES) * 2

# Ek girisler: tasiyor_mu, enerji
EXTRA_INPUTS = 2

INPUT_SIZE = (N_RAYS * RAY_FEATURES) + HOMING_INPUTS + PHEROMONE_INPUTS + EXTRA_INPUTS

# ---------------------------------------------------------------------------
# Feromon (karincalar antenle koklar; gercek karinca davranisinin temeli)
# ---------------------------------------------------------------------------
# Iki alan: HOME izi (besin tasimayan/aranan karincalar birakir) ve
# FOOD izi (besin tasiyan karincalar birakir -> digerleri besine ulasir).
PH_HOME = 0
PH_FOOD = 1
PH_DEPOSIT_HOME = 6.0       # adim basina home feromon birakimi
PH_DEPOSIT_FOOD = 16.0      # adim basina food feromon birakimi (daha guclu)
PH_EVAPORATION = 0.10       # saniyede buharlasma orani
PH_MAX = 200.0              # feromon tavani (normalize icin)
PH_DIFFUSE_EVERY = 0.5      # bu kadar saniyede bir hafif yayilim

# ---------------------------------------------------------------------------
# Sinir agi (LSTM + feedforward)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 16
OUTPUT_SIZE = 5                  # 0:hicbiri 1:ileri 2:geri 3:sol 4:sag

ACTION_NONE = 0
ACTION_FORWARD = 1
ACTION_BACK = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_NAMES = {
    ACTION_NONE: "bekle",
    ACTION_FORWARD: "ileri",
    ACTION_BACK: "geri",
    ACTION_LEFT: "sol",
    ACTION_RIGHT: "sag",
}

# ---------------------------------------------------------------------------
# Genetik algoritma
# ---------------------------------------------------------------------------
INITIAL_POP = 30                # baslangic karinca sayisi
MIN_POP = 12                    # bu sayinin altina dusulurse rastgele takviye
MAX_POP = 70                    # ust sinir
FOOD_PER_BIRTH = 3              # her 3 besin teslimati -> 1 yavru
N_PARENTS = 3                   # yavru kac ebeveynin genetiginden uretilir
MUTATION_RATE = 0.08            # gen basina mutasyon olasiligi
MUTATION_SCALE = 0.30           # mutasyon gauss siddeti

# Fitness takviyesi: populasyon dususte rastgele yerine en iyi hayatta
# kalanlardan ureme yapilir (bootstrap'i hizlandirir).
FITNESS_REINFORCE = True
TOP_SURVIVORS = 5               # takviye uremesinde kullanilan en iyi N hayatta kalan
# fitness = teslim*DELIVER_W + bulunan_besin*FIND_W
FITNESS_DELIVER_W = 10.0
FITNESS_FIND_W = 3.0

# Yuvada baslangicta uretilen besin sayisi yok; besinler haritadan gelir.
NEST_REGROW_FOOD = False        # True olursa besin yendikce yenilenir (kapali)

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
