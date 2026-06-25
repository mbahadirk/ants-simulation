"""
Genel yapilandirma / sabitler.
Tum moduller buradan beslenir. Degerleri buradan ayarlayabilirsin.
"""

# ---------------------------------------------------------------------------
# Ekran / dunya
# ---------------------------------------------------------------------------
CELL_SIZE = 20                  # bir izgara hucresinin piksel boyutu
GRID_W = 160                    # yatay hucre sayisi (80'den 2x -> daha buyuk dunya)
GRID_H = 100                    # dikey hucre sayisi (50'den 2x)

WORLD_W = GRID_W * CELL_SIZE     # dunya genisligi (piksel) = 3200
WORLD_H = GRID_H * CELL_SIZE     # dunya yuksekligi (piksel) = 2000

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
GRID_LINE_COLOR = (186, 57, 147)
BG_COLOR = (72 , 111 , 56 )

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
STARVE_TIME = 40.0             # bu kadar saniye besin bulamazsa olur (30'dan artirildi)
ENERGY_MAX = 1.0

# Cezalar - iki katmanli:
#  1) Enerji cezasi: ekstra enerji tuketir -> daha cabuk olur (hayatta kalma baskisi)
#     Eski degerler cok sert oldurorduyu (duvara yasli ~8 sn'de olum) -> yumusatildi;
#     secilim baskisinin agirligi FITNESS cezalarina kaydirildi (puanla elenir, olmez).
WALL_PENALTY_RATE = 0.08       # duvara/tasa carpinca saniyelik enerji cezasi (0.28'den)
IDLE_PENALTY_RATE = 0.0        # 'bekle' aksiyonu enerji TUKETMEZ (sadece fitness cezasi)
#  2) Fitness cezasi: dogrudan basari puanini dusurur (secilim baskisi).
WALL_FIT_PENALTY = 0.10        # adim basina duvara carpma fitness cezasi
IDLE_FIT_PENALTY = 0.06        # adim basina sabit durma fitness cezasi
# Sag-sol salinim (titreme) cezasi: TEK ters donus normal duzeltme sayilir
# (cezasiz); sadece ARDISIK 2+ ters donus (gercek titreme) cezalandirilir.
JITTER_FIT_PENALTY = 0.025     # adim basina titreme cezasi (0.04'ten dusuruldu)
# KOKUSUZ ALANDA DURMA cezasi: besin TASIMAYAN karinca, hic koku olmayan
# (last_odor < esik) bir yerde ILERLEMEDEN durursa (disp ~0) ufak ceza alir.
# Amac: yakinda besin bitip koku kaybolunca karincalar oradan AYRILIP uzak
# alanlara acilsin (kokusuz olu bolgede loiter etmek caydirilir). Hareket
# eden kasif cezasiz -> sadece DURAN cezalanir.
NO_ODOR_IDLE_PENALTY = 0.04    # adim basina, kokusuz alanda durunca fitness cezasi
ODOR_DEAD_THRESH = 0.02        # bu degerin altindaki yerel koku "kokusuz" sayilir
# En dis cerceveye (harita kenari) carpma EK cezasi (kenara sikismayi onler)
BORDER_FIT_PENALTY = 0.22      # adim basina ek fitness cezasi (sadece dis cerceve)
BORDER_PENALTY_RATE = 0.10     # saniyelik ek enerji cezasi (0.35'ten dusuruldu)
# DUVAR SURUNME cezasi: karincalar (ozellikle yuva kosedeyse) dis duvara
# surunerek yol buluyor, homing kullanmiyor. Dis cerceve seridindeyken
# (band) ufak ceza -> dogal/kisa (ic) yollar tercih edilsin. YUVA cevresi
# MUAF (yuva kosede ise yaklasirken duvara yakin olmak normal).
BORDER_HUG_CELLS = 2            # dis cerceveye bu kadar hucre yakinlik "serit"
BORDER_HUG_PENALTY = 0.06      # adim basina seritte olma cezasi
BORDER_HUG_NEST_EXEMPT_CELLS = 6  # yuvaya bu kadar yakin -> ceza yok (kose yaklasimi)
# Besin TASIRKEN yuvadan UZAKLASMA cezasi (piksel basina) -> yemi alip donmeyenler elenir
# KAPATILDI: tasi yan karincalarin tas/engel etrafindan dolanmasi gerektiginde
# (gecici olarak yuvadan uzaklasmak sart) titreyip kilitlenmelerine sebep oluyordu.
CARRY_AWAY_PENALTY_W = 0.0

# ---------------------------------------------------------------------------
# Gorus (180 derece SEKTOR tabanli, okluzyonlu)
# ---------------------------------------------------------------------------
# Karincanin onunde 180 derecelik gorus acisi acisal DILIMLERE (sektor) bolunur.
# Her dilimde o yondeki EN YAKIN nesne gorunur; arkasindaki nesneler bu yakin
# nesne tarafindan GIZLENIR (okluzyon). Boylece bir tasin arkasindaki besin
# gorunmez, ama 180 derece icindeki tum (farkli yonlerdeki) nesneler gorunur.
VISION_RANGE = 30.0             # gorus yaricapi (piksel) ~1.5 hucre - karincalar
                                 # kor sayilir ama tasi vaktinde fark edip
                                 # etrafindan dolanabilmeli (13'ten artirildi;
                                 # cok kisa menzil tasi gec gorup titremeye yol aciyordu)
VISION_FOV = 3.141592653589793  # 180 derece (on yari daire)
N_SECTORS = 8                   # gorus acisini kac dilime bolelim (12'den dusuruldu)
N_VIS_OBJ = 2                   # one-hot tipler: besin, engelli(tas+engel)
# Her sektor: [yakinlik(0..1), besin, engelli] = 3
# Yuva gorusten cikti (homing girdisi zaten karsilior); tas+engel "engelli" olarak birlesti.
# Diger karincalar da gorusten cikarildi: konum gurultusu yaratiyordu, ayrica
# karinca-karinca etkilesimi feromon/koku uzerinden zaten dolayli var.
SECTOR_FEATURES = 1 + N_VIS_OBJ
VISION_INPUTS = N_SECTORS * SECTOR_FEATURES
RAY_STEP = 6.0                  # (eski raycast yardimcisi icin)

# Homing (yon butunleme): yuva yonu (sin, cos) + normalize mesafe
HOMING_INPUTS = 3

# Kimyasal alan algilama (koku + feromon): ESKI yontem 3 ham nokta ornekliyordu;
# alan neredeyse her yerde doygun oldugu icin sol/orta/sag farki ~0 -> takip
# edilebilir yon yoktu. YENI yontem: alanin yerel GRADYANINI (yokus-yukari yonu)
# hesaplar ve ag'a heading'e gore (sin, cos) + buyukluk olarak verir. Boylece
# alan seviyesi ne olursa olsun NET bir yonlendirme sinyali olusur.
CHEM_GRAD_STEP = CELL_SIZE       # gradyan merkezi-fark adimi (piksel) ~1 hucre
CHEM_GRAD_NORM = 120.0           # gradyan buyuklugu normalizasyon carpani (~0..1 araligi)

# Besin kokusu: yokus-yukari yon (sin, cos) + buyukluk + yerel deger = 4
ODOR_INPUTS = 4
# TEK feromon: yokus-yukari yon (sin, cos) + yerel deger = 3
PHEROMONE_INPUTS = 3

# Ek girisler: tasiyor_mu, enerji
EXTRA_INPUTS = 2

INPUT_SIZE = (VISION_INPUTS + HOMING_INPUTS
              + PHEROMONE_INPUTS + ODOR_INPUTS + EXTRA_INPUTS)

# ---------------------------------------------------------------------------
# Feromon (TEK alan) - karincalar antenle koklar
# ---------------------------------------------------------------------------
# TEK bir feromon izi: tum karincalar ayni alana birakir; uzerinden gecildikce
# GUCLENIR (birikim, PH_MAX'a kadar). Buharlasma cok yavas -> izler kalici.
# Birakim ZAYIF tutulur ki dusuk-trafik bolgeler boyanmasin; sadece sik
# kullanilan yollar zamanla guclensin. Besin TASIYAN karinca 10x daha fazla birakir.
PH_DEPOSIT_BASE = 0.08      # bos gezen karincanin KAT EDILEN PIKSEL basina birakimi.
                            # Hareketle orantili -> sabit/donerken/titrerken birakmaz.
PH_CARRY_MULT = 10.0        # besin TASIYAN karinca bu kat daha fazla birakir
PH_EVAPORATION = 0.02     # saniyelik buharlasma (0.0003'ten 5x hizlandirildi)
PH_MAX = 200.0              # feromon tavani (normalize + birikim siniri)
# Goruntu: bu yogunlugun ustu "guclu iz" sayilir (parlak/mor cizilir)
PH_STRONG_THRESH = 140.0

# ---------------------------------------------------------------------------
# Besin kokusu (statik koku alani)
# ---------------------------------------------------------------------------
# Feromondan farkli: koku, besin KAYNAKLARININ kendisinden yayilir ve genis
# bir gradyan olusturur. Cok-kaynakli BFS ile besine olan (duvarlardan
# dolanarak) mesafe hesaplanir; koku = 1 - mesafe/menzil. Boylece koku
# HARITA BOYUNCA yayilir ve karincalar yogunlugu tirmanip besini bulur.
# Menzil 25 -> harita capinda neredeyse global koku -> gradyan duzlesir, yon yok.
# 12'ye dusuruldu: koku YEREL ve YONLU olur; karinca yokus-yukari gercekten
# ilerlemek zorunda kalir (yuva yaninda otomatik max koku farm'lanamaz).
ODOR_RANGE_CELLS = 48       # koku menzili (hucre): 48 hucre x 20px = 960px yaricap
                            # (32'den artirildi: uzak besine ulasmak icin ilk yolu
                            # acmak -> koku daha genis yayilsin, karincalar tirmanarak
                            # menzil-disi alanlara da yonelebilsin. Karisma riskine
                            # ragmen "uzaga acilma" oncelikli.)
ODOR_SAMPLE_DIST = 2.0 * CELL_SIZE  # (eski uyumluluk; artik gradyan kullaniliyor)

# ---------------------------------------------------------------------------
# Besin davranisi
# ---------------------------------------------------------------------------
# Her besin kaynaginin bir MIKTAR degeri vardir (editorde girilir). Besin
# her alindiginda miktar 1 azalir; 0 olunca kaynak biter (hucre bosalir).
FOOD_DEFAULT_AMOUNT = 10        # editorde yeni besin icin varsayilan miktar
FOOD_MAX_AMOUNT = 999

# Periyodik besin: her FOOD_SPAWN_INTERVAL saniyede bir, tas/engel olmayan
# rastgele bos bir hucrede FOOD_SPAWN_AMOUNT degerinde besin olusur.
# Ortami SABITLEMEK icin (gorev durup dururken ogrenilebilsin): daha sik, daha
# bol ve yuvadan UZAKTA spawn -> koloni surekli uzak forage hedefi bulur.
FOOD_SPAWN_INTERVAL = 25.0      # periyodik besin araligi
FOOD_SPAWN_AMOUNT = 12          # her spawn'da birim sayisi (5'ten artirildi)
FOOD_SPAWN_MIN_NEST_CELLS = 22  # spawn yuvadan en az bu kadar hucre uzakta olur (12'den artirildi)

# ---------------------------------------------------------------------------
# Sinir agi:  girdi -> Dense(encoder) -> LSTM -> Dense(cikis) -> argmax
# ---------------------------------------------------------------------------
# LSTM'den ONCE bir Dense kodlayici: cok sayidaki girdiyi (86) daha kucuk,
# anlamli bir temsile sikistirir -> LSTM girdisi kuculur, genom kuculur,
# GA icin arama uzayi daralir (daha hizli evrim).
# Beyin mimarisi: "mlp" (saf feedforward, ONERILEN) | "lstm" (recurrent, hafizali)
# Gorev neredeyse REAKTIF: homing/koku/feromon gradyanlari hazir verildigi icin
# karar anlik girdilerden cozulebilir -> hafiza (recurrence) sart degil. MLP daha
# kucuk genom + puruzsuz fitness manzarasi -> neuroevrimde daha hizli/kararli ogrenme.
# LSTM zamansal kredi atamasini zorlastirip arama uzayini sismelirir.
BRAIN_ARCH = "mlp"
MLP_HIDDEN = 32                 # MLP gizli katman boyutu (20'den buyutuldu; daha fazla odul
                                 # sekillendirme + girdi karisikligi icin kapasite artirildi)

ENCODER_SIZE = 16               # (LSTM modunda) LSTM oncesi Dense kodlayici boyutu
HIDDEN_SIZE = 12                # (LSTM modunda) LSTM gizli katman boyutu
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
INITIAL_POP = 100               # baslangic karinca sayisi (40'tan; buyuyen dunya icin)
MIN_POP = 40                    # bu sayinin altina dusulurse takviye
MAX_POP = 200                   # YUMUSAK ust sinir: normal teslimler bunun ustunde dogurmaz
HARD_MAX_POP = 250              # KATI ust sinir: elit teslimler buraya kadar zorla dogurabilir;
                                # bu sayiya ulasilirsa en ESKI nesilden karincalar oldurulur
# Ureme: yuvaya besin getiren HER karincadan, o karincanin genomundan
# (mutasyonla) yavru dogar. NORMAL karinca 1 yavru; o anki EN IYI (yasayan
# en yuksek fitness) karinca teslim ederse ELITE_OFFSPRING kadar yavru dogar
# ve YUMUSAK limit (MAX_POP) dolu olsa bile zorla dogurulur (en iyi genom
# ureyemeden kaybolmasin diye).
OFFSPRING_PER_DELIVERY = 1      # normal teslim eden karincadan kac yavru
ELITE_OFFSPRING = 2             # yasayan en iyi karinca teslim ederse kac yavru (limit asar)
N_PARENTS = 2                   # (onur listesi takviyesinde) kac ebeveyn birlestirilir
# KESIFCI moda geri donuldu: reward fonksiyonu cok degisti, populasyon eski
# (dusuk cesitlilik) ayarlarla erken yakinsamis ve fitness duz cizgi olmustu.
# Sifirdan baslayan yeni egitim icin daha yuksek mutasyon/cesitlilik.
MUTATION_RATE = 0.08             # kesif: gen basina mutasyon olasiligi
MUTATION_SCALE = 0.18            # kesif: daha buyuk gauss gurultusu
# _reinforce'ta tamamen rastgele genom enjeksiyon orani (cesitlilik vs kararlilik)
REINFORCE_RANDOM_FRAC = 0.12     # kesif: daha fazla rastgele giris -> cesitlilik
# _on_delivery'de HOF uyesiyle crossover olasiligi (yavru cesitliligi)
DELIVERY_CROSSOVER_FRAC = 0.35   # kesif: daha fazla caprazlama -> cesitlilik

# Fitness takviyesi: populasyon dususte rastgele yerine TUM ZAMANLARIN EN IYI
# karincalarindan (hall of fame) ureme yapilir. Olen karincalarin yerine
# gelenler bu onur listesindeki en basarili genomlardan uretilir.
FITNESS_REINFORCE = True
HALL_OF_FAME_SIZE = 12         # tum zamanlarin en iyi N karincasi saklanir
HOF_OFFER_EVERY = 0.5          # yasayan en iyi karinca bu sikligla onur listesine sunulur
# fitness = teslim*(DELIVER_W + mesafe) + (mesafeye gore bulma) + odul_sekil.
# Teslim odulu de mesafeye gore olceklenir: yuva yaninda topla-getir (kisa tur)
# uzun-mesafe forage ile AYNI puani vermesin -> gercek forager'lar HOF'a girer.
FITNESS_DELIVER_W = 20.0        # teslim basina sabit odul (azaltildi; agirlik mesafeye kaydi)
FITNESS_DELIVER_DIST_W = 0.06  # teslim mesafesi odulu: uzaktan getirmek cok daha degerli (2x artirildi)
# Besin bulma odulu YUVADAN UZAKLIGA gore olceklenir. Sabit taban KALDIRILDI
# (1.0 -> 0): yuvaya cok yakin besin bulmak neredeyse 0 puan -> nest-circling
# stratejisi odulsuz kalir. Sadece uzaga giden forage odullenir.
FITNESS_FIND_BASE = 0.0        # sabit taban kaldirildi (yakin besin ~0 puan)
FITNESS_FIND_DIST_W = 0.06     # piksel basina odul (yuvadan uzaklik) (artirildi)

# YOL VERIMLILIGI odulu: teslimat aninda duz_mesafe/kat_edilen_mesafe orani
# (1.0=mukemmel duz yol) ile ekstra bonus. Mevcut teslimat odulune EK olarak
# verilir (onu degistirmez) -> kisa/dogrudan yol bulan karincalar fazladan
# odullenir, dolambacli yol alanlar daha az bonus alir.
PATH_EFFICIENCY_W = 3.0        # teslimat basina, verimlilik oraniyla olcekli bonus

# Odul sekillendirme (reward shaping) - SEYREK ODUL TUZAGINI kirar:
RETURN_REWARD_W = 0.040        # besin tasirken yuvaya yaklasma odulu (piksel basina)
# YUVAYA BAKMA odulu: tasirken homing acisi (nrel) dustukce (yuva one yakin)
# odul artar. KAT EDILEN MESAFEYLE (disp) CARPILIR.
# DIKKAT: bu odul her hareket adiminda SINIRSIZ birikir (ratchet/cap yok).
# 0.05'te uzun omurlu bir karinca yuvaya/besine bakarak "dans ederek" binlerce
# fitness toplayip teslimat odulunu (~80) golgeliyordu -> secilim baskisi
# yok oluyordu. 10x DUSURULDU (sadece hafif yon durtusu, farming baskin degil).
FACE_NEST_REWARD_W = 0.005     # piksel basina odul (hizalanma * disp) (0.05'ten 10x dusuruldu)
EXPLORE_REWARD_W = 0.012       # bos gezerken yuvadan UZAKLASTIKCA odul (piksel basina ratchet)
# Koku takibini OGRETEN odul: bos gezerken besin kokusunu tirmandikca puan.
FORAGE_REWARD_W = 6.0          # bos gezerken besin kokusunu tirmanma odulu (0..1 artis)
# IZ TAKIP odulu (YENI): bos gezerken food-feromon gradyaninin YOKUS-YUKARI
# yonunde hareket etmek odullenir -> "izi takip et" davranisina DOGRUDAN
# secilim baskisi. Iz yoksa (basta) odul 0; basarili karincalar iz biraktikca
# digerleri izi takip etmeyi ogrenir (gercek karinca pozitif geri beslemesi).
TRAIL_FOLLOW_W = 9.0           # food-feromon gradyani yonunde hareket odulu
# GORUNEN besine YONELME odulu: bos gezerken gorus alaninda besin varsa,
# ona dogru ACISAL HIZALANARAK hareket etmek odullenir (FACE_NEST_REWARD_W
# ile ayni desen). Saf mesafe-azalma yeterli degildi -> dolanarak da mesafe
# zaman zaman azalabiliyordu, bu da "besin etrafinda dolanma" davranisina
# yol aciyordu. KAT EDILEN MESAFEYLE (disp) CARPILIR -> sabit durup besine
# bakarak farm edilemez.
# DIKKAT: FACE_NEST_REWARD_W ile ayni sinirsiz-birikim sorunu var (ratchet/cap
# yok). 0.05'te "besin etrafinda dans ederek" fitness farm'lanabiliyordu ->
# 10x DUSURULDU. Asil yon sinyali ratchet odullerden (FORAGE, EXPLORE) gelir.
FOOD_FACE_REWARD_W = 0.005     # piksel basina odul (hizalanma * disp) (0.05'ten 10x dusuruldu)

# ---------------------------------------------------------------------------
# Kayit (recording)
# ---------------------------------------------------------------------------
RECORD_DIR = "recordings"
RECORD_FPS = 30

# ---------------------------------------------------------------------------
# Dosyalar
# ---------------------------------------------------------------------------
MAP_FILE = "maps/default_map.json"
MAPS_DIR = "maps"                        # tum egitim haritalari (harita secimi/rotasyonu)
ANT_IMAGE = "ant.png"
CHECKPOINT_FILE = "sim_checkpoint.pkl"   # (eski) tekil kayit
DEMO_DIR = "demos"                       # H ile kaydedilen demolar (gecmis tutulur)

# ---------------------------------------------------------------------------
# Model bankasi (kalici hall-of-fame arsivi) + coklu harita egitimi
# ---------------------------------------------------------------------------
# En iyi genomlar diske kaydedilir; her yeni simulasyon (hangi haritada olursa
# olsun) baslangic populasyonunun bir kismini bankadan tohumlar. Farkli
# haritalarda egitilen modeller boylelikle kosudan kosuya GENELLESIR/guclenir.
MODEL_BANK_FILE = "models/model_bank.pkl"
MODEL_BANK_SIZE = 24        # bankada tutulan en iyi genom sayisi
BANK_SEED_FRAC = 0.5        # baslangic popun bu orani bankadan (mutasyonla) tohumlanir
BANK_MERGE_EVERY = 60.0     # bu kadar sim-saniyede bir hall -> banka birlestir + kaydet

# ---------------------------------------------------------------------------
# Istatistik (zaman serisi - T tusu ile gorsellestirilir)
# ---------------------------------------------------------------------------
STATS_SAMPLE_INTERVAL = 2.0     # bu kadar sim-saniyesinde bir anlik veri kaydet
STATS_RATE_WINDOW = 10.0        # oran grafikleri icin pencere (saniye) - "her 10 sn"
STATS_EXPORT_DIR = "stats_exports"   # E tusu ile disa aktarilan grafik/CSV klasoru
