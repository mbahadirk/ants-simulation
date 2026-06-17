"""
Training statistics visualization (time-series line charts).

Renders the simulation's recorded history (population, births, deaths,
deliveries, best fitness over time) as line charts. Used both live (T key
during a run) and for saved demos (Stats button in Demo History).
"""

import os
import csv
import time

import pygame

import config as C

_BG = (16, 18, 22)
_PANEL = (26, 30, 36)
_GRID = (45, 50, 58)
_AXIS = (90, 96, 105)
_TEXT = (220, 224, 230)
_MUTED = (150, 155, 162)


def _nice_max(v):
    if v <= 0:
        return 1.0
    return v * 1.15


def _rate(history, key, window_s=None):
    """Kumulatif 'key' degerinden pencere (window_s) basina ORAN hesaplar.
    Donus: (xs, rates) - rates[i] = son window_s saniyede olan miktar."""
    if window_s is None:
        window_s = C.STATS_RATE_WINDOW
    back = max(1, round(window_s / C.STATS_SAMPLE_INTERVAL))
    xs = [s["t"] for s in history]
    rates = []
    for i in range(len(history)):
        j = max(0, i - back)
        rates.append(history[i][key] - history[j][key])
    return xs, rates


def _smooth(ys, k=5):
    """Hareketli ortalama (merkezli pencere) -> ani sicramalari yumusatir
    ki egilim okunabilir olsun ('grafikleri normalize et')."""
    n = len(ys)
    if n < 3:
        return ys
    half = k // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def _draw_chart(surf, rect, font, small, title, xs, series):
    """series: list of (ys, color, label). xs: ortak x degerleri (zaman)."""
    x, y, w, h = rect
    pygame.draw.rect(surf, _PANEL, rect, border_radius=8)
    surf.blit(small.render(title, True, _TEXT), (x + 10, y + 6))

    pad_l, pad_r, pad_t, pad_b = 44, 12, 28, 22
    px, py = x + pad_l, y + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    if pw <= 2 or ph <= 2 or not xs or len(xs) < 2:
        surf.blit(small.render("not enough data", True, _MUTED), (x + 12, y + h // 2))
        return

    xmax = max(xs) or 1.0
    # y aralizi: NEGATIF degerleri de destekle (ymin..ymax). Boylece negatif
    # serilier (orn. DQN avg-return) grafigin altina tasmaz.
    all_vals = [v for ys, _, _ in series for v in ys] or [0.0]
    raw_max = max(all_vals)
    raw_min = min(all_vals)
    ymax = _nice_max(raw_max) if raw_max > 0 else (raw_max * 0.85 if raw_max < 0 else 1.0)
    ymin = raw_min * 1.15 if raw_min < 0 else 0.0
    span = (ymax - ymin) or 1.0

    def to_px(t, v):
        sx = px + int(pw * (t / xmax))
        sy = py + ph - int(ph * (v - ymin) / span)
        return sx, sy

    # eksenler + yatay izgara (ymin..ymax)
    for i in range(5):
        gy = py + ph - int(ph * i / 4)
        pygame.draw.line(surf, _GRID, (px, gy), (px + pw, gy), 1)
        val = ymin + span * i / 4
        lbl = small.render(f"{val:.0f}" if span >= 4 else f"{val:.1f}", True, _MUTED)
        surf.blit(lbl, (x + 6, gy - 8))
    pygame.draw.line(surf, _AXIS, (px, py), (px, py + ph), 1)
    pygame.draw.line(surf, _AXIS, (px, py + ph), (px + pw, py + ph), 1)
    # sifir cizgisi (negatif/pozitif sinir) belirgin cizilir
    if ymin < 0 < ymax:
        zy = py + ph - int(ph * (0.0 - ymin) / span)
        pygame.draw.line(surf, _AXIS, (px, zy), (px + pw, zy), 1)
    # x ekseni etiketleri (0 ve max sure)
    surf.blit(small.render("0s", True, _MUTED), (px - 4, py + ph + 4))
    tmax = small.render(f"{xmax:.0f}s", True, _MUTED)
    surf.blit(tmax, (px + pw - tmax.get_width(), py + ph + 4))

    # cizgiler
    for ys, color, _ in series:
        if len(ys) < 2:
            continue
        pts = [to_px(xs[i], ys[i]) for i in range(len(ys))]
        pygame.draw.lines(surf, color, False, pts, 2)

    # legend
    lx = px + 6
    ly = py + 2
    for ys, color, label in series:
        pygame.draw.rect(surf, color, (lx, ly + 3, 12, 6))
        t = small.render(label, True, _TEXT)
        surf.blit(t, (lx + 16, ly))
        lx += 16 + t.get_width() + 16


def render_stats(surf, history, title, font, big, small):
    surf.fill(_BG)
    surf.blit(big.render(title, True, (235, 200, 110)), (24, 18))

    if not history:
        surf.blit(font.render("No statistics recorded yet.", True, _MUTED), (24, 80))
        surf.blit(small.render("ESC / T / Enter: close", True, _MUTED),
                  (24, C.SCREEN_H - 30))
        return

    xs = [s["t"] for s in history]
    last = history[-1]
    summary = (f"samples: {len(history)}   time: {int(last['t'])}s   "
               f"generation: {last['gen']}   pop: {last['pop']}   "
               f"births: {last['births']}   deaths: {last['deaths']}   "
               f"delivered: {last['delivered']}   best fitness: {last['hof_best']:.1f}")
    surf.blit(font.render(summary, True, _TEXT), (24, 58))

    # 2x2 grafik izgarasi
    m = 24
    top = 92
    gw = (C.SCREEN_W - 3 * m) // 2
    gh = (C.SCREEN_H - top - 2 * m - 24) // 2
    r1 = (m, top, gw, gh)
    r2 = (m * 2 + gw, top, gw, gh)
    r3 = (m, top + gh + m, gw, gh)
    r4 = (m * 2 + gw, top + gh + m, gw, gh)

    win = int(C.STATS_RATE_WINDOW)
    _, br = _rate(history, "births")
    _, dr = _rate(history, "deaths")
    _, fr = _rate(history, "delivered")
    br, dr, fr = _smooth(br), _smooth(dr), _smooth(fr)   # gurultulu oranlari yumusat

    _draw_chart(surf, r1, font, small, f"Births / Deaths per {win}s", xs,
                [(br, (90, 220, 110), "births"),
                 (dr, (230, 90, 90), "deaths")])
    _draw_chart(surf, r2, font, small, "Births vs Deaths (cumulative)", xs,
                [([s["births"] for s in history], (90, 220, 110), "births"),
                 ([s["deaths"] for s in history], (230, 90, 90), "deaths")])
    _draw_chart(surf, r3, font, small, f"Food delivered per {win}s", xs,
                [(fr, (240, 200, 80), "delivered")])
    _draw_chart(surf, r4, font, small, "Best fitness (hall of fame)", xs,
                [([s["hof_best"] for s in history], (240, 150, 80), "best fitness")])

    surf.blit(small.render("E: export (PNG + CSV)    ESC / T / Enter: close", True, _MUTED),
              (24, C.SCREEN_H - 30))


def export_stats(screen, history):
    """Mevcut grafik ekranini PNG, veriyi CSV olarak disa aktarir. Yol doner."""
    os.makedirs(C.STATS_EXPORT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    png = os.path.join(C.STATS_EXPORT_DIR, f"stats_{stamp}.png")
    csvp = os.path.join(C.STATS_EXPORT_DIR, f"stats_{stamp}.csv")
    pygame.image.save(screen, png)
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "pop", "births", "deaths", "delivered", "hof_best", "gen"])
        for s in history:
            w.writerow([s["t"], s["pop"], s["births"], s["deaths"],
                        s["delivered"], s["hof_best"], s["gen"]])
    print(f"[STATS] exported -> {png} , {csvp}")
    return png


def show_stats_screen(screen, history, title="Training Statistics"):
    """Modal istatistik ekrani; ESC/T/Enter ile kapanir, E ile disa aktarir."""
    font = pygame.font.SysFont("consolas", 16)
    big = pygame.font.SysFont("consolas", 30, bold=True)
    small = pygame.font.SysFont("consolas", 13)
    clock = pygame.time.Clock()
    msg = ""
    msg_t = 0.0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_t, pygame.K_RETURN):
                    return "close"
                if event.key == pygame.K_e and history:
                    try:
                        path = export_stats(screen, history)
                        msg = f"Exported -> {os.path.basename(path)} (+ .csv)"
                    except Exception as e:
                        msg = f"Export error: {e}"
                    msg_t = 2.5
            elif event.type == pygame.MOUSEBUTTONDOWN:
                return "close"
        render_stats(screen, history, title, font, big, small)
        if msg_t > 0:
            msg_t -= 1.0 / 30.0
            img = big.render(msg, True, (255, 240, 160))
            bg = pygame.Surface((img.get_width() + 24, img.get_height() + 14),
                                pygame.SRCALPHA)
            bg.fill((20, 30, 20, 230))
            screen.blit(bg, (C.SCREEN_W // 2 - img.get_width() // 2 - 12,
                             C.SCREEN_H - 80))
            screen.blit(img, (C.SCREEN_W // 2 - img.get_width() // 2, C.SCREEN_H - 73))
        pygame.display.flip()
        clock.tick(30)
