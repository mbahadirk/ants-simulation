"""
Ant neuroevolution simulation - entry point.

Run:  python main.py

Simulation keys:
  D       : debug mode (vision fan + selected ant panel)
  Z       : zoom level
  O / P   : simulation speed - / +
  K / L   : ant lifespan - / + (15s)
  T       : show training statistics (charts)
  H       : save a demo (kept in Demo History, never overwritten)
  S       : start / stop screen recording (recordings/)
  Space   : pause / resume
  Arrows  : pan the map
  Click   : (debug) select & follow an ant
  R       : reset camera / stop following
  ESC     : back to menu
"""

import os
import json
import time
import glob

import pygame

import config as C
from world import World, make_default_world, make_random_world
from simulation import Simulation
from model_bank import ModelBank
from camera import Camera
from renderer import Renderer, SettingsPanel
from recorder import Recorder
from map_editor import MapEditor
from stats_view import show_stats_screen

PAN_SPEED = 600.0  # pixel/s (screen space)


# ---------------------------------------------------------------------------
# Demo history (H key saves timestamped demos; never overwritten)
# ---------------------------------------------------------------------------
def save_demo(sim):
    """Saves the simulation as a new timestamped demo (+ JSON summary)."""
    os.makedirs(C.DEMO_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(C.DEMO_DIR, f"demo_{stamp}")
    sim.save(base + ".pkl")
    s = sim.stats()
    summary = {
        "created": stamp,
        "time": int(s["time"]),
        "generation": s["generation"],
        "delivered": s["delivered"],
        "births": s["births"],
        "deaths": s["deaths"],
        "hof_best": round(s["hof_best"], 2),
        "history": sim.history,
    }
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f)
    return base + ".pkl"


def list_demos():
    """Returns demo summaries (newest first): list of (pkl_path, summary_dict)."""
    out = []
    for jpath in glob.glob(os.path.join(C.DEMO_DIR, "demo_*.json")):
        pkl = jpath[:-5] + ".pkl"
        if not os.path.exists(pkl):
            continue
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                out.append((pkl, json.load(f)))
        except Exception:
            continue
    out.sort(key=lambda t: t[1].get("created", ""), reverse=True)
    return out


def load_world(map_path=None):
    path = map_path or C.MAP_FILE
    if os.path.exists(path):
        try:
            return World.load(path)
        except Exception as e:
            print(f"[map] yuklenemedi ({e}), varsayilan harita kullaniliyor")
    w = make_default_world()
    try:
        w.save(C.MAP_FILE)  # editorde duzenlenebilsin diye varsayilani yaz
    except Exception:
        pass
    return w


def list_maps():
    """maps/ klasorundeki tum haritalar (alfabetik). En az MAP_FILE bulunur."""
    paths = sorted(glob.glob(os.path.join(C.MAPS_DIR, "*.json")))
    if not paths and os.path.exists(C.MAP_FILE):
        paths = [C.MAP_FILE]
    return paths


def create_random_map():
    """Yeni rastgele egitim haritasi uretip maps/ altina kaydeder; yolunu doner."""
    os.makedirs(C.MAPS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.MAPS_DIR, f"random_{stamp}.json")
    make_random_world().save(path)
    print(f"[map] rastgele harita olusturuldu: {path}")
    return path


# ===========================================================================
# Menu
# ===========================================================================
def run_menu(screen):
    font_big = pygame.font.SysFont("consolas", 46, bold=True)
    font = pygame.font.SysFont("consolas", 24)
    small = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    cx = C.SCREEN_W // 2
    has_demos = len(list_demos()) > 0
    buttons = [("Start Simulation (new)", "sim")]
    if has_demos:
        buttons.append(("Demo History", "demos"))
    buttons += [("Select Map", "maps"), ("Map Editor", "editor"), ("Quit", "quit")]

    rects = []
    y = 300
    for label, action in buttons:
        r = pygame.Rect(cx - 200, y, 400, 56)
        rects.append((r, action, label))
        y += 76

    while True:
        clock.tick(C.FPS)
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key in (pygame.K_RETURN, pygame.K_1):
                    return "sim"
                elif event.key == pygame.K_2 and has_demos:
                    return "demos"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for r, action, _ in rects:
                    if r.collidepoint(event.pos):
                        return action

        screen.fill((20, 18, 16))
        title = font_big.render("ANT NEUROEVOLUTION", True, (235, 200, 110))
        screen.blit(title, title.get_rect(center=(cx, 150)))
        sub = font.render("spiral experiment pygame", True, (170, 170, 170))
        screen.blit(sub, sub.get_rect(center=(cx, 205)))

        for r, action, label in rects:
            hot = r.collidepoint(mouse)
            col = (70, 90, 70) if hot else (45, 50, 45)
            pygame.draw.rect(screen, col, r, border_radius=10)
            pygame.draw.rect(screen, (20, 20, 20), r, 2, border_radius=10)
            t = font.render(label, True, (235, 235, 235))
            screen.blit(t, t.get_rect(center=r.center))

        tip_txt = ("Press H during a run to save a demo  |  open 'Demo History' for stats & resume"
                   if has_demos else "Press H during a run to save a demo -> 'Demo History' appears here")
        tip = small.render(tip_txt, True, (130, 130, 130))
        screen.blit(tip, tip.get_rect(center=(cx, y + 30)))
        pygame.display.flip()


# ===========================================================================
# Demo History
# ===========================================================================
def run_demo_history(screen):
    """Lists saved demos. Click a row to resume; 'Stats' to view charts.
    Returns 'menu', 'quit', or a .pkl path to resume."""
    font_big = pygame.font.SysFont("consolas", 32, bold=True)
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    cx = C.SCREEN_W // 2

    demos = list_demos()[:9]   # newest 9
    row_h, top = 64, 130
    while True:
        clock.tick(C.FPS)
        mouse = pygame.mouse.get_pos()
        # row rects + stats button rects
        rows = []
        for i, (pkl, s) in enumerate(demos):
            r = pygame.Rect(cx - 460, top + i * (row_h + 10), 920, row_h)
            sb = pygame.Rect(r.right - 130, r.y + 14, 110, row_h - 28)
            rows.append((r, sb, pkl, s))
        back = pygame.Rect(cx - 100, top + len(demos) * (row_h + 10) + 16, 200, 48)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "menu"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if back.collidepoint(event.pos):
                    return "menu"
                for r, sb, pkl, s in rows:
                    if sb.collidepoint(event.pos):
                        if show_stats_screen(screen, s.get("history", []),
                                             f"Demo {s.get('created','')}  Stats") == "quit":
                            return "quit"
                        break
                    elif r.collidepoint(event.pos):
                        return pkl

        screen.fill((20, 18, 16))
        screen.blit(font_big.render("DEMO HISTORY", True, (235, 200, 110)), (cx - 460, 60))
        screen.blit(small.render("Click a demo to resume it  |  'Stats' to view charts  |  ESC: back",
                                 True, (140, 140, 140)), (cx - 460, 100))
        for r, sb, pkl, s in rows:
            hot = r.collidepoint(mouse) and not sb.collidepoint(mouse)
            pygame.draw.rect(screen, (55, 62, 55) if hot else (40, 44, 42), r, border_radius=8)
            pygame.draw.rect(screen, (20, 20, 20), r, 2, border_radius=8)
            line1 = f"Demo {s.get('created','?')}"
            line2 = (f"time {s.get('time',0)}s   gen {s.get('generation',0)}   "
                     f"delivered {s.get('delivered',0)}   births {s.get('births',0)}   "
                     f"deaths {s.get('deaths',0)}   best fit {s.get('hof_best',0):.1f}")
            screen.blit(font.render(line1, True, (235, 235, 235)), (r.x + 16, r.y + 10))
            screen.blit(small.render(line2, True, (190, 195, 200)), (r.x + 16, r.y + 36))
            # Stats button
            sbhot = sb.collidepoint(mouse)
            pygame.draw.rect(screen, (60, 90, 120) if sbhot else (45, 65, 90), sb, border_radius=6)
            st = small.render("Stats", True, (235, 235, 235))
            screen.blit(st, st.get_rect(center=sb.center))

        bhot = back.collidepoint(mouse)
        pygame.draw.rect(screen, (70, 70, 80) if bhot else (45, 48, 55), back, border_radius=8)
        bt = font.render("Back (ESC)", True, (235, 235, 235))
        screen.blit(bt, bt.get_rect(center=back.center))
        pygame.display.flip()


# ===========================================================================
# Map picker (training map selection for the model bank system)
# ===========================================================================
def run_map_picker(screen):
    """Lists maps in maps/. Click to train on that map; 'New Random Map'
    generates a fresh randomized training map. Returns a map path,
    'menu', or 'quit'."""
    font_big = pygame.font.SysFont("consolas", 32, bold=True)
    font = pygame.font.SysFont("consolas", 18)
    small = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()
    cx = C.SCREEN_W // 2

    while True:
        maps = list_maps()[:10]
        row_h, top = 56, 150
        rows = []
        for i, p in enumerate(maps):
            r = pygame.Rect(cx - 380, top + i * (row_h + 8), 760, row_h)
            rows.append((r, p))
        y_btn = top + len(maps) * (row_h + 8) + 16
        newbtn = pygame.Rect(cx - 380, y_btn, 370, 48)
        back = pygame.Rect(cx + 10, y_btn, 370, 48)

        clock.tick(C.FPS)
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "menu"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if back.collidepoint(event.pos):
                    return "menu"
                if newbtn.collidepoint(event.pos):
                    try:
                        return create_random_map()
                    except Exception as e:
                        print(f"[map] rastgele harita olusturulamadi: {e}")
                        continue
                for r, p in rows:
                    if r.collidepoint(event.pos):
                        return p

        screen.fill((20, 18, 16))
        screen.blit(font_big.render("SELECT TRAINING MAP", True, (235, 200, 110)),
                    (cx - 380, 60))
        screen.blit(small.render(
            "Models are saved to the model bank and carried across maps "
            "(train on many maps -> stronger, general models)",
            True, (140, 140, 140)), (cx - 380, 104))
        screen.blit(small.render("M during a run: switch to the next map  |  ESC: back",
                                 True, (140, 140, 140)), (cx - 380, 122))

        for r, p in rows:
            hot = r.collidepoint(mouse)
            pygame.draw.rect(screen, (55, 62, 55) if hot else (40, 44, 42), r,
                             border_radius=8)
            pygame.draw.rect(screen, (20, 20, 20), r, 2, border_radius=8)
            name = os.path.basename(p)
            tag = "  (default)" if os.path.normpath(p) == os.path.normpath(C.MAP_FILE) else ""
            screen.blit(font.render(name + tag, True, (235, 235, 235)),
                        (r.x + 16, r.y + 16))

        nhot = newbtn.collidepoint(mouse)
        pygame.draw.rect(screen, (60, 90, 120) if nhot else (45, 65, 90), newbtn,
                         border_radius=8)
        nt = font.render("New Random Map", True, (235, 235, 235))
        screen.blit(nt, nt.get_rect(center=newbtn.center))

        bhot = back.collidepoint(mouse)
        pygame.draw.rect(screen, (70, 70, 80) if bhot else (45, 48, 55), back,
                         border_radius=8)
        bt = font.render("Back (ESC)", True, (235, 235, 235))
        screen.blit(bt, bt.get_rect(center=back.center))
        pygame.display.flip()


# ===========================================================================
# Simulation
# ===========================================================================
def run_simulation(screen, resume_path=None, map_path=None):
    camera = Camera()
    renderer = Renderer()
    recorder = Recorder()
    clock = pygame.time.Clock()

    flash_text = ""
    flash_timer = 0.0

    # model bankasi: onceki kosularin en iyi genomlari (haritalar arasi tasinir)
    bank = ModelBank()
    map_path = map_path or C.MAP_FILE
    map_name = os.path.basename(map_path)

    if resume_path and os.path.exists(resume_path):
        try:
            sim = Simulation.load(resume_path, bank=bank)
            flash_text = f"Resumed: t={int(sim.sim_time)}s  gen={sim.generation}"
            flash_timer = 3.0
            print(f"[SIM] Resumed from {resume_path}: t={int(sim.sim_time)}s")
        except Exception as e:
            print(f"[SIM] Load error ({e}), starting a new simulation")
            sim = Simulation(world=load_world(map_path), map_name=map_name, bank=bank)
    else:
        sim = Simulation(world=load_world(map_path), map_name=map_name, bank=bank)

    debug = False
    paused = False
    speed = C.SIM_SPEED_DEFAULT
    accumulator = 0.0
    sim.auto_spawn = True
    sim.auto_food  = True
    settings = SettingsPanel()
    follow_best = False   # B tusu: active best ajani 4x zoom ile takip et

    while True:
        dt = clock.tick(C.FPS) / 1000.0
        dt = min(dt, 0.05)  # buyuk sicramalari sinirlandir

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.save_bank(); recorder.stop(); return "quit"
            speed = settings.handle_event(event, sim, speed)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sim.save_bank(); recorder.stop(); return "menu"
                elif event.key == pygame.K_m:          # harita sec
                    sim.save_bank()
                    selected = run_map_picker(screen)
                    if selected == "quit":
                        recorder.stop(); return "quit"
                    elif isinstance(selected, str) and selected.endswith(".json"):
                        map_path = selected
                        map_name = os.path.basename(map_path)
                        sim = Simulation(world=load_world(map_path),
                                         map_name=map_name, bank=bank)
                        sim.auto_spawn = True
                        sim.auto_food  = True
                        camera.reset()
                        settings = SettingsPanel()
                        flash_text = f"Map: {map_name} (bank'tan tohumlandi)"
                        flash_timer = 3.0
                        print(f"[SIM] Switched to map {map_name}")
                    # "menu" veya None -> mevcut simulasyona devam
                elif event.key == pygame.K_d:
                    debug = not debug
                elif event.key == pygame.K_z:
                    camera.cycle_zoom(pygame.mouse.get_pos())
                elif event.key == pygame.K_s:
                    recorder.toggle(screen)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    camera.reset()
                elif event.key == pygame.K_h:          # save a NEW demo (history kept)
                    try:
                        path = save_demo(sim)
                        flash_text = f"Demo saved -> {os.path.basename(path)} (see Demo History)"
                        flash_timer = 2.5
                        print(f"[SIM] Demo saved: {path}")
                    except Exception as e:
                        flash_text = f"Save error: {e}"
                        flash_timer = 3.0
                        print(f"[SIM] Save error: {e}")
                elif event.key == pygame.K_t:          # training statistics screen
                    if show_stats_screen(screen, sim.history,
                                         "Live Training Statistics") == "quit":
                        recorder.stop(); return "quit"
                elif event.key == pygame.K_o:          # hizi yariya indir
                    speed = max(C.SIM_SPEED_MIN, speed / 2.0)
                elif event.key == pygame.K_p:          # hizi iki katina cikar
                    speed = min(C.SIM_SPEED_MAX, speed * 2.0)
                elif event.key == pygame.K_k:          # omur azalt (-15sn)
                    C.LIFESPAN_MIN = max(10.0, C.LIFESPAN_MIN - 15.0)
                    C.LIFESPAN_MAX = max(C.LIFESPAN_MIN + 15, C.LIFESPAN_MAX - 15.0)
                elif event.key == pygame.K_l:          # omur artir (+15sn)
                    C.LIFESPAN_MIN += 15.0
                    C.LIFESPAN_MAX += 15.0
                elif event.key == pygame.K_f:          # ajan otomatik dogum toggle
                    sim.auto_spawn = not sim.auto_spawn
                    state = "ON" if sim.auto_spawn else "OFF"
                    flash_text = f"Auto spawn: {state}"
                    flash_timer = 2.0
                elif event.key == pygame.K_g:          # besin otomatik spawn toggle
                    sim.auto_food = not sim.auto_food
                    state = "ON" if sim.auto_food else "OFF"
                    flash_text = f"Auto food: {state}"
                    flash_timer = 2.0
                elif event.key == pygame.K_b:          # active best ajani 4x takip toggle
                    follow_best = not follow_best
                    if follow_best:
                        best = sim.active_best_ant()
                        camera.follow = best
                        sim.selected = best       # debug panelini ac (uyum gosterir)
                        camera.set_zoom(1.6)
                        flash_text = "Follow active best: ON"
                    else:
                        camera.reset()            # ikinci basis -> kamerayi resetle
                        sim.selected = None       # paneli kapat
                        flash_text = "Follow active best: OFF"
                    flash_timer = 2.0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:  # orta tik
                wx, wy = camera.screen_to_world(*event.pos)
                col = int(wx // C.CELL_SIZE)
                row = int(wy // C.CELL_SIZE)
                if 0 <= row < C.GRID_H and 0 <= col < C.GRID_W:
                    tool = settings.brush_tool
                    cur  = int(sim.world.grid[row, col])
                    if cur == C.NEST:
                        pass   # yuvaya dokunma
                    elif tool == C.EMPTY:
                        sim.world.grid[row, col]        = C.EMPTY
                        sim.world.food_amount[row, col] = 0
                        sim.world.odor_dirty = True
                        flash_text = f"Erased ({col},{row})"
                        flash_timer = 0.8
                    elif tool == C.STONE:
                        sim.world.grid[row, col]        = C.STONE
                        sim.world.food_amount[row, col] = 0
                        flash_text = f"Stone placed ({col},{row})"
                        flash_timer = 0.8
                    elif tool == C.FOOD:
                        sim.world.grid[row, col]        = C.FOOD
                        sim.world.food_amount[row, col] = C.FOOD_SPAWN_AMOUNT
                        sim.world.odor_dirty = True
                        flash_text = f"Food placed ({col},{row})"
                        flash_timer = 0.8
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not settings.btn.collidepoint(event.pos) and \
                   not (settings.open and settings._panel_rect().collidepoint(event.pos)):
                    wx, wy = camera.screen_to_world(*event.pos)
                    ant = sim.select_at(wx, wy)
                    if ant is not None:
                        camera.follow = ant
            elif event.type == pygame.MOUSEWHEEL:
                camera.wheel_zoom(pygame.mouse.get_pos(), 1 if event.y > 0 else -1)

        # surekli pan (ok tuslari)
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * PAN_SPEED * dt
        dy = (keys[pygame.K_DOWN] - keys[pygame.K_UP]) * PAN_SPEED * dt
        if dx or dy:
            camera.pan(dx, dy)

        # sabit zaman adimi (hizdan bagimsiz kararli fizik)
        if not paused:
            accumulator += dt * speed
            steps = 0
            while accumulator >= C.FIXED_DT and steps < C.MAX_SUBSTEPS:
                sim.update(C.FIXED_DT)
                accumulator -= C.FIXED_DT
                steps += 1
            if steps >= C.MAX_SUBSTEPS:
                accumulator = 0.0  # cok yavas donanimda spiral'i kes

        # active best takip: her karede en iyiyi bul; degistiyse kamera VE debug
        # paneli otomatik yeni en iyiye gecer, zoom sabit kalir
        if follow_best:
            best = sim.active_best_ant()
            if best is not None:
                if best is not camera.follow:
                    camera.follow = best
                sim.selected = best       # debug paneli takip edilene uyum gostersin
                camera.set_zoom(1.6)
        camera.update()

        renderer.draw_world(screen, sim.world, camera, debug=debug)
        renderer.draw_ants(screen, sim, camera, debug=debug)
        renderer.draw_hud(screen, sim, camera, debug, recorder, paused=paused, speed=speed,
                          follow_best=follow_best)
        settings.draw(screen, sim, speed)

        # gecici bildirim (kayit/devam)
        if flash_timer > 0:
            flash_timer -= dt
            renderer.draw_flash(screen, flash_text)

        recorder.capture(screen)
        pygame.display.flip()


# ===========================================================================
# Editor
# ===========================================================================
def run_editor(screen):
    # Editoru bos harita ile ac; kullanici Load ile istedigini secebilir
    # ya da mevcut default_map varsa onu on-yukle.
    grid = None
    food_amount = None
    map_name = "default_map"
    if os.path.exists(C.MAP_FILE):
        try:
            w = World.load(C.MAP_FILE)
            grid = w.grid
            food_amount = w.food_amount
        except Exception:
            grid = None
    editor = MapEditor(screen, grid=grid, food_amount=food_amount, map_name=map_name)
    return editor.run()


# ===========================================================================
def main():
    pygame.init()
    pygame.display.set_caption("Ant Neuroevolution Simulation")
    # cift tamponlama -> daha akici/net goruntu
    screen = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H), pygame.DOUBLEBUF)

    state = "menu"
    while True:
        if state == "menu":
            state = run_menu(screen)
        elif state == "sim":
            # once egitim haritasi secilir (model bankasi haritalar arasi tasinir)
            state = run_map_picker(screen)
            if isinstance(state, str) and state.endswith(".json"):
                state = run_simulation(screen, map_path=state)
        elif state == "demos":
            state = run_demo_history(screen)
            # a .pkl path means "resume this demo"
            if isinstance(state, str) and state.endswith(".pkl"):
                state = run_simulation(screen, resume_path=state)
        elif state == "maps":
            state = run_map_picker(screen)
            if isinstance(state, str) and state.endswith(".json"):
                state = run_simulation(screen, map_path=state)
        elif state == "editor":
            state = run_editor(screen)
        elif state == "quit":
            break
        else:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
