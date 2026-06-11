"""
Karinca noroevrim simulasyonu - giris noktasi.

Calistir:  python main.py

Menu:
  - Simulasyonu Baslat
  - Harita Editoru
  - Cikis

Simulasyon tuslari:
  D       : debug modu (gorus isinlari + secili karinca paneli)
  Z       : zoom seviyesi (x1 / x1.6 / x2.5 / x4)
  O / P   : simulasyon hizi - / +
  K / L   : karinca omru - / + (15sn)
  H       : simulasyonu kaydet (menude 'Devam Et' ile geri yukle)
  S       : ekran kaydini baslat / durdur (recordings/)
  Space   : duraklat / devam
  Ok tus. : haritada gez (pan)
  Sol tik : (debug) karinca sec ve takip et
  R       : kamerayi sifirla / takibi birak
  ESC     : menuye don
"""

import os

import pygame

import config as C
from world import World, make_default_world
from simulation import Simulation
from camera import Camera
from renderer import Renderer
from recorder import Recorder
from map_editor import MapEditor

PAN_SPEED = 600.0  # piksel/sn (ekran uzayinda)


def load_world():
    if os.path.exists(C.MAP_FILE):
        try:
            return World.load(C.MAP_FILE)
        except Exception as e:
            print(f"[map] yuklenemedi ({e}), varsayilan harita kullaniliyor")
    w = make_default_world()
    try:
        w.save(C.MAP_FILE)  # editorde duzenlenebilsin diye varsayilani yaz
    except Exception:
        pass
    return w


# ===========================================================================
# Menu
# ===========================================================================
def run_menu(screen):
    font_big = pygame.font.SysFont("consolas", 46, bold=True)
    font = pygame.font.SysFont("consolas", 24)
    small = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    cx = C.SCREEN_W // 2
    has_checkpoint = os.path.exists(C.CHECKPOINT_FILE)
    buttons = [("Simulasyonu Baslat (yeni)", "sim")]
    if has_checkpoint:
        buttons.append(("Devam Et (checkpoint)", "resume"))
    buttons += [("Harita Editoru", "editor"), ("Cikis", "quit")]

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
                elif event.key == pygame.K_2:
                    return "resume" if has_checkpoint else "editor"
                elif event.key == pygame.K_3 and has_checkpoint:
                    return "editor"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for r, action, _ in rects:
                    if r.collidepoint(event.pos):
                        return action

        screen.fill((20, 18, 16))
        title = font_big.render("KARINCA  NOROEVRIM", True, (235, 200, 110))
        screen.blit(title, title.get_rect(center=(cx, 150)))
        sub = font.render("LSTM + Genetik Algoritma  /  pygame", True, (170, 170, 170))
        screen.blit(sub, sub.get_rect(center=(cx, 205)))

        for r, action, label in rects:
            hot = r.collidepoint(mouse)
            col = (70, 90, 70) if hot else (45, 50, 45)
            pygame.draw.rect(screen, col, r, border_radius=10)
            pygame.draw.rect(screen, (20, 20, 20), r, 2, border_radius=10)
            t = font.render(label, True, (235, 235, 235))
            screen.blit(t, t.get_rect(center=r.center))

        tip_txt = ("Sim sirasinda H: kaydet  |  Devam icin menude 'Devam Et'"
                   if has_checkpoint else "Sim sirasinda H ile kaydet -> menude 'Devam Et' cikar")
        tip = small.render(tip_txt, True, (130, 130, 130))
        screen.blit(tip, tip.get_rect(center=(cx, y + 30)))
        pygame.display.flip()


# ===========================================================================
# Simulasyon
# ===========================================================================
def run_simulation(screen, resume=False):
    camera = Camera()
    renderer = Renderer()
    recorder = Recorder()
    clock = pygame.time.Clock()

    flash_text = ""
    flash_timer = 0.0

    if resume and os.path.exists(C.CHECKPOINT_FILE):
        try:
            sim = Simulation.load(C.CHECKPOINT_FILE)
            flash_text = f"Devam edildi: t={int(sim.sim_time)}s  nesil={sim.generation}"
            flash_timer = 3.0
            print(f"[SIM] Checkpoint'ten devam: t={int(sim.sim_time)}s")
        except Exception as e:
            print(f"[SIM] Yukleme hatasi ({e}), yeni simulasyon baslatiliyor")
            sim = Simulation(world=load_world())
    else:
        sim = Simulation(world=load_world())

    debug = False
    paused = False
    speed = C.SIM_SPEED_DEFAULT
    accumulator = 0.0

    while True:
        dt = clock.tick(C.FPS) / 1000.0
        dt = min(dt, 0.05)  # buyuk sicramalari sinirlandir

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                recorder.stop(); return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    recorder.stop(); return "menu"
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
                elif event.key == pygame.K_h:          # simulasyonu kaydet
                    try:
                        sim.save(C.CHECKPOINT_FILE)
                        flash_text = f"Kaydedildi -> {C.CHECKPOINT_FILE} (menude 'Devam Et')"
                        flash_timer = 2.5
                        print(f"[SIM] Checkpoint kaydedildi: t={int(sim.sim_time)}s")
                    except Exception as e:
                        flash_text = f"Kayit hatasi: {e}"
                        flash_timer = 3.0
                        print(f"[SIM] Kayit hatasi: {e}")
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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                wx, wy = camera.screen_to_world(*event.pos)
                ant = sim.select_at(wx, wy)
                if ant is not None:
                    camera.follow = ant  # takip et
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
        camera.update()

        renderer.draw_world(screen, sim.world, camera, debug=debug)
        renderer.draw_ants(screen, sim, camera, debug=debug)
        renderer.draw_hud(screen, sim, camera, debug, recorder, paused=paused, speed=speed)

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
    # mevcut haritayi yukle (varsa) ki uzerinde duzenleme yapilabilsin
    grid = None
    food_amount = None
    if os.path.exists(C.MAP_FILE):
        try:
            w = World.load(C.MAP_FILE)
            grid = w.grid
            food_amount = w.food_amount
        except Exception:
            grid = None
    editor = MapEditor(screen, grid=grid, food_amount=food_amount)
    return editor.run()


# ===========================================================================
def main():
    pygame.init()
    pygame.display.set_caption("Karinca Noroevrim Simulasyonu")
    # cift tamponlama -> daha akici/net goruntu
    screen = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H), pygame.DOUBLEBUF)

    state = "menu"
    while True:
        if state == "menu":
            state = run_menu(screen)
        elif state == "sim":
            state = run_simulation(screen, resume=False)
        elif state == "resume":
            state = run_simulation(screen, resume=True)
        elif state == "editor":
            state = run_editor(screen)
        elif state == "quit":
            break
        else:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
