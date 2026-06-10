"""
Ekran kaydi.

s tusu ile baslar/durur. Pygame yuzeyini kare kare yakalar ve video yazar.
Oncelik sirasi: OpenCV (cv2) -> imageio -> ham PNG kare dizisi (bagimliliksiz).
Belgesel icin recordings/ klasorune .mp4 (ya da PNG kareleri) birakir.
"""

import os
import time

import numpy as np
import pygame

import config as C

# Hangi backend var?
_HAS_CV2 = False
_HAS_IMAGEIO = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    try:
        import imageio
        _HAS_IMAGEIO = True
    except Exception:
        pass


class Recorder:
    def __init__(self):
        self.recording = False
        self.backend = None
        self._writer = None
        self._frame_dir = None
        self._frame_no = 0
        self._size = None
        self._last_t = 0.0
        self._interval = 1.0 / C.RECORD_FPS

    @property
    def status(self):
        return self.backend if self.recording else None

    def toggle(self, surface):
        if self.recording:
            self.stop()
        else:
            self.start(surface)

    # ------------------------------------------------------------------ start
    def start(self, surface):
        os.makedirs(C.RECORD_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        w, h = surface.get_size()
        self._size = (w, h)
        self._frame_no = 0
        self._last_t = 0.0

        if _HAS_CV2:
            path = os.path.join(C.RECORD_DIR, f"ant_{stamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, C.RECORD_FPS, (w, h))
            self.backend = "cv2"
            self._out_path = path
        elif _HAS_IMAGEIO:
            path = os.path.join(C.RECORD_DIR, f"ant_{stamp}.mp4")
            self._writer = imageio.get_writer(path, fps=C.RECORD_FPS, macro_block_size=None)
            self.backend = "imageio"
            self._out_path = path
        else:
            self._frame_dir = os.path.join(C.RECORD_DIR, f"frames_{stamp}")
            os.makedirs(self._frame_dir, exist_ok=True)
            self.backend = "png"
            self._out_path = self._frame_dir

        self.recording = True
        print(f"[REC] Kayit basladi ({self.backend}) -> {self._out_path}")

    # ----------------------------------------------------------------- capture
    def capture(self, surface):
        if not self.recording:
            return
        now = time.time()
        if now - self._last_t < self._interval:
            return
        self._last_t = now

        # pygame yuzeyi -> (W,H,3) RGB
        arr = pygame.surfarray.array3d(surface)
        arr = np.transpose(arr, (1, 0, 2))  # -> (H,W,3)

        if self.backend == "cv2":
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self._writer.write(frame)
        elif self.backend == "imageio":
            self._writer.append_data(arr)
        else:  # png
            path = os.path.join(self._frame_dir, f"frame_{self._frame_no:06d}.png")
            pygame.image.save(surface, path)
        self._frame_no += 1

    # -------------------------------------------------------------------- stop
    def stop(self):
        if not self.recording:
            return
        try:
            if self.backend == "cv2":
                self._writer.release()
            elif self.backend == "imageio":
                self._writer.close()
        except Exception as e:
            print(f"[REC] kapatma hatasi: {e}")
        self.recording = False
        print(f"[REC] Kayit durdu -> {self._out_path} ({self._frame_no} kare)")
        self._writer = None
