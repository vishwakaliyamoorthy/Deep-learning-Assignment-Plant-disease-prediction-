"""
Camera module for plant leaf snapshot and live viewfinder streaming.
Uses OpenCV (cv2) with graceful fallback and safe hardware handle management.
"""
import glob
from typing import List, Optional
import cv2
from PIL import Image


class CameraModule:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._open_device()

    def _open_device(self) -> None:
        try:
            cap = cv2.VideoCapture(self.device_index)
            if cap.isOpened():
                self._cap = cap
            else:
                self._cap = None
        except Exception:
            self._cap = None

    def is_available(self) -> bool:
        """Returns True if the camera device opened successfully and is readable."""
        return self._cap is not None and self._cap.isOpened()

    def get_frame(self) -> Optional[Image.Image]:
        """
        Reads one frame from the camera, converts from BGR to RGB,
        and returns as a PIL Image. Returns None if unreadable.
        """
        if not self.is_available():
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        # Convert OpenCV BGR to RGB for PIL / Tkinter display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def release(self) -> None:
        """Safely release the hardware camera handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    @staticmethod
    def list_available_devices() -> List[int]:
        """Scans for available /dev/video* devices on Linux or tests indices 0..3."""
        available = []
        video_paths = sorted(glob.glob("/dev/video*"))
        if video_paths:
            for path in video_paths:
                try:
                    num = int(path.replace("/dev/video", ""))
                    cap = cv2.VideoCapture(num)
                    if cap.isOpened():
                        available.append(num)
                        cap.release()
                except Exception:
                    continue
        else:
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
        return available
