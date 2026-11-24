import unittest
from PIL import Image

class TestCameraModule(unittest.TestCase):
    def test_camera_init_and_availability(self):
        from camera import CameraModule
        cam = CameraModule(device_index=0)
        self.assertIsInstance(cam.is_available(), bool)
        cam.release()

    def test_camera_frame_capture_or_none(self):
        from camera import CameraModule
        cam = CameraModule(device_index=0)
        frame = cam.get_frame()
        if cam.is_available():
            if frame is not None:
                self.assertIsInstance(frame, Image.Image)
        else:
            self.assertIsNone(frame)
        cam.release()

    def test_invalid_device_index_graceful(self):
        from camera import CameraModule
        cam = CameraModule(device_index=999)
        self.assertFalse(cam.is_available())
        frame = cam.get_frame()
        self.assertIsNone(frame)
        cam.release()

if __name__ == "__main__":
    unittest.main()
