from typing import Generator, Optional
import cv2
import numpy as np

from .singleton import Singleton
#from .text_recognition import RealTimeTextRecognition
#from .circle_division import RealTimeTextRecognition
from .barometer_recognition import RealTimeTextRecognition

class BaseWebCamera:
    def __init__(self, cam_id: int = 0) -> None:
        self.cam_id = cam_id
        self.cam = cv2.VideoCapture(self.cam_id)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.width = int(self.cam.get(3))
        self.height = int(self.cam.get(4))
        self.fps = int(self.cam.get(5))

    def get_metadata(self) -> dict:
        frame_transform = ",".join(map(str, (self.height, self.width, 3)))
        return {
            "Frame-Transform": frame_transform,
            "Chunk-Size": "1024,1024",
            "FPS": str(self.fps),
        }

    def get_frame(self) -> np.ndarray:
        _, frame = self.cam.read()
        frame = cv2.flip(frame, 1)
        return frame


class WebCameraStream(BaseWebCamera, metaclass=Singleton):
    def stream_frame_bytes(self) -> Generator[bytes, None, None]:
        while self.cam.isOpened():
            yield self.get_frame().tobytes()

    def stream_img_bytes(self) -> Generator[bytes, None, None]:
        while self.cam.isOpened():
            _, buffer = cv2.imencode(".jpg", self.get_frame())
            img_frame = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + img_frame + b"\r\n"
    # ------------------------------------------------------------------------------------------------
    # Provide the path to the Tesseract executable

    #tesseract_cmd_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #recognizer = RealTimeTextRecognition(tesseract_cmd_path)
    #recognizer.capture_and_process_image()
    #recognizer.start()

    recognizer = RealTimeTextRecognition()
    recognizer.start()
    

    # ------------------------------------------------------------------------------------------------

class WebCameraRecoder(BaseWebCamera, metaclass=Singleton):
    def __init__(self, video_name: Optional[str] = None, cam_id: int = 0):
        self.video_name = video_name
        super().__init__(cam_id)

    def record_video(self) -> None:
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv2.VideoWriter(f"{self.video_name}.mp4", fourcc, self.fps, (self.width, self.height))

        while self.cam.isOpened():
            self.video_writer.write(self.get_frame())
