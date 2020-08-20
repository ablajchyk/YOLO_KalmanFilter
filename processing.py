import cv2
import numpy as np
from typing import Optional, Tuple


class Processing():
    def __init__(self, path: Optional[str] = None, frame_size: Tuple[int, int] = (320,320)):
        self.path = path
        self.stream = None
        self.output = None
        self.frame_size = frame_size

    def open_video(self, live: bool = False):
        if live or not self.path:
            self.stream = cv2.VideoCapture(0)
        else:
            self.stream = cv2.VideoCapture(self.path)

    def open_writer(self, name: str, fsp: int = 25):
        self.output = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fsp, self.frame_size)

    def get_frame(self) -> np.ndarray:
        return self.stream.read()[1]

    def write_video(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            self.output.write(cv2.resize(frame, self.frame_size))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    def close_video(self):
        self.output.release()
        self.stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    check = Processing('pedastrians.mp4')
    check.open_video()
    check.open_writer('output.avi')
    check.write_video()
    check.close_video()
