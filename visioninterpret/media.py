from PIL import Image 
import cv2 
import numpy as np 

from lib.processor import Processor 

class MediaProcessor:
    def load(self, *args, **kwargs):
        pass 
    
    def process(self, *args, **kwargs):
        pass 
    
    def save(self, *args, **kwargs):
        pass 
    
class ImageProcessor(MediaProcessor):
    def load(self, path):
        return Image.open(path)
    
    def process(self, path: str, processor: Processor):
        image = self.load(path)
        return processor.process(image)
    
    def save(self, path: str, image: np.ndarray | Image.Image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path, format="PNG")
    
class VideoProcessor(MediaProcessor):
    def load(self, path):
        return cv2.VideoCapture(path)
    
    def process(self, path: str, processor: Processor):
        video = self.load(path)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = processor.process(Image.fromarray(frame))
            yield frame
    
    def save(self, path: str, frames: list[np.ndarray | Image.Image]):
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))

        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)
        out.release()