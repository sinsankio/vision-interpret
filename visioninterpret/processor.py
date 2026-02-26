from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

class Processor(ABC):
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
    
    @abstractmethod
    def process(self, image: np.ndarray | Image.Image, *args, **kwargs):
        pass 

