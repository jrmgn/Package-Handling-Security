import cv2
import numpy as np
import random
import os

def create_synthetic_tamper(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return []
        
    h, w, _ = image.shape
    box_h, box_w = 100, 200
    
    x = random.randint(0, w - box_w)
    y = random.randint(0, h - box_h)
    
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    
    cv2.imwrite(output_path, image)
    
    return [[x, y, box_w, box_h]]
