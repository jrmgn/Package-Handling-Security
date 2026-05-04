import cv2
import numpy as np
import random
import os

def create_synthetic_tamper(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        return []
        
    h, w, _ = image.shape

    # 1. Define box size
    box_h, box_w = 100, 200
    
    # 2. Pick a random top-left corner
    # Stay within boundaries
    x = random.randint(0, w - box_w)
    y = random.randint(0, h - box_h)
    
    # 3. Apply the 'tamper' (Draw a black rectangle)
    # Using -1 for thickness fills the rectangle
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    
    # 4. Save the image
    cv2.imwrite(output_path, image)
    
    # 5. Return the COCO format bbox: [x, y, width, height]
    return [[x, y, box_w, box_h]]