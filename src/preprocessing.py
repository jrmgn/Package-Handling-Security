import numpy as np
import cv2
import os
import glob
import json
import random
import shutil
import xml.etree.ElementTree as ET
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def split_dataset(raw_dir, ann_dir, output_root, split_ratio=(0.8, 0.1, 0.1)):
    """Splits images and XMLs into train, val, and test folders."""
    images = glob.glob(os.path.join(raw_dir, "*.jpg")) + glob.glob(os.path.join(raw_dir, "*.png"))
    random.shuffle(images)

    train_end = int(len(images) * split_ratio[0])
    val_end = train_end + int(len(images) * split_ratio[1])

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split_name, split_images in splits.items():
        img_dest = os.path.join(output_root, split_name)
        os.makedirs(img_dest, exist_ok=True)
        
        for img_path in split_images:
            base_name = os.path.basename(img_path)
            xml_name = os.path.splitext(base_name)[0] + ".xml"
            xml_path = os.path.join(ann_dir, xml_name)

            shutil.copy(img_path, os.path.join(img_dest, base_name))
            if os.path.exists(xml_path):
                shutil.copy(xml_path, os.path.join(img_dest, xml_name))
    
    print(f"Dataset split completed: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test.")

def extract_ocr_from_xml(image_path, xml_path):
    """Crops regions from image based on XML and extracts text via OCR."""
    img = cv2.imread(image_path)
    if img is None: return {}
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ocr_results = {}

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin, ymin = int(bbox.find('xmin').text), int(bbox.find('ymin').text)
        xmax, ymax = int(bbox.find('xmax').text), int(bbox.find('ymax').text)

        crop = img[ymin:ymax, xmin:xmax]
        text = pytesseract.image_to_string(crop, config='--psm 6').strip()
        ocr_results[label] = text

    return ocr_results

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    ANN_DIR = "data/annotated"
    OUTPUT_ROOT = "data" 

    split_dataset(RAW_DIR, ANN_DIR, OUTPUT_ROOT)

from skimage.feature import hog, local_binary_pattern

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
    
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    
    return np.hstack([fd, lbp_hist])
