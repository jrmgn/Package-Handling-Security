import cv2
import numpy as np

def check_missing_fields(ocr_data, expected_fields=['total_amount', 'date', 'merchant_name']):
    """Returns a list of fields that were not found by OCR."""
    missing = [field for field in expected_fields if field not in ocr_data or not ocr_data[field]]
    return missing

def detect_font_inconsistency(image):
    """
    Uses Connected Components to check if text sizes vary wildly, 
    which often happens in 'copy-paste' forgeries.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    valid_heights = heights[(heights > 5) & (heights < 100)]
    
    if len(valid_heights) == 0: return 0.0
    
    return np.std(valid_heights)
