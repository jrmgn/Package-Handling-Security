import cv2
import joblib
import pytesseract
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ForgeryPipeline:
    def __init__(self, detector_weights, fusion_model_path):
        self.detector = YOLO('yolov8n.pt') 
        self.fusion_model = joblib.load(fusion_model_path)
        print("System: Full Pipeline Loaded.")
        
    def predict(self, image_path):
            img = cv2.imread(image_path)
            # Step 1: Preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Step 2: Detection
            results = self.detector(img, verbose=False)[0]
            
            # CRITICAL: This line must be here, outside the loop!
            final_results = [] 
            
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy()
                det_score = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = coords.astype(int)
                crop = gray[y1:y2, x1:x2]
                
                # Step 3: OCR
                text = pytesseract.image_to_string(crop).strip()
                
                # Step 4: Append to the list we just created
                final_results.append({
                    "box": coords.tolist(),
                    "detection_score": det_score,
                    "text": text,
                    "field_flag": "SUSPICIOUS" if det_score > 0.5 else "CLEAN"
                })
                
            return final_results