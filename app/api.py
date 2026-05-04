from flask import Flask, request, jsonify
import os
from src.infer import ForgeryPipeline

app = Flask(__name__)
pipeline = ForgeryPipeline(
    detector_weights="models/detector_weights.pt", 
    fusion_model_path="models/fusion_model_v1.pkl"
)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    file_path = "temp_upload.jpg"
    file.save(file_path)
    
    detections = api_response.get("suspicious_regions", [])
    
    image_score = max([d["detection_score"] for d in detections]) if detections else 0.0

    return jsonify({
        "image_level_forgery_score": round(image_score, 4), 
        "suspicious_regions": detections, 
        "extracted_content": [d["text"] for d in detections],
        "status": "success"
    })

@app.route('/')
def home():
    return "Forgery Detection API is Running!"

if __name__ == '__main__':
    app.run(port=5000)
