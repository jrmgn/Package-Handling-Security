Forgery Detection System

An AI-driven pipeline designed to verify the authenticity of documents (receipts, IDs, licenses) using a multi-stage approach involving object detection, OCR, and forgery scoring.

Deployment & Scaling Notes
1. Convert the YOLOv8 and fusion models to ONNX or TorchScript formats to reduce model size and improve latency.
2. While the current pipeline runs on CPU, high-volume production should utilize GPU inference to handle multiple concurrent requests efficiently.
3. Implement request batching in the Flask API to process multiple images in a single forward pass, maximizing hardware throughput.
4. For high-scale deployments, use a task queue like Celery to handle image processing in the background, preventing API timeouts.

Privacy & Data Handling
Since this system processes sensitive Government-Issued IDs, the following data handling protocols are advised:
1. The API should not store uploaded images permanently. Use the temp_upload.jpg pattern and ensure immediate deletion after processing.
2. All data in transit must be protected using TLS/SSL (HTTPS). Any data at rest (logs or metadata) should be encrypted.
3. Implement API Key authentication to ensure only authorized clients can access the endpoint.

Ethical Considerations & Bias
1. It is important to acknowledge the limitations and responsibilities of AI-driven verification:
2. The model may have  higher error rates or false positives on unusual ID designs, older license versions, or IDs from regions not represented in the training data.
3. This system should be used as a decision-support tool, not a final authority. High-risk "forgery" flags should always be reviewed by a human.

Setup & Installation
1. Clone the repository and navigate to the project folder.
2. Install dependencies: pip install -r requirements.txt
3. Run the API: python app.py

