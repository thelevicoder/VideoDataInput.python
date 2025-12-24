# hold_classifier_webapp.py
#
# Simple Flask web app to upload and classify hold images
# Run with: python hold_classifier_webapp.py

from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
import json
import io
import base64

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# Global model variables
MODEL = None
CLASSES = None
TRANSFORM = None
DEVICE = None


def load_model():
    """Load model once at startup."""
    global MODEL, CLASSES, TRANSFORM, DEVICE
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    models_dir = Path("models")
    model_path = models_dir / "hold_classifier_resnet18.pt"
    labels_path = models_dir / "hold_class_labels.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    
    with labels_path.open("r") as f:
        label_data = json.load(f)
    CLASSES = label_data["classes"]
    
    MODEL = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    MODEL.fc = nn.Linear(MODEL.fc.in_features, len(CLASSES))
    
    state = torch.load(model_path, map_location=DEVICE)
    MODEL.load_state_dict(state)
    MODEL.to(DEVICE)
    MODEL.eval()
    
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    print(f"✅ Model loaded with classes: {CLASSES}")


def classify_image(image_bytes):
    """Classify an image from bytes."""
    # Convert bytes to PIL Image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Transform and predict
    x = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
    
    cls_idx = int(idx.item())
    cls_name = CLASSES[cls_idx]
    conf_val = float(conf.item())
    
    # Get all probabilities
    all_probs = {
        CLASSES[i]: float(probs[i].item())
        for i in range(len(CLASSES))
    }
    
    return cls_name, conf_val, all_probs


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hold Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f1ff;
        }
        
        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8e9ff;
            transform: scale(1.02);
        }
        
        #fileInput {
            display: none;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .upload-text {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .upload-hint {
            color: #999;
            font-size: 0.85em;
        }
        
        #preview {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        #result {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            display: none;
            animation: slideIn 0.5s;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-class {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        
        .result-confidence {
            font-size: 1.2em;
            margin-bottom: 20px;
            opacity: 0.9;
        }
        
        .probabilities {
            background: rgba(255,255,255,0.15);
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
        }
        
        .prob-title {
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        
        .prob-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 0.85em;
        }
        
        .prob-bar {
            background: rgba(255,255,255,0.3);
            height: 6px;
            border-radius: 3px;
            margin-top: 4px;
            overflow: hidden;
        }
        
        .prob-fill {
            background: white;
            height: 100%;
            transition: width 0.5s;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
            color: #667eea;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧗 Hold Classifier</h1>
        <p class="subtitle">Upload an image of a climbing hold to classify it</p>
        
        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Click to upload or drag & drop</div>
            <div class="upload-hint">Supports JPG, PNG</div>
        </div>
        
        <input type="file" id="fileInput" accept="image/*" onchange="handleFile(this.files[0])">
        
        <img id="preview" alt="Preview">
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Classifying...</div>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });

        function handleFile(file) {
            if (!file) return;

            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Classify
            classifyImage(file);
        }

        function classifyImage(file) {
            loading.style.display = 'block';
            result.style.display = 'none';

            const formData = new FormData();
            formData.append('image', file);

            fetch('/classify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                displayResult(data);
            })
            .catch(error => {
                loading.style.display = 'none';
                alert('Error: ' + error);
            });
        }

        function displayResult(data) {
            const confidencePercent = (data.confidence * 100).toFixed(1);
            const confidenceClass = data.confidence > 0.7 ? '✅' : data.confidence > 0.4 ? '⚠️' : '❌';
            
            let probsHTML = '<div class="probabilities"><div class="prob-title">All Probabilities:</div>';
            
            // Sort by probability
            const sortedProbs = Object.entries(data.all_probabilities)
                .sort((a, b) => b[1] - a[1]);
            
            sortedProbs.forEach(([cls, prob]) => {
                const percent = (prob * 100).toFixed(1);
                probsHTML += `
                    <div class="prob-item">
                        <span>${cls}</span>
                        <span>${percent}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: ${percent}%"></div>
                    </div>
                `;
            });
            
            probsHTML += '</div>';

            result.innerHTML = `
                <div class="result-class">${confidenceClass} ${data.class_name}</div>
                <div class="result-confidence">Confidence: ${confidencePercent}%</div>
                ${probsHTML}
            `;
            result.style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/classify', methods=['POST'])
def classify():
    """Handle image classification."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image bytes
        image_bytes = file.read()
        
        # Classify
        cls_name, conf_val, all_probs = classify_image(image_bytes)
        
        return jsonify({
            'class_name': cls_name,
            'confidence': conf_val,
            'all_probabilities': all_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("HOLD CLASSIFIER WEB APP")
    print("="*60 + "\n")
    
    # Load model
    try:
        load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nMake sure you have:")
        print("  - models/hold_classifier_resnet18.pt")
        print("  - models/hold_class_labels.json")
        exit(1)
    
    print("\n🚀 Starting web server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)