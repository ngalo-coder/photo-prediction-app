from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import cv2
import logging

app = Flask(__name__)

# Load model
model = load_model('model/model.h5')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Enhanced preprocessing to match MNIST format"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Apply advanced preprocessing
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return img

def extract_digits(image_path):
    """Robust digit extraction with better contour handling"""
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return []
    
    # Find contours with better parameters
    contours, _ = cv2.findContours(
        processed_img, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    digit_images = []
    bounding_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        # More flexible filtering
        if w > 5 and h > 10 and 0.1 < aspect_ratio < 2.0:
            # Extract digit with border for better recognition
            border_size = 10
            digit = processed_img[y:y+h, x:x+w]
            
            # Pad and resize
            padded = cv2.copyMakeBorder(
                digit, 
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT, 
                value=0
            )
            resized = cv2.resize(padded, (28, 28))
            
            # Normalize and reshape for model
            normalized = resized.astype('float32') / 255.0
            digit_images.append(normalized.reshape(1, 28, 28, 1))
            bounding_boxes.append((x, y, w, h))

    # Sort left-to-right
    sorted_digits = [img for _, img in sorted(
        zip(bounding_boxes, digit_images),
        key=lambda pair: pair[0][0]
    )]
    
    return sorted_digits

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file uploaded"), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file selected"), 400

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)
    
    try:
        digits = extract_digits(img_path)
        if not digits:
            # Fallback to single digit processing
            img = Image.open(img_path).convert('L').resize((28, 28))
            img_array = 255 - np.array(img)  # Invert
            img_array = img_array.astype('float32') / 255.0
            digits = [img_array.reshape(1, 28, 28, 1)]

        predictions = []
        for digit in digits:
            pred = model.predict(digit)
            predictions.append(int(np.argmax(pred)))

        os.remove(img_path)  # Cleanup
        
        return jsonify(
            predictions=predictions,
            digit_count=len(predictions))
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)