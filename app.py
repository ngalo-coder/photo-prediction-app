from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import cv2
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load model - ensure this path is correct
model = load_model('model/model.h5')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_grid_image(img, cell_size=28):
    """Check if image is a grid with at least 2x2 cells"""
    if img.width < 2*cell_size or img.height < 2*cell_size:
        return False
    
    # Check if dimensions are close to multiples of cell_size
    w_ratio = img.width / cell_size
    h_ratio = img.height / cell_size
    
    return (abs(w_ratio - round(w_ratio)) < 0.05 and abs(h_ratio - round(h_ratio)) < 0.05)

def split_grid(img, cell_size=28):
    """Split grid image into individual digit cells"""
    rows = int(round(img.height / cell_size))
    cols = int(round(img.width / cell_size))
    digits = []
    
    for r in range(rows):
        for c in range(cols):
            # Calculate cell boundaries with padding compensation
            left = int(c * img.width / cols)
            upper = int(r * img.height / rows)
            right = int((c+1) * img.width / cols)
            lower = int((r+1) * img.height / rows)
            
            digit_img = img.crop((left, upper, right, lower))
            digit_img = digit_img.resize((28, 28))
            
            # Convert to array (keep original colors)
            digit_array = np.array(digit_img).astype('float32') / 255.0
            digits.append(digit_array.reshape(1, 28, 28, 1))
    return digits

def segment_digits_sequence(image_path):
    """Segment digits using contour detection with row/column sorting"""
    # Read image directly with PIL to match notebook processing
    pil_img = Image.open(image_path).convert('L')
    gray_image = np.array(pil_img)
    
    # Apply thresholding (like notebook)
    _, thresh = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    digit_images = []
    bounding_boxes = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        # Filter by size and aspect ratio (match notebook)
        if w > 10 and h > 20 and 0.2 < aspect_ratio < 1.0:
            # Extract digit from grayscale image (not thresholded)
            digit = gray_image[y:y+h, x:x+w]
            
            # Convert to PIL Image for processing (like notebook)
            digit_pil = Image.fromarray(digit).resize((28, 28))
            digit_array = np.array(digit_pil).astype('float32') / 255.0
            
            digit_images.append(digit_array.reshape(1, 28, 28, 1))
            bounding_boxes.append((x, y, w, h))  # Store position for sorting

    # Sort left-to-right (by x coordinate)
    sorted_digits = [img for _, img in sorted(
        zip(bounding_boxes, digit_images),
        key=lambda pair: pair[0][0]
    )]
    
    return sorted_digits

def prepare_single_image(img_path):
    """Process single digit image"""
    img = Image.open(img_path).convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    return [img_array.reshape(1, 28, 28, 1)]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)
    app.logger.info(f"Saved uploaded file to {img_path}")

    try:
        img = Image.open(img_path).convert('L')
        
        if is_grid_image(img):
            app.logger.info("Processing as grid image")
            digits = split_grid(img)
        else:
            app.logger.info("Processing with contour detection")
            digits = segment_digits_sequence(img_path)
            
            if not digits:
                app.logger.info("No digits found via contours, trying single digit")
                digits = prepare_single_image(img_path)

        if not digits:
            return jsonify(error="No digits detected"), 400

        predictions = []
        for d in digits:
            pred = model.predict(d)
            predictions.append(int(np.argmax(pred)))

        os.remove(img_path)  # Clean up uploaded file
        
        if len(predictions) == 1:
            return jsonify(prediction=predictions[0])
        else:
            return jsonify(predictions=predictions)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)