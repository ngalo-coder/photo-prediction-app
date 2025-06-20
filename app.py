from flask import Flask, request, jsonify, render_template
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import os
import cv2

app = Flask(__name__)

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='model/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
        # This list will hold the numpy arrays of digit images (1, 28, 28, 1) ready for model
        digit_image_list = []

        extracted_digits_from_cv2 = extract_digits(img_path) # Returns list of preprocessed np arrays

        if not extracted_digits_from_cv2:
            app.logger.info(f"extract_digits did not find contours from {img_path}, attempting fallback single image processing.")
            # Fallback: process img_path to a single np array
            img_pil = Image.open(img_path).convert('L').resize((28, 28))
            img_array_np = np.array(img_pil)
            # Invert colors (assuming standard MNIST: white digit on black background)
            # and normalize to [0, 1] as float32
            img_array_np = (255.0 - img_array_np.astype('float32')) / 255.0
            digit_image_list.append(img_array_np.reshape(1, 28, 28, 1))
        else:
            # Ensure all images from extract_digits are float32, though they should be already
            for digit_array in extracted_digits_from_cv2:
                 if digit_array.dtype != np.float32:
                    digit_image_list.append(digit_array.astype(np.float32))
                 else:
                    digit_image_list.append(digit_array)

        predictions = []
        if not digit_image_list:
            app.logger.error(f"No digit images to process from {img_path} after all attempts.")
            # Optionally, return a specific message to the user
            # return jsonify(error="Could not detect any digits in the image."), 400
        else:
            for digit_img_np in digit_image_list:
                # Ensure input tensor is float32 (it should be from preprocessing)
                # This double check might be redundant if extract_digits and fallback are reliable
                if digit_img_np.dtype != np.float32:
                    digit_img_np = digit_img_np.astype(np.float32)

                # Set tensor, invoke, get output
                interpreter.set_tensor(input_details[0]['index'], digit_img_np)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predictions.append(int(np.argmax(output_data)))

        os.remove(img_path)  # Cleanup
        
        return jsonify(
            predictions=predictions,
            digit_count=len(predictions))
        
    except Exception as e:
        # Use app.logger for Flask apps for better context and request info
        app.logger.error(f"Error during prediction for {img_path}: {str(e)}", exc_info=True)
        return jsonify(error=f"An internal error occurred during prediction."), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)