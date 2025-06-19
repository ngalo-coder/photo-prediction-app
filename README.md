# Photo Prediction App

This project is a web application that allows users to upload photos and receive predictions based on a pre-trained model. The application is built using Flask and utilizes a TensorFlow model saved in HDF5 format.

## Project Structure

```
photo-prediction-app
├── app.py                # Main application file
├── requirements.txt      # Dependencies for the project
├── templates             # HTML templates for the web interface
│   └── index.html       # User interface for photo upload and prediction
├── static                # Static files such as CSS
│   └── style.css        # Styles for the web application
├── model                 # Directory containing the model
│   └── model.h5         # Pre-trained model file
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd photo-prediction-app
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Flask server by running:
   ```
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000`.

## Usage

- Open your web browser and navigate to `http://127.0.0.1:5000`.
- Use the provided form to upload a photo.
- After uploading, the application will display the prediction results based on the uploaded image.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.