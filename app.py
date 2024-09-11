import os
import torch
from flask import Flask, request, jsonify
from torchvision import models, transforms
from torch import nn
from PIL import Image
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)

# Allow requests only from your specific frontend domain
CORS(app, resources={r"/*": {"origins": "https://med-cnn-frontend.vercel.app"}})

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # Adjust for 5 output classes
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Endpoint for predicting diabetic retinopathy
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print(f"File saved at {file_path}")

        # Preprocess the image
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Run the model prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            severity = int(predicted.item())
            print(f"Prediction completed with severity: {severity}")

        # Delete the file after prediction
        os.remove(file_path)
        print(f"File {file_path} deleted after prediction.")

        # Return the result as JSON
        return jsonify({'severity': severity})
    
    except Exception as e:
        print(f"Error processing the image: {e}")
        return jsonify({'error': 'Prediction failed due to an internal error.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
