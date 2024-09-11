import os
import torch
from flask import Flask, request, jsonify
from torchvision import models, transforms
from torch import nn
from PIL import Image
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React frontend
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # Adjust for 5 output classes
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

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
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess the image
    image = Image.open(file_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Run the model prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        severity = int(predicted.item())

    # Delete the file after prediction
    os.remove(file_path)

    # Return the result as JSON
    return jsonify({'severity': severity})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
