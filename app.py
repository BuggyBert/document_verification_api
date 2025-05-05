
from flask import Flask, request, jsonify
import pickle
import base64
from PIL import Image
import io
import requests

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1CAa0z4HrCKwMsXUjpl5MaVe2qAD8nmYd"

# Download and load model
def download_model():
    response = requests.get(MODEL_URL)
    with open("random_forest_model.pkl", "wb") as f:
        f.write(response.content)

download_model()
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/verify', methods=['POST'])
def verify_document():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data))

    # Dummy feature extraction
    features = [0.5, 0.8, 0.3]  # Replace with real features from image
    prediction = model.predict([features])[0]

    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run()
