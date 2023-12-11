from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import io

app = FastAPI()

# Load your trained model (update the path as needed)
model = load_model('mri-model.h5')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open('index.html', 'r') as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image and preprocess it
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image, target_size=(224, 224))

        # Make a prediction
        prediction = model.predict(processed_image)

        # Convert prediction to a categorical result
        prediction_score = prediction[0][0]
        result = "Patient has Tumor" if prediction_score > 0.5 else "Patient does not have tumor"

        return JSONResponse(content={"prediction": result, "score": str(prediction_score)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
