from fastapi import FastAPI, Request, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("./models/2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
@app.get("/ping1")
async def ping(request:Request):
    return f"your ip address is {request.client.host}"

def read_image (data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
     image  = read_image(await file.read())
     img_batch = np.expand_dims(image, 0)
     predictions = MODEL.predict(img_batch)
     predicted_class = CLASS_NAMES[np.argmax(predictions[0])] ##  np.argmax returns the index 
     confidence  = round(100 * (np.max(predictions[0])), 2)
     
     return {
         'class' : predicted_class,
         'confidence': float(confidence) 
     }
    
if __name__ == "__main__":
    uvicorn.run(app,host='0.0.0.0' ,port=7999) 
