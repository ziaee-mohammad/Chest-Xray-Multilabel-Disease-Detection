import io
from PIL import Image
import joblib
import os
import templates
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Form, Depends, Request, BackgroundTasks,UploadFile,File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from predictions import load_model,visualize_gradcams
model_path="model.pth"
model = load_model(model_path)  
UPLOAD_DIR = Path(r"static/uploaded_images")  # Create a folder named 'uploaded_images' in your project directory
UPLOAD_DIR.mkdir(exist_ok=True)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_output_images():
    output_dir = 'static/Output'
    images = []

    for filename in os.listdir(output_dir):
        if filename.lower().endswith(('.png')):
            images.append(os.path.join(output_dir, filename).replace("\\", "/"))

    return images


@app.get("/", response_class=HTMLResponse)
def display_landing_page(request: Request):
    return templates.TemplateResponse("landing_page1.html", {"request": request})


@app.get("/analyze", response_class=HTMLResponse)
def analyze_images(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.post("/upload")
async def upload_image(request: Request, image: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await image.read()
        image_data = Image.open(io.BytesIO(contents))
        print(image_data)
        # Define the path to save the image
        file_path = UPLOAD_DIR / image.filename

        with open(file_path, "wb") as f:
            f.write(contents)
        image_url = f"/static/uploaded_images/{image.filename}"




        disease_prob = visualize_gradcams(model, file_path)
        print("\nProbabilities array:", disease_prob)
        top_preds = [(disease, float(prob)) for disease, prob in disease_prob]
        print(top_preds)
        images=get_output_images()
        print(images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return templates.TemplateResponse("analysis.html", {"request": request,"image_url":image_url,"images":images,"top_preds":top_preds})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)