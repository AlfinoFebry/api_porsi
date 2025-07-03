from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cart.model import predict_from_input
from PIL import Image
import io
import numpy as np
import cv2
from paddleocr import PaddleOCR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://porsi.me", "http://localhost:3000"],  # Or set specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr = PaddleOCR(lang="en", use_angle_cls=True)

def preprocess_for_paddle(image: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

class InputData(BaseModel):
    JK: str
    Jurusan_SMA: str
    Pendidikan_Agama: float
    Pkn: float
    Bahasa_Indonesia: float
    Matematika_Wajib: float
    Sejarah_Indonesia: float
    Bahasa_Inggris: float
    Seni_Budaya: float
    Penjaskes: float
    PKWu: float
    Mulok: float
    Matematika_Peminatan: float
    Biologi: float
    Fisika: float
    Kimia: float
    Lintas_Minat: float
    Geografi: float
    Sejarah_Minat: float
    Sosiologi: float
    Ekonomi: float
    Hobi: str

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Convert PIL Image to numpy array
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run OCR
        result = ocr.ocr(img_np)

        # Extract texts
        texts = []
        for line in result[0]:  # result is a list with one item (for single image)
            text = line[1][0]   # line[1] = ("text", confidence)
            texts.append(text)

        return {"text": "\n".join(texts)}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "OCR failed", "details": str(e)})

    
@app.post("/cart")
def predict(input: InputData):
    try:
        result = predict_from_input(input.dict())
        return {"prediction": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "details": str(e)})
