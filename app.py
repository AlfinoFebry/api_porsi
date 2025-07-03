from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cart.model import predict_from_input
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://porsi.me", "http://localhost:3000"],  # Or set specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocessing
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"text": generated_text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/cart")
def predict(input: InputData):
    try:
        result = predict_from_input(input.dict())
        return {"prediction": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "details": str(e)})
