from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cart.model import predict_from_input
from PIL import Image
import io
import numpy as np
import cv2
import re
import easyocr

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://porsi.me", "http://localhost:3000"],  # Or set specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(['en'], gpu=False)

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

def extract_scores_from_text(text):
    data = {
        "Pendidikan_Agama": -1,
        "Pkn": -1,
        "Bahasa_Indonesia": -1,
        "Matematika_Wajib": -1,
        "Sejarah_Indonesia": -1,
        "Bahasa_Inggris": -1,
        "Seni_Budaya": -1,
        "Penjaskes": -1,
        "PKWu": -1,
        "Mulok": -1,
        "Matematika_Peminatan": -1,
        "Biologi": -1,
        "Fisika": -1,
        "Kimia": -1,
        "Lintas_Minat": -1,
        "Geografi": -1,
        "Sejarah_Minat": -1,
        "Sosiologi": -1,
        "Ekonomi": -1
    }

    keyword_map = {
        "Pendidikan_Agama": ["agama"],
        "Pkn": ["kewa", "pkn"],
        "Bahasa_Indonesia": ["indo", "ndon", "ndon", "hasa"],
        "Matematika_Wajib": ["mate", "math", "atika"],
        "Sejarah_Indonesia": ["sejarah indo", "jarah in"],
        "Bahasa_Inggris": ["ingg", "nggr", "gris"],
        "Seni_Budaya": ["seni", "budaya", "uday"],
        "Penjaskes": ["jasm", "penj", "olahr"],
        "PKWu": ["prak", "wira", "kwu"],
        "Mulok": ["muat", "ulok"],
        "Matematika_Peminatan": ["matika pemi"],
        "Biologi": ["biol", "iolo", "olog", "iogi"],
        "Fisika": ["fisi", "isik", "sika"],
        "Kimia": ["kimi", "imia", "kima"],
        "Geografi": ["geog", "eogr", "graf", "eofi"],
        "Sejarah_Minat": ["sejarah m", "jarah m"],
        "Sosiologi": ["sosi", "osio", "iolo"],
        "Ekonomi": ["eko", "kono", "onom", "nomi"]
    }

    lines = text.lower().split('\n')
    for line in lines:
        for field, keywords in keyword_map.items():
            if any(fragment in line for fragment in keywords):
                score = re.findall(r"\\b(\\d{2,3})\\b", line)
                if score:
                    data[field] = float(score[0])

    ipa_keys = ["Matematika_Peminatan", "Biologi", "Fisika", "Kimia"]
    ips_keys = ["Geografi", "Sejarah_Minat", "Sosiologi", "Ekonomi"]

    jurusan = None
    for line in lines:
        if "ipa" in line:
            jurusan = "IPA"
            break
        elif "ips" in line:
            jurusan = "IPS"
            break

    if not jurusan:
        ipa_count = sum(1 for k in ipa_keys if data[k] > 0)
        jurusan = "IPA" if ipa_count > 2 else "IPS"

    lintas = 0
    if jurusan == "IPA":
        for k in ips_keys:
            if data[k] > 0:
                lintas += data[k]
                data[k] = -1
    else:
        for k in ipa_keys:
            if data[k] > 0:
                lintas += data[k]
                data[k] = -1
    data["Lintas_Minat"] = round(lintas / 1, 1) if lintas > 0 else -1

    kelas = "X"
    semester = 1
    semester_hint = 1

    for line in lines:
        if "semester 2" in line or "semester genap" in line:
            semester_hint = 2

        if "xi" in line:
            kelas = "XI"
        elif "xii" in line:
            kelas = "XII"

    if kelas == "X":
        semester = semester_hint
    elif kelas == "XI":
        semester = 2 + semester_hint
    elif kelas == "XII":
        semester = 4 + semester_hint

    return {
        "Jurusan": jurusan,
        "Kelas": kelas,
        "Semester": semester,
        "Nilai": data
    }

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        result = reader.readtext(thresh, detail=0)
        text = "\n".join(result)
        processed = extract_scores_from_text(text)
        return processed
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "OCR failed", "details": str(e)})


@app.post("/cart")
def predict(input: InputData):
    try:
        result = predict_from_input(input.dict())
        return {"prediction": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Prediction failed", "details": str(e)})
