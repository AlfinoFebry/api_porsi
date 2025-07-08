from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cart.model import predict_from_input
from PIL import Image
import io
import numpy as np
import cv2
import re
import difflib
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
        "Pendidikan_Agama": ["agama", "islam", "krist", "katolik", "hindu", "budha"],
        "Pkn": ["kewa", "negara", "pendidikan", "panca"],
        "Bahasa_Indonesia": ["indo", "dones", "bahasa indo"],
        "Matematika_Wajib": ["mate", "math", "atika", "matematika", "matika umum", "matika"],
        "Sejarah_Indonesia": ["sejarah", "seja"],
        "Bahasa_Inggris": ["ingg", "nggr", "gris", "bahasa inggris"],
        "Seni_Budaya": ["seni", "budaya", "uday"],
        "Penjaskes": ["jasm", "penj", "olahr", "raga"],
        "PKWu": ["prak", "wira", "karya"],
        "Mulok": ["muat", "ulok" "daerah"],
        "Matematika_Peminatan": ["mat pemi", "pemina", "emina", "minat", "atika pem", "matematika peminatan"],
        "Biologi": ["biol", "iolo", "olog", "iogi", "biologi"],
        "Fisika": ["fisi", "isik", "sika" , "fisika"],
        "Kimia": ["kimi", "imia", "kima" , "kimia"],
        "Geografi": ["geog", "eogr", "graf", "eofi" , "geografi"],
        "Sejarah_Minat": ["sejarah m", "jarah m" , "sejarah peminatan"],
        "Sosiologi": ["sosi", "osio", "sosiologi"],
        "Ekonomi": ["eko", "kono", "onom", "nomi", "ekonomi"],
    }

    lines = text.lower().split('\n')
    for i, line in enumerate(lines):
        for field, keywords in keyword_map.items():
            for keyword in keywords:
                words = line.split()
                matches = difflib.get_close_matches(keyword, words, n=1, cutoff=0.75)
                if matches:
                    score = re.findall(r"\b(\d{2,3})\b", line)
                    if not score and i + 1 < len(lines):
                        score = re.findall(r"\b(\d{2,3})\b", lines[i + 1])
                    if score:
                        data[field] = float(score[0])
                        break

    ipa_keys = ["Matematika_Peminatan", "Biologi", "Fisika", "Kimia"]
    ips_keys = ["Geografi", "Sejarah_Minat", "Sosiologi", "Ekonomi"]

    jurusan = None
    for line in lines:
        if "ipa" in line:
            jurusan = "IPA"
            break
        elif "mia" in line:
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
        if any(keyword in line for keyword in ["semester 2", "semester dua", "sem 2", "semester genap"]):
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
        "Nilai": data,
        "Raw Text": text
    }

@app.middleware("http")
async def restrict_to_post_only(request: Request, call_next):
    if request.method != "POST":
        return JSONResponse(
            status_code=405,
            content={"detail": f"Method {request.method} not allowed. Only POST is allowed."}
        )
    return await call_next(request)

# @app.post("/ocr")
# async def ocr(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         npimg = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#         # Step 1: Resize image (scale up)
#         scale_percent = 150
#         width = int(img.shape[1] * scale_percent / 100)
#         height = int(img.shape[0] * scale_percent / 100)
#         img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

#         # Step 2: Denoise
#         img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

#         # Step 3: Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Step 4: Adaptive Gaussian Threshold
#         thresh = cv2.adaptiveThreshold(
#             gray, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             11, 2
#         )

#         # Step 5: OCR with allowlist
#         result = reader.readtext(
#             thresh,
#             detail=0,
#             paragraph=False,
#             allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#         )

#         # Combine OCR result
#         text = "\n".join(result)
#         processed = extract_scores_from_text(text)
#         return processed

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": "OCR failed", "details": str(e)})

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

        result = reader.readtext(
            thresh,
            detail=0,
            paragraph=False,
            allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )
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
