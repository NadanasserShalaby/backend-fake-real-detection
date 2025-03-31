from fastapi import FastAPI, File, UploadFile
import PyPDF2
import os
import cv2
import librosa
import numpy as np
from PIL import Image
import piexif
from pydub import AudioSegment
import shutil
from pydub.exceptions import CouldntDecodeError  # Handle conversion errors

# os.system("ffmpeg -version")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===================== PDF DETECTION FUNCTIONS ===================== #

def is_pdf_file(file_path):
    """Check if file starts with PDF signature (%PDF-)"""
    try:
        with open(file_path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except:
        return False

def is_valid_pdf(file_path):
    """Check if PDF is readable (not corrupt)"""
    try:
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            return len(pdf.pages) > 0
    except:
        return False

def has_valid_metadata(file_path):
    """Check for valid metadata (Fake PDFs may lack metadata)"""
    try:
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            metadata = pdf.metadata
            return bool(metadata and metadata.get("/Producer"))
    except:
        return False

def has_text_content(file_path):
    """Check if text is extractable (Real PDFs usually contain searchable text)"""
    try:
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            text = pdf.pages[0].extract_text()
            return bool(text and text.strip())  # True if text exists
    except:
        return False

def is_encrypted(file_path):
    """Check if PDF is encrypted (Fake PDFs may be locked)"""
    try:
        with open(file_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            return pdf.is_encrypted
    except:
        return False

# ===================== IMAGE DETECTION FUNCTIONS ===================== #

def is_valid_image(file_path):
    """Check if the file is a valid image"""
    try:
        img = Image.open(file_path)
        img.verify()  # Verifies if it's a real image
        return True
    except:
        return False

def has_metadata(file_path):
    """Check if EXIF metadata exists (Fake images often lack metadata)"""
    try:
        img = Image.open(file_path)
        exif_data = img.info.get("exif")
        return exif_data is not None
    except:
        return False

def detect_compression_artifacts(file_path):
    """Detects excessive compression artifacts (Fake images may be highly compressed)"""
    img = cv2.imread(file_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Edge detection (JPEG artifacts cause noise)
    
    # If too many edges are detected, assume heavy compression (Fake)
    return np.count_nonzero(edges) > 7000  # Tuned threshold

# ===================== AUDIO DETECTION FUNCTIONS ===================== #

def is_ffmpeg_installed():
    """Check if FFmpeg is installed"""
    try:
        AudioSegment.ffmpeg
        return True
    except:
        return False

def convert_to_wav(file_path):
    """Convert audio to WAV format"""
    if file_path.endswith(".wav"):
        return file_path  # Already WAV

    new_path = file_path.rsplit(".", 1)[0] + ".wav"
    
    try:
        # ✅ Check if ffmpeg is installed
        if not is_ffmpeg_installed():
            print("❌ FFmpeg is missing! Install it from https://ffmpeg.org/download.html")
            return None

        audio = AudioSegment.from_file(file_path)
        audio.export(new_path, format="wav")
        return new_path
    except CouldntDecodeError:
        print(f"❌ Error: Unsupported audio format for {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error converting to WAV: {e}")
        return None

def extract_mfcc(file_path):
    """Extract MFCC features, handling errors properly"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # ✅ Debugging: Check audio length
        print(f"📏 Audio Length: {len(y)/sr} seconds")

        if len(y) < sr:  # Less than 1 second of audio
            print("⚠️ Audio too short for processing")
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # ✅ Debugging: Print extracted MFCC features
        print(f"📊 MFCC Shape: {mfcc.shape}")
        print(f"📊 MFCC Mean: {np.mean(mfcc, axis=1)}")

        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"❌ Error extracting MFCC: {e}")
        return None

def detect_ai_voice(file_path):
    """Detect AI-generated voice using MFCC patterns"""
    features = extract_mfcc(file_path)

    if features is None:
        print("⚠️ No MFCC features extracted, likely silent or corrupted audio")
        return None

    avg_mfcc = np.mean(features)

    # ✅ Debugging: Print MFCC Mean
    print(f"📊 MFCC Mean: {avg_mfcc}")

    # ✅ Adjust threshold dynamically (Experiment with values)
    if avg_mfcc > -20:  # Adjust based on test data
        return True  # AI-generated
    else:
        return False  # Real human voice

# ===================== ENDPOINTS ===================== #

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Detects if a PDF is fake or real"""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    is_real = (
        is_pdf_file(file_path)
        and is_valid_pdf(file_path)
        and has_valid_metadata(file_path)
        and has_text_content(file_path)
        and not is_encrypted(file_path)  # Fake PDFs might be locked
    )

    os.remove(file_path)  # Cleanup

    return {"filename": file.filename, "is_real": is_real, "file_type": "pdf"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """Detects if an image is fake or real"""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    valid_image = is_valid_image(file_path)
    has_meta = has_metadata(file_path)
    compression_issues = detect_compression_artifacts(file_path)

    # Adjusted logic: Only mark fake if **multiple** issues are found
    if not valid_image:
        is_real = False
    elif has_meta and not compression_issues:
        is_real = True  # Likely real
    elif not has_meta and compression_issues:
        is_real = False  # Likely fake
    else:
        is_real = True  # If mixed results, assume real

    os.remove(file_path)  # Cleanup

    return {"filename": file.filename, "is_real": is_real, "file_type": "image"}



@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Detects if an audio recording is real or AI-generated"""
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # ✅ Check file size (avoid processing empty files)
    if os.stat(file_path).st_size == 0:
        return {"error": "Uploaded file is empty"}

    wav_path = convert_to_wav(file_path)
    if not wav_path:
        return {"error": "Failed to convert audio to WAV format"}

    is_real = detect_ai_voice(wav_path)

    if is_real is None:
        return {"error": "Failed to process audio"}

    os.remove(wav_path)  # Cleanup

    return {"filename": file.filename, "is_real": not is_real, "file_type": "audio"}
