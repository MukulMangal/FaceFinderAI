"""
main.py

FastAPI backend for FaceFinderAI.
"""

import os
import shutil

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.indexer import index_dataset
from src.search import search_faces


app = FastAPI(
    title="FaceFinderAI",
    version="2.0",
    description="Face Recognition Search Engine using InsightFace + FAISS",
)
app.mount(
    "/output",
    StaticFiles(directory="output"),
    name="output"
)

# ======================================================
# CORS
# ======================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================
# Health Check
# ======================================================

@app.get("/health", tags=["System"])
def health():
    """
    Check if the backend is running.
    """

    return {
        "status": "alive",
        "message": "FaceFinderAI Backend Running"
    }


# ======================================================
# Index Dataset
# ======================================================

@app.post("/index", tags=["Indexing"])
def index_images():
    """
    Index all images inside dataset/wedding_images.
    """

    try:

        index_dataset("dataset/wedding_images")

        return {
            "success": True,
            "message": "Dataset indexed successfully."
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ======================================================
# Search
# ======================================================

@app.post("/search", tags=["Search"])
def search(file: UploadFile = File(...)):
    """
    Search for matching wedding photos using a query image.
    """

    if file.filename is None:

        raise HTTPException(
            status_code=400,
            detail="No filename received."
        )

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in [".jpg", ".jpeg", ".png"]:

        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG files are supported."
        )

    os.makedirs("temp", exist_ok=True)

    temp_path = os.path.join(
        "temp",
        file.filename
    )

    try:

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = search_faces(temp_path)

        formatted = []

        for result in results:

            formatted.append(
    {
        "image_name": os.path.basename(result["image_path"]),
        "image_path": result["image_path"],
        "score": round(float(result["score"]) * 100, 2),
        "bbox": [float(x) for x in result["bbox"]],
        "output_image": f"http://127.0.0.1:8000/{result['output_image'].replace(os.sep, '/')}",
    }
)

        return {
            "success": True,
            "total_matches": len(formatted),
            "results": formatted,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)