"""
config.py

Central configuration for FaceFinderAI.
"""

# ==========================
# Paths
# ==========================

DATASET_DIR = "dataset/wedding_images"

EMBEDDINGS_DIR = "embeddings"

OUTPUT_DIR = "output"

TEMP_DIR = "temp"

FAISS_INDEX_PATH = "embeddings/faiss.index"

METADATA_PATH = "embeddings/metadata.json"

EMBEDDINGS_PATH = "embeddings/embeddings.npy"


# ==========================
# Face Recognition
# ==========================

MODEL_NAME = "buffalo_l"

DETECTION_SIZE = (640, 640)

TARGET_IMAGE_WIDTH = 800

SIMILARITY_THRESHOLD = 0.50

TOP_K = 10


# ==========================
# Drawing
# ==========================

BOX_COLOR = (0, 255, 0)

BOX_THICKNESS = 3

FONT_SCALE = 0.7