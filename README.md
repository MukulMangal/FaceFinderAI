# FaceFinderAI 👤🔍

An AI-powered face recognition application that allows users to upload a face image and search for matching faces from an indexed image dataset. The project combines deep learning-based face embeddings with fast vector similarity search to deliver accurate and efficient face retrieval through an interactive web interface.

---

## ✨ Features

- Upload a face image and search for matching faces
- Face detection and embedding generation using InsightFace
- Fast similarity search using FAISS
- Interactive web interface built with Streamlit
- REST API powered by FastAPI
- Displays similarity scores and annotated search results

---

## 🛠️ Tech Stack

- **Language:** Python
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Face Recognition:** InsightFace
- **Similarity Search:** FAISS
- **Computer Vision:** OpenCV
- **Data Processing:** NumPy

---

## 📂 Project Structure

```text
FaceFinderAI/
│── api/
│── frontend/
│── dataset/
│── models/
│── detector.py
│── indexer.py
│── search.py
│── utils.py
│── config.py
│── requirements.txt
│── README.md
```

---

## ⚙️ How It Works

```text
Upload Image
      │
      ▼
Face Detection
(InsightFace)
      │
      ▼
Generate Face Embeddings
      │
      ▼
Search Similar Embeddings
(FAISS)
      │
      ▼
Return Matching Images
```

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/MukulMangal/FaceFinderAI.git
cd FaceFinderAI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the backend

```bash
uvicorn api.main:app --reload
```

### Launch the frontend

```bash
streamlit run frontend/app.py
```

---

## 📸 Screenshots

### Home Page

> *Add a screenshot here*

### Search Results

> *Add a screenshot here*

---

## 📌 Future Improvements

- Improve scalability for larger image datasets
- Cloud deployment
- Batch image uploads
- GPU acceleration
- Docker support
- User authentication

---

## 👨‍💻 Author

**Mukul Mangal**

- GitHub: https://github.com/MukulMangal
- LinkedIn: *(Add your LinkedIn profile here)*

---

## ⭐ If you found this project interesting, consider giving it a star!
