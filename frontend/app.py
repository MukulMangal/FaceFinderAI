import requests
import streamlit as st
from pathlib import Path

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="FaceFinderAI",
    page_icon="📸",
    layout="wide"
)

# ==========================================
# Constants
# ==========================================

API_URL = "http://127.0.0.1:8000"

# ==========================================
# Load CSS
# ==========================================

css_file = Path("frontend/styles/style.css")

if css_file.exists():
    with open(css_file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================
# Header
# ==========================================

st.markdown("""
<div class="title">
📸 FaceFinderAI
</div>

<div class="subtitle">
AI Powered Wedding Photo Search Engine
</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.title("📊 Dashboard")

    st.success("Backend Connected")

    st.markdown("---")

    st.markdown("### Tech Stack")

    st.write("🐍 Python")
    st.write("⚡ FastAPI")
    st.write("🧠 InsightFace")
    st.write("🔍 FAISS")
    st.write("🎨 Streamlit")

    st.markdown("---")

    st.info("Version 2.0")

# ==========================================
# Main Layout
# ==========================================

left_col, right_col = st.columns([2, 1])

# Variables
data = None

# ==========================================
# LEFT COLUMN
# ==========================================

with left_col:

    st.markdown("## 📤 Upload Your Face")

    st.write(
        "Upload a clear photo of your face to search across all indexed wedding photographs."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        st.markdown("### 📸 Query Image")

        st.image(
            uploaded_file,
            width=350
        )

        search_clicked = st.button(
            "🔍 Search Wedding Photos",
            use_container_width=True,
            type="primary"
        )

        if search_clicked:

            with st.spinner("Searching for matching faces..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{API_URL}/search",
                    files=files
                )

                if response.status_code != 200:
                    st.error(response.text)
                    st.stop()

                data = response.json()

# ==========================================
# RIGHT COLUMN
# ==========================================

with right_col:

    st.markdown("## 📊 Search Stats")

    if data is None:

        st.info("Upload an image to begin searching.")

    else:

        st.metric(
            "Matches Found",
            data["total_matches"]
        )

        best_score = data["results"][0]["score"]

        st.metric(
            "Best Match",
            f"{best_score}%"
        )

        if best_score >= 80:
            st.success("✅ Face Found")
        else:
            st.warning("⚠ Low Confidence Match")

# ==========================================
# RESULTS
# ==========================================

if data is not None:

    st.success(
        f"Found {data['total_matches']} matching image(s)"
    )

    for result in data["results"]:

        match_number = data["results"].index(result)

        if match_number == 0:
            st.markdown("## 🥇 Best Match")
        elif match_number == 1:
            st.markdown("## 🥈 Match #2")
        elif match_number == 2:
            st.markdown("## 🥉 Match #3")
        else:
            st.markdown(f"## 🏅 Match #{match_number+1}")

        st.divider()

        image_col, info_col = st.columns([3, 1])

        with image_col:

            st.image(
                result["output_image"],
                use_container_width=True
            )

        with info_col:
            score = result["score"]

            st.metric(
                "Similarity",
                f"{score}%"
            )

            if score >= 90:
                st.success("🟢 Excellent Match")

            elif score >= 75:
                st.info("🔵 Strong Match")

            elif score >= 60:
                st.warning("🟡 Possible Match")

            else:
                st.error("🔴 Weak Match")

            st.write("### 📄 Filename")

            st.write(result["image_name"])

            image_path = result["output_image"]

            st.write(image_path)

            response = requests.get(image_path)

            st.write("Status Code:", response.status_code)

            image_bytes = response.content

            st.download_button(
                label="⬇ Download Image",
                data=image_bytes,
                file_name=result["image_name"],
                mime="image/jpeg",
                use_container_width=True,
                key=result["image_name"]
            )