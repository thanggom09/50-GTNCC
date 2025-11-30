import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import io
import cv2
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import gdown

# ================================
# 1. CÀI ĐẶT THÔNG SỐ MODEL
# ================================
models_info = {
    "ResNet50": {
        "path": "resnet50_rice_leaf.pth",
        "url": "https://drive.google.com/uc?id=1FrF_teTUh3lzb0mlwduwRU6pq4t4S6Lp",
        "constructor": lambda: __import__('torchvision.models').models.resnet50(weights=None)
    },
    "ViT": {
        "path": "ViT_rice_leaf.pth",
        "url": "https://drive.google.com/uc?id=1hVFE1nXSyn61fXoGug5yHEPyZEryO8KA",
        "constructor": lambda: __import__('torchvision.models').models.vit_b_16(weights=None)
    }

}

num_classes = 8
disease_labels = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# ================================
# 2. CHỌN MODEL TRONG APP
# ================================
st.sidebar.title("Chọn Model")
selected_model_name = st.sidebar.selectbox("Model dùng để dự đoán:", list(models_info.keys()))
model_info = models_info[selected_model_name]

# Tải model nếu chưa có
os.makedirs("model", exist_ok=True)
if not os.path.exists(model_info["path"]):
    st.info(f"Đang tải {selected_model_name} từ Google Drive...")
    gdown.download(model_info["url"], model_info["path"], quiet=False)

# Khởi tạo model
try:
    model = model_info["constructor"]()
    if selected_model_name.startswith("ResNet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # ViT
        in_features = model.heads[1].in_features
        model.heads = nn.Linear(in_features, num_classes)

    state = torch.load(model_info["path"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
except Exception as e:
    st.error(f"❌ Lỗi khi load model: {e}")

# ================================
# 3. TIỀN XỬ LÝ ẢNH CHUNG
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    return transform(image).unsqueeze(0)

# ================================
# 4. SAVE ẢNH THEO BỆNH
# ================================
def save_image(image_data, disease_name):
    disease_folder = os.path.join("images", disease_name)
    os.makedirs(disease_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(disease_folder, f"{timestamp}.jpg")
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    return image_path

# ================================
# 5. CSS
# ================================
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ================================
# 6. GIAO DIỆN STREAMLIT
# ================================
st.title("🌾 Phân Loại Bệnh Lá Lúa (PyTorch)")

with st.sidebar:
    option = option_menu(
        "MENU",
        ["Tải lên ảnh", "Chụp ảnh"],
        icons=["upload", "camera"],
        menu_icon="cast",
        default_index=0,
    )

# ================================
# TẢI ẢNH LÊN
# ================================
if option == "Tải lên ảnh":
    uploaded_image = st.file_uploader("Chọn ảnh lá lúa:", type=["jpg","jpeg","png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Predict
        img_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0].numpy()

        # Show result
        st.write("### 🔍 Kết quả dự đoán:")
        for idx, p in sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True):
            st.write(f"{disease_labels[idx]}: **{p*100:.2f}%**")

        # Save image
        predicted_label = disease_labels[np.argmax(probs)]
        save_image(uploaded_image.getvalue(), predicted_label)

# ================================
# CHỤP ẢNH WEBCAM
# ================================
elif option == "Chụp ảnh":

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Predict
            img_tensor = preprocess_image(pil_img)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0].numpy()

            label = disease_labels[np.argmax(probs)]

            # Draw label trên ảnh
            cv2.putText(img, f"{label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            return img

    webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)
