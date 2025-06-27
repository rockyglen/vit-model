import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download

# Configure page
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="glen-louis/cat-dogs",  
        filename="vit_model.pth"
    )
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Header
st.markdown(
    "<h1 style='text-align: center;'>Cat vs Dog Image Classifier</h1>",
    unsafe_allow_html=True
)

st.write(
    "This application uses a fine-tuned Vision Transformer (ViT) model to classify uploaded images as either a cat or a dog."
)

st.markdown("---")

# File uploader
file = st.file_uploader(
    "Upload an image (JPG, PNG, BMP, TIFF, WEBP):",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"]
)

# Prediction logic
if file is not None:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown("### Prediction")

        # Preprocess and predict
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

        class_labels = ["Cat", "Dog"]
        confidence = probabilities[predicted_class].item()

        # Confidence threshold to determine if image might be neither
        confidence_threshold = 0.75

        if confidence < confidence_threshold:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold; color: orange;'>Class: Neither Cat nor Dog</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>The model is not confident enough to classify this image as a cat or a dog (Confidence: {confidence:.2%}).</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold;'>Class: {class_labels[predicted_class]}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>Confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image or is corrupted.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
