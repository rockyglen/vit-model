import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model from Hugging Face (cached on first run)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="glen-louis/cat-dogs", filename="vit_model.pth")
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# UI
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image and I’ll tell you if it's a cat or dog!")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, 1).item()

    st.markdown(f"### Prediction: **{'Dog' if pred == 1 else 'Cat'}**")
