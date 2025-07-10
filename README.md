# 🐱🐶 Cat vs Dog Classifier (ViT-based)

This project demonstrates an end-to-end pipeline for training and deploying a **Vision Transformer (ViT)** model to classify images as either **Cat** or **Dog** using the [PetImages dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765). The model is trained using PyTorch and deployed via a simple **Streamlit** web app.

The trained model is **hosted on Hugging Face Hub**, which allows the Streamlit app to dynamically download and load it at runtime.



## 🧠 Model Overview

- **Architecture**: Vision Transformer (ViT-B/16) from the `timm` library
- **Dataset**: 25,000+ labeled cat and dog images
- **Training Split**: 70% train, 15% validation, 15% test
- **Accuracy**: 
  - Validation Accuracy: **98.85%**
  - Test Accuracy: **98.77%**
- **Model Hosting**: [Hugging Face Hub](https://huggingface.co/) (`glen-louis/cat-dogs`)






## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cat-dog-vit.git
cd cat-dog-vit
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # for macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
