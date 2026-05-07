<div align="center">

# 🐱 vs 🐶 Cat vs. Dog Image Classifier

### End-to-End Deep Learning Pipeline — From Data to Deployment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.x-FF6B6B?style=for-the-badge&logo=gradio&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📖 Project Overview

A complete **end-to-end Deep Learning project** that classifies images of cats and dogs using a custom-built **Convolutional Neural Network (CNN)**. This isn't just a model file — it's a **full production pipeline** that covers the entire machine learning lifecycle: automated data collection from Kaggle, rigorous data cleaning, model training with augmentation, and a fully deployed **Gradio web interface** hosted live on Hugging Face Spaces.

Whether you're a beginner exploring computer vision or a recruiter evaluating a full-stack ML project, this repository demonstrates a professional-grade workflow with clean code, proper documentation, and real-world deployment.

> 🚀 **[Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/rana789r/Cats-and-Dogs)** — Upload any cat or dog image and watch the model classify it in real time!

> 📂 **[View Model & Deployment Files on Hugging Face](https://huggingface.co/spaces/rana789r/Cats-and-Dogs/tree/main)** — The trained `.h5` model and deployment files are hosted here due to GitHub's file size limitations.

---

## ✨ Key Features

| # | Feature | Description |
|---|---------|-------------|
| 🔍 | **Automated Data Collection** | Downloads the dataset directly from Kaggle using the Kaggle API — no manual downloads or data uploads needed in the repo. |
| 🧹 | **Intelligent Data Cleaning** | A custom Python script using `os` and `PIL` scans every file and removes corrupted, invalid, or non-image files before training begins. |
| ✂️ | **Smart Dataset Splitting** | Uses `split-folders` to cleanly partition data into **Train**, **Validation**, and **Test** sets with controlled ratios. |
| 🔄 | **Data Augmentation** | Applies real-time augmentations — random rotations, width/height shifts, horizontal flips, and zooms — to artificially expand the training set and prevent overfitting. |
| 🧠 | **Custom CNN Architecture** | A from-scratch Sequential model built with multiple `Conv2D` + `MaxPooling2D` blocks, followed by `Flatten`, fully-connected `Dense` layers, and a `Dropout(0.5)` regularizer. |
| 🌐 | **Live Web Deployment** | The trained `.h5` model is wrapped in a **Gradio** web interface and deployed live on **Hugging Face Spaces** — accessible to anyone, anywhere. |

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Core programming language |
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white) / ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white) | Model building, training, and inference |
| **Web Interface** | ![Gradio](https://img.shields.io/badge/Gradio-FF6B6B?logo=huggingface&logoColor=white) | Interactive demo UI and Hugging Face deployment |
| **Notebook** | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) | Training pipeline, data processing, and experimentation |
| **Image Processing** | ![PIL](https://img.shields.io/badge/Pillow-8B6914?logo=python&logoColor=white) | Image validation and corruption detection |
| **Data Source** | ![Kaggle API](https://img.shields.io/badge/Kaggle_API-20BEFF?logo=kaggle&logoColor=white) | Automated dataset download |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white) | Training curves and data exploration |
| **Utilities** | `split-folders`, `os`, `numpy` | Data splitting, file operations, numerical computing |

---

## ⚙️ How It Works

### 1. Data Pipeline

```
Kaggle API  →  Raw Dataset  →  Data Cleaner (PIL + OS)  →  Split Folders  →  Train / Val / Test
```

The pipeline starts by pulling the dataset from Kaggle in a single command — no data files are stored in this repository. Before any training happens, a **custom data cleaning script** iterates through every file, attempting to open it with PIL. Any file that raises an exception — whether it's a truncated JPEG, a zero-byte file, or a disguised non-image — is immediately flagged and removed. This rigorous step prevents cryptic `OSError` crashes during `ImageDataGenerator`'s batching phase, which is one of the most common (and frustrating) pain points in real-world ML projects.

### 2. Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

The `ImageDataGenerator` applies a suite of stochastic transformations to each training image on-the-fly. This means the model never sees the exact same image twice, which drastically reduces overfitting and teaches the network to learn rotation-invariant, translation-invariant, and scale-invariant features — exactly the kind of robustness needed for real-world classification.

### 3. CNN Architecture

The heart of the classifier is a **custom Convolutional Neural Network** built from scratch:

```
Input (150×150×3)
    │
    ▼
┌─────────────────────┐
│  Conv2D (32 filters) │──▶ ReLU Activation
│  MaxPooling2D (2×2)  │
├─────────────────────┤
│  Conv2D (64 filters) │──▶ ReLU Activation
│  MaxPooling2D (2×2)  │
├─────────────────────┤
│  Conv2D (128 filters)│──▶ ReLU Activation
│  MaxPooling2D (2×2)  │
├─────────────────────┤
│  Flatten             │
│  Dense (512 units)   │──▶ ReLU Activation
│  Dropout (0.5)       │──▶ Regularization
│  Dense (1 unit)      │──▶ Sigmoid (Cat or Dog)
└─────────────────────┘
```

- **Conv2D layers** learn spatial hierarchies of features — edges in early layers, textures in middle layers, and semantic patterns (ears, eyes, fur) in deeper layers.
- **MaxPooling2D** progressively reduces the spatial dimensions, cutting computational cost and introducing translation invariance.
- **Dropout (0.5)** randomly deactivates 50% of neurons during training, forcing the network to learn redundant, robust representations instead of memorizing the training set.
- **Sigmoid output** produces a probability between 0 (cat) and 1 (dog).

### 4. Deployment with Gradio

The trained `.h5` model is loaded into `app.py`, a single file that handles three responsibilities:
1. **Model Loading** — Loads the trained Keras model from disk.
2. **Image Preprocessing** — Accepts user-uploaded images and reshapes/rescales them to match the model's expected input format.
3. **Gradio UI** — Renders a clean, interactive web interface where users can upload or drag-and-drop an image and instantly receive a classification result with confidence score.

The entire app is deployed on **[Hugging Face Spaces](https://huggingface.co/spaces/rana789r/Cats-and-Dogs/tree/main)** — no installation required for end users. Just open the link and start classifying!

> 💡 **Note on Model Hosting:** The trained model file (`model.h5`) exceeds GitHub's recommended file size limits, so it is hosted on Hugging Face Spaces alongside the deployment code. You can download it directly from the [Hugging Face repo](https://huggingface.co/spaces/rana789r/Cats-and-Dogs/tree/main).

---

## 🚀 Installation & How to Run Locally

### Prerequisites

- **Python 3.8+** installed
- A **Kaggle account** with an API key (`kaggle.json` in `~/.kaggle/`)
- (Optional) A **GPU** is recommended for faster training

### Step 1: Clone the Repository

```bash
git clone https://github.com/rana789r/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Make sure your Kaggle API key is set up:

```bash
# Place your kaggle.json at ~/.kaggle/kaggle.json
# Then run:
kaggle datasets download -d <dataset-name> -p ./data --unzip
```

### Step 5: Train the Model

Open the **`classifier.ipynb`** notebook and run all cells in order. The notebook handles everything:
- Data cleaning and corrupted file removal
- Dataset splitting into Train / Validation / Test sets
- Data augmentation setup
- CNN model building and compilation
- Model training with validation monitoring
- Saving the trained model as `model.h5`

### Step 6: Download the Pre-trained Model (Skip Step 5)

Don't want to train from scratch? Download the pre-trained model from Hugging Face:

```bash
# Download model.h5 from Hugging Face Spaces
wget https://huggingface.co/spaces/rana789r/Cats-and-Dogs/resolve/main/model.h5
```

### Step 7: Launch the Gradio App

```bash
python app.py
```

Open the local URL (usually `http://127.0.0.1:7860`) in your browser and start classifying!

---

## 📁 Folder Structure

### GitHub Repository (this repo)

```
cat-vs-dog-classifier/
│
├── app.py                  # Gradio web interface — loads model, preprocesses image, runs UI
├── classifier.ipynb        # Full training pipeline notebook (data, cleaning, augmentation, training)
├── requirements.txt        # All Python dependencies
├── README.md               # You are here 📍
└── .gitignore              # Excludes large model files and data directories
```

> **Note:** The dataset is **not** included in this repository. It downloads directly from Kaggle via the API during the training phase.

### Hugging Face Spaces (deployment + model)

```
Cats-and-Dogs/              # https://huggingface.co/spaces/rana789r/Cats-and-Dogs/tree/main
│
├── app.py                  # Gradio deployment file
├── model.h5                # Trained Keras model weights (too large for GitHub)
├── requirements.txt        # Dependencies for Hugging Face Spaces
└── README.md               # Hugging Face Spaces readme
```

> **Why two repos?** The trained model file (`model.h5`) exceeds GitHub's 100MB file size limit for direct pushes. Hugging Face Spaces handles large files natively and also provides free GPU/CPU hosting for the live demo.

---

## 🔮 Future Improvements

- [ ] **Transfer Learning:** Replace the custom CNN with a pre-trained model like **ResNet50**, **MobileNetV2**, or **EfficientNet** for significantly higher accuracy with less training time.
- [ ] **Multi-Class Classification:** Extend the project to classify more animal categories (birds, rabbits, horses, etc.).
- [ ] **Confidence Thresholding & Uncertainty:** Add a "not sure" category when the model's confidence is below a threshold, improving transparency.
- [ ] **Dockerized Deployment:** Package the entire application in a Docker container for reproducible, one-command deployment anywhere.
- [ ] **CI/CD Pipeline:** Automate testing, training, and deployment with GitHub Actions.
- [ ] **Model Optimization:** Apply quantization (TF Lite) to reduce the model size for edge and mobile deployment.
- [ ] **Explainability (Grad-CAM):** Add visual heatmaps showing which parts of the image the model focuses on for its prediction.

---

## 👤 Contact / Author

<div align="center">

**Rana**

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Spaces-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/rana789r/Cats-and-Dogs)

</div>

---

<div align="center">

**If you found this project useful, please consider giving it a ⭐ on GitHub!**

Made with ❤️ and lots of ☕

</div>
