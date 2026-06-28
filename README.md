# 👁️ Cataract Early Detection System (ResNet-152V2 & Flask)

An intelligent, deep learning-based clinical screening tool designed to detect cataract levels from eye images. Powered by a fine-tuned **ResNet-152V2** convolutional neural network, the system classifies eye images into three clinical stages: **No Cataract**, **Early Cataract**, or **Cataract**. The solution integrates a robust training pipeline with a modern, glassmorphic Flask web interface for instantaneous screening and diagnostic recommendations.

---

## 📌 Features

- **Multi-Class Early Detection:** Classifies images into three distinct categories:
  - `no_cataract`: Clean diagnosis.
  - `early_cataract`: Flags early-stage signs for proactive monitoring.
  - `cataract`: Directs immediate consultation for advanced cataracts.
- **Deep Learning Model (ResNet-152V2):** Leverages transfer learning on a state-of-the-art 152-layer deep residual network, pre-trained on ImageNet.
- **Robust Training Pipeline:** Implements customized real-time data augmentations, learning rate adjustments, early stopping, and automatic best-model checkpoint saving.
- **Modern Web Interface:** Built using Flask and a sleek, premium, glassmorphism-based CSS layout that accepts image uploads and produces results dynamically.
- **Actionable Clinical Cautions:** Automatically provides appropriate guidance and recommendation severity warnings based on prediction outcomes.

---

## 🏗️ Project Architecture

```filepath
cataract_early_detection_ResNet/
├── data/                       # Dataset directory (must contain subfolders for classes)
│   ├── cataract/               # Images containing mature cataracts
│   ├── early_cataract/         # Images containing early-stage cataracts
│   └── no_cataract/            # Images of healthy eyes
├── model/                      # Saved deep learning models
│   ├── best_resnet_model.h5    # Best checkpoint during training (highest val_accuracy)
│   └── resnet_model.h5         # Final saved fine-tuned model used for predictions
├── static/                     # Static web assets
│   ├── css/
│   │   └── style.css           # Styling with modern glassmorphism
│   ├── images/                 # Backgrounds and decorative images
│   └── uploads/                # Temporary storage for uploaded query images
├── templates/                  # Flask HTML view templates
│   ├── home.html               # Welcome portal and entry point
│   ├── predict.html            # File upload forms & prediction gateway
│   └── result.html             # Classification results & diagnostic cautions
├── app.py                      # Flask web application entry-point & model loading
├── train_model.py              # Model construction, augmentation, and training script
└── requirements                # Python dependency configuration
```

---

## 🛠️ Tech Stack & Dependencies

- **Backend Framework:** [Flask (2.2.5)](https://flask.palletsprojects.com/)
- **Deep Learning Framework:** [TensorFlow (2.12.0)](https://www.tensorflow.org/) & Keras
- **Numerical Processing:** [NumPy](https://numpy.org/)
- **Image Preprocessing:** [Pillow](https://python-pillow.org/) & [OpenCV](https://opencv.org/)
- **File I/O Helper:** [Werkzeug](https://werkzeug.palletsprojects.com/)

---

## 🚀 Getting Started

Follow the instructions below to set up your environment, train the model, and launch the web interface.

### 1. Prerequisites
- Python 3.8 to 3.10 is recommended (for full compatibility with TensorFlow 2.12).
- pip (Python Package Installer).

### 2. Environment Setup
Clone or navigate to the workspace, create a virtual environment, and install dependencies:

```bash
# Navigate to the project root directory
cd cataract_early_detection_ResNet

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install the required python packages
pip install -r requirements
```

### 3. Dataset Configuration
Organize your clinical cataract images under the `data/` directory using the following structure:
```text
data/
├── cataract/
│   ├── image1.jpg
│   └── image2.jpg
├── early_cataract/
│   ├── image3.jpg
│   └── image4.jpg
└── no_cataract/
    ├── image5.jpg
    └── image6.jpg
```
*Note: Make sure your dataset contains high-quality close-ups of the iris and lens for optimal training accuracy.*

---

## 🏋️ Model Training

To train the classification model from scratch using your dataset under `data/`, run:

```bash
python train_model.py
```

### Model & Training Parameters:
- **Base Architecture:** `ResNet152V2` initialized with pre-trained `ImageNet` weights.
- **Transfer Learning Strategy:** The base model layers are frozen, except for the last 30 layers which are unfrozen to allow fine-tuning on cataract-specific eye details.
- **Data Augmentation:** Real-time generation using `ImageDataGenerator` with features including:
  - 20% validation split (`validation_split=0.2`)
  - Random rotations (`rotation_range=20`)
  - Zoom scaling (`zoom_range=0.2`)
  - Horizontal flips (`horizontal_flip=True`)
  - Rescaling pixels to `[0, 1]` (`rescale=1./255`)
- **Optimization:** Trained using the Adam optimizer with a custom learning rate (`1e-5`) and Categorical Crossentropy loss.
- **Callbacks:**
  - `EarlyStopping`: Stops training early if validation accuracy fails to improve for 5 consecutive epochs (`patience=5`), preventing overfitting.
  - `ModelCheckpoint`: Automatically saves the model with the highest validation accuracy to `model/best_resnet_model.h5`.
- **Output:** The final fine-tuned model (without optimizer weights to reduce storage size) is saved directly to `model/resnet_model.h5` for fast loading in the Flask application.

---

## 🌐 Running the Web Application

Launch the Flask web app locally to screen eye images using the trained model:

```bash
python app.py
```

1. Once the application starts, open your browser and navigate to: **`http://127.0.0.1:5000/`**
2. Click **Start Screening** to navigate to the upload page.
3. Upload an eye image (e.g., JPEG/PNG format).
4. The system will pre-process the image, run inference using the loaded model, and display the predicted diagnostic classification along with medical recommendations.

---

## ⚕️ Medical Disclaimer

> [!IMPORTANT]
> **This system is an AI-based clinical screening tool intended for academic, educational, and research purposes only.** It does not constitute medical advice, professional diagnosis, opinion, treatment, or services. Machine learning models can produce false positives and false negatives. Always consult a certified healthcare professional or ophthalmologist for clinical diagnoses and treatment of eye conditions.
