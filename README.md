# PlantHealth AI

PlantHealth AI is an image classification application that identifies plant diseases from leaf images.

## Project Overview

The project uses a deep learning model based on MobileNetV2 to classify plant leaf images into different plant and disease categories.

The model was trained using the PlantVillage dataset containing 38 classes.

## Features

- Upload a plant leaf image
- Identify the plant
- Identify the detected disease or healthy condition
- Display prediction confidence
- Display the top 3 predictions
- Simple Streamlit web interface

## Technologies Used

- Python
- TensorFlow
- MobileNetV2
- Streamlit
- NumPy
- Pandas
- Pillow
- Matplotlib
- Scikit-learn

## Dataset

PlantVillage dataset with 38 plant/disease classes.

## Model

The application uses a MobileNetV2 transfer-learning model.

Input image size:

224 × 224 pixels

Validation accuracy:

92.53%

## How to Run

1. Open the project folder in VS Code.
2. Activate the virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt