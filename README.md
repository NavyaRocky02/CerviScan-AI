# CerviScan AI – AI-Based Cervical Vertebrae Classification

## Project Overview

CerviScan AI is an experimental deep learning-based image classification application designed to classify cervical vertebrae into seven categories (C1–C7).

The project uses a Convolutional Neural Network (CNN) to process synthetic images and predict their corresponding vertebra class. A web-based interface developed using Streamlit allows users to upload images and view classification results.

**Note:** This project is an experimental research prototype trained on synthetic images. It is not clinically validated and must not be used for medical diagnosis.

## Objectives

- Develop an image classification system using deep learning.
- Classify images into seven categories (C1–C7).
- Apply image preprocessing techniques.
- Build an interactive web application for image classification.
- Visualize model performance using evaluation metrics.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Streamlit |
| Deep Learning Framework | TensorFlow, Keras |
| Machine Learning | Scikit-learn |
| Numerical Processing | NumPy, Pandas |
| Image Processing | Pillow |
| Data Visualization | Matplotlib |
| Model Architecture | Convolutional Neural Network (CNN) |

## Key Features

- Image upload and preview.
- Image resizing and preprocessing.
- CNN-based image classification.
- Classification into C1–C7 categories.
- Prediction confidence scores.
- Training performance visualization.
- Confusion matrix visualization.

## Project Structure

```text
CerviScan_AI/
│
├── models/
│   ├── best_model.keras
│   ├── cerviscan_cnn.keras
│   └── class_names.json
│
├── outputs/
│   ├── synthetic/
│   └── preprocessed/
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_performance.png
│   └── evaluation_results.json
│
├── app.py
├── check_dataset.py
├── dataset_analysis.py
├── dataset_visualization.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NavyaRocky02/CerviScan-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd CerviScan-AI
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Model Information

- Model Architecture: Convolutional Neural Network (CNN)
- Input Image Size: 224 × 224 pixels
- Image Format: Grayscale
- Number of Classes: 7
- Classification Labels: C1, C2, C3, C4, C5, C6, C7
- Model Format: Keras (.keras)

## Model Evaluation

The model is evaluated using a test dataset. Evaluation results and visualizations are stored in the `results` folder.

The current experimental model achieved approximately **14.29% test accuracy** in the reported evaluation.

Further improvements in dataset quality, class distinguishability, and model training are required.

## Future Enhancements

- Use real and properly annotated medical image datasets.
- Improve classification accuracy.
- Experiment with advanced CNN architectures.
- Apply data augmentation techniques.
- Perform more comprehensive model evaluation.
- Improve the web application interface.

## Disclaimer

CerviScan AI is an academic experimental prototype developed for educational and research purposes.

It uses synthetic images and has not been clinically validated. Its predictions must not be interpreted as medical findings or used for diagnosis, treatment, or clinical decision-making.

## Author

**Navya Sree Challa**

GitHub: [NavyaRocky02](https://github.com/NavyaRocky02)

---

**Project Name:** CerviScan AI  
**Domain:** Artificial Intelligence and Deep Learning  
**Application Type:** Image Classification Web Application
