# Gym Health Classifier – From Scratch

## Overview
This project is an end-to-end AI application that classifies a gym trainee’s health status as Healthy or Unhealthy based on physical and activity-related indicators.

The core focus of this project is implementing Logistic Regression completely from scratch, without using machine learning libraries such as scikit-learn.

The system includes data generation, model training from scratch, an interactive GUI, visual health reporting, and PDF report export.

---

## Model Details
Algorithm: Logistic Regression  
Implementation: From scratch using NumPy  
Loss Function: Binary Cross Entropy  
Optimization Method: Gradient Descent  
Task: Binary Classification (Healthy / Unhealthy)

No pre-built machine learning models were used.

---

## Features
- Logistic Regression implemented from scratch
- Balanced synthetic dataset generation
- Training with loss tracking
- Interactive Streamlit GUI
- BMI calculation and interpretation
- BMI range visualization
- Prediction confidence visualization
- Explainable AI output with analysis and reasons
- Exportable PDF health report

---

## How It Works
1. The user enters physical and activity-related data.
2. The input data is normalized.
3. The trained logistic regression model predicts health status.
4. A detailed visual health report is generated.
5. The user can export the report as a PDF file.

---

## Technologies Used
Python, NumPy, Pandas, Streamlit, Matplotlib, ReportLab

---

## Project Structure
gym-health-classifier-from-scratch/  
app/ → app.py (Streamlit application with report and PDF export)  
data/ → generate_data.py (synthetic balanced dataset generation)  
model/ → logistic_regression.py, health_model.pkl, loss_curve.png  
train/ → train.py (model training script)  
reports/ → generated PDF health reports  
requirements.txt  
README.md  
.gitignore  

---

## Installation
Install dependencies using: pip install -r requirements.txt

---

## How to Run
Generate the dataset by running:  
cd data  
python generate_data.py  

Train the model from the project root:  
python -m train.train  

Run the application:  
streamlit run app/app.py  

---

## Output
The application provides health classification results, confidence scores, visual analysis charts, textual explanations, personalized recommendations, and a downloadable PDF health report.

---

## Disclaimer
This project is for educational and demonstration purposes only and does not replace professional medical or fitness advice.

---

## Author
Developed as a from-scratch AI portfolio project demonstrating machine learning fundamentals, manual model implementation, and end-to-end AI system development.
