# ---------- PATH FIX FOR STREAMLIT ----------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------- IMPORTS ----------
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------- SESSION STATE ----------
if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# ---------- LOAD MODEL ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "health_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Gym Health Report",
    layout="wide"
)

# ---------- HEADER ----------
st.title("Gym Health Assessment Report")
st.write(
    """
    This report evaluates general fitness health using a
    **Logistic Regression model implemented from scratch**.
    """
)

st.divider()

# ---------- INPUT SECTION ----------
st.subheader("1. Personal Information")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age (years)", 18, 60, 25)
    height = st.number_input("Height (cm)", 150, 200, 170)

with c2:
    weight = st.number_input("Weight (kg)", 40.0, 180.0, 70.0)
    activity = st.slider("Activity Level (1 = Low, 5 = High)", 1, 5, 3)

# ---------- BMI ----------
bmi = weight / ((height / 100) ** 2)

if bmi < 18.5:
    bmi_status = "Underweight"
elif bmi <= 24.9:
    bmi_status = "Normal"
elif bmi <= 29.9:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

st.divider()
st.subheader("2. Body Metrics")

st.write(f"**BMI:** {bmi:.2f}")
st.write(f"**BMI Category:** {bmi_status}")

# ---------- BMI VISUAL ----------
fig_bmi, ax = plt.subplots(figsize=(7, 2))
ax.axvspan(18.5, 24.9, alpha=0.3, label="Healthy Range")
ax.axvline(bmi, linewidth=2, label="Your BMI")
ax.set_xlim(10, 40)
ax.set_yticks([])
ax.set_xlabel("BMI Value")
ax.legend()
st.pyplot(fig_bmi)

st.divider()

# ---------- GENERATE REPORT ----------
if st.button("Generate Health Report"):
    st.session_state.report_ready = True

    X = np.array([[age, height, weight, activity]])
    X = (X - X.mean()) / X.std()

    prob = model.predict_proba(X)[0]
    prediction = "Healthy" if prob >= 0.5 else "Unhealthy"

    st.session_state.prediction = prediction
    st.session_state.prob = prob
    st.session_state.analysis = []

    if bmi < 18.5:
        st.session_state.analysis.append("BMI indicates underweight condition.")
    elif bmi > 24.9:
        st.session_state.analysis.append("BMI indicates excess body weight.")

    if activity < 3:
        st.session_state.analysis.append("Physical activity level is below recommended range.")

# ---------- REPORT VIEW ----------
if st.session_state.report_ready:
    st.subheader("3. Model Prediction")

    prediction = st.session_state.prediction
    prob = st.session_state.prob
    analysis_points = st.session_state.analysis

    st.write(f"**Health Status:** {prediction}")
    st.write(f"**Confidence Score:** {prob:.2f}")

    # ---------- PROBABILITY CHART ----------
    fig_prob, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Unhealthy", "Healthy"], [1 - prob, prob])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig_prob)

    # ---------- ANALYSIS ----------
    st.divider()
    st.subheader("4. Analysis")

    if analysis_points:
        for p in analysis_points:
            st.write("- ", p)
    else:
        st.write("All indicators are within healthy ranges.")

    # ---------- RECOMMENDATION ----------
    st.divider()
    st.subheader("5. Recommendations")

    if prediction == "Unhealthy":
        st.write(
            """
            - Increase physical activity frequency  
            - Improve dietary balance  
            - Monitor weight and BMI regularly  
            """
        )
    else:
        st.write(
            """
            - Maintain current workout routine  
            - Continue balanced nutrition  
            """
        )

    # ---------- PDF EXPORT ----------
    st.divider()
    st.subheader("6. Export Report")

    if st.button("Generate PDF Report"):
        report_dir = os.path.join(BASE_DIR, "..", "reports")
        os.makedirs(report_dir, exist_ok=True)

        filename = f"health_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        file_path = os.path.join(report_dir, filename)

        doc = SimpleDocTemplate(file_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Gym Health Assessment Report", styles["Title"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"Age: {age}", styles["Normal"]))
        story.append(Paragraph(f"Height: {height} cm", styles["Normal"]))
        story.append(Paragraph(f"Weight: {weight} kg", styles["Normal"]))
        story.append(Paragraph(f"Activity Level: {activity}/5", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"BMI: {bmi:.2f} ({bmi_status})", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"Prediction: {prediction}", styles["Normal"]))
        story.append(Paragraph(f"Confidence: {prob:.2f}", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Analysis:", styles["Heading2"]))
        if analysis_points:
            for p in analysis_points:
                story.append(Paragraph(p, styles["Normal"]))
        else:
            story.append(Paragraph("All indicators are healthy.", styles["Normal"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Recommendations:", styles["Heading2"]))

        if prediction == "Unhealthy":
            story.append(Paragraph("Increase activity and improve diet.", styles["Normal"]))
        else:
            story.append(Paragraph("Maintain current healthy lifestyle.", styles["Normal"]))

        doc.build(story)

        st.session_state.pdf_path = file_path
        st.success("PDF report generated successfully.")

    # ---------- DOWNLOAD ----------
    if st.session_state.pdf_path:
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=os.path.basename(st.session_state.pdf_path),
                mime="application/pdf"
            )

    st.caption(
        "Disclaimer: This report is for educational fitness assessment only."
    )
