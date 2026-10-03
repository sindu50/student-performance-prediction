import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's final score using Machine Learning."
)

st.divider()

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("data/student_data.csv")

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score"
    ]
]

y = data["final_score"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# USER INPUTS
# -----------------------------
st.subheader("👨‍🎓 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input(
        "📚 Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    attendance = st.number_input(
        "📅 Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

with col2:
    previous_score = st.number_input(
        "📝 Previous Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    assignment_score = st.number_input(
        "📋 Assignment Score",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

st.divider()

# -----------------------------
# PREDICTION
# -----------------------------
if st.button(
    "🔮 Predict Final Score",
    use_container_width=True
):

    new_student = [[
        study_hours,
        attendance,
        previous_score,
        assignment_score
    ]]

    prediction = model.predict(new_student)

    score = prediction[0]

    st.subheader("📊 Prediction Result")

    st.success(
        f"🎯 Predicted Final Score: {score:.2f}"
    )

    # Performance message
    if score >= 80:
        st.info(
            "🌟 Excellent expected performance!"
        )

    elif score >= 60:
        st.info(
            "👍 Good expected performance."
        )

    else:
        st.warning(
            "📖 More preparation may be helpful."
        )

    # Progress bar
    st.write("Performance Level")
    st.progress(
        min(max(int(score), 0), 100)
    )

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)
# Prediction Visualization
st.subheader("📊 Performance Visualization")

chart_data = pd.DataFrame({
    "Score": [score]
}, index=["Predicted Final Score"])

st.bar_chart(chart_data)