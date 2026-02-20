import streamlit as st
import pickle
import pandas as pd
import re
from docx import Document
import io

# ----------------------------
# Load Trained Model Files
# ----------------------------
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="AI Resume Screening System", layout="wide")
st.title("📄 AI Resume Screening System")

uploaded_files = st.file_uploader(
    "Upload Multiple Resumes (.docx)",
    type=["docx"],
    accept_multiple_files=True
)

# ----------------------------
# Helper Functions
# ----------------------------

def extract_text(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text


def extract_experience(text):
    pattern = r'(\d+)\+?\s*(years|yrs)'
    matches = re.findall(pattern, text.lower())
    if matches:
        return max([int(match[0]) for match in matches])
    return 0


skill_keywords = [
    "python", "java", "react", "sql", "machine learning",
    "deep learning", "excel", "power bi", "c++",
    "html", "css", "javascript", "linux", "aws"
]


def extract_skills(text):
    found_skills = []
    text_lower = text.lower()
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill)
    return ", ".join(found_skills)


# ----------------------------
# Main Processing
# ----------------------------

if st.button("Process All Resumes"):

    if not uploaded_files:
        st.warning("Please upload resume files first.")
    else:
        results = []

        for file in uploaded_files:
            text = extract_text(file)

            # Predict category
            text_tfidf = tfidf.transform([text])
            prediction = model.predict(text_tfidf)
            category = le.inverse_transform(prediction)[0]

            # Extract details
            experience = extract_experience(text)
            skills = extract_skills(text)

            results.append({
                "File Name": file.name,
                "Predicted Role": category,
                "Skills": skills,
                "Experience (Years)": experience
            })

        # Create DataFrame
        df = pd.DataFrame(results)

        # 🔥 Sort by Role and Experience (Ascending)
        df = df.sort_values(
            by=["Predicted Role", "Experience (Years)"],
            ascending=[True, True]
        ).reset_index(drop=True)

        # Display final single table
        st.success("Processing Completed Successfully ✅")
        st.dataframe(df, use_container_width=True)

        # ----------------------------
        # Excel Download
        # ----------------------------
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        st.download_button(
            label="⬇ Download Sorted Excel File",
            data=output,
            file_name="Sorted_Resume_Screening.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
