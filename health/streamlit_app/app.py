
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pypdf import PdfReader


# -----------------------------
# 1. Application configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

st.set_page_config(
    page_title="Blood Report Analyzer",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Blood Test Report Analyzer")
st.write(
    "Extract patient details and laboratory results "
    "from a blood test report using LangChain and Groq."
)

st.warning(
    "Educational tool only. This app does not diagnose diseases. "
    "Always confirm results with a qualified healthcare professional."
)


# -----------------------------
# 2. Initialize Groq model
# -----------------------------
@st.cache_resource
def get_model(api_key):
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0
    )


# -----------------------------
# 3. Read uploaded files
# -----------------------------
def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        pdf = PdfReader(uploaded_file)
        pages = []

        for page in pdf.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages)

    return uploaded_file.getvalue().decode("utf-8", errors="replace")


# -----------------------------
# 4. Extraction prompt
# -----------------------------
def build_prompt(blood_report):
    return f"""
You are a medical laboratory report information extraction assistant.

Extract information from the supplied report and return ONLY valid JSON.

Use this exact structure:

{{
  "patient_details": {{
    "name": null,
    "age": null,
    "gender": null,
    "report_date": null
  }},
  "tests": [
    {{
      "category": "CBC",
      "test_name": "Hemoglobin",
      "result": "15.1",
      "unit": "g/dL",
      "reference_range": "13.5-17.5 g/dL",
      "status": "Within range"
    }}
  ]
}}

Extraction requirements:
- Include patient details only when explicitly reported.
- Extract all available tests, including CBC, lipid panel,
  metabolic panel and liver function tests.
- Preserve reported results, units and reference ranges.
- Use null for missing values.
- Do not invent results or reference ranges.
- If a value is described as suggested, hypothetical or unverified,
  do not treat it as a confirmed laboratory result.
- For status, use "High", "Low", or "Within range" only when the
  result can be compared reliably with the supplied reference range.
- Otherwise use "Unable to determine".
- Do not diagnose medical conditions.
- Return valid JSON only. Do not use Markdown code fences.

BLOOD REPORT:
{blood_report}
"""


# -----------------------------
# 5. Upload or enter report
# -----------------------------
st.subheader("1. Provide your blood report")

uploaded_file = st.file_uploader(
    "Upload a TXT or PDF report",
    type=["txt", "pdf"]
)

manual_report = st.text_area(
    "Or paste the report text here",
    height=220,
    placeholder="Paste your blood test report here..."
)

blood_report = ""

if uploaded_file is not None:
    try:
        blood_report = read_uploaded_file(uploaded_file)
        st.success(f"Loaded file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")

elif manual_report.strip():
    blood_report = manual_report.strip()


# -----------------------------
# 6. Extract and analyze
# -----------------------------
if st.button("Extract Blood Test Results", type="primary"):

    if not blood_report.strip():
        st.error("Please upload a report or paste report text.")
        st.stop()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY was not found. Add it to your .env file "
            "in the same folder as app.py."
        )
        st.stop()

    try:
        with st.spinner("Extracting results with Groq..."):
            llm = get_model(api_key)
            prompt = build_prompt(blood_report)
            response = llm.invoke(prompt)

        output = response.content.strip()

        # Remove Markdown fences if the model includes them.
        if output.startswith("```"):
            lines = output.splitlines()
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            output = "\n".join(lines).strip()

        data = json.loads(output)

        st.session_state["blood_report_data"] = data
        st.session_state["blood_report_json"] = json.dumps(
            data, indent=2, ensure_ascii=False
        )

    except json.JSONDecodeError:
        st.error("The model did not return valid JSON. Please try again.")
        with st.expander("View model response"):
            st.code(output if "output" in locals() else "No response")
        st.stop()

    except Exception as e:
        st.error(f"Extraction failed: {e}")
        st.stop()


# -----------------------------
# 7. Display extracted results
# -----------------------------
if "blood_report_data" in st.session_state:
    data = st.session_state["blood_report_data"]

    st.divider()
    st.subheader("2. Patient Information")

    patient = data.get("patient_details", {})

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patient", patient.get("name") or "Not reported")
    col2.metric("Age", patient.get("age") or "Not reported")
    col3.metric("Gender", patient.get("gender") or "Not reported")
    col4.metric(
        "Report Date",
        patient.get("report_date") or "Not reported"
    )

    st.subheader("3. Extracted Laboratory Results")

    tests = data.get("tests", [])

    if tests:
        df = pd.DataFrame(tests)

        for column in [
            "category",
            "test_name",
            "result",
            "unit",
            "reference_range",
            "status"
        ]:
            if column not in df.columns:
                df[column] = None

        df = df[
            [
                "category",
                "test_name",
                "result",
                "unit",
                "reference_range",
                "status"
            ]
        ]

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="blood_test_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No laboratory tests were extracted.")

    st.subheader("4. Extracted JSON")

    st.json(data)

    st.download_button(
        "Download extracted JSON",
        data=st.session_state["blood_report_json"],
        file_name="blood_test_results.json",
        mime="application/json"
    )
