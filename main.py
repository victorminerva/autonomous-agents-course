# main.py

import os
import zipfile
import streamlit as st
from dotenv import load_dotenv
from agent import build_agent
from processor import list_csv_files, load_csv

# Page setup
st.set_page_config(page_title="CSV Question Agent", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 Intelligent CSV Agent</h1>", unsafe_allow_html=True)

load_dotenv()

# Two-column layout: 20% (left) for upload + selection, 80% (right) for preview + chat
left_col, right_col = st.columns([1, 4])

# --- LEFT: Upload and File Selection ---
with left_col:
    st.subheader("📤 Upload File")
    uploaded_file = st.file_uploader("Upload a `.csv` file or a `.zip` archive", type=["csv", "zip"])

    if uploaded_file:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall("data")
                st.success("✅ ZIP extracted successfully!")
            except zipfile.BadZipFile:
                st.error("❌ Invalid ZIP file.")
        else:
            st.success(f"✅ File `{uploaded_file.name}` uploaded.")

    # CSV file selection
    csv_files = list_csv_files("data")
    selected_file = None
    if csv_files:
        selected_file = st.selectbox("📂 Choose a CSV file:", csv_files)
    else:
        st.info("No CSV files found. Please upload one to continue.")

# --- RIGHT: Preview + Chat ---
with right_col:
    if selected_file:
        df = load_csv(os.path.join("data", selected_file))

        st.subheader("🔍 Preview CSV Content")
        with st.expander("View CSV table"):
            st.dataframe(df, use_container_width=True)

        st.subheader("💬 Ask a Question")

        # Initialize conversation history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("What would you like to know about the CSV data?")
        if st.button("🔎 Ask") and query:
            with st.spinner("🧠 Processing your question..."):
                query_engine = build_agent(data_path="data")
                result = query_engine.query(query)

            # Store question and answer in session state
            st.session_state.chat_history.append({
                "question": query,
                "answer": result.response
            })

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📚 Conversation History")
            for i, exchange in enumerate(st.session_state.chat_history[::-1], 1):
                st.markdown(f"**🟦 Q{i}:** {exchange['question']}")
                st.markdown(f"**🟩 A{i}:** {exchange['answer']}")
                st.markdown("---")
    else:
        st.info("Upload and select a CSV file to get started.")
