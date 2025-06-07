# main.py

import os
import streamlit as st
from agent import build_agent
from processor import list_csv_files, load_csv, unzip_file

# Page setup
st.set_page_config(page_title="CSV Question Agent", layout="wide")
st.markdown("<h1 style='text-align: center;'>🤖 Intelligent CSV Agent</h1>", unsafe_allow_html=True)

st.sidebar.header("📤 Upload File")
zip_file = st.sidebar.file_uploader("Upload a `.csv` file or a `.zip` archive", type=["csv", "zip"])

if zip_file:
    unzip_path = "data"
    os.makedirs(unzip_path, exist_ok=True)
    unzip_file(zip_file, extract_to=unzip_path)
    st.sidebar.success("Arquivo descompactado com sucesso!")

csv_files = list_csv_files("data")
selected_file = st.sidebar.selectbox("📂 Choose a CSV file:", csv_files)

if selected_file:
    df = load_csv(os.path.join("data", selected_file))
    st.sidebar.success(f"✅ File `{selected_file}` loaded successfully!")

    st.write("🔍 Preview CSV Content")
    with st.expander("View CSV table"):
        st.dataframe(df, use_container_width=True)

    agent, embeddings = build_agent(df)

    st.subheader("💬 Ask a Question")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    question = st.text_input("What would you like to know about the CSV data?")

    if st.button("🔎 Ask") and question:
        with st.spinner("🧠 Processing your question..."):
            try:
                result = agent.run(question)
                st.markdown(f"🟩 **Answer:** {result}")
                st.session_state['chat_history'].append({'question': question, 'answer': result})
            except Exception as e:
                st.error(f"❌ Error processing your question: {e}")

    # Display chat history (latest first)
    if st.session_state.chat_history:
        with st.expander('📚 Conversation History', expanded=True):
            st.markdown('<div style="max-height: 300px; overflow-y: auto;">', unsafe_allow_html=True)
            for i, exchange in enumerate(reversed(st.session_state['chat_history']), 1):
                st.markdown(f"**🟦 Q{len(st.session_state['chat_history']) - i + 1}:** {exchange['question']}")
                st.markdown(f"**🟩 A{len(st.session_state['chat_history']) - i + 1}:** {exchange['answer']}")
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)