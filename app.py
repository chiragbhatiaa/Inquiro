import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv
import os, re, sys
import pandas as pd

# ---------------------- Page Config ---------------------- #
st.set_page_config(
    page_title="Inquiro Bot | AI-Powered Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Load Environment ---------------------- #
load_dotenv()

# ---------------------- Init Session State ---------------------- #
def init_session():
    st.session_state.setdefault("api_key_valid", False)
    st.session_state.setdefault("file_uploaded", False)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("messages", [
        {"role": "assistant", "content": "Hello! I'm Inquiro Bot. Upload your dataset and ask away!"}
    ])

init_session()

# ---------------------- Custom CSS ---------------------- #
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------- Sidebar ---------------------- #
with st.sidebar:
    st.title("🧠 Inquiro Config")
    openai_api_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
    if openai_api_key:
        try:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            llm = OpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_api_key)
            st.session_state.api_key_valid = True
            st.session_state.llm = llm
            st.success("API key validated!")
        except Exception as e:
            st.session_state.api_key_valid = False
            st.error(f"API key error: {str(e)}")

    st.markdown("[🔗 Get your API Key](https://platform.openai.com/api-keys)", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        try:
            with st.spinner("Reading your file..."):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.session_state.file_uploaded = True
                st.success(f"Dataset Loaded! Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                st.dataframe(df.head(5))
        except Exception as e:
            st.session_state.file_uploaded = False
            st.error(f"Failed to read file: {str(e)}")

# ---------------------- Main UI ---------------------- #
st.title("🤖 Inquiro Bot")
st.markdown("Your **AI-powered assistant** for seamless data exploration.")

# Example Quick Queries
if st.session_state.file_uploaded:
    st.markdown("### ⚡ Try Quick Queries")
    cols = st.columns(3)
    queries = ["Show me summary statistics.", "Which product performed best?", "Any missing values?"]
    for i, q in enumerate(queries):
        if cols[i].button(q):
            st.session_state.messages.append({"role": "user", "content": q})

# Display Chat Interface
st.markdown("---")
st.subheader("💬 Chat with your Data")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Inquiro Bot:** {message['content']}")

if prompt := st.chat_input("Ask your data anything...", disabled=not (st.session_state.api_key_valid and st.session_state.file_uploaded)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Analyzing..."):
        def generate_response(df, user_input, llm):
            try:
                query_engine = PandasQueryEngine(df=df, verbose=True, llm=llm, synthesize_response=True)
                result = query_engine.query(user_input)
                return str(result), None
            except Exception as e:
                return None, str(e)

        response, error = generate_response(st.session_state.df, prompt, st.session_state.llm)
        if error:
            response = "⚠️ Error: " + error
        st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------------- Footer ---------------------- #
st.markdown("---")
st.markdown("🔍 Created by **Chirag Bhatia** · Built with [Streamlit](https://streamlit.io) + [LlamaIndex](https://www.llamaindex.ai)")
st.markdown("[GitHub Repo](https://github.com/yourusername/inquiro-bot) | [Connect on LinkedIn](https://www.linkedin.com/in/chirag--bhatia)")
