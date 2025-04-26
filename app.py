import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv
import os
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
    st.session_state.setdefault("messages", [
        {"role": "assistant", "content": "Hello! I'm Inquiro Bot. Upload your dataset and ask away!"}
    ])
    st.session_state.setdefault("pending_prompt", None)

init_session()

# ---------------------- Inline CSS Styling ---------------------- #
st.markdown("""
<style>
    .main { background-color: #1e1f29; color: #ffffff; }
    .block-container { max-width: 100%; padding-top: 1rem; }
    .stButton>button { background-color: #2d8ecd; color: white; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- Helper Functions ---------------------- #
def generate_response(df, user_input, llm):
    try:
        query_engine = PandasQueryEngine(df=df, verbose=True, llm=llm, synthesize_response=True)
        result = query_engine.query(user_input)
        return str(result), None
    except Exception as e:
        return None, str(e)

def clean_data(df, remove_duplicates, handle_missing, format_dates):
    cleaned_df = df.copy()

    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if handle_missing:
        cleaned_df = cleaned_df.fillna("N/A")

    if format_dates:
        for col in cleaned_df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
                except Exception:
                    pass

    return cleaned_df

# ---------------------- Sidebar ---------------------- #
with st.sidebar:
    st.title("🧠 Inquiro Config")

    # API Key input
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

    # File upload
    st.markdown("### 📄 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV/XLSX file", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        try:
            with st.spinner("Reading your file..."):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.session_state.file_uploaded = True
                st.success(f"Dataset Loaded! Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        except Exception as e:
            st.session_state.file_uploaded = False
            st.error(f"Failed to read file: {str(e)}")

    # Data Cleaning Assistant
    if st.session_state.file_uploaded:
        st.markdown("---")
        st.markdown("### 🧹 Data Cleaning Assistant")

        remove_duplicates = st.checkbox("Remove Duplicate Rows")
        handle_missing = st.checkbox("Fill Missing Values (with 'N/A')")
        format_dates = st.checkbox("Format Date/Time Columns")

        if st.button("Apply Cleaning"):
            st.session_state.df = clean_data(
                st.session_state.df,
                remove_duplicates,
                handle_missing,
                format_dates
            )
            st.success("Data cleaning applied!")

# ---------------------- Main UI ---------------------- #
st.title("🤖 Inquiro Bot")
st.markdown("Your **AI-powered assistant** for seamless data exploration.")

# Dataset Preview
if st.session_state.file_uploaded:
    st.markdown("### 📊 Dataset Preview")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

    # Quick Query Buttons
    st.markdown("### ⚡ Quick Queries")
    cols = st.columns(3)
    queries = ["Show me summary statistics.", "Which product performed best?", "Any missing values?"]
    for i, q in enumerate(queries):
        if cols[i].button(q):
            st.session_state.pending_prompt = q

# Chat Input
prompt = None
user_input = st.chat_input("Ask your data anything...", disabled=not (st.session_state.api_key_valid and st.session_state.file_uploaded))
prompt = user_input if user_input else st.session_state.pending_prompt
st.session_state.pending_prompt = None  # Clear after use

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.api_key_valid and st.session_state.file_uploaded:
        with st.spinner("Analyzing..."):
            response, error = generate_response(st.session_state.df, prompt, st.session_state.llm)
            if error:
                response = "⚠️ Error: " + error
            st.session_state.messages.append({"role": "assistant", "content": response})

# Chat History
st.markdown("---")
st.subheader("💬 Chat with your Data")

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Inquiro Bot:** {message['content']}")

# Footer
st.markdown("---")
st.markdown("🔍 Created by **Chirag Bhatia** · Built with [Streamlit](https://streamlit.io) + [LlamaIndex](https://www.llamaindex.ai)")
st.markdown("[GitHub Repo](https://github.com/yourusername/inquiro-bot) | [Connect on LinkedIn](https://www.linkedin.com/in/chirag--bhatia)")
