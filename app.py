import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv
import os, re, sys
import pandas as pd
import time

# Page configuration and styling
st.set_page_config(
    page_title="Inquiro Bot | Data Analysis Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .chat-container {
        border-radius: 10px;
        background-color: #f9f9f9;
        padding: 20px;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    .welcome-message {
        padding: 15px;
        border-radius: 5px;
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
    }
    .error-message {
        padding: 15px;
        border-radius: 5px;
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        margin-bottom: 15px;
    }
    .success-message {
        padding: 15px;
        border-radius: 5px;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        margin-bottom: 15px;
    }
    .info-box {
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .sidebar-content {
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Initialize session state
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "error" not in st.session_state:
    st.session_state.error = None

# Main header
st.markdown('<p class="main-header">Inquiro Bot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Data Analysis Assistant</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.subheader("Configuration")
    
    # API Key input
    openai_api_key = st.text_input('OpenAI API Key', type='password', help="Your API key is securely used and not stored permanently")
    
    if openai_api_key:
        try:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            llm = OpenAI(model="gpt-4o-mini", temperature=0.0, api_key=openai_api_key)
            st.session_state.api_key_valid = True
            st.session_state.llm = llm
            st.markdown('<div class="success-message">API key validated!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.session_state.api_key_valid = False
            st.markdown(f'<div class="error-message">API key error: {str(e)}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Need an API key? <a href="https://platform.openai.com/api-keys" target="_blank">Get one here</a></div>', unsafe_allow_html=True)
    
    # File uploader
    st.subheader("Dataset")
    uploaded_file = st.file_uploader(
        "Upload your dataset", 
        type=["csv", "xls", "xlsx"],
        help="Support for CSV and Excel files"
    )

    if uploaded_file:
        try:
            with st.spinner("Loading dataset..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.session_state.file_uploaded = True
                st.markdown(f'<div class="success-message">Dataset loaded successfully! <br>Rows: {df.shape[0]} | Columns: {df.shape[1]}</div>', unsafe_allow_html=True)
                
                # Display dataset preview in sidebar
                st.subheader("Dataset Preview")
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.session_state.file_uploaded = False
            st.markdown(f'<div class="error-message">Error loading file: {str(e)}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Used to stream sys output on the streamlit frontend
class StreamToContainer:
    def __init__(self, container):
        self.container = container
        self.buffer = []
    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)
        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.container.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

# Function to generate a relevant response to the user input query
def generate_response(df, user_input, llm):
    try:
        query_engine = PandasQueryEngine(df=df, verbose=True, llm=llm, synthesize_response=True)
        chatbot_response = query_engine.query(user_input)
        return str(chatbot_response), None
    except Exception as e:
        error_message = f"Analysis error: {str(e)}"
        return None, error_message

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # App description and guidance
    with st.expander("About Inquiro Bot", expanded=False):
        st.markdown("""
        ### How to use Inquiro Bot
        1. **Provide your OpenAI API key** in the sidebar
        2. **Upload a dataset** (CSV or Excel file)
        3. **Ask questions** about your data in natural language
        
        ### Example queries
        - "What's the average value in column X?"
        - "Show me the correlation between column A and B"
        - "What is the trend of sales over time?"
        - "Find outliers in the dataset"
        - "Summarize the key insights from this data"
        """)

    # Check if prerequisites are met
    requirements_met = st.session_state.api_key_valid and st.session_state.file_uploaded
    
    if not requirements_met:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        if not st.session_state.api_key_valid:
            st.warning("⚠️ Please enter a valid OpenAI API key in the sidebar")
        if not st.session_state.file_uploaded:
            st.warning("⚠️ Please upload a dataset in the sidebar")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader("💬 Chat with your data")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Inquiro Bot, your data analysis assistant. Upload a dataset and provide an API key to get started."}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data...", disabled=not requirements_met):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the new user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if requirements_met:
                with st.spinner("Analyzing data..."):
                    # System process container
                    system_container = st.container()
                    
                    with system_container:
                        st.markdown("#### System process")
                        sys_output = st.empty()
                        
                        # Redirect system output to our container
                        sys.stdout = StreamToContainer(sys_output)
                        
                        # Generate response
                        response, error = generate_response(st.session_state.df, prompt, st.session_state.llm)
                        
                        # Reset stdout
                        sys.stdout = sys.__stdout__
                    
                    # Handle response or error
                    if error:
                        st.error(error)
                        response = "I encountered an error while analyzing your data. Please try rephrasing your question or check if your data contains the information you're looking for."
                    
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                error_message = "Please provide a valid OpenAI API key and upload a dataset before asking questions."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Dataset information and tips
    if st.session_state.file_uploaded:
        st.markdown("### Dataset Information")
        df = st.session_state.df
        
        # Data statistics card
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.metric("Total Rows", df.shape[0])
        st.metric("Total Columns", df.shape[1])
        
        # Missing values analysis
        missing_values = df.isna().sum().sum()
        missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100
        st.metric("Missing Values", f"{missing_values} ({missing_percentage:.2f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Column information
        st.markdown("### Column Overview")
        column_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isna().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(column_info, use_container_width=True)
        
        # Query suggestions
        st.markdown("### Suggested Queries")
        query_suggestions = [
            f"Summarize the dataset",
            f"Find the average of {df.select_dtypes(include=['number']).columns[0] if not df.select_dtypes(include=['number']).empty else 'numeric columns'}",
            f"Show correlation between columns",
            f"Identify trends in {df.columns[0]}",
            f"Find outliers in the dataset"
        ]
        
        for suggestion in query_suggestions:
            if st.button(suggestion, key=suggestion):
                # Add suggestion to chat and simulate clicking send
                if "messages" in st.session_state:
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.experimental_rerun()
    else:
        # Welcome message when no file is uploaded
        st.markdown('<div class="welcome-message">', unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to Inquiro Bot!
        
        To get started:
        1. Enter your OpenAI API key in the sidebar
        2. Upload a CSV or Excel file
        3. Start asking questions about your data
        
        Inquiro Bot helps you analyze and understand your data through natural language conversations.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('---')
st.markdown('Inquiro Bot | Your intelligent data analysis assistant')
