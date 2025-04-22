import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from dotenv import load_dotenv
import os, re, sys
import pandas as pd

# Page configuration and styling
st.set_page_config(
    page_title="Inquiro Bot | Data Analysis Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling matching the screenshot
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --background-color: #1e1f29;
        --sidebar-color: #1a1b23;
        --text-color: #ffffff;
        --subtext-color: #a0a0a0;
        --accent-color: #2d8ecd;
        --warning-color: #f0ad4e;
        --error-color: #d9534f;
        --success-color: #5cb85c;
        --card-bg-color: #282a36;
        --input-bg-color: #2e303d;
    }
    
    /* Main container styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1wrcr25 {
        background-color: var(--sidebar-color);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.2rem;
        color: var(--accent-color);
        font-weight: 700;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: var(--subtext-color);
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Cards and containers */
    .highlight {
        background-color: var(--card-bg-color);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .chat-container {
        border-radius: 8px;
        background-color: var(--card-bg-color);
        padding: 20px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom button styling */
    .stButton > button {
        background-color: var(--accent-color);
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    /* Message styling */
    .welcome-message {
        padding: 15px;
        border-radius: 5px;
        background-color: rgba(45, 142, 205, 0.1);
        border-left: 5px solid var(--accent-color);
    }
    
    .error-message {
        padding: 15px;
        border-radius: 5px;
        background-color: rgba(217, 83, 79, 0.1);
        border-left: 5px solid var(--error-color);
        margin-bottom: 15px;
    }
    
    .success-message {
        padding: 15px;
        border-radius: 5px;
        background-color: rgba(92, 184, 92, 0.1);
        border-left: 5px solid var(--success-color);
        margin-bottom: 15px;
    }
    
    .warning-message {
        padding: 15px;
        border-radius: 5px;
        background-color: rgba(240, 173, 78, 0.1);
        border-left: 5px solid var(--warning-color);
        margin-bottom: 15px;
    }
    
    /* Info box */
    .info-box {
        padding: 10px;
        background-color: var(--input-bg-color);
        border-radius: 5px;
        font-size: 0.9rem;
        color: var(--subtext-color);
    }
    
    /* Sidebar content */
    .sidebar-content {
        padding: 20px 0;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: var(--input-bg-color);
        color: var(--text-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat input */
    .stChatInput {
        background-color: var(--input-bg-color);
        border-radius: 20px;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 50px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--subtext-color);
        font-size: 0.8rem;
    }
    
    /* Table styling */
    .dataframe {
        background-color: var(--card-bg-color);
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: var(--card-bg-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        color: var(--text-color);
    }
    
    /* System container */
    .system-container {
        background-color: var(--input-bg-color);
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: #cccccc;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: rgba(45, 142, 205, 0.1);
        padding: 10px 15px;
        border-radius: 18px 18px 2px 18px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
    }
    
    .bot-message {
        background-color: var(--input-bg-color);
        padding: 10px 15px;
        border-radius: 18px 18px 18px 2px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
    }
    
    /* Override Streamlit defaults */
    .stAlert > div {
        background-color: var(--card-bg-color);
        color: var(--text-color);
    }
    
    .stMetric {
        background-color: transparent;
    }
    
    /* File uploader styling */
    .css-1ekf893 {
        background-color: var(--input-bg-color);
        border: 1px dashed rgba(255, 255, 255, 0.3);
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
    st.markdown("#### OpenAI API Key")
    openai_api_key = st.text_input('', type='password', placeholder="Enter your API key", 
                                help="Your API key is securely used and not stored permanently", label_visibility="collapsed")
    
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
    
    st.markdown('<div class="info-box"><a href="https://platform.openai.com/api-keys" target="_blank">Get one here</a></div>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown("#### Dataset")
    
    # File upload area with instructions
    st.markdown("##### Upload your dataset")
    uploaded_file = st.file_uploader(
        "Drag and drop file here", 
        type=["csv", "xls", "xlsx"],
        label_visibility="collapsed"
    )
    st.caption("Limit 200MB per file • CSV, XLS, XLSX")

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
                st.dataframe(df.head(3), use_container_width=True)
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

# Main content area - single column layout like in the screenshot
main_container = st.container()

with main_container:
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
        # Warning message when requirements not met
        if not st.session_state.file_uploaded:
            st.markdown('<div class="warning-message">⚠️ Please upload a dataset in the sidebar</div>', unsafe_allow_html=True)
        elif not st.session_state.api_key_valid:
            st.markdown('<div class="warning-message">⚠️ Please enter a valid OpenAI API key in the sidebar</div>', unsafe_allow_html=True)
        
        # Welcome message and instructions
        st.markdown('<div class="welcome-message">', unsafe_allow_html=True)
        st.markdown("""
        ### Welcome to Inquiro Bot!
        
        To get started:
        
        1. Enter your OpenAI API key in the sidebar
        2. Upload a CSV or Excel file
        3. Start asking questions about your data
        
        Inquiro Bot helps you analyze and understand your data through natural language conversations.
        """)
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<h3>💬 Chat with your data</h3>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Inquiro Bot, your data analysis assistant. Upload a dataset and provide an API key to get started."}
        ]
    
    # Display chat messages with custom styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data...", disabled=not requirements_met):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the new user message
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
        
        # Generate and display assistant response
        if requirements_met:
            with st.spinner("Analyzing data..."):
                # System process container
                with st.expander("System process", expanded=False):
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
                
                st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            error_message = "Please provide a valid OpenAI API key and upload a dataset before asking questions."
            st.markdown(f'<div class="bot-message">{error_message}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Add dataset information if available
    if st.session_state.file_uploaded and requirements_met:
        with st.expander("Dataset Information", expanded=False):
            df = st.session_state.df
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", df.shape[0])
            
            with col2:
                st.metric("Total Columns", df.shape[1])
            
            with col3:
                # Missing values analysis
                missing_values = df.isna().sum().sum()
                missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Values", f"{missing_values} ({missing_percentage:.2f}%)")
            
            # Column information
            st.markdown("### Column Overview")
            column_info = pd.DataFrame({
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(column_info, use_container_width=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('Inquiro Bot | Your intelligent data analysis assistant', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
