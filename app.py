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

# Custom CSS for dark theme styling with cleaner interface (no highlighted containers)
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
    
    /* Full width content */
    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-bottom: 1rem;
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
    
    /* Seamless UI - no container boxes for main content */
    .seamless-section {
        margin-bottom: 20px;
        padding: 0;
    }
    
    /* Chat messages styling */
    .chat-message-container {
        margin-bottom: 15px;
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
    .warning-message {
        display: flex;
        align-items: center;
        padding: 12px 15px;
        border-radius: 4px;
        background-color: rgba(240, 173, 78, 0.2);
        color: #f0ad4e;
        margin-bottom: 15px;
    }
    .warning-message svg {
        margin-right: 10px;
    }
    
    /* Chat input */
    .stChatInput {
        background-color: var(--input-bg-color);
        border-radius: 20px;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--subtext-color);
        font-size: 0.8rem;
    }
    
    /* User message styling */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0;
    }
    
    .user-message-content {
        background-color: rgba(45, 142, 205, 0.2);
        padding: 12px 16px;
        border-radius: 18px 18px 2px 18px;
        max-width: 80%;
        color: var(--text-color);
    }
    
    /* Bot message styling */
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin: 10px 0;
    }
    
    .bot-message-content {
        background-color: var(--input-bg-color);
        padding: 12px 16px;
        border-radius: 18px 18px 18px 2px;
        max-width: 80%;
        color: var(--text-color);
    }
    
    /* Welcome section */
    .welcome-section {
        margin-top: 20px;
        margin-bottom: 30px;
    }
    
    .welcome-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: var(--text-color);
    }
    
    .welcome-text {
        color: var(--subtext-color);
        margin-bottom: 20px;
    }
    
    /* Steps list */
    .steps-list {
        list-style-type: none;
        padding-left: 0;
        margin-top: 15px;
    }
    
    .steps-list li {
        margin-bottom: 8px;
        display: flex;
        align-items: center;
    }
    
    .steps-list .step-number {
        background-color: var(--accent-color);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        margin-right: 10px;
        font-size: 0.9rem;
    }
    
    /* Chat section title */
    .section-title {
        margin-top: 30px;
        margin-bottom: 15px;
        font-size: 1.4rem;
        font-weight: 500;
        display: flex;
        align-items: center;
    }
    
    .section-title svg {
        margin-right: 10px;
    }
    
    /* Remove box shadows from expandable sections */
    .streamlit-expanderHeader {
        background-color: transparent;
        border: none;
        color: var(--text-color);
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        border: none;
        background-color: transparent;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: var(--input-bg-color);
        color: var(--text-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* File uploader */
    .css-1ekf893 {
        background-color: var(--input-bg-color);
        border: 1px dashed rgba(255, 255, 255, 0.3);
    }
    
    /* Hide default expander styling */
    .st-emotion-cache-1oe5cao {
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 20px 0;
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

# Main header - subtle and clean
st.markdown('<p class="main-header">Inquiro Bot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Data Analysis Assistant</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## Configuration")
    
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
            st.success("API key validated!")
        except Exception as e:
            st.session_state.api_key_valid = False
            st.error(f"API key error: {str(e)}")
    
    st.markdown('<a href="https://platform.openai.com/api-keys" target="_blank">Get one here</a>', unsafe_allow_html=True)
    
    st.divider()
    
    # File uploader
    st.markdown("## Dataset")
    
    # File upload area with instructions
    st.markdown("#### Upload your dataset")
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
                st.success(f"Dataset loaded successfully! Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                
                # Display dataset preview in sidebar
                st.markdown("#### Dataset Preview")
                st.dataframe(df.head(3), use_container_width=True)
        except Exception as e:
            st.session_state.file_uploaded = False
            st.error(f"Error loading file: {str(e)}")

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

# Main content area - No container boxes
main_container = st.container()

with main_container:
    # Discreet "About" section as an expander
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
    
    # Warning message only when requirements not met
    if not requirements_met:
        if not st.session_state.file_uploaded:
            st.markdown('<div class="warning-message">⚠️ Please upload a dataset in the sidebar</div>', unsafe_allow_html=True)
    
    # Welcome section - directly on the page, no container
    if not requirements_met:
        st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="welcome-header">Welcome to Inquiro Bot!</h2>', unsafe_allow_html=True)
        st.markdown('<p class="welcome-text">To get started:</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <ul class="steps-list">
            <li><span class="step-number">1</span> Enter your OpenAI API key in the sidebar</li>
            <li><span class="step-number">2</span> Upload a CSV or Excel file</li>
            <li><span class="step-number">3</span> Start asking questions about your data</li>
        </ul>
        <p class="welcome-text">Inquiro Bot helps you analyze and understand your data through natural language conversations.</p>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat section title - clean style
    st.markdown('<h3 class="section-title">💬 Chat with your data</h3>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm Inquiro Bot, your data analysis assistant. Upload a dataset and provide an API key to get started."}
        ]
    
    # Display chat messages with custom styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="user-message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <div class="bot-message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input - no containing box
    if prompt := st.chat_input("Ask a question about your data...", disabled=not requirements_met):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the new user message
        st.markdown(f"""
        <div class="user-message">
            <div class="user-message-content">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate and display assistant response
        if requirements_met:
            with st.spinner("Analyzing data..."):
                # System process container - hidden by default
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
                
                st.markdown(f"""
                <div class="bot-message">
                    <div class="bot-message-content">{response}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            error_message = "Please provide a valid OpenAI API key and upload a dataset before asking questions."
            st.markdown(f"""
            <div class="bot-message">
                <div class="bot-message-content">{error_message}</div>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Add dataset information if available - as an expander
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

# Footer - minimal and clean
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('Inquiro Bot | Your intelligent data analysis assistant', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
