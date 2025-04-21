import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.experimental.query_engine import PandasQueryEngine
from  dotenv import load_dotenv
import os,re,sys
import pandas as pd
 
load_dotenv()
 
#retrieve the user defined OpenAI api key 
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
os.environ['OPENAI_API_KEY']=openai_api_key
st.sidebar.write("Create a new OpenAI API Key  \nhttps://platform.openai.com/api-keys")

llm=OpenAI(model="gpt-4o-mini",temperature=0.0,api_key=openai_api_key)
 
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
            self.container.markdown(''.join(self.buffer) , unsafe_allow_html=True)
            self.buffer = []
 
 
 
#helper functions
#function to read & store spreadsheet in a temporary directory
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df=pd.read_csv(uploaded_file,index_col=0)
    else :
        df=pd.read_excel(uploaded_file,index_col=0)
    return df
 
 
#function to generate a relevant respond to the user input query.
def generate_response(df,user_input):
    try:
        query_engine = PandasQueryEngine(df=df, verbose=True,llm=llm,synthesize_response=True)
        chatbot_response=query_engine.query(user_input)
    except Exception as e:
        chatbot_response = f"An error occurred: {str(e)}"
    return str(chatbot_response)
 
#UI of the app
st.title("Inquiro bot")
st.subheader("Gain significant insights from your datasets",divider="red")
#st.divider()
 
with st.sidebar:
    uploaded_file=st.file_uploader(accept_multiple_files=False,label="Upload your dataset",type=["csv","xls","xlsx"])
 
if uploaded_file:
    df=read_file(uploaded_file)
 
#Initialize the chat message history
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Inquiro bot. Please upload a dataset to get started."}
    ]
 
 
# Prompt for user input and save to chat history
if prompt := st.chat_input("Please enter your query here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
 
# Display the prior chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
 
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant") :
        with st.spinner("Retrieving information.Please Wait."):
            with st.container(height=200,):
                st.write("**System process**")
                sys.stdout = StreamToContainer(st)
            #retrieve the response based on chat_engine and user 
                if openai_api_key and uploaded_file:
                    try:
                        response = generate_response(df,prompt)
                    except Exception as e:
                        response = f"An error occurred: {str(e)}"
                else:
                    response = "Please provide a valid OpenAI API key and upload a dataset to continue."
        # Display the response in the chat message
        st.write(response)
        message = {"role": "assistant", "content": response}
        # Add response to message history
        st.session_state.messages.append(message)