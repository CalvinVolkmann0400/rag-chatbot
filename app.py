import os

from dotenv import load_dotenv
from google.genai import errors
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import streamlit as st

load_dotenv()
DATA_DIR = "data/handbook"


def get_api_key():
    """Return the Gemini API key from .env, or stop the app if it is missing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add the line GEMINI_API_KEY=your-key "
                 "to the .env file in the project folder, then restart the app.")
        st.stop()
    return api_key


def check_data_dir():
    """Stop the app if DATA_DIR is missing or has no files to index."""
    if not os.path.isdir(DATA_DIR):
        st.error(f"Data folder '{DATA_DIR}' not found. Create it and put the "
                 "student handbook PDF inside, then restart the app.")
        st.stop()
    # Ignore hidden files like .DS_Store, since SimpleDirectoryReader skips them too
    if not [f for f in os.listdir(DATA_DIR) if not f.startswith(".")]:
        st.error(f"Data folder '{DATA_DIR}' is empty. Add the student handbook "
                 "PDF to it, then restart the app.")
        st.stop()


@st.cache_resource
def get_query_engine(api_key):
    """Configure the models, index the files in DATA_DIR, and return a query engine."""
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Babson Handbook Chatbot")

# Fail fast: the app can't do its job without a key, the data, and the engine
api_key = get_api_key()
check_data_dir()
try:
    query_engine = get_query_engine(api_key)
except errors.APIError as e:
    st.error(f"Gemini rejected the API key ({e.code}: {e.message}). Check that "
             "GEMINI_API_KEY in .env is correct, then restart the app.")
    st.stop()
except Exception as e:
    st.error(f"Couldn't build the chatbot: {e}. Check your internet connection "
             f"and the files in '{DATA_DIR}', then restart the app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the student handbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Graceful fallback: one failed question shouldn't stop the app
    try:
        response = query_engine.query(prompt).response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    except errors.APIError as e:
        st.error(f"Gemini couldn't answer ({e.code}: {e.message}). Wait a moment "
                 "and ask again.")
    except Exception as e:
        st.error(f"Couldn't get an answer: {e}. Check your internet connection "
                 "and ask again.")
