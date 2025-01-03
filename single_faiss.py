# main.py

import os
import streamlit as st
from dotenv import load_dotenv
import logging
import traceback
import warnings
import requests
from typing import List
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import ollama
from langchain_groq import ChatGroq

from langchain_community.vectorstores import FAISS
from utils import load_esab_logo, detect_machine_in_query
import preprocess  # Ensure preprocess.py is in the same directory or properly referenced

# ---------------------- Setup Logging ----------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Streamlit application started.")

# ------------------- Load Environment Variables -------------------
load_dotenv()
logger.info("Environment variables loaded.")
st.set_page_config(page_title="ESAB AI Assistant", layout="wide")
st.markdown("<style>body{font-family: 'Arial';}</style>", unsafe_allow_html=True)
warnings.filterwarnings("ignore")

# ------------------------ Configuration ------------------------
AWS_IP = "15.207.109.112"  # Adjust as needed
AWS_PORT = "11434"
pdf_dir = 'pdfs'
FAISS_DB_DIR = "faiss_dbs"

# ---------------------- Initialize Resources with Caching ----------------------
@st.cache_resource(show_spinner=False)
def get_resources():
    """
    Initializes and returns ESAB machines and FAISS database.

    Returns:
        Tuple[List[str], FAISS]: List of ESAB machines and FAISS database object.
    """
    logger.info("Initializing resources - should only appear once.")
    esab_machines, faiss_db = preprocess.initialize_resources()
    if not esab_machines:
        logger.error("No valid ESAB machines found during initialization.")
    return esab_machines, faiss_db

try:
    esab_machines, faiss_db = get_resources()
    logger.info(f"Loaded FAISS_DB with {len(faiss_db.index_to_docstore_id)} documents.")
except Exception as e:
    logger.error(f"Failed to initialize resources: {e}")
    logger.debug(traceback.format_exc())  # Detailed traceback for debugging
    st.error("Failed to initialize resources. Please check the logs for more details.")
    st.stop()

if not esab_machines:
    logger.error("No valid machine manuals found in the 'pdfs' directory.")
    st.error("No valid machine manuals found. Please ensure that the PDF files are present and contain 'dimensions' text.")
    st.stop()
else:
    logger.info(f"ESAB_MACHINES initialized with {len(esab_machines)} machines.")

# ---------------------- Streamlit State Initialization ----------------------
if 'machine_chat_history' not in st.session_state:
    st.session_state.machine_chat_history = {}
    logger.info("Initialized 'machine_chat_history' in session state.")
if 'current_machines' not in st.session_state:
    st.session_state.current_machines = []
    logger.info("Initialized 'current_machines' in session state.")
if 'retrieval_chain' not in st.session_state or 'memory' not in st.session_state:
    st.session_state.retrieval_chain, st.session_state.memory = None, None
    logger.info("Initialized 'retrieval_chain' and 'memory' in session state.")

# ---------------------- Sidebar Configuration ----------------------
manual_template = """
Welcome to the ESAB Knowledge Base!
We have troubleshooting manuals for various ESAB welding machines. These guides offer solutions for common issues, error codes, and maintenance tips to ensure optimal performance.

For specific troubleshooting steps or corrective actions, simply enter your query with the machine name or a general question about ESAB machines.
1) If you need help with a specific machine, mention the machine name in your query.
2) If you need info about machines that support a specific welding process, ask "Which machines handle TIG?".
3) Refresh the page if you want to ask questions generally about ESAB machines.
"""

st.sidebar.header("ESAB Machine Manuals")
st.sidebar.markdown(manual_template)
st.sidebar.subheader("Available Manuals")

# Use a single text block for listing manuals to reduce re-renders
manuals_list = "\n".join(f"* {m}" for m in esab_machines)
st.sidebar.write(manuals_list)

logger.info("Sidebar configured with available manuals.")

# Display ESAB Logo if exists
logo = load_esab_logo()
if logo:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{logo}" 
                 style="width:50px; height:50px; margin-right:10px;">
            <h1 style="margin: 0;">ESAB AI Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    logger.info("Displayed ESAB logo in the sidebar.")
else:
    st.sidebar.title("🔍 ESAB Welding Machines Knowledge Base")
    logger.info("Displayed default title as ESAB logo was not found.")

# ---------------------- Display Existing Conversation (If Any) ----------------------
if st.session_state.current_machines:
    key = "-".join(st.session_state.current_machines)
    logger.info(f"Displaying existing conversation for machines: {st.session_state.current_machines}.")
    for msg in st.session_state.machine_chat_history.get(key, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------------- LLM Initialization ----------------------
def get_llm():
    """
    Initializes the LLM (Ollama on AWS or fallback to Groq).

    Returns:
        LLM object.
    """
    logger.info("Initializing LLM.")
    if check_aws_connection():
        logger.info("Using Ollama LLM on AWS GPU server.")
        return ollama.Ollama(
            model="llama3",
            temperature=0.05,
            base_url=f"http://{AWS_IP}:{AWS_PORT}"
        )
    else:
        logger.warning("⚠️ Using Groq model instead of AWS GPU server.")
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API"),
            model_name="llama3-8b-8192",
            temperature=0.05
        )

def check_aws_connection() -> bool:
    """
    Checks connection to AWS GPU server. (Synchronous check)

    Returns:
        bool: True if connected, False otherwise.
    """
    logger.info("Checking connection to AWS GPU server.")
    try:
        response = requests.get(f"http://{AWS_IP}:{AWS_PORT}", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Connected to AWS GPU server.")
            st.sidebar.success("✅ Connected to AWS GPU server")
            return True
        else:
            logger.error(f"❌ Connection to AWS GPU server failed with status code {response.status_code}.")
            st.sidebar.error("❌ Connection to AWS GPU server failed")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Unable to connect to AWS GPU server: {e}")
        logger.debug(traceback.format_exc())  # Detailed traceback for debugging
        st.sidebar.error("❌ Unable to connect to AWS GPU server")
        return False

# ---------------------- Chain Setup ----------------------
def setup_chain(detected_machines: List[str]):
    """
    Sets up the RetrievalQA chain with a specialized prompt:
      - If user mentions a machine, focus on that machine's docs
      - Otherwise, use entire DB for general queries.

    Args:
        detected_machines (List[str]): List of detected machines in the query.

    Returns:
        Tuple[RetrievalQA, ConversationBufferMemory]: The QA chain and memory object.
    """
    logger.info(f"Setting up RetrievalQA chain for machines: {detected_machines if detected_machines else 'None'}.")
    llm = get_llm()
    prompt = PromptTemplate(
        template="""
You are an AI assistant for ESAB, knowledgeable about ESAB welding machines.

Instructions:
1. If the user specifies one or more machines, focus your answer on those machines' documents.
2. If the user does not specify any machine, use the entire knowledge base (including the welding process table).
3. Always ground your responses in the retrieved information—avoid speculations beyond what is in the knowledge base.
4. If the user asks "Which machines handle TIG?", look up the relevant 'welding_process_analysis' doc.
5. If the user asks "List out machines that you can answer," reference the 'machine_list' doc in the knowledge base.
6. **If you cannot find any relevant information to answer the question, respond generally with LLM MODEL. Do not include any instructions, related answers, or template steps in your response.**

Context: {context}
Question: {question}

Provide an accurate and concise answer, referencing only the knowledge you have retrieved.
Promote ESAB products and services where appropriate.
""",
        input_variables=["context", "question"]
    )

    if not faiss_db:
        logger.error("FAISS database is unavailable. Cannot set up RetrievalQA chain.")
        return None, None

    # If user specified machines, filter docs
    if detected_machines:
        logger.info(f"Filtering FAISS DB for specified machines: {detected_machines}.")
        relevant_docs = []
        for doc_id, doc in faiss_db.docstore._dict.items():
            machine_meta = doc.metadata.get("machine", "").lower()
            source_meta = doc.metadata.get("source", "")
            # Only keep docs for specified machines or process/machine_list docs
            if machine_meta in [m.lower() for m in detected_machines]:
                relevant_docs.append(doc)
            elif source_meta in ("welding_process_analysis", "machine_list"):
                relevant_docs.append(doc)
        if relevant_docs:
            # Make a smaller FAISS with those docs
            new_db = FAISS.from_documents(relevant_docs, faiss_db.embedding_function)
            retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 13})
            logger.info(f"Using filtered FAISS DB for {detected_machines} + analysis docs.")
        else:
            # Fallback: no matching docs => use entire DB
            retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 13})
            logger.warning("No matching docs for the specified machines. Using full FAISS DB.")
    else:
        # No machine => entire DB
        retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 13})
        logger.info("No specific machine detected. Using entire knowledge base.")

    # Build the QA chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("RetrievalQA chain successfully created.")
    except Exception as e:
        logger.error(f"Failed to create RetrievalQA chain: {e}")
        logger.debug(traceback.format_exc())  # Detailed traceback for debugging
        return None, None

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    logger.info("ConversationBufferMemory initialized for chat history.")
    return qa_chain, memory

# ---------------------- Process Query ----------------------
def process_query(user_query: str, detected_machines: List[str]):
    """
    Processes the user query with the chain.

    Args:
        user_query (str): The user's input query.
        detected_machines (List[str]): List of detected machines in the query.

    Returns:
        Dict: Response from the QA chain or error message.
    """
    logger.info(f"Processing user query: '{user_query}' with detected machines: {detected_machines}.")
    try:
        # NOTE: We are no longer forcing a fallback to st.session_state.current_machines
        # if user doesn't mention machines. We'll directly use `detected_machines`
        # to decide between machine-specific or general context.

        qa_chain, memory = setup_chain(detected_machines)
        if not qa_chain:
            logger.error("No RetrievalQA chain available. Cannot process query.")
            return {"error": "No FAISS database available. Cannot process query."}

        response = qa_chain.invoke({"query": user_query})
        logger.info("RetrievalQA chain invoked successfully.")

        # Keep chat memory for multi-turn
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(response["result"])

        logger.info("Chat memory updated with the latest interaction.")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.debug(traceback.format_exc())  # Detailed traceback for debugging
        return {"error": "Unable to process the query at the moment."}

# ---------------------- Chat Interface ----------------------
GREETING_RESPONSES = ["hi", "hello", "hey", "hola", "howdy", "greetings"]

if prompt := st.chat_input("Ask a question about ESAB or specific machines..."):
    logger.info(f"User submitted a new prompt: '{prompt}'.")
    detected_machines = detect_machine_in_query(prompt, esab_machines)

    # Handle simple greetings
    if prompt.lower() in GREETING_RESPONSES:
        st.chat_message("assistant").markdown("Hello, how may I assist you today?")
        st.session_state.machine_chat_history.setdefault('general', []).append({
            "role": "assistant",
            "content": "Hello, how may I assist you today?"
        })
        logger.info("Responded to user greeting.")
        st.stop()

    # If new machines are detected, or chain isn't set, build a new chain
    if detected_machines:
        # If user explicitly mentioned a machine, we switch context to that machine:
        st.session_state.current_machines = detected_machines
        st.session_state.retrieval_chain, st.session_state.memory = setup_chain(detected_machines)
        key = "-".join(detected_machines)
        st.session_state.machine_chat_history[key] = st.session_state.machine_chat_history.get(key, [])
        st.info(f"🔍 Context set for: {', '.join(detected_machines)}")
        logger.info(f"Context set for machines: {detected_machines}.")
    else:
        # If no machine is detected => always go to general context
        st.session_state.current_machines = []
        st.session_state.retrieval_chain, st.session_state.memory = setup_chain([])
        key = "general"
        st.session_state.machine_chat_history[key] = st.session_state.machine_chat_history.get(key, [])
        st.info("🔍 Using entire knowledge base for general queries.")
        logger.info("Using entire knowledge base for general queries.")

    # Ensure chain is ready
    if not st.session_state.retrieval_chain:
        st.session_state.retrieval_chain, st.session_state.memory = setup_chain(st.session_state.current_machines)
        logger.info("RetrievalQA chain ensured to be ready.")

    # Show the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.machine_chat_history.setdefault(key, []).append({
        "role": "user",
        "content": prompt
    })
    logger.info(f"User prompt added to chat history under key '{key}'.")

    # Generate the AI response
    with st.chat_message("assistant"):
        try:
            response = process_query(prompt, detected_machines)
            if "error" in response:
                st.markdown(response["error"])
                st.session_state.machine_chat_history[key].append({
                    "role": "assistant",
                    "content": response["error"]
                })
                logger.error(f"Error in response: {response['error']}")
            else:
                st.markdown(response["result"])
                st.session_state.machine_chat_history[key].append({
                    "role": "assistant",
                    "content": response["result"]
                })
                logger.info("AI response generated and added to chat history.")
                # Update memory
                if st.session_state.memory:
                    st.session_state.memory.save_context(
                        {"input": prompt},
                        {"output": response["result"]}
                    )
                    logger.info("Chat memory updated with the latest context.")
        except Exception as e:
            err_msg = f"An error occurred: {str(e)}"
            logger.error(f"Error generating response: {traceback.format_exc()}")
            st.error(err_msg)
            st.session_state.machine_chat_history[key].append({
                "role": "assistant",
                "content": err_msg
            })
            logger.info("Displayed error message to the user.")
