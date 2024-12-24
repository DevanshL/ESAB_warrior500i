## handles general queries also

import os
import base64
import re
import streamlit as st
import pdfplumber
import glob
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import ollama
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fuzzywuzzy import fuzz, process
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import warnings
import requests
from dotenv import load_dotenv
import logging
import traceback
from typing import Dict, List

# ---------------------- Setup Logging ----------------------
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------- Load Environment Variables -------------------
load_dotenv()
warnings.filterwarnings("ignore")

# ------------------------ Configuration ------------------------
AWS_IP = "15.207.109.112"  # Adjust as needed
AWS_PORT = "11434"

pdf_dir = 'pdfs'
FAISS_DB_DIR = "faiss_dbs"
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Greeting responses
GREETING_RESPONSES = ["hi", "hello", "hey", "hola", "howdy", "greetings"]

# --------------------------------------------------------------------------------
# Utility Functions to Convert DataFrame, Extract Sections, and Detect Processes
# --------------------------------------------------------------------------------

def dataframe_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Converts a DataFrame of welding processes and machines into
    a list of string-based Documents for indexing and retrieval.
    """
    documents = []
    for _, row in df.iterrows():
        process = row["Welding Process"]
        machines = row["Machines"]
        content = f"The welding process {process} is compatible with the following machines: {machines}."
        documents.append(Document(page_content=content))
    return documents

def extract_sections(pdf_paths: List[str], sections_to_extract: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Extracts specified sections from the given PDF files.
    Returns a nested dict: {machine_name: {section: "content", ...}, ...}
    """
    extracted_data = {}
    # Patterns for the headers we care about
    section_header_patterns = {
        section: [
            re.compile(rf'^\d+\.\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^[IVXLCDM]+\.\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^\d+\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^[IVXLCDM]+\s+{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^{re.escape(section.upper())}$', re.IGNORECASE),
            re.compile(rf'^\d+\.\s+{re.escape(section.upper())}[:\-]$', re.IGNORECASE),
            re.compile(rf'^[IVXLCDM]+\.\s+{re.escape(section.upper())}[:\-]$', re.IGNORECASE),
            re.compile(rf'^{re.escape(section.upper())}[:\-]$', re.IGNORECASE),
        ]
        for section in sections_to_extract
    }
    # Pattern for any new "unwanted" section header (stop capturing text if we hit a new one)
    any_section_header_pattern = re.compile(r'^\d+\s+[A-Z\s\-]+$', re.IGNORECASE)

    for pdf_path in pdf_paths:
        machine_name = os.path.splitext(os.path.basename(pdf_path))[0]
        extracted_data[machine_name] = {section: "" for section in sections_to_extract}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                current_section = None
                collected_text = {section: "" for section in sections_to_extract}

                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    lines = page_text.split('\n')

                    for i, line in enumerate(lines):
                        line_stripped = line.strip()
                        line_upper = line_stripped.upper()

                        # Check if line is a header for a desired section
                        matched_section = None
                        for section, patterns in section_header_patterns.items():
                            for pattern in patterns:
                                if pattern.match(line_stripped):
                                    matched_section = section
                                    break
                            if matched_section:
                                break

                        if matched_section:
                            current_section = matched_section
                            # Collect the rest of the page after the matched header
                            if i + 1 < len(lines):
                                collected_text[current_section] += '\n'.join(lines[i+1:]) + '\n'
                            break  # move to next page
                        elif any_section_header_pattern.match(line_upper):
                            # Found a new section not in sections_to_extract
                            current_section = None
                            continue

                        # If we're inside a desired section, accumulate text
                        if current_section:
                            collected_text[current_section] += line + '\n'

                # Final assignment of extracted text
                for section in sections_to_extract:
                    content = collected_text[section].strip()
                    if content:
                        extracted_data[machine_name][section] = content
                    else:
                        extracted_data[machine_name][section] = f"[INFO] No {section.upper()} section found."
        except Exception as e:
            logger.error(f"[ERROR] Processing {pdf_path}: {e}")

    return extracted_data

def detect_welding_processes(sections: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Detects which machines support MMA, MIG/MAG, TIG, FCAW, etc.
    Returns a DataFrame with 'Welding Process' and 'Compatible Machines'.
    """
    welding_processes = {
        "MMA": ["MMA", "STICK", "SMAW"],
        "MIG/MAG": ["MIG", "MAG", "GMAW"],
        "TIG": ["TIG", "GTAW"],
        "FCAW": ["FCAW", "FLUX CORED"]
    }
    process_machines = {proc: [] for proc in welding_processes.keys()}

    for machine, content_dict in sections.items():
        # Combine "INTRODUCTION" + "TECHNICAL DATA" text
        combined_content = ""
        for sec_name in ["INTRODUCTION", "TECHNICAL DATA"]:
            sec_text = content_dict.get(sec_name, "")
            combined_content += sec_text.upper() + " "

        # Match processes
        for process_name, keywords in welding_processes.items():
            for kw in keywords:
                pattern = r'\b' + re.escape(kw.upper()) + r'\b'
                if re.search(pattern, combined_content):
                    process_machines[process_name].append(machine)
                    break

    data = {"Welding Process": [], "Machines": []}
    for process_name, machines_list in process_machines.items():
        if machines_list:
            data["Welding Process"].append(process_name)
            data["Machines"].append(", ".join(machines_list))

    df = pd.DataFrame(data)
    df = df.sort_values("Welding Process").reset_index(drop=True)
    return df

# --------------------------------------------------------------------------------
# Existing Streamlit App Code
# --------------------------------------------------------------------------------

# 1) Validate PDFs by searching for "dimensions" text
def is_pdf_valid(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and "dimensions" in text.lower():
                    return True
        return False
    except Exception as e:
        logger.error(f"Error validating PDF {pdf_path}: {e}")
        return False

ESAB_MACHINES = []
for manual_name in os.listdir(pdf_dir):
    if manual_name.lower().endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, manual_name)
        machine_name = manual_name[:-4]
        if is_pdf_valid(pdf_path):
            ESAB_MACHINES.append(machine_name)
        else:
            logger.warning(f"PDF for '{machine_name}' is invalid or lacks 'dimensions' data. Skipping.")

if not ESAB_MACHINES:
    logger.error("No valid machine manuals found in the 'pdfs' directory.")
    st.error("No valid machine manuals found. Please ensure that the PDF files are present and contain 'dimensions' text.")
    st.stop()

GROQ_API_KEY = os.getenv("GROQ_API")
GOOGLE_API_KEY = os.getenv("GOOGLE_API")

def check_aws_connection():
    """
    Checks connection to AWS GPU server.
    """
    try:
        response = requests.get(f"http://{AWS_IP}:{AWS_PORT}", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ Connected to AWS GPU server")
            return True
        else:
            st.sidebar.error("❌ Connection to AWS GPU server failed")
            return False
    except requests.exceptions.RequestException:
        st.sidebar.error("❌ Unable to connect to AWS GPU server")
        return False

def get_llm():
    """
    Initializes the LLM (Ollama on AWS or fallback to Groq).
    """
    if check_aws_connection():
        return ollama.Ollama(
            model="llama3",
            temperature=0.05,
            base_url=f"http://{AWS_IP}:{AWS_PORT}"
        )
    else:
        st.sidebar.warning("⚠️ Using Groq model instead of AWS GPU server.")
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.05
        )

manual_template = """
Welcome to the ESAB Knowledge Base!
We have troubleshooting manuals for various ESAB welding machines. These guides offer solutions for common issues, error codes, and maintenance tips to ensure optimal performance.

For specific troubleshooting steps or corrective actions, simply enter your query with the machine name or a general question about ESAB machines.
1) If you need help with a specific machine, mention the machine name in your query.
2) If you need info about machines that support a specific welding process, ask "Which machines handle TIG?".
3) Refresh the page if you want to asks questions generally about ESAB machines.
"""

def extract_all_content_as_documents(pdf_paths):
    """
    Convert each PDF's text + tables into Document objects for FAISS indexing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = []
    for pdf_path in pdf_paths:
        machine_name = os.path.splitext(os.path.basename(pdf_path))[0]
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text:
                    split_texts = text_splitter.split_text(page_text)
                    for idx, chunk in enumerate(split_texts):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "machine": machine_name,
                                "page": page_number,
                                "chunk_idx": idx
                            }
                        ))
                # Convert tables to CSV
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0]).fillna("")
                            table_csv = df.to_csv(index=False)
                            documents.append(Document(
                                page_content=table_csv,
                                metadata={
                                    "machine": machine_name,
                                    "page": page_number,
                                    "table_idx": table_idx
                                }
                            ))
    return documents

@st.cache_resource
def load_or_create_faiss_db():
    """
    Builds or loads a local FAISS database of all PDF content + 
    the welding process analysis + the machine list doc.
    """
    faiss_db_path = os.path.join(FAISS_DB_DIR, "combined_faiss_db")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # If FAISS DB already exists, load it
    if os.path.exists(faiss_db_path):
        logger.info("Loading existing FAISS database...")
        db = FAISS.load_local(faiss_db_path, embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS database loaded successfully.")
        return db

    # Otherwise create a new one
    logger.info("Creating new FAISS database...")
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_paths:
        logger.error(f"No PDF files found in {pdf_dir} directory.")
        st.error(f"No PDF files found in {pdf_dir} directory.")
        return None

    # 1) Extract base documents from PDFs
    documents = extract_all_content_as_documents(pdf_paths)
    logger.info(f"Extracted {len(documents)} documents from PDFs.")

    # 2) Also create a doc listing all ESAB machines
    all_machines_text = "ESAB Machines List:\n" + "\n".join(ESAB_MACHINES)
    machine_list_doc = Document(
        page_content=all_machines_text,
        metadata={"source": "machine_list"}
    )
    documents.append(machine_list_doc)

    # 3) Extract sections and detect welding processes
    sections_to_extract = ["INTRODUCTION", "TECHNICAL DATA"]
    extracted_sections = extract_sections(pdf_paths, sections_to_extract)
    process_df = detect_welding_processes(extracted_sections)

    if not process_df.empty:
        # Convert that DataFrame into Document objects
        process_documents = dataframe_to_documents(process_df)
        # Add them to the list of documents
        documents.extend(process_documents)
        logger.info(f"Added {len(process_documents)} welding process documents to the knowledge base.")
    else:
        logger.info("No welding processes identified or no relevant sections found.")

    # 4) Build the FAISS DB
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(faiss_db_path)
    logger.info(f"FAISS database created and saved to {faiss_db_path}.")
    return db

def detect_machine_in_query(query):
    """
    Attempt to detect specific machine names from the user's query.
    """
    query_lower = query.lower()
    machine_names = {m.lower(): m for m in ESAB_MACHINES}
    detected = []

    # Exact match
    for machine_key in machine_names:
        pattern = re.compile(rf"\b{re.escape(machine_key)}\b", re.IGNORECASE)
        if pattern.findall(query_lower):
            detected.append(machine_names[machine_key])

    if detected:
        return list(set(detected))

    # Fuzzy matching
    for machine_key, machine_original in machine_names.items():
        partial_score = fuzz.partial_ratio(machine_key, query_lower)
        token_set_score = fuzz.token_set_ratio(machine_key, query_lower)
        if max(partial_score, token_set_score) > 70:
            detected.append(machine_original)

    # Simple "plural" check
    for machine_key, machine_original in machine_names.items():
        if machine_original.endswith('i') and not machine_original.endswith('ii'):
            plural = machine_original + 's'
            if plural.lower() in query_lower:
                detected.append(machine_original)

    return list(set(detected))

def setup_chain(detected_machines):
    """
    Sets up the RetrievalQA chain with a specialized prompt:
      - If user mentions a machine, focus on that machine's docs
      - Otherwise, use entire DB for general queries.
    """
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
6. **If you cannot find any relevant information to answer the question, respond generally with LLM MODEL." Do not include any instructions, related answers, or template steps in your response.**


Context: {context}
Question: {question}

Provide an accurate and concise answer, referencing only the knowledge you have retrieved.
Promote ESAB products and services where appropriate.
""",
        input_variables=["context", "question"]
    )

    db = load_or_create_faiss_db()

    # If user specified machines, filter docs
    if detected_machines:
        relevant_docs = []
        for doc_id, doc in db.docstore._dict.items():
            # If doc's 'machine' metadata is in the detected_machines list
            # or doc is a "welding_process_analysis" or "machine_list" doc, we can keep it
            machine_meta = doc.metadata.get("machine", "").lower()
            source_meta = doc.metadata.get("source", "")
            if machine_meta in [m.lower() for m in detected_machines]:
                relevant_docs.append(doc)
            elif source_meta in ("welding_process_analysis", "machine_list"):
                relevant_docs.append(doc)
        if relevant_docs:
            # Make a smaller FAISS with those docs
            new_db = FAISS.from_documents(relevant_docs, db.embedding_function)
            retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 13})
            logger.info(f"Using filtered FAISS DB for {detected_machines} + analysis docs.")
        else:
            # Fallback: no matching docs => use entire DB
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 13})
            logger.warning("No matching docs for that machine. Using full DB.")
    else:
        # No machine => entire DB
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 13})
        logger.info("No specific machine. Using entire knowledge base.")

    # Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    return qa_chain, memory

def process_query(user_query, detected_machines):
    """
    Processes the user query with the chain. 
    """
    try:
        if not detected_machines and st.session_state.current_machines:
            # Re-use any existing context if user had previously set machines
            detected_machines = st.session_state.current_machines

        qa_chain, memory = setup_chain(detected_machines)
        response = qa_chain.invoke({"query": user_query})

        # Keep chat memory for multi-turn
        memory.chat_memory.add_user_message(user_query)
        memory.chat_memory.add_ai_message(response["result"])

        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return {"error": "Unable to process the query at the moment."}

@st.cache_data
def load_esab_logo():
    """
    Loads and base64-encodes an ESAB logo (if you have esab-logo.png).
    """
    with open("esab-logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------------- Streamlit State Initialization ----------------------
if 'machine_chat_history' not in st.session_state:
    st.session_state.machine_chat_history = {}
if 'current_machines' not in st.session_state:
    st.session_state.current_machines = []
if 'retrieval_chain' not in st.session_state or 'memory' not in st.session_state:
    st.session_state.retrieval_chain, st.session_state.memory = None, None

# ---------------------- Sidebar Configuration ----------------------
st.sidebar.header("ESAB Machine Manuals")
st.sidebar.markdown(manual_template)
st.sidebar.subheader("Available Manuals")
for m in ESAB_MACHINES:
    st.sidebar.write(f"* {m}")

# Display ESAB Logo if exists
if os.path.exists("esab-logo.png"):
    logo = load_esab_logo()
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
else:
    st.title("🔍 ESAB Welding Machines Knowledge Base")

# ---------------------- Display Existing Conversation (If Any) ----------------------
if st.session_state.current_machines:
    key = "-".join(st.session_state.current_machines)
    for msg in st.session_state.machine_chat_history.get(key, []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---------------------- Chat Interface ----------------------
if prompt := st.chat_input("Ask a question about ESAB or specific machines..."):
    detected_machines = detect_machine_in_query(prompt)

    # Handle simple greetings
    if prompt.lower() in GREETING_RESPONSES:
        st.chat_message("assistant").markdown("Hello, how may I assist you today?")
        st.session_state.machine_chat_history.setdefault('general', []).append({
            "role": "assistant", 
            "content": "Hello, how may I assist you today?"
        })
        st.stop()

    # If new machines are detected, or chain isn't set, build a new chain
    if detected_machines:
        # Compare sets to see if we need a fresh context
        if (not st.session_state.retrieval_chain 
            or set(detected_machines) != set(st.session_state.current_machines)):
            st.session_state.current_machines = detected_machines
            st.session_state.retrieval_chain, st.session_state.memory = setup_chain(detected_machines)
            key = "-".join(detected_machines)
            st.session_state.machine_chat_history[key] = st.session_state.machine_chat_history.get(key, [])
            st.info(f"🔍 Context set for: {', '.join(detected_machines)}")
    else:
        # If no machine is detected => general context
        if not st.session_state.current_machines:
            st.session_state.current_machines = []
            st.session_state.retrieval_chain, st.session_state.memory = setup_chain([])
            key = "general"
            st.session_state.machine_chat_history[key] = st.session_state.machine_chat_history.get(key, [])
            st.info("🔍 Using entire knowledge base for general queries.")
        else:
            st.info(f"🔍 Continuing context for: {', '.join(st.session_state.current_machines)}")

    # Ensure chain is ready
    if not st.session_state.retrieval_chain:
        st.session_state.retrieval_chain, st.session_state.memory = setup_chain(st.session_state.current_machines)

    # Show the user's message
    key = "-".join(st.session_state.current_machines) if st.session_state.current_machines else "general"
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.machine_chat_history.setdefault(key, []).append({
        "role": "user",
        "content": prompt
    })

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
            else:
                st.markdown(response["result"])
                st.session_state.machine_chat_history[key].append({
                    "role": "assistant",
                    "content": response["result"]
                })
                # Update memory
                if st.session_state.memory:
                    st.session_state.memory.save_context(
                        {"input": prompt}, 
                        {"output": response["result"]}
                    )
        except Exception as e:
            err_msg = f"An error occurred: {str(e)}"
            logger.error(f"Error generating response: {traceback.format_exc()}")
            st.error(err_msg)
            st.session_state.machine_chat_history[key].append({
                "role": "assistant", 
                "content": err_msg
            })
