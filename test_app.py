import os
import base64
import streamlit as st
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from prompt_fabricator_em_400i_500i import get_prompt as get_fabricator_em_400i_500i_prompt
from prompt_fabricator_et_410ip import get_prompt as get_fabricator_et_410ip_prompt
from prompt_warrior_edge import get_prompt as get_warrior_edge_prompt
from prompt_warrior_500i import get_prompt as get_warrior_500i_prompt

# Define the list of ESAB machines with supported knowledge bases
ESAB_MACHINES = ["Warrior-Edge", "Warrior 500i", "Fabricator EM 400i&500i", "Fabricator ET 410iP"]

# Define the FAISS database directory
FAISS_DB_DIR = "faiss_dbs"
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Define greeting responses
GREETING_RESPONSES = ["hi", "hello", "hey", "hola", "howdy", "greetings"]

def extract_text_and_tables_from_pdf(pdf_path):
    text = ""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            page_tables = page.extract_tables()
            if page_tables:
                tables.extend(page_tables)

    # Format tables into structured text with headers and rows
    table_text = ""
    for table in tables:
        if table:
            headers = table[0]  # Use the first row as headers
            for row in table[1:]:  # Iterate over the rest of the rows
                # Check if row aligns with the number of headers, pad if needed
                row = row if len(row) == len(headers) else row + [""] * (len(headers) - len(row))
                row_data = {headers[i]: str(cell).replace('\n', ' ') if cell else "" for i, cell in enumerate(row)}
                row_text = " | ".join(f"{header}: {row_data[header]}" for header in headers)
                table_text += row_text + "\n"
            table_text += "\n"  # Add a newline after each table for readability

    return text + table_text

@st.cache_resource
def load_or_create_db(machine):
    db_path = os.path.join(FAISS_DB_DIR, f"db_{machine.lower().replace(' ', '_')}")
    
    if os.path.exists(db_path):
        print(f"Loading existing database for {machine}...")
        return FAISS.load_local(db_path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
    else:
        print(f"Creating new database for {machine}...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pdf_path = f"pdfs/{machine}.pdf"
        
        combined_text = extract_text_and_tables_from_pdf(pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_text(combined_text)
        
        document_objects = [Document(page_content=doc) for doc in documents]
        
        db = FAISS.from_documents(document_objects, embeddings)
        db.save_local(db_path)
        print(f"Database for {machine} created and saved.")
        return db

def get_prompt_template(machine):
    # Function to retrieve the prompt for the machine
    if machine == "Warrior-Edge":
        return get_warrior_edge_prompt()
    elif machine == "Warrior 500i":
        return get_warrior_500i_prompt()
    elif machine == "Fabricator EM 400i&500i":
        return get_fabricator_em_400i_500i_prompt()
    elif machine == "Fabricator ET 410iP":
        return get_fabricator_et_410ip_prompt()

def setup_chain(machine):
    llm = Ollama(model="llama3.1", temperature=0.05)
    prompt = get_prompt_template(machine)
    
    db = load_or_create_db(machine)
    retriever = db.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Use ConversationBufferMemory to track chat history
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    
    return retrieval_chain, memory

@st.cache_data
def load_esab_logo():
    with open("esab-logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

st.set_page_config(page_title="ESAB Machine Bot", layout="wide")

# Initialize session state variables
if 'machine_chat_history' not in st.session_state:
    st.session_state.machine_chat_history = {}

if 'current_machine' not in st.session_state:
    st.session_state.current_machine = None

if 'retrieval_chain' not in st.session_state or 'memory' not in st.session_state:
    st.session_state.retrieval_chain, st.session_state.memory = None, None

logo = load_esab_logo()
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo}" style="width:50px; height:50px; border-radius:50%; margin-right:10px;">
        <h1 style="margin: 0;">ESAB AI Assistant</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Display selectbox for machine selection
selected_machine = st.selectbox("Select an ESAB machine:", ESAB_MACHINES)
if st.session_state.current_machine != selected_machine:
    st.session_state.current_machine = selected_machine
    st.session_state.retrieval_chain, st.session_state.memory = setup_chain(selected_machine)
    st.session_state.machine_chat_history[selected_machine] = []  # Initialize chat history for the selected machine

if st.session_state.current_machine:
    st.info(f"🔍 Current knowledge base: {st.session_state.current_machine}")

# Display chat history
if st.session_state.current_machine:
    for message in st.session_state.machine_chat_history[st.session_state.current_machine]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle new user query input
if prompt := st.chat_input("Ask a question about ESAB or a specific machine"):
    if prompt.lower() in GREETING_RESPONSES:
        greeting_response = ("Hello, how may I assist you? If your query is regarding ESAB machines, "
                             "please select one from the dropdown above to start your query.")
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(greeting_response)
        st.session_state.machine_chat_history.setdefault('general', []).append({"role": "assistant", "content": greeting_response})
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.machine_chat_history[st.session_state.current_machine].append({"role": "user", "content": prompt})

        # Generate response based on selected machine and include chat history from memory
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                response = st.session_state.retrieval_chain.invoke({
                    "input": prompt,
                    "machine": st.session_state.current_machine,
                    "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]
                })
                answer = response['answer']
                status.update(label="Response ready!", state="complete", expanded=False)
            st.markdown(answer)
            st.session_state.machine_chat_history[st.session_state.current_machine].append({"role": "assistant", "content": answer})
            # Update the conversation memory with the latest interaction
            st.session_state.memory.save_context({"input": prompt}, {"answer": answer})
