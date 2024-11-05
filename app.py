<<<<<<< HEAD
import os
import base64
import streamlit as st
from fuzzywuzzy import process
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Import prompt functions from each machine's specific prompt file
from prompt_warrior_edge import get_prompt as get_warrior_edge_prompt
from prompt_warrior_500i import get_prompt as get_warrior_500i_prompt
from prompt_fabricator_em_400i_500i import get_prompt as get_fabricator_em_400i_500i_prompt
from prompt_fabricator_et_410ip import get_prompt as get_fabricator_et_410ip_prompt

# Define the list of ESAB machines with supported knowledge bases
ESAB_MACHINES = ["Warrior-Edge", "Warrior 500i", "Fabricator EM 400i&500i", "Fabricator ET 410iP"]

# Define the FAISS database directory
FAISS_DB_DIR = "faiss_dbs"
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Define greeting responses
GREETING_RESPONSES = ["hi", "hello", "hey", "hola", "howdy", "greetings"]

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
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(db_path)
        print(f"Database for {machine} created and saved.")
        return db

def get_prompt_template(machine):
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

    return retrieval_chain, ConversationBufferMemory(return_messages=True, memory_key="chat_history")

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

if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

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
    st.session_state.retrieval_chain, _ = setup_chain(selected_machine)
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
    # Check if input is a greeting
    if prompt.lower() in GREETING_RESPONSES:
        greeting_response = ("Hello, how may I assist you? If your query is regarding ESAB machines, "
                             "please select one from the dropdown above to start your query.")
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(greeting_response)
        st.session_state.machine_chat_history.setdefault('general', []).append({"role": "assistant", "content": greeting_response})
    else:
        # If it's not a greeting, process it normally
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.machine_chat_history[st.session_state.current_machine].append({"role": "user", "content": prompt})

        # Generate response based on selected machine
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                response = st.session_state.retrieval_chain.invoke({
                    "input": prompt,
                    "machine": st.session_state.current_machine,
                    "chat_history": st.session_state.machine_chat_history[st.session_state.current_machine]
                })
                answer = response['answer']
                status.update(label="Response ready!", state="complete", expanded=False)
            st.markdown(answer)
            st.session_state.machine_chat_history[st.session_state.current_machine].append({"role": "assistant", "content": answer})
=======
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
import pdfplumber
import textwrap
import warnings

warnings.filterwarnings("ignore")

llm = Ollama(model='llama3.1', temperature=0.05)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'greeting_given' not in st.session_state:
    st.session_state['greeting_given'] = False

# Improved function to extract text and tables from PDF
def extract_text_and_tables_from_pdf(file_path):
    text_content = []
    table_content = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract page text
            text = page.extract_text()
            if text:
                text_content.append(text)

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                if table:
                    # Clean and verify table data
                    cleaned_table = [["" if cell is None else str(cell) for cell in row] for row in table]
                    formatted_table = "\n".join(["\t".join(row) for row in cleaned_table if any(row)])
                    table_content.append(formatted_table)

    combined_text = "\n\n".join(text_content)
    combined_tables = "\n\n".join(table_content)

    return combined_text, combined_tables

# Function to split documents
def split_docs(docs, chunk_size=1000, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents=docs)
    return chunks

# Embedding model loader
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': normalize_embedding}
    )

# Create embeddings
def create_embeddings(chunks, embedding_model, storing_path='vectorstore'):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

# Updated prompt template
template = '''
    You are a technical assistant specialized in handling welding equipment, specifically the Warrior 500i CC/CV. 
    You provide precise and accurate responses based on the manual, focusing on technical data. 
    When responding to user queries, follow these instructions:
    
    - If the user asks about **safety**, make sure to include any relevant standards like ANSI/ASC Standard Z49.1.
    - If the user asks for a **full section or Table of Contents**, provide the complete content without omissions.
    - For longer content, make sure the response is split clearly into sub-sections for readability.
    - If the user asks for a **list of types, examples, or uses**, provide them in a bullet-point format.
    - If the user asks for **specific details or explanations**, provide them in a structured paragraph and some bullet points format.
    - If the user requests a **step-by-step guide**, break down the information into numbered steps.
    - If the user asks for **troubleshooting or errors**, list the errors and their corresponding solutions clearly, making sure not to skip any points.
    - When providing **key points** for a query response provide as bullet points.
    - For **comparisons**, ensure to highlight the differences and similarities clearly.
    
    Always ensure that the response is clear, concise, and contextual. Use the previous chat responses to maintain relevance but avoid unnecessary repetition.
    
    "{context}"

    "### User:"
    "{question}"

    ### Response:
'''

# Function to format responses
def format_response(response_text, user_query):
    # Keywords for different formats
    list_keywords = ['list', 'types', 'examples', 'uses', 'features', 'characteristics', 'mention', 'provide']
    step_keywords = ['steps', 'how to', 'process', 'procedure', 'guide']
    error_keywords = ['error', 'troubleshooting', 'fault']

    # Check if the question indicates a list or step-by-step format
    if any(keyword in user_query.lower() for keyword in list_keywords):
        response_lines = response_text.split('\n')
        formatted_response = "\n\n".join([f"- {line.strip()}" for line in response_lines if line.strip()])

    elif any(keyword in user_query.lower() for keyword in step_keywords):
        # Format response as numbered steps
        response_lines = response_text.split('\n')
        formatted_response = "\n\n".join([f"{idx + 1}. {line.strip().lstrip('1234567890. ')}" for idx, line in enumerate(response_lines) if line.strip()])

    elif any(keyword in user_query.lower() for keyword in error_keywords):
        # Specific format for errors in a troubleshooting context
        response_lines = response_text.split('\n')
        formatted_response = "\n\n".join([f"**{line.strip()}**" if "Error" in line else f"  - {line.strip()}" for line in response_lines if line.strip()])

    else:
        # Default to paragraph format for detailed explanations
        formatted_response = textwrap.fill(response_text, width=6000)

    return formatted_response

# QA Chain Loader
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Load embedding model
embed = load_embedding_model(model_path="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF and extract data
file_path = "Warrior 500i.pdf"
text_data, table_data = extract_text_and_tables_from_pdf(file_path)

# Combine content into Document structure
combined_content = [Document(page_content=text_data, metadata={'source': 'Text Data'}), Document(page_content=table_data, metadata={'source': 'Table Data'})]

# Split documents into chunks
documents = split_docs(combined_content)

# Create embeddings and vector store
vectorstore = create_embeddings(documents, embed)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create prompt
prompt = PromptTemplate.from_template(template)
chain = load_qa_chain(retriever, llm, prompt)

# Chat history management
def get_chat_history():
    chat_history_str = ""
    for chat in st.session_state['chat_history']:
        chat_history_str += f"User: {chat['user']}\nBot: {chat['bot']}\n"
    return chat_history_str.strip()

# Function to add chat bubbles with custom styles
def add_chat_bubble(role, text):
    if role == 'user':
        # User's message style
        st.markdown(
            f"""
            <div style='background-color: #f0f0f0; border-radius: 15px; padding: 10px; margin-bottom: 10px; width: fit-content; max-width: 70%; margin-left: auto;'>
                <strong style='color: #e74c3c;'>🧑</strong>
                <span style='margin-left: 10px;'>{text}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    elif role == 'bot':
        # Bot's message style
        st.markdown(
            f"""
            <div style='background-color: #e8f8e8; border-radius: 15px; padding: 10px; margin-bottom: 10px; width: fit-content; max-width: 70%;'>
                <strong style='color: #3498db;'>🤖</strong>
                <span style='margin-left: 10px;'>{text}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Streamlit UI setup
st.title("Warrior 500i Chatbot")

user_input = st.text_input("Ask a question about Warrior 500i equipment")

greetings = ['hi', 'hello', 'hey', 'hola', 'howdy', 'greetings']

if user_input:
    try:
        if any(greet in user_input.lower() for greet in greetings) and len(user_input.split()) <= 3 and not st.session_state['greeting_given']:
            bot_response = "Hello! How may I assist you today?"
            st.session_state['greeting_given'] = True
        else:
            # Build up chat history to pass as context
            chat_history = get_chat_history()
            context = chat_history + f"\n\n### New Query:\nUser: {user_input}\nBot:"

            # Get the response from the LLM with context
            response = chain({'query': user_input, 'context': context})
            formatted_response = format_response(response['result'], user_input)
            bot_response = formatted_response

        # Update the chat history with the current interaction
        st.session_state['chat_history'].append({'user': user_input, 'bot': bot_response})

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display chat history with chat bubbles
if st.session_state['chat_history']:
    for chat in st.session_state['chat_history']:
        add_chat_bubble('user', chat['user'])
        add_chat_bubble('bot', chat['bot'])
>>>>>>> ac2cad37fa9338de0369c29755612071a8a45d63
