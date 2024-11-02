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
