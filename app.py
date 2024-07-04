import os   
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from fpdf import FPDF
import time

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question, You are an expert in financial analysis. Extract and present detailed financial information from the provided context. 
    Ensure the accuracy and clarity of the information. When extracting numerical data, include the unit or currency prefix (e.g., dollars, rupees). 
    Additionally, provide any relevant extra information on the topic to give better context to the user. 
    If the answer is not available in the context, state, "Answer is not available in the context." Do not provide incorrect information.
    Always mention the page number from which you extracted the information.
    <context>
    {context}
    <context>
    Questions: {input}
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    chain = get_conversational_chain()
    retrieval_chain = create_retrieval_chain(retriever, chain)
    response = retrieval_chain.invoke({'input': user_question})
    return response['answer'], response['context']

def generate_pdf(chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for msg in chat_history:
        msg_type = msg['type'].capitalize()
        avatar = msg['avatar']
        content = msg['content']
        # Handling encoding issues
        content = content.encode('latin1', 'replace').decode('latin1')
        pdf.multi_cell(0, 10, f"{msg_type} ({avatar}): {content}")
    return pdf

def main():
    st.set_page_config(page_title="Chat with PDF", 
                       page_icon="https://i.postimg.cc/RZzRwFCw/tab-icon.png", 
                       layout="wide", 
                       initial_sidebar_state="expanded",
                       menu_items={'About': "# This is a header. This is an *extremely* cool app!"})
   
    logo_url = "https://i.postimg.cc/yY3dnD9S/logo.png"  
    col1, col2 = st.columns([1, 17])
    with col1:
        st.image(logo_url, width=55) 
    with col2:
        st.header("ChatPDF")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_confirmation" not in st.session_state:
        st.session_state.show_confirmation = False
    if "reset_confirmed" not in st.session_state:
        st.session_state.reset_confirmed = False

    for msg in st.session_state.history:
        st.chat_message(msg["type"], avatar=msg["avatar"]).write(msg["content"])
    
    if user_question := st.chat_input("Ask a Question from the PDF Files"):
        st.chat_message("human", avatar='https://i.postimg.cc/261JMMfm/user-3.png').write(user_question)
        answer, context = user_input(user_question)
        st.chat_message("ai", avatar='https://i.postimg.cc/fLSW0H9V/chat-16273634.png').write(answer)
        st.session_state.history.append({"type": "human", "content": user_question, "avatar": 'https://i.postimg.cc/261JMMfm/user-3.png'})
        st.session_state.history.append({"type": "ai", "content": answer, "avatar": 'https://i.postimg.cc/fLSW0H9V/chat-16273634.png'})

    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text_with_pages = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text_with_pages)
                    get_vector_store(text_chunks)
                    st.success("Done")

        if st.session_state.history:
            pdf = generate_pdf(st.session_state.history)
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button(
                label="Download Chat History",
                data=pdf_output,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

        if st.button("Reset Chat"):
            st.session_state.show_confirmation = True

        if st.session_state.show_confirmation:
            st.error("Are you sure you want to delete the chat history?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm"):
                    st.session_state.history = []
                    st.session_state.show_confirmation = False
                    st.session_state.reset_confirmed = True
                    st.experimental_rerun()
            with col2:
                if st.button("Close"):
                    st.session_state.show_confirmation = False
                    st.experimental_rerun()

        if st.session_state.reset_confirmed:
            st.success("Chat history has been reset")
            st.session_state.reset_confirmed = False 
        st.markdown("<p style=' margin-top: 200px;'>Powered by Llama 3</p>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()





