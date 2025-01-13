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
from PIL import Image
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

def get_conversational_chain(model_name):
    prompt_template = """
    Extract and present comprehensive insights from the provided context in the PDF, ensuring accuracy and clarity. Your response must address the following:

    1. **Insight Accuracy**: Provide detailed and accurate insights based only on the provided context, refraining from any assumptions.
    2. **Contextual Relevance**: Ensure the response is closely tied to the context, offering any additional or supplementary information that could enhance the user's understanding of the topic.
    3. **Numerical Data Representation**: If the information includes numerical data, always mention the relevant units or currency (e.g., dollars, rupees, percentages, etc.). Specify quantities in their exact form, ensuring no data is overlooked.
    4. **Specificity in Data**: Where applicable, break down data into categories or subcomponents (e.g., financial breakdowns by department, year, or region, etc.), and provide comparisons or trends if present in the context. 
    5. **Extra Insights**: If possible, identify any noteworthy patterns, anomalies, or areas for deeper exploration based on the provided data. Highlight key takeaways or summarizations to help the user grasp the significance of the data.
    6. **Page References**: Always mention the page number(s) from which you extracted the information to allow for easy reference.
    7. **Unavailable Information**: If the requested insight is not available in the provided context, explicitly state, "Answer is not available in the context." Do not speculate or provide vague information.
    8. **Clarity & Detail**: Use clear, concise language to avoid confusion. If there are technical terms, provide brief definitions or explanations if needed to ensure clarity.

    <context>
    {context}
    <context>
    Questions: {input}
    """
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)
    return chain

def user_input(user_question,model_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    chain = get_conversational_chain(model_name)
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
    st.set_page_config(page_title="ChatPDF", 
                       page_icon="assets/favicon.png", 
                       layout="wide", 
                       initial_sidebar_state="expanded",
                       menu_items={'About': "# This is a header. This is an *extremely* cool app!"})

    # logo_url = "https://i.postimg.cc/yY3dnD9S/logo.png"  
    # col1, col2 = st.columns([1, 17])
    # with col1:
    #     st.image(logo_url, width=55) 
    # with col2:
    #     st.header("ChatPDF")
    col1, col2, col3 = st.columns((1, 2, 1))
    with col2:
        st.image(Image.open("assets/Header.png"))

    
    st.markdown("""##### Here are some suggestions for you:
    ▶️ Summarize the document.
    ▶️ List keywords and Identify key terms.
    ▶️ What is the primary goal or objective of this document? """, unsafe_allow_html=True)   
    
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "show_confirmation" not in st.session_state:
        st.session_state.show_confirmation = False
    if "reset_confirmed" not in st.session_state:
        st.session_state.reset_confirmed = False

    # Model options for user selection
    model_options = {
        
        "Gemma2-9B": "gemma2-9b-it",
        "Llama3-8b":"llama3-8b-8192",
        "Llama3-70B": "llama3-70b-8192",
        "Llama 3.1 70B": "llama-3.1-70b-versatile",
        "Mixtral-8x7B": "mixtral-8x7b-32768",

    }
    
    with st.sidebar:
        st.title("Menu")
        
        # Add model selection in the sidebar
        selected_model = st.selectbox(
            "Select LLM Model", options=list(model_options.keys())
        )
        selected_model_name = model_options[selected_model]
        st.write(f"Selected model: **{selected_model_name}**")
        
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process", icon=":material/forward:",use_container_width=True):
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
                icon=":material/download:",
                mime="application/pdf",
                use_container_width=True
            )

        if st.button("Reset Chat", icon=":material/refresh:",use_container_width=True):
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
        st.markdown("<p style=' margin-top: 200px;'>Powered by Groq</p>", unsafe_allow_html=True)

    # Use the selected model for the conversation
    for msg in st.session_state.history:
        st.chat_message(msg["type"], avatar=msg["avatar"]).write(msg["content"])
    
    if user_question := st.chat_input("Ask a Question from the PDF Files"):
        st.chat_message("human", avatar='assets/human.png').write(user_question)
        answer, context = user_input(user_question, selected_model_name)
        st.chat_message("ai", avatar='assets/ai.png').write(answer)
        
        # Append conversation to history
        st.session_state.history.append({"type": "human", "content": user_question, "avatar": 'assets/human.png'})
        st.session_state.history.append({"type": "ai", "content": answer, "avatar": 'assets/ai.png'})
        
        # Store the last question for regenerating
        st.session_state.last_question = user_question

    # Regenerate button to redo the response for the last question
    if st.session_state.last_question:
        if st.button("Regenerate"):
            # Display the last question again
            st.chat_message("human", avatar='assets/human.png').write(st.session_state.last_question)
            
            # Generate new response for the same question
            answer, context = user_input(st.session_state.last_question, selected_model_name)
            
            # Display regenerated response
            st.chat_message("ai", avatar='assets/ai.png').write(answer)
            
            # Add regenerated response to history
            st.session_state.history.append({"type": "ai", "content": answer, "avatar": 'assets/ai.png'})


if __name__ == "__main__":
    main()




