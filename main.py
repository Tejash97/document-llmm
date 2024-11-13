import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import fitz  # PyMuPDF for extracting charts/images
import time

load_dotenv()

st.title("Document Q&A with Chart Recognition")

# Groq API Key and LLM setup
groq_api_key = "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"
llm = ChatGroq(
    temperature=0.4,
    groq_api_key=groq_api_key,
    model_name="llama-3.2-90b-text-preview"
)

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """Extracts images (charts) from a PDF."""
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)

    for i in range(len(doc)):
        page = doc[i]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = os.path.join(output_folder, f"page-{i+1}-{img_index}.{image_ext}")
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
    return output_folder

def embed_pdf_with_charts(uploaded_file):
    """Embedding logic for PDFs with image extraction."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        # Extract charts/images
        image_folder = extract_images_from_pdf(temp_file_path)
        st.session_state.images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

        # Load the saved temporary file using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        if not docs:
            st.write("No documents loaded. Please ensure valid PDFs are uploaded.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        if not final_documents:
            st.write("No documents were split. Please check document loading and splitting logic.")
            return

        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Button to trigger embedding
if uploaded_file:
    st.sidebar.button("Process PDF", on_click=embed_pdf_with_charts, args=(uploaded_file,))

# Display extracted charts/images
if "images" in st.session_state and st.session_state.images:
    st.write("Extracted Charts/Images:")
    for image_path in st.session_state.images:
        st.image(image_path, caption=os.path.basename(image_path))

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Generate Answer"):
    if "vectors" not in st.session_state:
        st.write("Please upload a PDF to generate embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response Time: {time.process_time() - start:.2f} seconds")
        st.write(response['answer'])
