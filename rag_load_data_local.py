import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_load_data_local.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Set the Hugging Face home directory
os.environ["HF_HOME"] = os.getenv("HF_HOME")

local_llm_model = "llama3.2:3b"
local_ollama_embed_model = "bge-m3" # alternatives all-minilm:l6-v2, bge:large-en-v1.5
# local_hb_embed_model = f"{os.getenv('HF_HOME')}/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/8b3219a92973c328a8e22fadcfa821b5dc75636a/"

data_path = "./data/"

def ollama_metadata_tagger(doc, llm):
    """Generate metadata using Ollama with improved parsing"""

    # Initialize source and page variables from the document metadata
    source = os.path.basename(doc.metadata.get("source", ""))
    page = doc.metadata.get("page", -1)

    # Define the output schema
    schema = [
        ResponseSchema(name="title", description="The title of the document."),
        ResponseSchema(name="keywords", description="A comma-separeted list of keywords related to the document."),
        ResponseSchema(name="contains_numerical_data", description="A boolean indicating whether the document contains numerical data"),
        ResponseSchema(name="names", description="A list of person, organization, location, product or service names mentioned in the document. Return as comma-separated string."),
        ResponseSchema(name="document_type", description="The type of document (e.g., study, report, article, scientific paper, financial report, etc.)"),
    ]

    parser = StructuredOutputParser.from_response_schemas(schema)

    prompt = ChatPromptTemplate.from_messages(["""
        ("system", "You are a helpful assistant that generates metadata for documents."),
        ("human", "Given the following text, generate metadata according to this format:
        {format_instructions}
        
        Text: {text}
        
        Respond with a JSON object containing only the metadata."""
    ]) 

    chain = prompt | llm

    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "text": doc.page_content
    })

    try:
        # Extract the content from the AIMessage
        content = result.content if hasattr(result, 'content') else str(result)

        # Parse the output
        metadata = parser.parse(content)

        # Ensure all required fields are present with default values if missing
        metadata.setdefault("source", source)
        metadata.setdefault("page", page)
        metadata.setdefault("title", "")
        metadata.setdefault("keywords", "")
        metadata.setdefault("contains_numerical_data", False)
        metadata.setdefault("names", "")
        metadata.setdefault("document_type", "")

        return metadata

    except Exception as e:
        logger.error(f"Error processing LLM output: {content if 'content' in locals() else 'No content available'}")
        logger.error(f"Error details: {str(e)}")
        return {"source": source, "page": page, "title": "", "keywords": "","contains_numerical_data": False, "names": "", "document_type": ""}

loader = PyPDFDirectoryLoader(data_path)
pages = loader.load()
logger.debug(f"Loaded {len(pages)} pages from the PDF.")
logger.debug(f"Metadata of the five first pages: {[page.metadata for page in pages[:5]]}")

cleaned_pages = [page for page in pages if len(page.page_content.split()) > 5]
logger.debug(f"Cleaned pages count: {len(cleaned_pages)}")

# cleaned_pages = []
# for page in pages:
#     if len(page.page_content.split(" ")) > 5:
#         cleaned_pages.append(page)
# logger.debug(f"Cleaned pages count: {len(cleaned_pages)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150, # Experiment with different chunk sizes and overlaps to see what works best for your data
    separators=["\n\n", "\n", " ", ""]
)

llm = ChatOllama(
    model=local_llm_model, 
    temperature=0,
)

# Apply the Ollama metadata tagger
tagged_docs = []
for doc in cleaned_pages:
    metadata = ollama_metadata_tagger(doc, llm)
    new_doc = doc.model_copy()
    new_doc.metadata.update(metadata)
    tagged_docs.append(new_doc)
    logger.debug(f"Document: {new_doc.page_content}")
    logger.debug(f"Tagged doc metadata: {new_doc.metadata}")

split_docs = text_splitter.split_documents(tagged_docs)
logger.debug(f"Split documents count: {len(split_docs)}")
logger.debug(split_docs)

embeddings = OllamaEmbeddings(
    model=local_ollama_embed_model,
)
logger.debug(f"Embeddings model initialized.")

# embeddings = HuggingFaceEmbeddings(
#     model_name=local_hb_embed_model,
# )
# logger.debug("Embeddings model initialized.")

# # 1. Using FAISS for vector store
# faiss_vector_store = FAISS.from_documents(split_docs, embeddings)
# # Save the vector store
# faiss_vector_store.save_local(folder_path="faiss_index", index_name="index")
# print("FAISS Vector store created and saved successfully.")

# 2. Using Chroma as a vector store
try:
    chroma_vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="index",
        client_settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
        ),
        # collection_metadata={"hnsw:space": "cosine"},
    )
    if os.path.exists("./chroma_db/chroma.sqlite3"):
        logger.info("Chroma vector store created and saved successfully.")
except Exception as e:
    logger.error(f"Error creating Chroma vector store: {str(e)}")
    # Log the metadata of the documents that caused the error
    for i, doc in enumerate(split_docs):
        logger.error(f"Document {i} metadata: {doc.metadata}")
        if i >= 4:  # Log only the first 5 documents to avoid overwhelming the log
            break
    # If it's a specific error related to metadata, log more details
    if "metadata" in str(e).lower():
        logger.error(f"Full error message: {str(e)}")
        logger.error(f"First document content: {split_docs[0].page_content}")