import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
import streamlit as st
import logging

# Initialize logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# Create a logger for debug level
debug_logger = logging.getLogger("debug_logger")
debug_logger.setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()

# Set up environment variables for Hugging Face for offline mode
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = os.getenv("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = os.getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ["HF_HOME"] = os.getenv("HF_HOME")
os.environ["DATASETS_CACHE"] = os.getenv("DATASETS_CACHE")

# Configuration
local_llm_model = "llama3.2:3b"
local_ollama_embed_model = "bge-m3"
local_rerank_model = f"{os.getenv('HF_HOME')}/hub/models--BAAI--bge-reranker-v2-m3"
vector_store = "chroma"

def load_vector_store(store_type=vector_store):
    """Load the vector store based on the specified store type."""
    embeddings = OllamaEmbeddings(model=local_ollama_embed_model)
    
    if store_type == "faiss":
        return FAISS.load_local(
            "faiss_index", 
            embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
    elif store_type == "chroma":
        return Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="index",
            client_settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            )
        )
    else:
        raise ValueError("Invalid store_type. Choose 'faiss' or 'chroma'.")

def get_documents_from_store(store_type=vector_store):
    """Get all documents from the vector store."""
    vector_store = load_vector_store(store_type)
    if store_type == "faiss":
        # For FAISS, we need to get documents from the docstore
        return [Document(page_content=doc.page_content, metadata=doc.metadata) 
                for doc in vector_store.docstore._dict.values()]
    elif store_type == "chroma":
        # For Chroma, we need to convert the returned dict to Document objects
        results = vector_store.get()
        return [
            Document(
                page_content=content,
                metadata=metadata if metadata else {}
            )
            for content, metadata in zip(
                results["documents"],
                results.get("metadatas", [{}] * len(results["documents"]))
            )
        ]

def create_hybrid_retriever(vector_store, documents, k=8):
    """Create a hybrid retriever combining vector similarity and BM25."""
    # Create vector store retriever
    vectorstore_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, bm25_retriever],
        weights=[0.4, 0.6]  # Weights for both retrievers (sum should be 1) to be experimented with
    )
    
    return ensemble_retriever

def query_data(query, documents, store_type=vector_store):
    """Query the data using enhanced RAG with hybrid search."""
    vector_store = load_vector_store(store_type)
    llm = ChatOllama(model=local_llm_model, temperature=0)

    # Create hybrid retriever
    base_retriever = create_hybrid_retriever(vector_store, documents)

    # Define the multi-query template
    multi_query_template = """
    Based on the following question, generate two specific, concise variations of the question that aim to retrieve the most relevant information from the provided context only.
    The new queries should be phrased in a way that they answer the original question directly based on the context provided.

    Original Question: {question}
    """

    query_generation_prompt = PromptTemplate.from_template(multi_query_template)

    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
        prompt=query_generation_prompt,
    )

    # Set up re-ranking
    reranker_model = HuggingFaceCrossEncoder(model_name=local_rerank_model)
    compressor = CrossEncoderReranker(model=reranker_model, top_n=15)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )

    def retrieve_and_rerank(query):
        compressed_docs = compression_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in compressed_docs)
        return {"context": context, "question": query}

    prompt_template = """
    You are a question answering assistant. Use the following pieces of context to provide a detailed, accurate, and informative answer to the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.

    Context:
    {context}
    Question: {question}
    """

    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    response_parser = StrOutputParser()

    rag_chain = (
        RunnablePassthrough()
        | retrieve_and_rerank
        | custom_rag_prompt
        | llm
        | response_parser
    )

    return rag_chain.invoke(query)

def refine_answer(answer, query, documents, store_type=vector_store):
    """Refine the answer using hybrid retrieval."""
    vector_store = load_vector_store(store_type)
    retriever = create_hybrid_retriever(vector_store, documents, k=6)

    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
        "answer": lambda _: answer
    }
    
    refine_template = """
    Your task is to refine the existing answer based on the original context and question. Follow these steps:

    1. Review the existing answer and the question:
    <existing_answer>
    {answer}
    </existing_answer>
    Question: {question}

    2. Carefully re-read the original context:
    <original_context>
    {context}
    </original_context>

    3. Refine the existing answer by:
    - Correcting any inaccuracies
    - Ensure that any numbers or statistics are accurate and consistent
    - Adding any crucial missing information directly relevant to the question
    - Removing any information not directly related to the question
    - Ensuring the answer is concise and to the point
    - If the answer is too general or tries to cover too many aspects when trying to answer a specific question, focus on the most relevant details

    4. Provide your refined answer below. Do not explain your changes or mention that this is a refined answer. Simply give the improved answer as if it were the original response:

    Refined answer:
    """

    refine_prompt = PromptTemplate.from_template(refine_template)
    llm = ChatOllama(model=local_llm_model, temperature=0)
    response_parser = StrOutputParser()

    refine_chain = (
        retrieve
        | refine_prompt
        | llm
        | response_parser
    )
    
    return refine_chain.invoke(query)

def process_query(query: str, use_refinement: bool = True):
    """Process the user query and return the response."""
    debug_logger.debug(f"Refinement is {'ON' if use_refinement else 'OFF'}")
    try:
        # Get initial answer
        answer = query_data(query, st.session_state.documents, store_type=vector_store)
        debug_logger.debug(f"Initial answer: {answer}")
        
        # Refine the answer if refinement is enabled
        if use_refinement:
            final_answer = refine_answer(answer, query, st.session_state.documents, store_type=vector_store)
            debug_logger.debug(f"Refined answer: {final_answer}")
        else:
            final_answer = answer
            
        return final_answer
    except Exception as e:
        st.error(f"An error occurred while processing your query: {str(e)}")
        return None
    
def initialize_session_state():
    """Initialize session state variables for the Streamlit app."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = get_documents_from_store(vector_store)

def main():
    # Set up the Streamlit page
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="centered",
    )
    
    # Add title and description
    st.title("🤖 RAG Chat Assistant")
    st.caption("""
    This chat assistant uses RAG (Retrieval Augmented Generation) to provide answers based on your documents.
    It combines vector similarity search with BM25 retrieval and uses multi-query expansion for better results.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Add sidebar controls
    with st.sidebar:
        st.title("RAG Chat Assistant")
        st.header("Settings")
        use_refinement = st.toggle("Enable Answer Refinement", value=True)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(f"""
        - Using local LLM: `{local_llm_model}`
        - Embeddings: `{local_ollama_embed_model}`
        - Reranker: `bge-reranker-v2-m3`
        - Vector Store: `{vector_store}`
        - Features: Hybrid Search, Multi-query Expansion, Answer Refinement
        """)
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input and generate response
    if query := st.chat_input("Ask a question about your documents..."):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Generate and display assistant response
        with st.spinner("Thinking..."):
            response = process_query(query, use_refinement)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()