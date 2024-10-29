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
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()

# Set up environment variables for Hugging Face for offline mode
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = os.getenv("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = os.getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ["HF_HOME"] = os.getenv("HF_HOME")
os.environ["DATASETS_CACHE"] = os.getenv("DATASETS_CACHE")

# Set the local LLM model, embeddings model, re-ranker model and vector store 
local_llm_model = "llama3.2:3b"
local_ollama_embed_model = "bge-m3"
local_rerank_model = f"{os.getenv('HF_HOME')}/hub/models--BAAI--bge-reranker-v2-m3"
vector_store = "chroma"
refine_is_on = True

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

def run_test_queries(store_type, use_refinement):
    """Run test queries with hybrid search."""
    print(f"\nRunning test queries using {store_type.upper()} vector store with hybrid search. Refinement is {'ON' if use_refinement else 'OFF'}.")
    
    # Get documents from the vector store
    documents = get_documents_from_store(store_type)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        answer = query_data(query, documents, store_type=store_type)
        
        if use_refinement:
            refined_answer = refine_answer(answer, query, documents, store_type=store_type)
            print(f"\nRefined Answer: {refined_answer}\n")
        else:
            print(f"\nAnswer: {answer}\n")
        
        print("-" * 80)

def main():
    if vector_store == "faiss":
        run_test_queries("faiss", use_refinement=refine_is_on)
    elif vector_store == "chroma":
        run_test_queries("chroma", use_refinement=refine_is_on)

# Your existing test_queries list here
test_queries = [
    "What is the name of Microsoft's AI companion announced in the report?",
    "What is Microsoft's total revenue for fiscal year 2023?",
    "How many organizations are using AI-powered capabilities in Power Platform?",
    "How many monthly active users does Microsoft Teams have?",
    "What is the annual recurring revenue of GitHub?",
    "How many organizations are using Azure OpenAI Service?",
    "What is the annual revenue of Microsoft's security business?",
    "How many LinkedIn members are there according to the report?",
    "What is the annual revenue of LinkedIn?",
    "How many organizations count on Microsoft's AI-powered security solutions?",
    "What is the name of Microsoft's employee experience platform?",
    "How many monthly active users does Microsoft Viva have?",
    "What is the annual revenue of Microsoft's Dynamics business?",
    "How many organizations have chosen GitHub Copilot for Business?",
    "What is Microsoft's goal for training and certifying people with digital economy skills by 2025?",
    "How many people has Microsoft helped with digital skills since July 2020?",
    "What is the value of discounted and donated technology Microsoft provided to nonprofits?",
    "How many nonprofits used Microsoft's cloud services?",
    "What is the name of Microsoft's AI-powered clinical documentation application?",
    "How many players did Starfield attract in its first month post-launch?",
]

if __name__ == "__main__":
    main()