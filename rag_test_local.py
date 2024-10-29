import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.multi_query import MultiQueryRetriever
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
local_ollama_embed_model = "bge-m3" # first attempt with all-minilm:l6-v2 with ~60% accuracy, second try-out with znbang/bge:large-en-v1.5-f16 with ~75% accuracy
local_rerank_model = f"{os.getenv('HF_HOME')}/hub/models--BAAI--bge-reranker-v2-m3"
vector_store = "chroma" # Choose between "faiss" or "chroma"
# Set the refine flag to enable or disable answer refinement
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

def query_data(query, store_type=vector_store):
    """Query the data using the RAG model with contextual compression."""
    # Load the vector store
    vector_store = load_vector_store(store_type)

    # Instantiate the language model for the local LLM
    llm = ChatOllama(
        model=local_llm_model,
        temperature=0
    )

    # Define the multi-query template
    multi_query_template = """
    Based on the following question, generate two specific, concise variations of the question that aim to retrieve the most relevant information from the provided context only. The new queries should focus strictly on asking the details mentioned within the context only and avoid any references to external sources such as news articles or reports.
    The new queries should be phrased in a way that they answer the original question directly based on the context provided.

    Original Question: {question}
    """

    query_generation_prompt = PromptTemplate.from_template(multi_query_template)

    # Create the multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8 # Experiment with different values for k to retrieve more or fewer documents
            },
        ), 
        llm=llm, 
        include_original=True, 
        prompt=query_generation_prompt, 
    )
    
    # Re-rank the retrieved documents using a cross-encoder model
    model = HuggingFaceCrossEncoder(model_name=local_rerank_model)
    compressor = CrossEncoderReranker(model=model, top_n=15)

    # Create the Contextual Compression Retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    # Create a function to retrieve and re-rank the documents
    def retrieve_and_rerank(query):
        compressed_docs = compression_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in compressed_docs)
        return {"context": context, "question": query}

    # Define the prompt template for the RAG model
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

    answer = rag_chain.invoke(query)
    
    return answer

def refine_answer(answer, query, store_type=vector_store):
    """ Refine the answer based on the original context and question. """
    vector_store = load_vector_store(store_type)
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6, # Experiment with different values for k to retrieve more or fewer documents
        },
    )

    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
        "question": RunnablePassthrough(),
        "answer": lambda _: answer  # Pass the original answer
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
    
    llm = ChatOllama(
        model=local_llm_model,
        temperature=0,
    )

    response_parser = StrOutputParser()

    refine_chain = (
        retrieve
        | refine_prompt
        | llm
        | response_parser
    )
    
    refined_answer = refine_chain.invoke(query)
    return refined_answer

# List of test queries
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

def run_test_queries(store_type, use_refinement):
    print(f"\nRunning test queries using {store_type.upper()} vector store. Refinement is {'ON' if use_refinement else 'OFF'}.")
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        answer = query_data(query, store_type=store_type)
        
        if use_refinement:
            refined_answer = refine_answer(answer, query, store_type=store_type)
            print(f"\nRefined Answer: {refined_answer}\n")
        else:
            print(f"\nAnswer: {answer}\n")
        
        print("-" * 80)

def main():
    if vector_store == "faiss":
        # Run tests with FAISS
        run_test_queries("faiss", use_refinement=refine_is_on)
    elif vector_store == "chroma":
        # Run tests with Chroma
        run_test_queries("chroma", use_refinement=refine_is_on)

if __name__ == "__main__":
    main()


# Note: The retrieval and re-ranking process can be customized further by modifying the query_data function.

# To add a prefilter, you can modify the retriever in the query_data function:
# def query_data(query, store_type="faiss"):
#     vector_store = load_vector_store(store_type)
#     retriever = vector_store.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 3,
#             "filter": lambda doc: not doc.metadata.get("hasCode", False),
#             "score_threshold": 0.01
#         },
#     )
#     ...