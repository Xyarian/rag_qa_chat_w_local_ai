
# Local AI Question-Answering (Q&A) with Retrieval-Augmented Generation (RAG) System

## Abstract

This experimental project demonstrates an enhanced **Retrieval-Augmented Generation (RAG)** system implementing advanced retrieval techniques including *hybrid search, multi-query retrieval, and cross-encoder reranking*. Built on the [LangChain](https://python.langchain.com/docs/tutorials/) framework, the system utilizes **local LLM models** via [Ollama](https://ollama.com/), [Chroma SDB](https://docs.trychroma.com/getting-started) for vector storage, and [Hugging Face](https://huggingface.co/welcome)'s cross-encoder models for reranking, providing a question-answering solution. A [Streamlit](https://streamlit.io/)-based user interface was developed to enable **interactive chat-like question-answering**.

## System Architecture

The system consists of four main components:
1. Data ingestion and indexing (`rag_load_data_local.py`)
2. Core RAG implementation (`rag_test_local.py`)
3. Enhanced retrieval system (`rag_test_local_v2.py`)
4. Streamlit UI for interactive chat (`chat.py`)

### 1. Data Ingestion Pipeline

The ingestion process (`rag_load_data_local.py`) includes:
- Document loading using `PyPDFDirectoryLoader`
- Text chunking via `RecursiveCharacterTextSplitter`
- Metadata enrichment using local LLM (llama3.2:3b)
- Embedding generation with `OllamaEmbeddings` (bge-m3)
- Vector storage in Chroma DB (or optionally FAISS)

### 2. Core RAG Implementation

The base system (`rag_test_local.py`) implements:
- Multi-query retrieval
- Cross-encoder reranking
- Contextual compression
- Answer generation and refinement

### 3. Enhanced Retrieval System

The enhanced system (`rag_test_local_v2.py`) introduces hybrid search combining dense and sparse retrievers:

```python
def create_hybrid_retriever(vector_store, documents, k=8):
    """Create a hybrid retriever combining vector similarity and BM25."""
    vectorstore_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, bm25_retriever],
        weights=[0.5, 0.5] # Weights for both retrievers (sum should be 1) to be experimented with
    )
    
    return ensemble_retriever
```

## Advanced Retrieval Techniques

### Multi-Query Retrieval

The system employs LLM-based query expansion to generate multiple search queries:

```python
multi_query_template = """
Based on the following question, generate two specific, concise variations 
of the question that aim to retrieve the most relevant information from 
the provided context only.

Original Question: {question}
"""

retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True,
    prompt=query_generation_prompt
)

query_generation_prompt = PromptTemplate.from_template(multi_query_template)
```

### Cross-Encoder Reranking

Document reranking is performed using BAAI's bge-reranker-v2-m3:

```python
reranker_model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3"
)
compressor = CrossEncoderReranker(
    model=reranker_model, 
    top_n=15
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 4. Streamlit UI for Interactive Chat

To complement the advanced retrieval and response generation capabilities of the RAG system, a Streamlit-based user interface was developed in the `chat.py` file. This interface provides an interactive, chat-like experience for users to ask questions and receive context-relevant answers.

The Streamlit UI offers the following key features:

- **Interactive Chat Interface:** Users can engage in a conversational-style interaction, typing questions and receiving real-time responses from the RAG system.
- **Answer Refinement Control:** A sidebar toggle allows users to enable or disable the answer refinement functionality, enabling them to explore the impact of this advanced technique on the quality of responses.
- **Chat History Clearing:** Users can clear the chat history using a dedicated button in the sidebar, enabling a fresh start for their exploration of the system.

By integrating the powerful RAG system with a user-friendly and intuitive Streamlit interface, project provides a solution for interactive, local AI-powered question-answering. The seamless combination of advanced retrieval techniques and an accessible user interface empowers end-users to leverage the capabilities of the RAG system in a practical manner.

To run the Streamlit app, execute:

```
streamlit run chat.py
```

## Technical Implementation Details

### Framework Integration

The system leverages several key frameworks:
- **LangChain**: Core framework for RAG pipeline implementation
- **Ollama**: Local LLM integration (llama3.2:3b)
- **Chroma DB**: Vector storage solutions
- **Hugging Face**: Cross-encoder models for reranking
- **Streamlit**: Interactive chat User-Interface

### Advanced Features

1. **Hybrid Search**
   - Combines dense (vector) and sparse (BM25) retrieval
   - Weighted ensemble for balanced results
   - Configurable k-parameter for result count

2. **Answer Refinement**
   - Two-stage answer generation
   - Context-aware refinement
   - Accuracy verification

```python
def refine_answer(answer, query, documents, store_type=vector_store):
    retriever = create_hybrid_retriever(vector_store, documents, k=6)
    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
        "answer": lambda _: answer
    }
    # ... refinement chain implementation
```

## Experimental Configuration

The system supports various configurations:
- Vector store selection (Chroma/FAISS)
- Retrieval parameters (k-values)
- Reranking parameters (top_n)
- Answer refinement toggle
- Model selection:
  - LLM: llama3.2:3b
  - Embeddings: bge-m3
  - Reranker: bge-reranker-v2-m3

## Research Applications

This system demonstrates several research-worthy aspects:

- Impact of hybrid retrieval on answer accuracy
- Query expansion effectiveness
- Reranking influence on context relevance
- Local LLM performance in RAG systems

## Further Research Directions

### 1. Data Ingestion 
- Table Detection and Conversion: Detect tables within documents and convert them into Markdown format for LLM processing. Use dynamic chunking methods to keep table data in one chunk, while applying standard chunking to textual content
- Support for OCR and image-processing and other advanced document layout analysis
- Multimodal LLMs: Question-answering over PDFs with complex layouts, diagrams, or scans it may be advantageous to skip the PDF parsing, instead casting a PDF page to an image and passing it to a model

### 2. Retrieval and Generation

- Dynamic weight adjustment for hybrid retrieval
- Automated parameter optimization
- Context length optimization
- Alternative reranking strategies
- Query expansion techniques evaluation

## Conclusion

This implementation showcases an advanced RAG system combining multiple retrieval strategies and introducing a Streamlit-based user interface for interactive question-answering. The hybrid retrieval approach, coupled with multi-query expansion and cross-encoder reranking, provides a foundation for context-relevant responses in local AI-powered question-answering systems. The Streamlit UI enhances the usability and accessibility of the system, enabling users to explore and leverage the capabilities of the enhanced RAG system. This project contributes to the development of local AI-powered question-answering solutions, paving the way for more efficient knowledge extraction.

https://github.com/Xyarian/rag_qa_chat_w_local_ai