# Research Paper Question Answering using RAG

##  About the Project

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about research papers in natural language. The application retrieves the most relevant information from the uploaded research papers using FAISS vector search and generates accurate responses using a Groq LLM.

##  Features

- Ask questions from research papers.
- Retrieval-Augmented Generation (RAG) pipeline.
- FAISS vector database for semantic search.
- HuggingFace embeddings for document representation.
- Structured responses with Introduction, Definition, and Explanation.
- Interactive Streamlit interface.

##  Tech Stack

- Python
- LangChain
- Groq (Gemma-7B)
- FAISS
- HuggingFace Embeddings
- Streamlit
- Pydantic

##  How to Run

1. Clone the repository.
2. Install the required dependencies.
3. Add your `GROQ_API_KEY` and `HF_TOKEN` to a `.env` file.
4. Place your research papers inside the `research_papers` folder.
5. Run the application:

```bash
streamlit run app.py
```

##  Learning Outcomes

- Implemented a Retrieval-Augmented Generation (RAG) pipeline.
- Built a semantic search system using FAISS.
- Generated context-aware answers using LangChain and Groq.
- Applied document chunking and vector embeddings for efficient retrieval.
