# WebRAG-Tavily

A Retrieval-Augmented Generation (RAG) system that combines local document knowledge with web search capabilities using LangChain, LangGraph, and Tavily API.

## Overview

WebRAG-Tavily is an intelligent question-answering system that:

1. **Retrieves** relevant information from local PDF documents using FAISS vector store
2. **Evaluates** retrieved document quality using LLM-based scoring
3. **Falls back** to web search via Tavily API when local documents are insufficient
4. **Refines** context through sentence-level filtering for optimal relevance
5. **Generates** accurate answers using Groq's Llama 3.3 70B model

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Retrieve    │────▶│  Eval Docs   │────▶│   Route     │
│  (FAISS)     │     │  (LLM)       │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
              ┌──────────┐             ┌──────────────┐             ┌──────────┐
              │ CORRECT   │             │  INCORRECT   │             │ AMBIGUOUS │
              └─────┬─────┘             └──────┬───────┘             └─────┬─────┘
                    │                          │                          │
                    ▼                          ▼                          ▼
              ┌──────────┐             ┌──────────────┐             ┌──────────┐
              │ Refine    │             │  Web Search  │             │ Generate  │
              └─────┬─────┘             └──────┬───────┘             └──────────┘
                    │                          │
                    └────────────┬─────────────┘
                                 ▼
                           ┌──────────┐
                           │ Generate  │
                           └──────────┘
```

## Features

- **Hybrid RAG Pipeline**: Combines local document retrieval with web search
- **Intelligent Document Evaluation**: LLM-based scoring to assess retrieved chunk relevance
- **Dynamic Routing**: Automatically falls back to web search when local documents are insufficient
- **Sentence-Level Refinement**: Filters and refines context for optimal answer generation
- **FAISS Vector Store**: Efficient similarity search for document retrieval
- **Groq Integration**: Fast inference using Llama 3.3 70B model
- **HuggingFace Embeddings**: Local embedding model for document vectorization

## Prerequisites

- Python 3.14+
- Groq API key
- Tavily API key
- OpenAI API key (optional, for additional features)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Webrag-Tavily-.git
cd Webrag-Tavily-
```

### 2. Create a virtual environment

```bash
# Using uv (recommended)
uv venv

# Or using python
python -m venv .venv
```

### 3. Activate the virtual environment

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure

```
Webrag-Tavily-/
├── app/
│   ├── core/
│   │   ├── config.py          # Settings management with Pydantic
│   │   └── tools/
│   │       └── constants/     # PDF documents for RAG
│   ├── agent/                 # Agent modules
│   └── webrag.py              # Main RAG pipeline implementation
├── main.py                    # Application entry point
├── pyproject.toml             # Project configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Basic Usage

```python
from app.webrag import app

# Run the RAG pipeline
result = app.invoke({
    "question": "What is machine learning?",
    "docs": [],
    "good_docs": [],
    "verdict": "",
    "reason": "",
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
    "answer": "",
})

print("Answer:", result["answer"])
print("Verdict:", result["verdict"])
```

### Running the Application

```bash
python main.py
```

## How It Works

### 1. Document Loading & Vectorization

- Loads PDF documents from `app/core/constants/`
- Splits documents into chunks (900 characters with 150 overlap)
- Creates FAISS vector store using HuggingFace embeddings (`all-MiniLM-L6-v2`)

### 2. Retrieval

- Retrieves top-6 similar document chunks using cosine similarity
- Returns chunks most relevant to the user's question

### 3. Evaluation

- LLM evaluates each retrieved chunk on a scale of 0.0 to 1.0
- Scores determine the retrieval quality:
  - **CORRECT** (Score > 0.6): Local documents contain sufficient information
  - **INCORRECT** (Score < 0.25): No relevant information in local documents
  - **AMBIGUOUS** (0.25 - 0.6): Partial information available

### 4. Routing

Based on evaluation verdict:
- **CORRECT**: Proceed to refine local context
- **INCORRECT**: Fall back to web search via Tavily API
- **AMBIGUOUS**: Generate answer with available context

### 5. Context Refinement

- Decomposes context into sentences
- Filters sentences for relevance using LLM
- Keeps only the most relevant information

### 6. Answer Generation

- Generates final answer using Groq's Llama 3.3 70B model
- Provides clear response based on refined context

## Configuration

### Model Settings

The system uses:
- **LLM**: `llama-3.3-70b-versatile` via Groq
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **Vector Store**: FAISS with similarity search

### Thresholds

- `UPPER_TH = 0.6`: Minimum score for "CORRECT" verdict
- `LOWER_TH = 0.25`: Minimum score to keep document chunk

## API Keys

| Service | Key | Purpose |
|---------|-----|---------|
| Groq | `GROQ_API_KEY` | LLM inference |
| Tavily | `TAVILY_API_KEY` | Web search |
| OpenAI | `OPENAI_API_KEY` | Additional features |

## Dependencies

- **LangChain**: Core RAG framework
- **LangGraph**: Graph-based workflow orchestration
- **FAISS**: Vector similarity search
- **HuggingFace**: Embedding models
- **Groq**: Fast LLM inference
- **Tavily**: Web search API
- **Pydantic**: Settings management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Groq](https://groq.com/) for fast LLM inference
- [Tavily](https://tavily.com/) for web search API
- [HuggingFace](https://huggingface.co/) for embedding models
- [FAISS](https://faiss.ai/) for vector similarity search
