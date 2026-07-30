import re
from typing import List, TypedDict

# from langchain_classic.retrievers import TavilySearchAPIRetriever
from langchain_community.tools import TavilySearchResults

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from app.core import settings

# -----------------------------------
# 1. Document Loading & Vector Store
# -----------------------------------
docs = (
    PyPDFLoader(r"app/core/constants/Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf").load()
    + PyPDFLoader(r"app/core/constants/01_Intro_to_DL.pdf").load()
)

chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu",
        "local_files_only": True  
    }
)

vector_store = FAISS.from_documents(chunks, embeddings)

# Top-k ko 4 se badha kar 6 kiya hai taaki context miss hone ka chance kam ho
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# -----------------------------------
# 2. LLM Setup
# -----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=settings.GROQ_API_KEY
)

UPPER_TH = 0.6  # Slightly lowered for practical retrieval
LOWER_TH = 0.25

class State(TypedDict):
    question: str
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_docs : List[Document]
    answer: str

def retrieved_node(state: State) -> State:
    q = state["question"]
    retrieved_docs = retriever.invoke(q)
    print(f"\n--- [DEBUG] Retrieved {len(retrieved_docs)} Chunks ---")
    for i, doc in enumerate(retrieved_docs[:2]):
        print(f"Chunk {i+1} Sample: {doc.page_content[:150]}...\n")
    return {"docs": retrieved_docs}


# -----------------------------------
# 3. Evaluator Node
# -----------------------------------
class BatchDocEval(BaseModel):
    scores: List[float]
    reasons: List[str]

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a retrieval evaluator for RAG.\n"
            "Evaluate if the chunk contains information helpful to answer the question.\n"
            "Return score between 0.0 and 1.0:\n"
            "- 0.8 - 1.0: Contains direct and sufficient answer\n"
            "- 0.3 - 0.7: Contains relevant concepts or partial info\n"
            "- 0.0 - 0.2: Completely irrelevant\n"
            "Output JSON only.",
        ),
        (
    "human",
    "Question:\n{question}\n\nRetrieved Chunks:\n{chunks}"
        ),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(BatchDocEval)

def eval_each_doc_node(state: State) -> State:
    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []

    for d in state["docs"]:
        try:
            out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
            scores.append(out.scores)
            if out.scores > LOWER_TH:
                good.append(d)
        except Exception as e:
            scores.append(0.0)

    if any(s >= UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"Retrieved context contains sufficient information (Score > {UPPER_TH}).",
        }

    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "Retrieved chunks do not contain relevant information for this question in the PDFs."
        }

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": "Partial information found, but not strongly decisive.",
    }


# -----------------------------------
# 4. Decomposer & Batched Filter Node
# -----------------------------------
def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]

class FilteredSentences(BaseModel):
    relevant_sentences: List[str]

batch_filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract ONLY the sentences from the candidate list that are relevant "
            "to answering the user's question. Ignore completely off-topic sentences.",
        ),
        (
            "human",
            "Question: {question}\n\nSentences:\n{sentences}",
        ),
    ]
)

batch_filter_chain = batch_filter_prompt | llm.with_structured_output(FilteredSentences)

def refine(state: State) -> State:
    q = state["question"]
    if state.get("verdict") == "CORRECT":
        context = "\n\n".join(d.page_content for d in state["good_docs"]).strip()
    else:
        context = "\n\n".join(d.page_content for d in state["web_docs"]).strip()

    strips = decompose_to_sentences(context)
    
    if not strips:
        return {"strips": [], "kept_strips": [], "refined_context": context}

    # Batch filtering in 1 API call instead of N calls
    try:
        res = batch_filter_chain.invoke({"question": q, "sentences": "\n".join(strips)})
        kept = res.relevant_sentences
    except Exception:
        kept = strips  # Fallback to all if parsing fails

    refined_context = "\n".join(kept).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context if refined_context else context,
    }


# -----------------------------------
# Websearch 
# -----------------------------------
tavily = TavilySearchResults(
    max_results=5,
    tavily_api_key=settings.TAVILY_API_KEY
)

def web_search_node(state: State) -> State:
    q= state["question"]
    result = tavily.invoke({"query":q})

    web_docs = []
    for r in result or []:

        title = r.get("title", "")
        url= r.get("url", "")
        content= r.get("content", "") or r.get("snippet", "")

        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(Document(page_content=text , metadata = {"url": url , "title": title}))

    return {"web_docs": web_docs}


# -----------------------------------
# 5. Generation Node
# -----------------------------------
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question directly using ONLY the provided context.\n"
            "If the context doesn't contain the answer, explicitly state that the documents do not cover this topic.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{refined_context}",
        ),
    ]
)

def generate(state: State) -> State:
    out = (answer_prompt | llm).invoke(
        {
            "question": state["question"],
            "refined_context": state["refined_context"],
        }
    )
    return {"answer": out.content}


def ambiguous_node(state: State) -> State:
    return {"answer": f"Ambiguous: {state['reason']}"}

def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"
    elif state["verdict"] == "INCORRECT":
        return "web_search"
    else:
        return "ambiguous" # Try refine even if ambiguous



# -----------------------------------
# 6. Graph Construction
# -----------------------------------
g = StateGraph(State)
g.add_node("retrieve", retrieved_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("ambiguous", ambiguous_node)
g.add_node("web_search", web_search_node)

g.add_node("refine", refine)
g.add_node("generate", generate)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
     "web_search": "web_search",
     "ambiguous": "ambiguous",
    },
)

g.add_edge("web_search","refine")
g.add_edge("refine", "generate")
g.add_edge("generate", END)
g.add_edge("ambiguous", END)    

app = g.compile()

# Test Run
res = app.invoke(
    {
        "question": "What is current AI news ?",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "answer": "",
    }
)

print("\n=== FINAL RESULT ===")
print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("\nOUTPUT:\n", res["answer"])