from langchain_chroma import Chroma                                                                                                   # type: ignore
from langchain_core.prompts import ChatPromptTemplate                                                                                                   # type: ignore
from langchain_core.pydantic_v1 import BaseModel, Field                                                                                                   # type: ignore
from langchain_ollama import ChatOllama                                                                                                   # type: ignore
import EmbeddingCreation 

# Section: Similarity Threshold Retriever Setup
chroma_db = embedding_creation.create_embeddings()
similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 3, "score_threshold": 0.3})

# Section: Document Grading Model Setup
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

llm = ChatOllama(model="phi-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not.
             """
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}

                     User question:
                     {question}
                  """),
    ]
)

doc_grader = (grade_prompt | structured_llm_grader)

def grade_docs(query):
    top3_docs = similarity_threshold_retriever.invoke(query)
    graded_docs = []
    for doc in top3_docs:
        grade = doc_grader.invoke({"question": query, "document": doc.page_content})
        if grade.binary_score == "yes":
            graded_docs.append(doc)
    return graded_docs
