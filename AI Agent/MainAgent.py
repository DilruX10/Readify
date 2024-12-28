from langchain_core.prompts import ChatPromptTemplate, prompts                                                                                                  # type: ignore
from langchain_ollama import ChatOllama                                                                                                                                                      # type: ignore
from langchain_core.runnables import RunnableLambda                                                                                                                                                  # type: ignore
from langchain_core.output_parsers import StrOutputParser                                                                                                                                                                         # type: ignore
from operator import itemgetter
from langgraph.graph import END, StateGraph                                                                                                                                                   # type: ignore
from langchain.schema import Document                                                                                                                                                             # type: ignore
from langchain_community.tools.duckduckgo_search import DuckDuckGoResults                                                                                                                     # type: ignore
from typing import List
from typing_extensions import TypedDict                                                                                                                                                              # type: ignore
import CorrectiveRag

# Section: Question Answering Chain Setup
prompt = """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If no context is present or if you don't know the answer, just say that you don't know the answer.
                Do not make up the answer unless it is there in the provided context.
                Give a detailed answer and to the point answer with regard to the question.

                Question:
                {question}

                Context:
                {context}

                Answer:
            """
prompt_template = ChatPromptTemplate.from_template(prompt)

chatgpt = ChatOllama(model_name='phi3-mini', temperature=0)

def format_docs(docs):
     return "\n\n".join(doc.page_content for doc in docs)

qa_rag_chain = (
     {
          "context": (itemgetter('context') | RunnableLambda(format_docs)),
          "question": itemgetter('question')
     }
     | prompt_template
     | chatgpt
     | StrOutputParser()
)

# State Graph Functions
def retrieve(state):
     question = state["question"]
     documents = corrective_rag.similarity_threshold_retriever.invoke(question)
     return {"documents": documents, "question": question}


def grade_documents(state):
     question = state["question"]
     documents = state["documents"]
     filtered_docs = []
     web_search_needed = "No"
     if documents:
          for d in documents:
                score = corrective_rag.doc_grader.invoke({"question": question, "document": d.page_content})
                grade = score.binary_score
                if grade == "yes":
                     filtered_docs.append(d)
                else:
                     web_search_needed = "Yes"
                     continue
     else:
          web_search_needed = "Yes"
     return {"documents": filtered_docs, "question": question, "web_search_needed": web_search_needed}


def decide_to_generate(state):
     web_search_needed = state["web_search_needed"]
     if web_search_needed == "Yes":
          return "rewrite_query"
     else:
          return "generate_answer"


def rewrite_query(state):
     question = state["question"]
     documents = state["documents"]
     better_question = question_rewriter.invoke({"question": question})  # type: ignore
     return {"documents": documents, "question": better_question}


def web_search(state):
     question = state["question"]
     documents = state["documents"]
     docs = DuckDuckGoResults.invoke(question)
     web_results = "\n\n".join([d["content"] for d in docs])
     web_results = Document(page_content=web_results)
     documents.append(web_results)
     return {"documents": documents, "question": question}


def generate_paragraph(state):
     topic = state["topic"]
     question = f"Generate a paragraph about {topic}"
     documents = corrective_rag.similarity_threshold_retriever.invoke(question)
     formatted_docs = format_docs(documents)
     paragraph = qa_rag_chain.invoke({"context": formatted_docs, "question": question})
     return {"paragraph": paragraph, "topic": topic}


def generate_question_01(state):
     paragraph = state["paragraph"]
     prompt = prompts.get_question_template_01(paragraph)
     question = corrective_rag.chatgpt.invoke(prompt)
     return {"question_01": question, "paragraph": paragraph, "topic": state["topic"]}


def generate_question_02(state):
     paragraph = state["paragraph"]
     prompt = prompts.get_question_template_02(paragraph)
     question = corrective_rag.chatgpt.invoke(prompt)
     return {"question_02": question, "paragraph": paragraph, "topic": state["topic"]}


def generate_question_03(state):
     paragraph = state["paragraph"]
     prompt = prompts.get_question_template_03(paragraph)
     question = corrective_rag.chatgpt.invoke(prompt)
     return {"question_03": question, "paragraph": paragraph, "topic": state["topic"]}


def generate_question_04(state):
     paragraph = state["paragraph"]
     prompt = prompts.get_question_template_04(paragraph)
     question = corrective_rag.chatgpt.invoke(prompt)
     return {"question_04": question, "paragraph": paragraph, "topic": state["topic"]}


def decide_question_type(state):
     question_type = state["question_type"]
     return f"generate_question_{question_type}"


# Function to validate user's answers
def validate_answers(state):
     paragraph = state["paragraph"]
     questions = {
          "01": state["question_01"],
          "02": state["question_02"],
          "03": state["question_03"],
          "04": state["question_04"]
     }
     user_answers = state["user_answers"]
     validation_results = []
     for key, question in questions.items():
          query = f"Context: {paragraph}\nQuestion: {question}\nUser Answer: {user_answers.get(key, '')}"
          context_documents = corrective_rag.similarity_threshold_retriever.invoke(query)
          context_data = format_docs(context_documents)
          feedback_prompt = prompts.get_feedback_template_advanced(paragraph, question, user_answers.get(key, ""))
          validation = qa_rag_chain.invoke({"context": context_data, "question": feedback_prompt})
          validation_results.append({"question_type": key, "validation": validation})
     return {"validation_results": validation_results, "paragraph": paragraph, "topic": state["topic"]}



# Workflow Manager
def manage_workflow(state):
     workflow = state["workflow"]
     if workflow == "generate_paragraph":
          return "generate_paragraph"
     elif workflow == "generate_question":
          return "retrieve_paragraph"
     elif workflow == "generate_feedback":
          return "retrieve_data"


# Section: State Graph Initialization
class GraphState(TypedDict):
     workflow: str
     topic: str
     paragraph: str
     question_01: str
     question_02: str
     question_03: str
     question_04: str
     user_answers: dict
     validation_results: List[dict]

agentic_rag = StateGraph(GraphState)

agentic_rag.add_node("retrieve", retrieve)
agentic_rag.add_node("grade_documents", grade_documents)
agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("web_search", web_search)

agentic_rag.add_node("generate_paragraph", generate_paragraph)
agentic_rag.add_node("generate_question_01", generate_question_01)
agentic_rag.add_node("generate_question_02", generate_question_02)
agentic_rag.add_node("generate_question_03", generate_question_03)
agentic_rag.add_node("generate_question_04", generate_question_04)
agentic_rag.add_node("decide_question_type", decide_question_type)
agentic_rag.add_node("validate_answers", validate_answers)
agentic_rag.add_node("manage_workflow", manage_workflow)


# Workflow to generate paragraph
agentic_rag.add_entry_point("generate_paragraph")
agentic_rag.add_edge("generate_paragraph", "retrieve")
agentic_rag.add_edge("retrieve", "grade_documents")
agentic_rag.add_conditional_edges(
     "grade_documents",
     decide_to_generate,
     {"rewrite_query": "rewrite_query", "generate_paragraph": "generate_paragraph"},
)

agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_paragraph")
agentic_rag.add_edge("web_search", END)


# Workflow to generate question
agentic_rag.add_entry_point("retrieve_paragraph")
agentic_rag.add_conditional_edges(
     "retrieve_paragraph",
     decide_question_type,
     {
          "01": "generate_question_01",
          "02": "generate_question_02",
          "03": "generate_question_03",
          "04": "generate_question_04"
     }
)

agentic_rag.add_edge("generate_question_01", END)
agentic_rag.add_edge("generate_question_02", END)
agentic_rag.add_edge("generate_question_03", END)
agentic_rag.add_edge("generate_question_04", END)


# Workflow to generate feedback
agentic_rag.add_entry_point("retrieve_data")
agentic_rag.add_edge("retrieve_data", "grade_documents")
agentic_rag.add_conditional_edges(
     "grade_documents",
     decide_to_generate,
     {"rewrite_query": "rewrite_query", "generate_feedback": "generate_feedback"},
)
agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_feedback")
agentic_rag.add_edge("generate_feedback", END)

agentic_rag = agentic_rag.compile()


# Section: State Graph Execution
def execute_workflow(state):
     initial_node = agentic_rag.get_node(state["workflow"])
     response = agentic_rag.invoke(state, start_at=initial_node)
     return response


if __name__ == '__main__':

     # Generate Paragraph
     topic = "India"
     initial_state_paragraph = {"workflow": "generate_paragraph", "topic": topic, "user_answers": {}}
     paragraph_result = execute_workflow(initial_state_paragraph)
     print(paragraph_result)
     
     
     # Generate Question
     initial_state_question = {"workflow": "retrieve_paragraph", "topic": topic, "question_type": "01", "paragraph": paragraph_result["paragraph"]}
     question_result = execute_workflow(initial_state_question)
     print(question_result)
     
     
     # Generate Feedback
     initial_state_feedback = {"workflow": "retrieve_data", "question": "What is the capital of India?", "paragraph": paragraph_result["paragraph"], "user_answers": {"01": "New Delhi"}}
     feedback_result = execute_workflow(initial_state_feedback)
     print(feedback_result)
