import streamlit as st
from typing import TypedDict
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import urllib
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    question: Annotated[str, "User input question"]
    query: Annotated[str, "Generated SQL query"]
    result: Annotated[str, "Query result"]
    answer: Annotated[str, "Final answer from LLM"]

llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name="llama3-70b-8192"
)

raw_password = os.getenv('DB_PASSWORD')
safe_password = urllib.parse.quote_plus(raw_password)
SUPABASE_DB_URL = (
    f"postgresql://postgres.znxqptwscjjbduccopap:{safe_password}"
    "@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
)
db = SQLDatabase.from_uri(SUPABASE_DB_URL)

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results.

Never query for all the columns from a specific table, only ask for the
few relevant columns given the question.

Only use the following tables:
{table_info}
"""

query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "Question: {input}")
])

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: dict) -> dict:
    question = state["question"]
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": question,
    })

    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    state["query"] = result["query"]
    return state

def execute_query(state: State) -> State:
    result = QuerySQLDatabaseTool(db=db).invoke(state["query"])
    state["result"] = result
    return state

def generate_answer(state: State) -> State:
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question clearly.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    state["answer"] = response.content.strip()
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()

st.set_page_config(page_title="SQL Chatbot", page_icon="🤖")
st.title("📊 SQL Chatbot with Supabase")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Ask your question about the placement data:")

if st.button("Submit") and question:
    state = {"question": question}
    for step in graph.stream(state, stream_mode="updates"):
        for node_name, node_state in step.items():
            if node_name == "write_query":
                st.session_state.history.append(("Generated Query", node_state["query"]))
            if node_name == "execute_query":
                st.session_state.history.append(("SQL Result", node_state["result"]))
            if node_name == "generate_answer":
                st.session_state.history.append(("Answer", node_state["answer"]))

for label, content in st.session_state.history:
    st.markdown(f"**{label}:**")
    st.code(content if isinstance(content, str) else str(content))
