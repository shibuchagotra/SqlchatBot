 
# 🧠 SQL Assistant with LangGraph, Streamlit & Supabase

This is a conversational app that converts natural language questions into SQL queries, executes them on a Supabase PostgreSQL database (with Row-Level Security), and returns human-readable answers using Groq’s LLaMA3-70B.

## 🔧 Tech Stack
- **LangGraph** for flow orchestration
- **LangChain SQLToolkit** to run safe read-only SQL queries
- **Groq (LLaMA3-70B)** for query + answer generation
- **Supabase** PostgreSQL DB with RLS enabled
- **Streamlit** frontend for chatbot UI

## ▶️ How It Works
1. User types a question in natural language (e.g., “How many CSE students were placed in 2024?”)
2. LLaMA3 generates the corresponding SQL query.
3. The query is executed on Supabase via the transaction pooler (port 6543).
4. LLaMA3 explains the result clearly.
5. User can ask another question, continuing the loop.

## 🧪 Environment Variables
Create a `.env` file with:
