import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# Page
st.set_page_config(page_title="AI Study Agent")

st.title("📚 AI Study Agent")
st.write("Ask any study question and get an answer.")


# LLM Model
model = ChatOllama(
    model="qwen3:1.7b",
    think=False
)


# System Message
system_message = SystemMessage(
    content="""
You are an AI Study Agent.

Your job is to help students learn.

You can answer questions about:
- Computer Networking
- Python
- Programming
- Operating Systems
- Database
- AI and Machine Learning
- General Computer Science

Rules:
1. Give correct and simple answers.
2. Explain difficult topics step by step.
3. Give examples when useful.
4. Use simple language.
5. If the user asks a calculation, solve it clearly.
"""
)


# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# User Input
question = st.chat_input("Ask your study question...")


if question:

    # Show user question
    with st.chat_message("user"):
        st.write(question)

    # Save user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Create Human Message
    human_message = HumanMessage(
        content=question
    )

    try:

        # Call LLM
        response = model.invoke([
            system_message,
            human_message
        ])

        answer = response.content

        # Show AI answer
        with st.chat_message("assistant"):
            st.write(answer)

        # Save AI answer
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:

        st.error(f"Error: {e}")