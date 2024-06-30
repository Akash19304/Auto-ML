# from langchain.agents import create_csv_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv
import os
import streamlit as st

def ask_csv(dataset):
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model='llama3-8b-8192',
        temperature=0,
        api_key=groq_api_key
    )
    if dataset: 

        agent = create_csv_agent(
           llm, dataset, verbose=True, allow_dangerous_code=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))




