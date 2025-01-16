from langchain_groq import ChatGroq
from typing import TypedDict, List
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Langgraph Chatbot", page_icon="")

st.title("Langgraph chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.2-11b-vision-preview")

class State(TypedDict):
    messages: Annotated[List, add_messages]
    
graph = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm.invoke(state['messages'])]}

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

app = graph.compile()


def stream_graph_updates(user_input: str):
    initial_state = {"messages": [("user", user_input)]}
    responses = []
    for events in app.stream(initial_state):
        for value in events.values():
            responses.append(value['messages'][-1].content)
                
    return responses


if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    

def chatbot_sidebar():   
    user_input = st.sidebar.text_input("Your message", key="input", placeholder="Type your message here...")
    
    if st.sidebar.button("Submit"):
        if user_input:
            submit_message(user_input)
    
def submit_message(user_input):
    if user_input:
        st.session_state['messages'].append(f"**You:** {user_input}")
        
    responses = stream_graph_updates(user_input)
    
    for response in responses:
        st.session_state['messages'].append(f"**Assistant:** {response}")
        
        
def display_chat():
    st.write("### Conversation:")
    for message in st.session_state['messages']:
        st.write(message)
        
        
chatbot_sidebar()
display_chat()