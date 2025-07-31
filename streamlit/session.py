import streamlit as st

def get_session_key(key):
    return f"session_{key}"

def initialize_session():
    if get_session_key("mode") not in st.session_state:
        st.session_state[get_session_key("mode")] = "child"
    if get_session_key("parent_authenticated") not in st.session_state:
        st.session_state[get_session_key("parent_authenticated")] = False

def reset_session():
    keys = [get_session_key(k) for k in ["mode", "parent_authenticated", "chat_history", "child_history", "today_question", "question_date", "ai_report"]]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
    initialize_session()