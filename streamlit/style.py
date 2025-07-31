import streamlit as st

def render_css():
    st.markdown("""
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --bg-light: #fafbfc;
            --text-muted: #64748b;
            --positive: #10b981;
            --negative: #ef4444;
        }
        .stApp { background: var(--bg-light); }
        .main .block-container { max-width: 1200px; padding: 2rem 3rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, var(--primary) 0%, var(--secondary) 100%); padding-top: 2rem; }
        [data-testid="stSidebar"] * { color: white !important; }
        [data-testid="stSidebar"] .stRadio label { font-size: 1.1rem; font-weight: 500; padding: 0.75rem 1rem; border-radius: 10px; transition: all 0.3s ease; display: block; margin: 0.5rem 0; }
        [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.1); }
        [data-testid="stSidebar"] [data-baseweb="radio"] { background-color: rgba(255,255,255,0.2) !important; }
        .main-header { background: white; padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        .main-header h1 { color: var(--primary); margin: 0; font-size: 2.5rem; font-weight: 700; }
        .main-header p { color: var(--text-muted); margin: 0.5rem 0 0 0; font-size: 1.1rem; }
        .chat-wrapper { background: white; border-radius: 20px; padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 600px; display: flex; flex-direction: column; }
        .chat-messages { flex: 1; overflow-y: auto; padding-right: 1rem; margin-bottom: 1rem; }
        .metric-card { background: white; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%; }
        .metric-value { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
        .metric-label { color: var(--text-muted); font-size: 0.9rem; }
        .question-box { background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center; }
        .question-text { color: #4c1d95; font-size: 1.3rem; font-weight: 600; margin: 0; }
        .stButton > button { background: var(--primary); color: white; border: none; border-radius: 10px; padding: 0.6rem 1.5rem; font-weight: 600; transition: all 0.2s ease; width: 100%; }
        .stButton > button:hover { background: #4f46e5; transform: translateY(-1px); }
        .emotion-tag { display: inline-block; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin: 0.2rem; }
        .tag-positive { background: var(--positive); color: #065f46; }
        .tag-negative { background: var(--negative); color: #991b1b; }
        @media (max-width: 768px) {
            .main .block-container { padding: 1rem; }
            .metric-card { padding: 1rem; }
        }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        .auth-container { max-width: 400px; margin: 4rem auto; background: white; border-radius: 20px; padding: 3rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); text-align: center; }
        [data-testid="stTabs"] { display: none; }
        .chat-bubble { max-width: 70%; padding: 0.9em 1.2em; border-radius: 1.5em; margin-bottom: 0.5em; font-size: 1.08em; line-height: 1.5; word-break: break-word; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
        .bubble-left { background: #f3f4f6; color: #22223b; border-bottom-left-radius: 0.4em; margin-right: auto; margin-left: 0; text-align: left; display: flex; align-items: flex-end; gap: 0.5em; }
        .bubble-right { background: var(--primary); color: #fff; border-bottom-right-radius: 0.4em; margin-left: auto; margin-right: 0; text-align: right; display: flex; align-items: flex-end; flex-direction: row-reverse; gap: 0.5em; }
        .avatar { width: 2.2em; height: 2.2em; border-radius: 50%; background: #e0e7ff; display: flex; align-items: center; justify-content: center; font-size: 1.5em; margin-bottom: 0.1em; }
        button:focus { outline: 2px solid #fff; outline-offset: 2px; }
    </style>
    """, unsafe_allow_html=True)