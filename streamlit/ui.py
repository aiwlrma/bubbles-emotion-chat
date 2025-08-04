import streamlit as st
import random
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from datetime import date, datetime

# 환경 변수 로드
load_dotenv()
PARENT_CODE = os.getenv("PARENT_CODE", "1234")

from emotion import load_model_and_tokenizer, enhanced_emotion_classification, get_openai_client
from rag import load_rag_documents, generate_rag_based_report, create_pdf_report
from session import get_session_key
from i18n import t
from style import render_css

def get_today_question():
    today = date.today().isoformat()
    key = get_session_key("today_question")
    date_key = get_session_key("question_date")
    if key not in st.session_state or st.session_state.get(date_key) != today:
        questions = get_questions()
        question = random.choice(questions)
        st.session_state[key] = question
        st.session_state[date_key] = today
    return st.session_state[key]

@st.cache_data
def get_questions():
    return [
        t("questions.positive_moment"),
        t("questions.gratitude"),
        t("questions.new_learn"),
        t("questions.friend_talk"),
        t("questions.color_day"),
        t("questions.fun_moment"),
        t("questions.hard_moment"),
        t("questions.tomorrow_plan")
    ]

def render_app():
    st.set_page_config(page_title=t("app.title"), page_icon="💝", layout="wide")
    render_css()

    tokenizer, model = load_model_and_tokenizer()
    rag_folder = os.path.join(os.path.dirname(__file__), "rag")
    rag_docs = load_rag_documents(rag_folder)
    font_path = os.path.join(os.path.dirname(__file__), "NotoSansKR-Regular.ttf")

    render_sidebar()
    mode = st.session_state.get(get_session_key("mode"), "child")
    if mode == "child":
        render_child_mode(tokenizer, model)
    else:
        render_parent_mode(tokenizer, model, rag_docs, font_path)

def render_sidebar():
    with st.sidebar:
        st.markdown(t("sidebar.header"), unsafe_allow_html=True)
        st.markdown("---")
        mode_label = ["👶 " + t("sidebar.child_mode"), "👨‍👩‍👧 " + t("sidebar.parent_mode")]
        index = 0 if st.session_state.get(get_session_key("mode"), "child") == "child" else 1
        mode = st.radio(t("sidebar.mode"), mode_label, index=index, label_visibility="collapsed")
        st.session_state[get_session_key("mode")] = "child" if "👶" in mode else "parent"
        st.markdown("---")
        with st.expander(t("sidebar.guide")):
            st.markdown(t("sidebar.guide_child") + "\n\n" + t("sidebar.guide_parent"))
        st.markdown(t("sidebar.footer"), unsafe_allow_html=True)

def render_child_mode(tokenizer, model):
    st.markdown(t("child.header"), unsafe_allow_html=True)
    question = get_today_question()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="question-box" aria-label="{t("child.question_box")}">'
            f'<p class="question-text">{question}</p></div>', 
            unsafe_allow_html=True
        )

        if get_session_key("chat_history") not in st.session_state:
            st.session_state[get_session_key("chat_history")] = [
                {"role": "assistant", "content": f"{t('child.greeting')} {question}"}
            ]

        with st.container():
            for msg in st.session_state[get_session_key("chat_history")][-10:]:
                with st.chat_message(msg["role"], avatar="💝" if msg["role"] == "assistant" else "👶"):
                    st.write(msg["content"])

        if user_input := st.chat_input(t("child.input"), key="child_input"):
            st.session_state[get_session_key("chat_history")].append({"role": "user", "content": user_input})
            with st.spinner(t("child.processing")):
                emotion, confidence, _ = enhanced_emotion_classification(user_input, tokenizer, model)
                if get_session_key("child_history") not in st.session_state:
                    st.session_state[get_session_key("child_history")] = []
                st.session_state[get_session_key("child_history")].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "answer": user_input,
                    "emotion": emotion,
                    "confidence": confidence
                })
                messages = [
                    {"role": "system", "content": t("child.system_prompt")},
                    *st.session_state[get_session_key("chat_history")]
                ]
                try:
                    client = get_openai_client()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.8,
                        max_tokens=150
                    )
                    bot_response = response.choices[0].message.content.strip()
                except ValueError as e:
                    st.error(f"API 키 오류: {e}")
                    st.stop()
                except Exception:
                    bot_response = "죄송해요, 지금은 대답하기 어려워요. 잠시 후 다시 시도해주세요."
                st.session_state[get_session_key("chat_history")].append({"role": "assistant", "content": bot_response})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def render_parent_mode(tokenizer, model, rag_docs, font_path):
    if not require_parent_auth():
        return

    st.markdown(t("parent.header"), unsafe_allow_html=True)

    history = st.session_state.get(get_session_key("child_history"), [])
    today = date.today().isoformat()
    # 안전하게 필터링: timestamp, emotion, answer 포함된 항목만
    today_data = [
        h for h in history
        if isinstance(h, dict)
        and h.get("timestamp", "").startswith(today)
        and "emotion" in h
        and "answer" in h
        and "confidence" in h
    ]

    if not today_data:
        st.info(t("parent.no_data"))
        return

    total, positive, negative, pos_ratio, avg_confidence = calculate_metrics(today_data)
    render_metrics(total, positive, negative, pos_ratio, avg_confidence)
    render_charts(today_data, pos_ratio, positive, negative)
    render_conversations(today_data)
    render_actions(today_data, rag_docs, font_path)

def calculate_metrics(today_data):
    filtered = [h for h in today_data if h.get("emotion") in ("Positive", "Negative")]
    total = len(filtered)
    positive = sum(1 for h in filtered if h.get("emotion") == "Positive")
    negative = total - positive
    pos_ratio = (positive / total * 100) if total > 0 else 0
    avg_confidence = (sum(h.get("confidence", 0) for h in filtered) / total) if total > 0 else 0
    return total, positive, negative, pos_ratio, avg_confidence
