import streamlit as st
import random
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from emotion import load_model_and_tokenizer, enhanced_emotion_classification, client
from rag import load_rag_documents, generate_rag_based_report, create_pdf_report
from session import reset_session, get_session_key
from i18n import t
from style import render_css
from datetime import date, datetime

# Load environment variables
load_dotenv()
PARENT_CODE = os.getenv("PARENT_CODE", "1234")  # Default to "1234" if not set in .env

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
    render_css()
    st.set_page_config(page_title=t("app.title"), page_icon="💝", layout="wide")

    tokenizer, model = load_model_and_tokenizer()
    rag_folder = r"C:\Users\lemon\Desktop\AID\streamlit\rag"
    rag_docs = load_rag_documents(rag_folder)
    font_path = r"C:\Users\lemon\Desktop\AID\streamlit\NotoSansKR-Regular.ttf"

    render_sidebar()
    if st.session_state.get(get_session_key("mode"), "child") == "child":
        render_child_mode(tokenizer, model)
    else:
        render_parent_mode(tokenizer, model, rag_docs, font_path)

def render_sidebar():
    with st.sidebar:
        st.markdown(t("sidebar.header"), unsafe_allow_html=True)
        st.markdown("---")
        mode = st.radio(
            t("sidebar.mode"),
            ["👶 " + t("sidebar.child_mode"), "👨‍👩‍👧 " + t("sidebar.parent_mode")],
            index=0 if st.session_state.get(get_session_key("mode"), "child") == "child" else 1,
            label_visibility="collapsed",
        )
        st.session_state[get_session_key("mode")] = "child" if "아이" in mode else "parent"
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
            f'<div class="question-box" aria-label="{t("child.question_box")}"><p class="question-text">{question}</p></div>',
            unsafe_allow_html=True,
        )

        if get_session_key("chat_history") not in st.session_state:
            st.session_state[get_session_key("chat_history")] = [
                {"role": "assistant", "content": f"{t('child.greeting')} {question}"}
            ]

        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state[get_session_key("chat_history")][-10:]):
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
                messages = [{"role": "system", "content": t("child.system_prompt")}, *st.session_state[get_session_key("chat_history")]]
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.8,
                    max_tokens=150,
                )
                bot_response = response.choices[0].message.content.strip()
                st.session_state[get_session_key("chat_history")].append({"role": "assistant", "content": bot_response})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def render_parent_mode(tokenizer, model, rag_docs, font_path):
    if not require_parent_auth():
        return

    st.markdown(t("parent.header"), unsafe_allow_html=True)

    history = st.session_state.get(get_session_key("child_history"), [])
    today = date.today().isoformat()
    today_data = [h for h in history if h["timestamp"].startswith(today)]

    if not today_data:
        st.info(t("parent.no_data"))
        return

    total, positive, negative, pos_ratio, avg_confidence = calculate_metrics(today_data)
    render_metrics(total, positive, negative, pos_ratio, avg_confidence)
    render_charts(today_data, pos_ratio, positive, negative)
    render_conversations(today_data)
    render_actions(today_data, rag_docs, font_path)

def calculate_metrics(today_data):
    total = len(today_data)
    positive = sum(1 for h in today_data if h["emotion"] == "Positive")
    negative = total - positive
    pos_ratio = (positive / total * 100) if total > 0 else 0
    avg_confidence = sum(h["confidence"] for h in today_data) / total
    return total, positive, negative, pos_ratio, avg_confidence

def render_metrics(total, positive, negative, pos_ratio, avg_confidence):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{t("parent.metric.total")}</div>'
            f'<div class="metric-value" style="color: #6366f1;">{total}</div>'
            f'<div class="metric-label">{t("parent.metric.count")}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{t("parent.metric.positive")}</div>'
            f'<div class="metric-value" style="color: #10b981;">{positive}</div>'
            f'<div class="metric-label">{t("parent.metric.percent", value=pos_ratio)}%</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{t("parent.metric.negative")}</div>'
            f'<div class="metric-value" style="color: #ef4444;">{negative}</div>'
            f'<div class="metric-label">{t("parent.metric.percent", value=100-pos_ratio)}%</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        mood = t(f"parent.mood.{'good' if pos_ratio >= 70 else 'neutral' if pos_ratio >= 40 else 'bad'}")
        color = "#10b981" if pos_ratio >= 70 else "#f59e0b" if pos_ratio >= 40 else "#ef4444"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{t("parent.metric.mood")}</div>'
            f'<div class="metric-value" style="color: {color}; font-size: 1.5rem;">{mood}</div>'
            f'<div class="metric-label">{t("parent.metric.confidence", value=avg_confidence)}%</div></div>',
            unsafe_allow_html=True,
        )

def render_charts(today_data, pos_ratio, positive, negative):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📊 " + t("parent.chart.trend"))
        times = [datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M") for h in today_data]
        emotions = [1 if h["emotion"] == "Positive" else -1 for h in today_data]
        fig = go.Figure(
            go.Scatter(
                x=times,
                y=emotions,
                mode='markers+lines',
                marker=dict(size=12, color=['#10b981' if e == 1 else '#ef4444' for e in emotions]),
                line=dict(color='#e5e7eb', width=2),
                hovertemplate='%{x|%H:%M}<br>감정: %{y}<extra></extra>',
            )
        )
        fig.update_layout(
            xaxis_title=t("parent.chart.time"),
            yaxis=dict(
                tickvals=[-1, 0, 1],
                ticktext=[t("parent.chart.negative"), t("parent.chart.neutral"), t("parent.chart.positive")],
            ),
            height=300,
            showlegend=False,
            hovermode='x',
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 " + t("parent.chart.mood"))
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pos_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': t("parent.chart.positive_index"), 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#6366f1"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#fee2e2'},
                        {'range': [50, 100], 'color': '#d1fae5'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                },
            )
        )
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "#4a5568"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric(t("parent.metric.positive_short"), positive)
        st.metric(t("parent.metric.negative_short"), negative)
        st.markdown('</div>', unsafe_allow_html=True)

def render_conversations(today_data):
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 💬 " + t("parent.conversations"))
    for i, item in enumerate(today_data[-3:]):
        time = item['timestamp'].split()[1]
        tag_class = "tag-positive" if item['emotion'] == "Positive" else "tag-negative"
        emotion_text = t(f"parent.emotion.{'positive' if item['emotion'] == 'Positive' else 'negative'}")
        st.markdown(
            f'<div style="margin-bottom: 1rem;">'
            f'<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">'
            f'<span style="color: #64748b; font-size: 0.9rem;">{time}</span>'
            f'<span class="emotion-tag {tag_class}" aria-label="{emotion_text}">{emotion_text}</span>'
            f'</div>'
            f'<div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem;">'
            f'{item["answer"][:50]}{"..." if len(item["answer"]) > 50 else ""}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

def render_actions(today_data, rag_docs, font_path):
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("parent.guide_button"), use_container_width=True):
            with st.spinner(t("parent.processing")):
                st.session_state[get_session_key("ai_report")] = generate_rag_based_report(today_data, rag_docs)
    with col2:
        if st.button(t("parent.pdf_button"), use_container_width=True):
            report_content = st.session_state.get(
                get_session_key("ai_report"),
                generate_rag_based_report(today_data, rag_docs),
            )
            pdf_buffer = create_pdf_report(today_data, report_content, font_path)
            if pdf_buffer is None:
                st.error("PDF 생성에 실패했습니다.")
            else:
                try:
                    st.download_button(
                        label=t("parent.download"),
                        data=pdf_buffer.getvalue(),
                        file_name=f"emotion_report_{date.today().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"다운로드 처리 중 오류 발생: {e}")
    with col3:
        if st.button(t("parent.refresh"), use_container_width=True):
            st.rerun()

    if st.session_state.get(get_session_key("ai_report")):
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 " + t("parent.advice"))
        st.write(st.session_state[get_session_key("ai_report")])
        st.markdown('</div>', unsafe_allow_html=True)

def require_parent_auth():
    if not st.session_state.get(get_session_key("parent_authenticated"), False):
        st.markdown(t("auth.container"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            code = st.text_input(t("auth.code"), type="password", placeholder="****", label_visibility="hidden")
            if st.button(t("auth.submit"), use_container_width=True):
                if code == PARENT_CODE:
                    st.session_state[get_session_key("parent_authenticated")] = True
                    st.success(t("auth.success"))
                    st.rerun()
                else:
                    st.error(t("auth.failure"))
        return False
    return True
