import streamlit as st
import random
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from datetime import date, datetime

# Try to import modules with error handling
try:
    from emotion import load_model_and_tokenizer, enhanced_emotion_classification
    # Import client separately as it might not be available
    try:
        from emotion import client
    except ImportError:
        # Create a fallback OpenAI client
        try:
            from openai import OpenAI
            import os
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
            client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        except:
            client = None
except ImportError as e:
    st.error(f"Could not import emotion module: {e}")
    # Provide fallback functions
    def load_model_and_tokenizer():
        return None, None
    def enhanced_emotion_classification(text, tokenizer=None, model=None):
        return "Neutral", 0.5, "fallback"
    client = None

try:
    from rag import load_rag_documents, generate_rag_based_report, create_pdf_report
except ImportError as e:
    st.error(f"Could not import rag module: {e}")
    # Provide fallback functions
    def load_rag_documents(folder):
        return []
    def generate_rag_based_report(data, docs):
        return "Report generation not available"
    def create_pdf_report(data, content, font_path):
        return b"PDF generation not available"

try:
    from session import reset_session, get_session_key
except ImportError as e:
    st.error(f"Could not import session module: {e}")
    # Provide fallback functions
    def get_session_key(key):
        return f"session_{key}"
    def reset_session():
        pass

try:
    from i18n import t
except ImportError as e:
    st.error(f"Could not import i18n module: {e}")
    # Provide fallback translation function
    def t(key, **kwargs):
        # Basic English translations as fallback
        translations = {
            "app.title": "Emotion Chat",
            "sidebar.header": "🌟 Emotion Chat",
            "sidebar.mode": "Select Mode",
            "sidebar.child_mode": "Child Mode",
            "sidebar.parent_mode": "Parent Mode",
            "sidebar.guide": "User Guide",
            "sidebar.guide_child": "Child mode: Answer daily questions and chat with the AI assistant.",
            "sidebar.guide_parent": "Parent mode: View emotion analysis and get guidance reports.",
            "sidebar.footer": "Made with ❤️",
            "child.header": "# 👶 Child Mode",
            "child.question_box": "Today's Question",
            "child.greeting": "Hello! Let's talk about:",
            "child.input": "Type your answer here...",
            "child.processing": "Processing...",
            "child.system_prompt": "You are a friendly AI assistant talking to a child. Be encouraging and positive.",
            "parent.header": "# 👨‍👩‍👧 Parent Dashboard",
            "parent.no_data": "No data available for today.",
            "parent.metric.total": "Total Responses",
            "parent.metric.positive": "Positive",
            "parent.metric.negative": "Negative",
            "parent.metric.count": "responses",
            "parent.metric.percent": "{value:.1f}",
            "parent.metric.mood": "Overall Mood",
            "parent.metric.confidence": "Confidence: {value:.1f}",
            "parent.mood.good": "😊 Good",
            "parent.mood.neutral": "😐 Neutral", 
            "parent.mood.bad": "😟 Needs Attention",
            "parent.chart.trend": "Emotion Trend",
            "parent.chart.mood": "Mood Overview",
            "parent.chart.time": "Time",
            "parent.chart.negative": "Negative",
            "parent.chart.neutral": "Neutral",
            "parent.chart.positive": "Positive",
            "parent.chart.positive_index": "Positivity Index",
            "parent.metric.positive_short": "Positive",
            "parent.metric.negative_short": "Negative",
            "parent.conversations": "Recent Conversations",
            "parent.emotion.positive": "Positive",
            "parent.emotion.negative": "Negative",
            "parent.guide_button": "Generate AI Guidance",
            "parent.pdf_button": "Create PDF Report",
            "parent.download": "Download PDF",
            "parent.refresh": "Refresh",
            "parent.processing": "Generating report...",
            "parent.advice": "AI Guidance & Recommendations",
            "auth.container": '<div style="text-align: center; padding: 2rem;"><h3>🔒 Parent Authentication Required</h3></div>',
            "auth.code": "Enter Parent Code",
            "auth.submit": "Access Dashboard",
            "auth.success": "Access granted!",
            "auth.failure": "Invalid code. Please try again.",
            "questions.positive_moment": "What made you happy today?",
            "questions.gratitude": "What are you grateful for today?",
            "questions.new_learn": "What new thing did you learn today?",
            "questions.friend_talk": "Tell me about your friends today.",
            "questions.color_day": "If today was a color, what would it be and why?",
            "questions.fun_moment": "What was the most fun part of your day?",
            "questions.hard_moment": "Was there anything difficult today?",
            "questions.tomorrow_plan": "What are you looking forward to tomorrow?"
        }
        
        # Handle nested keys like "parent.metric.percent"
        if key in translations:
            result = translations[key]
            # Handle format strings
            if "{value" in result and kwargs:
                try:
                    return result.format(**kwargs)
                except:
                    return result
            return result
        return key

try:
    from style import render_css
except ImportError as e:
    st.error(f"Could not import style module: {e}")
    # Provide fallback CSS function
    def render_css():
        st.markdown("""
        <style>
        .content-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .question-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .question-text {
            font-size: 1.2rem;
            font-weight: 600;
            margin: 0;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .emotion-tag {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .tag-positive {
            background: #d1fae5;
            color: #065f46;
        }
        .tag-negative {
            background: #fee2e2;
            color: #991b1b;
        }
        </style>
        """, unsafe_allow_html=True)

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
    st.set_page_config(page_title=t("app.title"), page_icon="💝", layout="wide")
    render_css()
    
    tokenizer, model = load_model_and_tokenizer()
    rag_folder = os.path.join(os.path.dirname(__file__), "rag")
    rag_docs = load_rag_documents(rag_folder)
    font_path = os.path.join(os.path.dirname(__file__), "NotoSansKR-Regular.ttf")
    
    render_sidebar()
    if st.session_state.get(get_session_key("mode"), "child") == "child":
        render_child_mode(tokenizer, model)
    else:
        render_parent_mode(tokenizer, model, rag_docs, font_path)

def render_sidebar():
    with st.sidebar:
        st.markdown(t("sidebar.header"), unsafe_allow_html=True)
        st.markdown("---")
        current_mode = st.session_state.get(get_session_key("mode"), "child")
        mode = st.radio(
            t("sidebar.mode"), 
            ["👶 " + t("sidebar.child_mode"), "👨‍👩‍👧 " + t("sidebar.parent_mode")], 
            index=0 if current_mode == "child" else 1, 
            label_visibility="collapsed"
        )
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
        st.markdown(f'<div class="question-box" aria-label="{t("child.question_box")}"><p class="question-text">{question}</p></div>', unsafe_allow_html=True)
        
        chat_history_key = get_session_key("chat_history")
        if chat_history_key not in st.session_state:
            st.session_state[chat_history_key] = [{"role": "assistant", "content": f"{t('child.greeting')} {question}"}]
        
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state[chat_history_key][-10:]):
                with st.chat_message(msg["role"], avatar="💝" if msg["role"] == "assistant" else "👶"):
                    st.write(msg["content"])
        
        if user_input := st.chat_input(t("child.input"), key="child_input"):
            st.session_state[chat_history_key].append({"role": "user", "content": user_input})
            with st.spinner(t("child.processing")):
                emotion, confidence, _ = enhanced_emotion_classification(user_input, tokenizer, model)
                
                child_history_key = get_session_key("child_history")
                if child_history_key not in st.session_state:
                    st.session_state[child_history_key] = []
                
                st.session_state[child_history_key].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "answer": user_input,
                    "emotion": emotion,
                    "confidence": confidence
                })
                
                # Generate bot response
                if client:
                    try:
                        messages = [{"role": "system", "content": t("child.system_prompt")}, *st.session_state[chat_history_key]]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo", 
                            messages=messages, 
                            temperature=0.8, 
                            max_tokens=150
                        )
                        bot_response = response.choices[0].message.content.strip()
                    except Exception as e:
                        bot_response = "Thank you for sharing! Can you tell me more?"
                else:
                    bot_response = "Thank you for sharing! That's wonderful to hear."
                
                st.session_state[chat_history_key].append({"role": "assistant", "content": bot_response})
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
    avg_confidence = sum(h["confidence"] for h in today_data) / total if total > 0 else 0
    return total, positive, negative, pos_ratio, avg_confidence

def render_metrics(total, positive, negative, pos_ratio, avg_confidence):
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.markdown(f'<div class="metric-card"><div class="metric-label">{t("parent.metric.total")}</div><div class="metric-value" style="color: #6366f1;">{total}</div><div class="metric-label">{t("parent.metric.count")}</div></div>', unsafe_allow_html=True)
    with col2: 
        st.markdown(f'<div class="metric-card"><div class="metric-label">{t("parent.metric.positive")}</div><div class="metric-value" style="color: #10b981;">{positive}</div><div class="metric-label">{t("parent.metric.percent", value=pos_ratio)}</div></div>', unsafe_allow_html=True)
    with col3: 
        st.markdown(f'<div class="metric-card"><div class="metric-label">{t("parent.metric.negative")}</div><div class="metric-value" style="color: #ef4444;">{negative}</div><div class="metric-label">{t("parent.metric.percent", value=100-pos_ratio)}</div></div>', unsafe_allow_html=True)
    with col4:
        mood = t(f"parent.mood.{'good' if pos_ratio >= 70 else 'neutral' if pos_ratio >= 40 else 'bad'}")
        color = "#10b981" if pos_ratio >= 70 else "#f59e0b" if pos_ratio >= 40 else "#ef4444"
        st.markdown(f'<div class="metric-card"><div class="metric-label">{t("parent.metric.mood")}</div><div class="metric-value" style="color: {color}; font-size: 1.5rem;">{mood}</div><div class="metric-label">{t("parent.metric.confidence", value=avg_confidence*100)}</div></div>', unsafe_allow_html=True)

def render_charts(today_data, pos_ratio, positive, negative):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📊 " + t("parent.chart.trend"))
        times = [datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M") for h in today_data]
        emotions = [1 if h["emotion"] == "Positive" else -1 for h in today_data]
        fig = go.Figure(go.Scatter(
            x=times, 
            y=emotions, 
            mode='markers+lines', 
            marker=dict(size=12, color=['#10b981' if e == 1 else '#ef4444' for e in emotions]), 
            line=dict(color='#e5e7eb', width=2), 
            hovertemplate='%{x|%H:%M}<br>감정: %{customdata}<extra></extra>',
            customdata=[t("parent.chart.positive") if e == 1 else t("parent.chart.negative") for e in emotions]
        ))
        fig.update_layout(
            xaxis_title=t("parent.chart.time"), 
            yaxis=dict(
                tickvals=[-1, 0, 1], 
                ticktext=[t("parent.chart.negative"), t("parent.chart.neutral"), t("parent.chart.positive")]
            ), 
            height=300, 
            showlegend=False, 
            hovermode='x', 
            plot_bgcolor='white', 
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 " + t("parent.chart.mood"))
        fig = go.Figure(go.Indicator(
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
                'threshold': {
                    'line': {'color': "red", 'width': 4}, 
                    'thickness': 0.75, 
                    'value': 90
                }
            }
        ))
        fig.update_layout(
            height=200, 
            margin=dict(l=0, r=0, t=30, b=0), 
            paper_bgcolor='rgba(0,0,0,0)', 
            font={'color': "#4a5568"}
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
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span style="color: #64748b; font-size: 0.9rem;">{time}</span>
                <span class="emotion-tag {tag_class}" aria-label="{emotion_text}">{emotion_text}</span>
            </div>
            <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem;">
                {item["answer"][:50]}{"..." if len(item["answer"]) > 50 else ""}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_actions(today_data, rag_docs, font_path):
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1: 
        if st.button(t("parent.guide_button"), use_container_width=True):
            with st.spinner(t("parent.processing")): 
                report = generate_rag_based_report(today_data, rag_docs)
                st.session_state[get_session_key("ai_report")] = report
    with col2:
        if st.button(t("parent.pdf_button"), use_container_width=True):
            report_content = st.session_state.get(get_session_key("ai_report"), generate_rag_based_report(today_data, rag_docs))
            pdf_buffer = create_pdf_report(today_data, report_content, font_path)
            st.download_button(
                label=t("parent.download"), 
                data=pdf_buffer, 
                file_name=f"emotion_report_{date.today().strftime('%Y%m%d')}.pdf", 
                mime="application/pdf", 
                use_container_width=True
            )
    with col3: 
        if st.button(t("parent.refresh"), use_container_width=True): 
            st.rerun()
    
    ai_report_key = get_session_key("ai_report")
    if st.session_state.get(ai_report_key):
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 " + t("parent.advice"))
        st.write(st.session_state[ai_report_key])
        st.markdown('</div>', unsafe_allow_html=True)

def require_parent_auth():
    auth_key = get_session_key("parent_authenticated")
    if not st.session_state.get(auth_key, False):
        st.markdown(t("auth.container"), unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            code = st.text_input(t("auth.code"), type="password", placeholder="****", label_visibility="hidden")
            if st.button(t("auth.submit"), use_container_width=True):
                if code == PARENT_CODE:
                    st.session_state[auth_key] = True
                    st.success(t("auth.success"))
                    st.rerun()
                else:
                    st.error(t("auth.failure"))
        return False
    return True
