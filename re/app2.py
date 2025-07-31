import os
import random
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# --- 1. 페이지 설정 및 커스텀 CSS ---
st.set_page_config(
    page_title="우리 아이 마음 일기",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .stApp {
        background: linear-gradient(to bottom right, #fef3c7, #fef9e7);
    }
    
    /* 메인 컨테이너 스타일 */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* 제목 스타일 */
    .app-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 아이 뷰 스타일 */
    .child-view-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 30px;
        padding: 3rem;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    .question-box {
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    .question-text {
        font-size: 1.8rem;
        font-weight: 600;
        line-height: 1.6;
        text-align: center;
    }
    
    /* 이모션 카드 */
    .emotion-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .emotion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* 채팅 메시지 스타일 */
    .stChatMessage {
        background: rgba(255,255,255,0.9);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        background: linear-gradient(to bottom, #f3e7fc, #e7e0fc);
    }
    
    /* 프로그레스 바 */
    .progress-container {
        background: #e0e7ff;
        border-radius: 20px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* 감정 뱃지 */
    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .emotion-positive {
        background: #86efac;
        color: #166534;
    }
    
    .emotion-negative {
        background: #fca5a5;
        color: #991b1b;
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* 툴팁 */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 환경 변수 및 API/인증 코드 로드 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PARENT_CODE = os.getenv("PARENT_CODE", "1234")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- 3. 오늘의 질문 (확장된 질문 세트) ---
@st.cache_data
def get_questions():
    return {
        "feelings": [
            "오늘 가장 기뻤던 일은 뭐였어? 🌟",
            "오늘 속상했던 일이 있었니? 🥺",
            "오늘 하루 중 가장 힘들었던 순간은 언제였어? 💪",
            "오늘 가장 많이 웃었던 순간은? 😄",
            "오늘 어떤 기분이 가장 많이 들었어? 🎨"
        ],
        "social": [
            "친구와 재미있게 놀았던 순간을 이야기해줄래? 👫",
            "오늘 누군가에게 고마웠던 일이 있었니? 💝",
            "오늘 엄마(아빠)에게 해주고 싶은 말이 있니? 💬",
            "오늘 누군가를 도와준 적이 있니? 🤝",
            "오늘 새로운 친구를 만났니? 🌈"
        ],
        "learning": [
            "오늘 학교(유치원)에서 배운 것 중 기억에 남는 게 있니? 📚",
            "오늘 새로운 것을 시도해본 적이 있니? 🚀",
            "오늘 무언가를 성공했을 때 어떤 기분이었어? 🏆",
            "오늘 가장 재미있었던 활동은 뭐였어? 🎯",
            "오늘 배운 것 중에 집에서도 해보고 싶은 게 있니? 🏠"
        ],
        "self": [
            "오늘 혼자만의 시간이 필요했던 적이 있었니? 🌙",
            "오늘 나에게 칭찬해주고 싶은 일이 있니? ⭐",
            "오늘 하루를 한 단어로 표현한다면? 💭",
            "내일은 뭘 하고 싶어? 🌅",
            "오늘 하루 중 가장 나다웠던 순간은? 🦄"
        ]
    }

def get_today_question():
    today = date.today().isoformat()
    if "today_question" not in st.session_state or st.session_state.get("question_date") != today:
        questions = get_questions()
        all_questions = []
        for category in questions.values():
            all_questions.extend(category)
        question = random.choice(all_questions)
        st.session_state["today_question"] = question
        st.session_state["question_date"] = today
    return st.session_state["today_question"]

# --- 4. 모델/토크나이저 로드 (캐싱) ---
@st.cache_resource(show_spinner="🤖 AI 모델을 준비하고 있어요...")
def load_model_and_tokenizer():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    config = AutoConfig.from_pretrained("klue/bert-base", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", config=config)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return tokenizer, model

# --- 5. 감정 분류 로직 (Rule → Model → API) ---
POSITIVE_KEYWORDS = ["좋아요", "행복", "기뻐", "재미", "웃", "고마워", "신나", "즐거", "사랑", "최고", "멋져", "훌륭"]
NEGATIVE_KEYWORDS = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워", "속상", "걱정", "불안"]

def rule_based_emotion(text: str):
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            return "Positive", 1.0, "룰 기반"
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            return "Negative", 1.0, "룰 기반"
    return None, None, None

def predict_emotion(text: str, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
            label = "Positive" if pred == 1 else "Negative"
            confidence = float(probs[pred])
        return label, confidence, "모델"
    except Exception as e:
        return None, None, f"모델 예외: {e}"

def api_emotion(text: str):
    prompt = (
        f"아래 어린이의 답변을 감정(Positive/Negative)으로 분류하고, 신뢰도(0~1)를 함께 알려줘.\n"
        f"답변: \"{text}\"\n"
        f"결과는 JSON 형식으로:\n"
        f"{{\"emotion\": \"Positive|Negative\", \"confidence\": 0.0~1.0}}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100,
        )
        import json
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get("emotion", "Unknown"), float(result.get("confidence", 0.0)), "API"
    except Exception as e:
        return "분석 실패", 0.0, f"API 예외: {e}"

def classify_emotion(text: str, tokenizer, model, conf_threshold=0.7):
    label, conf, method = rule_based_emotion(text)
    if label:
        return label, conf, method
    label, conf, method = predict_emotion(text, tokenizer, model)
    if label and conf is not None and conf >= conf_threshold:
        return label, conf, method
    label, conf, method = api_emotion(text)
    return label, conf, method

# --- 6. 부모용 리포트 생성 (개선된 버전) ---
def generate_parent_report(today_data):
    if not today_data:
        return "아직 오늘 아이의 답변이 없습니다. 아이와 함께 대화를 시작해보세요! 💬"
    
    prompt = (
        "아래는 아이의 오늘 답변과 감정 분석 결과입니다.\n"
        "부모님께 다음 내용을 포함해서 조언해주세요:\n"
        "1. 아이의 오늘 감정 상태 요약\n"
        "2. 긍정적인 감정을 강화하는 대화법\n"
        "3. 부정적인 감정을 다루는 방법\n"
        "4. 오늘 밤 아이와 나눌 수 있는 구체적인 대화 주제 2-3개\n"
        "답변은 따뜻하고 실용적으로 작성해주세요.\n\n"
    )
    
    for i, item in enumerate(today_data, 1):
        prompt += (
            f"{i}. 답변: \"{item['answer']}\"\n"
            f"   감정: {item['emotion']} (신뢰도: {item['confidence']:.2f})\n"
        )
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"리포트 생성 중 오류가 발생했습니다: {e}"

# --- 7. 감정 통계 시각화 함수들 ---
def create_emotion_gauge(positive_pct):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = positive_pct,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "오늘의 긍정 지수", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#fca5a5'},
                {'range': [25, 50], 'color': '#fde047'},
                {'range': [50, 75], 'color': '#bef264'},
                {'range': [75, 100], 'color': '#86efac'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#f8fafc",
        font={'color': "darkblue", 'family': "Arial"},
        height=300
    )
    
    return fig

def create_emotion_timeline(history):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['emotion_score'] = df['emotion'].apply(lambda x: 1 if x == "Positive" else -1)
    
    fig = go.Figure()
    
    # 긍정 감정
    positive_df = df[df['emotion'] == 'Positive']
    fig.add_trace(go.Scatter(
        x=positive_df['timestamp'],
        y=[1] * len(positive_df),
        mode='markers+text',
        name='긍정',
        marker=dict(size=15, color='#86efac', symbol='star'),
        text=positive_df['answer'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x),
        textposition="top center",
        hovertext=positive_df['answer'],
        hoverinfo='text'
    ))
    
    # 부정 감정
    negative_df = df[df['emotion'] == 'Negative']
    fig.add_trace(go.Scatter(
        x=negative_df['timestamp'],
        y=[-1] * len(negative_df),
        mode='markers+text',
        name='부정',
        marker=dict(size=15, color='#fca5a5', symbol='circle'),
        text=negative_df['answer'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x),
        textposition="bottom center",
        hovertext=negative_df['answer'],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="오늘의 감정 타임라인",
        xaxis_title="시간",
        yaxis_title="감정",
        yaxis=dict(
            tickmode='array',
            tickvals=[-1, 0, 1],
            ticktext=['😢 부정', '😐 중립', '😊 긍정']
        ),
        height=400,
        hovermode='closest',
        showlegend=True
    )
    
    return fig

def create_weekly_emotion_chart(history):
    if not history:
        return None
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # 최근 7일 데이터
    end_date = date.today()
    start_date = end_date - timedelta(days=6)
    
    daily_stats = []
    for i in range(7):
        current_date = start_date + timedelta(days=i)
        day_data = df[df['date'] == current_date]
        
        pos_count = len(day_data[day_data['emotion'] == 'Positive'])
        neg_count = len(day_data[day_data['emotion'] == 'Negative'])
        total = pos_count + neg_count
        
        daily_stats.append({
            'date': current_date,
            'day': current_date.strftime('%a'),
            'positive': pos_count,
            'negative': neg_count,
            'total': total,
            'pos_ratio': (pos_count / total * 100) if total > 0 else 0
        })
    
    stats_df = pd.DataFrame(daily_stats)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats_df['day'],
        y=stats_df['positive'],
        name='긍정',
        marker_color='#86efac',
        text=stats_df['positive'],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        x=stats_df['day'],
        y=stats_df['negative'],
        name='부정',
        marker_color='#fca5a5',
        text=stats_df['negative'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="주간 감정 분석",
        xaxis_title="요일",
        yaxis_title="답변 수",
        barmode='stack',
        height=350,
        showlegend=True
    )
    
    return fig

# --- 8. 세션/인증 관리 ---
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def require_parent_auth():
    if not st.session_state.get("parent_authenticated", False):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="main-container fade-in">
                <h3 style="text-align:center; color:#667eea;">🔒 부모님 인증</h3>
                <p style="text-align:center; color:#6b7280; margin-bottom:2rem;">
                    자녀의 감정 기록을 보호하기 위해 인증이 필요합니다.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            code = st.text_input("인증 코드를 입력하세요", type="password", placeholder="****")
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                login = st.button("🔓 로그인", use_container_width=True)
            
            if login:
                if code == PARENT_CODE:
                    st.session_state["parent_authenticated"] = True
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ 인증 코드가 올바르지 않습니다.")
        st.stop()

# --- 9. 사이드바 ---
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <h1 style="color:#667eea; margin:0;">🌈</h1>
            <h3 style="color:#667eea; margin:0;">우리 아이 마음 일기</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 네비게이션
        view = st.radio(
            "👥 사용자 선택",
            options=["👶 아이 모드", "👨‍👩‍👧 부모 모드"],
            index=0 if st.session_state.get("current_view", "child") == "child" else 1,
            key="view_selector"
        )
        
        st.session_state["current_view"] = "child" if "아이" in view else "parent"
        
        st.markdown("---")
        
        # 통계 미리보기
        if "child_history" in st.session_state:
            history = st.session_state["child_history"]
            today_data = [h for h in history if h["timestamp"].startswith(date.today().isoformat())]
            
            st.markdown("### 📊 오늘의 기록")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("전체 답변", f"{len(today_data)}개")
            with col2:
                pos = sum(1 for h in today_data if h["emotion"] == "Positive")
                st.metric("긍정 답변", f"{pos}개")
        
        st.markdown("---")
        
        # 로그아웃
        if st.button("🚪 로그아웃", use_container_width=True):
            reset_session()
            st.rerun()
        
        # 정보
        with st.expander("ℹ️ 앱 정보"):
            st.markdown("""
            **우리 아이 마음 일기**는 
            AI를 활용해 아이의 감정을 
            이해하고 소통을 돕는 
            스마트 육아 도우미입니다.
            
            - 버전: 2.0
            - 문의: support@example.com
            """)

# --- 10. Child View (개선된 UI) ---
def child_view(tokenizer, model):
    # 헤더
    st.markdown("""
    <div class="child-view-container fade-in">
        <h1 style="text-align:center; color:white; font-size:2.5rem; margin-bottom:0;">
            안녕! 오늘은 어떤 하루였니? 🌟
        </h1>
        <p style="text-align:center; color:rgba(255,255,255,0.8); font-size:1.2rem;">
            너의 이야기를 들려줘!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 오늘의 질문
    question = get_today_question()
    
    st.markdown(f"""
    <div class="question-box fade-in">
        <div class="question-text">
            {question}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 챗봇 히스토리 초기화
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": f"안녕! 나는 너의 친구 무지개야! 🌈\n\n오늘은 이런 이야기를 들려줄래?\n\n**{question}**"
        })
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    with chat_container:
        # 채팅 메시지 표시
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🌈"):
                    st.write(msg["content"])
            else:
                with st.chat_message("user", avatar="👶"):
                    st.write(msg["content"])
    
    # 입력 영역
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.chat_input("여기에 답변을 입력해줘... 💭", key="child_input")
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        
        # 챗봇 응답 생성
        with st.spinner("무지개가 생각하고 있어요... 🤔"):
            chat_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]
            
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": """당신은 아이들과 대화하는 친절하고 재미있는 AI 친구 '무지개'입니다. 
                        아이의 감정을 공감하고 긍정적으로 반응하며, 추가 질문을 통해 대화를 이어가세요.
                        이모티콘을 적절히 사용하고, 쉽고 따뜻한 언어를 사용하세요."""},
                        *chat_msgs
                    ],
                    temperature=0.7,
                    max_tokens=200,
                )
                bot_resp = resp.choices[0].message.content.strip()
            except Exception as e:
                bot_resp = "앗, 잠깐 문제가 생겼어요! 다시 한 번 이야기해줄래? 🙏"
        
        # 응답 저장 및 표시
        st.session_state["chat_history"].append({"role": "assistant", "content": bot_resp})
        
        # 감정 분석
        label, confidence, method = classify_emotion(user_input, tokenizer, model)
        
        # 히스토리에 저장
        history = st.session_state.setdefault("child_history", [])
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "answer": user_input,
            "emotion": label,
            "confidence": confidence,
            "method": method,
        })
        st.session_state["child_history"] = history
        
        # 감정에 따른 피드백
        emotion_feedback = ""
        if label == "Positive":
            emotion_feedback = "✨ 좋은 일이 있었구나! 정말 기뻐!"
            st.success(emotion_feedback)
        elif label == "Negative":
            emotion_feedback = "🤗 힘든 일이 있었구나. 괜찮아, 내가 있잖아!"
            st.info(emotion_feedback)
        
        st.rerun()
    
    # 하단 도움말
    with st.expander("💡 도움말"):
        st.markdown("""
        - 오늘 있었던 일을 자유롭게 이야기해줘요
        - 기쁜 일, 슬픈 일, 재미있었던 일 모두 좋아요
        - 무지개는 항상 너의 이야기를 들을 준비가 되어있어요! 🌈
        """)

# --- 11. Parent View (고급 대시보드) ---
def parent_view():
    require_parent_auth()
    
    # 헤더
    st.markdown("""
    <h1 class="app-title fade-in">부모님을 위한 마음 대시보드</h1>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    history = st.session_state.get("child_history", [])
    today = date.today().isoformat()
    today_data = [h for h in history if h["timestamp"].startswith(today)]
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 오늘의 감정", "📈 주간 리포트", "💬 대화 가이드", "📝 전체 기록"])
    
    with tab1:
        st.markdown("### 오늘의 감정 분석")
        
        if today_data:
            # 메트릭 카드
            col1, col2, col3, col4 = st.columns(4)
            
            pos = sum(1 for h in today_data if h["emotion"] == "Positive")
            neg = sum(1 for h in today_data if h["emotion"] == "Negative")
            total = pos + neg
            pos_pct = int(pos / total * 100) if total else 0
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">전체 대화</div>
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">회</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #86efac 0%, #22c55e 100%);">
                    <div class="metric-label">긍정 감정</div>
                    <div class="metric-value">{pos}</div>
                    <div class="metric-label">회 ({pos_pct}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #fca5a5 0%, #ef4444 100%);">
                    <div class="metric-label">부정 감정</div>
                    <div class="metric-value">{neg}</div>
                    <div class="metric-label">회 ({100-pos_pct}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_conf = sum(h["confidence"] for h in today_data) / len(today_data)
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);">
                    <div class="metric-label">평균 신뢰도</div>
                    <div class="metric-value">{avg_conf:.1%}</div>
                    <div class="metric-label">정확도</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 감정 게이지
            col1, col2 = st.columns([2, 3])
            with col1:
                fig_gauge = create_emotion_gauge(pos_pct)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # 타임라인
                fig_timeline = create_emotion_timeline(today_data)
                if fig_timeline:
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            # 오늘의 대화 내용
            st.markdown("### 💬 오늘의 대화 내용")
            for i, item in enumerate(today_data, 1):
                emotion_class = "emotion-positive" if item["emotion"] == "Positive" else "emotion-negative"
                emotion_icon = "😊" if item["emotion"] == "Positive" else "😢"
                
                st.markdown(f"""
                <div class="emotion-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{i}. {item['timestamp'].split()[1]}</strong>
                            <span class="emotion-badge {emotion_class}">{emotion_icon} {item['emotion']}</span>
                        </div>
                        <div class="tooltip">
                            <span style="color: #6b7280;">신뢰도: {item['confidence']:.1%}</span>
                            <span class="tooltiptext">{item['method']} 방식으로 분석됨</span>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; color: #374151;">
                        "{item['answer']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("아직 오늘의 대화 기록이 없습니다. 아이와 대화를 시작해보세요! 💬")
    
    with tab2:
        st.markdown("### 📈 주간 감정 추이")
        
        if history:
            # 주간 차트
            fig_weekly = create_weekly_emotion_chart(history)
            if fig_weekly:
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            # 주간 통계
            st.markdown("### 📊 주간 통계 요약")
            
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            week_data = df[df['date'] >= date.today() - timedelta(days=6)]
            
            if not week_data.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_week = len(week_data)
                    st.metric("주간 총 대화", f"{total_week}회")
                
                with col2:
                    pos_week = len(week_data[week_data['emotion'] == 'Positive'])
                    pos_week_pct = int(pos_week / total_week * 100) if total_week else 0
                    st.metric("주간 긍정률", f"{pos_week_pct}%", f"{pos_week}회")
                
                with col3:
                    avg_daily = total_week / 7
                    st.metric("일평균 대화", f"{avg_daily:.1f}회")
        else:
            st.info("주간 데이터를 표시하기 위한 충분한 기록이 없습니다.")
    
    with tab3:
        st.markdown("### 🤝 맞춤형 부모 대화 가이드")
        
        if st.button("💡 오늘의 대화 가이드 생성", type="primary", use_container_width=True):
            if today_data:
                with st.spinner("AI가 맞춤형 조언을 준비하고 있어요... 🤖"):
                    report = generate_parent_report(today_data)
                    st.session_state["parent_report"] = report
            else:
                st.warning("오늘의 대화 기록이 필요합니다.")
        
        if st.session_state.get("parent_report"):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%); 
                        padding: 2rem; border-radius: 20px; margin-top: 1rem;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
                <h4 style="color: #0369a1; margin-bottom: 1rem;">🌟 오늘의 대화 팁</h4>
                <div style="color: #0c4a6e; line-height: 1.8;">
            """, unsafe_allow_html=True)
            
            st.markdown(st.session_state["parent_report"])
            
            st.markdown("</div></div>", unsafe_allow_html=True)
        
        # 대화 팁 카드들
        st.markdown("### 💝 일반 대화 팁")
        
        tips = [
            {
                "icon": "👂",
                "title": "경청하기",
                "content": "아이가 말할 때 눈을 맞추고 고개를 끄덕이며 들어주세요."
            },
            {
                "icon": "🤗",
                "title": "공감하기",
                "content": "\"그랬구나\", \"힘들었겠다\" 같은 공감 표현을 사용하세요."
            },
            {
                "icon": "❓",
                "title": "열린 질문",
                "content": "\"어떤 기분이었어?\" 같은 열린 질문으로 대화를 이어가세요."
            },
            {
                "icon": "🎉",
                "title": "긍정 강화",
                "content": "작은 성취도 크게 칭찬하고 격려해주세요."
            }
        ]
        
        cols = st.columns(2)
        for i, tip in enumerate(tips):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="emotion-card">
                    <h4>{tip['icon']} {tip['title']}</h4>
                    <p style="color: #6b7280; margin-top: 0.5rem;">{tip['content']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### 📝 전체 대화 기록")
        
        if history:
            # 필터링 옵션
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date_filter = st.date_input("날짜 선택", value=date.today())
            
            with col2:
                emotion_filter = st.selectbox("감정 필터", ["전체", "Positive", "Negative"])
            
            with col3:
                sort_order = st.selectbox("정렬 순서", ["최신순", "오래된순"])
            
            # 데이터 필터링
            filtered_data = []
            for h in history:
                h_date = datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M").date()
                if h_date == date_filter:
                    if emotion_filter == "전체" or h["emotion"] == emotion_filter:
                        filtered_data.append(h)
            
            # 정렬
            if sort_order == "최신순":
                filtered_data.reverse()
            
            # 테이블 표시
            if filtered_data:
                df = pd.DataFrame(filtered_data)
                df['시간'] = df['timestamp'].apply(lambda x: x.split()[1])
                df['감정'] = df['emotion'].apply(lambda x: "😊 긍정" if x == "Positive" else "😢 부정")
                df['신뢰도'] = df['confidence'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    df[['시간', 'answer', '감정', '신뢰도', 'method']].rename(columns={
                        'answer': '대화 내용',
                        'method': '분석 방법'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # 다운로드 버튼
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV로 다운로드",
                    data=csv,
                    file_name=f"emotion_log_{date_filter}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("선택한 조건에 맞는 기록이 없습니다.")
        else:
            st.info("아직 저장된 대화 기록이 없습니다.")

# --- 12. 메인 실행 ---
def main():
    # 사이드바
    sidebar()
    
    # 모델 로드
    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        st.error(f"⚠️ 모델 로딩 중 오류가 발생했습니다: {e}")
        st.info("API 기반 감정 분석으로 전환됩니다.")
        tokenizer, model = None, None
    
    # 뷰 렌더링
    if st.session_state.get("current_view", "child") == "child":
        child_view(tokenizer, model)
    else:
        parent_view()

if __name__ == "__main__":
    main()