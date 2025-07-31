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
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import io
import base64
import json

# --- 1. 페이지 설정 및 커스텀 CSS ---
st.set_page_config(
    page_title="마음이 - AI 감정 일기",
    page_icon="💝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 모던하고 깔끔한 CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: #fafbfc;
    }
    
    /* 메인 컨테이너 조정 */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 3rem;
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.1rem;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        display: block;
        margin: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background-color: rgba(255,255,255,0.2) !important;
    }
    
    /* 메인 헤더 */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .main-header h1 {
        color: #6366f1;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #64748b;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* 카드 스타일 */
    .content-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* 채팅 컨테이너 */
    .chat-wrapper {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        height: 600px;
        display: flex;
        flex-direction: column;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding-right: 1rem;
        margin-bottom: 1rem;
    }
    
    /* 대시보드 메트릭 */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        height: 100%;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* 질문 박스 */
    .question-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .question-text {
        color: #4c1d95;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: #6366f1;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #4f46e5;
        transform: translateY(-1px);
    }
    
    /* 감정 태그 */
    .emotion-tag {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .tag-positive {
        background: #d1fae5;
        color: #065f46;
    }
    
    .tag-negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* 인증 폼 */
    .auth-container {
        max-width: 400px;
        margin: 4rem auto;
        background: white;
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* 탭 숨김 */
    [data-testid="stTabs"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 환경 변수 및 API 로드 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PARENT_CODE = os.getenv("PARENT_CODE", "1234")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- 3. RAG 문서 저장소 ---
EMOTION_GUIDANCE_DOCS = {
    "positive_reinforcement": """
    긍정적 감정 강화 가이드:
    - 아이가 긍정적인 감정을 표현했을 때는 구체적으로 칭찬해주세요
    - "정말 잘했네!", "네가 행복해하니 나도 기뻐" 같은 공감 표현 사용
    - 긍정적 경험을 더 자세히 이야기하도록 격려
    - 감정을 표현한 것 자체를 칭찬
    """,
    "negative_support": """
    부정적 감정 지원 가이드:
    - 먼저 아이의 감정을 인정하고 공감해주세요
    - "많이 속상했겠구나", "힘들었겠네" 같은 표현 사용
    - 해결책을 바로 제시하기보다 충분히 들어주기
    - 안전하고 편안한 분위기 조성
    """,
    "conversation_tips": """
    효과적인 대화 팁:
    - 눈높이를 맞추고 대화하기
    - 열린 질문으로 대화 이어가기
    - 판단하지 않고 경청하기
    - 아이의 속도에 맞춰 대화하기
    """,
    "emotional_development": """
    감정 발달 이해:
    - 연령별 감정 표현의 차이 이해하기
    - 감정 어휘를 확장시켜주기
    - 다양한 감정을 인정하고 수용하기
    - 감정 조절 방법을 함께 찾아가기
    """
}

# --- 4. 오늘의 질문 ---
@st.cache_data
def get_questions():
    return [
        "오늘 가장 기뻤던 순간은 언제였어? 😊",
        "오늘 누군가에게 고마움을 느꼈니? 💝",
        "오늘 새롭게 배운 것이 있다면 뭐야? 📚",
        "오늘 친구와 어떤 이야기를 나눴어? 👫",
        "오늘 하루를 색깔로 표현한다면 무슨 색일까? 🎨",
        "오늘 가장 재미있었던 일은 뭐야? 🎉",
        "오늘 조금 힘들었던 일이 있었니? 🤗",
        "내일은 뭘 하고 싶어? ✨"
    ]

def get_today_question():
    today = date.today().isoformat()
    if "today_question" not in st.session_state or st.session_state.get("question_date") != today:
        questions = get_questions()
        question = random.choice(questions)
        st.session_state["today_question"] = question
        st.session_state["question_date"] = today
    return st.session_state["today_question"]

# --- 5. 모델 로드 ---
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    try:
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
    except Exception:
        return None, None

# --- 6. 감정 분류 (모델 + API 교차 검증) ---
def enhanced_emotion_classification(text: str, tokenizer, model):
    """모델과 API를 함께 사용하여 정확한 감정 분류"""
    
    # 키워드 기반
    positive_keywords = ["좋아", "행복", "기뻐", "재미", "웃", "고마워", "신나", "즐거", "사랑"]
    negative_keywords = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워"]
    
    for kw in positive_keywords:
        if kw in text:
            return "Positive", 0.95, "키워드"
    for kw in negative_keywords:
        if kw in text:
            return "Negative", 0.95, "키워드"
    
    # 모델 예측
    model_emotion, model_conf = None, 0
    if tokenizer and model:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                model_emotion = "Positive" if pred == 1 else "Negative"
                model_conf = float(probs[pred])
        except:
            pass
    
    # API 예측
    api_emotion, api_conf = None, 0
    try:
        prompt = f"""
        아이의 답변을 분석해서 감정을 분류해주세요.
        답변: "{text}"
        
        형식: {{"emotion": "Positive" 또는 "Negative", "confidence": 0.0~1.0}}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        api_emotion = result.get("emotion", "Unknown")
        api_conf = float(result.get("confidence", 0))
    except:
        pass
    
    # 종합 판단
    if model_emotion and api_emotion:
        if model_emotion == api_emotion:
            return model_emotion, (model_conf + api_conf) / 2, "종합"
        else:
            return (model_emotion if model_conf > api_conf else api_emotion,
                   max(model_conf, api_conf), "모델" if model_conf > api_conf else "API")
    elif model_emotion:
        return model_emotion, model_conf, "모델"
    elif api_emotion:
        return api_emotion, api_conf, "API"
    else:
        return "Neutral", 0.5, "기본값"

# --- 7. RAG 기반 리포트 생성 ---
def generate_rag_based_report(history_data):
    """RAG를 활용한 맞춤형 리포트 생성"""
    
    # 감정 분석
    positive_count = sum(1 for h in history_data if h["emotion"] == "Positive")
    negative_count = len(history_data) - positive_count
    pos_ratio = (positive_count / len(history_data) * 100) if history_data else 0
    
    # 관련 문서 선택
    relevant_docs = []
    if pos_ratio >= 70:
        relevant_docs.append(EMOTION_GUIDANCE_DOCS["positive_reinforcement"])
    if negative_count > 0:
        relevant_docs.append(EMOTION_GUIDANCE_DOCS["negative_support"])
    relevant_docs.append(EMOTION_GUIDANCE_DOCS["conversation_tips"])
    
    # 프롬프트 생성
    context = "\n\n".join(relevant_docs)
    
    conversations = "\n".join([
        f"- {h['timestamp'].split()[1]}: {h['answer']} (감정: {h['emotion']})"
        for h in history_data[:5]
    ])
    
    prompt = f"""
    다음 가이드라인을 참고하여 부모님을 위한 맞춤형 조언을 작성해주세요:
    
    [가이드라인]
    {context}
    
    [오늘의 아이 대화 기록]
    {conversations}
    
    [감정 분석 결과]
    - 긍정: {positive_count}회 ({pos_ratio:.0f}%)
    - 부정: {negative_count}회
    
    위 정보를 바탕으로 부모님께 다음 내용을 포함한 조언을 해주세요:
    1. 오늘 아이의 감정 상태 요약
    2. 구체적인 대화 방법 3가지
    3. 주의사항 및 권장사항
    
    따뜻하고 실용적인 조언으로 작성해주세요.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"리포트 생성에 실패했습니다: {str(e)}"

# --- 8. PDF 리포트 생성 (한글 지원) ---
def create_pdf_report(history_data, report_content):
    """PDF 리포트 생성"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # 스타일 설정
    styles = getSampleStyleSheet()
    
    # 제목 스타일
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6366f1'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        fontName='Helvetica'
    )
    
    # 제목
    story.append(Paragraph("Child Emotion Report", title_style))
    story.append(Paragraph(f"{date.today().strftime('%Y-%m-%d')}", normal_style))
    story.append(Spacer(1, 0.5*inch))
    
    # 통계 요약
    story.append(Paragraph("Today's Summary", heading_style))
    
    positive_count = sum(1 for h in history_data if h["emotion"] == "Positive")
    negative_count = len(history_data) - positive_count
    total = len(history_data)
    
    summary_data = [
        ['Category', 'Value'],
        ['Total Conversations', f"{total}"],
        ['Positive Emotions', f"{positive_count} ({positive_count/total*100:.0f}%)"],
        ['Negative Emotions', f"{negative_count} ({negative_count/total*100:.0f}%)"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.5*inch))
    
    # 대화 내용
    story.append(Paragraph("Conversation History", heading_style))
    
    for i, item in enumerate(history_data[:10], 1):
        time = item['timestamp'].split()[1]
        emotion = "Positive" if item['emotion'] == "Positive" else "Negative"
        confidence = item['confidence']
        
        conv_text = f"<b>{i}. Time:</b> {time} | <b>Emotion:</b> {emotion} ({confidence:.1%})<br/>"
        conv_text += f"<b>Content:</b> {item['answer']}<br/><br/>"
        
        story.append(Paragraph(conv_text, normal_style))
    
    # AI 조언
    if report_content:
        story.append(PageBreak())
        story.append(Paragraph("AI Recommendations", heading_style))
        story.append(Paragraph(report_content, normal_style))
    
    # PDF 생성
    doc.build(story)
    buffer.seek(0)
    
    return buffer

# --- 9. 인증 관리 ---
def require_parent_auth():
    if not st.session_state.get("parent_authenticated", False):
        st.markdown("""
        <div class="auth-container">
            <h2 style="color: #6366f1; margin-bottom: 1rem;">🔒 부모님 인증</h2>
            <p style="color: #64748b; margin-bottom: 2rem;">
                자녀의 소중한 감정 기록을 보호합니다
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            code = st.text_input("인증 코드", type="password", placeholder="****", label_visibility="hidden")
            if st.button("확인", use_container_width=True):
                if code == PARENT_CODE:
                    st.session_state["parent_authenticated"] = True
                    st.success("✅ 인증되었습니다!")
                    st.rerun()
                else:
                    st.error("❌ 올바른 코드가 아닙니다")
        return False
    return True

# --- 10. 사이드바 ---
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: white; margin: 0;">마음이 💝</h1>
            <p style="color: rgba(255,255,255,0.8); margin-top: 0.5rem;">AI 감정 일기</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 모드 선택
        mode = st.radio(
            "사용자 선택",
            options=["👶 아이 모드", "👨‍👩‍👧 부모 모드"],
            index=0 if st.session_state.get("mode", "child") == "child" else 1,
            label_visibility="collapsed"
        )
        
        st.session_state["mode"] = "child" if "아이" in mode else "parent"
        
        st.markdown("---")
        
        # 도움말
        with st.expander("💡 사용 가이드"):
            st.markdown("""
            **아이 모드**
            - 오늘의 질문에 답하기
            - 마음이와 대화하기
            
            **부모 모드**
            - 감정 분석 확인
            - AI 조언 받기
            - PDF 리포트 다운로드
            """)
        
        # 하단 정보
        st.markdown("""
        <div style="position: absolute; bottom: 1rem; left: 1rem; right: 1rem; text-align: center;">
            <p style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">
                © 2024 마음이<br>
                v2.0
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- 11. 아이 모드 ---
def render_child_mode(tokenizer, model):
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>안녕! 오늘은 어떤 하루였니? 🌈</h1>
        <p>마음이가 너의 이야기를 들어줄게</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 오늘의 질문
    question = get_today_question()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        # 질문 표시
        st.markdown(f"""
        <div class="question-box">
            <p class="question-text">{question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 채팅 히스토리 초기화
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"안녕! 나는 마음이야 💝\n\n{question}"
            })
        
        # 채팅 메시지 표시
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_history"][-10:]:  # 최근 10개만 표시
                with st.chat_message(msg["role"], avatar="💝" if msg["role"] == "assistant" else "👶"):
                    st.write(msg["content"])
        
        # 입력
        if user_input := st.chat_input("이야기를 들려줘..."):
            # 사용자 메시지 추가
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            
            # 감정 분석 및 저장
            with st.spinner("마음이가 듣고 있어요..."):
                emotion, confidence, method = enhanced_emotion_classification(user_input, tokenizer, model)
                
                # 히스토리 저장
                if "child_history" not in st.session_state:
                    st.session_state["child_history"] = []
                
                st.session_state["child_history"].append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "answer": user_input,
                    "emotion": emotion,
                    "confidence": confidence,
                    "method": method
                })
                
                # AI 응답 생성
                try:
                    messages = [
                        {"role": "system", "content": """
                        너는 아이들의 친구 '마음이'야. 
                        아이의 감정을 공감하고, 긍정적으로 반응해줘.
                        짧고 따뜻하게, 이모티콘을 사용해서 대답해줘.
                        추가 질문으로 대화를 이어가되, 한 번에 하나씩만 물어봐.
                        """},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.8,
                        max_tokens=150
                    )
                    
                    bot_response = response.choices[0].message.content.strip()
                except:
                    bot_response = "정말 좋은 이야기야! 더 들려줄래? 😊"
                
                st.session_state["chat_history"].append({"role": "assistant", "content": bot_response})
            
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # 오늘의 감정 요약
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 오늘의 감정")
        
        if "child_history" in st.session_state:
            today_history = [h for h in st.session_state["child_history"] 
                           if h["timestamp"].startswith(date.today().isoformat())]
            
            if today_history:
                positive = sum(1 for h in today_history if h["emotion"] == "Positive")
                negative = len(today_history) - positive
                
                # 감정 게이지
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = positive / len(today_history) * 100 if today_history else 0,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "긍정 지수", 'font': {'size': 16}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
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
                    font={'color': "#4a5568", 'family': "Arial"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 감정 카운트
                st.metric("😊 긍정", positive)
                st.metric("😢 부정", negative)
            else:
                st.info("아직 오늘의 대화가 없어요")
        else:
            st.info("대화를 시작해보세요!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- 12. 부모 모드 ---
def render_parent_mode(tokenizer, model):
    if not require_parent_auth():
        return
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>부모님을 위한 감정 분석 대시보드</h1>
        <p>AI가 분석한 우리 아이의 마음 상태</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 준비
    history = st.session_state.get("child_history", [])
    today = date.today().isoformat()
    today_data = [h for h in history if h["timestamp"].startswith(today)]
    
    if not today_data:
        st.info("📝 아직 오늘의 대화 기록이 없습니다. 아이 모드에서 대화를 시작해보세요.")
        return
    
    # 통계 계산
    total = len(today_data)
    positive = sum(1 for h in today_data if h["emotion"] == "Positive")
    negative = total - positive
    pos_ratio = (positive / total * 100) if total > 0 else 0
    avg_confidence = sum(h["confidence"] for h in today_data) / total
    
    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">오늘의 대화</div>
            <div class="metric-value" style="color: #6366f1;">{total}</div>
            <div class="metric-label">회</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">긍정 감정</div>
            <div class="metric-value" style="color: #10b981;">{positive}</div>
            <div class="metric-label">회 ({pos_ratio:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">부정 감정</div>
            <div class="metric-value" style="color: #ef4444;">{negative}</div>
            <div class="metric-label">회 ({100-pos_ratio:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        mood = "😊 좋음" if pos_ratio >= 70 else "😐 보통" if pos_ratio >= 40 else "😢 관심필요"
        color = "#10b981" if pos_ratio >= 70 else "#f59e0b" if pos_ratio >= 40 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">전체 기분</div>
            <div class="metric-value" style="color: {color}; font-size: 1.5rem;">{mood}</div>
            <div class="metric-label">신뢰도 {avg_confidence:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 차트와 대화 내용
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 📊 오늘의 감정 변화")
        
        # 시간대별 감정 차트
        times = []
        emotions = []
        for h in today_data:
            time = datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M")
            times.append(time)
            emotions.append(1 if h["emotion"] == "Positive" else -1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=emotions,
            mode='markers+lines',
            marker=dict(
                size=12,
                color=['#10b981' if e == 1 else '#ef4444' for e in emotions],
            ),
            line=dict(color='#e5e7eb', width=2),
            hovertemplate='%{x|%H:%M}<br>감정: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="시간",
            yaxis=dict(
                tickvals=[-1, 0, 1],
                ticktext=['😢 부정', '😐 중립', '😊 긍정'],
                range=[-1.5, 1.5]
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
        st.markdown("### 💬 최근 대화")
        
        for item in today_data[-3:]:  # 최근 3개만
            time = item['timestamp'].split()[1]
            tag_class = "tag-positive" if item['emotion'] == "Positive" else "tag-negative"
            emotion_text = "긍정" if item['emotion'] == "Positive" else "부정"
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="color: #64748b; font-size: 0.9rem;">{time}</span>
                    <span class="emotion-tag {tag_class}">{emotion_text}</span>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem;">
                    {item['answer'][:50]}{'...' if len(item['answer']) > 50 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 액션 버튼
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 AI 대화 가이드", use_container_width=True):
            with st.spinner("AI가 분석 중입니다..."):
                report = generate_rag_based_report(today_data)
                st.session_state["ai_report"] = report
    
    with col2:
        if st.button("📄 PDF 리포트 생성", use_container_width=True):
            report_content = st.session_state.get("ai_report", "")
            if not report_content:
                with st.spinner("리포트 생성 중..."):
                    report_content = generate_rag_based_report(today_data)
                    st.session_state["ai_report"] = report_content
            
            pdf_buffer = create_pdf_report(today_data, report_content)
            
            st.download_button(
                label="📥 다운로드",
                data=pdf_buffer,
                file_name=f"emotion_report_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with col3:
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()
    
    # AI 리포트 표시
    if st.session_state.get("ai_report"):
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 AI 맞춤 조언")
        st.write(st.session_state["ai_report"])
        st.markdown('</div>', unsafe_allow_html=True)

# --- 13. 메인 함수 ---
def main():
    # 세션 초기화
    if "mode" not in st.session_state:
        st.session_state["mode"] = "child"
    
    # 모델 로드
    with st.spinner("AI를 준비하고 있어요..."):
        tokenizer, model = load_model_and_tokenizer()
    
    # 사이드바 렌더링
    render_sidebar()
    
    # 메인 컨텐츠
    if st.session_state["mode"] == "child":
        render_child_mode(tokenizer, model)
    else:
        render_parent_mode(tokenizer, model)

if __name__ == "__main__":
    main()