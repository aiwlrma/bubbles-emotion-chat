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
import io
import base64

# --- 1. 페이지 설정 및 커스텀 CSS ---
st.set_page_config(
    page_title="마음이 - AI 감정 일기",
    page_icon="💝",
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 기본 숨김
)

# 깔끔하고 모던한 CSS
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: #f8fafc;
    }
    
    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* 카드 스타일 */
    .dashboard-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    /* 숫자 강조 */
    .big-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6366f1;
        margin: 0;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    /* 감정 지표 */
    .emotion-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .emotion-positive {
        background: #d1fae5;
        color: #065f46;
    }
    
    .emotion-negative {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .emotion-neutral {
        background: #e0e7ff;
        color: #3730a3;
    }
    
    /* 아이 모드 채팅 */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        max-width: 800px;
        margin: 0 auto;
    }
    
    .today-question {
        background: linear-gradient(135deg, #ddd6fe 0%, #c7d2fe 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .today-question h3 {
        color: #5b21b6;
        margin: 0;
        font-size: 1.3rem;
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
    }
    
    .stButton > button:hover {
        background: #4f46e5;
        transform: translateY(-1px);
    }
    
    /* 부모 인증 */
    .auth-container {
        max-width: 400px;
        margin: 4rem auto;
        background: white;
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        border: 2px solid #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: #6366f1;
        color: white;
        border-color: #6366f1;
    }
    
    /* 애니메이션 */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: slideIn 0.3s ease-out;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.8rem; }
        .big-number { font-size: 2rem; }
        .dashboard-card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 환경 변수 및 API/인증 코드 로드 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PARENT_CODE = os.getenv("PARENT_CODE", "1234")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- 3. 오늘의 질문 ---
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
        "내일은 뭘 하고 싶어? ✨",
        "오늘 받은 칭찬이 있다면 들려줄래? ⭐",
        "오늘 하루 중 가장 나다웠던 순간은? 🦄"
    ]

def get_today_question():
    today = date.today().isoformat()
    if "today_question" not in st.session_state or st.session_state.get("question_date") != today:
        questions = get_questions()
        question = random.choice(questions)
        st.session_state["today_question"] = question
        st.session_state["question_date"] = today
    return st.session_state["today_question"]

# --- 4. 모델/토크나이저 로드 ---
@st.cache_resource(show_spinner="AI를 준비하고 있어요... 🤖")
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

# --- 5. 개선된 감정 분류 (모델 + API 교차 검증) ---
POSITIVE_KEYWORDS = ["좋아", "행복", "기뻐", "재미", "웃", "고마워", "신나", "즐거", "사랑", "최고", "멋져"]
NEGATIVE_KEYWORDS = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워", "속상", "걱정"]

def enhanced_emotion_classification(text: str, tokenizer, model):
    """모델과 API를 함께 사용하여 더 정확한 감정 분류"""
    
    # 1. 룰 기반 체크
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            return "Positive", 0.95, "키워드"
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            return "Negative", 0.95, "키워드"
    
    # 2. 모델 예측
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
    
    # 3. API 예측
    api_emotion, api_conf = None, 0
    try:
        prompt = f"""
        다음 아이의 답변을 분석해서 감정을 분류해주세요.
        답변: "{text}"
        
        형식: {{"emotion": "Positive" 또는 "Negative", "confidence": 0.0~1.0}}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        
        import json
        result = json.loads(response.choices[0].message.content.strip())
        api_emotion = result.get("emotion", "Unknown")
        api_conf = float(result.get("confidence", 0))
    except:
        pass
    
    # 4. 종합 판단
    if model_emotion and api_emotion:
        # 둘 다 있으면 평균 신뢰도로 결정
        if model_emotion == api_emotion:
            return model_emotion, (model_conf + api_conf) / 2, "종합"
        else:
            # 의견이 다르면 신뢰도가 높은 쪽 선택
            if model_conf > api_conf:
                return model_emotion, model_conf, "모델"
            else:
                return api_emotion, api_conf, "API"
    elif model_emotion:
        return model_emotion, model_conf, "모델"
    elif api_emotion:
        return api_emotion, api_conf, "API"
    else:
        return "Neutral", 0.5, "기본값"

# --- 6. PDF 리포트 생성 ---
def create_pdf_report(history_data, today_stats):
    """깔끔한 PDF 리포트 생성"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # 커스텀 스타일
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6366f1'),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=12
    )
    
    # 제목
    story.append(Paragraph("마음이 감정 일기 리포트", title_style))
    story.append(Paragraph(f"{date.today().strftime('%Y년 %m월 %d일')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # 오늘의 요약
    story.append(Paragraph("📊 오늘의 감정 요약", heading_style))
    
    summary_data = [
        ['항목', '수치'],
        ['전체 대화 수', f"{today_stats['total']}회"],
        ['긍정 감정', f"{today_stats['positive']}회 ({today_stats['pos_ratio']:.0f}%)"],
        ['부정 감정', f"{today_stats['negative']}회 ({today_stats['neg_ratio']:.0f}%)"],
        ['평균 신뢰도', f"{today_stats['avg_confidence']:.1%}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.5*inch))
    
    # 대화 내용
    story.append(Paragraph("💬 오늘의 대화 내용", heading_style))
    
    for item in history_data:
        time = item['timestamp'].split()[1]
        emotion_text = "😊 긍정" if item['emotion'] == "Positive" else "😢 부정"
        
        conversation_text = f"""
        <b>시간:</b> {time}<br/>
        <b>감정:</b> {emotion_text} (신뢰도: {item['confidence']:.1%})<br/>
        <b>대화:</b> {item['answer']}<br/>
        """
        story.append(Paragraph(conversation_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # 부모님을 위한 조언
    if today_stats.get('advice'):
        story.append(PageBreak())
        story.append(Paragraph("💡 부모님을 위한 맞춤 조언", heading_style))
        story.append(Paragraph(today_stats['advice'], styles['Normal']))
    
    # PDF 생성
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 7. 세션/인증 관리 ---
def require_parent_auth():
    if not st.session_state.get("parent_authenticated", False):
        st.markdown("""
        <div class="auth-container animate-in">
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
        st.stop()

# --- 8. 네비게이션 ---
def render_navigation():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1>마음이 💝</h1>
            <p>AI와 함께하는 우리 아이 감정 일기</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 모드 선택
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👶 아이 모드", use_container_width=True, 
                    type="primary" if st.session_state.get("mode", "child") == "child" else "secondary"):
            st.session_state["mode"] = "child"
            st.rerun()
    with col2:
        if st.button("👨‍👩‍👧 부모 모드", use_container_width=True,
                    type="primary" if st.session_state.get("mode", "child") == "parent" else "secondary"):
            st.session_state["mode"] = "parent"
            st.rerun()

# --- 9. 아이 모드 (심플하고 재미있게) ---
def child_mode(tokenizer, model):
    st.markdown('<div class="chat-container animate-in">', unsafe_allow_html=True)
    
    # 오늘의 질문
    question = get_today_question()
    st.markdown(f"""
    <div class="today-question">
        <h3>{question}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 채팅 히스토리
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": f"안녕! 나는 마음이야 💝 {question}"
        })
    
    # 메시지 표시
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="💝" if msg["role"] == "assistant" else "👶"):
            st.write(msg["content"])
    
    # 입력
    if prompt := st.chat_input("이야기를 들려줘..."):
        # 사용자 메시지 추가
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        
        # 감정 분석
        with st.spinner("마음이가 듣고 있어요..."):
            emotion, confidence, method = enhanced_emotion_classification(prompt, tokenizer, model)
            
            # 히스토리 저장
            if "child_history" not in st.session_state:
                st.session_state["child_history"] = []
            
            st.session_state["child_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "answer": prompt,
                "emotion": emotion,
                "confidence": confidence,
                "method": method
            })
            
            # AI 응답 생성
            try:
                system_prompt = """
                너는 아이들의 친구 '마음이'야. 
                아이의 감정을 공감하고, 긍정적으로 반응해줘.
                짧고 따뜻하게, 이모티콘을 사용해서 대답해줘.
                추가 질문으로 대화를 이어가되, 한 번에 하나씩만 물어봐.
                """
                
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]])
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.8,
                    max_tokens=150
                )
                
                bot_response = response.choices[0].message.content.strip()
            except:
                bot_response = "와, 정말 좋은 이야기야! 더 들려줄래? 😊"
            
            st.session_state["chat_history"].append({"role": "assistant", "content": bot_response})
        
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 10. 부모 모드 (핵심만 간결하게) ---
def parent_mode(tokenizer, model):
    require_parent_auth()
    
    # 데이터 준비
    history = st.session_state.get("child_history", [])
    today = date.today().isoformat()
    today_data = [h for h in history if h["timestamp"].startswith(today)]
    
    # 통계 계산
    total = len(today_data)
    positive = sum(1 for h in today_data if h["emotion"] == "Positive")
    negative = total - positive
    pos_ratio = (positive / total * 100) if total > 0 else 0
    neg_ratio = 100 - pos_ratio
    avg_confidence = sum(h["confidence"] for h in today_data) / total if total > 0 else 0
    
    # 대시보드 헤더
    st.markdown("## 📊 오늘의 감정 대시보드")
    
    # 핵심 지표 (간결하게)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <p class="big-number">{}</p>
            <p class="subtitle">오늘의 대화</p>
        </div>
        """.format(total), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <p class="big-number" style="color: #10b981;">{}%</p>
            <p class="subtitle">긍정 비율</p>
        </div>
        """.format(int(pos_ratio)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <p class="big-number" style="color: #ef4444;">{}%</p>
            <p class="subtitle">부정 비율</p>
        </div>
        """.format(int(neg_ratio)), unsafe_allow_html=True)
    
    with col4:
        overall_mood = "😊 좋음" if pos_ratio >= 70 else "😐 보통" if pos_ratio >= 40 else "😢 관심필요"
        color = "#10b981" if pos_ratio >= 70 else "#f59e0b" if pos_ratio >= 40 else "#ef4444"
        st.markdown(f"""
        <div class="dashboard-card">
            <p class="big-number" style="color: {color}; font-size: 1.8rem;">{overall_mood}</p>
            <p class="subtitle">오늘의 기분</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 간단한 감정 차트
    if today_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 시간대별 감정 분포
            fig = go.Figure()
            
            times = [datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M").hour for h in today_data]
            emotions = [1 if h["emotion"] == "Positive" else -1 for h in today_data]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=emotions,
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=['#10b981' if e == 1 else '#ef4444' for e in emotions],
                    symbol=['circle' if e == 1 else 'x' for e in emotions]
                ),
                line=dict(color='#e5e7eb', width=2),
                name='감정 변화'
            ))
            
            fig.update_layout(
                title="오늘의 감정 변화",
                xaxis_title="시간",
                yaxis=dict(
                    tickvals=[-1, 0, 1],
                    ticktext=['부정', '중립', '긍정'],
                    range=[-1.5, 1.5]
                ),
                height=300,
                showlegend=False,
                plot_bgcolor='white',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💡 빠른 인사이트")
            
            # 가장 긍정적인 시간대
            positive_times = [datetime.strptime(h["timestamp"], "%Y-%m-%d %H:%M").hour 
                            for h in today_data if h["emotion"] == "Positive"]
            if positive_times:
                most_positive_hour = max(set(positive_times), key=positive_times.count)
                st.info(f"**가장 행복한 시간**: {most_positive_hour}시")
            
            # 대화 추천
            if neg_ratio > 30:
                st.warning("**오늘은 아이와 더 많은 대화가 필요해요**")
            else:
                st.success("**아이가 좋은 하루를 보냈어요!**")
    
    # 액션 버튼들
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 AI 대화 가이드 받기", use_container_width=True):
            if today_data:
                with st.spinner("AI가 분석중..."):
                    # AI 조언 생성
                    prompt = f"""
                    아이의 오늘 감정 데이터:
                    - 긍정: {positive}회, 부정: {negative}회
                    - 주요 대화: {', '.join([h['answer'][:20] + '...' for h in today_data[:3]])}
                    
                    부모님께 3가지 핵심 조언을 간단명료하게 제공해주세요.
                    """
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=300
                        )
                        advice = response.choices[0].message.content.strip()
                        st.session_state["today_advice"] = advice
                        
                        st.markdown("### 🌟 오늘의 대화 팁")
                        st.info(advice)
                    except:
                        st.error("조언 생성에 실패했습니다.")
    
    with col2:
        if st.button("📄 PDF 리포트 다운로드", use_container_width=True):
            if today_data:
                # PDF 생성을 위한 데이터 준비
                today_stats = {
                    'total': total,
                    'positive': positive,
                    'negative': negative,
                    'pos_ratio': pos_ratio,
                    'neg_ratio': neg_ratio,
                    'avg_confidence': avg_confidence,
                    'advice': st.session_state.get('today_advice', '')
                }
                
                # PDF 생성
                pdf_buffer = create_pdf_report(today_data, today_stats)
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 리포트 다운로드",
                    data=pdf_buffer,
                    file_name=f"마음이_리포트_{date.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("오늘의 대화 기록이 없습니다.")
    
    # 오늘의 대화 목록 (간단하게)
    if today_data:
        st.markdown("---")
        st.markdown("### 💬 오늘의 대화")
        
        for i, item in enumerate(today_data[-5:], 1):  # 최근 5개만 표시
            time = item['timestamp'].split()[1]
            emotion_badge = "emotion-positive" if item['emotion'] == "Positive" else "emotion-negative"
            emotion_icon = "😊" if item['emotion'] == "Positive" else "😢"
            
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f'<span class="emotion-indicator {emotion_badge}">{emotion_icon} {time}</span>', 
                          unsafe_allow_html=True)
            with col2:
                st.write(item['answer'])
        
        if len(today_data) > 5:
            st.info(f"... 그 외 {len(today_data) - 5}개의 대화가 더 있습니다")

# --- 11. 메인 함수 ---
def main():
    # 모델 로드
    tokenizer, model = load_model_and_tokenizer()
    
    # 네비게이션
    render_navigation()
    
    # 모드별 렌더링
    mode = st.session_state.get("mode", "child")
    
    if mode == "child":
        child_mode(tokenizer, model)
    else:
        parent_mode(tokenizer, model)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; padding: 1rem;">
        마음이와 함께 아이의 마음을 이해해보세요 💝<br>
        <a href="#" style="color: #6366f1;">도움말</a> | 
        <a href="#" style="color: #6366f1;">문의하기</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()