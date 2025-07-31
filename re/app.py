import os
import random
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, date
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# --- 1. 환경 변수 및 API/인증 코드 로드 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PARENT_CODE = os.getenv("PARENT_CODE", "1234")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- 2. 오늘의 질문 ---
@st.cache_data
def get_questions():
    return [
        "오늘 가장 기뻤던 일은 뭐였어?",
        "오늘 속상했던 일이 있었니?",
        "친구와 재미있게 놀았던 순간을 이야기해줄래?",
        "오늘 학교(유치원)에서 배운 것 중 기억에 남는 게 있니?",
        "오늘 누군가에게 고마웠던 일이 있었니?",
        "오늘 하루 중 가장 힘들었던 순간은 언제였어?",
        "오늘 엄마(아빠)에게 해주고 싶은 말이 있니?",
        "오늘 새로운 것을 시도해본 적이 있니?",
        "오늘 가장 많이 웃었던 순간은?",
        "오늘 혼자만의 시간이 필요했던 적이 있었니?"
    ]

def get_today_question():
    today = date.today().isoformat()
    if "today_question" not in st.session_state or st.session_state.get("question_date") != today:
        questions = get_questions()
        question = random.choice(questions)
        st.session_state["today_question"] = question
        st.session_state["question_date"] = today
    return st.session_state["today_question"]

# --- 3. 모델/토크나이저 로드 (캐싱) ---
@st.cache_data(show_spinner="모델 로딩 중...", persist=True)
def load_model_and_tokenizer():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    config = AutoConfig.from_pretrained("klue/bert-base", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", config=config)
    state = torch.load(model_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return tokenizer, model

# --- 4. 감정 분류 로직 (Rule → Model → API) ---
POSITIVE_KEYWORDS = ["좋아요", "행복", "기뻐", "재미", "웃", "고마워", "신나요", "즐거", "사랑"]
NEGATIVE_KEYWORDS = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워", "속상"]

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

# --- 5. 부모용 리포트 생성 ---
def generate_parent_report(today_data):
    if not today_data:
        return "아직 오늘 아이의 답변이 없습니다."
    prompt = (
        "아래는 아이의 오늘 답변과 감정 분석 결과입니다.\n"
        "부모가 아이와 대화할 때 참고할 수 있는 대화법, 격려, 주의점 등을 3~5줄로 제안해줘.\n"
        "답변은 친근하고 따뜻한 말투로 작성해줘.\n\n"
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
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"리포트 생성 중 오류가 발생했습니다: {e}"

# --- 6. 세션/인증 관리 ---
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def require_parent_auth():
    if not st.session_state.get("parent_authenticated", False):
        st.markdown("### 🔒 부모 인증")
        code = st.text_input("인증 코드", type="password")
        login = st.button("로그인")
        if login:
            if code == PARENT_CODE:
                st.session_state["parent_authenticated"] = True
                st.rerun()
            else:
                st.error("❌ 코드 오류")
        st.stop()

# --- 7. 사이드바 및 뷰 전환 ---
def sidebar():
    st.sidebar.title("👨‍👩‍👧‍👦 메뉴")
    view = st.sidebar.radio(
        "화면 선택",
        options=["Child View", "Parent View"],
        index=0 if st.session_state.get("current_view", "Child View") == "Child View" else 1,
        key="current_view"
    )
    if st.sidebar.button("로그아웃"):
        reset_session()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2024 Child-Parent Emotion App")

# --- 8. Child View (챗봇 인터페이스) ---
def child_view(tokenizer, model):
    st.markdown(
        "<h2 style='color:#2563eb;'>🧒 오늘의 질문</h2>",
        unsafe_allow_html=True
    )
    question = get_today_question()

    # 챗봇 히스토리 초기화
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": f"안녕! 오늘은 이 질문에 답해줄래? 💬\n\n**{question}**"
        })

    # 챗봇 메시지 표시 (오직 assistant 메시지만)
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])

    # 답변 입력
    user_input = st.chat_input("여기에 답변을 입력해 주세요.", key="child_input")
    if user_input:
        # 1. 챗봇 응답 생성 (OpenAI API, 전체 히스토리 전달)
        chat_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]
        chat_msgs.append({"role": "user", "content": user_input})
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 친절한 아동용 챗봇입니다."},
                    *chat_msgs
                ],
                temperature=0.6,
                max_tokens=200,
            )
            bot_resp = resp.choices[0].message.content.strip()
        except Exception as e:
            bot_resp = f"죄송해요, 답변 생성에 문제가 발생했어요. ({e})"

        # 2. 히스토리 저장 및 assistant 메시지만 렌더
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.session_state["chat_history"].append({"role": "assistant", "content": bot_resp})
        st.chat_message("assistant").write(bot_resp)

        # 3. 감정 분류 (백그라운드 저장)
        label, confidence, method = classify_emotion(user_input, tokenizer, model)
        history = st.session_state.setdefault("child_history", [])
        history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "answer": user_input,
            "emotion": label,
            "confidence": confidence,
            "method": method,
        })
        st.session_state["child_history"] = history

        st.rerun()

# --- 9. Parent View ---
def parent_view():
    require_parent_auth()
    st.markdown(
        "<h2 style='color:#f59e42;'>👨‍👩‍👧 부모 대시보드</h2>",
        unsafe_allow_html=True
    )
    history = st.session_state.get("child_history", [])
    today = date.today().isoformat()
    today_data = [h for h in history if h["timestamp"].startswith(today)]

    # 긍/부정 비율
    pos = sum(1 for h in today_data if h["emotion"] == "Positive")
    neg = sum(1 for h in today_data if h["emotion"] == "Negative")
    total = pos + neg
    pos_pct = int(pos / total * 100) if total else 0
    neg_pct = int(neg / total * 100) if total else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("오늘 긍정 비율", f"{pos_pct}%", f"{pos}회")
    with col2:
        st.metric("오늘 부정 비율", f"{neg_pct}%", f"{neg}회")

    # 대화 가이드
    if st.button("부모 대화 가이드 생성", type="primary"):
        with st.spinner("가이드를 생성하고 있어요..."):
            report = generate_parent_report(today_data)
        st.session_state["parent_report"] = report

    if st.session_state.get("parent_report"):
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);padding:1.5em;border-radius:12px;margin-top:1em;border-left:4px solid #0ea5e9;">
                <h4 style="color:#0ea5e9;margin-top:0;">🤗 부모님을 위한 대화 팁</h4>
                <div style="line-height:1.6;color:#374151;">
                    {st.session_state['parent_report']}
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    # 오늘 이력 테이블
    st.markdown("### 오늘 답변 이력")
    if today_data:
        import pandas as pd
        df = pd.DataFrame(today_data)
        df = df[["timestamp", "answer", "emotion", "confidence"]]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("아직 오늘 답변 이력이 없습니다.")

# --- 10. 메인 실행 ---
def main():
    st.set_page_config(
        page_title="Child-Parent Emotion App",
        page_icon="👨‍👩‍👧‍👦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    sidebar()
    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
        st.stop()
    if st.session_state.get("current_view", "Child View") == "Child View":
        child_view(tokenizer, model)
    else:
        parent_view()

if __name__ == "__main__":
    main()