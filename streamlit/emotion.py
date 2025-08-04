import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
import re

load_dotenv()

@st.cache_resource(show_spinner=False)
def get_openai_client():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    api_key = (api_key or "").strip()
    if not api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Secrets 또는 환경변수에 `OPENAI_API_KEY`로 넣어주세요.")
        st.stop()
    if not api_key.startswith("sk-"):
        st.warning("API 키 형식이 평소와 다릅니다. 제대로 된 키인지 확인하세요.")
    # 디버그용 마스킹 출력 (배포 시 제거)
    masked = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    st.write("사용 중인 OpenAI 키 (마스킹):", masked)
    return OpenAI(api_key=api_key)

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

def enhanced_emotion_classification(text: str, tokenizer, model):
    positive_keywords = ["좋아", "행복", "기뻐", "재미", "웃", "고마워", "신나", "즐거", "사랑"]
    negative_keywords = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워"]

    for kw in positive_keywords:
        if kw in text:
            return "Positive", 0.95, "키워드"
    for kw in negative_keywords:
        if kw in text:
            return "Negative", 0.95, "키워드"

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
        except Exception:
            pass

    api_emotion, api_conf = None, 0
    try:
        client = get_openai_client()
        prompt = (
            f'아이의 답변을 분석해서 감정을 분류해주세요.\n'
            f'답변: "{text}"\n'
            '응답은 정확히 JSON 객체 하나로만 해주세요. 예: {"emotion": "Positive", "confidence": 0.87}'
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        content = response.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            result = json.loads(m.group())
            api_emotion = result.get("emotion")
            api_conf = float(result.get("confidence", 0))
    except Exception:
        pass

    if model_emotion and api_emotion:
        if model_emotion == api_emotion:
            return model_emotion, (model_conf + api_conf) / 2, "종합"
        if model_conf > api_conf:
            return model_emotion, model_conf, "모델"
        else:
            return api_emotion, api_conf, "API"
    elif model_emotion:
        return model_emotion, model_conf, "모델"
    elif api_emotion:
        return api_emotion, api_conf, "API"
    return "Neutral", 0.5, "기본값"
