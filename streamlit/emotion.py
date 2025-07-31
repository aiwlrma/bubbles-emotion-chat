import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)  # Module-level client

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
        except:
            pass
    
    api_emotion, api_conf = None, 0
    try:
        prompt = f'아이의 답변을 분석해서 감정을 분류해주세요.\n답변: "{text}"\n형식: {{"emotion": "Positive" 또는 "Negative", "confidence": 0.0~1.0}}'
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=50)
        result = eval(response.choices[0].message.content.strip())
        api_emotion = result.get("emotion", "Unknown")
        api_conf = float(result.get("confidence", 0))
    except:
        pass
    
    if model_emotion and api_emotion:
        if model_emotion == api_emotion:
            return model_emotion, (model_conf + api_conf) / 2, "종합"
        return (model_emotion if model_conf > api_conf else api_emotion, max(model_conf, api_conf), "모델" if model_conf > api_conf else "API")
    elif model_emotion:
        return model_emotion, model_conf, "모델"
    elif api_emotion:
        return api_emotion, api_conf, "API"
    return "Neutral", 0.5, "기본값"