import os
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
import re
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")

load_dotenv()

# OpenAI 클라이언트 반환 함수
def get_openai_client():
    """OpenAI 클라이언트를 반환합니다. Streamlit 환경에서만 secrets를 확인합니다."""
    api_key = None
    
    # Streamlit이 import되어 있는 경우에만 secrets 확인
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("OPENAI_API_KEY")
    except ImportError:
        pass
    except Exception:
        pass
    
    # 환경변수에서 키 확인
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    
    api_key = (api_key or "").strip()
    
    if not api_key:
        raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경변수에 OPENAI_API_KEY를 설정해주세요.")
    
    if not api_key.startswith("sk-"):
        print("Warning: API 키 형식이 평소와 다릅니다.")
    
    return OpenAI(api_key=api_key)

# 감정 분류 모델/토크나이저 로드
def load_model_and_tokenizer():
    """감정 분류 모델과 토크나이저를 로드합니다."""
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    try:
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        
        # 먼저 모델이 있는지 확인
        if os.path.exists(model_path):
            # 저장된 모델이 있으면 config와 함께 로드
            config = AutoConfig.from_pretrained("klue/bert-base", num_labels=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "klue/bert-base", 
                    config=config, 
                    ignore_mismatched_sizes=True
                )
            
            # 저장된 가중치 로드
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        else:
            # 모델 파일이 없으면 pretrained 모델 사용
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "klue/bert-base", 
                    num_labels=2, 
                    ignore_mismatched_sizes=True
                )
        
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None, None

def enhanced_emotion_classification(text: str, tokenizer, model):
    """텍스트의 감정을 분류합니다."""
    positive_keywords = ["좋아", "행복", "기뻐", "재미", "웃", "고마워", "신나", "즐거", "사랑"]
    negative_keywords = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워"]

    # 키워드 기반 분류
    for kw in positive_keywords:
        if kw in text:
            return "Positive", 0.95, "키워드"
    for kw in negative_keywords:
        if kw in text:
            return "Negative", 0.95, "키워드"

    # 모델 기반 분류
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
        except Exception as e:
            print(f"모델 추론 중 오류: {e}")

    # API 기반 분류
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
    except Exception as e:
        print(f"API 호출 중 오류: {e}")

    # 결과 종합
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
