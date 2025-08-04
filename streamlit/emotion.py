import os
import streamlit as st
from dotenv import load_dotenv
import torch
import torch.nn.functional as F

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client
client = None
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    st.error("OpenAI package not installed. Please install it with: pip install openai")
except Exception as e:
    st.error(f"Error initializing OpenAI client: {e}")

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    """Load the emotion classification model and tokenizer"""
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
        
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
        config = AutoConfig.from_pretrained("klue/bert-base", num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", config=config)
        
        # Load custom weights if available
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                st.info("Custom model weights loaded successfully")
            except Exception as e:
                st.warning(f"Could not load custom model weights: {e}. Using default weights.")
        
        model.eval()
        return tokenizer, model
    except ImportError:
        st.error("Transformers package not installed. Please install it with: pip install transformers torch")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def enhanced_emotion_classification(text: str, tokenizer=None, model=None):
    """
    Enhanced emotion classification using multiple methods:
    1. Keyword-based classification
    2. BERT model classification (if available)
    3. OpenAI API classification (if available)
    """
    
    # Method 1: Keyword-based classification
    positive_keywords = ["좋아", "행복", "기뻐", "재미", "웃", "고마워", "신나", "즐거", "사랑", "최고", "완전", "대박"]
    negative_keywords = ["슬퍼", "싫어", "화나", "짜증", "힘들", "울", "무서워", "아파", "외로워", "최악", "별로", "답답"]
    
    # Check for strong keyword matches first
    text_lower = text.lower()
    for kw in positive_keywords:
        if kw in text_lower:
            return "Positive", 0.95, "키워드"
    for kw in negative_keywords:
        if kw in text_lower:
            return "Negative", 0.95, "키워드"
    
    # Method 2: BERT model classification
    model_emotion, model_conf = None, 0
    if tokenizer and model:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(probs.argmax())
                model_emotion = "Positive" if pred == 1 else "Negative"
                model_conf = float(probs[pred])
        except Exception as e:
            st.warning(f"Model classification failed: {e}")
    
    # Method 3: OpenAI API classification
    api_emotion, api_conf = None, 0
    if client:
        try:
            prompt = f'''아이의 답변을 분석해서 감정을 분류해주세요.
답변: "{text}"

다음 형식으로만 응답해주세요:
{{"emotion": "Positive" 또는 "Negative", "confidence": 0.0~1.0}}'''
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.1, 
                max_tokens=50
            )
            
            result_text = response.choices[0].message.content.strip()
            # Clean up the response to ensure it's valid JSON
            result_text = result_text.replace("'", '"')
            
            try:
                import json
                result = json.loads(result_text)
                api_emotion = result.get("emotion", "Unknown")
                api_conf = float(result.get("confidence", 0))
            except json.JSONDecodeError:
                # Fallback parsing
                if "Positive" in result_text:
                    api_emotion = "Positive"
                elif "Negative" in result_text:
                    api_emotion = "Negative"
                api_conf = 0.7  # Default confidence
                
        except Exception as e:
            st.warning(f"API classification failed: {e}")
    
    # Combine results
    if model_emotion and api_emotion:
        if model_emotion == api_emotion:
            # Both methods agree
            combined_conf = (model_conf + api_conf) / 2
            return model_emotion, combined_conf, "종합"
        else:
            # Methods disagree, use the one with higher confidence
            if model_conf > api_conf:
                return model_emotion, model_conf, "모델"
            else:
                return api_emotion, api_conf, "API"
    elif model_emotion:
        return model_emotion, model_conf, "모델"
    elif api_emotion:
        return api_emotion, api_conf, "API"
    else:
        # No classification method worked, default to neutral
        return "Neutral", 0.5, "기본값"

# Test function
def test_emotion_classification():
    """Test function for emotion classification"""
    test_cases = [
        "오늘 정말 행복했어요!",
        "친구들과 놀아서 너무 재미있었어요",
        "숙제가 너무 어려워서 힘들어요",
        "오늘은 그냥 보통이었어요"
    ]
    
    tokenizer, model = load_model_and_tokenizer()
    
    st.write("### Emotion Classification Test Results:")
    for text in test_cases:
        emotion, confidence, method = enhanced_emotion_classification(text, tokenizer, model)
        st.write(f"**Text:** {text}")
        st.write(f"**Emotion:** {emotion} (Confidence: {confidence:.2f}, Method: {method})")
        st.write("---")

if __name__ == "__main__":
    # This will run if the file is executed directly
    test_emotion_classification()
