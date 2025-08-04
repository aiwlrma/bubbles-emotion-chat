import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import json
import streamlit as st
from datetime import date
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = None
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    st.error("OpenAI package not installed. Please install it with: pip install openai")
except Exception as e:
    st.warning(f"OpenAI client initialization failed: {e}")

def load_rag_documents(rag_folder):
    """Load RAG documents for guidance generation"""
    EMOTION_GUIDANCE_DOCS = {
        "positive_reinforcement": """긍정적 감정 강화 가이드:
- 아이가 긍정적인 감정을 표현했을 때는 구체적으로 칭찬해주세요
- '정말 잘했네!', '네가 행복해하니 나도 기뻐' 같은 공감 표현을 사용하세요
- 긍정적 경험을 더 자세히 이야기하도록 격려해주세요
- 감정을 표현한 것 자체를 칭찬해주세요
- 긍정적인 순간들을 기록하고 나중에 다시 이야기해보세요""",
        
        "negative_support": """부정적 감정 지원 가이드:
- 먼저 아이의 감정을 인정하고 공감해주세요
- '많이 속상했겠구나', '힘들었겠네' 같은 표현을 사용하세요
- 해결책을 바로 제시하기보다 충분히 들어주세요
- 안전하고 편안한 분위기를 조성해주세요
- 아이가 감정을 표현할 수 있는 다양한 방법을 제시해주세요
- 부정적 감정도 자연스럽고 소중한 것임을 알려주세요""",
        
        "conversation_tips": """효과적인 대화 팁:
- 아이의 눈높이에 맞춰 대화하세요
- 열린 질문으로 대화를 이어가세요 ('어떻게 느꼈어?', '더 자세히 말해줄래?')
- 판단하지 않고 경청하세요
- 아이의 속도에 맞춰 대화하세요
- 일상적인 순간들을 대화의 기회로 활용하세요
- 아이의 감정을 반영해서 다시 말해주세요""",
        
        "emotional_development": """감정 발달 이해:
- 연령별 감정 표현의 차이를 이해하세요
- 감정 어휘를 확장시켜주세요 ('기쁘다' 외에 '신나다', '뿌듯하다' 등)
- 다양한 감정을 인정하고 수용해주세요
- 감정 조절 방법을 함께 찾아가세요
- 감정과 행동을 분리해서 이해하도록 도와주세요
- 감정 표현의 모델이 되어주세요"""
    }
    
    # Try to load additional documents from rag folder if it exists
    if os.path.exists(rag_folder) and os.path.isdir(rag_folder):
        try:
            for filename in os.listdir(rag_folder):
                if filename.endswith(".txt"):
                    file_path = os.path.join(rag_folder, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        key = os.path.splitext(filename)[0]
                        EMOTION_GUIDANCE_DOCS[key] = file.read().strip()
        except Exception as e:
            st.warning(f"Could not load additional RAG documents: {e}")
    
    return EMOTION_GUIDANCE_DOCS

def generate_rag_based_report(history_data, rag_docs):
    """Generate AI-based guidance report using RAG documents"""
    
    # Check if we have data
    if not history_data:
        return "아직 오늘 아이의 답변이 없습니다. 아이가 대화를 시작하면 분석을 제공하겠습니다."
    
    # Check API key
    if not client:
        return """OpenAI API를 사용할 수 없어 기본 가이드를 제공합니다:

📊 **오늘의 감정 분석**
- 총 대화: {}회
- 아이가 다양한 감정을 표현하고 있습니다.

💡 **부모님을 위한 조언**
1. **경청하기**: 아이의 이야기를 끝까지 들어주세요
2. **공감하기**: "그랬구나", "힘들었겠네" 같은 공감 표현을 사용하세요  
3. **격려하기**: 감정을 표현한 것 자체를 칭찬해주세요

🌟 **추천 대화법**
- "오늘 어떤 기분이었어?"
- "그때 어떤 생각이 들었어?"
- "네 마음을 이해할 것 같아"

이런 방식으로 아이와 꾸준히 소통해보세요!""".format(len(history_data))
    
    # Calculate metrics
    positive_count = sum(1 for h in history_data if h["emotion"] == "Positive")
    negative_count = len(history_data) - positive_count
    total_count = len(history_data)
    pos_ratio = (positive_count / total_count * 100) if total_count > 0 else 0
    
    # Select relevant documents based on emotion analysis
    relevant_docs = []
    if pos_ratio >= 70:
        relevant_docs.append(rag_docs.get("positive_reinforcement", ""))
    if negative_count > 0:
        relevant_docs.append(rag_docs.get("negative_support", ""))
    
    # Always include conversation tips and emotional development
    relevant_docs.append(rag_docs.get("conversation_tips", ""))
    relevant_docs.append(rag_docs.get("emotional_development", ""))
    
    # Create context from relevant documents
    context = "\n\n".join(filter(None, relevant_docs))
    
    # Create conversation summary (limit to last 5 conversations for context)
    conversations = []
    for h in history_data[-5:]:
        time = h['timestamp'].split()[1] if ' ' in h['timestamp'] else h['timestamp']
        conversations.append(f"- {time}: {h['answer'][:100]}... (감정: {h['emotion']}, 신뢰도: {h['confidence']:.1%})")
    
    conversations_text = "\n".join(conversations)
    
    # Create prompt for OpenAI
    prompt = f"""당신은 아동 심리 전문가입니다. 다음 정보를 바탕으로 부모님을 위한 맞춤형 조언을 작성해주세요.

[전문가 가이드라인]
{context}

[오늘의 아이 대화 기록]
{conversations_text}

[감정 분석 결과]
- 총 대화: {total_count}회
- 긍정적 감정: {positive_count}회 ({pos_ratio:.0f}%)
- 부정적 감정: {negative_count}회 ({100-pos_ratio:.0f}%)

위 정보를 바탕으로 부모님께 다음 내용을 포함한 조언을 작성해주세요:

1. **오늘의 감정 상태 요약** (2-3문장)
2. **구체적인 대화 방법 3가지** (실제 사용할 수 있는 문장 예시 포함)
3. **주의사항 및 권장사항** (2-3개)
4. **격려 메시지** (부모님을 위한)

따뜻하고 실용적이며 전문적인 조언으로 작성해주세요. 한국어로 작성하고, 이모지를 적절히 사용해서 읽기 쉽게 해주세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.error(f"AI 리포트 생성 중 오류가 발생했습니다: {e}")
        
        # Provide fallback report
        return f"""📊 **오늘의 감정 분석**

총 {total_count}회의 대화에서 긍정적 감정이 {pos_ratio:.0f}%를 차지했습니다.

💡 **부모님을 위한 조언**

1. **경청의 힘**: 아이의 이야기를 끝까지 들어주세요. "그랬구나", "더 말해볼래?" 같은 반응을 보여주세요.

2. **감정 인정하기**: {"긍정적인 감정을 더 격려해주세요. '네가 기뻐하니 나도 기뻐!' 같은 표현을 사용하세요." if pos_ratio >= 70 else "부정적인 감정도 자연스러운 것임을 알려주세요. '힘들었겠구나' 하며 공감해주세요."}

3. **꾸준한 소통**: 매일 조금씩이라도 아이와 감정에 대해 이야기해보세요.

🌟 **오늘 하루도 아이와 소중한 시간을 보내셨네요. 꾸준한 관심과 사랑이 아이의 감정 발달에 큰 도움이 됩니다!**"""

def create_pdf_report(history_data, report_content, font_path):
    """Create PDF report with emotion data and AI guidance"""
    
    if not history_data:
        st.warning("PDF를 생성할 데이터가 없습니다.")
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Try to register Korean font
    try:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('NotoSansCJKkr', font_path))
            font_name = 'NotoSansCJKkr'
        else:
            st.warning("한글 폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            font_name = 'Helvetica'
    except Exception as e:
        st.warning(f"폰트 로딩 실패: {e}. 기본 폰트를 사용합니다.")
        font_name = 'Helvetica'
    
    # Define styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6366f1'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName=font_name
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#4f46e5'),
        spaceAfter=12,
        fontName=font_name
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        fontName=font_name
    )
    
    # Add title and date
    story.append(Paragraph("Child Emotion Report", title_style))
    story.append(Paragraph(f"Report Date: {date.today().strftime('%Y-%m-%d')}", normal_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Add summary section
    story.append(Paragraph("Today's Summary", heading_style))
    positive_count = sum(1 for h in history_data if h["emotion"] == "Positive")
    negative_count = len(history_data) - positive_count
    total = len(history_data)
    
    summary_data = [
        ['Category', 'Value'],
        ['Total Conversations', str(total)],
        ['Positive Emotions', f"{positive_count} ({positive_count/total*100:.0f}%)"],
        ['Negative Emotions', f"{negative_count} ({negative_count/total*100:.0f}%)"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.5*inch))
    
    # Add conversation history
    story.append(Paragraph("Conversation History", heading_style))
    for i, item in enumerate(history_data[:10], 1):
        time = item['timestamp'].split()[1] if ' ' in item['timestamp'] else item['timestamp']
        emotion = "Positive" if item["emotion"] == "Positive" else "Negative"
        confidence = item.get('confidence', 0)
        
        conv_text = f"""<b>{i}. Time:</b> {time} | <b>Emotion:</b> {emotion} ({confidence:.0%})<br/>
<b>Content:</b> {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}<br/><br/>"""
        
        story.append(Paragraph(conv_text, normal_style))
    
    # Add AI recommendations if available
    if report_content and report_content.strip():
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("AI Recommendations", heading_style))
        # Clean up the report content for PDF
        clean_content = report_content.replace('**', '<b>').replace('**', '</b>')
        clean_content = clean_content.replace('*', '•')
        story.append(Paragraph(clean_content, normal_style))
    
    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")
        return None
