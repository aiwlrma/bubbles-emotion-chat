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
import json  # ← 추가
import streamlit as st  # ← 추가
from openai import OpenAI  # ← 추가

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_rag_documents(rag_folder):
    EMOTION_GUIDANCE_DOCS = {
        "positive_reinforcement": "긍정적 감정 강화 가이드: 아이가 긍정적인 감정을 표현했을 때는 구체적으로 칭찬해주세요. '정말 잘했네!', '네가 행복해하니 나도 기뻐' 같은 공감 표현 사용. 긍정적 경험을 더 자세히 이야기하도록 격려. 감정을 표현한 것 자체를 칭찬.",
        "negative_support": "부정적 감정 지원 가이드: 먼저 아이의 감정을 인정하고 공감해주세요. '많이 속상했겠구나', '힘들었겠네' 같은 표현 사용. 해결책을 바로 제시하기보다 충분히 들어주기. 안전하고 편안한 분위기 조성.",
        "conversation_tips": "효과적인 대화 팁: 눈높이를 맞추고 대화하기. 열린 질문으로 대화 이어가기. 판단하지 않고 경청하기. 아이의 속도에 맞춰 대화하기.",
        "emotional_development": "감정 발달 이해: 연령별 감정 표현의 차이 이해하기. 감정 어휘를 확장시켜주기. 다양한 감정을 인정하고 수용하기. 감정 조절 방법을 함께 찾아가기."
    }
    if os.path.exists(rag_folder) and os.path.isdir(rag_folder):
        for filename in os.listdir(rag_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(rag_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    key = os.path.splitext(filename)[0]
                    EMOTION_GUIDANCE_DOCS[key] = file.read().strip()
    return EMOTION_GUIDANCE_DOCS

def generate_rag_based_report(history_data, rag_docs):
    # API 키 확인 추가
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        return "API 키 설정 필요"
    
    if not history_data:
        return "아직 오늘 아이의 답변이 없습니다."
    
    positive_count = sum(1 for h in history_data if h["emotion"] == "Positive")
    negative_count = len(history_data) - positive_count
    pos_ratio = (positive_count / len(history_data) * 100) if history_data else 0
    
    relevant_docs = []
    if pos_ratio >= 70:
        relevant_docs.append(rag_docs["positive_reinforcement"])
    if negative_count > 0:
        relevant_docs.append(rag_docs["negative_support"])
    relevant_docs.append(rag_docs["conversation_tips"])
    
    context = "\n\n".join(relevant_docs)
    conversations = "\n".join([f"- {h['timestamp'].split()[1]}: {h['answer']} (감정: {h['emotion']})" for h in history_data[:5]])
    
    prompt = f"""
    다음 가이드라인을 참고하여 부모님을 위한 맞춤형 조언을 작성해주세요:
    [가이드라인]\n{context}
    [오늘의 아이 대화 기록]\n{conversations}
    [감정 분석 결과]\n- 긍정: {positive_count}회 ({pos_ratio:.0f}%)\n- 부정: {negative_count}회
    위 정보를 바탕으로 부모님께 다음 내용을 포함한 조언을 해주세요:
    1. 오늘 아이의 감정 상태 요약
    2. 구체적인 대화 방법 3가지
    3. 주의사항 및 권장사항
    따뜻하고 실용적인 조언으로 작성해주세요.
    """

    
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1000)
        return response.choices[0].message.content.strip()
    except:
        return "리포트 생성에 실패했습니다."

def create_pdf_report(history_data, report_content, font_path):
    # date import 추가 필요
    from datetime import date  # ← 파일 상단에 추가하거나 여기에 추가
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, encoding='UTF-8')
    story = []
    
    try:
        pdfmetrics.registerFont(TTFont('NotoSansCJKkr', font_path))
    except Exception as e:
        st.error(f"폰트 파일을 로드하지 못했습니다: {e}. 'NotoSansKR-Regular' 파일을 경로에 추가하세요.")
        return None
    
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#6366f1'), alignment=TA_CENTER, spaceAfter=30, fontName='NotoSansCJKkr')
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16, textColor=colors.HexColor('#4f46e5'), spaceAfter=12, fontName='NotoSansCJKkr')
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=11, leading=16, fontName='NotoSansCJKkr')
    
    story.append(Paragraph("Child Emotion Report", title_style))
    story.append(Paragraph(f"{date.today().strftime('%Y-%m-%d')}", normal_style))
    story.append(Spacer(1, 0.5*inch))
    
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
        ('FONTNAME', (0, 0), (-1, 0), 'NotoSansCJKkr'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("Conversation History", heading_style))
    for i, item in enumerate(history_data[:10], 1):
        time = item['timestamp'].split()[1]
        emotion = "Positive" if item["emotion"] == "Positive" else "Negative"
        confidence = item['confidence']
        conv_text = f"<b>{i}. Time:</b> {time} | <b>Emotion:</b> {emotion} ({confidence:.1%})<br/><b>Content:</b> {item['answer']}<br/><br/>"
        story.append(Paragraph(conv_text, normal_style))
    
    if report_content:
        story.append(Paragraph("AI Recommendations", heading_style))
        story.append(Paragraph(report_content, normal_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer