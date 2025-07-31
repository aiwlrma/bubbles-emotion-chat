import gettext

# Set up translation
translator = gettext.translation('messages', localedir='locales', languages=['ko'], fallback=True)
t = translator.gettext

# Translation strings
strings = {
    "ko": {
        "app.title": "감정 일기",
        "sidebar.header": "<h3>설정</h3>",
        "sidebar.mode": "모드 선택",
        "sidebar.child_mode": "아이 모드",
        "sidebar.parent_mode": "부모 모드",
        "sidebar.guide": "사용 가이드",
        "sidebar.guide_child": "1. 질문을 보고 답변을 입력하세요.\n2. AI와 대화를 나누며 감정을 표현하세요.",
        "sidebar.guide_parent": "1. 부모 인증 후 아이의 감정 데이터를 확인하세요.\n2. PDF 보고서를 다운로드하세요.",
        "sidebar.footer": "<small>© 2025 xAI</small>",
        "child.header": "<h1>💝 아이 모드</h1>",
        "child.question_box": "오늘의 질문",
        "child.greeting": "안녕! 오늘의 질문은",
        "child.input": "답변을 입력하세요...",
        "child.processing": "처리 중...",
        "child.system_prompt": "너는 아이와 대화를 나누는 친절한 AI야. 감정에 맞춰 따뜻하게 응답해줘.",
        "parent.header": "<h1>👨‍👩‍👧 부모 모드</h1>",
        "parent.no_data": "오늘의 데이터가 없습니다.",
        "parent.metric.total": "총 대화",
        "parent.metric.positive": "긍정",
        "parent.metric.negative": "부정",
        "parent.metric.percent": "{value}%",  # Changed from "{}%" to "{value}%"
        "parent.metric.mood": "기분",
        "parent.metric.confidence": "{value}%",  # Changed from "{}%" to "{value}%"
        "parent.metric.count": "회",
        "parent.mood.good": "좋음",
        "parent.mood.neutral": "보통",
        "parent.mood.bad": "나쁨",
        "parent.chart.trend": "감정 추이",
        "parent.chart.time": "시간",
        "parent.chart.positive": "긍정",
        "parent.chart.neutral": "중립",
        "parent.chart.negative": "부정",
        "parent.chart.mood": "감정 지표",
        "parent.chart.positive_index": "긍정 지수",
        "parent.conversations": "최근 대화",
        "parent.emotion.positive": "긍정",
        "parent.emotion.negative": "부정",
        "parent.guide_button": "AI 가이드 보기",
        "parent.pdf_button": "PDF 다운로드",
        "parent.download": "다운로드",
        "parent.refresh": "새로고침",
        "parent.processing": "처리 중...",
        "parent.advice": "AI 조언",
        "auth.container": "<div style='text-align: center; padding: 1rem;'><h3>부모 인증</h3></div>",
        "auth.code": "인증 코드",
        "auth.submit": "제출",
        "auth.success": "인증 성공!",
        "auth.failure": "인증 실패. 다시 시도하세요.",
        "chat.edit": "수정",
        "chat.delete": "삭제",
        "chat.edit_input": "수정할 내용을 입력하세요",
        "chat.save": "저장",
        "questions.positive_moment": "오늘 가장 기뻤던 순간은?",
        "questions.gratitude": "오늘 고마웠던 일은?",
        "questions.new_learn": "오늘 새로 배운 것은?",
        "questions.friend_talk": "친구와 나눴던 재미있는 이야기는?",
        "questions.color_day": "오늘 하루를 색깔로 표현한다면?",
        "questions.fun_moment": "오늘 가장 재미있었던 일은?",
        "questions.hard_moment": "오늘 힘들었던 순간은?",
        "questions.tomorrow_plan": "내일 하고 싶은 계획은?",
        "parent.metric.positive_short": "긍정",
        "parent.metric.negative_short": "부정"
    }
}

# Custom t function to handle dictionary-based translations
def t(key, **kwargs):
    if key in strings.get('ko', {}):
        return strings['ko'][key].format(**kwargs)
    return translator.gettext(key).format(**kwargs)