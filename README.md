# bubbles-emotion-chat — AI Emotion Journal Chat

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.9%2B-important)]()  
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange)]()  
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-lightgrey)]()

> **“아이의 감정을 듣고, 부모에게 실용적이고 따뜻한 조언을 전하는”**  
> Streamlit 기반의 RAG + LLM 감정 일기/대시보드 및 챗 플랫폼.

## 개요
`bubbles-emotion-chat`은 아이의 하루 대화를 실시간으로 받고 감정을 다층으로 분석하여, 부모에게 맞춤 조언과 PDF 리포트를 제공하는 시스템이다.  
로컬 감정 분류 모델, 룰 기반 키워드, OpenAI API 결과를 앙상블하고, RAG 컨텍스트를 활용해 부모용 설명을 생성한다. UI/UX는 Streamlit으로 구성되며, 한글 지원 PDF 리포트를 자동으로 만든다.

## 폴더 / 모듈 구조 (현재 기준)
```
streamlit/
│
├── app.py                # 진입점: 세션 초기화 후 UI 렌더링 호출
├── emotion.py            # 감정 분류 모델 로딩, 키워드 룰, OpenAI API 앙상블 (모델 + 감정 분류 로직)
├── rag.py                # 다국어/문자열 리소스 및 번역(현재 strings 정의된 localization 모듈)
├── ui.py                 # 전체 Streamlit 화면 구성: 아이/부모 모드, 대시보드, 입력, 액션 버튼 등
├── i18n.py               # OpenAI 클라이언트, 모델/토크나이저 로딩 및 enhanced_emotion_classification (이름과 역할이 섞여 있으므로 리팩토링 고려)
├── session.py            # 세션 상태 관리, RAG 문서 로딩, 리포트 생성, PDF 제작, 부모 인증 처리
├── style.py              # CSS 및 세션 기본값 셋업 (mode, authentication 등)
├── requirements.txt      # Python 의존성
├── best_model.pt        # 사전 학습/파인튜닝된 감정 분류 체크포인트 (optional)
├── .env                 # 환경 변수 (비공개로 관리)
└── rag/                 # 사용자 정의 RAG 컨텍스트 문서 (.txt)
    └── ... (예: positive_reinforcement.txt 등)
```

## 주요 기능
- **아이 모드**: 오늘의 질문 제시, 사용자의 답변을 받아 감정 분류(키워드 + 로컬 모델 + OpenAI API), 대화 히스토리 유지  
- **부모 모드**: 부모 인증, 당일 감정 통계/추세 시각화, 최근 대화 요약, AI 기반 맞춤 조언 생성, PDF 리포트 다운로드  
- **RAG 기반 맞춤 조언**: `rag/` 폴더 및 기본 내장 가이드 문서를 컨텍스트로 삼아 부모에게 조언 생성  
- **PDF 리포트**: 한글 폰트(Noto Sans KR)를 사용한 감정 요약 리포트 자동 생성  
- **로컬 + API 앙상블 감정분류**: 키워드 룰, Transformers 모델, OpenAI GPT API 결과를 통합하여 최종 감정 판단  
- **다국어/문구 관리**: `rag.py`에서 문자열 템플릿 관리, 확장 가능  

## 빠른 시작

### 1. 클론
```bash
git clone https://github.com/<your-org>/bubbles-emotion-chat.git
cd bubbles-emotion-chat/streamlit
```

### 2. 가상환경 및 설치
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# .\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 환경 변수 설정
`.env` 파일에 아래를 설정한다 (`.env.example`로 보관하고 실제는 커밋하지 말 것):
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
PARENT_CODE=1234
```

### 4. 리소스 준비
- `best_model.pt`: 감정 분류용 로컬 체크포인트 (없으면 OpenAI 기반 fallback)  
- `rag/` 디렉터리에 `.txt` 형태의 가이드 문서 추가 (예: `positive_reinforcement.txt`, `negative_support.txt`)  
- 한글 PDF용 폰트 경로 설정 (`NotoSansKR-Regular.ttf` 등)  

### 5. 실행
```bash
python app.py
```
또는
```bash
streamlit run app.py
```

## 환경 변수 (필수)
| 이름 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI ChatCompletion API 호출용 키 |
| `PARENT_CODE` | 부모 인증을 위한 간단 코드 (UI 보호용) |

## 책임 분리 & 리팩토링 제안
- `app.py`: 앱 시작, 세션 초기화, UI 진입  
- `emotion.py` / `i18n.py`: 감정 분류 + OpenAI API. 현재 역할 혼선이 있으므로 `i18n.py`는 진짜 국제화로, `emotion.py`는 모델/분류로 분리 고려  
- `session.py`: 상태 키/초기화, RAG 문서 로딩, 보고서 작성, PDF 생성, 인증  
- `ui.py`: 화면 흐름, 차트, 대화, 리포트 관련 액션  
- `style.py`: CSS 주입 및 기본 세션 관리  
- `rag/`: 확장 가능한 텍스트 기반 컨텍스트  
- `rag.py`: localization 문자열 관리 (향후 진정한 i18n 확장 가능)

## 배포 & CI/CD 권장
- **브랜치 전략**: `main` (안정), `develop` (통합), `feature/*` (기능)  
- **버전 태깅**: Semantic Versioning (예: `v1.0.0`)  
- **CI 예시**: GitHub Actions에서 lint (`flake8`), 타입 검사 (`mypy`), smoke test 실행  
- **비밀 관리**: `.env`는 절대 커밋하지 말고 CI/CD 시크릿 스토어에 보관  

### GitHub Actions 예시 (`.github/workflows/ci.yml`)
```yaml
name: CI

on: [push, pull_request]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Lint
        run: |
          pip install flake8
          flake8 .
      - name: Type check (optional)
        run: |
          pip install mypy
          mypy .
```

## 보안 & 프라이버시
- `.env`에 민감 정보가 있으므로 **절대 커밋 금지** (`.gitignore`에 포함)  
- 아이의 대화/감정 데이터는 민감 정보이므로 저장/전송 시 암호화 고려  
- OpenAI API 호출 모니터링 및 실패/비정상 트래픽 대응  

## 기여
1. 저장소 Fork  
2. `feature/xxx` 브랜치 생성  
3. 의미 있는 커밋 메시지 작성 (Conventional Commits 권장)  
4. Pull Request 생성  
5. 리뷰 후 병합  

## 커밋 메시지 예시
- `feat: add PDF report download button`  
- `fix: handle missing environment variable gracefully`  
- `refactor: separate localization from core logic`  
- `docs: update README for bubbles-emotion-chat structure`  

## 향후 개선
- 부모 로그인/세션 강화 (JWT, OAuth)  
- 감정 분류 다중 축/개인화 (강도, 복합 감정)  
- 대화 요약 + 하이라이트 자동 생성 (추가 RAG 레이어)  
- 이상 감정 탐지 알림  

## 라이선스
MIT License © 2025 [AID]
