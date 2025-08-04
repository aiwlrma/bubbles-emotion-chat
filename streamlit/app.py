import streamlit as st
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ui 모듈에서 render_app 함수 import
try:
    from ui import render_app
except ImportError as e:
    st.error(f"UI 모듈을 불러올 수 없습니다: {e}")
    st.stop()

if __name__ == "__main__":
    render_app()
