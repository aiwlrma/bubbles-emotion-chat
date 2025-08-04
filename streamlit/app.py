import streamlit as st
from session import initialize_session
from ui import render_app  # 패키지로 구성했으니 상대가 아니라 일반 import 가능

def main():
    initialize_session()
    render_app()

if __name__ == "__main__":
    main()
