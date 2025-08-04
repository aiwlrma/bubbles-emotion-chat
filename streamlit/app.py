import streamlit as st
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    try:
        # Import after path is set
        from ui import render_app
        render_app()
    except ImportError as e:
        st.error(f"모듈을 불러오는 중 오류가 발생했습니다: {e}")
        st.info("필요한 파일들이 모두 있는지 확인해주세요.")
        st.stop()
    except Exception as e:
        st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
