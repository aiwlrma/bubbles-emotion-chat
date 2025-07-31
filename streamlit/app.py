import streamlit as st
from ui import render_app
from session import initialize_session

def main():
    initialize_session()
    render_app()

if __name__ == "__main__":
    main()