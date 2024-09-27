import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from utils import clean_text
from chains import Chain

def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail Generator")
    url_input = st.text_input("Enter Job URL: ")
    name_input = st.text_input("Enter your name: ")
    role_input = st.text_input("Enter your current role: ")
    job_input = st.text_input("Enter your college/current company name: ")

    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            jobs = llm.extract_jobs(data)
            for job in jobs:
                email = llm.write_mail(job, name_input, role_input, job_input)
                st.code(email, language="markdown")
        except Exception as e:
            st.error(f"An Error as occured: {e}")


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, clean_text)