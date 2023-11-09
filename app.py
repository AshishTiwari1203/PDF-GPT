import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title = "PDF's GPT" , page_icon=":books:")
    st.header("Chat With Your PDF's :books:")
    st.text_input("Ask Me Anything..")

    with st.sidebar:
        st.subheader("It reads your Documents")
        st.file_uploader("Import Your PDF's here")
        st.button("Submit")


if __name__ == '__main__':
    main()