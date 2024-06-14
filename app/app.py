import streamlit as st

from common import set_page_config, set_sidebar_headings, S_RESULTS, Results



def main():
    set_page_config()
    set_sidebar_headings()
    if S_RESULTS not in st.session_state:
        st.session_state[S_RESULTS] = Results()


main()
