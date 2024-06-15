import io

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml, load_digits, load_iris

from common import S_RESULTS, fix_data, V_DATA_SUMMARY_FEAT, V_DATA_SUMMARY_HIST, V_DATA_SUMMARY_COLOR, initialize
from common import set_page_config, set_sidebar_headings


def lp_metric(p):
    return lambda x, y: np.linalg.norm(x - y, ord=p)


@st.cache_data
def cached_load_digits():
    return load_digits(return_X_y=True, as_frame=True)


@st.cache_data
def cached_load_iris():
    return load_iris(return_X_y=True, as_frame=True)


@st.cache_data
def cached_fetch_openml(source):
    return fetch_openml(source, return_X_y=True, as_frame=True)


def get_data_caption(df_X, df_y):
    if df_X.empty:
        return 'No data source found'
    if df_y.empty:
        return f'{len(df_X)} instances, {len(df_X.columns)} features'
    return f'''{len(df_X)} instances,
        {len(df_X.columns)} + {len(df_y.columns)} features'''



def _update_data(data_source):
    X, y = pd.DataFrame(), pd.DataFrame()
    if isinstance(data_source, io.BytesIO):
        X, y = pd.read_csv(data_source), pd.DataFrame()
    elif data_source == 'Digits':
        X, y = cached_load_digits()
    elif data_source == 'Iris':
        X, y = cached_load_iris()
    elif isinstance(data_source, str):
        try:
            X, y = cached_fetch_openml(data_source)
        except ValueError as err:
            st.toast(f'# {err}', icon='🚨')
    df_X, df_y = fix_data(X), fix_data(y)
    st.session_state[S_RESULTS].set_df(df_X, df_y)
    st.toast('Successfully Loaded Data', icon='✅')


def data_section():
    data_source_type = st.selectbox(
        'Source',
        options=['Example', 'OpenML', 'CSV'])
    if data_source_type == 'Example':
        data_source = st.selectbox('Name', options=['Digits', 'Iris'])
    elif data_source_type == 'OpenML':
        data_source = st.text_input('Name', placeholder='Name', help=f'Search on [OpenML]({OPENML_URL})')
    elif data_source_type == 'CSV':
        data_source = st.file_uploader('Upload')
    load_button = st.button(
        '📦 Load',
        use_container_width=True)
    if load_button:
        _update_data(data_source)


def summary_output():
    
    df_summary = st.session_state[S_RESULTS].df_summary
    if not df_summary.empty:
        df_summary = df_summary[[
            V_DATA_SUMMARY_FEAT,
            V_DATA_SUMMARY_HIST
        ]]
    st.data_editor(
        df_summary,
        height=400,
        hide_index=True,
        use_container_width=True,
        column_config={
            V_DATA_SUMMARY_HIST: st.column_config.AreaChartColumn(
                width='large'),
            V_DATA_SUMMARY_FEAT: st.column_config.TextColumn(
                width='small',
                disabled=True),
        })


def data_caption():
    df_X = st.session_state[S_RESULTS].df_X
    df_y = st.session_state[S_RESULTS].df_y
    st.caption(get_data_caption(df_X, df_y))


def data_output():
    df_all = st.session_state[S_RESULTS].df_all
    st.dataframe(
        df_all.head(50),
        use_container_width=True,
        height=400)


def data_download_button():
    df_all = st.session_state[S_RESULTS].df_all
    df_all_data = df_all.to_json()
    st.download_button(
        '📥 Download Cleaned Data',
        disabled=df_all.empty,
        use_container_width=True,
        data=df_all_data)



def main():
    initialize()
    with st.sidebar:
        data_section()
    data_caption()
    col_0, col_1 = st.columns([1, 3])
    with col_0:
        summary_output()
    with col_1:
        data_output()
    data_download_button()


main()