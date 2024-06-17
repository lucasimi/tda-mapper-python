import pandas as pd
import plotly.graph_objects as go

import networkx as nx 
import streamlit as st

from tdamapper.plot import MapperLayoutInteractive


MAX_NODES = 1000

MAX_SAMPLES = 1000

SAMPLE_FRAC = 0.1

OPENML_URL = 'https://www.openml.org/search?type=data&sort=runs&status=active'

DATA_INFO = 'Non-numeric and NaN features get dropped. NaN rows get replaced by mean'

GIT_REPO_URL = 'https://github.com/lucasimi/tda-mapper-python'

REPORT_BUG = f'{GIT_REPO_URL}/issues'

ABOUT = f'{GIT_REPO_URL}/blob/main/README.md'

DESCRIPTION = f'''
    This app leverages the *Mapper Algorithm* from Topological Data Analysis 
    (TDA) to provide an efficient and intuitive way to gain insights from your
    datasets.

    More details on **[GitHub]({GIT_REPO_URL})**.
    '''

FOOTER = f'''
    If you find this app useful, please consider leaving a :star: on **[GitHub]({GIT_REPO_URL})**.
    '''

ICON_URL = f'{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-icon.png'

LOGO_URL = f'{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png'

APP_TITLE = 'TDA Mapper App'

# V_* are reusable values for widgets

V_LENS_IDENTITY = 'Identity'

V_LENS_PCA = 'PCA'

V_COVER_TRIVIAL = 'Trivial'

V_COVER_BALL = 'Ball'

V_COVER_CUBICAL = 'Cubical'

V_CLUSTERING_TRIVIAL = 'Trivial'

V_CLUSTERING_AGGLOMERATIVE = 'Agglomerative'

V_DATA_SUMMARY_FEAT = 'feature'

V_DATA_SUMMARY_HIST = 'histogram'

V_DATA_SUMMARY_COLOR = 'color'

V_DATA_SUMMARY_BINS = 5

# VD_* are reusable default values for widgets

VD_SEED = 42

VD_DIM = 3

# S_* are reusable manually managed stored objects

S_RESULTS = 'stored_results'


class Results:

    def __init__(self):
        self.df_X = pd.DataFrame()
        self.df_y = pd.DataFrame()
        self.df_X_sample = pd.DataFrame()
        self.df_y_sample = pd.DataFrame()
        self.df_all = pd.DataFrame()
        self.X = self.df_X.to_numpy()
        self.df_summary = pd.DataFrame()
        self.mapper_graph = nx.Graph()
        self.mapper_plot = None
        self.mapper_fig = go.Figure()
        self.auto_rendering = None

    def set_df(self, X, y):
        self.df_X = fix_data(X)
        self.df_y = fix_data(y)
        self.df_X_sample = get_sample(self.df_X)
        self.df_y_sample = get_sample(self.df_y)
        self.df_all = pd.concat([self.df_y, self.df_X], axis=1)
        self.X = self.df_X.to_numpy()
        self.df_summary = get_data_summary(self.df_X, self.df_y)
        self.mapper_graph = nx.Graph()
        self.mapper_plot = None
        self.mapper_fig = go.Figure()
        self._init_fig()
        self.auto_rendering = None

    def _init_fig(self):
        fig = go.Figure(
            data=[go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode='markers'
            )])
        fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                ticks='outside'
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                ticks='outside'
            ),
            zaxis=dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                ticks='outside'
            )),
        )

        self.mapper_fig = fig


    def set_mapper(self, mapper_graph):
        self.mapper_graph = mapper_graph
        self.mapper_plot = MapperLayoutInteractive(
            self.mapper_graph,
            dim=VD_DIM,
            height=450,
            width=450,
            colors=self.X,
            seed=VD_SEED)
        self.mapper_fig = go.Figure()
        nodes_num = mapper_graph.number_of_nodes()
        if nodes_num <= MAX_NODES:
            self.auto_rendering = True
        else:
            self.auto_rendering = False

    def set_mapper_fig(self, mapper_fig):
        self.mapper_fig = mapper_fig
        self.auto_rendering = None


def get_data_summary(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_hist = pd.DataFrame({x: df[x].value_counts(bins=V_DATA_SUMMARY_BINS, sort=False).values for x in df.columns}).T
    df_summary = pd.DataFrame({
        V_DATA_SUMMARY_FEAT: df.columns,
        V_DATA_SUMMARY_HIST: df_hist.values.tolist()})
    df_summary[V_DATA_SUMMARY_COLOR] = False
    return df_summary


@st.cache_data
def get_sample(df: pd.DataFrame, frac=SAMPLE_FRAC, max_n=MAX_SAMPLES, rand=42):
    if frac * len(df) > max_n:
        return df.sample(n=max_n, random_state=rand)
    return df.sample(frac=frac, random_state=rand)


def fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include='number')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def set_page_config():
    st.set_page_config(
        layout='wide',
        page_icon=ICON_URL,
        page_title=APP_TITLE,
        menu_items={
            'Report a bug': REPORT_BUG,
            'About': ABOUT})


def get_data_caption(df_X, df_y):
    if df_X.empty:
        return 'No data source found'
    if df_y.empty:
        return f'{len(df_X)} instances, {len(df_X.columns)} features'
    return f'''{len(df_X)} instances,
        {len(df_X.columns)} + {len(df_y.columns)} features'''


def set_sidebar_headings():
    #with st.sidebar:
    st.image(LOGO_URL, width=150)
    #st.markdown('#')
    #st.markdown(DESCRIPTION)
    #st.markdown('#')


def initialize():
    set_page_config()
    #set_sidebar_headings()
    if S_RESULTS not in st.session_state:
        st.session_state[S_RESULTS] = Results()