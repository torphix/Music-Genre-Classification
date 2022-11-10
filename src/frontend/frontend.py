import pathlib
import streamlit as st
from src.frontend.utils.tabs import overview_tab_ui, data_analysis_tab_ui, models_tab_ui
# Constants
ROOT = pathlib.Path(__file__).parent.parent.parent
DATA_URL = f'{ROOT}/data/features_30_sec.csv'

# App UI
def main_ui():
    st.set_page_config(layout="wide")
    st.markdown('''
    <div style="text-align: center; font-size:50px">
        <b>Music Genre Analysis<b/>
    </div>
    ''', unsafe_allow_html=True)

    overview_tab, data_analysis_tab, models_tab = st.tabs(["Overview", "Data Analysis", "Models"])

    with overview_tab:  
        overview_tab_ui()
    
    with data_analysis_tab:
        data_analysis_tab_ui(DATA_URL)
    
    with models_tab:
        models_tab_ui()    
   

if __name__ == '__main__':
    main_ui()
