import streamlit as st
import pandas as pd
import graphviz
import itertools
import re
from copy import deepcopy
import xml.etree.ElementTree as ET

# Keep all the existing functions from the original code

def main():
    # default settings of the page
    st.set_page_config(page_title="PM-training (Alpha Miner)", page_icon=":rocket:", 
                       layout="wide", initial_sidebar_state="expanded")
    # hide right menu and logo at the bottom 
    hide_streamlit_style = """
                           <style>
                           #MainMenu {visibility: unhidden;}
                           footer {visibility: hidden;}
                           </style>
                           """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)              

    # =============================================================================
    LNG = 'en'                  # interface language
    md_text = get_dict_text()   # dict with markdown texts
    # =============================================================================
    # left panel      
    # =============================================================================
    page = st.sidebar.radio('**Alpha Miner**', 
                            ['Alpha Algorithm']) 
    graph_orientation = st.sidebar.radio('**Graph orientation (Left → Right or Top → Bottom)**',['LR','TB'],index = 0, horizontal = True)
    st.sidebar.markdown('---')
    st.sidebar.markdown(md_text['left_block_author_refs',LNG])                    
    # =============================================================================   
    # central panel
    # =============================================================================   
    st.markdown('##### Process Mining training. Alpha Miner (Bottom-Up Process Discovery)')   

    # New section for file upload
    st.subheader("Upload Event Log")
    uploaded_file = st.file_uploader("Choose a CSV or XES file", type=['csv', 'xes'])

    if uploaded_file is not None:
        df_log = process_uploaded_file(uploaded_file)
        if df_log is not None:
            st.write("Event Log Preview:")
            st.write(df_log.head())
    else:
        # Original code for selecting default event logs
        st.markdown(md_text['common_block',LNG])    
        with st.expander("Select an event log for training", expanded = True):
            st.markdown(md_text['log_list',LNG])
            
            col1_log, col2_log = st.columns(2)
            st_radio_select_log = col1_log.radio('Choose one of the default event logs', 
                                         ('L1','L2','L3','L4','L5','L6','L7','L8'), 
                                           index = 0, horizontal = True)
            col2_log.write(''); col2_log.write('')
            st_check_edit_log = col2_log.checkbox('Modify selected event log',value = False)
            
            if st_check_edit_log:
                selected_log = st.text_input('Edit your event log as a string', value = get_default_event_log(st_radio_select_log))
                st.markdown(md_text['user_log_format_requirements',LNG])
            else:
                selected_log = get_default_event_log(st_radio_select_log)

        # Original code for checking the selected event log
        with st.expander("Check the selected event log in the table format", expanded = True): 
            try:
                df_log = get_df_log(selected_log)
                if len(df_log)==0: raise Exception ('Error! Check your input data')   
                st.write(df_log)   # show DataFrame
            except Exception as ex_msg: st.warning(ex_msg)        

    # Rest of the original code...
    if page == 'Alpha Algorithm':  
        st.markdown('##### Constructing an Accepting Petri-Net based on the simple Event Log using the Alpha Algorithm')  
        with st.form('Applying the Alpha Algorithm'):
            st_submitt_start_alpha_algorithm = st.form_submit_button('Start the Alpha Algorithm')
            if st_submitt_start_alpha_algorithm:
                # Your existing alpha algorithm code here
                # Make sure to use the df_log dataframe

def process_uploaded_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
        return df
    elif file_extension == 'xes':
        return parse_xes(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or XES file.")
        return None

def parse_xes(file):
    tree = ET.parse(file)
    root = tree.getroot()

    data = []
    for trace in root.findall('{http://www.xes-standard.org/}trace'):
        case_id = trace.find('{http://www.xes-standard.org/}string[@key="concept:name"]').get('value')
        for event in trace.findall('{http://www.xes-standard.org/}event'):
            event_data = {'case:concept:name': case_id}
            for attr in event:
                if attr.get('key') in ['concept:name', 'time:timestamp']:
                    event_data[attr.get('key')] = attr.get('value')
            data.append(event_data)

    return pd.DataFrame(data)

# Keep all other functions from the original code

if __name__ == "__main__":
    main()
