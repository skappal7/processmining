import streamlit as st
import pandas as pd
import graphviz
import itertools
from io import StringIO
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Process Mining App", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def parse_xes(file):
    tree = ET.parse(file)
    root = tree.getroot()
    data = []
    for trace in root.findall('{http://www.xes-standard.org/}trace'):
        case_id = trace.find('{http://www.xes-standard.org/}string[@key="concept:name"]').get('value')
        for event in trace.findall('{http://www.xes-standard.org/}event'):
            event_data = {'case:concept:name': case_id}
            for attr in event:
                if attr.get('key') in ['concept:name', 'time:timestamp', 'org:resource']:
                    event_data[attr.get('key')] = attr.get('value')
            data.append(event_data)
    return pd.DataFrame(data)

@st.cache_data
def get_dfg(df, case_id_col, activity_col):
    traces = df.groupby(case_id_col)[activity_col].agg(list)
    frequencies = traces.value_counts()
    
    activities = df[activity_col].value_counts().reset_index()
    activities.columns = ['activity', 'frequency']
    
    pairs = traces.apply(lambda x: list(zip(x[:-1], x[1:])))
    pairs_count = pairs.explode().value_counts().reset_index()
    pairs_count.columns = ['pair', 'frequency']
    
    return activities, pairs_count

@st.cache_data
def get_footprint(pairs):
    activities = sorted(set(pairs['pair'].explode()))
    footprint = pd.DataFrame(index=activities, columns=activities, data='#')
    
    for _, row in pairs.iterrows():
        a, b = row['pair']
        footprint.at[a, b] = '→'
        if footprint.at[b, a] == '→':
            footprint.at[a, b] = footprint.at[b, a] = '||'
    
    return footprint

def main():
    st.title("Process Mining Application")
    
    uploaded_file = st.file_uploader("Choose an event log file", type=['csv', 'xes'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = parse_xes(uploaded_file)
        
        st.subheader("Event Log Preview")
        st.dataframe(df.head())
        
        st.subheader("Column Selection")
        columns = df.columns.tolist()
        
        case_id_col = st.selectbox("Select Case ID column", options=columns, index=0 if columns else None)
        activity_col = st.selectbox("Select Activity Name column", options=columns, index=1 if len(columns) > 1 else None)
        timestamp_col = st.selectbox("Select Timestamp column", options=columns, index=2 if len(columns) > 2 else None)
        resource_col = st.selectbox("Select Resource column (optional)", options=['None'] + columns, index=0)
        
        if resource_col == 'None':
            resource_col = None
        
        if case_id_col and activity_col and timestamp_col:
            st.success("Columns selected successfully. You can now proceed with the analysis.")
            
            if st.button("Discover Process Model"):
                activities, pairs = get_dfg(df, case_id_col, activity_col)
                
                st.subheader("Activity Frequencies")
                st.dataframe(activities)
                
                st.subheader("Directly-Follows Graph")
                dot = graphviz.Digraph(comment='Directly-Follows Graph')
                for _, row in activities.iterrows():
                    dot.node(row['activity'], f"{row['activity']} ({row['frequency']})")
                for _, row in pairs.iterrows():
                    dot.edge(row['pair'][0], row['pair'][1], label=str(row['frequency']))
                st.graphviz_chart(dot)
                
                st.subheader("Footprint")
                footprint = get_footprint(pairs)
                st.dataframe(footprint)
                
                # Here you would add your alpha miner algorithm and Petri net visualization
                st.info("Alpha Miner algorithm and Petri Net visualization would be implemented here.")
        else:
            st.warning("""
            Please select the appropriate columns to start the analysis:
            - Case ID: Unique identifier for each process instance
            - Activity Name: The name of the activity performed
            - Timestamp: When the activity was performed
            - Resource (Optional): Who performed the activity
            
            Once you've selected these columns, you can proceed with the process discovery.
            """)

if __name__ == "__main__":
    main()
