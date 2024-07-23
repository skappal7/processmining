import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.statistics.start_activities.log import get as start_activities
from pm4py.statistics.end_activities.log import get as end_activities
from pm4py.algo.filtering.log.variants import variants_filter
from datetime import datetime
import io

# Set page config
st.set_page_config(page_title="Process Mining App", layout="wide")

# Sidebar
st.sidebar.title("Process Mining App")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Convert to event log
    event_log = log_converter.apply(df)

    # Main content
    st.title("Process Mining Dashboard")

    # 1. Process Map Visualization
    st.header("1. Process Map Visualization")
    try:
        dfg = dfg_discovery.apply(event_log)
        gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.FREQUENCY)
        st.graphviz_chart(gviz)
    except Exception as e:
        st.error(f"An error occurred while generating the process map: {str(e)}")
        st.write("Displaying a simplified process flow instead.")
        activities = df['concept:name'].unique()
        G = nx.DiGraph()
        G.add_nodes_from(activities)
        for i in range(len(df) - 1):
            if df.iloc[i]['case:concept:name'] == df.iloc[i+1]['case:concept:name']:
                G.add_edge(df.iloc[i]['concept:name'], df.iloc[i+1]['concept:name'])
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, arrows=True)
        st.pyplot(fig)

    # 2. Variant Analysis
    st.header("2. Variant Analysis")
    try:
        variants_count = variants_filter.get_variants(event_log)
        variants_df = pd.DataFrame([(k, v) for k, v in variants_count.items()], columns=['Variant', 'Count'])
        variants_df['Variant'] = variants_df['Variant'].apply(lambda x: '->'.join(x))
        variants_df = variants_df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        fig = px.bar(variants_df.head(10), x='Variant', y='Count', title='Top 10 Variants')
        st.plotly_chart(fig)
        
        st.write(variants_df)
    except Exception as e:
        st.error(f"An error occurred during variant analysis: {str(e)}")
        st.write("Unable to perform variant analysis.")

    # 3. Process Summary
    st.header("3. Process Summary")
    try:
        num_cases = df['case:concept:name'].nunique()
        num_events = len(df)
        num_activities = df['concept:name'].nunique()
        avg_case_duration = df.groupby('case:concept:name').apply(lambda x: (x['time:timestamp'].max() - x['time:timestamp'].min()).total_seconds() / 86400).mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Number of Cases", num_cases)
        col2.metric("Number of Events", num_events)
        col3.metric("Number of Activities", num_activities)
        col4.metric("Avg Case Duration (days)", f"{avg_case_duration:.2f}")
        
        start_acts = start_activities.get_start_activities(event_log)
        end_acts = end_activities.get_end_activities(event_log)
        
        st.write("Start Activities:", start_acts)
        st.write("End Activities:", end_acts)
    except Exception as e:
        st.error(f"An error occurred while calculating process summary: {str(e)}")
        st.write("Unable to calculate detailed process summary.")

    # 4. Performance Analysis
    st.header("4. Performance Analysis")
    try:
        case_durations = df.groupby('case:concept:name').apply(lambda x: (x['time:timestamp'].max() - x['time:timestamp'].min()).total_seconds() / 86400)
        
        fig = px.histogram(case_durations, nbins=20, labels={'value': 'Case Duration (days)'}, title='Case Duration Distribution')
        st.plotly_chart(fig)
        
        activity_durations = df.groupby('concept:name').apply(lambda x: (x['time:timestamp'].max() - x['time:timestamp'].min()).total_seconds() / 86400)
        
        fig = px.bar(x=activity_durations.index, y=activity_durations.values, 
                     labels={'x': 'Activity', 'y': 'Average Duration (days)'}, title='Average Activity Duration')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred during performance analysis: {str(e)}")
        st.write("Unable to perform detailed performance analysis.")

    # 5. Social Network Analysis
    st.header("5. Social Network Analysis")
    if 'org:resource' in df.columns:
        try:
            handover_df = df.sort_values(['case:concept:name', 'time:timestamp'])
            handover_df['next_resource'] = handover_df.groupby('case:concept:name')['org:resource'].shift(-1)
            handover_df = handover_df[handover_df['org:resource'] != handover_df['next_resource']]
            handover_count = handover_df.groupby(['org:resource', 'next_resource']).size().reset_index(name='count')
            
            G = nx.from_pandas_edgelist(handover_count, 'org:resource', 'next_resource', 'count')
            pos = nx.spring_layout(G)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    )
                )
            )

            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_text.append(f'{list(G.nodes())[node]}: {len(adjacencies[1])} connections')

            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Social Network Analysis',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )

            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred during social network analysis: {str(e)}")
            st.write("Unable to perform social network analysis.")
    else:
        st.write("Resource column not found. Social Network Analysis is not available.")

    # 6. Time-based Analysis
    st.header("6. Time-based Analysis")
    try:
        df['date'] = df['time:timestamp'].dt.date
        daily_cases = df.groupby('date')['case:concept:name'].nunique().reset_index()
        daily_cases.columns = ['date', 'num_cases']
        
        fig = px.line(daily_cases, x='date', y='num_cases', title='Daily Number of Cases')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred during time-based analysis: {str(e)}")
        st.write("Unable to perform time-based analysis.")
    
    # 7. Filtering and Drill-down Capabilities
    st.header("7. Filtering and Drill-down Capabilities")
    
    st.write("Select a date range to filter the data:")
    try:
        date_range = st.date_input("Date range", value=(df['time:timestamp'].min().date(), df['time:timestamp'].max().date()))
        
        filtered_df = df[(df['time:timestamp'].dt.date >= date_range[0]) & (df['time:timestamp'].dt.date <= date_range[1])]
        
        st.write(f"Number of cases in selected date range: {filtered_df['case:concept:name'].nunique()}")
        
        selected_case = st.selectbox("Select a case for detailed view:", filtered_df['case:concept:name'].unique())
        
        case_details = filtered_df[filtered_df['case:concept:name'] == selected_case].sort_values('time:timestamp')
        st.write(case_details)
    except Exception as e:
        st.error(f"An error occurred during filtering and drill-down: {str(e)}")
        st.write("Unable to perform filtering and drill-down analysis.")

else:
    st.write("Please upload a CSV file to begin the analysis.")

# Add download button for the processed data
if 'df' in locals():
    try:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download processed data as CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"An error occurred while preparing the download: {str(e)}")
        st.write("Unable to provide download option for processed data.")
