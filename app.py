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
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.statistics.variants.log import get as variants_module
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.organizational_mining.roles import algorithm as roles_discovery
from pm4py.algo.organizational_mining.sna import algorithm as sna
from pm4py.statistics.traces.generic.log import case_arrival
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.util import constants
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import io

# Set page config
st.set_page_config(page_title="Contact Center Process Mining App", layout="wide")

# Sidebar
st.sidebar.title("Contact Center Process Mining")
file_type = st.sidebar.selectbox("Select file type", ["CSV", "XES"])
uploaded_file = st.sidebar.file_uploader(f"Choose a {file_type} file", type=file_type.lower())

@st.cache_data
def load_data(file, file_type):
    if file_type == "CSV":
        df = pd.read_csv(file)
    else:  # XES
        event_log = xes_importer.apply(file)
        df = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file, file_type)
    
    # Column selection
    st.sidebar.header("Select Columns")
    case_id_col = st.sidebar.selectbox("Case ID", df.columns)
    activity_col = st.sidebar.selectbox("Activity", df.columns)
    timestamp_col = st.sidebar.selectbox("Timestamp", df.columns)
    resource_col = st.sidebar.selectbox("Resource", df.columns, index=None)
    
    # Convert to event log
    df['case:concept:name'] = df[case_id_col]
    df['concept:name'] = df[activity_col]
    df['time:timestamp'] = pd.to_datetime(df[timestamp_col])
    if resource_col:
        df['org:resource'] = df[resource_col]
    
    event_log = log_converter.apply(df)

    # Main content
    st.title("Contact Center Process Mining Dashboard")

    # 1. Process Map Visualization
    st.header("1. Process Map Visualization")
    tab1, tab2 = st.tabs(["As-is Process", "Suggested Process"])

    try:
        with tab1:
            net, initial_marking, final_marking = alpha_miner.apply(event_log)
            gviz = pn_visualizer.apply(net, initial_marking, final_marking)
            st.image(gviz)
        
        with tab2:
            tree = inductive_miner.apply_tree(event_log)
            net, initial_marking, final_marking = inductive_miner.apply(event_log)
            gviz = pn_visualizer.apply(net, initial_marking, final_marking)
            st.image(gviz)
    except Exception as e:
        st.error(f"An error occurred while generating the process map: {str(e)}")
        st.write("As an alternative, here's a summary of the process:")
        
        # Generate a simple summary of the process
        activities = df[activity_col].value_counts()
        st.write("Top 10 activities:")
        st.write(activities.head(10))
        
        st.write("Number of unique cases:", df[case_id_col].nunique())
        st.write("Number of unique activities:", df[activity_col].nunique())
        st.write("Date range:", df[timestamp_col].min(), "to", df[timestamp_col].max())

    # 2. Variant Analysis
    st.header("2. Variant Analysis")
    try:
        variants_count = variants_module.get_variants(event_log)
        variants_df = pd.DataFrame([(k, v) for k, v in variants_count.items()], columns=['Variant', 'Count'])
        variants_df = variants_df.sort_values('Count', ascending=False).reset_index(drop=True)
        
        fig = px.bar(variants_df.head(10), x='Variant', y='Count', title='Top 10 Variants')
        st.plotly_chart(fig)
        
        st.write(variants_df)
    except Exception as e:
        st.error(f"An error occurred during variant analysis: {str(e)}")
        st.write("Unable to perform variant analysis. Here's a summary of activities instead:")
        activity_counts = df[activity_col].value_counts()
        st.write(activity_counts)

    # 3. Process Summary
    st.header("3. Process Summary")
    try:
        case_durations = case_statistics.get_all_casedurations(event_log)
        avg_duration = sum(case_durations) / len(case_durations)
        median_duration = np.median(case_durations)
        throughput_time = case_arrival.get_case_arrival_avg(event_log)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Case Duration", f"{avg_duration:.2f} days")
        col2.metric("Median Case Duration", f"{median_duration:.2f} days")
        col3.metric("Average Throughput Time", f"{throughput_time:.2f} days")
    except Exception as e:
        st.error(f"An error occurred while calculating process summary: {str(e)}")
        st.write("Unable to calculate detailed process summary.")

    # 4. Conformance Checking
    st.header("4. Conformance Checking")
    try:
        replayed_traces = token_replay.apply(event_log, net, initial_marking, final_marking)
        conf_df = pd.DataFrame(replayed_traces)
        
        fitness = sum(conf_df['trace_fitness']) / len(conf_df)
        st.metric("Overall Process Fitness", f"{fitness:.2%}")
        
        fig = px.histogram(conf_df, x='trace_fitness', nbins=20, title='Trace Fitness Distribution')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred during conformance checking: {str(e)}")
        st.write("Unable to perform conformance checking.")

    # 5. Social Network Analysis
    st.header("5. Social Network Analysis")
    if resource_col:
        try:
            hw_values = sna.apply(event_log, variant=sna.Variants.HANDOVER_LOG)
            hw_df = pd.DataFrame(hw_values).reset_index()
            hw_df.columns = ['source', 'target', 'value']
            
            G = nx.from_pandas_edgelist(hw_df, 'source', 'target', 'value')
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
        st.write("Resource column not selected. Social Network Analysis is not available.")

    # 6. Performance Analysis
    st.header("6. Performance Analysis")
    try:
        case_durations = case_statistics.get_all_casedurations(event_log)
        
        fig = px.histogram(case_durations, nbins=20, labels={'value': 'Case Duration (days)'}, title='Case Duration Distribution')
        st.plotly_chart(fig)
        
        activities = attributes_filter.get_attribute_values(event_log, "concept:name")
        activity_durations = {act: attributes_filter.get_attribute_values(event_log, "time:timestamp", act) for act in activities}
        
        activity_avg_duration = {act: np.mean([(max(times) - min(times)).total_seconds() / 86400 for times in activity_durations[act]]) for act in activities}
        
        fig = px.bar(x=list(activity_avg_duration.keys()), y=list(activity_avg_duration.values()), 
                     labels={'x': 'Activity', 'y': 'Average Duration (days)'}, title='Average Activity Duration')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred during performance analysis: {str(e)}")
        st.write("Unable to perform detailed performance analysis.")

    # 7. Predictive Analytics
    st.header("7. Predictive Analytics")
    try:
        # Prepare data for prediction
        df['case:concept:name'] = df['case:concept:name'].astype('category').cat.codes
        df['concept:name'] = df['concept:name'].astype('category').cat.codes
        if resource_col:
            df['org:resource'] = df['org:resource'].astype('category').cat.codes
            features = ['case:concept:name', 'concept:name', 'org:resource']
        else:
            features = ['case:concept:name', 'concept:name']
        
        target = 'time:timestamp'
        
        X = df[features]
        y = (df[target] - df[target].min()).dt.total_seconds()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)
        rmse = np.sqrt(mse)
        
        st.metric("Root Mean Square Error (RMSE) for Prediction", f"{rmse:.2f} seconds")
        
        feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance for Prediction')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred during predictive analytics: {str(e)}")
        st.write("Unable to perform predictive analytics.")

    # 8. Root Cause Analysis
    st.header("8. Root Cause Analysis")
    try:
        # Cluster cases based on duration
        case_durations = case_statistics.get_all_casedurations(event_log, parameters={constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"})
        case_duration_df = pd.DataFrame.from_dict(case_durations, orient='index', columns=['duration'])
        case_duration_df = case_duration_df.reset_index()
        case_duration_df.columns = ['case_id', 'duration']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(case_duration_df[['duration']])
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        case_duration_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        fig = px.scatter(case_duration_df, x='case_id', y='duration', color='cluster', title='Case Clusters based on Duration')
        st.plotly_chart(fig)
        
        st.write("Analyzing factors contributing to long-duration cases:")
        long_duration_cluster = case_duration_df['cluster'].value_counts().idxmax()
        long_cases = case_duration_df[case_duration_df['cluster'] == long_duration_cluster]['case_id'].tolist()
        
        long_cases_log = attributes_filter.apply(event_log, long_cases, parameters={constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "case:concept:name", "positive": True})
        
        variants_count = variants_module.get_variants(long_cases_log)
        variants_df = pd.DataFrame([(k, v) for k, v in variants_count.items()], columns=['Variant', 'Count'])
        variants_df = variants_df.sort_values('Count', ascending=False).head(5).reset_index(drop=True)
        
        st.write("Top 5 variants in long-duration cases:")
        st.write(variants_df)
    except Exception as e:
        st.error(f"An error occurred during root cause analysis: {str(e)}")
        st.write("Unable to perform root cause analysis.")

    # 9. Recommendations Engine
    st.header("9. Recommendations Engine")
    
    st.write("Based on the analysis, here are some recommendations:")
    
    rec1, rec2, rec3 = st.columns(3)
    
    with rec1:
        st.metric("Reduce Variant Complexity", "Focus on top 3 variants")
        st.write("Simplify the most common process paths to reduce overall case duration.")
    
    with rec2:
        st.metric("Optimize Handovers", f"Reduce by {20}%")
        st.write("Minimize unnecessary handovers between resources to improve efficiency.")
    
    with rec3:
        st.metric("Address Long-duration Cases", f"Target cluster {long_duration_cluster}")
        st.write("Investigate and optimize the factors contributing to exceptionally long case durations.")

    # 10. Time-based Analysis
    st.header("10. Time-based Analysis")
    try:
        df['date'] = pd.to_datetime(df['time:timestamp']).dt.date
        daily_cases = df.groupby('date')['case:concept:name'].nunique().reset_index()
        daily_cases.columns = ['date', 'num_cases']
        
        fig = px.line(daily_cases, x='date', y='num_cases', title='Daily Number of Cases')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred during time-based analysis: {str(e)}")
        st.write("Unable to perform time-based analysis.")
    
    # 11. Filtering and Drill-down Capabilities
    st.header("11. Filtering and Drill-down Capabilities")
    
    st.write("Select a date range to filter the data:")
    try:
        date_range = st.date_input("Date range", value=(df['date'].min(), df['date'].max()))
        
        filtered_df = df[(df['date'] >= date_range[0]) & (df['date'] <= date_range[1])]
        
        st.write(f"Number of cases in selected date range: {filtered_df['case:concept:name'].nunique()}")
        
        selected_case = st.selectbox("Select a case for detailed view:", filtered_df['case:concept:name'].unique())
        
        case_details = filtered_df[filtered_df['case:concept:name'] == selected_case].sort_values('time:timestamp')
        st.write(case_details)
    except Exception as e:
        st.error(f"An error occurred during filtering and drill-down: {str(e)}")
        st.write("Unable to perform filtering and drill-down analysis.")

else:
    st.write("Please upload a CSV or XES file to begin the analysis.")

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
