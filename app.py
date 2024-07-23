import streamlit as st
import pandas as pd
import graphviz
import itertools
from io import StringIO
import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime

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

def sanitize_node_name(name):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', str(name))

def alpha_miner(activities, pairs):
    # Step 1: Get all activities
    T = set(activities['activity'])
    
    # Step 2: Find initial and final activities
    Ti = set(pair[0] for pair in pairs['pair'])
    To = set(pair[1] for pair in pairs['pair'])
    start_activities = T - To
    end_activities = T - Ti
    
    # Step 3: Find direct succession, causality, parallel and choice relations
    relations = defaultdict(lambda: defaultdict(str))
    for _, row in pairs.iterrows():
        a, b = row['pair']
        relations[a][b] = '>'
    
    for a in T:
        for b in T:
            if relations[a][b] == '>' and relations[b][a] == '>':
                relations[a][b] = relations[b][a] = '||'
            elif relations[a][b] == '>' and relations[b][a] != '>':
                relations[a][b] = '->'
            elif relations[a][b] != '>' and relations[b][a] == '>':
                relations[a][b] = '<-'
            else:
                relations[a][b] = '#'
    
    # Step 4: Find maximal pairs
    maximal_pairs = set()
    for a_set in powerset(T):
        for b_set in powerset(T):
            a_set, b_set = set(a_set), set(b_set)  # Ensure these are sets
            if a_set and b_set and all(relations[a][b] == '->' for a in a_set for b in b_set):
                if not any(a_set.issubset(other_a) and b_set.issubset(other_b) 
                           for other_a, other_b in maximal_pairs 
                           if (a_set, b_set) != (other_a, other_b)):
                    maximal_pairs.add((frozenset(a_set), frozenset(b_set)))
    
    # Step 5: Create places
    places = [f"p_{'_'.join(sorted(a))}__{'_'.join(sorted(b))}" for a, b in maximal_pairs]
    places = ["start"] + places + ["end"]
    
    # Step 6: Create flow relations
    flow = set()
    for a_set, b_set in maximal_pairs:
        for a in a_set:
            flow.add((a, f"p_{'_'.join(sorted(a_set))}__{'_'.join(sorted(b_set))}"))
        for b in b_set:
            flow.add((f"p_{'_'.join(sorted(a_set))}__{'_'.join(sorted(b_set))}", b))
    
    for start in start_activities:
        flow.add(("start", start))
    for end in end_activities:
        flow.add((end, "end"))
    
    return T, places, flow

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def visualize_petri_net(T, places, flow):
    dot = graphviz.Digraph(comment='Petri Net')
    dot.attr(rankdir='LR')
    
    # Add transitions
    for t in T:
        dot.node(sanitize_node_name(t), t, shape='rect')
    
    # Add places
    for p in places:
        if p == "start":
            dot.node(p, "", shape='circle', style='filled', fillcolor='gray')
        elif p == "end":
            dot.node(p, "", shape='circle', peripheries='2')
        else:
            dot.node(sanitize_node_name(p), "", shape='circle')
    
    # Add flow relations
    for f in flow:
        dot.edge(sanitize_node_name(f[0]), sanitize_node_name(f[1]))
    
    return dot

def variant_analysis(df, case_id_col, activity_col):
    variants = df.groupby(case_id_col)[activity_col].agg(lambda x: ','.join(x))
    variant_counts = variants.value_counts().reset_index()
    variant_counts.columns = ['Variant', 'Count']
    variant_counts['Percentage'] = variant_counts['Count'] / variant_counts['Count'].sum() * 100
    return variant_counts

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
                dot_dfg = graphviz.Digraph(comment='Directly-Follows Graph')
                for _, row in activities.iterrows():
                    sanitized_name = sanitize_node_name(row['activity'])
                    dot_dfg.node(sanitized_name, f"{row['activity']} ({row['frequency']})")
                for _, row in pairs.iterrows():
                    sanitized_start = sanitize_node_name(row['pair'][0])
                    sanitized_end = sanitize_node_name(row['pair'][1])
                    dot_dfg.edge(sanitized_start, sanitized_end, label=str(row['frequency']))
                st.graphviz_chart(dot_dfg)
                
                st.subheader("Footprint")
                footprint = get_footprint(pairs)
                st.dataframe(footprint)
                
                st.subheader("Alpha Miner and Petri Net")
                T, places, flow = alpha_miner(activities, pairs)
                dot_pn = visualize_petri_net(T, places, flow)
                st.graphviz_chart(dot_pn)
                
                st.subheader("Variant Analysis")
                variants = variant_analysis(df, case_id_col, activity_col)
                st.dataframe(variants)
                
                # Visualize top variants
                top_variants = variants.head(5)  # Show top 5 variants
                fig = px.bar(top_variants, x='Variant', y='Percentage', 
                             title='Top 5 Process Variants',
                             labels={'Variant': 'Process Variant', 'Percentage': 'Percentage of Cases'})
                st.plotly_chart(fig)

        else:
            st.warning("""
            Please select the appropriate columns to start the analysis:
            - Case ID: Unique identifier for each process instance
            - Activity Name: The name of the activity performed
            - Timestamp: When the activity was performed
            - Resource (Optional): Who performed the activity
            
            Once you've selected these columns, you can proceed with the process discovery.
            """)

def conformance_checking(df, case_id_col, activity_col, discovered_model):
    # Simple conformance checking based on trace existence
    traces = df.groupby(case_id_col)[activity_col].agg(list)
    conforming_traces = 0
    non_conforming_traces = 0
    
    for trace in traces:
        if check_trace_conformance(trace, discovered_model):
            conforming_traces += 1
        else:
            non_conforming_traces += 1
    
    fitness = conforming_traces / (conforming_traces + non_conforming_traces)
    return fitness, conforming_traces, non_conforming_traces

def check_trace_conformance(trace, model):
    # Check if the trace follows the discovered model
    current_state = "start"
    for activity in trace:
        if (current_state, activity) in model:
            current_state = activity
        else:
            return False
    return current_state in model.get("end", set())

def performance_analysis(df, case_id_col, activity_col, timestamp_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    case_durations = df.groupby(case_id_col).apply(lambda x: (x[timestamp_col].max() - x[timestamp_col].min()).total_seconds() / 3600)
    activity_durations = df.groupby(activity_col).apply(lambda x: x[timestamp_col].diff().mean().total_seconds() / 60)
    return case_durations, activity_durations

def process_enhancement(df, case_id_col, activity_col, timestamp_col):
    # Identify bottlenecks
    activity_durations = df.groupby(activity_col).apply(lambda x: x[timestamp_col].diff().mean().total_seconds() / 60)
    bottlenecks = activity_durations.nlargest(3)
    
    # Identify rework
    activity_frequencies = df.groupby(case_id_col)[activity_col].apply(lambda x: x.value_counts().to_dict())
    rework = activity_frequencies.apply(lambda x: {k: v for k, v in x.items() if v > 1})
    
    return bottlenecks, rework

def interactive_dfg(activities, pairs):
    G = nx.DiGraph()
    for _, row in activities.iterrows():
        G.add_node(row['activity'], weight=row['frequency'])
    for _, row in pairs.iterrows():
        G.add_edge(row['pair'][0], row['pair'][1], weight=row['frequency'])
    
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

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]} - # of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Directly-Follows Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def main():
    # ... (keep existing code up to the "Discover Process Model" button)

    if st.button("Discover Process Model"):
        activities, pairs = get_dfg(df, case_id_col, activity_col)
        
        st.subheader("Activity Frequencies")
        st.dataframe(activities)
        
        st.subheader("Directly-Follows Graph")
        dot_dfg = graphviz.Digraph(comment='Directly-Follows Graph')
        for _, row in activities.iterrows():
            sanitized_name = sanitize_node_name(row['activity'])
            dot_dfg.node(sanitized_name, f"{row['activity']} ({row['frequency']})")
        for _, row in pairs.iterrows():
            sanitized_start = sanitize_node_name(row['pair'][0])
            sanitized_end = sanitize_node_name(row['pair'][1])
            dot_dfg.edge(sanitized_start, sanitized_end, label=str(row['frequency']))
        st.graphviz_chart(dot_dfg)
        
        st.subheader("Interactive Directly-Follows Graph")
        fig_dfg = interactive_dfg(activities, pairs)
        st.plotly_chart(fig_dfg)
        
        st.subheader("Footprint")
        footprint = get_footprint(pairs)
        st.dataframe(footprint)
        
        st.subheader("Alpha Miner and Petri Net")
        T, places, flow = alpha_miner(activities, pairs)
        dot_pn = visualize_petri_net(T, places, flow)
        st.graphviz_chart(dot_pn)
        
        st.subheader("Conformance Checking")
        discovered_model = {(f[0], f[1]) for f in flow}
        fitness, conforming, non_conforming = conformance_checking(df, case_id_col, activity_col, discovered_model)
        st.write(f"Fitness: {fitness:.2f}")
        st.write(f"Conforming traces: {conforming}")
        st.write(f"Non-conforming traces: {non_conforming}")
        
        st.subheader("Performance Analysis")
        case_durations, activity_durations = performance_analysis(df, case_id_col, activity_col, timestamp_col)
        st.write("Case Durations (hours):")
        st.dataframe(case_durations.describe())
        st.write("Activity Durations (minutes):")
        st.dataframe(activity_durations)
        
        st.subheader("Process Enhancement")
        bottlenecks, rework = process_enhancement(df, case_id_col, activity_col, timestamp_col)
        st.write("Top 3 Bottlenecks (Activity, Average Duration in minutes):")
        st.dataframe(bottlenecks)
        st.write("Rework Activities:")
        st.write(rework)
        
        st.subheader("Variant Analysis")
        variants = variant_analysis(df, case_id_col, activity_col)
        st.dataframe(variants)
        
        # Visualize top variants
        top_variants = variants.head(5)  # Show top 5 variants
        fig = px.bar(top_variants, x='Variant', y='Percentage', 
                     title='Top 5 Process Variants',
                     labels={'Variant': 'Process Variant', 'Percentage': 'Percentage of Cases'})
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
