import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import networkx as nx

# Set page config
st.set_page_config(page_title="Advanced Process Mining App", layout="wide")

# Function to load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to correct datetime format
def correct_datetime(df, timestamp_col):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], infer_datetime_format=True)
    return df

# Function to create process map
def create_process_map(df, case_id, activity):
    G = nx.DiGraph()
    edges = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:]))).explode()
    edge_counts = edges.value_counts()

    for (source, target), count in edge_counts.items():
        G.add_edge(source, target, weight=count)

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
        node_text.append(f'{adjacencies[0]}<br># of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Process Map',
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

# Function to create transition matrix
def create_transition_matrix(df, case_id, activity):
    transitions = df.groupby(case_id)[activity].apply(lambda x: list(zip(x, x[1:])))
    transition_counts = transitions.explode().value_counts().unstack(fill_value=0)
    return transition_counts

# Main function
def main():
    st.title("Advanced Process Mining App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("File successfully loaded!")

            # Column selection
            st.sidebar.subheader("Column Selection")
            case_id = st.sidebar.selectbox("Select Case ID column", df.columns)
            activity = st.sidebar.selectbox("Select Activity column", df.columns)
            timestamp = st.sidebar.selectbox("Select Timestamp column", df.columns)

            # Correct datetime format
            df = correct_datetime(df, timestamp)

            # Process Overview
            st.subheader("Process Overview")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Tickets", df[case_id].nunique())
            col2.metric("Activities", df[activity].nunique())
            col3.metric("Events", len(df))
            col4.metric("Users", df['user'].nunique() if 'user' in df.columns else "N/A")
            col5.metric("Products", df['product'].nunique() if 'product' in df.columns else "N/A")
            col6.metric("Variants", df.groupby(case_id)[activity].agg(tuple).nunique())

            # Time range
            st.write(f"Time Range: {df[timestamp].min().date()} to {df[timestamp].max().date()}")

            # Number of cases per month
            st.subheader("Number of cases per month by start date")
            df['month'] = df[timestamp].dt.to_period('M')
            cases_per_month = df.groupby('month')[case_id].nunique().reset_index()
            cases_per_month['month'] = cases_per_month['month'].astype(str)
            fig = px.bar(cases_per_month, x='month', y=case_id, labels={'month': 'Month', case_id: 'Number of Cases'})
            st.plotly_chart(fig, use_container_width=True)

            # Process Map
            st.subheader("Process Map")
            process_map = create_process_map(df, case_id, activity)
            st.plotly_chart(process_map, use_container_width=True)

            # Variant Analysis
            st.subheader("Variant Analysis")
            variants = df.groupby(case_id)[activity].agg(lambda x: tuple(x)).value_counts()
            fig = px.bar(variants.reset_index(), x='index', y='count', labels={'index': 'Variant', 'count': 'Frequency'})
            fig.update_layout(xaxis_title="Variant", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Activity Distribution
            st.subheader("Activity Distribution")
            activity_counts = df[activity].value_counts()
            fig = px.bar(activity_counts.reset_index(), x='index', y=activity, labels={'index': 'Activity', activity: 'Frequency'})
            fig.update_layout(xaxis_title="Activity", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Transition Matrix
            st.subheader("Transition Matrix")
            try:
                transition_matrix = create_transition_matrix(df, case_id, activity)
                fig = px.imshow(transition_matrix, labels=dict(x="Next Activity", y="Current Activity", color="Frequency"), 
                                x=transition_matrix.columns, y=transition_matrix.index)
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as e:
                st.warning(f"Unable to create transition matrix. Error: {str(e)}")
                st.info("This might occur if there are cases with only one activity or if activities aren't properly paired.")

    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()
