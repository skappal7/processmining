import streamlit as st
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import networkx as nx
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(page_title="Process Mining App", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stSelectbox {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_log(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        return df
    elif file.name.endswith('.xes'):
        log = xes_importer.apply(file)
        return log
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XES file.")

def convert_df_to_event_log(df, case_id, activity, timestamp):
    df = df.rename(columns={case_id: 'case:concept:name', 
                            activity: 'concept:name', 
                            timestamp: 'time:timestamp'})
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    event_log = log_converter.apply(df)
    return event_log

def visualize_process_model(log, algorithm):
    try:
        if algorithm == "Alpha Miner":
            net, initial_marking, final_marking = pm4py.discover_petri_net_alpha(log)
        elif algorithm == "Heuristics Miner":
            net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(log)
        else:
            tree = inductive_miner.apply(log)
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)

        return visualize_using_networkx(net)
    except Exception as e:
        st.warning(f"Error in process discovery. Please try a different algorithm or check your data.")
        return None

def visualize_using_networkx(net):
    G = nx.DiGraph()
    for place in net.places:
        G.add_node(place.name, node_type='place')
    for transition in net.transitions:
        G.add_node(transition.name, node_type='transition')
    for arc in net.arcs:
        G.add_edge(arc.source.name, arc.target.name)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, arrows=True)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def analyze_process(log):
    # Variant analysis
    variants_count = case_statistics.get_variant_statistics(log)
    variants_df = pd.DataFrame(variants_count).sort_values("count", ascending=False)

    # Activity frequency
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    activities_df = pd.DataFrame.from_dict(activities, orient='index', columns=['frequency']).sort_values('frequency', ascending=False)

    return variants_df, activities_df

def main():
    st.title("Process Mining App")

    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload event log (CSV or XES)", type=["csv", "xes"])

    if uploaded_file is not None:
        data = load_log(uploaded_file)

        if isinstance(data, pd.DataFrame):
            st.sidebar.subheader("Column Selection")
            case_id = st.sidebar.selectbox("Select case ID column", data.columns)
            activity = st.sidebar.selectbox("Select activity column", data.columns)
            timestamp = st.sidebar.selectbox("Select timestamp column", data.columns)

            if st.sidebar.button("Process Data"):
                with st.spinner("Processing data..."):
                    try:
                        log = convert_df_to_event_log(data, case_id, activity, timestamp)
                        st.session_state['log'] = log
                        st.success("Data processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.session_state['log'] = None
        else:
            log = data
            st.session_state['log'] = log
            st.success("XES file loaded successfully!")

        if 'log' in st.session_state and st.session_state['log'] is not None:
            log = st.session_state['log']

            st.sidebar.subheader("Process Discovery")
            algorithm = st.sidebar.selectbox("Select mining algorithm", ["Alpha Miner", "Heuristics Miner", "Inductive Miner"])

            if st.sidebar.button("Discover Process Model"):
                with st.spinner("Discovering process model..."):
                    img_bytes = visualize_process_model(log, algorithm)
                    if img_bytes:
                        st.subheader("Process Model")
                        st.image(img_bytes)

            if st.sidebar.button("Analyze Process"):
                with st.spinner("Analyzing process..."):
                    variants_df, activities_df = analyze_process(log)

                    st.subheader("Variant Analysis")
                    st.dataframe(variants_df)

                    st.subheader("Activity Frequency")
                    st.bar_chart(activities_df['frequency'])

            st.sidebar.subheader("Filtering")
            activities = pm4py.get_event_attribute_values(log, "concept:name")
            selected_activities = st.sidebar.multiselect("Select activities to include", list(activities.keys()))

            if selected_activities:
                filtered_log = pm4py.filter_event_attribute_values(log, "concept:name", selected_activities, level="event")
                st.sidebar.text(f"Filtered log: {len(filtered_log)} traces")

                if st.sidebar.button("Apply Filter"):
                    with st.spinner("Applying filter and updating visualizations..."):
                        img_bytes = visualize_process_model(filtered_log, algorithm)
                        if img_bytes:
                            st.subheader("Filtered Process Model")
                            st.image(img_bytes)

                        variants_df, activities_df = analyze_process(filtered_log)
                        
                        st.subheader("Filtered Variant Analysis")
                        st.dataframe(variants_df)

                        st.subheader("Filtered Activity Frequency")
                        st.bar_chart(activities_df['frequency'])

        else:
            st.warning("Please process the data first by selecting the appropriate columns and clicking 'Process Data'.")

    else:
        st.info("Please upload a CSV or XES file to begin process mining.")

if __name__ == "__main__":
    main()
