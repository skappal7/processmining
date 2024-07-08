import streamlit as st
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.convert import process_tree_to_petri_net
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.organizational_mining.roles import algorithm as roles_discovery
import matplotlib.pyplot as plt
import networkx as nx
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

@st.cache(allow_output_mutation=True)
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
            # Fallback to Inductive Miner
            tree = inductive_miner.apply(log)
            net, initial_marking, final_marking = process_tree_to_petri_net(tree)

        gviz = pn_visualizer.apply(net, initial_marking, final_marking)
        img_bytes = pn_visualizer.save(gviz, "temp.png")
        return img_bytes
    except Exception as e:
        st.warning(f"Graphviz not available. Using alternative visualization method. Error: {str(e)}")
        tree = inductive_miner.apply(log)
        gviz = pt_visualizer.apply(tree)
        img_bytes = pt_visualizer.save(gviz, "temp.png")
        return img_bytes

def analyze_process(log):
    # Variant analysis
    variants_count = case_statistics.get_variant_statistics(log)
    variants_df = pd.DataFrame(variants_count).sort_values("count", ascending=False)

    # Activity frequency
    activities = attributes_filter.get_attribute_values(log, "concept:name")
    activities_df = pd.DataFrame.from_dict(activities, orient='index', columns=['frequency']).sort_values('frequency', ascending=False)

    # Roles discovery
    roles = roles_discovery.apply(log)
    roles_df = pd.DataFrame([(k, v) for k, v in roles.items()], columns=['Resource', 'Activities'])

    return variants_df, activities_df, roles_df

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
                    st.subheader("Process Model")
                    st.image(img_bytes)

            if st.sidebar.button("Analyze Process"):
                with st.spinner("Analyzing process..."):
                    variants_df, activities_df, roles_df = analyze_process(log)

                    st.subheader("Variant Analysis")
                    st.dataframe(variants_df)

                    st.subheader("Activity Frequency")
                    st.bar_chart(activities_df['frequency'])

                    st.subheader("Resource Roles")
                    st.dataframe(roles_df)

            st.sidebar.subheader("Filtering")
            activities = attributes_filter.get_attribute_values(log, "concept:name")
            selected_activities = st.sidebar.multiselect("Select activities to include", list(activities.keys()))

            if selected_activities:
                filtered_log = attributes_filter.apply_events(log, selected_activities)
                st.sidebar.text(f"Filtered log: {len(filtered_log)} traces")

                if st.sidebar.button("Apply Filter"):
                    with st.spinner("Applying filter and updating visualizations..."):
                        img_bytes = visualize_process_model(filtered_log, algorithm)
                        st.subheader("Filtered Process Model")
                        st.image(img_bytes)

                        variants_df, activities_df, roles_df = analyze_process(filtered_log)
                        
                        st.subheader("Filtered Variant Analysis")
                        st.dataframe(variants_df)

                        st.subheader("Filtered Activity Frequency")
                        st.bar_chart(activities_df['frequency'])

                        st.subheader("Filtered Resource Roles")
                        st.dataframe(roles_df)
        else:
            st.warning("Please process the data first by selecting the appropriate columns and clicking 'Process Data'.")

    else:
        st.info("Please upload a CSV or XES file to begin process mining.")

if __name__ == "__main__":
    main()
