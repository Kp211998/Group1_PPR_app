import streamlit as st  # Importing the Streamlit library
from streamlit_option_menu import option_menu  # Importing a custom module for creating an option menu
from tabs import (upload_graph, store_graph,
                  visualize_graph, analyze_graph, export_graph,
                  system_analysis_function, node_functions, edge_functions)

# Initialize session state variables if not present
if __name__ == '__main__':
    st.set_page_config(layout='wide')

    if "node_list" not in st.session_state:
        st.session_state["node_list"] = []
    if "edge_list" not in st.session_state:
        st.session_state["edge_list"] = []
    if "graph_dict" not in st.session_state:
        st.session_state["graph_dict"] = []
    if "view_graphs" not in st.session_state:
        st.session_state["view_graphs"] = []
    if "tab_index" not in st.session_state:
        st.session_state['tab_index'] = 0


    # Set title of the Streamlit app
    st.title("Graph Analyzer by Group 1")

    # List of tabs for the Streamlit app
    tab_list = [
        "Import Graph",
        "Node Functions",
        "Edge Functions",
        "Store the Graph",
        "Visualize the Graph",
        "Analyze the Graph",
        "System Analysis",
        "Export the Graph"
    ]

    if "Store the Graph" in tab_list and not st.session_state["node_list"]:
        tab_list.remove("Store the Graph")
    if "Visualize the Graph" in tab_list and not st.session_state["node_list"]:
        tab_list.remove("Visualize the Graph")
    if "Analyze the Graph" in tab_list and not st.session_state["node_list"]:
        tab_list.remove("Analyze the Graph")
    if "Export the Graph" in tab_list and not st.session_state["node_list"]:
        tab_list.remove("Export the Graph")
    if "System Analysis" in tab_list and not st.session_state["node_list"]:
        tab_list.remove("System Analysis")
    index = st.session_state["tab_index"]

    # Create a sidebar with an option menu for selecting the main menu
    with st.sidebar:
        selected = option_menu("Main Menu", tab_list,
                               menu_icon="cast",
                               default_index=index,
                               )

    # Handle the selected tab and call the corresponding function
    if selected == "Import Graph":
        upload_graph()  # Function call to upload graph
    elif selected == "Node Functions":
        node_functions()
    elif selected == "Edge Functions":
        edge_functions()
    elif selected == "Store the Graph":
        store_graph()  # Function call to store the graph
    elif selected == "Visualize the Graph":
        visualize_graph()  # Function call to visualize the graph
    elif selected == "Analyze the Graph":
        analyze_graph()  # Function call to analyze the graph
    elif selected == "System Analysis":
        system_analysis_function()  # Function call to analyze the graph
    elif selected == "Export the Graph":
        export_graph()  # Function call to export the graph
