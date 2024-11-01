import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from groq import Groq
from openai import OpenAI
import plotly.express as px
from typing import Dict, List
import os

os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'


# Initialize Groq client with API key
API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY)
CLUSTER_RANGE = range(2, 10)

# Data processing functions
def detect_text_columns(dataframe, threshold=0.7, unique_value_limit=10):
    text_columns = []
    for col in dataframe.columns:
        non_numeric_values = dataframe[col].apply(lambda x: isinstance(x, str)).mean()
        unique_values = dataframe[col].nunique()
        if non_numeric_values > threshold and unique_values >= unique_value_limit:
            text_columns.append(col)
    return text_columns

def get_column_sample(column, num_samples=5):
    return column.dropna().sample(n=min(num_samples, len(column))).tolist()

# LLM interaction functions
def classify_column_with_llm(column_name, column_samples):
    sample_text = "\n".join(f"- {sample}" for sample in column_samples)
    prompt = (f"Here are some sample values from the column '{column_name}':\n\n{sample_text}\n\n"
              f"Based on these samples, would you classify this column as containing 'qualitative text', 'categorical data', or 'other'? "
              f"Please provide your classification ONLY, with no other text or commentary.")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with classifying columns based on sample data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

def analyze_candidate_columns_with_llm(dataframe, candidate_columns):
    """Analyze columns to identify true qualitative text columns"""
    qualitative_columns = []
    for col in candidate_columns:
        column_samples = get_column_sample(dataframe[col])
        if not column_samples:
            continue
        
        # More specific prompt for identifying qualitative text
        sample_text = "\n".join(f"- {sample}" for sample in column_samples)
        prompt = (
            f"Here are some sample values from the column '{col}':\n\n{sample_text}\n\n"
            f"Is this qualitative text (like open-ended responses, comments, or detailed feedback) "
            f"or some other type of data (like IDs, categories, or structured data)? "
            f"Answer ONLY with 'qualitative' or 'other'."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at identifying qualitative text data in datasets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10
        )
        
        classification = response.choices[0].message.content.strip().lower()
        #st.write(f"Column '{col}' classified as: {classification}")
        
        if 'qualitative' in classification:
            qualitative_columns.append(col)
    return qualitative_columns

def get_cluster_name(summary, existing_names):
    """
    Get a unique cluster name, taking into account existing names
    """
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert in analytics. You will be given a description of demographic clusters, "
                "and your task is to provide a short, descriptive name for each cluster. "
                f"The following names are already in use and should be avoided: {', '.join(existing_names)}. "
                "Please provide a unique name that clearly distinguishes this group."
            )
        },
        {
            "role": "user", 
            "content": f"Here is a description of a cluster:\n\n{summary}\n\nCan you suggest a concise and suitable name for this group? Just give the name, no other text."
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def ask_cluster_question(cluster_num, question, conversation_history, qual_data_by_cluster, cluster_names):
    """
    Ask a question to a specific cluster, using only that cluster's qualitative responses
    """
    # Get only the qualitative responses for this cluster
    cluster_responses = qual_data_by_cluster.get(cluster_num, [])
    
    if not cluster_responses:
        return "I don't have any qualitative responses from this cluster to base my answer on."
    
    # Join all qualitative responses for this cluster and clean them up
    qual_text = "\n".join(f"Response: {str(response).strip()}" 
                         for response in cluster_responses 
                         if str(response).strip())  # Only include non-empty responses
    
    # Create system message that focuses on the cluster's identity and responses
    system_message = (
        f"You are a representative member of the group '{cluster_names[cluster_num]}'. "
        f"Below are actual responses from your group members. Use these responses to inform your answers:\n\n"
        f"{qual_text}\n\n"
        f"Answer as if you are a member of this group, using only the information from these responses. "
        f"If you can't find relevant information in the responses to answer a question, say so."
    )
    
    # Create messages array with conversation history
    messages = [
        {"role": "system", "content": system_message}
    ] + conversation_history + [
        {"role": "user", "content": question}
    ]
    
    # Get response from LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=600
    )
    
    return response.choices[0].message.content.strip()

# Data analysis functions
def detect_column_types(data):
    """
    Automatically detect and categorize columns into numerical, categorical, and text types.
    Returns dictionaries of column names and their types.
    """
    numerical_columns = []
    categorical_columns = []
    text_columns = []
    
    for column in data.columns:
        # Get number of unique values and total values
        unique_count = data[column].nunique()
        total_count = len(data[column])
        unique_ratio = unique_count / total_count
        
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            if unique_ratio < 0.05:  # If very few unique values, treat as categorical
                categorical_columns.append(column)
            else:
                numerical_columns.append(column)
        else:
            # For non-numeric columns
            if unique_count < 10:  # Threshold for categorical
                categorical_columns.append(column)
            else:
                text_columns.append(column)
    
    return {
        'numerical': numerical_columns,
        'categorical': categorical_columns,
        'text': text_columns
    }

def preprocess_data(data):
    """
    Preprocess data with automatic column type detection and appropriate encoding
    """
    # Detect column types
    column_types = detect_column_types(data)
    
    # Create a copy of the data to avoid modifying the original
    processed_data = data.copy()
    
    # Handle categorical columns
    if column_types['categorical']:
        # Convert categorical columns to string type before encoding
        categorical_data = processed_data[column_types['categorical']].astype(str)
        
        # Updated OneHotEncoder initialization
        onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        categorical_encoded = onehot_encoder.fit_transform(categorical_data)
        categorical_encoded_df = pd.DataFrame(
            categorical_encoded,
            columns=onehot_encoder.get_feature_names_out(column_types['categorical'])
        )
    else:
        categorical_encoded_df = pd.DataFrame(index=data.index)
    
    # Handle numerical columns
    if column_types['numerical']:
        numerical_data = data[column_types['numerical']]
    else:
        numerical_data = pd.DataFrame(index=data.index)
    
    # Combine numerical and encoded categorical data
    combined_data = pd.concat([numerical_data, categorical_encoded_df], axis=1)
    
    # Scale the combined data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    return scaled_data, column_types

def find_optimal_clusters(scaled_data):
    inertia = []
    sil_scores = []
    for k in CLUSTER_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    
    # Plot silhouette scores
    fig = px.line(x=list(CLUSTER_RANGE), y=sil_scores, 
                  title='Silhouette Score vs Number of Clusters',
                  labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'})
    
    return sil_scores.index(max(sil_scores)) + 2

def apply_clustering(data, scaled_data, optimal_clusters):
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
    return data

def calculate_ratios(column):
    """Calculate value ratios while handling mixed types"""
    # Convert all values to strings to ensure consistent comparison
    str_column = column.astype(str)
    return str_column.value_counts(normalize=True) * 100

def generate_cluster_summary(data, cluster_summary, column_types):
    """
    Generate cluster summary with automatically detected categorical columns
    """
    categorical_ratios = {}
    for cluster_num in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster_num]
        # Create a dictionary to store ratios for each categorical column
        ratio_summary = {}
        for col in column_types['categorical']:
            # Convert column values to strings before calculating ratios
            ratio_summary[col] = calculate_ratios(cluster_data[col])
        # Create DataFrame with consistent index type
        categorical_ratios[cluster_num] = pd.DataFrame(ratio_summary)
    return categorical_ratios

def cluster_to_summary(cluster_row, cluster_num, categorical_ratios, column_types):
    """
    Generate cluster summary text with dynamic column handling
    """
    ratio_data = categorical_ratios[cluster_num]
    
    # Generate summaries for each categorical column
    categorical_summaries = []
    for col in column_types['categorical']:
        if col in ratio_data:
            summary = ", ".join([f"{str(val)}: {ratio:.1f}%" 
                               for val, ratio in ratio_data[col].dropna().items()])
            categorical_summaries.append(f"{col} distribution: {summary}")
    
    # Generate summaries for numerical columns
    numerical_summaries = []
    for col in column_types['numerical']:
        if col in cluster_row:
            numerical_summaries.append(f"{col}: {cluster_row[col]:.1f}")
    
    # Combine all summaries
    summary = (
        f"Cluster {cluster_num} characteristics:\n"
        f"Numerical metrics: {'; '.join(numerical_summaries)}.\n"
        f"Categorical distributions:\n"
        f"{'; '.join(categorical_summaries)}."
    )
    
    return summary

# Add this cache decorator to prevent recomputing clusters
@st.cache_data
def process_data_and_cluster(data):
    """Cache the data processing and clustering results"""
    scaled_data, column_types = preprocess_data(data)
    
    # Analyze text columns with LLM to identify true qualitative columns
    if column_types['text']:
        candidate_columns = column_types['text']
        qualitative_columns = analyze_candidate_columns_with_llm(data, candidate_columns)
        # Update text columns to only include confirmed qualitative columns
        column_types['text'] = qualitative_columns
    
    optimal_clusters = find_optimal_clusters(scaled_data)
    clustered_data = apply_clustering(data, scaled_data, optimal_clusters)
    
    # Get numerical summaries
    cluster_summary = clustered_data.groupby('Cluster')[column_types['numerical']].mean()
    categorical_ratios = generate_cluster_summary(clustered_data, cluster_summary, column_types)
    
    # Generate cluster summaries and names
    cluster_summaries = {}
    cluster_names = {}
    existing_names = set()  # Keep track of names we've already used
    
    for cluster_num, cluster_row in cluster_summary.iterrows():
        summary = cluster_to_summary(cluster_row, cluster_num, categorical_ratios, column_types)
        cluster_summaries[cluster_num] = summary
        # Pass existing names to ensure uniqueness
        name = get_cluster_name(summary, list(existing_names))
        cluster_names[cluster_num] = name
        existing_names.add(name)
    
    # Handle qualitative data - combine all qualitative text columns
    qual_data_by_cluster = {}
    if column_types['text']:
        for cluster_num in clustered_data['Cluster'].unique():
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster_num]
            # Combine responses from all qualitative columns
            all_responses = []
            for text_col in column_types['text']:
                responses = cluster_data[text_col].dropna().tolist()
                all_responses.extend([str(r).strip() for r in responses if str(r).strip()])
            if all_responses:  # Only add if there are valid responses
                qual_data_by_cluster[cluster_num] = all_responses
    
    return {
        'clustered_data': clustered_data,
        'column_types': column_types,
        'cluster_names': cluster_names,
        'qual_data_by_cluster': qual_data_by_cluster
    }

# Add this session state initializations at the start of main()
def initialize_session_state():
    if 'conversation_histories' not in st.session_state:
        # Dictionary to store conversation history for each cluster
        st.session_state.conversation_histories = {}
    if 'cluster_data' not in st.session_state:
        st.session_state.cluster_data = None
    if 'column_types' not in st.session_state:
        st.session_state.column_types = None
    if 'cluster_names' not in st.session_state:
        st.session_state.cluster_names = {}
    if 'qual_data_by_cluster' not in st.session_state:
        st.session_state.qual_data_by_cluster = None

# Modify the main function
def main():
    st.title("Cluster Analysis & Insights")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload section at the top
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        
        # Only process data if it hasn't been processed yet
        if st.session_state.cluster_data is None:
            with st.spinner("Processing data..."):
                results = process_data_and_cluster(data)
                st.session_state.cluster_data = results['clustered_data']
                st.session_state.column_types = results['column_types']
                st.session_state.cluster_names = results['cluster_names']
                st.session_state.qual_data_by_cluster = results['qual_data_by_cluster']

        # Create two columns for the main layout
        analysis_col, chat_col = st.columns([1, 1])

        # Analysis Column
        with analysis_col:
            st.subheader("Cluster Analysis Results")
            # Add your analysis visualizations here
            st.write("Cluster Overview:")
            for cluster_num, name in st.session_state.cluster_names.items():
                st.write(f"Cluster {cluster_num}: {name}")

        # Chat Column
        with chat_col:
            st.subheader("Interactive Cluster Insights")

            # Custom CSS for chat interface
            st.markdown("""
                <style>
                    .user-message {
                        background-color: #2e3136;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 5px 0;
                    }
                    .assistant-message {
                        background-color: #36393f;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 5px 0;
                    }
                    .stSelectbox {
                        margin-bottom: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Cluster selection and clear chat in a row
            sel_col, clear_col = st.columns([3, 1])
            with sel_col:
                selected_cluster = st.selectbox(
                    "Select Cluster:",
                    options=list(st.session_state.cluster_names.keys()),
                    format_func=lambda x: f"Cluster {x}: {st.session_state.cluster_names[x]}"
                )
            with clear_col:
                if st.button("Clear Chat", key="clear_chat"):
                    st.session_state.conversation_histories[selected_cluster] = []
                    st.rerun()

            # Initialize conversation history for this cluster if it doesn't exist
            if selected_cluster not in st.session_state.conversation_histories:
                st.session_state.conversation_histories[selected_cluster] = []

            # Chat display container
            chat_placeholder = st.container()
            with chat_placeholder:
                st.markdown(f"<div class='chat-container'>", unsafe_allow_html=True)
                for message in st.session_state.conversation_histories[selected_cluster]:
                    if message["role"] == "user":
                        st.markdown(f"<div class='user-message'>👤 You: {message['content']}</div>", 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div class='assistant-message'>🤖 {st.session_state.cluster_names[selected_cluster]}: "
                            f"{message['content']}</div>", 
                            unsafe_allow_html=True
                        )
                st.markdown("</div>", unsafe_allow_html=True)

            # Input area at the bottom
            with st.container():
                input_col, button_col = st.columns([4, 1])
                with input_col:
                    question = st.text_input("Your question:", key="question_input")
                with button_col:
                    if st.button("Ask", key="ask_button") and question:
                        answer = ask_cluster_question(
                            selected_cluster,
                            question,
                            st.session_state.conversation_histories[selected_cluster],
                            st.session_state.qual_data_by_cluster,
                            st.session_state.cluster_names
                        )
                        
                        st.session_state.conversation_histories[selected_cluster].extend([
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ])
                        st.rerun()

if __name__ == "__main__":
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_LOGGER_LEVEL'] = 'error'
    main()
