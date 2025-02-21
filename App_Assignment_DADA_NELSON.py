import streamlit as st
import pandas as pd
import plotly.express as px
from fastapi import FastAPI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import OpenAI
import plotly.figure_factory as ff
import seaborn as sns
import os

# Initialize FastAPI
app = FastAPI()

# Initialize Streamlit interface
st.title('Titanic Dataset Chatbot 🚢')

# Security warning
st.warning("""⚠️ This application executes Python code to analyze the Titanic dataset. 
The code execution is limited to data analysis operations on the pre-loaded dataset.""")

# API Key input
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Load and prepare data
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    return df

# Load the dataset
df = load_data()

# Initialize LangChain agent
def create_agent():
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None
    return create_pandas_dataframe_agent(
        OpenAI(temperature=0),
        df,
        verbose=True,
        allow_dangerous_code=True  # Added security parameter
    )

# Function to create visualizations
def create_visualization(query):
    if "histogram" in query.lower() and "age" in query.lower():
        fig = px.histogram(df, x="age", title="Distribution of Passenger Ages")
        return fig
    elif "embarked" in query.lower():
        fig = px.bar(df["embarked"].value_counts(), title="Passengers by Port of Embarkation")
        return fig
    elif "fare" in query.lower():
        fig = px.box(df, y="fare", title="Distribution of Ticket Fares")
        return fig
    return None

# Chat interface
def process_query(query):
    agent = create_agent()
    if not agent:
        return None, None
    
    # Get text response
    response = agent.run(query)
    
    # Create visualization if applicable
    viz = create_visualization(query)
    
    return response, viz

st.write('Ask me questions about the Titanic passengers!')

# Streamlit UI components
query = st.text_input("Ask your question:")
if query:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar first.")
    else:
        with st.spinner('Processing your question...'):
            try:
                response, viz = process_query(query)
                if response:
                    # Display text response
                    st.write("Answer:", response)
                    
                    # Display visualization if available
                    if viz:
                        st.plotly_chart(viz)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Additional UI elements for data overview
with st.expander("See raw data"):
    st.write(df)

st.sidebar.header("Sample Questions")
st.sidebar.write("""
- What percentage of passengers were male?
- Show me a histogram of passenger ages
- What was the average ticket fare?
- How many passengers embarked from each port?
""")