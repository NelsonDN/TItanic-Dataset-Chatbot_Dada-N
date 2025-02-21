import streamlit as st
import pandas as pd
import plotly.express as px
from fastapi import FastAPI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import OpenAI
import plotly.figure_factory as ff
import seaborn as sns
import os

# Launch FastAPI
app = FastAPI()

# User interface with Streamlit
st.title('Titanic Dataset Chatbot 🚢')

# Small warning to let users know that Python code will be executed
st.warning("""⚠️ This application executes Python code to analyze the Titanic dataset. 
Execution is limited to data analysis operations on the preloaded dataset.""")

# Request OpenAI API key
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Function to load data (cached to avoid reloading on every run)
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')  # Fetch Titanic dataset via Seaborn
    return df

# Load the dataset once
# This prevents reloading the dataset every time the user interacts
df = load_data()

# Create a LangChain agent to answer questions
# It will only be created if the API key is provided
def create_agent():
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None
    return create_pandas_dataframe_agent(
        OpenAI(temperature=0),  # Use OpenAI model with low temperature for more deterministic responses
        df,
        verbose=True,
        allow_dangerous_code=True  # ⚠️ Warning: This allows code execution! Use with caution.
    )

# Generate visualizations based on user queries
def create_visualization(query):
    if "histogram" in query.lower() and "age" in query.lower():
        fig = px.histogram(df, x="age", title="Passenger Age Distribution")
        return fig
    elif "embarked" in query.lower():
        fig = px.bar(df["embarked"].value_counts(), title="Passengers by Port of Embarkation")
        return fig
    elif "fare" in query.lower():
        fig = px.box(df, y="fare", title="Ticket Fare Distribution")
        return fig
    return None

# Handle user queries
def process_query(query):
    agent = create_agent()
    if not agent:
        return None, None
    
    # Get textual response from the agent
    response = agent.run(query)
    
    # Check if a visualization matches the query
    viz = create_visualization(query)
    
    return response, viz

st.write("Ask me questions about Titanic passengers!")

# User interaction interface
query = st.text_input("Ask your question:")
if query:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar first.")
    else:
        with st.spinner('Processing your question...'):
            try:
                response, viz = process_query(query)
                if response:
                    # Display textual response
                    st.write("Answer:", response)
                    
                    # Display chart if relevant
                    if viz:
                        st.plotly_chart(viz)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Bonus: Show raw data if the user wants to see it
with st.expander("View raw data"):
    st.write(df)

# Some sample questions in the sidebar
st.sidebar.header("Sample Questions")
st.sidebar.write("""
- What percentage of passengers were male?
- Show me a histogram of passenger ages.
- What was the average ticket fare?
- How many passengers embarked from each port?
""")
