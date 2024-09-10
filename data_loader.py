import pandas as pd
import streamlit as st

def load_data(file):
    """Loads the data from the uploaded file."""
    df = pd.read_csv(file)
    return df

def preprocess_data(df):
    """Perform basic preprocessing tasks like handling missing values, encoding, etc."""
    df = df.dropna() 
    return df