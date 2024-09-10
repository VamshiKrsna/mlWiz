import streamlit as st
import pandas as pd
from data_loader import preprocess_data
from data_visualization import plot_data_distribution
from model_builder import identify_problem_type, evaluate_models
from sklearn.model_selection import train_test_split

def load_data(uploaded_file):
    """
    Load data from uploaded file based on file type.
    """
    # Identify file type based on the file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
    elif file_extension == 'tsv':
        df = pd.read_csv(uploaded_file, delimiter='\t')
    else:
        st.error("Unsupported file format!")
        return None
    
    return df

def main():
    st.title("mlWiz - AutoML")
    st.write("Your personal automated Machine Learning companion, Data Scientist")

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV, XLSX, XLS, TSV format)", 
        type=['csv', 'xlsx', 'xls', 'tsv']
    )

    # Navigation options
    options = ["Home", "Data Visualization", "Data Analysis", "Model Building & Evaluation"]
    selected_option = st.sidebar.selectbox("Select an Option", options)

    # If a file is uploaded
    if uploaded_file is not None:
        # Load and preprocess dataset
        df = load_data(uploaded_file)
        
        # Check if the dataframe is loaded successfully
        if df is not None:
            df = preprocess_data(df)
            
            # Display Home Section
            if selected_option == "Home":
                st.subheader("Dataset Preview")
                st.write(df.head())

            # Display Data Visualization Section
            elif selected_option == "Data Visualization":
                st.subheader("Data Visualization")
                target_column = st.selectbox("Select the target column", df.columns)
                if target_column:
                    plot_data_distribution(df, target_column)

            # Display Data Analysis Section
            elif selected_option == "Data Analysis":
                st.subheader("Data Analysis")
                st.write("Basic Data Analysis and Summary Statistics")
                st.write(df.describe())
                st.write("Correlation Matrix")
                st.write(df.corr())

            # Display Model Building & Evaluation Section
            elif selected_option == "Model Building & Evaluation":
                st.subheader("Model Building & Evaluation")
                target_column = st.selectbox("Select the target column", df.columns)
                
                if target_column:
                    problem_type = identify_problem_type(df, target_column)
                    st.write(f"Identified Problem Type: {problem_type}")

                    # Split the data
                    X = df.drop(target_column, axis=1)
                    y = df[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Evaluate models
                    results = evaluate_models(X_train, X_test, y_train, y_test, problem_type)
                    st.write("Model Performance:")
                    st.write(results)

                    # Display best model
                    best_model = (
                        min(results, key=results.get)
                        if problem_type == "Regression"
                        else max(results, key=results.get)
                    )
                    st.write(f"Best Model: {best_model}")

    else:
        st.write("Please upload a CSV, XLSX, XLS, or TSV file from the navbar to get started.")

if __name__ == "__main__":
    main()