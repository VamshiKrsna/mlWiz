import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_data(file):
    """Load data from an uploaded file."""
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        return pd.read_csv(file)
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file)
    elif file_extension == 'tsv':
        return pd.read_csv(file, delimiter='\t')
    else:
        st.error("Unsupported file format!")
        return None

def plot_data_distribution(df, target_column, selected_features=None, plot_type="univariate"):
    """
    Plots data distribution based on the selected plot type and features.
    """
    if plot_type == "multivariate":
        if selected_features:
            st.write(f"Multivariate Visualization of Selected Features")
            
            # Handle multivariate visualizations
            plt.figure(figsize=(12, 6))
            sns.pairplot(df[selected_features], diag_kind='kde')
            st.pyplot(plt)
        else:
            st.error("Please select features for multivariate visualization.")

    elif plot_type == "bivariate":
        if selected_features and len(selected_features) == 2:
            st.write(f"Bivariate Visualization of {selected_features[0]} vs {selected_features[1]}")
            
            # Handle bivariate visualizations
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]])
            st.pyplot(plt)
        else:
            st.error("Please select exactly two features for bivariate visualization.")

    else:
        # Default heatmap of the correlation matrix
        st.write("Heatmap of the Correlation Matrix")
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)

def main():
    st.title("Data Visualization App")
    st.write("Upload your dataset and visualize data distributions.")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV, XLSX, XLS, TSV format)", 
        type=['csv', 'xlsx', 'xls', 'tsv']
    )

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Data Loaded Successfully")

            # Select the type of visualization
            plot_type = st.radio("Select the type of visualization", ["Univariate", "Bivariate", "Multivariate"])
            
            if plot_type == "Univariate":
                target_column = st.selectbox("Select the target column", df.columns)
                if target_column:
                    st.subheader("Data Distribution")
                    plot_data_distribution(df, target_column)

            elif plot_type == "Bivariate":
                st.subheader("Bivariate Visualization")
                features = st.multiselect("Select exactly two features", df.columns)
                if len(features) == 2:
                    plot_data_distribution(df, None, features, plot_type="bivariate")
                else:
                    st.warning("Please select exactly two features.")

            elif plot_type == "Multivariate":
                st.subheader("Multivariate Visualization")
                features = st.multiselect("Select features for multivariate analysis", df.columns)
                if features:
                    plot_data_distribution(df, None, features, plot_type="multivariate")
                else:
                    st.warning("Please select features for multivariate analysis.")

if __name__ == "__main__":
    main()
