import streamlit as st
from pycaret.classification import (
    setup as classification_setup,
    compare_models as classification_compare_models,
    pull as classification_pull,
    save_model as classification_save_model,
    load_model as classification_load_model,
    predict_model as classification_predict_model
)
from pycaret.regression import (
    setup as regression_setup,
    compare_models as regression_compare_models,
    pull as regression_pull,
    save_model as regression_save_model,
    load_model as regression_load_model,
    predict_model as regression_predict_model
)
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from csv_agent import ask_csv

# Load dataset if it exists

# if "file" not in st.session_state:
#     file = None

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

# Sidebar with navigation and logo
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1DDKqIu5OZ5tXnze34exCf_O0-nxlDh3IcQ&s")
    st.title("Machine Learning App")
    st.markdown("### Choose an action:")
    choice = st.radio("Navigation", ["Upload","Ask CSV", "Data Profiling", "Train Model", "Download", "Predict"])
    st.info("This application is designed to upload clean data and train a model based on it. Both classification and regression are supported.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    st.markdown("### Upload a CSV file containing your dataset:")
    uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if uploaded_file:
        # Save the uploaded file to a temporary location
        with open('uploaded_dataset.csv', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.file_path = 'uploaded_dataset.csv'
        df = pd.read_csv(st.session_state.file_path, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.success("File uploaded successfully!")
        st.dataframe(df)

if choice == "Ask CSV" and 'df' in locals():
    st.title("Ask your dataset")
    if 'file_path' in st.session_state:
        ask_csv(st.session_state.file_path)
        st.dataframe(df)
    else:
        st.warning("Please upload a CSV file first.")

if choice == "Data Profiling" and 'df' in locals():
    st.title("Perform Exploratory Data Analysis")
    st.markdown("### EDA Report:")
    profile_df = ProfileReport(df)
    st_profile_report(profile_df)

if choice == "Train Model" and 'df' in locals():
    st.title("Select Task Type and Target Column to Train Model")
    task_type = st.selectbox('Choose the Task Type', ['Classification', 'Regression'])
    chosen_target = st.selectbox('Choose the Target Column', df.columns)

    # Encode categorical variables
    df = pd.get_dummies(df)
    feature_columns = df.columns.drop(chosen_target)

    if st.button('Run Modelling'):
        if task_type == 'Classification':
            classification_setup(df, target=chosen_target, verbose=False)
            setup_df = classification_pull()
            st.dataframe(setup_df)
            best_model = classification_compare_models()
            compare_df = classification_pull()
            st.dataframe(compare_df)
            classification_save_model(best_model, 'best_model')
            feature_columns.to_series().to_csv('feature_columns.csv', index=False)
            with open('task_type.txt', 'w') as f:
                f.write('Classification')
            st.success("Best classification model saved!")

        elif task_type == 'Regression':
            regression_setup(df, target=chosen_target, verbose=False)
            setup_df = regression_pull()
            st.dataframe(setup_df)
            best_model = regression_compare_models()
            compare_df = regression_pull()
            st.dataframe(compare_df)
            regression_save_model(best_model, 'best_model')
            feature_columns.to_series().to_csv('feature_columns.csv', index=False)
            with open('task_type.txt', 'w') as f:
                f.write('Regression')
            st.success("Best regression model saved!")

if choice == "Download" and os.path.exists('best_model.pkl'):
    st.title("Download the Best Model")
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

if choice == "Predict":
    st.title("Upload Your Test Dataset")
    st.markdown("### Upload a CSV file containing your test dataset:")
    file = st.file_uploader("Upload Your Test Dataset", type=["csv"])
    if file and os.path.exists('best_model.pkl') and os.path.exists('task_type.txt'):
        test_df = pd.read_csv(file, index_col=None)
        with open('task_type.txt', 'r') as f:
            task_type = f.read()
        feature_columns = pd.read_csv('feature_columns.csv').squeeze().tolist()
        test_df = pd.get_dummies(test_df)
        test_df = test_df.reindex(columns=feature_columns, fill_value=0)
        model = classification_load_model('best_model') if task_type == 'Classification' else regression_load_model('best_model')
        predictions = classification_predict_model(model, data=test_df) if task_type == 'Classification' else regression_predict_model(model, data=test_df)
        st.write(predictions)
        predictions.to_csv('predictions.csv', index=False)
        with open('predictions.csv', 'rb') as f:
            st.download_button('Download Predictions', f, file_name="predictions.csv")
