import streamlit as st
from pycaret.classification import setup as classification_setup, compare_models as classification_compare_models, pull as classification_pull, save_model as classification_save_model
from pycaret.regression import setup as regression_setup, compare_models as regression_compare_models, pull as regression_pull, save_model as regression_save_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

# Load dataset if it exists
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1DDKqIu5OZ5tXnze34exCf_O0-nxlDh3IcQ&s")
    st.title("Choose the action:")
    choice = st.radio("Navigation", ["Upload","Data_Profiling","Train_Model", "Download"])
    st.info("This application is designed to upload a clean data and train a model based on it.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Data_Profiling" and 'df' in locals(): 
    st.title("Perform Exploratory Data Analysis")
    profile_df = ProfileReport(df)
    st_profile_report(profile_df)

if choice == "Train_Model" and 'df' in locals(): 
    st.title("Select Task Type and Target Column to Train Model")
    task_type = st.selectbox('Choose the Task Type', ['Classification', 'Regression'])
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    # Encode categorical variables
    df = pd.get_dummies(df)
    
    if st.button('Run Modelling'): 
        if task_type == 'Classification':
            classification_setup(df, target=chosen_target, verbose=False)
            setup_df = classification_pull()
            st.dataframe(setup_df)
            best_model = classification_compare_models()
            compare_df = classification_pull()
            st.dataframe(compare_df)
            classification_save_model(best_model, 'best_model')
        
        elif task_type == 'Regression':
            regression_setup(df, target=chosen_target, verbose=False)
            setup_df = regression_pull()
            st.dataframe(setup_df)
            best_model = regression_compare_models()
            compare_df = regression_pull()
            st.dataframe(compare_df)
            regression_save_model(best_model, 'best_model')

if choice == "Download" and os.path.exists('best_model.pkl'): 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
