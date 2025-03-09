import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'cancer_risk_model.pkl'
model = joblib.load(model_filename)

# Define the app title
st.title("Cancer Risk Prediction App")

# Function to get user inputs
@st.cache_data
def user_inputs():
    # Define the input fields
    input_data = {}

    # Categorical variables with 3 or more unique values
    categorical_columns = {
        'sex': ['Male', 'Female'],
        'composition': ['Solid', 'Mixed', 'Cystic'],
        'echogenicity': ['High', 'Medium', 'Low'],
        'margins': ['Smooth', 'Irregular'],
        'calcifications': ['None', 'Micro', 'Macro'],
        'tirads': ['1', '2', '3', '4', '5']
    }

    for column, options in categorical_columns.items():
        input_data[column] = st.selectbox(f"Select {column}", options)

    # Numerical variables
    numerical_columns = ['age', 'Malignant_percentage']
    for column in numerical_columns:
        input_data[column] = st.number_input(f"Enter {column}", min_value=0.0, step=0.1)

    return pd.DataFrame([input_data])

# Collect user inputs
input_df = user_inputs()

# Encode the inputs as the model expects
encoded_input = pd.get_dummies(input_df, drop_first=True)

# Ensure the input has all required columns
model_columns = joblib.load('model_columns.pkl')  # Save model columns during training
encoded_input = encoded_input.reindex(columns=model_columns, fill_value=0)

# Predict the cancer risk
if st.button("Calculate Cancer Risk"):
    prediction = model.predict(encoded_input)[0]
    st.write(f"The predicted Cancer Risk is: **{prediction:.2f}%**")
