import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from PIL import Image

# Load the model
model = load_model('insurance_model')

# Prediction function
def predict(model, input_df):
    prediction_df = predict_model(estimator=model, data=input_df)
    prediction = prediction_df['Label'][0]  # Adjusted based on PyCaret's output column for prediction
    return prediction

# Main function to run the Streamlit app
def run():
    # Load images
    image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    # Display images
    st.image(image, use_column_width=False)
    st.sidebar.image(image_hospital)

    # Sidebar options
    st.sidebar.title("Insurance Charges Prediction App")
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.success('https://www.pycaret.org')

    # App title
    st.title("Insurance Charges Prediction App")

    # Online prediction
    if add_selectbox == "Online":
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=25)
        children = st.selectbox('Children', [0, 1, 2, 3, 4, 5])
        smoker = st.selectbox('Smoker', ['yes', 'no'])
        region = st.selectbox('Region', ['southeast', 'southwest', 'northeast', 'northwest'])

        # Dictionary for input values
        input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])

        # Predict and display result
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            st.success(f'The predicted insurance charge is ₹{output}')

    # Batch prediction
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

# Run the app
if __name__ == '__main__':
    run()
