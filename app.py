import streamlit as st
import pandas as pd
st.header('AutoML Project',divider="blue")


# Initialize session state for the button click if it doesn't exist
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Function to set the session state when the button is clicked
def click_button():
    st.session_state.button_clicked = True

# Display the button
st.button("Upload Dataset", on_click=click_button)

# Conditionally display the file uploader based on the session state
if st.session_state.button_clicked:
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"],accept_multiple_files=False)

    if uploaded_file is not None:
        st.write("You selected the file:", uploaded_file.name)
   
        df= pd.read_csv(uploaded_file)
        
        st.write(df.head())
        st.text('Shape of the Dataset')
        st.write(df.shape)
        st.write(df.isnull().sum())
