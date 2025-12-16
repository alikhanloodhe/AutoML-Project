import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

st.header('AutoML Project', divider="blue")

# Mode selection (must be chosen before uploading a dataset)
mode = st.selectbox("Select mode (required before uploading dataset)", ["", "Beginner", "Expert"])

if mode == "":
    st.info("Please select a mode to continue.")
else:
    st.session_state['mode'] = mode
    st.write(f"Mode selected: {mode}")
    if mode == "Expert":
        st.write("Expert mode: advanced options will be available after upload.")

    # Initialize session state for the button click if it doesn't exist
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    # Function to set the session state when the button is clicked
    def click_button():
        st.session_state.button_clicked = True

    # Display the button (only after mode selection)
    st.button("Upload Dataset", on_click=click_button)

    # Conditionally display the file uploader based on the session state
    if st.session_state.button_clicked:
        uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "csv"], accept_multiple_files=False)

        if uploaded_file is not None:
            st.write("You selected the file:", uploaded_file.name)
            try:
                if uploaded_file.name.lower().endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                    used_encoding = None
                else:
                    content = uploaded_file.read()
                    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
                    df = None
                    used_encoding = None
                    for enc in encodings:
                        try:
                            text = content.decode(enc)
                            df = pd.read_csv(io.StringIO(text))
                            used_encoding = enc
                            break
                        except Exception:
                            continue

                    if df is None:
                        # final fallback: decode with replacement to avoid errors
                        try:
                            text = content.decode('utf-8', errors='replace')
                            df = pd.read_csv(io.StringIO(text))
                            used_encoding = 'utf-8 (errors=replace)'
                        except Exception as e:
                            raise e
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df = None

            if df is not None:
                st.text('Shape of the Dataset')
                st.write(df.shape)
                st.text('Nulls in the Dataset')
                st.write(df.isnull().sum())

                # detect columns with more than 50% missing values
                null_frac = df.isnull().mean()
                cols_to_flag = null_frac[null_frac > 0.5].index.tolist()

                if cols_to_flag:
                    st.error(f"Columns with more than 50% nulls detected: {', '.join(cols_to_flag)}")
                    # highlight flagged columns in red for the displayed table
                    try:
                        styled = df.head().style.apply(
                            lambda col: ["background-color: red" if col.name in cols_to_flag else "" for _ in col],
                            axis=0,
                        )
                        st.write(styled)
                    except Exception:
                        # fallback: just show head if styling fails
                        st.write(df.head())
                else:
                    st.write(df.head())

                if used_encoding:
                    st.caption(f"Detected/used encoding: {used_encoding}")

                # Run EDA button (univariate analysis)
                if 'run_eda' not in st.session_state:
                    st.session_state.run_eda = False

                def run_eda():
                    st.session_state.run_eda = True
                st.write('We have to do cleaning first before EDA..!!')
                st.button("Run EDA (Univariate)", on_click=run_eda)

                if st.session_state.run_eda:
                    st.subheader("Univariate Analysis")
                    # determine column types
                    num_cols = df.select_dtypes(include=['number']).columns.tolist()
                    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                    # Numeric: histograms
                    for col in num_cols:
                        st.write(f"Numeric: {col}")
                        fig, ax = plt.subplots()
                        try:
                            ax.hist(df[col].dropna(), bins=30, color='steelblue')
                            ax.set_xlabel(col)
                            ax.set_ylabel('Count')
                            st.pyplot(fig)
                        except Exception as e:
                            st.write(f"Could not plot {col}: {e}")
                        plt.close(fig)

                    # Categorical: bar charts
                    for col in cat_cols:
                        st.write(f"Categorical: {col}")
                        counts = df[col].fillna('<<MISSING>>').value_counts()
                        fig, ax = plt.subplots()
                        try:
                            counts.plot(kind='bar', ax=ax, color='lightcoral')
                            ax.set_ylabel('Count')
                            st.pyplot(fig)
                        except Exception as e:
                            st.write(f"Could not plot {col}: {e}")
                        plt.close(fig)
    
