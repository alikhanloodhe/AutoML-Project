"""
Dataset upload and validation module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from modules.utils import (
    format_bytes, detect_column_types, get_numerical_summary,
    get_categorical_summary, get_class_distribution, is_imbalanced,
    detect_potential_targets, validate_dataframe
)


def load_file(uploaded_file):
    """
    Load a CSV or XLSX file with multiple encoding attempts.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Tuple of (DataFrame, encoding_used, error_message)
    """
    if uploaded_file is None:
        return None, None, "No file uploaded"
    
    file_size = uploaded_file.size
    max_size = 200 * 1024 * 1024  # 200MB
    
    if file_size > max_size:
        return None, None, f"File size ({format_bytes(file_size)}) exceeds maximum allowed (200MB)"
    
    try:
        if uploaded_file.name.lower().endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
            return df, "xlsx", None
        
        elif uploaded_file.name.lower().endswith('.csv'):
            content = uploaded_file.read()
            encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
            
            for enc in encodings:
                try:
                    text = content.decode(enc)
                    df = pd.read_csv(io.StringIO(text))
                    return df, enc, None
                except Exception:
                    continue
            
            # Final fallback with error replacement
            try:
                text = content.decode('utf-8', errors='replace')
                df = pd.read_csv(io.StringIO(text))
                return df, 'utf-8 (with replacements)', None
            except Exception as e:
                return None, None, f"Failed to parse CSV: {str(e)}"
        
        else:
            return None, None, "Unsupported file format. Please upload CSV or XLSX files."
    
    except Exception as e:
        return None, None, f"Error reading file: {str(e)}"


def display_file_info(uploaded_file, df):
    """Display file information after successful upload."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", format_bytes(uploaded_file.size))
    with col3:
        memory_usage = df.memory_usage(deep=True).sum()
        st.metric("Memory Usage", format_bytes(memory_usage))


def display_dataset_metadata(df):
    """Display comprehensive dataset metadata."""
    st.subheader("Dataset Overview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Values", f"{missing_pct:.2f}%")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicates:,}")
    
    # Column types
    column_types = detect_column_types(df)
    
    st.markdown("---")
    st.subheader("Column Types")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.info(f"Numerical: {len(column_types['numerical'])}")
    with col2:
        st.info(f"Categorical: {len(column_types['categorical'])}")
    with col3:
        st.info(f"Text: {len(column_types.get('text', []))}")
    with col4:
        st.info(f"Datetime: {len(column_types['datetime'])}")
    with col5:
        st.info(f"Boolean: {len(column_types['boolean'])}")
    
    # Column details table
    with st.expander("View Column Details", expanded=False):
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            unique = df[col].nunique()
            
            col_info.append({
                'Column': col,
                'Data Type': dtype,
                'Non-Null Count': non_null,
                'Missing': null_count,
                'Missing %': f"{(null_count/len(df))*100:.2f}%",
                'Unique Values': unique
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True, hide_index=True)
    
    return column_types


def display_data_preview(df):
    """Display first and last rows of the dataset."""
    st.subheader("Data Preview")
    
    tab1, tab2 = st.tabs(["First 10 Rows", "Last 5 Rows"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(df.tail(5), use_container_width=True)


def display_summary_statistics(df, column_types):
    """Display summary statistics for numerical and categorical columns."""
    st.subheader("Summary Statistics")
    
    tab1, tab2 = st.tabs(["Numerical Columns", "Categorical Columns"])
    
    with tab1:
        if column_types['numerical']:
            num_stats = []
            for col in column_types['numerical']:
                summary = get_numerical_summary(df, col)
                if summary:
                    summary['column'] = col
                    num_stats.append(summary)
            
            if num_stats:
                stats_df = pd.DataFrame(num_stats)
                cols_order = ['column', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis', 'unique']
                stats_df = stats_df[cols_order]
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numerical columns found in the dataset.")
    
    with tab2:
        if column_types['categorical']:
            cat_stats = []
            for col in column_types['categorical']:
                summary = get_categorical_summary(df, col)
                if summary:
                    summary['column'] = col
                    # Convert top 5 values to string for display
                    top_5_str = ", ".join([f"{k}: {v}" for k, v in list(summary['top_5_values'].items())[:3]])
                    summary['top_values'] = top_5_str
                    del summary['top_5_values']
                    cat_stats.append(summary)
            
            if cat_stats:
                stats_df = pd.DataFrame(cat_stats)
                cols_order = ['column', 'count', 'unique', 'mode', 'missing', 'missing_pct', 'top_values']
                stats_df = stats_df[cols_order]
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No categorical columns found in the dataset.")


def target_variable_selection(df):
    """UI for selecting target variable."""
    st.subheader("Target Variable Selection")
    
    # Detect potential targets
    potential_targets = detect_potential_targets(df)
    
    if potential_targets:
        st.info(f"Potential target columns detected: {', '.join(potential_targets)}")
    
    # Selection dropdown
    all_columns = df.columns.tolist()
    target_col = st.selectbox(
        "Select the target column for classification:",
        options=[""] + all_columns,
        index=0,
        help="Choose the column you want to predict"
    )
    
    if target_col:
        # Store the selected target column name
        st.session_state['target_column'] = target_col
        
        # Display class distribution
        st.markdown("#### Class Distribution")
        
        dist_df = get_class_distribution(df[target_col])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        with col2:
            import plotly.express as px
            fig = px.bar(
                dist_df, 
                x='Class', 
                y='Count',
                color='Class',
                text='Percentage',
                title='Class Distribution'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Check for imbalance
        imbalanced, minority_class, minority_pct, severity = is_imbalanced(df[target_col])
        
        if severity == 'severe':
            st.error(f"**Severe Class Imbalance Detected!** Class '{minority_class}' represents only {minority_pct:.2f}% of the data.")
            st.info("⚠️ Recommendation: Use SMOTE, class weights, or collect more data for the minority class.")
        elif severity == 'moderate':
            st.warning(f"**Moderate Class Imbalance Detected!** Class '{minority_class}' represents {minority_pct:.2f}% of the data.")
            st.info("💡 Recommendation: Consider using SMOTE or class weights during training.")
        else:
            st.success(f"Class distribution appears balanced. Minority class '{minority_class}': {minority_pct:.2f}%")
        
        # Check for missing values in target
        target_missing = df[target_col].isnull().sum()
        if target_missing > 0:
            st.error(f"Target variable has {target_missing} missing values! These rows will need to be removed.")
        
        return target_col
    
    return None


def render_upload_page():
    """Render the complete upload page."""
    st.header("Upload Dataset")
    
    st.markdown("""
    Upload your dataset to begin the AutoML process. 
    - **Supported formats:** CSV, XLSX
    - **Maximum file size:** 200MB
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing your dataset"
    )
    
    # Sample datasets option
    st.markdown("---")
    st.markdown("**Or use a sample dataset:**")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        if st.button("Iris Dataset", use_container_width=True):
            st.session_state['use_sample'] = 'iris'
    
    with sample_col2:
        if st.button("Titanic Dataset", use_container_width=True):
            st.session_state['use_sample'] = 'titanic'
    
    with sample_col3:
        if st.button("Credit Fraud Dataset", use_container_width=True):
            st.session_state['use_sample'] = 'credit_fraud'
    
    # Load sample dataset if selected
    if 'use_sample' in st.session_state and st.session_state['use_sample']:
        sample_name = st.session_state['use_sample']
        try:
            df = pd.read_csv(f"sample_datasets/{sample_name}.csv")
            st.session_state['data'] = df
            st.session_state['file_name'] = f"{sample_name}.csv"
            st.success(f"Loaded {sample_name} sample dataset!")
            st.session_state['use_sample'] = None
        except Exception as e:
            st.error(f"Error loading sample dataset: {str(e)}")
    
    # Process uploaded file
    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            df, encoding, error = load_file(uploaded_file)
        
        if error:
            st.error(f"{error}")
        else:
            # Validate dataset
            is_valid, errors = validate_dataframe(df)
            
            if not is_valid:
                for err in errors:
                    st.error(f"{err}")
            else:
                st.session_state['data'] = df
                st.session_state['file_name'] = uploaded_file.name
                
                if encoding:
                    st.caption(f"Detected encoding: {encoding}")
                
                st.success("Dataset loaded successfully!")
    
    # Display dataset info if loaded
    if 'data' in st.session_state and st.session_state['data'] is not None:
        df = st.session_state['data']
        
        st.markdown("---")
        
        # File info
        if 'file_name' in st.session_state:
            st.info(f"Current dataset: **{st.session_state['file_name']}**")
        
        # Metadata
        column_types = display_dataset_metadata(df)
        st.session_state['column_types'] = column_types
        
        # Preview
        display_data_preview(df)
        
        # Summary statistics
        display_summary_statistics(df, column_types)
        
        # Target selection
        st.markdown("---")
        target_col = target_variable_selection(df)
        
        if target_col:
            st.success("Target variable selected. Proceed to Exploratory Data Analysis.")
            
            # Continue button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Continue to EDA", type="primary", use_container_width=True):
                    st.session_state['current_page'] = 'eda'
                    st.session_state['scroll_to_top'] = True
                    st.rerun()
