"""
Data preprocessing pipeline module.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from modules.text_processor import (
    preprocess_text_column, vectorize_text_tfidf, vectorize_text_count
)
from modules.utils import detect_identifier_columns

# Try to import SMOTE, but make it optional
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from modules.utils import detect_column_types



class PreprocessingPipeline:
    """Class to manage the preprocessing pipeline."""
    
    def __init__(self):
        self.steps = []
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.label_encoder = None
        self.feature_names = []
        self.target_name = None
    
    def add_step(self, step_name, action, params, affected_columns):
        """Add a preprocessing step to the pipeline."""
        # Convert affected_columns to string for consistent display
        if isinstance(affected_columns, list):
            affected_cols_str = ', '.join(map(str, affected_columns)) if affected_columns else 'None'
        else:
            affected_cols_str = str(affected_columns)
        
        self.steps.append({
            'step': step_name,
            'action': action,
            'params': params,
            'affected_columns': affected_cols_str
        })
    
    def get_summary(self):
        """Get a summary of all preprocessing steps."""
        return pd.DataFrame(self.steps)


def impute_missing_values(df, column_types, strategy_num='median', strategy_cat='mode', knn_neighbors=5):
    """
    Impute missing values in the dataset.
    
    Args:
        df: DataFrame
        column_types: Dict with column type information
        strategy_num: Strategy for numerical columns ('mean', 'median', 'mode', 'knn')
        strategy_cat: Strategy for categorical columns ('mode', 'constant', 'missing')
        knn_neighbors: Number of neighbors for KNN imputation
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    # Numerical columns
    num_cols = [col for col in column_types.get('numerical', []) if col in df_imputed.columns]
    num_cols_with_missing = [col for col in num_cols if df_imputed[col].isnull().any()]
    
    if num_cols_with_missing:
        if strategy_num == 'knn':
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df_imputed[num_cols_with_missing] = imputer.fit_transform(df_imputed[num_cols_with_missing])
        elif strategy_num in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy_num)
            df_imputed[num_cols_with_missing] = imputer.fit_transform(df_imputed[num_cols_with_missing])
        elif strategy_num == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[num_cols_with_missing] = imputer.fit_transform(df_imputed[num_cols_with_missing])
    
    # Categorical columns
    cat_cols = [col for col in column_types.get('categorical', []) if col in df_imputed.columns]
    cat_cols_with_missing = [col for col in cat_cols if df_imputed[col].isnull().any()]
    
    if cat_cols_with_missing:
        if strategy_cat == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[cat_cols_with_missing] = imputer.fit_transform(df_imputed[cat_cols_with_missing])
        elif strategy_cat == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
            df_imputed[cat_cols_with_missing] = imputer.fit_transform(df_imputed[cat_cols_with_missing])
        elif strategy_cat == 'missing':
            for col in cat_cols_with_missing:
                df_imputed[col] = df_imputed[col].fillna('Missing')
    
    return df_imputed


def handle_outliers(df, column_types, method='cap', lower_pct=5, upper_pct=95):
    """
    Handle outliers in numerical columns.
    
    Args:
        df: DataFrame
        column_types: Dict with column type information
        method: 'remove', 'cap', 'log', 'none'
        lower_pct: Lower percentile for capping
        upper_pct: Upper percentile for capping
    
    Returns:
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    
    num_cols = [col for col in column_types.get('numerical', []) if col in df_clean.columns]
    
    if method == 'none':
        return df_clean
    
    for col in num_cols:
        if method == 'cap':
            lower = df_clean[col].quantile(lower_pct / 100)
            upper = df_clean[col].quantile(upper_pct / 100)
            df_clean[col] = df_clean[col].clip(lower, upper)
        
        elif method == 'remove':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        
        elif method == 'log':
            # Only apply log to positive values
            if (df_clean[col] > 0).all():
                df_clean[col] = np.log1p(df_clean[col])
    
    return df_clean


def scale_features(df, column_types, method='standard', target_col=None):
    """
    Scale numerical features.
    
    Args:
        df: DataFrame
        column_types: Dict with column type information
        method: 'standard', 'minmax', 'robust', 'none'
        target_col: Target column to exclude from scaling
    
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_scaled = df.copy()
    
    num_cols = [col for col in column_types.get('numerical', []) if col in df_scaled.columns]
    
    # Exclude target column from scaling
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    
    if method == 'none' or not num_cols:
        return df_scaled, None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return df_scaled, None
    
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    
    return df_scaled, scaler


def encode_categorical(df, column_types, method='onehot', target_col=None):
    """
    Encode categorical features.
    
    Args:
        df: DataFrame
        column_types: Dict with column type information
        method: 'onehot', 'label', 'target', 'frequency'
        target_col: Target column to exclude from encoding
    
    Returns:
        Tuple of (encoded DataFrame, encoder dict, dropped columns list)
    """
    df_encoded = df.copy()
    
    cat_cols = [col for col in column_types.get('categorical', []) if col in df_encoded.columns]
    
    # Exclude target column from encoding
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    
    encoders = {}
    dropped_cols = []
    
    if not cat_cols:
        return df_encoded, encoders, dropped_cols
    
    # Drop high-cardinality categorical columns (likely identifiers like names, IDs)
    # These have >50 unique values or >80% unique ratio
    for col in cat_cols[:]:  # Use slice to iterate over copy
        n_unique = df_encoded[col].nunique()
        unique_ratio = n_unique / len(df_encoded)
        
        if n_unique > 50 or unique_ratio > 0.8:
            st.info(f"Dropping '{col}' (high cardinality: {n_unique} unique values, {unique_ratio*100:.1f}% unique). Likely an identifier.")
            df_encoded = df_encoded.drop(columns=[col])
            cat_cols.remove(col)
            dropped_cols.append(col)
    
    if not cat_cols:
        return df_encoded, encoders, dropped_cols
    
    if method == 'onehot':
        # Check if one-hot encoding would create too many columns
        total_categories = sum(df_encoded[col].nunique() for col in cat_cols)
        
        if total_categories > 100:
            st.warning(f"One-hot encoding would create {total_categories} columns. Consider using target encoding for high cardinality columns.")
        
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True, dtype=int)
    
    elif method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    elif method == 'frequency':
        for col in cat_cols:
            freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
            df_encoded[col] = df_encoded[col].map(freq_map)
            encoders[col] = freq_map
    
    elif method == 'target' and target_col:
        for col in cat_cols:
            target_mean = df_encoded.groupby(col)[target_col].mean()
            df_encoded[col] = df_encoded[col].map(target_mean)
            encoders[col] = target_mean.to_dict()
    
    return df_encoded, encoders, dropped_cols


def split_data(df, target_col, test_size=0.2, stratify=True, random_state=42):
    """
    Split data into training and test sets.
    
    Args:
        df: DataFrame
        target_col: Target column name
        test_size: Proportion of data for test set
        stratify: Whether to stratify the split
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if stratify else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=stratify_param, 
            random_state=random_state
        )
    except ValueError:
        # If stratification fails (e.g., class has only 1 sample), split without stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        st.warning("Stratified split failed. Using random split instead.")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE for handling class imbalance."""
    if not SMOTE_AVAILABLE:
        st.warning("SMOTE is not available (imbalanced-learn not installed or incompatible). Using original data.")
        return X_train, y_train, False
    
    try:
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled, True
    except Exception as e:
        st.warning(f"SMOTE failed: {str(e)}. Using original data.")
        return X_train, y_train, False


def encode_target(y_train, y_test):
    """Encode target variable if it's categorical."""
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        return y_train_encoded, y_test_encoded, le
    return y_train, y_test, None


def preprocess_text_features(df, column_types, target_col=None, method='tfidf', 
                             max_features=3000, remove_stopwords_flag=True):
    """
    Preprocess text columns by cleaning and vectorizing.
    
    Args:
        df: DataFrame
        column_types: Dict with column type information
        target_col: Target column to exclude
        method: 'tfidf' or 'count' for vectorization
        max_features: Maximum number of features to extract
        remove_stopwords_flag: Whether to remove stopwords
    
    Returns:
        Tuple of (processed DataFrame, text vectorizers dict, processed column types)
    """
    text_cols = column_types.get('text', [])
    
    # Remove target from text cols if present
    if target_col and target_col in text_cols:
        text_cols = [col for col in text_cols if col != target_col]
    
    if not text_cols:
        return df, {}, column_types
    
    df_processed = df.copy()
    text_vectorizers = {}
    
    for col in text_cols:
        st.info(f"Processing text column: {col}")
        
        # Clean and preprocess text
        cleaned_text = preprocess_text_column(
            df_processed[col],
            remove_stopwords_flag=remove_stopwords_flag,
            use_stemming=False,
            use_lemmatization=False,
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=False
        )
        
        # Store the cleaned text (will be vectorized during train/test split)
        df_processed[f'{col}_cleaned'] = cleaned_text
        
        # Drop the original text column to prevent it from being treated as categorical
        df_processed = df_processed.drop(columns=[col])
        
        # Mark original text column for removal later
        text_vectorizers[col] = {
            'method': method,
            'max_features': max_features,
            'cleaned_col': f'{col}_cleaned'
        }
    
    # Update column types - remove text columns from original types
    # and also ensure they're not in categorical
    new_column_types = column_types.copy()
    new_column_types['text'] = []
    
    # Remove text columns from categorical if they were added there
    if 'categorical' in new_column_types:
        new_column_types['categorical'] = [
            col for col in new_column_types['categorical'] 
            if col not in text_cols
        ]
    
    return df_processed, text_vectorizers, new_column_types


def vectorize_text_for_split(X_train, X_test, text_vectorizers):
    """
    Vectorize text columns after train/test split.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        text_vectorizers: Dict with text column info
    
    Returns:
        Tuple of (X_train with text vectors, X_test with text vectors, fitted vectorizers)
    """
    if not text_vectorizers:
        return X_train, X_test, {}
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    fitted_vectorizers = {}
    
    for original_col, config in text_vectorizers.items():
        cleaned_col = config['cleaned_col']
        method = config['method']
        max_features = config['max_features']
        
        if cleaned_col not in X_train_copy.columns:
            continue
        
        # Check if column has sufficient non-empty text
        train_non_empty = X_train_copy[cleaned_col].fillna('').str.strip().astype(bool).sum()
        if train_non_empty < 5:  # Skip if less than 5 non-empty entries
            st.warning(f"Skipping '{original_col}' - insufficient text data ({train_non_empty} entries)")
            X_train_copy = X_train_copy.drop(columns=[cleaned_col], errors='ignore')
            X_test_copy = X_test_copy.drop(columns=[cleaned_col], errors='ignore')
            continue
        
        # Adjust min_df based on data size (more lenient for small datasets)
        adaptive_min_df = max(1, min(2, train_non_empty // 100))
        
        try:
            # Vectorize with adaptive parameters
            if method == 'tfidf':
                train_vectors, test_vectors, vectorizer, feature_names = vectorize_text_tfidf(
                    X_train_copy[cleaned_col],
                    X_test_copy[cleaned_col],
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=adaptive_min_df,
                    max_df=0.95
                )
            else:  # count
                train_vectors, test_vectors, vectorizer, feature_names = vectorize_text_count(
                    X_train_copy[cleaned_col],
                    X_test_copy[cleaned_col],
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=adaptive_min_df,
                    max_df=0.95
                )
        except ValueError as e:
            st.warning(f"Could not vectorize '{original_col}': {str(e)}. Skipping this column.")
            X_train_copy = X_train_copy.drop(columns=[cleaned_col], errors='ignore')
            X_test_copy = X_test_copy.drop(columns=[cleaned_col], errors='ignore')
            continue
        
        # Add vector columns with prefix
        for feat in feature_names:
            safe_feat = f"{original_col}_{feat}".replace(' ', '_').replace('-', '_')
            X_train_copy[safe_feat] = train_vectors[feat].values
            X_test_copy[safe_feat] = test_vectors[feat].values
        
        # Remove cleaned text column
        X_train_copy = X_train_copy.drop(columns=[cleaned_col])
        X_test_copy = X_test_copy.drop(columns=[cleaned_col])
        
        # Also remove original text column if it exists
        if original_col in X_train_copy.columns:
            X_train_copy = X_train_copy.drop(columns=[original_col])
            X_test_copy = X_test_copy.drop(columns=[original_col])
        
        fitted_vectorizers[original_col] = vectorizer
        
        st.success(f"Vectorized '{original_col}' into {len(feature_names)} features using {method.upper()}")
    
    return X_train_copy, X_test_copy, fitted_vectorizers


def render_preprocessing_page():
    """Render the preprocessing page."""
    st.header("Data Preprocessing")
    
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state['data']
    target_col = st.session_state.get('target_column', None)
    column_types = st.session_state.get('column_types', detect_column_types(df))
    mode = st.session_state.get('mode', 'Beginner')
    
    if not target_col:
        st.warning("Please select a target variable in the Upload Dataset page first!")
        return
    
    st.info(f"Dataset: {len(df)} rows × {len(df.columns)} columns | Target: {target_col}")
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline()
    
    if mode == "Beginner":
        st.markdown("### Beginner Mode - Automatic Preprocessing")
        st.markdown("We'll apply smart defaults for preprocessing. Click the button below to proceed.")
        
        if st.button("Apply Automatic Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                progress = st.progress(0)
                status = st.empty()
                
                # Step 0: Encode target variable first (label encode)
                status.text("Step 0/7: Encoding target variable...")
                df_processed = df.copy()
                
                # Label encode target if it's categorical/text
                if df_processed[target_col].dtype == 'object' or df_processed[target_col].dtype.name == 'category':
                    target_le = LabelEncoder()
                    df_processed[target_col] = target_le.fit_transform(df_processed[target_col].astype(str))
                    st.session_state['target_encoder_early'] = target_le
                    pipeline.add_step("Target Encoding", "Label Encoding", {}, target_col)
                    st.success(f"Target variable '{target_col}' label encoded.")
                progress.progress(5)
                
                # ═══════════════════════════════════════════════════════════
                # Step 0.5: EARLY-STAGE COLUMN REMOVAL (Before any processing)
                # ═══════════════════════════════════════════════════════════
                status.text("Step 0.5/7: Detecting and removing identifier columns...")
                
                cols_to_drop = []
                drop_reasons = {}
                
                # 1. Detect identifier columns (user_id, name, email, etc.)
                identifier_detection = detect_identifier_columns(
                    df_processed, 
                    target_col=target_col,
                    text_columns=column_types.get('text', [])
                )
                
                identifier_cols = identifier_detection['identifier_cols']
                if identifier_cols:
                    for col in identifier_cols:
                        cols_to_drop.append(col)
                        drop_reasons[col] = f"IDENTIFIER ({identifier_detection['metadata'][col]['details']})"
                    st.info(f"🔍 Detected {len(identifier_cols)} identifier columns: {', '.join(identifier_cols)}")
                
                # 2. Check for user-marked columns (from issue resolution)
                issue_resolutions = st.session_state.get('issue_resolutions', {})
                for issue_id, action in issue_resolutions.items():
                    if action.startswith('drop_'):
                        col_name = action.replace('drop_', '')
                        if col_name in df_processed.columns and col_name != target_col and col_name not in cols_to_drop:
                            cols_to_drop.append(col_name)
                            drop_reasons[col_name] = "User-selected in issue resolution"
                
                # 3. Check for >50% missing values
                for col in df_processed.columns:
                    if col != target_col and col not in cols_to_drop:
                        missing_pct = df_processed[col].isnull().sum() / len(df_processed) * 100
                        if missing_pct > 50:
                            cols_to_drop.append(col)
                            drop_reasons[col] = f">50% missing ({missing_pct:.1f}%)"
                
                # Execute dropping
                if cols_to_drop:
                    df_processed = df_processed.drop(columns=cols_to_drop)
                    
                    # Update column types after dropping
                    for col in cols_to_drop:
                        for col_type in column_types:
                            if col in column_types[col_type]:
                                column_types[col_type].remove(col)
                    
                    # Store metadata for reproducibility
                    st.session_state['dropped_columns_metadata'] = {
                        'columns': cols_to_drop,
                        'reasons': drop_reasons,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    # Add to pipeline
                    pipeline.add_step("Remove Identifier & Bad Columns", 
                                    "Identifiers, user-selected, >50% missing", 
                                    {}, ', '.join(cols_to_drop))
                    
                    # Display summary
                    st.success(f"✅ Removed {len(cols_to_drop)} columns")
                    with st.expander("View dropped columns details"):
                        for col in cols_to_drop:
                            st.write(f"• **{col}**: {drop_reasons[col]}")
                
                progress.progress(8)
                
                # Step 1: Handle text columns if present
                text_vectorizers = {}
                if column_types.get('text', []):
                    status.text("Step 1/7: Preprocessing text columns...")
                    df_processed, text_vectorizers, column_types = preprocess_text_features(
                        df_processed, column_types, target_col, method='tfidf', max_features=3000
                    )
                    if text_vectorizers:
                        text_col_names = ', '.join(text_vectorizers.keys())
                        pipeline.add_step("Text Preprocessing", "TF-IDF Vectorization", 
                                        {"max_features": 3000}, text_col_names)
                progress.progress(12)
                
                # Step 2: Impute missing values
                status.text("Step 2/7: Imputing missing values...")
                df_processed = impute_missing_values(df_processed, column_types, 'median', 'mode')
                pipeline.add_step("Missing Value Imputation", "Median (num) / Mode (cat)", {}, "All columns")
                progress.progress(28)
                
                # Step 3: Handle outliers
                status.text("Step 3/7: Handling outliers...")
                df_processed = handle_outliers(df_processed, column_types, 'cap', 5, 95)
                pipeline.add_step("Outlier Handling", "Cap at 5th/95th percentile", {"lower": 5, "upper": 95}, column_types.get('numerical', []))
                progress.progress(44)
                
                # Step 4: Encode categorical (excluding target)
                status.text("Step 4/7: Encoding categorical features...")
                df_processed, encoders, dropped_cols = encode_categorical(df_processed, column_types, 'onehot', target_col)
                cat_cols_encoded = [col for col in column_types.get('categorical', []) if col != target_col and col not in dropped_cols]
                if dropped_cols:
                    pipeline.add_step("Drop High Cardinality Columns", "Removed identifiers", {}, ', '.join(dropped_cols))
                pipeline.add_step("Categorical Encoding", "One-Hot Encoding", {}, ', '.join(cat_cols_encoded) if cat_cols_encoded else 'None')
                progress.progress(60)
                
                # Step 5: Scale features
                status.text("Step 5/7: Scaling features...")
                df_processed, scaler = scale_features(df_processed, detect_column_types(df_processed), 'standard', target_col)
                pipeline.add_step("Feature Scaling", "StandardScaler", {}, "Numerical columns")
                progress.progress(72)
                
                # Step 6: Split data
                status.text("Step 6/7: Splitting data...")
                X_train, X_test, y_train, y_test = split_data(df_processed, target_col, 0.2, True, 42)
                progress.progress(80)
                
                # Step 7: Vectorize text if present and remove text columns
                if text_vectorizers:
                    status.text("Step 7/7: Vectorizing text features...")
                    X_train, X_test, fitted_vectorizers = vectorize_text_for_split(
                        X_train, X_test, text_vectorizers
                    )
                    st.session_state['text_vectorizers'] = fitted_vectorizers
                    
                    # Remove any remaining text/object columns
                    X_train = X_train.select_dtypes(exclude=['object'])
                    X_test = X_test.select_dtypes(exclude=['object'])
                progress.progress(90)
                
                # Target is already encoded, no need to encode again
                status.text("Finalizing...")
                target_encoder = st.session_state.get('target_encoder_early', None)
                
                # Apply SMOTE if needed
                if st.session_state.get('apply_smote', False):
                    X_train, y_train, smote_success = apply_smote(X_train, y_train)
                    if smote_success:
                        pipeline.add_step("SMOTE", "Oversampling minority class", {}, target_col)
                
                progress.progress(100)
                status.text("Preprocessing complete!")
                
                # Store results
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['preprocessing_pipeline'] = pipeline
                st.session_state['target_encoder'] = target_encoder
                st.session_state['feature_scaler'] = scaler
                st.session_state['preprocessing_done'] = True
                
                # Display summary
                st.success("Preprocessing complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Test Samples", len(X_test))
                with col3:
                    st.metric("Features", X_train.shape[1])
                with col4:
                    n_classes = len(np.unique(y_train))
                    st.metric("Classes", n_classes)
    
    else:  # Expert mode
        st.markdown("### Expert Mode - Custom Preprocessing")
        
        # Text preprocessing (if text columns exist)
        text_method = 'tfidf'
        text_max_features = 100
        text_remove_stopwords = True
        
        if column_types.get('text', []):
            with st.expander("0 Text Preprocessing", expanded=True):
                st.info(f"Text columns detected: {', '.join(column_types['text'])}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    text_method = st.selectbox(
                        "Vectorization method:",
                        ["tfidf", "count"],
                        help="TF-IDF weighs words by importance, Count is raw frequency"
                    )
                with col2:
                    text_max_features = st.number_input(
                        "Max features:", 
                        min_value=10, 
                        max_value=3000, 
                        value=500,
                        help="Maximum number of text features to extract"
                    )
                with col3:
                    text_remove_stopwords = st.checkbox(
                        "Remove stopwords",
                        value=True,
                        help="Remove common words like 'the', 'is', 'and'"
                    )
        
        # Missing value imputation
        with st.expander("1 Missing Value Imputation", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                strategy_num = st.selectbox(
                    "Strategy for numerical columns:",
                    ["median", "mean", "mode", "knn"],
                    help="Median is robust to outliers"
                )
            with col2:
                strategy_cat = st.selectbox(
                    "Strategy for categorical columns:",
                    ["mode", "constant", "missing"],
                    help="Mode uses the most frequent value"
                )
        
        # Outlier handling
        with st.expander("2 Outlier Handling", expanded=True):
            outlier_method = st.selectbox(
                "Outlier handling method:",
                ["cap", "remove", "log", "none"],
                help="Cap: clip values at percentiles, Remove: delete outlier rows"
            )
            
            if outlier_method == "cap":
                col1, col2 = st.columns(2)
                with col1:
                    lower_pct = st.slider("Lower percentile", 1, 25, 5)
                with col2:
                    upper_pct = st.slider("Upper percentile", 75, 99, 95)
            else:
                lower_pct, upper_pct = 5, 95
        
        # Categorical encoding
        with st.expander("3⃣ Categorical Encoding", expanded=True):
            encoding_method = st.selectbox(
                "Encoding method:",
                ["onehot", "label", "frequency", "target"],
                help="One-hot: creates binary columns, Label: assigns integers"
            )
        
        # Feature scaling
        with st.expander("4 Feature Scaling", expanded=True):
            scaling_method = st.selectbox(
                "Scaling method:",
                ["standard", "minmax", "robust", "none"],
                help="Standard: mean=0, std=1. MinMax: scale to 0-1"
            )
        
        # Train-test split
        with st.expander("5 Train-Test Split", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
            with col2:
                stratify = st.checkbox("Stratified split", value=True)
            with col3:
                random_state = st.number_input("Random state", 0, 999, 42)
        
        # SMOTE option
        apply_smote_option = st.checkbox(
            "Apply SMOTE (oversample minority class)",
            value=st.session_state.get('apply_smote', False)
        )
        
        # Apply preprocessing
        if st.button("Apply Custom Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Preprocessing data..."):
                progress = st.progress(0)
                
                # ═══════════════════════════════════════════════════════════
                # Step 0: EARLY-STAGE IDENTIFIER DETECTION & REMOVAL
                # ═══════════════════════════════════════════════════════════
                
                df_processed = df.copy()
                cols_to_drop = []
                drop_reasons = {}
                
                # Detect identifier columns
                identifier_detection = detect_identifier_columns(
                    df_processed,
                    target_col=target_col,
                    text_columns=column_types.get('text', [])
                )
                
                identifier_cols = identifier_detection['identifier_cols']
                if identifier_cols:
                    for col in identifier_cols:
                        cols_to_drop.append(col)
                        drop_reasons[col] = f"IDENTIFIER ({identifier_detection['metadata'][col]['details']})"
                    st.info(f"🔍 Detected {len(identifier_cols)} identifier columns: {', '.join(identifier_cols)}")
                
                # Execute dropping
                if cols_to_drop:
                    df_processed = df_processed.drop(columns=cols_to_drop)
                    
                    # Update column types
                    for col in cols_to_drop:
                        for col_type in column_types:
                            if col in column_types[col_type]:
                                column_types[col_type].remove(col)
                    
                    # Store metadata
                    st.session_state['dropped_columns_metadata'] = {
                        'columns': cols_to_drop,
                        'reasons': drop_reasons,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    pipeline.add_step("Remove Identifier Columns", 
                                    "Conservative detection", 
                                    {}, ', '.join(cols_to_drop))
                    
                    st.success(f"✅ Removed {len(identifier_cols)} identifier columns")
                
                progress.progress(5)
                
                # Step 1: Text preprocessing
                text_vectorizers = {}
                if column_types.get('text', []):
                    df_processed, text_vectorizers, column_types = preprocess_text_features(
                        df_processed, column_types, target_col, method=text_method, 
                        max_features=text_max_features,
                        remove_stopwords_flag=text_remove_stopwords
                    )
                    pipeline.add_step("Text Preprocessing", f"{text_method.upper()} Vectorization", 
                                    {"max_features": text_max_features}, column_types.get('text', []))
                progress.progress(20)
                
                # Step 2: Impute
                df_processed = impute_missing_values(df_processed, column_types, strategy_num, strategy_cat)
                pipeline.add_step("Missing Value Imputation", f"{strategy_num} (num) / {strategy_cat} (cat)", {}, "All columns")
                progress.progress(35)
                
                # Step 3: Outliers
                df_processed = handle_outliers(df_processed, column_types, outlier_method, lower_pct, upper_pct)
                pipeline.add_step("Outlier Handling", outlier_method, {"lower": lower_pct, "upper": upper_pct}, column_types.get('numerical', []))
                progress.progress(50)
                
                # Step 4: Encode
                df_processed, encoders, dropped_cols = encode_categorical(df_processed, column_types, encoding_method, target_col)
                cat_cols_encoded = [col for col in column_types.get('categorical', []) if col != target_col and col not in dropped_cols]
                if dropped_cols:
                    pipeline.add_step("Drop High Cardinality Columns", "Removed identifiers", {}, ', '.join(dropped_cols))
                pipeline.add_step("Categorical Encoding", encoding_method, {}, ', '.join(cat_cols_encoded) if cat_cols_encoded else 'None')
                progress.progress(65)
                
                # Step 5: Scale
                df_processed, scaler = scale_features(df_processed, detect_column_types(df_processed), scaling_method, target_col)
                pipeline.add_step("Feature Scaling", scaling_method, {}, "Numerical columns")
                progress.progress(80)
                
                # Step 6: Split
                X_train, X_test, y_train, y_test = split_data(df_processed, target_col, test_size, stratify, random_state)
                progress.progress(90)
                
                # Step 5b: Vectorize text if present
                if text_vectorizers:
                    X_train, X_test, fitted_vectorizers = vectorize_text_for_split(
                        X_train, X_test, text_vectorizers
                    )
                    st.session_state['text_vectorizers'] = fitted_vectorizers
                progress.progress(90)
                
                # Encode target
                y_train, y_test, target_encoder = encode_target(y_train, y_test)
                
                # SMOTE
                if apply_smote_option:
                    X_train, y_train, smote_success = apply_smote(X_train, y_train, random_state)
                    if smote_success:
                        pipeline.add_step("SMOTE", "Oversampling", {}, target_col)
                
                progress.progress(100)
                
                # Store
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['preprocessing_pipeline'] = pipeline
                st.session_state['target_encoder'] = target_encoder
                st.session_state['feature_scaler'] = scaler
                st.session_state['preprocessing_done'] = True
                
                st.success("Preprocessing complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Test Samples", len(X_test))
                with col3:
                    st.metric("Features", X_train.shape[1])
                with col4:
                    n_classes = len(np.unique(y_train))
                    st.metric("Classes", n_classes)
    
    # Show preprocessing summary if done
    if st.session_state.get('preprocessing_done', False):
        st.markdown("---")
        st.subheader("Preprocessing Summary")
        
        pipeline = st.session_state.get('preprocessing_pipeline')
        if pipeline:
            st.dataframe(pipeline.get_summary(), use_container_width=True, hide_index=True)
        
        # Class distribution in train set
        y_train = st.session_state.get('y_train')
        if y_train is not None:
            st.markdown("#### Training Set Class Distribution")
            unique, counts = np.unique(y_train, return_counts=True)
            dist_df = pd.DataFrame({'Class': unique, 'Count': counts, 'Percentage': (counts / len(y_train) * 100).round(2)})
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        st.success("Data is ready for model training. Proceed to Feature Engineering or Model Training.")
        
        # Continue button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Feature Engineering", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'features'
                st.rerun()
