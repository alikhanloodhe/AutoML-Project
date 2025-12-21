"""
Feature selection and engineering module.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import plotly.express as px


def remove_low_variance_features(X, threshold=0.01):
    """
    Remove features with variance below threshold.
    
    Args:
        X: Feature DataFrame or array
        threshold: Variance threshold
    
    Returns:
        Tuple of (filtered X, removed feature names, selector)
    """
    selector = VarianceThreshold(threshold=threshold)
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_filtered = selector.fit_transform(X)
        
        mask = selector.get_support()
        kept_features = [f for f, m in zip(feature_names, mask) if m]
        removed_features = [f for f, m in zip(feature_names, mask) if not m]
        
        X_filtered = pd.DataFrame(X_filtered, columns=kept_features, index=X.index)
    else:
        X_filtered = selector.fit_transform(X)
        removed_features = []
        kept_features = list(range(X_filtered.shape[1]))
    
    return X_filtered, removed_features, selector


def remove_correlated_features(X, threshold=0.95, target=None):
    """
    Remove highly correlated features, keeping the one with higher target correlation.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold
        target: Target variable for deciding which feature to keep
    
    Returns:
        Tuple of (filtered X, removed feature names)
    """
    if not isinstance(X, pd.DataFrame):
        return X, []
    
    corr_matrix = X.corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = []
    
    for col in upper.columns:
        correlated_cols = upper.index[upper[col] > threshold].tolist()
        
        if correlated_cols:
            for corr_col in correlated_cols:
                if corr_col not in to_drop:
                    # If target is provided, keep the one with higher target correlation
                    if target is not None:
                        corr_with_target_col = abs(X[col].corr(target))
                        corr_with_target_corr = abs(X[corr_col].corr(target))
                        
                        if corr_with_target_col >= corr_with_target_corr:
                            to_drop.append(corr_col)
                        else:
                            to_drop.append(col)
                    else:
                        to_drop.append(corr_col)
    
    to_drop = list(set(to_drop))
    X_filtered = X.drop(columns=to_drop)
    
    return X_filtered, to_drop


def get_feature_importance_rf(X, y, n_estimators=100, random_state=42):
    """
    Get feature importances using Random Forest.
    
    Args:
        X: Feature DataFrame or array
        y: Target variable
        n_estimators: Number of trees
        random_state: Random state
    
    Returns:
        DataFrame with feature importances
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importances, rf


def select_k_best_features(X, y, k=10, score_func='f_classif'):
    """
    Select K best features using statistical tests.
    
    Args:
        X: Feature DataFrame or array
        y: Target variable
        k: Number of features to select (or 'all')
        score_func: Scoring function ('f_classif', 'chi2', 'mutual_info')
    
    Returns:
        Tuple of (filtered X, selected feature names, selector)
    """
    if score_func == 'f_classif':
        scorer = f_classif
    elif score_func == 'chi2':
        # Chi2 requires non-negative values
        scorer = chi2
    else:
        scorer = mutual_info_classif
    
    if k == 'all':
        k = X.shape[1]
    
    selector = SelectKBest(score_func=scorer, k=min(k, X.shape[1]))
    
    try:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_selected = selector.fit_transform(X, y)
            
            mask = selector.get_support()
            selected_features = [f for f, m in zip(feature_names, mask) if m]
            
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        else:
            X_selected = selector.fit_transform(X, y)
            selected_features = list(range(X_selected.shape[1]))
        
        return X_selected, selected_features, selector
    except Exception as e:
        st.warning(f"Feature selection failed: {str(e)}. Returning original features.")
        return X, X.columns.tolist() if isinstance(X, pd.DataFrame) else list(range(X.shape[1])), None


def apply_pca(X, n_components=0.95, random_state=42):
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        X: Feature DataFrame or array
        n_components: Number of components or variance ratio to keep
        random_state: Random state
    
    Returns:
        Tuple of (transformed X, PCA object, explained variance)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # Create column names
    n_comps = X_pca.shape[1]
    columns = [f"PC{i+1}" for i in range(n_comps)]
    
    if isinstance(X, pd.DataFrame):
        X_pca = pd.DataFrame(X_pca, columns=columns, index=X.index)
    
    explained_variance = pca.explained_variance_ratio_
    
    return X_pca, pca, explained_variance


def render_feature_engineering_page():
    """Render the feature engineering page."""
    st.header("Feature Engineering")
    
    if not st.session_state.get('preprocessing_done', False):
        st.warning("Please complete preprocessing first!")
        return
    
    X_train = st.session_state.get('X_train')
    X_test = st.session_state.get('X_test')
    y_train = st.session_state.get('y_train')
    y_test = st.session_state.get('y_test')
    
    if X_train is None:
        st.warning("No training data available. Please complete preprocessing first!")
        return
    
    st.info(f"Current features: {X_train.shape[1]} | Training samples: {len(X_train)}")
    
    mode = st.session_state.get('mode', 'Beginner')
    
    # Track original feature count
    original_features = X_train.shape[1]
    
    if mode == "Beginner":
        st.markdown("### Beginner Mode - Automatic Feature Selection")
        
        if st.button("Apply Automatic Feature Selection", type="primary", use_container_width=True):
            with st.spinner("Selecting features..."):
                progress = st.progress(0)
                
                X_train_fe = X_train.copy()
                X_test_fe = X_test.copy()
                removed_features = []
                
                # Step 1: Remove low variance
                progress.progress(25)
                X_train_fe, removed_low_var, _ = remove_low_variance_features(X_train_fe, threshold=0.01)
                if removed_low_var:
                    X_test_fe = X_test_fe.drop(columns=[c for c in removed_low_var if c in X_test_fe.columns], errors='ignore')
                    removed_features.extend(removed_low_var)
                
                # Step 2: Remove highly correlated
                progress.progress(50)
                if isinstance(X_train_fe, pd.DataFrame):
                    y_series = pd.Series(y_train, index=X_train_fe.index)
                    X_train_fe, removed_corr = remove_correlated_features(X_train_fe, threshold=0.95, target=y_series)
                    if removed_corr:
                        X_test_fe = X_test_fe.drop(columns=[c for c in removed_corr if c in X_test_fe.columns], errors='ignore')
                        removed_features.extend(removed_corr)
                
                # Step 3: Get feature importance
                progress.progress(75)
                importances, _ = get_feature_importance_rf(X_train_fe, y_train)
                
                # Keep top 80% important features (or at least 5)
                n_keep = max(5, int(len(importances) * 0.8))
                top_features = importances.head(n_keep)['Feature'].tolist()
                
                if len(top_features) < X_train_fe.shape[1]:
                    X_train_fe = X_train_fe[top_features]
                    X_test_fe = X_test_fe[[c for c in top_features if c in X_test_fe.columns]]
                
                progress.progress(100)
                
                # Update session state
                st.session_state['X_train'] = X_train_fe
                st.session_state['X_test'] = X_test_fe
                st.session_state['feature_importances'] = importances
                st.session_state['feature_engineering_done'] = True
                
                # Display results
                st.success("Feature selection complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Features", original_features)
                with col2:
                    st.metric("Final Features", X_train_fe.shape[1])
                with col3:
                    st.metric("Removed", original_features - X_train_fe.shape[1])
                
                if removed_features:
                    with st.expander("Removed Features", expanded=False):
                        st.write(removed_features)
                
                # Feature importance chart
                st.markdown("#### Feature Importance (Top 20)")
                fig = px.bar(
                    importances.head(20),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 20 Feature Importances (Random Forest)'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Expert mode
        st.markdown("### Expert Mode - Custom Feature Selection")
        
        # Options
        col1, col2 = st.columns(2)
        
        with col1:
            remove_low_var = st.checkbox("Remove low variance features", value=True)
            if remove_low_var:
                var_threshold = st.slider("Variance threshold", 0.0, 0.1, 0.01, 0.005)
            else:
                var_threshold = 0.01
        
        with col2:
            remove_corr = st.checkbox("Remove highly correlated features", value=True)
            if remove_corr:
                corr_threshold = st.slider("Correlation threshold", 0.8, 1.0, 0.95, 0.01)
            else:
                corr_threshold = 0.95
        
        # Feature selection method
        st.markdown("#### Feature Selection Method")
        selection_method = st.selectbox(
            "Select method:",
            ["None", "Random Forest Importance", "SelectKBest", "PCA"]
        )
        
        if selection_method == "Random Forest Importance":
            n_top = st.slider("Keep top N features", 5, min(100, X_train.shape[1]), min(20, X_train.shape[1]))
        elif selection_method == "SelectKBest":
            k_best = st.slider("Select K best features", 5, min(100, X_train.shape[1]), min(20, X_train.shape[1]))
            score_func = st.selectbox("Scoring function", ["f_classif", "mutual_info"])
        elif selection_method == "PCA":
            variance_ratio = st.slider("Explained variance ratio to keep", 0.8, 0.99, 0.95, 0.01)
        
        if st.button("Apply Feature Engineering", type="primary", use_container_width=True):
            with st.spinner("Engineering features..."):
                X_train_fe = X_train.copy()
                X_test_fe = X_test.copy()
                
                # Remove low variance
                if remove_low_var:
                    X_train_fe, removed, selector = remove_low_variance_features(X_train_fe, var_threshold)
                    if removed:
                        X_test_fe = X_test_fe.drop(columns=[c for c in removed if c in X_test_fe.columns], errors='ignore')
                        st.info(f"Removed {len(removed)} low variance features")
                
                # Remove correlated
                if remove_corr and isinstance(X_train_fe, pd.DataFrame):
                    y_series = pd.Series(y_train, index=X_train_fe.index)
                    X_train_fe, removed = remove_correlated_features(X_train_fe, corr_threshold, y_series)
                    if removed:
                        X_test_fe = X_test_fe.drop(columns=[c for c in removed if c in X_test_fe.columns], errors='ignore')
                        st.info(f"Removed {len(removed)} highly correlated features")
                
                # Apply selection method
                if selection_method == "Random Forest Importance":
                    importances, _ = get_feature_importance_rf(X_train_fe, y_train)
                    top_features = importances.head(n_top)['Feature'].tolist()
                    X_train_fe = X_train_fe[top_features]
                    X_test_fe = X_test_fe[[c for c in top_features if c in X_test_fe.columns]]
                    st.session_state['feature_importances'] = importances
                
                elif selection_method == "SelectKBest":
                    X_train_fe, selected, _ = select_k_best_features(X_train_fe, y_train, k_best, score_func)
                    X_test_fe = X_test_fe[[c for c in selected if c in X_test_fe.columns]]
                
                elif selection_method == "PCA":
                    X_train_fe, pca_model, explained = apply_pca(X_train_fe, variance_ratio)
                    X_test_fe = pca_model.transform(X_test_fe)
                    X_test_fe = pd.DataFrame(X_test_fe, columns=X_train_fe.columns, index=X_test.index)
                    st.session_state['pca_model'] = pca_model
                    
                    # Show explained variance
                    st.markdown("#### PCA Explained Variance")
                    cum_var = np.cumsum(explained)
                    fig = px.line(
                        x=range(1, len(cum_var) + 1),
                        y=cum_var,
                        labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
                        title='PCA Cumulative Explained Variance'
                    )
                    fig.add_hline(y=variance_ratio, line_dash="dash", annotation_text=f"{variance_ratio*100}%")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Update session state
                st.session_state['X_train'] = X_train_fe
                st.session_state['X_test'] = X_test_fe
                st.session_state['feature_engineering_done'] = True
                
                st.success("Feature engineering complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Features", original_features)
                with col2:
                    st.metric("Final Features", X_train_fe.shape[1])
                with col3:
                    st.metric("Removed", original_features - X_train_fe.shape[1])
    
    # Show feature importance if available
    if 'feature_importances' in st.session_state and st.session_state.get('feature_engineering_done', False):
        importances = st.session_state['feature_importances']
        
        with st.expander("All Feature Importances", expanded=False):
            st.dataframe(importances, use_container_width=True, hide_index=True)
    
    # Skip button
    if not st.session_state.get('feature_engineering_done', False):
        st.markdown("---")
        if st.button("Skip Feature Engineering", use_container_width=True):
            st.session_state['feature_engineering_done'] = True
            st.success("Feature engineering skipped. Proceed to Model Training.")
    
    # Continue button
    if st.session_state.get('feature_engineering_done', False):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Model Training", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'training'
                st.rerun()
