"""
Model training and hyperparameter tuning module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from config.model_configs import HYPERPARAMETERS, HYPERPARAMETERS_FAST, MODEL_DESCRIPTIONS


def get_model_instance(model_name):
    """Get a model instance by name."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'Rule-Based Classifier': DecisionTreeClassifier(max_depth=1, random_state=42)
    }
    return models.get(model_name)


def train_single_model(model_name, X_train, y_train, X_test, y_test, 
                       use_grid_search=True, fast_mode=False, class_weight=None):
    """
    Train a single model with hyperparameter tuning.
    
    Args:
        model_name: Name of the model
        X_train, y_train: Training data
        X_test, y_test: Test data
        use_grid_search: Use GridSearchCV (True) or RandomizedSearchCV (False)
        fast_mode: Use reduced hyperparameter grid
        class_weight: Class weights for imbalanced data
    
    Returns:
        Dict with model results
    """
    start_time = time.time()
    
    model = get_model_instance(model_name)
    
    if model is None:
        return None
    
    # Get hyperparameters
    param_grid = HYPERPARAMETERS_FAST.get(model_name, {}) if fast_mode else HYPERPARAMETERS.get(model_name, {})
    
    # Add class weight if applicable
    if class_weight and hasattr(model, 'class_weight'):
        model.set_params(class_weight=class_weight)
    
    # Determine number of classes for scoring
    n_classes = len(np.unique(y_train))
    scoring = 'f1_weighted' if n_classes > 2 else 'f1'
    
    try:
        if param_grid:
            if use_grid_search:
                search = GridSearchCV(
                    model, param_grid, 
                    cv=5, scoring=scoring, 
                    n_jobs=-1, error_score='raise'
                )
            else:
                search = RandomizedSearchCV(
                    model, param_grid, 
                    n_iter=20, cv=5, scoring=scoring,
                    n_jobs=-1, random_state=42, error_score='raise'
                )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
            cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring).mean()
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        if n_classes > 2:
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            if y_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except:
                    roc_auc = None
            else:
                roc_auc = None
        else:
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if y_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                except:
                    roc_auc = None
            else:
                roc_auc = None
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'model_name': model_name,
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'training_time': training_time,
            'confusion_matrix': conf_matrix,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        training_time = time.time() - start_time
        return {
            'model_name': model_name,
            'model': None,
            'best_params': {},
            'cv_score': None,
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'roc_auc': None,
            'training_time': training_time,
            'confusion_matrix': None,
            'y_pred': None,
            'y_proba': None,
            'success': False,
            'error': str(e)
        }


def train_all_models(X_train, y_train, X_test, y_test, 
                     selected_models=None, use_grid_search=True, 
                     fast_mode=False, class_weight=None, progress_callback=None):
    """
    Train all selected models.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        selected_models: List of model names to train
        use_grid_search: Use GridSearchCV
        fast_mode: Use reduced hyperparameter grid
        class_weight: Class weights for imbalanced data
        progress_callback: Callback function for progress updates
    
    Returns:
        Dict of model results keyed by model name
    """
    if selected_models is None:
        selected_models = list(HYPERPARAMETERS.keys())
    
    results = {}
    total_models = len(selected_models)
    
    for idx, model_name in enumerate(selected_models):
        if progress_callback:
            progress_callback(idx, total_models, model_name)
        
        result = train_single_model(
            model_name, X_train, y_train, X_test, y_test,
            use_grid_search, fast_mode, class_weight
        )
        
        if result:
            results[model_name] = result
    
    return results


def save_model(model, filename):
    """Save a trained model to disk."""
    joblib.dump(model, filename)


def load_model(filename):
    """Load a trained model from disk."""
    return joblib.load(filename)


def render_model_training_page():
    """Render the model training page."""
    st.header("Model Training")
    
    if not st.session_state.get('preprocessing_done', False):
        st.warning("Please complete preprocessing first!")
        return
    
    X_train = st.session_state.get('X_train')
    X_test = st.session_state.get('X_test')
    y_train = st.session_state.get('y_train')
    y_test = st.session_state.get('y_test')
    
    if X_train is None:
        st.warning("No training data available!")
        return
    
    # Info
    st.info(f"Training: {len(X_train)} samples, {X_train.shape[1]} features | Test: {len(X_test)} samples")
    
    mode = st.session_state.get('mode', 'Beginner')
    use_class_weights = st.session_state.get('use_class_weights', False)
    
    # Model selection
    all_models = list(HYPERPARAMETERS.keys())
    
    if mode == "Beginner":
        st.markdown("### Beginner Mode - Train All Models")
        st.markdown("All 7 classifiers will be trained with smart hyperparameter tuning.")
        
        selected_models = all_models
        fast_mode = True
        use_grid_search = False
        
        # Show model descriptions
        with st.expander("Model Descriptions", expanded=False):
            for model_name, description in MODEL_DESCRIPTIONS.items():
                st.markdown(f"**{model_name}:** {description}")
    
    else:  # Expert mode
        st.markdown("### Expert Mode - Custom Training")
        
        # Model selection
        selected_models = st.multiselect(
            "Select models to train:",
            all_models,
            default=all_models
        )
        
        col1, col2 = st.columns(2)
        with col1:
            search_method = st.radio(
                "Hyperparameter search method:",
                ["Grid Search (thorough)", "Random Search (faster)"]
            )
            use_grid_search = "Grid" in search_method
        
        with col2:
            fast_mode = st.checkbox("Use reduced hyperparameter grid (faster)", value=False)
        
        use_class_weights = st.checkbox(
            "Use class weights (for imbalanced data)",
            value=use_class_weights
        )
    
    # Train button
    if st.button("Train Models", type="primary", use_container_width=True, disabled=len(selected_models) == 0):
        
        class_weight = 'balanced' if use_class_weights else None
        
        # Progress container
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(idx, total, model_name):
            progress = (idx / total)
            progress_bar.progress(progress)
            status_text.markdown(f"**Training:** {model_name} ({idx + 1}/{total})")
        
        # Train models
        with st.spinner("Training models..."):
            results = train_all_models(
                X_train, y_train, X_test, y_test,
                selected_models=selected_models,
                use_grid_search=use_grid_search,
                fast_mode=fast_mode,
                class_weight=class_weight,
                progress_callback=progress_callback
            )
        
        progress_bar.progress(1.0)
        status_text.markdown("**Training complete!**")
        
        # Store results
        st.session_state['model_results'] = results
        st.session_state['models_trained'] = True
        
        # Display summary
        st.success(f"Trained {len(results)} models successfully!")
        
        # Quick results table
        results_data = []
        for name, result in results.items():
            if result['success']:
                results_data.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.4f}" if result['accuracy'] else 'N/A',
                    'F1-Score': f"{result['f1_score']:.4f}" if result['f1_score'] else 'N/A',
                    'ROC-AUC': f"{result['roc_auc']:.4f}" if result['roc_auc'] else 'N/A',
                    'Time (s)': f"{result['training_time']:.2f}"
                })
            else:
                results_data.append({
                    'Model': name,
                    'Accuracy': 'Failed',
                    'F1-Score': 'Failed',
                    'ROC-AUC': 'Failed',
                    'Time (s)': f"{result['training_time']:.2f}",
                    'Error': result['error']
                })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Best model
        successful_results = {k: v for k, v in results.items() if v['success']}
        if successful_results:
            best_model_name = max(successful_results, key=lambda x: successful_results[x]['f1_score'] or 0)
            best_result = successful_results[best_model_name]
            
            st.markdown("---")
            st.markdown("### Best Model")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model", best_model_name)
            with col2:
                st.metric("Accuracy", f"{best_result['accuracy']:.4f}")
            with col3:
                st.metric("F1-Score", f"{best_result['f1_score']:.4f}")
            with col4:
                if best_result['roc_auc']:
                    st.metric("ROC-AUC", f"{best_result['roc_auc']:.4f}")
            
            st.session_state['best_model_name'] = best_model_name
            st.session_state['best_model'] = best_result['model']
        
        st.info("Proceed to Model Comparison for detailed analysis and visualizations.")
    
    # Show previous results if available
    elif st.session_state.get('models_trained', False):
        st.markdown("---")
        st.subheader("Previous Training Results")
        
        results = st.session_state.get('model_results', {})
        
        if results:
            results_data = []
            for name, result in results.items():
                if result['success']:
                    results_data.append({
                        'Model': name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'Precision': f"{result['precision']:.4f}",
                        'Recall': f"{result['recall']:.4f}",
                        'F1-Score': f"{result['f1_score']:.4f}",
                        'ROC-AUC': f"{result['roc_auc']:.4f}" if result['roc_auc'] else 'N/A',
                        'Time (s)': f"{result['training_time']:.2f}"
                    })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            if st.button("Retrain Models"):
                st.session_state['models_trained'] = False
                st.rerun()
    
    # Continue button if models are trained
    if st.session_state.get('models_trained', False):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Model Comparison", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'comparison'
                st.rerun()
