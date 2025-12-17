"""
Model evaluation and comparison module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib
import io
import json

from modules.utils import close_figure, get_color_palette


def create_comparison_dataframe(results):
    """Create a comparison DataFrame from model results."""
    data = []
    
    for name, result in results.items():
        if result['success']:
            data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'] if result['roc_auc'] else None,
                'Training Time (s)': result['training_time'],
                'Best Parameters': str(result['best_params'])
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('F1-Score', ascending=False)
    
    return df


def get_model_rankings(results, metric='f1_score'):
    """Get model rankings based on a specific metric."""
    successful = {k: v for k, v in results.items() if v['success'] and v.get(metric)}
    
    if not successful:
        return []
    
    sorted_models = sorted(successful.items(), key=lambda x: x[1][metric] or 0, reverse=True)
    
    rankings = []
    for rank, (name, result) in enumerate(sorted_models, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        rankings.append({
            'Rank': medal,
            'Model': name,
            metric.replace('_', ' ').title(): result[metric]
        })
    
    return rankings


def plot_model_comparison_bar(results, metric='f1_score'):
    """Create bar chart comparing models on a metric."""
    data = []
    for name, result in results.items():
        if result['success'] and result.get(metric):
            data.append({
                'Model': name,
                'Value': result[metric]
            })
    
    df = pd.DataFrame(data).sort_values('Value', ascending=True)
    
    fig = px.bar(
        df, x='Value', y='Model', orientation='h',
        color='Value', color_continuous_scale='Blues',
        title=f'Model Comparison: {metric.replace("_", " ").title()}'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig


def plot_all_metrics_grouped(results):
    """Create grouped bar chart showing all metrics for each model."""
    data = []
    
    for name, result in results.items():
        if result['success']:
            data.append({'Model': name, 'Metric': 'Accuracy', 'Value': result['accuracy']})
            data.append({'Model': name, 'Metric': 'Precision', 'Value': result['precision']})
            data.append({'Model': name, 'Metric': 'Recall', 'Value': result['recall']})
            data.append({'Model': name, 'Metric': 'F1-Score', 'Value': result['f1_score']})
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df, x='Model', y='Value', color='Metric',
        barmode='group', title='All Metrics by Model'
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig


def plot_training_time_comparison(results):
    """Create horizontal bar chart of training times."""
    data = []
    for name, result in results.items():
        if result['success']:
            data.append({
                'Model': name,
                'Training Time (s)': result['training_time']
            })
    
    df = pd.DataFrame(data).sort_values('Training Time (s)', ascending=True)
    
    fig = px.bar(
        df, x='Training Time (s)', y='Model', orientation='h',
        color='Training Time (s)', color_continuous_scale='Reds',
        title='Training Time Comparison'
    )
    
    return fig


def plot_confusion_matrix(conf_matrix, class_names=None, title='Confusion Matrix'):
    """Create confusion matrix heatmap."""
    if class_names is None:
        class_names = [str(i) for i in range(conf_matrix.shape[0])]
    
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        title=title,
        text_auto=True
    )
    
    return fig


def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models (binary classification)."""
    fig = go.Figure()
    
    colors = get_color_palette(len(results))
    
    for idx, (name, result) in enumerate(results.items()):
        if result['success'] and result['y_proba'] is not None:
            try:
                if result['y_proba'].shape[1] == 2:  # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, result['y_proba'][:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f'{name} (AUC={roc_auc:.3f})',
                        line=dict(color=colors[idx % len(colors)])
                    ))
            except:
                pass
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.6, y=0.1)
    )
    
    return fig


def plot_precision_recall_curves(results, y_test):
    """Plot Precision-Recall curves for all models."""
    fig = go.Figure()
    
    colors = get_color_palette(len(results))
    
    for idx, (name, result) in enumerate(results.items()):
        if result['success'] and result['y_proba'] is not None:
            try:
                if result['y_proba'].shape[1] == 2:  # Binary classification
                    precision, recall, _ = precision_recall_curve(y_test, result['y_proba'][:, 1])
                    ap = average_precision_score(y_test, result['y_proba'][:, 1])
                    
                    fig.add_trace(go.Scatter(
                        x=recall, y=precision,
                        name=f'{name} (AP={ap:.3f})',
                        line=dict(color=colors[idx % len(colors)])
                    ))
            except:
                pass
    
    fig.update_layout(
        title='Precision-Recall Curves Comparison',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.1, y=0.1)
    )
    
    return fig


def get_downloadable_results(results, comparison_df):
    """Prepare downloadable results in various formats."""
    # CSV
    csv_buffer = io.StringIO()
    comparison_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    # JSON
    json_data = {}
    for name, result in results.items():
        if result['success']:
            json_data[name] = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'roc_auc': result['roc_auc'],
                'training_time': result['training_time'],
                'best_params': result['best_params']
            }
    json_str = json.dumps(json_data, indent=2)
    
    return csv_data, json_str


def render_model_comparison_page():
    """Render the model comparison page."""
    st.header("📈 Model Comparison Dashboard")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first!")
        return
    
    results = st.session_state.get('model_results', {})
    y_test = st.session_state.get('y_test')
    
    if not results:
        st.warning("⚠️ No model results available!")
        return
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(results)
    
    # Summary metrics
    successful_models = sum(1 for r in results.values() if r['success'])
    failed_models = len(results) - successful_models
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Trained", successful_models)
    with col2:
        st.metric("Failed", failed_models)
    with col3:
        best_f1 = comparison_df['F1-Score'].max() if len(comparison_df) > 0 else 0
        st.metric("Best F1-Score", f"{best_f1:.4f}")
    with col4:
        best_model = comparison_df.iloc[0]['Model'] if len(comparison_df) > 0 else 'N/A'
        st.metric("Best Model", best_model)
    
    st.markdown("---")
    
    # Comparison Table
    st.subheader("📊 Model Comparison Table")
    
    # Metric selection for sorting
    sort_metric = st.selectbox(
        "Sort by:",
        ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Training Time (s)'],
        index=0
    )
    
    sorted_df = comparison_df.sort_values(sort_metric, ascending=(sort_metric == 'Training Time (s)'))
    
    # Style the dataframe
    def highlight_best(s):
        if s.name in ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']:
            is_best = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_best]
        elif s.name == 'Training Time (s)':
            is_best = s == s.min()
            return ['background-color: #90EE90' if v else '' for v in is_best]
        return ['' for _ in s]
    
    styled_df = sorted_df.style.apply(highlight_best)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Rankings
    st.subheader("🏆 Model Rankings")
    
    metric_for_ranking = st.selectbox(
        "Rank by:",
        ['f1_score', 'accuracy', 'precision', 'recall', 'roc_auc'],
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    rankings = get_model_rankings(results, metric_for_ranking)
    if rankings:
        rankings_df = pd.DataFrame(rankings)
        st.dataframe(rankings_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Metric Comparison", "All Metrics", "Training Time", "ROC Curves", "Confusion Matrices"
    ])
    
    with tab1:
        metric_to_plot = st.selectbox(
            "Select metric:",
            ['f1_score', 'accuracy', 'precision', 'recall', 'roc_auc'],
            format_func=lambda x: x.replace('_', ' ').title(),
            key='metric_bar'
        )
        fig = plot_model_comparison_bar(results, metric_to_plot)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = plot_all_metrics_grouped(results)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = plot_training_time_comparison(results)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            fig = plot_roc_curves(results, y_test)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Precision-Recall Curves")
            fig_pr = plot_precision_recall_curves(results, y_test)
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.info("ROC curves are displayed for binary classification. For multiclass, see individual model confusion matrices.")
    
    with tab5:
        model_for_cm = st.selectbox(
            "Select model:",
            [name for name, r in results.items() if r['success']],
            key='cm_model'
        )
        
        if model_for_cm:
            result = results[model_for_cm]
            if result['confusion_matrix'] is not None:
                target_encoder = st.session_state.get('target_encoder')
                if target_encoder:
                    class_names = target_encoder.classes_.tolist()
                else:
                    class_names = [str(i) for i in range(result['confusion_matrix'].shape[0])]
                
                fig = plot_confusion_matrix(
                    result['confusion_matrix'],
                    class_names,
                    f'Confusion Matrix - {model_for_cm}'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                with st.expander("📋 Classification Report"):
                    y_test = st.session_state.get('y_test')
                    report = classification_report(y_test, result['y_pred'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
    
    st.markdown("---")
    
    # Best Model Details
    st.subheader("🌟 Best Model Details")
    
    best_idx = comparison_df['F1-Score'].idxmax()
    best_model_name = comparison_df.loc[best_idx, 'Model']
    best_result = results[best_model_name]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {best_model_name}")
        st.markdown(f"**Accuracy:** {best_result['accuracy']:.4f}")
        st.markdown(f"**Precision:** {best_result['precision']:.4f}")
        st.markdown(f"**Recall:** {best_result['recall']:.4f}")
        st.markdown(f"**F1-Score:** {best_result['f1_score']:.4f}")
        if best_result['roc_auc']:
            st.markdown(f"**ROC-AUC:** {best_result['roc_auc']:.4f}")
        st.markdown(f"**Training Time:** {best_result['training_time']:.2f}s")
    
    with col2:
        st.markdown("**Best Hyperparameters:**")
        if best_result['best_params']:
            for param, value in best_result['best_params'].items():
                st.markdown(f"- `{param}`: {value}")
        else:
            st.markdown("Default parameters used")
    
    st.markdown("---")
    
    # Downloads
    st.subheader("📥 Download Results")
    
    csv_data, json_data = get_downloadable_results(results, comparison_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name="model_results.json",
            mime="application/json"
        )
    
    with col3:
        # Download best model
        if best_result['model'] is not None:
            model_bytes = io.BytesIO()
            joblib.dump(best_result['model'], model_bytes)
            model_bytes.seek(0)
            
            st.download_button(
                label="🤖 Download Best Model",
                data=model_bytes,
                file_name=f"{best_model_name.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )
    
    st.success("✅ Model comparison complete! Proceed to Generate Report for a comprehensive PDF report.")
