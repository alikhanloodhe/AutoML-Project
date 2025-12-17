"""
PDF and HTML report generation module.
"""

import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import tempfile
from datetime import datetime
import base64

from modules.utils import close_figure


class AutoMLReport(FPDF):
    """Custom PDF report class for AutoML results."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'AutoML Classification Report', 0, 0, 'L')
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'R')
        self.line(10, 20, 200, 20)
        self.ln(5)
    
    def footer(self):
        """Add footer with page number."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add a chapter title."""
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 102, 204)
        self.cell(0, 10, title, 0, 1, 'L')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
    
    def section_title(self, title):
        """Add a section title."""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
    
    def body_text(self, text):
        """Add body text."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def add_metric(self, label, value):
        """Add a metric line."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(51, 51, 51)
        self.cell(60, 6, label + ':', 0, 0)
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, str(value), 0, 1)
    
    def add_table(self, headers, data, col_widths=None):
        """Add a table to the report."""
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(0, 102, 204)
        self.set_text_color(255, 255, 255)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, str(header), 1, 0, 'C', True)
        self.ln()
        
        # Data rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)
            
            for i, cell in enumerate(row):
                cell_text = str(cell)[:30] if len(str(cell)) > 30 else str(cell)
                self.cell(col_widths[i], 6, cell_text, 1, 0, 'C', True)
            self.ln()
            fill = not fill
        
        self.ln(3)
    
    def add_image_from_fig(self, fig, width=180):
        """Add matplotlib figure to PDF."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name, format='png', dpi=150, bbox_inches='tight')
            tmp_path = tmp.name
        
        try:
            self.image(tmp_path, x=15, w=width)
            self.ln(5)
        finally:
            os.unlink(tmp_path)


def generate_pdf_report(session_state):
    """Generate a comprehensive PDF report."""
    pdf = AutoMLReport()
    
    # Cover Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(0, 102, 204)
    pdf.ln(40)
    pdf.cell(0, 15, 'AutoML Classification Report', 0, 1, 'C')
    
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(100, 100, 100)
    pdf.ln(10)
    
    file_name = session_state.get('file_name', 'Unknown Dataset')
    pdf.cell(0, 8, f'Dataset: {file_name}', 0, 1, 'C')
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 11)
    pdf.cell(0, 8, 'Prepared by AutoML Classification System', 0, 1, 'C')
    
    # Team info
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(51, 51, 51)
    pdf.cell(0, 8, 'Team Members:', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, 'Ali Asghar Khan Lodhi (478734)', 0, 1, 'C')
    pdf.cell(0, 6, 'Muhammad Saad Akhtar (458102)', 0, 1, 'C')
    
    # Executive Summary
    pdf.add_page()
    pdf.chapter_title('1. Executive Summary')
    
    # Get best model info
    results = session_state.get('model_results', {})
    best_model_name = session_state.get('best_model_name', 'N/A')
    
    if best_model_name != 'N/A' and best_model_name in results:
        best_result = results[best_model_name]
        pdf.body_text(f"This report presents the results of an automated machine learning analysis "
                     f"performed on the dataset '{file_name}'. After training and evaluating 7 different "
                     f"classification models, {best_model_name} emerged as the best performing model "
                     f"with an F1-Score of {best_result['f1_score']:.4f}.")
        
        pdf.ln(5)
        pdf.section_title('Key Findings')
        pdf.add_metric('Best Model', best_model_name)
        pdf.add_metric('Accuracy', f"{best_result['accuracy']:.4f}")
        pdf.add_metric('Precision', f"{best_result['precision']:.4f}")
        pdf.add_metric('Recall', f"{best_result['recall']:.4f}")
        pdf.add_metric('F1-Score', f"{best_result['f1_score']:.4f}")
        if best_result['roc_auc']:
            pdf.add_metric('ROC-AUC', f"{best_result['roc_auc']:.4f}")
    
    # Dataset Overview
    pdf.add_page()
    pdf.chapter_title('2. Dataset Overview')
    
    df = session_state.get('data')
    if df is not None:
        pdf.section_title('Basic Statistics')
        pdf.add_metric('Total Rows', len(df))
        pdf.add_metric('Total Columns', len(df.columns))
        pdf.add_metric('Memory Usage', f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        target_col = session_state.get('target_column')
        if target_col:
            pdf.add_metric('Target Variable', target_col)
            pdf.add_metric('Number of Classes', df[target_col].nunique())
        
        # Column types
        pdf.ln(5)
        pdf.section_title('Column Information')
        
        column_types = session_state.get('column_types', {})
        pdf.add_metric('Numerical Columns', len(column_types.get('numerical', [])))
        pdf.add_metric('Categorical Columns', len(column_types.get('categorical', [])))
        
        # Class distribution
        if target_col and target_col in df.columns:
            pdf.ln(5)
            pdf.section_title('Target Variable Distribution')
            
            class_dist = df[target_col].value_counts()
            headers = ['Class', 'Count', 'Percentage']
            data = [[str(cls), count, f"{count/len(df)*100:.2f}%"] 
                   for cls, count in class_dist.items()]
            pdf.add_table(headers, data[:10], [60, 60, 70])
    
    # EDA Summary
    pdf.add_page()
    pdf.chapter_title('3. Exploratory Data Analysis')
    
    missing_analysis = session_state.get('missing_analysis')
    if missing_analysis is not None and isinstance(missing_analysis, pd.DataFrame):
        cols_with_missing = missing_analysis[missing_analysis['Missing Count'] > 0]
        if len(cols_with_missing) > 0:
            pdf.section_title('Missing Values Summary')
            headers = ['Column', 'Missing Count', 'Missing %']
            data = cols_with_missing[['Column', 'Missing Count', 'Missing %']].values.tolist()[:10]
            pdf.add_table(headers, data, [80, 50, 60])
        else:
            pdf.body_text("No missing values were found in the dataset.")
    
    outlier_analysis = session_state.get('outlier_analysis')
    if outlier_analysis is not None and isinstance(outlier_analysis, pd.DataFrame):
        cols_with_outliers = outlier_analysis[outlier_analysis['Total (IQR)'] > 0]
        if len(cols_with_outliers) > 0:
            pdf.ln(5)
            pdf.section_title('Outlier Detection Summary')
            headers = ['Column', 'IQR Outliers', 'Outlier %']
            data = cols_with_outliers[['Column', 'IQR Outliers', 'Outlier %']].values.tolist()[:10]
            pdf.add_table(headers, data, [80, 50, 60])
    
    # Preprocessing Summary
    pdf.add_page()
    pdf.chapter_title('4. Preprocessing Decisions')
    
    pipeline = session_state.get('preprocessing_pipeline')
    if pipeline:
        pdf.section_title('Applied Preprocessing Steps')
        summary = pipeline.get_summary()
        if len(summary) > 0:
            headers = ['Step', 'Action', 'Parameters']
            data = []
            for _, row in summary.iterrows():
                data.append([row['step'], row['action'], str(row['params'])[:40]])
            pdf.add_table(headers, data, [60, 70, 60])
    
    X_train = session_state.get('X_train')
    X_test = session_state.get('X_test')
    if X_train is not None and X_test is not None:
        pdf.ln(5)
        pdf.section_title('Train-Test Split')
        pdf.add_metric('Training Samples', len(X_train))
        pdf.add_metric('Test Samples', len(X_test))
        pdf.add_metric('Final Features', X_train.shape[1])
    
    # Model Training Results
    pdf.add_page()
    pdf.chapter_title('5. Model Training & Tuning')
    
    if results:
        pdf.section_title('Models Trained')
        pdf.body_text("The following 7 classifiers were trained with hyperparameter optimization:")
        
        model_list = [
            "1. Logistic Regression",
            "2. K-Nearest Neighbors (KNN)",
            "3. Decision Tree",
            "4. Naive Bayes",
            "5. Random Forest",
            "6. Support Vector Machine (SVM)",
            "7. Rule-Based Classifier"
        ]
        for model in model_list:
            pdf.body_text(f"  {model}")
        
        pdf.ln(5)
        pdf.section_title('Training Summary')
        
        for name, result in results.items():
            if result['success']:
                pdf.set_font('Helvetica', 'B', 10)
                pdf.cell(0, 6, name, 0, 1)
                pdf.set_font('Helvetica', '', 9)
                pdf.cell(0, 5, f"  Training Time: {result['training_time']:.2f}s", 0, 1)
                if result['best_params']:
                    params_str = str(result['best_params'])[:80]
                    pdf.cell(0, 5, f"  Best Params: {params_str}", 0, 1)
                pdf.ln(2)
    
    # Model Comparison
    pdf.add_page()
    pdf.chapter_title('6. Model Performance Comparison')
    
    if results:
        pdf.section_title('Performance Metrics')
        
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        data = []
        
        for name, result in results.items():
            if result['success']:
                data.append([
                    name[:20],
                    f"{result['accuracy']:.4f}",
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['f1_score']:.4f}",
                    f"{result['roc_auc']:.4f}" if result['roc_auc'] else 'N/A'
                ])
        
        # Sort by F1-Score
        data.sort(key=lambda x: float(x[4]), reverse=True)
        pdf.add_table(headers, data, [35, 25, 30, 25, 30, 30])
        
        # Create comparison chart
        pdf.ln(5)
        pdf.section_title('Performance Visualization')
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        models = [r['model_name'][:15] for r in results.values() if r['success']]
        f1_scores = [r['f1_score'] for r in results.values() if r['success']]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(models)))
        bars = ax.barh(models, f1_scores, color=colors)
        ax.set_xlabel('F1-Score')
        ax.set_title('Model Comparison by F1-Score')
        
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        pdf.add_image_from_fig(fig, width=170)
        close_figure(fig)
    
    # Best Model Details
    pdf.add_page()
    pdf.chapter_title('7. Best Model Analysis')
    
    if best_model_name != 'N/A' and best_model_name in results:
        best_result = results[best_model_name]
        
        pdf.section_title(f'Best Model: {best_model_name}')
        
        pdf.body_text(f"{best_model_name} achieved the highest F1-Score among all tested models, "
                     f"making it the recommended choice for this classification task.")
        
        pdf.ln(5)
        pdf.section_title('Performance Metrics')
        pdf.add_metric('Accuracy', f"{best_result['accuracy']:.4f}")
        pdf.add_metric('Precision', f"{best_result['precision']:.4f}")
        pdf.add_metric('Recall', f"{best_result['recall']:.4f}")
        pdf.add_metric('F1-Score', f"{best_result['f1_score']:.4f}")
        if best_result['roc_auc']:
            pdf.add_metric('ROC-AUC', f"{best_result['roc_auc']:.4f}")
        pdf.add_metric('Training Time', f"{best_result['training_time']:.2f} seconds")
        
        pdf.ln(5)
        pdf.section_title('Optimal Hyperparameters')
        if best_result['best_params']:
            for param, value in best_result['best_params'].items():
                pdf.add_metric(param, str(value))
        else:
            pdf.body_text("Default parameters were used.")
        
        # Confusion matrix
        if best_result['confusion_matrix'] is not None:
            pdf.ln(5)
            pdf.section_title('Confusion Matrix')
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - {best_model_name}')
            plt.tight_layout()
            pdf.add_image_from_fig(fig, width=120)
            close_figure(fig)
    
    # Recommendations
    pdf.add_page()
    pdf.chapter_title('8. Recommendations')
    
    pdf.section_title('Model Deployment')
    pdf.body_text(f"Based on the analysis, we recommend deploying {best_model_name} for production use. "
                 f"The model has been saved and can be loaded for making predictions on new data.")
    
    pdf.ln(5)
    pdf.section_title('Potential Improvements')
    improvements = [
        "1. Collect more training data if possible to improve model generalization.",
        "2. Perform feature engineering to create more informative features.",
        "3. Try ensemble methods combining multiple models for better performance.",
        "4. Monitor model performance in production and retrain periodically.",
        "5. Consider class balancing techniques if dealing with imbalanced data."
    ]
    for imp in improvements:
        pdf.body_text(imp)
    
    pdf.ln(5)
    pdf.section_title('Limitations')
    pdf.body_text("This automated analysis provides a good starting point but should be validated "
                 "by domain experts. Model performance may vary on out-of-sample data.")
    
    # Generate PDF bytes
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    return pdf_output


def generate_html_report(session_state):
    """Generate an HTML report."""
    results = session_state.get('model_results', {})
    best_model_name = session_state.get('best_model_name', 'N/A')
    file_name = session_state.get('file_name', 'Unknown Dataset')
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Classification Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #0066cc;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #0066cc;
            margin-top: 30px;
        }}
        h3 {{
            color: #333;
        }}
        .metric {{
            display: inline-block;
            background: #e3f2fd;
            padding: 15px 25px;
            margin: 5px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #0066cc;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best-model {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .best-model h3 {{
            color: white;
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AutoML Classification Report</h1>
        <p><strong>Dataset:</strong> {file_name}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
        
        <h2>📊 Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>Time (s)</th>
            </tr>
    """
    
    # Add model results
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['f1_score'] if x[1]['success'] else 0, 
                           reverse=True)
    
    for name, result in sorted_results:
        if result['success']:
            html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{result['accuracy']:.4f}</td>
                <td>{result['precision']:.4f}</td>
                <td>{result['recall']:.4f}</td>
                <td>{result['f1_score']:.4f}</td>
                <td>{result['roc_auc']:.4f if result['roc_auc'] else 'N/A'}</td>
                <td>{result['training_time']:.2f}</td>
            </tr>
            """
    
    html += "</table>"
    
    # Best model section
    if best_model_name in results and results[best_model_name]['success']:
        best = results[best_model_name]
        html += f"""
        <div class="best-model">
            <h3>🏆 Best Model: {best_model_name}</h3>
            <div class="metric">
                <div class="metric-value" style="color: white;">{best['accuracy']:.4f}</div>
                <div class="metric-label" style="color: #ddd;">Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: white;">{best['f1_score']:.4f}</div>
                <div class="metric-label" style="color: #ddd;">F1-Score</div>
            </div>
            <div class="metric">
                <div class="metric-value" style="color: white;">{best['roc_auc']:.4f if best['roc_auc'] else 'N/A'}</div>
                <div class="metric-label" style="color: #ddd;">ROC-AUC</div>
            </div>
        </div>
        """
    
    html += """
        <h2>👥 Team Members</h2>
        <ul>
            <li>Ali Asghar Khan Lodhi (478734)</li>
            <li>Muhammad Saad Akhtar (458102)</li>
        </ul>
        
        <div class="footer">
            <p>Generated by AutoML Classification System</p>
            <p>CS-245 Machine Learning Course - NUST SEECS</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


def generate_markdown_report(session_state):
    """Generate a Markdown report."""
    results = session_state.get('model_results', {})
    best_model_name = session_state.get('best_model_name', 'N/A')
    file_name = session_state.get('file_name', 'Unknown Dataset')
    
    md = f"""# AutoML Classification Report

**Dataset:** {file_name}  
**Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M")}

---

## Executive Summary

"""
    
    if best_model_name in results and results[best_model_name]['success']:
        best = results[best_model_name]
        md += f"""The best performing model is **{best_model_name}** with:
- Accuracy: {best['accuracy']:.4f}
- F1-Score: {best['f1_score']:.4f}
- ROC-AUC: {best['roc_auc']:.4f if best['roc_auc'] else 'N/A'}

"""
    
    md += """## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Time (s) |
|-------|----------|-----------|--------|----------|---------|----------|
"""
    
    for name, result in sorted(results.items(), 
                               key=lambda x: x[1]['f1_score'] if x[1]['success'] else 0,
                               reverse=True):
        if result['success']:
            md += f"| {name} | {result['accuracy']:.4f} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1_score']:.4f} | {result['roc_auc']:.4f if result['roc_auc'] else 'N/A'} | {result['training_time']:.2f} |\n"
    
    md += """
---

## Team Members

- Ali Asghar Khan Lodhi (478734)
- Muhammad Saad Akhtar (458102)

---

*Generated by AutoML Classification System*  
*CS-245 Machine Learning Course - NUST SEECS*
"""
    
    return md


def render_report_generation_page():
    """Render the report generation page."""
    st.header("📄 Generate Report")
    
    if not st.session_state.get('models_trained', False):
        st.warning("⚠️ Please train models first before generating a report!")
        return
    
    st.markdown("""
    Generate a comprehensive report of your AutoML analysis including:
    - Executive Summary
    - Dataset Overview
    - EDA Results
    - Preprocessing Steps
    - Model Training Results
    - Performance Comparison
    - Best Model Analysis
    - Recommendations
    """)
    
    st.markdown("---")
    
    # Report options
    st.subheader("📝 Report Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_charts = st.checkbox("Include Charts", value=True)
        include_confusion_matrix = st.checkbox("Include Confusion Matrix", value=True)
    
    with col2:
        include_hyperparams = st.checkbox("Include Hyperparameters", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    st.markdown("---")
    
    # Generate buttons
    st.subheader("📥 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_bytes = generate_pdf_report(st.session_state)
                    st.session_state['pdf_report'] = pdf_bytes
                    st.success("✅ PDF report generated!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
    
    with col2:
        if st.button("🌐 Generate HTML Report", type="primary", use_container_width=True):
            with st.spinner("Generating HTML report..."):
                try:
                    html_content = generate_html_report(st.session_state)
                    st.session_state['html_report'] = html_content
                    st.success("✅ HTML report generated!")
                except Exception as e:
                    st.error(f"Error generating HTML: {str(e)}")
    
    with col3:
        if st.button("📝 Generate Markdown Report", type="primary", use_container_width=True):
            with st.spinner("Generating Markdown report..."):
                try:
                    md_content = generate_markdown_report(st.session_state)
                    st.session_state['md_report'] = md_content
                    st.success("✅ Markdown report generated!")
                except Exception as e:
                    st.error(f"Error generating Markdown: {str(e)}")
    
    st.markdown("---")
    
    # Download buttons
    st.subheader("📥 Download Generated Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'pdf_report' in st.session_state:
            st.download_button(
                label="📄 Download PDF",
                data=st.session_state['pdf_report'],
                file_name="automl_report.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Generate PDF first")
    
    with col2:
        if 'html_report' in st.session_state:
            st.download_button(
                label="🌐 Download HTML",
                data=st.session_state['html_report'],
                file_name="automl_report.html",
                mime="text/html"
            )
        else:
            st.info("Generate HTML first")
    
    with col3:
        if 'md_report' in st.session_state:
            st.download_button(
                label="📝 Download Markdown",
                data=st.session_state['md_report'],
                file_name="automl_report.md",
                mime="text/markdown"
            )
        else:
            st.info("Generate Markdown first")
    
    # Preview HTML report
    if 'html_report' in st.session_state:
        st.markdown("---")
        st.subheader("👀 Report Preview")
        
        with st.expander("Preview HTML Report", expanded=False):
            st.components.v1.html(st.session_state['html_report'], height=600, scrolling=True)
