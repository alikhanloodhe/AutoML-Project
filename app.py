"""
AutoML Classification System - Main Application
A production-ready AutoML web application for binary and multiclass classification.

Team Members:
- Ali Asghar Khan Lodhi (478734)
- Muhammad Saad Akhtar (458102)

CS-245 Machine Learning Course - NUST SEECS
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="AutoML Classifier",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with HCI Principles
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.1);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255, 255, 255, 0.05);
        padding: 12px 16px;
        border-radius: 10px;
        margin: 4px 0;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateX(5px);
    }
    
    /* Main Content */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    h1 {
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        margin: 16px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Feature Card */
    .feature-card {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: var(--primary-gradient);
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 10px 24px;
        color: #ffffff;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-1px);
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
    }
    
    /* Progress */
    .stProgress > div > div > div > div {
        background: var(--primary-gradient);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 32px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 40px 20px;
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 32px;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.7);
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Mode Cards */
    .mode-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
    }
    
    .mode-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
    }
    
    .mode-card.active {
        border-color: #667eea;
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
    }
    
    .mode-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 12px;
    }
    
    .mode-desc {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Step Indicator */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 20px;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 16px;
        width: fit-content;
    }
    
    .step-number {
        background: var(--primary-gradient);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-complete {
        background: rgba(56, 239, 125, 0.15);
        color: #38ef7d;
        border: 1px solid rgba(56, 239, 125, 0.3);
    }
    
    .status-pending {
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 32px 0;
    }
    
    /* Team Card */
    .team-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .team-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    .team-roll {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
    }
    
    /* Hide Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import modules
from modules.data_loader import render_upload_page
from modules.eda import render_eda_page
from modules.issue_detector import render_issue_detection_page
from modules.preprocessor import render_preprocessing_page
from modules.feature_engineering import render_feature_engineering_page
from modules.model_trainer import render_model_training_page
from modules.evaluator import render_model_comparison_page
from modules.report_generator import render_report_generation_page


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data': None,
        'file_name': None,
        'target_column': None,
        'column_types': None,
        'mode': 'Beginner',
        'eda_complete': False,
        'issues_resolved': False,
        'preprocessing_done': False,
        'feature_engineering_done': False,
        'models_trained': False,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'model_results': None,
        'best_model': None,
        'best_model_name': None,
        'apply_smote': False,
        'use_class_weights': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_home_page():
    """Render the home page."""
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">AutoML Classifier</div>
        <div class="hero-subtitle">
            Transform your data into powerful machine learning models with our automated classification system. 
            No coding required.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown("## Choose Your Experience")
    
    col1, col2 = st.columns(2)
    
    current_mode = st.session_state.get('mode', 'Beginner')
    
    with col1:
        beginner_active = "active" if current_mode == "Beginner" else ""
        st.markdown(f"""
        <div class="mode-card {beginner_active}">
            <div class="mode-title">Beginner Mode</div>
            <div class="mode-desc">
                Perfect for newcomers to machine learning.<br><br>
                - Automated preprocessing<br>
                - Smart default settings<br>
                - Guided step-by-step workflow<br>
                - Simplified explanations
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Beginner Mode", key="beginner_btn", use_container_width=True):
            st.session_state['mode'] = 'Beginner'
            st.rerun()
    
    with col2:
        expert_active = "active" if current_mode == "Expert" else ""
        st.markdown(f"""
        <div class="mode-card {expert_active}">
            <div class="mode-title">Expert Mode</div>
            <div class="mode-desc">
                Full control for data scientists.<br><br>
                - Custom preprocessing options<br>
                - Advanced hyperparameter tuning<br>
                - Detailed analytics<br>
                - Complete model control
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Expert Mode", key="expert_btn", use_container_width=True):
            st.session_state['mode'] = 'Expert'
            st.rerun()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Features Section
    st.markdown("## Features")
    
    features = [
        ("Smart Upload", "CSV and Excel support with auto-encoding"),
        ("Auto EDA", "Complete analysis with visualizations"),
        ("Issue Detection", "Find and fix data problems"),
        ("Preprocessing", "Automated data cleaning"),
        ("Feature Engineering", "Smart feature selection"),
        ("7 ML Models", "Train multiple classifiers"),
        ("Comparison", "Interactive model analysis"),
        ("PDF Reports", "Professional documentation")
    ]
    
    cols = st.columns(4)
    for idx, (title, desc) in enumerate(features):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Workflow Section
    st.markdown("## How It Works")
    
    workflow_steps = [
        ("1", "Upload", "Import your dataset"),
        ("2", "Analyze", "Explore your data"),
        ("3", "Clean", "Preprocess and engineer"),
        ("4", "Train", "Build 7 ML models"),
        ("5", "Compare", "Evaluate performance"),
        ("6", "Export", "Download reports")
    ]
    
    cols = st.columns(6)
    for idx, (num, title, desc) in enumerate(workflow_steps):
        with cols[idx]:
            st.markdown(f"""
            <div style="text-align: center; padding: 16px;">
                <div class="step-number" style="width: 40px; height: 40px; margin: 0 auto 12px; font-size: 1.2rem;">
                    {num}
                </div>
                <div style="font-weight: 600; color: #fff; margin-bottom: 4px;">{title}</div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.5);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Team Section
    st.markdown("## Team")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("""
            <div class="team-card">
                <div class="team-name">Ali Asghar Khan Lodhi</div>
                <div class="team-roll">Roll #: 478734</div>
            </div>
            """, unsafe_allow_html=True)
        with tcol2:
            st.markdown("""
            <div class="team-card">
                <div class="team-name">Muhammad Saad Akhtar</div>
                <div class="team-roll">Roll #: 458102</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 32px; color: rgba(255,255,255,0.4); font-size: 0.9rem;">
        CS-245 Machine Learning Course - NUST SEECS - 2024
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar navigation."""
    
    # Logo and title
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0; margin-bottom: 20px;">
        <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin-bottom: 8px;">AutoML</div>
        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5);">Machine Learning Made Easy</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current mode badge
    mode = st.session_state.get('mode', 'Beginner')
    mode_color = "#38ef7d" if mode == "Beginner" else "#667eea"
    
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 12px 16px; border-radius: 10px; 
                margin-bottom: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
        <span style="color: {mode_color}; font-weight: 500;">{mode} Mode</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Section
    st.sidebar.markdown("### Progress")
    
    progress_items = [
        ("Data Loaded", st.session_state.get('data') is not None),
        ("Target Selected", st.session_state.get('target_column') is not None),
        ("EDA Complete", st.session_state.get('eda_complete', False)),
        ("Preprocessed", st.session_state.get('preprocessing_done', False)),
        ("Models Trained", st.session_state.get('models_trained', False))
    ]
    
    for label, done in progress_items:
        status_class = "status-complete" if done else "status-pending"
        check = "[Done]" if done else "[Pending]"
        st.sidebar.markdown(f"""
        <div class="status-badge {status_class}" style="margin: 4px 0; width: 100%; justify-content: flex-start;">
            <span>{check}</span>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="custom-divider" style="margin: 16px 0;"></div>', unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    
    pages = [
        ("Home", "home"),
        ("Upload Dataset", "upload"),
        ("Exploratory Analysis", "eda"),
        ("Issue Detection", "issues"),
        ("Preprocessing", "preprocessing"),
        ("Feature Engineering", "features"),
        ("Model Training", "training"),
        ("Model Comparison", "comparison"),
        ("Generate Report", "report")
    ]
    
    selected_page = st.sidebar.radio(
        "Select Page",
        [page[0] for page in pages],
        label_visibility="collapsed"
    )
    
    page_key = next((p[1] for p in pages if p[0] == selected_page), "home")
    
    st.sidebar.markdown('<div class="custom-divider" style="margin: 16px 0;"></div>', unsafe_allow_html=True)
    
    # Quick actions
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Dataset info
    if st.session_state.get('data') is not None:
        st.sidebar.markdown('<div class="custom-divider" style="margin: 16px 0;"></div>', unsafe_allow_html=True)
        st.sidebar.markdown("### Dataset Info")
        df = st.session_state['data']
        
        st.sidebar.markdown(f"""
        <div class="glass-card" style="padding: 16px; margin: 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: rgba(255,255,255,0.6);">Rows</span>
                <span style="font-weight: 600;">{len(df):,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: rgba(255,255,255,0.6);">Columns</span>
                <span style="font-weight: 600;">{len(df.columns)}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: rgba(255,255,255,0.6);">Target</span>
                <span style="font-weight: 600;">{st.session_state.get('target_column', 'Not set')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return page_key


def main():
    """Main application entry point."""
    init_session_state()
    
    page_key = render_sidebar()
    
    # Page routing
    if page_key == "home":
        render_home_page()
    
    elif page_key == "upload":
        st.markdown('<div class="step-indicator"><div class="step-number">1</div>Upload Dataset</div>', unsafe_allow_html=True)
        render_upload_page()
    
    elif page_key == "eda":
        st.markdown('<div class="step-indicator"><div class="step-number">2</div>Exploratory Data Analysis</div>', unsafe_allow_html=True)
        if st.session_state.get('data') is not None:
            render_eda_page()
        else:
            st.warning("Please upload a dataset first!")
    
    elif page_key == "issues":
        st.markdown('<div class="step-indicator"><div class="step-number">3</div>Issue Detection</div>', unsafe_allow_html=True)
        if st.session_state.get('data') is not None:
            render_issue_detection_page()
        else:
            st.warning("Please upload a dataset first!")
    
    elif page_key == "preprocessing":
        st.markdown('<div class="step-indicator"><div class="step-number">4</div>Data Preprocessing</div>', unsafe_allow_html=True)
        if st.session_state.get('data') is not None:
            render_preprocessing_page()
        else:
            st.warning("Please upload a dataset first!")
    
    elif page_key == "features":
        st.markdown('<div class="step-indicator"><div class="step-number">5</div>Feature Engineering</div>', unsafe_allow_html=True)
        if st.session_state.get('preprocessing_done', False):
            render_feature_engineering_page()
        else:
            st.warning("Please complete preprocessing first!")
    
    elif page_key == "training":
        st.markdown('<div class="step-indicator"><div class="step-number">6</div>Model Training</div>', unsafe_allow_html=True)
        if st.session_state.get('preprocessing_done', False):
            render_model_training_page()
        else:
            st.warning("Please complete preprocessing first!")
    
    elif page_key == "comparison":
        st.markdown('<div class="step-indicator"><div class="step-number">7</div>Model Comparison</div>', unsafe_allow_html=True)
        if st.session_state.get('models_trained', False):
            render_model_comparison_page()
        else:
            st.warning("Please train models first!")
    
    elif page_key == "report":
        st.markdown('<div class="step-indicator"><div class="step-number">8</div>Generate Report</div>', unsafe_allow_html=True)
        if st.session_state.get('models_trained', False):
            render_report_generation_page()
        else:
            st.warning("Please train models first!")


if __name__ == "__main__":
    main()
