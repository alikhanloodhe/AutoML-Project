"""
Exploratory Data Analysis module.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from modules.utils import (
    detect_column_types, calculate_skewness_kurtosis,
    close_figure, get_color_palette, get_text_summary
)

# Try to import wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


def analyze_missing_values(df):
    """Analyze and visualize missing values."""
    st.subheader("Missing Value Analysis")
    
    missing_data = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_data.append({
            'Column': col,
            'Missing Count': missing_count,
            'Missing %': round(missing_pct, 2)
        })
    
    missing_df = pd.DataFrame(missing_data)
    missing_df = missing_df.sort_values('Missing %', ascending=False)
    
    # Total missing
    total_cells = len(df) * len(df.columns)
    total_missing = df.isnull().sum().sum()
    total_missing_pct = (total_missing / total_cells) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Missing Values", f"{total_missing:,}")
    with col2:
        st.metric("Total Missing %", f"{total_missing_pct:.2f}%")
    with col3:
        cols_with_missing = (missing_df['Missing Count'] > 0).sum()
        st.metric("Columns with Missing", f"{cols_with_missing}")
    
    # Table
    cols_missing = missing_df[missing_df['Missing Count'] > 0]
    if len(cols_missing) > 0:
        st.dataframe(cols_missing, use_container_width=True, hide_index=True)
        
        # Heatmap visualization
        with st.expander("Missing Values Heatmap", expanded=False):
            if len(cols_missing) > 0 and len(cols_missing) <= 30:
                fig, ax = plt.subplots(figsize=(12, 6))
                missing_matrix = df[cols_missing['Column'].tolist()].isnull()
                sns.heatmap(missing_matrix.head(100), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
                ax.set_title('Missing Values Pattern (First 100 rows)')
                ax.set_xlabel('Columns')
                st.pyplot(fig)
                close_figure(fig)
    else:
        st.success("No missing values found in the dataset!")
    
    return missing_df


def detect_outliers_iqr(series):
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_below = (series < lower_bound).sum()
    outliers_above = (series > upper_bound).sum()
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_below': outliers_below,
        'outliers_above': outliers_above,
        'total_outliers': outliers_below + outliers_above
    }


def detect_outliers_zscore(series, threshold=3):
    """Detect outliers using Z-score method."""
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return {'total_outliers': 0, 'outlier_indices': []}
    
    z_scores = np.abs(stats.zscore(clean_series))
    outlier_mask = z_scores > threshold
    
    return {
        'total_outliers': outlier_mask.sum(),
        'outlier_indices': clean_series[outlier_mask].index.tolist()
    }


def analyze_outliers(df, column_types):
    """Analyze and visualize outliers in numerical columns."""
    st.subheader("Outlier Detection")
    
    if not column_types['numerical']:
        st.info("No numerical columns available for outlier detection.")
        return None
    
    outlier_data = []
    
    for col in column_types['numerical']:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        
        iqr_results = detect_outliers_iqr(series)
        zscore_results = detect_outliers_zscore(series)
        
        outlier_pct = (iqr_results['total_outliers'] / len(series)) * 100
        
        outlier_data.append({
            'Column': col,
            'IQR Outliers': iqr_results['total_outliers'],
            'Z-Score Outliers': zscore_results['total_outliers'],
            'Total (IQR)': iqr_results['total_outliers'],
            'Outlier %': round(outlier_pct, 2),
            'Lower Bound': round(iqr_results['lower_bound'], 2),
            'Upper Bound': round(iqr_results['upper_bound'], 2)
        })
    
    if outlier_data:
        outlier_df = pd.DataFrame(outlier_data)
        outlier_df = outlier_df.sort_values('Outlier %', ascending=False)
        
        # Summary
        total_outliers = outlier_df['Total (IQR)'].sum()
        cols_with_outliers = (outlier_df['Total (IQR)'] > 0).sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Outliers Detected (IQR)", f"{total_outliers:,}")
        with col2:
            st.metric("Columns with Outliers", f"{cols_with_outliers}")
        
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        # Box plots for top columns with outliers
        with st.expander("Box Plots", expanded=False):
            cols_to_plot = outlier_df[outlier_df['Total (IQR)'] > 0]['Column'].head(6).tolist()
            
            if cols_to_plot:
                n_cols = min(3, len(cols_to_plot))
                n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                for idx, col in enumerate(cols_to_plot):
                    if idx < len(axes):
                        axes[idx].boxplot(df[col].dropna(), vert=True)
                        axes[idx].set_title(f'{col}')
                        axes[idx].set_ylabel('Value')
                
                # Hide empty subplots
                for idx in range(len(cols_to_plot), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                close_figure(fig)
        
        return outlier_df
    
    st.info("No outliers detected in numerical columns.")
    return None


def analyze_correlations(df, column_types, target_col=None):
    """Analyze and visualize correlations."""
    st.subheader("Correlation Analysis")
    
    numerical_cols = column_types['numerical']
    
    if len(numerical_cols) < 2:
        st.info("Need at least 2 numerical columns for correlation analysis.")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Heatmap
    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "High Correlations", "Target Correlations"])
    
    with tab1:
        # Limit columns for better visualization
        display_cols = numerical_cols[:20] if len(numerical_cols) > 20 else numerical_cols
        display_corr = df[display_cols].corr()
        
        fig = px.imshow(
            display_corr,
            labels=dict(color="Correlation"),
            x=display_cols,
            y=display_cols,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            zmin=-1,
            zmax=1,
            text_auto='.2f'
        )
        fig.update_traces(textfont_size=10)
        fig.update_layout(title='Correlation Matrix', height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': round(corr_val, 4)
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.warning(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8)")
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
        else:
            st.success("No highly correlated feature pairs found (|r| > 0.8)")
    
    with tab3:
        if target_col and target_col in numerical_cols:
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            
            target_corr_df = pd.DataFrame({
                'Feature': target_corr.index,
                'Correlation with Target': target_corr.values.round(4)
            })
            
            st.dataframe(target_corr_df, use_container_width=True, hide_index=True)
            
            # Bar chart
            fig = px.bar(
                target_corr_df.head(15),
                x='Correlation with Target',
                y='Feature',
                orientation='h',
                color='Correlation with Target',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(title=f'Top 15 Correlations with {target_col}', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select a numerical target column to see target correlations.")
    
    return corr_matrix


def analyze_distributions(df, column_types):
    """Analyze and visualize feature distributions."""
    st.subheader("Distribution Analysis")
    
    tab1, tab2 = st.tabs(["Numerical Distributions", "Categorical Distributions"])
    
    with tab1:
        if column_types['numerical']:
            selected_num_cols = st.multiselect(
                "Select numerical columns to visualize:",
                column_types['numerical'],
                default=column_types['numerical'][:6]
            )
            
            if selected_num_cols:
                for col in selected_num_cols:
                    with st.expander(f"{col}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram with KDE
                            fig = px.histogram(
                                df, x=col, 
                                marginal='box',
                                title=f'Distribution of {col}'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Statistics
                            series = df[col].dropna()
                            skew, kurt = calculate_skewness_kurtosis(series)
                            
                            stats_data = {
                                'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
                                'Value': [
                                    round(series.mean(), 4),
                                    round(series.median(), 4),
                                    round(series.std(), 4),
                                    skew if skew else 'N/A',
                                    kurt if kurt else 'N/A'
                                ]
                            }
                            st.table(pd.DataFrame(stats_data))
                            
                            # Normality indicator
                            if skew is not None:
                                if abs(skew) < 0.5:
                                    st.success("Approximately symmetric")
                                elif skew > 0.5:
                                    st.warning("Right-skewed (positive skew)")
                                else:
                                    st.warning("Left-skewed (negative skew)")
        else:
            st.info("No numerical columns available.")
    
    with tab2:
        if column_types['categorical']:
            selected_cat_cols = st.multiselect(
                "Select categorical columns to visualize:",
                column_types['categorical'],
                default=column_types['categorical'][:4]
            )
            
            if selected_cat_cols:
                for col in selected_cat_cols:
                    with st.expander(f"{col}", expanded=False):
                        value_counts = df[col].value_counts().head(20)
                        
                        fig = px.bar(
                            x=value_counts.index.astype(str),
                            y=value_counts.values,
                            labels={'x': col, 'y': 'Count'},
                            title=f'Distribution of {col} (Top 20)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stats
                        st.caption(f"Unique values: {df[col].nunique()} | Mode: {df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'}")
        else:
            st.info("No categorical columns available.")


def analyze_multivariate(df, column_types, target_col=None):
    """Multivariate analysis including pairplots."""
    st.subheader("Multivariate Analysis")
    
    if not target_col:
        st.info("Select a target variable to enable multivariate analysis.")
        return
    
    numerical_cols = column_types['numerical']
    
    if len(numerical_cols) < 2:
        st.info("Need at least 2 numerical columns for multivariate analysis.")
        return
    
    # Get top correlated features with target (if numerical)
    if target_col in numerical_cols:
        corr_with_target = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top_features = corr_with_target.head(6).index.tolist()
        if target_col in top_features:
            top_features.remove(target_col)
            top_features = top_features[:5]
    else:
        top_features = numerical_cols[:5]
    
    if top_features:
        st.markdown("#### Top Correlated Features with Target")
        
        selected_features = st.multiselect(
            "Select features for pairplot:",
            numerical_cols,
            default=top_features[:4]
        )
        
        if selected_features and len(selected_features) >= 2:
            with st.spinner("Generating pairplot..."):
                if target_col in df.columns:
                    plot_df = df[selected_features + [target_col]].dropna()
                    
                    if len(plot_df) > 500:
                        plot_df = plot_df.sample(n=500, random_state=42)
                    
                    fig = px.scatter_matrix(
                        plot_df,
                        dimensions=selected_features,
                        color=target_col,
                        title='Feature Relationships by Target Class'
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Grouped statistics
        st.markdown("#### Grouped Statistics by Target")
        if target_col in df.columns:
            grouped_stats = df.groupby(target_col)[numerical_cols].mean()
            st.dataframe(grouped_stats.round(3), use_container_width=True)


def analyze_text_data(df, column_types):
    """Analyze and visualize text data columns."""
    st.subheader("Text Data Analysis")
    
    text_cols = column_types.get('text', [])
    
    if not text_cols:
        st.info("No text columns detected in the dataset.")
        return
    
    st.markdown(f"**Text Columns Detected:** {len(text_cols)}")
    
    for col in text_cols:
        with st.expander(f"📝 {col}", expanded=True):
            # Get text statistics
            text_summary = get_text_summary(df, col)
            
            if text_summary:
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", f"{text_summary['count']:,}")
                with col2:
                    st.metric("Unique Values", f"{text_summary['unique']:,}")
                with col3:
                    st.metric("Avg Length (chars)", f"{text_summary['avg_length']:.0f}")
                with col4:
                    st.metric("Avg Words", f"{text_summary['avg_words']:.1f}")
                
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Min Length", f"{text_summary['min_length']}")
                with col6:
                    st.metric("Max Length", f"{text_summary['max_length']}")
                with col7:
                    st.metric("Missing", f"{text_summary['missing']:,} ({text_summary['missing_pct']:.1f}%)")
                
                # Text length distribution
                st.markdown("##### Text Length Distribution")
                text_series = df[col].dropna().astype(str)
                lengths = text_series.str.len()
                
                fig = px.histogram(
                    x=lengths,
                    nbins=50,
                    title=f'Character Length Distribution - {col}',
                    labels={'x': 'Character Count', 'y': 'Frequency'}
                )
                fig.update_layout(
                    showlegend=False,
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Word cloud
                if WORDCLOUD_AVAILABLE and len(text_series) > 0:
                    st.markdown("##### Word Cloud")
                    
                    try:
                        # Combine all text
                        all_text = ' '.join(text_series.head(1000))  # Limit to first 1000 for performance
                        
                        if len(all_text.strip()) > 0:
                            # Generate word cloud
                            wordcloud = WordCloud(
                                width=800,
                                height=400,
                                background_color='white',
                                colormap='viridis',
                                max_words=100,
                                relative_scaling=0.5,
                                min_font_size=10
                            ).generate(all_text)
                            
                            # Display word cloud
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(f'Word Cloud - {col}', fontsize=14, pad=10)
                            st.pyplot(fig)
                            close_figure(fig)
                        else:
                            st.info("Not enough text data to generate word cloud.")
                    
                    except Exception as e:
                        st.warning(f"Could not generate word cloud: {str(e)}")
                else:
                    if not WORDCLOUD_AVAILABLE:
                        st.info("WordCloud library not available. Install it to see word clouds.")
                
                # Sample texts
                st.markdown("##### Sample Texts")
                sample_texts = text_series.head(5).tolist()
                for i, text in enumerate(sample_texts, 1):
                    # Truncate long texts
                    display_text = text if len(text) <= 200 else text[:200] + "..."
                    st.text(f"{i}. {display_text}")
            
            else:
                st.warning(f"No valid text data found in column '{col}'")


def render_eda_page():
    """Render the complete EDA page with improved UI."""
    st.header("Exploratory Data Analysis")
    
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state['data']
    column_types = st.session_state.get('column_types', detect_column_types(df))
    target_col = st.session_state.get('target_column', None)
    
    # Dataset overview card
    st.markdown("""
    <div style="background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 16px; padding: 20px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{:,}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">Rows</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">Columns</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">Numerical</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">Categorical</div>
            </div>
            <div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{}</div>
                <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">Text</div>
            </div>
        </div>
    </div>
    """.format(len(df), len(df.columns), len(column_types.get('numerical', [])), 
               len(column_types.get('categorical', [])), len(column_types.get('text', []))), 
    unsafe_allow_html=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run EDA analyses
    status_text.text("Analyzing missing values...")
    progress_bar.progress(20)
    missing_df = analyze_missing_values(df)
    st.session_state['missing_analysis'] = missing_df
    
    st.markdown('<div style="height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 32px 0;"></div>', unsafe_allow_html=True)
    
    status_text.text("Detecting outliers...")
    progress_bar.progress(40)
    outlier_df = analyze_outliers(df, column_types)
    st.session_state['outlier_analysis'] = outlier_df
    
    st.markdown('<div style="height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 32px 0;"></div>', unsafe_allow_html=True)
    
    status_text.text("Analyzing correlations...")
    progress_bar.progress(60)
    corr_matrix = analyze_correlations(df, column_types, target_col)
    st.session_state['correlation_matrix'] = corr_matrix
    
    st.markdown('<div style="height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 32px 0;"></div>', unsafe_allow_html=True)
    
    status_text.text("Analyzing distributions...")
    progress_bar.progress(80)
    analyze_distributions(df, column_types)
    
    st.markdown('<div style="height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 32px 0;"></div>', unsafe_allow_html=True)
    
    # Analyze text data if present
    if column_types.get('text', []):
        status_text.text("Analyzing text data...")
        progress_bar.progress(85)
        analyze_text_data(df, column_types)
        
        st.markdown('<div style="height:1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 32px 0;"></div>', unsafe_allow_html=True)
    
    status_text.text("Performing multivariate analysis...")
    progress_bar.progress(90)
    analyze_multivariate(df, column_types, target_col)
    
    progress_bar.progress(100)
    status_text.text("EDA Complete!")
    
    st.session_state['eda_complete'] = True
    
    # Success message with next step
    st.markdown("""
    <div style="background: linear-gradient(145deg, rgba(56, 239, 125, 0.1), rgba(17, 153, 142, 0.1));
                border: 1px solid rgba(56, 239, 125, 0.3); border-radius: 12px; padding: 20px; margin-top: 24px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5rem;"></span>
            <div>
                <div style="font-weight: 600; color: #38ef7d;">Exploratory Data Analysis Complete!</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 4px;">
                    Proceed to Issue Detection to identify and fix data quality problems.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Continue button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Continue to Issue Detection", type="primary", use_container_width=True):
            st.session_state['current_page'] = 'issues'
            st.rerun()
