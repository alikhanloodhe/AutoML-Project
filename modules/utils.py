"""
Utility functions for the AutoML application.
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import gc
import sys
from scipy import stats


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def format_bytes(size_bytes):
    """Convert bytes to human-readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def detect_column_types(df):
    """
    Detect and categorize column types using smart heuristics.
    
    Returns:
        dict with 'numerical', 'categorical', 'datetime', 'boolean', 'text' keys
    """
    column_types = {
        'numerical': [],
        'categorical': [],
        'datetime': [],
        'boolean': [],
        'text': []
    }
    
    # Identifier keywords that suggest a column is an ID/name, not meaningful text
    identifier_keywords = ['id', 'name', 'key', 'code', 'index', 'identifier', 'uuid', 'guid']
    
    # Text-suggesting column names (common in datasets)
    text_keywords = ['message', 'text', 'review', 'comment', 'description', 'body', 'content', 
                     'tweet', 'post', 'email', 'subject', 'title', 'summary', 'note']
    
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            column_types['boolean'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types['numerical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
        else:
            # Try to parse as datetime
            try:
                pd.to_datetime(df[col], errors='raise')
                column_types['datetime'].append(col)
            except:
                # Distinguish between categorical, text, and identifiers
                non_null = df[col].dropna().astype(str)
                if len(non_null) == 0:
                    column_types['categorical'].append(col)
                    continue
                
                # Calculate text statistics
                avg_length = non_null.str.len().mean()
                median_length = non_null.str.len().median()
                max_length = non_null.str.len().max()
                unique_ratio = df[col].nunique() / len(df)
                n_unique = df[col].nunique()
                
                # Check column name for hints
                col_lower = col.lower()
                is_likely_identifier = any(keyword in col_lower for keyword in identifier_keywords)
                suggests_text = any(keyword in col_lower for keyword in text_keywords)
                
                # Count words in sample entries (text usually has multiple words)
                sample_size = min(100, len(non_null))
                sample = non_null.sample(sample_size, random_state=42)
                avg_word_count = sample.str.split().str.len().mean()
                
                # Check for spaces (text usually has spaces between words)
                has_spaces_ratio = non_null.str.contains(' ', regex=False).sum() / len(non_null)
                
                # Text characteristics (multiple words + spaces indicates real text, not IDs)
                has_text_characteristics = (avg_word_count > 2 and has_spaces_ratio > 0.5) or avg_length > 40
                
                # SMART TEXT DETECTION STRATEGY:
                # Priority 1: Column name suggests text content
                if suggests_text and avg_length > 20:
                    column_types['text'].append(col)
                
                # Priority 2: Definite text - has multiple words and spaces (even if highly unique)
                elif has_text_characteristics and avg_word_count > 3:
                    column_types['text'].append(col)
                
                # Priority 3: Identifier detection (only if NO text characteristics)
                elif is_likely_identifier and not has_text_characteristics:
                    column_types['categorical'].append(col)
                
                # Priority 3: Identifier detection (only if NO text characteristics)
                elif is_likely_identifier and not has_text_characteristics:
                    column_types['categorical'].append(col)
                
                # Priority 4: High unique ratio (>90%) with short strings = likely IDs
                elif unique_ratio > 0.9 and avg_length < 30 and avg_word_count < 2:
                    column_types['categorical'].append(col)
                
                # Priority 5: Long text with multiple words (messages, reviews, etc.)
                elif avg_length > 40 and avg_word_count > 3:
                    column_types['text'].append(col)
                
                # Priority 6: Moderate length text with spaces (sentences)
                elif avg_length > 25 and has_spaces_ratio > 0.7:
                    column_types['text'].append(col)
                
                # Priority 7: High unique ratio with reasonable length + text features (unique text content)
                elif unique_ratio > 0.5 and avg_length > 30 and avg_word_count > 2:
                    column_types['text'].append(col)
                
                # Priority 7: High unique ratio with reasonable length + text features (unique text content)
                elif unique_ratio > 0.5 and avg_length > 30 and avg_word_count > 2:
                    column_types['text'].append(col)
                
                # Priority 8: Very few unique values = categorical
                elif n_unique <= 20:
                    column_types['categorical'].append(col)
                
                # Priority 9: Short strings with low unique ratio = categorical
                elif avg_length < 25 and unique_ratio < 0.5:
                    column_types['categorical'].append(col)
                
                # Priority 10: If unclear but has moderate length and spaces, lean towards text
                elif avg_length > 20 and has_spaces_ratio > 0.5 and avg_word_count > 2:
                    column_types['text'].append(col)
                
                # Final fallback: categorical
                else:
                    column_types['categorical'].append(col)
    
    return column_types


def calculate_skewness_kurtosis(series):
    """Calculate skewness and kurtosis for a numerical series."""
    try:
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return None, None
        skewness = stats.skew(clean_series)
        kurtosis = stats.kurtosis(clean_series)
        return round(skewness, 4), round(kurtosis, 4)
    except:
        return None, None


def get_numerical_summary(df, column):
    """Get comprehensive summary statistics for a numerical column."""
    series = df[column].dropna()
    
    if len(series) == 0:
        return None
    
    skew, kurt = calculate_skewness_kurtosis(series)
    
    summary = {
        'count': int(series.count()),
        'mean': round(series.mean(), 4),
        'std': round(series.std(), 4),
        'min': round(series.min(), 4),
        '25%': round(series.quantile(0.25), 4),
        '50%': round(series.quantile(0.50), 4),
        '75%': round(series.quantile(0.75), 4),
        'max': round(series.max(), 4),
        'skewness': skew,
        'kurtosis': kurt,
        'unique': int(series.nunique())
    }
    
    return summary


def get_categorical_summary(df, column):
    """Get summary statistics for a categorical column."""
    series = df[column]
    
    value_counts = series.value_counts()
    top_5 = value_counts.head(5).to_dict()
    
    summary = {
        'count': int(series.count()),
        'unique': int(series.nunique()),
        'mode': series.mode().iloc[0] if len(series.mode()) > 0 else None,
        'missing': int(series.isna().sum()),
        'missing_pct': round(series.isna().mean() * 100, 2),
        'top_5_values': top_5
    }
    
    return summary


def get_text_summary(df, column):
    """Get summary statistics for a text column."""
    series = df[column].dropna().astype(str)
    
    if len(series) == 0:
        return None
    
    word_counts = series.str.split().str.len()
    char_counts = series.str.len()
    
    summary = {
        'count': int(df[column].count()),
        'unique': int(df[column].nunique()),
        'missing': int(df[column].isna().sum()),
        'missing_pct': round(df[column].isna().mean() * 100, 2),
        'avg_length': round(char_counts.mean(), 2),
        'min_length': int(char_counts.min()),
        'max_length': int(char_counts.max()),
        'avg_words': round(word_counts.mean(), 2),
        'total_chars': int(char_counts.sum())
    }
    
    return summary


def save_figure_to_bytes(fig):
    """Save a matplotlib figure to bytes for PDF/report generation."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


def clean_memory():
    """Clean up memory by running garbage collection."""
    gc.collect()


def close_figure(fig):
    """Safely close a matplotlib figure to free memory."""
    plt.close(fig)
    clean_memory()


def format_percentage(value, decimals=2):
    """Format a decimal value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def get_class_distribution(series):
    """Get class distribution as DataFrame."""
    counts = series.value_counts()
    percentages = series.value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({
        'Class': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values.round(2)
    })
    
    return dist_df


def is_imbalanced(series, threshold=0.2):
    """
    Check if classes are imbalanced.
    
    Args:
        series: Target variable series
        threshold: Minimum fraction for any class (default 20%)
    
    Returns:
        Tuple of (is_imbalanced, minority_class, minority_pct, severity)
        severity: 'severe' (<10%), 'moderate' (10-20%), 'balanced' (>20%)
    """
    percentages = series.value_counts(normalize=True)
    min_pct = percentages.min()
    minority_class = percentages.idxmin()
    min_pct_value = min_pct * 100
    
    # Determine severity
    if min_pct_value < 10:
        severity = 'severe'
    elif min_pct_value < 20:
        severity = 'moderate'
    else:
        severity = 'balanced'
    
    is_imb = min_pct < threshold
    
    return is_imb, minority_class, min_pct_value, severity


def detect_potential_targets(df, max_unique=20, max_unique_ratio=0.05):
    """
    Detect columns that could be potential target variables.
    
    Args:
        df: DataFrame
        max_unique: Maximum number of unique values
        max_unique_ratio: Maximum ratio of unique values to total rows
    
    Returns:
        List of potential target column names
    """
    potential_targets = []
    n_rows = len(df)
    
    for col in df.columns:
        n_unique = df[col].nunique()
        unique_ratio = n_unique / n_rows
        
        # Skip columns with mostly unique values (likely ID columns)
        if n_unique > 1 and n_unique <= max_unique and unique_ratio <= max_unique_ratio:
            potential_targets.append(col)
    
    return potential_targets


def create_styled_metric(label, value, delta=None, delta_color="normal"):
    """Create a styled metric display."""
    return st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def display_dataframe_with_download(df, filename="data.csv", key=None):
    """Display a dataframe with download button."""
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key
    )


def get_color_palette(n_colors):
    """Get a color palette for charts."""
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    return colors[:n_colors]


def truncate_string(s, max_length=50):
    """Truncate a string to a maximum length."""
    if len(str(s)) > max_length:
        return str(s)[:max_length-3] + "..."
    return str(s)


def validate_dataframe(df):
    """
    Validate a dataframe for ML processing.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if df is None or len(df) == 0:
        errors.append("Dataset is empty")
        return False, errors
    
    if len(df.columns) < 2:
        errors.append("Dataset must have at least 2 columns (features + target)")
    
    if len(df) < 10:
        errors.append("Dataset must have at least 10 rows")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        errors.append("Dataset has duplicate column names")
    
    return len(errors) == 0, errors
