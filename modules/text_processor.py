"""
Text processing and NLP module for AutoML pipeline.
Handles tokenization, text cleaning, and feature extraction for text data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# Try to import NLTK and handle if not installed
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data silently
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False


def clean_text(text, lowercase=True, remove_punctuation=True, remove_numbers=False, 
               remove_extra_spaces=True):
    """
    Clean text data with various options.
    
    Args:
        text: Text string to clean
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        remove_numbers: Remove numeric characters
        remove_extra_spaces: Remove extra whitespace
    
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers if requested
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation if requested
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text, language='english'):
    """
    Remove stopwords from text.
    
    Args:
        text: Text string
        language: Language for stopwords (default: 'english')
    
    Returns:
        Text without stopwords
    """
    if not NLTK_AVAILABLE:
        return text
    
    if pd.isna(text) or text == "":
        return ""
    
    try:
        stop_words = set(stopwords.words(language))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    except Exception:
        return text


def stem_text(text):
    """
    Apply stemming to text.
    
    Args:
        text: Text string
    
    Returns:
        Stemmed text
    """
    if not NLTK_AVAILABLE:
        return text
    
    if pd.isna(text) or text == "":
        return ""
    
    try:
        stemmer = PorterStemmer()
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    except Exception:
        return text


def lemmatize_text(text):
    """
    Apply lemmatization to text.
    
    Args:
        text: Text string
    
    Returns:
        Lemmatized text
    """
    if not NLTK_AVAILABLE:
        return text
    
    if pd.isna(text) or text == "":
        return ""
    
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    except Exception:
        return text


def preprocess_text_column(series, remove_stopwords_flag=True, use_stemming=False, 
                           use_lemmatization=False, lowercase=True, 
                           remove_punctuation=True, remove_numbers=False):
    """
    Preprocess a text column with various options.
    
    Args:
        series: Pandas Series containing text
        remove_stopwords_flag: Whether to remove stopwords
        use_stemming: Whether to apply stemming
        use_lemmatization: Whether to apply lemmatization
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation
        remove_numbers: Remove numbers
    
    Returns:
        Preprocessed Series
    """
    # Clean text
    cleaned = series.apply(lambda x: clean_text(
        x, 
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers
    ))
    
    # Remove stopwords
    if remove_stopwords_flag and NLTK_AVAILABLE:
        cleaned = cleaned.apply(remove_stopwords)
    
    # Stemming
    if use_stemming and NLTK_AVAILABLE:
        cleaned = cleaned.apply(stem_text)
    
    # Lemmatization
    if use_lemmatization and NLTK_AVAILABLE:
        cleaned = cleaned.apply(lemmatize_text)
    
    return cleaned


def vectorize_text_tfidf(train_series, test_series=None, max_features=100, 
                         ngram_range=(1, 1), min_df=1, max_df=1.0):
    """
    Vectorize text using TF-IDF.
    
    Args:
        train_series: Training text data
        test_series: Test text data (optional)
        max_features: Maximum number of features
        ngram_range: N-gram range (e.g., (1,1) for unigrams, (1,2) for unigrams+bigrams)
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    
    Returns:
        Tuple of (train_vectors, test_vectors, vectorizer, feature_names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
    )
    
    # Fit and transform training data
    train_vectors = vectorizer.fit_transform(train_series.fillna(''))
    
    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    train_df = pd.DataFrame(
        train_vectors.toarray(), 
        columns=feature_names,
        index=train_series.index
    )
    
    # Transform test data if provided
    test_df = None
    if test_series is not None:
        test_vectors = vectorizer.transform(test_series.fillna(''))
        test_df = pd.DataFrame(
            test_vectors.toarray(),
            columns=feature_names,
            index=test_series.index
        )
    
    return train_df, test_df, vectorizer, feature_names


def vectorize_text_count(train_series, test_series=None, max_features=100,
                         ngram_range=(1, 1), min_df=1, max_df=1.0):
    """
    Vectorize text using Count Vectorization (Bag of Words).
    
    Args:
        train_series: Training text data
        test_series: Test text data (optional)
        max_features: Maximum number of features
        ngram_range: N-gram range
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    
    Returns:
        Tuple of (train_vectors, test_vectors, vectorizer, feature_names)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    
    # Fit and transform training data
    train_vectors = vectorizer.fit_transform(train_series.fillna(''))
    
    # Convert to DataFrame
    feature_names = vectorizer.get_feature_names_out()
    train_df = pd.DataFrame(
        train_vectors.toarray(),
        columns=feature_names,
        index=train_series.index
    )
    
    # Transform test data if provided
    test_df = None
    if test_series is not None:
        test_vectors = vectorizer.transform(test_series.fillna(''))
        test_df = pd.DataFrame(
            test_vectors.toarray(),
            columns=feature_names,
            index=test_series.index
        )
    
    return train_df, test_df, vectorizer, feature_names


def get_text_statistics(series):
    """
    Get comprehensive statistics for a text column.
    
    Args:
        series: Pandas Series containing text
    
    Returns:
        Dictionary with text statistics
    """
    non_null = series.dropna().astype(str)
    
    if len(non_null) == 0:
        return None
    
    # Character statistics
    char_lengths = non_null.str.len()
    
    # Word statistics
    word_counts = non_null.str.split().str.len()
    
    # Vocabulary
    all_words = ' '.join(non_null).split()
    unique_words = len(set(all_words))
    
    stats = {
        'total_documents': len(non_null),
        'total_characters': int(char_lengths.sum()),
        'total_words': int(word_counts.sum()),
        'unique_words': unique_words,
        'avg_char_length': round(char_lengths.mean(), 2),
        'avg_word_count': round(word_counts.mean(), 2),
        'min_char_length': int(char_lengths.min()),
        'max_char_length': int(char_lengths.max()),
        'min_word_count': int(word_counts.min()),
        'max_word_count': int(word_counts.max()),
    }
    
    return stats


def extract_text_features(series):
    """
    Extract additional features from text (length, word count, etc.).
    
    Args:
        series: Pandas Series containing text
    
    Returns:
        DataFrame with extracted features
    """
    features = pd.DataFrame(index=series.index)
    
    text_series = series.fillna('').astype(str)
    
    # Length features
    features['text_char_count'] = text_series.str.len()
    features['text_word_count'] = text_series.str.split().str.len()
    
    # Special character counts
    features['text_punctuation_count'] = text_series.str.count(r'[^\w\s]')
    features['text_uppercase_count'] = text_series.str.count(r'[A-Z]')
    features['text_digit_count'] = text_series.str.count(r'\d')
    
    # Average word length
    features['text_avg_word_length'] = features['text_char_count'] / features['text_word_count'].replace(0, 1)
    
    return features
