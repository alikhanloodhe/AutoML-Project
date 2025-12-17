"""
Hyperparameter search spaces for all classifiers.
"""

HYPERPARAMETERS = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500]
    },
    
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    },
    
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    },
    
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },
    
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4]
    },
    
    'Rule-Based Classifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }
}

# Reduced hyperparameters for faster training (Beginner mode)
HYPERPARAMETERS_FAST = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'max_iter': [200]
    },
    
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean']
    },
    
    'Decision Tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    
    'Naive Bayes': {
        'var_smoothing': [1e-9, 1e-7]
    },
    
    'Random Forest': {
        'n_estimators': [50, 100],
        'criterion': ['gini'],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    },
    
    'SVM': {
        'C': [1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale']
    },
    
    'Rule-Based Classifier': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best']
    }
}

# Model descriptions for UI
MODEL_DESCRIPTIONS = {
    'Logistic Regression': 'A linear model for classification that uses logistic function. Best for linearly separable data.',
    'KNN': 'K-Nearest Neighbors classifies based on the majority class of k closest training examples.',
    'Decision Tree': 'A tree-based model that makes decisions based on feature thresholds. Highly interpretable.',
    'Naive Bayes': 'A probabilistic classifier based on Bayes theorem with strong independence assumptions.',
    'Random Forest': 'An ensemble of decision trees that reduces overfitting through averaging.',
    'SVM': 'Support Vector Machine finds an optimal hyperplane to separate classes.',
    'Rule-Based Classifier': 'A simple decision stump that makes predictions based on a single decision rule.'
}
