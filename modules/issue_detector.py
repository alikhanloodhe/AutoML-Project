"""
Data quality issue detection module.
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.utils import detect_column_types, is_imbalanced


class Issue:
    """Class representing a data quality issue."""
    
    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    
    def __init__(self, severity, issue_type, column, message, suggestion, actions):
        self.severity = severity
        self.issue_type = issue_type
        self.column = column
        self.message = message
        self.suggestion = suggestion
        self.actions = actions  # List of (action_name, action_key)
        self.resolved = False
        self.reviewed = False
        self.action_taken = None
    
    def get_id(self):
        """Get unique identifier for this issue."""
        return f"{self.issue_type}_{self.column}"


def detect_missing_value_issues(df):
    """Detect missing value issues."""
    issues = []
    
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        
        if missing_pct > 50:
            issues.append(Issue(
                severity=Issue.CRITICAL,
                issue_type="High Missing Values",
                column=col,
                message=f"Column '{col}' has {missing_pct:.1f}% missing values",
                suggestion="Consider dropping this column or using advanced imputation",
                actions=[("Drop Column", f"drop_{col}"), ("Keep & Impute", f"impute_{col}")]
            ))
        elif missing_pct > 5:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            suggestion = "Impute with mean/median/KNN" if is_numeric else "Impute with mode/constant value"
            
            issues.append(Issue(
                severity=Issue.MODERATE,
                issue_type="Missing Values",
                column=col,
                message=f"Column '{col}' has {missing_pct:.1f}% missing values",
                suggestion=suggestion,
                actions=[("Impute", f"impute_{col}"), ("Drop Column", f"drop_{col}"), ("Skip", f"skip_{col}")]
            ))
    
    return issues


def detect_target_issues(df, target_col):
    """Detect target variable issues."""
    issues = []
    
    if target_col is None:
        return issues
    
    # Missing values in target
    target_missing = df[target_col].isnull().sum()
    if target_missing > 0:
        issues.append(Issue(
            severity=Issue.CRITICAL,
            issue_type="Target Missing Values",
            column=target_col,
            message=f"Target variable has {target_missing} missing values",
            suggestion="Remove rows with missing target values",
            actions=[("Remove Rows", "remove_target_missing")]
        ))
    
    # Class imbalance
    imbalanced, minority_class, minority_pct, severity = is_imbalanced(df[target_col].dropna())
    
    if imbalanced:
        if minority_pct < 5:
            issues.append(Issue(
                severity=Issue.CRITICAL,
                issue_type="Severe Class Imbalance",
                column=target_col,
                message=f"Severe class imbalance detected. Class '{minority_class}': {minority_pct:.1f}%",
                suggestion="Consider SMOTE, class weights, or collecting more data",
                actions=[("Apply SMOTE", "apply_smote"), ("Use Class Weights", "use_weights")]
            ))
        else:
            issues.append(Issue(
                severity=Issue.MODERATE,
                issue_type="Class Imbalance",
                column=target_col,
                message=f"Class imbalance detected. Class '{minority_class}': {minority_pct:.1f}%",
                suggestion="Apply class weights or SMOTE",
                actions=[("Apply Weights", "use_weights"), ("Apply SMOTE", "apply_smote"), ("Continue", "skip_imbalance")]
            ))
    
    return issues


def detect_outlier_issues(df, column_types):
    """Detect outlier issues in numerical columns."""
    issues = []
    
    for col in column_types.get('numerical', []):
        series = df[col].dropna()
        if len(series) < 4:
            continue
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        outlier_pct = (outliers / len(series)) * 100
        
        if outlier_pct > 5:
            issues.append(Issue(
                severity=Issue.MODERATE,
                issue_type="Outliers Detected",
                column=col,
                message=f"{outliers} outliers detected in '{col}' ({outlier_pct:.1f}% of data)",
                suggestion="Outliers can skew model performance. Keep if meaningful (e.g., fraud detection)",
                actions=[("Remove Outliers", f"remove_outliers_{col}"), ("Cap at Percentiles", f"cap_{col}"), ("Keep Outliers", f"keep_{col}")]
            ))
    
    return issues


def detect_cardinality_issues(df, column_types):
    """Detect high cardinality issues in categorical columns."""
    issues = []
    
    for col in column_types.get('categorical', []):
        n_unique = df[col].nunique()
        
        if n_unique > 50:
            issues.append(Issue(
                severity=Issue.MODERATE,
                issue_type="High Cardinality",
                column=col,
                message=f"Column '{col}' has {n_unique} unique categories",
                suggestion="High cardinality can cause memory issues with one-hot encoding",
                actions=[("Keep Top 20", f"top20_{col}"), ("Drop Column", f"drop_{col}"), ("Use Target Encoding", f"target_enc_{col}")]
            ))
    
    return issues


def detect_constant_features(df):
    """Detect constant and near-constant features."""
    issues = []
    
    for col in df.columns:
        n_unique = df[col].nunique()
        
        if n_unique == 1:
            issues.append(Issue(
                severity=Issue.MINOR,
                issue_type="Constant Feature",
                column=col,
                message=f"Column '{col}' has constant values (only 1 unique value)",
                suggestion="This feature provides no information for prediction",
                actions=[("Drop Column", f"drop_{col}"), ("Keep", f"keep_{col}")]
            ))
        elif n_unique == 2:
            # Check if near-constant (one value dominates >95%)
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.95:
                dominant_pct = value_counts.iloc[0] * 100
                issues.append(Issue(
                    severity=Issue.MINOR,
                    issue_type="Near-Constant Feature",
                    column=col,
                    message=f"Column '{col}' is {dominant_pct:.1f}% constant",
                    suggestion="Low variance feature may not be informative",
                    actions=[("Drop Column", f"drop_{col}"), ("Keep", f"keep_{col}")]
                ))
    
    return issues


def detect_correlation_issues(df, column_types):
    """Detect highly correlated feature pairs."""
    issues = []
    
    numerical_cols = column_types.get('numerical', [])
    
    if len(numerical_cols) < 2:
        return issues
    
    corr_matrix = df[numerical_cols].corr()
    
    # Find highly correlated pairs
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.95:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                issues.append(Issue(
                    severity=Issue.MINOR,
                    issue_type="High Feature Correlation",
                    column=f"{col1}, {col2}",
                    message=f"'{col1}' and '{col2}' are highly correlated ({corr_val:.3f})",
                    suggestion="Remove one to reduce multicollinearity",
                    actions=[("Drop " + col1, f"drop_{col1}"), ("Drop " + col2, f"drop_{col2}"), ("Keep Both", f"keep_{col1}_{col2}")]
                ))
    
    return issues


def detect_all_issues(df, target_col=None, column_types=None):
    """Detect all data quality issues."""
    if column_types is None:
        column_types = detect_column_types(df)
    
    all_issues = []
    
    # Detect issues
    all_issues.extend(detect_missing_value_issues(df))
    all_issues.extend(detect_target_issues(df, target_col))
    all_issues.extend(detect_outlier_issues(df, column_types))
    all_issues.extend(detect_cardinality_issues(df, column_types))
    all_issues.extend(detect_constant_features(df))
    all_issues.extend(detect_correlation_issues(df, column_types))
    
    # Sort by severity
    severity_order = {Issue.CRITICAL: 0, Issue.MODERATE: 1, Issue.MINOR: 2}
    all_issues.sort(key=lambda x: severity_order[x.severity])
    
    return all_issues


def display_issue(issue, idx, mode="expert"):
    """Display a single issue with action buttons.
    
    Args:
        issue: Issue object to display
        idx: Index for unique button keys
        mode: "expert" or "beginner" mode
        
    Returns:
        Tuple (action_key, action_name) where:
        - action_key: The specific action key (e.g., "impute_Age", "skip_Age")
        - action_name: The action name for display
    """
    severity_icons = {
        Issue.CRITICAL: "🔴",
        Issue.MODERATE: "🟡",
        Issue.MINOR: "🟢"
    }
    
    icon = severity_icons[issue.severity]
    
    # Issue container
    st.markdown(f"## {icon} {issue.issue_type}")
    
    if issue.severity == Issue.CRITICAL:
        st.error(f"**{issue.message}**")
    elif issue.severity == Issue.MODERATE:
        st.warning(f"**{issue.message}**")
    else:
        st.info(f"**{issue.message}**")
    
    st.markdown(f"**Suggestion:** {issue.suggestion}")
    
    # Action buttons based on mode
    if mode == "expert" and issue.actions:
        # Display actual action buttons from the issue
        num_actions = len(issue.actions)
        cols = st.columns(num_actions)
        
        for col_idx, (action_name, action_key) in enumerate(issue.actions):
            with cols[col_idx]:
                if st.button(action_name, key=f"{action_key}_{idx}", use_container_width=True):
                    return (action_key, action_name)
    
    st.markdown("---")
    
    return (None, None)


def apply_issue_fix(df, action_key, issue):
    """Apply a fix based on the action selected."""
    modified_df = df.copy()
    message = ""
    
    # Handle imputation actions
    if action_key.startswith("impute_mean_"):
        col_name = action_key.replace("impute_mean_", "")
        if col_name in modified_df.columns:
            mean_val = modified_df[col_name].mean()
            modified_df[col_name].fillna(mean_val, inplace=True)
            message = f"Imputed '{col_name}' with mean ({mean_val:.2f})"
    
    elif action_key.startswith("impute_median_"):
        col_name = action_key.replace("impute_median_", "")
        if col_name in modified_df.columns:
            median_val = modified_df[col_name].median()
            modified_df[col_name].fillna(median_val, inplace=True)
            message = f"Imputed '{col_name}' with median ({median_val:.2f})"
    
    elif action_key.startswith("impute_knn_"):
        col_name = action_key.replace("impute_knn_", "")
        if col_name in modified_df.columns:
            try:
                from sklearn.impute import KNNImputer
                
                # Count non-missing samples
                non_missing_count = modified_df[col_name].notna().sum()
                
                # Adjust n_neighbors based on available data
                n_neighbors = min(5, max(1, non_missing_count - 1))
                
                if n_neighbors >= 1 and non_missing_count >= 2:
                    # Get numeric columns for KNN
                    numeric_cols = modified_df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if col_name in numeric_cols and len(numeric_cols) > 0:
                        # Apply KNN imputation
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                        modified_df[numeric_cols] = imputer.fit_transform(modified_df[numeric_cols])
                        message = f"Imputed '{col_name}' using KNN (n_neighbors={n_neighbors})"
                    else:
                        # Fallback to mean for single column or non-numeric
                        mean_val = modified_df[col_name].mean()
                        modified_df[col_name].fillna(mean_val, inplace=True)
                        message = f"Imputed '{col_name}' with mean ({mean_val:.2f}) - KNN not applicable"
                else:
                    # Not enough data for KNN, use mean
                    mean_val = modified_df[col_name].mean()
                    modified_df[col_name].fillna(mean_val, inplace=True)
                    message = f"Imputed '{col_name}' with mean ({mean_val:.2f}) - insufficient data for KNN"
            except Exception as e:
                # Fallback to mean if KNN fails
                mean_val = modified_df[col_name].mean()
                modified_df[col_name].fillna(mean_val, inplace=True)
                message = f"Imputed '{col_name}' with mean ({mean_val:.2f}) - KNN failed"
    
    elif action_key.startswith("impute_mode_"):
        col_name = action_key.replace("impute_mode_", "")
        if col_name in modified_df.columns:
            mode_val = modified_df[col_name].mode()[0] if len(modified_df[col_name].mode()) > 0 else 'Unknown'
            modified_df[col_name].fillna(mode_val, inplace=True)
            message = f"Imputed '{col_name}' with mode ('{mode_val}')"
    
    elif action_key.startswith("impute_constant_"):
        col_name = action_key.replace("impute_constant_", "")
        if col_name in modified_df.columns:
            modified_df[col_name].fillna('Unknown', inplace=True)
            message = f"Imputed '{col_name}' with constant 'Unknown'"
    
    elif action_key.startswith("impute_missing_"):
        col_name = action_key.replace("impute_missing_", "")
        if col_name in modified_df.columns:
            modified_df[col_name].fillna('Missing', inplace=True)
            message = f"Imputed '{col_name}' with 'Missing' category"
    
    elif action_key.startswith("drop_"):
        col_name = action_key.replace("drop_", "")
        if col_name in modified_df.columns:
            modified_df = modified_df.drop(columns=[col_name])
            message = f"Dropped column '{col_name}'"
    
    elif action_key.startswith("remove_outliers_"):
        col_name = action_key.replace("remove_outliers_", "")
        if col_name in modified_df.columns:
            Q1 = modified_df[col_name].quantile(0.25)
            Q3 = modified_df[col_name].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before_count = len(modified_df)
            modified_df = modified_df[(modified_df[col_name] >= lower) & (modified_df[col_name] <= upper)]
            removed = before_count - len(modified_df)
            message = f"Removed {removed} outlier rows from '{col_name}'"
    
    elif action_key.startswith("cap_"):
        col_name = action_key.replace("cap_", "")
        if col_name in modified_df.columns:
            lower = modified_df[col_name].quantile(0.05)
            upper = modified_df[col_name].quantile(0.95)
            modified_df[col_name] = modified_df[col_name].clip(lower, upper)
            message = f"Capped outliers in '{col_name}' at 5th and 95th percentiles"
    
    elif action_key.startswith("keep_"):
        # Keep action - extract column name and check issue type
        col_name = action_key.replace("keep_", "")
        # Track as ignored
        if 'ignored_issues' not in st.session_state:
            st.session_state['ignored_issues'] = []
        issue_id = f"{issue.issue_type}_{issue.column}"
        if issue_id not in st.session_state['ignored_issues']:
            st.session_state['ignored_issues'].append(issue_id)
        
        if "outlier" in issue.issue_type.lower():
            message = f"Kept outliers in '{col_name}' - moved to ignored issues"
        else:
            message = f"Kept '{col_name}' - moved to ignored issues"
    
    elif action_key.startswith("skip"):
        # Skip - track as ignored
        if 'ignored_issues' not in st.session_state:
            st.session_state['ignored_issues'] = []
        issue_id = f"{issue.issue_type}_{issue.column}"
        if issue_id not in st.session_state['ignored_issues']:
            st.session_state['ignored_issues'].append(issue_id)
        message = f"Skipped '{issue.column}' - moved to ignored issues"
    
    elif action_key.startswith("top20_"):
        col_name = action_key.replace("top20_", "")
        if col_name in modified_df.columns:
            top_20 = modified_df[col_name].value_counts().head(20).index.tolist()
            modified_df[col_name] = modified_df[col_name].apply(lambda x: x if x in top_20 else 'Other')
            message = f"Kept top 20 categories in '{col_name}', replaced others with 'Other'"
    
    elif action_key.startswith("target_enc_"):
        col_name = action_key.replace("target_enc_", "")
        if col_name in modified_df.columns:
            # Mark for target encoding during preprocessing
            if 'target_encode_cols' not in st.session_state:
                st.session_state['target_encode_cols'] = []
            if col_name not in st.session_state['target_encode_cols']:
                st.session_state['target_encode_cols'].append(col_name)
            message = f"Marked '{col_name}' for target encoding during preprocessing"
    
    elif action_key == "remove_target_missing":
        target_col = issue.column
        before_count = len(modified_df)
        modified_df = modified_df.dropna(subset=[target_col])
        removed = before_count - len(modified_df)
        message = f"Removed {removed} rows with missing target values"
    
    elif action_key in ["apply_smote", "use_weights"]:
        # These will be handled in preprocessing
        if action_key == "apply_smote":
            st.session_state['apply_smote'] = True
            message = "SMOTE will be applied during preprocessing"
        else:
            st.session_state['use_class_weights'] = True
            message = "Class weights will be used during model training"
    
    elif action_key == "skip_imbalance" or action_key == "continue" or action_key == "continue_anyway":
        message = "Continued with imbalanced data"
    
    return modified_df, message


def render_issue_detection_page():
    """Render the issue detection page."""
    st.header("Issue Detection & Resolution")
    
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.warning("Please upload a dataset first!")
        return
    
    df = st.session_state['data']
    target_col = st.session_state.get('target_column', None)
    column_types = st.session_state.get('column_types', detect_column_types(df))
    
    # Initialize reviewed issues tracking
    if 'reviewed_issues' not in st.session_state:
        st.session_state['reviewed_issues'] = set()
    
    # Detect issues (only if not already detected or if requested)
    if 'detected_issues' not in st.session_state or st.session_state.get('redetect_issues', False):
        with st.spinner("Detecting data quality issues..."):
            issues = detect_all_issues(df, target_col, column_types)
            
            # Filter out already reviewed issues
            reviewed_set = st.session_state.get('reviewed_issues', set())
            unreviewed_issues = [issue for issue in issues if issue.get_id() not in reviewed_set]
            
            st.session_state['detected_issues'] = unreviewed_issues
            st.session_state['redetect_issues'] = False
    
    issues = st.session_state['detected_issues']
    
    # Summary metrics
    critical_count = sum(1 for i in issues if i.severity == Issue.CRITICAL)
    moderate_count = sum(1 for i in issues if i.severity == Issue.MODERATE)
    minor_count = sum(1 for i in issues if i.severity == Issue.MINOR)
    active_issues_count = len(issues)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("ACTIVE ISSUES", active_issues_count)
    with col2:
        st.metric("CRITICAL", critical_count)
    with col3:
        st.metric("MODERATE", moderate_count)
    with col4:
        st.metric("MINOR", minor_count)
    with col5:
        reviewed_count = len(st.session_state.get('reviewed_issues', set()))
        st.metric("REVIEWED", reviewed_count)
    
    st.markdown("---")
    
    if active_issues_count == 0:
        st.success("✅ No active data quality issues! Your dataset is ready for preprocessing.")
        st.info("💡 Issues will be handled automatically during preprocessing based on your selections.")
        st.session_state['issues_resolved'] = True
        
        # Continue button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Preprocessing", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'preprocessing'
                st.rerun()
        return
    
    # Mode selection
    mode = st.session_state.get('mode', 'Beginner')
    
    if mode == "Beginner":
        st.info("**Beginner Mode:** Review detected issues and their recommended solutions. Click 'Auto-Fix All Issues' to apply smart fixes.")
        
        # Display issues with recommended fixes
        st.markdown("### Detected Issues & Recommended Fixes")
        
        for idx, issue in enumerate(issues):
            severity_color = {
                Issue.CRITICAL: "🔴",
                Issue.MODERATE: "🟡",
                Issue.MINOR: "🟢"
            }[issue.severity]
            
            with st.expander(f"{severity_color} {issue.issue_type} - Column: `{issue.column}`", expanded=False):
                st.markdown(f"**Issue:** {issue.message}")
                st.markdown(f"**Recommendation:** {issue.suggestion}")
                
                if issue.actions:
                    best_action = issue.actions[0]  # First action is the recommended one
                    st.markdown(f"**Recommended Fix:** {best_action[0]}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Auto-Fix All Issues", type="primary", use_container_width=True):
                modified_df = df.copy()
                applied_fixes = []
                
                for issue in issues:
                    # Mark as reviewed
                    st.session_state['reviewed_issues'].add(issue.get_id())
                    
                    if issue.issue_type == "High Missing Values":
                        col = issue.column
                        if col in modified_df.columns:
                            modified_df = modified_df.drop(columns=[col])
                            applied_fixes.append(f"✅ Dropped column '{col}' (>50% missing)")
                    
                    elif issue.issue_type == "Target Missing Values":
                        target = issue.column
                        before = len(modified_df)
                        modified_df = modified_df.dropna(subset=[target])
                        applied_fixes.append(f"✅ Removed {before - len(modified_df)} rows with missing target")
                    
                    elif issue.issue_type == "Severe Class Imbalance" or issue.issue_type == "Class Imbalance":
                        st.session_state['apply_smote'] = True
                        applied_fixes.append("✅ Enabled SMOTE for class balancing (will apply during preprocessing)")
                    
                    elif issue.issue_type == "High Cardinality":
                        col = issue.column
                        if col in modified_df.columns:
                            top_20 = modified_df[col].value_counts().head(20).index.tolist()
                            modified_df[col] = modified_df[col].apply(lambda x: x if x in top_20 else 'Other')
                            applied_fixes.append(f"✅ Kept top 20 categories in '{col}'")
                    
                    elif issue.issue_type == "Constant Feature":
                        col = issue.column
                        if col in modified_df.columns:
                            modified_df = modified_df.drop(columns=[col])
                            applied_fixes.append(f"✅ Dropped constant column '{col}'")
                    
                    elif issue.issue_type == "Missing Values":
                        # Will be handled in preprocessing
                        applied_fixes.append(f"ℹ️ Missing values in '{issue.column}' will be imputed during preprocessing")
                
                st.session_state['data'] = modified_df
                st.session_state['issues_resolved'] = True
                st.session_state['column_types'] = detect_column_types(modified_df)
                
                st.success("✅ Applied automatic fixes!")
                for fix in applied_fixes:
                    st.write(fix)
                
                st.info(f"📊 Dataset now has {len(modified_df)} rows and {len(modified_df.columns)} columns.")
                
                # Clear detected issues
                del st.session_state['detected_issues']
                st.rerun()
        
        with col2:
            if st.button("⏭️ Skip & Continue", use_container_width=True):
                # Mark all as reviewed but don't fix
                for issue in issues:
                    st.session_state['reviewed_issues'].add(issue.get_id())
                st.session_state['issues_resolved'] = True
                del st.session_state['detected_issues']
                st.rerun()
    
    else:  # Expert mode
        st.info("**Expert Mode:** Review each issue and decide how to handle it. Issues are only marked for resolution - actual fixes apply during preprocessing.")
        
        # Display issues
        if issues:
            st.markdown("### Active Issues")
            for idx, issue in enumerate(issues):
                action_key, action_name = display_issue(issue, idx, mode="expert")
                
                if action_key:
                    # Check if this is a "skip" action (user wants to ignore the issue)
                    if action_key.startswith("skip_"):
                        # Mark as reviewed/ignored (no resolution)
                        st.session_state['reviewed_issues'].add(issue.get_id())
                        st.success(f"✅ Issue skipped: {issue.issue_type} on '{issue.column}'")
                        del st.session_state['detected_issues']
                        st.rerun()
                    else:
                        # Mark for resolution - store the action for preprocessing
                        st.session_state['reviewed_issues'].add(issue.get_id())
                        
                        if 'issue_resolutions' not in st.session_state:
                            st.session_state['issue_resolutions'] = {}
                        
                        # Store the action key for preprocessing to use
                        st.session_state['issue_resolutions'][issue.get_id()] = action_key
                        st.success(f"✅ Marked for resolution: {action_name} for '{issue.column}' - Will apply during preprocessing")
                        
                        del st.session_state['detected_issues']
                        st.rerun()
        else:
            st.success("✅ All issues reviewed!")
        
        # Mark as resolved button
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Mark All Issues as Reviewed", type="primary", use_container_width=True):
                for issue in issues:
                    st.session_state['reviewed_issues'].add(issue.get_id())
                st.session_state['issues_resolved'] = True
                del st.session_state['detected_issues']
                st.rerun()
        
        with col2:
            if st.button("🔄 Re-detect Issues", use_container_width=True):
                st.session_state['reviewed_issues'] = set()
                st.session_state['redetect_issues'] = True
                del st.session_state['detected_issues']
                st.rerun()
    
    # Continue button if issues are resolved
    if st.session_state.get('issues_resolved', False):
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Continue to Preprocessing", type="primary", use_container_width=True):
                st.session_state['current_page'] = 'preprocessing'
                st.rerun()
