# Hotfix support code for AI_Driven_Financial_Risk_Analytics_25K_FIXED_executed.ipynb
# Paste/run these helper cells in the notebook if needed.

from collections import Counter
import pandas as pd


def preserve_protected_attribute(df, column='personal_status_sex'):
    """Preserve raw protected attribute before one-hot encoding or dataframe replacement."""
    if column not in df.columns:
        raise KeyError(f"{column} not found in dataframe before preprocessing.")
    return df[column].copy()


def apply_smote_if_imbalanced(X_train, y_train, imbalance_threshold=0.40, random_state=42):
    """Apply SMOTE only to the training split when the minority class ratio is below threshold."""
    print('Before SMOTE:', Counter(y_train))
    minority_ratio = pd.Series(y_train).value_counts(normalize=True).min()
    if minority_ratio < imbalance_threshold:
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise ImportError('Install imbalanced-learn: pip install imbalanced-learn') from exc
        k_neighbors = min(5, max(1, pd.Series(y_train).value_counts().min() - 1))
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print('After SMOTE:', Counter(y_train_balanced))
        return X_train_balanced, y_train_balanced
    print('SMOTE not applied; class balance is acceptable.')
    return X_train, y_train


def get_safe_protected_values(test_idx, df=None, X_test=None, protected_attribute_series=None, column='personal_status_sex'):
    """Avoid KeyError when transformed df no longer contains the raw protected column."""
    if protected_attribute_series is not None:
        return protected_attribute_series.loc[test_idx]
    if df is not None and column in df.columns:
        return df.loc[test_idx, column]
    if X_test is not None and hasattr(X_test, 'columns') and column in X_test.columns:
        return X_test[column]
    raise KeyError(
        f"{column} is missing. Save it before preprocessing using: "
        f"protected_attribute_series = df['{column}'].copy()"
    )


def add_issue_fraud_signal(fraud_df, issue_col='Issue'):
    """Check whether Issue text contains fraud/anomaly signal and add binary flag."""
    fraud_terms = [
        'fraud', 'scam', 'identity theft', 'unauthorized', 'suspicious',
        'stolen', 'forgery', 'fake', 'misrepresentation', 'dispute',
        'error', 'incorrect', 'not mine', 'account opening', 'theft'
    ]
    if issue_col not in fraud_df.columns:
        print(f'WARNING: {issue_col} column not found in fraud dataset.')
        return fraud_df
    issue_text = fraud_df[issue_col].astype(str).str.lower()
    fraud_df = fraud_df.copy()
    fraud_df['issue_fraud_signal'] = issue_text.apply(lambda x: int(any(term in x for term in fraud_terms)))
    print('Issue fraud/anomaly signal distribution:')
    print(fraud_df['issue_fraud_signal'].value_counts(dropna=False))
    if fraud_df['issue_fraud_signal'].sum() == 0:
        print('WARNING: Issue feature does not appear to contain fraud/anomaly signal.')
    else:
        print('Issue feature contains fraud/anomaly-related complaint patterns.')
    return fraud_df
