###final code-1

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)


def main():
    # Step 1: Load the Dataset
    DATA_PATH= os.path.join("data", "data.csv")
    df=pd.read_csv(DATA_PATH)
    df.drop(columns=['id', 'Unnamed: 32'], errors='ignore', inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    df.fillna(df.median(), inplace=True)

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    #  Step 2: Train-Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Step 3: SHAP Feature Selection
    print("\nRunning SHAP on training set...")
    base_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train_full, y_train_full)
    explainer = shap.Explainer(base_model, X_train_full)
    shap_values = explainer(X_train_full)

    shap_array = shap_values.values
    if shap_array.ndim == 3 and shap_array.shape[2] == 2:
        shap_array = shap_array[:, :, 1]

    importance = np.abs(shap_array).mean(axis=0)
    top_features = pd.Series(importance, index=X_train_full.columns).nlargest(15).index.tolist()
    print("\nTop 15 SHAP-selected features:")
    for i, feat in enumerate (top_features, 1):
    print (f"{i:2d}.{feat}")

    # Step 4: Scale Features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full[top_features])
    X_test_scaled = scaler.transform(X_test[top_features])
    y_train_full = y_train_full.reset_index(drop=True)

    #  Step 5: Model Definitions 
    et = ExtraTreesClassifier(n_estimators=200, random_state=42)
    lgbm = LGBMClassifier(n_estimators=400, learning_rate=0.02, max_depth=4,
                          num_leaves=30, colsample_bytree=0.7, subsample=0.6,
                          min_child_samples=20, reg_lambda=3.5, reg_alpha=0.7,
                          random_state=42)
    xgb = XGBClassifier(n_estimators=400, learning_rate=0.02, max_depth=4,
                        subsample=0.6, colsample_bytree=0.7, gamma=0.4,
                        reg_lambda=4.0, reg_alpha=0.8, use_label_encoder=False,
                        eval_metric='logloss', random_state=42)
    svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)

    ensemble = VotingClassifier(
        estimators=[('ET', et), ('LGBM', lgbm), ('XGB', xgb), ('SVM', svm)],
        voting='soft', weights=[1, 3, 3, 2]
    )

    #  Step 6: Cross-Validation 
    print("\nPerforming 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    threshold = 0.30

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_full)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        ensemble.fit(X_tr, y_tr)
        y_prob = ensemble.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > threshold).astype(int)

        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred))
        metrics['recall'].append(recall_score(y_val, y_pred))
        metrics['f1'].append(f1_score(y_val, y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_val, y_prob))

        print(f"Fold {fold+1} - AUC: {metrics['roc_auc'][-1]:.4f}, Recall: {metrics['recall'][-1]:.4f}")

    #  Step 6.1: Report Mean ± Std 
    print("\n=== Cross-Validation Results (Mean ± Std) ===")
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # ======= Step 6.2: Report Median ± IQR =======
    print("\n=== Cross-Validation Results (Median ± IQR) ===")
    for k, v in metrics.items():
        median = np.median(v)
        iqr = np.percentile(v, 75) - np.percentile(v, 25)
        print(f"{k.capitalize():<10}: {median:.4f} ± {iqr:.4f}")

    #  Step 7: Final Test Set Evaluation 
    print("\nFinal Test Evaluation...")
    ensemble.fit(X_train_scaled, y_train_full)
    y_prob_test = ensemble.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_prob_test > threshold).astype(int)

    print(f"Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_test):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred_test):.4f}")
    print(f"AUC-ROC : {roc_auc_score(y_test, y_prob_test):.4f}")

if __name__ == '__main__':
    main()
