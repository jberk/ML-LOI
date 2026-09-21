#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


df1 = pd.read_csv('56d_MICE_imputed_dataset_1.csv')
df2 = pd.read_csv('56d_MICE_imputed_dataset_2.csv')
df3 = pd.read_csv('56d_MICE_imputed_dataset_3.csv')
df4 = pd.read_csv('56d_MICE_imputed_dataset_4.csv')
df5 = pd.read_csv('56d_MICE_imputed_dataset_5.csv')


# In[ ]:


df_list = [df1, df2, df3, df4, df5]


# In[ ]:


for df in df_list:

    df.drop(columns = ['index'], inplace=True)


    for col in df:

        if df[col].dtype == 'int64':

            if col != 'year':

                df[col] = df[col].astype('category')


    df.drop(columns = ['LOS'], inplace=True)

    print(df.info())

    print(df['Long_stay'].value_counts())






# In[ ]:


j = 1
results = []
importances = []


for df in df_list:

    rf_model = joblib.load(f"brf_56d_avgp_MICE_new_{j}.joblib")
    gbm_model = joblib.load(f"gbm_56d_avgp_MICE_new_{j}.joblib")
    log_model = joblib.load(f"log_56d_avgp_MICE_new_{j}.joblib")
    catboost_model = joblib.load(f"cat_56d_avgp_MICE_new_{j}.joblib")


    df_train_years = df[df['year'] <= 2018]

    df_train_years = df_train_years.drop(columns = ['year'])

    #print(df_train_years.info())

    df_test_years = df[df['year'] >= 2019]

    df_test_years = df_test_years.drop(columns = ['year'])

    #print(df_test_years.info())

    df_manual_test = df_test_years

    df_model = df_train_years

    df_model_final = df_model.copy()

    X_train_full = df_model.drop(columns=['Long_stay'])
    y_train_full = df_model['Long_stay'].astype('category')

    #already manually set aside test set

    X_test = df_test_years.drop(columns=['Long_stay'])
    y_test = df_test_years['Long_stay'].astype('category')

    base_models = [rf_model, gbm_model, log_model, catboost_model]

    # Nested CV setup
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Store metrics - for final training
    metrics_summary = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "pr_auc": [],
        "balanced_accuracy": [],
        "npv": [],
        "specificty":[],
        "roc-auc":[]
    }


    # === 1. Retrain base models on full training set ===

    final_base_models = []
    train_meta_features = np.zeros((X_train_full.shape[0], len(base_models)))
    test_meta_features = np.zeros((X_test.shape[0], len(base_models)))

    for i, base_model in enumerate(base_models):

        model_clone = clone(base_model)
        model_clone.fit(X_train_full, y_train_full)
        final_base_models.append(model_clone)

        # Meta features for training
        train_meta_features[:, i] = model_clone.predict_proba(X_train_full)[:, 1]

        # Meta features for test
        test_meta_features[:, i] = model_clone.predict_proba(X_test)[:, 1]

    final_meta_model = GradientBoostingClassifier(
        random_state=42, n_estimators=200, learning_rate=0.1,max_depth=3)

    calibrated_final_meta = CalibratedClassifierCV(
        estimator=final_meta_model,
        method="isotonic",     # isotonic since relatively large dataset
        cv=5                  # internal calibration split on TRAIN only
    )

    calibrated_final_meta.fit(train_meta_features, y_train_full)


    # === 3. Eval on test (CALIBRATED) ===

    y_prob_test = calibrated_final_meta.predict_proba(test_meta_features)[:, 1]

    # Tune threshold here
    y_pred_test = (y_prob_test >= 0.05).astype(int)


    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    pr_auc = average_precision_score(y_test, y_prob_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    print("\nFinal Test Set Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"NPV: {npv:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

    specificty = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"Specificity (TNR): {specificty:.4f}")



    roc_auc = roc_auc_score(y_test, y_prob_test)

    print(roc_auc)


    metrics_summary["accuracy"].append(acc)
    metrics_summary["precision"].append(prec)
    metrics_summary["recall"].append(rec)
    metrics_summary["f1"].append(f1)
    metrics_summary["pr_auc"].append(pr_auc)
    metrics_summary["balanced_accuracy"].append(bal_acc)
    metrics_summary["npv"].append(npv)
    metrics_summary["specificty"].append(specificty)
    metrics_summary["roc-auc"].append(roc_auc)

    results.append(metrics_summary)

    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1], [0,1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid()
    plt.show()


    importances = np.mean(
    [clf.estimator.feature_importances_
     for clf in calibrated_final_meta.calibrated_classifiers_],
    axis=0
    )

    fi_df = pd.DataFrame({
        "base_model": base_models,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(fi_df)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)


    # Step 2: Compute Youden's J statistic for each threshold
    youden_j = tpr - fpr  # Equivalent to Sensitivity + Specificity - 1

    # Step 3: Find the threshold that maximizes J
    max_j_index = np.argmax(youden_j)
    optimal_threshold = thresholds[max_j_index]
    max_j_value = youden_j[max_j_index]

    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Maximum Youden's J: {max_j_value:.4f}")

    print('______________________________________')

    #importances.append(fi_df)





# In[ ]:





# In[ ]:


print(results)


# In[ ]:


avg_metrics = {
    metric: np.mean([d[metric][0] for d in results])
    for metric in results[0]
}

print(avg_metrics)


# In[ ]:


custom_index = ['result']


# In[ ]:


avg_metrics_df = pd.DataFrame(avg_metrics, index=custom_index)


# In[ ]:


print(avg_metrics_df)


# In[ ]:


avg_metrics_df.to_csv('Stacked 56d RIDOC MICE.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:








# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
