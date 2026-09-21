#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


df1 = pd.read_csv('JDI_56d_MICE_imputed_dataset_1.csv')
df2 = pd.read_csv('JDI_56d_MICE_imputed_dataset_2.csv')
df3 = pd.read_csv('JDI_56d_MICE_imputed_dataset_3.csv')
df4 = pd.read_csv('JDI_56d_MICE_imputed_dataset_4.csv')
df5 = pd.read_csv('JDI_56d_MICE_imputed_dataset_5.csv')


# In[ ]:


df_list = [df1, df2, df3, df4, df5]


# In[ ]:


print(df1.info())


# In[ ]:


print(df1['Long_stay'].value_counts())


# In[ ]:


cat_cols = [
    "Race_Ethnicity_Standardized",
    "Sex_Gender_Standardized","State", "Top_Charge","Num_Bookings_Grouped"
]

num_cols = [
    "Bond_Amount",'SVI'
]


# In[ ]:


for df in df_list:

    df['Num_Bookings_Grouped'] = df['Num_Bookings_Grouped'].astype('int64')

    df.drop(columns = ['index'],inplace=True)

print(df1.info())


# In[ ]:


print(df1['Long_stay'].value_counts())


# In[ ]:


results = []

i = 0

for df in df_list:

    i += 1

    df_train_years = df[df['Year'] <= 2023]

    print(df_train_years['Year'].value_counts())


    df_train_years = df_train_years.drop(columns = ['Year'])

    print(df_train_years.info()) #34,715 entries

    print(df_train_years['Long_stay'].value_counts())

    df_test_years = df[df['Year'] >= 2024]

    print(df_test_years['Year'].value_counts())

    df_test_years = df_test_years.drop(columns = ['Year'])

    print(df_test_years.info()) #4,977 entries

    print(df_test_years['Long_stay'].value_counts())


    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": 1,
        "n_jobs": -1,
        "is_unbalance": True
    }

    num_round = 100


    X_train = df_train_years.drop(columns=['Long_stay'])
    y_train = df_train_years['Long_stay']

    X_test = df_test_years.drop(columns=['Long_stay'])
    y_test = df_test_years['Long_stay']

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols, free_raw_data=False)

    lgb_test  = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=cat_cols, free_raw_data=False)

    final_gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        valid_names=["train", "valid"],
    )

    y_pred_proba = final_gbm.predict_proba(X_test)[:, 1]

    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    #auc = roc_auc_score(y_test, y_pred_binary)
    cm = confusion_matrix(y_test, y_pred_binary)

    print(f"Accuracy on test set: {accuracy}")
    print(f"AUC on test set: {auc}")
    print(f"Confusion Matrix:\n{cm}")


    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Log loss:", log_loss(y_test, y_pred_proba))
    print(classification_report(y_test, y_pred_binary))

    joblib.dump(final_gbm,f"lightgbm_JDI_56d_{i}.joblib")


    accuracy = accuracy_score(y_test, y_pred_binary)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred_binary
    ).ravel()

    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_binary)

    pr_auc = auc(recall_vals, precision_vals)

    print(f"PR AUC: {pr_auc}")


    metrics_dict = {
        'Accuracy': round(accuracy,4),
        'Balanced Accuracy': round(balanced_accuracy,4),
        'Precision': round(precision,4),
        'Sensitivity / Recall': round(recall,4),
        'Specificity': round(specificity,4),
        'NPV': round(npv,4),
        'F1 Score': round(f1,4),
        'PR-AUC': round(pr_auc,4),
        'ROC-AUC': round(roc_auc,4)
    }

    metrics_df = pd.DataFrame([metrics_dict])

    print(metrics_df)

    results.append(metrics_df)

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1], [0,1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid()
    plt.show()


# In[ ]:


print(results)


# In[ ]:


pooled_results = sum(results) / len(results)

print(pooled_results)


# In[ ]:


pooled_results.to_csv('JDI_LightGBM_MICE_56d.csv')
