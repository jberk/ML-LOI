#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


#pip install "numpy<2.0"


# In[ ]:


#pip install --upgrade imbalanced-learn


# In[ ]:





# In[ ]:





# In[2]:


df = pd.read_csv('56d_RIDOC_complete_case.csv')


# In[3]:


print(df.info())


# In[ ]:





# In[4]:


df = df.drop(columns = ['Unnamed: 0'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


df.drop(columns = ['index'], inplace=True)


for col in df:

        if df[col].dtype == 'int64':

            if col != 'year':

                df[col] = df[col].astype('category')

print(df.info())

long_stay_def = 55
df['Long_stay'] = df['LOS'].apply(lambda x: 1 if x > long_stay_def else 0)

df.drop(columns = ['LOS'], inplace=True)

print(df.info())

print(df['Long_stay'].value_counts())




# In[14]:


print(df['Long_stay'].value_counts())


# In[15]:


results = []
best_scores = []

i = 0


i += 1

df_train_years = df[df['year'] <= 2018]

df_train_years = df_train_years.drop(columns = ['year'])

#print(df_train_years.info())

df_test_years = df[df['year'] >= 2019]

df_test_years = df_test_years.drop(columns = ['year'])

    #print(df_test_years.info())

df_manual_test = df_test_years

df_model = df_train_years

df_model_final = df_model.copy()




    # Extract labels
labels = df_model['Long_stay'].astype('category').cat.codes
    #Drop target variable from df for df_model
df_model = df_model.drop(columns=['Long_stay'])

    # Outer Cross-Validation initialization
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


numeric_features = df_train_years.select_dtypes(include=['float64']).columns.tolist()

print(numeric_features)


    # Parameter grid options that will be tested in grid search
param_grid = {
        'n_estimators': [100, 200], #default 100
        'max_depth': [None, 2], #None = default
        'min_samples_split': [2, 5], #2 = default
        'min_samples_leaf': [1,2], #1 = default
        'class_weight':['balanced']
}

np.random.seed(123)  #for reproducibility

print(df_model)



all_best_params = []
best_fold_pr_auc = 0 #initialize as 0
best_pr_auc = 0

    # For each fold in the outer cross-validation
for fold_number, (train_index, test_index) in enumerate(outer_cv.split(df_model, labels), start=1):

        # Print the current fold number
        print(f'Fitting for fold {fold_number}')

        # Ensure 80% data goes to training/20% goes to testing within each fold
        indices = np.arange(len(train_index))
        train_indices = np.random.choice(indices, size=int(0.8 * len(train_index)), replace=False)
        test_indices = np.setdiff1d(indices, train_indices)

        # Obtain the actual data indices from train_index using the selected train_indices
        actual_train_index = train_index[train_indices]
        actual_test_index = train_index[test_indices]

        X_train = df_model.iloc[actual_train_index]
        y_train = labels.iloc[actual_train_index]
        X_test = df_model.iloc[actual_test_index]
        y_test = labels.iloc[actual_test_index]

        # Inner Cross-Validation initialization for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)


        grid_search = GridSearchCV(
            estimator=BalancedRandomForestClassifier(random_state=123),
            param_grid=param_grid,
            cv=inner_cv,
            scoring='average_precision',
            n_jobs=-1
        )

        # Train the model using grid_search
        grid_search.fit(X_train, y_train)

        best_gbm = grid_search.best_estimator_


        best_params = grid_search.best_params_

        all_best_params.append(best_params)

        predictions = best_gbm.predict(X_test)

        predictions_proba = best_gbm.predict_proba(X_test)[:, 1]
        predicted_classes = (predictions_proba > 0.5).astype(int)

        precision, recall, thresholds = precision_recall_curve(y_test, predictions_proba)
        pr_auc = auc(recall, precision)

        if pr_auc > best_fold_pr_auc:
            best_fold_pr_auc = pr_auc
            best_model = best_gbm
            best_fold_params = best_params
            best_fold_number = fold_number
            best_pr_auc = pr_auc


print(best_fold_params)

print(best_fold_number)

best_scores.append(best_pr_auc)


final_params = best_fold_params

final_model = BalancedRandomForestClassifier(**final_params)

    #final_model.set_params(**final_params)


X_train_final = df_model_final.drop(columns=['Long_stay'])
Y_train_final = df_model_final['Long_stay']

final_model.fit(X_train_final, Y_train_final) #train model using best parameters identified

X_test_final = df_manual_test.drop(columns=['Long_stay']) #use unseen data set aside before for testing
Y_test_final = df_manual_test['Long_stay']

    # Step 4: Make predictions (if needed)
final_predictions = final_model.predict(X_test_final)

final_predictions_proba = final_model.predict_proba(X_test_final)[:, 1]  # Probabilities for PR AUC

threshold = 0.5
final_predictions = (final_predictions_proba >= threshold).astype(int)
    # Step 3: Compute the metrics

    # Accuracy
accuracy = accuracy_score(Y_test_final, final_predictions)

    # Balanced Accuracy
balanced_accuracy = balanced_accuracy_score(Y_test_final, final_predictions)

    # Precision
precision = precision_score(Y_test_final, final_predictions)

    # Sensitivity / Recall
sensitivity = recall_score(Y_test_final, final_predictions)

roc_auc = roc_auc_score(Y_test_final, final_predictions)


    # Confusion Matrix for specificity calculation
tn, fp, fn, tp = confusion_matrix(Y_test_final, final_predictions).ravel()

    # Specificity
specificity = tn / (tn + fp)

    # F1 Score
f1 = f1_score(Y_test_final, final_predictions)

    # Precision-Recall AUC
precision_vals, recall_vals, thresholds = precision_recall_curve(Y_test_final, final_predictions_proba)
pr_auc = auc(recall_vals, precision_vals)

    # Remove the last element from precision_vals and recall_vals to match the length of thresholds
precision_vals = precision_vals[:-1]
recall_vals = recall_vals[:-1]

    # Check all arrays have the same length
print(f"Length of precision_vals: {len(precision_vals)}")
print(f"Length of recall_vals: {len(recall_vals)}")
print(f"Length of thresholds: {len(thresholds)}")

    # Save precision, recall, and thresholds to a DataFrame
pr_curve_df = pd.DataFrame({
        'Threshold': thresholds,  # Now thresholds match the length of precision and recall
        'Precision': precision_vals,
        'Recall': recall_vals
})


    # Print all metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity / Recall: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
tn, fp, fn, tp = confusion_matrix(y_test, predicted_classes).ravel()
npv = tn / (tn + fn)
print(f"NPV: {npv:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")


final_cm = confusion_matrix(Y_test_final, final_predictions)
print(f'Final Confusion Matrix:\n{final_cm}')


joblib.dump(final_model,f"brf_56d_cc_lim.joblib")
print('Model saved!')

    # Prepare the metrics in the desired order
metrics_dict = {
        'Accuracy': round(accuracy,4),
        'Balanced Accuracy': round(balanced_accuracy,4),
        'Precision': round(precision,4),
        'Sensitivity / Recall': round(sensitivity,4),
        'Specificity': round(specificity,4),
        'NPV': round(npv,4),
        'F1 Score': round(f1,4),
        'PR-AUC': round(pr_auc,4),
        'ROC-AUC': round(roc_auc,4)
}

    # Create a DataFrame for the metrics
metrics_df = pd.DataFrame([metrics_dict])

    # Add the confusion matrix as a single string in one column
confusion_matrix_str = f'[[{tn}, {fp}], [{fn}, {tp}]]'
    #metrics_df['Confusion Matrix'] = confusion_matrix_str

    # Add the best parameters as a column (converted to string for consistency)
    #metrics_df['Best Parameters'] = str(best_params)

    # Add the unique model name
    #model_name = f'Model_{i}'

    #metrics_df['Model'] = model_name

results.append(metrics_df)

prob_true, prob_pred = calibration_curve(Y_test_final, final_predictions_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0,1], [0,1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve (Reliability Diagram)')
plt.legend()
plt.grid()
plt.show()


# In[16]:


print(results)


# In[17]:


print(best_scores)


# In[ ]:





# In[18]:


pooled_results = sum(results) / len(results)

print(pooled_results)


# In[19]:


pooled_results.to_csv('BRF_56d_CC_full.csv')
