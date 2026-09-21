#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv('National_ML_Jan26_noNanBond.csv')


# In[3]:


print(df.info())


# In[4]:


df = df.drop(columns = ['Unnamed: 0'])


# In[5]:


#nan Bond already dropped


# In[ ]:





# In[6]:


print(df['meta.first_seen'].value_counts())


# In[7]:


#print(df['meta.first_seen'].min())


# In[8]:


#print(df['meta.first_seen'].max())


# In[9]:


df['meta.first_seen'] = pd.to_datetime(df['meta.first_seen'], errors='coerce')

df['Year'] = df['meta.first_seen'].dt.year


# In[10]:


print(df['Year'].value_counts())


# In[11]:


print(df['Top_Charge'].value_counts())


# In[12]:


type_counts = df['Top_Charge'].value_counts()

top_5_types = type_counts.head(5).index

print(top_5_types)


# In[13]:


df = df[df['Top_Charge'].isin(top_5_types)]


# In[14]:


print(df.info())


# In[15]:


print(df['Num_Bookings'].value_counts())


# In[16]:


bins = [0, 1, 2, 3, 4, float('inf')]  # float('inf') represents "5+" category

labels = ['1', '2', '3', '4', '5+']

df['Num_Bookings_Grouped'] = pd.cut(df['Num_Bookings'], bins=bins, labels=labels, right=True)

print(df['Num_Bookings_Grouped'].value_counts())


# In[17]:


plt.hist(df['Num_Bookings'], bins = 50)
plt.xlim(0,50)


# In[18]:


#need to consolidate into incarceration events
#check meta.first seen vs. id

# Step 1: Drop duplicates based on 'ID' and 'date'
unique_id_date_pairs = df[['_id', 'meta.first_seen']].drop_duplicates()

# Step 2: Count the number of unique ID and date pairs
num_unique_pairs = unique_id_date_pairs.shape[0]

# Show the result
print("Number of unique ID and date pairs:", num_unique_pairs)



# In[19]:


print(len(df)) #already indexed by incarceration event


# In[20]:


SVI_df = pd.read_csv('SVI_2018_US_county.csv')

column_list = SVI_df.columns.to_list()

print(column_list)


# In[21]:


SVI_df = SVI_df.rename(columns={'COUNTY': 'County'})

SVI_df['County'] = SVI_df['County'].astype(str)
SVI_df['RPL_THEMES'] = SVI_df['RPL_THEMES'].astype(str)


themes_df = SVI_df[['County', 'RPL_THEMES']]
print(themes_df)


# In[22]:


themes_df['RPL_THEMES'] = pd.to_numeric(themes_df['RPL_THEMES'], errors = 'coerce')

themes_df = themes_df[themes_df['RPL_THEMES'] >= 0]


# In[23]:


print(themes_df.info())


# In[24]:


themes_df = themes_df.drop_duplicates()


# In[25]:


themes_df = themes_df.drop_duplicates(subset = 'County')


# In[26]:


print(themes_df.info())


# In[27]:


df = df.drop_duplicates()
print(df.info())


# In[28]:


merged_df = pd.merge(df, themes_df, on='County', how='left')
print(merged_df.shape)


# In[29]:


print(merged_df.info())


# In[30]:


merged_df = merged_df.drop(columns = ['County'])

merged_df = merged_df.rename(columns = {'RPL_THEMES':'SVI'})


# In[31]:


print(merged_df['SVI'].value_counts())


# In[32]:


merged_df['SVI'] = pd.to_numeric(merged_df['SVI'], errors = 'coerce')


# In[33]:


plt.hist(merged_df['SVI'])


# In[34]:


#just group SVI for manuscript table, use actual values in model

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]  # float('inf') represents "5+" category

labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

merged_df['SVI_Grouped'] = pd.cut(merged_df['SVI'], bins=bins, labels=labels, right=True)

print(merged_df['SVI_Grouped'].value_counts())


# In[35]:


print(merged_df.info())


# In[36]:


plt.hist(merged_df['LOS'], bins=70)
plt.xlim(0,200)


# In[37]:


#cleaning/processing

merged_df = merged_df[merged_df['LOS'] > 3]

print(merged_df.info())

merged_df['Long_stay'] = (merged_df['LOS'] >= 28).astype(int)

print(merged_df['Long_stay'].value_counts())


# In[38]:


merged_df = merged_df[merged_df['Age_Standardized'] >= 18]
print(merged_df['Age_Standardized'].value_counts())
print(merged_df.info())


# In[39]:


#merged_df = merged_df.drop(columns = ['LOS'])


# In[40]:


print(merged_df.info())


# In[41]:


merged_df = merged_df.drop(columns = ['_id', 'meta.first_seen', 'Num_Bookings', 'SVI_Grouped'])


# In[42]:


print(merged_df.info())


# In[43]:


merged_df['Num_Bookings_Grouped'] = merged_df['Num_Bookings_Grouped'].replace('5+', 5)


# In[44]:


merged_df['Num_Bookings_Grouped'] = pd.to_numeric(merged_df['Num_Bookings_Grouped'], errors = 'coerce')


# In[45]:


print(merged_df['Num_Bookings_Grouped'].value_counts())


# In[46]:


plt.hist(merged_df['LOS'], range = (0,500), bins = 50)
plt.xlabel('Length of Incarceration (Days)')
plt.ylabel('Number of Incarceration Events')
plt.title('LOI Distribution')


# In[47]:


print(merged_df.info())


# In[48]:


numeric = ['Age_Standardized', 'Bond_Amount', 'SVI']

categorical = ['Sex_Gender_Standardized', 'Race_Ethnicity_Standardized', 'Top_Charge', 'State', 'Num_Bookings_Grouped']


# In[49]:


df_encode = merged_df.copy()

encoder = LabelEncoder()

# Initialize encoding_mappings as an empty list
encoding_mappings = []

# Loop through each column and encode
for y in categorical:
    # Fit and transform the column to encode categorical data
    df_encode[y] = encoder.fit_transform(merged_df[y])

    column_mapping = pd.DataFrame({
        'Category': encoder.classes_,
        'Encoded_Label': range(len(encoder.classes_))
    })

    column_mapping['Column'] = y

    encoding_mappings.append(column_mapping)

encoding_df = pd.concat(encoding_mappings, ignore_index=True)

print(encoding_df)



# In[50]:


print(df_encode.head())


# In[51]:


print(df_encode.info())


# In[52]:


print(df_encode.isna().sum())



# In[53]:


#just drop values with unknown year, not many
#same with Num Bookings

df_encode = df_encode.dropna(subset = ['Year', 'Num_Bookings_Grouped'])


# In[54]:


print(df_encode.info())


# In[55]:


#so imputed variable = SVI


# In[ ]:


#MICE IMPUTATION
import miceforest as mf

df_encode = df_encode.reset_index()

kernel = mf.ImputationKernel(
    data=df_encode,
    num_datasets=5,
    random_state=1 # set a seed for reproducibility
)

kernel.mice(10)



# In[ ]:


imputed_datasets = []

for i in range(5):
    completed_data = kernel.complete_data(dataset=i)
    imputed_datasets.append(completed_data)

    print(completed_data.head())
    print(completed_data.info())

    filename = f'JDI_28d_MICE_imputed_dataset_{i+1}.csv'
    completed_data.to_csv(filename, index=False)
    print(f"Saved {filename}")



# In[ ]:


imputed_datasets = []

for i in range(5):
    completed_data = kernel.complete_data(dataset=i)

    plt.hist(completed_data['Number_Of_Children'])
    plt.title('NOC')
    plt.show()
    plt.close()

    plt.title('SVI')
    plt.hist(completed_data['SVI'])
    plt.show()
    plt.close()
