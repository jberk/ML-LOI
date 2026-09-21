#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pickle


# In[ ]:


#pip show miceforest


# In[ ]:


#pip install miceforest


# In[ ]:


#Part 1 - Data Cleaning


# In[ ]:


#check for dropping after consolidating

df = pd.read_csv('ridoc_release.csv')

print(df.info())

df = df.rename(columns={'RandID#': 'ID', 'Total Stay (in Days)': 'LOS', 'Admission Age': 'Age',
                        'Bail Status (Lagged Across all Records)': 'Bail_Status',
                        'Bail Amount (to be paid)': 'Bail_Amount_TBP',
                        'Bail Type (Charge Specific)': 'Bail Type',
                        'Offense Type': 'Offense_Type', 'Admission Type': 'Admission_Type',
                        'Marital Status': 'Marital_Status',
                        'Number of Children': 'Number_Of_Children',
                        'Offense Type':'Offense_Type', 'Most Serious Offense Type':'Most_Serious_Offense_Type'})

# Convert Admission Date
df['Admission_Date'] = pd.to_datetime(df['Admission Date'], format='%m/%d/%y')
df['Admission_Date'] = np.where(df['Admission_Date'].dt.year < 100,
                                pd.to_datetime(df['Admission_Date'].dt.strftime('20%y-%m-%d'), format='%Y-%m-%d'),
                                df['Admission_Date'])


print(df['Admission_Date'].value_counts())

df['year'] = df['Admission_Date'].dt.year



# In[ ]:


print(df['year'].value_counts())


# In[ ]:


zipdf = pd.read_csv('zip_income copy.csv')


# Drop unnecessary columns, including 'Charge Description' and 'Count'
df.drop(columns=['Charge Description', 'Release Type', 'RandIncar#', 'Release Date',
                 'Release Type Collapsed', 'Detainers1', 'Detainers2', 'Detainers3',
                  'Release Age', 'Count','Release Age Category', 'Bail Type', 'Bail Amount',
                 'Release Security', 'Admission Age Category'], inplace=True)


# Filter data to include only entries from 2012 onwards (before 2012 would be error)
df = df[df['year'] >= 2012]

print(df.info())


# In[ ]:


# Add season column based on admission date
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

df['Season'] = df['Admission_Date'].dt.month.apply(get_season)

# Only include ages above 18
df = df[df['Age'] >= 18]

# Function to map median household income by ZIP code
def match_zip_median_income(df, df_zip):
    df_zip['ZIPCODE'] = df_zip['ZIPCODE'].astype(str)
    zip_to_income = dict(zip(df_zip['ZIPCODE'], df_zip['ACS_MEDIAN_HH_INC_ZC']))
    df.loc[df['Zip'].astype(str).str.contains('-'), 'Zip'] = np.nan
    df['Median_Income'] = df['Zip'].astype(str).map(zip_to_income)
    return None

# Assuming zipdf is already loaded
match_zip_median_income(df, zipdf)

# Drop ZIP code (no longer needed)
df.drop(columns=['Zip'], inplace=True)

# Define long stay threshold and classify long stay
long_stay_def = 55
df['Long_stay'] = df['LOS'].apply(lambda x: 1 if x > long_stay_def else 0)

### Bail Status ###

#If held w/o bail, set to unattainable amount ($1 million)
df.loc[df['Bail_Status'] == 'held w/o bail', 'Bail_Amount_TBP'] = 1000000

#Create binary version of Bail_Status - No_Bail

df['No_Bail'] = (df['Bail_Status'] == 'held w/o bail').astype(int)

df['No_Bail'] = df['No_Bail'].replace({0: 'Bail', 1: 'No bail'})

df = df.drop(columns=['Bail_Status']) #no longer needed

### Bail Amount ###

## Replace non-numeric values with NaN
df['Bail_Amount_TBP'] = pd.to_numeric(df['Bail_Amount_TBP'], errors='coerce')

### Nativity ###

#group into top 5 + other - United States, Puerto Rico, Dominican Republic, Cape Verde, Guatamala

# Find the top 5 categories
top_5_categories = df['Nativity'].value_counts().nlargest(5).index

# Replace categories not in the top 5 with 'Other'
df['Nativity'] = df['Nativity'].apply(lambda x: x if x in top_5_categories else 'Other')

### Children ###

df['Number_Of_Children'] = pd.to_numeric(df['Number_Of_Children'], errors='coerce')

bins = [-1, 0, 1, 2, 3, 4, float('inf')]
labels = [0, 1, 2, 3, 4, 5] #5 represents 5+ children

# Categorize the number of children
df['Number_Of_Children'] = pd.cut(df['Number_Of_Children'], bins=bins, labels=labels)

# Map offenses to six main categories
offense_map = {
    "B&E": "B&E",
    "Drug": "Drug",
    "Drug Possession": "Drug",
    "Escape": "Nonviolent",
    "Fraud": "Nonviolent",
    "MV": "Nonviolent",
    "Nonviolent": "Nonviolent",
    "Weapons": "Nonviolent",
    "Pending": "Pending",
    "Sex": "Sex_Offense",
    "Domestic Violence": "Violent",
    "Violent": "Violent",
    "Violpred": "Violent"
}

# Map offenses to categories
df['Offense_Category'] = df['Offense_Type'].map(offense_map)

# Create binary columns for each offense category
offense_categories = ['B&E', 'Drug', 'Nonviolent', 'Pending', 'Sex_Offense', 'Violent']
for category in offense_categories:
    df[category] = df['Offense_Category'].apply(lambda x: 1 if x == category else 0)

# Aggregate to consolidate dataset by ID and Admission_Date
aggregation_dict = {category: 'sum' for category in offense_categories}  # Sum offense counts for each offense category
shared_columns = [col for col in df.columns if col not in ['Offense_Type', 'Offense_Category'] + offense_categories] #all common columns (so exclude offense_type categories)
aggregation_dict.update({col: 'first' for col in shared_columns})  # Take 'first' for other shared columns - does not
                                                                   #matter since should be the same values

# Perform the aggregation (consolidating to single incarceration event)
df = df.groupby(['ID', 'Admission_Date'], as_index=False).agg(aggregation_dict)

# Drop columns that are no longer needed
df.drop(columns=['ID'], inplace = True)

print(df)
print(df.columns)

print(df.info())


# In[ ]:


df = df.drop(columns = ['Admission Date', 'Admission_Date'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


print(df['Number_Of_Children'].value_counts())


# In[ ]:


#Part 2 - Processing


# In[ ]:


### Long stay ###

#filter to only include observations greater than 3 days

df = df[df['LOS'] > 3]

#df = df.drop(columns=['LOS']) #no longer needed

#handling missing values

print(df.info())

df_nandrop = df.copy()
#use this df later


# In[ ]:


import matplotlib.pyplot as plt
plt.hist(df['LOS'], bins = 30, range=(0, 200))



# In[ ]:





# In[ ]:





# In[ ]:


#check again for missing values

print(df.info())


# In[ ]:


#base processing is filtering for LOS < 3 days and imputing missing values
#df.to_csv('LOS_Imputed_NoFilteredYrs.csv')


# In[ ]:


#add label encoding

df_encode = df.copy()

encoder = LabelEncoder()

#encode nominal categorical - Most Serious Offense Type, Admission Type, Season, Race,
#Nativity, Marital Status, Sex

nominal_cat = 'Most_Serious_Offense_Type', 'Admission_Type', 'Season', 'Race', 'Nativity', 'Marital_Status', 'Sex', 'No_Bail'

# Initialize encoding_mappings as an empty list
encoding_mappings = []

# Loop through each column and encode
for y in nominal_cat:
    # Fit and transform the column to encode categorical data
    df_encode[y] = encoder.fit_transform(df[y])

    # Create a DataFrame with the mapping for this column
    column_mapping = pd.DataFrame({
        'Category': encoder.classes_,
        'Encoded_Label': range(len(encoder.classes_))
    })

    # Add the column name to the mapping DataFrame
    column_mapping['Column'] = y

    # Append the column mapping to encoding_mappings
    encoding_mappings.append(column_mapping)

# Combine all individual column mappings into one DataFrame
encoding_df = pd.concat(encoding_mappings, ignore_index=True)

# Print the encoding mappings DataFrame
print(encoding_df)

# Save the encoding mapping to CSV
#encoding_df.to_csv('encoding_mappings.csv', index=False)


# In[ ]:


#encode ordinal categorical manually - Number_Of_Children and Education

#Education
print(df_encode['Education'].value_counts())

education_mapping = {
    'High School Diploma/GED Completion': 0,
    'Some High School': 1,
    'Some College': 2,
    'Less than 9th Grade': 3,
    'Associate\'s Degree': 4,
    'Bachelor\'s Degree': 5,
    'More than Bachelor\'s Degree': 6,
    'Unknown': -1  # or you can choose another code for 'Unknown' if needed
}

# Apply the mapping to the Education column
df_encode['Education'] = df_encode['Education'].map(education_mapping)

print(df['Education'].value_counts())

#Number_Of_Children already encoded, just need to make integer


# In[ ]:


df_encode['Number_Of_Children'] = df_encode['Number_Of_Children'].astype('category')

print(df_encode['Number_Of_Children'].value_counts())

#df_encode.drop(columns = ['Admission_Date'])


# In[ ]:


print(df_encode.info())


# In[ ]:


df_encode['LOS'] = df_encode['LOS'].astype('float64')


# In[ ]:


df_encode['Age'] = df_encode['Age'].astype('float64')


# In[ ]:


df_encode['B&E'] = df_encode['B&E'].astype('float64')


# In[ ]:


df_encode['Drug'] = df_encode['Drug'].astype('float64')


# In[ ]:


df_encode['Nonviolent'] = df_encode['Nonviolent'].astype('float64')


# In[ ]:


df_encode['Pending'] = df_encode['Pending'].astype('float64')


# In[ ]:


df_encode['Sex_Offense'] = df_encode['Sex_Offense'].astype('float64')


# In[ ]:


df_encode['Violent'] = df_encode['Violent'].astype('float64')


# In[ ]:


df_encode['year'] = df_encode['year'].astype('category')


# In[ ]:


print(df_encode.info())


# In[ ]:


for col in df_encode:

     if df_encode[col].dtype == 'int64':

            df_encode[col] = df_encode[col].astype('category')


# In[ ]:


print(df_encode.info())


# In[ ]:


plt.hist(df_encode['Number_Of_Children'])


# In[ ]:


plt.hist(df_encode['Median_Income'])


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

    filename = f'56d_MICE_imputed_dataset_{i+1}.csv'
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

    plt.title('Median_Income')
    plt.hist(completed_data['Median_Income'])
    plt.show()
    plt.close()


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
