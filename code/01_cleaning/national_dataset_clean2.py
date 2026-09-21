import pandas as pd

NE_df = pd.read_csv('NE_Jail_data_9.10.csv')
Mountain_df = pd.read_csv('Mountain_Jail_data_9.10.csv')
South_Atl_df = pd.read_csv('SouthAtl_Jail_data_9.10.csv')
Pacific_df = pd.read_csv('Pacific_Jail_data_9.10.csv')
ENC_df = pd.read_csv('ENC_Jail_data_9.10.csv')
WNC_df = pd.read_csv('WNC_Jail_data_9.10.csv')
WE_South_df = pd.read_csv('WESouth_Jail_data_9.10.csv')


df = pd.concat([NE_df, Pacific_df, Mountain_df, ENC_df, WNC_df, South_Atl_df, WE_South_df], ignore_index=True)

print(df.info())

print(len(df['_id'].unique()))

import numpy as np



print(df['Bond_Amount'].isna().sum())


# 2. Determine the top N categories (e.g., top 5)
N = 5
top_n_categories = df['Top_Charge'].value_counts().nlargest(N).index
# print(top_n_categories)

# 3. Create a new column to store the grouped categories
# Use a lambda function with np.where to assign 'Other' to values not in the top N list
df['Grouped_Category'] = np.where(df['Top_Charge'].isin(top_n_categories),
                                  df['Top_Charge'],
                                  'Other')


print(df['Grouped_Category'].value_counts())
# 4. (Optional) Group by the new 'Grouped_Category' column to see the aggregated results
# For example, sum the 'Value' for each grouped category
grouped_df = df.groupby('Grouped_Category')['Value'].sum().reset_index()
# print(grouped_df)


print(df['Top_Charge'].value_counts())




keep_columns = ['_id', 'Race_Ethnicity_Standardized', 'Sex_Gender_Standardized',
                'State', 'Num_Bookings', 'LOS', 'Bond_Amount', 'Age_Standardized',
                'Top_Charge', 'Jail_Name', 'meta.first_seen', 'County']



df_fil = df[keep_columns]

print(df_fil.info())

df_na = df_fil.dropna(subset = ['meta.first_seen']) #drops about 5 mil entries

print(df_na.info())

print(df_fil['State'].value_counts())

print(len(df_fil['State'].value_counts()))

df_fil = df_fil.drop_duplicates() #none dropped

print(df_fil.info())


df_fil = df_fil.drop(columns = ['Jail_Name'])

print(df_fil.info())

print(df_fil['Bond_Amount'].isna().sum())

print(df_fil['Top_Charge'].value_counts())


df_fil = df_fil.dropna(subset = ['Bond_Amount'])

print(df_fil.info())


df_paper = df_fil[df_fil['Age_Standardized'] >= 18]

df_paper = df_paper[df_paper['LOS']>3]

df_paper = df_paper[df_paper['Top_Charge'].isin(top_n_categories)]

print(df_paper.info())

import matplotlib.pyplot as plt


plt.hist(df_paper['LOS'], range = (0,500), bins = 50)
plt.xlabel('Length of Incarceration (Days)')
plt.ylabel('Number of Incarceration Events')
plt.title('LOI Distribution')

#df_filtered = df[df[column_name].isin(top_values)]


df_fil.to_csv('National_ML_Jan26_noNanBond.csv')



#CT_df, MA_df, PA_df
