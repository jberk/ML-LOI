import pandas as pd


keep_columns = ['_id', 'Race_Ethnicity_Standardized', 'Sex_Gender_Standardized',
                'meta.State', 'meta.County', 'meta.num_bookings', 'meta.length_of_stay',
                'Bond_Standardized.value', 'Age_Standardized', 'Top_Charge', 'meta.Facility_Name','meta.County',
                'meta.first_seen', 'meta.Jail_ID']

#Northeast#


NJ_df = pd.read_csv('NJ_people.csv')

col_list = NJ_df.columns.to_list()

print(col_list)

NJ_df['meta.first_seen'] = pd.to_datetime(NJ_df['meta.first_seen'])

print(NJ_df['meta.first_seen'].max())
print(NJ_df['meta.first_seen'].min())


#event_counts = NJ_df.groupby(["_id", "Commitment_Date"]).size().reset_index(name="Count")

#print(event_counts)

#print(event_counts.value_counts())

#duplicates = event_counts[event_counts["Count"] > 1]
#print(duplicates)



#CT_df = pd.read_csv('CT_people.csv') exclude for now since no top charge
ME_df = pd.read_csv('ME_people.csv')
NH_df = pd.read_csv('NH_people.csv')
NY_df = pd.read_csv('NY_people.csv')
#MA_df = pd.read_csv('MA_people.csv') exclude for now since no top charge or bond value
#PA_df = pd.read_csv('PA_people.csv') exclude for now since no top charge or bond value


NJ_df_fil = NJ_df[keep_columns]
#CT_df_fil = CT_df[keep_columns]
ME_df_fil = ME_df[keep_columns]
NH_df_fil = NH_df[keep_columns]
NY_df_fil = NY_df[keep_columns]
#MA_df_fil = MA_df[keep_columns]
#PA_df_fil = PA_df[keep_columns]



df = pd.concat([NJ_df_fil, ME_df_fil, NH_df_fil, NY_df_fil], ignore_index=True)

NorthEast_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

print(NorthEast_df.info())

NorthEast_df.to_csv('NE_Jail_data_9.10.csv')

#Pacific#

CA_df = pd.read_csv('CA_people.csv')
WA_df =  pd.read_csv('WA_people.csv')
OR_df =  pd.read_csv('OR_people.csv')


CA_df_fil = CA_df[keep_columns]
WA_df_fil = WA_df[keep_columns]
OR_df_fil = OR_df[keep_columns]


df = pd.concat([CA_df_fil,WA_df_fil,OR_df_fil], ignore_index=True)

Pacific_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

print(Pacific_df.info())

Pacific_df.to_csv('Pacific_Jail_data_9.10.csv')


### Mountain ###

CO_df = pd.read_csv('CO_people.csv')
AZ_df = pd.read_csv('AZ_people.csv')
UT_df = pd.read_csv('UT_people.csv')
MT_df = pd.read_csv('MT_people.csv')
NV_df = pd.read_csv('NV_people.csv')
NM_df = pd.read_csv('NM_people.csv')
WY_df = pd.read_csv('WY_people.csv')
ID_df = pd.read_csv('ID_people.csv')


CO_df_fil = CO_df[keep_columns]
AZ_df_fil = AZ_df[keep_columns]
UT_df_fil = UT_df[keep_columns]
MT_df_fil = MT_df[keep_columns]
NV_df_fil = NV_df[keep_columns]
NM_df_fil = NM_df[keep_columns]
WY_df_fil = WY_df[keep_columns]
ID_df_fil = ID_df[keep_columns]



df = pd.concat([CO_df_fil,AZ_df_fil,UT_df_fil,MT_df_fil,NV_df_fil,NM_df_fil,WY_df_fil,ID_df_fil], ignore_index=True)


Mountain_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

Mountain_df.to_csv('Mountain_Jail_data_9.10.csv')

print(Mountain_df.info())

### Midwest - ENC ###

IL_df = pd.read_csv('IL_people.csv')
IN_df = pd.read_csv('IN_people.csv')
MI_df = pd.read_csv('MI_people.csv')
OH_df = pd.read_csv('OH_people.csv')
WI_df = pd.read_csv('WI_people.csv')


IL_df_fil = IL_df[keep_columns]
IN_df_fil = IN_df[keep_columns]
MI_df_fil = MI_df[keep_columns]
OH_df_fil = OH_df[keep_columns]
WI_df_fil = WI_df[keep_columns]

df = pd.concat([IL_df_fil,IN_df_fil,MI_df_fil,OH_df_fil,WI_df_fil], ignore_index=True)




ENC_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

print(ENC_df.info())

ENC_df.to_csv('ENC_Jail_data_9.10.csv')


### Midwest - WNC ###

IA_df = pd.read_csv('IA_people.csv')
KS_df = pd.read_csv('KS_people.csv')
MN_df = pd.read_csv('MN_people.csv')
MO_df = pd.read_csv('MO_people.csv')
NE_df = pd.read_csv('NE_people.csv')
ND_df = pd.read_csv('ND_people.csv')
SD_df = pd.read_csv('SD_people.csv')


IA_df_fil = IA_df[keep_columns]
KS_df_fil = KS_df[keep_columns]
MN_df_fil = MN_df[keep_columns]
MO_df_fil = MO_df[keep_columns]
NE_df_fil = NE_df[keep_columns]
ND_df_fil = ND_df[keep_columns]
SD_df_fil = SD_df[keep_columns]

df = pd.concat([IA_df_fil,KS_df_fil,MN_df_fil,MO_df_fil,NE_df_fil,ND_df_fil,SD_df_fil], ignore_index=True)

WNC_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

print(WNC_df.info())


WNC_df.to_csv('WNC_Jail_data_9.10.csv')


import pandas as pd
import numpy as np


#South Atl#

FL_df = pd.read_csv('FL_people.csv')
GA_df = pd.read_csv('GA_people.csv')
MD_df = pd.read_csv('MD_people.csv')
NC_df = pd.read_csv('NC_people.csv')
VA_df = pd.read_csv('VA_people.csv')
SC_df = pd.read_csv('SC_people.csv')
WV_df = pd.read_csv('WV_people.csv')

keep_columns = ['_id', 'Race_Ethnicity_Standardized', 'Sex_Gender_Standardized',
                'meta.State', 'meta.County', 'meta.num_bookings', 'meta.length_of_stay',
                'Bond_Standardized.value', 'Age_Standardized', 'Top_Charge', 'meta.Facility_Name',
                'meta.first_seen', 'meta.Jail_ID']


keep_columns_no_date = ['_id', 'Race_Ethnicity_Standardized', 'Sex_Gender_Standardized',
                'meta.State', 'meta.County', 'meta.num_bookings', 'meta.length_of_stay',
                'Bond_Standardized.value', 'Age_Standardized', 'Top_Charge', 'meta.Facility_Name',
                'meta.Jail_ID']


FL_df_fil = FL_df[keep_columns]
GA_df_fil = GA_df[keep_columns]
MD_df_fil = MD_df[keep_columns]
NC_df_fil = NC_df[keep_columns]
SC_df_fil = SC_df[keep_columns]
VA_df_fil = VA_df[keep_columns]
WV_df_fil = WV_df[keep_columns_no_date]

WV_df_fil['meta.first_seen'] = np.nan

"""
for name, frame in zip(
    ['FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV'],
    [FL_df_fil, GA_df_fil, MD_df_fil, NC_df_fil, SC_df_fil, VA_df_fil, WV_df_fil]
):
    dup_cols = frame.columns[frame.columns.duplicated()]
    if not dup_cols.empty:
        print(f"{name} has duplicate columns: {dup_cols.tolist()}")
"""


df = pd.concat([FL_df_fil, GA_df_fil, MD_df_fil, NC_df_fil, SC_df_fil, VA_df_fil, WV_df_fil], ignore_index=True)

SouthAtl_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

print(SouthAtl_df.info())


print(SouthAtl_df['State'].value_counts())


SouthAtl_df.to_csv('SouthAtl_Jail_data_9.10.csv')


import pandas as pd

keep_columns = ['_id', 'Race_Ethnicity_Standardized', 'Sex_Gender_Standardized',
                'meta.State', 'meta.County', 'meta.num_bookings', 'meta.length_of_stay',
                'Bond_Standardized.value', 'Age_Standardized', 'Top_Charge', 'meta.Facility_Name','meta.County',
                'meta.first_seen', 'meta.Jail_ID']
#W+E South#

OK_df = pd.read_csv('OK_people.csv')
AR_df = pd.read_csv('AR_people.csv')
LA_df = pd.read_csv('LA_people.csv')
AL_df = pd.read_csv('AL_people.csv')
MS_df = pd.read_csv('MS_people.csv')
KY_df = pd.read_csv('KY_people.csv')
TN_df = pd.read_csv('TN_people.csv')
TX_df = pd.read_csv('TX_people.csv')



col_list = TX_df.columns.to_list()

print(col_list)

TX_df['meta.first_seen'] = pd.to_datetime(TX_df['meta.first_seen'], errors='coerce')

print(TX_df['meta.first_seen'].max())
print(TX_df['meta.first_seen'].min())



print(TX_df['meta.first_seen'].isna().sum())


print(TX_df.info())



TX_df_fil = TX_df[keep_columns]
OK_df_fil = OK_df[keep_columns]
AL_df_fil = AL_df[keep_columns]
AR_df_fil = AR_df[keep_columns]
KY_df_fil = KY_df[keep_columns]
MS_df_fil = MS_df[keep_columns]
TN_df_fil = TN_df[keep_columns]
LA_df_fil = LA_df[keep_columns]


df = pd.concat([TX_df_fil, OK_df_fil, AL_df_fil, AR_df_fil, KY_df_fil, MS_df_fil, TN_df_fil, LA_df_fil], ignore_index=True)

WESouth_df = df.rename(columns={'meta.County': 'County', 'meta.State': 'State', 'Bond_Standardized.value':'Bond_Amount',
                        'meta.length_of_stay':'LOS', 'meta.num_bookings':'Num_Bookings', 'meta.Facility_Name':'Jail_Name'})

print(WESouth_df.info())

WESouth_df.to_csv('WESouth_Jail_data_9.10.csv')

#national_df = pd.concat(NorthEast_df, Pacific_df, Mountain_df,ENC_df,WNC_df,SouthAtl_df,WESouth_df)

#print(national_df.info())
