import pandas as pd
rd_schema_2018 = pd.read_json('pu2018_schema.json')
rd_schema_2019 = pd.read_json('pu2019_schema.json')
rd_schema_2020 = pd.read_json('pu2020_schema.json')
df_data2018 = pd.read_csv("pu2018.csv",\
                      names=rd_schema_2018['name'],
                      sep='|',\
                      header=0,\
                      usecols = [
                          'SSUID','PNUM','MONTHCODE','SPANEL','SWAVE',\
                          'WPFINWGT','RIN_UNIV',\
                          'ESEX','TAGE','ERACE','EORIGIN','EEDUC','EMS','EJB1_TYPPAY1',\
                          'TPTRNINC','TPTOTINC','RTANF_MNYN','EMD_SCRNR', 'RSNAP_MNYN', 'TSNAP_AMT', 'TSSSAMT', 'RMESR','EPAYHELP','TTANF_AMT', 'TWIC_AMT'\
                    ]
)
df_data2019 = pd.read_csv("pu2019.csv",\
                      names=rd_schema_2019['name'],\
                      sep='|',\
                      header=0,\
                      usecols = [
                          'SSUID','PNUM','MONTHCODE','SPANEL','SWAVE',\
                          'WPFINWGT','RIN_UNIV',\
                          'ESEX','TAGE','ERACE','EORIGIN','EEDUC','EMS','EJB1_TYPPAY1',\
                          'TPTRNINC','TPTOTINC','RTANF_MNYN','EMD_SCRNR', 'RSNAP_MNYN', 'TSNAP_AMT', 'TSSSAMT', 'RMESR','EPAYHELP','TTANF_AMT', 'TWIC_AMT'\
                    ]
)
df_data2020 = pd.read_csv("pu2020.csv",\
                      names=rd_schema_2020['name'],\
                      sep='|',\
                      header=0,\
                      usecols = [
                          'SSUID','PNUM','MONTHCODE','SPANEL','SWAVE',\
                          'WPFINWGT','RIN_UNIV',\
                          'ESEX','TAGE','ERACE','EORIGIN','EEDUC','EMS','EJB1_TYPPAY1',\
                          'TPTRNINC','TPTOTINC','RTANF_MNYN','EMD_SCRNR', 'RSNAP_MNYN', 'TSNAP_AMT', 'TSSSAMT', 'RMESR','EPAYHELP','TTANF_AMT', 'TWIC_AMT'\
                    ]
)
df_data2019 = df_data2019.loc[(df_data2019['ESEX'] == 2) & (df_data2019['TAGE'] >=  16) & (df_data2019['TAGE'] <= 55) & (df_data2019['EEDUC'] <= 39) & (df_data2019['EMS'] >= 2)]
df_data2020 = df_data2020.loc[(df_data2020['ESEX'] == 2) & (df_data2020['TAGE'] >=  16) & (df_data2020['TAGE'] <= 55) & (df_data2020['EEDUC'] <= 39)& (df_data2020['EMS'] >= 2)]
df_data2018 = df_data2018.loc[(df_data2018['ESEX'] == 2) & (df_data2018['TAGE'] >=  16) & (df_data2018['TAGE'] <= 55) & (df_data2018['EEDUC'] <= 39)& (df_data2018['EMS'] >= 2)]
all_data = pd.concat([df_data2018, df_data2019, df_data2020])
weights = pd.read_csv("lgtwgt2020yr3.csv",sep='|',header=0)
new_df = pd.merge(weights, all_data,  how='inner', left_on=['ssuid', 'pnum', 'panel'], right_on = ['SSUID', 'PNUM', 'SPANEL'])
new_df = new_df.drop(['ssuid', 'pnum', 'panel'], axis=1)
new_df[['TSSSAMT', 'TTANF_AMT', 'TSNAP_AMT', 'TWIC_AMT']] = new_df[['TSSSAMT', 'TTANF_AMT', 'TSNAP_AMT', 'TWIC_AMT']].fillna(0)
new_df['SEAM'] = new_df.apply(lambda x: 1 if x['MONTHCODE'] == 12 else 0, axis=1)
new_df.to_csv('sipp.csv') 