import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore



sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
df_raw = pd.read_csv('dataset_2019_2020.csv')

# Pivot the long format into wide format
# df_wide = df.pivot(index='Country Name', columns='Series Name', values='2010 [YR2010]').reset_index()
df = df_raw.drop(columns=["Series Code", "Country Code", ])
# Rename columns for simplicity
fertility_df = df[df['Series Name'] == "Fertility rate, total (births per woman)"].copy()
fertility_df = fertility_df.rename(columns={"2010 [YR2010]": "fertility_2010", "2019 [YR2019]": "fertility_2019"})
fertility_df = fertility_df.drop(columns=["Series Name"])
fertility_df['fertility_2010'] = pd.to_numeric(fertility_df['fertility_2010'], errors='coerce')
fertility_df['fertility_2019'] = pd.to_numeric(fertility_df['fertility_2019'], errors='coerce')
# print(fertility_df.head())
final_df = pd.DataFrame()
gdp_df = df[df['Series Name'] == "GDP per capita growth (annual %)"].copy()
gdp_df = gdp_df.rename(columns={"2010 [YR2010]": "gdp_growth_2010", "2019 [YR2019]": "gdp_growth_2019"})
gdp_df = gdp_df.drop(columns=["Series Name"])
gdp_df['gdp_growth_2010'] = pd.to_numeric(gdp_df['gdp_growth_2010'], errors='coerce')
gdp_df['gdp_growth_2019'] = pd.to_numeric(gdp_df['gdp_growth_2019'], errors='coerce')

migration_df = df[df['Series Name'] == "Net migration"].copy()
migration_df = migration_df.rename(columns={"2010 [YR2010]": "net_migration_2010", "2019 [YR2019]": "net_migration_2019"})
migration_df = migration_df.drop(columns=["Series Name"])
migration_df['net_migration_2010'] = pd.to_numeric(migration_df['net_migration_2010'], errors='coerce')
migration_df['net_migration_2019'] = pd.to_numeric(migration_df['net_migration_2019'], errors='coerce')

life_exp_df = df[df['Series Name'] == "Life expectancy at birth, total (years)"].copy()
life_exp_df = life_exp_df.rename(columns={"2010 [YR2010]": "life_exp_2010", "2019 [YR2019]": "life_exp_2019"})
life_exp_df = life_exp_df.drop(columns=["Series Name"])
life_exp_df['life_exp_2010'] = pd.to_numeric(life_exp_df['life_exp_2010'], errors='coerce')
life_exp_df['life_exp_2019'] = pd.to_numeric(life_exp_df['life_exp_2019'], errors='coerce')


net_migration_df = df[df['Series Name'] == "Net migration"].copy()
net_migration_df = net_migration_df.rename(columns={"2010 [YR2010]": "net_migration_2010", "2019 [YR2019]": "net_migration_2019"})
net_migration_df = net_migration_df.drop(columns=["Series Name"])
net_migration_df['net_migration_2010'] = pd.to_numeric(net_migration_df['net_migration_2010'], errors='coerce')
net_migration_df['net_migration_2019'] = pd.to_numeric(net_migration_df['net_migration_2019'], errors='coerce')


pop_df = df[df['Series Name'] == "Population, total"].copy()
pop_df = pop_df.rename(columns={"2010 [YR2010]": "pop_2010", "2019 [YR2019]": "pop_2019"})
pop_df = pop_df.drop(columns=["Series Name"])
pop_df['pop_2010'] = pd.to_numeric(pop_df['pop_2010'], errors='coerce')
pop_df['pop_2019'] = pd.to_numeric(pop_df['pop_2019'], errors='coerce')

# Calculate migration rate (per 1000 people) for better comparability
final_df['migration_rate_2010'] = (net_migration_df['net_migration_2010'] / pop_df['pop_2010']) * 1000

# Calculate overall population change from 2010 to 2019
final_df['pop_change'] = pop_df['pop_2019'] - pop_df['pop_2010']
final_df['pop_change_percent'] = (final_df['pop_change'] / pop_df['pop_2010']) * 100
# set index in final_df as the country name

print(final_df.head())
# merge all dfs into final_df
# final_df = final_df.merge(fertility_df, on='Country Name', how='left')
# final_df = final_df.merge(gdp_df, on='Country Name', how='left')
# final_df = final_df.merge(life_exp_df, on='Country Name', how='left')
# final_df = final_df.merge(net_migration_df, on='Country Name', how='left')
# final_df = final_df.merge(pop_df, on='Country Name', how='left')


# print(final_df.head())
# Remove rows with any missing values in key columns
# key_columns = ['fertility_2010', 'gdp_growth_2010', 'life_exp_2010',
#                'migration_rate_2010', 'pop_2010', 'pop_2019']