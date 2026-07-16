import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

df_raw = pd.read_csv('dataset_2019_2020.csv')

df_pivot = df_raw.pivot_table(
    index=['Country Name', 'Country Code'],
    columns='Series Name',
    values=['2010 [YR2010]', '2019 [YR2019]'],
    aggfunc='first'
)
print(df_pivot.columns.values)
print(max(df_pivot['2010 [YR2010]', 'Fertility rate, total (births per woman)']))

df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
df_pivot.reset_index(inplace=True)
df_pivot.columns = [col.replace('2010 [YR2010]_', '2010_').replace('2019 [YR2019]_', '2019_')
                    for col in df_pivot.columns]

df = pd.DataFrame({
    'Country': df_pivot['Country Name'],
    'Country_Code': df_pivot['Country Code'],
    'Fertility_2010': pd.to_numeric(df_pivot['2010_Fertility rate, total (births per woman)'], errors='coerce'),
    'GDP_Growth_2010': pd.to_numeric(df_pivot['2010_GDP per capita growth (annual %)'], errors='coerce'),
    'Life_Expectancy_2010': pd.to_numeric(df_pivot['2010_Life expectancy at birth, total (years)'], errors='coerce'),
    'Net_Migration_2010': pd.to_numeric(df_pivot['2010_Net migration'], errors='coerce'),
    'Population_2010': pd.to_numeric(df_pivot['2010_Population, total'], errors='coerce'),
    'Population_2019': pd.to_numeric(df_pivot['2019_Population, total'], errors='coerce'),
    'Pop_Growth_Rate_2019': pd.to_numeric(df_pivot['2019_Population growth (annual %)'], errors='coerce')
})

# migration rate = net migration / population * 1000
df['Migration_Rate_2010'] = (df['Net_Migration_2010'] / df['Population_2010']) * 1000

df['Population_Change'] = df['Population_2019'] - df['Population_2010']
df['Population_Change_Percent'] = ((df['Population_2019'] - df['Population_2010']) / df['Population_2010']) * 100
df['Population_Change_Millions'] = df['Population_Change'] / 1_000_000



key_columns = ['Fertility_2010', 'GDP_Growth_2010', 'Life_Expectancy_2010',
               'Migration_Rate_2010', 'Population_2010', 'Population_2019']

# need to specify the key columns
df_clean = df.dropna(subset=key_columns).copy()



# SHOW CORRELATION

corr_vars = ['Fertility_2010', 'GDP_Growth_2010', 'Life_Expectancy_2010',
             'Migration_Rate_2010', 'Population_Change_Percent']
correlation_matrix = df_clean[corr_vars].corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', square=True, linewidths=1)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# SHOW SCATTER PLOT 
## Display  realtion between fertility rate and population change
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
ax1 = axes[0, 0]
scatter1 = ax1.scatter(df_clean['Fertility_2010'], df_clean['Population_Change_Percent'],
                       c=df_clean['Migration_Rate_2010'], cmap='viridis', s=100, alpha=0.6)
ax1.set_xlabel('Fertility Rate (2010)')
ax1.set_ylabel('Population Change %')
ax1.set_title('Fertility vs Population Growth')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Migration Rate')


# relation between migration rate and population change 
ax2 = axes[0, 1]
scatter2 = ax2.scatter(df_clean['Migration_Rate_2010'], df_clean['Population_Change_Percent'],
                       c=df_clean['GDP_Growth_2010'], cmap='plasma', s=100, alpha=0.6)
ax2.set_xlabel('Migration Rate per 1000')
ax2.set_ylabel('Population Change %')
ax2.set_title('Migration vs Population Growth')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='GDP Growth %')

# relation between gdp growth and population change
ax3 = axes[1, 0]
scatter3 = ax3.scatter(df_clean['GDP_Growth_2010'], df_clean['Population_Change_Percent'],
                       c=df_clean['Fertility_2010'], cmap='RdYlGn_r', s=100, alpha=0.6)
ax3.set_xlabel('GDP Growth %')
ax3.set_ylabel('Population Change %')
ax3.set_title('GDP Growth vs Population Growth')
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Fertility Rate')

# relation between life expectancy and population change
ax4 = axes[1, 1]
scatter4 = ax4.scatter(df_clean['Life_Expectancy_2010'], df_clean['Population_Change_Percent'],
                       s=np.abs(df_clean['Population_Change_Millions']) * 20,
                       c=df_clean['Fertility_2010'], cmap='coolwarm', alpha=0.6)
ax4.set_xlabel('Life Expectancy')
ax4.set_ylabel('Population Change %')
ax4.set_title('Life Expectancy vs Population Growth')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.3)
plt.colorbar(scatter4, ax=ax4, label='Fertility Rate')

plt.tight_layout()
plt.show()


# calculate zscore
df_clean['z_fertility'] = zscore(df_clean['Fertility_2010'])
df_clean['z_migration'] = zscore(df_clean['Migration_Rate_2010'])
df_clean['z_life_exp'] = zscore(df_clean['Life_Expectancy_2010'])
df_clean['z_gdp'] = zscore(df_clean['GDP_Growth_2010'])
#
df_clean['Score_V1'] = (df_clean['z_fertility'] + df_clean['z_migration'] - df_clean['z_life_exp'])
df_clean['Score_V2'] = (df_clean['z_fertility'] + df_clean['z_migration'] - df_clean['z_life_exp'] + 0.5 * df_clean['z_gdp'])

corr_v1 = df_clean['Score_V1'].corr(df_clean['Population_Change_Percent'])
corr_v2 = df_clean['Score_V2'].corr(df_clean['Population_Change_Percent'])

# plt.figure(figsize=(10,6))

# # Score_V1
# sns.histplot(df_clean['Score_V1'], bins=20, kde=True, color='skyblue', label='Score V1')

# # Score_V2
# sns.histplot(df_clean['Score_V2'], bins=20, kde=True, color='orange', label='Score V2', alpha=0.6)

# plt.xlabel('Composite Score')
# plt.ylabel('Number of Countries')
# plt.title('Distribution of Composite Scores')
# plt.legend()
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax1 = axes[0]
ax1.scatter(df_clean['Score_V1'], df_clean['Population_Change_Percent'],
            alpha=0.6, s=100, color='steelblue')
ax1.set_xlabel('Score V1')
ax1.set_ylabel('Population Change %')
ax1.set_title(f'Score V1 vs Actual Growth (r={corr_v1:.3f})')
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.3)

z = np.polyfit(df_clean['Score_V1'], df_clean['Population_Change_Percent'], 1)
p = np.poly1d(z)
ax1.plot(df_clean['Score_V1'].sort_values(), p(df_clean['Score_V1'].sort_values()),
         "r--", linewidth=2, alpha=0.7, label='Trend')
ax1.legend()

ax2 = axes[1]
ax2.scatter(df_clean['Score_V2'], df_clean['Population_Change_Percent'],
            alpha=0.6, s=100, color='darkgreen')
ax2.set_xlabel('Score V2')
ax2.set_ylabel('Population Change %')
ax2.set_title(f'Score V2 vs Actual Growth (r={corr_v2:.3f})')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.3)

z2 = np.polyfit(df_clean['Score_V2'], df_clean['Population_Change_Percent'], 1)
p2 = np.poly1d(z2)
ax2.plot(df_clean['Score_V2'].sort_values(), p2(df_clean['Score_V2'].sort_values()),
         "r--", linewidth=2, alpha=0.7, label='Trend')
ax2.legend()

plt.tight_layout()
plt.show()


def assign_demographic_group(row):
    fertility = row['Fertility_2010']
    migration = row['Migration_Rate_2010']
    life_exp = row['Life_Expectancy_2010']

    if fertility > 3.5 and migration > 0:
        return 'High Growth'
    elif fertility > 2.5 and migration > -2:
        return 'Moderate Growth'
    elif fertility < 2.1 and life_exp > 75:
        return 'Low Growth'
    elif fertility < 2.1 and migration < 0:
        return 'Declining'
    else:
        return 'Stable'

df_clean['Predicted_Group'] = df_clean.apply(assign_demographic_group, axis=1)

def categorize_actual_growth(percent):
    if percent > 15:
        return 'High Growth'
    elif percent > 8:
        return 'Moderate Growth'
    elif percent > 2:
        return 'Low Growth'
    elif percent > -2:
        return 'Stable'
    else:
        return 'Declining'

df_clean['Actual_Group'] = df_clean['Population_Change_Percent'].apply(categorize_actual_growth)

group_colors = {
    'High Growth': '#27ae60',
    'Moderate Growth': '#3498db',
    'Low Growth': '#f39c12',
    'Stable': '#95a5a6',
    'Declining': '#e74c3c'
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for group in df_clean['Predicted_Group'].unique():
    data = df_clean[df_clean['Predicted_Group'] == group]
    ax1.scatter(data['Fertility_2010'], data['Migration_Rate_2010'],
                c=group_colors.get(group, '#000000'), label=group, s=100, alpha=0.7)
ax1.set_xlabel('Fertility Rate 2010')
ax1.set_ylabel('Migration Rate 2010')
ax1.set_title('Predicted Groups Based on Fertility & Migration')
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
ax1.axvline(x=2.1, color='red', linestyle='--', alpha=0.5, label='Replacement Rate')
ax1.legend()
ax1.grid(True, alpha=0.3)

for group in df_clean['Actual_Group'].unique():
    data = df_clean[df_clean['Actual_Group'] == group]
    ax2.scatter(data['Fertility_2010'], data['Migration_Rate_2010'],
                c=group_colors.get(group, '#000000'), label=group, s=100, alpha=0.7)
ax2.set_xlabel('Fertility Rate 2010')
ax2.set_ylabel('Migration Rate 2010')
ax2.set_title('Actual Groups Based on Population Change')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
ax2.axvline(x=2.1, color='red', linestyle='--', alpha=0.5, label='Replacement Rate')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

output_df = df_clean[['Country', 'Country_Code', 'Fertility_2010', 'Migration_Rate_2010',
                       'GDP_Growth_2010', 'Life_Expectancy_2010',
                       'Population_Change_Percent', 'Score_V1', 'Score_V2',
                       'Predicted_Group', 'Actual_Group']]
output_df.to_csv('analysis_results.csv', index=False)
