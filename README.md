# PORJECT
PROJECT FINAL : BOSTON HOUSING
# ==========================================
# Boston Housing Data Analysis
# ==========================================

# ------------------------------------------
# 1. Install Required Libraries (JupyterLite)
# ------------------------------------------
import piplite
await piplite.install(['numpy', 'pandas', 'seaborn', 'statsmodels'])

# ------------------------------------------
# 2. Import Libraries
# ------------------------------------------
import io

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

from js import fetch
from scipy.stats import levene, ttest_ind, f_oneway, pearsonr

# ------------------------------------------
# 3. Load Dataset
# ------------------------------------------
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv"

resp = await fetch(URL)
boston_url = io.BytesIO((await resp.arrayBuffer()).to_py())

boston_df = pd.read_csv(boston_url)

# ------------------------------------------
# 4. Explore Dataset
# ------------------------------------------
print(boston_df.head())

print("\nMissing Values:")
print(boston_df.isna().sum())

print("\nDuplicate Rows:")
print(boston_df.duplicated().sum())

print("\nCorrelation Matrix:")
print(boston_df.corr(numeric_only=True))

print("\nDescriptive Statistics:")
print(boston_df.describe(include='all'))

# ==========================================
# Data Visualization
# ==========================================

# ------------------------------------------
# 5. Box Plot of MEDV
# ------------------------------------------
plt.figure(figsize=(8,6))
plt.boxplot(boston_df['MEDV'])
plt.title("Box Plot of MEDV")
plt.ylabel("MEDV")
plt.show()

# ------------------------------------------
# 6. Bar Plot of CHAS
# ------------------------------------------
plt.figure(figsize=(6,4))

chas_counts = boston_df['CHAS'].value_counts().sort_index()

plt.bar(chas_counts.index.astype(str), chas_counts.values)

plt.title("Distribution of CHAS")
plt.xlabel("CHAS")
plt.ylabel("Count")

plt.show()

# ------------------------------------------
# 7. Create AGE Groups
# ------------------------------------------
boston_df['AGE_Group'] = pd.cut(
    boston_df['AGE'],
    bins=3,
    labels=['Low', 'Medium', 'High']
)

# ------------------------------------------
# 8. Box Plot: MEDV vs AGE Group
# ------------------------------------------
plt.figure(figsize=(8,6))

sns.boxplot(
    x='AGE_Group',
    y='MEDV',
    data=boston_df
)

plt.title("MEDV by AGE Groups")
plt.xlabel("AGE Group")
plt.ylabel("MEDV")

plt.show()

# ------------------------------------------
# 9. Scatter Plot (NOX vs INDUS)
# ------------------------------------------
plt.figure(figsize=(8,6))

sns.scatterplot(
    x='NOX',
    y='INDUS',
    data=boston_df
)

plt.title("Scatter Plot of NOX vs INDUS")
plt.xlabel("NOX")
plt.ylabel("INDUS")

plt.show()

# ------------------------------------------
# 10. Histogram of PTRATIO
# ------------------------------------------
plt.figure(figsize=(8,6))

plt.hist(
    boston_df['PTRATIO'],
    bins=20
)

plt.title("Histogram of PTRATIO")
plt.xlabel("PTRATIO")
plt.ylabel("Frequency")

plt.show()

# ==========================================
# Statistical Analysis
# ==========================================

# ------------------------------------------
# 11. Levene Test
# ------------------------------------------
chas_yes = boston_df[boston_df['CHAS'] == 1]['MEDV']
chas_no = boston_df[boston_df['CHAS'] == 0]['MEDV']

stat, p = levene(chas_yes, chas_no)

print("\nLevene Test")
print("Statistic:", stat)
print("P-value:", p)

# ------------------------------------------
# 12. Independent T-Test
# ------------------------------------------
t_stat, p_value = ttest_ind(chas_yes, chas_no)

print("\nIndependent T-Test")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# ------------------------------------------
# 13. One-Way ANOVA
# ------------------------------------------
group1 = boston_df[boston_df['AGE_Group'] == 'Low']['MEDV']
group2 = boston_df[boston_df['AGE_Group'] == 'Medium']['MEDV']
group3 = boston_df[boston_df['AGE_Group'] == 'High']['MEDV']

f_stat, p_value = f_oneway(group1, group2, group3)

print("\nOne-Way ANOVA")
print("F-statistic:", f_stat)
print("P-value:", p_value)

# ------------------------------------------
# 14. Pearson Correlation
# ------------------------------------------
corr, p_value = pearsonr(
    boston_df['NOX'],
    boston_df['INDUS']
)

print("\nPearson Correlation")
print("Correlation:", corr)
print("P-value:", p_value)

# ------------------------------------------
# 15. Simple Linear Regression
# ------------------------------------------
X = boston_df['DIS']
y = boston_df['MEDV']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\nLinear Regression Summary")
print(model.summary())


import statsmodels.api as sm

# Independent variable
X = boston_df['DIS']

# Dependent variable
y = boston_df['MEDV']

# Add intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display results
print(model.summary())


# Independent variable
X = boston_df['DIS']

# Dependent variable
y = boston_df['MEDV']

# Add intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Display results
print(model.summary())
