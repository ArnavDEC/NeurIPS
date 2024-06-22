import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

# Load the data
df = pd.read_excel("/content/Dropout Rate ANOVA Test Statistics_Zenodo Repository (3).xlsx")
income_df = pd.read_excel("/content/Book2.xlsx")
poverty_df = pd.read_excel("/content/Poverty.xlsx")
unemp_df = pd.read_excel("/content/Unemployed.xlsx")
union_df = pd.read_excel("/content/Union.xlsx")

# Multiply dropout rate by 100
df['DRate_9_12'] = pd.to_numeric(df['DRate_9_12'].apply(lambda x: x * 100), errors='coerce')

# Rename columns to uppercase
df.rename(columns={'school': 'SCHOOL', 'Year': 'YEAR', 'School District': 'SCHOOL DISTRICT'}, inplace=True)

# Ensure all relevant columns are numeric
numeric_columns = ['%Minority', 'Tch_Salary', '%LEP', '%At Risk', 'Enrollment 9_12', '%OS_Susp', '%T9', 'Tru_Rate', 'Exp_Stdnt', '%Tchr', 'Avg_ACT', '%Female', '%IS_Susp', 'Tchr_Avg_Expr', '%Black', '%White', '%Multiple']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Combine data from income and poverty datasets
income_dict = income_df.set_index('Parish')[' FIPS'].to_dict()
poverty_dict = poverty_df.set_index('Parish')['Poverty'].to_dict()
unemp_dict = unemp_df.set_index('Parish')['Unemployed'].to_dict()
union_dict = union_df.set_index('Parish')['Union'].to_dict()

def get_fips(parish):
    return income_dict.get(parish, np.nan)

def get_poverty(parish):
    return poverty_dict.get(parish, np.nan)

def get_unemp(parish):
    return unemp_dict.get(parish, np.nan)

def get_union(parish):
    return union_dict.get(parish, np.nan)

# Create the FIPS and Poverty columns based on Parish values
df[' FIPS'] = df['SCHOOL DISTRICT'].apply(get_fips)
df['Poverty'] = df['SCHOOL DISTRICT'].apply(get_poverty)
df['unemp'] = df['SCHOOL DISTRICT'].apply(get_unemp)
df['union'] = df['SCHOOL DISTRICT'].apply(get_union)

# Fill missing values with the mean of the columns
mean_fips = df[' FIPS'].mean()
df[' FIPS'].fillna(mean_fips, inplace=True)
mean_pov = df['Poverty'].mean()
df['Poverty'].fillna(mean_pov, inplace=True)
mean_unemp = df['unemp'].mean()
df['unemp'].fillna(mean_unemp, inplace=True)
mean_union = df['union'].mean()
df['union'].fillna(mean_union, inplace=True)

# Add FIPS and Poverty to the list of numeric columns
numeric_columns.extend([' FIPS', 'Poverty', 'unemp', 'union'])

# Sort by SCHOOL and YEAR
df = df.sort_values(by=['SCHOOL', 'YEAR'])

# Drop rows with NaN values
df = df.dropna(subset=numeric_columns + ['Att_Rate'])

# Define features and target
X = df[numeric_columns]
y = df['Att_Rate']

# Handle missing values
X = X.fillna(X.mean())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#DEAR VIEWER FEEL FREE TO EDIT TRAIN TEST SPLIT


# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to train and evaluate a regression model
def prediction(model):
    reg = model
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, pred, squared=True)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2)
    return pd.DataFrame({'y_Test': y_test, 'Pred': pred})

# Create interaction terms
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interactions = interaction.fit_transform(X)

# Convert the interaction terms into a DataFrame
interaction_features = interaction.get_feature_names_out(X.columns)
X_interactions_df = pd.DataFrame(X_interactions, columns=interaction_features)

# Calculate the correlation matrix including the interaction terms
interaction_correlation_matrix = X_interactions_df.copy()
interaction_correlation_matrix['Att_Rate'] = y.values
correlation_matrix_interactions = interaction_correlation_matrix.corr()

# Identify the highest correlation interaction terms with the target variable
correlations_with_target = correlation_matrix_interactions['Att_Rate'].drop('Att_Rate').sort_values(ascending=False)
print("\nHighest Correlations with Attendance Rate:")
print(correlations_with_target.head(10))

# Identify the 10 lowest correlation interaction terms with the target variable
lowest_correlations_with_target = correlation_matrix_interactions['Att_Rate'].drop('Att_Rate').sort_values(ascending=True).head(10)
print("\n10 Lowest Correlations with Attendance Rate:")
print(lowest_correlations_with_target)

# Create a correlation matrix with only Att_Rate and other variables
correlation_matrix_att_rate = df[numeric_columns + ['Att_Rate']].corr()
att_rate_correlations = correlation_matrix_att_rate['Att_Rate'].drop('Att_Rate')

# Plot the correlation matrix for Att_Rate
plt.figure(figsize=(10, 1))
sns.heatmap(att_rate_correlations.to_frame().T, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation of Att_Rate with Other Variables')
plt.show()

# Train-test split with interaction terms
X_train_interactions, X_test_interactions, y_train, y_test = train_test_split(X_interactions_df, y, test_size=0.01, random_state=42)

# Standardize the data
sc_interactions = StandardScaler()
X_train_interactions = sc_interactions.fit_transform(X_train_interactions)
X_test_interactions = sc_interactions.transform(X_test_interactions)

# Function to train and evaluate a regression model with interaction terms
def prediction_interaction(model):
    reg = model
    reg.fit(X_train_interactions, y_train)
    pred = reg.predict(X_test_interactions)
    mse = mean_squared_error(y_test, pred, squared=True)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2)
    return pd.DataFrame({'y_Test': y_test, 'Pred': pred})

# Example with LinearRegression including interaction terms
print("\nLinearRegression with Interactions:")
result_lr_interactions = prediction_interaction(LinearRegression())
print(result_lr_interactions.head())

