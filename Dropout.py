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
from sklearn.tree import export_graphviz
import os

# Load the data
df = pd.read_excel("/content/Dropout Rate ANOVA Test Statistics_Zenodo Repository (3).xlsx")
income_df = pd.read_excel("/content/Book2.xlsx")
poverty_df = pd.read_excel("/content/Poverty.xlsx")
unemp_df= pd.read_excel("/content/Unemployed.xlsx")
union_df= pd.read_excel("/content/Union.xlsx")

# Multiply dropout rate by 100
df['DRate_9_12'] = pd.to_numeric(df['DRate_9_12'].apply(lambda x: x * 100), errors='coerce')

# Rename columns to uppercase
df.rename(columns={'school': 'SCHOOL', 'Year': 'YEAR', 'School District': 'SCHOOL DISTRICT'}, inplace=True)

# Ensure all relevant columns are numeric
numeric_columns = ['%Minority', 'Tch_Salary', '%LEP', '%At Risk', 'Enrollment 9_12', '%OS_Susp', '%T9', 'Tru_Rate', 'Att_Rate', 'Exp_Stdnt', '%Tchr', 'Avg_ACT', '%Female', '%IS_Susp', 'Tchr_Avg_Expr', '%Black', '%White', '%Multiple']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Combine data from income and poverty datasets
income_dict = income_df.set_index('Parish')[' FIPS'].to_dict()
poverty_dict = poverty_df.set_index('Parish')['Poverty'].to_dict()
unemp_dict = unemp_df.set_index('Parish')['Unemployed'].to_dict()
union_dict =  union_df.set_index('Parish')['Union'].to_dict()

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
numeric_columns.extend([' FIPS', 'Poverty','unemp','union'])

# Sort by SCHOOL and YEAR
df = df.sort_values(by=['SCHOOL', 'YEAR'])

# Drop rows with NaN values
df = df.dropna(subset=numeric_columns + ['DRate_9_12'])

# Define features and target
X = df[numeric_columns]
y = df['DRate_9_12']
z= df['SCHOOL']
# Handle missing values
X = X.fillna(X.mean())

# Display basic statistics
print("Data Description:\n", df.describe())

# Visualize relationships
sns.pairplot(df, x_vars=numeric_columns, y_vars='DRate_9_12', kind='reg')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

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
    print('R²:', r2)
    if isinstance(model, RandomForestRegressor):  # Check if the model is RandomForestRegressor
        for i, tree_in_forest in enumerate(model.estimators_):
            export_graphviz(tree_in_forest,
                            out_file=f'tree_{i}.dot',
                            feature_names=X.columns,
                            filled=True,
                            rounded=True)
            os.system(f'dot -Tpng tree_{i}.dot -o tree_{i}.png')
    return pd.DataFrame({'y_Test': y_test, 'Pred': pred})


print("RandomForestRegressor:")
result_rf = prediction(RandomForestRegressor())
print(result_rf.head())

# Example with LinearRegression
print("\nLinearRegression:")
result_lr = prediction(LinearRegression())
print(result_lr.head())

# Example with XGBRegressor
print("\nXGBRegressor:")
result_xgb = prediction(XGBRegressor())
print(result_xgb.head())

# Example with DecisionTreeRegressor
print("\nDecisionTreeRegressor:")
result_dt = prediction(DecisionTreeRegressor())
print(result_dt.head())

# Example with SVR
print("\nSupport Vector Regressor (SVR):")
result_svr = prediction(SVR())
print(result_svr.head())

# Example with KNeighborsRegressor
print("\nKNeighborsRegressor:")
result_knn = prediction(KNeighborsRegressor())
print(result_knn.head())

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_columns + ['DRate_9_12']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
