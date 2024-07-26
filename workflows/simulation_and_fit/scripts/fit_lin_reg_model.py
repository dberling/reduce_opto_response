import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

df = pd.concat([pd.read_csv(fname, index_col=0) for fname in list(snakemake.input)])

# Calculate the sum of all entities at each time point
df['sum_all_entities'] = df.iloc[:, 2:].sum(axis=1)

# We need to encode the 'condition' column if it's categorical
df['condition'] = df['condition'].astype('category').cat.codes

# Calculate mutual information
mi = mutual_info_regression(df.iloc[:, 1:-1], df['sum_all_entities'])
mi_df = pd.DataFrame({'feature': df.columns[1:-1], 'mi': mi})
mi_df = mi_df.sort_values(by='mi', ascending=False)

# Select top 100 entities plus the condition feature
top_100_features = mi_df.head(100)['feature'].values
df_selected = df[['time [ms]', 'condition'] + list(top_100_features) + ['sum_all_entities']]

# Splitting data with stratification on 'condition'
X = df_selected.drop(['time [ms]', 'sum_all_entities'], axis=1)
y = df_selected['sum_all_entities']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['condition'], random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sum')
plt.plot(y_pred, label='Predicted Sum', linestyle='--')
plt.plot((y_pred-y_test.values), label='Difference', linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Sum of Observables')
plt.title('Actual vs Predicted Sum of Observables')
plt.savefig(str(snakemake.output[1]))

# Additionally, evaluate the model performance across different conditions
conditions = X_test['condition'].unique()
for condition in conditions:
    idx = X_test['condition'] == condition
    mse_condition = mean_squared_error(y_test[idx], y_pred[idx])
    print(f'Mean Squared Error for condition {condition}: {mse_condition}')


with open(str(snakemake.input[0]), "wb") as output_file:
    pickle.dump(model, output_file)
