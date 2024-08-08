import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import ast

conds_per_condition = [np.load(fname) for fname in ast.literal_eval(str(snakemake.input[:-1]))]
sum_per_condition = [conds.sum(axis=1) for conds in conds_per_condition]

times = np.arange(0,250,1)
conditions = np.arange(0,len(conds_per_condition))
comps = ['_'.join([str(int(name)), str(x)]) for (name, x) in np.load(str(snakemake.input[-1]))[:,:2]]
conds_per_condition = [np.array([times for comp in comps]) for condition in conditions]

dfs = []
for condition, conductances_per_comp in zip(conditions, conds_per_condition):
    df = pd.DataFrame()
    df['time [ms]'] = times
    for comp, conductance_over_time in zip(comps, conductances_per_comp):
        df[str(comp)] = conductance_over_time
    dfs.append(df)
df = pd.concat(dfs)

# Calculate the sum of all entities at each time point
df['sum_all_entities'] = df.iloc[:, 2:].sum(axis=1)

# Calculate mutual information
mi = mutual_info_regression(df.iloc[:, 1:-1], df['sum_all_entities'])
mi_df = pd.DataFrame({'feature': df.columns[1:-1], 'mi': mi})
mi_df = mi_df.sort_values(by='mi', ascending=False)

# Select and save top 100 entities
top_100_features = mi_df.head(100)['feature'].values
np.save(str(snakemake.output[2]), top_100_features)
df_selected = df[['time [ms]'] + list(top_100_features) + ['sum_all_entities']]

# Splitting data with stratification on 'condition'
X = df_selected.drop(['time [ms]', 'sum_all_entities'], axis=1)
y = df_selected['sum_all_entities']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


with open(str(snakemake.output[0]), "wb") as output_file:
    pickle.dump(model, output_file)
