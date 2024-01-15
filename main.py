import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sb
import warnings
import time

warnings.filterwarnings('ignore')
df = pd.read_excel('data.xlsx')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 14}
plt.figure(figsize=(15, 5))
plt.plot(df['target'])
plt.title('Index', fontdict=font_properties)
plt.ylabel('Temperature', fontdict=font_properties)
plt.savefig('tempt.png', dpi=300)
plt.show()
X = df[['w', 't', 'a', 'k']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {'LinearRegression': LinearRegression(),
          'SVR': SVR(kernel='linear'),
          'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
          'DecisionTree': DecisionTreeRegressor(random_state=42)}

for model_name, model in models.items():
    start_time = time.time() * 1000
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    end_time = time.time() * 1000

    processing_time = end_time - start_time
    print(f"{model_name} Processing Time: {processing_time:.2f} milliseconds")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Metrics:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2) score: {r2}")

    indices = range(len(y_test))
    plt.plot(indices, y_test, label='Actual')
    plt.plot(indices, y_pred, label='Predicted')
    plt.xlabel('Index', fontdict=font_properties)
    plt.ylabel('Value', fontdict=font_properties)
    plt.legend()
    plt.savefig(f'{model_name}.png', bbox_inches='tight', dpi=300)
    plt.show()
