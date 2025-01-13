from taipy import Gui
import pandas as pd
import numpy as np
import os

#LOADING THE DATA:
# data = pd.read_csv("hotel_bookings.csv")
# df = pd.DataFrame(data)
# Process the CSV file in chunks
chunks = pd.read_csv("hotel_bookings.csv", chunksize=1000)
processed_chunks = []
for chunk in chunks:
    # Perform any processing you need on the chunk
    chunk.fillna(method='ffill', inplace=True)
    processed_chunks.append(chunk)

df = pd.concat(processed_chunks)
# Optimize memory usage
for col in df.select_dtypes(include=["float", "int"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="float" if df[col].dtype == "float" else "integer")
# Convert categorical columns to category type
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype("category")

df.fillna(method='ffill', inplace=True)
# df_encoded = pd.get_dummies(df, drop_first=True)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop="first", sparse=True)  # Use sparse matrix to save memory
df_encoded = encoder.fit_transform(df.select_dtypes(include=["category"]))

# Create a summary DataFrame for visualization
columns = df.columns
non_null_counts = df.notnull().sum()
data_types = df.dtypes

# Prepare the summary table
summary_data = {
    "Column Name": columns,
    "Non-Null Values": non_null_counts.values,
}

# iterative_imputer = IterativeImputer()
# df = pd.DataFrame(iterative_imputer.fit_transform(df), columns=df.columns)

#Checking which hotels are cancelled more:
city_hotel_cancellations = df.loc[df["hotel"] == "City Hotel", "is_canceled"].sum()
resort_hotel_cancellations = df.loc[df["hotel"] == "Resort Hotel", "is_canceled"].sum()
total_cancellations = df["is_canceled"].count()

city_hotel = (city_hotel_cancellations / total_cancellations) * 100
resort_hotel = (resort_hotel_cancellations / total_cancellations) * 100


cancellation_data = {
    "Hotel Type": ["City Hotel", "Resort Hotel"],
    "Cancellations": [city_hotel_cancellations, resort_hotel_cancellations],
}

# Group data by 'deposit_type' and calculate cancellation counts and rates
deposit_groups = df.groupby("deposit_type")["is_canceled"].agg(["sum", "count"]).reset_index()
deposit_groups.rename(columns={"sum": "Cancellations", "count": "Total Bookings"}, inplace=True)
deposit_groups["Cancellation Rate (%)"] = (deposit_groups["Cancellations"] / deposit_groups["Total Bookings"]) * 100

# Prepare data for visualization
visualization_data = {
    "Deposit Type": deposit_groups["deposit_type"].tolist(),
    "Cancellations": deposit_groups["Cancellations"].tolist(),
    "Non-Cancellations": (deposit_groups["Total Bookings"] - deposit_groups["Cancellations"]).tolist(),
}

#SPLITTING THE DATA:
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_encoded, test_size=0.2, random_state=42)


x_train = train.drop(columns=["is_canceled","reservation_status_Check-Out"])
y_train = train["is_canceled"]
x_test = test.drop(columns=["is_canceled","reservation_status_Check-Out"])
y_test = test["is_canceled"]


#Creating the model:
from sklearn.ensemble import HistGradientBoostingClassifier
model = HistGradientBoostingClassifier(categorical_features=True)

from sklearn.preprocessing import StandardScaler

# Remove constant columns to prevent scaling issues
x_train = x_train.loc[:, (x_train != x_train.iloc[0]).any()]
x_test = x_test.loc[:, x_train.columns]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled, y_train)

# Feature importance
# importances = model.feature_importances_
# feature_names = x_train.columns
# sorted_importances = sorted(zip(importances, feature_names), reverse=True)
# print("Feature Importances:")
# for importance, feature in sorted_importances:
#     print(f"{feature}: {importance:.2f}")

#Prediction
predictions = model.predict(x_test_scaled)
df_pred = pd.DataFrame({"Predictions": predictions, "Actual": y_test})
y_actual = np.array(y_test)
y_predicted = np.array(predictions)

#Metrics
# Avoid division by zero by filtering out zero values
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
# Compute metrics
non_zero_actual = y_actual != 0
mpae = (
    np.mean(np.abs((y_actual[non_zero_actual] - y_predicted[non_zero_actual]) / y_actual[non_zero_actual])) * 100
    if non_zero_actual.any()
    else 0
)
mse = mean_squared_error(y_actual, y_predicted)
r_squared = r2_score(y_actual, y_predicted)
print(mpae, mse, r_squared)

metrics_data = {
    "Metric": ["Mean Absolute Percentage Error (MAPE)", "Mean Squared Error (MSE)", "R-Squared"],
    "Value": [mpae, mse, r_squared],
}


page = """
<|text-center|

# Predicting Hotel Cancellations with Machine Learning


## Introduction
Understanding cancellation patterns is crucial for optimizing hotel operations. In this project, we explore data from a hotel booking dataset to identify factors influencing cancellations and predict cancellations using a regression model.

---

## Loading the Data
<|{summary_data}|table|height=300px|>

The dataset is loaded into a pandas DataFrame to explore its structure and identify missing values or data types that might need preprocessing.

---

## Analyzing Cancellation Patterns
<|layout|columns=1 1|
<|{cancellation_data}|table|>

<|{cancellation_data}|chart|type=bar|x=Hotel Type|y=Cancellations|>
|>

<h6>By grouping the dataset by hotel type, we analyze which type of hotel (City or Resort) has a higher rate of cancellations. This helps identify patterns that may require different strategies for the two hotel types.</h6>

---
# Impact of Deposit Type on Cancellations
<|layout|columns=1 1|
<|{visualization_data}|table|height=300px|>

<|{visualization_data}|chart|type=bar|x=Deposit Type|y[1]=Cancellations|y[2]=Non-Cancellations|stacked=True|width=700px|height=400px|>
|>
## Preprocessing the Data
```python
# Filling missing values
from sklearn.impute import IterativeImputer

df.fillna(method='ffill', inplace=True)
df_encoded = pd.get_dummies(df, drop_first=True)
```
To prepare the dataset for modeling, missing values are filled using forward fill, and categorical variables are encoded using one-hot encoding.

---

## Splitting the Data
```python
# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_encoded, test_size=0.2, random_state=42)

x_train = train.drop(columns=["is_canceled"])
y_train = train["is_canceled"]
x_test = test.drop(columns=["is_canceled"])
y_test = test["is_canceled"]
```
The dataset is split into training and testing sets to ensure unbiased model evaluation. The target variable is `is_canceled`.

---

## Scaling the Data
```python
# Scaling features for better model performance
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```
Feature scaling is performed to normalize the data and improve the performance of the regression model.

---

## Training the Model
```python
# Training the regression model
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor()
model.fit(x_train_scaled, y_train)
```
We train a `HistGradientBoostingRegressor` to predict hotel cancellations. This model is robust and efficient for large datasets.

---

## Making Predictions
```python
# Making predictions on the test set
predictions = model.predict(x_test_scaled)
```
The model's predictions are compared with actual values from the test set, providing insights into its accuracy.

---
# Model Performance Metrics

### Numeric Widgets
# Mean Absolute Percentage Error (MAPE) : <|{mpae}|metric|format=%.2f%%|>
# Mean Squared Error (MSE) : <|{mse}|metric|format=%.2f|>
# R-Squared : <|{r_squared}|metric|format=%.2f|>

### Bar Chart
<|{metrics_data}|chart|type=bar|x=Metric|y=Value|>

## Evaluating the Model
```python
# Calculating the Mean Percentage Absolute Error (MPAE)
import numpy as np

y_actual = np.array(y_test)
y_predicted = np.array(predictions)

non_zero_actual = y_actual != 0
mpae_1 = np.mean(np.abs((y_actual[non_zero_actual] - y_predicted[non_zero_actual]) / y_actual[non_zero_actual])) * 100


```
To assess model performance, we calculate the Mean Percentage Absolute Error (MPAE). A lower MPAE indicates a better model fit.

---

## Conclusion
By analyzing cancellation patterns and building a regression model, we gain actionable insights into factors influencing hotel cancellations. This approach can help hotels optimize booking strategies and reduce cancellation rates.
|>
"""


if __name__== "__main__":
    app = Gui(page)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)