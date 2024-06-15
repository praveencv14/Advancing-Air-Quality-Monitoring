# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:23:56 2024

@author: prave
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN

# Load the dataset
file_path = r'C:\Users\prave\Downloads\Raw_data_1Hr_2023_site_288_Velachery_Res._Area_Chennai_CPCB_1Hr (1).csv'
data = pd.read_csv(file_path)

# Convert the Timestamp column to datetime format and set as index
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data = data.set_index('Timestamp')

# Sample 1000 records from the dataset
data = data.sample(n=1000, random_state=42)


# Exploratory Data Analysis (EDA)
def eda(data):
    print("Basic Information:")
    print(data.info())
    print("\nStatistical Summary:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isna().sum())
    
    # Plotting time series data
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)']])
    plt.title('Time Series Plot of PM2.5, PM10, and NO2')
    plt.xlabel('Timestamp')
    plt.ylabel('Concentration (µg/m³)')
    plt.legend(['PM2.5', 'PM10', 'NO2'])
    plt.show()
    
    # Pairplot for selected features
    sns.pairplot(data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'RH (%)', 'WS (m/s)', 'WD (deg)']])
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(14, 12))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()

eda(data)

# Preprocessing
# Impute missing values with mean for numerical columns
data.fillna(data.mean(), inplace=True)

# Drop columns with no records if necessary
data.dropna(axis=1, how='all', inplace=True)

# Select features and target
features = data[['RH (%)', 'WS (m/s)', 'WD (deg)']]
target_pm25 = data['PM2.5 (µg/m³)']

# Scaling features and target independently
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(features)
scaled_target_pm25 = scaler_target.fit_transform(target_pm25.values.reshape(-1, 1))

# Create dataset for LSTM and CNN
def create_dataset(features, target, look_back=1):
    X, Y = [], []
    for i in range(len(features) - look_back - 1):
        a = features[i:(i + look_back), :]
        X.append(a)
        Y.append(target[i + look_back])
    return np.array(X), np.array(Y)

look_back = 10
X, Y_pm25 = create_dataset(scaled_features, scaled_target_pm25, look_back)

# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

# Define test sizes
test_sizes = [0.20, 0.30]

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for test_size in test_sizes:
    train_rmse_list = []
    test_rmse_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y_pm25[train_index], Y_pm25[test_index]

        # Build the LSTM model
        model_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)  # Output layer for single target variable
        ])

        # Compile the LSTM model
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Train the LSTM model
        model_lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2)

        # Make LSTM predictions
        y_pred_train_lstm = model_lstm.predict(X_train)
        y_pred_test_lstm = model_lstm.predict(X_test)

        # Build the CNN model
        model_cnn = Sequential([
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, X.shape[2])),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)  # Output layer for single target variable
        ])

        # Compile the CNN model
        model_cnn.compile(optimizer='adam', loss='mean_squared_error')

        # Train the CNN model
        model_cnn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=2)

        # Make CNN predictions
        y_pred_train_cnn = model_cnn.predict(X_train)
        y_pred_test_cnn = model_cnn.predict(X_test)

        # Inverse scaling for plotting and interpretation
        y_train_inv = scaler_target.inverse_transform(y_train)
        y_test_inv = scaler_target.inverse_transform(y_test)
        y_pred_train_lstm_inv = scaler_target.inverse_transform(y_pred_train_lstm)
        y_pred_test_lstm_inv = scaler_target.inverse_transform(y_pred_test_lstm)
        y_pred_train_cnn_inv = scaler_target.inverse_transform(y_pred_train_cnn)
        y_pred_test_cnn_inv = scaler_target.inverse_transform(y_pred_test_cnn)

        # Calculate RMSE for both train and test data
        train_rmse_lstm = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_lstm_inv))
        test_rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_lstm_inv))
        train_rmse_cnn = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_cnn_inv))
        test_rmse_cnn = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_cnn_inv))
        train_rmse_list.append((train_rmse_lstm, train_rmse_cnn))
        test_rmse_list.append((test_rmse_lstm, test_rmse_cnn))

    # Print average RMSE over all folds
    avg_train_rmse_lstm = np.mean([x[0] for x in train_rmse_list])
    avg_test_rmse_lstm = np.mean([x[0] for x in test_rmse_list])
    avg_train_rmse_cnn = np.mean([x[1] for x in train_rmse_list])
    avg_test_rmse_cnn = np.mean([x[1] for x in test_rmse_list])
    print(f'Test size {test_size*100}% - LSTM PM2.5 Train RMSE: {avg_train_rmse_lstm:.3f}, Test RMSE: {avg_test_rmse_lstm:.3f}')
    print(f'Test size {test_size*100}% - CNN PM2.5 Train RMSE: {avg_train_rmse_cnn:.3f}, Test RMSE: {avg_test_rmse_cnn:.3f}')

# Training Data Plot for LSTM
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train_inv)), y_train_inv, label='Actual')
plt.plot(range(len(y_pred_train_lstm_inv)), y_pred_train_lstm_inv, label='Predicted')
plt.title(f'Training Data with Test Size: {test_size*100}% for PM2.5 (LSTM)')
plt.xlabel('Time Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()

plt.show()

# Test Data Plot for LSTM
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test_inv)), y_test_inv, label='Actual')
plt.plot(range(len(y_pred_test_lstm_inv)), y_pred_test_lstm_inv, label='Predicted')
plt.title(f'Test Data with Test Size: {test_size*100}% for PM2.5 (LSTM)')
plt.xlabel('Time Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()

plt.show()

# Training Data Plot for CNN
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train_inv)), y_train_inv, label='Actual')
plt.plot(range(len(y_pred_train_cnn_inv)), y_pred_train_cnn_inv, label='Predicted')
plt.title(f'Training Data with Test Size: {test_size*100}% for PM2.5 (CNN)')
plt.xlabel('Time Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()

plt.show()

# Test Data Plot for CNN
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test_inv)), y_test_inv, label='Actual')
plt.plot(range(len(y_pred_test_cnn_inv)), y_pred_test_cnn_inv, label='Predicted')
plt.title(f'Test Data with Test Size: {test_size*100}% for PM2.5 (CNN)')
plt.xlabel('Time Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()

plt.show()

# Future Predictions

# Extend the time index for future predictions (e.g., next 100 hours)
future_time_indices = np.arange(data.shape[0], data.shape[0] + 100).reshape(-1, 1)

# Generate input features for future time indices (e.g., estimated or constant values for simplicity)
# Here we use the mean values of the existing features for simplicity
future_features = np.tile(scaler_features.transform(features.mean().values.reshape(1, -1)), (100, 1))

# Create the future dataset for LSTM input
X_future, _ = create_dataset(np.vstack([scaled_features, future_features]), np.vstack([scaled_target_pm25, np.zeros((100, 1))]), look_back)
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], X_future.shape[2]))

# Use the LSTM model to predict future PM2.5 values
future_predictions_scaled_lstm = model_lstm.predict(X_future[-100:])
future_predictions_lstm = scaler_target.inverse_transform(future_predictions_scaled_lstm)

# Plot the future predictions along with the historical data for LSTM
plt.figure(figsize=(14, 7))

# Plot historical data
plt.plot(range(len(target_pm25)), target_pm25, label='Historical PM2.5')

# Plot future predictions
plt.plot(range(len(target_pm25), len(target_pm25) + 100), future_predictions_lstm, label='Future PM2.5 Predictions (LSTM)', color='red')

plt.xlabel('Time Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.title('Historical Data and Future Predictions for PM2.5 (LSTM)')
plt.show()

# Use the CNN model to predict future PM2.5 values
future_predictions_scaled_cnn = model_cnn.predict(X_future[-100:])
future_predictions_cnn = scaler_target.inverse_transform(future_predictions_scaled_cnn)

# Plot the future predictions along with the historical data for CNN
plt.figure(figsize=(14, 7))

# Plot historical data
plt.plot(range(len(target_pm25)), target_pm25, label='Historical PM2.5')

# Plot future predictions
plt.plot(range(len(target_pm25), len(target_pm25) + 100), future_predictions_cnn, label='Future PM2.5 Predictions (CNN)', color='red')

plt.xlabel('Time Index')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.title('Historical Data and Future Predictions for PM2.5 (CNN)')
plt.show()

# Clustering with Elbow Method
def elbow_method(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, 'bo-', markersize=8)
    plt.title('Elbow Method For Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Square)')
    plt.show()

# Running Elbow Method to find the optimal number of clusters
elbow_method(data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'RH (%)', 'WS (m/s)', 'WD (deg)']])

# Clustering with KMeans
def clustering(data, num_clusters=4):
    # Example clustering with KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'RH (%)', 'WS (m/s)', 'WD (deg)']])
    
    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=data, x='PM2.5 (µg/m³)', y='PM10 (µg/m³)', hue='Cluster', palette='viridis')
    plt.title('Clustering of Air Quality Data')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('PM10 (µg/m³)')
    plt.show()
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'RH (%)', 'WS (m/s)', 'WD (deg)']], data['Cluster'])
    print(f'Silhouette Score: {silhouette_avg:.3f}')
    
    # Additional visualizations
    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=data, x='PM2.5 (µg/m³)', y='NO2 (µg/m³)', hue='Cluster', palette='viridis')
    plt.title('Clustering of PM2.5 and NO2 with KMeans')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('NO2 (µg/m³)')
    plt.show()

    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=data, x='PM2.5 (µg/m³)', y='WS (m/s)', hue='Cluster', palette='viridis')
    plt.title('Clustering of PM2.5 and Wind Speed with KMeans')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('Wind Speed (m/s)')
    plt.show()

clustering(data, num_clusters=4)

# DBSCAN Clustering
eps_values = [0.5, 0.6, 0.7]
min_samples_values = [5, 10, 15]
best_silhouette = -1
best_eps = 0
best_min_samples = 0

# We need to use a separate scaler for clustering to avoid the mismatch error
scaler_clustering = MinMaxScaler(feature_range=(0, 1))
scaled_clustering_features = scaler_clustering.fit_transform(data[['PM2.5 (µg/m³)', 'PM10 (µg/m³)', 'NO2 (µg/m³)', 'RH (%)', 'WS (m/s)', 'WD (deg)']])

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(scaled_clustering_features)
        
        if len(set(dbscan_labels)) > 1:  # Avoid scoring when all points are in the same cluster
            silhouette_avg = silhouette_score(scaled_clustering_features, dbscan_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_eps = eps
                best_min_samples = min_samples

print(f'Best DBSCAN Silhouette Score: {best_silhouette:.3f} (eps={best_eps}, min_samples={best_min_samples})')

# Best DBSCAN Clustering
if best_eps != 0:  # Check if best_eps is not 0
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan_labels = dbscan.fit_predict(scaled_clustering_features)
else:
    print("Warning: best_eps is 0, which is invalid for DBSCAN clustering.")

# Visualizing DBSCAN Clusters
plt.figure(figsize=(10, 6))
plt.scatter(scaled_clustering_features[:, 0], scaled_clustering_features[:, 1], c=dbscan_labels, cmap='viridis')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster Label')
plt.show()
