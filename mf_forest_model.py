# train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load your data
data = pd.read_csv('mf.csv')

#new
relevant_data = data[['scheme_name', 'returns_1yr', 'returns_3yr', 'returns_5yr']]
# relevant_data.set_index('scheme_name', inplace=True)

# Save the relevant data to a joblib file
joblib.dump(relevant_data, 'scheme_returns.pkl')

data.drop('fund_manager', axis=1, inplace=True)

# Define label and features
label = 'scheme_name'
features = data.drop(label, axis=1)
features.drop(['sortino', 'sd', 'sub_category', 'amc_name', 'fund_size_cr', 'fund_age_yr', 'rating'], axis=1, inplace=True)

# Handle missing values
features.replace('-', np.nan, inplace=True)
numeric_columns = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 'beta', 'sharpe', 'risk_level', 'returns_1yr', 'returns_3yr', 'returns_5yr']
features[numeric_columns] = features[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing numeric values with mean
for feature in features.columns:
    if pd.api.types.is_numeric_dtype(features[feature]):
        mean_value = features[feature].mean()
        features[feature].fillna(mean_value, inplace=True)

# One-hot encode categorical features
features = pd.get_dummies(features, columns=['category'], drop_first=True)

# Scale numeric features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[numeric_columns])
features_scaled_df = pd.DataFrame(features_scaled, columns=numeric_columns)

# Combine scaled numeric data with one-hot encoded categorical data
features_final = pd.concat([features_scaled_df.reset_index(drop=True), features.drop(numeric_columns, axis=1).reset_index(drop=True)], axis=1)

# Encode target variable (scheme_name)
label_encoder = LabelEncoder()
data['scheme_name_encoded'] = label_encoder.fit_transform(data['scheme_name'])
target = data['scheme_name_encoded']

# Handle class imbalance using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(features_final, target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Save the model, scaler, and label encoder to disk
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')



print("Model trained and saved successfully.")
