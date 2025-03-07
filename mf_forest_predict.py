# # predict_using_model.py

# import joblib
# import pandas as pd
# import numpy as np

# # Load the saved model, scaler, and label encoder
# rf_model = joblib.load('random_forest_model.pkl')
# scaler = joblib.load('scaler.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

# # Define input data for prediction (example input)
# input_data = {
#     'min_sip': 100, 'min_lumpsum': 100, 'expense_ratio': 0.17,
#     'alpha': -0.72, 'beta': 1, 'sharpe': 1.3, 'risk_level': 6,
#     'returns_1yr': 0.9, 'returns_3yr': 26.1, 'returns_5yr': 11.9, 'category': 'other'
# }

# # Define the feature columns (must match the training phase)
# numeric_features = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 'beta', 'sharpe', 'risk_level', 'returns_1yr', 'returns_3yr', 'returns_5yr']

# # def preprocess_input(user_input, scaler, feature_names, category_columns):
# #     input_df = pd.DataFrame([user_input])
    
# #     # Scale numeric features
# #     scaled_numeric_data = scaler.transform(input_df[numeric_features])
    
# #     # One-hot encode categorical features
# #     categorical_data = pd.get_dummies(input_df[['category']])
    
# #     # Combine the scaled numeric data and the categorical data
# #     combined_data = pd.DataFrame(scaled_numeric_data, columns=numeric_features)
# #     final_data = pd.concat([combined_data, categorical_data], axis=1)
    
# #     # Make sure all category columns are present
# #     for category in category_columns:
# #         if category not in final_data.columns:
# #             final_data[category] = 0
# #     final_data = final_data[feature_names]
    
# #     return final_data 

# def preprocess_input(user_input, scaler, feature_names, category_columns):
#     input_df = pd.DataFrame([user_input])
    
#     # Define numeric features that need to be scaled
#     numeric_features = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 'beta', 'sharpe', 'risk_level', 'returns_1yr', 'returns_3yr', 'returns_5yr']
    
#     # Scale numeric data using the scaler
#     scaled_numeric_data = scaler.transform(input_df[numeric_features])
    
#     # Convert the scaled data to DataFrame with the correct column names
#     scaled_numeric_data_df = pd.DataFrame(scaled_numeric_data, columns=numeric_features)
    
#     # Process categorical data (one-hot encoding)
#     categorical_data = pd.get_dummies(input_df[['category']])
    
#     # Combine scaled numeric data and categorical data
#     combined_data = pd.concat([scaled_numeric_data_df, categorical_data], axis=1)
    
#     # Add any missing category columns with 0 if necessary
#     for category in category_columns:
#         if category not in combined_data.columns:
#             combined_data[category] = 0
    
#     # Reorder columns to match the final model input features
#     final_data = combined_data[feature_names]
    
#     return final_data


# def predict_scheme(model, scaler, feature_names, label_encoder, user_input, category_columns, top_n=8):
#     preprocessed_input = preprocess_input(user_input, scaler, feature_names, category_columns)
    
#     # Get prediction probabilities for all classes
#     class_probabilities = model.predict_proba(preprocessed_input)
    
#     # Get the top N predicted schemes
#     top_n_indices = class_probabilities.argsort()[0][-top_n:][::-1]
#     top_n_schemes = label_encoder.inverse_transform(top_n_indices)
    
#     # Print the predicted schemes and their probabilities
#     for scheme in top_n_schemes:
#         print(f" {scheme}")
    
#     return top_n_schemes

# # Feature names and category columns should match what was used in the training phase
# # feature_names = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 'beta', 'sharpe', 'risk_level', 'returns_1yr', 'returns_3yr', 'returns_5yr', 'category_Other']
# # Feature names
# feature_names = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 
#                  'beta', 'sharpe', 'risk_level', 'returns_1yr', 
#                  'returns_3yr', 'returns_5yr', 'category_Equity', 
#                   'category_Hybrid', 'category_Other','category_Solution Oriented']
# category_columns = ['category_Equity', 'category_Hybrid', 'category_Other','category_Solution Oriented']
# # category_columns = features.columns.difference(numeric_columns).tolist()

# # Run prediction
# predicted_schemes = predict_scheme(rf_model, scaler, feature_names, label_encoder, input_data, category_columns, top_n=10)



# predict_using_model.py

import joblib
import pandas as pd

import joblib
print(joblib.__version__)

# Load the saved model, scaler, and label encoder
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def preprocess_input(user_input, scaler, feature_names, category_columns):
    input_df = pd.DataFrame([user_input])
    
    # Define numeric features that need to be scaled
    numeric_features = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 'beta', 'sharpe', 'risk_level', 'returns_1yr', 'returns_3yr', 'returns_5yr']
    
    # Scale numeric data using the scaler
    scaled_numeric_data = scaler.transform(input_df[numeric_features])
    
    # Convert the scaled data to DataFrame with the correct column names
    scaled_numeric_data_df = pd.DataFrame(scaled_numeric_data, columns=numeric_features)
    
    # Process categorical data (one-hot encoding)
    categorical_data = pd.get_dummies(input_df[['category']])
    
    # Combine scaled numeric data and categorical data
    combined_data = pd.concat([scaled_numeric_data_df, categorical_data], axis=1)
    
    # Add any missing category columns with 0 if necessary
    for category in category_columns:
        if category not in combined_data.columns:
            combined_data[category] = 0
    
    # Reorder columns to match the final model input features
    final_data = combined_data[feature_names]
    
    return final_data


def predict_scheme(user_input, top_n=8):
    feature_names = ['min_sip', 'min_lumpsum', 'expense_ratio', 'alpha', 
                     'beta', 'sharpe', 'risk_level', 'returns_1yr', 
                     'returns_3yr', 'returns_5yr', 'category_Equity', 
                     'category_Hybrid', 'category_Other','category_Solution Oriented']
    category_columns = ['category_Equity', 'category_Hybrid', 'category_Other','category_Solution Oriented']
    
    preprocessed_input = preprocess_input(user_input, scaler, feature_names, category_columns)
    
    # Get prediction probabilities for all classes
    class_probabilities = rf_model.predict_proba(preprocessed_input)
    
    # Get the top N predicted schemes
    top_n_indices = class_probabilities.argsort()[0][-top_n:][::-1]
    top_n_schemes = label_encoder.inverse_transform(top_n_indices)
    
    # Return the predicted schemes and their probabilities
    return top_n_schemes

