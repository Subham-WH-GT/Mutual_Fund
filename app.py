

from flask import Flask, request, jsonify, render_template,send_from_directory, url_for
from flask import send_file
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
data=joblib.load('scheme_returns.pkl')
# print("hi",data.columns)

# Preprocess input function (same as defined in your predict_using_model.py)
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


# Predict function
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

    # print(data.columns)


    return_values = []
    for scheme in top_n_schemes:
        scheme_data = data[data['scheme_name'] == scheme].iloc[0]  # Getting the first matching scheme
        return_values.append([scheme_data['returns_1yr'], scheme_data['returns_3yr'], scheme_data['returns_5yr']])
    
    # Create a DataFrame for the predicted schemes and returns
    return_df = pd.DataFrame(return_values, columns=['1 Year Return', '3 Year Return', '5 Year Return'], index=top_n_schemes)
    
    # Plot the returns
    plot_path = 'static/returns_plot.png'
    return_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Top N Scheme Returns for 1 Year, 3 Year, and 5 Year')
    plt.xlabel('Scheme Names')
    plt.ylabel('Returns (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Returns Period')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close() 
    # plt.show()    
    
    # Return the predicted schemes
    return top_n_schemes



# Route for the homepage
@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/documentation')
def documentation():
    return render_template('doc.html')

@app.route('/diag')
def diag():
    return send_file('static/Fund Diagrams.pdf', as_attachment=False)


# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        user_input = {
            'min_sip': float(request.form['min_sip']),
            'min_lumpsum': float(request.form['min_lumpsum']),
            'expense_ratio': float(request.form['expense_ratio']),
            'alpha': float(request.form['alpha']),
            'beta': float(request.form['beta']),
            'sharpe': float(request.form['sharpe']),
            'risk_level': float(request.form['risk_level']),
            'returns_1yr': float(request.form['returns_1yr']),
            'returns_3yr': float(request.form['returns_3yr']),
            'returns_5yr': float(request.form['returns_5yr']),
            'category': request.form['category']
        }
        
        # Make prediction
        predictions = predict_scheme(user_input)
        
        # Return the results
        return render_template('result.html', predictions=predictions,plot_url=url_for('view_plot'))


@app.route('/view_plot')
def view_plot():
    return send_file('static/returns_plot.png')      

if __name__ == '__main__':
    app.run(debug=True)

