

from flask import Flask, request, jsonify, render_template,send_from_directory, url_for,redirect
from flask import send_file
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import spacy
import dash
from dash import html, dash_table

app = Flask(__name__)


rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
data=joblib.load('scheme_returns.pkl')


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






@app.route('/', methods=["GET", "POST"])
def home():
    videos=[
        {"title":"What is Mutual Fund?", "url": "https://www.youtube.com/watch?v=rsFBpGUAZWA"},
        {"title": "How to Start Investing in Mutual Funds", "url": "https://www.youtube.com/watch?v=rcA2PycBQr4"},
        {"title": "Mutual Fund Investment Tips", "url": "https://www.youtube.com/watch?v=vA53lR5wh8M"}
    ]

    if request.method == "POST":
        user_message = request.form.get("user_message")
        if user_message.lower() == "welcome":
            bot_response = (
                "Hey! What would you like to know more about?<br>"
                "1. How this platform Works?<br>"
                "2. Why should you use this Platform?<br>"
                "3. Connect to our Technical Support.<br>" 
                "4. We always value your Feedback,Please share, if any."          
            )
        else:
            intent = get_intent(user_message)
            bot_response = intent_responses.get(intent, "I'm sorry, I didn't understand that. Could you rephrase?")
        return jsonify(bot_response=bot_response)

    return render_template('index.html')


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



intent_keywords = {
    "technical_issue": ["technical", "issue", "problem", "error", "bug"],
    "feedback": ["feedback", "suggestion", "opinion", "thoughts"],
    "support": ["support", "help", "assist", "assistance"],
    "platform_use": ["platform", "use", "functionality", "how to", "guide"],
    "welcome": ["welcome", "hello", "hi", "hey", "greetings"]
}

@app.route('/view_plot')
def view_plot():
    return send_file('static/returns_plot.png')    

@app.route('/page1')
def page1():
    return render_template('page1.html')


nlp = spacy.load("en_core_web_sm")

intent_responses = {    
    "technical_issue": "Please describe the technical issue you're facing.",
    "feedback": "We value your feedback! Please share your thoughts.",
    "support": "All of our Techies are congested at the moment, please drop your query at subhamprof685@gmail.com. We will try to reach you out at the earliest.",
    "platform_use": "Our platform helps you predict the best mutual funds based on your input data. We recommed you to go through the expert videos before you start your investment journey."
}

def get_intent(user_message):
    doc = nlp(user_message.lower())
    # if user_message.lower() == "welcome":
    #     return "welcome"

    # if "technical" in user_message:
    #     return "technical_issue"
    # elif "feedback" in user_message:
    #     return "feedback"
    # elif "support" in user_message:
    #     return "support"
    # elif "platform" in user_message:
    #     return "platform_use"
    # else:
    #     return "unknown"
    for intent, keywords in intent_keywords.items():
        if any(keyword in user_message for keyword in keywords):
            return intent

    # Default response if no intent matches
    return "unknown"



# @app.route("/chat", methods=["GET", "POST"])
# def chatbot():
#     if request.method == "POST":
#         user_message = request.form.get("user_message")
#         if user_message.lower() == "welcome":
#             bot_response = ("Hey Buddy! What would you like to know more about?<br>"
#                             "1.How this platform Works?<br>"
#                             "2. Why should you use this Platform?<br>"
#                             "3. Connect to our Technical Support<br>" 
#                             "4. We always value your Feedback,Please share if any")
#         else:
#             intent = get_intent(user_message)
#             bot_response = intent_responses.get(intent, "I'm sorry, I didn't understand that. Could you rephrase?")
#         return jsonify(bot_response=bot_response)
#     return render_template("index3.html")


BROKERAGE_RATES = {
    "zerodha": {"rate": 0.0003, "max_brokerage": 20},
    "groww": {"rate": 0.0002, "max_brokerage": 15},
    "angelone": {"rate": 0.0005, "max_brokerage": 30},
    "upstox": {"rate": 0.00025, "max_brokerage": 20},
    "kotak":{},
    "hdfc":{},
    "icici":{},
    "motilal":{}
}

broker_data=pd.read_csv('static/Broker Charge.csv')

# @app.route('/')
# def select_broker():
#     return render_template('broker_selection.html')  # Render broker selection page

# Route to render calculator page with selected broker
# @app.route('/calculator/<broker>')
# def calculator(broker):
#     if broker not in BROKERAGE_RATES:
#         return "Broker not found!", 404
#     return render_template('calculator.html', broker=broker.capitalize())

# Route to handle brokerage calculation
@app.route('/calculate-brokerage', methods=['POST'])
def calculate_brokerage():
    data = request.get_json()
    # broker = data['broker'].lower()
    # print('hi',broker)
    quantity = int(data['quantity'])
    price = float(data['price'])
    # broker_row = broker_data[broker_data['Name'] == broker]
    # broker_info = broker_row.iloc[0].to_dict()
    # print(broker_info)

    broker = request.json.get('broker', '').strip().lower()  # Normalize input
    print(broker)

    # Normalize the 'Name' column in the DataFrame to avoid mismatches
    broker_data['Name'] = broker_data['Name'].str.strip().str.lower()

    # Filter the DataFrame for the selected broker
    broker_row = broker_data[broker_data['Name'] == broker]

    broker_info = broker_row.iloc[0].to_dict()
    rate=quantity * price * (broker_info['Broker_Delivery']/100)
    print(rate)
    # print(type(broker_info))
    turnover = quantity * price
    print('turnover',turnover)
    print('sebi',broker_info['SEBI_Delivery'])
    tranx=broker_info['Exchange_Deli']/100
    print('Tranx',broker_info['Exchange_Deli'])
    gst=0.18*(rate+(broker_info['SEBI_Delivery']/100)+(broker_info['Exchange_Deli']/100))
    rounded_gst = round(gst, 6)
    print('gst',gst)
    print('round_gst',rounded_gst)
    sebi=broker_info['SEBI_Delivery']/100
    print('sebi',sebi)
    stt=quantity*price*(broker_info['STT_Delivery']/100)
    print('stt',stt)
    charges=rate+stt+tranx+gst+sebi
    print('total:',charges)
    credit=turnover-charges
    stamp=0
    # Get the brokerage rates for the selected broker
    if broker not in BROKERAGE_RATES:
        return jsonify({'error': 'Invalid broker selected'}), 400

    
    # rates = broker_data[broker]
    # print(rates)
    
    # brokerage = min(rates['rate'] * turnover, rates['max_brokerage'])
    # gst = 0.18 * brokerage  # 18% GST on brokerage
    # total_charges = brokerage + gst

    return jsonify({'brokerage': round(charges, 2),'turnover':round(turnover, 2),'rate':round(rate, 2),'stt':round(stt, 2),'tranx':tranx,'gst':gst,'sebi':sebi,'stamp':stamp, 'total':round(charges, 2),'credit':round(credit, 2)})

if __name__ == '__main__':
    app.run(debug=True)

