

from flask import Flask, request, jsonify, render_template,send_from_directory, url_for,redirect
from flask import send_file
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import spacy
import requests
from bs4 import BeautifulSoup
import dash
from dash import html, dash_table 
from utils import get_sector_metrics 
import numpy as np
from scipy.optimize import minimize
import json

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
    "feedback": ["feedback", "suggestion", "opinion", "thoughts","4"],
    "support": ["support", "help", "assist", "assistance","3"],
    "platform_use": ["platform", "use", "functionality", "how to", "guide","1"],
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

@app.route('/test', methods=['GET'])
def test():
    url = 'https://api.ipoalerts.in/ipos?status=closed'
    headers = {
        'x-api-key': 'a518696c60a470172ff2f77c0ab8746ea559d720f8cb59ed6c907020c671eea8'  # Replace with your actual API key
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return jsonify(data)  # Return API response as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json()
        P = float(data["amount"])  # Principal
        r = float(data["rate"]) / 100  # Convert percentage to decimal
        t = int(data["years"])  # Time in years

        if(t>50):
            t=50

        # Compound Interest Formula
        A = P * (1 + r) ** t
        amount=P 
        earnings=A-P


        return jsonify({
            "future_value": round(A, 2),
        "deposited_amount": round(amount, 2),
        "total_earnings": round(earnings, 2)
        }
        )

    except Exception as e:
        return jsonify({"error": str(e)})  
    
@app.route("/calculate_emi", methods=["POST"])
def calculate_emi():
    try:
        data = request.get_json()
        P = float(data["emiamount"])  # Principal
        r = float(data["emirate"]) / 100 /12  # Convert percentage to decimal
        t = int(data["emiyears"])*12  # Time in years
        

        if(t>30*12):
            t=30*12

        # Compound Interest Formula
        if r == 0:  # Handling zero interest rate
            emi_value = P / (t)
        else:
            # emi_value = (P * r * (1 + r) ** t) / ((1 + r) ** t - 1)
            emi_value = (P * r * ((1 + r) ** t)) / (((1 + r) ** t) - 1)
           


        return jsonify({
            "emi_value": round(emi_value, 2),
        "loan_amount": round(P, 2),
        
        }
        )

    except Exception as e:
        return jsonify({"error": str(e)})  


def scrape_market_news():
    url = "https://www.cnbctv18.com/market"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": "Failed to fetch data"}

    soup = BeautifulSoup(response.text, "html.parser")
    
    news_list = []
    articles = soup.find_all("a", class_="jsx-42d95dab3970c9b7")

    for article in articles:
        title_tag = article.find("h3")  # Extract title
        title = title_tag.text if title_tag else "No Title"
        link = article.get("href")  # Extract link

        timestamp_div = article.find("div", class_="mkt-ts")
        if timestamp_div:
            full_text = timestamp_div.get_text(strip=True)
        else:
            full_text='no'    
    
        # print(f"Title: {title}")
        # print(f"Link: {link}\n")
        # print(f"Timestamp: {full_text}\n")
        news_list.append({"title": title, "link": link, "timestamp": full_text})

    return news_list

@app.route("/api/news", methods=["GET"])
def get_news():
    news = scrape_market_news()
    return jsonify(news)


def convert_numpy(obj):
    """ Convert NumPy objects to JSON serializable formats """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()  # Converts single values properly
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

def optimize_portfolio2(expected_returns, covariance_matrix, risk_free_rate=0.04, max_sector_weight=0.3):
    if not expected_returns or covariance_matrix.empty:
        print("Error: Insufficient data for optimization.")
        return None

    sectors = list(expected_returns.keys())
    num_assets = len(sectors)
    expected_returns_array = np.array([expected_returns[sector] for sector in sectors])

    # Initial equal weights
    initial_weights = np.ones(num_assets) / num_assets

    # Constraints: Weights sum to 1 + sector diversification
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights must be 1
        *[
            {'type': 'ineq', 'fun': lambda w, i=i: max_sector_weight - w[i]}  # Limit individual sector weight
            for i in range(num_assets)
        ]
    ]
    
    # Bounds: No short selling, upper bound set to diversification limit
    bounds = [(0, max_sector_weight) for _ in range(num_assets)]

    # Objective function: Negative Sharpe Ratio
    def negative_sharpe_ratio(weights):
        port_return = np.dot(weights, expected_returns_array)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return -(port_return - risk_free_rate) / (port_volatility + 1e-6)  # Avoid division by zero
        

    # Optimization
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
   

    return result.x if result.success else initial_weights 








def calculate_portfolio_metrics(user_input):

    # data = request.json
    # total_investment = float(data.get("total_investment", 0))
    # current_allocation = {sector: float(value) for sector, value in data.get("current_allocation", {}).items()}
    # optimisation_type = data.get("optimisation_type", "mpt")



    current_allocation = user_input["current_allocation"]
    total_investment = user_input["total_investment"]
    risk_free_rate = 0.04  # 6% annualized risk-free rate

    # Fetch real expected returns and sector risk
    expected_returns, sector_risk, sentiment_scores,expected_returns2,covariance_matrix2 = get_sector_metrics()
    
    if not sentiment_scores:
        print("Error: Sentiment scores are empty. Check yfinance data retrieval.")
        return None

    

    invested_sectors = {sector: amount for sector, amount in current_allocation.items()}
    invested_sectors2 = {s: a for s, a in current_allocation.items()}
    # Calculate portfolio weights (only for invested sectors)

    invested_returns2 = {s: expected_returns2[s] for s in invested_sectors2.keys()}
    invested_cov_matrix2 = covariance_matrix2.loc[invested_sectors2.keys(), invested_sectors2.keys()]

    weights2 = np.array([current_allocation[sector] / total_investment for sector in invested_sectors])
   
    portfolio_return2 = np.dot(weights2, list(invested_returns2.values()))
    # portfolio_volatility2 = np.sqrt(np.dot(weights2.T, np.dot(invested_cov_matrix2, weights2)))
    portfolio_volatility2 = np.sqrt(np.dot(weights2.T, np.dot(covariance_matrix2, weights2)))

    sharpe_ratio2 = (portfolio_return2 - risk_free_rate) / (portfolio_volatility2 + 1e-6)
    optimized_weights2 = optimize_portfolio2(invested_returns2, covariance_matrix2)
    print("optimised Weights",optimized_weights2)
    if optimized_weights2 is None:
        print("Error: Optimization failed.")
        return None

    optimized_allocation2 = {sector: round(total_investment * weights2) for sector, weights2 in zip(invested_sectors2.keys(), optimized_weights2)}
    # optimized_allocation2 = {sector: round(total_investment * optimized_weights2[sector]) for sector in invested_sectors2.keys()}

    optimized_return2 = np.dot(optimized_weights2, list(invested_returns2.values()))
    optimized_volatility2 = np.sqrt(np.dot(optimized_weights2.T, np.dot(invested_cov_matrix2, optimized_weights2)))
    optimized_sharpe2 = (optimized_return2 - risk_free_rate) / (optimized_volatility2 + 1e-6)

    result2 = {
        "Current Portfolio": {
            "Expected Return": np.round(portfolio_return2 * 100, 2),
            "Risk": round(portfolio_volatility2 * 100, 2),
            "Sharpe Ratio": np.round(sharpe_ratio2, 2)*10
        },
        "Optimized Portfolio": {
            "Allocation": optimized_allocation2,
            "Expected Return": np.round(optimized_return2 * 100, 2),
            "Risk": round(optimized_volatility2 * 100, 2),
            "Sharpe Ratio": np.round(optimized_sharpe2, 2)*10
        }
    }

    print("\nCurrent Portfolio Metrics MVO:", result2["Current Portfolio"])
    print("\nOptimized Portfolio Allocation MVO:", result2["Optimized Portfolio"])


    weights = {sector: amount / total_investment for sector, amount in invested_sectors.items()}

    # Expected portfolio return
    portfolio_return = sum(weights[sector] * expected_returns.get(sector, 0) for sector in weights)

    # Portfolio risk calculation using variance-covariance matrix
    weight_array = np.array([weights.get(sector, 0) for sector in invested_sectors.keys()])
    std_devs = np.array([sector_risk.get(sector, 0) for sector in invested_sectors.keys()])
    covariance_matrix = np.outer(std_devs, std_devs) * np.identity(len(std_devs))
    portfolio_variance = np.dot(weight_array, np.dot(covariance_matrix, weight_array.T))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Sharpe ratio (handle case where volatility is zero)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

    # Optimized Allocation Based on Sentiment (only for invested sectors)
    sentiment_weights = {k: v for k, v in sentiment_scores.items() if k in invested_sectors}
    total_sentiment = sum(sentiment_weights.values())

    if total_sentiment > 0:
        sentiment_weights = {sector: weight / total_sentiment for sector, weight in sentiment_weights.items()}
    else:
        sentiment_weights = {sector: 1 / len(invested_sectors) for sector in invested_sectors}  # Equal weights fallback

    optimized_allocation = {sector: round(total_investment * sentiment_weights[sector]) for sector in sentiment_weights}

    # Expected return and risk after optimization
    optimized_return = sum(sentiment_weights[sector] * expected_returns.get(sector, 0) for sector in sentiment_weights)
    optimized_risk = np.sqrt(np.dot(list(sentiment_weights.values()), np.dot(covariance_matrix, list(sentiment_weights.values()))))
    optimized_sharpe = (optimized_return - risk_free_rate) / optimized_risk if optimized_risk > 0 else 0

    result_sentimentbased = {
        "Current Portfolio": {
            "Expected Return": round(portfolio_return * 100, 2),
            "Risk": round(portfolio_volatility * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2)*10
        },
        "Optimized Portfolio": {
            "Allocation": optimized_allocation,
            "Expected Return": round(optimized_return * 100, 2),
            "Risk": round(optimized_risk * 100, 2),
            "Sharpe Ratio": round(optimized_sharpe, 2)*10
        }
    }



    print("Current Portfolio Metrics market Sentiment based:", result_sentimentbased["Current Portfolio"])
    print("Optimized Portfolio Allocation market sentiment based:", result_sentimentbased["Optimized Portfolio"])
    return result_sentimentbased,result2



@app.route('/optimise',methods=["GET", "POST"])
def optimise():
    if request.method=='POST':
        
        data = request.json
        total_investment = float(data.get("total_investment", 0))
        current_allocation = {sector: float(value) for sector, value in data.get("current_allocation", {}).items()}
        optimization_type = data.get("optimization_type", "mpt")  # User-selected optimization type

    # Call function to calculate portfolio metrics
        sentiment_result, mpt_result = calculate_portfolio_metrics({
        "total_investment": total_investment,
        "current_allocation": current_allocation
        })

    # Return result based on user selection
        if optimization_type == "mpt":
            return json.dumps({k: convert_numpy(v) for k, v in mpt_result.items()}), 200, {'Content-Type': 'application/json'}
        else:
            return jsonify(sentiment_result)
    
    else:
        return render_template('optimise.html')

if __name__ == '__main__':
    app.run(debug=True)

