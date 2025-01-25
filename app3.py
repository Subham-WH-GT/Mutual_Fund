# from flask import Flask, request, render_template, jsonify

# app = Flask(__name__)

# # Route to render the form
# @app.route('/')
# def home():
#     return render_template('form.html')  # Render the form page

# # Route to handle the brokerage calculation
# @app.route('/calculate-brokerage', methods=['POST'])
# def calculate_brokerage():
#     data = request.get_json()
#     quantity = int(data['quantity'])
#     price = float(data['price'])
#     order_type = data['orderType']
#     exchange = data['exchange']

#     # Example calculation logic (can be customized)
#     turnover = quantity * price
#     brokerage = min(0.0003 * turnover, 20)  # Example: 0.03% of turnover or max ₹20
#     gst = 0.18 * brokerage  # 18% GST on brokerage
#     total_charges = brokerage + gst

#     return jsonify({'brokerage': round(total_charges, 2)})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Define brokerage rates and rules for each broker
BROKERAGE_RATES = {
    "zerodha": {"rate": 0.0003, "max_brokerage": 20},
    "groww": {"rate": 0.0002, "max_brokerage": 15},
    "angelone": {"rate": 0.0005, "max_brokerage": 30},
    "upstox": {"rate": 0.00025, "max_brokerage": 20},
}

# Route to render broker selection page
@app.route('/')
def select_broker():
    return render_template('broker_selection.html')  # Render broker selection page

# Route to render calculator page with selected broker
@app.route('/calculator/<broker>')
def calculator(broker):
    if broker not in BROKERAGE_RATES:
        return "Broker not found!", 404
    return render_template('calculator.html', broker=broker.capitalize())

# Route to handle brokerage calculation
@app.route('/calculate-brokerage', methods=['POST'])
def calculate_brokerage():
    data = request.get_json()
    broker = data['broker']
    quantity = int(data['quantity'])
    price = float(data['price'])

    # Get the brokerage rates for the selected broker
    if broker not in BROKERAGE_RATES:
        return jsonify({'error': 'Invalid broker selected'}), 400

    rates = BROKERAGE_RATES[broker]
    turnover = quantity * price
    brokerage = min(rates['rate'] * turnover, rates['max_brokerage'])
    gst = 0.18 * brokerage  # 18% GST on brokerage
    total_charges = brokerage + gst

    return jsonify({'brokerage': round(total_charges, 2)})

if __name__ == '__main__':
    app.run(debug=True)

