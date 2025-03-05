

from utils import get_sector_metrics 
import numpy as np
from scipy.optimize import minimize

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
        "Current Portfolio2": {
            "Expected Return2": np.round(portfolio_return2 * 100, 2),
            "Risk2": round(portfolio_volatility2 * 100, 2),
            "Sharpe Ratio2": np.round(sharpe_ratio2, 2)*10
        },
        "Optimized Portfolio2": {
            "Allocation2": optimized_allocation2,
            "Expected Return2": np.round(optimized_return2 * 100, 2),
            "Risk2": round(optimized_volatility2 * 100, 2),
            "Sharpe Ratio2": np.round(optimized_sharpe2, 2)*10
        }
    }

    print("\nCurrent Portfolio Metrics MVO:", result2["Current Portfolio2"])
    print("\nOptimized Portfolio Allocation MVO:", result2["Optimized Portfolio2"])


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


# Example User Input
user_input = {
    
    "total_investment": 70000,   # Total investment amount in INR
    "current_allocation": {
        "Auto": 10000, "Bank": 9000, "Energy": 10000,
        "Pharma": 15500, "FMCG": 1000, "IT": 14000,
        "Reality": 1000,  "PSU": 1000,
        "PSE": 500, "Infra": 0, "Metal": 7000, "Media": 0
    }
}

calculate_portfolio_metrics(user_input)





