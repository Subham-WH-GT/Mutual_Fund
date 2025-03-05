import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Sector-to-stock mapping
sector_tickers = {
    "Auto": "^CNXAUTO",
    "Bank": "^NSEBANK",
    "Energy": "^CNXENERGY",
    "Pharma": "SUNPHARMA.NS",
    "FMCG": "^CNXFMCG",
    "IT": "^CNXIT",
    "Reality": "^CNXREALTY",
    "PSU": "^CNXPSUBANK",
    "PSE": "^CNXPSE",
    "Infra": "^CNXINFRA",
    "Metal": "^CNXMETAL",
    "Media": "^CNXMEDIA"
}

# Fetch returns & covariance matrix
def get_sector_data():
    sector_returns = {}
    expected_returns = {}

    for sector, ticker in sector_tickers.items():
        try:
            # data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1y", interval="1d", auto_adjust=True)
            
            if "Adj Close" not in data:
                print(f"Skipping {sector}: No 'Adj Close' data available.")
                continue  # Skip if no valid data

            returns = data['Adj Close'].pct_change().dropna()

            if returns.empty:
                print(f"Skipping {sector}: No return data available.")
                continue  # Skip if no return data

            expected_returns[sector] = returns.mean() * 252  # Annualized return
            sector_returns[sector] = returns  # Store full return series

        except Exception as e:
            print(f"Error fetching {sector}: {e}")

    # Convert returns to DataFrame & compute covariance matrix
    if not sector_returns:
        print("Error: No valid sector data fetched. Exiting.")
        return None, None

    returns_df = pd.DataFrame(sector_returns)
    covariance_matrix = returns_df.cov() * 252  # Annualized covariance

    return expected_returns, covariance_matrix

# Portfolio Optimization
def optimize_portfolio(expected_returns, covariance_matrix, risk_free_rate=0.05):
    if not expected_returns or covariance_matrix.empty:
        print("Error: Insufficient data for optimization.")
        return None

    sectors = list(expected_returns.keys())
    num_assets = len(sectors)
    expected_returns_array = np.array([expected_returns[sector] for sector in sectors])

    # Initial equal weights
    initial_weights = np.ones(num_assets) / num_assets

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Bounds: No negative allocations
    bounds = [(0, 1) for _ in range(num_assets)]

    # Objective function: Negative Sharpe Ratio
    def negative_sharpe_ratio(weights):
        port_return = np.dot(weights, expected_returns_array)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        return -(port_return - risk_free_rate) / (port_volatility + 1e-6)  # Avoid division by zero

    # Optimization
    result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x if result.success else initial_weights

# Main function
def calculate_portfolio_metrics(user_input):
    total_investment = user_input["total_investment"]
    current_allocation = user_input["current_allocation"]

    # Fetch sector returns & covariance matrix
    expected_returns, covariance_matrix = get_sector_data()

    if expected_returns is None or covariance_matrix is None:
        print("Error: Could not retrieve market data.")
        return None

    invested_sectors = {s: a for s, a in current_allocation.items() if a > 0 and s in expected_returns}

    if not invested_sectors:
        print("Error: No valid invested sectors found.")
        return None

    invested_returns = {s: expected_returns[s] for s in invested_sectors.keys()}
    invested_cov_matrix = covariance_matrix.loc[invested_sectors.keys(), invested_sectors.keys()]

    # Calculate current portfolio return & risk
    weights = np.array([current_allocation[sector] / total_investment for sector in invested_sectors])
    portfolio_return = np.dot(weights, list(invested_returns.values()))
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(invested_cov_matrix, weights)))

    # Sharpe Ratio
    risk_free_rate = 0.05
    sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_volatility + 1e-6)

    # Optimized Weights
    optimized_weights = optimize_portfolio(invested_returns, invested_cov_matrix)

    if optimized_weights is None:
        print("Error: Optimization failed.")
        return None

    # Convert weights to allocation
    optimized_allocation = {sector: round(total_investment * weight) for sector, weight in zip(invested_sectors.keys(), optimized_weights)}

    # Calculate optimized portfolio metrics
    optimized_return = np.dot(optimized_weights, list(invested_returns.values()))
    optimized_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(invested_cov_matrix, optimized_weights)))
    optimized_sharpe = (optimized_return - risk_free_rate) / (optimized_volatility + 1e-6)

    result = {
        "Current Portfolio": {
            "Expected Return": round(portfolio_return * 100, 2),
            "Risk": round(portfolio_volatility * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2)
        },
        "Optimized Portfolio": {
            "Allocation": optimized_allocation,
            "Expected Return": round(optimized_return * 100, 2),
            "Risk": round(optimized_volatility * 100, 2),
            "Sharpe Ratio": round(optimized_sharpe, 2)
        }
    }

    print("\nCurrent Portfolio Metrics:", result["Current Portfolio"])
    print("\nOptimized Portfolio Allocation:", result["Optimized Portfolio"])
    return result

# Example User Input
user_input = {
    "risk_tolerance": "Medium",
    "total_investment": 50000,
    "current_allocation": {
        "Auto": 10000, "Bank": 2000, "Energy": 0,
        "Pharma": 15000, "FMCG": 1000, "IT": 12000,
        "Reality": 1000,  "PSU": 0,
        "PSE": 1000, "Infra": 0, "Metal": 7000, "Media": 0
    }
}

# Run Optimization
calculate_portfolio_metrics(user_input)
