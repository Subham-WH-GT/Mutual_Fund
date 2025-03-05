import yfinance as yf
import numpy as np
import pandas as pd

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

def get_sector_metrics():
    expected_returns = {}
    sector_returns = {}  # Stores daily return Series for each sector
    expected_returns2 = {}
    sector_risk = {}
    sentiment_scores = {}

    for sector, ticker in sector_tickers.items():
        try:
            data = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)

            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                print(f"Warning: No price data available for {sector}")
                continue  # Skip this sector if no price data is found

            daily_returns = prices.pct_change().dropna()

            # ✅ Store valid returns only
            if not daily_returns.empty and daily_returns.notnull().all():
                sector_returns[sector] = daily_returns
            else:
                print(f"Warning: No valid return data for {sector}")
                continue

            annual_return = daily_returns.mean() * 252
            expected_returns2[sector] = annual_return

            expected_returns[sector] = round(annual_return, 4)

            annual_volatility = daily_returns.std() * np.sqrt(252)
            sector_risk[sector] = round(annual_volatility, 4)

            if annual_volatility > 0:
                sentiment_scores[sector] = round(max(0.001, (annual_return / annual_volatility) * 10), 3)
            else:
                sentiment_scores[sector] = 1

        except Exception as e:
            print(f"Failed to fetch data for {sector}: {e}")

    # ✅ Only compute covariance matrix if we have multiple valid sectors
    if len(sector_returns) > 1:
        returns_df = pd.DataFrame(sector_returns)
        covariance_matrix = returns_df.cov() * 252
    else:
        covariance_matrix = None
        print("Warning: Not enough valid sectors for covariance matrix.")

    return expected_returns, sector_risk, sentiment_scores, expected_returns2, covariance_matrix

# Example usage
if __name__ == "__main__":
    returns, risk, sentiment_scores, expected_returns2, covariance_matrix = get_sector_metrics()
    print("\nExpected Returns:", returns)
    print("Sector Risk (Volatility):", risk)
    print("Sentiment Scores:", sentiment_scores)
    print("Expected Returns by MVO:", expected_returns2)
    print("Covariance Matrix by MVO:\n", covariance_matrix)
