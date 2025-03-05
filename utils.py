# import yfinance as yf
# import numpy as np
# import pandas as pd

# # Mapping of sectors to stock symbols (NSE sectoral indices)
# sector_tickers = {
#     "Auto": "^CNXAUTO",      # Nifty IT Index
#     "Bank": "^NSEBANK",
#     "Energy": "^CNXENERGY",
#     "Pharma":"SUNPHARMA.NS",
#     "FMCG":"^CNXFMCG",
#     "IT":"^CNXIT",
#     "Reality":"^CNXREALTY",
#     "Finance":"BAJFINANCE.NS",
#     "PSU":"^CNXPSUBANK",
#     "PSE":"^CNXPSE",
#     "Infra":"^CNXINFRA",
#     "Metal":"^CNXMETAL",
#     "Media":"^CNXMEDIA"
# }

# # Function to fetch historical returns and risk for each sector
# def get_sector_metrics():
#     expected_returns = {}
#     sector_risk = {}
#     sentiment_scores = {}

#     for sector, ticker in sector_tickers.items():
#         try:
#             # Download 1 year of historical data with auto_adjust=False
#             data = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)

#             # Use 'Close' if 'Adj Close' is unavailable
#             if 'Adj Close' in data.columns:
#                 prices = data['Adj Close']
#             elif 'Close' in data.columns:
#                 prices = data['Close']
#             else:
#                 raise ValueError(f"No 'Close' or 'Adj Close' data available for {sector}")

#             # Calculate daily returns
#             daily_returns = prices.pct_change().dropna()

#             # Expected return (annualized)
#             annual_return = daily_returns.mean() * 252
            
#             expected_returns[sector] = round(annual_return.iloc[0] if isinstance(annual_return, pd.Series) else annual_return, 4)

#             # Risk (annualized volatility)
#             annual_volatility = daily_returns.std() * np.sqrt(252)

#             print(f"{sector}: Annual Return = {annual_return}, Volatility = {annual_volatility}")


#             sector_risk[sector] = round(annual_volatility.iloc[0] if isinstance(annual_volatility, pd.Series) else annual_volatility, 4)
#             score = (annual_return / annual_volatility) if annual_volatility > 0 else 0
#             sentiment_scores[sector] = max(1, round(score * 10))

#         except Exception as e:
#             print(f"Failed to fetch data for {sector}: {e}")

#     return expected_returns, sector_risk, sentiment_scores

# # Example of using this function
# if __name__ == "__main__":
#     returns, risk, sentiment_scores = get_sector_metrics()
#     print("Expected Returns:", returns)
#     print("Sector Risk (Volatility):", risk)
#     print("Sentiment:",sentiment_scores)


import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Mapping of sectors to stock symbols (NSE sectoral indices)
sector_tickers = {
    "Auto": "^CNXAUTO",      # Nifty IT Index
    "Bank": "^NSEBANK",
    "Energy": "^CNXENERGY",
    "Pharma": "SUNPHARMA.NS",
    "FMCG": "^CNXFMCG",
    "IT": "^CNXIT",
    "Reality": "^CNXREALTY",
    # "Finance": "BAJFINANCE.NS",
    "PSU": "^CNXPSUBANK",
    "PSE": "^CNXPSE",
    "Infra": "^CNXINFRA",
    "Metal": "^CNXMETAL",
    "Media": "^CNXMEDIA"
}


def get_sector_metrics():
    expected_returns = {}
    sector_returns={}
    expected_returns2={}
    sector_risk = {}
    sentiment_scores = {}
    # covariance_matrix=None
    covariance_matrix = None  
    for sector, ticker in sector_tickers.items():
        try:
           
            data = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)

            # Use 'Close' if 'Adj Close' is unavailable
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                raise ValueError(f"No 'Close' or 'Adj Close' data available for {sector}")

            # Calculate daily returns
            daily_returns = prices.pct_change().dropna()

            # Expected return (annualized)
            expected_returns2[sector]=daily_returns.mean()*252
            annual_return = daily_returns.mean() * 252
            sector_returns[sector] = daily_returns
            # returns_df = pd.DataFrame(sector_returns)
            # covariance_matrix = returns_df.cov() * 252
            # returns_df = pd.concat(sector_returns, axis=1)
            # returns_df.dropna(inplace=True) 
            # covariance_matrix = returns_df.cov() * 252
            annual_return_value = annual_return.iloc[0] if isinstance(annual_return, pd.Series) else annual_return
            expected_returns[sector] = round(annual_return_value, 4)

            # Risk (annualized volatility)
            annual_volatility = daily_returns.std() * np.sqrt(252)
            annual_volatility_value = annual_volatility.iloc[0] if isinstance(annual_volatility, pd.Series) else annual_volatility
            sector_risk[sector] = round(annual_volatility_value, 4)

            print(f"{sector}: Annual Return = {annual_return_value}, Volatility = {annual_volatility_value}")

            # Sentiment Score Calculation
            if annual_volatility_value > 0:
                score = annual_return_value / annual_volatility_value
                
                # sentiment_scores[sector] = max(1, round(score * 10))
                sentiment_scores[sector] = round(max(0.001, score * 10), 3)  # Keep float, with a small floor to avoid zero allocation

                # if(sentiment_scores[sector]>3):
                #     sentiment_scores[sector]=3
            else:
                sentiment_scores[sector] = 1  # Assigning minimum sentiment score

        except Exception as e:
            print(f"Failed to fetch data for {sector}: {e}")
    returns_df = pd.concat(sector_returns, axis=1)
    # print("hi",returns_df)
    returns_df.dropna(inplace=True) 
    covariance_matrix = returns_df.cov() * 252
    return expected_returns, sector_risk, sentiment_scores, expected_returns2,covariance_matrix 

# Example of using this function
if __name__ == "__main__":
    returns, risk, sentiment_scores,expected_returns2,covariance_matrix  = get_sector_metrics()
    print("Expected Returns:", returns)
    print("Sector Risk (Volatility):", risk)
    print("Sentiment Scores:", sentiment_scores)
    print("expected by MVO:",expected_returns2)
    print("covariance by MVO:",covariance_matrix)

