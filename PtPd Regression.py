import requests

session = requests.Session()
session.verify = False
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})

import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm


tickers = {
    'Platinum': 'PL=F',
    'Palladium': 'PA=F',
    'Gold': 'GC=F',
    'SPY': 'SPY',
    'VIX': '^VIX',
    'Crude_Oil': 'CL=F',
    'Copper': 'HG=F',
    'USD': 'DX-Y.NYB',
    'Lumber': 'LBS=F',
    'Silver': 'SI=F',
    #'US10Y': 'ZN=F',
    'JM':'JMAT.L',
    'Luxury_goods':'LUXE',
    'Rivian':'RIVN',

    # Addition 1
    
    'Bitcoin': 'BTC-USD',
    'IBM': 'IBM',
    'Semiconductors': 'SOXX',
    'JPMorgan': 'JPM',
    'AAPL': 'AAPL',
    'Google': 'GOOGL',
    'Tesla': 'TSLA',
    'AI_Proxy': 'MSFT',
    'Nvidia': 'NVDA',
    'Small_Cap': 'IWM'

}

# Add these to your master dictionary

tickers.update({

    # Tech / Growth
    'Meta': 'META',
    'Amazon': 'AMZN',
    'Netflix': 'NFLX',
    'Adobe': 'ADBE',

    # Semiconductors (important for macro industrial signals)
    'AMD': 'AMD',
    'Intel': 'INTC',
    'TSMC': 'TSM',
    'ASML': 'ASML',

    # Financials
    'Bank_of_America': 'BAC',
    'Goldman_Sachs': 'GS',
    'Morgan_Stanley': 'MS',
    'Wells_Fargo': 'WFC',

    # Industrials
    'Caterpillar': 'CAT',
    'Deere': 'DE',
    'General_Electric': 'GE',

    # Energy
    'Exxon': 'XOM',
    'Chevron': 'CVX',
    'Nat_Gas': 'NG=F',

    # Commodities
    'Aluminum': 'ALI=F',
    #'Nickel': 'NI=F',
    'Zinc': 'ZN=F',

    # FX / Macro
    'EURUSD': 'EURUSD=X',
    'JPY': 'JPY=X',
    'GBPUSD': 'GBPUSD=X',

    # Rates / Bonds
    'US30Y': 'ZB=F',
    'TLT': 'TLT',

    # Indices
    'NASDAQ': '^IXIC',
    'DAX': '^GDAXI',

    # Crypto / Alternatives
    'Ethereum': 'ETH-USD',

    # Volatility / Risk
    'VVIX': '^VVIX'
})



def run_correlations(variables, lags, include_lags=True):

    # 2. Set Time Horizon: Past 15 Years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=15 * 365.25) # Account for leap years

    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    data = yf.download(list(tickers.values()), start=start_date, end=end_date, session=session)['Close']

    # Rename columns from ticker symbols to readable names
    data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)

    # 3. Calculate Returns
    returns = np.log(data / data.shift(1)).dropna()

    # 4. Generate 10-Day Lagged Variables
    # We create lags so that 'Gold_Lag10' represents what the return of Gold was 10 days prior
    df_combined = returns[variables].copy()

    if include_lags:
        for col in variables:
            for lag in lags:
                df_combined[f'{col}_lag{lag}'] = df_combined[col].shift(lag)

    # Drop the first 10 rows which now have NaN values due to the shift
    df_combined = df_combined.dropna()

    # Optional: Reorder columns to group base variables together, and lags together
    base_cols = variables

    if include_lags:
        lag_cols = [f'{col}_lag{lag}' for col in base_cols for lag in lags]
        df_combined = df_combined[base_cols + lag_cols]
    else:
        df_combined = df_combined[base_cols]

    # 5. Calculate Correlation Matrix
    corr_matrix = df_combined.corr()

    # Save the dataset to CSV if you want to inspect the exact numbers for your regressions later
    df_combined.to_csv("returns_and_lags_15yr.csv")

    # 6. Plot the Heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if 'lag' in col1 and 'lag' in col2:
                mask[i, j] = True

    plt.figure(figsize=(18, 14))
    plt.title(f'Correlation Matrix (Lags:{lags}, Past 15 Years)', fontsize=18, fontweight='bold', pad=20)

    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True,          
        fmt=".2f",           
        cmap="coolwarm",     
        center=0,            
        linewidths=0.5,      
        cbar_kws={"label": "Correlation"}
    )

    plt.tight_layout()
    plt.show()


##########

# Correlation Scanner

#########

def scan_correlations(
    ticker_dict,
    variables=None,
    lags=[1, 3, 5],
    threshold=0.3,
    self_threshold=0.6,
    contemp_threshold=0.5,
    pt_pd_only=False,
    start_date=None,
    end_date=None, 
    lookback_years=15
):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta

    # =========================
    # 1. Select variables
    # =========================
    if variables is None:
        variables = list(ticker_dict.keys())

    # =========================
    # 2. Download data
    # =========================
    if end_date is None:
        end_date = datetime.today()

    if start_date is None:
        start_date = end_date - timedelta(days=lookback_years * 365.25)

    data = yf.download(list(ticker_dict.values()),
                       start=start_date,
                       end=end_date)['Close']

    data.rename(columns={v: k for k, v in ticker_dict.items()}, inplace=True)

    # =========================
    # 3. Log returns
    # =========================
    returns = np.log(data / data.shift(1)).dropna()

    df = returns[variables].copy()

    # =========================
    # 4. Create lags
    # =========================
    for var in variables:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

    df = df.dropna()

    # =========================
    # 5. Correlation matrix
    # =========================
    corr = df.corr()

    # =========================
    # 6. Convert to list + filter
    # =========================
    results = []

    cols = corr.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):  # avoids duplicates
            var1 = cols[i]
            var2 = cols[j]
            value = corr.iloc[i, j]

            # -------------------------
            # Skip lag–lag relationships
            # -------------------------
            if 'lag' in var1 and 'lag' in var2:
                continue

            # -------------------------
            # Pt/Pd filter
            # -------------------------
            if pt_pd_only:
                if not (
                    'Platinum' in var1 
                    #or 'Platinum' in var2 
                    or 'Palladium' in var1 
                    #or 'Palladium' in var2
                ):
                    continue

            # -------------------------
            # Determine relationship type
            # -------------------------
            same_base = var1.split('_')[0] == var2.split('_')[0]
            is_contemp = ('lag' not in var1) and ('lag' not in var2)

            # -------------------------
            # Apply thresholds
            # -------------------------
            abs_val = abs(value)

            if same_base:
                if abs_val < self_threshold:
                    continue
            elif is_contemp:
                if abs_val < contemp_threshold:
                    continue
            else:
                if abs_val < threshold:
                    continue

            results.append((var1, var2, value))

    # =========================
    # 7. Sort results
    # =========================
    results.sort(key=lambda x: abs(x[2]), reverse=True)

    # =========================
    # 8. Print results
    # =========================
    print("\n===== TOP CORRELATIONS =====")
    for r in results:
        print(f"{r[0]}  vs  {r[1]}  →  {round(r[2], 3)}")

    return results




# Regression



def run_regression(
    target,
    variables,
    lags=[1, 5, 10],
    start_date=None,
    end_date=None,
    lookback_years=15
):
    """
    Run flexible regression on financial time series.

    Parameters:
    -----------
    target : str
        Dependent variable (e.g. 'Platinum')

    variables : list
        List of independent variables (e.g. ['Gold', 'SPY'])

    lags : list
        Lags to include (e.g. [1, 3, 5, 10])

    start_date, end_date : datetime
        Optional explicit date range

    lookback_years : int
        Used if start_date not provided

    Returns:
    --------
    model : statsmodels result object
    df_combined : dataframe used
    """

    # =========================
    # 2. Handle dates
    # =========================
    if end_date is None:
        end_date = datetime.today()

    if start_date is None:
        start_date = end_date - timedelta(days=lookback_years * 365.25)

    # =========================
    # 3. Download data
    # =========================
    data = yf.download(list(tickers.values()),
                       start=start_date,
                       end=end_date)['Close']

    data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)

    # =========================
    # 4. Log returns
    # =========================
    returns = np.log(data / data.shift(1)).dropna()

    # =========================
    # 5. Build dataset
    # =========================
    df = returns[[target] + variables].copy()

    # Add lagged features
    for var in [target] + variables:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

    df = df.dropna()

    # =========================
    # 6. Define regression
    # =========================
    y = df[target]

    # Drop contemporaneous target (avoid leakage)
    X = df.drop(columns=[target])

    # Add intercept
    X = sm.add_constant(X)

    # =========================
    # 7. Fit model (robust SE)
    # =========================
    model = sm.OLS(y, X).fit(cov_type='HC3')

    # =========================
    # 8. Output
    # =========================
    print(f"\n===== Regression: {target} =====")
    print(f"Variables: {variables}")
    print(f"Lags: {lags}")
    print(f"Period: {start_date.date()} → {end_date.date()}")
    print(model.summary())

    return model, df



def build_regression_spec(
    target,
    variables,
    lags=[1, 5, 10],
    start_date=None,
    end_date=None,
    lookback_years=15
):
    spec = {
        'target': target,
        'variables': variables,
        'lags': lags,
        'start_date': start_date,
        'end_date': end_date,
        'lookback_years': lookback_years
    }
    return spec



def execute_regression(spec, include_contemp=True):
    
    # unpack spec
    target = spec['target']
    variables = spec['variables']
    lags = spec['lags']
    start_date = spec.get('start_date')
    end_date = spec.get('end_date')
    lookback_years = spec.get('lookback_years', 15)


    # handle dates
    if end_date is None:
        end_date = datetime.today()

    if start_date is None:
        start_date = end_date - timedelta(days=lookback_years * 365.25)

    # download
    
    relevant_names = list(set([target] + variables))
    relevant_tickers = [tickers[name] for name in relevant_names]
    
    data = yf.download(relevant_tickers,
                       start=start_date,
                       end=end_date)['Close']

    data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)

    # returns
    returns = np.log(data / data.shift(1)).dropna()

    # dataset
    if include_contemp:
        df = returns[[target] + variables].copy()
    else:
        df = returns[[target]].copy()

    for var in variables:
        for lag in lags:
            df[f'{var}_lag{lag}'] = returns[var].shift(lag)

    df = df.dropna()

    y = df[target]
    X = df.drop(columns=[target])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type='HC3')

    return model, df



def run_batch(regressions, include_contemp=True):
    results = []

    for spec in regressions:
        model, df = execute_regression(spec, include_contemp=include_contemp)

        print("\n==============================")
        print(f"Target: {spec['target']}")
        print(f"Variables: {spec['variables']}")
        print(f"Lags: {spec['lags']}")
        print(model.summary())

        results.append(model)

    return results

def plot_correlations(
    pairs,
    ticker_dict=tickers,
    lag=0,
    start_date=None,
    end_date=None,
    lookback_years=15
):
   

    # =========================
    # 1. Handle dates
    # =========================
    if end_date is None:
        end_date = datetime.today()

    if start_date is None:
        start_date = end_date - timedelta(days=lookback_years * 365.25)

    # =========================
    # 2. Download data
    # =========================
    data = yf.download(list(ticker_dict.values()),
                       start=start_date,
                       end=end_date)['Close']

    data.rename(columns={v: k for k, v in ticker_dict.items()}, inplace=True)

    # =========================
    # 3. Log returns
    # =========================
    returns = np.log(data / data.shift(1)).dropna()

    # =========================
    # 4. Create plots
    # =========================
    n = len(pairs)
    cols = 2
    rows = (n + 1) // cols

    plt.figure(figsize=(12, 5 * rows))

    for i, (x_var, y_var) in enumerate(pairs, 1):

        plt.subplot(rows, cols, i)

        if lag > 0:
            x = returns[x_var].shift(lag)
            y = returns[y_var]
            label = f"{x_var}_lag{lag} vs {y_var}"
        else:
            x = returns[x_var]
            y = returns[y_var]
            label = f"{x_var} vs {y_var}"

        df = pd.concat([x, y], axis=1).dropna()

        x = df[x_var]
        y = df[y_var]

        # Scatter
        plt.scatter(x, y, alpha=0.5)

        # Regression line
        coef = np.polyfit(x, y, 1)
        poly = np.poly1d(coef)
        plt.plot(x, poly(x))

        # Correlation
        corr = np.corrcoef(x, y)[0, 1]

        plt.title(f"{label}\nCorr: {corr:.3f}")
        plt.xlabel(x_var)
        plt.ylabel(y_var)

    plt.tight_layout()
    plt.show()




def get_fred_data(series_dict, api_key, start_date=None, end_date=None):
    """
    Pull multiple FRED series into one dataframe.

    Parameters
    ----------
    series_dict : dict
        Dictionary where keys are your desired column names
        and values are FRED series IDs.

        Example:
        {
            'US_Auto_Sales': 'TOTALSA',
            'Industrial_Production': 'INDPRO'
        }

    api_key : str
        Your FRED API key.

    start_date : str, optional
        Format: 'YYYY-MM-DD'

    end_date : str, optional
        Format: 'YYYY-MM-DD'

    Returns
    -------
    df_final : pandas DataFrame
        DataFrame indexed by Date with one column per FRED series.
    """

    all_series = []

    for name, series_id in series_dict.items():

        url = "https://api.stlouisfed.org/fred/series/observations"

        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json"
        }

        if start_date is not None:
            params["observation_start"] = start_date

        if end_date is not None:
            params["observation_end"] = end_date

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()["observations"]

        df = pd.DataFrame(data)

        df = df[["date", "value"]].copy()
        df.rename(columns={"date": "Date", "value": name}, inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df[name] = pd.to_numeric(df[name], errors="coerce")

        all_series.append(df)

    # Merge all series on Date
    df_final = all_series[0]

    for df in all_series[1:]:
        df_final = pd.merge(df_final, df, on="Date", how="outer")

    df_final = df_final.sort_values("Date").reset_index(drop=True)

    # =========================
    # Transform series
    # =========================

    fed_funds_col = "Fed_Funds_Rate"
    manufacturer_confidence_col = "Manufacturer_Confidence"

    for col in df_final.columns:
        if col == "Date":
            continue

    # Fed funds: raw change, not percent change
        if col == fed_funds_col:
            df_final[col] = df_final[col].diff()

    # Manufacturer confidence: already percent change
        elif col == manufacturer_confidence_col:
            df_final[col] = df_final[col] / 100

        # Everything else: percent change as decimal
        else:
            df_final[col] = df_final[col].pct_change()

    # Drop rows created by pct_change/diff
    df_final = df_final.dropna().reset_index(drop=True)

    return df_final

def execute_monthly_indicator_regression(
    target,
    market_variables,
    fred_series,
    ticker_dict,
    fred_api_key,
    lags=[1],
    start_date=None,
    end_date=None,
    include_contemporaneous=True
):
    
    # =========================
    # 1. Handle dates
    # =========================
    if end_date is None:
        end_date = datetime.today()

    if start_date is None:
        start_date = end_date - timedelta(days=15 * 365.25)

    # =========================
    # 2. Download relevant market data only
    # =========================
    market_names = list(dict.fromkeys([target] + market_variables))
    market_tickers = [ticker_dict[name] for name in market_names]

    market_data = yf.download(
        market_tickers,
        start=start_date,
        end=end_date,
        progress=False
    )["Close"]

    # Handle case where only one ticker is downloaded
    if isinstance(market_data, pd.Series):
        market_data = market_data.to_frame(name=market_tickers[0])

    market_data.rename(
        columns={v: k for k, v in ticker_dict.items()},
        inplace=True
    )

    # =========================
    # 3. Convert market data to monthly log returns
    # =========================
    monthly_prices = market_data.resample("ME").last()
    monthly_market_returns = np.log(monthly_prices / monthly_prices.shift(1))

    monthly_market_returns = monthly_market_returns.reset_index()
    monthly_market_returns["Month"] = monthly_market_returns["Date"].dt.to_period("M")

    # =========================
    # 4. Pull FRED data
    # =========================
    fred_df = get_fred_data(
        series_dict=fred_series,
        api_key=fred_api_key,
        start_date=start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else start_date,
        end_date=end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else end_date
    )

    fred_df["Date"] = pd.to_datetime(fred_df["Date"])
    fred_df["Month"] = fred_df["Date"].dt.to_period("M")

    # Drop Date so we don't duplicate it in merge
    fred_df = fred_df.drop(columns=["Date"])

    # =========================
    # 5. Merge market + FRED data by month
    # =========================
    df = pd.merge(
        monthly_market_returns,
        fred_df,
        on="Month",
        how="inner"
    )

    df = df.drop(columns=["Date"])
    df["Date"] = df["Month"].dt.to_timestamp("M")
    df = df.drop(columns=["Month"])

    # Put Date first
    cols = ["Date"] + [col for col in df.columns if col != "Date"]
    df = df[cols]

    # =========================
    # 6. Build regression dataset
    # =========================
    predictors = market_variables + list(fred_series.keys())

    if include_contemporaneous:
        df_model = df[["Date", target] + predictors].copy()
    else:
        df_model = df[["Date", target]].copy()

    # Add lags of all predictors
    for var in predictors:
        for lag in lags:
            df_model[f"{var}_lag{lag}"] = df[var].shift(lag)

    df_model = df_model.dropna().reset_index(drop=True)

    # =========================
    # 7. Run regression
    # =========================
    y = df_model[target]
    X = df_model.drop(columns=["Date", target])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    # =========================
    # 8. Print summary
    # =========================
    print("\n===== MONTHLY INDICATOR REGRESSION =====")
    print(f"Target: {target}")
    print(f"Market variables: {market_variables}")
    print(f"FRED variables: {list(fred_series.keys())}")
    print(f"Lags: {lags}")
    print(f"Include contemporaneous variables: {include_contemporaneous}")
    print(f"Observations: {len(df_model)}")
    print("Root MSE:", np.sqrt(model.mse_resid))
    print(model.summary())

    return model, df_model
    #return df

regressions = [
    {
        'target': 'Rivian',
        'variables': ['Tesla'],
        'lags': [5], 'lookback_years': 5
    },
    {
        'target': 'Palladium',
        'variables': ['JM'],
        'lags': [1], 'lookback_years': 5
    },
    {
        'target': 'Palladium',
        'variables': ['Tesla', 'AAPL', 'Nvidia', 'AMD', 'NASDAQ', 'Zinc', 'Semiconductors'
                      ,'Morgan_Stanley'],
        'lags': [3], 'lookback_years': 5
    },
    {
        'target': 'Platinum',
        'variables': ['Crude_Oil', 'Bitcoin'],
        'lags': [3], 'lookback_years': 5
    }
]


'''
Functions you have:
1. run correlations 2. scan correlations 3. run regression / exec 4. run batch
5. run plots 6. get fred data 7. monthly regressions
'''

tix = ['Platinum', 'Palladium', 'Gold', 'SPY', 'VIX', 'Copper', 'Lumber',
         'Crude_Oil', 'Silver', 'USD']
lags1 = [1, 5, 10]




'''scan_correlations(
    ticker_dict=tickers,
    lags=[3],
    threshold= 0.1, self_threshold=0.3, contemp_threshold=0.5, lookback_years=5,
    pt_pd_only=True
)
'''


'''run_correlations(variables=[tix[i] for i in [0,1]] + ['JM'], lags=[1,3,20,50, 200], 
                 include_lags=True)'''

'''
model, df =execute_regression(spec=regressions[0], include_contemp=True)
print(model.summary())
'''

'''
run_batch([regressions[i] for i in [0, 2,3]], include_contemp=False)
'''

pairs1 = [('Platinum', 'Gold'), ('Platinum', 'Silver'), ('Platinum', 'Copper'),
         ('Platinum', 'Crude_Oil') ]

pairs2 = [('Palladium', 'Gold'), ('Palladium', 'Silver'), ('Palladium', 'Copper'),
         ('Palladium', 'Crude_Oil')]

pairscrude = [('Crude_Oil', 'Platinum'), ('Crude_Oil', 'Palladium')]

pairstesla = [('Tesla', 'Platinum'), ('Tesla', 'Palladium')]

'''
plot_correlations(pairs=pairstesla, lookback_years=5, lag=3)
'''

fred_series = {
    "US_Auto_Sales": "TOTALSA",
    "Industrial_Production": "INDPRO",
    "Unemployment_Rate": "UNRATE",
    "CPI": "CPIAUCSL",
    'US_Auto_Production': 'DAUPSA',
    'Fed_Funds_Rate': 'FEDFUNDS',
    'Home_Prices': 'CSUSHPINSA',
    'Manufacturer_Confidence': 'BSCICP02USM460S',
    'Jewelry_store_sales':'MRTSSM44831USS',

    # Additions
    'Military_Spending': 'USGOVFEDMILRGSP',
    'PPI_Mining':'PCU333131333131',
    'Global_metal':'PMETAINDEXM'

}

fred_series_2 = {
    'Military_aircraft_parts': 'WPU142501033',
    'PPI_Mining':'PCU333131333131',
    'Global_metal':'PMETAINDEXM'

}


fred_api_key = "3f624180fb969eaad9508ce267505bd5"

fred_df = get_fred_data(
    series_dict=fred_series_2,
    api_key=fred_api_key,
    start_date="2010-01-01",
    end_date="2025-12-31"
)


'''model, df_monthly = execute_monthly_indicator_regression(
    target="Platinum",
    market_variables=['Gold', 'SPY', 'Crude_Oil'],
    fred_series=fred_series_2,
    ticker_dict=tickers,
    fred_api_key=fred_api_key,
    lags=[1],
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2025, 12, 31),
    include_contemporaneous=False
)'''


'''model, df_monthly = execute_monthly_indicator_regression(
    target="Platinum",
    market_variables=[],
    fred_series={"US_Auto_Sales": "TOTALSA"},
    ticker_dict=tickers,
    fred_api_key=fred_api_key,
    lags=[1],
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2025, 12, 31),
    include_contemporaneous=False
)'''



# Functions to Demo

'''scan_correlations(
    ticker_dict=tickers,
    lags=[3],
    threshold= 0.1, self_threshold=0.3, contemp_threshold=0.5, lookback_years=5,
    pt_pd_only=True
)'''

run_correlations(variables=tix, lags=[3], 
                 include_lags=False)


'''plot_correlations(pairs=pairstesla, lookback_years=5, lag=3)
'''
'''
run_batch([regressions[i] for i in [1,3]], include_contemp=False)
'''
'''
model, df_monthly = execute_monthly_indicator_regression(
    target="Platinum",
    market_variables=['Gold'],
    fred_series=fred_series_2,
    ticker_dict=tickers,
    fred_api_key=fred_api_key,
    lags=[1],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2025, 12, 31),
    include_contemporaneous=False
)'''