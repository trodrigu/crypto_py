from coinbase.rest import RESTClient
from datetime import datetime, timedelta
import numpy as np
import uuid
import pandas as pd
import time
import requests
import argparse
import talib
import sqlite3

# Initialize the client with your Coinbase API credentials
client = RESTClient()

OVERLAP_STUDIES = ['BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MAMA', 'MAVP', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA']
MOMENTUM_INDICATORS = ['ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX', 'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 'ULTOSC', 'WILLR']
VOLUME_INDICATORS = ['AD', 'ADOSC', 'OBV']
CYCLE_INDICATORS = ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE']
VOLATILITY_INDICATORS = ['ATR', 'NATR', 'TRANGE']
PATTERN_RECOGNITION = ['CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS']
STATISTIC_FUNCTIONS = ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR']

def fetch_data(api_client, product_id, start, end, granularity="FIFTEEN_MINUTE"):
    while True:
        try:
            print(f"Fetching data for {product_id} from {start} to {end} with granularity {granularity}")
            response = api_client.get_candles(
                product_id=product_id,
                start=start,
                end=end,
                granularity=granularity
            )
            return response['candles']
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
                time.sleep(60)
            else:
                raise e

def process_data(data, product_id):
    df = pd.DataFrame(data, columns=["start", "low", "high", "open", "close", "volume"])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='s')
    df['low'] = df['low'].astype(float)
    df['high'] = df['high'].astype(float)
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['price_change'] = df['close'] - df['open']
    df['volatility'] = df['high'] - df['low']
    df = calculate_indicators_for_base(df, product_id)
    return df

def analyze_intervals(df):
    grouped = df.groupby([df['start'].dt.time]).agg({
        'volatility': 'mean',
        'price_change': 'sum'
    }).reset_index()
    return grouped

def find_cyclical_patterns(product_id, start_date, end_date, granularity="FIFTEEN_MINUTE"):
    conn = sqlite3.connect('crypto_data.db')
    cursor = conn.cursor()

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    query = '''
        SELECT * FROM candles
        WHERE start >= ? AND start <= ?
    '''
    cursor.execute(query, (start_timestamp, end_timestamp))
    all_data = cursor.fetchall()
    conn.close()

    df = process_data(all_data, product_id)
    analyzed_df = analyze_intervals(df)
    
    # Find the intervals with the least volatility on average
    best_intervals = analyzed_df.sort_values(by='volatility').head(10)
    print("Best Cyclical Intervals for 100x Leverage Trading:")
    print(best_intervals)
    # Monitor significant volume changes and generate signals
    signals = monitor_significant_volume_changes(df)
    for signal in signals:
        print(f"Date: {signal[0]}, Signal: {signal[1]}")

    return best_intervals

def signal_buy_opportunity(product_id, days_back=7, granularity="ONE_DAY"):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())
    
    try:
        candles = client.get_candles(
            product_id=product_id, 
            start=start_time_unix,
            end=end_time_unix, 
            granularity=granularity
        )
    except Exception as e:
        print(f"Error fetching historical data for {product_id}: {e}")
        return None

    close_prices = [float(candle['close']) for candle in candles['candles']]
    volumes = [float(candle['volume']) for candle in candles['candles']]
    timestamps = [datetime.utcfromtimestamp(int(candle['start'])) for candle in candles['candles']]

    if len(volumes) > 1:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change = 0

    try:
        A_INDEX = np.argmax(close_prices)
        C_INDEX = len(close_prices) - 1
        B_INDEX = np.argmin(close_prices[A_INDEX:C_INDEX]) + A_INDEX
        
        A_point = (timestamps[A_INDEX], close_prices[A_INDEX], volumes[A_INDEX])
        B_point = (timestamps[B_INDEX], close_prices[B_INDEX], volumes[B_INDEX])
        C_point = (timestamps[C_INDEX], close_prices[C_INDEX], volumes[C_INDEX])

        fib_retracement_levels = [0.382, 0.5, 0.618]
        potential_buy_signal = False
        detailed_reason = ""
        price_recovery_ratio = (C_point[1] - B_point[1]) / (A_point[1] - B_point[1])
        
        with_scores = []
        for level in fib_retracement_levels:
            fib_retracement = A_point[1] - (A_point[1] - B_point[1]) * level
            if close_prices[-1] > fib_retracement and A_point[1] > B_point[1] and C_point[2] > B_point[2] and volumes[-1] > volumes[B_INDEX]:
                potential_buy_signal = True
                detailed_reason = f"Price recovered beyond the {level*100}% Fibonacci level with increased volume."
                with_score = {"product_id": product_id, "score": calculate_score(volume_change, level, price_recovery_ratio), "details": detailed_reason}
                with_scores.append(with_score)
        return with_scores
            
    except Exception as e:
        return []

def calculate_score(volume_change, fib_level, price_recovery_ratio):
    score = 0
    if volume_change > 0.5: score += 3
    elif volume_change > 0.2: score += 2
    else: score += 1

    if fib_level == 0.5: score += 3
    elif fib_level == 0.382: score += 2
    elif fib_level == 0.618: score += 1

    if price_recovery_ratio > 0.8: score += 1
    elif price_recovery_ratio > 0.6: score += 2
    else: score += 3

    return score

def get_best_bid_ask(product_id):
    try:
        order_book = client.get_product_book(product_id, level=1)
    except Exception as e:
        print(f"Error fetching order book for {product_id}: {e}")
        return None, None

    best_bid = order_book['pricebook']['bids'][0]['price']
    best_ask = order_book['pricebook']['asks'][0]['price']
    return float(best_bid), float(best_ask)

def fetch_all_product_ids():
    products = client.get_products()['products']
    return [product['product_id'] for product in products]

def fetch_filtered_products():
    all_products = client.get_products()
    filtered_products = [product for product in all_products['products'] if product['product_id'].endswith('-USD') or product['product_id'].endswith('-USDT')]
    return filtered_products

def compare_prices_for_arbitrage(product_ids):
    prices = {product_id: get_best_bid_ask(product_id) for product_id in product_ids}
    
    max_bid_product = max(prices.items(), key=lambda x: x[1][0])
    min_ask_product = min(prices.items(), key=lambda x: x[1][1])
    
    buy_price = min_ask_product[1][1]
    sell_price = max_bid_product[1][0]
    buy_size = 100 / buy_price
    sell_size = buy_size
    total_buy_cost = buy_price * buy_size
    total_sell_revenue = sell_price * sell_size

    buy_fee = total_buy_cost * 0.008
    sell_fee = total_sell_revenue * 0.008

    net_profit = total_sell_revenue - total_buy_cost - buy_fee - sell_fee

    if net_profit > 0:
        buy_product_id = min_ask_product[0]
        sell_product_id = max_bid_product[0]

        print(f"Potential arbitrage opportunity: Buy from {min_ask_product[0]} at {min_ask_product[1][1]} and sell on {max_bid_product[0]} at {max_bid_product[1][0]}")

        buy_order_response = place_market_order(buy_product_id, 'buy', buy_price, buy_size)
        sell_order_response = place_market_order_sell(sell_product_id, 'sell', sell_price, sell_size, buy_size, buy_price)

def get_unique_first_part_of_product_ids(product_ids):
    return list(set([product_id.split('-')[0] for product_id in product_ids]))

def place_market_order(product_id, side, price, size):
    client_order_id = str(uuid.uuid4())
    if side == 'buy':
        order_response = client.market_order_buy(client_order_id=client_order_id, product_id=product_id, quote_size=str(size))
    elif side == 'sell':
        order_response = client.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=str(size))
    return order_response

def place_market_order_sell(product_id, side, price, size, buy_size, buy_price):
    client_order_id = str(uuid.uuid4())
    fee_percentage = 0.008

    total_buy_cost = buy_price * buy_size
    buy_fee = total_buy_cost * fee_percentage

    actual_currency_purchased = (total_buy_cost - buy_fee) / buy_price

    sell_size = actual_currency_purchased
    order_response = client.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=sell_size)
    return order_response

def get_product_ids_which_match_first_part(product_ids, first_part):
    return [product_id for product_id in product_ids if product_id.startswith(first_part)]

def group_by_product_ids(product_ids):
    first_parts = get_unique_first_part_of_product_ids(product_ids)
    return {first_part: get_product_ids_which_match_first_part(product_ids, first_part) for first_part in first_parts}

def discover_abc_pattern_with_volume_fibonacci(product_id, days_back=1, granularity="ONE_DAY"):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    candles = get_candle_data_filtered_by_date("crypto_data.db", product_id, start_time, end_time)
    candles = process_data(candles, product_id)

    close_prices = candles['close']
    volumes = candles['volume']
    timestamps = candles['start']

    try:
        A_INDEX = np.argmax(close_prices)
        C_INDEX = len(close_prices) - 1
        B_INDEX = np.argmin(close_prices[A_INDEX:C_INDEX]) + A_INDEX
        
        A_point = (timestamps[A_INDEX], close_prices[A_INDEX], volumes[A_INDEX])
        B_point = (timestamps[B_INDEX], close_prices[B_INDEX], volumes[B_INDEX])
        C_point = (timestamps[C_INDEX], close_prices[C_INDEX], volumes[C_INDEX])

        fib_retracement = A_point[1] - (A_point[1] - B_point[1]) * 0.382

        if A_point[1] < C_point[1] and A_point[1] > B_point[1] and C_point[2] > B_point[2] and C_point[1] > fib_retracement:
            print(f"Potential ABC pattern detected for {product_id}:")
            print(f"A (Peak): {A_point}, B (Correction): {B_point}, C (Final Rise): {C_point}")
            print(f"Fibonacci Retracement Level (38.2% from A to B): {fib_retracement}")
            return A_point, B_point, C_point, fib_retracement
        else:
            return None
    except Exception as e:
        return None

def get_price_changes_for_interval(api_client, product_id, start_date, end_date, interval_time, granularity="FIFTEEN_MINUTE"):
    current_date = start_date
    all_data = []
    request_count = 0
    
    while current_date < end_date:
        if request_count >= 10000:
            print("Hourly request limit reached. Waiting for an hour before continuing...")
            time.sleep(3600)  # Wait for an hour
            request_count = 0
        
        next_date = current_date + timedelta(days=1)
        start_timestamp = int(current_date.timestamp())
        end_timestamp = int(next_date.timestamp())
        
        data = fetch_data(api_client, product_id, start_timestamp, end_timestamp, granularity)
        all_data.extend(data)
        request_count += 1
        
        current_date = next_date
        time.sleep(0.2)  # Sleep for 200ms to ensure we stay within 5 requests per second
    
    df = process_data(all_data, product_id)
    df['interval_time'] = df['start'].dt.time
    
    # Filter data for the specified interval time
    interval_data = df[df['interval_time'] == interval_time]
    price_changes = interval_data[['start', 'price_change']]
    
    return price_changes

def get_price_change_last_24_hours(api_client, product_id, interval_time, granularity="FIFTEEN_MINUTE"):
    if isinstance(interval_time, str):
        try:
            end_time = datetime.strptime(interval_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Invalid datetime format for interval_time. Please use YYYY-MM-DD HH:MM:SS format.")
            return None
    elif isinstance(interval_time, datetime):
        end_time = interval_time
    elif isinstance(interval_time, pd.Timestamp):
        end_time = interval_time
    else:
        print("Unsupported type for interval_time. Please provide a string or Timestamp.")
        return None

    start_time = end_time - timedelta(days=1)
    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())

    data = fetch_data(api_client, product_id, start_time_unix, end_time_unix, granularity)
    if data:
        open_price = float(data[-1]['open'])
        close_price = float(data[0]['close'])
        return ((close_price - open_price) / open_price) * 100
    return 0

def get_volume_change_last_24_hours(api_client, product_id, interval_time, granularity="FIFTEEN_MINUTE"):
    if isinstance(interval_time, str):
        try:
            end_time = datetime.strptime(interval_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Invalid datetime format for interval_time. Please use YYYY-MM-DD HH:MM:SS format.")
            return None
    elif isinstance(interval_time, datetime):
        end_time = interval_time
    elif isinstance(interval_time, pd.Timestamp):
        end_time = interval_time
    else:
        print("Unsupported type for interval_time. Please provide a string or Timestamp.")
        return None

    start_time = end_time - timedelta(days=1)
    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())

    data = fetch_data(api_client, product_id, start_time_unix, end_time_unix, granularity)
    if data:
        volume_change = (float(data[0]['volume']) - float(data[-1]['volume'])) / float(data[-1]['volume'])
        return volume_change
    return 0

def get_price_change_since_beginning_of_day(api_client, product_id, interval_time, granularity="ONE_HOUR"):
    try:
        end_time = datetime.strptime(interval_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("Invalid datetime format for interval_time. Please use YYYY-MM-DD HH:MM:SS format.")
        return None

    start_of_day = datetime(end_time.year, end_time.month, end_time.day)  # Midnight of the specified day

    start_time_unix = int(start_of_day.timestamp())
    end_time_unix = int(end_time.timestamp())

    data = fetch_data(api_client, product_id, start_time_unix, end_time_unix, granularity)
    if data:
        open_price = float(data[0]['open'])
        close_price = float(data[-1]['close'])
        return ((close_price - open_price) / open_price) * 100
    return 0

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = pd.Series(data).ewm(span=fast, adjust=False).mean()
    exp2 = pd.Series(data).ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_rsi(data, period=14):
    delta = np.diff(data)
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta

    avg_gain = np.average(gain[:period])
    avg_loss = np.average(loss[:period])

    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def monitor_significant_volume_changes(df, volume_threshold=0.05):
    significant_volume_changes = df[df['Volume_Change'].pct_change() > volume_threshold]
    signals = []
    for index, row in significant_volume_changes.iterrows():
        if row['Volume_Change'] > 0 and row['RSI'] > 50 and row['MACD'] > row['MACD_signal']:
            signals.append((row['start'], 'Consider long position'))
        elif row['Volume_Change'] < 0 and row['RSI'] < 50 and row['MACD'] < row['MACD_signal']:
            signals.append((row['start'], 'Consider short position'))
    return signals

def calculate_correlation_interval_with_volume_twenty_four(client, product_id, start_date, end_date, interval_time, granularity="FIFTEEN_MINUTE"):
    data = get_price_changes_for_interval_from_db('crypto_data.db', product_id, start_date, end_date, interval_time)
    for index, row in data.iterrows():
        row_time = row['start']  # Assuming 'start' is the column with the datetime information
        data.at[index, 'volume_change_since_twenty_four_hours'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time)
    correlation_with_volume_change_since_twenty_four_hours = data['price_change'].corr(data['volume_change_since_twenty_four_hours'])
    print(f"Correlation with volume change since twenty four hours: {correlation_with_volume_change_since_twenty_four_hours}")

def calculate_composite_score(df, product_id):
    df['Standardized_Volume_Change'] = (df['Volume_Change'] - df['Volume_Change'].mean()) / df['Volume_Change'].std()
    df['Standardized_RSI'] = (df['RSI'] - df['RSI'].mean()) / df['RSI'].std()
    df['Standardized_MACD'] = (df['MACD'] - df['MACD'].mean()) / df['MACD'].std()

    for index, row in df.iterrows():
        row_time = row['start']  # Assuming 'start' is the column with the datetime information
        df.at[index, 'volume_change_since_twenty_four_hours'] = get_volume_change_last_24_hours(client, product_id, row_time)

    df['Standardized_Volume_Change_Last_24'] = (df['volume_change_since_twenty_four_hours'] - df['volume_change_since_twenty_four_hours'].mean()) / df['volume_change_since_twenty_four_hours'].std()
    
    # Combine the standardized indicators into a composite score
    df['Composite_Score'] = (df['Standardized_Volume_Change_Last_24'] + df['Standardized_RSI'] + df['Standardized_MACD']) / 3
    return df

def calculate_correlation_with_composite(api_client, product_id, start_date, end_date, interval_time, granularity="FIFTEEN_MINUTE"):
    current_date = start_date
    all_data = []
    request_count = 0
    
    while current_date < end_date:
        if request_count >= 10000:
            print("Hourly request limit reached. Waiting for an hour before continuing...")
            time.sleep(3600)  # Wait for an hour
            request_count = 0
        
        next_date = current_date + timedelta(days=1)
        start_timestamp = int(current_date.timestamp())
        end_timestamp = int(next_date.timestamp())
        
        data = fetch_data(api_client, product_id, start_timestamp, end_timestamp, granularity)
        all_data.extend(data)
        request_count += 1
        
        current_date = next_date
        time.sleep(0.2)  # Sleep for 200ms to ensure we stay within 5 requests per second
    
    df = process_data(all_data, product_id)
    
    # Calculate Composite Score
    data = calculate_composite_score(df, product_id)
    
    # Calculate correlation
    correlation_with_composite_score = data['price_change'].corr(data['Composite_Score'])
    
    print(f"Correlation with Composite Score: {correlation_with_composite_score}")
    return correlation_with_composite_score

def calculate_correlation_interval_with_twenty_four(client, product_id, start_date, end_date, interval_time, granularity="FIFTEEN_MINUTE"):
    data = get_price_changes_for_interval_from_db('crypto_data.db', product_id, start_date, end_date, interval_time)
    for index, row in data.iterrows():
        row_time = row['start']  # Assuming 'start' is the column with the datetime information
        data.at[index, 'price_change_since_twenty_four_hours'] = get_price_change_last_24_hours_from_db('crypto_data.db', product_id, row_time)
    correlation_with_price_change_since_twenty_four_hours = data['price_change'].corr(data['price_change_since_twenty_four_hours'])
    print(f"Correlation with price change since twenty four hours: {correlation_with_price_change_since_twenty_four_hours}")

def determine_granularity(start_time, end_time):
    total_minutes = (end_time - start_time).total_seconds() / 60
    if total_minutes <= 300:
        return "ONE_MINUTE"
    elif total_minutes <= 1500:
        return "FIVE_MINUTES"
    elif total_minutes <= 7200:
        return "FIFTEEN_MINUTES"
    elif total_minutes <= 14400:
        return "ONE_HOUR"
    else:
        return "ONE_DAY"

def calculate_indicators_for_base(df, product_id):
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Price_Change'] = df['close'].diff()
    df['Volume_Change'] = df['volume'].diff()

    for index, row in df.iterrows():
        row_time = row['start']  # Assuming 'start' is the column with the datetime information
        df.at[index, 'volume_change_since_twenty_four_hours'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time)
        df.at[index, 'volume_change_since_one_hour'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=1)
        df.at[index, 'volume_change_since_six_hours'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=6)
        df.at[index, 'volume_change_since_twelve_hours'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=12)
        df.at[index, 'volume_change_since_seven_days'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=7*24)
        df.at[index, 'volume_change_since_thirty_days'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=30*24)
    df.dropna(inplace=True)  # Drop NaN values that result from indicator calculations
    return df

def import_data_to_sqlite(data, db_name="crypto_data.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            start INTEGER,
            low REAL,
            high REAL,
            open REAL,
            close REAL,
            volume REAL
        )
    ''')
    
    # Convert list of dictionaries to list of tuples
    data_tuples = [(d['start'], d['low'], d['high'], d['open'], d['close'], d['volume']) for d in data]
    
    cursor.executemany('''
        INSERT INTO candles (start, low, high, open, close, volume)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', data_tuples)
    
    conn.commit()
    conn.close()

def get_price_change_last_24_hours_from_db(db_name, product_id, interval_time):
    if isinstance(interval_time, str):
        try:
            end_time = datetime.strptime(interval_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Invalid datetime format for interval_time. Please use YYYY-MM-DD HH:MM:SS format.")
            return None
    elif isinstance(interval_time, pd.Timestamp):
        end_time = interval_time
    elif isinstance(interval_time, datetime):
        end_time = interval_time
    else:
        print("Unsupported type for interval_time. Please provide a string or Timestamp.")
        return None
    
    start_time = end_time - timedelta(days=1)
    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    query = '''
        SELECT open, close FROM candles
        WHERE start >= ? AND start <= ?
        ORDER BY start ASC
    '''
    cursor.execute(query, (start_time_unix, end_time_unix))
    data = cursor.fetchall()
    conn.close()
    if data:
        close_price = data[-1][1]
        open_price = data[0][1]
        price_change = (float(close_price) - float(open_price)) / float(open_price)
        return price_change
    return 0

def get_volume_change_last_hours_from_db(db_name, product_id, interval_time, hours=24):
    if isinstance(interval_time, str):
        try:
            end_time = datetime.strptime(interval_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("Invalid datetime format for interval_time. Please use YYYY-MM-DD HH:MM:SS format.")
            return None
    elif isinstance(interval_time, datetime):
        end_time = interval_time
    elif isinstance(interval_time, pd.Timestamp):
        end_time = interval_time
    else:
        print("Unsupported type for interval_time. Please provide a string or Timestamp.")
        return None

    start_time = end_time - timedelta(hours=hours)
    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    query = '''
        SELECT volume FROM candles
        WHERE start >= ? AND start <= ?
        ORDER BY start ASC
    '''
    cursor.execute(query, (start_time_unix, end_time_unix))
    data = cursor.fetchall()
    conn.close()

    if data:
        close_volume = data[-1][0]
        open_volume = data[0][0]
        volume_change = (float(close_volume) - float(open_volume)) / float(open_volume)
        return volume_change
    return 0

def get_candle_data_filtered_by_date(db_name, product_id, start_date, end_date):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    query = '''
        SELECT start, low, high, open, close, volume FROM candles
        WHERE start >= ? AND start <= ?
        ORDER BY start ASC
    '''
    cursor.execute(query, (start_timestamp, end_timestamp))
    data = cursor.fetchall()
    conn.close()

    return data

def get_price_changes_for_interval_from_db(db_name, product_id, start_date, end_date, interval_time):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    query = '''
        SELECT start, open, close FROM candles
        WHERE start >= ? AND start <= ?
        ORDER BY start ASC
    '''
    cursor.execute(query, (start_timestamp, end_timestamp))
    data = cursor.fetchall()
    conn.close()

    if not data:
        return pd.DataFrame(columns=['start', 'price_change'])

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['start', 'open', 'close'])
    df['start'] = pd.to_datetime(df['start'], unit='s')
    df['price_change'] = df['close'] - df['open']
    df['interval_time'] = df['start'].dt.time

    # Filter data for the specified interval time
    interval_data = df[df['interval_time'] == interval_time]
    price_changes = interval_data[['start', 'price_change']]

    return price_changes

def get_volume_changes_for_interval_from_db(db_name, product_id, start_date, end_date, interval_time):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    query = '''
    SELECT start, open, close, volume FROM candles
    WHERE start >= ? AND start <= ?
    ORDER BY start ASC
    '''

    cursor.execute(query, (start_timestamp, end_timestamp))
    data = cursor.fetchall()
    conn.close()

    if not data:
        return pd.DataFrame(columns=['start', 'volume_change'])

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['start', 'open', 'close', 'volume'])
    df['start'] = pd.to_datetime(df['start'], unit='s')
    df['volume_change'] = df['volume'].diff()
    df['price_change'] = df['close'] - df['open']
    df['interval_time'] = df['start'].dt.time

    # Filter data for the specified interval time
    interval_data = df[df['interval_time'] == interval_time]
    for index, row in interval_data.iterrows():
        row_time = row['start']  # Assuming 'start' is the column with the datetime information
        interval_data.loc[index, '24'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time)
        interval_data.loc[index, '1'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=1)
        interval_data.loc[index, '6'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=6)
        interval_data.loc[index, '12'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=12)
        interval_data.loc[index, '7'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=7*24)
        interval_data.loc[index, '30'] = get_volume_change_last_hours_from_db('crypto_data.db', product_id, row_time, hours=30*24)
    interval_data.dropna(inplace=True)  # Drop NaN values that result from indicator calculations
    volume_changes = interval_data[['start', '24', '1', '6', '12', '7', '30', 'price_change']]
    return volume_changes



def calculate_correlation(df, interval_time, volume_change_column='volume_change_since_twenty_four_hours'):
    df['interval_time'] = df['start'].dt.time
    interval_data = df[df['interval_time'] == interval_time]
    if len(interval_data) < 2:
        return None
    return interval_data['price_change'].corr(interval_data[volume_change_column])

def find_best_correlation(db_name, product_id, start_date, end_date):
    all_data = get_candle_data_filtered_by_date(db_name, product_id, start_date, end_date)
    df = process_data(all_data, product_id)

    best_correlation_twenty_four = -1
    best_correlation_one_hour = -1
    best_correlation_six_hours = -1
    best_correlation_twelve_hours = -1
    best_correlation_seven_days = -1
    best_correlation_thirty_days = -1
    best_interval_twenty_four = None
    best_interval_one_hour = None
    best_interval_six_hours = None
    best_interval_twelve_hours = None
    best_interval_seven_days = None
    best_interval_thirty_days = None

    for hour in range(24):
        for minute in range(0, 60, 15):  # Checking every 15 minutes
            interval_time = (datetime.min + timedelta(hours=hour, minutes=minute)).time()
            correlation_twenty_four = calculate_correlation(df, interval_time)
            correlation_one_hour = calculate_correlation(df, interval_time, volume_change_column='volume_change_since_one_hour')
            correlation_six_hours = calculate_correlation(df, interval_time, volume_change_column='volume_change_since_six_hours')
            correlation_twelve_hours = calculate_correlation(df, interval_time, volume_change_column='volume_change_since_twelve_hours')
            correlation_seven_days = calculate_correlation(df, interval_time, volume_change_column='volume_change_since_seven_days')
            correlation_thirty_days = calculate_correlation(df, interval_time, volume_change_column='volume_change_since_thirty_days')
            if correlation_twenty_four is not None and correlation_twenty_four > best_correlation_twenty_four:
                best_correlation_twenty_four = correlation_twenty_four
                best_interval_twenty_four = interval_time
            if correlation_one_hour is not None and correlation_one_hour > best_correlation_one_hour:
                best_correlation_one_hour = correlation_one_hour
                best_interval_one_hour = interval_time
            if correlation_six_hours is not None and correlation_six_hours > best_correlation_six_hours:
                best_correlation_six_hours = correlation_six_hours
                best_interval_six_hours = interval_time
            if correlation_twelve_hours is not None and correlation_twelve_hours > best_correlation_twelve_hours:
                best_correlation_twelve_hours = correlation_twelve_hours
                best_interval_twelve_hours = interval_time
            if correlation_seven_days is not None and correlation_seven_days > best_correlation_seven_days:
                best_correlation_seven_days = correlation_seven_days
                best_interval_seven_days = interval_time
            if correlation_thirty_days is not None and correlation_thirty_days > best_correlation_thirty_days:
                best_correlation_thirty_days = correlation_thirty_days
                best_interval_thirty_days = interval_time

    return best_interval_twenty_four, best_correlation_twenty_four, best_interval_one_hour, best_correlation_one_hour, best_interval_six_hours, best_correlation_six_hours, best_interval_twelve_hours, best_correlation_twelve_hours, best_interval_seven_days, best_correlation_seven_days, best_interval_thirty_days, best_correlation_thirty_days

def calculate_indicators(indicators, product_id, start_date, end_date, granularity):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    data = get_candle_data_filtered_by_date("crypto_data.db", product_id, start_date, end_date)

    df = pd.DataFrame(data, columns=['start', 'low', 'high', 'open', 'close', 'volume'])
    df['start'] = pd.to_datetime(df['start'], unit='s')

    for indicator in indicators:
        try:
            if indicator == 'BBANDS':
                result = talib.BBANDS(df['close'])
                print(f"{indicator}: {result}")
            elif indicator == 'MAMA':
                result = talib.MAMA(df['close'])
                print(f"{indicator}: {result}")
            elif indicator == 'MAVP':
                result = talib.MAVP(df['close'], df['volume'])  # Example, adjust as needed
                print(f"{indicator}: {result}")
            elif indicator == 'MIDPRICE':
                result = talib.MIDPRICE(df['high'], df['low'])
                print(f"{indicator}: {result}")
            elif indicator == 'SAR':
                result = talib.SAR(df['high'], df['low'])
                print(f"{indicator}: {result}")
            elif indicator == 'SAREXT':
                result = talib.SAREXT(df['high'], df['low'])
                print(f"{indicator}: {result}")
            elif indicator in ['DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 'MIDPOINT', 'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA']:
                result = getattr(talib, indicator)(df['close'])
                print(f"{indicator}: {result}")
            elif indicator in talib.get_function_groups()['Pattern Recognition']:
                result = getattr(talib, indicator)(df['open'], df['high'], df['low'], df['close'])
                result = result[result != 0]
                if not result.empty:
                    print(f"{indicator}: {result}")
            else:
                result = getattr(talib, indicator)(df['close'])
        except Exception as e:
            print(f"Error calculating {indicator}: {e}")

def backtest_strategy(product_id, start_date, end_date, interval, volume_threshold=0.05, price_threshold=0.005):
    conn = sqlite3.connect('crypto_data.db')
    cursor = conn.cursor()

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    query = '''
        SELECT start, low, high, open, close, volume FROM candles
        WHERE start >= ? AND start <= ?
        ORDER BY start ASC
    '''
    cursor.execute(query, (start_timestamp, end_timestamp))
    data = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(data, columns=['start', 'low', 'high', 'open', 'close', 'volume'])
    df['start'] = pd.to_datetime(df['start'], unit='s')
    df['price_change'] = df['close'].diff()
    df['volume_change'] = df['volume'].pct_change()

    # Filter data for the specified interval time
    df['interval_time'] = df['start'].dt.time
    interval_data = df[df['interval_time'] == interval]

    # Backtest logic
    positions = []
    for index, row in interval_data.iterrows():
        if row['volume_change'] > volume_threshold and row['price_change'] > price_threshold:
            entry_price = row['close']
            stop_loss = entry_price * 0.98
            take_profit = entry_price * 1.03
            positions.append({
                'entry_time': row['start'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })

    # Evaluate positions
    results = []
    for position in positions:
        entry_time = position['entry_time']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']

        future_data = df[df['start'] > entry_time]
        for _, future_row in future_data.iterrows():
            if future_row['low'] <= stop_loss:
                results.append({'entry_time': entry_time, 'exit_time': future_row['start'], 'exit_price': stop_loss, 'result': 'stop_loss'})
                break
            elif future_row['high'] >= take_profit:
                results.append({'entry_time': entry_time, 'exit_time': future_row['start'], 'exit_price': take_profit, 'result': 'take_profit'})
                break
        else:
            results.append({'entry_time': entry_time, 'exit_time': future_data.iloc[-1]['start'], 'exit_price': future_data.iloc[-1]['close'], 'result': 'hold'})

    return results

def main():
    parser = argparse.ArgumentParser(description='Crypto analysis tool.')
    parser.add_argument('--product_id', type=str, help='Product ID for the crypto asset')
    parser.add_argument('--find-cyclical-patterns', action='store_true', help='Find cyclical patterns for the given product ID')
    parser.add_argument('--price-changes-for-interval', action='store_true', help='Get price changes for a specific interval')
    parser.add_argument('--volume-changes-for-interval', action='store_true', help='Get volume changes for a specific interval')
    parser.add_argument('--previous-twenty-four-hours', action='store_true', help='Get the overall price change for the last 24 hours')
    parser.add_argument('--since-beginning-of-day', action='store_true', help='Get the price change since the beginning of the specified day')
    parser.add_argument('--rsi', action='store_true', help='Calculate RSI for the given product ID up to the specified interval')
    parser.add_argument('--macd', action='store_true', help='Calculate MACD for the given product ID up to the specified interval')
    parser.add_argument('--interval_time', type=str, help='Interval time in YYYY-MM-DD HH:MM:SS format for calculations')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--calculate_correlation_interval_with_twenty_four', action='store_true', help='Calculate correlation between price changes and indicators')
    parser.add_argument('--calculate_correlation_interval_with_twenty_four_volume', action='store_true', help='Calculate correlation between price changes and indicators')
    parser.add_argument('--calculate_correlation_interval_with_composite', action='store_true', help='Calculate correlation between price changes and indicators')
    parser.add_argument('--granularity', type=str, default='ONE_MINUTE', choices=['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'SIX_HOURS', 'ONE_DAY'],
                        help='Granularity of the data to fetch. Default is ONE_MINUTE.')
    parser.add_argument('--importdb', action='store_true', help='Import fetched data into SQLite database')
    parser.add_argument('--find-best-correlation', action='store_true', help='Find the best correlation between price changes and volume changes for the given product ID')
    parser.add_argument('--discover-abc-pattern', action='store_true', help='Discover ABC pattern with volume and Fibonacci levels')
    parser.add_argument('--days_back', type=int, default=7, help='Number of days to look back for the ABC pattern')

    # Add switches for each category of indicators
    parser.add_argument('--overlap-studies', action='store_true', help='Calculate Overlap Studies indicators')
    parser.add_argument('--momentum-indicators', action='store_true', help='Calculate Momentum Indicators')
    parser.add_argument('--volume-indicators', action='store_true', help='Calculate Volume Indicators')
    parser.add_argument('--cycle-indicators', action='store_true', help='Calculate Cycle Indicators')
    parser.add_argument('--volatility-indicators', action='store_true', help='Calculate Volatility Indicators')
    parser.add_argument('--pattern-recognition', action='store_true', help='Calculate Pattern Recognition indicators')
    parser.add_argument('--statistic-functions', action='store_true', help='Calculate Statistic Functions')
    parser.add_argument('--backtest', action='store_true', help='Backtest the strategy with the given interval')

    args = parser.parse_args()

    if args.importdb:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        all_data = []
        current_date = start_date

        while current_date < end_date:
            next_date = current_date + timedelta(days=1)
            start_timestamp = int(current_date.timestamp())
            end_timestamp = int(next_date.timestamp())
            
            data = fetch_data(client, args.product_id, start_timestamp, end_timestamp, args.granularity)
            all_data.extend(data)
            
            current_date = next_date
            time.sleep(0.2)  # Sleep for 200ms to ensure we stay within 5 requests per second
        
        import_data_to_sqlite(all_data)
        print(f"Data imported into SQLite database successfully.")
        return

    if args.discover_abc_pattern:
        granularities = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'SIX_HOURS', 'ONE_DAY']
        days_back_options = [7, 14, 30, 60, 90]

        patterns_found = []

        for granularity in granularities:
            for days_back in days_back_options:
                result = discover_abc_pattern_with_volume_fibonacci(args.product_id, days_back, granularity)
                if result:
                    A_point, B_point, C_point, fib_retracement = result
                    patterns_found.append({
                        'granularity': granularity,
                        'days_back': days_back,
                        'A_point': A_point,
                        'B_point': B_point,
                        'C_point': C_point,
                        'fib_retracement': fib_retracement
                    })

        if patterns_found:
            print("ABC Patterns found:")
            for pattern in patterns_found:
                print(f"Granularity: {pattern['granularity']}, Days Back: {pattern['days_back']}")
                print(f"A: {pattern['A_point']}, B: {pattern['B_point']}, C: {pattern['C_point']}, Fibonacci Retracement: {pattern['fib_retracement']}")
        else:
            print("No ABC patterns found.")

    if args.overlap_studies:
        calculate_indicators(OVERLAP_STUDIES, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.momentum_indicators:
        calculate_indicators(MOMENTUM_INDICATORS, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.volume_indicators:
        calculate_indicators(VOLUME_INDICATORS, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.cycle_indicators:
        calculate_indicators(CYCLE_INDICATORS, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.volatility_indicators:
        calculate_indicators(VOLATILITY_INDICATORS, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.pattern_recognition:
        calculate_indicators(PATTERN_RECOGNITION, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.statistic_functions:
        calculate_indicators(STATISTIC_FUNCTIONS, args.product_id, args.start_date, args.end_date, args.granularity)

    if args.find_cyclical_patterns:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        find_cyclical_patterns(args.product_id, start_date, end_date, args.granularity)

    if args.price_changes_for_interval:
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        price_changes = get_price_changes_for_interval_from_db('crypto_data.db', args.product_id, start_date, end_date, interval_time)
        print(price_changes)

    if args.previous_twenty_four_hours:
        interval_time = args.interval_time if args.interval_time else datetime.now()
        print(f"interval_time: {interval_time}")
        price_change = get_price_change_last_24_hours_from_db('crypto_data.db', args.product_id, interval_time)
        if price_change is not None:
            print(f"Price change in the last 24 hours for {args.product_id} as of {interval_time}: {price_change}%")

    if args.since_beginning_of_day:
        if args.interval_time:
            price_change = get_price_change_since_beginning_of_day(client, args.product_id, args.interval_time)
            if price_change is not None:
                print(f"Price change since the beginning of the day for {args.product_id} on {args.interval_time}: {price_change}%")
        else:
            print("Error: Date for --since-beginning-of-day not provided.")

    if args.rsi or args.macd:
        if not args.interval_time:
            print("Error: Please specify --interval_time for RSI or MACD calculations.")
            return

        end_time = datetime.strptime(args.interval_time, "%Y-%m-%d %H:%M:%S")
        start_time = end_time - timedelta(days=30)  # Assuming 30 days data is needed for calculation

        # Determine appropriate granularity based on the time range
        granularity = determine_granularity(start_time, end_time)

        try:
            data = fetch_data(client, args.product_id, int(start_time.timestamp()), int(end_time.timestamp()), granularity)
            close_prices = [float(d['close']) for d in data]

            if args.rsi:
                rsi_value = calculate_rsi(close_prices)
                print(f"RSI as of {args.interval_time}: {rsi_value}")

            if args.macd:
                macd_value, signal_line = calculate_macd(close_prices)
                print(f"MACD as of {args.interval_time}: {macd_value}, Signal Line: {signal_line}")
        except Exception as e:
            print(f"Error fetching data: {e}")

    if args.calculate_correlation_interval_with_twenty_four:
        if not args.product_id or not args.interval_time:
            print("Error: Please specify both --product_id and --interval_time for correlation calculations.")
            return
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        calculate_correlation_interval_with_twenty_four(client, args.product_id, start_date, end_date, interval_time, args.granularity)

    if args.calculate_correlation_interval_with_twenty_four_volume:
        if not args.product_id or not args.interval_time:
            print("Error: Please specify both --product_id and --interval_time for correlation calculations.")
            return
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        calculate_correlation_interval_with_volume_twenty_four(client, args.product_id, start_date, end_date, interval_time, args.granularity)

    if args.calculate_correlation_interval_with_composite:
        if not args.product_id or not args.interval_time:
            print("Error: Please specify both --product_id and --interval_time for correlation calculations.")
            return
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        calculate_correlation_with_composite(client, args.product_id, start_date, end_date, interval_time, args.granularity)

    if args.volume_changes_for_interval:
        if not args.product_id or not args.interval_time:
            print("Error: Please specify both --product_id and --interval_time for correlation calculations.")
            return
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        volume_changes = get_volume_changes_for_interval_from_db("crypto_data.db", args.product_id, start_date, end_date, interval_time)
        print(volume_changes.to_string())
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            # print(volume_changes)

    if args.find_best_correlation:
        db_name = "crypto_data.db"
        product_id = args.product_id
        # end_date = datetime.now()
        # start_date = end_date - timedelta(days=3*365)  # Look back 3 years
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        best_interval_twenty_four, best_correlation_twenty_four, best_interval_one_hour, best_correlation_one_hour, best_interval_six_hours, best_correlation_six_hours, best_interval_twelve_hours, best_correlation_twelve_hours, best_interval_seven_days, best_correlation_seven_days, best_interval_thirty_days, best_correlation_thirty_days = find_best_correlation(db_name, product_id, start_date, end_date)
        print(f"24 hour Best interval: {best_interval_twenty_four}, Correlation: {best_correlation_twenty_four}")
        print(f"1 hour Best interval: {best_interval_one_hour}, Correlation: {best_correlation_one_hour}")
        print(f"6 hours Best interval: {best_interval_six_hours}, Correlation: {best_correlation_six_hours}")
        print(f"12 hours Best interval: {best_interval_twelve_hours}, Correlation: {best_correlation_twelve_hours}")
        print(f"7 days Best interval: {best_interval_seven_days}, Correlation: {best_correlation_seven_days}")
        print(f"30 days Best interval: {best_interval_thirty_days}, Correlation: {best_correlation_thirty_days}")

    if args.discover_abc_pattern:
        result = discover_abc_pattern_with_volume_fibonacci(args.product_id)
        if result:
            A_point, B_point, C_point, fib_retracement = result
            print(f"ABC Pattern found: A={A_point}, B={B_point}, C={C_point}, Fibonacci Retracement={fib_retracement}")
        else:
            print("No ABC pattern found.")

    if args.backtest:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        results = backtest_strategy(args.product_id, start_date, end_date, interval_time)
        for result in results:
            print(result)
        return

if __name__ == "__main__":
    main()



