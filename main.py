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
    df = calculate_indicators(df, data, product_id)
    return df

def analyze_intervals(df):
    grouped = df.groupby([df['start'].dt.time]).agg({
        'volatility': 'mean',
        'price_change': 'sum'
    }).reset_index()
    return grouped

def find_cyclical_patterns(api_client, product_id, start_date, end_date, granularity="FIFTEEN_MINUTE"):
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
        A_index = np.argmax(close_prices)
        C_index = len(close_prices) - 1
        B_index = np.argmin(close_prices[A_INDEX:C_INDEX]) + A_INDEX
        
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

def discover_abc_pattern_with_volume_fibonacci(product_id, days_back=7, granularity="ONE_DAY"):
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

    try:
        A_index = np.argmax(close_prices)
        C_index = len(close_prices) - 1
        B_index = np.argmin(close_prices[A_INDEX:C_INDEX]) + A_INDEX
        
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
        data.at[index, 'volume_change_since_twenty_four_hours'] = get_volume_change_last_24_hours_from_db('crypto_data.db', product_id, row_time)
    print(data)
    correlation_with_volume_change_since_twenty_four_hours = data['price_change'].corr(data['volume_change_since_twenty_four_hours'])
    print(f"Correlation with volume change since twenty four hours: {correlation_with_volume_change_since_twenty_four_hours}")

def calculate_composite_score(df, product_id):
    print("hi")
    print(df)
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
    print(data)
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

def calculate_indicators(df, raw_data, product_id):
    print(df)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['Price_Change'] = df['close'].diff()
    df['Volume_Change'] = df['volume'].diff()

    for index, row in df.iterrows():
        row_time = row['start']  # Assuming 'start' is the column with the datetime information
        df.at[index, 'volume_change_since_twenty_four_hours'] = get_volume_change_last_24_hours_from_db('crypto_data.db', product_id, row_time)
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
        price_change = (float(data[0][1]) - float(data[-1][1])) / float(data[-1][1])
        return price_change
    return 0

def get_volume_change_last_24_hours_from_db(db_name, product_id, interval_time):
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
        volume_change = (float(data[0][0]) - float(data[-1][0])) / float(data[-1][0])
        return volume_change
    return 0

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

    if args.find_cyclical_patterns:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        find_cyclical_patterns(client, args.product_id, start_date, end_date, args.granularity)

    if args.price_changes_for_interval:
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        price_changes = get_price_changes_for_interval(client, args.product_id, start_date, end_date, interval_time, args.granularity)
        print(price_changes)

    if args.previous_twenty_four_hours:
        interval_time = args.interval_time if args.interval_time else datetime.now()
        price_change = get_price_change_last_24_hours(client, args.product_id, interval_time)
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


if __name__ == "__main__":
    main()



