from coinbase.rest import RESTClient
from datetime import datetime, timedelta
import numpy as np
import uuid
import pandas as pd
import time
import requests
import argparse

# Initialize the client with your Coinbase API credentials
client = RESTClient()

def fetch_data(api_client, product_id, start, end, granularity="FIFTEEN_MINUTE"):
    while True:
        try:
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

def process_data(data):
    df = pd.DataFrame(data, columns=["start", "low", "high", "open", "close", "volume"])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='s')
    df['low'] = df['low'].astype(float)
    df['high'] = df['high'].astype(float)
    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['price_change'] = df['close'] - df['open']
    df['volatility'] = df['high'] - df['low']
    return df

def analyze_intervals(df):
    grouped = df.groupby([df['start'].dt.time]).agg({
        'volatility': 'mean',
        'price_change': 'sum'
    }).reset_index()
    return grouped

def find_cyclical_patterns(api_client, product_id, start_date, end_date):
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
        
        data = fetch_data(api_client, product_id, start_timestamp, end_timestamp)
        all_data.extend(data)
        request_count += 1
        
        current_date = next_date
        time.sleep(0.2)  # Sleep for 200ms to ensure we stay within 5 requests per second
    
    df = process_data(all_data)
    analyzed_df = analyze_intervals(df)
    
    # Find the intervals with the least volatility on average
    best_intervals = analyzed_df.sort_values(by='volatility').head(10)
    print("Best Cyclical Intervals for 100x Leverage Trading:")
    print(best_intervals)
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

def get_price_changes_for_interval(api_client, product_id, start_date, end_date, interval_time):
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
        
        data = fetch_data(api_client, product_id, start_timestamp, end_timestamp)
        all_data.extend(data)
        request_count += 1
        
        current_date = next_date
        time.sleep(0.2)  # Sleep for 200ms to ensure we stay within 5 requests per second
    
    df = process_data(all_data)
    df['interval_time'] = df['start'].dt.time
    
    # Filter data for the specified interval time
    interval_data = df[df['interval_time'] == interval_time]
    price_changes = interval_data[['start', 'price_change']]
    
    return price_changes

def get_price_change_last_24_hours(api_client, product_id, interval_time, granularity="FIFTEEN_MINUTE"):
    try:
        end_time = datetime.strptime(interval_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("Invalid datetime format for interval_time. Please use YYYY-MM-DD HH:MM:SS format.")
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
        open_price = float(data[-1]['open'])
        close_price = float(data[0]['close'])
        return ((close_price - open_price) / open_price) * 100
    return 0

def main():
    parser = argparse.ArgumentParser(description='Crypto analysis tool.')
    parser.add_argument('--product_id', type=str, help='Product ID for the crypto asset')
    parser.add_argument('--find-cyclical-patterns', action='store_true', help='Find cyclical patterns for the given product ID')
    parser.add_argument('--price-changes-for-interval', action='store_true', help='Get price changes for a specific interval')
    parser.add_argument('--previous-twenty-four-hours', action='store_true', help='Get the overall price change for the last 24 hours')
    parser.add_argument('--since-beginning-of-day', action='store_true', help='Get the price change since the beginning of the specified day')
    parser.add_argument('--interval_time', type=str, help='Interval time in YYYY-MM-DD HH:MM:SS format for the previous 24 hours and beginning of day calculations')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')

    args = parser.parse_args()

    if args.find_cyclical_patterns:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        find_cyclical_patterns(client, args.product_id, start_date, end_date)

    if args.price_changes_for_interval:
        try:
            interval_time = datetime.strptime(args.interval_time, "%H:%M:%S").time()
        except ValueError:
            print("Invalid time format for interval_time. Please use HH:MM:SS format.")
            return
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        price_changes = get_price_changes_for_interval(client, args.product_id, start_date, end_date, interval_time)
        print(price_changes)

    if args.previous_twenty_four_hours:
        price_change = get_price_change_last_24_hours(client, args.product_id, args.interval_time)
        if price_change is not None:
            print(f"Price change in the last 24 hours for {args.product_id} as of {args.interval_time}: {price_change}%")

    if args.since_beginning_of_day:
        if args.interval_time:
            price_change = get_price_change_since_beginning_of_day(client, args.product_id, args.interval_time)
            if price_change is not None:
                print(f"Price change since the beginning of the day for {args.product_id} on {args.interval_time}: {price_change}%")
        else:
            print("Error: Date for --since-beginning-of-day not provided.")

if __name__ == "__main__":
    main()


