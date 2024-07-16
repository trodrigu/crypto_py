# from coinbase.wallet.client import Client
from coinbase.rest import RESTClient
from datetime import datetime, timedelta
import numpy as np
import uuid

# Initialize the client with your Coinbase API credentials
client = RESTClient()

def signal_buy_opportunity(product_id, days_back=7, granularity="ONE_DAY"):
    """Signal a potential buy opportunity based on the anticipation of the C leg development."""
    # The existing setup to fetch historical data remains the same
    
    # Adjusted logic to identify the end of B and the start of C
    # Assuming the same setup for fetching and preparing the data as in `discover_abc_pattern_with_volume_fibonacci`
    # Calculate start and end times for historical data fetch
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())
    
    try:
        # Fetch historical data
        candles = client.get_candles(
            product_id=product_id, 
            start=start_time_unix,
            end=end_time_unix, 
            granularity=granularity
        )
    except Exception as e:
        print(f"Error fetching historical data for {product_id}: {e}")
        return None

    # Extract close prices, volumes, and timestamps from the data
    close_prices = [float(candle['close']) for candle in candles['candles']]
    volumes = [float(candle['volume']) for candle in candles['candles']]
    timestamps = [datetime.utcfromtimestamp(int(candle['start'])) for candle in candles['candles']]

    # Calculate 24-hour volume change if data covers at least 2 days
    if len(volumes) > 1:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change = 0

    try:
        A_index = np.argmax(close_prices)  # Highest price point in the period
        C_index = len(close_prices) - 1  # Last price point
        B_index = np.argmin(close_prices[A_index:C_index]) + A_index  # Lowest point after A and before C
        
        A_point = (timestamps[A_index], close_prices[A_index], volumes[A_index])
        B_point = (timestamps[B_index], close_prices[B_index], volumes[B_index])
        C_point = (timestamps[C_index], close_prices[C_index], volumes[C_index])

        fib_retracement_levels = [0.382, 0.5, 0.618]  # Common Fibonacci levels to check for B-C transition
        potential_buy_signal = False
        detailed_reason = ""
        price_recovery_ratio = (C_point[1] - B_point[1]) / (A_point[1] - B_point[1])
        
        with_scores = []
        for level in fib_retracement_levels:
            fib_retracement = A_point[1] - (A_point[1] - B_point[1]) * level
            if close_prices[-1] > fib_retracement and A_point[1] > B_point[1] and C_point[2] > B_point[2] and volumes[-1] > volumes[B_index]:
                # This implies we're seeing a price recovery beyond a Fibonacci level with increasing volume
                potential_buy_signal = True
                detailed_reason = f"Price recovered beyond the {level*100}% Fibonacci level with increased volume."
                with_score = {"product_id": product_id, "score": calculate_score(volume_change, level, price_recovery_ratio), "details": detailed_reason}
                with_scores.append(with_score)
        return with_scores
            # else:

                # break

        # if potential_buy_signal and volume_change > 0.2:
            # print(f"Significant 24-hour volume change detected for {product_id} with an increase of {volume_change*100}%")
            # print(f"Potential buy opportunity detected for {product_id}. {detailed_reason}")
            # print("\n")
            # return True, detailed_reason
        # else:
            # # print("No clear buy opportunity detected.")
            # return False, "No clear buy opportunity detected."
            
    except Exception as e:
        # print(f"Error analyzing data for {product_id}: {e}")
        # return None, str(e)
        return []

# Example usage:
# product_id = 'BTC-USD'
# signal_buy_opportunity(product_id)

def calculate_score(volume_change, fib_level, price_recovery_ratio):
    """Example scoring function."""
    score = 0
    # Volume change scoring (arbitrary scoring example)
    if volume_change > 0.5: score += 3
    elif volume_change > 0.2: score += 2
    else: score += 1

    # Fibonacci level scoring
    # invert for fun
    if fib_level == 0.5: score += 3
    elif fib_level == 0.382: score += 2
    # if fib_level == 0.618: score += 3
    # elif fib_level == 0.5: score += 2
    elif fib_level == 0.618: score += 1

    # Price recovery ratio scoring (closer to A point scores higher)
    # invert for fun
    if price_recovery_ratio > 0.8: score += 1
    elif price_recovery_ratio > 0.6: score += 2
    else: score += 3
    # if price_recovery_ratio > 0.8: score += 3
    # elif price_recovery_ratio > 0.6: score += 2
    # else: score += 1

    return score

def get_best_bid_ask(product_id):
    """Fetch the best bid and ask for a given product."""
    try:
        order_book = client.get_product_book(product_id, level=1)
    except exception as e:
        print(f"error: {e}")
        print(f"Error fetching order book for {product_id}")


    best_bid = order_book['pricebook']['bids'][0]['price']
    best_ask = order_book['pricebook']['asks'][0]['price']
    return float(best_bid), float(best_ask)

# fetch all products and get their ids
def fetch_all_product_ids():
    """Fetch all products from Coinbase."""
    # map over products and get their ids
    products = client.get_products()['products']
    # get the product id if the key is product_id
    return [product['product_id'] for product in products]

def fetch_filtered_products():
    """Fetch and filter products based on criteria (e.g., '-USD' or '-USDT' suffix)."""
    all_products = client.get_products()
    filtered_products = [product for product in all_products['products'] if product['product_id'].endswith('-USD') or product['product_id'].endswith('-USDT')]
    return filtered_products

def compare_prices_for_arbitrage(product_ids):
    """Compare prices across products to find arbitrage opportunities."""
    prices = {product_id: get_best_bid_ask(product_id) for product_id in product_ids}
    
    # Find product with the highest bid and the lowest ask
    max_bid_product = max(prices.items(), key=lambda x: x[1][0])
    min_ask_product = min(prices.items(), key=lambda x: x[1][1])
    
    # difference = max_bid_product[1][0] - min_ask_product[1][1]
    # fee = 0.008 * max_bid_product[1][0] + 0.008 * min_ask_product[1][1]
    # Assuming buy_size and sell_size are correctly calculated based on your strategy
    buy_price = min_ask_product[1][1]  # The price at which you'll buy
    sell_price = max_bid_product[1][0]
    # convert $140 USD to what buy price is and return buy size
    buy_size = 100 / buy_price
    # convert buy size to sell size
    sell_size = buy_size
    total_buy_cost = buy_price * buy_size
    total_sell_revenue = sell_price * sell_size

    # Apply the fee to the total transaction amount
    buy_fee = total_buy_cost * 0.008
    sell_fee = total_sell_revenue * 0.008

    # Calculate net profit considering fees
    net_profit = total_sell_revenue - total_buy_cost - buy_fee - sell_fee

    if net_profit > 0:
        # calculate if difference if greater than 0.05%
        # what is the total fee if there are 2 transactions?
        # if difference > fee:
        buy_product_id = min_ask_product[0]
        sell_product_id = max_bid_product[0]

        print(f"Potential arbitrage opportunity: Buy from {min_ask_product[0]} at {min_ask_product[1][1]} and sell on {max_bid_product[0]} at {max_bid_product[1][0]}")

        # Place buy limit order
        buy_order_response = place_market_order(buy_product_id, 'buy', buy_price, buy_size)
        # Place sell limit order
        sell_order_response = place_market_order_sell(sell_product_id, 'sell', sell_price, sell_size, buy_size, buy_price)

        # else:
            # print("Arbitrage opportunity found but difference is less than 0.05%")
    # else:
        # print("No arbitrage opportunity found.")

def get_unique_first_part_of_product_ids(product_ids):
    """Get the unique first part of product IDs."""
    return list(set([product_id.split('-')[0] for product_id in product_ids]))

def place_market_order(product_id, side, price, size):
    """Place a limit order on Coinbase."""
    # This is a placeholder function. You'll need to replace it with the actual function call from the Coinbase SDK.
    # Ensure you have the correct permissions and API keys setup for trading.
    # generate random UUID
    client_order_id = str(uuid.uuid4())
    if side == 'buy':
        order_response = client.market_order_buy(client_order_id=client_order_id, product_id=product_id, quote_size=str(size))
    elif side == 'sell':
        order_response = client.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=str(size))
    return order_response

def place_market_order_sell(product_id, side, price, size, buy_size, buy_price):
    """Place a limit order on Coinbase."""
    # This is a placeholder function. You'll need to replace it with the actual function call from the Coinbase SDK.
    # Ensure you have the correct permissions and API keys setup for trading.
    # generate random UUID
    client_order_id = str(uuid.uuid4())
    fee_percentage = 0.008  # Example fee percentage (0.8%)

    # Assuming buy_price and buy_size have been defined
    total_buy_cost = buy_price * buy_size
    buy_fee = total_buy_cost * fee_percentage

    # Calculate the actual amount of currency purchased after fees
    # This is simplified and assumes you receive exactly buy_size units of currency for the cost, which may not always be accurate due to price fluctuations and fees applied in terms of the purchased currency.
    actual_currency_purchased = (total_buy_cost - buy_fee) / buy_price

    # For sell_size, you want to sell all the actual_currency_purchased
    sell_size = actual_currency_purchased
    order_response = client.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=sell_size)
    return order_response

def get_product_ids_which_match_first_part(product_ids, first_part):
    """Get product IDs which match the first part."""
    return [product_id for product_id in product_ids if product_id.startswith(first_part)]

def group_by_product_ids(product_ids):
    """Group product IDs by their first part."""
    first_parts = get_unique_first_part_of_product_ids(product_ids)
    return {first_part: get_product_ids_which_match_first_part(product_ids, first_part) for first_part in first_parts}

def discover_abc_pattern_with_volume_fibonacci(product_id, days_back=7, granularity="ONE_DAY"):
    """Discover unfolding ABC pattern with volume and Fibonacci retracement considerations."""
    # Calculate start and end times for historical data fetch
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)

    end_time_unix = int(end_time.timestamp())
    start_time_unix = int(start_time.timestamp())
    
    try:
        # Fetch historical data
        candles = client.get_candles(
            product_id=product_id, 
            start=start_time_unix,
            end=end_time_unix, 
            granularity=granularity
        )
    except Exception as e:
        print(f"Error fetching historical data for {product_id}: {e}")
        return None

    # Extract close prices, volumes, and timestamps from the data
    close_prices = [float(candle['close']) for candle in candles['candles']]
    volumes = [float(candle['volume']) for candle in candles['candles']]
    timestamps = [datetime.utcfromtimestamp(int(candle['start'])) for candle in candles['candles']]

    try:
        A_index = np.argmax(close_prices)  # Highest price point in the period
        C_index = len(close_prices) - 1  # Last price point
        B_index = np.argmin(close_prices[A_index:C_index]) + A_index  # Lowest point after A and before C
        
        A_point = (timestamps[A_index], close_prices[A_index], volumes[A_index])
        B_point = (timestamps[B_index], close_prices[B_index], volumes[B_index])
        C_point = (timestamps[C_index], close_prices[C_index], volumes[C_index])

        # Calculate Fibonacci retracement level from A to B
        # fib_retracement = A_point[1] - (A_point[1] - B_point[1]) * 0.618
        # fib_retracement = A_point[1] - (A_point[1] - B_point[1]) * 0.50
        fib_retracement = A_point[1] - (A_point[1] - B_point[1]) * 0.382

        # Verify if ABC pattern criteria are met, including volume increase from B to C
        if A_point[1] < C_point[1] and A_point[1] > B_point[1] and C_point[2] > B_point[2] and C_point[1] > fib_retracement:
            print(f"Potential ABC pattern detected for {product_id}:")
            print(f"A (Peak): {A_point}, B (Correction): {B_point}, C (Final Rise): {C_point}")
            print(f"Fibonacci Retracement Level (61.8% from A to B): {fib_retracement}")
            return A_point, B_point, C_point, fib_retracement
        else:
            # print("No clear ABC pattern detected.")
            return None
    except Exception as e:
        # print(f"Error analyzing data for {product_id}: {e}")
        return None



# Example usage
products = fetch_filtered_products()
print(f"products: {products}")
product_ids = [product['product_id'] for product in products]
# unique_first_parts = get_unique_first_part_of_product_ids(product_ids)
grouped_product_ids = group_by_product_ids(product_ids)
# # remve group with only one product
grouped_product_ids = {k: v for k, v in grouped_product_ids.items() if len(v) > 1}
# # filter out FORT and TRB
grouped_product_ids = {k: v for k, v in grouped_product_ids.items() if k not in ['FORT', 'T']}
# # for each check if there is arbitrage
for k, v in grouped_product_ids.items():
    compare_prices_for_arbitrage(v)

# product_ids = ['BTC-USD', 'ETH-USD']  # Example product IDs
# compare_prices_for_arbitrage(product_ids)


# Example usage:
# product_id = 'BTC-USD'
# discover_abc_pattern_with_volume_fibonacci(product_id)

# Use the above and fetch all product ids and iterate
# product_ids = fetch_all_product_ids()
# opportunities = []
# for product_id in product_ids:
    # signals = signal_buy_opportunity(product_id, days_back=0.125, granularity="ONE_MINUTE")
    # if signals:
        # opportunities = opportunities + signals

# opportunities.sort(key=lambda x: x['score'], reverse=True)

# for opportunity in opportunities:
    # print(opportunity)
    # print("\n")


# buy_order_response = place_market_order('ENJ-USDT', 'buy', 0, 5)
# print(buy_order_response)
# # Place sell limit order
# sell_order_response = place_market_order_sell('ENJ-USD', 'sell', 0, 10)
# print(sell_order_response)
