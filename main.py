# from coinbase.wallet.client import Client
from coinbase.rest import RESTClient

# Initialize the client with your Coinbase API credentials
client = RESTClient()

def get_best_bid_ask(product_id):
    """Fetch the best bid and ask for a given product."""
    order_book = client.get_product_book(product_id, level=1)
    best_bid = order_book['pricebook']['bids'][0]['price']
    best_ask = order_book['pricebook']['asks'][0]['price']
    return float(best_bid), float(best_ask)

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
    
    if max_bid_product[1][0] > min_ask_product[1][1]:
        difference = max_bid_product[1][0] - min_ask_product[1][1]
        # calculate if difference if greater than 0.05%
        # what is the total fee if there are 2 transactions?
        fee = 0.008 * max_bid_product[1][0] + 0.005 * min_ask_product[1][1]
        if difference > fee:
            buy_product_id = min_ask_product[0]
            sell_product_id = max_bid_product[0]
            buy_price = min_ask_product[1][1]  # The price at which you'll buy
            sell_price = max_bid_product[1][0]
            # convert $140 USD to what buy price is and return buy size
            buy_size = 140 / buy_price
            # convert buy size to sell size
            sell_size = buy_size

            # Place buy limit order
            buy_order_response = place_limit_order(buy_product_id, 'buy', buy_price, buy_size)
            # Place sell limit order
            sell_order_response = place_limit_order(sell_product_id, 'sell', sell_price, sell_size)

            print(f"Potential arbitrage opportunity: Buy from {min_ask_product[0]} at {min_ask_product[1][1]} and sell on {max_bid_product[0]} at {max_bid_product[1][0]}")
            print(f"Difference: {difference}")
        # else:
            # print("Arbitrage opportunity found but difference is less than 0.05%")
    # else:
        # print("No arbitrage opportunity found.")

def get_unique_first_part_of_product_ids(product_ids):
    """Get the unique first part of product IDs."""
    return list(set([product_id.split('-')[0] for product_id in product_ids]))

def place_limit_order(product_id, side, price, size):
    """Place a limit order on Coinbase."""
    # This is a placeholder function. You'll need to replace it with the actual function call from the Coinbase SDK.
    # Ensure you have the correct permissions and API keys setup for trading.
    if side == 'buy':
        # Place a buy limit order
        order_response = client.place_order(product_id=product_id, side='buy', order_type='limit', price=str(price), size=str(size))
    elif side == 'sell':
        # Place a sell limit order
        order_response = client.place_order(product_id=product_id, side='sell', order_type='limit', price=str(price), size=str(size))
    return order_response

def get_product_ids_which_match_first_part(product_ids, first_part):
    """Get product IDs which match the first part."""
    return [product_id for product_id in product_ids if product_id.startswith(first_part)]

def group_by_product_ids(product_ids):
    """Group product IDs by their first part."""
    first_parts = get_unique_first_part_of_product_ids(product_ids)
    return {first_part: get_product_ids_which_match_first_part(product_ids, first_part) for first_part in first_parts}


# Example usage
products = fetch_filtered_products()
product_ids = [product['product_id'] for product in products]
unique_first_parts = get_unique_first_part_of_product_ids(product_ids)
grouped_product_ids = group_by_product_ids(product_ids)
# remve group with only one product
grouped_product_ids = {k: v for k, v in grouped_product_ids.items() if len(v) > 1}
# filter out FORT and TRB
grouped_product_ids = {k: v for k, v in grouped_product_ids.items() if k not in ['FORT', 'T']}
# for each check if there is arbitrage
for k, v in grouped_product_ids.items():
    compare_prices_for_arbitrage(v)

# product_ids = ['BTC-USD', 'ETH-USD']  # Example product IDs
# compare_prices_for_arbitrage(product_ids)

