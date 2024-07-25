# Setup
```
export COINBASE_API_KEY="your_api_key"
export COINBASE_API_SECRET="your_api_secret"
```

# Run
```
python main.py --help
usage: main.py [-h] [--product_id PRODUCT_ID] [--find-cyclical-patterns] [--price-changes-for-interval] [--volume-changes-for-interval] [--previous-twenty-four-hours] [--since-beginning-of-day] [--rsi] [--macd] [--interval_time INTERVAL_TIME]
               [--start_date START_DATE] [--end_date END_DATE] [--calculate_correlation_interval_with_twenty_four] [--calculate_correlation_interval_with_twenty_four_volume] [--calculate_correlation_interval_with_composite]
               [--granularity {ONE_MINUTE,FIVE_MINUTE,FIFTEEN_MINUTE,ONE_HOUR,SIX_HOURS,ONE_DAY}] [--importdb] [--find-best-correlation] [--discover-abc-pattern] [--days_back DAYS_BACK] [--overlap-studies] [--momentum-indicators] [--volume-indicators]
               [--cycle-indicators] [--volatility-indicators] [--pattern-recognition] [--statistic-functions]

Crypto analysis tool.

options:
  -h, --help            show this help message and exit
  --product_id PRODUCT_ID
                        Product ID for the crypto asset
  --find-cyclical-patterns
                        Find cyclical patterns for the given product ID
  --price-changes-for-interval
                        Get price changes for a specific interval
  --volume-changes-for-interval
                        Get volume changes for a specific interval
  --previous-twenty-four-hours
                        Get the overall price change for the last 24 hours
  --since-beginning-of-day
                        Get the price change since the beginning of the specified day
  --rsi                 Calculate RSI for the given product ID up to the specified interval
  --macd                Calculate MACD for the given product ID up to the specified interval
  --interval_time INTERVAL_TIME
                        Interval time in YYYY-MM-DD HH:MM:SS format for calculations
  --start_date START_DATE
                        Start date in YYYY-MM-DD format
  --end_date END_DATE   End date in YYYY-MM-DD format
  --calculate_correlation_interval_with_twenty_four
                        Calculate correlation between price changes and indicators
  --calculate_correlation_interval_with_twenty_four_volume
                        Calculate correlation between price changes and indicators
  --calculate_correlation_interval_with_composite
                        Calculate correlation between price changes and indicators
  --granularity {ONE_MINUTE,FIVE_MINUTE,FIFTEEN_MINUTE,ONE_HOUR,SIX_HOURS,ONE_DAY}
                        Granularity of the data to fetch. Default is ONE_MINUTE.
  --importdb            Import fetched data into SQLite database
  --find-best-correlation
                        Find the best correlation between price changes and volume changes for the given product ID
  --discover-abc-pattern
                        Discover ABC pattern with volume and Fibonacci levels
  --days_back DAYS_BACK
                        Number of days to look back for the ABC pattern
  --overlap-studies     Calculate Overlap Studies indicators
  --momentum-indicators
                        Calculate Momentum Indicators
  --volume-indicators   Calculate Volume Indicators
  --cycle-indicators    Calculate Cycle Indicators
  --volatility-indicators
                        Calculate Volatility Indicators
  --pattern-recognition
                        Calculate Pattern Recognition indicators
  --statistic-functions
                        Calculate Statistic Functions



python main.py --product_id=SOL-USD --start_date=2024-01-01 --end_date=2024-07-31 --granularity FIFTEEN_MINUTE --importdb
python main.py --product_id=SOL-USD --find-best-correlation --start_date=2024-07-01 --end_date=2024-07-31
```

# Notes
- SQLite for storing data so we don't have to worry about rate limits
- Coinbase rest client for fetching data
- TA-Lib for technical analysis
- Pandas for data manipulation

