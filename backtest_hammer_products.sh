if [ -f crypto_data.db ]; then
    rm crypto_data.db
fi

/usr/local/bin/python3 main.py --product_id=SOL-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id SOL-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db
 
/usr/local/bin/python3 main.py --product_id=SUI-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id SUI-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=ETH-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id ETH-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=BTC-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id BTC-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=DASH-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id DASH-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=DOGE-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id DOGE-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=BONK-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id BONK-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=RNDR-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id RNDR-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=AERO-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id AERO-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=CVX-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id CVX-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=MATIC-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id MATIC-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db

/usr/local/bin/python3 main.py --product_id=OP-USD --start_date=2024-07-01 --end_date=2024-08-01 --granularity FIFTEEN_MINUTE --importdb
/usr/local/bin/python3 main.py --product_id OP-USD --backtest-hammer --start_date 2024-07-01 --end_date 2024-08-01 --input 4621899
rm crypto_data.db
