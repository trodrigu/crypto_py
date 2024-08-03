RANDOM_FILENAME="backtest_results_$(date +%s).txt"
./backtest_hammer_products.sh | tee "$RANDOM_FILENAME"
/usr/local/bin/python3 main.py --parse-backtest-hammer --filename "$RANDOM_FILENAME"

