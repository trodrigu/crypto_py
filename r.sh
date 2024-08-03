RANDOM_FILENAME="backtest_results_$(date +%s).txt"
./backtest_hammer_products.sh | tee "$RANDOM_FILENAME"
python main.py --parse-backtest-hammer --filename "$RANDOM_FILENAME"

