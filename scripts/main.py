from algotrading import AlgoTrading
import os

def main() -> None:
    # Set current working directory
    main_dir = os.path.dirname(os.path.abspath(__file__))

    algo = AlgoTrading(main_dir)

    # Run the selected task
    algo.run()

    return None
    
if __name__ == "__main__":
    main()