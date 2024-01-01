
## Introduction
Welcome to Alpaca RL Trader, a reinforcement learning-based trading system that interfaces with Alpaca's Paper Trading API for a simulated trading environment. This system also utilizes historical stock data from the Polygon API.

## System Requirements
- Python 3.11

## Installation Steps

1. **Install Python 3.11**  
   Ensure Python 3.11 is installed on your system.

2. **Create a Virtual Environment**  
   `python -m venv venv`

3. **Activate the Virtual Environment**  
   - Windows: `venv\Scripts\activate`  
   - Linux/MacOS: `source venv/bin/activate`

4. **Install Dependencies**  
   `pip install -r requirements.txt`

5. **Environment Variables**  
   Create a `.env` file in the root directory and add your API keys:

    APCA_API_KEY_ID=your_alpaca_api_key_id 
    APCA_API_SECRET_KEY=your_alpaca_api_secret_key 
    POLYGON_API_KEY=your_polygon_api_key

6. **Download some data**  
   Run `python data/historical_data.py` to download historical data from the Polygon API and store it in the SQL DB.

7. **Run the Program**  
- `python train.py`: Trains the model, creates a `.pth` file in the models directory, and logs to the log directory.
- `python test.py`: Tests the model.
- `python live.py`: Runs the model live, trading on the Alpaca paper trading account.

## Project Structure

- **`env/`** - Contains the environment for the reinforcement learning model.
- **`notebook/`** - Jupyter Notebooks for data analysis and testing ideas.
- **`util/`** - Utility functions such as trading API helpers, neural network, and callback functions.
- **`data/`** - Contains `data_loader.py` for loading data from SQL DB and creating test splits and features. `historical_data.py` downloads historical data from Polygon API and stores it in the SQL DB.

## Usage

- **Training**: Use `train.py` to train your trading model.
- **Testing**: Use `test.py` to evaluate your model's performance.
- **Live Trading**: Use `live.py` for live trading simulations.

## Notebooks
Explore our Jupyter Notebooks in the `notebook/` directory for in-depth data analysis and experimentation.

## Contributing
Contributions to Alpaca RL Trader are welcome. Please read our contribution guidelines for more information.
