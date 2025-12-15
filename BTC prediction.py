import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

url = "https://finance.yahoo.com/quote/BTC-USD/history/"
headers = {'User-Agent': 'Mozilla/5.0'}

def get_data():   # This function scrape the data from yahoo finance and return dataframe
    result = requests.get(url, headers=headers)
    soup = BeautifulSoup(result.text, "html.parser")
    table = soup.find('table')
    
    data = []
    if table:
        tbody = table.find("tbody")
        rows = tbody.find_all("tr")
        
        for row in rows:
            columns = row.find_all("td")
            if len(columns) >= 5: 
                date = columns[0].text.strip()
                close = columns[4].text.strip()
                data.append([date, close])
                
    df = pd.DataFrame(data, columns=['Date', 'Close'])
    return df

def clean_data(df):   # This function remove comma and convert string to number
    # Remove comma from price (e.g. 98,000 to 98000)
    df['Close'] = df['Close'].str.replace(',', '').astype(float)
    
    # Yahoo gives data newest first, we need oldest first for ARIMA
    df = df.iloc[::-1].reset_index(drop=True) 
    return df

def predict_price(df):   # This function use ARIMA model to predict next 20 days
    train_data = df['Close']
    
    # Train the model (Order 5,1,0 is standard for this)
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Predict 20 steps ahead
    forecast = model_fit.forecast(steps=20)
    return forecast

def save_and_show(df, forecast):   # This function save data to csv and show the plot
    # Save to CSV
    df.to_csv("btc_prices.csv", index=False)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='History')
    
    # Make X-axis for prediction (continue from history)
    last_index = df.index[-1]
    forecast_index = range(last_index + 1, last_index + 21)
    
    plt.plot(forecast_index, forecast, label='Prediction', color='red')
    plt.legend()
    plt.title("Bitcoin Price Prediction")
    
    # Save plot as image
    plt.savefig("btc_plot.png")
    print("Plot saved and data saved.")
    plt.show()

# Main program execution
if __name__ == "__main__":
    df = get_data()
    if not df.empty:
        df = clean_data(df)
        forecast = predict_price(df)
        save_and_show(df, forecast)
    else:
        print("No data found")