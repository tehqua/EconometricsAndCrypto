import streamlit as st
import requests
import plotly.graph_objs as go
import pandas as pd
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima.utils import ndiffs


def get_crypto_prices():
    url = "https://api.binance.com/api/v3/ticker/price"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_crypto_history(symbol, interval='1d', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def crypto_currency_overview():
    st.title("Crypto Price Viewer")

    crypto_prices = get_crypto_prices()
    crypto_symbols = [crypto['symbol'] for crypto in crypto_prices]

    selected_crypto = st.selectbox("Select Cryptocurrency", crypto_symbols)

    if selected_crypto:
        st.write(f"### Real-time Prices for {selected_crypto}")
        st.write("Price (USDT)")
        for crypto in crypto_prices:
            if crypto['symbol'] == selected_crypto:
                st.write(crypto['price'])

        crypto_history = get_crypto_history(selected_crypto)
        if crypto_history:
            df = pd.DataFrame(crypto_history, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            
            if df.isnull().sum().any():
                st.warning("Missing values detected! Filling with mean values.")
                df.fillna(df.mean(), inplace=True)

            
            if df['close'].dtype == 'object':
                df['close'] = pd.to_numeric(df['close'], errors='coerce')

            
            if len(df) >= 100:
                
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['taker_buy_base_asset_volume'] = pd.to_numeric(df['taker_buy_base_asset_volume'], errors='coerce')

                st.subheader("Price Chart")
                
                model, forecast_mean, forecast_conf_int = sarima_forecast(df['close'])

                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

               
                fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)
                fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)

               
                fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(dash='dash')), row=1, col=1)

                
                fig.add_trace(go.Scatter(x=forecast_conf_int.index,
                                         y=forecast_conf_int.iloc[:, 0],
                                         mode='lines',
                                         line=dict(color='grey'),
                                         showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=forecast_conf_int.index,
                                         y=forecast_conf_int.iloc[:, 1],
                                         mode='lines',
                                         line=dict(color='grey'),
                                         fill='tonexty',
                                         fillcolor='rgba(0,100,80,0.2)',
                                         showlegend=False), row=1, col=1)

                
                volume_color = ['green' if buy_volume > sell_volume else 'red' for buy_volume, sell_volume in zip(df['taker_buy_base_asset_volume'], df['volume'] - df['taker_buy_base_asset_volume'])]
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=volume_color), row=2, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)

                fig.update_layout(title=f"Price and Volume Chart for {selected_crypto}", height=600)

                st.plotly_chart(fig)
            else:
                st.error("Insufficient data for SARIMA analysis.")


def get_crypto_prices():
    url = "https://api.binance.com/api/v3/ticker/price"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_crypto_history(symbol, interval='1d', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def sarima_forecast(series, forecast_period=30, alpha=0.05):
    
    d = ndiffs(series, test='adf')
    model = SARIMAX(series, order=(1, d, 1), seasonal_order=(1, d, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=forecast_period)
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int(alpha=alpha)
    
    return model_fit, forecast_mean, forecast_conf_int


def econometrics():
    st.title("Econometrics")

    crypto_prices = get_crypto_prices()
    if crypto_prices:
        crypto_symbols = [crypto['symbol'] for crypto in crypto_prices]

        selected_crypto = st.selectbox("Select Cryptocurrency", crypto_symbols)

        if selected_crypto:
            st.write(f"### Real-time Prices for {selected_crypto}")
            st.write("Price (USDT)")
            for crypto in crypto_prices:
                if crypto['symbol'] == selected_crypto:
                    st.write(crypto['price'])

            crypto_history = get_crypto_history(selected_crypto)
            if crypto_history:
                df = pd.DataFrame(crypto_history, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                
                if df.isnull().sum().any():
                    st.warning("Missing values detected! Filling with mean values.")
                    df.fillna(df.mean(), inplace=True)

               
                df['close'] = df['close'].astype(float)

                
                if len(df) >= 100:  
                    st.subheader("Econometrics")
                    # Get SARIMA forecast
                    model, forecast_mean, forecast_conf_int = sarima_forecast(df['close'])

                    # Plot historical data and forecast
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

                    # Add price trace to subplot 1
                    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Price'), row=1, col=1)
                    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)

                    # Add volume trace to subplot 2
                    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='blue'), row=2, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)

                    # Add forecast mean to subplot 1
                    fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, mode='lines', name='Forecast', line=dict(dash='dash')), row=1, col=1)

                    # Add prediction interval to subplot 1
                    fig.add_trace(go.Scatter(x=forecast_conf_int.index,
                                             y=forecast_conf_int.iloc[:, 0],
                                             mode='lines',
                                             line=dict(color='grey'),
                                             showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=forecast_conf_int.index,
                                             y=forecast_conf_int.iloc[:, 1],
                                             mode='lines',
                                             line=dict(color='grey'),
                                             fill='tonexty',
                                             fillcolor='rgba(0,100,80,0.2)',
                                             showlegend=False), row=1, col=1)

                    fig.update_layout(title=f"Price and Volume Chart for {selected_crypto}", height=600)

                    st.plotly_chart(fig)
                    # Display model summary
                    st.subheader("Model Summary")
                    st.write(model.summary())
                else:
                    st.info("Insufficient data for econometrics analysis. Skipping...")

            else:
                st.error("Failed to fetch cryptocurrency price history.")
    else:
        st.error("Failed to fetch cryptocurrency prices.")

def main():
    st.title("Crypto Analysis App")
    page = st.sidebar.selectbox("Select Page", ["Crypto Currency Overview", "Econometrics"])
    if page == "Crypto Currency Overview":
        crypto_currency_overview()
    elif page == "Econometrics":
        econometrics()

if __name__ == "__main__":
    main()
