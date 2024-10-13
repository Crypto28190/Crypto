# app.py
from flask import Flask, request, jsonify
from crypto import get_stock_data
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# قائمة بأشهر 10 عملات رقمية (بما في ذلك Baby Doge Coin)
supported_symbols = [
    'BTC-USD',  # Bitcoin
    'ETH-USD',  # Ethereum
    'BNB-USD',  # Binance Coin
    'BABYDOGE-USD', # Baby Doge Coin
    'XRP-USD',  # Ripple
    'ADA-USD',  # Cardano
    'SOL-USD',  # Solana
    'DOGE-USD', # Dogecoin
    'DOT-USD',  # Polkadot
    'LTC-USD'   # Litecoin
]

@app.route('/results', methods=['POST'])
def results():
    # ... (بقية الكود كما هو)
    
    # عملية التنبؤ بناءً على المدخلات
    for symbol in supported_symbols:
        try:
            # جلب البيانات بناءً على الفترة المحددة
            stock_data = get_stock_data(symbol, period)
            
            # إذا كان هناك خطأ في جلب البيانات
            if 'error' in stock_data:
                results[symbol] = {"error": stock_data['error']}
                continue

            close_prices = stock_data['Close']

            # التحقق من وجود قيم مفقودة
            if close_prices.isnull().any():
                close_prices = close_prices.dropna()

            # إذا كانت البيانات غير كافية، تجاهل العملة
            if len(close_prices) < 10:
                results[symbol] = {"error": "Insufficient data"}
                continue
            
            # التنبؤ باستخدام النموذج المختار
            train_data = close_prices
            
            # -------- ARIMA Model --------
            if model_type == 'arima':
                model1 = ARIMA(train_data, order=(5, 1, 0))  
                fitted_model1 = model1.fit()
                forecast_arima = fitted_model1.forecast(steps=1)[0]
                results[symbol] = {
                    "current_price": float(close_prices.iloc[-1]),
                    "forecast_arima": float(forecast_arima)
                }

            # -------- XGBoost Model --------
            elif model_type == 'xgboost':
                X = np.arange(len(train_data)).reshape(-1, 1)
                y = train_data.values

                model2 = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                model2.fit(X, y)
                next_index = np.array([[len(train_data)]])
                forecast_xgboost = model2.predict(next_index)[0]
                results[symbol] = {
                    "current_price": float(close_prices.iloc[-1]),
                    "forecast_xgboost": float(forecast_xgboost)
                }

            # -------- LSTM Model --------
            elif model_type == 'lstm':
                # Normalization
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

                # Reshaping data for LSTM
                X_lstm, y_lstm = [], []
                for i in range(10, len(train_data_scaled)):
                    X_lstm.append(train_data_scaled[i-10:i, 0])
                    y_lstm.append(train_data_scaled[i, 0])
                
                X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

                # LSTM model architecture
                lstm_model = tf.keras.Sequential()
                lstm_model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
                lstm_model.add(tf.keras.layers.LSTM(units=50, return_sequences=False))
                lstm_model.add(tf.keras.layers.Dense(units=25))
                lstm_model.add(tf.keras.layers.Dense(units=1))

                # Compile and fit LSTM model
                lstm_model.compile(optimizer='adam', loss='mean_squared_error')
                lstm_model.fit(X_lstm, y_lstm, epochs=1, batch_size=1, verbose=2)

                # Preparing the last data for prediction
                last_10_days = train_data_scaled[-10:]
                X_test = np.reshape(last_10_days, (1, last_10_days.shape[0], 1))

                # Forecast with LSTM
                forecast_lstm = lstm_model.predict(X_test)
                forecast_lstm = scaler.inverse_transform(forecast_lstm)[0][0]

                results[symbol] = {
                    "current_price": float(close_prices.iloc[-1]),
                    "forecast_lstm": float(forecast_lstm)
                }
        
        except Exception as e:
            # إذا حدث خطأ، سجل رسالة الخطأ للعملة المحددة
            results[symbol] = {"error": str(e)}

    # إرجاع جميع النتائج كـ JSON
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
