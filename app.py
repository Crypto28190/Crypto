from flask import Flask, request, jsonify
from crypto import get_stock_data
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import numpy as np
import os

app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# قائمة رموز العملات المدعومة
supported_symbols = [
    'BTC-USD',  # Bitcoin
    'ETH-USD',  # Ethereum
    'LTC-USD',  # Litecoin
    'XRP-USD',  # Ripple
    'BCH-USD',  # Bitcoin Cash
    'LINK-USD', # Chainlink
    'DOT-USD',  # Polkadot
    'ADA-USD',  # Cardano
    'SOL-USD',  # Solana
    'DOGE-USD'  # Dogecoin
]

@app.route('/results', methods=['POST'])
def results():
    # قراءة المدة الزمنية من الطلب (مثل '1mo' أو '2w')
    period = request.form.get('period', '3mo')  # إذا لم يتم تحديد المدة، افترض '1mo'
    
    results = {}
    
    for symbol in supported_symbols:
        try:
            # جلب البيانات بناءً على الفترة المحددة
            stock_data = get_stock_data(symbol, period)
            close_prices = stock_data['Close']
            
            # استخدام جميع البيانات للتنبؤ
            train_data = close_prices

            # نموذج ARIMA
            model1 = ARIMA(train_data, order=(5, 1, 0))  
            fitted_model1 = model1.fit()

            # نموذج XGBoost
            X = np.arange(len(train_data)).reshape(-1, 1)
            y = train_data.values

            model2 = XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model2.fit(X, y)

            # توقع السعر لنهاية اليوم الحالي باستخدام ARIMA و XGBoost
            forecast_arima = fitted_model1.forecast(steps=1)[0]
            next_index = np.array([[len(train_data)]])  # الخطوة التالية في X
            forecast_xgboost = model2.predict(next_index)[0]
            
            # السعر الحالي
            current_price = close_prices.iloc[-1]

            # تخزين النتائج لكل عملة
            results[symbol] = {
                "current_price": float(current_price),
                "forecast_arima": float(forecast_arima),
                "forecast_xgboost": float(forecast_xgboost)
            }
        
        except Exception as e:
            # إذا حدث خطأ، سجل رسالة الخطأ للعملة المحددة
            results[symbol] = {"error": str(e)}

    # إرجاع جميع النتائج كـ JSON
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
