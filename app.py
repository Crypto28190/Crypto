from flask import Flask, request, jsonify
from crypto import get_stock_data
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import numpy as np

app = Flask(__name__)

@app.route('/results', methods=['POST'])
def results():
    # تحديد رمز العملة الافتراضي (على سبيل المثال Bitcoin مقابل الدولار الأمريكي)
    symbol = 'BTC-USD'
    
    # قراءة المدة الزمنية من الطلب (على سبيل المثال '2mo' لشهرين أو '7d' لأسبوع)
    period = request.form['period']
    
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

    # إرجاع النتيجة كـ JSON
    result = {
        "symbol": symbol,
        "current_price": float(current_price),
        "forecast_arima": float(forecast_arima),
        "forecast_xgboost": float(forecast_xgboost)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=3003)
