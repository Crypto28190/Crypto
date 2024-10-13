@app.route('/results', methods=['POST'])
def results():

    # قراءة المدخلات من الطلب
    period = request.form.get('period', '1mo')  # إذا لم يتم تحديد المدة، افترض '1mo'
    model_type = request.form.get('model', 'arima')  # افتراض استخدام arima إذا لم يتم تحديد النموذج
    
    # التحقق من صحة المدخلات
    if model_type not in ['arima', 'xgboost', 'lstm']:
        return jsonify({"error": "Invalid model type. Choose from 'arima', 'xgboost', 'lstm'"}), 400

    if period not in ['1mo', '3mo', '6mo', '1y', '2y']:
        return jsonify({"error": "Invalid period. Choose from '1mo', '3mo', '6mo', '1y', '2y'"}), 400

    results = {}
    
    # عملية التنبؤ بناءً على المدخلات
    for symbol in supported_symbols:
        try:
            # جلب البيانات بناءً على الفترة المحددة
            stock_data = get_stock_data(symbol, period)
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
