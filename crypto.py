import yfinance as yf
import pandas as pd

def get_stock_data(symbol, period):
    try:
        # استخدم yfinance لجلب البيانات بناءً على رمز العملة والمدة الزمنية
        stock_data = yf.download(symbol, period=period)

        # التحقق من أن البيانات ليست فارغة
        if stock_data.empty:
            raise ValueError("No data found for the given symbol and period.")
        
        # حذف أي قيم مفقودة في بيانات الإغلاق
        stock_data = stock_data.dropna(subset=['Close'])

        return stock_data
    except Exception as e:
        return {"error": str(e)}
