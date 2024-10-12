import yfinance as yf
import pandas as pd

def get_stock_data(symbol, period):
    # استخدم yfinance لجلب البيانات بناءً على رمز العملة والمدة الزمنية
    stock_data = yf.download(symbol, period=period)
    return stock_data
