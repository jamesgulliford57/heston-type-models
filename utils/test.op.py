import yfinance as yf
stock = yf.Ticker("MSFT")
print(stock.options)