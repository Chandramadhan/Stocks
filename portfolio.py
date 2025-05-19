import streamlit as st
import yfinance as yf
import pandas as pd
import ta.momentum
import matplotlib.pyplot as plt
from io import BytesIO

# Nifty 50 tickers list (can be expanded)
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "BAJFINANCE.NS",
    "AXISBANK.NS", "MARUTI.NS", "HDFC.NS", "LT.NS", "NESTLEIND.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "DIVISLAB.NS", "POWERGRID.NS", "TECHM.NS", "ONGC.NS",
    "JSWSTEEL.NS", "BAJAJFINSV.NS", "EICHERMOT.NS", "HCLTECH.NS", "TATASTEEL.NS", "DRREDDY.NS"
]

# --- Helper functions ---

def calc_rsi(data):
    close_prices = data['Close']
    if len(close_prices.shape) > 1:
        close_prices = close_prices.squeeze()
    rsi_indicator = ta.momentum.RSIIndicator(close_prices)
    return rsi_indicator.rsi().iloc[-1]  # latest RSI value

def fetch_data(ticker):
    data = yf.download(ticker, period='6mo', progress=False)
    return data

def get_latest_price(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.history(period='1d')['Close'].iloc[-1]
        return price
    except Exception:
        return None

def calc_portfolio_metrics(holdings):
    data = []
    total_value = 0
    for ticker, qty in holdings.items():
        price = get_latest_price(ticker)
        if price is None:
            market_value = None
        else:
            market_value = price * qty
            total_value += market_value

        # Get RSI
        stock_data = fetch_data(ticker)
        rsi = calc_rsi(stock_data) if not stock_data.empty else None

        # Status based on RSI
        if rsi is None:
            status = "No Data"
        elif rsi < 30:
            status = "OVERSOLD (BUY)"
        elif rsi > 70:
            status = "OVERBOUGHT (SELL)"
        else:
            status = "HOLD"

        data.append({
            "Ticker": ticker,
            "Quantity": qty,
            "Price (₹)": round(price, 2) if price else "N/A",
            "Market Value (₹)": round(market_value, 2) if market_value else "N/A",
            "RSI": round(rsi, 2) if rsi else "N/A",
            "Status": status
        })
    return pd.DataFrame(data), total_value

def plot_portfolio_weights(holdings, prices):
    labels = []
    sizes = []
    for ticker, qty in holdings.items():
        price = prices.get(ticker, 0)
        if price is not None:
            labels.append(ticker)
            sizes.append(price * qty)
    if sizes:
        plt.figure(figsize=(7,7))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Portfolio Allocation')
        st.pyplot(plt)
    else:
        st.write("No valid data to plot pie chart.")

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Portfolio')
    writer.close()  # <-- use close() instead of save()
    processed_data = output.getvalue()
    return processed_data

def project_wealth(monthly_investment, years, annual_return=0.12):
    # Compound monthly investment projection
    months = years * 12
    monthly_rate = (1 + annual_return) ** (1/12) - 1
    future_value = monthly_investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
    return future_value

# --- Streamlit UI ---

st.title("🔥 Brutal Portfolio Dashboard with Full Features")

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}

menu = st.sidebar.selectbox("Select Option", ["Select Holdings", "View Portfolio", "RSI Scan", "Investment Projection"])

if menu == "Select Holdings":
    st.header("Select Holdings & Quantities")

    selected = st.multiselect("Select tickers from Nifty 50", NIFTY_50_TICKERS,
                              default=list(st.session_state['portfolio'].keys()))
    
    new_portfolio = {}
    for ticker in selected:
        qty = st.number_input(f"Quantity for {ticker}", min_value=1, step=1,
                              value=st.session_state['portfolio'].get(ticker, 1), key=ticker)
        new_portfolio[ticker] = qty
    
    if st.button("Save Portfolio"):
        st.session_state['portfolio'] = new_portfolio
        st.success(f"Portfolio updated with {len(new_portfolio)} holdings")

elif menu == "View Portfolio":
    st.header("Your Portfolio Overview")
    if not st.session_state['portfolio']:
        st.warning("No holdings selected yet.")
    else:
        df, total_value = calc_portfolio_metrics(st.session_state['portfolio'])
        st.write(f"### Total Portfolio Value: ₹{total_value:,.2f}")
        st.dataframe(
            df.style.apply(
                lambda x: ['background-color: #ffcccc' if v == 'OVERSOLD (BUY)' else '' for v in x], subset=['Status']
            )
        )
        prices = {row['Ticker']: float(row['Price (₹)']) if row['Price (₹)'] != "N/A" else 0 for _, row in df.iterrows()}
        plot_portfolio_weights(st.session_state['portfolio'], prices)

        # Export buttons
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name='portfolio.csv', mime='text/csv')

        excel_data = to_excel(df)
        st.download_button(label="Download Excel", data=excel_data, file_name='portfolio.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

elif menu == "RSI Scan":
    st.header("RSI Scan for Your Portfolio")
    if not st.session_state['portfolio']:
        st.warning("No holdings selected yet.")
    else:
        df, total_value = calc_portfolio_metrics(st.session_state['portfolio'])
        st.dataframe(
            df[['Ticker', 'RSI', 'Status']].style.apply(
                lambda x: ['background-color: #ffcccc' if v == 'OVERSOLD (BUY)' else '' for v in x], subset=['Status']
            )
        )

elif menu == "Investment Projection":
    st.header("Monthly SIP Projection to ₹1.5 Crore Target")

    current_age = st.number_input("Your Current Age", min_value=18, max_value=50, value=25)
    target_age = 50
    years_left = target_age - current_age

    monthly_sip = st.number_input("Monthly Investment Amount (₹)", min_value=100, max_value=100000, value=5000, step=500)

    expected_annual_return = st.slider("Expected Annual Return (%)", min_value=5, max_value=20, value=12, step=1) / 100

    future_value = project_wealth(monthly_sip, years_left, expected_annual_return)
    st.write(f"At age {target_age}, your investment could grow to approximately **₹{future_value:,.0f}**.")

    if future_value >= 1.5e7:
        st.success("You are on track to reach your ₹1.5 Crore goal! 🎉")
    else:
        diff = 1.5e7 - future_value
        extra_per_month = diff / (years_left * 12)
        st.warning(f"You might need to increase your monthly investment by approx ₹{extra_per_month:,.0f} to reach ₹1.5 Crore.")

