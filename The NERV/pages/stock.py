import pandas as pd 
import streamlit as st 
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
from firebase_admin import credentials, firestore
st.set_page_config(layout="wide", page_title="FinFlex", page_icon=":money_with_wings:")

options = pd.read_csv("E:/FinFlex/pages/stock.csv")["Stock"]

@st.cache_data
def recommendStock():
    user = st.session_state["user"]
    db = firestore.client() 
    doc_ref = db.collection("users").document(user)
    doc = doc_ref.get()
    features = ["RISK TOLERANCE","FINANCIAL KNOWLEDGE","INCOME LEVEL",
                "GOALS","STRATEGY","RISK MANAGEMENT SKILLS"]
    user_features = []
    for feature in features:
        user_features.append(doc.get(f'`{feature}`'))

    stock_features = pd.read_csv("E:/FinFlex/pages/stock.csv")[:100][["Stock","Beta","marketcap","trailingPE","forwardPE"]]

    client = OpenAI(api_key="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")

    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": 
             f'''You are a professional person financial assistant. Given the customer with features below:
             RISK TOLERANCE,FINANCIAL KNOWLEDGE,INCOME LEVEL,GOALS,STRATEGY,RISK MANAGEMENT SKILLS 
             which are all encoded in numeric values from 1-5 or 1-3, higher number implies that better except for
             STRATEGY which represents conservative to aggressive. Now the values of these features will be given
             in the same order as above: {user_features}.
             Now given some stock with these features Stock Symbol,Beta,Market Cap,trailing PE,forward PE as below:
             {stock_features}.
             Please recommend the top 3 stocks to the customer.
             Return the symbols of stock in such manner:
             "Stock 1"
             "Stock 2"
             "Stock 3"
             "Stock 4"
             "Stock 5"
             '''},
            {"role": "user", "content": '''
             Please recommend me five suitable stocks from the stock provided to you and list their symbols in such manner:
             "Stock 1"
             "Stock 2"
             "Stock 3"
             "Stock 4"
             "Stock 5"
             Do not explain any thing, just gives the symbols of the stocks.
             '''}
        ]
    )
    stocks = completion.choices[0].message.content
    stock = stocks.split("\n")
    for idx in range(len(stock)):
        stock[idx] = stock[idx].replace('"', "")
        y = yf.Ticker(stock[idx]).info
        stock[idx] += f"({y["shortName"]})"

    

    product1_name = stock[0].replace('"', "")
    product1_description = "Description of Product 1"
    product1_price = "$19.99"

    product2_name = stock[1].replace('"', "")
    product2_description = "Description of Product 2"
    product2_price = "$29.99"

    product3_name = stock[2].replace('"', "")
    product3_description = "Description of Product 3"
    product3_price = "$39.99"

    product4_name = stock[3].replace('"', "")
    product4_description = "Description of Product 3"
    product4_price = "$39.99"

    product5_name = stock[4].replace('"', "")
    product5_description = "Description of Product 3"
    product5_price = "$39.99"

    # Render the HTML code with Python variables
    st.markdown(f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Product Cards</title>
        <style>
            /* Your CSS styles here */
            * {{
                box-sizing: border-box;
            }}

            body {{
                font-family: Arial, Helvetica, sans-serif;
            }}

            /* Remove extra left and right margins, due to padding in columns */
            .row {{
                margin: 0 -5px;
            }}

            /* Clear floats after the columns */
            .row:after {{
                content: "";
                display: table;
                clear: both;
            }}

            /* Style the counter cards */
            .card {{
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); /* this adds the "card" effect */
                padding: 24px;
                text-align: center;
                background-color: #000000;
                border-radius: 5px; /* 5px rounded corners */
                margin-bottom: 20px; /* Add margin between cards */
            }}

            /* Responsive columns - one column layout (vertical) on small screens */
            @media screen and (max-width: 600px) {{
                .column {{
                    width: 100%;
                    display: block;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="column">
            <div class="card">
                <h3>{product1_name}</h3>
            </div>
            <div class="card">
                <h3>{product2_name}</h3>
            </div>
            <div class="card">
                <h3>{product3_name}</h3>
            </div>
            <div class="card">
                <h3>{product4_name}</h3>
            </div>
            <div class="card">
                <h3>{product5_name}</h3>
            </div>
        </div>
    </body>
    </html>
    ''', unsafe_allow_html=True)


def stock_app():
    st.sidebar.header(f"Hello {st.session_state['user']}")
    st.sidebar.page_link("pages/main.py", label="# JARVIS")
    st.sidebar.page_link("pages/stock.py", label="# Stock")
    st.sidebar.page_link("pages/advisor.py", label="# Advisor")

    # Setting Title
    st.title("Stock")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        # Setting Header
        st.subheader("Choose a stock and duration to see the graph")
        
        # Creating a dropdown
        symbol = st.selectbox("Select an option", options)

        # Taking Date Inputs
        startdate = str(st.date_input("Select a start date", date.today() - timedelta(7)))

        endtdate = str(st.date_input("Select an end date", date.today()))

        # Getting stock Data from yahoo finance
        tickerData = yf.Ticker(symbol)
        tickerDf = tickerData.history(period='1d', interval="1m", start=startdate, end=endtdate)
        submit = st.button("Get Graphs")
        # st.markdown(f"### OHLC and Volume of {symbol}")

        if submit:
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.update_layout(title_text=f"OHLC and Volume of {symbol}", 
                    title_font=dict(size=32, family='Arial'),)
            # include candlestick with rangeselector
            fig.add_trace(go.Candlestick(x=tickerDf.index,
                            open=tickerDf['Open'], high=tickerDf['High'],
                            low=tickerDf['Low'], close=tickerDf['Close'], name="OHLC"),
                        secondary_y=True)

            # include a go.Bar trace for volumes
            fig.add_trace(go.Bar(x=tickerDf.index, y=tickerDf['Volume'], name="Volume"),
                        secondary_y=False)

            fig.layout.yaxis2.showgrid=False

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        st.title("Stock Recommendation")
        recommendStock()


stock_app()



