import pandas as pd 
import streamlit as st 
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px 
from firebase_admin import firestore
from openai import OpenAI


st.set_page_config(layout="wide", page_title="FinFlex", page_icon=":money_with_wings:")


@st.cache_data
def recommendAdvisor():
    user = st.session_state["user"]
    db = firestore.client() 
    doc_ref = db.collection("users").document(user)
    doc = doc_ref.get()
    features = ["RISK TOLERANCE","FINANCIAL KNOWLEDGE","INCOME LEVEL",
                "GOALS","STRATEGY","RISK MANAGEMENT SKILLS"]
    user_features = []
    for feature in features:
        user_features.append(doc.get(f'`{feature}`'))

    stock_features = pd.read_csv("E:/FinFlex/pages/advisor.csv")[["NAME","RETURN ON INVESTMENT (ROI)","CLIENT RETENTION RATE","ASSET UNDER MANAGEMENT (AUM)","PORTFOLIO PERFORMANCE METRICS","FEE STRUCTURE AND REVENUE METRICS"]]

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
             Now given some financial advisor with these features NAME,RETURN ON INVESTMENT (ROI),CLIENT RETENTION RATE,ASSET UNDER MANAGEMENT (AUM),PORTFOLIO PERFORMANCE METRICS,FEE STRUCTURE AND REVENUE METRICS as below:
             {stock_features}.
             where each features are explained as following:
             Return on Investment (ROI):

            Assign a numerical scale to ROI percentages. For instance:
            Excellent ROI: 5
            Good ROI: 4
            Average ROI: 3
            Below Average ROI: 2
            Poor ROI: 1
            Client Retention Rate:

            Use a percentage scale to represent the retention rate:
            90% and above: 5
            80% - 89%: 4
            70% - 79%: 3
            60% - 69%: 2
            Below 60%: 1
            Asset Under Management (AUM):

            Encode AUM into different brackets or ranges and assign corresponding values:
            High AUM: 5
            Medium-High AUM: 4
            Medium AUM: 3
            Medium-Low AUM: 2
            Low AUM: 1
            Portfolio Performance Metrics:

            Define thresholds for performance metrics like Sharpe Ratio, Standard Deviation, and Tracking Error, and assign scores accordingly:
            Excellent Performance: 5
            Good Performance: 4
            Average Performance: 3
            Below Average Performance: 2
            Poor Performance: 1
            Fee Structure and Revenue Metrics:

            Use a scale to represent fee revenue, fee-to-AUM ratio, and fee transparency:
            High Revenue and Transparency, Low Fee-to-AUM Ratio: 5
            Moderate Revenue and Transparency, Moderate Fee-to-AUM Ratio: 3
            Low Revenue and Transparency, High Fee-to-AUM Ratio: 1

             Please recommend the top 5 advisors to the customer.
             Return the name of advisors in such manner:
             "Advisor 1"
             "Advisor 2"
             "Advisor 3"
             "Advisor 4"
             "Advisor 5"
             '''},
            {"role": "user", "content": '''
             Please recommend me 5 suitable advisors and list their names in such manner:
             "Advisor 1"
             "Advisor 2"
             "Advisor 3"
             "Advisor 4"
             "Advisor 5"
             Do not explain any thing, just gives the name of the advisors.
             '''}
        ]
    )
    stocks = completion.choices[0].message.content
    stock = stocks.split("\n")
    for idx in range(len(stock)):
        stock[idx] = stock[idx].replace('"', "")


    

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


def advisor_app():
    st.sidebar.header(f"Hello {st.session_state['user']}")
    st.sidebar.page_link("pages/main.py", label="# JARVIS")
    st.sidebar.page_link("pages/stock.py", label="# Stock")
    st.sidebar.page_link("pages/advisor.py", label="# Advisor")



    # Setting Title
    st.title("Match with your Advisor")
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        # Setting Header
        st.subheader("Your Financial Summary")

        df = st.session_state["data"]
        path = ["PAYMENT METHOD", "CATEGORY", "TRANSACTION DETAILS"]
        # Fill null value
        for p in path:
            df[p].fillna("No Stated", inplace=True)
        # Strip the strings
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df[df['CATEGORY'] != 'Income/Salary']
        fig = px.sunburst(df, path=path, values=df["WITHDRAWAL AMT"], width=700, height=700)
        fig.update_layout(
                          uniformtext=dict(minsize=12, mode='hide'))
        fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
        fig.update_traces(insidetextorientation='radial')
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with col2:
        st.title("Advisor Recommendation")
        recommendAdvisor()

advisor_app()



