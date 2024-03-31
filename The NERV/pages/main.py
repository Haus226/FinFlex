import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from io import StringIO
from openai import OpenAI

import yfinance as yf

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose  
from statsmodels.tsa.arima.model import ARIMA  
from statsmodels.tsa.statespace.sarimax import SARIMAX  
import matplotlib.pyplot as plt

import numpy as np
import plotly.express as px
st.set_page_config(layout="centered", page_title="FinFlex", page_icon=":money_with_wings:")



def main_app():
    client = OpenAI(api_key="OPENAI_API_KEY")
    st.sidebar.header(f"Hello {st.session_state['user']}")
    st.sidebar.page_link("pages/main.py", label="# JARVIS")
    st.sidebar.page_link("pages/stock.py", label="# Stock")
    st.sidebar.page_link("pages/advisor.py", label="# Advisor")

    st.title('FinFlex')
    st.markdown("## Your Finance _JARVIS_")
    tab1, tab3 = st.tabs(["Analyze", "Predict"])
    if "data" in st.session_state:
        df = st.session_state["data"]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.replace("Income/Salary", "Income")
    else:
        df = None

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append(
            {"role": "system", "content":
             '''
             You are a professional financial assistant, customer will provide his/her data in following format:
             DATE, TRANSACTION DETAILS, DESCRIPTION, PAYMENT METHOD, WITHDRAWAL AMT(EXPENSES), DEPOSIT AMT(INCOME)
             Please analyze the data given and answer the question asked.The more accurate the feedbacks or analysis the more tips you will gain.
             '''}
            )
        df_filtered = df.drop(columns=["TRANSACTION ID", "DATE", "DESCRIPTION"])
        
        d = df[["DATE", "TRANSACTION DETAILS", "DESCRIPTION", "PAYMENT METHOD", "WITHDRAWAL AMT", "DEPOSIT AMT"]]
        # mapping_category = {}
        mapping_payment = {}
        mapping_transactions = {}
        # for idx, category in enumerate(d["CATEGORY"].unique().tolist()):
        #     d.replace(category, idx)
        #     mapping_category[idx] = category
        for idx, payment in enumerate(d["PAYMENT METHOD"].unique().tolist()):
            d.replace(payment, idx)
            mapping_payment[idx] = payment
        for idx, transactions in enumerate(d["TRANSACTION DETAILS"].unique().tolist()):
            d.replace(transactions, idx)
            mapping_transactions[idx] = transactions
        d["DEPOSIT AMT"].fillna(0, inplace=True)
        d = d.to_string()
        
        # trans = df_filtered.groupby(["TRANSACTION DETAILS"])["WITHDRAWAL AMT"].sum().to_string()
        # category = df_filtered.groupby(["CATEGORY"])["WITHDRAWAL AMT"].sum().to_string()
        # daily = df.groupby(["DATE"])["WITHDRAWAL AMT"].sum().to_string()
        # payment = df_filtered.groupby(["PAYMENT METHOD"])["WITHDRAWAL AMT"].sum().to_string()
        st.session_state.messages.append(
            {"role": "user", "content":
            f'''
            Great to see you, I am worry about my financial spending and saving.The data is in detail as possible as I can, please retrieve the data carefully.
            '''
            }
            )

        st.session_state.messages.append(
            {"role": "user", "content":
            f'''
            Below are my DATE, TRANSACTION DETAILS, DESCRIPTION, CATEGORY, PAYMENT_METHOD, WITHDRAWAL AMT(EXPENSES), DEPOSIT AMT(INCOME)
            {d}
            Please use the following mapping relationships to recognize what the numeric values under the columns TRANSACTION DETAILS, and PAYMENT METHOD represent
            1)The mapping between integers and TRANSACTION DETAILS: {mapping_transactions}
            2)The mapping between integers and PAYMENT METHOD: {mapping_payment}
            Do not mention about any data in your response and remember the relationships without confusing yourself.Summarize your feedback as possible as you can, prevent any fancy stuffs.
            '''
            }
        )
        if prompt := st.chat_input(""):
            with st.chat_message("user"):
                st.markdown(prompt)
                st.session_state.messages.append({
                    "role":"user",
                    "content":prompt
                })

            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=True,
                )
                st.write_stream(stream)
                st.session_state.messages.pop()
                # st.session_state.messages.append({"role": "assistant", "content": response})

    with tab3:
        choices = df["CATEGORY"].unique().tolist()  + df["TRANSACTION DETAILS"].unique().tolist()
        categories = st.multiselect(
            "Which category would you like to predict?",
            choices,
            placeholder="Select your categories"
        )
        predict_button = st.button("Predict")
        if predict_button:
            if categories == ["Daily"]:
                fig = plt.figure()
                ex = pd.ExcelFile("UMH24 - FinTech Dataset.xlsx")
                df = pd.read_excel(ex, sheet_name=st.session_state["user"])
                df = df[df['CATEGORY'] != 'Income/Salary']
                s = df.groupby(["DATE"])["WITHDRAWAL AMT"].sum()

                s_ = s.diff(1).dropna()
                d = 0
                D = 0
                date = s.index
                category = "Daily"
                for idx in range(10):
                    adf_test = adfuller(s)
                    if adf_test[1] > 0.05:
                        s = s.diff()
                        d += 1
                    else:
                        break
                    adf_test = adfuller(s_)
                    if adf_test[1] > 0.05:
                        s_ = s_.diff()
                        D += 1
                    else:
                        break

                nlags = 15
                acf_vals = acf(s, nlags=nlags)
                pacf_vals = pacf(s, nlags=nlags)
                res = pd.DataFrame(np.array([acf_vals, pacf_vals]).reshape(nlags + 1, 2), columns=["ACF", "PACF"])

                n = len(s)
                threshold = 1.96 / np.sqrt(n)

                p = sum(abs(pacf_vals[1:]) > threshold)
                q = sum(abs(acf_vals[1:]) > threshold) 
                
                nlags = 15
                acf_vals = acf(s_, nlags=nlags)
                pacf_vals = pacf(s_, nlags=nlags)
                res = pd.DataFrame(np.array([acf_vals, pacf_vals]).reshape(nlags + 1, 2), columns=["ACF", "PACF"])

                n = len(s_)
                threshold = 1.96 / np.sqrt(n)

                P = sum(abs(pacf_vals[1:]) > threshold)
                Q = sum(abs(acf_vals[1:]) > threshold) 
                model = SARIMAX(s_, order=(p, d, q), seasonal_order=(P, D, Q, 30))
                
                res = model.fit(maxiter=15)
                forecast = res.get_forecast(steps=30)
                
                plt.plot(date, s.values, label=category)
                plt.plot(pd.date_range(start=date[-1], periods=30), 
                            forecast.predicted_mean, label=category + " Forecast")
                
            else:
                
                ex = pd.ExcelFile("UMH24 - FinTech Dataset.xlsx")
                df = pd.read_excel(ex, sheet_name=st.session_state["user"])
                df = df[df['CATEGORY'] != 'Income']
                new_date = pd.date_range(start=df["DATE"].min(), end=df["DATE"].max(), freq="D")
                
                fig = plt.figure()

                for category in categories:
                    d = 0
                    D = 0
                    if category in df["CATEGORY"].unique().tolist():
                        s = df.groupby(["DATE", "CATEGORY"])["WITHDRAWAL AMT"].sum()
                        s = s.reindex(pd.MultiIndex.from_product([new_date, s.index.levels[1]], names=['DATE', 'CATEGORY']), fill_value=0)
                        s = s.loc[s.index.get_level_values('CATEGORY') == category]

                        s_ = s.diff(12).dropna()
                        date = s.index.get_level_values('DATE')
                        
                        for idx in range(10):
                            print(category)
                            adf_test = adfuller(s)
                            if adf_test[1] > 0.05:
                                s = s.diff()
                                d += 1
                            else:
                                break
                            adf_test = adfuller(s_)
                            if adf_test[1] > 0.05:
                                s_ = s_.diff()
                                D += 1
                            else:
                                break
                        nlags = 15
                        acf_vals = acf(s, nlags=nlags)
                        pacf_vals = pacf(s, nlags=nlags)
                        res = pd.DataFrame(np.array([acf_vals, pacf_vals]).reshape(nlags + 1, 2), columns=["ACF", "PACF"])

                        n = len(s)
                        threshold = 1.96 / np.sqrt(n)

                        p = sum(abs(pacf_vals[1:]) > threshold)
                        q = sum(abs(acf_vals[1:]) > threshold) 

                        nlags = 15
                        acf_vals = acf(s_, nlags=nlags)
                        pacf_vals = pacf(s_, nlags=nlags)
                        res = pd.DataFrame(np.array([acf_vals, pacf_vals]).reshape(nlags + 1, 2), columns=["ACF", "PACF"])

                        n = len(s_)
                        threshold = 1.96 / np.sqrt(n)

                        P = sum(abs(pacf_vals[1:]) > threshold)
                        Q = sum(abs(acf_vals[1:]) > threshold) 

                        model = SARIMAX(s_, order=(p, d, q), seasonal_order=(P, D, Q, 31))
                        res = model.fit(maxiter=15)
                        forecast = res.get_forecast(steps=30)
                        plt.plot(date, s.values, label=category)
                        f = forecast.predicted_mean.apply(lambda x: max(0, x))

                        plt.plot(pd.date_range(start=date[-1], periods=30), 
                                 f, label=category + " Forecast")
                        
    
                    elif category in df["TRANSACTION DETAILS"].unique().tolist():
                        new_date = pd.date_range(start=df["DATE"].min(), end=df["DATE"].max(), freq="D")
                        t = df.groupby(["DATE", "TRANSACTION DETAILS"])["WITHDRAWAL AMT"].sum()
                        t = t.reindex(pd.MultiIndex.from_product([new_date, t.index.levels[1]], names=['DATE', 'TRANSACTION DETAILS']), fill_value=0)

                        t = t.loc[t.index.get_level_values('TRANSACTION DETAILS') == category]

                        t_ = t.diff(12).dropna()
                        date = t.index.get_level_values('DATE')
                        
                        for idx in range(10):
                            print(category)
                            adf_test = adfuller(t)
                            if adf_test[1] > 0.05:
                                t = t.diff()
                                d += 1
                            else:
                                break
                            adf_test = adfuller(t_)
                            if adf_test[1] > 0.05:
                                t_ = t_.diff()
                                D += 1
                            else:
                                break
                        nlags = 15
                        acf_vals = acf(t, nlags=nlags)
                        pacf_vals = pacf(t, nlags=nlags)
                        res = pd.DataFrame(np.array([acf_vals, pacf_vals]).reshape(nlags + 1, 2), columns=["ACF", "PACF"])

                        n = len(t)
                        threshold = 1.96 / np.sqrt(n)

                        p = sum(abs(pacf_vals[1:]) > threshold)
                        q = sum(abs(acf_vals[1:]) > threshold) 

                        nlags = 15
                        acf_vals = acf(t_, nlags=nlags)
                        pacf_vals = pacf(t_, nlags=nlags)
                        res = pd.DataFrame(np.array([acf_vals, pacf_vals]).reshape(nlags + 1, 2), columns=["ACF", "PACF"])

                        n = len(t_)
                        threshold = 1.96 / np.sqrt(n)

                        P = sum(abs(pacf_vals[1:]) > threshold)
                        Q = sum(abs(acf_vals[1:]) > threshold) 

                        model = SARIMAX(t_, order=(p, d, q), seasonal_order=(P, D, Q, 31))
                        res = model.fit()
                        forecast = res.get_forecast(steps=30)
                        f = forecast.predicted_mean.apply(lambda x: max(0, x))
                        plt.plot(date, t.values, label=category)
                        plt.plot(pd.date_range(start=date[-1], periods=30), 
                                 f, label=category + " Forecast")
                plt.legend()
                plt.grid()
            st.pyplot(fig)



main_app()

