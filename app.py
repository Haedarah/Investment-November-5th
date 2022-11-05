import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
import time
import altair as alt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

mapping={
"Dollar":"INR=X",
"Euro":"EURINR=X", 
"Bitcoin":"BTC-INR",
"Ethereum":"ETH-INR", 
"Infosys":"INFY.NS",
"TATA Steel":"TATASTEEL.NS",
"ICICI Bank":"ICICIBANK.NS",
"State Bank of India":"SBIN.NS",
"Reliance Industries limited":"RELIANCE.NS"
}

times={
"3 Months":90,
"6 Months":180,
"1 Year":365
}

risk={
"Dollar":-1,
"Euro":-1, 
"Bitcoin":-1,
"Ethereum":-1, 
"Infosys":0,
"TATA Steel":1,
"ICICI Bank":0,
"State Bank of India":0,
"Reliance Industries limited":-1
}

periods={
"1 Day":"1d",
"5 Days":"5d",
"1 Month":"1mo",
"3 Months":"3mo",
"6 Months":"6mo",
"1 Year":"1y",
"2 Years":"2y",
"5 Years":"5y",
"10 Years":"10y",
"Max":"max"
}

intervals={
"1 Minute":"1m",
"2 Minutes":"2m",
"5 Minutes":"5m",
"15 Minutes":"15m",
"30 Minutes":"30m",
"60 Minutes":"60m",
"90 Minutes":"90m",
"1 Day":"1d",
"5 Days":"5d",
"1 Week":"1wk",
"1 Month":"1mo",
"3 Months":"3mo"
}

st.set_page_config(
	page_icon="chart_with_upwards_trend",
    page_title="Investment Dashboard",
    layout="wide",
)


header=st.container()
st.write("""---""")
UpperBlock=st.container()
st.write("""---""")
MiddleBlock=st.container()
LowerBlock=st.container()
st.write("""---""")
footer=st.container()





with header:
	col1,col2=st.columns([1,4],gap="large")

	with col1:
		st.image("Objects/LOGO1.png",use_column_width=True)

	with col2:
		st.title("Indian Investment Dashboard - Beta Version")
		st.write("Prototype of an Indian Investment Dashboard.")




with UpperBlock:
	
	col1, col2=st.columns([8,3],gap="large")
	with col1:
		display=st.selectbox("Select a stock:",mapping.keys())
		st.header("5-Year Historical performance of "+display)
		data1=yf.download(tickers=mapping[display], period="5y")	
		data1=data1.reset_index()	

		plotting1=px.line(data1,x="Date",y="Close",height=800).update_layout(xaxis_title="Date", yaxis_title="Close/₹",height=600)
		plotting1.update_traces(line_color="blue", line_width=3)
		st.plotly_chart(plotting1,use_container_width=True)


	with col2:
		st.subheader("Global Indicators/₹")
		
		end=datetime.now()
		start=end-timedelta(hours=end.hour,minutes=end.minute,seconds=end.second)

		try:
			sub_data_1=yf.download(tickers="INR=X", start=start, end=end, interval = "1m")
			dol_close=sub_data_1['Close'][-1]
			dol_diff=dol_close-sub_data_1['Open'][0]
			st.metric("United States Dollar",str('1$ = '+'{:.2f}'.format(dol_close)+' ₹'),str('{:.5f}'.format(dol_diff)+' ₹'))
		except:
			sub_data_11 = yf.download(tickers="INR=X",period="5d",interval="1h")
			st.metric("United States Dollar",str('1$ = '+'{:.2f}'.format(sub_data_11['Close'][-1])+' ₹'),str("showing the last close value"))


		try:	
			sub_data_2=yf.download(tickers="EURINR=X", start=start, end=end, interval = "1m")
			eur_close=sub_data_2['Close'][-1]
			eur_diff=eur_close-sub_data_2['Open'][0]	
			st.metric("EURO",str('1€ = '+'{:.2f}'.format(eur_close)+' ₹'),str('{:.5f}'.format(eur_diff))+' ₹')
		except:
			sub_data_22 = yf.download(tickers="EURINR=X",period="5d",interval="1h")
			st.metric("EURO",str('1€ = '+'{:.2f}'.format(sub_data_22['Close'][-1])+' ₹'),str("showing the last close value"))


		try:
			sub_data_3=yf.download(tickers="BTC-INR", start=start, end=end, interval = "1m")
			btc_close=sub_data_3['Close'][-1]
			btc_diff=btc_close-sub_data_3['Open'][0]
			st.metric("Bitcoin",str('1₿ = '+'{:.2f}'.format(btc_close)+' ₹'),str('{:.5f}'.format(btc_diff))+' ₹')

		except:
			sub_data_33 = yf.download(tickers="BTC-INR",period="5d",interval="1h")
			st.metric("Bitcoin",str('1₿ = '+'{:.2f}'.format(sub_data_33['Close'][-1])+' ₹'),str("showing the last close value"))


		try:
			sub_data_4=yf.download(tickers="ETH-INR", start=start, end=end, interval = "1m")
			eth_close=sub_data_4['Close'][-1]
			eth_diff=eth_close-sub_data_4['Open'][0]
			st.metric("Ethereum",str('1Ξ = '+'{:.2f}'.format(eth_close)+' ₹'),str('{:.5f}'.format(eth_diff))+' ₹')
		except:
			sub_data_44 = yf.download(tickers="ETH-INR",period="5d",interval="1h")
			st.metric("Ethereum",str('1Ξ = '+'{:.2f}'.format(sub_data_44['Close'][-1])+' ₹'),str("showing the last close value"))

		st.write("""---""")
		st.button("Update")


with MiddleBlock:
	st.header("Analysis")
	st.subheader("# Next 30 Days Forecasted Performance of "+display)
	prediction_data=yf.download(tickers=mapping[display],period="10y",interval="1d")
	
	prediction_data=prediction_data['Close']
	prediction_data=prediction_data.dropna()
	prediction_data=prediction_data.reset_index()
	prediction_data['Close']=np.log(prediction_data['Close']) #Normalization

	train=prediction_data[:-15]
	test=prediction_data[-15:]

	#Making room for the predictions
	current_time=prediction_data["Date"][len(prediction_data)-1]
	for i in range(15):
		current_time=current_time+timedelta(days=1)
		new_row=pd.DataFrame({"Date":[current_time],"Close":[None]})
		prediction_data=pd.concat([prediction_data,new_row],ignore_index=True)
	
	auto_arima_train=auto_arima(train['Close'],seasonal=False,stepwise=False)
	validate=auto_arima_train.predict(n_periods=len(test))
	prediction_data['Validation']=[None]*len(train)+list(validate)+[None]*15

	arima_all=ARIMA(prediction_data['Close'][:-15],order=(auto_arima_train.get_params().get("order")))
	arima_all=arima_all.fit()
	predict=arima_all.forecast(15)
	prediction_data['Validation'][-15:]=predict

	prediction_data["Close"]=np.e**prediction_data["Close"]
	prediction_data["Validation"]=np.e**prediction_data["Validation"]

	plotting2=go.Figure()
	plotting2.add_trace(go.Scatter(x=prediction_data['Date'][-100:],y=prediction_data['Close'][-100:],mode='lines',
		line=dict(color="blue",width=4),name="Historical Data"))
	plotting2.add_trace(go.Scatter(x=prediction_data['Date'][-100:],y=prediction_data['Close'][-100:],mode='markers',
		marker=dict(color="blue",size=10),name="Historical Values"))

	plotting2.add_trace(go.Scatter(x=prediction_data['Date'][-30:-15],y=prediction_data['Validation'][-30:-15],mode='lines',
		line=dict(color="red",width=4),name="Validation"))
	plotting2.add_trace(go.Scatter(x=prediction_data['Date'][-30:-15],y=prediction_data['Validation'][-30:-15],mode='markers',
		marker=dict(color="red",size=10),name="Validation Values"))

	plotting2.add_trace(go.Scatter(x=prediction_data['Date'][-15:],y=prediction_data['Validation'][-15:],mode='lines',
		line=dict(color="green",width=4),name="Forecast"))
	plotting2.add_trace(go.Scatter(x=prediction_data['Date'][-15:],y=prediction_data['Validation'][-15:],mode='markers',
		marker=dict(color="green",size=10),name="Forecast Values"))	

	plotting2.update_layout(xaxis_title="Date", yaxis_title="Close/₹",height=600)

	st.plotly_chart(plotting2,use_container_width=True)

	prediction_data["Prediction"]=prediction_data["Validation"]

	st.subheader("Next 5 Days Forecasting Values")
	for i in prediction_data[-15:-10].index:
		st.write(str(prediction_data['Date'][i].date())+"  :  "+str(prediction_data['Prediction'][i]))
		




with LowerBlock:
	st.subheader("# Candlestick Chart")

	col1,col2,col3,col4=st.columns([1,1,1,1],gap="large")
	
	with col1:
		st.write("")
		st.write("")
		if risk[display]==1 :
			st.markdown(display+":"+"<span style='color:Red;'> High risk stock !!</span>",unsafe_allow_html=True)
		elif risk[display]==0:
			st.markdown(display+":"+"<span style='color:Green;'> Low risk stock</span>",unsafe_allow_html=True)
		else:
			st.markdown(str(display+":"+"<span style='color:Gray;'> Can't determine risk of the stock.</span>"),unsafe_allow_html=True)

	with col2:
		per=st.selectbox("Select the period:",periods.keys(),index=6)

	with col3:
		inter=st.selectbox("Select the intervals:",intervals.keys(),index=7)

	

	data3=yf.download(tickers=mapping[display],period=periods[per],interval=intervals[inter])
	data3.index.name='Date'
	data3=data3.reset_index()

	st.write("The size of the dataset is: "+str(len(data3)))

	try:
		if len(data3)==0:
			x=1/0
		plotting3=go.Figure(data=[go.Candlestick(x=data3['Date'],open=data3['Open'],high=data3['High'],low=data3['Low'],close=data3['Close'])])
		plotting3.update_layout(xaxis_title="Date", yaxis_title="Close/₹",height=600)
		st.plotly_chart(plotting3,use_container_width=True)

	except:
		st.write("Please adjust the parameters in order to plot the data properly.")
		st.write("If the size of the dataset is 0, consider increasing the interval value.")
	

	



with footer:
	st.write("AMBILIO Technology")

