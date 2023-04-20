import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/kaggle/input/retail-store-sales-transactions/scanner_data.csv")



### CLV

# We are trying to understand the data.

def check_df(dataframe, head=7):
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#################### Head ####################")
    print(dataframe.head(head))

check_df(df)



# Data Preparation

# We are removing the variable that does not carry any information
df.drop("Unnamed: 0", inplace=True, axis=1) 

# We are finding the unit price of the product.
df["Unit_Price"] = df["Sales_Amount"] / df["Quantity"]

# We are accessing the values required for the formulas.
cltv = df.groupby('Customer_ID').agg({'Transaction_ID': lambda x: x.nunique(),  # total transaction
                                        'Sales_Amount': lambda x: x.sum()})     # total price

# We are changing the names of the variables we have created.
cltv.columns = ['total_transaction', 'total_price']

cltv.head()



# Average Order Value (average_order_value = total_price / total_transaction)
cltv["average_order_value"] = cltv["total_price"] / cltv["total_transaction"]

# Purchase Frequency (total_transaction / total_number_of_customers)
cltv.shape[0]  # total number of customers
cltv["purchase_frequency"] = cltv["total_transaction"] / cltv.shape[0]

# Repeat Rate & Churn Rate
# (number of customers who make multiple purchases / all customers)
repeat_rate = cltv[cltv["total_transaction"] > 1].shape[0] / cltv.shape[0]
churn_rate = 1 - repeat_rate

# Profit Margin (total price * profit margin rate)
# We are setting the profit margin rate as 0.10.
cltv['profit_margin'] = cltv['total_price'] * 0.10

# Customer Value (customer_value = average_order_value * purchase_frequency)
cltv['customer_value'] = cltv['average_order_value'] * cltv["purchase_frequency"]

# Customer Lifetime Value (CLV = (customer_value / churn_rate) x profit_margin)
cltv["cltv"] = (cltv["customer_value"] / churn_rate) * cltv["profit_margin"]

cltv.head()



# Creating the Segments

# We are dividing the CLTV values into 4 parts and creating a segment variable.
cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

cltv.sort_values(by="cltv", ascending=False).head()



# We are creating count, mean, and sum values for all other variables based on the segments.
cltv.groupby("segment").agg({"count", "mean", "sum"})

# Thus, we can perform operations on customers based on their segments.




### CLV Prediction

def check_df(dataframe, head=10):
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#################### Head ####################")
    print(dataframe.head(head))

check_df(df)


# Data Preparation

# We are converting the date in object format to datetime format using the datetime module.
df["Date"] = pd.to_datetime(df["Date"])

# We will find the last date and make transactions starting from 2 days later.
df["Date"].max()  # '31-12-2016'
today_date = pd.to_datetime("2017-01-02")

# We are calculating the necessary values for our calculations.
cltv_pr = df.groupby('Customer_ID').agg(
    {'Date': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,  # tx: recency degeri
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],         # T: customer's age
     'Transaction_ID': lambda Invoice: Invoice.nunique(),    # x: frequency degeri 
     'Sales_Amount': lambda TotalPrice: TotalPrice.sum()})

# The variables in the output are displayed in two rows. We need to remove the first row.
cltv_pr.columns = cltv_pr.columns.droplevel(0)

# We are changing the names of the variables we have created.
cltv_pr.columns = ['recency', 'T', 'frequency', 'monetary']

# mx: monetary degeri. total price/frequency
cltv_pr["monetary"] = cltv_pr["monetary"] / cltv_pr["frequency"]

# The frequency refers to customers who made more than one purchase.
cltv_pr = cltv_pr[(cltv_pr['frequency'] > 1)]

# We are converting the recency value into weekly time.
cltv_pr["recency"] = cltv_pr["recency"] / 7

# We are converting the customer's age into weekly time.
cltv_pr["T"] = cltv_pr["T"] / 7

cltv_pr.head()



# Setting up the BG-NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_pr['frequency'],       
        cltv_pr['recency'],
        cltv_pr['T'])


# Which are the top 10 customers that we expect to have the most purchases within the a month?

cltv_pr["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_pr['frequency'],
                                               cltv_pr['recency'],
                                               cltv_pr['T'])


# What is the expected number of sales for the entire company in 3 months?

cltv_pr["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_pr['frequency'],
                                               cltv_pr['recency'],
                                               cltv_pr['T'])

cltv_pr.head()


# Evaluation of Prediction Results

plot_period_transactions(bgf)
plt.show()


# Setting up the Gamma-Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.01)  

ggf.fit(cltv_pr['frequency'], cltv_pr['monetary'])


# We are seeing the profit prediction for customers.
cltv_pr["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_pr['frequency'],
                                                                             cltv_pr['monetary'])


cltv_pr.sort_values("expected_average_profit", ascending=False).head(5)


# Calculation of CLV prediction using BG-NBD and GG models.

# CLTV Prediction: BG/NBD * Gamma gamma submodel

#We are calculating the 3-month CLV prediction.
cltv_pred = ggf.customer_lifetime_value(bgf,
                                   cltv_pr['frequency'],
                                   cltv_pr['recency'],
                                   cltv_pr['T'],
                                   cltv_pr['monetary'],
                                   time=3,  
                                   freq="W",  # T and frequency time interval; week(W)
                                   discount_rate=0.01)

cltv_pred = cltv_pred.reset_index()

cltv_pred.head()


# We combine CLV predictions with other variables.
cltv_prediction = cltv_pr.merge(cltv_pred, on="Customer_ID", how="left")

cltv_prediction.head()


# Creating Segments Based on CLV

# We are dividing the CLV predictions into 4 parts and creating a segment variable.
cltv_prediction["segment"] = pd.qcut(cltv_prediction["clv"], 4, labels=["D", "C", "B", "A"])

cltv_prediction.sort_values(by="clv", ascending=False).head()


# We are creating count, mean, and sum values for all other variables based on the segments.
cltv_prediction.groupby("segment").agg(
    {"count", "mean", "sum"})


# Thus, we can perform operations on customers based on their segments.


