# CLV-and-CLV-Prediction

##What is Customer Lifetime Value(CLV)?

The monetary value a customer will bring to a company during their relationship and communication is known as "customer lifetime value."

We will achieve this by using the formulas below:

Repeat rate: number of customers who make multiple purchases / all customers

Churn rate: 1 - repeat rate

Purchase frequency: total transactions / total number of unique customers

Average order value: total price / total transactions

Customer value: average order value * purchase frequency

Profit margin: total price * profit margin rate(provided by the company)

CLV = (customer value / churn rate) * profit margin

Customers are segmented based on the generated CLV value, and operations are carried out according to these segments.


##CLV Prediction with BG-NBD and Gamma-Gamma

CLTV Prediction: BG/NBD * Gamma gamma submodel

We are performing CLTV prediction with BG/NBD and Gamma-Gamma.

Expected Number of Transactions with BG/NBD.

BG/NBD is used as a standalone sales prediction model, that is; it predicts the expected number of purchases per customer.

The information we need to use in this model and learn from the customer is:

x: Number of repeated purchases by the customer (more than 1) (frequency)

tx: Time between a customer's first and last purchase

T: Time since the customer's first purchase (customer's age)

Gamma Gamma SubModel

It is used to estimate how much profit a customer can bring per transaction on average.

The information we need to use in this model and learn from the customer is:

x: Number of repeated purchases by the customer (more than 1) (frequency)

mx: These are the observed transaction values, i.e., the monetary value, i.e., total price/total transaction.
