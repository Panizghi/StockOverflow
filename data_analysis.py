import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

# Load preprocessed data
data_df = pd.read_csv("https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv")

# Data analysis
temp = data_df["Class"].value_counts()
df = pd.DataFrame({'Class': temp.index, 'values': temp.values})

trace = go.Bar(
    x=df['Class'], y=df['values'],
    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
    marker=dict(color="Red"),
    text=df['values']
)
data = [trace]
layout = dict(
    title='Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
    xaxis=dict(title='Class', showticklabels=True),
    yaxis=dict(title='Number of transactions'),
    hovermode='closest', width=600,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig = dict(data=data, layout=layout)
iplot(fig, filename='class')
fig.savefig('fraud-graphs/class.png', transparent=True)

# Continue with additional analysis steps...
