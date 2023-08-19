import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats


# df = pd.read_csv("./data/book_sales.csv", index_col='Date', parse_dates=['Date'],
#                  ).drop('Paperback', axis=1)
#
# df['Time'] = np.arange(len(df.index))
# plt.rc(
#     "figure",
#     autolayout=True,
#     figsize=(11, 4),
#     titlesize=18,
#     titleweight='bold',
# )
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=16,
#     titlepad=10,
# )
#
# fig, ax = plt.subplots()
# ax.plot('Time', 'Hardcover', data=df, color='0.75')
# ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_title('Time Plot of Hardcover Sales')
#
# # plt.show()
#
# df['Lag_1'] = df['Hardcover'].shift(1)
# df = df.reindex(columns=['Hardcover', 'Lag_1'])
#
# print(df.head())
#
# fig, ax = plt.subplots()
# ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_aspect('equal')
# ax.set_title('Lag Plot of Hardcover Sales');
#


from pathlib import Path
from warnings import simplefilter

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)



# Load Tunnel Traffic dataset
data_dir = Path("./data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])

tunnel = tunnel.set_index("Day")

# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.
tunnel = tunnel.to_period()

print(tunnel.head())

df = tunnel.copy()

df['Time'] = np.arange(len(tunnel.index))

print(df.head())

from sklearn.linear_model import LinearRegression

# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)

