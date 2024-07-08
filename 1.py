# Import Packages
import csv
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import scipy.stats as st
import gp1
from gp1 import genetic
from gp1 import functions
from gp1.functions import make_function
from gp1.genetic import SymbolicTransformer
from gp1.fitness import make_fitness
import logging
logging.getLogger().setLevel(logging.ERROR)
from scipy.stats import rankdata,zscore
import tqdm
import pandas_ta as ta
from numpy import power, mod, maximum, minimum, hypot, arctan2, clip, round, floor, ceil, trunc, fmod, remainder, reciprocal, square, positive, negative



# # import and clean data
# raw_data = pd.read_pickle('RU.pkl')
# raw_data.index = raw_data.index.droplevel(0)
# raw_data = raw_data.reset_index()

# # select 1 yr data
# raw_data['Year']= raw_data['datetime'].dt.year
# selected_data = raw_data[raw_data['Year']==2023]
# df = selected_data.copy()

# df['return'] = df['close'].pct_change()
# df['Predicted'] = df['return'].shift(-1)

# # group by date then compute the rolling 60 volatility
# df['Volatility'] = df.groupby('trading_date')['return'].rolling(60*8*60).std().reset_index(0,drop=True)

# # Select train and test data
# base_Factor = ["open",'high','low','volume',"open_interest","Volatility"]
# df = df.dropna()
# X_train = df[base_Factor] # remove the last row
# y_train = df['Predicted']# remove the last row
# future_codes = ['RU']

# import and clean data
raw_data = pd.read_pickle('RM.pkl')
raw_data.index = raw_data.index.droplevel(0)
raw_data = raw_data.reset_index()

# select 1 yr data
raw_data['Year']= raw_data['datetime'].dt.year
selected_data = raw_data[raw_data['Year']>=2023]
df = selected_data.copy()

df['return'] = df['close'].pct_change()
df['Predicted'] = df['return'].shift(-1)

# group by date then compute the rolling 60 volatility
df['Volatility'] = df['return'].rolling(60*8*60).std().reset_index(0,drop=True)

# Assuming df is your DataFrame with 'open', 'high', 'close', 'low' columns
# Set the number of trading hours per day
hrs_trading = 8

# Calculate Simple Moving Averages (SMA)
df['SMA_5m'] = df['close'].rolling(window=5).mean()
df['SMA_15d'] = df['close'].rolling(window=15*60*hrs_trading).mean()

# Exponential Moving Average (EMA)
df['EMA_5m'] = df['close'].ewm(span=5, adjust=False).mean()
df['EMA_15d'] = df['close'].ewm(span=15 * 60 * hrs_trading, adjust=False).mean()

# Moving Average Convergence Divergence (MACD)
df['EMA_12m'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26m'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12m'] - df['EMA_26m']
df['Signal_Line'] = df['MACD'].ewm(span=9 * 60 * hrs_trading, adjust=False).mean()

# Bollinger Bands
df['Middle_BB'] = df['close'].rolling(window=20 * 60 * hrs_trading).mean()
df['Upper_BB'] = df['Middle_BB'] + 2 * df['close'].rolling(window=20 * 60 * hrs_trading).std()
df['Lower_BB'] = df['Middle_BB'] - 2 * df['close'].rolling(window=20 * 60 * hrs_trading).std()

# Relative Strength Index (RSI)
delta = df['close'].diff()
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0
roll_up = up.rolling(window=14 * 60 * hrs_trading).mean()
roll_down = down.abs().rolling(window=14 * 60 * hrs_trading).mean()
RS = roll_up / roll_down
df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

# Average True Range (ATR)
df['High-Low'] = df['high'] - df['low']
df['High-PrevClose'] = abs(df['high'] - df['close'].shift())
df['Low-PrevClose'] = abs(df['low'] - df['close'].shift())
df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14 * 60 * hrs_trading).mean()

# Percentage Price Oscillator (PPO)
df['PPO'] = ((df['EMA_12m'] - df['EMA_26m']) / df['EMA_26m']) * 100

# Stochastic Oscillator
df['14-high'] = df['high'].rolling(window=14 * 60 * hrs_trading).max()
df['14-low'] = df['low'].rolling(window=14 * 60 * hrs_trading).min()
df['%K'] = (df['close'] - df['14-low'])*100 / (df['14-high'] - df['14-low'])
df['%D'] = df['%K'].rolling(window=3 * 60 * hrs_trading).mean()

# Williams %R
df['Williams_%R'] = (df['14-high'] - df['close']) / (df['14-high'] - df['14-low']) * -100

# On-Balance Volume (OBV)
df['OBV'] = np.where(df['close'] > df['close'].shift(), df['volume'], 
                     np.where(df['close'] < df['close'].shift(), -df['volume'], 0)).cumsum()

# Select train and test data
base_Factor = ['close', 'open_interest', 'total_turnover',
       'high', 'open', 'low', 'volume', 'return',
       'SMA_5m', 'SMA_15d', 'EMA_5m', 'EMA_15d', 'EMA_12m',
       'EMA_26m', 'MACD', 'Signal_Line', 'Middle_BB', 'Upper_BB', 'Lower_BB',
       'RSI', 'High-Low', 'High-PrevClose', 'Low-PrevClose', 'TR', 'ATR',
       'PPO', '14-high', '14-low', '%K', '%D', 'Williams_%R', 'OBV']
df.to_pickle('RM_ready.pkl')
# set time for train 
df = df[df['Year']==2023].copy()
df = df.dropna(axis = 1, how = 'all')
df = df.dropna()

X_train = df[base_Factor] # remove the last row
y_train = df['Predicted']# remove the last row
future_codes = ['RU']

# def funcs
def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 1e-10, np.divide(x1, x2), 1.)

def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))

def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 1e-10, np.log(np.abs(x1)), 0.)

def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 1e-10, 1. / x1, 0.)

def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))

def gp_add(x, y):
    return x + y

def gp_sub(x, y):
    return x - y

def gp_mul(x, y):
    return x * y

def gp_div(x, y):
    return _protected_division(x, y)

def gp_sqrt(data):
    return _protected_sqrt(data)

def gp_log(data):
    return _protected_log(data)

def gp_neg(data):
    return np.negative(data)

def gp_inv(data):
    return _protected_inverse(data)

def gp_abs(data):
    return np.abs(data)

def gp_sin(data):
    return np.sin(data)

def gp_cos(data):
    return np.cos(data)

def gp_tan(data):
    return np.tan(data)

def gp_sig(data):
    return _sigmoid(data)

def gp_exp(data):
    return np.exp(data)

# Creating custom functions for gplearn
gp_add = make_function(function=gp_add, name='gp_add', arity=2)
gp_sub = make_function(function=gp_sub, name='gp_sub', arity=2)
gp_mul = make_function(function=gp_mul, name='gp_mul', arity=2)
gp_div = make_function(function=gp_div, name='gp_div', arity=2)

gp_sqrt = make_function(function=gp_sqrt, name='gp_sqrt', arity=1)
gp_log = make_function(function=gp_log, name='gp_log', arity=1)
gp_neg = make_function(function=gp_neg, name='gp_neg', arity=1)
gp_inv = make_function(function=gp_inv, name='gp_inv', arity=1)
gp_abs = make_function(function=gp_abs, name='gp_abs', arity=1)
gp_sin = make_function(function=gp_sin, name='gp_sin', arity=1)
gp_cos = make_function(function=gp_cos, name='gp_cos', arity=1)
gp_tan = make_function(function=gp_tan, name='gp_tan', arity=1)
gp_sig = make_function(function=gp_sig, name='gp_sig', arity=1)
gp_exp = make_function(function=gp_exp, name='gp_exp', arity=1)


# Power operation
def pow_func(x):
    return np.sign(x)*(np.abs(x)) ** (3)
pow_func = make_function(function=pow_func, name='pow', arity=1)

# Modulo operation
def mod_func(x, y):
    return np.ceil(np.mod(x,y+1e-10))
    # return np.where(y != 0, np.mod(x, y), 0)
mod_func = make_function(function=mod_func, name='mod', arity=2)

# Maximum of two numbers
def max_func(x, y):
    return maximum(x, y)
max_func = make_function(function=max_func, name='max', arity=2)

# Minimum of two numbers
def min_func(x, y):
    return minimum(x, y)
min_func = make_function(function=min_func, name='min', arity=2)

# Hypotenuse using Pythagoras theorem
def hypot_func(x, y):
    return hypot(x, y)
hypot_func = make_function(function=hypot_func, name='hypot', arity=2)

# Angle between x and y coordinate
def atan2_func(x, y):
    return arctan2(x, y)
atan2_func = make_function(function=atan2_func, name='atan2', arity=2)

# Clip (limit) the values in an array
def clip_func(x, a_min, a_max):
    return clip(x, a_min, a_max)
clip_func = make_function(function=clip_func, name='clip', arity=3)

# Round to the nearest integer
def round_func(x):
    return round(x)
round_func = make_function(function=round_func, name='round', arity=1)

# Floor division
def floor_func(x):
    return floor(x)
floor_func = make_function(function=floor_func, name='floor', arity=1)

# Ceiling division
def ceil_func(x):
    return ceil(x)
ceil_func = make_function(function=ceil_func, name='ceil', arity=1)

# Truncate the decimal part
def trunc_func(x):
    return trunc(x)
trunc_func = make_function(function=trunc_func, name='trunc', arity=1)

# Returns the remainder of division
def fmod_func(x, y):
    return np.fmod(x, y+1e-10)
fmod_func = make_function(function=fmod_func, name='fmod', arity=2)

# Returns element-wise remainder of division
def remainder_func(x, y):
    return remainder(x, y+1e-10)
remainder_func = make_function(function=remainder_func, name='remainder', arity=2)

# Returns the reciprocal of the argument, element-wise
def reciprocal_func(x):

    return reciprocal(x+1e-100)
reciprocal_func = make_function(function=reciprocal_func, name='reciprocal', arity=1)

# Returns the element-wise square of the input
# def square_func(x):
#     return min(square(x),2**10)
# square_func = make_function(function=square_func, name='square', arity=1)

# Numerical positive, element-wise
def positive_func(x):
    return positive(x)
positive_func = make_function(function=positive_func, name='positive', arity=1)

# Numerical negative, element-wise
def negative_func(x):
    return negative(x)
negative_func = make_function(function=negative_func, name='negative', arity=1)


# Fitness function 
def w_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return corr
    return 0

def w_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return w_pearson(y_pred_ranked, y_ranked, w)



def weighted_spearman_icir_multiperiod(y,y_pred,w):
    """Calculate the weighted Spearman correlation coefficient for multi period data.
    input data y, y_pred must  be sort by tradedate and classname together

    """

    icall = []
    k = len(future_codes) #输入，品种数要对应
    step = len(future_codes)#初始值输入，品种数对应
    while k <= len(y):

        if w[k - step:k].sum() == 0 :
            icall.append(0)
        else:
            ic = w_spearman(y[k - step:k], y_pred[k - step:k], w[k - step:k])
            icall.append(ic)

        k += step

    icall = np.array(icall)
    if icall.std() ==0:
        return 0
    else:
        return abs(icall.mean())/icall.std()*np.sqrt(50)


def weighted_spearman_ic_multiperiod(y,y_pred,w):
    """Calculate the weighted Spearman correlation coefficient for multi period data.
    input data y, y_pred must  be sort by tradedate and classname together

    """

    icall = []
    k = len(future_codes) #输入，品种数要对应
    step = len(future_codes)#初始值输入，品种数对应
    while k <= len(y):

        if w[0:k].sum() == 0 :
            icall.append(0)
        else:
            ic = w_spearman(y[k - step:k], y_pred[k - step:k], w[k - step:k])
            icall.append(ic)

        k += step

    icall = np.array(icall)
    return abs(icall.mean())


def IC(x,y,w):
    a = np.argsort(x)
    b = np.argsort(y)

    return np.corrcoef(a, b)[0,1]


icir = gp1.fitness.make_fitness(function=IC, greater_is_better=True, wrap=True)


print("开始挖掘因子...")

generations = 7
population_size = 20000
random_state=61 # random seed

# function_set = [ gp_add, gp_sub, gp_mul, gp_div,
#                  gp_sqrt, gp_log, gp_neg, gp_inv,
#                  gp_abs,  gp_sin,
#                  gp_cos, gp_tan, gp_sig, ta_ema_apply, ta_rsi_apply
#                ]
function_set = [ gp_add, gp_sub, gp_mul, gp_div,
                 gp_sqrt, gp_log, gp_neg, gp_inv,
                 gp_abs,  gp_sin,
                 gp_cos, gp_tan, gp_sig,
                 pow_func, mod_func, max_func,
                 min_func, hypot_func, atan2_func,
                 clip_func, round_func, floor_func,
                 ceil_func, trunc_func, fmod_func,
                 remainder_func, reciprocal_func
               ]

# function_set = [ gp_add, gp_sub, gp_mul, gp_div,
#                  gp_sqrt, gp_log, gp_neg, gp_inv,
#                  gp_abs,  gp_sin,
#                  gp_cos, gp_tan, gp_sig,
#                  sma_5,sma_10,sma_20,
#                  ts_sum_5,ts_sum_10,ts_sum_20,
#                  ts_rank_5,ts_rank_10,ts_rank_20,
#                  stddev_5,stddev_10,stddev_20,
#                  product_5,product_10,product_20,
#                  ts_min_5,ts_min_10,ts_min_20,
#                  ts_max_5,ts_max_10,ts_max_20,
#                  pow_func, mod_func, max_func,
#                  min_func, hypot_func, atan2_func,
#                  clip_func, round_func, floor_func,
#                  ceil_func, trunc_func, fmod_func,
#                  remainder_func, reciprocal_func,
#                 ,delta_5,0,delta_20,
#                  delay_1,delay_5,delay_10,delay_20,
#                 ts_normal_cdf,ts_lognormal_cdf,
#                 ts_normal_pdf,ts_lognormal_pdf,
#                 ta_ema_apply,ta_rsi_apply
#                ]

est_gp = genetic.SymbolicTransformer(feature_names=base_Factor,
                            init_method = 'half and half',         # half and half 倾向于长出balance和unbalanced的树， full 会长处balance的树
                            function_set=function_set ,             #+ function_set,function_base
                            generations=generations,
                            stopping_criteria=0.8,
                            metric=icir,                           #my_metric_group,my_metric_ud,'spearman'
                            population_size=population_size,
                            tournament_size=300,
                            init_depth=(2,4),
                            random_state=random_state,
                            n_components=10,
                            parsimony_coefficient = 'auto',
                            const_range=None,
                            n_jobs=-1,
                            low_memory=True,
                            verbose=1,
                         )

est_gp.fit(X_train, y_train)

# print result
best_programs = est_gp._best_programs
best_programs_dict = {}

for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    
    best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}
    print(f'\n\n{factor_name}: \nfitness: {p.fitness_}\nexpression: {str(p)}\ndelth: {p.length_}\nlength: {p.length_}\n\n')
best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict[best_programs_dict.fitness.abs() >= 0.0]
best_programs_dict = best_programs_dict.sort_values(by='fitness',ascending=False)

print(best_programs_dict)
