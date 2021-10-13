#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:27:18 2021

@author: kevinlu
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs
from scipy.stats.mstats import winsorize
# pip install pandas_datareader
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#pip install tabulate
from tabulate import tabulate
import scipy.stats as scs
#%%
##############################################################################
#1. MONTHLY STOCK RETURN DATA (Compustat)

# 1.1. Read Compustat monthly stock returns data 
df1 = pd.read_csv('proj_stockret.csv', parse_dates = ['datadate'])
df1 = df1.sort_values(by=['gvkey','datadate'])
df1 = df1.dropna()

# 1.2. Create portfolio formation year (pfy) variable, where
#      pfy = current year for Jul-Dec dates and previous year for Jan-Jun dates.
df1['year'], df1['month'] = df1['datadate'].dt.year, df1['datadate'].dt.month
df1['pfy'] = np.where(df1.month > 6 , df1.year, df1.year - 1)

# 1.3. Compute monthly return compounding factor (1+monthly return)
df1['mretfactor'] = 1 + df1.trt1m/100
df1 = df1.sort_values(by=['gvkey','pfy'])
df2 = df1[['gvkey', 'conm', 'datadate', 'pfy', 'mretfactor']]
df2.head(20)

# 1.4. Compound monthly returns to get annual returns at end-June of each pfy,
#      ensuring only firm-years with 12 mths of return data from Jul-Jun are selected.
df2['yret'] = df2.groupby(['gvkey', 'pfy'])['mretfactor'].cumprod() - 1
df3 = df2.groupby(['gvkey', 'pfy']).nth(11) 


# 1.5. Winsorize outliers
df3['yret'] = winsorize(df3['yret'], limits=[0.025,0.025])
df3 = df3.drop(['mretfactor'], axis=1)    # "axis=1" means to drop column
df3.head()

#%%
##############################################################################
#2. YEARLY STOCK ACCOUNTING DATA (Compustat)
# 2.1. Read Compustat accounting data
df4 = pd.read_csv('proj_acctdata.csv', parse_dates = ['datadate'])
df4 = df4.sort_values(by=['gvkey','datadate'])

# 2.2. Create portfolio formation year (pfy) variable, where
#      pfy = current year for Jan-Mar year-end dates and next year for Apr-Dec year-end dates.
#      This is to facilitate compounding returns over July-June by pfy below.
df4['year'], df4['month'] = df4['datadate'].dt.year, df4['datadate'].dt.month
df4['pfy'] = np.where(df4.month < 4, df4.year, df4.year + 1)
#%%
# 2.3. Compute accounting variables from Compustat data, keep relevant variables, delete missing values
# a) Profitability ratios
# F_ROA
df4["pre_at"] = df4.groupby(["gvkey"])["at"].shift(1)
df4["roa"] = df4["ib"]/df4["pre_at"]
df4["f_roa"] = np.where(df4["roa"] > 0, 1, 0)

# F_CFO
df4["cfo"] = df4["oancf"]/df4["pre_at"]
df4["f_cfo"] = np.where(df4["cfo"] > 0, 1, 0)

# F_CHG_ROA
df4["pre_roa"] = df4.groupby(["gvkey"])["roa"].shift(1)
df4["f_chg_roa"] = np.where(df4["roa"] > df4["pre_roa"], 1, 0)

# F_Accrual
df4["accrual"] = (df4["ib"] - df4["oancf"])/df4["pre_at"]
df4["f_accrual"] = np.where(df4["cfo"] > df4["roa"], 1, 0)

# b) Leverage, liquidity, source of funds
# F_LEVER
df4["ldta"] = df4["dltt"]/np.mean([df4["at"],df4["pre_at"]])
df4["pre_ldta"] = df4.groupby(["gvkey"])["ldta"].shift(1)
df4["f_lever"] = np.where(df4["ldta"] > df4["pre_ldta"], 0, 1) #increase = 0

# F_LIQUID
df4["curr_ratio"] = df4["act"]/df4["lct"]
df4["pre_curr_ratio"] = df4.groupby(["gvkey"])["curr_ratio"].shift(1)
df4["f_liquid"] = np.where(df4["curr_ratio"] > df4["pre_curr_ratio"], 1, 0)

# EQ_OFFER
df4["pre_cshi"] = df4.groupby(["gvkey"])["cshi"].shift(1)
df4["eq_offer"] = np.where(df4["cshi"] > df4["pre_cshi"], 0, 1) # increase = 0 because issued shares this year


# c) Operating efficiency
# F_CHG_MARGIN
df4['margin'] =  df4["gp"]/df4["revt"]
df4['pre_margin'] =  df4.groupby(["gvkey"])["margin"].shift(1)
df4['f_chg_margin'] =  np.where(df4['margin'] > df4['pre_margin'], 1, 0)

# F_CHG_TURN
df4['turn'] =  df4['revt'] / df4['pre_at'] 
df4['pre_turn'] = df4.groupby(["gvkey"])["margin"].shift(1)
df4['f_chg_turn']= np.where(df4['turn'] > df4['pre_turn'], 1, 0)

# d) Value ratio
# BM ratio
df4["bm"] = (df4["at"] - df4["lt"])/ df4["mkvalt"]
#%%
# 2.4 Compute F_SCORE
df4["f_score"] = df4[["f_roa", "f_cfo", "f_chg_roa", "f_accrual", "f_lever", "f_liquid", "eq_offer", "f_chg_margin", "f_chg_turn"]].sum(axis=1)

#%%
# 2.5. Merge accounting dataset (df4) with returns dataset (df3)
df5 = pd.merge(df3, df4, how='inner', on=['gvkey', 'pfy'])
df5 = df5[['gvkey','tic', 'conm_x', 'pfy', 'yret', 'f_score', 'bm', 'mkvalt', "datadate_x"]]
df5 = df5.dropna()

#%%
##############################################################################
# 3. Year-by-year OLS regression of annual returns against accounting variables
def olsreg(d, yvar, xvars):
    Ygrp = d[yvar]
    Xgrp = sm.add_constant(d[xvars])
    reg = sm.OLS(Ygrp, Xgrp).fit()
    return reg.params

df_group = df5.groupby('pfy')
yearcoef = df_group.apply(olsreg, 'yret', ['f_score', 'bm', 'mkvalt'])
print('Coefficients of year-by-year regressions\n', yearcoef, '\n'*3)
tstat, pval = scs.ttest_1samp(yearcoef, 0)
print('T-statistics and p-values of year-by-year coefficients: \n')
print(pd.DataFrame({'t-stat': tstat.round(4), 'p-value': pval.round(4)}, 
                   index=['const', 'f_score', 'bm', 'mkvalt']), '\n'*5)

#%%
##############################################################################
# 4. Extract S&P 500 Data and Risk free rate data

# 4.1 S&P 500
spy = pdr.get_data_yahoo("^GSPC", "2000-01-01", "2021-03-01", interval="m")[["Adj Close"]]

spy["benchmark_ret"] = (spy['Adj Close'].pct_change()+ 1).cumprod() - 1
spy['year'], spy['month'] = spy.index.year, spy.index.month
spy['pfy'] = np.where(spy.month > 6, spy.year, spy.year - 1)

spy_ret = spy.groupby(["pfy"])["benchmark_ret"].nth(11)

#4.2 Risk free rate
# Risk Free Rate is US 10-year Treasury Bill, extracted from Fama French Data Library
tbill = pd.read_csv("fama_french_risk_free.csv")
tbill["pfy"] = pd.to_datetime(tbill["pfy"].astype(str), format="%Y")
tbill["rf"] = tbill["rf"]/100

#%%
##############################################################################
# 5. Functions for later computation

# 5.1 Construct Long-Only Strategy
def long_only_strategy(dfg, mkval, bm_cut, f_cut):
    
    portfolio_return = {"pfy":[],
                    "companies":[],
                    "no_of_comp":[],
                   "return":[]}
    
    for i, j in dfg.groupby(["pfy"]):
        df = j.copy()
        
        # Using signals to filter companies
        df = df[df["mkvalt"] >= df["mkvalt"].quantile(mkval)]
        df = df[df["bm"]>= df["bm"].quantile(bm_cut)]
        df = df[df["f_score"] >= f_cut]

        portfolio_return["pfy"].append(i)
        portfolio_return["companies"].append(df["gvkey"].to_list())
        portfolio_return["no_of_comp"].append(df["conm_x"].count())
        portfolio_return["return"].append(df["yret"].mean())

    portfolio = pd.DataFrame(portfolio_return)
    
    final = pd.merge(portfolio, spy_ret, how = "right", on="pfy")
    final["pfy"] = pd.to_datetime(final["pfy"].astype(str), format="%Y")
    final["cum_return"] = (final["return"] + 1).cumprod() - 1
    final["cum_return_snp"] = (final["benchmark_ret"] + 1).cumprod() - 1
    final["strat_$_return"] = final["cum_return"] * 1000
    final["snp_$_return"] = final["cum_return_snp"] * 1000
#     final = final.dropna()
    
        
    return final
#%%
# 5.2 Construct Long/Short Strategy
def long_short_strategy(dfg, mkval, bm_long_cut, f_long_cut, bm_short_cut, f_short_cut):
    portfolio_return2 = {"pfy":[],
                     "companies":[],
                    "no_of_comp":[],
                   "return":[]}

    for i, j in dfg.groupby(["pfy"]):
        # Using signals to filter companies
        df = j.copy()
        df = df[df["mkvalt"] >= df["mkvalt"].quantile(mkval)]
        df_long = df[df["bm"]>= df["bm"].quantile(bm_long_cut)]
        df_long = df_long[df_long["f_score"] >= f_long_cut]

        df_short = df[df["bm"]<=df["bm"].quantile(bm_short_cut)]
        df_short = df_short[df_short["f_score"] <=f_short_cut]
        df_short["yret"] = -df_short["yret"]

        df_final = pd.concat([df_long, df_short])
        
        portfolio_return2["pfy"].append(i)
        portfolio_return2["companies"].append(df_final["gvkey"].to_list())
        portfolio_return2["no_of_comp"].append(df_final["conm_x"].count())
        portfolio_return2["return"].append(df_final["yret"].mean())

    portfolio = pd.DataFrame(portfolio_return2)
    
    final2 = pd.merge(portfolio, spy_ret, how = "right", on="pfy")
    final2["pfy"] = pd.to_datetime(final2["pfy"].astype(str), format="%Y")
    final2["cum_return"] = (final2["return"] + 1).cumprod() - 1
    final2["cum_return_snp"] = (final2["benchmark_ret"] + 1).cumprod() - 1
    final2["strat_$_return"] = final2["cum_return"] * 1000
    final2["snp_$_return"] = final2["cum_return_snp"] * 1000
#     final2 = final2.dropna()
    
    return final2

#%%
# 5.3 Compute CAGR
def compute_cagr(df):
    strat_cagr = ((df["cum_return"].tail(1))**(1/len(df))-1)
    sp500_cagr = ((df["cum_return_snp"].tail(1))**(1/len(df))-1)
    cagr = list([float(strat_cagr.values*100), float(sp500_cagr.values*100)])
    return cagr

#%%
# 5.4 Plot Cumulative Returns
def plot_cum_return(df, strat, output):
    st = compute_cagr(df)
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_ylabel("Cumulative return $", fontsize=15)
    ax.set_xlabel("Year", fontsize=15)
    min_year = min(df["pfy"].dt.year)
    max_year = max(df["pfy"].dt.year)
    strat_ret = float(df["strat_$_return"].tail(1))
    snp_ret = float(df["snp_$_return"].tail(1))
    style = dict(size=15, color='red')
    
    if strat == "l":
        ax.set_title("Annual investment performance of Long-Only Strategy {}-{}".format(min_year, max_year), fontsize=20)
        ax.plot(df["pfy"], df["strat_$_return"], label="Long-Only Strategy")
        ax.plot(df["pfy"], df["snp_$_return"], label="S&P500")

        ax.text(pd.Timestamp("2019-01-01"), df["strat_$_return"].tail(1)+100, "Long-Only Strategy: ${:,.2f}, CAGR: {}%".format(strat_ret, round(st[0], 2)), ha='right', **style)
        ax.text(pd.Timestamp("2019-01-01"), df["snp_$_return"].tail(1), "S&P500: ${:,.2f}, CAGR: {}%".format(snp_ret, round(st[1],2)), ha='right', **style)
        ax.legend(prop={'size': 15})
        plt.savefig(output)

    else:
        ax.set_title("Annual investment performance of Long/Short Strategy {}-{}".format(min_year, max_year), fontsize=20)
        ax.plot(df["pfy"], df["strat_$_return"], label="Long/Short Strategy")
        ax.plot(df["pfy"], df["snp_$_return"], label="S&P500")

        ax.text(pd.Timestamp("2019-01-01"), df["strat_$_return"].tail(1)+1, "Long/Short Strategy: ${:,.2f}, CAGR: {}%".format(strat_ret, round(st[0], 2)), ha='right', **style)
        ax.text(pd.Timestamp("2019-01-01"), df["snp_$_return"].tail(1), "S&P500 ${:,.2f}, CAGR: {:,.2f}%".format(snp_ret, round(st[1],2)), ha='right', **style)
        ax.legend(prop={'size': 15})
        plt.savefig(output)
        
#%%
# 5.5 Compute metrics
def compute_metrics(portfolio):
    final = pd.merge(portfolio, tbill, on="pfy", how="left")
    
    final["alpha_ret_port"] = final["return"] - final["rf"]
    final["alpha_ret_snp"] = final["benchmark_ret"] - final["rf"]
    
    # Compute CAGR
    cagr = compute_cagr(portfolio)
    cagr_port = cagr[0]
    cagr_snp = cagr[1]

    # Compute annualized Sharpe ratio
    sharpe_port = final['alpha_ret_port'].mean() / final['alpha_ret_port'].std()
    sharpe_snp = final['alpha_ret_snp'].mean() / final['alpha_ret_snp'].std()
    
    # Compute Sortino ratio 
    rbar = 0.0
    sortino_port = final['alpha_ret_port'].mean() / final[final["alpha_ret_port"] < rbar]['alpha_ret_port'].std()
    sortino_snp = final['alpha_ret_snp'].mean() / final[final["alpha_ret_snp"] < rbar]['alpha_ret_snp'].std()
    
    
    # Compute Maximum drawdown
    cumret_port = np.cumprod(final['return'] + 1)
    cumret_snp = np.cumprod(final['benchmark_ret'] + 1)
    hwm_port = cumret_port.cummax()
    hwm_snp = cumret_snp.cummax()
    
    max_drawdown_port = ((hwm_port - cumret_port) / hwm_port).round(2).max()
    max_drawdown_snp = ((hwm_snp - cumret_snp) / hwm_snp).round(2).max()
    
    # Tabulate output
    tab = tabulate([["CAGR", "{}%".format(round(cagr_port,2)), "{}%".format(round(cagr_snp,2))], 
                    ["Sharpe Ratio", "{}".format(round(sharpe_port,2)), "{}".format(round(sharpe_snp,2))], 
                   ["Sortino Ratio", "{}".format(round(sortino_port,2)), "{}".format(round(sortino_snp,2))],
                   ["Maximum Drawdown", "{}%".format(round(max_drawdown_port*100,2)), "{}%".format(round(max_drawdown_snp*100,2))]], 
                  headers = ["Metrics", "Strategy", "S&P 500"])
    print(tab)
    
    # Compute t-stats
    final = final.dropna()
    tstat, pval = scs.ttest_ind(final["return"], final["benchmark_ret"], equal_var=False)
    print("t-statistic of difference in annual returns: {}(p={})".format(tstat.round(4), pval.round(4)))

#%%
##############################################################################
# 6. Strategy 1: Long-Only Portfolio
long_only = long_only_strategy(df5, 0.4, 0.90, 8)
print(long_only.tail())
plot_cum_return(long_only, "l", "long_only.jpg")
compute_metrics(long_only)

#%%
##############################################################################
# 7. Strategy 2: Long-Only 
long_short = long_short_strategy(df5, 0.4, 0.90, 8, 0.10, 1)
print(long_short.tail())
plot_cum_return(long_short, "ls", "long_short.jpg")
compute_metrics(long_short)

##############################################################################
# 8. Strategy 3: Optimized Long-Only 
long_short_opt = long_short_strategy(df5, 0.35, 0.95, 9, 0.05, 2)
print(long_short_opt.tail())
plot_cum_return(long_short_opt, "ls", "long_short_opt.jpg")
compute_metrics(long_short_opt)

##############################################################################
# Optional: output comparison chart of cumulative returns
#long_ret = float(long_only["strat_$_return"].tail(1))
#long_short_ret = float(long_short["strat_$_return"].tail(1))
#long_short_opt_ret = float(long_short_opt["strat_$_return"].tail(1))
#snp_ret = float(long_only["snp_$_return"].tail(1))

#fig, ax = plt.subplots(figsize=(16,9))
#ax.set_ylabel("Cumulative return $", fontsize=15)
#ax.set_xlabel("Year", fontsize=15)
#min_year = min(long_only["pfy"].dt.year)
#max_year = max(long_only["pfy"].dt.year)

#style = dict(size=15, color='red')
#ax.set_title("Annual investment performance of Strategies and S&P500 {}-{}".format(min_year, max_year), fontsize=20)
#ax.plot(long_only["pfy"], long_only["strat_$_return"], label="Long-Only Strategy")
#ax.plot(long_only["pfy"], long_short["strat_$_return"], label="Long/Short Strategy")
#ax.plot(long_only["pfy"], long_short_opt["strat_$_return"], label="Optimized Long/Short Strategy")
#ax.plot(long_only["pfy"], long_only["snp_$_return"], label="S&P500")
#ax.text(pd.Timestamp("2019-01-01"), long_only["strat_$_return"].tail(1)+100, "Long-Only Strategy: ${:,.2f}".format(long_ret), ha='right', **style)
#ax.text(pd.Timestamp("2019-01-01"), long_short["strat_$_return"].tail(1)+2000, "Long/Short Strategy: ${:,.2f}".format(long_short_ret), ha='right', **style)
#ax.text(pd.Timestamp("2019-01-01"), long_short_opt["strat_$_return"].tail(1)+100, "Optimized Long/Short Strategy: ${:,.2f}".format(long_short_opt_ret), ha='right', **style)
#ax.text(pd.Timestamp("2019-01-01"), long_only["snp_$_return"].tail(1)-3000, "S&P500: ${:,.2f}".format(snp_ret), ha='right', **style)
#ax.legend(prop={'size': 15})
#plt.savefig("Long-only vs Long-short.jpg")
