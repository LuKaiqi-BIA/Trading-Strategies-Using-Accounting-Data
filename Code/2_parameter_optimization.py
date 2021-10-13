#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:02:36 2021

@author: kevinlu
"""

import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")
# Output file is already included, don't need to run this section
#%%
##############################################################################
# Create parameters for Long strategy

# mkval_para = [i for i in np.arange(0.3, 1.0, 0.05)]
# bm_para = [i for i in np.arange(0.3, 1.0, 0.05)]
# f_para = [i for i in range(4, 10, 1)]
# params = [mkval_para, bm_para, f_para]

# param_list = list(itertools.product(*params))
# len_params = len(param_list)

#%%
# def simulation(dfg, param):
#     mkval = param[0]
#     bm_cut = param[1]
#     f_cut = param[2]
    
#     portfolio_return = {"pfy":[],
#                         "mkval":[],
#                         "bm_cut":[],
#                         "f_cut":[],
#                    "return":[]}
    
#     for i, j in dfg.groupby(["pfy"]):
#         df = j.copy()
#         df = df[df["mkvalt"] >= df["mkvalt"].quantile(mkval)]
#         df = df[df["bm"]>= df["bm"].quantile(bm_cut)]
#         df = df[df["f_score"] >= f_cut]

#         portfolio_return["pfy"].append(i)
#         portfolio_return["return"].append(df["yret"].mean())
#         portfolio_return["mkval"].append(mkval)
#         portfolio_return["bm_cut"].append(bm_cut)
#         portfolio_return["f_cut"].append(f_cut)

#     portfolio = pd.DataFrame(portfolio_return)
    
#     final = pd.merge(portfolio, spy_ret, how = "left", on="pfy")
#     final["pfy"] = pd.to_datetime(final["pfy"].astype(str), format="%Y")
#     final["cum_return"] = (final["return"] + 1).cumprod() - 1
#     final["cum_return_snp"] = (final["benchmark_ret"] + 1).cumprod() - 1
#     final = final.dropna()
    
#     return final.tail(1)
#%%
# collection = pd.DataFrame()
# counter = 1
# for i in param_list:
#     collection = collection.append(simulation(df5, i))
#     print("{}/{} completed".format(counter, len_params))
#     counter += 1
# collection.to_csv("long_strategy_simulation.csv")
#%%
long_strat = pd.read_csv("long_strategy_simulation.csv").drop("Unnamed: 0", axis=1)
long_strat["pfy"] = pd.to_datetime(long_strat["pfy"].astype(str))
long_strat_2019 = long_strat[long_strat["pfy"].dt.year == 2019]
long_strat.sort_values("cum_return", ascending=False).head()
len(long_strat)

#%%
##############################################################################
# Create parameter for long/short strategy
# mkval_para = [i for i in np.arange(0.3, 1.0, 0.05)]
# bm_l_para = [i for i in np.arange(0.55, 1.0, 0.05)]
# f_l_para = [i for i in range(5, 10, 1)]
# bm_s_para = [i for i in np.arange(0.05, 0.5, 0.05)]
# f_s_para = [i for i in range(0, 5, 1)]
# params = [mkval_para, bm_l_para, f_l_para, bm_s_para, f_s_para]

# param_list2 = list(itertools.product(*params))
# len_params2 = len(param_list2)
#%%
# def simulation2(dfg, param):
#     mkval = param[0]
#     bm_l_cut = param[1]
#     f_l_cut = param[2]
#     bm_s_cut = param[3]
#     f_s_cut = param[4]
    
#     portfolio_return = {"pfy":[],
#                         "mkval":[],
#                         "bm_l_cut":[],
#                         "f_l_cut":[],
#                         "bm_s_cut":[],
#                         "f_s_cut":[],
#                    "return":[]}
    
#     for i, j in dfg.groupby(["pfy"]):

#         df = j.copy()
#         df = df[df["mkvalt"] >= df["mkvalt"].quantile(mkval)]
#         df_long = df[df["bm"]>= df["bm"].quantile(bm_l_cut)]
#         df_long = df_long[df_long["f_score"] >= f_l_cut]

#         df_short = df[df["bm"]<=df["bm"].quantile(bm_s_cut)]
#         df_short = df_short[df_short["f_score"] <=f_s_cut]
#         df_short["yret"] = -df_short["yret"]

#         df_final = pd.concat([df_long, df_short])
        
#         portfolio_return["pfy"].append(i)
#         portfolio_return["return"].append(df_final["yret"].mean())
#         portfolio_return["mkval"].append(mkval)
#         portfolio_return["bm_l_cut"].append(bm_l_cut)
#         portfolio_return["f_l_cut"].append(f_l_cut)
#         portfolio_return["bm_s_cut"].append(bm_s_cut)
#         portfolio_return["f_s_cut"].append(f_s_cut)

#     portfolio = pd.DataFrame(portfolio_return)
    
#     final = pd.merge(portfolio, spy_ret, how = "left", on="pfy")
#     final["pfy"] = pd.to_datetime(final["pfy"].astype(str), format="%Y")
#     final["cum_return"] = (final["return"] + 1).cumprod() - 1
#     final["cum_return_snp"] = (final["benchmark_ret"] + 1).cumprod() - 1
#     final = final.dropna()
    
#     return final.tail(1)
#%%
# collection2 = pd.DataFrame()
# counter = 1
# for i in param_list2:
#     collection2 = collection2.append(simulation2(df5, i))
#     print("{}/{} completed".format(counter, len_params2))
#     counter += 1
# collection2.to_csv("ls_strategy_simulation.csv")
#%%
ls_strat = pd.read_csv("ls_strategy_simulation.csv").drop("Unnamed: 0", axis=1)
ls_strat["pfy"] = pd.to_datetime(ls_strat["pfy"].astype(str))
ls_strat_2019 = ls_strat[ls_strat["pfy"].dt.year == 2019]
ls_strat.sort_values("cum_return", ascending=False).head()
len(ls_strat)