import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from easydl import clear_output

from kao_get_stock_data import kao_get_stock_data as kao_get_stock_data
from kao_get_Taiwan_stock_data import kao_get_Taiwan_stock_data as kao_get_Taiwan_stock_data

import yfinance as yf

def kao_com_foreign_stock_data(stock_sets,s_d,e_d=[]):
    
    data = kao_get_Taiwan_stock_data(stock_sets,s_d,e_d) #必須要先有感興趣臺灣資料
    df_all = pd.DataFrame(data,index=data.index) #
    
    I_I = ['^GSPC','^DJI','^SOX','^IXIC','^XAX','^RUT','^N100','^N225','^HSI','000001.SS','^KS11','^TWII']
    for x in I_I:
        stk = yf.Ticker(x)
        data_I_I = stk.history(start = s_d)
        data_I_I = data_I_I[['Open','High','Low','Close','Volume']]
        data_I_I['Spread'] = 100*(data_I_I['Close']-data_I_I['Close'].shift(1))/data_I_I['Close'].shift(1)
        data_I_I.columns = data_I_I.columns + x
        df_all = df_all.merge(data_I_I, how='left',right_index = True,left_index = True)
    df_all['Next'] = ((pd.Series(df_all.index).shift(-1) - pd.Series(df_all.index)) / np.timedelta64(1, 'D')).values
    df_all= df_all.drop(columns=['Volume^SOX','Volume^XAX'])
    df_all= df_all.ffill()
    
    print('台股合併外國資料維度:',df_all.shape)
    print('台股合併外國資料欄位:\n',df_all.columns)
    
    return df_all,df_all.columns

def kao_com_foreign_select_data(stock_sets,s_d,features,e_d=[]):
    df_all = kao_com_foreign_stock_data(stock_sets,s_d,e_d)[0]
    data = df_all[features]

    #df_corr = round(data.corr(),1)
    #plt.subplots(figsize=(20, 20)) # 設定畫面大小
    #sns.heatmap(df_corr, annot=True, vmax=1, square=True, cmap="Blues")
    #plt.show()
    
    print('台股合併外國挑選維度:',data .shape)
    print('台股合併外國挑選欄位:\n',data .columns)
    
    return data

def kao_com_foreign_select_corelation(stock_sets,s_d,rho,e_d=[]):
    
    df_all = kao_com_foreign_stock_data(stock_sets,s_d,e_d)[0]

    fix = ["Close^GSPC","Volume^GSPC","Spread^GSPC","Close^XAX","Spread^XAX","Volume^N100","Spread^N100",
     "Volume^N225","Spread^N225","Close^HSI","Volume^HSI","Spread^HSI",
     "Close000001.SS","Volume000001.SS","Spread000001.SS","Volume^KS11","Spread^KS11"]

    auto = ["close_","volume_","spread_","lspread_","hspread_","macd_","macdhist_","rsi_","slowk_"]

    ff = [auto[i] + stock_sets[0] for i in range(0,len(auto))]
    for j in range(1,len(stock_sets)):
        ff = ff + [auto[i] + stock_sets[j] for i in range(1,len(auto))]
        
    features = ["Next"] + ff + fix

    ##################################

    ts = ['spread_','lspread_','hspread_']
    tss = [ts[i] + stock_sets[0] for i in range(0,len(ts))]
    for j in range(1,len(stock_sets)):
        tss = tss + [ts[i] + stock_sets[j] for i in range(0,len(ts))]
   
    data = df_all[features]
    df_corr = round(data.corr(),2)
    clear_output()

    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > rho)]
    feature_d = set(features) - set(to_drop)

    tss = list(set(tss) & set(feature_d))

    if len(set(tss) - set(feature_d)) != 0:
        print("這些變數相關性高於",rho,":",list(set(tss) - set(feature_d)))

    return {'df_corr':df_corr,'feature_d':list(feature_d),'target_spread':tss}





