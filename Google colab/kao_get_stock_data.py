
def kao_get_stock_data(s_n,s_d,e_d=[]):
    from FinMind.data import DataLoader
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from talib import abstract
    dl = DataLoader()
    stock_get = dl.taiwan_stock_daily(stock_id=s_n, start_date=s_d,end_date=e_d)
    stock_get.set_index("date" , inplace=True)
    stock_get = stock_get.set_index(pd.DatetimeIndex(pd.to_datetime(stock_get.index)))

    data_stock_get = stock_get[["open","max","min","close","Trading_Volume",'spread','Trading_turnover','Trading_money']]
    #原本資料有的 7個 columns 

    # 改成 TA-Lib 可以辨識的欄位名稱
    ad = '_'+s_n
    data_stock_get.columns = ['open','high','low','close','volume','spread','Trading_turnover','Trading_money']
    
    abnormal = np.where(data_stock_get['open'] == 0)[0].tolist() #[57, 318, 422]
    repair_last = np.sum([abnormal,[-1] * len(abnormal )],axis=0).tolist() #[56, 317, 421]
    data_stock_get.iloc[abnormal] = data_stock_get.iloc[repair_last]
    data_stock_get[data_stock_get['open'] == 0]

    data_stock_get= data_stock_get.ffill()
    # 將原本的 7個 columns 重新命名

    ta_list = ['MACD','RSI','MOM','STOCH']
    # 快速計算與整理因子 新增了 6個技術指標

    for x in ta_list:
        output = eval('abstract.'+x+'(data_stock_get)')
        output.name = x.lower() if type(output) == pd.core.series.Series else None
        data_stock_get = pd.merge(data_stock_get, pd.DataFrame(output), left_on = data_stock_get.index, right_on = output.index)
        data_stock_get = data_stock_get.set_index('key_0')

    data_stock_get["rate"] = 100*data_stock_get["spread"] / data_stock_get["close"].shift(1)
    data_stock_get["hspread"] = 100*(data_stock_get["high"]-data_stock_get["close"].shift(1)) / data_stock_get["close"].shift(1)
    data_stock_get["lspread"] = 100*(data_stock_get["low"]-data_stock_get["close"].shift(1)) / data_stock_get["close"].shift(1)

    data_stock_get["spread"] = data_stock_get["rate"]
    data_stock_get= data_stock_get.drop(columns=['rate'])

    data_stock_get.columns = data_stock_get.columns + ad #增加個股欄位名稱
    data_stock_get = data_stock_get.dropna() # 前33 日有些技術指標 算不出來 去除 = 4649 - 33 = 4617
    
    return data_stock_get
