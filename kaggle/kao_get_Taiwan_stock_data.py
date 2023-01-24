def kao_get_Taiwan_stock_data(stock_sets,s_d,e_d=[]):
    from kao_get_stock_data import kao_get_stock_data as kao_get_stock_data
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    data = kao_get_stock_data(stock_sets[0],s_d,e_d)
    print(stock_sets[0],'資料維度:',data.shape)
    
    for i in range(0,len(stock_sets)-1):
        print(stock_sets[i+1],'資料維度:',kao_get_stock_data(stock_sets[i+1],s_d,e_d).shape)
        data = pd.concat([data,kao_get_stock_data(stock_sets[i+1],s_d,e_d)], axis=1)
    
    #data = data.reset_index().copy()
    #data.rename(columns={'key_0': 'Date'},inplace=True)
    #data.rename(index={'key_0': 'Date'},inplace=True)
    data.index.names = ['Date']
    #print('台股合併資料維度:',data.shape)
    return data
