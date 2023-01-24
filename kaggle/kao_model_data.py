from kao_get_stock_data import kao_get_stock_data as kao_get_stock_data
from kao_get_Taiwan_stock_data import kao_get_Taiwan_stock_data as kao_get_Taiwan_stock_data
from kao_com_foreign_stock_data import kao_com_foreign_select_data as kao_com_foreign_select_data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tqdm
import os

def Y(train_y,test_y,Total_y,day_mean,n):    
    new_train_y = []
    new_test_y = []
    new_Total_y = []    
    for i in range(0,len(train_y)-day_mean+1): 
        new_train_y.append(np.mean(pd.DataFrame(train_y).iloc[i:i+day_mean])) #現有資料+30天的Y
    for j in range(1,day_mean): 
        new_train_y.append(np.mean(pd.DataFrame(train_y).iloc[len(train_y)-day_mean+j:len(train_y)]))

    for i in range(0,len(test_y)-day_mean+1): 
        new_test_y.append(np.mean(pd.DataFrame(test_y).iloc[i:i+day_mean])) #現有資料+30天的Y
    for j in range(1,day_mean): 
        new_test_y.append(np.mean(pd.DataFrame(test_y).iloc[len(test_y)-day_mean+j:len(test_y)]))

    for i in range(0,len(Total_y)-day_mean+1):
        new_Total_y.append(np.mean(Total_y.iloc[i:i+day_mean])) #現有資料+30天的Y
    for j in range(1,day_mean):  
        new_Total_y.append(np.mean(Total_y.iloc[len(Total_y)-day_mean+j:len(Total_y)]))

    return {"new_train_y":np.array(new_train_y),"new_test_y":np.array(new_test_y),"new_Total_y":np.array(new_Total_y)}

def kao_model_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean,e_d=[]):
    
    data = kao_com_foreign_select_data(stock_sets,s_d,features,e_d)
    scaler = MinMaxScaler()
    data = data.dropna() # 前33 日有些技術指標 算不出來 去除 = 4649 - 33 = 4617
    data_scaler =  pd.DataFrame(scaler.fit_transform(data), columns=data.columns).copy()
    #data_scaler = data.copy()
    split_point = int(len(data)*split_p) # 決定切割比例為 70%:30%
    train = data_scaler.iloc[:split_point,:].copy()  
    test = data_scaler.iloc[split_point-n:,:].copy() 
    print('去除掉遺失值data:',data.shape)      # 一開始 全部資料
    print('拆成 train 個數:',train.shape)     # 拆成 train 個數 970 
    print('test從train - n + 1 開始取:',test.shape)      # 但由於 取 n 觀察日 僅剩 train - n 940 的資料，但是 test X 可以 從 train - n + 1 開始取  941~1022 = 82

    train_spread  = data.iloc[:split_point,:][target_spread].copy()  
    train_spread  = pd.DataFrame(train_spread)

    test_spread  = data.iloc[split_point-n:,:][target_spread].copy() 
    test_spread  = pd.DataFrame(test_spread)
    
    feature_names = list(train.columns)
    train_X = []
    train_y = []
    indexes = []
    norm_data_x = train[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(train)-n)): 
      train_X.append(norm_data_x.iloc[i:i+n]. values) 
      train_y.append(train_spread.iloc[i+n]) #現有資料+30天的Y
      indexes.append(train.index[i+n-1]) #Y的日期
    train_X=np.array(train_X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
    train_y=np.array(train_y)

    feature_names = list(test.columns)
    test_X = []
    test_y = []
    indexes = []
    norm_data_x = test[feature_names]
    for i in tqdm.notebook.tqdm(range(0,len(test)-n)): 
      test_X.append(norm_data_x.iloc[i:i+n].values) 
      test_y.append(test_spread.iloc[i+n]) #現有資料+30天的Y
      indexes.append(test.index[i+n-1]) #Y的日期
    test_X=np.array(test_X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
    test_y=np.array(test_y)

    Total_y = data[target_spread]

    out_test = pd.DataFrame(test).copy()
    out_test.index = Total_y.index[split_point-n:]
    out_test_y = pd.DataFrame(test_y,columns = [target_spread[i]+'_t' for i in range(0,len(target_spread))],index = Total_y.index[split_point:])
    out_test_all = out_test.merge(out_test_y,how='left',right_index = True,left_index = True)

    out_train = pd.DataFrame(train).copy()
    out_train.index = Total_y.index[:split_point]
    out_train_y = pd.DataFrame(train_y,columns = [target_spread[i]+'_t' for i in range(0,len(target_spread))],index = Total_y.index[n:split_point])
    out_train_all = out_train.merge(out_train_y,how='left',right_index = True,left_index = True)

    predict_X = []
    predict_X.append(norm_data_x.iloc[len(test)-n:len(test)+1].values) 
    predict_X = np.array(predict_X)

    os.chdir('/kaggle/working')
    out_test_all.to_csv('out_test_all_now.csv')
    out_train_all.to_csv('out_train_all_now.csv')

    future3 = Y(train_y,test_y,Total_y,3,n)
    future5 = Y(train_y,test_y,Total_y,5,n)


    train_Y = np.concatenate([train_y,future3['new_train_y'],future5['new_train_y']], axis=1)
    test_Y = np.concatenate([test_y,future3['new_test_y'],future5['new_test_y']], axis=1)
    new_Total_Y = np.concatenate([Total_y,future3['new_Total_y'],future5['new_Total_y']], axis=1)

    conames = target_spread
    for i in range (0,len(target_spread)):
        conames = conames+ [target_spread[i]+'_f3']
    for i in range (0,len(target_spread)):    
        conames = conames+ [target_spread[i]+'_f5']

    new_Total_Y = pd.DataFrame(new_Total_Y,columns = conames,index = Total_y.index)

    return {
   "train_X": train_X,"test_X": test_X,"train": train,"test": test,"train_y": train_Y,"test_y": test_Y,"predict_X":predict_X,
    "Total_y":Total_y,"new_Total_y":new_Total_Y,
    "train_y_spread":train_y,"test_y_spread":test_y,
    "out_train_all":out_train_all,"out_test_all":out_test_all}    

   
def kao_model_pca_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean,components,e_d=[],cols=[]):

    from sklearn.decomposition import PCA
    
    data = kao_com_foreign_select_data(stock_sets,s_d,features,e_d)
    scaler = MinMaxScaler()
    data = data.dropna() # 前33 日有些技術指標 算不出來 去除 = 4649 - 33 = 4617
    
    split_point = int(len(data)*split_p) # 決定切割比例為 70%:30%

    if cols == []:
        data_scaler =  pd.DataFrame(scaler.fit_transform(data), columns=data.columns).copy()
    else:
        cols2 = [cols[i] -1 for i in range(0,len(cols))]
        data_scaler =  pd.DataFrame(scaler.fit_transform(data), columns=data.columns).copy()
        data_scaler =  data_scaler.iloc[:,cols2]
        
    train = data_scaler.iloc[:split_point,:].copy()  
    test = data_scaler.iloc[split_point-n:,:].copy() 
    print('去除掉遺失值data:',data.shape)      # 一開始 全部資料
    print('拆成 train 個數:',train.shape)     # 拆成 train 個數 970 
    print('test從train - n + 1 開始取:',test.shape)      # 但由於 取 n 觀察日 僅剩 train - n 940 的資料，但是 test X 可以 從 train - n + 1 開始取  941~1022 = 82

    train_spread  = data.iloc[:split_point,:][target_spread].copy()  
    train_spread  = pd.DataFrame(train_spread)

    test_spread  = data.iloc[split_point-n:,:][target_spread].copy() 
    test_spread  = pd.DataFrame(test_spread)

    ###################################################
    pca_model = PCA(n_components=components)
    # 將資料集輸入模型
    pca_model.fit(train)
    # 對資料集進行轉換對映
    train_pca = pd.DataFrame(pca_model.transform(train))
    test_pca = pd.DataFrame(pca_model.transform(test))
    
    # 獲得轉換後的所有主成分
    components = pca_model.components_
    # 獲得各主成分的方差
    components_var = pca_model.explained_variance_
    # 獲取主成分的方差佔比
    components_var_ratio = pca_model.explained_variance_ratio_

    print('解釋變異:',100*np.round(components_var_ratio,3))
    print('總解釋變異:',round(sum(100*components_var_ratio),3))

    feature_names = list(train_pca.columns)

    train_X = []
    train_y = []
    indexes = []
    norm_data_x = train_pca[feature_names]

    for i in tqdm.tqdm_notebook(range(0,len(train)-n)): 
        train_X.append(norm_data_x.iloc[i:i+n].values) 
        train_y.append(train_spread.iloc[i+n]) #現有資料+30天的Y
        indexes.append(train.index[i+n-1]) #Y的日期
    train_X=np.array(train_X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
    train_y=np.array(train_y)

    feature_names = list(test_pca.columns)

    test_X = []
    test_y = []
    indexes = []
    norm_data_x = test_pca[feature_names]

    for i in tqdm.notebook.tqdm(range(0,len(test)-n)): 
        test_X.append(norm_data_x.iloc[i:i+n].values) 
        test_y.append(test_spread.iloc[i+n]) #現有資料+30天的Y
        indexes.append(test.index[i+n-1]) #Y的日期
    test_X=np.array(test_X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
    test_y=np.array(test_y)

    predict_X = []
    predict_X.append(norm_data_x.iloc[len(test)-n:len(test)+1].values) 
    predict_X = np.array(predict_X)

    ##########################################################################

    Total_y = data[target_spread]

    out_test = pd.DataFrame(test).copy()
    out_test.index = Total_y.index[split_point-n:]
    out_test_y = pd.DataFrame(test_y,columns = [target_spread[i]+'_t' for i in range(0,len(target_spread))],index = Total_y.index[split_point:])
    out_test_all = out_test.merge(out_test_y,how='left',right_index = True,left_index = True)

    out_train = pd.DataFrame(train).copy()
    out_train.index = Total_y.index[:split_point]
    out_train_y = pd.DataFrame(train_y,columns = [target_spread[i]+'_t' for i in range(0,len(target_spread))],index = Total_y.index[n:split_point])
    out_train_all = out_train.merge(out_train_y,how='left',right_index = True,left_index = True)

    os.chdir('/kaggle/working')
    out_test_all.to_csv('out_test_all_now.csv')
    out_train_all.to_csv('out_train_all_now.csv')

    future2 = Y(train_y,test_y,Total_y,2,n)
    future3 = Y(train_y,test_y,Total_y,3,n)


    train_Y = np.concatenate([train_y,future2['new_train_y']], axis=1)
    test_Y = np.concatenate([test_y,future2['new_test_y']], axis=1)
    new_Total_Y = np.concatenate([Total_y,future2['new_Total_y']], axis=1)

    conames = target_spread
    for i in range (0,len(target_spread)):
        conames = conames+ [target_spread[i]+'_f2']
    #for i in range (0,len(target_spread)):    
    #    conames = conames+ [target_spread[i]+'_f3']

    new_Total_Y = pd.DataFrame(new_Total_Y,columns = conames,index = Total_y.index)

    return {
   "train_X": train_X,"test_X": test_X,"train": train,"test": test,"train_y": train_Y,"test_y": test_Y,"predict_X":predict_X,
    "Total_y":Total_y,"new_Total_y":new_Total_Y,
    "train_y_spread":train_y,"test_y_spread":test_y,
    "out_train_all":out_train_all,"out_test_all":out_test_all}

def kao_model_origin_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean,e_d=[]):
    
    data = kao_com_foreign_select_data(stock_sets,s_d,features,e_d)
    scaler = MinMaxScaler()
    data = data.dropna() # 前33 日有些技術指標 算不出來 去除 = 4649 - 33 = 4617
    #data_scaler =  pd.DataFrame(scaler.fit_transform(data), columns=data.columns).copy()
    data_scaler = data.copy()
    #data_scaler = data.copy()
    split_point = int(len(data)*split_p) # 決定切割比例為 70%:30%
    train = data_scaler.iloc[:split_point,:].copy()  
    test = data_scaler.iloc[split_point-n:,:].copy() 
    print('去除掉遺失值data:',data.shape)      # 一開始 全部資料
    print('拆成 train 個數:',train.shape)     # 拆成 train 個數 970 
    print('test從train - n + 1 開始取:',test.shape)      # 但由於 取 n 觀察日 僅剩 train - n 940 的資料，但是 test X 可以 從 train - n + 1 開始取  941~1022 = 82

    train_spread  = data.iloc[:split_point,:][target_spread].copy()  
    train_spread  = pd.DataFrame(train_spread)

    test_spread  = data.iloc[split_point-n:,:][target_spread].copy() 
    test_spread  = pd.DataFrame(test_spread)
    
    feature_names = list(train.columns)
    train_X = []
    train_y = []
    indexes = []
    norm_data_x = train[feature_names]
    for i in tqdm.tqdm_notebook(range(0,len(train)-n)): 
      train_X.append(norm_data_x.iloc[i:i+n]. values) 
      train_y.append(train_spread.iloc[i+n]) #現有資料+30天的Y
      indexes.append(train.index[i+n-1]) #Y的日期
    train_X=np.array(train_X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
    train_y=np.array(train_y)

    feature_names = list(test.columns)
    test_X = []
    test_y = []
    indexes = []
    norm_data_x = test[feature_names]
    for i in tqdm.notebook.tqdm(range(0,len(test)-n)): 
      test_X.append(norm_data_x.iloc[i:i+n].values) 
      test_y.append(test_spread.iloc[i+n]) #現有資料+30天的Y
      indexes.append(test.index[i+n-1]) #Y的日期
    test_X=np.array(test_X) # RD外的6個自變量，記憶體=30，EX:預測12/31的Y，用12/1~12/30的X
    test_y=np.array(test_y)

    Total_y = data[target_spread]

    out_test = pd.DataFrame(test).copy()
    out_test.index = Total_y.index[split_point-n:]
    out_test_y = pd.DataFrame(test_y,columns = [target_spread[i]+'_t' for i in range(0,len(target_spread))],index = Total_y.index[split_point:])
    out_test_all = out_test.merge(out_test_y,how='left',right_index = True,left_index = True)

    out_train = pd.DataFrame(train).copy()
    out_train.index = Total_y.index[:split_point]
    out_train_y = pd.DataFrame(train_y,columns = [target_spread[i]+'_t' for i in range(0,len(target_spread))],index = Total_y.index[n:split_point])
    out_train_all = out_train.merge(out_train_y,how='left',right_index = True,left_index = True)

    predict_X = []
    predict_X.append(norm_data_x.iloc[len(test)-n:len(test)+1].values) 
    predict_X = np.array(predict_X)

    os.chdir('/kaggle/working')
    out_test_all.to_csv('out_test_all_now.csv')
    out_train_all.to_csv('out_train_all_now.csv')

    future2 = Y(train_y,test_y,Total_y,2,n)
    future3 = Y(train_y,test_y,Total_y,3,n)


    train_Y = np.concatenate([train_y,future2['new_train_y']], axis=1)
    test_Y = np.concatenate([test_y,future2['new_test_y']], axis=1)
    new_Total_Y = np.concatenate([Total_y,future2['new_Total_y']], axis=1)

    conames = target_spread
    for i in range (0,len(target_spread)):
        conames = conames+ [target_spread[i]+'_f2']
    #for i in range (0,len(target_spread)):    
    #    conames = conames+ [target_spread[i]+'_f3']

    new_Total_Y = pd.DataFrame(new_Total_Y,columns = conames,index = Total_y.index)

    return {
   "train_X": train_X,"test_X": test_X,"train": train,"test": test,"train_y": train_Y,"test_y": test_Y,"predict_X":predict_X,
    "Total_y":Total_y,"new_Total_y":new_Total_Y,
    "train_y_spread":train_y,"test_y_spread":test_y,
    "out_train_all":out_train_all,"out_test_all":out_test_all}
