import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM,GRU, TimeDistributed, RepeatVector

from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

def kao_analyze(Total_y,train_X,train_y,test_X,test_y,target_spread,
                layer,units,Dropout_p,loss_f,method,epochs_1,batch_size_1,epochs_2,batch_size_2,
                callbacks= EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="min")):
    
    GRU_spread = Sequential() # Initialising the RNN 
    # Adding the first GRU layer and some Dropout regularisation
    if method == "LSTM":
        for i in range(0,layer-1):
            GRU_spread.add(LSTM(units = units, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
            GRU_spread.add(Dropout(Dropout_p))

        # Adding a fourth GRU layer and some Dropout regularisation
        GRU_spread.add(LSTM(units = units))
        GRU_spread.add(Dropout(Dropout_p))

    else:
        for i in range(0,layer-1):
            GRU_spread.add(GRU(units = units, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
            GRU_spread.add(Dropout(Dropout_p))

        # Adding a fourth GRU layer and some Dropout regularisation
        GRU_spread.add(GRU(units = units))
        GRU_spread.add(Dropout(Dropout_p))

    # Adding the output layer
    GRU_spread.add(Dense(units = test_y.shape[1]))

    # Compiling
    GRU_spread.compile(optimizer = 'adam', loss = loss_f)

    GRU_spread_history =  GRU_spread.fit(train_X, train_y, epochs = epochs_1, batch_size = batch_size_1,
                                         validation_data=(test_X, test_y),callbacks=[callbacks],verbose = 0)
    
    train_loss_1 = GRU_spread_history.history["loss"][-1]
    test_loss_1 = GRU_spread_history.history["val_loss"][-1]
    
    plt.figure(figsize=(14,4))#全部的資料4616
    plt.title('spread_train_loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.plot(GRU_spread_history.history["loss"])
    plt.plot(GRU_spread_history.history["val_loss"])
    plt.show()

    ###############################################################################################
    if train_y.shape[1] == len(target_spread):
        
        conames = [target_spread[i]+'預測值' for i in range(0,len(target_spread))]
        
    else:
        conames = [target_spread[i]+'預測值' for i in range(0,len(target_spread))]

        for i in range (0,len(target_spread)):
            conames = conames+ [target_spread[i]+'_f2預測值']
        #for i in range (0,len(target_spread)):    
        #    conames = conames+ [target_spread[i]+'_f3預測值']
        
    conames2 = [i.replace('預測值','實際值') for i in conames]
    conames3 = [i.replace('預測值','正確率') for i in conames]  

    GRU_spread_train_predictions = GRU_spread.predict(train_X).copy()
    GRU_spread_train_predictions_frame = pd.DataFrame(GRU_spread_train_predictions).copy()
    GRU_spread_train_predictions_frame.columns = conames
    GRU_spread_train_predictions_frame.index = Total_y.index[train_X.shape[1]:train_X.shape[1]+len(train_y)]

    GRU_spread_test_predictions = GRU_spread.predict(test_X)
    GRU_spread_test_predictions_frame = pd.DataFrame(GRU_spread_test_predictions).copy()
    GRU_spread_test_predictions_frame.columns = conames
    GRU_spread_test_predictions_frame.index = Total_y.index[len(train_y)+train_X.shape[1]:len(Total_y)]

    train_X_2 = train_X.copy() #預測更新最新的日期
    train_y_2 = train_y.copy()
    GRU_spread_test_predictions_frame_2 = GRU_spread_test_predictions_frame.copy()
    
    val_loss_process = [GRU_spread_history.history["val_loss"][-1]] # val_loss_process 初始值
    loss_process = [GRU_spread_history.history["loss"][-1]] # loss_process 初始值
    for i in range(0,len(test_y)-1): 
        train_X_2 = np.concatenate([train_X_2,test_X[i].reshape(1,train_X.shape[1],train_X.shape[2])])
        train_y_2 = np.concatenate([train_y_2,test_y[i].reshape(1,test_y.shape[1])])
        GRU_spread_history = GRU_spread.fit(train_X_2, train_y_2, epochs = epochs_2, batch_size = batch_size_2,
                                            validation_data=(test_X[i+1:], test_y[i+1:]), callbacks=[callbacks],verbose = 0)
        GRU_spread_test_predictions_frame_2.iloc[i+1] = GRU_spread.predict(test_X)[i+1]
        val_loss_process.append(GRU_spread_history.history["val_loss"][-1]) # val_loss_process 過程值新增
        loss_process.append(GRU_spread_history.history["loss"][-1]) # loss_process 過程值新增

    #for i in range(0,len(test_y)-1): 
    #print(test_X[i+1:].shape, test_y[i+1:].shape,test_y[i+1:][0]) # test set 的減少 證明

    train_loss_2 = GRU_spread_history.history["loss"][-1]
    test_loss_2 = sum(val_loss_process)/len(val_loss_process) # test_loss_2 顯示最後的模型loss 已無意義，顯示平均

    os.chdir('/kaggle/working')
    GRU_spread.save('layer_'+str(layer)+'units_'+str(units)+'.h5')
    
    #plt.figure(figsize=(12,10))#全部的資料4616
    #plt.title('spread_train_loss')
    #plt.ylabel('loss')
    #plt.xlabel('Epoch')
    #plt.plot(GRU_spread_history.history["loss"])
    #plt.plot(GRU_spread_history.history["val_loss"])
    #plt.show()

    #clear_output()

    GRU_spread_mae = sum(abs(test_y - GRU_spread_test_predictions))/len(test_y) #一開始沒更新最後一日
    GRU_spread_mae_2 = abs(test_y - GRU_spread_test_predictions_frame_2).sum(axis = 0)/len(test_y) #有更新到最後一日

    GRU_spread_mae = pd.DataFrame(GRU_spread_mae,columns = ["mae"])
    GRU_spread_mae_2 = pd.DataFrame(GRU_spread_mae_2,columns = ["mae_2"])

    GRU_spread_mae.index = conames
    GRU_spread_mae_2.index = conames

    final_0 = np.where(test_y<0, 0, 1) - np.where(GRU_spread_test_predictions_frame_2<0, 0, 1)

    conditions = [final_0 <0,final_0 ==0,final_0 >0]
    choices = ["低估", "正確", "高估"]

    final_0 = np.select(conditions, choices, default='未分類')
    final_0 = pd.DataFrame(final_0,columns = conames3)

    GRU_spread_test_predictions_frame_real = pd.concat([GRU_spread_test_predictions_frame_2,
    pd.DataFrame(test_y,columns = conames2,index = GRU_spread_test_predictions_frame_2.index)], axis=1)
    GRU_spread_test_predictions_frame_real['val_loss_process'] = val_loss_process #將過程的 val_loss 寫在這
    GRU_spread_test_predictions_frame_real['loss_process'] = loss_process #將過程的 loss 寫在這 
    
    test_accurary = final_0[0:1].copy()
    test_accurary.index = ["test_accurary"]
    for i in range(0,len(conames3)):
        test_accurary.iloc[0][i] = round(100*final_0[conames3[i]].value_counts()/len(final_0),2)['正確']
        print(conames3[i],round(100*final_0[conames3[i]].value_counts()/len(final_0),2)['正確'])
    
    for i in range(0,len(Total_y.columns)):
        plt.figure(figsize = (16,4)) # 全部的資料 4616
        ax = Total_y.iloc[len(train_y)+train_X.shape[1]:len(Total_y),i].plot(color='black', title=Total_y.columns[i])
        ax = GRU_spread_test_predictions_frame.iloc[:,i].plot(ax=ax, color='red')
        ax = GRU_spread_test_predictions_frame_2.iloc[:,i].plot(ax=ax, color='green')
        plt.show()

    plt.figure(figsize = (14,4)) # 全部的資料 4616
    ax = GRU_spread_test_predictions_frame_real['val_loss_process'].plot(color='green',title= 'val_loss_process')
    ax = GRU_spread_test_predictions_frame_real['loss_process'].plot(color='red')
    plt.show()
    
    return {'T_para':GRU_spread.count_params(),
            'train_loss_1':round(train_loss_1,2),
            'test_loss_1':round(test_loss_1,2),
            'train_loss_2':round(train_loss_2,2),
            'test_loss_2':round(test_loss_2,2),
            'test_accurary':test_accurary,
            'GRU_spread_mae':GRU_spread_mae,
            'GRU_spread_mae_2':GRU_spread_mae_2,
            'GRU_spread_train_predictions_frame':GRU_spread_train_predictions_frame,
            'GRU_spread_test_predictions_frame': GRU_spread_test_predictions_frame,
            'GRU_spread_test_predictions_frame_2': GRU_spread_test_predictions_frame_2,
            'GRU_spread_test_predictions_frame_real': GRU_spread_test_predictions_frame_real
           }

def kao_analyze_no_show(Total_y,train_X,train_y,test_X,test_y,target_spread,
                layer,units,Dropout_p,loss_f,method,epochs_1,batch_size_1,epochs_2,batch_size_2,
                callbacks= EarlyStopping(monitor="val_loss", patience=100, verbose=1, mode="min")):
    
    GRU_spread = Sequential() # Initialising the RNN 
    # Adding the first GRU layer and some Dropout regularisation
    if method == "LSTM":
        for i in range(0,layer-1):
            GRU_spread.add(LSTM(units = units, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
            GRU_spread.add(Dropout(Dropout_p))

        # Adding a fourth GRU layer and some Dropout regularisation
        GRU_spread.add(LSTM(units = units))
        GRU_spread.add(Dropout(Dropout_p))

    else:
        for i in range(0,layer-1):
            GRU_spread.add(GRU(units = units, return_sequences = True, input_shape = (train_X.shape[1], train_X.shape[2])))
            GRU_spread.add(Dropout(Dropout_p))

        # Adding a fourth GRU layer and some Dropout regularisation
        GRU_spread.add(GRU(units = units))
        GRU_spread.add(Dropout(Dropout_p))

    # Adding the output layer
    GRU_spread.add(Dense(units = test_y.shape[1]))

    # Compiling
    GRU_spread.compile(optimizer = 'adam', loss = loss_f)

    GRU_spread_history =  GRU_spread.fit(train_X, train_y, epochs = epochs_1, batch_size = batch_size_1,
                                         validation_data=(test_X, test_y),callbacks=[callbacks],verbose = 0)
    
    train_loss_1 = GRU_spread_history.history["loss"][-1]
    test_loss_1 = GRU_spread_history.history["val_loss"][-1]
    
    #plt.figure(figsize=(14,4))#全部的資料4616
    #plt.title('spread_train_loss')
    #plt.ylabel('loss')
    #plt.xlabel('Epoch')
    #plt.plot(GRU_spread_history.history["loss"])
    #plt.plot(GRU_spread_history.history["val_loss"])
    #plt.show()

    ###############################################################################################
    if train_y.shape[1] == len(target_spread):
        
        conames = [target_spread[i]+'預測值' for i in range(0,len(target_spread))]
        
    else:
        conames = [target_spread[i]+'預測值' for i in range(0,len(target_spread))]

        for i in range (0,len(target_spread)):
            conames = conames+ [target_spread[i]+'_f2預測值']
        #for i in range (0,len(target_spread)):    
        #    conames = conames+ [target_spread[i]+'_f3預測值']
        
    conames2 = [i.replace('預測值','實際值') for i in conames]
    conames3 = [i.replace('預測值','正確率') for i in conames]  

    GRU_spread_train_predictions = GRU_spread.predict(train_X).copy()
    GRU_spread_train_predictions_frame = pd.DataFrame(GRU_spread_train_predictions).copy()
    GRU_spread_train_predictions_frame.columns = conames
    GRU_spread_train_predictions_frame.index = Total_y.index[train_X.shape[1]:train_X.shape[1]+len(train_y)]

    GRU_spread_test_predictions = GRU_spread.predict(test_X)
    GRU_spread_test_predictions_frame = pd.DataFrame(GRU_spread_test_predictions).copy()
    GRU_spread_test_predictions_frame.columns = conames
    GRU_spread_test_predictions_frame.index = Total_y.index[len(train_y)+train_X.shape[1]:len(Total_y)]

    train_X_2 = train_X.copy() #預測更新最新的日期
    train_y_2 = train_y.copy()
    GRU_spread_test_predictions_frame_2 = GRU_spread_test_predictions_frame.copy()
    
    val_loss_process = [GRU_spread_history.history["val_loss"][-1]] # val_loss_process 初始值
    loss_process = [GRU_spread_history.history["loss"][-1]] # loss_process 初始值
    for i in range(0,len(test_y)-1): 
        train_X_2 = np.concatenate([train_X_2,test_X[i].reshape(1,train_X.shape[1],train_X.shape[2])])
        train_y_2 = np.concatenate([train_y_2,test_y[i].reshape(1,test_y.shape[1])])
        GRU_spread_history = GRU_spread.fit(train_X_2, train_y_2, epochs = epochs_2, batch_size = batch_size_2,
                                            validation_data=(test_X[i+1:], test_y[i+1:]), callbacks=[callbacks],verbose = 0)
        GRU_spread_test_predictions_frame_2.iloc[i+1] = GRU_spread.predict(test_X)[i+1]
        val_loss_process.append(GRU_spread_history.history["val_loss"][-1]) # val_loss_process 過程值新增
        loss_process.append(GRU_spread_history.history["loss"][-1]) # loss_process 過程值新增

    #for i in range(0,len(test_y)-1): 
    #print(test_X[i+1:].shape, test_y[i+1:].shape,test_y[i+1:][0]) # test set 的減少 證明

    train_loss_2 = GRU_spread_history.history["loss"][-1]
    test_loss_2 = sum(val_loss_process)/len(val_loss_process) # test_loss_2 顯示最後的模型loss 已無意義，顯示平均

    os.chdir('/kaggle/working')
    GRU_spread.save('layer_'+str(layer)+'units_'+str(units)+'.h5')
    
    #plt.figure(figsize=(12,10))#全部的資料4616
    #plt.title('spread_train_loss')
    #plt.ylabel('loss')
    #plt.xlabel('Epoch')
    #plt.plot(GRU_spread_history.history["loss"])
    #plt.plot(GRU_spread_history.history["val_loss"])
    #plt.show()

    #clear_output()

    GRU_spread_mae = sum(abs(test_y - GRU_spread_test_predictions))/len(test_y) #一開始沒更新最後一日
    GRU_spread_mae_2 = abs(test_y - GRU_spread_test_predictions_frame_2).sum(axis = 0)/len(test_y) #有更新到最後一日

    GRU_spread_mae = pd.DataFrame(GRU_spread_mae,columns = ["mae"])
    GRU_spread_mae_2 = pd.DataFrame(GRU_spread_mae_2,columns = ["mae_2"])

    GRU_spread_mae.index = conames
    GRU_spread_mae_2.index = conames

    final_0 = np.where(test_y<0, 0, 1) - np.where(GRU_spread_test_predictions_frame_2<0, 0, 1)

    conditions = [final_0 <0,final_0 ==0,final_0 >0]
    choices = ["低估", "正確", "高估"]

    final_0 = np.select(conditions, choices, default='未分類')
    final_0 = pd.DataFrame(final_0,columns = conames3)

    GRU_spread_test_predictions_frame_real = pd.concat([GRU_spread_test_predictions_frame_2,
    pd.DataFrame(test_y,columns = conames2,index = GRU_spread_test_predictions_frame_2.index)], axis=1)
    GRU_spread_test_predictions_frame_real['val_loss_process'] = val_loss_process #將過程的 val_loss 寫在這
    GRU_spread_test_predictions_frame_real['loss_process'] = loss_process #將過程的 loss 寫在這 
    
    test_accurary = final_0[0:1].copy()
    test_accurary.index = ["test_accurary"]
    for i in range(0,len(conames3)):
        test_accurary.iloc[0][i] = round(100*final_0[conames3[i]].value_counts()/len(final_0),2)['正確']
    #    print(conames3[i],round(100*final_0[conames3[i]].value_counts()/len(final_0),2)['正確'])
    
    #for i in range(0,len(Total_y.columns)):
    #    plt.figure(figsize = (16,4)) # 全部的資料 4616
    #    ax = Total_y.iloc[len(train_y)+train_X.shape[1]:len(Total_y),i].plot(color='black', title=Total_y.columns[i])
    #    ax = GRU_spread_test_predictions_frame.iloc[:,i].plot(ax=ax, color='red')
    #    ax = GRU_spread_test_predictions_frame_2.iloc[:,i].plot(ax=ax, color='green')
    #    plt.show()

    #plt.figure(figsize = (14,4)) # 全部的資料 4616
    #ax = GRU_spread_test_predictions_frame_real['val_loss_process'].plot(color='green',title= 'val_loss_process')
    #ax = GRU_spread_test_predictions_frame_real['loss_process'].plot(color='red')
    #plt.show()
    
    return {'T_para':GRU_spread.count_params(),
            'train_loss_1':round(train_loss_1,2),
            'test_loss_1':round(test_loss_1,2),
            'train_loss_2':round(train_loss_2,2),
            'test_loss_2':round(test_loss_2,2),
            'test_accurary':test_accurary,
            'GRU_spread_mae':GRU_spread_mae,
            'GRU_spread_mae_2':GRU_spread_mae_2,
            'GRU_spread_train_predictions_frame':GRU_spread_train_predictions_frame,
            'GRU_spread_test_predictions_frame': GRU_spread_test_predictions_frame,
            'GRU_spread_test_predictions_frame_2': GRU_spread_test_predictions_frame_2,
            'GRU_spread_test_predictions_frame_real': GRU_spread_test_predictions_frame_real
           }   
