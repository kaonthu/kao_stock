from kao_get_stock_data import kao_get_stock_data as kao_get_stock_data
from kao_get_Taiwan_stock_data import kao_get_Taiwan_stock_data as kao_get_Taiwan_stock_data
from kao_com_foreign_stock_data import kao_com_foreign_select_data as kao_com_foreign_select_data
from kao_model_data import kao_model_data  as kao_model_data 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

def kao_tree_data(target_spread,train_X,train_y,test_X,test_y,Total_y,out_train_all,train_predictions_frame,percent):

    n= train_X.shape[1]

    data_tree_x = out_train_all.iloc[n-1:-1].drop(columns=[target_spread[i] + '_t' for i in range(0,len(target_spread))])
    data_tree_y = pd.DataFrame(train_y,index = data_tree_x.index,columns = Total_y.columns)

    train_predictions_frame.index = data_tree_x.index 

    tree_data_dis = train_predictions_frame - np.array(data_tree_y)
    tree_data_dis.columns = [Total_y.columns[i] + '_dis' for i in range(0,len(Total_y.columns))]

    tree_data =  data_tree_x.merge(train_predictions_frame,how='left',right_index = True,left_index = True)
    tree_data['tree_data_dis_s'] = abs(tree_data_dis).sum(axis = 1)

    tree_data['tree_data_dis_type'] = tree_data['tree_data_dis_s'] > np.percentile(tree_data['tree_data_dis_s'],percent)
    tree_data['tree_data_dis_type'] = tree_data['tree_data_dis_type']+0

    tree_data['tree_data_dis_weight'] = tree_data['tree_data_dis_type']
    tree_data_dis_type_0 = tree_data['tree_data_dis_s'][tree_data['tree_data_dis_type']==0].index
    tree_data_dis_type_1 = tree_data['tree_data_dis_s'][tree_data['tree_data_dis_type']==1].index
    tree_data['tree_data_dis_weight'][tree_data_dis_type_0] = tree_data['tree_data_dis_s'][tree_data_dis_type_0].rank(ascending=False)/len(tree_data_dis_type_0)
    tree_data['tree_data_dis_weight'][tree_data_dis_type_1] = tree_data['tree_data_dis_s'][tree_data_dis_type_1].rank()/len(tree_data_dis_type_1)

    train_X_tree = tree_data.drop(columns=['tree_data_dis_type','tree_data_dis_s','tree_data_dis_weight'])
    train_y_tree = tree_data[['tree_data_dis_type']]

    return {'tree_data':tree_data,'train_X_tree':train_X_tree,'train_y_tree':train_y_tree}

def kao_tree_stock_data(stock_sets,target_spread,train_X,train_y,test_X,test_y,Total_y,out_train_all,train_predictions_frame,percent,factor=[]):

    n= train_X.shape[1]

    data_tree_x = out_train_all.iloc[n-1:-1].drop(columns=[target_spread[i] + '_t' for i in range(0,len(target_spread))])
    data_tree_y = pd.DataFrame(train_y,index = data_tree_x.index,columns = Total_y.columns)

    train_predictions_frame.index = data_tree_x.index 

    tree_data_stock = pd.DataFrame(columns = list(data_tree_x.columns)+['stock','stock_t','real','prediction'])

    for j in range(0,len(stock_sets)):
        
        matching = [s for s in list(data_tree_y.columns) if stock_sets[j] in s]

        for i in range(0,len(matching)):
            tree_data_1 =  data_tree_x.copy()
            tree_data_1['stock'] = stock_sets[j]
            tree_data_1['stock_t'] = matching[i]

            data_tree_y1 = data_tree_y[[matching[i]]]
            data_tree_y1.rename(columns = {matching[i]:'real'}, inplace = True)

            tree_data_1 = tree_data_1.merge(data_tree_y1,how='left',right_index = True,left_index = True)
            tree_data_1 = tree_data_1.merge(train_predictions_frame[matching[i] + '預測值'],how='left',right_index = True,left_index = True)
            tree_data_1.rename(columns = {matching[i] + '預測值':'prediction'}, inplace = True)
            tree_data_stock = pd.concat([tree_data_stock,tree_data_1])

    #tree_data_stock.loc['2015-03-09']
    #全部混合在一起算dis 然後分一半
    tree_data_stock['dis'] = abs(tree_data_stock['prediction'] - tree_data_stock['real'])
    tree_data_stock['dis_type'] = tree_data_stock['dis'] > np.percentile(tree_data_stock['dis'],percent)
    tree_data_stock['dis_type'] = tree_data_stock['dis_type']+0

    print('0:優質點,1:劣質點',tree_data_stock.groupby('dis_type').size())
    #重新設 index
    #tree_data_stock.reset_index(inplace=True) 
    # 優質點與劣質點各算weight 
    tree_data_stock['tree_data_dis_weight'] = tree_data_stock['dis_type']
    tree_data_stock_dis_type_0 = tree_data_stock['dis'][tree_data_stock['dis_type']==0].index #被分好優質點的位置
    tree_data_stock_dis_type_1 = tree_data_stock['dis'][tree_data_stock['dis_type']==1].index #被分到劣質點的位置
    tree_data_stock['tree_data_dis_weight'][tree_data_stock_dis_type_0] = tree_data_stock['dis'][tree_data_stock_dis_type_0].rank(ascending=False)/len(tree_data_stock_dis_type_0)
    tree_data_stock['tree_data_dis_weight'][tree_data_stock_dis_type_1] = tree_data_stock['dis'][tree_data_stock_dis_type_1].rank()/len(tree_data_stock_dis_type_1)

    tree_data_stock['YEAR'] = [tree_data_stock.index[i].strftime("%Y") for i in range(0,len(tree_data_stock.index))]
    #tree_data_stock['MONTH'] = [tree_data_stock['Date'][i].strftime("%m") for i in range(0,len(tree_data_stock['Date']))]
    tree_data_stock = tree_data_stock.astype({'YEAR': int})

    if factor == []:
        train_X_tree = tree_data_stock.drop(columns=['dis_type','dis','real','tree_data_dis_weight'])
    else:
        train_X_tree = tree_data_stock[factor]

    train_stock = tree_data_stock['stock']

    train_y_tree = tree_data_stock[['dis_type']]

    return {'tree_data_stock':tree_data_stock,'train_X_tree':train_X_tree,'train_y_tree':train_y_tree}    

def k_tree_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean,train_predictions_frame,percent):

    data = kao_model_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean)

    Total_y = data['new_Total_y']
    train_X = data["train_X"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    data_tree_x = data["out_train_all"].iloc[n-1:-1].drop(columns=[target_spread[i] + '_t' for i in range(0,len(target_spread))])
    data_tree_y = pd.DataFrame(train_y,index = data_tree_x.index,columns = Total_y.columns)

    train_predictions_frame.index = data_tree_x.index 

    tree_data_dis = train_predictions_frame - np.array(data_tree_y)
    tree_data_dis.columns = [Total_y.columns[i] + '_dis' for i in range(0,len(Total_y.columns))]

    tree_data =  data_tree_x.merge(train_predictions_frame,how='left',right_index = True,left_index = True)
    tree_data['tree_data_dis_s'] = abs(tree_data_dis).sum(axis = 1)

    tree_data['tree_data_dis_type'] = tree_data['tree_data_dis_s'] > np.percentile(tree_data['tree_data_dis_s'],percent)
    tree_data['tree_data_dis_type'] = tree_data['tree_data_dis_type']+0

    tree_data['tree_data_dis_weight'] = tree_data['tree_data_dis_type']
    tree_data_dis_type_0 = tree_data['tree_data_dis_s'][tree_data['tree_data_dis_type']==0].index
    tree_data_dis_type_1 = tree_data['tree_data_dis_s'][tree_data['tree_data_dis_type']==1].index
    tree_data['tree_data_dis_weight'][tree_data_dis_type_0] = tree_data['tree_data_dis_s'][tree_data_dis_type_0].rank(ascending=False)/len(tree_data_dis_type_0)
    tree_data['tree_data_dis_weight'][tree_data_dis_type_1] = tree_data['tree_data_dis_s'][tree_data_dis_type_1].rank()/len(tree_data_dis_type_1)

    train_X_tree = tree_data.drop(columns=['tree_data_dis_type','tree_data_dis_s','tree_data_dis_weight'])
    train_y_tree = tree_data[['tree_data_dis_type']]

    return {'tree_data':tree_data,'train_X_tree':train_X_tree,'train_y_tree':train_y_tree}


def k_tree_stock_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean,train_predictions_frame,percent,factor=[]):

    data = kao_model_data(target_spread,stock_sets,s_d,features,split_p,n,day_mean)

    Total_y = data['new_Total_y']
    train_X = data["train_X"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    data_tree_x = data["out_train_all"].iloc[n-1:-1].drop(columns=[target_spread[i] + '_t' for i in range(0,len(target_spread))])
    data_tree_y = pd.DataFrame(train_y,index = data_tree_x.index,columns = Total_y.columns)

    train_predictions_frame.index = data_tree_x.index 

    tree_data_stock = pd.DataFrame(columns = list(data_tree_x.columns)+['stock','stock_t','real','prediction'])

    for j in range(0,len(stock_sets)):
        
        matching = [s for s in list(data_tree_y.columns) if stock_sets[j] in s]

        for i in range(0,len(matching)):
            tree_data_1 =  data_tree_x.copy()
            tree_data_1['stock'] = stock_sets[j]
            tree_data_1['stock_t'] = matching[i]

            data_tree_y1 = data_tree_y[[matching[i]]]
            data_tree_y1.rename(columns = {matching[i]:'real'}, inplace = True)

            tree_data_1 = tree_data_1.merge(data_tree_y1,how='left',right_index = True,left_index = True)
            tree_data_1 = tree_data_1.merge(train_predictions_frame[matching[i] + '預測值'],how='left',right_index = True,left_index = True)
            tree_data_1.rename(columns = {matching[i] + '預測值':'prediction'}, inplace = True)
            tree_data_stock = pd.concat([tree_data_stock,tree_data_1])

    #tree_data_stock.loc['2015-03-09']
    #全部混合在一起算dis 然後分一半
    tree_data_stock['dis'] = abs(tree_data_stock['prediction'] - tree_data_stock['real'])
    tree_data_stock['dis_type'] = tree_data_stock['dis'] > np.percentile(tree_data_stock['dis'],percent)
    tree_data_stock['dis_type'] = tree_data_stock['dis_type']+0

    print('0:優質點,1:劣質點',tree_data_stock.groupby('dis_type').size())
    #重新設 index
    #tree_data_stock.reset_index(inplace=True) 
    # 優質點與劣質點各算weight 
    tree_data_stock['tree_data_dis_weight'] = tree_data_stock['dis_type']
    tree_data_stock_dis_type_0 = tree_data_stock['dis'][tree_data_stock['dis_type']==0].index #被分好優質點的位置
    tree_data_stock_dis_type_1 = tree_data_stock['dis'][tree_data_stock['dis_type']==1].index #被分到劣質點的位置
    tree_data_stock['tree_data_dis_weight'][tree_data_stock_dis_type_0] = tree_data_stock['dis'][tree_data_stock_dis_type_0].rank(ascending=False)/len(tree_data_stock_dis_type_0)
    tree_data_stock['tree_data_dis_weight'][tree_data_stock_dis_type_1] = tree_data_stock['dis'][tree_data_stock_dis_type_1].rank()/len(tree_data_stock_dis_type_1)

    tree_data_stock['YEAR'] = [tree_data_stock.index[i].strftime("%Y") for i in range(0,len(tree_data_stock.index))]
    #tree_data_stock['MONTH'] = [tree_data_stock['Date'][i].strftime("%m") for i in range(0,len(tree_data_stock['Date']))]
    tree_data_stock = tree_data_stock.astype({'YEAR': int})

    if factor == []:
        train_X_tree = tree_data_stock.drop(columns=['dis_type','dis','real','tree_data_dis_weight'])
    else:
        train_X_tree = tree_data_stock[factor]

    train_stock = tree_data_stock['stock']

    train_y_tree = tree_data_stock[['dis_type']]

    return {'tree_data_stock':tree_data_stock,'train_X_tree':train_X_tree,'train_y_tree':train_y_tree}

def kao_tree_analyze(tree_data,train_X_tree,train_y_tree,depth):  
    
    model_tree = DecisionTreeClassifier(max_depth = depth)
    model_tree.fit(train_X_tree, train_y_tree)

    model_tree_weight = DecisionTreeClassifier(max_depth = depth)
    model_tree_weight.fit(train_X_tree, train_y_tree,sample_weight=tree_data['tree_data_dis_weight'])

    model_tree_dot = export_graphviz(model_tree, out_file = None,feature_names = train_X_tree.columns,
                           filled = True, rounded = True,class_names = True,special_characters = True)

    model_tree_dot_weight = export_graphviz(model_tree_weight, out_file = None,feature_names = train_X_tree.columns,
                           filled = True, rounded = True,class_names = True,special_characters = True)

    model_tree_dot_graph = graphviz.Source(model_tree_dot)
    
    model_tree_dot_weight_graph = graphviz.Source(model_tree_dot_weight)

    return {'model_tree':model_tree,'model_tree_weight':model_tree_weight,
            'model_tree_dot_graph':model_tree_dot_graph,'model_tree_dot_weight_graph':model_tree_dot_weight_graph}


def kao_tree_stock_analyze(tree_data_stock,train_X_tree,train_y_tree,depth):

    model_tree_stock = DecisionTreeClassifier(max_depth = depth)

    model_tree_stock_weight = DecisionTreeClassifier(max_depth = depth)

    model_tree_stock.fit(pd.get_dummies(train_X_tree), train_y_tree)

    model_tree_stock_weight = DecisionTreeClassifier(max_depth = depth)

    model_tree_stock_weight.fit(pd.get_dummies(train_X_tree), train_y_tree,sample_weight=tree_data_stock['tree_data_dis_weight'])

    model_tree_stock_dot = export_graphviz(model_tree_stock, out_file = None,feature_names = pd.get_dummies(train_X_tree).columns,
                           filled = True, rounded = True,class_names = True,special_characters = True)

    model_tree_stock_dot_weight = export_graphviz(model_tree_stock_weight, out_file = None,feature_names = pd.get_dummies(train_X_tree).columns,
                           filled = True, rounded = True,class_names = True,special_characters = True)

    model_tree_dot_graph = graphviz.Source(model_tree_stock_dot)
    
    model_tree_dot_weight_graph = graphviz.Source(model_tree_stock_dot_weight)
    
    return {'model_tree_stock':model_tree_stock,'model_tree_stock_weight':model_tree_stock_weight,
            'model_tree_dot_graph':model_tree_dot_graph,'model_tree_dot_weight_graph':model_tree_dot_weight_graph}



    
