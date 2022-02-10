def deciling(data,var,asc = 1):
    '''
    equal frequency binning 
    
    sort the dataframe var and then  divide equally in  10 groups
    
    '''
    size = data.shape[0]
    data =data.sort_values(by =var,ascending = asc)
    li = []
    interval =  int(size/10)
    interval
    print(size,interval)
    for i in range(1,11):
        print(i)
        l2 = [i]*interval
        print("----------------",l2[0])
        li = li+l2
    delta = size - len(li)
    li= li+[i]*delta
    print(len(li))
    data["ibin"] = np.array(li)
    return data
  
  
  
  
  def cardinality (data):
    '''calculates number of unique in series.Check this as this may equvalant to nuniqes function'''
    t
    index = []
    card = []
    for col in data.columns:
        index.append(col)
        card.append(data[col].nunique())
    return pd.Series(index=index,data=card)
    
    
def metadata(data):
    '''
    this function depends on cardinality function and it returns  metadata for dataframe provided as \
    input
    
    '''
    meta_data = pd.DataFrame(columns=data.columns,index=["type","desc","missing","miss%","cardinality"])
    meta_data.loc["missing"] =  data.isna().sum()
    meta_data.loc["cardinality"] = cardinality(data)
    meta_data.loc["shape"]= data.shape[0]
    meta_data.loc["miss%"] =meta_data.loc["missing"]/data.shape[0]
    meta_data.loc["type"] = data.dtypes
    meta_data = meta_data.append(data.describe(percentiles=np.linspace(0.1,1,10)))
    meta_data.transpose()
    
    
    def OutlierCap(data,up = 1,down = 99):
    """
    This function caps value at given percentile
    default is 1,99
    
    """
    df=  data.copy()
    variable_outlier_map  = df.describe(percentiles=[up/100,down/100]).transpose()[["{up}%".format(up = str(up)),"{down}%".format(down = str(down))]]
    for var in data.select_dtypes(exclude = "O").columns:
        #print(var)
        lower = variable_outlier_map.loc[var][0]
        upper = variable_outlier_map.loc[var][1]
        #print(lower,upper)
        df[var]  =  df[var].clip(lower,upper)
    return df , variable_outlier_map
        
    
import pandas as pd

# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
def ks(data=None,target=None, prob=None ,score_ = False, n_bins = 10):
    data['target0'] = 1 - data[target]
    data['bucket'],cutbin = pd.qcut(data[prob], n_bins,retbins=True)
    print(cutbin)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    if  score_ == True:
        kstable = kstable.sort_values(by="min_prob", ascending=True).reset_index(drop = True)
    else:
        kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100

    
    
    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,n_bins+1)
    kstable.index.rename('Decile', inplace=True)
    kstable["count"] = kstable["events"]+kstable["nonevents"]
    kstable["decile_flow"] = kstable["events"]/kstable["count"]
    pd.set_option('display.max_columns', 11)
   # print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable ,cutbin)




def ks_apply(data,target, prob ,bin_, extend_boundry = False,score_ = False):
    data['target0'] = 1 - data[target]
    if extend_boundry == True:
        bin_[0] = np.inf
        bin_[len(bin_)-1] = np.inf
    data['bucket'] = pd.cut(data[prob],bins=bin_,include_lowest=True)
    print("cut_bins___\n",data['bucket'])
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    if  score_ == True:
        kstable = kstable.sort_values(by="min_prob", ascending=True).reset_index(drop = True)
    else:
        kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

    
    
    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,len(bin_))
    kstable.index.rename('Decile', inplace=True)
    kstable["count"] = kstable["events"]+kstable["nonevents"]
    kstable["decile_flow"] = kstable["events"]/kstable["count"]
    pd.set_option('display.max_columns', 11)
   # print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)




def prob_to_score(proba):
    '''
    
    Probabilities -series :- Probability of 1
    
'''
    probabilities  =  pd.DataFrame(proba)
    probabilities[ 0 ] = 1 - probabilities["prob"]
    probabilities[ 1 ] = probabilities["prob"]
    probabilities['a'] = np.log(51200)
    probabilities['f'] = np.log(probabilities[0]/(probabilities[1]))
    probabilities['c'] = np.log(0.048828125)
    probabilities['MinMaxAs1'] = 1
    probabilities['min'] = probabilities[['a', 'f']].min(axis = 1)
    probabilities['logOdds'] = probabilities[['min', 'c']].max(axis = 1)
    probabilities['factor'] = 60/np.log(2) #Factor = pdo / ln(2)
    probabilities['constant'] = 700
    probabilities['offset'] = probabilities['constant'] - (probabilities['factor'] * np.log(50)) #Offset = Score — {Factor * ln(Odds)}
    probabilities['score'] = (round(probabilities['offset'] + (probabilities['factor'] * probabilities['logOdds']))).astype(int)
    return probabilities





def convert_to_single_dtype(data):
    '''    convert all  columns value to single dtype'''
    type_ = data.dtypes
    data = data.fillna(999999)
    for col in data.columns:
        if type_[col] == 'O':
            data[col] =data[col].astype('str')
        else:
            data[col] =data[col].astype('float64')
    return data







from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices 
import statsmodels.api as sm

def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1])]
    return(vif)


def iter_calc_vif(data, count, drop_list =  False,bins=None):
    '''depends on score cardpy package
    
    
    iv=iv_from_bins(bins)
    plot = sc.woebin_plot(bins)
    print("{var} variabel found".format(var = len(data.columns)))
    drop_list = []
    while (len(data.columns)>=count):
        top_vif = pd.DataFrame(calc_vif(data).sort_values('VIF',ascending=0).variables.iloc[0:5])
        
        g = iv
        g = g.reset_index()
        g["varWoe"] = g["index"]+"_woe"
        iv = g.set_index("varWoe")
        top_vif = top_vif.set_index("variables")
        
        top_vif["info_value"] = iv.info_value
        print(top_vif)
        print([col[:-4] for col in top_vif.index])
        var_to_drop = [col[:-4] for col in top_vif.index]
        #var_to_drop_1 = dict((k,plot[k]) for k in var_to_drop if k in plot)
        for var  in var_to_drop:
            plot[var]
        drop_var = input()
        
        
        #drop_var = top_vif.sort_values("info_value",ascending=0).index[0]
       
        
        print("droping {drop_var}".format(drop_var = drop_var))
        
        
        
        #drop_var= calc_vif(data).sort_values('VIF',ascending=0).variables.iloc[0]
        vif = calc_vif(data).sort_values('VIF',ascending=0).VIF.iloc[0]
        drop_list.append(drop_var)
        yield  calc_vif(data).sort_values('VIF',ascending=0)

        print("\n \n \n droping {drop_var} @ VIF =  {VIF}\n \n \n".format(drop_var=drop_var,VIF =  vif))
        data = data.drop(columns=[drop_var])
        del(vif)
    
    print("done",  len(data.columns)>=count,len(data.columns),count)
    if drop_list == True:
        print("selected %d  of variable droped %d variable".format(len(data.columns),len(drop_list)))
        return  calc_vif(data) , drop_list
    else :
        return  calc_vif(data)
    
    #data = iter_calc_vif(vifData, count = 15, drop_list =  True)
#output = iter_calc_vif(vifData, count = 15, drop_list =  True)#  argument data frame ,how many variable
for  i in  iter_calc_vif(vifData, count = 10, drop_list =  True,bins = bins_b0):
    print(i)
