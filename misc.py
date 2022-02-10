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
        
