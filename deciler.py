import pandas as pd


def create_decile(data,prob,bin_ = None ,n_bins = 10):
  '''
  data:pandas data frame
  prob:variable onwhich you wants to create deciles
  bins:default None , you can pass list of bins threshold if you want create decile on prior thresholds
  n_bins = default = 10 , create 10 splits which is deciles if bin_ is not None then ignore 
  
  
  
  '''
    if bin_ ==  None:
        data['bucket'],cutbin = pd.qcut(data[prob], n_bins,retbins=True)
    else :
        data['bucket'] = pd.cut(data[prob],bins=bin_,include_lowest=True)
    
    table = ks_data.groupby("bucket").mean("prob").sort_values("prob",ascending=False).reset_index().reset_index()
    table["index"] = table["index"]+1
    table = table[["index","bucket"]]
    return pd.merge(ks_data,table,on="bucket")
        
