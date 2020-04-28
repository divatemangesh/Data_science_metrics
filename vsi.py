def vsi(var1,var2):
    '''
        vsi is used to calculate csi between same variable from 2 diffrent dataset
        and it gives you  metric how stable/unstable is varibale.
        var1:- pandas_series  
        var2:- pandas_series

        Returns CSI,csi_table

    '''
    csi_seg1 = pd.Dataframe()
    csi_seg1['counts'] = var1.value_counts(dropna = False)
    csi_seg1 = csi_seg1.reset_index().rename(columns =  {"index":"category"})
    csi_seg1["percent"] =  csi_seg1["count"] / var1.shape[0]


    csi_seg2 = pd.Dataframe()
    csi_seg2['counts'] = var2.value_counts(dropna = False)
    csi_seg2 = csi_seg1.reset_index().rename(columns =  {"index":"category"})
    csi_seg2["percent"] =  csi_seg2["count"] / var2.shape[0]


    csi_seg =  csi_seg1.merge(csi_seg2,on="category")

    csi_seg["percent_diff"] = csi_seg["percent_x"]-csi_seg["percent_y"]
    csi_seg["ln_ratio"] =  np.log(csi_seg["percent_x"]/csi_seg["percent_y"])
    csi_seg["index"] = csi_seg["percent_diff"] * csi_seg["ln_ratio"]
    csi = csi_seg["index"].sum()
    return csi,csi_seg

    



