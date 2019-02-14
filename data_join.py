#encoding:utf-8
import pandas as pd
from pybert.config.basic_config import configs as config

data1 = pd.read_csv(config['raw_data_path1'],sep = '\t',header = None)
data2 = pd.read_csv(config['raw_data_path2'],sep = '\t',header = None)

data = pd.concat([data1,data2],axis=0)
data = data.drop(0,axis =1)
data.to_csv(config['raw_data_join_path'],index=False,header=None,sep = '\t')



