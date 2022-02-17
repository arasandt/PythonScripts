#OaM cost sheet

import pandas as pd
#help(pd.read_fwf)


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
filepaths = ['D:\Arasan\Common\OaM\MFFiles\C0177T.C0253.STOR.YEAR.TXT','D:\Arasan\Common\OaM\MFFiles\C0177T.C0251.STOR.YEAR.TXT','D:\Arasan\Common\OaM\MFFiles\C0177T.C4041.STOR.YEAR.TXT','D:\Arasan\Common\OaM\MFFiles\C0177T.C4043.STOR.YEAR.TXT']

def readfwf(x):
    return pd.read_fwf(x, header=None, skiprows=4)
#df = pd.concat(map(pd.read_fwf, filepaths))
df = pd.concat(map(readfwf, filepaths))
#df = pd.read_fwf(filepaths, header=None, skiprows=4)
#df = pd.read_fwf("D:\Arasan\Common\OaM\MFFiles\C0177T.C0253.STOR.YEAR.TXT", header=None, skiprows=4,names=['SYSTEM_ID','OCCUPANCY_DT','PROJECT_CD','MEDIA_CD','ID1_CD','ENVIRONMENT_CD','MIRROR_FACTOR_NM','STOR_QTY','POST_TS'])
#df = pd.read_csv("D:\Arasan\Common\OaM\MFFiles\C0177T.C0253.STOR.YEAR.TXT1", 
#                 names=['SYSTEM_ID','OCCUPANCY_DT','PROJECT_CD','MEDIA_CD','ID1_CD','ENVIRONMENT_CD','MIRROR_FACTOR_NM','STOR_QTY','POST_TS'],
#                 skiprows=None) #,delimiter=' ')
df.columns = ['SYSTEM_ID','OCCUPANCY_DT','PROJECT_CD','MEDIA_CD','ID1_CD','ENVIRONMENT_CD','MIRROR_FACTOR_NM','STOR_QTY','POST_TS']

#print(df.head())
#print(df.count())
df.drop(columns=['SYSTEM_ID', 'OCCUPANCY_DT','ENVIRONMENT_CD','POST_TS'],inplace=True)
df['STORAGE'] = df['MIRROR_FACTOR_NM'] * df['STOR_QTY']
df.drop(columns=['MIRROR_FACTOR_NM', 'STOR_QTY'],inplace=True)
#print(df.head())
gdf = df.groupby(['PROJECT_CD','MEDIA_CD','ID1_CD'])['STORAGE'].sum()
print(dict(gdf))



# =============================================================================
# =============================================================================
# filepaths = ['D:\Arasan\Common\OaM\MFFiles\C0177T.C0251.CPU.YEAR.TXT','D:\Arasan\Common\OaM\MFFiles\C0177T.C0253.CPU.YEAR.TXT','D:\Arasan\Common\OaM\MFFiles\C0177T.C4041.CPU.YEAR.TXT','D:\Arasan\Common\OaM\MFFiles\C0177T.C4043.CPU.YEAR.TXT']
# 
# # =============================================================================
# # x = ''
# # for fname in filepaths:
# #     with open(fname) as infile:
# #         x += infile.read()
# # 
# # x = x.replace('NOT APPLICABLE','NOTAPPLICABLE').replace('DIST DB2','DISTDB2').replace('AUTH ID','AUTHID')
# # =============================================================================
# def readfwf(x):
#     return pd.read_fwf(x, header=None, skiprows=4,widths=[12,12,12,22,22,22,12,18,12,12,21,22,26])
# #df = pd.concat(map(pd.read_fwf, filepaths))
# df = pd.concat(map(readfwf, filepaths))
# #df = pd.read_fwf(filepaths, header=None, skiprows=4)
# #df = pd.read_fwf(x, header=None, skiprows=4)
# #df = pd.read_fwf("D:\Arasan\Common\OaM\MFFiles\C0177T.C0253.STOR.YEAR.TXT", header=None, skiprows=4,names=['SYSTEM_ID','OCCUPANCY_DT','PROJECT_CD','MEDIA_CD','ID1_CD','ENVIRONMENT_CD','MIRROR_FACTOR_NM','STOR_QTY','POST_TS'])
# #df = pd.read_csv("D:\Arasan\Common\OaM\MFFiles\C0177T.C0253.STOR.YEAR.TXT1", 
# #                 names=['SYSTEM_ID','OCCUPANCY_DT','PROJECT_CD','MEDIA_CD','ID1_CD','ENVIRONMENT_CD','MIRROR_FACTOR_NM','STOR_QTY','POST_TS'],
# #                 skiprows=None) #,delimiter=' ')
# #df.columns = ['SYSTEM_ID','ACCOUNTED_DT','CYCLE_DT','ID_NM1','ID_NM2','ID_NM31','ID_NM32','COST_DRIVER_NM1','COST_DRIVER_NM2','SHIFT_IND','ENVIRONMENT_CD','PROJECT_CD','CPU_USAGE','ASSIGNMENT_RSN_CD1','ASSIGNMENT_RSN_CD2','POST_TS']
# df.columns = ['SYSTEM_ID','ACCOUNTED_DT','CYCLE_DT','ID_NM1','ID_NM2','ID_NM3','COST_DRIVER_NM','SHIFT_IND','ENVIRONMENT_CD','PROJECT_CD','CPU_USAGE','ASSIGNMENT_RSN_CD','POST_TS']
# #df.fillna('',inplace=True)
# #print_full(df.ASSIGNMENT_RSN_CD.head())
# #df['ID_NM3'] = df['ID_NM31'] + df['ID_NM32']
# #df['COST_DRIVER_NM'] = df['COST_DRIVER_NM1'] + df['COST_DRIVER_NM2']
# #df['ASSIGNMENT_RSN_CD'] = df['ASSIGNMENT_RSN_CD1'] + df['ASSIGNMENT_RSN_CD2']
# #print_full(df.tail())
# #df.drop(columns=['ID_NM31', 'ID_NM32'],inplace=True)
# #df.drop(columns=['COST_DRIVER_NM1', 'COST_DRIVER_NM2'],inplace=True)
# #df.drop(columns=['ASSIGNMENT_RSN_CD1', 'ASSIGNMENT_RSN_CD2'],inplace=True)
# 
# df.drop(columns=['COST_DRIVER_NM','SYSTEM_ID', 'ACCOUNTED_DT','CYCLE_DT','ID_NM1','ID_NM2','ID_NM3','ENVIRONMENT_CD','ASSIGNMENT_RSN_CD','POST_TS'],inplace=True)
# #print_full(df.head())
# gdf = df.groupby(['SHIFT_IND','PROJECT_CD'])['CPU_USAGE'].sum()
# print_full(dict(gdf))
#  
# 
# 
# 
# =============================================================================
