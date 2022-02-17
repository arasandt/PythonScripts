#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import os
import pandas as pd

from LenelJobDAO import LenelJobDAO


if __name__ == '__main__':
    """
    """

    conn = LenelJobDAO.utils_obj.get_db_conn()
    #SelectQuery = 'SELECT FILE_ID, FILE_PATH, FRAME_RT, SRVR_ID, CHNL_NO, FILE_CRT_TS FROM T_LNR_FILE_INFO where SRVR_ID = ? and CHNL_NO = ? and CONVERT(DATE, FILE_CRT_TS) = ? order by FILE_CRT_TS'
    #file_date = '2020-02-17'
    #channel = 104
    #serverid = 6
    SelectQuery = 'SELECT A.FILE_ID, A.FILE_PATH, A.FRAME_RT, A.SRVR_ID, A.CHNL_NO, A.FILE_CRT_TS from t_file B, T_LNR_FILE_INFO A WHERE a.file_id = b.file_id AND B.outputfileframes <> 0 and a.FILE_ID not in (253107, 70361, 251730, 19263)'
    #result_df = pd.read_sql(SelectQuery, con=conn, params=(serverid, channel, file_date))     
    result_df = pd.read_sql(SelectQuery, con=conn)     
    conn.close()
    print('Selected {0} rows'.format(len(result_df)))
    InputTriggerpath = os.path.join(os.getcwd(),"input")
    inp = input("Proceed to create {0} trigger files (Y/N) : ".format(len(result_df)))
    if inp == 'Y':
        for index, row in result_df.iterrows():
            filn = os.path.join(InputTriggerpath, str(row['FILE_ID']) + '.trg')
            content = str(row['FILE_PATH']) + "|" + str(row['FRAME_RT'])
            with open(filn,'w') as wr:
                wr.write(content)
    else:
        print('Exiting..')
        
        
    
