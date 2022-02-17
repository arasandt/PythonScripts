import pandas as pd, os
from datetime import datetime, date, timedelta 
import wget

expiry_file = 'foExp.js'
index_file = '^NSEI_yahoo.csv'
vix_file = 'VIX_nse.csv'
option_file = 'OptionPrice_dumpall.csv'
sp_nearer = 100
start_date = date(2020,5,12)
index_option = 'NIFTY'
options_df = None

def refresh_expiry_dates():
    with open(expiry_file,'r') as exp:
        exp_data =  exp.readlines()
    exp_data = [i.replace('\n','')[-12:-2:] for i in exp_data if 'var ' not in i if 'vixExpryDt' not in i if 'stkExpryDt' not in i]
    exp_data_formatted = [datetime.strptime(i, '%d-%m-%Y') for i in exp_data]
    df = pd.DataFrame({'ExpiryDateFormatted': exp_data_formatted})    
    df = df.sort_values(['ExpiryDateFormatted'])
    df['ExpiryDate'] = pd.to_datetime(df['ExpiryDateFormatted'], format='%d-%b-%y')
    df.drop(['ExpiryDateFormatted'], axis=1, inplace=True)
    df.to_csv(expiry_file + '.csv',header=True, sep=',', index=False)

def load_expiry_dates():
    index_cutoff_date = date(2019,2,8)
    df = pd.read_csv(expiry_file + '.csv')
    df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'], format='%Y-%m-%d')
    df['ExpiryMonth'] = df['ExpiryDate'].apply(lambda x: str(x.year) + '-' + str(x.month) )   
    df1 = df[df['ExpiryDate'] <= pd.Timestamp(index_cutoff_date)]
    df1_grp = df1.groupby(['ExpiryMonth']).last()
    df1_grp = df1_grp[df1_grp.ExpiryDate != '2019-02-07']
    df2 = df[df['ExpiryDate'] > pd.Timestamp(index_cutoff_date)]
    df2 = df2[df2.ExpiryDate != '2019-06-21']
    df2 = df2[df2.ExpiryDate != '2019-03-15']
    df = df1_grp.append(df2, ignore_index=True, sort=True)
    df.reset_index(inplace=True,drop=True)
    df.drop(['ExpiryMonth'], axis=1, inplace=True)
    df = df[df['ExpiryDate'] >= pd.Timestamp(start_date)]
    return df

def load_index_data(days_window):
    holidays_in_month = 0
    ddays = holidays_in_month + days_window
    ddate = start_date - timedelta(days=ddays)
    df = pd.read_csv(index_file)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df[df['Date'] >= pd.Timestamp(ddate)]
    #df.drop(['Volume','Adj Close','Open','High','Low'], axis=1, inplace=True)
    return df

def load_vix():
    df = pd.read_csv(vix_file)
    df['Date'] = pd.to_datetime(df['Date '], format='%d-%b-%Y')
    df = df[df['Date'] >= pd.Timestamp(start_date)]
    df.drop(['Prev. Close ','Change ','Open ','High ','Low ', '% Change', 'Date '], axis=1, inplace=True)
    df.rename(columns={"Close ": "Vix"}, inplace=True)
    df['Vix'] = df['Vix'].apply(lambda x: round(x,2))
    return df    

def load_option_prices():
    if os.path.exists(option_file):
        df = pd.read_csv(option_file)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['ExpiryDay'] = pd.to_datetime(df['ExpiryDay'], format='%Y-%m-%d')
    else:
        df = pd.DataFrame(columns=['Date', 'Name', 'Type', 'StrikePrice', 'ExpiryDay','Price'])
    return df    

def myround(x, base):
    return base * round(x/base)

def get_url(dt):
    base_url = "http://www1.nseindia.com/content/historical/DERIVATIVES/" #2020/FEB/fo06FEB2020bhav.csv.zip
    yr = dt.year
    day = dt.day
    mon = dt.strftime('%b').upper()
    filename = 'fo{2:02d}{1}{0}bhav.csv.zip'.format(yr,mon,day)
    return '{0}{1}/{2}/{3}'.format(base_url,yr,mon,filename), filename


def return_dayname(day):
    day = str(day)
    dayname = { '0': 'Intraday', 
                '1': 'Thursday',
                '2': 'Wednesday',
                '3': 'Tuesday',
                '4': 'Monday',
                '5': 'Friday',
                '6': 'Thursday-1',
                '30': 'Month'}
    return dayname[day]    

def options_get_history(row):
    # if opt == 'PE':
    #     strike_price = row['PutStrikePrice']
    # else:
    #     strike_price = row['CallStrikePrice']
    
    put_strike_price = row['PutStrikePrice']
    call_strike_price = row['CallStrikePrice']

    global options_df
    
    found_price_put = options_df.loc[(options_df['Date'] == row['Date']) & (options_df['Name'] == index_option) & (options_df['Type'] == 'PE') & (options_df['StrikePrice'] == put_strike_price) & (options_df['ExpiryDay'] == row['ExpiryDay'])]
    found_price_call = options_df.loc[(options_df['Date'] == row['Date']) & (options_df['Name'] == index_option) & (options_df['Type'] == 'CE') & (options_df['StrikePrice'] == call_strike_price) & (options_df['ExpiryDay'] == row['ExpiryDay'])]
    try:
        if found_price_put.empty: # or found_price_call.empty :
            print('Fetching Price from NSE dumps {0}_{1}_PE on {2}'.format(row['ExpiryDay'].date(), int(put_strike_price), row['Date'].date()))

            #http://www1.nseindia.com/content/historical/DERIVATIVES/2020/FEB/fo06FEB2020bhav.csv.zip

            url, filename = get_url(row['Date'])
            op_file = os.path.join('downloads',filename)
            if not os.path.exists(op_file):
                wget.download(url, out='./downloads')
                print()

            df = pd.read_csv(op_file)
            df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'], format='%d-%b-%Y')
            df = df[df['INSTRUMENT'] == 'OPTIDX']
            df = df[df['SYMBOL'] == index_option]
            df = df[df['STRIKE_PR'] == put_strike_price]
            df = df[df['EXPIRY_DT'] == row['ExpiryDay']]

            putopenprice = df[df['OPTION_TYP'] == 'PE']['OPEN'].values[0]
            puthighprice = df[df['OPTION_TYP'] == 'PE']['HIGH'].values[0]
            putlowprice = df[df['OPTION_TYP'] == 'PE']['LOW'].values[0]
            putcloseprice = df[df['OPTION_TYP'] == 'PE']['CLOSE'].values[0]
            #print(putopenprice , puthighprice, putlowprice, putcloseprice)
            temp_dict = {}
            temp_dict[row['Date']] = {'Name'          : index_option, 
                                        'Type'        : 'PE', 
                                        'StrikePrice' : put_strike_price,
                                        'ExpiryDay'   : row['ExpiryDay'],
                                        'ClosePrice'  : putcloseprice,
                                        'LowPrice'    : putlowprice,
                                        'HighPrice'   : puthighprice,
                                        'OpenPrice'   : putopenprice,
                                        }
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
            temp_df['Date'] = temp_df.index
            temp_df.reset_index(inplace=True,drop=True)
            options_df = options_df.append(temp_df, ignore_index = True,sort=False)
            temp_df = None  
        else:
            #print('Fetching Price from local {0}_{1}_PE on {2}'.format(row['ExpiryDay'].date(), int(put_strike_price), row['Date'].date()))
            putopenprice = found_price_put['OpenPrice'].values[0]
            puthighprice = found_price_put['HighPrice'].values[0]
            putcloseprice = found_price_put['ClosePrice'].values[0]

        if found_price_call.empty :
            print('Fetching Price from NSE dumps {0}_{1}_CE on {2}'.format(row['ExpiryDay'].date(), int(call_strike_price), row['Date'].date()))

            #http://www1.nseindia.com/content/historical/DERIVATIVES/2020/FEB/fo06FEB2020bhav.csv.zip

            url, filename = get_url(row['Date'])
            op_file = os.path.join('downloads',filename)
            if not os.path.exists(op_file):
                wget.download(url, out='./downloads')
                print()

            df = pd.read_csv(op_file)
            df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'], format='%d-%b-%Y')
            df = df[df['INSTRUMENT'] == 'OPTIDX']
            df = df[df['SYMBOL'] == index_option]
            df = df[df['STRIKE_PR'] == call_strike_price]
            df = df[df['EXPIRY_DT'] == row['ExpiryDay']]

            callopenprice = df[df['OPTION_TYP'] == 'CE']['OPEN'].values[0]
            callhighprice = df[df['OPTION_TYP'] == 'CE']['HIGH'].values[0]
            calllowprice = df[df['OPTION_TYP'] == 'CE']['LOW'].values[0]
            callcloseprice = df[df['OPTION_TYP'] == 'CE']['CLOSE'].values[0]
            #print(callopenprice , callhighprice, calllowprice, callcloseprice)
            temp_dict = {}
            temp_dict[row['Date']] = {'Name'          : index_option, 
                                        'Type'        : 'CE', 
                                        'StrikePrice' : call_strike_price,
                                        'ExpiryDay'   : row['ExpiryDay'],
                                        'ClosePrice'  : callcloseprice,
                                        'LowPrice'    : calllowprice,
                                        'HighPrice'   : callhighprice,
                                        'OpenPrice'   : callopenprice,
                                        }
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
            temp_df['Date'] = temp_df.index
            temp_df.reset_index(inplace=True,drop=True)
            options_df = options_df.append(temp_df, ignore_index = True,sort=False)
            temp_df = None  
        else:
            #print('Fetching Price from local {0}_{1}_CE on {2}'.format(row['ExpiryDay'].date(), int(call_strike_price), row['Date'].date()))
            callopenprice = found_price_call['OpenPrice'].values[0]
            callhighprice = found_price_call['HighPrice'].values[0]
            callcloseprice = found_price_call['ClosePrice'].values[0]
        #print(putopenprice , callopenprice)
        return putopenprice , callopenprice, puthighprice , callhighprice, putcloseprice , callcloseprice
        # else:
        #     print('Fetching Price from local {0}_{1}_CE/PE on {2}'.format(row['ExpiryDay'].date(), int(strike_price), row['Date'].date()))
        #     return found_price_put['Price'].values[0], found_price_call['Price'].values[0]
    except Exception as e:
        print("Error") 
        print(e)

def run():

    #refresh_expiry_dates()
    expdt_df = load_expiry_dates()
    expdt_df = expdt_df[expdt_df.ExpiryDate != '2019-06-21']
    expdt_df = expdt_df[expdt_df.ExpiryDate != '2019-03-15']
    expdt_df.reset_index(inplace=True, drop=True)    
    #print(expdt_df.head())
    #print(expdt_df.tail())
    
    days_window = 0
    #curr_date = datetime(2020,5,22,0,0,0)
    #nifty_close = 9039.10
    #vix_close = 32.38
    
    index_df = load_index_data(days_window)
    temp_dict = {}
    #temp_dict[0] = {'Date' : curr_date,'Close' : nifty_close}
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    index_df = index_df.append(temp_df, ignore_index = True, sort=False)
    index_df.reset_index(inplace=True,drop=True)
    #print(index_df.tail())   
    
    vix_df = load_vix()
    temp_dict = {}
    #temp_dict[0] = {'Date' : curr_date, 'Vix' : vix_close}
    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    vix_df = vix_df.append(temp_df, ignore_index = True, sort=False)    
    vix_df.reset_index(inplace=True,drop=True)
    #print(vix_df.tail())   
    
    global options_df
    options_df = load_option_prices()
   
    index_df.reset_index(inplace=True,drop=True)
    index_df = pd.merge(index_df, vix_df, on='Date', how='left')
    index_df.fillna(0, inplace=True)
    #print(index_df.head())     
    #print(index_df.tail())   
    
    
    def next_exp_Date(d):
        dat = d.iloc[0]
        #return list(expdt_df[expdt_df['ExpiryDate'] >= pd.Timestamp('2020-01-13 00:00:00')].iloc[0])[0]
        filter_df = expdt_df[expdt_df['ExpiryDate'] >= pd.Timestamp(dat)]
        if len(filter_df) > 0 :
            return list(filter_df.iloc[0])[0]
        else:
            return 0

    index_df['ExpiryDay'] = index_df.apply(lambda x: next_exp_Date(x), axis=1)
    index_df['StrikePrice'] = index_df['Open'].apply(lambda x: int(myround(x, sp_nearer)))
    index_df['PutStrikePrice'] = index_df['StrikePrice'] 
    index_df['PutOpenPrice'] = 0
    index_df['PutHighPrice'] = 0
    index_df['PutClosePrice'] = 0
    index_df['CallStrikePrice'] = index_df['StrikePrice']
    index_df['CallOpenPrice'] = 0
    index_df['CallHighPrice'] = 0
    index_df['CallClosePrice'] = 0

    def get_sweet_spot(row):
        diff = 50
        maxprice = 50
        po, co, ph, ch, pc, cc = options_get_history(row)
        while True:
            if po > maxprice or po == 0:
                row['PutStrikePrice'] = int(row['PutStrikePrice']) - diff
            if co > maxprice or co == 0:
                row['CallStrikePrice'] = int(row['CallStrikePrice']) + diff   
            po, co, ph, ch, pc, cc = options_get_history(row)
            if co < maxprice and po < maxprice:
                break        
        #print(row['PutStrikePrice'] , row['CallStrikePrice'], po, co, ph, ch, pc, cc)
        return row['PutStrikePrice'] , row['CallStrikePrice'], po, co, ph, ch, pc, cc

    index_df['PutStrikePrice'], index_df['CallStrikePrice'], index_df['PutOpenPrice'], index_df['CallOpenPrice'], index_df['PutHighPrice'], index_df['CallHighPrice'], index_df['PutClosePrice'], index_df['CallClosePrice'] = zip(*index_df.apply(get_sweet_spot,  axis=1))

    index_df['MarginPricePut'] = index_df['PutHighPrice'] - index_df['PutOpenPrice']
    index_df['MarginPriceCall'] = index_df['CallHighPrice'] - index_df['CallOpenPrice']
    index_df['MarginPrice'] = index_df[["MarginPricePut", "MarginPriceCall"]].max(axis=1)
    
    index_df['PointsDiff'] = index_df['PutOpenPrice'] - index_df['PutClosePrice'] + index_df['CallOpenPrice'] - index_df['CallClosePrice']

    index_df.drop(['MarginPriceCall','MarginPricePut', 'Volume','Adj Close', 'StrikePrice'], axis=1, inplace=True)

    index_df.to_csv('{0}_output_lead-{1}.csv'.format(index_file, return_dayname(days_window)),header=True, sep=',', index=False)
    options_df.to_csv(option_file, header=True, sep=',', index=False)

if __name__ == '__main__':
    run()