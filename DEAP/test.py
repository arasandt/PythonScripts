#import pandas as pd
#import numpy as np
#
#import itertools
#
#module   = ['Science','Maths','Physics','Chemistry','English']
#stgroups = range(1,11)
#day      = ['Mon','Tue','Wed','Thu','Fri']
#location = ['Hall A','Hall B','Hall C','Hall D','Hall E']
#timing   = range(9,18)
#
#def evaltimetable(individual):
#    print('Evaluate : ', len(individual))
#    #print('*****************')
#    #print(individual)
#    
#    df = pd.DataFrame(individual, columns = ['Module' , 'StudentGroup', 'Weekday', 'Avenue', 'Time']) 
#    #print(df.head())
#    
#    stg = df['StudentGroup'].unique()
#    #print(stg)
#    
#    fit1 = [] 
#    fit2 = []
#    fit3 = []
#    #daily_classes = pd.DataFrame(list(itertools.product(module, day)),columns = ['Module','Weekday']).reset_index()
#    #daily_hours = pd.DataFrame(list(itertools.product(timing, day)),columns = ['Time','Weekday']).reset_index()
#    daily_hours = len(timing) * len(day)
#    
#    fit = []
#    
#    for i in stg:
#        dfstg = df[df['StudentGroup'] == i].copy()
#        #print(dfstg)
#        
#        #time conflicts within student group . minimize this
#        dfstg['fit1'] = 0 
#        #group by date and time and make there is only one entry for each.
#        dff = dfstg[dfstg.groupby(['Weekday','Time'])['Time'].transform('count') > 1]
#        #if entry is more than 1 then maximize this fitness for this individual
#        dfstg.loc[dff.index,'fit1'] = 1
#        
#        
#        # make sure classes cover all days. maximize this.
#        dfstg['fit2'] = 1
#        #get all lecture days for the group
#        dff = dfstg.groupby(['Weekday'])['Weekday'].agg('count')
#        y = dff.index.tolist()
#        # assign fitness based on missing lecture days
#        dfstg['fit2'] = len(y)/len(day)
#
#        # make sure classes cover all times. maximize this.
#        dfstg['fit3'] = 1
#        #print(dfstg)
#        #print('***************************************')
#        #get all lecture time and days for the group
#        dff = dfstg.groupby(['Weekday','Time'])['Weekday'].unique().agg('count')
#        # assign fitness based on missing times per day
#        dfstg['fit3'] = dff / len(daily_hours)
#        #print(dfstg)
#        #print('***************************************')        
#        
#        # make sure each lecture is not more than 2 in a singleday. maxmise this.
#        dfstg['fit4'] = 1
#        #print('***************************************')        
#        #print(dfstg)       
#        # take all classes which are more than 2 for each day
#        dff = dfstg[dfstg.groupby(['Weekday','Module'])['Module'].transform('count') > 2]
#        #print(dff)
#        # for those classes alone have inverse fitness.
#        if len(dff) > 0:
#            dfstg.loc[dff.index,'fit4'] = 1 / len(dff)
#        #print(dfstg)
#        #print('***************************************')     
#        
#        # make sure each lecture is taken atleast one day. Maximize this
#        dfstg['fit5'] = 1
#        #print('***************************************')     
#        #print(dfstg) 
#        # take count of each module for each day
#        dff = dfstg.groupby(['Weekday','Module'])['Module'].agg('count')
#        y = dff.index.tolist()
#        #print(dff)
#        #print(y)
#        dfstg['fit5'] = len(y) / (len(module) * len(day))
#        #print(dfstg) 
#        #print('***************************************')     
#        
#        # hall conflict at the same time and date with other student groups. minimize this
#        dfstg['fit6'] = 0
#        # select other student groups
#        df1 = df[df['StudentGroup'] != i]
#        # Count all records
#        all_recs = len(df)
#        # get conflicts using inner join
#        conf_df = pd.merge(df1, dfstg, on=['Avenue','Weekday','Time'], how='inner').fillna(0)
#        conflicts = len(conf_df)
#        dfstg['fit6'] = conflicts / all_recs
#        
#        print(dfstg)
#        fit1 = sum(dfstg.fit1.tolist())/len(dfstg)
#        fit2 = sum(dfstg.fit2.tolist())/len(dfstg)
#        fit3 = sum(dfstg.fit3.tolist())/len(dfstg)
#        fit4 = sum(dfstg.fit4.tolist())/len(dfstg)
#        fit5 = sum(dfstg.fit5.tolist())/len(dfstg)
#        fit6 = sum(dfstg.fit6.tolist())/len(dfstg)
#        
#        fit.append((fit1, fit2, fit3, fit4, fit5, fit6))
#    
#    print(fit)
#    fit = list(zip(*fit))
#    print(fit)
#    fit = [sum(i)/len(i) for i in fit]
#    print(fit)
#    return fit
#        
#        
#        
#
#    
#
#    
#
#    
#        
#if __name__ == "__main__":
#    #ind = [('English', 1, 'Tue', 'Hall D', 12), ('Science', 5, 'Mon', 'Hall B', 13), ('Maths', 7, 'Fri', 'Hall A', 14), ('Chemistry', 3, 'Tue', 'Hall A', 13), ('Chemistry', 4, 'Thu', 'Hall A', 16), ('Chemistry', 8, 'Fri', 'Hall C', 13), ('Chemistry', 9, 'Wed', 'Hall D', 11), ('English', 2, 'Wed', 'Hall D', 12), ('English', 3, 'Thu', 'Hall E', 16), ('Maths', 3, 'Mon', 'Hall A', 15), ('Chemistry', 5, 'Tue', 'Hall E', 16)]
#    ind = [('English', 1, 'Tue', 'Hall D', 11), ('English', 1, 'Tue', 'Hall D', 11), ('English', 1, 'Tue', 'Hall D', 13), ('English', 1, 'Tue', 'Hall D', 14), ('Maths', 7, 'Fri', 'Hall A', 14), ('Chemistry', 3, 'Tue', 'Hall A', 13), ('Chemistry', 4, 'Thu', 'Hall A', 16), ('Chemistry', 8, 'Fri', 'Hall C', 13), ('Chemistry', 9, 'Wed', 'Hall D', 11), ('English', 2, 'Tue', 'Hall D', 12), ('English', 3, 'Thu', 'Hall E', 16), ('Maths', 3, 'Mon', 'Hall A', 15), ('Chemistry', 5, 'Tue', 'Hall E', 16)]
#    evaltimetable(ind)


def cxSet(ind1, ind2):
    import random
#    iteration = len(ind1) if len(ind1) < len(ind2) else len(ind2)
#    for i in range(iteration):
#        if random.random() < 0.5:
#            print(ind1)
#            print(ind1[i])
#            tmp = set(ind1[i])
#            ind1[i] = ind2[i]
#            ind2[i] = tmp
#            #ind1.add(ind2[i])
#            #ind2.add(tmp)
    print('Before crossover : ', ind1)
    #ind1len = len(ind1)
    print('Before crossover : ', ind2)
    #ind2len = len(ind2)
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                   # Symmetric Difference (inplace)
    
    temp = set()
    for i,j in enumerate(ind2):
        if i % 2 == 0:
            ind1.add(j)
        else:
            temp.add(j)
    ind2 = temp
    
    print('After crossover : ', ind1)
    print('After crossover : ', ind2)
    print(ind1)
    print(ind2)


if __name__ == "__main__":
    ind1 = {('Elective3', 5, 'Thu', 'Hall B', 17), ('Elective3', 3, 'Tue', 'Hall B', 17), ('Elective3', 5, 'Mon', 'Hall C', 12), ('Elective1', 9, 'Tue', 'Hall B', 12), ('Elective2', 9, 'Wed', 'Hall E', 13), ('Elective5', 2, 'Wed', 'Hall C', 13), ('Elective5', 2, 'Tue', 'Hall A', 11), ('Elective4', 1, 'Thu', 'Hall A', 12), ('Elective3', 6, 'Thu', 'Hall B', 12), ('Elective1', 8, 'Mon', 'Hall D', 11)}
    ind2 = {('Elective4', 8, 'Fri', 'Hall E', 15), ('Elective2', 5, 'Wed', 'Hall E', 11), ('Elective1', 4, 'Thu', 'Hall A', 12), ('Elective4', 1, 'Thu', 'Hall A', 10), ('Elective3', 5, 'Wed', 'Hall D', 15), ('Elective2', 9, 'Tue', 'Hall D', 12), ('Elective1', 9, 'Mon', 'Hall A', 17), ('Elective3', 5, 'Wed', 'Hall B', 12), ('Elective1', 7, 'Tue', 'Hall A', 14), ('Elective4', 3, 'Mon', 'Hall E', 16)}
    cxSet(ind1, ind2)

