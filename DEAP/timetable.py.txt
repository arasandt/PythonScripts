import random
import pandas as pd
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

NBR_ATTRBS = 5
IND_INIT_SIZE = 10

#moduletext   = ['Subject1','Subject2','Subject3','Subject4','Subject5']
moduletext   = ['Subject1','Subject2'] 
module       = range(len(moduletext))
#stgroupstext = ['StudentGroup1','StudentGroup2','StudentGroup3','StudentGroup4','StudentGroup5']
stgroupstext = ['StudentGroup1','StudentGroup2'] 
stgroups     = range(len(stgroupstext))
#daytext      = ['Mon','Tue','Wed','Thu','Fri']
daytext      = ['Mon','Tue']
day          = range(len(daytext))
#locationtext = ['Hall A','Hall B','Hall C','Hall D','Hall E']
locationtext = ['Hall A','Hall B']
location     = range(len(locationtext))
#timing       = range(9,18)
starttime, endtime = 9, 12
timing       = range(starttime, endtime + 1)

creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
creator.create("Individual", set, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def sel_attr(count):
    return (random.choice(module), random.choice(stgroups), random.choice(day), random.choice(location), random.choice(timing))
    
toolbox.register("attr_class", sel_attr, NBR_ATTRBS)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_class, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaltimetable(individual):
    df = pd.DataFrame(list(individual), columns = ['Module' , 'StudentGroup', 'Weekday', 'Venue', 'Time']) 
    stg = df['StudentGroup'].unique()
   
    daily_hours = len(timing) * len(day)

    fit = []
    
    # limit the population growth beyond a certain limit. maximize but not beyond certain limit
    if len(individual) > (len(stgroups) * len(day) * len(timing)):
        return 1, 1, 0, 0, 0, 0, 0
    
    for i in stg:
        dfstg = df[df['StudentGroup'] == i].copy()
        #time conflicts within student group . minimize this
        dfstg['fit1'] = 0 
        #group by date and time and make there is only one entry for each.
        dff = dfstg[dfstg.groupby(['Weekday','Time'])['Time'].transform('count') > 1]
        #if entry is more than 1 then maximize this fitness for this individual
        dfstg.loc[dff.index,'fit1'] = 1
        
        
        # make sure classes cover all days. maximize this.
        dfstg['fit2'] = 1
        #get all lecture days for the group
        dff = dfstg.groupby(['Weekday'])['Weekday'].agg('count')
        y = dff.index.tolist()
        # assign fitness based on missing lecture days
        dfstg['fit2'] = len(y)/len(day)


        # make sure classes cover all times. maximize this.
        dfstg['fit3'] = 1
        #get all lecture time and days for the group
        dff = dfstg.groupby(['Weekday','Time'])['Weekday'].unique().agg('count')
        # assign fitness based on missing times per day
        dfstg['fit3'] = dff / daily_hours
        

        # make sure each lecture is not more than 2 in a singleday. maxmise this.
        dfstg['fit4'] = 1
        # take all classes which are more than 2 for each day
        dff = dfstg[dfstg.groupby(['Weekday','Module'])['Module'].transform('count') > 2]
        # for those classes alone have inverse fitness.
        if len(dff) > 0:
            dfstg.loc[dff.index,'fit4'] = 1 / len(dff)
        

        # make sure each lecture is taken atleast one day. Maximize this
        dfstg['fit5'] = 1
        # take count of each module for each day
        dff = dfstg.groupby(['Weekday','Module'])['Module'].agg('count')
        y = dff.index.tolist()
        dfstg['fit5'] = len(y) / (len(module) * len(day))
        

        # hall conflict at the same time and date with other student groups. minimize this
        dfstg['fit6'] = 0
        # select other student groups
        df1 = df[df['StudentGroup'] != i].copy()
        # Count all records
        all_recs = len(df)
        # get conflicts using inner join
        conf_df = pd.merge(df1, dfstg, on=['Venue','Weekday','Time'], how='inner').fillna(0)
        conflicts = len(conf_df)
        dfstg['fit6'] = conflicts / all_recs
        

        fit1 = sum(dfstg.fit1.tolist())/len(dfstg)
        fit2 = sum(dfstg.fit2.tolist())/len(dfstg)
        fit3 = sum(dfstg.fit3.tolist())/len(dfstg)
        fit4 = sum(dfstg.fit4.tolist())/len(dfstg)
        fit5 = sum(dfstg.fit5.tolist())/len(dfstg)
        fit6 = sum(dfstg.fit6.tolist())/len(dfstg)
        
        fit.append((fit1, fit6, fit2, fit3, fit4, fit5))
    
    # check if all student groups are present in the entire population. maximize this.
    fit7 = len(stg) / len(stgroups)
    

    if fit:
        fit = list(zip(*fit))
        fit = [sum(i)/len(i) for i in fit]
        fit.append(fit7)
        return tuple(fit)
    else:
        return 1, 1, 0, 0, 0, 0, 0


       
def cxSet(ind1, ind2):
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
    
    return ind1, ind2
    

def mutSet(individual):
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add((random.choice(module), random.choice(stgroups), random.choice(day), random.choice(location), random.choice(timing)))
    return individual,


toolbox.register("evaluate", evaltimetable)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


def main():
    random.seed(32)
    NGEN = 100 # of generations 
    MU = 15 # of individuals to select for next generations
    LAMBDA = MU * 2 # of childern to produce
    CXPB = 0.7
    MUTPB = 0.2
    
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront() #contains all the non-dominated individuals that ever lived in the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    #stats.register("std", np.std, axis=0)
    #stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof, verbose=True)
    return pop, stats, hof
                 
if __name__ == "__main__":
    pop, stats, hof = main()
    fittest = max([i.fitness.values for i in hof])
    fitt = list([i for i in hof if i.fitness.values == fittest][0])
    
    csvlist = pd.DataFrame(fitt, columns = ['Module' , 'StudentGroup', 'Weekday', 'Venue', 'Time'])
    csvlist['Module']       = csvlist['Module'].apply(lambda x: moduletext[int(x)])
    csvlist['StudentGroup'] = csvlist['StudentGroup'].apply(lambda x: stgroupstext[int(x)])
    csvlist['Weekday']      = csvlist['Weekday'].apply(lambda x: daytext[int(x)])
    csvlist['Venue']       = csvlist['Venue'].apply(lambda x: locationtext[int(x)])
    csvlist['Time']         = csvlist['Time'].apply(lambda x: '{0} to {1}'.format(str(x), str(x+1)) )
    print(csvlist.head())    
    csvlist.to_csv('result.csv', header=True, sep=',', index=False)
    