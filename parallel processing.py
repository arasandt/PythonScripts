import multiprocessing as mp
import numpy as np
#
#
#pool = mp.Pool(mp.cpu_count())
results = []
#
# Prepare data
np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[2, 2])
data = arr.tolist()
data[:5]

print(data)
print(arr)
#
# Step 1: Redefine, to accept `i`, the iteration number
def howmany_within_range2(i, row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return (i, count)
#
#
# Step 2: Define callback function to collect the output in `results`
#def collect_result(result):
#    global results
#    results.append(result)

#
## Step 3: Use loop to parallelize
#for i, row in enumerate(data):
#    print(i, row)
#    pool.apply_async(howmany_within_range2, args=(i, row, 4, 8), callback=collect_result)
#
## Step 4: Close Pool and let all the processes complete    
#pool.close()
#pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
#
## Step 5: Sort results [OPTIONAL]
#results.sort(key=lambda x: x[0])
#results_final = [r for i, r in results]
#
#print(results_final[:10])
##> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]


def cube(x):
    return x**3


def call_process():
    pool = mp.Pool(processes=4)
    #results = [pool.apply(cube, args=(x,)) for x in range(1,7)]
    results = [pool.apply(howmany_within_range2, args=(i, row, 4, 8)) for i, row in enumerate(data)]
    #results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
    print(results)
    
    
    #results = [pool.apply_async(howmany_within_range2, args=(i, row, 4, 8)) for i, row in enumerate(data)]
    #output = [p.get() for p in results]
    #print(output)
    


if __name__ == '__main__':
    call_process()
