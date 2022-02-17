# =============================================================================
# import dateutil.parser as parser
# from dateutil import tz
# import collections, json
# 
# temp = dict()
# with open('../Input/result.json') as tdump:
#     videos_time_ordered = json.load(tdump)
# for ts, all_data in videos_time_ordered['timestamp'].items():
#     ts = parser.parse(ts)
#     ts = ts.astimezone(tz.gettz('America/New_York') )
#     temp[ts] = all_data
# videos_time_ordered['timestamp'] = collections.OrderedDict(sorted(temp.items()))
# videos_time_orderednew = dict()
# videos_time_orderednew['timestamp'] = videos_time_ordered['timestamp']
# for i,j in videos_time_ordered['timestamp'].items():
#     videos_time_orderednew['timestamp'][i] = dict()
#     for x,y,z in j:
#         try:
#             videos_time_orderednew['timestamp'][i][z] += list([[x,y]])
#         except KeyError:
#             videos_time_orderednew['timestamp'][i][z] = list([[x,y]])
#     for f in videos_time_ordered['timestamp']:
#         for x,y,z in videos_time_orderednew['timestamp'][i]:
#             if f in z:
#                 pass
#             else:
#                 videos_time_orderednew['timestamp'][i][z] += list([['NA',f]])
# 
# #print(videos_time_orderednew['timestamp'])
# videos_time_ordered['timestamp'] = videos_time_orderednew['timestamp']
# 
# temp = dict()
# for ts, all_data in videos_time_ordered['timestamp'].items():
#     temp[str(ts)] = all_data
# videos_time_ordered['timestamp'] = temp
# 
# with open('../Input/result_mod.json', 'w') as tdump:
#     json.dump(videos_time_ordered, tdump)
# 
# =============================================================================


from win32api import GetSystemMetrics

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

print("Width =", width)
print("Height =", height)

#import numpy as np
#c = np.arange(width*height).reshape((width,height))
#a1 = np.split(c,10,axis=1) 
#print(a1[0])

p = ceiling(sqrt(n))
best_merit_yet = merit_function(p,p,0)
best_configuration_yet = (p,p)
for p from floor(sqrt(n)) to downward:
  # we need pq >= n and q as near to p as possible, which means (since p is too small) as small as possible
  q = ceiling(n/p)
  if max(merit_function(n/p,n/q,0), merit_function(n/q,n/p,0)) < best_merit_yet:
    break
  n_wasted = p*q-n
  merit1 = merit_function(n/p,n/q,n_wasted)
  merit2 = merit_function(n/q,n/p,n_wasted)
  if max(merit1,merit2) > best_merit_yet:
    if merit1 > merit2:
      best_configuration_yet = (p,q)
      best_merit_yet = merit1
    else:
      best_configuration_yet = (q,p)
      best_merit_yet = merit2