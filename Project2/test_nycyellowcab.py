import sys
import numpy as np
import timeit
sys.path.append('./build')

import cudf
import haversine_library

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

taxi = cudf.read_parquet("/data/csc59866_f24/tlcdata/yellow_tripdata_2009-01.parquet", columns=['Start_Lon', 'Start_Lat', 'End_Lon', 'End_Lat'],
                                                                                        filters=[('Start_Lon', '>=', -74.15), ('Start_Lat', '>=', 40.5774),
                                                                                        ('End_Lon', '<=', -73.7004), ('End_Lat', '<=', 40.9176)])

for x in range(2,13):
    taxi_aux = cudf.read_parquet(f"/data/csc59866_f24/tlcdata/yellow_tripdata_2009-{x:02d}.parquet", columns=['Start_Lon', 'Start_Lat', 'End_Lon', 'End_Lat'],
                                                                                        filters=[('Start_Lon', '>=', -74.15), ('Start_Lat', '>=', 40.5774),
                                                                                        ('End_Lon', '<=', -73.7004), ('End_Lat', '<=', 40.9176)])
    taxi = cudf.concat([taxi, taxi_aux], ignore_index=True)
    del taxi_aux


x1=taxi['Start_Lon'].to_numpy()
y1=taxi['Start_Lat'].to_numpy()
x2=taxi['End_Lon'].to_numpy()
y2=taxi['End_Lat'].to_numpy()
size=len(x1)
dist=np.zeros(size)
print(size)
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)


def haversine(size , x1, y1, x2, y2, dist):
    radius = 6371
    x1, y1, x2, y2 = map(np.radians, [x1, y1, x2, y2])

    dLon = x2 - x1
    dLat = y2 - y1

    a = np.sin(dLat / 2)**2 + np.cos(y1) * np.cos(y2) * np.sin(dLon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a), np.sqrt(1-a))
    dist = c * radius
    return dist

time = '''
haversine(size, x1, y1, x2, y2, dist)
'''
timing = timeit.timeit(setup="from __main__ import haversine, size, x1, y1, x2, y2, dist", stmt=time, number=5)

print("Average time:", timing / 10, "seconds")


del taxi