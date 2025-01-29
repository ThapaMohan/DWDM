import numpy
print("Mohan Thapa Masharangi");
speed_of_cars = [80,100,89,99,94,70,79,120,86,88,75,85]
x= numpy.mean(speed_of_cars)
print("Mean:",x)
y = numpy.median(speed_of_cars)
print("Median:",y)
from scipy import stats
z= stats.mode(speed_of_cars)
print(f"Mode number: {int(z.mode)} and Frequency: {int(z.count)}")