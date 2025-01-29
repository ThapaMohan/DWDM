#data visualization
import pandas as pd
import matplotlib.pyplot as plt
#reading the database
data=pd.read_csv("Housing.csv")
#Bar chart with day against tip
plt.bar(data['furnishingstatus'],data['area'])
plt.title("Bar Chart")
#Setting the X and Y labels
plt.xlabel('Furnished status')
plt.ylabel('Area')
#Adding the legends
plt.show()