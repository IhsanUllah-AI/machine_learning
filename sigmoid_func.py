import numpy as np
import matplotlib.pyplot as plt


inp_array=np.array([1,2,3,4])
exp_array=np.exp(inp_array)

print("input array  :",inp_array)
print("exp array  :",exp_array)


#use sigmoid function gu=iive result bet zero and one
def sigmoid(x):
    g=1/(1+np.exp(-x))
    return g

z_tmp=np.arange(-10,11) #create array from -10 to plus 10
y=sigmoid(z_tmp)

#plot them
fig,ax=plt.subplots()
ax.plot(z_tmp,y,color='red')
plt.title("sigmoid function")
plt.xlabel("Z")
plt.ylabel("g(z)")
plt.show()

#logictic reg used for classification it has done in two step
#z=wx+b of linear regression
#use sigmiod function


import pandas as pd
from sklearn.linear_model import LinearRegression
lr=LinearRegression()


df=pd.DataFrame({
    "tumer_size":[1.1,1.4,1.8,2.5,3.4,3.9,4,5.6,7],
    "malognant_benign":[0,0,0,0,0,1,1,1,1]
})

x=df[['tumer_size']]
y=df['malognant_benign']

lr.fit(x,y)

#step 1
z=lr.predict(x)

#step 2
def sigmoid_fun(z):
    g=1/(1+np.exp(-z))
    return g

y=sigmoid_fun(z)

plt.plot(z,y,color='blue')
plt.xlabel("Z")
plt.ylabel("g(z)")
plt.title("Logistic Regression")
plt.show()


