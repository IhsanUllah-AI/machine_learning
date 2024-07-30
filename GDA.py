import numpy as np

def gradient_decent(x,y):
    w_curr=b_curr=0      #start with some value zero 
    iteration=10000
    learning_rate=0.08
    for i in range(iteration):
        y_pred=w_curr*x+b_curr
        n=len(x)
        cost=(1/n)*sum([val**2 for val in (y-y_pred)])
        wd=-(2/n)*sum(x*(y-y_pred))
        bd=-(2/n)*sum(y-y_pred)
        w_curr=w_curr-learning_rate*wd
        b_curr=b_curr-learning_rate*bd
        print("w= {},    b= {},    cost= {},    iteraion {}".format(w_curr,b_curr,cost,i))

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])

gradient_decent(x,y)
