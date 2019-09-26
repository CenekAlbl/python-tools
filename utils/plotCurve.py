import numpy as np
#Plot a curve 
def plotCurve(plt,fn,xlim,step=1,color='r'):   
    t = range(xlim[0],xlim[1],step)
    y = list(map(fn, t))
    plt.plot(t,y,color)