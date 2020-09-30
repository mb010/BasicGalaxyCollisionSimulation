import angled2dplot as plotme
import numpy as np

x = np.loadtxt("Merger Values/Final008MERGERxtime.txt",dtype = np.float64, delimiter = ',')
y = np.loadtxt("Merger Values/Final008MERGERytime.txt",dtype = np.float64, delimiter = ',')
z = np.loadtxt("Merger Values/Final008MERGERztime.txt",dtype = np.float64, delimiter = ',')
plotme.finalnonsliderplt(0,-2,0,x,y,z,0) #more of a y angle I think.
#if z axis is labeled on your left then the rotation is: "positive"
#IF y axis is on the bottom, then rot is: "positive"

"""Final Solution seems to be: time=445myr
x = np.loadtxt("Merger Values/Final008MERGERxtime.txt",dtype = np.float64, delimiter = ',')
y = np.loadtxt("Merger Values/Final008MERGERytime.txt",dtype = np.float64, delimiter = ',')
z = np.loadtxt("Merger Values/Final008MERGERztime.txt",dtype = np.float64, delimiter = ',')
plotme.finalnonsliderplt(0,-2,8,x,y,z,0)"""
