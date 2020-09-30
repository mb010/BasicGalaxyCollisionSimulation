#STEP SLIDER PLOT v15
#   Inserting required modules
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from numpy import arctan2
import ValueCheck

filename="gal2 N[20 10],vFac[[1.32 1.03]_ [1.32 1.03]_ [0.51 0.4 ]],sCrit50000001.0,eta1e-06,deltat1.0,NoG3"
NumberOfGalaxies=3
N=[20,10]
radius=10

#Instead of calling function, we load in the data from the saved files under the respective titles
x_timed = np.loadtxt("Calculated Values/{}xtime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
y_timed = np.loadtxt("Calculated Values/{}yTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
z_timed = np.loadtxt("Calculated Values/{}zTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
step_time = np.loadtxt("Calculated Values/{}sTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
nsteps = x_timed.shape[0]

#Setup plot:
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
#step = np.arange(0,nsteps,1)
step_init = 4
data, = plt.plot(x_timed[0],y_timed[0], 'm.')
data1, = plt.plot(x_timed[0,-1],y_timed[0,-1],'k.')
if (NumberOfGalaxies>=2):
	data2, = plt.plot(x_timed[-1,np.sum(N)+1],y_timed[-1,np.sum(N)+1],'k^')
if (NumberOfGalaxies>=3):
	data3, = plt.plot(x_timed[2*(np.sum(N)+1),-1],y_timed[2*(np.sum(N)+1),-1],'k.')
plt.axis([-radius*2e3, radius*2e3, -radius*2e3, radius*2e3])
plt.title('Recreation Plot')


axstep = plt.axes([0.25,0.1,0.65,0.03], facecolor = 'lightgoldenrodyellow')
sstep = Slider(axstep, 'Step', valmin=2,valmax = x_timed.shape[0], valinit = step_init)	#What unit is step in? [myr]?
  
def update(val):
	step = int(round(sstep.val,0))
	data.set_xdata(x_timed[step-1])
	data.set_ydata(y_timed[step-1])
	xy = x_timed[step-1,0],y_timed[step-1,0]
	data1.set_xdata(x_timed[step-1,-1])
	data1.set_ydata(y_timed[step-1,-1])
	if (NumberOfGalaxies>=2):
		data2.set_xdata(x_timed[step-1,np.sum(N)])
		data2.set_ydata(y_timed[step-1,np.sum(N)])
		if (NumberOfGalaxies>=3):
			data3.set_xdata(x_timed[step-1,-1])
			data3.set_ydata(y_timed[step-1,-1])
	#ax.annotate('%s' % 'o', xy=xy, textcoords='data') # Adding the Annotation to the particle, Use to track particle and remove to create images (for now)
	#ax.annotate.remove()	#Attempt to continuously remove the annotations. - didnt work.
	fig.canvas.draw_idle()

sstep.on_changed(update)
ax.set_xlabel('x [pc]')
ax.set_ylabel('y [pc]')
plt.show()
#ann = plt.annotate (...)
#ann.remove()
