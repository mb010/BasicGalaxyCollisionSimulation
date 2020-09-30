#STEP SLIDER PLOT v15

#   Inserting required modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arctan2
from mpmath import linspace
from cmath import pi
import ValueCheck

#Saving the plotable data
filename = ValueCheck.filename



#startingconds = ['N',N,'eta',eta,'deltat',deltat,'tcrit',tcrit,'eps2',eps2,'BlackHoleOnOff',BlackHoleOnOff,'velFactor',velFactor,'radius',radius,'BHMass',BHMass,'NoOfRings',NoOfRings,'x_timed',x_timed,'y_timed',y_timed,'z_timed',z_timed,'nsteps',nsteps,'step_time',step_time,]
#data = [x_timed,y_timed,z_timed,step_time]
def speedcheck(filename):
	v = np.loadtxt("Calculated Values/{}v.txt".format(filename),	dtype = np.float64 , delimiter = ',' , ndmin = 2)
	N = v.shape[1]
	stepnumber = v.shape[0]
	Color_array = np.ndarray([N,3],dtype=float)
	print('stepno:',stepnumber)
	for i in range(N):
		Color_array[i,0] = 1-(i+1)/N
		Color_array[i,1] = 1-((N-i))/N
		Color_array[i,2] = (i+1)/N

	#Setup plot:
	fig, ax = plt.subplots()
	for i in range(v.shape[1]):
		plt.plot(v[:,i], color=(Color_array[i,0],Color_array[i,1],Color_array[i,2]),label='Particle {}'.format(i))
		#Invalid rgba argument. Use something to distinguish individual lines. Repeat and put into defined function for each value which you want to plot

	plt.axis([-10,stepnumber,-10,14000])#max(v[:,1])])
	plt.title('Data analysis of simulation (Particle Speed [pc/myr])')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number')
	ax.set_ylabel('Speed	[pc/myr]')
	plt.show()

speedcheck(filename)
