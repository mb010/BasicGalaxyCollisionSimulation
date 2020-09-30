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

def timecheck(filename):
	stept	= np.loadtxt("Calculated Values/{}sTime.txt".format(filename),	dtype = np.float64 , ndmin = 1)
	stepnumber = stept.shape[0]
	Color_array = np.ndarray([3],dtype=float)
	print('stepno:',stepnumber)

	#Setup plot:
	fig, ax = plt.subplots()
	plt.plot(stept, 'r.')

	plt.axis([-10,stepnumber,-0.005,1])
	plt.title('Data analysis of simulation: Time of System')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number')
	ax.set_ylabel('Time [myr]')
	plt.show()

timecheck(filename)
