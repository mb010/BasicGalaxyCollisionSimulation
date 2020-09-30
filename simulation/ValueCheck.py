#STEP SLIDER PLOT v15

#   Inserting required modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arctan2
#from mpmath import linspace
#from cmath import pi


#Saving the plotable data
filename = "N[10 20],vFac[1.03 1.32],sCrit20000000.0,eta1e-06,deltat1.0,NoG2"
#filename = galaxy.datalabel



#startingconds = ['N',N,'eta',eta,'deltat',deltat,'tcrit',tcrit,'eps2',eps2,'BlackHoleOnOff',BlackHoleOnOff,'velFactor',velFactor,'radius',radius,'BHMass',BHMass,'NoOfRings',NoOfRings,'x_timed',x_timed,'y_timed',y_timed,'z_timed',z_timed,'nsteps',nsteps,'step_time',step_time,]
#data = [x_timed,y_timed,z_timed,step_time]
def speedcheck(filelabel):
	v = np.loadtxt("Calculated Values/{}v.txt".format(filelabel),	dtype = np.float64 , delimiter = ',' , ndmin = 2)
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
	plt.title('Data analysis of simulation: Particle Speed')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number')
	ax.set_ylabel('Speed [pc/myr]')
	plt.show()
	
def distcheck(filelabel):
	Dist 	= np.loadtxt("Calculated Values/{}DistO.txt".format(filelabel),	dtype = np.float64 , delimiter = ',' , ndmin = 2)
	N = Dist.shape[1]
	stepnumber = Dist.shape[0]
	Color_array = np.ndarray([N,3],dtype=float)
	print('stepno:',stepnumber)
	for i in range(N):
		Color_array[i,0] = 1-(i+1)/N
		Color_array[i,1] = 1-((N-i))/N
		Color_array[i,2] = (i+1)/N

	#Setup plot:
	fig, ax = plt.subplots()
	for i in range(Dist.shape[1]):
		plt.plot(Dist[:,i], color=(Color_array[i,0],Color_array[i,1],Color_array[i,2]),label='Particle {}'.format(i))
		#Invalid rgba argument. Use something to distinguish individual lines. Repeat and put into defined function for each value which you want to plot

	plt.axis([-10,stepnumber,-10,max(Dist[0:4000,1])])
	plt.title('Data analysis of simulation: Distance from Origin')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number')
	ax.set_ylabel('Distance [pc]')
	plt.show()
	
def echeck(filelabel):
	ECheck	= np.loadtxt("Calculated Values/{}ECheck.txt".format(filelabel),	dtype = np.float64 , ndmin = 1)
	stepnumber = ECheck.shape[0]
	Color_array = np.ndarray([3],dtype=float)
	print('stepno:',stepnumber)

	#Setup plot:
	fig, ax = plt.subplots()
	#ax.set_yscale('log')	
	plt.plot(ECheck, 'k.',label='Energy of System')
	

	plt.axis([0,stepnumber,min(ECheck),max(ECheck)])
#	plt.title('Single Galaxy System Energy \n')
#	plt.legend(loc='best')
	 
	ax.set_xlabel('Time [myr]')
	ax.set_ylabel('Energy [J]')

	plt.show()

def timecheck(filelabel):
	stept	= np.loadtxt("Calculated Values/{}sTime.txt".format(filelabel),	dtype = np.float64 , ndmin = 1)
	stepnumber = stept.shape[0]
	Color_array = np.ndarray([3],dtype=float)
	print('stepno:',stepnumber)

	#Setup plot:
	fig, ax = plt.subplots()
	plt.plot(stept, 'r.',label='Time')

	plt.axis([-10,stepnumber,-0.005,max(stept)])
	plt.title('Data analysis of simulation: System Time Progression')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number')
	ax.set_ylabel('Time [myr]')
	plt.show()
