#STEP SLIDER PLOT v15
#   Inserting required modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ValueCheck


def pltinthreed(nstepSlice,filename):	#nstepSlice is the time at which you want to view the system in 3d.
					#filename is the filename of the data group that you want to read in.
	#Instead of calling function, we load in the data from the saved files under the respective titles
	x_timed = np.loadtxt("Calculated Values/{}xtime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	y_timed = np.loadtxt("Calculated Values/{}yTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	z_timed = np.loadtxt("Calculated Values/{}zTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	#nsteps = x_timed.shape[0]

	print(x_timed[nstepSlice],'\n',y_timed[nstepSlice],'\n',z_timed[nstepSlice])

	#Setting up 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = x_timed[nstepSlice]
	ys = y_timed[nstepSlice]
	zs = z_timed[nstepSlice]
	ax.scatter(xs,ys,zs,c='k',marker='o')
	ax.set_xlabel('x position [pc]')
	ax.set_ylabel('y position [pc]')
	ax.set_zlabel('z position [pc]')

	plt.show()

pltinthreed(2058,"N[20 10],vFac[1.32 1.03],sCrit60000001.0,eta1e-06,deltat1.0,NoG2")
