#STEP SLIDER PLOT v19
#   Inserting required modules
import numpy as np
import time
import random
import math
import example_cy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import ValueCheck


def simulate(datalabel,anglefactor,radiusfactor,TCRIT): 
	starttime = time.time()
	#Starting Conditions:
	#---------------#filename = "N[ 6 10 14],vFac[1.35 1.54 2.  ],rad10,sCrit100000,"	#Quite pretty.v10
	#Primary Values:
	#---------------
	N 			= np.array([20,10],dtype=np.int32)			#N[0] is inner most ring of particles with N[0] particles, etc.
	velFactor 		= np.asarray([[1.32,1.03],[1.32,1.03],[0.258,0.194],[0.65,.5],[0.35,.265]],dtype=np.float64)		# velFactor[0] is factor for inner most ring (then indices move outwards along number of rings), must be the same length as N
	radius 			= 22.5#[kpc]					#R/N[-1] is radius factor. RadFactor*N[i]/N[-1] for radius of ring
	stepscrit		= 5.*10**(7)+00+1				#Use final digits to differentiate save files

	eta 			= 1e-6#[myr]					#eta and deltat are inextricably linked for propper outputting of data.
	deltat 			= 1e0#[myr]					#Lowest order combination found thus far: [1e-10,3e-5](CanBeImproved)Other combinations include: [1e-6,1e-2],[5e-2,1e0]
	eps2 			= 1e-2						# Value found to be most accurate so far: 0.000000	
	## Secondary Values:							#Values which stay mostly constant and are not listed in the title of the files.
	tcrit 			= 300#[myr]					# Limited by the 10**10 value used in the algorithm
	BHMass 			= 1e11#np.sum(N)#[SolMass]			#BHMass does not affect the nsteps for which the programme runs (Or the final time of the run).

	#Transformation Conditions & Galaxy
	NumberOfGalaxies = 3
	GalaxyMassRatios = np.array([1.,1.,1/20,3/10,1/10],dtype=np.float64)
	xangle = np.array([-10*np.pi/180,	20*np.pi/180,		+90.*np.pi/180],dtype=np.float64)#[rad]#Must have length >= NumberOfGalaxies , Angle around the x angle around which to rotate
	yangle = np.array([np.pi+30*np.pi/180.,	np.pi-30.*np.pi/180,	0.0],dtype=np.float64)#[rad]#Must have length >= NumberOfGalaxies , Angle around the y angle around which to rotate
	zangle = np.array([0.*np.pi/180,	0.,			0*np.pi/180.],dtype=np.float64)#[rad]#Must have length >= NumberOfGalaxies , Angle around the z angle around which to rotate

	#"""#Transformations for playing around with
	r_transform = np.array([[-40.,+25,0.],#first galaxy
				[+40.,-25,0.],#second galaxy
				[12.,-12,75]],#third galaxy
				dtype=np.float64)#[kpc]
	#THIS IS REALLY GOOD, BUT I NEED TO BREAK THE BLUE TAIL UP

	v_transform = np.array([[+110.,0.,0.],#first galaxy
				[-110.,0.,0.],#second galaxy
				[0.,0.,-170.]],#third galaxy
				dtype=np.float64)#[pc/myr]
	#first dimension length must be greater than NumberOfGalaxies


	#Call function to get values:
	BlackHoleOnOff = 1
	radius = radius*radiusfactor
	x_timed , y_timed , z_timed , v_timed , nsteps , step_time , Earr = example_cy.NBody(N,eta,deltat,TCRIT+1,eps2,BlackHoleOnOff,velFactor,radius*1000/N[0],BHMass,stepscrit,r_transform*1000,v_transform,xangle,yangle,zangle,NumberOfGalaxies,GalaxyMassRatios,anglefactor)

	x_timed = x_timed*1e-3	#converting from pc to kpc
	y_timed = y_timed*1e-3	#converting from pc to kpc
	z_timed = z_timed*1e-3	#converting from pc to kpc
	R = np.sqrt(np.square(y_timed)+np.square(x_timed)+np.square(z_timed))
	#Saving the plotable data
	np.savetxt("Merger Values/{}xtime.txt".format(datalabel) ,	 x_timed , delimiter = ',' , newline = '\n' , header = 'x_timed [pc]:    '+datalabel)
	np.savetxt("Merger Values/{}yTime.txt".format(datalabel) ,	 y_timed , delimiter = ',' , newline = '\n' , header = 'y_timed [pc]:    '+datalabel)
	np.savetxt("Merger Values/{}zTime.txt".format(datalabel) ,	 z_timed , delimiter = ',' , newline = '\n' , header = 'z_timed [pc]:    '+datalabel)
	nsteps = x_timed.shape[0]

	endtime = time.time()
	minute = round(endtime-starttime,3)/60
	seconds = round(endtime-starttime,3)%60
	#print(datalabel + ' calculations: Runtime = ',round(minute,2),"min ")# ,round(seconds,0),"seconds
	return (x_timed, y_timed, z_timed)


