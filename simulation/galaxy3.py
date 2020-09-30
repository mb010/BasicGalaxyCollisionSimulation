#STEP SLIDER PLOT v18
#   Inserting required modules
import numpy as np
import time
import random
import math
import example_cy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import ValueCheck

starttime = time.time()
	#Starting Conditions:
#---------------#filename = "N[ 6 10 14],vFac[1.35 1.54 2.  ],rad10,sCrit100000,"	#Quite pretty.v10
#Primary Values:
#---------------
TitleTag		= "gal3.0 "
N 			= np.array([20,10],dtype=np.int32)			#N[0] is inner most ring of particles with N[0] particles, etc.
velFactor 		= np.asarray([[1.32,1.03],[0.35,.265],[0.85,.655],[0.51,0.4],[0.258,0.194],[0.65,.5],[0.75,.59],[0.67,.525]],dtype=np.float64)		# velFactor[0] is factor for inner most ring (then indices move outwards along number of rings), must be the same length as N
radius 			= 30#[kpc]					#R/N[-1] is radius factor. RadFactor*N[i]/N[-1] for radius of ring
stepscrit		= 5.*10**(8)+00+1				#Use final digits to differentiate save files

eta 			= 1e-6#[myr]					#eta and deltat are inextricably linked for propper outputting of data.
deltat 			= 1e0#[myr]					#Lowest order combination found thus far: [1e-10,3e-5](CanBeImproved)Other combinations include: [1e-6,1e-2],[5e-2,1e0]
eps2 			= 1e-2						# Value found to be most accurate so far: 0.000000	
## Secondary Values:							#Values which stay mostly constant and are not listed in the title of the files.
tcrit 			= 5000#[myr]					# Limited by the 10**10 value used in the algorithm
BlackHoleOnOff 		= 1						#1 for on, 0 for off.
BHMass 			= 1e11#np.sum(N)#[SolMass]			#BHMass does not affect the nsteps for which the programme runs (Or the final time of the run).

#Transformation Conditions & Galaxy
NumberOfGalaxies = 1
GalaxyMassRatios = np.array([1,1/10,1/2,1/5,1/20,3/10,4/10,1/3],dtype=np.float64)
xangle = np.array([1,			0,		60.*np.pi/180],dtype=np.float64)#[rad]#Must have length >= NumberOfGalaxies , Angle around the x angle around which to rotate
yangle = np.array([0*np.pi/180.,	-30.*np.pi/180,	0.*np.pi/180],dtype=np.float64)#[rad]#Must have length >= NumberOfGalaxies , Angle around the y angle around which to rotate
zangle = np.array([0.,			0.,		15*np.pi/180.],dtype=np.float64)#[rad]#Must have length >= NumberOfGalaxies , Angle around the z angle around which to rotate

#"""#Transformations for playing around with
r_transform = np.array([[10.,0.,0.],#first galaxy
			[15.,35.,0.],#second galaxy
			[-70.,-50.,40.]],#third galaxy
			dtype=np.float64)#[kpc]

v_transform = np.array([[0.,3.,5.],#first galaxy
			[-60.,-5.,0.],#second galaxy
			[-5.,-15.,-160.]],#third galaxy
			dtype=np.float64)#[pc/myr]#first dimension length must be greater than NumberOfGalaxies

"""r_transform	 = np.array([[-12.,-35.,0.],#first galaxy
			     [12.,35.,0.],#second galaxy
			     [20.,-100.,30.]],#third galaxy
			dtype=np.float64)#[kpc]

v_transform	 = np.array([[60.,5.,0.],#first galaxy
			     [-60.,-5.,0.],#second galaxy
			     [-5.,-5.,-150.]],#third galaxy
			dtype=np.float64)#[pc/myr]	#first dimension length must be greater than NumberOfGalaxies
#"""
"""#transformations for previously best double tail

GalaxyMassRatios = np.array([1.,1.,1/10],dtype=np.float64)
xangle		 = np.array([0,				0,				np.pi/2.-15.*np.pi/180],dtype=np.float64)#[rad]		#Must have length >= NumberOfGalaxies , Angle around the x angle around which to rotate
yangle		 = np.array([np.pi/6.-0.*np.pi/360,	-(np.pi/6.-0.*np.pi/180),	0.],dtype=np.float64)#[rad]		#Must have length >= NumberOfGalaxies , Angle around the y angle around which to rotate
zangle		 = np.array([0.,			0.,				0*np.pi/2.],dtype=np.float64)#[rad]		#Must have length >= NumberOfGalaxies , Angle around the z angle around which to rotate

r_transform	 = np.array([[-15.,-35.,0.],#first galaxy
			     [15.,35.,0.],#second galaxy
			     [-80.,-65.,40.]],#third galaxy
			dtype=np.float64)#[kpc]

v_transform	 = np.array([[60.,5.,0.],#first galaxy
			     [-60.,-5.,0.],#second galaxy
			     [-5.,-5.,-180.]],#third galaxy
			dtype=np.float64)#[pc/myr]	#first dimension length must be greater than NumberOfGalaxies
#"""

#Call function to get values:
x_timed , y_timed , z_timed , v_timed , nsteps , step_time , Earr = example_cy.NBody(N,eta,deltat,tcrit,eps2,BlackHoleOnOff,velFactor,radius*1000/N[0],BHMass,stepscrit,r_transform*1000,v_transform,xangle,yangle,zangle,NumberOfGalaxies,GalaxyMassRatios,0)

x_timed = x_timed*1e-3	#converting from pc to kpc
y_timed = y_timed*1e-3	#converting from pc to kpc
z_timed = z_timed*1e-3	#converting from pc to kpc
R = np.sqrt(np.square(y_timed)+np.square(x_timed)+np.square(z_timed))
#Saving the plotable data
datalabel = "{}N{},vFac{},sCrit{},eta{},deltat{},NoG{}".format(TitleTag,N,velFactor,stepscrit,eta,deltat,NumberOfGalaxies)
np.savetxt("Calculated Values/{}xtime.txt".format(datalabel) ,	 x_timed , delimiter = ',' , newline = '\n' , header = 'x_timed [pc]:    '+datalabel)
np.savetxt("Calculated Values/{}yTime.txt".format(datalabel) ,	 y_timed , delimiter = ',' , newline = '\n' , header = 'y_timed [pc]:    '+datalabel)
np.savetxt("Calculated Values/{}zTime.txt".format(datalabel) ,	 z_timed , delimiter = ',' , newline = '\n' , header = 'z_timed [pc]:    '+datalabel)
np.savetxt("Calculated Values/{}v.txt".format(datalabel) ,	 v_timed , delimiter = ',' , newline = '\n' , header = 'v_timed [pc/myr]:    '+datalabel)
np.savetxt("Calculated Values/{}sTime.txt".format(datalabel) ,	 step_time , delimiter = ',' , newline = '\n' , header = 'step_time [myr]:    '+datalabel)
np.savetxt("Calculated Values/{}ECheck.txt".format(datalabel) ,	 Earr , delimiter = ',' , newline = '\n' , header = 'Energy [J]:    '+datalabel)
np.savetxt("Calculated Values/{}DistO.txt".format(datalabel),	 R , delimiter = ',' , newline = '\n' , header = 'distance from Origin [pc]:    '+datalabel)
nsteps = x_timed.shape[0]

endtime = time.time()
minute = round(endtime-starttime,3)/60
seconds = round(endtime-starttime,3)%60
print('No.2: Runtime = ',round(minute,2),"min ")# ,round(seconds,0),"seconds")



#Setup plot:
fig = plt.figure(figsize=(20/2.54, 20/2.54))
ax = fig.add_subplot(111, projection='3d')

# Slider initialisation
step_init = 4
axstep = plt.axes([0.25,0.1,0.65,0.03], facecolor = 'lightgoldenrodyellow')
sstep = Slider(axstep, 'Time [myr]', valmin=1,valmax = x_timed.shape[0]-1, valinit = step_init)	#step_init [myr]

# 3D Slider plot
xs = x_timed[0]
ys = y_timed[0]
zs = z_timed[0]

ax.scatter(
	xs[:np.sum(N)],
	ys[:np.sum(N)],
	zs[:np.sum(N)],c='#01148c',marker='.')
ax.scatter(xs[np.sum(N)],
	ys[np.sum(N)],
	zs[np.sum(N)], c='k',marker='o')
if (NumberOfGalaxies > 1):
	ax.scatter(
		xs[np.sum(N)+1:2*np.sum(N)+1],
		ys[np.sum(N)+1:2*np.sum(N)+1],
		zs[np.sum(N)+1:2*np.sum(N)+1], c='#8c0101', marker='.')
	ax.scatter(xs[2*np.sum(N)+1],
		ys[2*np.sum(N)+1],
		zs[2*np.sum(N)+1], c='k',marker='p')
		
if (NumberOfGalaxies > 2):
	ax.scatter(
		xs[2*np.sum(N)+2:-1],
		ys[2*np.sum(N)+2:-1],
		zs[2*np.sum(N)+2:-1],c='#038c01',marker='.')
	ax.scatter(xs[-1],ys[-1],zs[-1], c='k',marker='.')
#ax.scatter(0,0,0,c='r',marker='x')

ax.set_xlabel('2x [kpc]')
ax.set_ylabel('2y [kpc]')
ax.set_zlabel('2z [kpc]')
ax.set_xlim(-110,110)
ax.set_ylim(-110,110)
ax.set_zlim(-110,110)


def update(val):
	step = int(round(sstep.val,0))
	ax.clear()
	
	xs = x_timed[step]
	ys = y_timed[step]
	zs = z_timed[step]

	ax.scatter(
		xs[:np.sum(N)],
		ys[:np.sum(N)],
		zs[:np.sum(N)],c='#01148c',marker='.')
	ax.scatter(xs[np.sum(N)],
		ys[np.sum(N)],
		zs[np.sum(N)], c='k',marker='o')
	if (NumberOfGalaxies > 1):
		ax.scatter(
			xs[np.sum(N)+1:2*np.sum(N)+1],
			ys[np.sum(N)+1:2*np.sum(N)+1],
			zs[np.sum(N)+1:2*np.sum(N)+1], c='#8c0101', marker='.')
		ax.scatter(xs[2*np.sum(N)+1],
			ys[2*np.sum(N)+1],
			zs[2*np.sum(N)+1], c='k',marker='p')
			
	if (NumberOfGalaxies > 2):
		ax.scatter(
			xs[2*np.sum(N)+2:-1],
			ys[2*np.sum(N)+2:-1],
			zs[2*np.sum(N)+2:-1],c='#038c01',marker='.')
		ax.scatter(xs[-1],ys[-1],zs[-1], c='k',marker='.')
	
	#ax.scatter(0,0,0,c='k',marker='x')
	
	ax.set_xlim(-110,110)
	ax.set_ylim(-110,110)
	ax.set_zlim(-110,110)
	ax.set_xlabel('2x [kpc]')
	ax.set_ylabel('2y [kpc]')
	ax.set_zlabel('2z [kpc]')
	fig.canvas.draw_idle()

E1 = 100*(1-min(Earr)/Earr[0])
E2 = 100*(max(Earr)/Earr[0]-1)
print('Energy change maximum: '+str(max(E1,E2))+'%')
sstep.on_changed(update)
plt.show()


#ValueCheck.echeck(datalabel)
#ValueCheck.distcheck(datalabel)
#ValueCheck.speedcheck(datalabel)
#ValueCheck.timecheck(datalabel)
