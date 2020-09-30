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
import angled2dplot as twodview

starttime = time.time()
	#Starting Conditions:
#---------------#filename = "N[ 6 10 14],vFac[1.35 1.54 2.  ],rad10,sCrit100000,"	#Quite pretty.v10
#Primary Values:
#---------------
TitleTag		= "gal2 "
N 			= np.array([20,10],dtype=np.int32)			#N[0] is inner most ring of particles with N[0] particles, etc.
velFactor 		= np.asarray([[1.32,1.03],[1.32,1.03],[0.51,0.4]],dtype=np.float64)		# velFactor[0] is factor for inner most ring (then indices move outwards along number of rings), must be the same length as N
#velFactor 		= np.asarray([[1.1,0.6],[1.1,0.6],[1,.88]],dtype=np.float64)		# velFactor[0] is factor for inner most ring (then indices move outwards along number of rings), must be the same length as N
radius 			= 22.5#[kpc]					#R/N[-1] is radius factor. RadFactor*N[i]/N[-1] for radius of ring
stepscrit		= 5.*10**(7)+00+1				#Use final digits to differentiate save files

eta 			= 1e-6#[myr]					#eta and deltat are inextricably linked for propper outputting of data.
deltat 			= 1e0#[myr]					#Lowest order combination found thus far: [1e-10,3e-5](CanBeImproved)Other combinations include: [1e-6,1e-2],[5e-2,1e0]
eps2 			= 1e-2						# Value found to be most accurate so far: 0.000000	
## Secondary Values:							#Values which stay mostly constant and are not listed in the title of the files.
tcrit 			= 600#[myr]					# Limited by the 10**10 value used in the algorithm
BHMass 			= 1e11#np.sum(N)#[SolMass]			#BHMass does not affect the nsteps for which the programme runs (Or the final time of the run).


#Transformation Conditions & Galaxy
NumberOfGalaxies = 3
GalaxyMassRatios = np.array([1.,1.,2/10],dtype=np.float64)
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
BlackHoleOnOff 		= 1						#1 for on, 0 for off.
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
np.savetxt("Calculated Values/{}ECheck.txt".format(datalabel) ,	 Earr/(1e0) , delimiter = ',' , newline = '\n' , header = 'Energy [J]:    '+datalabel)
np.savetxt("Calculated Values/{}DistO.txt".format(datalabel),	 R , delimiter = ',' , newline = '\n' , header = 'distance from Origin [pc]:    '+datalabel)
nsteps = x_timed.shape[0]

endtime = time.time()
minute = round(endtime-starttime,3)/60
seconds = round(endtime-starttime,3)%60
print('No.1: Runtime = ',round(minute,2),"min ")# ,round(seconds,0),"seconds")



#Setup plot:
fig = plt.figure(figsize=(15/2.54, 15/2.54))
ax = fig.add_subplot(111, projection='3d')

# Slider initialisation
step_init = 4
axstep = plt.axes([0.25,0.1,0.65,0.03], facecolor = 'lightgoldenrodyellow')
sstep = Slider(axstep, 'Time [myr]', valmin=1,valmax = x_timed.shape[0]-1, valinit = step_init)	#step_init [myr]

# 3D Slider plot
xs = x_timed[0]
ys = y_timed[0]
zs = z_timed[0]

c1 = '#01148c'
c2 = '#8c0101'
c3 = '#038c01'
alph = 0.5
size1mass = 30
size3galmass = size1mass*GalaxyMassRatios[2]

ax.scatter(
	xs[:np.sum(N)],
	ys[:np.sum(N)],
	zs[:np.sum(N)],c=c1,marker='.',s=size1mass,alpha=alph)
ax.scatter(xs[np.sum(N)],
	ys[np.sum(N)],
	zs[np.sum(N)], c=c1,marker='.',s=size1mass*30,alpha=alph)
#ax.set_title('Results')
#ax.set_facecolor('k')
if (NumberOfGalaxies > 1):
	ax.scatter(
		xs[np.sum(N)+1:2*np.sum(N)+1],
		ys[np.sum(N)+1:2*np.sum(N)+1],
		zs[np.sum(N)+1:2*np.sum(N)+1], c=c2, marker='.',s=size1mass,alpha=alph)
	ax.scatter(xs[2*np.sum(N)+1],
		ys[2*np.sum(N)+1],
		zs[2*np.sum(N)+1], c=c2,marker='.',s=size1mass*30,alpha=alph)
		
if (NumberOfGalaxies > 2):
	ax.scatter(
		xs[2*np.sum(N)+2:-1],
		ys[2*np.sum(N)+2:-1],
		zs[2*np.sum(N)+2:-1],c=c3,marker='.',s=size3galmass,alpha=alph)
	ax.scatter(xs[-1],ys[-1],zs[-1], c=c3,marker='.',s=size3galmass*30,alpha=alph)
#ax.scatter(0,0,0,c='r',marker='x')

ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_zlabel('z [kpc]')
ax.set_xlim(-55,55)
ax.set_ylim(-55,55)
ax.set_zlim(-55,55)


def update(val):
	step = int(round(sstep.val,0))
	ax.clear()
	
	xs = x_timed[step]
	ys = y_timed[step]
	zs = z_timed[step]

	ax.scatter(
		xs[:np.sum(N)],
		ys[:np.sum(N)],
		zs[:np.sum(N)],c=c1,marker='.',s=size1mass,alpha=alph)
	ax.scatter(xs[np.sum(N)],
		ys[np.sum(N)],
		zs[np.sum(N)], c=c1,marker='.',s=size1mass*30,alpha=alph)
	#ax.set_facecolor('k')
	if (NumberOfGalaxies > 1):
		ax.scatter(
			xs[np.sum(N)+1:2*np.sum(N)+1],
			ys[np.sum(N)+1:2*np.sum(N)+1],
			zs[np.sum(N)+1:2*np.sum(N)+1], c=c2, marker='.',s=size1mass,alpha=alph)
		ax.scatter(xs[2*np.sum(N)+1],
			ys[2*np.sum(N)+1],
			zs[2*np.sum(N)+1], c=c2,marker='.',s=size1mass*30,alpha=alph)
			
	if (NumberOfGalaxies > 2):
		ax.scatter(
			xs[2*np.sum(N)+2:-1],
			ys[2*np.sum(N)+2:-1],
			zs[2*np.sum(N)+2:-1],c=c3,marker='.',s=size3galmass,alpha=alph)
		ax.scatter(xs[-1],ys[-1],zs[-1], c=c3,marker='.',s=size3galmass*30,alpha=alph)
	
	#ax.scatter(0,0,0,c='k',marker='x')
	
	#ax.set_title('gal2 results')
	ax.set_xlim(-55,55)
	ax.set_ylim(-55,55)
	ax.set_zlim(-55,55)
	ax.set_xlabel('x [kpc]')
	ax.set_ylabel('y [kpc]')
	ax.set_zlabel('z [kpc]')
	fig.canvas.draw_idle()

E1 = 100*(1-min(Earr)/Earr[0])
E2 = 100*(max(Earr)/Earr[0]-1)
print('Energy change maximum: '+str(max(E1,E2))+'%')
sstep.on_changed(update)
plt.show()

#twodview.finalsliderplt(0,3,0,x_timed,y_timed,z_timed)

#.echeck(datalabel)
#ValueCheck.distcheck(datalabel)
#ValueCheck.speedcheck(datalabel)
#ValueCheck.timecheck(datalabel)"""
