#v19
import numpy as np
#from mpmath import linspace
#from cmath import pi
from math import floor,ceil

# Functions which rotate my vector structures by angles around the given axis (x,y,z respectively)
def xrot(xangle, vector, N):
	for i in range(0, N):
		vec_saved   = np.cos(xangle)*vector[1,i] - np.sin(xangle)*vector[2,i]
		vector[2,i] = np.sin(xangle)*vector[1,i] + np.cos(xangle)*vector[2,i]
		vector[1,i] = vec_saved
	return (vector)
def yrot(yangle, vector, N):
	for i in range(0, N):
		vec_saved   = np.cos(yangle)*vector[0,i] + np.sin(yangle)*vector[2,i]
		vector[2,i] =-np.sin(yangle)*vector[0,i] + np.cos(yangle)*vector[2,i]
		vector[0,i] = vec_saved
	return (vector)
def zrot(zangle, vector, N):
	for i in range(0, N):
		vec_saved   = np.cos(zangle)*vector[0,i] - np.sin(zangle)*vector[1,i]
		vector[1,i] = np.sin(zangle)*vector[0,i] + np.cos(zangle)*vector[1,i]
		vector[0,i] = vec_saved
	return (vector)

# Used in calculations for orbital velocity
def ReducedMass(body,R,N,BHM):
	mass_rsum = BHM*0
	mass_sum = BHM
	for i in range(0,N):
		mass_rsum += R[i]*body[i]
		mass_sum += body[i]
	mass_red = mass_rsum/mass_sum
	return mass_red

# Orbital velocity calculations
def v_orb(R,mass_red,Gpcmyr,BHM):
	v_orb = np.sqrt(Gpcmyr*BHM/R)
	return v_orb #[pc/myr]
	

def positions(R, phi, N, BH, r_transform, xangle, yangle, zangle):
	# Setting positions for galaxy around origin
	x0 = np.zeros([3,N+2*BH],np.float64)
	for k in range(0,N):
		x0[0,k] = R[k] * np.cos(phi[k]) #[pc]
		x0[1,k] = R[k] * np.sin(phi[k]) #[pc]
		x0[2,k] = 0 #[pc]
	x0 = xrot(xangle,x0,N)
	x0 = yrot(yangle,x0,N)
	x0 = zrot(zangle,x0,N)
	
	# Translating initial positions
	x0[0] = x0[0] + r_transform[0]
	x0[1] = x0[1] + r_transform[1]
	x0[2] = x0[2] + r_transform[2]
	return (x0)

def velocities(body, R, N, BHM, kv, phi, theta, NP, Gpcmyr, BH, v_transform, xangle, yangle, zangle):
	# Setting velocity of initial system (only x-y plane atm)
	x0dot = np.zeros([3,N+2*BH], np.float64)
	mass_red = ReducedMass(body, R, N, BHM)
	j,k = 0,0
	while (j < NP.size):
		i = 0
		while (i < NP[j]):
			x0dot[0,k] = -kv[j] * np.sin(phi[k]) * v_orb(R[k],mass_red,Gpcmyr,BHM) #[pc/myr]
			x0dot[1,k] =  kv[j] * np.cos(phi[k]) * v_orb(R[k],mass_red,Gpcmyr,BHM) #[pc/myr]
			x0dot[2,k] =  0 #[pc/myr]
			i += 1
			k += 1
		j += 1
	x0dot = xrot(xangle, x0dot, N)
	x0dot = yrot(yangle, x0dot, N)
	x0dot = zrot(zangle, x0dot, N)
	
	#Translating initial velocities
	x0dot[0] = x0dot[0] + v_transform[0]
	x0dot[1] = x0dot[1] + v_transform[1]
	x0dot[2] = x0dot[2] + v_transform[2]
	return (x0dot)


def StartPositions(Np, BH, r, KV, body, BHM, Gpcmyr, sPERmyr, r_transform, v_transform, xangle, yangle, zangle, GalNum, GalMass, anglefactor):
	# Setting Constants
	kmPERpc = (3.085677581*10**13) #[km/pc]
	#print('G in start pos calcs:',Gpcmyr)
	#print('r_transform:    ',np.asarray(r_transform))
	#Numpifying the Memory Views from the cython file
	NP = np.asarray(Np)
	kv = np.asarray(KV)
	GalMassRatios = np.asarray(GalMass)
	r_transform = np.asarray(r_transform)
	v_transform = np.asarray(v_transform)
	xangle = np.asarray(xangle)
	yangle = np.asarray(yangle)
	zangle = np.asarray(zangle)
	#print('xangle:',xangle,'yangle:',yangle,'zangle:',zangle)

	# Finding total number of stars in rings on a single galaxy
	N = 0
	for i in range(0,NP.size):
		N += NP[i]
	#print(N)

	# Creating zero arrays for values to be stored.

	R = np.zeros(N,np.float64)
	phi = np.zeros(N,np.float64)
	theta = np.zeros(N,np.float64)

	
	# Evenly spacing the particles in each individual ring (ie. setting phi)
	j,k = 0,0
	while (j < NP.size):
		i = 0
		while (i < NP[j]):
			R[k] = NP[j]*r	# Each ring has the same density per pc of particles, since the radius is proportional to the number of particles in the ring.
			phi0 = np.linspace(0,2*np.pi,NP[j],endpoint = False, dtype = np.float64)
			phi[k] = phi0[i]+anglefactor*phi0[1]
			i += 1
			k += 1
		j += 1
	
	x0 = np.zeros([3,(N+1)*GalNum] , dtype=np.float64 , order='F')
	x0dot = np.zeros([3,(N+1)*GalNum] , dtype=np.float64 , order='F')
	# Calculating positions and velocities for multiple independant galaxies
	i,j=0,0
	while (i<GalNum):
		# RADIUS ADJUSTMENT FOR MASS RATIO PROPORTIONALITY: Changes are subjective
		MassPropRadius = R**GalMassRatios[i]	#According to https://arxiv.org/abs/1206.2532 if you set the log ratios equal one another. see notes 12/09
		# Problem: This is very small. VERY small. Makes the radius less than 3, instead of larger than 3, which was my goal.
		RadiusAdjusted0 = R
		RadiusAdjusted1 = R + (1-GalMassRatios[i])*R/2
		RadiusAdjusted2 = R + (GalMassRatios[i]-1)*R/2
		RadiusAdjusted3 = R + (GalMassRatios[i]-1)*R/3

		x0_temp	 = positions(RadiusAdjusted0,phi,N,BH,r_transform[i],xangle[i],yangle[i],zangle[i])
		vel_temp = velocities(body,RadiusAdjusted3,N,BHM,kv[i],phi,theta,NP,Gpcmyr,BH,v_transform[i],xangle[i],yangle[i],zangle[i])
		for n in range(0,(N+BH)):
			for k in range(0,3):
				x0[k,j+n]    = x0_temp[k,n]
				x0dot[k,j+n] = vel_temp[k,n]
				#print(k,j,n)
		i += 1
		j = (N+1)*i
	#print(x0)
	#x0dot = x0dot*sPERmyr/kmPERpc
	return(x0,phi,x0dot)
#1 pc = (3.085677581*10**13) km
