#v19 Python Adaptation (Original made use of Cython) # Tried not to change any line numbers to enable easy comparison with original example_cy.pyx file.
#N-Body Program

#cython: boundscheck=False, wraparound=False
from __future__ import division
from simulation.StartPositions import StartPositions
#from cython.parallel import prange
#from cython.view cimport array as cvarray
#from cython cimport view
import numpy as np
#cimport numpy as np
#from libc.math cimport sqrt


DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t


# Functions from allowing starting functions to be computed in python
def returnMV2d(a):	#Returns a fortran contiguous array from input np.ndarray
	narr_view = np.asfortranarray(a, dtype=np.float64)
	return narr_view
def returnMV1d(a):	#Returns a fortran contiguous array from input np.ndarray
	narr_view = np.asfortranarray(a, dtype=np.float64)
	return narr_view
	
def sqr(x):
	return np.sqrt(x)

#NBody simulation main Function
def NBody(NP, eta, deltat, tcrit, eps2, BH, kv, r, BHM, stepscrit, r_transform, v_transform, xangle, yangle, zangle, GalNum, GalMass, anglefactor):
	TESTVALUE = 0.0
	
	time = 0.0#[myr]
	tnext = 0.0
	Energy = 0.0#[J]
	nsteps = 0
	graphCounter = 0
	
	kmPERpc = 3.085677581e13#[km/pc]
	kgPERSM = 1.989e30#[kg/SolarMass]
	sPERmyr = 3.154e13
	G = 0.004301#[pc*SolMass*(km/s)**2]
	Gpc = G/(kmPERpc*kmPERpc) #[pc**3*SolMass**-1*s**-2]
	Gpcmyr = Gpc*sPERmyr**2 #[pc**3*SolMass**-1*myr**-2]
	
	#print('G main fnc:',G)
	#print('Gpc main fnc:',Gpc)
	#print('Gpcmyr main fnc:',Gpcmyr)
	#print('kmPERpc',kmPERpc)
	#print('kgPERSM',kgPERSM)
	#cdef int i, j, k
	fmag = 0.
	f1mag = 0.
	f2mag = 0.
	f3mag = 0.
	N = BH	
	n = 0
	for i in range(NP.shape[0]):
		N += NP[i]
	n = N
	N = N*GalNum
	# Data Arrays - np.ndarrays defined as memory views
	# Values for each individual particle
	x = np.ndarray([3,N], dtype=DTYPE, order='F')      	#r_t		Updated positions
	x0dot = np.ndarray([3,N], dtype=DTYPE, order='F')	#v_0		Starting velocity
	x0 = np.ndarray([3,N], dtype=DTYPE, order='F')		#r_0		Starting positions & 4th Order Predicted positions
	t0 = np.ndarray([N], dtype=DTYPE)      			#t_0		Starting time
	body = np.ndarray([N], dtype=DTYPE)			#m		Mass
	step = np.ndarray([N], dtype=DTYPE) 		    	#delt (_i)	Step for each individual particle
	f = np.ndarray([3,N], dtype=DTYPE, order='F')        	#F		Force
	fdot = np.ndarray([3,N], dtype=DTYPE, order='F')    	#F^(1)		Force derivative
	d1 = np.ndarray([3,N], dtype=DTYPE, order='F')       	#D^1	
	d2 = np.ndarray([3,N], dtype=DTYPE, order='F')      	#D^2
	d3 = np.ndarray([3,N], dtype=DTYPE, order='F')       	#D^3
	t1 = np.ndarray([N], dtype=DTYPE)       			#t_1	
	t2 = np.ndarray([N], dtype=DTYPE)       			#t_2	
	t3 = np.ndarray([N], dtype=DTYPE)       			#t_3	
	# Temporarily shared / global variables
	a = np.ndarray([17,N], dtype=DTYPE, order='F')        	#
	f1 = np.ndarray([3], dtype=DTYPE)        			#	
	f1dot = np.ndarray([3], dtype=DTYPE)     			#	
	f2dot = np.ndarray([3], dtype=DTYPE)     			#	
	f3dot = np.ndarray([3], dtype=DTYPE)    			#	
	
	# Values for Analysis
	R = np.ndarray([N], dtype=DTYPE, order='F')
	velocity= np.ndarray([N], dtype=DTYPE)
	xout = np.zeros([N*8000], dtype=DTYPE)
	yout = np.zeros([N*8000], dtype=DTYPE)
	zout = np.zeros([N*8000], dtype=DTYPE)
	Eout = np.zeros([N*8000], dtype=DTYPE)
	vout = np.zeros([N*8000], dtype=DTYPE)
	tout = np.zeros([N*8000], dtype=DTYPE)
	
	# Temporary constants
	s = np.ndarray([N], dtype=DTYPE)
	dt = 0.
	dt1 = 0.
	dt2 = 0.
	dt3 = 0.
	t1pr = 0.
	t2pr = 0.
	t3pr = 0.


	# Reading in Starting positions
	if (BH == 1):
		for i in range(0, GalNum):
			for j in range(0, n): #n is the number of particles in each galaxy (with central mass)
				body[j+i*n] = BHM/(n-BH)*GalMass[i]	#Mass of centre = mass of surrounding stars
		for i in range(0, GalNum):
			body[(i+1)*n-1] = BHM*GalMass[i] 	#Setting mass of galaxy centre
		x0_np, theta_np, x0dot_np = StartPositions(NP, BH, r, kv, body, BHM, Gpcmyr, sPERmyr, r_transform, v_transform, xangle, yangle, zangle, GalNum, GalMass, anglefactor)
		x0dot = returnMV2d(x0dot_np)
		#print('x0dot final:\n',np.asarray(x0dot))
		x0 = returnMV2d(x0_np)#[pc]
		theta = returnMV1d(theta_np)#[rad]

	
	else:
		for i in range(0,N):
			body[i] = BHM/(N-1)
		x0_np, theta_np , x0dot_np = StartPositions(NP, BH, r, kv, body, BHM, Gpcmyr, sPERmyr, r_transform, v_transform, xangle, yangle, zangle, GalNum, anglefactor)
		x0 = returnMV2d(x0_np) #[pc]
		theta = returnMV1d(theta_np) #[rad]
		x0dot = returnMV2d(x0dot_np) #[pc/myr]

	for i in range(0,N):
		velocity[i] = np.sqrt(x0dot_np[0,i]**2+x0dot_np[1,i]**2+x0dot_np[2,i]**2)
		#print('velocity of particle {}'.format(i),velocity[i])
	for i in range(0,N):
		xout[i+N*graphCounter] = x0[0,i]
		yout[i+N*graphCounter] = x0[1,i]
		zout[i+N*graphCounter] = x0[2,i]
		vout[i+N*graphCounter] = velocity[i]
   
	# Obtain total force and first derivative for each body
	for i in range (0, N):
		for k in range (0, 3):
			f[k,i] = 0
			fdot[k,i] = 0
			d2[k,i] = 0
			d3[k,i] = 0
		for j in range(0, N):
			if (j != i):
				for k in range (0, 3):
					a[k,i] = x0[k,j] - x0[k,i]	#R in eqn2.4					#delR[k] between particle i and j
					a[k+3,i] = x0dot[k,j] - x0dot[k,i]					#delv[k] between particle i and j
				a[6,i] = 1/(a[0,i]**2 + a[1,i]**2 + a[2,i]**2 + eps2)	#1/\R\**2
				a[7,i] = body[j]*a[6,i]**(3/2) #m_j/\R\**3		[solm*pc^-3]
				a[8,i] = 3*(a[0,i]*a[3,i] + a[1,i]*a[4,i] + a[2,i]*a[5,i])*a[6,i]	#3*(R*V)/\R\**2
				for k in range (0,3):
					f[k,i] += a[k,i]*a[7,i]	#eqn2.4a [pc^-2*solm]*G=[expected units]=[pc*solm*myr^-2]
					fdot[k,i] += (a[k+3,i] - a[k,i]*a[8,i])*a[7,i]	#eqn2.4b

	# Form Second and third derivative
	for i in range(0,N):	# Not paralelisable as each thread will overwrite f*dot[k] (No prange because each thread will overwrite f1dot[k],f2dot[k],f3dot[k])
		for j in range(0,N):
			if j != i:
				for k in range(0,3):
					a[k,i] = x0[k,j] - x0[k,i]
					a[k+3,i] = x0dot[k,j] - x0dot[k,i]
					a[k+6,i] = f[k,j] - f[k,i]
					a[k+9,i] = fdot[k,j] - fdot[k,i]
				a[12,i] = 1/(a[0,i]**2 + a[1,i]**2 + a[2,i]**2 + eps2)	#1/\R\**2
				a[13,i] = body[j]*a[12,i]**(3/2)	#m_j/\R\**3
				a[14,i] = (a[0,i]*a[3,i] + a[1,i]*a[4,i] + a[2,i]*a[5,i])*a[12,i]	#term in the differential 		#R*V/\R\**2
				a[15,i] = (a[3,i]**2 + a[4,i]**2 + a[5,i]**2 + a[0,i]*a[6,i] + a[1,i]*a[7,i] + a[2,i]*a[8,i])*a[12,i] + a[14,i]**2	#term in the differential		#(|V|**2 * R*delF)/\R\**2 + (R*V/\R\**2)**2
				a[16,i] = (9*(a[3,i]*a[6,i] + a[4,i]*a[7,i] + a[5,i]*a[8,i]) + 3*(a[0,i]*a[9,i] + a[1,i]*a[10,i] + a[2,i]*a[11,i]))*a[12,i] + a[14,i]*(9*a[15,i] -12*a[14,i]**2)	#term in the differential		#(9*(V*delF + 3*(R*delFdot))/\R\**2 + R*V/\R\**2 * (9*((|V|**2 * R*delF)/\R\**2 + (R*V/\R\**2)**2) - 12*(R*V/\R\**2)**2)
				for k in range(0,3):	#Every write depends on k value
					f1dot[k] = a[k+3,i] - 3*a[14,i]*a[k,i]	#V-3*(R*V/|R|**2) * |R|		#Currently stores a factor used in differentials
					f2dot[k] = (a[k+6,i] - 6*a[14,i]*f1dot[k] - 3*a[15,i]*a[k,i])*a[13,i]	#(delF - 6*(R*V/|R|**2) * V - 3*(R*V/|R|**2)*|R| - 3*(|V|**2*R*delF/R*delF/\R\**2+(R*V/\R\**2)**2) * R) * m_j/|R|^(3/2)		#~eqn2.5a: Check that this (and the following eqn for eqn2.5b) are both acctually what they are.???
					f3dot[k] = (a[k+9,i] - 9*a[15,i]+f1dot[k] - a[16,i]*a[k,i])*a[13,i]	#(delFdot - 9*((|V|**2 * R*delF)/\R\**2 + (R*V/\R\**2)**2) + f1dot - a[16]* R) * m_j/\R\**3		#~eqn2.5b?	#a[16] is the same as a[16,i], but would lead to an unreasonably long comment.
					d2[k,i] += f2dot[k]	#Currently stores "f dot dot" or F^(2)    (Calculated by differentiating F. All previous values in these loops are calculatiosns for this, and F^(3) )
					d3[k,i] += f3dot[k] - 9*a[14,i]*f2dot[k]	#Currently stores "f dot dot dot" or F^(3)


    # Initialise integration steps and convert to force differences
	for i in range(0,N):
		step[i] = sqr(eta*sqr((f[0,i]**2 + f[1,i]**2 + f[2,i]**2)/(d2[0,i]**2 + d2[1,i]**2 + d2[2,i]**2))) #[myr] #Should this be the same time step calculation as the other ones?	#Timestep for each individual particle.
		t0[i] = time #[myr] #Values of time for each individual particle: e.g. fig 2.1
		t1[i] = time - step[i] #[myr]
		t2[i] = time - 2*step[i] #[myr]
		t3[i] = time - 3*step[i] #[myr]
		for k in range(0,3):
			d1[k,i] = (d3[k,i]*step[i]/6- 0.5*d2[k,i])*step[i] + fdot[k,i]	#eqn2.7a
			d2[k,i] = 0.5*d2[k,i] - 0.5*d3[k,i]*step[i]	#~eqn2.7b QUESTION: Why is the term 0.5 and not 1/6 for d3
			d3[k,i] = d3[k,i]/6	#eqn2.7c
			f[k,i] = f[k,i]/2	#/2 so we dont have to do it in the coordinate prediction step (~eqn2.10)
			fdot[k,i] = fdot[k,i]/6	#/6 so we dont have to do it in the coordinate prediction step (~eqn2.10)
	

    # Starting the while loop to move the particles
	while (time < tcrit and nsteps < stepscrit):
		if (time >= tnext):	#time-tnext>=0 (see fortran numeric/arithmetic if) (For more details on how output works, see: 24/05 in notebook)
			# Energy Check and output
			Energy = 0.
			#print('IF: DT step time difference for particle 0:',tnext-np.asarray(t0))
			graphCounter += 1
			for i in range(0,N): # Not paralelisable (no prange: dt not i dependant)
				dt = time - t0[i]#[myr]

				for k in range(0,3):
					f2dot[k] = d3[k,i]*((t0[i] - t1[i]) + (t0[i] - t2[i])) + d2[k,i]	#eqn2.3b
					x[k,i] = (Gpcmyr*(((0.05*d3[k,i]*dt + f2dot[k]/12)*dt + fdot[k,i])*dt + f[k,i])*dt + x0dot[k,i])*dt + x0[k,i]	#eqn2.22 and 2.23
					a[k,i] = Gpcmyr*(((0.25*d3[k,i]*dt + f2dot[k]/3)*dt + 3*fdot[k,i])*dt + 2*f[k,i])*dt + x0dot[k,i]		#Velocity of the particles after they are moved
				velocity[i] = sqr(a[0,i]**2+a[1,i]**2+a[2,i]**2)
		
				Energy = Energy + 0.5*body[i]*(a[0,i]**2 + a[1,i]**2 + a[2,i]**2)
				#[SM*pc**2*myr**-2]		#Sum the kinetic energy of the particles 	#sum(0.5 * m_j*v**2)	#Could I use the viral energy? eqn2.9 (would not require a second velocity evaluation)
			for i in range(0,N):
				xout[i+N*graphCounter] = x[0,i]
				yout[i+N*graphCounter] = x[1,i]
				zout[i+N*graphCounter] = x[2,i]
				vout[i+N*graphCounter] = velocity[i]
		
			# FORTRAN WRITE FUNCTION IS HERE - Output for graphing
			tout[graphCounter] = time
			for i in range(0,N):
				for j in range(0,N):
					if j > i:	#Got 1 error on following line with eps2=0. This is a possibility IF eps2=0
						Energy = Energy - Gpcmyr*body[i]*body[j]/((x[0,i]-x[0,j])**2 + (x[1,i]-x[1,j])**2 + (x[2,i]-x[2,j])**2+eps2)**(0.5)
						#[SM*pc**2*myr**-2]	#eqn1.2 potential energy
			Eout[graphCounter-1] = Energy*10**6*kmPERpc**2*kgPERSM#[J]
			tnext = tnext + deltat#[myr]

		# Find next body to be advanced and set new time 
		time=1e10
		"""time = t0[0] + step[0]#[myr]
		#i=0"""
		for j in range(0,N):	#no prange: time must be checked against each elements possible "time"
			if (time > t0[j] + step[j]):
				i=j				#Essentially outputting the element label which is to be moved next
				time = t0[j] + step[j]#[myr]
		# Predict all coordinates to the first order in force derivative
		for j in range(0,N):	#prange since all elements are contained within the loop/dont have any changed uses with prange
			s[j] = time - t0[j]#[myr]
			x[0,j] = ((fdot[0,j]*s[j] + f[0,j])*s[j]*Gpcmyr + x0dot[0,j])*s[j] + x0[0,j]#[pc]	#eqn2.10 (factors differ due to setting of fdot = fdot/6 and f=f/2)
			x[1,j] = ((fdot[1,j]*s[j] + f[1,j])*s[j]*Gpcmyr + x0dot[1,j])*s[j] + x0[1,j]#[pc]	#eqn2.10 (factors differ due to setting of fdot = fdot/6 and f=f/2)
			x[2,j] = ((fdot[2,j]*s[j] + f[2,j])*s[j]*Gpcmyr + x0dot[2,j])*s[j] + x0[2,j]#[pc]	#eqn2.10 (factors differ due to setting of fdot = fdot/6 and f=f/2)
		# Include second and third order and obtain the velocity
		dt = time - t0[i]
		for k in range(0,3):    
			f2dot[k] = d3[k,i]*((t0[i] - t1[i]) + (t0[i] - t2[i])) + d2[k,i]	#eqn2.3
			x[k,i] = (0.05*d3[k,i]*dt + f2dot[k]/12)*Gpcmyr*dt**4 + x[k,i]#[pc/myr]		#Improving position by 2 orders for particle i only
			x0dot[k,i] = (((0.25*d3[k,i]*dt + f2dot[k]/3)*dt + 3*fdot[k,i])*dt + 2*f[k,i])*Gpcmyr*dt + x0dot[k,i]#[pc/myr] #Improving position by 2 orders for particle i only #Would this be the change in the energy factor?
			f1[k] = 0
		# Obtain the current Force on the i'th Body
		for j in range(0,N):	#prange: NOT ok. f1[0],f1[1],f1[2] are not summed correctly
			if j != i:
				a[0,j] = x[0,j] - x[0,i]
				a[1,j] = x[1,j] - x[1,i]
				a[2,j] = x[2,j] - x[2,i]
				a[3,j] = 1./(a[0,j]**2 + a[1,j]**2 +a[2,j]**2 + eps2)
				a[4,j] = body[j]*a[3,j]*sqr(a[3,j])
				f1[0] += a[0,j]*a[4,j]	#eqn2.4a
				f1[1] += a[1,j]*a[4,j]	#eqn2.4a
				f1[2] += a[2,j]*a[4,j]	#eqn2.4a
		# Set time intervals for new differences and update the times 
		dt1 = time - t1[i]#[myr]
		dt2 = time - t2[i]#[myr]
		dt3 = time - t3[i]#[myr]
		t1pr = t0[i] - t1[i]#[myr]	#t1 prediction? previous?
		t2pr = t0[i] - t2[i]#[myr]
		t3pr = t0[i] - t3[i]#[myr]
		t3[i] = t2[i]#[myr]
		t2[i] = t1[i]#[myr]
		t1[i] = t0[i]#[myr]
		t0[i] = time#[myr]
		# Form new differences and include fourth-order semi iteration
		for k in range(0,3):
			a[k,i] = (f1[k] - 2*f[k,i])/dt	#eqn2.2 for k=1 (D**(1)[t_0,t_1])
			a[k+3,i] = (a[k,i] - d1[k,i])/dt1	#eqn2.2 for k=2 (D**(2)[t_0,t_2])
			a[k+6,i] = (a[k+3,i] - d2[k,i])/dt2	#eqn2.2 for k=3 (D**(3)[t_0,t_3])
			a[k+9,i] = (a[k+6,i] - d3[k,i])/dt3	#eqn2.2 for k=4	(a[k+9] := D**(4)[t_0,t_4]) (Or d4[k,i] in our notation)
			d1[k,i] = a[k,i]	#Saving to use in next iteration as "aged" values of the force differences
			d2[k,i] = a[k+3,i]	#Saving to use in next iteration as "aged" values of the force differences
			d3[k,i] = a[k+6,i]	#Saving to use in next iteration as "aged" values of the force differences
			f1dot[k] = t1pr*t2pr*t3pr*a[k+9,i]	#D4*t1*t2*t3	#D4 term of eqn2.3a	#D4*a from eqn2.11
			f2dot[k] = (t1pr*t2pr + t3pr*(t1pr + t2pr))*a[k+9,i]	#(t1*t2 + t3*t1 + t3*t2)*D4	#D4 term of eqn2.3b	#D4*b from eqn2.11
			f3dot[k] = (t1pr + t2pr + t3pr)*a[k+9,i]	#(t1 + t2 +t3)*D4	#D4 term of eqn2.3c	#D4*c from eqn2.11
			#x0[k,i] = Gpcmyr*(((a[k+9,i]*dt/30 + 0.05*f3dot[k])*dt + f2dot[k]/12)*dt + f1dot[k]/6)*dt**3 + x[k,i]	#eqn2.11a/24
			x0[k,i] = 24*Gpcmyr*(((a[k+9,i]*dt/30 + 0.05*f3dot[k])*dt + f2dot[k]/12)*dt + f1dot[k]/6)*dt**3 + x[k,i]#[pc]	#eqn2.11a
			x0dot[k,i] = Gpcmyr*(((0.2*a[k+9,i]*dt + 0.25*f3dot[k])*dt + f2dot[k]/3)*dt + 0.5*f1dot[k])*dt**2 + x0dot[k,i]#[pc/myr]	#eqn2.11b
		# Scale f and fdot by factorials and set new integration step 
		for k in range(0,3):
			f[k,i] = 0.5*f1[k]	#/2 so we dont have to do it in the coordinate prediction step (~eqn2.10)
			fdot[k,i] = ((d3[k,i]*dt1 + d2[k,i])*dt + d1[k,i])/6	#eqn2.3a	#/6 so we dont have to do it in the coordinate prediction step
			f2dot[k] = 2*(d3[k,i]*(dt + dt1) + d2[k,i])	#eqn2.3b
		fmag = sqr(f1[0]**2 + f1[1]**2 + f1[2]**2) 
		f1mag = 6*sqr(fdot[0,i]**2+fdot[1,i]**2+fdot[2,i]**2)	#*6 since it was preemptively /6 for factor reasons in predictions.
		f2mag = sqr(f2dot[0]**2 + f2dot[1]**2 + f2dot[2]**2)
		f3mag = 6*sqr(d3[0,i]**2+d3[1,i]**2+d3[2,i]**2)#*6 since d3 is the difference. eqn2.7c
		#print('i',i,'fmag:',fmag,'f1mag',f1mag,'f2mag',f2mag,'f3mag',f3mag)
		#step[i] = sqr(eta*sqr((f1[0]**2 + f1[1]**2 + f1[2]**2)/(f2dot[0]**2 + f2dot[1]**2 + f2dot[2]**2)))	#eqn2.12 (Improvable to 2.13)
		step[i] = np.divide(sqr(eta*(fmag*f2mag+f1mag**2)),(sqr((f1mag*f3mag+f2mag**2))))#[myr]
		if (step[i]*1.2>dt):
			step[i] = 1.2*dt
		#step[i] = sqr(eta*(sqr((f1[0]**2 + f1[1]**2 + f1[2]**2)*(f2dot[0]**2 + f2dot[1]**2 + f2dot[2]**2))+(36*fdot[0,i]**2+36*fdot[1,i]**2+36*fdot[2,i]**2))/(sqr((36*fdot[0,i]**2+36*fdot[1,i]**2+36*fdot[2,i]**2)*(36*d3[0,i]**2+36*d3[1,i]**2+36*d3[2,i]**2))+(f2dot[0]**2 + f2dot[1]**2 + f2dot[2]**2)))	#eqn2.13: F**(3) calculated using eqn2.13c-d4 term, fdot**2*(6)**2 is where the 36 term comes from, since fdot was already /6 in preperation for the coordinate calculation.

		#reseting for while loop
		nsteps += 1
		#print('x_timed',x_timed)
	Energy = 0.0
	#print('FinalOut: DT step time difference for particle 0:',time-t0[0])
	graphCounter += 1
	for i in range(0,N): #no prange: dt not i dependant
		dt = time - t0[i]#[myr]

		for k in range(0,3):
			f2dot[k] = d3[k,i]*((t0[i] - t1[i]) + (t0[i] - t2[i])) + d2[k,i]	#eqn2.3b
			x[k,i] = Gpcmyr*((((f2dot[k]/12)*dt + fdot[k,i])*dt + f[k,i])*dt + x0dot[k,i])*dt + x0[k,i]	#~eqn2.10: But up to the third order force derivative
			a[k,i] = Gpcmyr*(((0.25*d3[k,i]*dt + f2dot[k]/3)*dt + 3*fdot[k,i])*dt + 2*f[k,i])*dt + x0dot[k,i]		#Velocity of the particles after they are moved
		velocity[i] = sqr(a[0,i]**2+a[1,i]**2+a[2,i]**2)
		
		Energy = Energy + 0.5*body[i]*(a[1,i]**2 + a[2,i]**2 + a[3,i]**2)#[SM*pc**2*myr**-2]		#Sum the kinetic energy of the particles 	#sum(0.5 * m_j*v**2)	#Could I use the viral energy? eqn2.9 (would not require a second velocity evaluation)
		xout[i+N*graphCounter] = x[0,i]
		yout[i+N*graphCounter] = x[1,i]
		zout[i+N*graphCounter] = x[2,i]
		vout[i+N*graphCounter] = velocity[i]
		
	# FORTRAN WRITE FUNCTION IS HERE - Output for graphing
	tout[graphCounter] = time
	for i in range(0,N):
		for j in range(0,N):
			if j != i:	#Got 1 error on following line with eps2=0. This is a possibility IF eps2=0
				Energy = Energy - 0.5*Gpcmyr*body[i]*body[j]/sqr((x[0,i] - x[0,j])**2 + (x[1,i] - x[1,j])**2 + (x[2,i] - x[2,j])**2 + eps2)#[SM*pc**2*myr**-2]	#eqn1.2 potential energy #WHY THE HALF!?
	Eout[graphCounter-1] = Energy*10**6*kmPERpc**2*kgPERSM*sPERmyr**2#[J]
	tnext = tnext + deltat#[myr]
	# Define arrays for final output
	x_timed = np.ndarray([N,graphCounter], dtype=DTYPE,order='F')
	y_timed = np.ndarray([N,graphCounter], dtype=DTYPE,order='F')
	z_timed = np.ndarray([N,graphCounter], dtype=DTYPE,order='F')
	v_timed = np.ndarray([N,graphCounter], dtype=DTYPE,order='F')
	E_arr = np.ndarray([graphCounter-1], dtype=DTYPE)
	step_time = np.ndarray([graphCounter], dtype=DTYPE)
	
	x_timed = np.resize(xout,(graphCounter,N))
	y_timed = np.resize(yout,(graphCounter,N))
	z_timed = np.resize(zout,(graphCounter,N))
	v_timed = np.resize(vout,(graphCounter,N))
	E_arr 	= np.resize(Eout,(graphCounter-1))
	step_time=np.resize(tout,(graphCounter))
	print("Steps: ",nsteps, "\n Time: ",time)
	return (x_timed, y_timed, z_timed, v_timed, nsteps, step_time, E_arr)


#To convert make sure all titles match AND that loops start at 0 and go to the full value (which isnt included), AND that every reference that ISNT using a running number (i,k,j) has one subtracted off of it, since it also runs from 0,N-1
#use a variable to store type-set numbers for implimentation (0 specifically)
