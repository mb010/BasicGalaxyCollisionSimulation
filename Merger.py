#v19
import numpy as np
import random
from mpmath import linspace
import math
import example_cy
import matplotlib.pyplot as plt
import SingleSimulation
import time
import angled2dplot as twodview


def merger(datalabel,OutTime):
	
	starttime = time.time()
        #N = np.asarray([20])
	N = np.asarray([20,10])
	noAngleShifts = 5
	noRadialShifts = 3
	NoMergedSimulations = noAngleShifts*noRadialShifts
	anglefact_array = np.linspace(0, 1, noAngleShifts, endpoint = False, dtype = np.float64)
	radiusfactor_array = np.linspace(0.97,1.03,noRadialShifts, endpoint = True, dtype = np.float64)
	
	NumberOfGalaxies = 2
	x = np.append([],NoMergedSimulations)
	y = np.append([],NoMergedSimulations)
	z = np.append([],NoMergedSimulations)
	for j in range(0,noRadialShifts):
		for i in range(0,noAngleShifts):
			anglefactor = anglefact_array[i]
			radiusfactor = radiusfactor_array[j]
			x_timed, y_timed, z_timed = SingleSimulation.simulate(datalabel +str(i) +' '+ str(NoMergedSimulations)+' '+str(j), anglefactor, radiusfactor,OutTime)
			if (OutTime<= x_timed.shape[0]):
				x = np.append(x,x_timed[OutTime])
				y = np.append(y,y_timed[OutTime])
				z = np.append(z,z_timed[OutTime])
			else:
				NoMergedSimulations -=1

	# 2d plot of 3d data rotated to angles rotations around axis xyz for better/consistent viewing angles and measurements
	np.savetxt("Merger Values/Final010MERGERxtime.txt", x , delimiter = ',' , newline = '\n' , header = 'x [pc] 550myr: ')
	np.savetxt("Merger Values/Final010MERGERytime.txt", y , delimiter = ',' , newline = '\n' , header = 'y [pc] 550myr: ')
	np.savetxt("Merger Values/Final010MERGERztime.txt", z , delimiter = ',' , newline = '\n' , header = 'z [pc] 550myr: ')
	
	twodview.finalnonsliderplt(0,-2,0,x,y,z,OutTime)

	alpha_BR 	= 0.3	#galaxies 1&2
	alpha_g		= 0.3	#galaxy 3
	alpha_black 	= 0.3	#galaxies 1&2&3 - centre particles
	sizerings_BR	= 30	#galaxies 1&2
	sizering_G	= 3.	#galaxy 3
	sizeinner_big	= 900	#galaxies 1&2
	sizeinner_small	= 90	#galaxy 3
	c1 		= '#01148c'
	c2		= '#8c0101'
	c3		= '#038c01'

	# Setup plot:
	fig = plt.figure(figsize=(20/2.54, 20/2.54))
        #ax = fig.add_subplot()
	ax = fig.add_subplot(111, projection='3d')
	plt.grid(b=False)
	xs = x[1:]
	ys = y[1:]
	zs = z[1:]
	for i in range(0,NoMergedSimulations):
		shift = i*NumberOfGalaxies*(np.sum(N)+1)
		ax.scatter(
			xs[0+shift:np.sum(N)+shift],
			ys[0+shift:np.sum(N)+shift],
			zs[0+shift:np.sum(N)+shift],c=c1,marker='.',s=sizerings_BR,alpha=alpha_BR)	#blue#01148c
		ax.scatter(
			xs[np.sum(N)+shift],
			ys[np.sum(N)+shift],
			zs[np.sum(N)+shift], c=c1,marker='.',s=sizeinner_big,alpha=alpha_BR)
		if (NumberOfGalaxies > 1):
			ax.scatter(
				xs[np.sum(N)+1+shift:2*np.sum(N)+1+shift],
				ys[np.sum(N)+1+shift:2*np.sum(N)+1+shift],
				zs[np.sum(N)+1+shift:2*np.sum(N)+1+shift], c=c2, marker='.',s=sizerings_BR,alpha=alpha_BR)	#red#8c0101
			ax.scatter(
				xs[2*np.sum(N)+1+shift],
				ys[2*np.sum(N)+1+shift],
				zs[2*np.sum(N)+1+shift], c=c2,marker='.',s=sizeinner_big,alpha=alpha_BR)
				
		if (NumberOfGalaxies > 2):
			ax.scatter(
				xs[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],
				ys[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],
				zs[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],c=c3,marker='.',s=sizering_G,alpha=alpha_BR)	#green#038c01
			ax.scatter(
				xs[shift+3*(np.sum(N)+1)-1],
				ys[shift+3*(np.sum(N)+1)-1],
				zs[shift+3*(np.sum(N)+1)-1], c=c3,marker='.',s=sizeinner_small,alpha=alpha_BR)
	ax.scatter(0,0,0,c='r',marker='x')
	ax.set_xlabel(str(OutTime)+' x [kpc]')
	ax.set_ylabel('y [kpc]')
	ax.set_zlabel(str(OutTime)+'[myr]  z [kpc]')
	ax.set_xlim(-50,50)
	ax.set_ylim(-50,50)
	ax.set_zlim(-50,50)
	endtime = time.time()
	minute = round(endtime-starttime,3)/60
	seconds = round(endtime-starttime,3)%60
	print('Total Time Required: Runtime = ',round(minute,2),"min ")# ,round(seconds,0),"seconds
	plt.show()

###
merger('Triple Tail 2 ', 445) #Find time where the two primary arms are more angled.

