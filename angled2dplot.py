#Version 19
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def test():
        #xrot = 90*np.pi/180
	#yrot = 90*np.pi/180
	#zrot = 90*np.pi/180
	xrot = 90*np.pi/180
	yrot = 90*np.pi/180
	zrot = 90*np.pi/180
	Rot_x = np.array([[1,0,0],[0,np.cos(xrot),-np.sin(xrot)],[0,np.sin(xrot),np.cos(xrot)]])
	Rot_y = np.array([[np.cos(yrot),0,np.sin(yrot)],[0,1,0],[-np.sin(yrot),0,np.cos(yrot)]])
	Rot_z = np.array([[np.cos(zrot),-np.sin(zrot),0],[np.sin(zrot),np.cos(zrot),0],[0,0,1]])
	x = np.array([[1],[0],[0]])
	y = np.array([[0],[1],[0]])
	z = np.array([[0],[0],[1]])
	#print(Rot_x)
	print(np.matmul(Rot_y,x)) #should be [0,1,0]
	print(np.matmul(Rot_z,y)) #should be [1,0,0]
	print(np.matmul(Rot_x,z)) #should be [0,1,0]

def RotateAroundX(theta,x,y,z):
	y0 = y*np.cos(theta)-z*np.sin(theta)
	z0 = y*np.sin(theta)+z*np.cos(theta)
	return (x,y0,z0)
def RotateAroundY(theta,x,y,z):
	x0 = x*np.cos(theta)+z*np.sin(theta)
	z0 = -x*np.sin(theta)+z*np.cos(theta)
	return (x0,y,z0)
def RotateAroundZ(theta,x,y,z):
	x0 = x*np.cos(theta)-y*np.sin(theta)
	y0 = x*np.sin(theta)+y*np.cos(theta)
	return (x0,y0,z)	


# 2d plot of data at given viewing angle xyz
def finalsliderplt(xrot,yrot,zrot,x_timed,y_timed,z_timed):
	xrot = (xrot-45)*np.pi/180
	yrot = (yrot+45)*np.pi/180
	zrot = (zrot+30)*np.pi/180
	
	x_timed,y_timed,z_timed = RotateAroundX(xrot,x_timed,y_timed,z_timed)
	x_timed,y_timed,z_timed = RotateAroundY(yrot,x_timed,y_timed,z_timed)
	x_timed,y_timed,z_timed = RotateAroundZ(zrot,x_timed,y_timed,z_timed)


#	x_timed = xs
#	y_timed = ys
#	z_timed = zs

	#Setup plot:
	fig = plt.figure(figsize=(15/2.54, 15/2.54))
	ax = fig.add_subplot(111, projection='3d')

	# Slider initialisation
	step_init = 4
	axstep = plt.axes([0.25,0.1,0.65,0.03], facecolor = 'lightgoldenrodyellow')
	sstep = Slider(axstep, 'Time [myr]', valmin=1,valmax = x_timed.shape[0]-1, valinit = step_init)	#step_init [myr]
        #N = np.asarray([20])
	N=[20,10]
	NumberOfGalaxies =2
	# 3D Slider plot
	xs = x_timed[0]
	ys = y_timed[0]
	zs = z_timed[0]

	ax.scatter(
		xs[:np.sum(N)],
		#ys[:np.sum(N)],
		zs[:np.sum(N)],
		c='#01148c',marker='.',alpha=0.5)
	ax.scatter(
		xs[np.sum(N)],
		#ys[np.sum(N)],
		zs[np.sum(N)], 
		c='k',marker='o',alpha=0.5)
	#ax.set_facecolor('k')
	if (NumberOfGalaxies > 1):
		ax.scatter(
			xs[np.sum(N)+1:2*np.sum(N)+1],
			#ys[np.sum(N)+1:2*np.sum(N)+1],
			zs[np.sum(N)+1:2*np.sum(N)+1], 
			c='#8c0101', marker='.',alpha=0.5)
		ax.scatter(
			xs[2*np.sum(N)+1],
			#ys[2*np.sum(N)+1],
			zs[2*np.sum(N)+1], 
			c='k',marker='p',alpha=0.5)
			
	if (NumberOfGalaxies > 2):
		ax.scatter(
			xs[2*np.sum(N)+2:-1],
			#ys[2*np.sum(N)+2:-1],
			zs[2*np.sum(N)+2:-1],
			c='#038c01',marker='.',alpha=0.5)
		ax.scatter(
			xs[-1],
			#ys[-1],
			zs[-1], 
			c='k',marker='.',alpha=0.5)
	#ax.scatter(0,0,0,c='r',marker='x')

	ax.set_xlabel('x [kpc]')
	#ax.set_ylabel('y [kpc]')
	ax.set_zlabel('1z [kpc]')
	ax.set_xlim(-200,200)
	#ax.set_ylim(-200,200)
        #ax.set_zlim(-200,200)
	ax.set_zlim(-110,110)


	def update(val):
		step = int(round(sstep.val,0))
		ax.clear()
		
		xs = x_timed[step]
		ys = y_timed[step]
		zs = z_timed[step]

		ax.scatter(
			xs[:np.sum(N)],
			#ys[:np.sum(N)],
			zs[:np.sum(N)],
			c='#01148c',marker='.',alpha=0.5)
		ax.scatter(
			xs[np.sum(N)],
			#ys[np.sum(N)],
			zs[np.sum(N)], 
			c='k',marker='o',alpha=0.5)
		if (NumberOfGalaxies > 1):
			ax.scatter(
				xs[np.sum(N)+1:2*np.sum(N)+1],
				#ys[np.sum(N)+1:2*np.sum(N)+1],
				zs[np.sum(N)+1:2*np.sum(N)+1], 
				c='#8c0101', marker='.',alpha=0.5)
			ax.scatter(
				xs[2*np.sum(N)+1],
				#ys[2*np.sum(N)+1],
				zs[2*np.sum(N)+1], 
				c='k',marker='p',alpha=0.5)
				
		if (NumberOfGalaxies > 2):
			ax.scatter(
				xs[2*np.sum(N)+2:-1],
				#ys[2*np.sum(N)+2:-1],
				zs[2*np.sum(N)+2:-1],
				c='#038c01',marker='.',alpha=0.5)
			ax.scatter(
				xs[-1],
				#ys[-1],
				zs[-1], 
				c='k',marker='.',alpha=0.5)
		
		#ax.scatter(0,0,0,c='k',marker='x')
		
		ax.set_xlim(-110,110)
		#ax.set_ylim(-110,110)
		ax.set_zlim(-110,110)
		ax.set_xlabel('x [kpc]')
		#ax.set_ylabel('y [kpc]')
		ax.set_zlabel('1z [kpc]')
		fig.canvas.draw_idle()

	sstep.on_changed(update)
	plt.show()

# non slider plot of given time at viewing angle xyz
def finalnonsliderplt(xrot,yrot,zrot,x_timed,y_timed,z_timed,time):
	xrot = (xrot-45)*np.pi/180
	yrot = (yrot+45)*np.pi/180
	zrot = (zrot+30)*np.pi/180
	
	xs,ys,zs = RotateAroundX(xrot,x_timed,y_timed,z_timed)
	xs,ys,zs = RotateAroundY(yrot,xs,ys,zs)
	xs,ys,zs = RotateAroundZ(zrot,xs,ys,zs)
	
	alpha_BR 	= 0.1	#galaxies 1&2
	alpha_g		= 0.1	#galaxy 3
	alpha_black 	= 0.1	#galaxies 1&2&3 - centre particles
	sizerings_BR	= 100	#galaxies 1&2
	sizering_G	= sizerings_BR*1/20	#galaxy 3
	sizeinner_big	= sizerings_BR*30
	sizeinner_small	= sizering_G*30	#galaxy 3
	c1 		= '#01148c'
	c2		= '#8c0101'
	c3		= '#038c01'

	# defining outlying values
        #N = np.array([20])
	N = np.array([20,10])
	NumberOfGalaxies = 2
	NoMergedSimulations = 15
	xs = xs[1:]
	ys = ys[1:]
	zs = zs[1:]
	
	# Setup plot:
	fig = plt.figure(figsize=(15/2.54, 15/2.54))
	ax = fig.add_subplot(111)
	#print(xs.shape,xs.shape[0]/15)
	for i in range(0,NoMergedSimulations):
		shift = i*NumberOfGalaxies*(np.sum(N)+1)
		ax.scatter(
			xs[0+shift:np.sum(N)+shift],			
			#ys[0+shift:np.sum(N)+shift],
			zs[0+shift:np.sum(N)+shift],
			c=c1,marker='.',alpha=alpha_BR,s=sizerings_BR)	#blue
		ax.scatter(
			xs[np.sum(N)+shift],
			#ys[np.sum(N)+shift],
			zs[np.sum(N)+shift],
			c=c1,marker='.',alpha=alpha_black,s=sizeinner_big)
		if (NumberOfGalaxies > 1):
			ax.scatter(
				xs[np.sum(N)+1+shift:2*np.sum(N)+1+shift],
				#ys[np.sum(N)+1+shift:2*np.sum(N)+1+shift],
				zs[np.sum(N)+1+shift:2*np.sum(N)+1+shift],
				c=c2, marker='.',alpha=alpha_BR,s=sizerings_BR)	#red
			ax.scatter(
				xs[2*np.sum(N)+1+shift],
				#ys[2*np.sum(N)+1+shift],
				zs[2*np.sum(N)+1+shift],
				c=c2,marker='.',alpha=alpha_black,s=sizeinner_big)
				
		if (NumberOfGalaxies > 2):
			ax.scatter(
				xs[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],
				#ys[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],
				zs[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],
				c=c3,marker='.',alpha=alpha_g,s=sizering_G)	#green
			ax.scatter(
				xs[shift+3*(np.sum(N)+1)-1],
				#ys[shift+3*(np.sum(N)+1)-1],
				zs[shift+3*(np.sum(N)+1)-1],
				c=c3,marker='.',alpha=alpha_black,s=sizeinner_small)
	#ax.scatter(0,0,0,c='r',marker='x')
	ax.set_xlabel("x' [kpc]")
	ax.set_ylabel("y' [kpc]")
	ax.set_xlim(-80,80)
	ax.set_ylim(-80,80)
	plt.show()

def onesystemfinalnonsliderplt(xrot,yrot,zrot,x_timed,y_timed,z_timed,time):
	xrot = (xrot-45)*np.pi/180
	yrot = (yrot+45)*np.pi/180
	zrot = (zrot+30)*np.pi/180
	Rot_x = np.array([[1,0,0],[0,np.cos(xrot),-np.sin(xrot)],[0,np.sin(xrot),np.cos(xrot)]])
	Rot_y = np.array([[np.cos(yrot),0,np.sin(yrot)],[0,1,0],[-np.sin(yrot),0,np.cos(yrot)]])
	Rot_z = np.array([[np.cos(zrot),-np.sin(zrot),0],[np.sin(zrot),np.cos(zrot),0],[0,0,1]])
	
	xs,ys,zs = RotateAroundX(xrot,x_timed[time],y_timed[time],z_timed[time])
	xs,ys,zs = RotateAroundY(yrot,xs,ys,zs)
	xs,ys,zs = RotateAroundZ(zrot,xs,ys,zs)


	# defining outlying values
        #N = np.array([20])
	N = np.array([20,10])
	NumberOfGalaxies = 2
	#xs = xs[1:]
	#ys = ys[1:]
	#zs = zs[1:]
	alpha_BR 	= 0.2	#galaxies 1&2
	alpha_g		= 0.2	#galaxy 3
	alpha_black 	= 0.2	#galaxies 1&2&3 - centre particles
	sizerings_BR	= 30	#galaxies 1&2
	sizering_G	= sizerings_BR*2/10	#galaxy 3
	sizeinner_big	= sizerings_BR*30
	sizeinner_small	= sizering_G*30	#galaxy 3
	c1 		= 'k'#01148c'
	c2		= 'k'#8c0101'
	c3		= 'k'#038c01'
	
	# Setup plot:
	fig = plt.figure(figsize=(15/2.54, 15/2.54))
	ax = fig.add_subplot(111)
	#print(xs.shape,xs.shape[0]/15)
	ax.scatter(
		xs[0+shift:np.sum(N)+shift],			
		#ys[:np.sum(N)],
		zs[:np.sum(N)],
		c=c1,marker='o',alpha=alpha_BR,s=sizerings_BR)	#blue
	ax.scatter(
		xs[np.sum(N)+shift],
		#ys[np.sum(N)],
		zs[np.sum(N)],
		c='k',marker='s',alpha=alpha_black,s=sizeinner_big)
	if (NumberOfGalaxies > 1):
		ax.scatter(
			xs[np.sum(N)+1+shift:2*np.sum(N)+1+shift],
			#ys[np.sum(N)+1:2*np.sum(N)+1],
			zs[np.sum(N)+1:2*np.sum(N)+1],
			c=c2, marker='o',alpha=alpha_BR,s=sizerings_BR)	#red
		ax.scatter(
			xs[2*np.sum(N)+1+shift],
			#ys[2*np.sum(N)+1],
			zs[2*np.sum(N)+1],
			c='k',marker='p',alpha=alpha_black,s=sizeinner_big)
			
	if (NumberOfGalaxies > 2):
		ax.scatter(
			xs[2*(np.sum(N)+1)+shift:shift+3*(np.sum(N)+1)-1],
			#ys[2*(np.sum(N)+1):3*(np.sum(N)+1)-1],
			zs[2*(np.sum(N)+1):3*(np.sum(N)+1)-1],
			c='#038c01',marker='.',alpha=alpha_g,s=sizering_G)	#green
		ax.scatter(
			xs[shift+3*(np.sum(N)+1)-1],
			#ys[3*(np.sum(N)+1)-1],
			zs[3*(np.sum(N)+1)-1],
			c=c3,marker='.',alpha=alpha_black,s=sizeinner_small)
	#ax.scatter(0,0,0,c='r',marker='x')
	ax.set_xlabel(str(time)+'[myr]  x [kpc]')
	ax.set_ylabel('y [kpc]')
	ax.set_xlim(-110,110)
	ax.set_ylim(-110,110)
	plt.show()


def sliderplt(filename):
	#This code is copied and modified from plot.py
	x_timed = np.loadtxt("{}xtime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	y_timed = np.loadtxt("{}yTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	z_timed = np.loadtxt("{}zTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	#Setup plot:
#	fig = plt.figure(figsize=(20/2.54, 20/2.54))
	fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54))
	plt.subplots_adjust(left=0.25, bottom=0.25)
	#step = np.arange(0,nsteps,1)
	step_init = 4
	data, = plt.plot(x_timed[0],z_timed[0], '.',c='#01148c',markersize=4)
	data1, = plt.plot(x_timed[0,-1],z_timed[0,-1],'k.',markersize=6)
	#data2, = plt.plot(x_timed[np.sum(N)+1,-1],y_timed[np.sum(N)+1,-1],'k.')
	plt.axis([-100, 100, -100, 100])
	#plt.title(filename[57:])
	
	axstep = plt.axes([0.25,0.1,0.65,0.03], facecolor = 'lightgoldenrodyellow')
	sstep = Slider(axstep, 'Time [myr]', valmin=1,valmax = x_timed.shape[0]-1, valinit = step_init)
	  
	def update(val):
		step = int(round(sstep.val,0))
		data.set_xdata(x_timed[step-1])
		data.set_ydata(z_timed[step-1])
		xy = x_timed[step-1,0],y_timed[step-1,0]
		data1.set_xdata(x_timed[step-1,-1])
		data1.set_ydata(z_timed[step-1,-1])

		#data2.set_xdata(x_timed[step-1,-1])
		#data2.set_ydata(y_timed[step-1,-1])
		"""if (NumberOfGalaxies>=3):
				data3.set_xdata(x_timed[step-1,-1])
				data3.set_ydata(z_timed[step-1,-1])"""
		#ax.annotate('%s' % 'o', xy=xy, textcoords='data') # Adding the Annotation to the particle, Use to track particle and remove to create images (for now)
		#ax.annotate.remove()	#Attempt to continuously remove the annotations. - didnt work.
		fig.canvas.draw_idle()
	
	sstep.on_changed(update)
	ax.set_xlabel('x [kpc]')
	ax.set_ylabel('y [kpc]')
	plt.show()


