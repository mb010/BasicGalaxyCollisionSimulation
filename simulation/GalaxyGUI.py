#Version 17
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTk, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import angled2dplot as angledplt

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import ttk
import numpy as np

LARGE_FONT= ("Verdana", 12)

### Functions I have defined to use to display data
def dataselection():
        tk.Tk.filename = filedialog.askopenfilename(initialdir = "Calculated Values/",title = "Select file",filetypes = (("velocity files","*v.txt"),("txt files","*.txt"),("all files", "*.*")))        
        tk.Tk.filename = tk.Tk.filename[:-5]

def speedcheck(filelabel):
	v = np.loadtxt("{}v.txt".format(filelabel),	dtype = np.float64 , delimiter = ',' , ndmin = 2)
	v = v/1000
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
		plt.plot(v[:,i], color=(Color_array[i,0],Color_array[i,1],Color_array[i,2]))#,label='Particle {}'.format(i))
		#Invalid rgba argument. Use something to distinguish individual lines. Repeat and put into defined function for each value which you want to plot

	plt.axis([0,stepnumber,0,3.06594845])#max(v[:,1])]) #306.594845 kpc/myr is is the speed of light in kpc/myr.
	plt.title('Data analysis of simulation: Particle Speed')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Time Passed [myr]')
	ax.set_ylabel('Speed [kpc/myr]')
	plt.show()
	
def distcheck(filelabel):
	Dist 	= np.loadtxt("{}DistO.txt".format(filelabel),	dtype = np.float64 , delimiter = ',' , ndmin = 2)
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
	ECheck	= np.loadtxt("{}ECheck.txt".format(filelabel),	dtype = np.float64 , ndmin = 1)
	stepnumber = ECheck.shape[0]
	print('stepno:',stepnumber)

	#Setup plot:
	fig, ax = plt.subplots()
	#ax.set_yscale('log')	
	plt.plot(ECheck, 'k.',markersize=4)
	

	plt.axis([0,stepnumber,min(ECheck),max(ECheck)])
#	plt.title('Single Galaxy System Energy \n')
#	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number / Time [Myr]')
	ax.set_ylabel('Energy [J]')

	plt.show()

def timecheck(filelabel):
	stept	= np.loadtxt("{}sTime.txt".format(filelabel),	dtype = np.float64 , ndmin = 1)
	stepnumber = stept.shape[0]
	Color_array = np.ndarray([3],dtype=float)
	print('stepno:',stepnumber)

	#Setup plot:
	fig, ax = plt.subplots()
	plt.plot(stept, 'r.',label='Time')
	#ax.set_yscale('log')

	plt.axis([0,stepnumber,0,max(stept)])
	plt.title('Data analysis of simulation: System Time Progression \n')
	plt.legend(loc='best')
	 
	ax.set_xlabel('Step Number')
	ax.set_ylabel('Time [myr]')
	plt.show()

def pltinthreed(filename):	#N is the number of particles per galaxy
					#filename is the filename of the data group that you want to read in.
	#We load in the data from the saved files under the respective titles
	x_timed = np.loadtxt("{}xtime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	y_timed = np.loadtxt("{}yTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	z_timed = np.loadtxt("{}zTime.txt".format(filename) , dtype = np.float64 , delimiter = ',' , ndmin = 2)
	
	NumberOfGalaxies = int(np.asarray(filename[-1]))
	N = x_timed.shape[1]-1
	N = int(N/NumberOfGalaxies)
	print(N)
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
	size3galmass = size1mass*0.1

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
	#ax.scatter(0,0,0,c='r',marker='x')

	ax.set_xlabel('x [kpc]')
	ax.set_ylabel('y [kpc]')
	ax.set_zlabel('z [kpc]')
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
		
		ax.set_xlim(-110,110)
		ax.set_ylim(-110,110)
		ax.set_zlim(-110,110)
		ax.set_xlabel('x [kpc]')
		ax.set_ylabel('y [kpc]')
		ax.set_zlabel('z [kpc]')
		fig.canvas.draw_idle()
	
	sstep.on_changed(update)
	plt.show()
	
"""
	print(x_timed[nstepSlice],'\n',y_timed[nstepSlice],'\n',z_timed[nstepSlice])

	#Setting up 3D plot
	fig = plt.figure()
	#fig = plt.figure(figsize=(5,5))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim3d(-80000,80000)
	ax.set_ylim3d(-80000,80000)
	ax.set_zlim3d(-80000,80000)
	xs = x_timed[nstepSlice]
	ys = y_timed[nstepSlice]
	zs = z_timed[nstepSlice]
	ax.scatter(xs,ys,zs,c='b',marker='o')
	ax.set_xlabel('x position [pc]')
	ax.set_ylabel('y position [pc]')
	ax.set_zlabel('z position [pc]')

	plt.show()"""

def sliderpltcheck(filename):
	if (filename != "Please select a file from the \"Select Simulation\" menu." and filename != ""):
		sliderplt(filename)
	else:
		print("Please select a file from the \"Select Simulation\" menu.")

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
	data, = plt.plot(x_timed[0],y_timed[0], '.',c='#01148c',markersize=4)
	data1, = plt.plot(x_timed[0,-1],y_timed[0,-1],'k.',markersize=6)
	#data2, = plt.plot(x_timed[np.sum(N)+1,-1],y_timed[np.sum(N)+1,-1],'k.')
	plt.axis([-100, 100, -100, 100])
	#plt.title(filename[57:])
	
	axstep = plt.axes([0.25,0.1,0.65,0.03], facecolor = 'lightgoldenrodyellow')
	sstep = Slider(axstep, 'Time [myr]', valmin=1,valmax = x_timed.shape[0]-1, valinit = step_init)
	  
	def update(val):
		step = int(round(sstep.val,0))
		data.set_xdata(x_timed[step-1])
		data.set_ydata(y_timed[step-1])
		xy = x_timed[step-1,0],y_timed[step-1,0]
		data1.set_xdata(x_timed[step-1,-1])
		data1.set_ydata(y_timed[step-1,-1])

		#data2.set_xdata(x_timed[step-1,-1])
		#data2.set_ydata(y_timed[step-1,-1])
		"""if (NumberOfGalaxies>=3):
				data3.set_xdata(x_timed[step-1,-1])
				data3.set_ydata(y_timed[step-1,-1])"""
		#ax.annotate('%s' % 'o', xy=xy, textcoords='data') # Adding the Annotation to the particle, Use to track particle and remove to create images (for now)
		#ax.annotate.remove()	#Attempt to continuously remove the annotations. - didnt work.
		fig.canvas.draw_idle()
	
	sstep.on_changed(update)
	ax.set_xlabel('x [kpc]')
	ax.set_ylabel('y [kpc]')
	plt.show()

def timeForPlot(filename):
	if (filename != "Please select a file from the \"Select Simulation\" menu." and filename != " "):
		#tk.Tk.Time3dPlot = simpledialog.askinteger("Time Entry", "Enter time of desired 3d representation.",initialvalue=2,minvalue=1)
		#print (tk.Tk.Time3dPlot)
		pltinthreed(filename)
	else:
		print("Please select a file from the \"Select Simulation\" menu.")

tk.Tk.filename = "Please select a file from the \"Select Simulation\" menu."

### Initialisation of GUI
class SimulationGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, "Simulation Analysis GUI")
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(10, weight=1)
        container.grid_columnconfigure(10, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame(StartPage)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
    


class StartPage(tk.Frame):
        
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Select a simulation run you would like to inspect.\nUse the buttons below to call specific plots\nfor the chosen simulation.")
        label.pack(pady=10,padx=10)

        button = tk.Button(self, text="Select Simulation",
                            command=lambda:dataselection())
        plotbutton = tk.Button(self, text="Slider Plot", 
                            command=lambda:sliderpltcheck(tk.Tk.filename))
        button2 = ttk.Button(self, text="3D Plot ",
                            command=lambda: timeForPlot(tk.Tk.filename))
        ebutton = ttk.Button(self, text="Energy Plot",
                            command=lambda: echeck(tk.Tk.filename))
        speedbutton = ttk.Button(self, text="Speed Plot",
                            command=lambda: speedcheck(tk.Tk.filename))
        distbutton = ttk.Button(self, text="Distance from Origin Plot",
                            command=lambda: distcheck(tk.Tk.filename))
        timebutton = ttk.Button(self, text="Time Progression Plot\n   (should be linear)",
                            command=lambda: timecheck(tk.Tk.filename))

        button3 = ttk.Button(self, text="Print selected file name",
                            command=lambda: print(tk.Tk.filename))


        button.pack()
        plotbutton.pack()
        button2.pack()
        ebutton.pack()
        speedbutton.pack()
        distbutton.pack()
        timebutton.pack()
        
        button3.pack()
        
        
        #Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=W, pady=4)
        #Button(master, text='Show', command=show_entry_fields).grid(row=3, column=1, sticky=W, pady=4)

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Sliding Graph", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

        canvas = FigureCanvasTk(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)



app = SimulationGUI()
app.mainloop()
