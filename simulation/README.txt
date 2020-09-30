Predictor Corrector Galaxy Interaction Simulation (v19)
---

FILE NAME			TASK

changes.txt			Lists the changes I made upto this version.
galaxy*.py			*=1-4 are seperate starting conditions. I used these to run similar but slightly varied simulations which I compared and can move on from the best conditions
	...		
setup.py			This file explains how to compile (and is used to compile) example_cy.pyx
example_cy.pyx		This is the acctual calculations of the simulation. It calls StartPositions.py but calculates the mass of the systems itself.
StartPositions.py	This file sets the starting coordinates and velocities of the particles according to the starting conditions given in galaxy*.py
GalaxyGUI.py		This file opens a GUI where you can select a file name (which is saved automatically) and then plot the respective data related to that file (without having to re-run the simulation)
Merger.py			This file gives the number of simulations you wish to run and how you split them up (angle and radial shifts) and then calls SingleSimulation.py to run the same starting conditions with the altered starting conditions.
SingleSimulation.py	This file works in conjunction with Merger.py and sets the remaining starting conditions for your merged simulation.
PlotMerger2d.py		This file takes the saved data from your merged simulation and plots it in a 2d projection (with a rotated viewing angle - in my implementation plotting the rotated y and z axis).
Noteable Starting Conditions.ods 		This are my "stable" starting conditions along with how long they lived according to my own definitions.

BT_TripleGalacticMerger.pdf				If you have questions for the initial starting conditions, they are explained individually in some depth here.


I used:
Ubuntu 18.04.1 LTS
Python 3.6.7
Cython version 0.28.2