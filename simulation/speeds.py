import timeit


cy = timeit.timeit('example_cy.NBody(8,0.0001,0.001,100,0.0,1,1,20,20,1,10000)', setup = 'import example_cy', number = 20)
#py = timeit.timeit('v05example_cy.NBody(8,0.01,0.001,100,0.05,1,0.56,5,8000,1)', setup = 'import example_cy', number = 10)

print(cy)#,py)
