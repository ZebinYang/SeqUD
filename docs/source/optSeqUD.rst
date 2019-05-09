SeqUDOptimizer
===============
Here we give several examples to show the usage of this package.

Example 1: Cliff-Shaped function
------------------------------------------------
Optimization for Cliff-Shaped function based on Sequential Uniform Design::

		from sklearn import svm
		from SeqMM.pySeqUD import SeqUDOptimizer
		def cliff(parameters):
			x1 = parameters['x1']
			x2 = parameters['x2']
			term1 = -0.5*x1**2/100
			term2 = -0.5*(x2+0.03*x1**2-3)**2
			y = np.exp(term1 + term2)
			return  y
		ParaSpace =  {'x1': {'Type': 'continuous', 'Range': [-20,20], 'wrapper': lambda x: x}, 
			   'x2': {'Type': 'continuous', 'Range': [-10,5], 'wrapper': lambda x: x}}
		Level_Number = 20
		clf = SeqUDOptimizer(cliff, ParaSpace, Level_Number, n_jobs = 10, verbose = True)
		clf.search()

Example 2: Octopus-Shaped function
------------------------------------------------
Optimization for Octopus-Shaped function based on Sequential Uniform Design::

		from sklearn import svm
		from SeqMM.pySeqUD import SeqUDOptimizer
		def octopus(parameters):
			x1 = parameters['x1']
			x2 = parameters['x2']
			y = 2*np.cos(10*x1)*np.sin(10*x2)+np.sin(10*x1*x2)
			return  y
		ParaSpace = {'x1': {'Type': 'continuous', 'Range': [0,1], 'Wrapper': lambda x: x}, 
			   'x2': {'Type': 'continuous', 'Range': [0,1], 'Wrapper': lambda x: x}}
		Level_Number = 20
		clf = SeqUDOptimizer(octopus, ParaSpace, Level_Number, n_jobs = 10, verbose = True)
		clf.search()
