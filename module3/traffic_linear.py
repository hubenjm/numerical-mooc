import numpy
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

from matplotlib import animation
from JSAnimation.IPython_display import display_animation

def rho_green_light(nx, rho_light):
    """Computes "green light" initial condition with shock, and linear distribution behind

    Parameters
    ----------
    nx        : int
        Number of grid points in x
    rho_light : float
        Density of cars at stoplight

    Returns
    -------
    rho_initial: array of floats
        Array with initial values of density
    """  
    rho_initial = numpy.arange(nx)*2./nx*rho_light  # Before stoplight
    rho_initial[(nx-1)/2:] = 0
    
    return rho_initial
    
    
def rho_red_light(nx, rho_max, rho_in):
	"""Computes "red light" initial condition with shock

    Parameters
    ----------
    nx        : int
        Number of grid points in x
    rho_max   : float
        Maximum allowed car density
    rho_in    : float
        Density of incoming cars 

    Returns
    -------
    rho: array of floats
        Array with initial values of density
	"""
	rho = rho_max*numpy.ones(nx)
	rho[:(nx-1)*3./4.] = rho_in
	return rho
    
def computeF(u_max, rho_max, rho):
    """Computes flux F=V*rho

    Parameters
    ----------
    u_max  : float
        Maximum allowed velocity
    rho    : array of floats
        Array with density of cars at every point x
    rho_max: float
        Maximum allowed car density
        
    Returns
    -------
    F : array
        Array with flux at every point x
    """
    return u_max*rho*(1-rho/rho_max)
    
def ftbs(rho, nt, dt, dx, rho_max, u_max):
    """ Computes the solution with forward in time, backward in space
    
    Parameters
    ----------
    rho    : array of floats
            Density at current time-step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing
    rho_max: float
            Maximum allowed car density
    u_max  : float
            Speed limit
    
    Returns
    -------
    rho_n : array of floats
            Density after nt time steps at every point x
    """
    
    #initialize our results array with dimensions nt by nx
    rho_n = numpy.zeros((nt,len(rho)))      
    #copy the initial u array into first row of our new array
    rho_n[0,:] = rho.copy()              
    
    for t in range(1,nt):
        F = computeF(u_max, rho_max, rho)
        rho_n[t,1:] = rho[1:] - dt/dx*(F[1:]-F[:-1])
        rho_n[t,0] = rho[0]
        rho_n[t,-1] = rho[-1]
        rho = rho_n[t].copy()

    return rho_n
    
    
def laxfriedrichs(rho, nt, dt, dx, rho_max, u_max):
    """ Computes the solution with Lax-Friedrichs scheme
    
    Parameters
    ----------
    rho    : array of floats
            Density at current time-step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing
    rho_max: float
            Maximum allowed car density
    u_max  : float
            Speed limit
    
    Returns
    -------
    rho_n : array of floats
            Density after nt time steps at every point x
    """
    
    #initialize our results array with dimensions nt by nx
    rho_n = numpy.zeros((nt,len(rho)))      
    #copy the initial u array into each row of our new array
    rho_n[:,:] = rho.copy()              
    
    '''
    Now, for each timestep, we're going to calculate rho^n+1, 
    then set the value of rho equal to rho^n+1 so we can calculate 
    the next iteration.  For every timestep, the entire vector
    rho^n is saved in a single row of our results array rho_n.
    '''
    for t in range(1,nt):
        F = computeF(u_max, rho_max, rho)
        rho_n[t,1:-1] = .5*(rho[2:]+rho[:-2]) - dt/(2*dx)*(F[2:]-F[:-2])
        rho_n[t,0] = rho[0]
        rho_n[t,-1] = rho[-1]
        rho = rho_n[t].copy()
        
    return rho_n

def Jacobian(u_max, rho_max, rho):
	return u_max*(1-2*rho/rho_max)

def laxwendroff(rho, nt, dt, dx, rho_max, u_max):
	""" Computes the solution with Lax-Wendroff scheme

	Parameters
	----------
	rho    : array of floats
		    Density at current time-step
	nt     : int
		    Number of time steps
	dt     : float
		    Time-step size
	dx     : float
		    Mesh spacing
	rho_max: float
		    Maximum allowed car density
	u_max  : float
		    Speed limit

	Returns
	-------
	rho_n : array of floats
		    Density after nt time steps at every point x
	"""

	#initialize our results array with dimensions nt by nx
	rho_n = numpy.zeros((nt,len(rho)))      
	#copy the initial u array into each row of our new array
	rho_n[:,:] = rho.copy()              
    
	for t in range(1,nt):
		F = computeF(u_max, rho_max, rho)
		J = Jacobian(u_max, rho_max, rho)

		rho_n[t,1:-1] = rho[1:-1] - dt/(2*dx)*(F[2:]-F[:-2]) \
				           + dt**2/(4*dx**2) * ( (J[2:]+J[1:-1])*(F[2:]-F[1:-1]) \
				           - (J[1:-1]+J[:-2])*(F[1:-1]-F[:-2]) )

		rho_n[t,0] = rho[0]
		rho_n[t,-1] = rho[-1]
		rho = rho_n[t].copy()
		
	return rho_n

def maccormack(rho, nt, dt, dx, rho_max, u_max):
    """ Computes the solution with MacCormack scheme
    
    Parameters
    ----------
    rho    : array of floats
            Density at current time-step
    nt     : int
            Number of time steps
    dt     : float
            Time-step size
    dx     : float
            Mesh spacing
    rho_max: float
            Maximum allowed car density
    u_max  : float
            Speed limit
    
    Returns
    -------
    rho_n : array of floats
            Density after nt time steps at every point x
    """
    
    rho_n = numpy.zeros((nt,len(rho)))
    rho_star = numpy.empty_like(rho)
    rho_n[:,:] = rho.copy()
    rho_star = rho.copy()
    
    for t in range(1,nt):
        F = computeF(u_max, rho_max, rho)
        rho_star[:-1] = rho[:-1] - dt/dx * (F[1:]-F[:-1])
        Fstar = computeF(u_max, rho_max, rho_star)
        rho_n[t,1:] = .5 * (rho[1:]+rho_star[1:] - dt/dx * (Fstar[1:] - Fstar[:-1]))
        rho = rho_n[t].copy()
        
    return rho_n

def test_red_light():
	"""
	tests out the 2nd order in space/time MacCormack method for the red light problem.
	
	"""
	L = 4.0
	nx = 81
	nt = 300
	dx = L/(nx-1)
	
	rho_max = 10.
	u_max = 1.
	rho_in = 7.
	
	x = numpy.linspace(0,L,nx)
	rho = rho_red_light(nx, rho_max, rho_in)
	
	sigma = 0.5
	dt = sigma*dx/u_max
	#rho_n = maccormack(rho,nt,dt,dx, rho_max, u_max)
	rho_n = laxfriedrichs(rho, nt, dt, dx, rho_max, u_max)
	#rho_n = laxwendroff(rho, nt, dt, dx, rho_max, u_max)
	
	
	fig = pyplot.figure();
	ax = pyplot.axes(xlim=(0,L),ylim=(0,11),xlabel=('Distance'),ylabel=('Traffic density'));
	line, = ax.plot([],[],color='#003366', lw=2);
	
	def animate(data):
		x = numpy.linspace(0,L,nx)
		y = data
		line.set_data(x,y)
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=rho_n, interval=50)
	pyplot.show()
	
def test_green_light():
	
	L = 4.0
	nx = 81
	nt = 120
	dx = L/(nx-1)
	
	rho_max = 10.
	u_max = 1.
	rho_light = 3.5
	
	x = numpy.linspace(0,L,nx)
	rho_initial = rho_green_light(nx, rho_light)
	
	sigma = 0.5
	dt = sigma*dx/u_max
	#rho_n = maccormack(rho_initial, nt, dt, dx, rho_max, u_max)
	#rho_n = laxfriedrichs(rho_initial, nt, dt, dx, rho_max, u_max)
	rho_n = ftbs(rho_initial, nt, dt, dx, rho_max, u_max)
	
	fig = pyplot.figure();
	ax = pyplot.axes(xlim=(0,L),ylim=(-1,8),xlabel=('Distance'),ylabel=('Traffic density'));
	line, = ax.plot([],[],color='#003366', lw=2);
	
	def animate(data):
		x = numpy.linspace(0,L,nx)
		y = data
		line.set_data(x,y)
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=rho_n, interval=50)
	pyplot.show()

	
if __name__ == '__main__':
	test_green_light()
	#test_red_light()
