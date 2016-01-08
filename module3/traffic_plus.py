import numpy
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

from matplotlib import animation
from JSAnimation.IPython_display import display_animation

def compute_A_B(rho_max, u_max, u_star):
	T1 = rho_max**2*u_max*(u_max - 2*u_star)
	T2 = rho_max*(2*u_max - 3*u_star)**2
	
	A1 = 1./(2*T1)*(T2 - numpy.sqrt(-rho_max*u_star*T2*(4*u_max - 9*u_star)))
	A2 = 1./(2*T1)*(T2 + numpy.sqrt(-rho_max*u_star*T2*(4*u_max - 9*u_star)))
	
	A = numpy.max(A1,A2)
	assert A > 0, "infeasible choice of parameters"
	
	B = 1./(rho_max**2)*(-A*rho_max + 1)
	return numpy.array([A,B])


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
    
def computeF(u_max, rho, aval, bval):
	"""
	Assuming the flux F is given by the model
	F = u_max*rho*(1 - A*rho - B*rho**2)

	Parameters
	----------
	u_max	: float
	rho	: array of floats
	aval
	bval
	
	Returns
	-------
	float
	"""
	return u_max*rho*(1 - aval*rho - bval*rho**2)
    
def ftbs(rho, nt, dt, dx, rho_max, u_max, aval, bval):
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
        F = computeF(u_max, rho, aval, bval)
        rho_n[t,1:] = rho[1:] - dt/dx*(F[1:]-F[:-1])
        rho_n[t,0] = rho[0]
        rho_n[t,-1] = rho[-1]
        rho = rho_n[t].copy()

    return rho_n
    
    
def laxfriedrichs(rho, nt, dt, dx, rho_max, u_max, aval, bval):
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
        F = computeF(u_max, rho, aval, bval)
        rho_n[t,1:-1] = .5*(rho[2:]+rho[:-2]) - dt/(2*dx)*(F[2:]-F[:-2])
        rho_n[t,0] = rho[0]
        rho_n[t,-1] = rho[-1]
        rho = rho_n[t].copy()
        
    return rho_n

def maccormack(rho, nt, dt, dx, rho_max, u_max, aval, bval):
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
        F = computeF(u_max, rho, aval, bval)
        rho_star[:-1] = rho[:-1] - dt/dx * (F[1:]-F[:-1])
        Fstar = computeF(u_max, rho_star, aval, bval)
        rho_n[t,1:] = .5 * (rho[1:]+rho_star[1:] - dt/dx * (Fstar[1:] - Fstar[:-1]))
        rho = rho_n[t].copy()
        
    return rho_n

def test_red_light():
	"""
	tests out the 2nd order in space/time MacCormack method for the red light problem.
	
	"""
	L = 4.0
	nx = 81
	nt = 30
	dx = L/(nx-1)
	
	rho_max = 10.
	u_max = 1.
	u_star = 0.7
	rho_in = 3.
	
	aval, bval = compute_A_B(rho_max, u_max, u_star)
	
	x = numpy.linspace(0,L,nx)
	rho = rho_red_light(nx, rho_max, rho_in)
	
	pyplot.plot(x, rho, color='#003366', ls='-', lw=3)
	pyplot.ylabel('Traffic density')
	pyplot.xlabel('Distance')
	pyplot.ylim(-0.5,11.);
	pyplot.show()
	
	sigma = 0.5
	dt = sigma*dx/u_max
	#rho_n = maccormack(rho,nt,dt,dx, rho_max, u_max, aval, bval)
	rho_n = laxfriedrichs(rho, nt, dt, dx, rho_max, u_max, aval, bval)
	
	fig = pyplot.figure();
	ax = pyplot.axes(xlim=(0,L),ylim=(0,11),xlabel=('Distance'),ylabel=('Traffic density'));
	line, = ax.plot([],[],color='#003366', lw=2);
	
	def animate(data):
		x = numpy.linspace(0,L,nx)
		y = data
		line.set_data(x,y)
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=rho_n, interval=50)
	#display_animation(anim, default_mode='once')
	pyplot.show()
	
def test_green_light():
	
	L = 4.0
	nx = 81
	nt = 30
	dx = L/(nx-1)
	
	rho_max = 10.
	u_max = 1.
	u_star = 0.7
	rho_light = 3.5
	aval, bval = compute_A_B(rho_max, u_max, u_star)
	
	x = numpy.linspace(0,L,nx)
	rho_initial = rho_green_light(nx, rho_light)
	
	sigma = 0.5
	dt = sigma*dx/u_max
	rho_n = maccormack(rho_initial, nt, dt, dx, rho_max, u_max, aval, bval)
	#rho_n = laxfriedrichs(rho_initial, nt, dt, dx, rho_max, u_max, aval, bval)
	#rho_n = ftbs(rho_initial, nt, dt, dx, rho_max, u_max, aval, bval)
	
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
	
def main():
	#Basic initial condition parameters
	#defining grid size, time steps
	nx = 81
	nt = 30
	dx = 4.0/(nx-1)
	
	rho_max = 10.
	u_max = 1.
	u_star = 0.7
	aval, bval = compute_A_B(rho_max, u_max, u_star)

	x = numpy.linspace(0,4,nx)
	rho_light = 5.5
	rho_initial = rho_green_light(nx, rho_light)
	
#	pyplot.plot(x, rho, color='#003366', ls='-', lw=3)
#	pyplot.ylabel('Traffic density')
#	pyplot.xlabel('Distance')
#	pyplot.ylim(-0.5,11.);
	
	sigma = 1.
	dt = sigma*dx/u_max
	rho_n = ftbs(rho_initial, nt, dt, dx, rho_max, u_max, aval, bval)
	
	fig = pyplot.figure();
	ax = pyplot.axes(xlim=(0,4.0),ylim=(-1,8),xlabel=('Distance'),ylabel=('Traffic density'));
	line, = ax.plot([],[],color='#003366', lw=2);
	
	def animate(data):
		x = numpy.linspace(0,4.0,nx)
		y = data
		line.set_data(x,y)
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=rho_n, interval=50)
	#display_animation(anim, default_mode='once')
	pyplot.show()
	
def test_a_b():
	rho_max = 10.
	u_max = 1.
	u_star = 0.7
	aval, bval = compute_A_B(rho_max, u_max, u_star)
	
	print(aval)
	print(bval)
	print(type(aval))

def exercise_A_B():
	rho_max = 15.
	u_max = 2.
	u_star = 1.5
	A,B = compute_A_B(rho_max, u_max, u_star)
	print(A,B)
	
	vertex = -A/(3*B)
	S1 = vertex + numpy.sqrt(A**2 + 3*B)/(-3*B)
	S2 = vertex - numpy.sqrt(A**2 + 3*B)/(-3*B)
	
	print(S1,S2)
	
if __name__ == '__main__':
	#test_a_b()
	#main()
	#test_green_light()
	#test_red_light()
	exercise_A_B()

