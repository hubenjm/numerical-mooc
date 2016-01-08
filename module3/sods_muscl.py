import numpy 
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
from matplotlib import animation

from sods import compute_f

def compute_u_i(a, b, barrier, nx, gamma, params):
	"""
	Generates initial values of vector u at each point in space
	
	Parameters
	----------
	a		: float
	b		: float
	barrier : float
			we require that a < barrier < b
	nx		: int
	gamma 	: float
	params	: array of floats
			length 6 array of [
	"""
	assert a < barrier and barrier < b and len(params) == 6, "must have a < barrier < b and len(params) = 6)"
	assert nx > 0, "nx must be positive"
	
	u_initial = numpy.zeros((3,nx))
	rho_L, vel_L, p_L = params[:3]
	rho_R, vel_R, p_R = params[3:]
	
	dx = (b-a)/(nx-1)
	i_barrier = numpy.floor((barrier-a)/dx)
	
	u_initial[:, :i_barrier] = numpy.array([rho_L, rho_L*vel_L, rho_L*(p_L/((gamma-1)*rho_L) + 0.5*vel_L**2)]).reshape((3,1))
	u_initial[:, i_barrier:] = numpy.array([rho_R, rho_R*vel_R, rho_R*(p_R/((gamma-1)*rho_R) + 0.5*vel_R**2)]).reshape((3,1))

	return u_initial


def minmod(e, dx):
	"""
	Compute the minmod approximation to the slope

	Parameters
	----------
	e : 3 by n array of float 
		input data
	dx : float 
		spacestep

	Returns
	-------
	sigma : 3 by n array of float 
			minmod slope
	"""

	sigma = numpy.zeros_like(e)
	de_minus = numpy.ones_like(e)
	de_plus = numpy.ones_like(e)

	de_minus[:,1:] = (e[:,1:] - e[:,:-1])/dx
	de_plus[:,:-1] = (e[:,1:] - e[:,:-1])/dx

	# The following is inefficient but easy to read
	for i in range(3):
		for j in range(1, e.shape[1]-1):
			if (de_minus[i,j] * de_plus[i,j] < 0.0):
				sigma[i,j] = 0.0
			elif (numpy.abs(de_minus[i,j]) < numpy.abs(de_plus[i,j])):
				sigma[i,j] = de_minus[i,j]
			else:
				sigma[i,j] = de_plus[i,j]
	
	return sigma
	
def sods_muscl(u, nt, dt, dx):
	"""Computes the solution to the 1D shock equation using the MUSCL scheme

	Parameters
	---------
	u		: array of floats
			3 by nx array of initial values of [rho, u, p]'
	nt		: int
			number of timesteps to compute over
	dt		: float
			time stepsize
	nx		: int
			n
		
	Returns
	-------
	u_1, u2, u3 : arrays of floats
			nt by u_initial.shape[1] array of values of rho, u, p at each time and spatial value
	"""

	nx = u.shape[1]

	#initialize our results arrays with dimensions nt by nx
	u1 = numpy.zeros((nt,nx))
	u2 = numpy.zeros((nt,nx))
	u3 = numpy.zeros((nt,nx))

	#current array of u values
	u_n = numpy.zeros((3,nx))

	#copy the initial u array into each row of our new array
	u_n[:,:] = u.copy() #u at next time step
	u1[:,:] = u[0,:].copy()
	u2[:,:] = u[1,:].copy()
	u3[:,:] = u[2,:].copy()
		         
	#set up some temporary arrays
	u_plus = numpy.zeros_like(u)
	u_minus = numpy.zeros_like(u)
	flux = numpy.zeros_like(u)
	u_star = numpy.zeros_like(u)

	for t in range(1,nt):       
		sigma = minmod(u,dx) #calculate minmod slope

		#reconstruct values at cell boundaries
		u_left = u + sigma*dx/2.
		u_right = u - sigma*dx/2.     

		flux_left = compute_f(u_left) 
		flux_right = compute_f(u_right)

		#flux i = i + 1/2
		flux[:,:-1] = 0.5 * (flux_right[:,1:] + flux_left[:,:-1] - dx/dt *\
					      (u_right[:,1:] - u_left[:,:-1] ))

		#rk2 step 1
		u_star[:,1:-1] = u[:,1:-1] + dt/dx*(flux[:,:-2] - flux[:,1:-1])

		u_star[:,0] = u[:,0]
		u_star[:,-1] = u[:,-1]

		sigma = minmod(u_star,dx) #calculate minmod slope

		#reconstruct values at cell boundaries
		u_left = u_star + sigma*dx/2.
		u_right = u_star - sigma*dx/2.

		flux_left = compute_f(u_left) 
		flux_right = compute_f(u_right)

		flux[:,:-1] = 0.5*(flux_right[:,1:] + flux_left[:,:-1] - dx/dt*(u_right[:,1:] - u_left[:,:-1] ))

		u_n[:,1:-1] = .5 * (u[:,1:-1] + u_star[:,1:-1] + dt/dx * (flux[:,:-2] - flux[:,1:-1]))
		u_n[:,0] = u[:,0]
		u_n[:,-1] = u[:,-1]
	
		u1[t,:] = u_n[0,:]
		u2[t,:] = u_n[1,:]
		u3[t,:] = u_n[2,:]
	
		u = u_n.copy()

	return u1,u2,u3


def main():

	a = -10.0
	b = 10.0
	barrier = 0.0
	params = numpy.array([1.,0.,100000., 0.125, 0., 10000.])
	dt = 0.0002
	nt = 201
	nx = 162
	dx = (b-a)/(nx-1)
	gamma = 1.4 #parameter for computing pressure

	u_initial = compute_u_i(a, b, barrier, nx, gamma, params)
	u1,u2,u3 = sods_muscl(u_initial, nt, dt, dx)

	fig = pyplot.figure()
	ax = pyplot.axes(xlim=(a,b), ylim=(0,5),xlabel=('Position'),ylabel=('Density'))
	line, = ax.plot([],[],color='#003366', lw=2)

	def animate(data):
		x = numpy.linspace(a,b,nx)
		y = data
		line.set_data(x,y)
		return line,

	anim = animation.FuncAnimation(fig, animate, frames=u1, interval=50)
	pyplot.show()
	
if __name__ == "__main__":
	main()
