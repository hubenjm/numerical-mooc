import numpy
from matplotlib import pyplot
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

from matplotlib import animation
from JSAnimation.IPython_display import display_animation

gamma = 1.4 #parameter for computing pressure

def compute_u_i(a, b, barrier, nx, params):
	"""
	Generates initial values of density, velocity, pressure at each point in space
	
	Parameters
	----------
	a		: float
	b		: float
	barrier : float
			we require that a < barrier < b
	nx		: int
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

def compute_f(u):
	"""
	Compute the vector function f(u)
	
	Parameters
	----------
	u	: array of floats
			3 by nx array of values of rho, rho*vel, rho*e_T
			
	Returns
	-------
	f		: array of floats
			3 by nx array with columns [rho*vel, rho*vel**2 + p, (rho*e_T + p)*vel]
			where e_T = p/((gamma-1)*rho) + 0.5*vel**2	
	"""
	
	f = numpy.zeros_like(u)
	f[0,:] = u[1,:]
	f[1,:] = (u[1,:]**2)/u[0,:] + (gamma - 1)*(u[2,:] - 0.5*(u[1,:]**2)/u[0,:])
	f[2,:] = (u[1,:]/u[0,:])*(u[2,:] + (gamma - 1)*(u[2,:] - 0.5*(u[1,:]**2)/u[0,:]))
	
	return f
	
def richtmyer(u, nt, dt, dx):
	"""
	Computes the solution to the 1D shock equation using the Richtmyer method
	
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
	
	u1 = numpy.zeros((nt,nx))
	u2 = numpy.zeros((nt,nx))
	u3 = numpy.zeros((nt,nx))
	u1[:,:] = u[0,:].copy()
	u2[:,:] = u[1,:].copy()
	u3[:,:] = u[2,:].copy()
	
	f_n = compute_f(u)
	u_half = numpy.zeros_like(u)
	f_half = numpy.zeros_like(u)
	u_n = numpy.zeros_like(u)
	u_n[:,:] = u.copy()
	
	for t in range(1,nt):
		u_half[:,:-1] = 0.5*(u[:,1:] - u[:,:-1]) - dt/(2*dx)*(f_n[:,1:] - f_n[:,:-1])
		u_half[:,-1] = u[:,-1]
		f_half = compute_f(u_half)
		u_n[:,:-1] = u[:,:-1] - (dt/dx)*(f_half[:,1:] - f_half[:,:-1])
		u[:,:] = u_n.copy()
		f_n = compute_f(u_n)
		u1[t,:] = u_n[0,:]
		u2[t,:] = u_n[1,:]
		u3[t,:] = u_n[2,:]
		
	return u1,u2,u3	
		 
		
def main():
	a = -10.0
	b = 10.0
	barrier = 0.0
	params = numpy.array([1.,0.,100., 0.125, 0., 10.])
	dt = 0.01
	nt = 10
	nx = 200
	dx = (b-a)/(nx-1)
	
	u_initial = compute_u_i(a, b, barrier, nx, params)
	u1,u2,u3 = richtmyer(u_initial, nt, dt, dx)
	
	#compute density, velocity, pressure from u
	
	print(u1)
	print(u2)
	print(u3)

if __name__ == "__main__":
	main()
