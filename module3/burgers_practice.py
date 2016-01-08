import numpy
import matplotlib.pyplot as plt
from matplotlib import animation

def u_initial(nx):
	#fill in code here
	u = numpy.zeros(nx)
	u[:nx/2]=1

	return u
	
def computeF(u):
	return 0.5*u**2
	

def maccormack(u, nt, dt, dx):
	epsilon = 0.01
	un = numpy.zeros((nt,len(u)))
	un[:] = u.copy()
	ustar = u.copy()    

	for n in range(1,nt):
		F = computeF(u)
		ustar[1:-1] = un[n-1,1:-1] - (dt/dx)*(F[1:-1] - F[:-2])\
		 + epsilon*(un[n-1,2:] - 2*un[n-1,1:-1] + un[n-1,:-2])
		Fstar = computeF(ustar)
		un[n,1:] = 0.5*(un[n-1,1:] + ustar[1:] - (dt/dx)*(Fstar[1:] - Fstar[:-1]))
		u = un[n].copy()

	return un
	
def main():

	nx = 81
	nt = 70
	dx = 4.0/(nx-1)

	def animate(data):
		x = numpy.linspace(0,4,nx)
		y = data
		line.set_data(x,y)
		return line,

	u = u_initial(nx)
	sigma = .5
	dt = sigma*dx

	un = maccormack(u,nt,dt,dx)

	fig = plt.figure();
	ax = plt.axes(xlim=(0,4),ylim=(-.5,2));
	line, = ax.plot([],[],lw=2);

	anim = animation.FuncAnimation(fig, animate, frames=un, interval=50)
	#display_animation(anim, default_mode='once')
	plt.show()
	
if __name__ == "__main__":
	main()
	
	
