import numpy as np
import scipy
import sys
sys.path.append("/home/mark/Math/JSAnimation/")
import sympy
from sympy.utilities.lambdify import lambdify
from sympy import init_printing

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


L = 11.0
nx = 51.0
dt = 0.001
dx = L/(nx-1)
rho_max = 250.0

x = np.linspace(0,L,nx)
rho0 = np.ones(nx)*10
rho0[10:20] = 50

def velocity(rho0, V_max, t_f):
	"""
	computes the average traffic velocity at time t_f
	the velocity function is given by V(rho) = V_max * (1 - rho/rho_max)
	rho = rho(x,t)
	t_f is given in minutes
	return in meters/s
	"""
	rho = rho0.copy()
	nt = int(t_f/(60*dt))
	rho_n = np.empty(nx)
	
	for j in range(nt):
		
		rho_n = rho.copy()
		rho[1:] = V_max*dt/dx*(2*rho_n[1:]/rho_max - 1)*(rho_n[1:] - rho_n[:-1]) + rho_n[1:]
	
	return V_max*(1 - rho/rho_max)*1000.0/3600.0

if __name__ == "__main__":
	#part A
	rho0 = rho0 = np.ones(nx)*10
	rho0[10:20] = 50
	V_max = 80.0
	print(np.min(velocity(rho0, V_max, 0)))
	print(np.mean(velocity(rho0, V_max, 3)))
	print(np.min(velocity(rho0, V_max, 6)))
	
	#part B
	rho0 = np.ones(nx)*20 ##note this change
	rho0[10:20] = 50
	V_max = 136.0
	print(np.min(velocity(rho0, V_max, 0)))
	print(np.mean(velocity(rho0, V_max, 3)))
	print(np.min(velocity(rho0, V_max, 3)))
	
	
