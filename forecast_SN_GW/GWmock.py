from matplotlib.pyplot import *
import numpy as np
from numpy import *
from scipy.integrate import quad
from scipy.interpolate import interp1d
ion()
show()

H0 = 67.74
Om0 = 0.315
Or0 = 8.47 * 10**-5
OK0 = 0.0
OL0 = 0.685


def Hz(z):
	return H0 * np.sqrt(Om0*(1.+z)**3 + Or0*(1.+z)**4 + OK0*(1.+z)**2 + OL0)

def invHz(z):
	return 1.0/Hz(z)

def comovdist(z):
	return quad(invHz, 0, z)[0]

def Rz(z):
	if z<=1:
		return 1.+2.*z 
	elif z <5:
		return 3./4*(5 - z)
	else:
		return 0.0

def Pz(zs):
	listPz = []
	for z in zs:
		listPz.append(4*np.pi*comovdist(z)**2*Rz(z) / Hz(z)/(1.+z))

	return np.array(listPz)

def GenerateRedshift(zmax,nsample):
	
	zrange = np.linspace(0, zmax, 10000)
	Pzs = Pz(zrange)
	Pzs = Pzs/sum(Pzs)
	cum_Pz = np.cumsum(Pzs)

	uniform_randoms = np.random.uniform(0, 1, nsample)
	invfcum_Pz = interp1d(cum_Pz, zrange)
	projection = invfcum_Pz(uniform_randoms)
	return projection
	

def GeneratedL(zs):
	listdL = []
	for z in zs:
		listdL.append((1.0+z)*comovdist(z))
	return np.array(listdL)

def SigmadL():

	return dL/rho


zmax = 5
nsample = 10000
z_sample = GenerateRedshift(zmax, nsample)
dL_sample = GeneratedL(z_sample)



