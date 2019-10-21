#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:34:25 2019

@author: adrienkuntz
"""

from matplotlib.pyplot import *
import numpy as np
from numpy import *
from scipy import integrate
from scipy import interpolate
ion()
show()

H0 = 70e3 #en [m s-1 Mpc-1] 
Om0 = 0.295323	


G_new = 6.674 * 10**-11 # [m3 kg−1 s−2]
c_lum = 299792458 # [m/s]

m_sol =  1.98847*10**30  # [kg]

#######FIRST PART : THE REDSHIFT DISTRIBUTION

def Hz(z):
	return H0 * np.sqrt(Om0*(1.+z)**3 + (1-Om0))

def dist_como(z):
	return c_lum * integrate.quad(lambda zp : 1.0/Hz(zp), 0, z)[0]

def Rz(z):
	if z<=1:
		return 1.+2.*z 
	elif z <5:
		return 3./4*(5 - z)
	else:
		return 0.0
    
    
def Pz(z) :
    return 4*np.pi*dist_como(z)**2*Rz(z) / (Hz(z)*(1.+z))

def dist_lum(z) :
    return (1.0+z) * dist_como(z)

###Get the inverse function of the redshift distribution
zrange = np.linspace(0, 5, 10000)
Pzs = np.vectorize(Pz)(zrange)
Pzs = Pzs/sum(Pzs)
cum_Pz = np.cumsum(Pzs)
invfcum_Pz = interpolate.interp1d(cum_Pz, zrange)

def GenerateRedshift():
	uniform_randoms = np.random.uniform(0, 1)
	projection = invfcum_Pz([uniform_randoms])[0]
	return projection

#############SECOND PART : THE SIGNAL TO NOISE RATIO
    
###The ET sensitivity
design = 'D'

if design == 'B' :
    ETSensitivity = np.loadtxt("ET-0002A-18_ETBSensitivityCurveTxtFile.txt")
    freq = ETSensitivity[:,0]
    Sn = ETSensitivity[:,1]**2 #! Sn is the square of what ET gives
elif design == 'D' :
    ETSensitivity = np.loadtxt("ET-0000A-18_ETDSensitivityCurveTxtFile.txt")
    freq = ETSensitivity[:,0]
    Sn = ETSensitivity[:,3]**2


loglog(freq, Sn)


###Draw masses and angles

def chirpmass(m1_int, m2_int, z):
	M = m1_int+m2_int
	eta = m1_int*m2_int/M/M
	Mc = M*eta**(3./5.)
	return (1.+z) * Mc

def generate_intrinsic_masses():
	m1_int = m_sol * np.random.uniform(1, 2)
	m2_int = m_sol * np.random.uniform(1, 2)
	return m1_int, m2_int


def generateangle():
    theta = np.arccos(np.random.uniform(-1,1)) #Draw cos(theta) uniformly
    iota = np.arccos(np.random.uniform(np.cos(20.*np.pi/180),1)) #Draw cos(iota) uniformly between 0 and 20 degrees
    return theta, iota

def fmax(M_int, z) :
    return c_lum**3 / (6. * np.sqrt(6) * (1.+z) * G_new * M_int)

def curlyA(iota, theta) :
    return 1./64. * (35. + 28. * np.cos(2.*iota) + np.cos(4.*iota)) * (35. + 28. * np.cos(2.*theta) + np.cos(4.*theta))

def IntTrap(x, y) :
    npts = x.shape[0]
    h = (x[npts-1] - x[0]) / (npts - 1)
    return h/2. * (2. * np.sum(y[1:-1]) + y[0] + y[npts-1]) #np.sum est bien plus efficace qu'une boucle for

def integralfreq(fmax) : #Defines the integral in rho
    index = np.searchsorted(freq, fmax)
    fonction = 1/(freq[:index]**(7./3.) * Sn[:index])
    return IntTrap(freq[:index], fonction)

def signalToNoise(z) :
    m1_int, m2_int = generate_intrinsic_masses()
    theta, iota = generateangle()
    dl_m = dist_lum(z) * 3.085e22  #convert MPc in meters
    
    #print('iota = %f degrees'%(iota * 180 / np.pi))
    #print('fmax = %f'%fmax(m1_int+m2_int,z))
    
    return np.sqrt(15./512./np.pi**(4./3.)/c_lum**3 * curlyA(iota, theta)
            * (G_new * chirpmass(m1_int,m2_int,z))**(5./3.) / dl_m**2 
            * integralfreq(fmax(m1_int+m2_int,z)))
    

####Draw 1000 redshifts and distances
    
i = 0
mockCatalogue = []
while i < 1000 :
    z = GenerateRedshift()
    dl = dist_lum(z)
    rho = signalToNoise(z)
    
    if rho > 8. :
        i += 1
        sigma_dl = 2 * dl / rho
        random_mu = 5*np.log10(np.random.normal(loc=dl, scale = sigma_dl)*10**6)-5 #unite jla
        sigma_mu = 5 /(np.log(10)*np.random.normal(loc=dl, scale = sigma_dl))*sigma_dl
        mockCatalogue.append([z, random_mu, sigma_mu])
        
    
print(mockCatalogue)
np.savetxt('mock_catalogue.txt', np.array(mockCatalogue))

