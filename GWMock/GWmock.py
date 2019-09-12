from matplotlib.pyplot import *
import numpy as np
from numpy import *
import scipy
from scipy import interpolate
ion()
show()

H0 = 67.74e3 #en m/s/Mpc #6.62606896e-27 
Om0 = 0.315
Or0 = 8.47 * 10**-5
OK0 = 0.0
OL0 = 0.685

G_new = 6.674 * 10**-11 # [m3 kg−1 s−2]
c_lum = 299792458 # [m/s]

m_sol =  1.98847*10**30 #G_new/c_lum**3 # [kg]

def Hz(z):
	return H0 * np.sqrt(Om0*(1.+z)**3 + Or0*(1.+z)**4 + OK0*(1.+z)**2 + OL0)

def dist_como(z):
	return c_lum * scipy.integrate.quad(lambda z : 1.0/Hz(z), 0, z)[0]

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
		listPz.append(4*np.pi*dist_como(z)**2*Rz(z) / Hz(z)/(1.+z))

	return np.array(listPz)

def GenerateRedshift(zmax,nsample):
	
	zrange = np.linspace(0, zmax, 10000)
	Pzs = Pz(zrange)
	Pzs = Pzs/sum(Pzs)
	cum_Pz = np.cumsum(Pzs)

	uniform_randoms = np.random.uniform(0, 1, nsample)
	invfcum_Pz = scipy.interpolate.interp1d(cum_Pz, zrange)
	projection = invfcum_Pz(uniform_randoms)
	return projection
	

def GeneratedL(zs):
	listdL = []
	for z in zs:
		listdL.append((1.0+z)*dist_como(z))
	return np.array(listdL)

#def SigmadL():
#
#	return dL/rho


##################################################################
##################################################################
##################################################################


# plot(ETDSensitivity[:, 0], ETDSensitivity[:, 1])
# ylabel(r'$\rm strain[1/\sqrt{freq[Hz]}$')
# xlabel(r'$\rm freq \,[Hz]$')


def Sh(f):
	S0 = 1.449*10**(-52)
	f0 = 200.0
	x = f/f0
	p1 = -4.05
	p2 = -0.69
	a1 = 185.62
	a2 = 232.56
	b1 = 31.18
	b2 = -64.72
	b3 = 52.24
	c1 = 13.58
	c2 = -36.46
	p2 = -0.69
	a2 = 232.56
	b4 = -42.16
	b5 = 10.17
	b6 = 11.53
	c3 = 18.56
	c4 = 27.43

	num = 1.0 + b1*x + b2*x**2 + b3*x**3 + b4*x**4 + b5*x**5 + b6*x**6
	den = 1.0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4
	return S0*(x**p1 + a1*x**p2 + a2*num/den)

def Fp1(theta, phi, psi):
	return 0.5*np.sqrt(3)*(0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi) - np.cos(theta)*np.sin(2*phi)*np.sin(2*psi))

def Fx1(theta, phi, psi):
	return 0.5*np.sqrt(3)*(0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi) + np.cos(theta)*np.sin(2*phi)*np.cos(2*psi))

def Fp2(theta, phi, psi):
	return Fp1(theta, phi+np.pi/3.*2. , psi)

def Fx2(theta, phi, psi):
	return Fx1(theta, phi+np.pi/3.*2. , psi)

def Fp3(theta, phi, psi):
	return Fp1(theta, phi+np.pi/3.*4. , psi)

def Fx3(theta, phi, psi):
	return Fx3(theta, phi+np.pi/3.*4. , psi)
	
def moduleHcal(freq, Acal):
	'''
	Compute the module of Fourier transform of the time domaine wave-form.
	'''
	return Acal * freq**(-7./3.) / Sh_interp(freq)

# def Acal(omega, dL, Mc, theta, phi, psi, Fp, Fx):
# 	'''
# 	Fourier amplitude
# 	'''
# 	Fp2 = Fp(theta, phi, psi)**2
# 	Fx2 = Fx(theta, phi, psi)**2

# 	sqrt1 = np.sqrt(Fp2 * (1.0 + np.cos(omega)**2)**2 + 4*Fx2*np.cos(omega)**2)
# 	sqrt2 = np.sqrt(5*np.pi / 96.) * np.pi**(-7./6.) * Mc**(5./6.)
# 	return 1.0/dL * sqrt1 * sqrt2


def sumAcal(theta, Mc, dL):
	factor = 5*np.pi / 96. * np.pi**(-7./3.) * Mc**(5./3.)
	return 3.0/dL**2 * 3./2. * factor * (0.25 * (1 + np.cos(theta)**2)**2 + np.cos(theta)**2) 


def chirpmass(m1_obs, m2_obs):
	M = m1_obs+m2_obs
	eta = m1_obs*m2_obs/M/M
	Mc = M*eta**(3./5.)
	return Mc

def generate_intrinsic_masses(nsample):
	m1_int = m_sol * np.random.uniform(1, 2, nsample)
	m2_int = m_sol * np.random.uniform(1, 2, nsample)
	return m1_int, m2_int


def generateangle(nsample):
	theta_incl = np.random.uniform(0, np.pi, nsample) #[0, pi]
	phi_azi = np.random.uniform(0, 2*np.pi, nsample) #[0, 2pi]
	psi = np.random.uniform(0, np.pi, nsample) #[0, 2pi]
	return theta_incl, phi_azi, psi

def innerprod(flow, fupp, Acal):
	return 4 * scipy.integrate.quad(moduleHcal, flow, fupp, args=(Acal))[0]


##### generate redshift z and distance lumonisty dL

zmax = 5
nsample = 100
z_sample = GenerateRedshift(zmax, nsample)
dL_sample = GeneratedL(z_sample)

figure()
clf()
z_array = np.arange(0, 5, 0.01)
dL_array = GeneratedL(z_array)
plot(z_array, dL_array)
ylabel('dL')
xlabel('z')


##### distance luminosity error sigma_dL
'''
probably bugged, since rho ~ 1e29 
planck constant, ligh speed, or grav constant misplaced? forgotten ?

'''

ETDSensitivity = np.loadtxt("ET-0002A-18_ETBSensitivityCurveTxtFile.txt")
Sh_interp = scipy.interpolate.interp1d(ETDSensitivity[:, 0], ETDSensitivity[:, 1])


theta_incl, phi_azi, psi = generateangle(nsample)
m1_int, m2_int = generate_intrinsic_masses(nsample)
m1_obs, m2_obs = m1_int*(z_sample+1), m2_int*(z_sample+1)
Mc = chirpmass(m1_obs, m2_obs)
allsumAcal = sumAcal(theta_incl, Mc, dL_sample)

rho2 = []
for samp in np.arange(nsample):
	Acal = allsumAcal[samp]
	Mobs = m1_obs[samp]+m2_obs[samp]
	fLSO = 1.0/(6**(3./2) * 2*np.pi * Mobs)
	flow, fupp = (ETDSensitivity[0, 0], ETDSensitivity[-1, 0]) #(ETDSensitivity[0, 0], 2*fLSO)
	rho2.append(innerprod(flow, fupp, Acal))

rho = np.sqrt(rho2)
sigma_dL = dL_sample/rho

figure()
clf()
plot(z_sample, sigma_dL, '.')


errorbar(z_sample, dL_sample, yerr=sigma_dL, fmt='.')
plot(z_sample, sigma_dL, fmt='.')





