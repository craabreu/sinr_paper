import numpy as np
import matplotlib.pyplot as plt

params = {
	'font.size': 8,
	'text.usetex': True,
	'lines.linewidth': 0.5,
	'legend.frameon': False,
	'savefig.format': 'eps',
	'savefig.dpi': 600,
	'savefig.bbox': 'tight',
}
plt.rcParams.update(params)

fig = plt.figure('axes', figsize=(3.37,2))
ax = fig.add_subplot(111)
ax.set_xlabel('$r$ (\AA)')
ax.set_ylabel('$E$ (kcal/mol)')

epsilon = 0.1554253     # kcal/mol
sigma = 3.16549212      # angstroms
chargeprod = (-0.84)**2 # e
Kc = 332.063709         # kcal/(mol*angstrom*e**2),
rmin = 4.0              # angstroms
rcut = 9.0             # angstroms
rc0 = 8.0               # angstroms
rs0 = 5.0               # angstroms

def switched(r):
  u = np.heaviside(r-rs0,1)*(r-rs0)/(rc0-rs0)
  S = 1 + u**3*(15*u - 6*u**2 - 10)
  potential = 4*epsilon*((sigma/r)**12-(sigma/r)**6) + Kc*chargeprod/r
  return np.heaviside(rc0-r,1)*S*potential

def shifted_switched(r):
  u = np.heaviside(r-rs0,1)*(r-rs0)/(rc0-rs0)
  S = 1 + u**3*(15*u - 6*u**2 - 10)
  potential = 4*epsilon*((sigma/r)**12-(sigma/r)**6) + Kc*chargeprod/r
  shift = 4*epsilon*((sigma/rc0)**12-(sigma/rc0)**6) + Kc*chargeprod/rc0
  return np.heaviside(rc0-r,1)*S*(potential - shift)

def force_swithed(r):
	u = np.heaviside(r-rs0,1)*(r-rs0)/(rc0-rs0)
	b = rs0/(rc0-rs0)
	R = u/b+1
	f1 = 1+5*(b+1)**2*(6*b**3*R*np.log(R)-6*b**2*u-3*b*u**2+u**3)+u**4*(3*u-5*b-10)/2
	f6 = 1+(6*b**2-3*b+1)*(b**3*(R**6-1)-6*b**2*u-15*b*u**2-20*u**3)+45*(1-2*b)*u**4-36*u**5
	f12 = 1+(6*b**2-21*b+28)*(b**3*(R**12-1)-12*b**2*u-66*b*u**2-220*u**3)/462+45*(7-2*b)*u**4/14-72*u**5/7
	f1c = (30*(1+b))*(b**2*(1+b)**2*np.log(1/b+1)-b**3-(3/2)*b**2-(1/3)*b+1/12)
	f6c = (1+b)**3/b**3
	f12c = (1+b)**3*(b**6+3*b**5+(30/7)*b**4+(25/7)*b**3+(25/14)*b**2+(1/2)*b+2/33)/b**9
	potential = 4*epsilon*(f12*(sigma/r)**12-f6*(sigma/r)**6) + Kc*chargeprod*f1/r
	shift = 4*epsilon*(f12c*(sigma/rc0)**12-f6c*(sigma/rc0)**6) + Kc*chargeprod*f1c/rc0
	return np.heaviside(rc0-r,1)*(potential - shift)

def total(r):
  return 4*epsilon*((sigma/r)**12-(sigma/r)**6) + Kc*chargeprod/r

Es = np.vectorize(switched)
Ess = np.vectorize(shifted_switched)
Efs = np.vectorize(force_swithed)
Et = np.vectorize(total)

r = np.linspace(rmin, rcut, 200)
ax.plot(r, Es(r), 'b--')
ax.plot(r, Ess(r), 'g--')
ax.plot(r, Efs(r), 'r--')
ax.plot(r, Et(r)-Es(r), 'b-', label=r'switched')
ax.plot(r, Et(r)-Ess(r), 'g-', label=r'shifted \& switched')
ax.plot(r, Et(r)-Efs(r), 'r-', label=r'force-switched')
ax.plot(r, Et(r), 'k-')
ax.legend()

plt.show()
fig.savefig('potentials')
