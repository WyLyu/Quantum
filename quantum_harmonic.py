# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:58:52 2020

@author: Adimn
"""

#%%
from scipy.integrate import odeint,quad,trapz,solve_ivp, ode
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import scipy.linalg as linalg
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from matplotlib import cm
from scipy import optimize
from scipy.optimize import fsolve
from pylab import rcParams
import seaborn as sns
from scipy.special import eval_hermite
from scipy.special import gamma
from scipy.special import genlaguerre
fal = 30 # fontsize axis labels

ftl = 20 # fontsize tick labels

mpl.rcParams['xtick.labelsize'] = ftl

mpl.rcParams['ytick.labelsize'] = ftl

# mpl.rcParams['ztick.labelsize'] = ftl

mpl.rcParams['axes.labelsize'] = fal

#m = cm.ScalarMappable(cmap=cm.jet)

mpl.rcParams['font.weight'] = 'normal'



lw = 3.0 # linewidth for line plots

mpl.rcParams['mathtext.fontset'] = 'cm'

mpl.rcParams['mathtext.rm'] = 'serif'



plt.style.use('seaborn')



rcParams['figure.figsize'] = 8, 8 

axis_fs = 30

savefig_flag = True


#%%
import quantum_1DOF


#%% Comparison of the analytical solution of the harmonic oscilattor with its numerical solution computed using the FGH method.
omega=1.0
N = 599
a = -4
b = 4
h = 1
H=quantum_1DOF.Qdist_H_jk_FGH(a, b, N, h, omega, 'harmonic')


#%%
evalue,evec=np.linalg.eig(H)
idx = evalue.argsort()[::-1]   
evalue = evalue[idx]
evec = evec[:,idx]
x_range = np.linspace(-4,4,1000)

n = 2
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plot2 = ax.plot(a+np.arange(N)*(b-a)/(N-1),evec[:,-n]*np.sqrt((N-1)/(b-a)), '-',lw=lw,label=r'$E_1 = %.2f$, numer' %(evalue[-n]))

plot2an = ax.plot(x_range,quantum_1DOF.harmonic_1dof_analytical(x_range, omega, n - 1, 1), '-',lw=lw,label=r'$E_n = %.2f$, analy' %(h*omega*(n-1+0.5)))

legend = ax.legend(loc='best')
ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'normalised energy eigenvector $\psi_{n}$', fontsize=axis_fs)
ax.set_xlim(-5, 5)
ax.set_ylim(-1, 1)

plt.show()
# plt.savefig('quantum_harm_eigenvec_1.pdf', format='pdf', \
            # bbox_inches='tight')
            

#%% Comparison of the analytical solution of the harmonic oscilattor with its numerical solution using the FD method.
omega=1.0
N = 599
a = -4
b = 4
h = 1
H=quantum_1DOF.Qdist_H_jk_FD(a, b, N, h, omega, 'harmonic')


#%%
evalue,evec=np.linalg.eig(H)
idx = evalue.argsort()[::-1]   
evalue = evalue[idx]
evec = evec[:,idx]
x_range = np.linspace(-4,4,1000)
evec_wbound = np.zeros((N, N-2))
evec_wbound[1:N-1, :] = evec

n=2
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plot = ax.plot((a+np.arange(N)*(b-a)/(N-1)),evec_wbound[:,-n]*np.sqrt((N-1)/(b-a)), '-',lw=lw,label=r'$E_1 = %.2f$, numer' %(evalue[-n]))

plotan = ax.plot(x_range,quantum_1DOF.harmonic_1dof_analytical(x_range, omega, n-1, 1), '-',lw=lw,label=r'$E_1 = %.2f$, analy' %(h*omega*(n-1+0.5)))

legend = ax.legend(loc='best')
ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'normalised energy eigenvector $\psi_{n}$', fontsize=axis_fs)
ax.set_xlim(-5, 5)
ax.set_ylim(-1, 1)


#%% Comparison of the analytical solution of the Morse oscilattor with its numerical solution using the FGH method.
MASS_A=1.0
MASS_B=1.0
EPSILON_S=0.0
aa=1.0
De = 2
xe = 1
N = 179
parameters = np.array([aa, De, xe])
a = -3
b = 10
h = 1
H=quantum_1DOF.Qdist_H_jk_FGH(a, b, N, h, parameters, 'morse')


#%%
evalue,evec=np.linalg.eig(H)
idx = evalue.argsort()[::-1]   
evalue = evalue[idx]
evec = evec[:,idx]
x_range = np.linspace(-3,10,1000)

n=2
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()

plot = ax.plot(a+np.arange(N)*(b-a)/(N-1),evec[:,-n]*np.sqrt((N-1)/(b-a)), '-',lw=lw,label=r'$E_1 = %.2f$, numer' %(evalue[-n]))

E_n, psi_n = quantum_1DOF.morse_1dof_analytical(x_range, parameters, n-1, h)
plotan = ax.plot(x_range,psi_n, '-',lw=lw,label=r'$E_n = %.2f$, analy' %(E_n))


legend = ax.legend(loc='best')
ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'normalised energy eigenvector $\psi_{n}$', fontsize=axis_fs)
ax.set_xlim(-3, 10)
ax.set_ylim(-1, 1)
# plt.show()
# plt.savefig('quantum_morse_eigenvec_1.pdf', format='pdf', \
#             bbox_inches='tight')


#%% Comparison of the analytical solution of the Morse oscilattor with its numerical solution using the FD method. 
MASS_A=1.0
MASS_B=1.0
EPSILON_S=0.0
aa=1.0
De = 2
xe = 1
N = 179
parameters = np.array([aa, De, xe])
a = -3
b = 10
h = 1
H=quantum_1DOF.Qdist_H_jk_FD(a, b, N, h, parameters, 'morse')


#%%
evalue,evec=np.linalg.eig(H)
idx = evalue.argsort()[::-1]   
evalue = evalue[idx]
evec = evec[:,idx]
x_range = np.linspace(-3,10,1000)
evec_wbound = np.zeros((N, N-2))
evec_wbound[1:N-1, :] = evec

n = 2
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plot = ax.plot((a+np.arange(N)*(b-a)/(N-1)),evec_wbound[:,-2]*np.sqrt((N-1)/(b-a)), '-',lw=lw,label=r'$E_1 = %.2f$, numer' %(evalue[-2]))

E_n, psi_n = quantum_1DOF.morse_1dof_analytical(x_range,parameters, n-1, h)
plotan = ax.plot(x_range,psi_n, '-',lw=lw,label=r'$E_n = %.2f$, analy' %(E_n))

legend = ax.legend(loc='best')
ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'normalised energy eigenvector $\psi_{n}$', fontsize=axis_fs)
ax.set_xlim(-3, 10)
ax.set_ylim(-1, 1)


#%% numerical solution of the Wigner function for 1 DOF harmonic oscillator, eigenfunction computed using the FGH method
omega=1.0
N = 199
a = -4
b = 4
h = 1
H=quantum_1DOF.Qdist_H_jk_FGH(a, b, N, h, omega, 'harmonic')


#%%
evalue,evec=np.linalg.eig(H)
idx = evalue.argsort()[::-1]   
evalue = evalue[idx]
evec = evec[:,idx]

n = 2
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plotx = np.linspace(a, b, N)
# plotp = np.linspace(-(N-1) // 2, (N-1) // 2 + 1, N) * 2 * np.pi / (b - a)
plotp = np.linspace(a, b, N)
X,P = np.meshgrid(plotx, plotp)
W = quantum_1DOF.Wigner(a, b, N, h, evec[:,-n]*np.sqrt((N-1)/(b-a)))
# W = quantum_1DOF.Wigner(a, b, N, h, quantum_1DOF.harmonic_1dof_analytical(np.linspace(a, b, N), parameters, 4, 1))
cset1 = plt.contour(X, P, W, 20, \
                   cmap='RdGy')
ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'$p$', fontsize=axis_fs)
ax.set_xlim(a, b)
# ax.set_ylim(-(N-1) // 2 * np.pi / (b - a), (N-1) // 2 * np.pi / (b - a))
ax.set_ylim(a, b)
plt.colorbar()
plt.show()
#plt.savefig('quantum_harm1DOF_wigner_num_1_N599.pdf', format='pdf', \
#            bbox_inches='tight')
# plt.savefig('quantum_harm1DOF_wigner_anaevec_1_N599.pdf', format='pdf', \
            # bbox_inches='tight')
            

#%% analytical solution of the Wigner function for 1 DOF harmonic oscillator 
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plotx = np.linspace(a, b, N)
plotp = np.linspace(a, b, N)
# plotp = np.linspace(-(N-1) // 2 * np.pi / (b - a), (N-1) // 2 * np.pi / (b - a), N)
X,P = np.meshgrid(plotx, plotp)
W = np.zeros((N, N))
for j in range(N):
    for k in range(N):
        W[j, k] = quantum_1DOF.Wigner_analytic_harmonic_1(plotx[j], plotp[k], h, 1)
cset1 = plt.contour(X, P, W, 20, \
                   cmap='RdGy')

ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'$p$', fontsize=axis_fs)
ax.set_xlim(a, b)
# ax.set_ylim(-(N-1) // 2 * np.pi / (b - a), (N-1) // 2 * np.pi / (b - a))
plt.colorbar()
plt.show()
# plt.savefig('quantum_harm1DOF_wigner_ana_1.pdf', format='pdf', \
#             bbox_inches='tight')


#%% numerical solution of the Husimi function for 1 DOF harmonic oscillator 
evalue,evec=np.linalg.eig(H)
idx = evalue.argsort()[::-1]   
evalue = evalue[idx]
evec = evec[:,idx]

n = 2
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plotx = np.linspace(a, b, N)
plotp = np.linspace(-(N-1) // 2, (N-1) // 2 + 1, N) * 2 * np.pi / (b - a)
plotp = np.linspace(a, b, N)
X,P = np.meshgrid(plotx, plotp)
H_1DOF = quantum_1DOF.Husimi(a, b, N, h, 1, evec[:,-n]*np.sqrt((N-1)/(b-a)))
# H_1DOF = quantum_1DOF.Husimi(a,b,N, h, 1, quantum_1DOF.harmonic_1dof_analytical(np.linspace(a, b, N), parameters, n-1, 1))
cset1 = plt.contour(X, P, H_1DOF, 20, \
                   cmap='RdGy')
ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'$p$', fontsize=axis_fs)
ax.set_xlim(a, b)
# ax.set_ylim(-(N-1) // 2 * np.pi / (b - a), (N-1) // 2 * np.pi / (b - a))
ax.set_ylim(a, b)
plt.colorbar()
plt.show()
#plt.savefig('quantum_harm1DOF_husimi_num_1_N599.pdf', format='pdf', \
#            bbox_inches='tight')
# plt.savefig('quantum_harm1DOF_husimi_anaevec_1_N599.pdf', format='pdf', \
            # bbox_inches='tight')


#%% analytical solution of the Husimi function for 1 DOF harmonic oscillator 
sns.set(font_scale = 2)
sns.set_style("whitegrid")
ax = plt.gca()
plotx = np.linspace(a, b, N)
plotp = np.linspace(a, b, N)
# plotp = np.linspace(-(N-1) // 2 * np.pi / (b - a), (N-1) // 2 * np.pi / (b - a), N)
X,P = np.meshgrid(plotx, plotp)
H_1DOF = np.zeros((N, N))
for j in range(N):
    for k in range(N):
        H_1DOF[j, k] = quantum_1DOF.Husimi_analytic_harmonic_1(plotx[j], plotp[k], h, 1, 1)
cset1 = plt.contour(X, P, H_1DOF, 20, \
                   cmap='RdGy')

ax.set_xlabel(r'$x$', fontsize=axis_fs)
ax.set_ylabel(r'$p$', fontsize=axis_fs)
ax.set_xlim(a, b)
# ax.set_ylim(-(N-1) // 2 * np.pi / (b - a), (N-1) // 2 * np.pi / (b - a))
plt.colorbar()
plt.show()
# plt.savefig('quantum_harm1DOF_husimi_ana_1.pdf', format='pdf', \
#             bbox_inches='tight')

