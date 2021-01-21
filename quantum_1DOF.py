# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:04:12 2020

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
import sympy
from sympy.solvers import solve
from sympy import Symbol

#%% functions for the 1 DOF system 
# class potential_energy():
#     def __init__(self, x=0, par =np.zeros(5)):
#         self.x = x
#         self.par= par

#         return None
#     def V_SN_1dof(self):
#         V = (1/3)*self.par[3]*self.x**3 - math.sqrt(self.par[4])*self.x**2
#         return V
    
#     def V_harmonic_1dof(self):
#         V =  0.5*self.par[4]**2*self.x**2
#         return V
    
#     def V_morse_1dof(self):
#         x = self.x
#         aa = self.par[0]
#         xe = self.par[1]
#         De = self.par[2]
#         V =  De*((1 - np.e**(-aa*(x-xe)))**2 - 1)
#         return V
#     pass
    
def V_SN_1dof(x, par):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy of the 1 DOF saddle node system evaluated at x.

    """
    
    V = (1/3)*par[3]*x**3 - math.sqrt(par[4])*x**2
    return V

def V_harmonic_1dof(x, par):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy of the 1 DOF harmonic oscillator system evaluated at x.

    """
    
    V =  0.5*par**2*x**2
    return V

def V_invert_harmonic_1dof(x, par):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy of the 1 DOF inverted harmonic oscillator system evaluated at x.

    """
    
    V =  -0.5*par**2*x**2
    return V

def V_morse_1dof(x, par):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy of the 1 DOF morse oscillator system evaluated at x.

    """
    
    aa = par[0]
    De = par[1]
    xe = par[2]
    V =  De * ((1 - np.e**(-aa*(x-xe)))**2 - 1)
    return V

def harmonic_1dof_analytical(x, par, n, h):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.
    n : TYPE
        nth excited energy state of the system, starting from zero.
    h : TYPE
        reduced planck's constant of the system.

    Returns
    -------
    psi : TYPE
        analytical solution of the nth energy eigenfunction of the 1 DOF harmonic oscillator system evaluated at x.

    """
    
    psi = 1 / np.sqrt(2**n * math.factorial(n)) * (par/ h / np.pi)** (0.25) * np.exp(-par * x**2 / 2 / h) \
        * eval_hermite(n,np.sqrt(par/h)*x)
    return psi

def morse_1dof_analytical(x, par, n, h):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.
    n : TYPE
        nth excited energy state of the system, starting from zero.
    h : TYPE
        reduced planck's constant of the system.

    Returns
    -------
    psi : TYPE
        analytical solution of the nth energy eigenfunction of the 1 DOF morse oscillator system evaluated at x.
    E : TYPE
        analytical solution of the nth energy eigenvalue of the 1 DOF morse oscillator system.

    """
    
    aa = par[0]
    De = par[1]
    xe = par[2]
    lamb = np.sqrt(2 * De) / aa / h
    epsilon = - (lamb - n - 0.5)**2
    E = aa**2 * h**2 * epsilon / 2
    N_n = (math.factorial(n) * (2 * lamb - 2 * n - 1) / gamma(2 * lamb - n) )**(0.5)
    z = 2 * lamb * np.e**(-aa*(x-xe))
    psi = N_n * z ** (lamb - n - 0.5) * np.e**(- 0.5 * z) * genlaguerre(n,2 * lamb - 2 * n - 1 )(z)
    return E, psi

def invert_harmonic_1dof_evalue_analytical(par, h, N_a, L):
    """
    

    Parameters
    ----------
    par : TYPE
        parameters of the potential energy function.
    h : TYPE
        reduced planck's constant of the system.
    N_a : TYPE
        highest order of the polynomial that has nonzero terms. 
    L : TYPE
        boundary value that is used for solving the schrodinger equation.

    Returns
    -------
    a_odd_real : TYPE
        real solution of the odd function psi_2(z) that satisfy the boundary condition at L, used to determine the energy eigenvalues.
    a_even_odd : TYPE
        real solution of the even function psi_1(z) that satisfy the boundary condition at L, used to determine the energy eigenvalues.

    """
    a = Symbol('a')
    L_0 = L * np.sqrt(2 * par / h)
    c_a = []
    c_a.append(0)
    c_a.append(0)
    c_a.append(1)
    c_a.append(1)
    sum_odd = c_a[2]*L_0
    sum_even = c_a[3]
    for i in range(4, (N_a+2)):
        c_a.append(a * c_a[i-2] - 0.25 * (i-2-2) * (i-2-3) * c_a[i-4])
        if i % 2 == 1:
            sum_odd = sum_odd + c_a[i]*L_0**(i-2)/math.factorial(i-2)
        else:
            sum_even = sum_even + c_a[i]*L_0**(i-2)/math.factorial(i-2)
                
    a_odd = solve(sum_odd, a)
    a_even = solve(sum_even, a)
    a_odd_real = []
    a_even_real = []
    for i in range(len(a_even)):
        if a_even[i].is_real==True:
            a_even_real.append(float(a_even[i]))
    for i in range(len(a_odd)):
        if a_odd[i].is_real==True:
            a_odd_real.append(float(a_odd[i]))
    
    return a_odd_real, a_even_real

def invert_harmonic_1dof_evec_analytical(x, par, h, N_a, a, model):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.
    h : TYPE
        reduced planck's constant of the system.
    N_a : TYPE
        highest order of the polynomial that has nonzero terms. 
    a : TYPE
        parameter that is related to the energy eigenvalue
    model : TYPE
        return old/even function of the analytical solution of the energy eigenfunction

    Returns
    -------
    f_odd : TYPE
        odd function of the analytical solution of the energy eigenfunction of the 1 DOF inverted harmonic oscillator system evaluated at x.
    f_even : TYPE
        even function of the analytical solution of the energy eigenfunction of the 1 DOF inverted harmonic oscillator system evaluated at x.

    """
    
    c = np.zeros(N_a+2)
    c[2] = 1
    c[3] = 1
    f_even = 1
    f_odd = np.sqrt(2 * par / h)*x
    for i in range(4, (N_a+2)):
        c[i] = a * c[i-2] - 0.25 * (i-2-2) * (i-2-3) * c[i-4]
        if i % 2 == 1:
            f_odd = f_odd + c[i]*(np.sqrt(2 * par / h)*x)**(i-2)/math.factorial(i-2)
        else:
            f_even = f_even + c[i]*(np.sqrt(2 * par / h)*x)**(i-2)/math.factorial(i-2)
    if model == 'odd':
        return f_odd
    elif model == 'even':
        return f_even

def invert_harmonic_1dof_analytical(x, par, n, h, N_a, L):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.
    n : TYPE
        nth excited energy state of the system, starting from zero.
    h : TYPE
        reduced planck's constant of the system.

    Returns
    -------
    E : TYPE
        analytical solution of the nth energy eigenvalue of the 1 DOF inverted harmonic oscillator system.
    psi : TYPE
        analytical solution of the nth energy eigenfunction of the 1 DOF inverted harmonic oscillator system evaluated at x.

    """
    
    a_odd_real, a_even_real = invert_harmonic_1dof_evalue_analytical(par, h, N_a, L)
    a_real = np.concatenate((a_even_real, a_odd_real), axis=None)
    index = np.argsort(a_real)[-n-1]
    # print(index)
    if index >= len(a_even_real):
        evec_ana = invert_harmonic_1dof_evec_analytical(x, par, h, N_a, a_real[index], 'odd')
    else:
        evec_ana = invert_harmonic_1dof_evec_analytical(x, par, h, N_a, a_real[index], 'even')
    e_vec_ana_norm = evec_ana/np.linalg.norm(evec_ana)
    E = -h*par*a_real[index]
    psi = e_vec_ana_norm
    return E, psi

def potential_energy_1dof(x, par, model):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    par : TYPE
    
        parameters of the potential energy function.
    model : TYPE
        model chosen for the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy evaluated at x given the model system.

    """
    
    dispatcher = {'invertharmonic' : V_invert_harmonic_1dof, 'harmonic' : V_harmonic_1dof, 'saddlenode' : V_SN_1dof, 'morse' : V_morse_1dof}
    try:
        V = dispatcher[model](x, par)
        # print(V)
        return V
    except:
        return "Invalid function"


def Qdist_H_jk_FGH(a, b, N, h, par, model):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval chosen for solving the time independent schrodinger equation.
    N : TYPE
        number of grid points, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.
    model : TYPE
        model chosen for the potential energy function.

    Returns
    -------
    H_jk : TYPE
        Hamiltonian matrix which is a discretisation of the Hamiltonian operator, computed using the FGH method.

    """
    
    dx = (b-a)/(N-1)
    dp = 2*np.pi / (N-1) / dx
    H_jk = np.zeros((N,N))
    for j in range(N):
        #print(j)
        H_jk[j,j] = H_jk[j,j] + sum(2/(N-1) * h**2 * (np.arange(int((N+1)/2))*dp)**2 / 2) \
            + potential_energy_1dof((a+j*dx),par, model)
        for k in range((j+1),N):
            H_jk[j,k] = H_jk[j,k] + sum(2/(N-1) * h**2 * (np.arange(int((N+1)/2))*dp)**2 / 2 * np.cos(2*np.pi*np.arange(int((N+1)/2))*(j-k)/(N-1)))
            H_jk[k,j] = H_jk[j,k] 
    return H_jk

def Qdist_H_jk_FD(a, b, N, h, par, model):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval chosen for solving the time independent schrodinger equation.
    N : TYPE
        number of grid points, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.
    model : TYPE
        model chosen for the potential energy function.

    Returns
    -------
    H_jk : TYPE
        Hamiltonian matrix which is a discretisation of the Hamiltonian operator, computed using the FD method.

    """
    
    x_range = np.linspace(a,b,N)
    dx = (b-a)/(N-1)
    T_jk = np.ones((N-2,N-2))
    V_jk = np.zeros((N-2,N-2))
    V_jk = np.diag(potential_energy_1dof(x_range[1:N-1], par, model))
    T_jk = np.tril(T_jk, k=1)
    T_jk = np.triu(T_jk, k=-1)
    T_jk = - (T_jk - np.diag(np.ones(N-2))*3) * 0.5 / dx / dx * h * h
    H_jk = T_jk + V_jk
    return H_jk


# Wigner function 
def Wigner(a, b, N, h, eigenvec):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval chosen for solving the time independent schrodinger equation.
    N : TYPE
        number of grid points, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    eigenvec : TYPE
        normalised energy eigenfunction.

    Returns
    -------
    W : TYPE
        Wigner function based on the normalised energy eigenfunction, eigenvec.

    """
    
    dx = (b-a)/(N-1)
    # dp = 2*np.pi / (N-1) / dx
    W = np.zeros((N,N))
    for j in range(N):
        #print(j)
        for k in range(N):
            #p_k = (N - 1) / 2 * dp + k * dp
            p_k = a + k * dx
            # for l in range(min((N-1-j),j) + 1):
            #     W[j, k] = W[j, k] + 2 / np.pi * eigenvec[j + l] * eigenvec[j - l] \
            #               * np.cos(p_k * (2 * dx * l) ) * dx
            W[j, k] = 1 / np.pi / h * sum(eigenvec[j + np.arange(-min((N-1-j),j), min((N-1-j),j) + 1)] * eigenvec[j - np.arange(-min((N-1-j),j), min((N-1-j),j) + 1)] \
                      * np.cos(p_k * (2 * dx * np.arange(-min((N-1-j),j), min((N-1-j),j) + 1)) / h )) * dx
            
    return W

def Wigner_analytic_harmonic_0(x, p, h, omega):
    """
    

    Parameters
    ----------
    x : TYPE
        position value, as an input of the Wigner function.
    p : TYPE
        momentum value, as an input of the Wigner function.
    h : TYPE
        reduced planck's constant of the system.
    omega : TYPE
        parameter value of the 1 DOF harmonic oscillator.

    Returns
    -------
    W : TYPE
        analytical solution of the Wigner function of the ground state for 1 DOF harmonic oscillator.

    """
    
    a_square = h / omega
    W = 1 / np.pi / h * (np.exp(- x * x / a_square - a_square * p * p / h / h))
    return W

def Wigner_analytic_harmonic_1(x, p, h, omega):
    """
    

    Parameters
    ----------
    x : TYPE
        position value, as an input of the Wigner function.
    p : TYPE
        momentum value, as an input of the Wigner function.
    h : TYPE
        reduced planck's constant of the system.
    omega : TYPE
        parameter value of the 1 DOF harmonic oscillator.

    Returns
    -------
    W : TYPE
        analytical solution of the Wigner function of the 1st excited state for 1 DOF harmonic oscillator.

    """
    
    a_square = h / omega
    W = 1 / np.pi / h * (np.exp(- x * x / a_square - a_square * p * p / h / h)) * (-1 + 2 * a_square * p * p / h / h + 2 * x * x / a_square)
    return W

# Husimi function 
def Husimi(a, b, N, h, sigma, eigenvec):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval chosen for solving the time independent schrodinger equation.
    N : TYPE
        number of grid points, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    sigma : TYPE
        width of the minimum certainty state.
    eigenvec : TYPE
        normalised energy eigenfunction.

    Returns
    -------
    H : TYPE
        Husimi function based on the normalised energy eigenfunction, eigenvec.

    """
    
    dx = (b-a)/(N-1)
    # dp = 2*np.pi / (N-1) / dx
    H = np.zeros((N,N))
    for j in range(N):
        #print(j)
        for k in range(N):
            #p_k = (N - 1) / 2 * dp + k * dp
            p_k = a + k * dx
            # for l in range(N):
                # H[j, k] = 1 / 2 / np.pi * np.sqrt(np.pi)* sigma * (sum(np.exp(-(l-j)*dx*(l-j)*dx/2/sigma/sigma)* np.sin((l-j)*dx*p_k)*eigenvec) * dx)**2 \
                # + 1 / 2 / np.pi * np.sqrt(np.pi)* sigma * (sum(np.exp(-(l-j)*dx*(l-j)*dx/2/sigma/sigma)* np.cos((l-j)*dx*p_k)*eigenvec) * dx)**2
            H[j, k] = 1 / 2 / np.pi / np.sqrt(np.pi) / sigma / h * (sum(np.exp(-(np.arange(N) - j) * dx * (np.arange(N) - j) * dx / 2 / sigma / sigma) * np.sin((a + np.arange(N) * dx) * p_k / h) * eigenvec) * dx)**2\
                + 1 / 2 / np.pi / np.sqrt(np.pi) / sigma / h * (sum(np.exp(-(np.arange(N) - j) * dx * (np.arange(N) - j) * dx / 2/ sigma / sigma) * np.cos((a + np.arange(N) * dx) * p_k / h) * eigenvec) * dx)**2
            
    return H

def Husimi_analytic_harmonic_0(x, p, h, omega, sigma):
    """
    

    Parameters
    ----------
    x : TYPE
        position value, as an input of the Husimi function.
    p : TYPE
        momentum value, as an input of the Husimi function.
    h : TYPE
        reduced planck's constant of the system.
    omega : TYPE
        parameter value of the 1 DOF harmonic oscillator.
    sigma : TYPE
        width of the minimum certainty state.

    Returns
    -------
    H : TYPE
        analytical solution of the Husimi function of the ground state for 1 DOF harmonic oscillator.

    """
    
    a_square = h / omega
    H = np.sqrt(a_square)* sigma  / np.pi / h / (a_square + sigma**2) \
        *(np.exp(- x * x / (a_square + sigma**2) - a_square * p * p *sigma * sigma/ h / h / (a_square + sigma**2)))
    return H

def Husimi_analytic_harmonic_1(x, p, h, omega, sigma):
    """
    

    Parameters
    ----------
    x : TYPE
        position value, as an input of the Husimi function.
    p : TYPE
        momentum value, as an input of the Husimi function.
    h : TYPE
        reduced planck's constant of the system.
    omega : TYPE
        parameter value of the 1 DOF harmonic oscillator.
    sigma : TYPE
        width of the minimum certainty state.

    Returns
    -------
    H : TYPE
        analytical solution of the Husimi function of the 1st excited state for 1 DOF harmonic oscillator.

    """
    
    a_square = h / omega
    H = 2 * np.sqrt(a_square)**3 * sigma**5 / np.pi / h  / (a_square + sigma**2) / (a_square + sigma**2) / (a_square + sigma**2) \
        *( x * x / (sigma**4) + p * p / h / h)\
        *(np.exp(- x * x / (a_square + sigma**2) - a_square * p * p * sigma * sigma/ h / h / (a_square + sigma**2)))
    return H

#%% functions for the 2 DOF system 
def V_harmonic_2dof(x, y, par):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    y : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy of the 2 DOF harmonic oscillator system evaluated at x, y.

    """
    
    V = 0.5*par[6]*x*x + 0.5*par[6]*y*y
    return V

def V_SN_2dof(x, y, par):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    y : TYPE
        independent variable value of the potential energy function.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy of the 2 DOF saddle node system evaluated at x, y.

    """
    
    V = (1/3)*par[3]*x**3 - np.sqrt(par[4])*x**2 + 0.5*par[6]*y**2 + 0.5*par[5]*(x-y)**2
    return V

def potential_energy_2dof(x, y, par, model):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    y : TYPE
        independent variable value of the potential energy function.
    par : TYPE
    
        parameters of the potential energy function.
    model : TYPE
        model chosen for the potential energy function.

    Returns
    -------
    V : TYPE
        potential energy evaluated at x, y given the model system.

    """
    
    dispatcher = {'harmonic' : V_harmonic_2dof, 'saddlenode' : V_SN_2dof}
    try:
        V = dispatcher[model](x, y, par)
        #print(V)
        return V
    except:
        return "Invalid function"

def Qdist_H_lm_jk_FGH(a, b, c , d, N_x, N_y, h, par, model):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    c : TYPE
        left end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    d : TYPE
        right end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    N_x : TYPE
        number of grid points in x coordinate, must be an odd integer value.
    N_y : TYPE
        number of grid points in y coordinate, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    H_lm_jk : TYPE
        Hamiltonian matrix which is a discretisation of the Hamiltonian operator, computed using the FGH method.

    """
    
    dx = (b-a)/(N_x-1)
    dpx = 2*np.pi / (N_x-1) / dx
    dy = (d-c)/(N_y-1)
    dpy = 2*np.pi / (N_y-1) / dy
    H_lm_jk = np.zeros((N_x*N_y,N_x*N_y))
    for i1 in range(N_x*N_y):
        #print(i1)
        for i2 in range(i1, N_x*N_y):
            k = (i2 // N_x)
            j = (i2 % N_x)
            m = (i1 // N_x)
            l = (i1 % N_x)
            sum_rhalf = sum(np.cos(2*np.pi*np.arange(1, int((N_x+1)/2))*(j-l)/(N_x-1)))
            sum_r = 2 * sum_rhalf + 1
            sum_shalf = sum(np.cos(2*np.pi*np.arange(1, int((N_y+1)/2))*(k-m)/(N_y-1)))
            sum_s = 2 * sum_shalf + 1
            if j == l and k == m:
                H_lm_jk[m*N_x+l,k*N_x+j] = H_lm_jk[m*N_x+l,k*N_x+j] + 2 * 1/(N_x-1) /(N_y-1)* h**2 * sum((np.arange(1, int((N_x+1)/2))*dpx)**2 / 2 * np.cos(2*np.pi*np.arange(1, int((N_x+1)/2))*(j-l)/(N_x-1))) * sum_s \
                    + 2 * 1/(N_x-1) /(N_y-1)* h**2 * sum((np.arange(1, int((N_y+1)/2))*dpy)**2 / 2 * np.cos(2*np.pi*np.arange(1, int((N_y+1)/2))*(k-m)/(N_y-1))) * sum_r 
                H_lm_jk[m*N_x+l,k*N_x+j] = H_lm_jk[m*N_x+l,k*N_x+j] + potential_energy_2dof((a+j*dx),(c+k*dy),par, model)
                H_lm_jk[k*N_x+j,m*N_x+l] = H_lm_jk[m*N_x+l,k*N_x+j]
            else:
                H_lm_jk[m*N_x+l,k*N_x+j] = H_lm_jk[m*N_x+l,k*N_x+j] + 2 * 1/(N_x-1) /(N_y-1)* h**2 * sum((np.arange(1, int((N_x+1)/2))*dpx)**2 / 2 * np.cos(2*np.pi*np.arange(1, int((N_x+1)/2))*(j-l)/(N_x-1))) * sum_s \
                    + 2 * 1/(N_x-1) /(N_y-1)* h**2 * sum((np.arange(1, int((N_y+1)/2))*dpy)**2 / 2 * np.cos(2*np.pi*np.arange(1, int((N_y+1)/2))*(k-m)/(N_y-1))) * sum_r
                H_lm_jk[k*N_x+j,m*N_x+l] = H_lm_jk[m*N_x+l,k*N_x+j] 

    return H_lm_jk

def Qdist_H_lm_jk_FGHmod(a, b, c , d, N_x, N_y, h, par, model):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    c : TYPE
        left end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    d : TYPE
        right end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    N_x : TYPE
        number of grid points in x coordinate, must be an odd integer value.
    N_y : TYPE
        number of grid points in y coordinate, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    H_lm_jk : TYPE
        Hamiltonian matrix which is a discretisation of the Hamiltonian operator, computed using a modification of the FGH method.

    """
    
    dx = (b-a)/(N_x - 1)
    dpx = 2*np.pi / (N_x - 1) / dx
    dy = (d-c)/(N_y - 1)
    dpy = 2*np.pi / (N_y - 1) / dy
    H_lm_jk = np.zeros((N_x*N_y,N_x*N_y))
    for i1 in range(N_x*N_y):
        #print(i1)
        for i2 in range(i1, N_x*N_y):
            k = (i2 // (N_x))
            j = (i2 % N_x)
            m = (i1 // N_x)
            l = (i1 % N_x)
            if j == l and k == m:
                H_lm_jk[m*N_x+l,k*N_x+j] = sum(2/(N_x - 1) * h**2 * ((np.arange(int((N_x+1)/2))*dpx)**2) / 2 * np.cos(2*np.pi*np.arange(int((N_x+1)/2))*(j-l)/(N_x - 1))*int(m == k)) \
                    + sum(2/(N_y - 1) * h**2 * ((np.arange(int((N_y+1)/2))*dpy)**2) / 2 *np.cos(2*np.pi*np.arange(int((N_y+1)/2))*(k-m)/(N_y - 1))*int(l == j))
                H_lm_jk[m*N_x+l,k*N_x+j] = H_lm_jk[m*N_x+l,k*N_x+j] + potential_energy_2dof((a+(j)*dx),(c+(k)*dy), par, model)
                H_lm_jk[k*N_x+j,m*N_x+l] = H_lm_jk[m*N_x+l,k*N_x+j] 
            else:
                H_lm_jk[m*N_x+l,k*N_x+j] = sum(2/(N_x - 1) * h**2 * ((np.arange(int((N_x+1)/2))*dpx)**2) / 2 * np.cos(2*np.pi*np.arange(int((N_x+1)/2))*(j-l)/(N_x - 1))*int(m == k)) \
                    + sum(2/(N_y - 1) * h**2 * ((np.arange(int((N_y+1)/2))*dpy)**2) / 2 *np.cos(2*np.pi*np.arange(int((N_y+1)/2))*(k-m)/(N_y - 1))*int(l == j))
                H_lm_jk[k*N_x+j,m*N_x+l] = H_lm_jk[m*N_x+l,k*N_x+j] 

    return H_lm_jk

def Qdist_H_lm_jk_FD(a, b, c , d, N_x, N_y, h, par, model):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    c : TYPE
        left end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    d : TYPE
        right end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    N_x : TYPE
        number of grid points in x coordinate, must be an odd integer value.
    N_y : TYPE
        number of grid points in y coordinate, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    H_lm_jk : TYPE
        Hamiltonian matrix which is a discretisation of the Hamiltonian operator, computed using the FD method.

    """
    
    dx = (b-a) / (N_x-1)
    #dpx = 2*np.pi / (N_x-1) / dx
    dy = (d-c) / (N_y-1)
    #dpy = 2*np.pi / (N_y-1) / dy
    V_jk = np.zeros(((N_x-2)*(N_y-2), (N_x-2)*(N_y-2)))
    T1_row = np.zeros((N_x-2)*(N_y-2))  
    T1_row[0] = -2
    T1_row[1] = 1
    
    T2_row = np.zeros((N_x-2)*(N_y-2))
    T2_row[0] = -2
    T2_row[N_x-2] = 1

    T1_jk = linalg.toeplitz(T1_row, T1_row)
    T2_jk = linalg.toeplitz(T2_row, T2_row)
    for i in range((N_x-2)*(N_y-2)):
        k = i // (N_x - 2)
        j = i % (N_x - 2)
        V_jk[i, i] = potential_energy_2dof((a+(j+1)*dx),(c+(k+1)*dy), par, model)
    H_lm_jk =  -0.5 / dx / dx * h * h * T1_jk - 0.5 / dy / dy * h * h * T2_jk + V_jk
    return H_lm_jk

def Husimi_2dof(a, b, c, d, N_x, N_y, h, sigma_1, sigma_2, psi_x_y):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    c : TYPE
        left end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    d : TYPE
        right end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    N_x : TYPE
        number of grid points in x coordinate, must be an odd integer value.
    N_y : TYPE
        number of grid points in y coordinate, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    sigma_1 : TYPE
        width of the minimum certainty state in x coordinate.
    sigma_2 : TYPE
        width of the minimum certainty state in y coordinate.
    psi_x_y : TYPE
        2 DOF normalised energy eigenfunction.

    Returns
    -------
    H_2dof : TYPE
        2 DOF Husimi function with zero p_x and zero p_y, based on the 2 DOF normalised energy eigenfunction, psi_x_y.

    """
    
    dx = (b-a) / (N_x-1)
    dy = (d-c) / (N_y-1)
    H_2dof = np.zeros((N_x,N_y))
    p_x = 0
    p_y = 0
    for j in range(N_x):
        #print(j)
        for k in range(N_y):
            sum_cos = 0
            sum_sin = 0
            for l_1 in range(N_x):
                for l_2 in range(N_y):
                    sum_sin = sum_sin + np.exp(-(l_1-j)*dx*(l_1-j)*dx/2/sigma_1/sigma_1-(l_2-k)*dy*(l_2-k)*dy/2/sigma_2/sigma_2)* np.sin((a+l_1*dx)*p_x/h + (c+l_2*dy)*p_y/h)*psi_x_y[l_1, l_2] * dx * dy
                    sum_cos = sum_cos + np.exp(-(l_1-j)*dx*(l_1-j)*dx/2/sigma_1/sigma_1-(l_2-k)*dy*(l_2-k)*dy/2/sigma_2/sigma_2)* np.cos((a+l_1*dx)*p_x/h + (c+l_2*dy)*p_y/h)*psi_x_y[l_1, l_2] * dx * dy
            H_2dof[j, k] = 1 / (2 * np.pi * h) / (2 * np.pi * h) / np.pi / sigma_1 / sigma_2 * (sum_cos**2 + sum_sin**2)
                
    return H_2dof

def Wigner_2dof(a, b, c, d, N_x, N_y, h, psi_x_y):
    """
    

    Parameters
    ----------
    a : TYPE
        left end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    b : TYPE
        right end point of the interval in x coordinate chosen for solving the time independent schrodinger equation.
    c : TYPE
        left end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    d : TYPE
        right end point of the interval in y coordinate chosen for solving the time independent schrodinger equation.
    N_x : TYPE
        number of grid points in x coordinate, must be an odd integer value.
    N_x : TYPE
        number of grid points in y coordinate, must be an odd integer value.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    W_2dof : TYPE
        2 DOF Wigner function with zero p_x and zero p_y, based on the 2 DOF normalised energy eigenfunction, psi_x_y.

    """
    
    dx = (b-a) / (N_x-1)
    dy = (d-c) / (N_y-1)
    W_2dof = np.zeros((N_x,N_y))
    p_x = 0
    p_y = 0
    for j in range(N_x):
        #print(j)
        for k in range(N_y):
            for l1 in range(-min((N_x-1-j),j), min((N_x-1-j),j) + 1):
                for l2 in range(-min((N_y-1-k),k), (min((N_y-1-k),k) + 1)):
                    W_2dof[j, k] = W_2dof[j, k] +  1 / np.pi / np.pi / h /h * psi_x_y[j + l1, k + l2] * psi_x_y[j - l1, k - l2] \
                      * np.cos(p_x * (2 * dx * l1) /h +  p_y * (2 * dy * l2) /h) * dx * dy
    return W_2dof 

def harmonic_2DOF_analytical(x, y, par, n_x, n_y, h):
    """
    

    Parameters
    ----------
    x : TYPE
        independent variable value of the potential energy function.
    y : TYPE
        independent variable value of the potential energy function.
    n_x : TYPE
        quantum number in x coordinate.
    n_y : TYPE
        quantum number in y coordinate.
    h : TYPE
        reduced planck's constant of the system.
    par : TYPE
        parameters of the potential energy function.

    Returns
    -------
    psi : TYPE
        analytical solution of the (n_x, n_y) th energy eigenfunction of the 2 DOF harmonic oscillator system evaluated at (x, y).

    """
    
    psi = harmonic_1dof_analytical(x, par, n_x, h) * harmonic_1dof_analytical(y, par, n_y, h)
    return psi

def Wigner_2dof_harmonic(x, y, p_x, p_y, n_x, n_y, h, omega_x, omega_y):
    """
    

    Parameters
    ----------
    x : TYPE
        position value, as an input of the Wigner function.
    y : TYPE
        position value, as an input of the Wigner function.
    p_x : TYPE
        momentum value, as an input of the Wigner function.
    p_y : TYPE
        momentum value, as an input of the Wigner function.
    n_x : TYPE
        quantum number in x coordinate.
    n_y : TYPE
        quantum number in y coordinate.
    h : TYPE
        reduced planck's constant of the system.
    omega_x : TYPE
        parameter value of the 1 DOF harmonic oscillator in x coordinate.
    omega_y : TYPE
        parameter value of the 1 DOF harmonic oscillator in y coordinate.

    Returns
    -------
    W_2dof : TYPE
        analytical solution of the 2 DOF Wigner function of the (n_x, n_y) th excited state for 2 DOF harmonic oscillator.
        this function only supports n_x,n_y = 0, 1.

    """
    
    if n_x == 0 and n_y == 0:
        W_2dof = Wigner_analytic_harmonic_0(x, p_x, h, omega_x) * Wigner_analytic_harmonic_0(y, p_y, h, omega_x)
    elif n_x == 0 and n_y == 1:
        W_2dof = Wigner_analytic_harmonic_0(x, p_x, h, omega_x) * Wigner_analytic_harmonic_1(y, p_y, h, omega_x)
    elif n_x == 1 and n_y == 0:
        W_2dof = Wigner_analytic_harmonic_1(x, p_x, h, omega_x) * Wigner_analytic_harmonic_0(y, p_y, h, omega_x)
    return W_2dof

def Husimi_2dof_harmonic(x, y, p_x, p_y, n_x, n_y, h, omega_x, omega_y, sigma_x, sigma_y):
    """
    

    Parameters
    ----------
    x : TYPE
        position value, as an input of the Husimi function.
    y : TYPE
        position value, as an input of the Husimi function.
    p_x : TYPE
        momentum value, as an input of the Husimi function.
    p_y : TYPE
        momentum value, as an input of the Husimi function.
    n_x : TYPE
        quantum number in x coordinate.
    n_y : TYPE
        quantum number in y coordinate.
    h : TYPE
        reduced planck's constant of the system.
    omega_x : TYPE
        parameter value of the 1 DOF harmonic oscillator in x coordinate.
    omega_y : TYPE
        parameter value of the 1 DOF harmonic oscillator in y coordinate.
    sigma_1 : TYPE
        width of the minimum certainty state in x coordinate.
    sigma_2 : TYPE
        width of the minimum certainty state in y coordinate.

    Returns
    -------
    H_2dof : TYPE
        analytical solution of the 2 DOF Husimi function of the (n_x, n_y) th excited state for 2 DOF harmonic oscillator.
        this function only supports n_x,n_y = 0, 1.

    """
    
    if n_x == 0 and n_y == 0:
        H_2dof = Husimi_analytic_harmonic_0(x, p_x, h, omega_x, sigma_x) * Husimi_analytic_harmonic_0(y, p_y, h, omega_x, sigma_x)
    elif n_x == 0 and n_y == 1:
        H_2dof = Husimi_analytic_harmonic_0(x, p_x, h, omega_x, sigma_x) * Husimi_analytic_harmonic_1(y, p_y, h, omega_x, sigma_x)
    elif n_x == 1 and n_y == 0:
        H_2dof = Husimi_analytic_harmonic_1(x, p_x, h, omega_x, sigma_x) * Husimi_analytic_harmonic_0(y, p_y, h, omega_x, sigma_x)
    return H_2dof