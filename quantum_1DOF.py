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

#%%
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
    H_jk : TYPE
        DESCRIPTION.

    """
    
    dispatcher = {'harmonic' : V_harmonic_1dof, 'saddlenode' : V_SN_1dof, 'morse' : V_morse_1dof}
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
        print(j)
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
        print(j)
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
        print(j)
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