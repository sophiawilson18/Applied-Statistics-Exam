# coding: utf-8

#Author: Sophia Wilson


import numpy as np                                     # Matlab like syntax for linear algebra and functions
import matplotlib as mpl
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
import sys                                             # Modules to see files and folders in directories
from scipy import stats
from scipy.stats import binom, poisson, norm           # Functions from SciPy Stats...
import math

sys.path.append('External_Functions')
from ExternalFunctions import Chi2Regression, UnbinnedLH, BinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax 

# ======================================
# GENERAL
# ======================================


def weightedmean(x,sigma):
    mean = sum(x/sigma**2)/sum(sigma**-2)
    uncertainty = np.sqrt(1/sum(sigma**-2))
    return mean, uncertainty


def compare(mu1,mu2,sigma1,sigma2):
    from scipy.stats import norm
    from math import erf, sqrt
    
    dmu = np.abs(mu1-mu2) 
    dsigma = np.sqrt(sigma1**2+sigma2**2)
    nsigma = dmu/dsigma
    
    p = norm.cdf(0, dmu, dsigma)*2
    return dmu, dsigma, nsigma, p 

def Chauvenet(datapoint, data):
    mu = data.mean()
    std = data.std()
    nsigma = compare(mu, datapoint, std, 0)
    return nsigma*len(data)


def pull(x, mu, sigma, plot = False, range = None):
    '''Pull distribution / Chauvenets critereon'''
    z = (x-mu)/sigma
    
    if plot == True:
        Nbins = int(np.sqrt(len(z)))
        bin_width = (max(z)-min(z))/Nbins
        
        fig, ax = plt.subplots()
        counts, bins = np.histogram(z)
        plt.hist(z, bins = Nbins, range = range) 
        
        d = {'Entries':     len(z),
             'Mean':        z.mean(),
             'Std':         z.std(),
             'Area':        sum(counts*bin_width)
        }
        
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.02, 0.95, text, ax, fontsize=12)
        
        density = counts / (sum(counts) * np.diff(bins))
        test = np.sum(density * np.diff(bins))
        
    return z.mean(), z.std(), test, Nbins

def chisquare(x_exp, x_obs, x_obs_sigma, N_par):
    
    chi2_value = sum((x_obs-x_exp)**2/x_obs_sigma**2)
    Ndof_value = len(x_obs)-N_par
    chi2_prob = stats.chi2.sf(chi2_value, Ndof_value)
    
    return chi2_value, chi2_prob
    
    



# ======================================
# PROBABILITY DENSITY / MASS FUNCTIONS
# ======================================


def binomial_pmf(x, n, p):
    """Biominal distribution """
    return binom.pmf(x, n, p)

def poisson_pmf(x, lamb):
    """Poisson distribution"""
    return poisson.pmf(x, lamb)

def gaussian_pdf(x, mu, sigma):
    """Gaussian distribution"""
    return norm.pdf(x, mu, sigma)

def gaussian_unit_own(x, mu, sigma):
    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def gauss_extended_own(x, N, mu, sigma) :
    """Extended (non-normalized) Gaussian"""
    return N * gaussian_pdf(x, mu, sigma)



# ======================================
# CHI SQUARE FIT
# ======================================



def chisquarefit(x, y, ysigma, fitfunction, startparameters, plot=False):
    'Chi-square fit'
    chi2fit = Chi2Regression(fitfunction, x, y, ysigma)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()
    
    'Parameters and uncertainties'
    par = minuit_chi2.values[:]   
    par_err = minuit_chi2.errors[:]
    
    'Chi-square value, number of degress of freedom and probability'
    chi2_value = minuit_chi2.fval 
    Ndof_value = len(x)-len(par)
    chi2_prob = stats.chi2.sf(chi2_value, Ndof_value)
    
    'Plotting'
    if plot==True:
        x_axis = np.linspace(min(x), max(x), 1000)
        
        d = {'Chi2':     chi2_value,
             'Prob':     chi2_prob,
            }
            
        fig, ax = plt.subplots(figsize=(14,10))
        ax.plot(x, y, 'k.', label='X')
        ax.plot(x_axis, fitfunction(x_axis, *minuit_chi2.values[:]), '-r', label='Chi2 fit model result') 
        ax.set(xlabel='X', ylabel='X', title='X')
        ax.errorbar(x, y, ysigma, fmt='ro', ecolor='k', elinewidth=2, capsize=2, capthick=1)
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.02, 0.95, text, ax, fontsize=12)
    
    return chi2_value, Ndof_value, chi2_prob, par, par_err


def chisquarefit_histogram(X, fitfunction, startparameters, xmin, xmax, N_bins, plot=False, verbose = False):
    '''CHI SQUARE FIT FOR HISTROGRAMS'''
    
    'Histrogram'
    counts, bin_edges = np.histogram(X, bins=N_bins, range=(xmin, xmax), density=True)
    
    'Define x, y, sy. Makes sure all bins are nonzero'
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0])
    
    'Chi-square fit'
    chi2fit = Chi2Regression(fitfunction, x, y, sy)
    minuit_chi2 = Minuit(chi2fit, *startparameters)
    minuit_chi2.errordef = 1.0     
    minuit_chi2.migrad()   
    
    'Parameters and uncertainties'
    par = minuit_chi2.values[:]
    par_err = minuit_chi2.errors[:] 
    par_name = minuit_chi2.parameters[:]
    chi2_value = minuit_chi2.fval           
    N_NotEmptyBin = np.sum(y > 0)
    Ndof_value = N_NotEmptyBin - minuit_chi2.nfit
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value) 
    
    'Printing Chi2 value, Ndof, prob, value and errors of fit parameters'
    if verbose == True:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3f}")
        
        for name in minuit_chi2.parameters:
            value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
            print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")

    'Plot histogram and fit'
    if plot == True:
        x_axis = np.linspace(xmin, xmax, 1000)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        hist_trans = ax.hist(X, bins=N_bins, range=(xmin, xmax), histtype='step', label='Histogram', density=True)
        ax.set(xlabel="x", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1))
        ax.plot(x_axis,fitfunction(x_axis,*minuit_chi2.values), label='Fitted function')
        
        for name in minuit_chi2.parameters:
            value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
            d = {'Chi2':     chi2_value,
                 'ndf':      Ndof_value,
                 'Prob':     Prob_value,
                 f'{name}':     [value, error],   
                 }
            
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(0.10, 0.95, text, ax, fontsize=14)
        ax.legend(loc='best')
        fig.tight_layout()  
    

# ======================================
# MONTE CARLO; ACCEPT AND REJECT 
# ======================================
def acceptandreject(func, xmin, xmax, N_points, N_bins):
    'Random number with fixed seed'
    r = np.random
    r.seed(42)
    
    'Generate random numbers within the fixed box'
    xaxis = np.linspace(xmin,xmax,N_points) 
    X_rnd = r.uniform(xmin, xmax, size=N_points)                       #random x values  
    Y_rnd = r.uniform(min(func(xaxis)),max(func(xaxis)),size=N_points) #random y values 
    
    'Accept and reject'
    fX = func(X_rnd)                    #fit function used on random x values
    fX_accepted = fX > Y_rnd            #condition for accept / reject
    X_accepted = X_rnd[fX_accepted]     #accepted X values
    eff = len(X_accepted) / len(X_rnd)  #efficiency 
    return X_accepted, eff

def acceptandrejectwpar(func, par, xmin, xmax, N_points, N_bins):
    'Random number with fixed seed'
    r = np.random
    r.seed(42)
    
    'Generate random numbers within the fixed box'
    xaxis = np.linspace(xmin,xmax,N_points) 
    X_rnd = r.uniform(xmin, xmax, size=N_points)                       #random x values  
    Y_rnd = r.uniform(min(func(xaxis, *par)),max(func(xaxis, *par)),size=N_points) #random y values 
    
    'Accept and reject'
    fX = func(X_rnd, *par)                    #fit function used on random x values
    fX_accepted = fX > Y_rnd            #condition for accept / reject
    X_accepted = X_rnd[fX_accepted]     #accepted X values
    eff = len(X_accepted) / len(X_rnd)  #efficiency 
    return X_accepted, eff



# ======================================
# COMPARE TWO HISTOGRAMS E.G. TRANSFORMATION AND ACCEPTANDREJECT 
# ======================================
def comparehistograms(hist1, hist2, verbose=False):
    'Terms in the chi2 sum'
    terms_in_sum = (hist_trans[0]-hist_accept[0])**2/(hist_trans[0]+hist_accept[0])
    
    'Chi2, Ndof and prob'
    chi2_value = sum(terms_in_sum)
    Ndof_value = len(terms_in_sum)
    Prob_value = stats.chi2.sf(chi_square,Ndof_value)
    
    'Printing Chi2 value, Ndof and prob value'
    if verbose == True:
        print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}   Prob(Chi2,Ndof) = {Prob_value:5.3f}")
    
    return chi2_value, Prob_value






# ======================================
# PLOTTING 
# ======================================
def plot_hist(X, xmin, xmax, N_points, N_bins):
    fig, ax = plt.subplots(figsize=(12, 6))
    hist_trans = ax.hist(X, bins=N_bins, range=(xmin, xmax), histtype='step', label='Histogram', density=True)
    ax.set(xlabel="x", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1))

    d = {'Entries': len(X),
         'Mean': X.mean(),
         'Std': X.std(ddof=1),
         }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax, fontsize=14)

    ax.legend(loc='best')
    fig.tight_layout()  


def plot_histandfunc(X, func, xmin, xmax, N_points, N_bins):
    '''Plot of histogram with fit with no parameters'''
    fig, ax = plt.subplots(figsize=(12, 6))
    hist_trans = ax.hist(X, bins=N_bins, range=(xmin, xmax), histtype='step', label='Histogram', density=True)
    ax.set(xlabel="x", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1))

    x_axis = np.linspace(xmin, xmax, 1000)
    y_axis = func(x_axis)
    ax.plot(x_axis, y_axis, 'r-', label='Plotted function')

    d = {'Entries': len(X),
         'Mean': X.mean(),
         'Std': X.std(ddof=1),
         }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax, fontsize=14)

    ax.legend(loc='best')
    fig.tight_layout()  
    
def plot_histandfuncwpar(X, func, par, xmin, xmax, N_points, N_bins):
    '''Plot of histogram with fit with parameters'''
    fig, ax = plt.subplots(figsize=(12, 6))
    hist_trans = ax.hist(X, bins=N_bins, range=(xmin, xmax), histtype='step', label='Histogram', density=True)
    ax.set(xlabel="x", ylabel="Frequency", xlim=(xmin-0.1, xmax+0.1))

    x_axis = np.linspace(xmin, xmax, 1000)
    y_axis = func(x_axis, *par)
    ax.plot(x_axis, y_axis, 'r-', label='Plotted function')

    d = {'Entries': len(X),
         'Mean': X.mean(),
         'Std': X.std(ddof=1),
         }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.05, 0.95, text, ax, fontsize=14)

    ax.legend(loc='best')
    fig.tight_layout()  





#JONATHANS
def ophob(f,variables,values,uncertainties, cov = True, ftype = 'Sympy', verbose  = False):
    from sympy.tensor.array import derive_by_array
    from numpy import identity, array, dot, matmul
    from latex2sympy2 import latex2sympy
    
    if ftype == 'LaTeX':
        f = latex2sympy(f)


    if cov == True:
        cov = identity(len(variables))

    subs_dict = dict(zip(variables, values))
    gradient = derive_by_array(f,variables).subs(subs_dict)
    
    VECTOR = array([element.evalf() for element in gradient])* uncertainties


    if verbose:
        from sympy.printing.latex import print_latex
        print(          '-- Value --' )
        print(f.subs(subs_dict).evalf())
        
        print('\n         -- python --         ')
        print(derive_by_array(f,variables))

        print('\n         -- LaTeX  --         ')
        print_latex(derive_by_array(f,variables)) 
        
        print("\n         -- (Gradient \cdot \sigma)^2 --")
        print(VECTOR**2)
        
        print("\n         -- Gradient^2 --")
        print((array([element.evalf() for element in gradient])**2))

        print('\n         -- variables  --         ')
        print(variables)


    return float(dot(VECTOR  , matmul(cov , VECTOR))**0.5)

















## NOT DONE WEEK 2 POISSONGAUSSBIOMINAL
def biominal_chisquare(data, N_experiment, N_trials, p):
    counts, bin_edges = np.histogram(data, bins=N_trials+1, range=(-0.5, N_trials+0.5)) 
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    
    x = bin_centers[counts>0]
    y = counts[counts>0]
    
    
    Ndof_bino = 0
    chi2_bino = 0.0  
    
    for N_obs, x_i in zip(y, x):
        N_exp = func_binomial(x_i, N_experiment, N_trials, p)
        if (N_obs > 0) :
            chi2_bino += (N_obs-N_experiment)**2 / N_obs
            Ndof_bino += 1         # We simply count the number of DoF (nothing to be subtracted given no parameters)

    # Also calculate Ndof and Prob:
    Prob_bino = stats.chi2.sf(chi2_bino, Ndof_bino)
    print(f"Binomial:   chi2 = {chi2_bino:.2f}   N_dof = {Ndof_bino:d}   Prob = {Prob_bino:6.4f}")
    
    
    
'''
    
    ### From external functions

class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2
        
'''
