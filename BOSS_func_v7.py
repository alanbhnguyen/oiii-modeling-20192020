"""
Gaussian models for BOSS seyferts

Author: Alan Nguyen

Date: 02-06-2020
"""

import numpy as np
from numpy.linalg import norm
# =============================================================================
# import matplotlib
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.table import Table, Column
# from scipy.optimize import curve_fit, leastsq
# from astropy.cosmology import WMAP9 as cosmo
# from astropy import units as u
# import os
# import math as m
# from astropy import constants as const
# import time
# =============================================================================

def to_angstrom_OIII(vel):
    gauss_center = 5007 * ( ( vel / 300000 ) + 1 )
    return gauss_center

def to_vel_OIII(angstrom):
    z = (angstrom - 5007) / 5007
    vel = 300000 * z
    return vel

def line_width_OIII(vel_sigma):
    angstrom_width = (vel_sigma * 0.007087) #5007 / 300000 / 2.355
    return angstrom_width  

def gaussian( x, params ):
    (amp, vel, vel_sigma) = params
    return amp * np.exp( - (x - to_angstrom_OIII(vel))**2.0 / (2.0 * (line_width_OIII(vel_sigma))**2.0) )

def linear_continuum(x, params ):
    (m, b) = params
    return m * x + b

def double_gaussian_lincont( x, params ):
    (amp1, vel1, vel_sigma1, amp2, vel2, vel_sigma2, slope, yint) = params
    return gaussian(x, [amp1, vel1, vel_sigma1]) + gaussian(x,  [amp2, vel2, vel_sigma2]) + linear_continuum(x, [slope, yint])

def double_gaussian_lincont_fit( params, wave, fluxden, error ):
    (amp1, mu1, vel_sigma1, amp2, mu2, vel_sigma2, slope, yint) = params
    fit = double_gaussian_lincont( wave, params )
    return (fit - fluxden) / error

def double_gaussian_constrain_postive( params, wave, fluxden, error ):
    (amp1, mu1, vel_sigma1, amp2, mu2, vel_sigma2, slope, yint) = params
    if np.any(params[0:6] < 0):
        return 2 * norm(params[0:6][params[0:6]<0]) * double_gaussian_lincont_fit( params, wave, fluxden, error )
    else:
        return double_gaussian_lincont_fit( params, wave, fluxden, error )

def single_gaussian_lincont( x, params ):
    (amp1, mu1, vel_sigma1, slope, yint) = params
    return gaussian(x, [amp1, mu1, vel_sigma1]) + linear_continuum(x, [slope, yint])

def single_gaussian_lincont_fit( params, wave, fluxden, error ):
    (amp1, mu1, vel_sigma1, slope, yint) = params
    fit = single_gaussian_lincont( wave, params )
    return (fit - fluxden) / error 
    
def amp_gauss(wave, fluxden):
    max_fluxden = np.max(fluxden)
    
    avg1 = np.mean(fluxden[0:10])
    avg2 = np.mean(fluxden[0:-10])
    
    avg = ((avg1 + avg2) / 2.0)
    
    amp_gauss = max_fluxden - avg
    
    return amp_gauss
    
def single_gauss_params(wave, fluxden, k): #(wavelength array, fluxden array, k = 1 + z)
    amp_OIII = amp_gauss(wave, fluxden)
    
    amp = 11 * amp_OIII / 12
    z = k - 1
    v = z * 300000
    vel_sigma = 141
    
    ####################

    m = (fluxden[-1] - fluxden[0]) / (wave[-1] - wave[0])
    b = (v * -wave[0]) + fluxden[0]
    
    return [amp, m, vel_sigma, m, b]

def identify_wing(wave, fluxden, k):
    max_fluxden = np.max(fluxden)
    
    wave_max = wave[fluxden == max_fluxden]
    
    center_index = np.where(wave == wave_max)[0][0]
    
    r = len(fluxden) - center_index
    
    blu_wave = wave[(center_index - r):center_index + 1]
    blu_fluxden = fluxden[(center_index - r):center_index + 1]
    
    red_wave = wave[center_index:(center_index + r)]
    red_fluxden = fluxden[center_index:(center_index + r)]
    
    blusum = np.sum(blu_fluxden)
    redsum = np.sum(red_fluxden)

    z = k - 1

    v_c = z * 300000    
    
    if blusum > 1.01 * redsum:
        v2 = v_c * (5002 / 5007)
    
    elif redsum > 1.01 * blusum:
        v2 = v_c * (5012 / 5007)
    
    else:
        v2 = v_c
    
    return v2
    
def double_gauss_params(wave, fluxden, k): #(wavelength array, fluxden array, k = 1 + z)
    amp_OIII = amp_gauss(wave, fluxden)
    
    amp_narrow = 11 * amp_OIII / 12
    
    z = k - 1
    v1 = z * 300000
    vel_sigma_narrow = 141

    amp_broad = amp_OIII / 4
    vel_sigma_broad = 300
    
    ####################
    
    v2 = identify_wing(wave, fluxden, k)
    
    m = (fluxden[-1] - fluxden[0]) / (wave[-1] - wave[0])
    b = (m * -wave[0]) + fluxden[0]
    
    return [amp_narrow, v1, vel_sigma_narrow, amp_broad, v2, vel_sigma_broad, m, b]

def wing_check(fit):
    
    wing_check1 = fit[2] #width
    wing_check2 = fit[5] #width
    
    comp1 = np.zeros(3)
    comp2 = np.zeros(3)

    if wing_check1 < wing_check2:
        comp2 = fit[3:6]
        comp1 = fit[0:3]

    elif wing_check1 > wing_check2:
        comp2 = fit[0:3]
        comp1 = fit[3:6]
    
    return [comp1, comp2] # core wing
    
def flag_spec(fit_parameters, wave, fluxden, error):
    
    ###############################
    ###############################
    
    #0 bad data, 1 no outflow, 2 blu wing, 3 red wing, 4 broad comp, no outflow
    
    ###############################
    ###############################
    
    FLAG = 0
    
    csq = np.sum(((fluxden - double_gaussian_lincont(wave, fit_parameters[0])) / error) ** 2)
    rcsq = csq / len(fluxden)
    
    ###############################
    ###############################
    
    components = wing_check(fit_parameters[0])
    
    
    C = components[0]
    W = components[1]
    
    if C[1] > W[1]:
        FLAG = 2
    elif C[1] < W[1]:
        FLAG = 3
        
    ###############################
    ###############################
        
    if fit_parameters[0][0] == 0: #second gaussian gives zero thus single
        FLAG = 1

    if fit_parameters[0][3] == 0: #second gaussian gives zero thus single
        FLAG = 1

    if fit_parameters[0][2] > 1900: #second gaussian un realistically wide thus single
        FLAG = 1

    if fit_parameters[0][5] > 1900: #second gaussian un realistically wide thus single
        FLAG = 1

    if fit_parameters[0][2] < 0: #negative width?? essentially flat
        FLAG = 1

    if fit_parameters[0][5] < 0: #negative width?? essentially flat
        FLAG = 1

    if fit_parameters[0][1] < min(to_vel_OIII(wave)): #fits second gauss elsewhere for somereason??
        FLAG = 1

    if fit_parameters[0][4] < min(to_vel_OIII(wave)): #fits second gauss elsewhere for somereason??
        FLAG = 1

    if fit_parameters[0][1] > max(to_vel_OIII(wave)): #fits second gauss elsewhere for somereason??
        FLAG = 1

    if fit_parameters[0][4] > max(to_vel_OIII(wave)): #fits second gauss elsewhere for somereason??
        FLAG = 1
        
    if fit_parameters[0][2] < -1900: #second gaussian un realistically wide thus single
        FLAG = 1

    if fit_parameters[0][5] < -1900: #second gaussian un realistically wide thus single
        FLAG = 1
        
    if fit_parameters[0][0] < 0: #negative amp bad data
        FLAG = 0

    if fit_parameters[0][3] < 0: #negative amp bad data
        FLAG = 0

    if rcsq > 20: #poor fit bad data
        FLAG = 0
        
    return FLAG


    


