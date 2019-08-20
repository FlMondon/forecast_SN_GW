#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:23:09 2018

@author: florian
"""
import numpy as np
from scipy import integrate

from .constant import CLIGHT, H0
# ------------------ #
#   Cosmology        #
# ------------------ #

def int_cosmo(z, Omega_M=0.3):   
    """
    """
    return 1./np.sqrt(Omega_M*(1+z)**3+(1.-Omega_M))
    
def luminosity_distance(zcmb, zhl):
    """
    """
    if type(zcmb)==np.ndarray:
        integr = np.zeros_like(zcmb)
        for i in range(len(zcmb)):
            integr[i]=integrate.quad(int_cosmo, 0, zcmb[i])[0]
    else:
        integr = integrate.quad(int_cosmo, 0, zcmb)[0]

    return (1+zhl)*(CLIGHT/H0)*integr
 
def distance_modulus_th(zcmb, zhl):
    """
    """
    return 5.*np.log(luminosity_distance(zcmb, zhl))/np.log(10.)-5.