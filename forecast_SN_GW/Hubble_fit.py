#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:56:16 2018

@author: florian
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import inv
from scipy import optimize
try:
   import cPickle as pkl
except ModuleNotFoundError:
   import pickle as pkl
import iminuit as minuit
from .cosmo_tools import distance_modulus_th
from .math_toolbox import comp_rms


def make_method(obj):
    """Decorator to make the function a method of *obj*.
    In the current context::
      @make_method(Axes)
      def toto(ax, ...):
          ...
    makes *toto* a method of `Axes`, so that one can directly use::
      ax.toto()
    COPYRIGHT: from Yannick Copin
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f
    return decorate


def get_hubblefit(x, cov_x, zhl, zcmb, sig_z,  sig_int, sig_lens, PARAM_NAME=np.asarray(['alpha1','alpha2',"alpha3","beta","delta", "delta2", "delta3"])):
    """
    Parameters
    ----------
    x: type
        infor
    
    cov_x ....
    
    
    parameters: list of int to specify if you want to remove some parameters in the fit in PARAM_NAME. 
                By default is None and this take all parameters. If you don't want to put a correction use 
                parameters=[]
        example: 
            for usual salt x1 c correction if you want only color correction, use parameters = [2]. 
            Warning if you have more than 5 parameters, add parameters in PARAM_NAME
    """

    n_corr =  np.shape(x)[1]-1 
    class hubble_fit_case(Hubble_fit):
        freeparameters = ["Mb"]+PARAM_NAME[:n_corr].tolist()
        


    h = hubble_fit_case(x, cov_x, zhl, zcmb, sig_z,  sig_int, sig_lens)
    return h


class Hubble_fit(object):
    """
    """
    
    def __new__(cls,*arg,**kwargs):
        """ Upgrade of the New function to enable the
        the _minuit_ black magic
        """
        obj = super(Hubble_fit,cls).__new__(cls)
        
        exec ("@make_method(Hubble_fit)\n"+\
             "def _minuit_chi2_(self,%s): \n"%(", ".join(obj.freeparameters)+', omgM')+\
             "    parameters = %s \n"%(", ".join(obj.freeparameters))+\
             "    return self.get_chi2(parameters, omgM)\n")


        return obj
    

    def __init__(self, X, cov_X, zhl, zcmb, sig_z,  sig_int, sig_lens, guess=None):
        self.variable = X
        self.cov = cov_X
        self.zcmb = zcmb
        self.zhl = zhl
        self.dmz = (5*sig_z)/(np.log(10)*self.zcmb)  #adding peculiar velocity
        self.sig_int = sig_int
        self.sig_lens = sig_lens
        self.dof = len(X)-len(self.freeparameters)  


    def distance_modulus(self, params):
        """
        (mb + alpha * v1 + beta * v2 .....) - Mb
        """

        return  np.sum(np.concatenate([[1],params[1:]]).T * self.variable, axis=1) - params[0]     
    
    def get_chi2(self, params, omgM):
        """
        """
        self.Cmu = np.zeros_like(self.cov[::len(params),::len(params)])
        pcorr = np.concatenate([[1],params[1:]])

        for i, coef1 in enumerate(pcorr):
            for j, coef2 in enumerate(pcorr):
                self.Cmu += (coef1 * coef2) * self.cov[i::len(params)-1, j::len(params)-1] 
                
                
        self.Cmu[np.diag_indices_from(self.Cmu)] += self.sig_int**2 + self.dmz**2 + self.sig_lens**2
        self.C = inv(self.Cmu)
        self.distance_modulus_table =  self.distance_modulus(params)
        L = self.distance_modulus_table - distance_modulus_th(omgM, self.zcmb, self.zhl)
        self.residuals = L
        self.var = np.diag(self.Cmu)
        return np.dot(L, np.dot(self.C,L))        


    def setup_guesses(self,**kwargs):
        """ Defines the guesses, boundaries and fixed values
        that will be passed to the given model.
        For each variable `v` of the model (see freeparameters)
        the following array will be defined and set to param_input:
           * v_guess,
           * v_boundaries,
           * v_fixed.
        Three arrays (self.paramguess, self.parambounds,self.paramfixed)
        will be accessible that will point to the defined array.
        Parameter
        ---------
        **kwargs the v_guess, v_boundaries and, v_fixed for as many
        `v` (from the freeparameter list).
        All the non-given `v` values will be filled either by pre-existing
        values in the model or with: 0 for _guess, False for _fixed, and
        [None,None] for _boundaries
        Return
        ------
        Void, defines param_input (and consquently paramguess, parambounds and paramfixed)
        """
        def _test_it_(k,info):
            param = k.split(info)[0]
            if param not in self.freeparameters:
                raise ValueError("Unknown parameter %s"%param)

        self.param_input = {}
        # -- what you hard coded
        for name in self.freeparameters:
            for info in ["_guess","_fixed","_boundaries"]:
                if hasattr(self, name+info):
                    self.param_input[name+info] = eval("self.%s"%(name+info))
                    
        # -- Then, whatever you gave
        for k,v in kwargs.items():
            if "_guess" in k:
                _test_it_(k,"_guess")
            elif "_fixed" in k:
                _test_it_(k,"_fixed")
            elif "_boundaries" in k:
                _test_it_(k,"_boundaries")
            else:
                raise ValueError("I am not able to parse %s ; not _guess, _fixed nor _boundaries"%k)
            self.param_input[k] = v

        # -- Finally if no values have been set, let's do it
        for name in self.freeparameters:
            if name+"_guess" not in self.param_input.keys():
                self.param_input[name+"_guess"] = 0.
            if name+"_fixed" not in self.param_input.keys():
                self.param_input[name+"_fixed"] = False
            if name+"_boundaries" not in self.param_input.keys():
                self.param_input[name+"_boundaries"] = [None,None]
    
    def fit(self, **kwargs):
        """
        How to use kwargs 
        For each variable `v` of the model (see freeparameters)
        the following array will be defined and set to param_input:
           * v_guess,
           * v_boundaries,
           * v_fixed.
        Three arrays (self.paramguess, self.parambounds,self.paramfixed)
        will be accessible that will point to the defined array.
        Parameter
        ---------
        **kwargs the v_guess, v_boundaries and, v_fixed for as many
        `v` (from the freeparameter list).
        All the non-given `v` values will be filled either by pre-existing
        values in the model or with: 0 for _guess, False for _fixed, and
        [None,None] for _boundaries
        """
        self._loopcount = 0
        self.sig_int = 0.
        self.setup_guesses(**kwargs)
        
        self.first_iter = self._fit_minuit_()
        # - Final steps      
        return self._fit_readout_()

        
    def _setup_minuit_(self):
        """
        """
        # == Minuit Keys == #
        minuit_kwargs = {}
        for param in self.freeparameters:
            minuit_kwargs[param] = self.param_input["%s_guess"%param]
            

        self.minuit = minuit.Minuit(self._minuit_chi2_, **minuit_kwargs)
    
    def _fit_minuit_(self):
        """
        """
        self._setup_minuit_()
        self._migrad_output_ = self.minuit.migrad()
        
        if self._migrad_output_[0]["is_valid"] is False:
            print("migrad is not valid")
            
            
        self.resultsfit = np.asarray([self.minuit.values[k]
                              for k in self.freeparameters])
        self.chi2_per_dof = self.minuit.fval/self.dof
        
    def _fit_readout_(self):
        """ Computes the main numbers """
        return comp_rms(self.residuals, self.dof, err=True, variance=self.var) 
           
    def dump_pkl(self, outpath='../../lsfr_analysis/HD_results_sugar.pkl'):
        HD_results = {}
        err = np.sqrt(np.diag(self.cov))
        HD_results['minuit_results'] = self.minuit.values
        HD_results['minuit_results_errors'] = self.minuit.errors
        HD_results['modefit_values'] =  self.step_fitvalues
        HD_results['sig_int'] = self.sig_int
        HD_results['data'] = {}
        for i, name in enumerate (self.sn_name):
            HD_results['data'][name] = {}
            HD_results['data'][name]['residuals'] = self.residuals[i]
            if self.lssfr is not None:
                HD_results['data'][name]['lssfr'] = self.lssfr[i]
                HD_results['data'][name]['proba'] = self.proba[i]
            HD_results['data'][name]['var'] = self.var[i]
            
            HD_results['data'][name]['mb'] = self.variable[i,0]
           
            HD_results['data'][name]['mb_err'] = err[i*len(self.variable[0,:])]
            
            for l in range(len(self.variable[0,:-1])):
                HD_results['data'][name]['param_'+str(l+1)] = self.variable[i,l+1]
                HD_results['data'][name]['param_'+str(l+1)+'_err'] = []
                HD_results['data'][name]['param_'+str(l+1)+'_err'] = err[i*len(self.variable[0,:])+l+1]
               
            
                
        pkl.dump(HD_results, open(outpath, 'w'))
        
        
        
