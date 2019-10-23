#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:56:16 2018

@author: florian
"""

import numpy as np
from numpy.linalg import inv
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


def get_hubblefit(x, cov_x, zhl, zcmb,  sig_int, sig_lens,
                  fit_cosmo=True, sirens=1.,
                  PARAM_NAME=np.asarray(['alpha1', 'alpha2', "alpha3", "beta",
                                         "delta", "delta2", "delta3"])):
    """
    Parameters
    ----------
    x: type
        infor

    cov_x ....


    parameters: list of int to specify if you want to remove some parameters in
                the fit in PARAM_NAME.
                By default is None and this take all parameters. If you don't
                want to put a correction use parameters=[]
        example:
            for usual salt x1 c correction if you want only color correction,
            use parameters = [2].
            Warning if you have more than 5 parameters, add parameters in
            PARAM_NAME
    """

    n_corr = np.shape(x)[1]-1

    class hubble_fit_case(Hubble_fit):
        freeparameters = ["Mb"]+PARAM_NAME[:n_corr].tolist()

    h = hubble_fit_case(x, cov_x, zhl, zcmb,
                        sig_int, sig_lens, fit_cosmo=fit_cosmo,
                        sirens=sirens)
    return h


class Hubble_fit(object):
    """
    """

    def __new__(cls, *arg, **kwargs):
        """ Upgrade of the New function to enable the
        the _minuit_ black magic
        """
        obj = super(Hubble_fit, cls).__new__(cls)

        exec("@make_method(Hubble_fit)\n" +
             "def _minuit_chi2_(self,%s): \n" % (", ".join(obj.freeparameters)+', omgM, omgK, w') +
             "    parameters = %s \n" % (", ".join(obj.freeparameters)) +
             "    return self.get_chi2(parameters, omgM, omgK, w)\n")

        return obj

    def __init__(self, X, cov_X, zhl, zcmb, sig_int, sig_lens, fit_cosmo=True, guess=None, sirens=1.):
        self.variable = X
        self.cov = cov_X
        self.zcmb = zcmb
        self.zhl = zhl
        self.pecvel = (5 * 150 / 3e5) / (np.log(10.) *
                                         self.zcmb)  # adding peculiar velocity
        

        self.Mb_fact = sirens
        self.sig_int = sig_int
        self.sig_lens = sig_lens
        self.dof = len(X)-len(self.freeparameters)
        self.fit_cosmo = fit_cosmo

    def distance_modulus(self, params):
        """
        (mb + alpha * v1 + beta * v2 .....) - Mb
        """

        return np.sum(np.concatenate([[1], params[1:]]).T * self.variable,
                      axis=1) - params[0]*self.Mb_fact

    def get_chi2(self, params, omgM, omgK, w):
        """
        """
        self.Cmu = np.zeros_like(self.cov[::len(params)-1, ::len(params)-1])
        pcorr = np.concatenate([[1], params[1:-1]])
        for i, coef1 in enumerate(pcorr):
            for j, coef2 in enumerate(pcorr):
                self.Cmu += (coef1 * coef2) *self.cov[i::len(params)-1, j::len(params)-1]

        self.Cmu[np.diag_indices_from(
            self.Cmu)] += self.sig_int**2 + self.pecvel**2 + self.sig_lens**2
        self.C = inv(self.Cmu)
        self.distance_modulus_table = self.distance_modulus(params)
        L = self.distance_modulus_table - distance_modulus_th(self.zcmb, self.zhl, 
                                                              omgM=omgM, omgK=omgK, w=w)
        self.residuals = L
        self.var = np.diag(self.Cmu)
        return np.dot(L, np.dot(self.C, L))
    
     

    def setup_guesses(self, **kwargs):
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
        Void, defines param_input (and consquently paramguess, parambounds and
        paramfixed)
        """
        def _test_it_(k, info):
            param = k.split(info)[0]
            if param not in self.freeparameters:
                raise ValueError("Unknown parameter %s" % param)

        self.param_input = {}
        # -- what you hard coded
        for name in self.freeparameters:
            for info in ["_guess", "_fixed", "_boundaries"]:
                if hasattr(self, name+info):
                    self.param_input[name+info] = eval("self.%s" % (name+info))

        # -- Then, whatever you gave
        for k, v in kwargs.items():
            if "_guess" in k:
                _test_it_(k, "_guess")
            elif "_fixed" in k:
                _test_it_(k, "_fixed")
            elif "_boundaries" in k:
                _test_it_(k, "_boundaries")
            else:
                raise ValueError(
                    "I am not able to parse %s ; not _guess, _fixed nor _boundaries" % k)
            self.param_input[k] = v

        # -- Finally if no values have been set, let's do it
        for name in self.freeparameters:
            if name+"_guess" not in self.param_input.keys():
                self.param_input[name+"_guess"] = 0.
            if name+"_fixed" not in self.param_input.keys():
                self.param_input[name+"_fixed"] = False
            if name+"_boundaries" not in self.param_input.keys():
                self.param_input[name+"_boundaries"] = [None, None]

    def fit(self, fix_omgM=False,
        fix_omgK=True, fix_w=True, **kwargs):
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
        self.fix_omgM = fix_omgM
        self.fix_omgK = fix_omgK
        self.fix_w = fix_w
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
            minuit_kwargs[param] = self.param_input["%s_guess" % param]
            minuit_kwargs['Mb'] = -19.05
            minuit_kwargs['omgM'] = 0.3
            minuit_kwargs['omgK'] = 0.
            minuit_kwargs['w'] = -1.
            minuit_kwargs['limit_omgM'] =(0.,1.)
            minuit_kwargs['limit_omgK'] =(-1.,1.)
            if self.fit_cosmo == False:
                minuit_kwargs['fix_omgM'] = True
                minuit_kwargs['fix_omgK'] = True
                minuit_kwargs['fix_w'] = True
            else: 
                minuit_kwargs['fix_omgM'] = self.fix_omgM
                minuit_kwargs['fix_omgK'] = self.fix_omgK
                minuit_kwargs['fix_w'] = self.fix_w  
                
        self.minuit = minuit.Minuit(self._minuit_chi2_,
                                    pedantic=False,
                                    print_level=1,
                                    **minuit_kwargs)

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
        HD_results['modefit_values'] = self.step_fitvalues
        HD_results['sig_int'] = self.sig_int
        HD_results['data'] = {}
        for i, name in enumerate(self.sn_name):
            HD_results['data'][name] = {}
            HD_results['data'][name]['residuals'] = self.residuals[i]
            if self.lssfr is not None:
                HD_results['data'][name]['lssfr'] = self.lssfr[i]
                HD_results['data'][name]['proba'] = self.proba[i]
            HD_results['data'][name]['var'] = self.var[i]

            HD_results['data'][name]['mb'] = self.variable[i, 0]

            HD_results['data'][name]['mb_err'] = err[i *
                                                     len(self.variable[0, :])]

            for l in range(len(self.variable[0, :-1])):
                HD_results['data'][name]['param_' +
                                         str(l+1)] = self.variable[i, l+1]
                HD_results['data'][name]['param_'+str(l+1)+'_err'] = []
                HD_results['data'][name]['param_' + str(l+1)+'_err'] =\
                    err[i*len(self.variable[0, :])+l+1]

        pkl.dump(HD_results, open(outpath, 'w'))

    def compute_contour(self,varx,vary,nsigma=1,nbinX=7,quick=True,results='Hubblefit/results/'):
        '''
        Function which computes contours fromp a converged imnuit object
        inputs:
            - mobject: the converges minuit object
            - varX : the X variable for the contour. Should be included in mobject.parameters
            - varY : the Y variable for the contour. Should be included in mobject.parameters
            - quick : False is the real version of contour but it's take very long time to run
                    a quick version of contour (quick=True) can give a first idea of the contour
        returns:
            - contourline to be used with matplotlib.fill_between
        '''

        assert varx in self.minuit.parameters
        assert vary in self.minuit.parameters
        import copy
        import scipy.optimize as opt
        # prepare intermediate minuit results        
        chi2min = self.minuit.fval
        marg = copy.copy(self.minuit.fitarg)
        m = self.minuit
        #fix or unfix all parameters
        for p in self.minuit.parameters:
            marg['fix_'+p] = True
        
        #fix w when using omgK and fix omgK when using w
        if vary =='w':
            marg['fix_omgK'] = True
        elif vary=='omgK':
            marg['fix_w'] = True
        else:
            raise ValueError('Error: this function need w or omgK in vary')
    ##        
    #    #Fix alpha and beta accelerates this function      
    #    marg['fix_alpha']=True
    #    marg['fix_beta']=True
    ##    
    #    initialisation 
        yplus = list()
        yminus = list()
        xlist = list()
        xmin = m.values[varx]
    #
    #    
        # prepare functions to be zeroed
        if quick==False :    #real version of contour
            def f(y):
                # var x and var y should already be fixed
                # other should be left free
                # value of x is already set before looking for the 0 of f
                for p in self.minuit.parameters:
                    marg['fix_'+p] = False
                m.values[vary] = y
                marg[vary] = y
                marg['fix_omgK'] = True
                marg['fix_'+varx] = True
                marg['fix_w'] = True
                m2 = minuit.Minuit(self.minuit.fcn,**marg)
                m2.migrad()
                for p in self.minuit.parameters:
                    marg['fix_'+p] = True        
                # this part to be switched off if fast approach is used
                return m2.fcn(**m2.values) - chi2min - nsigma**2
            def g(x):    
                # var x and var y should already be fixed
                # other should be left free
                # value of x is already set before looking for the 0 of f
                for p in self.minuit.parameters:
                    marg['fix_'+p]= False
                m.values[varx] = x
                marg[varx] = x
                marg['fix_omgK'] = True
                marg['fix_'+varx] = True
                marg['fix_w'] = True
                m2=minuit.Minuit(self.minuit.fcn,**marg)
                m2.migrad()
                for p in self.minuit.parameters:
                    marg['fix_'+p] = True        
                # this part to be switched off if fast approach is used
                return m.fcn(**m2.values) - chi2min - nsigma**2
        else:
            def f(y):
                m.values[vary] = y
                return m.fcn(**m.values) - chi2min - nsigma**2
            def g(x):
                m.values[varx] = x
                return m.fcn(**m.values) - chi2min - nsigma**2
            
        #initiate the convergence to the ellipse extremum
        ydif = 1
    
    ##
        xplus = opt.brentq(g, xmin, xmin+m.errors[varx]*nsigma*2)
        xplusinit = xplus
    
        xminus = opt.brentq(g, xmin-m.errors[varx]*nsigma, xmin)
        xminusinit = xminus

    #
        t = 0
        #converge to the contour max
        while ydif >= 0.01 and t<=12 and marg[varx]>0.:
   
            marg['fix_'+varx] = True
            marg['fix_'+vary] = False
            marg[varx]=xplus       
            try:
                m = minuit.Minuit(self.minuit.fcn,**marg)
                m.migrad()        
                ymin = m.values[vary]
                yp = opt.brentq(f, ymin, ymin+m.errors[vary]*nsigma*2)
                ym = opt.brentq(f, ymin-m.errors[vary]*nsigma*2, ymin)
                xlist.append(xplus)
                yplus.append(yp)
                yminus.append(ym)
                ydif = yp-ym
                ymid = (yp+ym)/2
                marg[vary] = ymid
                marg['fix_'+varx] = False
                marg['fix_'+vary] = True
                m = minuit.Minuit(self.minuit.fcn,**marg)
                m.migrad()
                xmin = m.values[varx]
                xplus = opt.brentq(g, xmin, xmin+m.errors[varx]*nsigma*2)
                xminus = opt.brentq(g, xmin-m.errors[varx]*nsigma*2, xmin)
            except:
                marg[varx] = marg[varx]+0.01            
                t+= 1        
            t+= 1
    
        #if the first while don't converge to the ellypse max this one find the max
        #but warning this can be very slow 
        t =True
        xplus = xplus-0.1
        while ydif >= 0.01 and t==True and marg[varx]>0.:
            print t
            marg['fix_'+varx] = True
            marg['fix_'+vary] = False
            marg[varx] = xplus+0.001*nsigma
            m = minuit.Minuit(self.minuit.fcn,**marg)
            m.migrad()        
            ymin = m.values[vary]
    
            # *2 : safety margin
            try:
                yp = opt.brentq(f, ymin, ymin+m.errors[vary]*nsigma*2)
                ym = opt.brentq(f, ymin-m.errors[vary]*nsigma*2, ymin)   
                xlist.append(marg[varx])
                xplus = marg[varx]        
                yplus.append(yp)
                yminus.append(ym)
                ydif = yp-ym

            except:
                t=False
    
    
        
        #initialisation of the next while
        xplus = xplusinit
        xminus = xminusinit
        ydif = 1
        t = 0
       
    #   converge to the contour min
        while ydif >= 0.01 and t<=12 and xminus >= -0.05 and marg[varx]>0.:
            marg['fix_'+varx] = True
            marg['fix_'+vary] = False
            marg[varx] = xminus 

            try:
                # *2 : safety margin
                m = minuit.Minuit(self.minuit.fcn,**marg)
                m.migrad()        
                ymin=m.values[vary]
                yp=opt.brentq(f, ymin, ymin+m.errors[vary]*nsigma*2)
                ym=opt.brentq(f, ymin-m.errors[vary]*nsigma*2, ymin)
                xlist.append(xminus)
                yplus.append(yp)
                yminus.append(ym)
                ydif = yp-ym
                ymid = (yp+ym)/2
                marg[vary] = ymid
                marg['fix_'+varx] = False
                marg['fix_'+vary] = True
                m = minuit.Minuit(self.minuit.fcn,**marg)
                m.migrad()
                xmin = m.values[varx]
                xplus = opt.brentq(g, xmin, xmin+m.errors[varx]*nsigma*2)
                xminus = opt.brentq(g, xmin-m.errors[varx]*nsigma*2, xmin)  
            except:
                marg[varx] = marg[varx]+0.01            
                t+= 1          
            t+= 1

        #if the first while don't converge to the contour min this one find the min
        #but warnig this can be very slow 
        t=True
     
        while ydif >= 0.01 and t==True and xminus >= -0.05 and marg[varx]>0.:
            print t
            marg['fix_'+varx]=True
            marg['fix_'+vary]=False
            marg[varx] = xminus-0.001*nsigma
            m = minuit.Minuit(self.minuit.fcn,**marg)
            m.migrad()        
            ymin = m.values[vary]
    
            # *2 : safety margin
            try:
                yp = opt.brentq(f, ymin, ymin+m.errors[vary]*nsigma*2)
                ym = opt.brentq(f, ymin-m.errors[vary]*nsigma*2, ymin)   
                xlist.append(marg[varx])
                xminus = marg[varx]        
                yplus.append(yp)
                yminus.append(ym)
                ydif = yp-ym
            except:
                t=False        
    ##        
            
            
        #make more bins in t    
        xvals= np.linspace(self.minuit.values[varx]-nsigma*self.minuit.errors[varx],
                           self.minuit.values[varx]+self.minuit.errors[varx]*nsigma,
                           nbinX)
    #    xvals=N.linspace(0.1,0.3,nbinX)

#
#        for x in xvals:
#            #find y which realizes chi2 min
#            marg['fix_'+varx]=True
#            marg['fix_'+vary]=False    
#            marg[varx] = x
#            xlist.append(x)
#            m = minuit.Minuit(self.minuit.fcn,**marg)
#            m.migrad()
#            ymin  =m.values[vary]       
#            if f(ymin)<0:
#                # *2 : safety margin
#                yp = opt.brentq(f, ymin, ymin + m.errors[vary]*nsigma*2)
#                yplus.append(yp)
#                ym = opt.brentq(f, ymin-m.errors[vary]*nsigma*2, ymin)
#                yminus.append(ym)            
#            else:
#                yplus.append(np.nan)
#                yminus.append(np.nan)
        
        #transform the list in array
        xlist = np.array(xlist)
        yplus = np.array(yplus)
        yminus = np.array(yminus)
        
        #make order between the differents elements on the array    
        argso = np.argsort(xlist)
        xlist = xlist[argso]
        yplus = yplus[argso]
        yminus = yminus[argso]
        
        return xlist[np.isfinite(yplus)], yminus[np.isfinite(yplus)], yplus[np.isfinite(yplus)]
        