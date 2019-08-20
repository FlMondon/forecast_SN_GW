#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:56:16 2018

@author: florian
"""

import sncosmo
from matplotlib import pyplot as plt
import copy
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import inv
from scipy import optimize

try:
   import cPickle as pkl
except ModuleNotFoundError:
   import pickle as pkl
import iminuit as minuit

from .builtins import register_SNf_bands_width, mag_sys_SNF_width,  builtins_jla_bandpasses, mag_sys_jla
from .load_sugar import register_SUGAR
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

class read_input_data_SNf(object):
    """
    """
    def __init__(self, res_dict_path='../../sugar_analysis_data/resfitlc_SNf_5new_fU_10new_fI_10_sugar.pkl', 
                 model_name='sugar', 
                 select=None, 
                 standard='mb',
                 step_data=None,
                 update=False,
                 modeldir='../../sugar_model/'):
        
        self.model_name = model_name
        try:
            dic =  pkl.load(open(res_dict_path))
        except:
            dic = pkl.load(open(res_dict_path,'rb'),encoding='latin1')
        self.dic_res = dic['data']
        self.t0_fix = dic['info']['t0 fix']
        self.select = select
        self.update = update
        self.modeldir = modeldir
        self.standard = standard
        self.step_data = step_data
        self.transformed = False
        if model_name == 'sugar' :
            if update: 
                if self.standard == 'mb' :
                    self.param_name = ['mb', 'q1', 'q2', 'q3', 'q4', 'Av']
                else: 
                    self.param_name = ['Mgr', 'q1', 'q2', 'q3', 'q4', 'Av']
            else:
                if self.standard == 'mb' :
                    self.param_name = ['mb', 'q1', 'q2', 'q3', 'Av']
                elif self.standard == 'Mgr':
                    self.param_name = ['Mgr', 'q1', 'q2', 'q3', 'Av']
                else:
                    self.param_name = ['Xgr', 'q1', 'q2', 'q3', 'Av']
                
        elif model_name == 'salt2':
            if self.standard == 'mb' :
                self.param_name = ['mb', 'x1', 'c']
            elif self.standard == 'log10_x0' :
                self.param_name = ['log10_x0', 'x1', 'c']
            else: 
                self.param_name = ['x0', 'x1', 'c']
        else:
            raise ValueError('Model name have to be salt2 or sugar')
            
    def delete_fit_fail(self, dic):
        """
        """
        self.fit_fail = []
        dic_del = {}
        for sn_name in dic.keys():
            if dic[sn_name]['res'] != 'fit fail' :       
#                if dic[sn_name]['res']['success'] == False:
#                    print sn_name
                if type(dic[sn_name]['res']['covariance']) == np.ndarray:
                     dic_del[sn_name] = dic[sn_name]
                else:
                    self.fit_fail.append(sn_name)
            else:
                self.fit_fail.append(sn_name)
        return dic_del 
    
    def selection(self):
        
        if self.select is not None:
            dic = {}
            sn_out_selec = []
            for sn_name in self.select:
                if sn_name in self.dic_res.keys():
                    dic[sn_name] = self.dic_res[sn_name]
                else:
                    sn_out_selec.append(sn_name)
            self.sn_out_selec = sn_out_selec
            self.dic_res = dic
                   
    def trans_data_form(self):
        self.dic_res = self.delete_fit_fail(self.dic_res)
        self.selection()
        list_param = []
        list_cov = []
        list_zhl = []
        list_zcmb = []
        list_zerr = []
        list_sn_name = []
        for sn_name in self.dic_res.keys():
            
            list_param.append(list(self.dic_res[sn_name]['res']['parameters'][2:-2]))
            if self.t0_fix:
                list_cov.append(self.dic_res[sn_name]['res']['covariance'])
            else:
                list_cov.append(np.delete(np.delete(self.dic_res[sn_name]['res']['covariance'], 0, 0),0,1))
            list_zhl.append(self.dic_res[sn_name]['zhl'])
            list_sn_name.append(sn_name)
            try:
                list_zcmb.append(self.dic_res[sn_name]['zcmb'])
                list_zerr.append(self.dic_res[sn_name]['zerr'])
            except:
                list_zcmb.append(None)
                list_zerr.append(None)
        self.zhl = np.array(list_zhl)
        self.zcmb = np.array(list_zcmb)
        self.zerr = np.array(list_zerr)
        self.params = np.array(list_param)
        self.cov = np.array(list_cov)
        self.sn_name = np.array(list_sn_name)
        self.transformed = True
        
    def Xgr_to_Mgr(self):
        for j in range(len(self.cov[:,0])):
            self.cov[j,0,:] = self.cov[j,0,:]* (1./(-2.5*np.log(10)*self.params[j,0]))
            self.cov[j,:,0] = self.cov[j,:,0]* (1./(-2.5*np.log(10)*self.params[j,0]))
        self.params[:,0] = -2.5*np.log10(self.params[:,0])

    def  Xgr_to_mb(self):
        builtins_jla_bandpasses()
        mag_sys_jla()
        # Calculation of m_b
        register_SUGAR(modeldir=self.modeldir)
        source = sncosmo.get_source('sugar')
        model = sncosmo.Model(source=source)
        self.mb = np.zeros(len(self.dic_res.keys()))
        cov_copy = copy.deepcopy(self.cov)
        for i, sn_name in enumerate (self.dic_res.keys()):
            parameters = self.dic_res[sn_name]['res']['parameters']
            h = 10**-9
            if self.update:
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                       q2=parameters[4], q3=parameters[5], q4=parameters[6], A=parameters[7])
                self.mb[i] = model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) 
                
                #dmbxgr
                model.set(t0=parameters[1], Xgr=(parameters[2]+h*10**-15), q1=parameters[3],    #10**-15 oder scale of Xgr
                           q2=parameters[4], q3=parameters[5], q4=parameters[6], A=parameters[7])
                dmbxgr = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbq1
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3]+h,
                           q2=parameters[4], q3=parameters[5], q4=parameters[6], A=parameters[7])
                dmbq1 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbq2
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4]+h, q3=parameters[5], q4=parameters[6], A=parameters[7])
                dmbq2 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbq3
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4], q3=parameters[5]+h, q4=parameters[6], A=parameters[7])
                dmbq3 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                
                #dmb4
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4], q3=parameters[5], q4=parameters[6]+h, A=parameters[7])
                dmbq4 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h                                      
                #dmbA
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4], q3=parameters[5], q4=parameters[6], A=parameters[7]+h)            
                dmbA = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                dray = np.array([dmbxgr, dmbq1, dmbq2, dmbq3, dmbq4, dmbA])
                if sum(self.params[:,2]) != 0. and   sum(self.params[:,3]) != 0. :
                    dray = np.array([dmbxgr, dmbq1, dmbq2, dmbq3, dmbA])
                elif sum(self.params[:,2]) != 0. :
                    dray = np.array([dmbxgr, dmbq1, dmbq2, dmbA])
                elif sum(self.params[:,3]) != 0. :
                    dray = np.array([dmbxgr, dmbq1, dmbq3, dmbA])      
                else :
                    dray = np.array([dmbxgr, dmbq1,  dmbA])


            else:
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                       q2=parameters[4], q3=parameters[5], A=parameters[6])
                self.mb[i] = model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) 
                #dmbxgr
                model.set(t0=parameters[1], Xgr=(parameters[2]+h*10**-15), q1=parameters[3],    #10**-15 oder scale of Xgr
                           q2=parameters[4], q3=parameters[5], A=parameters[6])
                dmbxgr = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbq1
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3]+h,
                           q2=parameters[4], q3=parameters[5], A=parameters[6])
                dmbq1 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbq2
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4]+h, q3=parameters[5], A=parameters[6])
                dmbq2 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbq3
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4], q3=parameters[5]+h, A=parameters[6])
                dmbq3 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                #dmbA
                model.set(t0=parameters[1], Xgr=parameters[2], q1=parameters[3],
                           q2=parameters[4], q3=parameters[5], A=parameters[6]+h)            
                dmbA = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
                if sum(self.params[:,2]) != 0. and   sum(self.params[:,3]) != 0. :
                    dray = np.array([dmbxgr, dmbq1, dmbq2, dmbq3, dmbA])
                elif sum(self.params[:,2]) != 0. :
                    dray = np.array([dmbxgr, dmbq1, dmbq2, dmbA])
                elif sum(self.params[:,3]) != 0. :
                    dray = np.array([dmbxgr, dmbq1, dmbq3, dmbA])      
                else :
                    dray = np.array([dmbxgr, dmbq1,  dmbA])

#            for j in range(len(dray)):
#                if j ==0:
#                    self.cov[i,0,j] = np.dot(np.dot(dray.T,cov_copy[i]),dray)
#                else:
#                    self.cov[i,0,j] = np.dot(cov_copy[i,j,:],dray)
#                    self.cov[i,j,0] = self.cov[i,0,j]
#
#        if sum(self.params[:,3]) == 0. and sum(self.params[:,2]) == 0.:
#            self.params = np.delete(self.params, 3,1)
#            self.params = np.delete(self.params, 2,1)
#        elif  sum(self.params[:,3]) == 0.:
#            self.params = np.delete(self.params, 3,1)        
#        elif  sum(self.params[:,2]) == 0.:
#            self.params = np.delete(self.params, 2,1)           
        self.params[:,0] = self.mb

    def x0_to_mb(self):
        builtins_jla_bandpasses()
        mag_sys_jla()
        # Calculation of m_b
        source = sncosmo.get_source('salt2', version='2.4')
        model = sncosmo.Model(source=source)
        self.mb = np.zeros(len(self.dic_res.keys()))
        cov_copy = copy.deepcopy(self.cov)        
        for i, sn_name in enumerate (self.dic_res.keys()):
            parameters = self.dic_res[sn_name]['res']['parameters']
            model.set(t0=parameters[1], x0=parameters[2], x1=parameters[3],
                       c=parameters[4])
            self.mb[i] = model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) 
            h = 10**-9
            model.set(t0=parameters[1], x0=parameters[2]+h, x1=parameters[3],
                       c=parameters[4])         
            dmbx0 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h
            model.set(t0=parameters[1], x0=parameters[2], x1=parameters[3]+h,
                       c=parameters[4])         
            dmbx1 = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h            
            model.set(t0=parameters[1], x0=parameters[2], x1=parameters[3],
                       c=parameters[4]+h)         
            dmbc = (model.bandmag('jla_STANDARD::B', 'jla_VEGA2_mb', [parameters[1]]) - self.mb[i])/h  
            dray = np.array([dmbx0, dmbx1,dmbc])
            for j in range(len(parameters[2:-2])):
                if j ==0:
                    self.cov[i,0,j] = np.dot(np.dot(dray.T,cov_copy[i]),dray)
                else:
                    self.cov[i,0,j] = np.dot(cov_copy[i,j,:],dray)
                    self.cov[i,j,0] = self.cov[i,0,j]
        
        self.params[:,0] = self.mb

    def build_host(self, global_mass='../../lsfr_analysis/lssfr_paper_full_sntable.csv'):
        if self.transformed == False:
            self.trans_data_form()
        self.host = np.loadtxt(global_mass,skiprows=1,delimiter=',', dtype='str')
        self.gmass = self.host[:,4].astype(float)
        self.lssfr = self.host[:,10].astype(float)
        self.new_sn_name = self.host[:,16]
        self.P_highmass = self.host[:,17].astype(float)
        self.P_Prompt = self.host[:,18].astype(float) # (1 - p_delayed I guess     

        Filtre =  np.array([True]*len(self.sn_name))
        for i, sn_name in enumerate(self.sn_name): 
            if sn_name not in self.new_sn_name:
                Filtre[i] = False
            if self.zhl[i] < 0.01:
                Filtre[i] = False
            if abs(self.params[i,1])>15:
                Filtre[i] = False
                
        self.cov_save, self.cov = self.cov_save[Filtre], self.cov_save[Filtre]
        self.params = self.params[Filtre]
        self.zcmb = self.zcmb[Filtre]
        self.zhl = self.zhl[Filtre]
        self.zerr = self.zerr[Filtre]
        self.sn_name = np.array(self.sn_name)[Filtre]
        
        Filtre =  np.array([True]*len(self.new_sn_name))
        for j, new_sn_name in enumerate(self.new_sn_name): 
            if new_sn_name not in self.sn_name:
                Filtre[j] = False

        self.host = self.host[Filtre]
        self.gmass = self.gmass[Filtre]
        self.lssfr = self.lssfr[Filtre]
        self.P_highmass = self.P_highmass[Filtre]
        self.P_Prompt = self.P_Prompt[Filtre]   
        self.new_sn_name = self.new_sn_name[Filtre]
                                            
                                            
        argsort = np.array([1]*len(self.new_sn_name))
        for j, new_sn_name in enumerate(self.new_sn_name): 
            for i , sn_name in enumerate(self.sn_name):
                if sn_name == new_sn_name:
                    argsort[i] = j     
                    

        self.host = self.host[argsort]
        self.gmass = self.gmass[argsort]
        self.lssfr = self.lssfr[argsort]
        self.P_highmass = self.P_highmass[argsort]
        self.P_Prompt = self.P_Prompt[argsort]
        self.P_delayed = 1 - self.P_Prompt

        if self.step_data is not None:
            if self.step_data == 'lssfr':
                self.params= np.column_stack((self.params, self.P_delayed))
            elif self.step_data == 'mass':
                self.params= np.column_stack((self.params, self.P_highmass))
            elif self.step_data == 'lssfr+mass':
                self.params= np.column_stack((self.params, self.P_delayed))
                self.params= np.column_stack((self.params, self.P_highmass))
                self.cov_fin = []
                for i in range(len(self.cov)):
                    cov_int= np.column_stack((self.cov[i,:,:], np.zeros(len(self.cov[i,:,:]))))
                    self.cov_fin.append(np.vstack((cov_int, np.zeros(len(self.cov[i,:,:])+1))))
                self.cov = np.array(self.cov_fin)  
            else:
                raise ValueError('step_data have to be None, mass, lssfr or lssfr+mass')
                
            self.cov_fin = []
            for i in range(len(self.cov)):
                cov_int= np.column_stack((self.cov[i,:,:], np.zeros(len(self.cov[i,:,:]))))
                self.cov_fin.append(np.vstack((cov_int, np.zeros(len(self.cov[i,:,:])+1))))
            self.cov = self.cov_fin
            self.cov = list(self.cov)
            self.cov = block_diag(*self.cov)
        else:
            self.cov = list(self.cov)
            self.cov = block_diag(*self.cov)
                        
    def build_HD_data(self):
        if self.transformed == False:
            self.trans_data_form()
        if self.model_name=='sugar':
            if self.standard == 'Mgr':
                self.Xgr_to_Mgr()
            elif self.standard == 'mb':
                self.Xgr_to_mb()
            elif self.standard is not 'Xgr':
                 raise ValueError( 'Warning: with sugar standard have to be Mgr, mb or Xgr')
        if self.model_name=='salt2':
            if self.standard == 'mb':
                self.x0_to_mb()
            elif self.standard == 'log10_x0' :
                self.params[:,0] = -2.5*np.log10(self.params[:,0])
                for i in range(len(self.sn_name)):
                        self.cov[i,0,0] = self.cov[i,0,0]*(1.0857362047581294 /self.params[i,0])**2
                        self.cov[i,0,1] = self.cov[i,0,1]*(1.0857362047581294 /self.params[i,0])
                        self.cov[i,0,2] = self.cov[i,0,2]*(1.0857362047581294 /self.params[i,0])
                        self.cov[i,:,0] = self.cov[i,0,:]
            elif self.standard is not  'x0':
                raise ValueError( 'Warning: with sugar standard have to be mb, log10_x0 or x0')
                
        self.cov_save = np.array(self.cov) 
        self.cov_list = list(self.cov)
        self.cov = block_diag(*self.cov_list)
        

    def write_data_yaml(self, outfile = '../../lsfr_analysis/meta_sugar_flo.yaml'):
        import yaml
        dico = {'params_'+self.model_name :self.params,
                'cov_params' : self.cov_pf,
                'z_cmb' : self.zcmb,
                'sn_name' : self.sn_name,
                'z_helio' : self.zhl}
        yaml.dump(dico, open(outfile,'w'))

def generate_fake_data(N=200, slopes=None, stds=None, stds_err=None, sigma_int=None, step=None, SEED=42):
    """
    En chantier (Work in progress)
    """
    np.random.seed(SEED)
    # mu = m - M
    Mb = -19.3
    noise_level = 0.05
    z = np.random.uniform(0.01,0.1,size=N)
    mb = distance_modulus_th(z, z) + Mb

    if slopes is not None:
        data = np.zeros((len(mb), len(slopes)+1))
        for i in range(len(slopes)):
            data[:,i+1] = np.random.normal(scale=stds[i], size=N)
            mb += slopes[i] * data[:,i+1]

        
    if stds_err is not None:
        data_cov = np.zeros((len(mb), len(slopes)+1, len(slopes)+1))
        for i in range(len(slopes)):
            data[:,i+1] += np.random.normal(scale=stds_err[i], size=N)
            data_cov[:,i+1,i+1] = stds_err[i]**2
            
    if sigma_int is not None:
        mb += np.random.normal(scale=sigma_int,size=N)

    if step is not None: 
        mass = np.random.uniform(5,15,size=N)
        proba = np.zeros(N)
        Filtre = mass > 10.
        mb[Filtre] += step
        proba[Filtre] = 1.

    noise = np.random.normal(scale=0.05,size=N)
    mb += noise
    mb_err = np.ones_like(mb) * noise_level
    if slopes is not None:
        data[:,0] = mb
        data_cov[:,0,0] = mb_err**2
        data_cov = block_diag(*data_cov)
        return  data, data_cov, z
    else:
        return z, mb, mb_err          


def get_hubblefit(x, cov_x, zhl, zcmb, zerr, PARAM_NAME=np.asarray(['alpha1','alpha2',"alpha3","beta","delta", "delta2", "delta3"]), lssfr=None, sn_name=None):
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
        

    h = hubble_fit_case(x, cov_x, zhl, zcmb, zerr, lssfr=lssfr, sn_name=sn_name)
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
             "def _minuit_chi2_(self,%s): \n"%(", ".join(obj.freeparameters))+\
             "    parameters = %s \n"%(", ".join(obj.freeparameters))+\
             "    return self.get_chi2(parameters)\n")


        return obj
    

    def __init__(self, X, cov_X, zhl, zcmb, zerr,  guess=None, lssfr=None, sn_name=None):
        self.variable = X
        self.cov = cov_X
        self.zcmb = zcmb
        self.zhl = zhl
        self.zerr = zerr
        self.dmz = 5/np.log(10) * np.sqrt(self.zerr**2 + 0.001**2) / self.zcmb #adding peculiar velocity
        self.sn_name = sn_name
        self.dof = len(X)-len(self.freeparameters)  

        if lssfr is not None:
            self.lssfr = lssfr[:,0]
            self.proba = lssfr[:,1]
        else:
            self.lssfr = None
            self.proba = None
    

    def distance_modulus(self, params):
        """
        (mb + alpha * v1 + beta * v2 .....) - Mb
        """

        return  np.sum(np.concatenate([[1],params[1:]]).T * self.variable, axis=1) - params[0]     
    
    def get_chi2(self, params):
        """
        """
        self.Cmu = np.zeros_like(self.cov[::len(params),::len(params)])
        pcorr = np.concatenate([[1],params[1:]])

        for i, coef1 in enumerate(pcorr):
            for j, coef2 in enumerate(pcorr):
                self.Cmu += (coef1 * coef2) * self.cov[i::len(params), j::len(params)] 
                
                
        self.Cmu[np.diag_indices_from(self.Cmu)] += self.sig_int**2 + self.dmz**2 
        self.C = inv(self.Cmu)
        self.distance_modulus_table =  self.distance_modulus(params)
        L = self.distance_modulus_table - distance_modulus_th(self.zcmb, self.zhl)
        self.residuals = L
        self.var = np.diag(self.Cmu)
        return np.dot(L, np.dot(self.C,L))        


    def fit_intrinsic(self, intrinsic_guess=0.1):
        """ Get the most optimal intrinsic dispersion given the current fitted standardization parameters. 
        
        The optimal intrinsic magnitude dispersion is the value that has to be added in quadrature to 
        the magnitude errors such that the chi2/dof is 1.
        Returns
        -------
        float (intrinsic magnitude dispersion)
        """
        def get_intrinsic_chi2dof(intrinsic):
            self.sig_int = intrinsic
            return np.abs(self.get_chi2(self.resultsfit) / self.dof -1)
        
        return optimize.fmin(get_intrinsic_chi2dof,
                             intrinsic_guess)

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
    
    def fit(self, fit_intrinsic=True, **kwargs):
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
        # - Intrinsic disposerion Fit?
        if fit_intrinsic:
            while (np.abs(self.chi2_per_dof - 1) > 0.001 and self._loopcount < 30):
                self.sig_int =  self.fit_intrinsic(np.sqrt(np.mean(self.var))*2. / self.chi2_per_dof)
                self._fit_minuit_()
                self._loopcount += 1
                
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
        return comp_rms(self.residuals, self.dof, err=True, variance=self.var) , self.sig_int  
    
    def comp_mass_step_modefit(self, xcut=-10.8, xlabel='$\log(lsSFR)$', PRINT_WRMS=False, model_name=None, fig_path='../../lsfr_analysis/lsfr_') :
        import modefit
        if (self.proba == self.variable[:,-1]).all():
            raise ValueError('Step done in the gloabal fit')
        if model_name == None:
            raise ValueError('model_name have to be salt2 or sugar')
        self.fit(Mb_guess = -19.05)
        print ('chi2', self.minuit.fval, 'dof ', self.dof)
        print ('chi2 per dof ', self.chi2_per_dof )
        step = modefit.stepfit(self.lssfr, self.residuals, np.sqrt(self.var-0.1**2),
                               proba=self.proba, dx=None, xcut=xcut, masknan=True, names=None)
        step.fit()
        FIG = plt.figure(figsize=(9,6))
        PLOT = step.show(figure= FIG)
        ax = PLOT['ax'][0]
        ax.set_ylim(-0.6,0.6)
 

        ylabel='$\Delta \mu$ '+ model_name
        ax.set_ylabel(ylabel,fontsize=16)
        ax.set_xlabel(xlabel,fontsize=16)
        PLOT['fig'].subplots_adjust(top=0.97,right=0.99)
        print ('sig int ', self.sig_int)
        import sugar_training as st
        wrms, wrms_err = st.comp_rms(self.residuals, 1, variance=self.var)

        if PRINT_WRMS:
            ax.set_title('$wRMS = (%.2f \pm %.2f)$ mag, $\Delta_Y=(%.2f \pm %.2f)$ mag'%((wrms, wrms_err,
                                                                                          abs(step.modelstep[0]), 
                                                                                          step.modelstep[1])),
                         fontsize=14)
        else:
            ax.set_title('$\Delta_Y=(%.2f \pm %.2f)$ mag'%((abs(step.modelstep[0]), 
                                                            step.modelstep[1])),
                         fontsize=18)

        print (model_name+' step: ', step.modelstep)
        plt.savefig(fig_path+model_name+'.png')
        self.step_fitvalues = step.fitvalues


       
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
        
        
        