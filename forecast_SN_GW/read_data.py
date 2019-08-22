import numpy as np
import pandas as pd
from astropy.io import fits

d_ps1 = pd.read_csv('../Data/SN_ps_snls_sdss_hst.csv', sep=' ', index_col='CID')
ps1_pd = d_ps1[d_ps1['IDSURVEY'] == 15]

d_X = pd.read_csv('../Data/jla_lcparams.txt', sep=' ', index_col='#name')
X = np.array([d_X['zcmb'].values, d_X['mb'].values, d_X['color'].values,
             d_X['x1'].values])

d_sigma = pd.read_csv('../Data/covmat/sigma_mu.txt', sep=' ')
S = np.array([d_sigma['#sigma_coh'].values, d_sigma['#sigma_lens'].values,
              d_sigma['#z'].values])

bias_data = fits.getdata('../Data/covmat/C_bias.fits')
cal_data = fits.getdata('../Data/covmat/C_cal.fits')
dust_data = fits.getdata('../Data/covmat/C_dust.fits')
host_data = fits.getdata('../Data/covmat/C_host.fits')
model_data = fits.getdata('../Data/covmat/C_model.fits')
nonia_data = fits.getdata('../Data/covmat/C_nonia.fits')
pecvel_data = fits.getdata('../Data/covmat/C_pecvel.fits')
stat_data = fits.getdata('../Data/covmat/C_stat.fits')

covX = bias_data + cal_data + dust_data + host_data + model_data + nonia_data + pecvel_data + stat_data
