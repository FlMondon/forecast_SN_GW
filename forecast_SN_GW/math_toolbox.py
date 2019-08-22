""" Math toolbox, that include all the usefull tools."""

import numpy as np

def comp_rms(residuals, dof, err=True, variance=None):
    """
    Compute the RMS or WRMS of a given distribution.

    :param 1D-array residuals: the residuals of the fit.
    :param int dof: the number of degree of freedom of the fit.
    :param bool err: return the error on the RMS (WRMS) if set to True.
    :param 1D-aray variance: variance of each point. If given,
                             return the weighted RMS (WRMS).

    :return: rms or rms, rms_err
    """
    if variance is None:                # RMS
        rms = float(np.sqrt(np.sum(residuals**2)/dof))
        rms_err = float(rms / np.sqrt(2*dof))
    else:                               # Weighted RMS
        assert len(residuals) == len(variance)
        rms = float(np.sqrt(np.sum((residuals**2)/variance) / np.sum(1./variance)))
        #rms_err = float(N.sqrt(1./N.sum(1./variance)))
        rms_err = np.sqrt(2.*len(residuals)) / (2*np.sum(1./variance)*rms)
    if err:
        return rms, rms_err
    else:
        return rms


    