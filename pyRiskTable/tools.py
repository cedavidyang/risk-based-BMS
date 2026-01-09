# currently, there is no turn-key implementation for cascading hazard yet
# since it requires a defintiion of 2d hazard curve
# (i.e., conditional probability of secondary intensity)

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.interpolate as intp
 
 
def export_primary_scenarios(ims=None, scalers=None, likes=None, cqs=None,
                             hazard_curve=None, vec_func=False,
                             filepath=None):

    scenario_df = pd.DataFrame()
    scenario_df['Likelihood'] = likes
    scenario_df['Intensity'] = ims
    scenario_df['Consequences'] = cqs
    scenario_df['Weights'] = scalers

    if hazard_curve is not None:
        if vec_func:
            vfunc = hazard_curve
        else:
            vfunc = np.vectorize(hazard_curve)
            
        rates = vfunc(ims)
        scenario_df['Return Period'] = 1/rates
    
    if filepath is None:
        filepath = "tmp.csv"
    scenario_df.to_csv(filepath)


def risk_from_primary_scenarios(filepath=None, new_consequence=None):

    assert filepath is not None, "Must provide filepath"
    
    scenario_df = pd.read_csv(filepath)
    
    likes = scenario_df['Likelihood']

    if new_consequence is None:
        cqs = scenario_df['Consequences']
    else:
        cqs = new_consequence
    
    scalers = scenario_df['Weights']
    
    risk = np.sum(likes*cqs*scalers)

    return risk


def user_hazard_likelihood(
    filepath=None, im_key='Intensity', like_key='Likelihood',
    like_type='likelihood', method='linear', space='linear',
    diff_kwgs={}, intp_kwgs={},
):

    assert filepath is not None, "Must provide filepath"

    # check if filepath is a path or a dataframe
    if isinstance(filepath, str):
        hazard_df = pd.read_csv(filepath)
        im_arr, like_arr = hazard_df[im_key].values, hazard_df[like_key].values
    elif isinstance(filepath, pd.DataFrame):
        im_arr, like_arr = filepath[im_key].values, filepath[like_key].values
    else:
        raise ValueError(f"Invalid filepath={filepath}")

    # drop nan values
    im_arr, like_arr = im_arr[~np.isnan(like_arr)], like_arr[~np.isnan(like_arr)]
    
    if like_type == 'likelihood' and method == 'linear':
        func = lambda im: np.interp(im, im_arr, like_arr, **intp_kwgs)

    elif like_type == 'likelihood' and method == 'cubic_spline':
        res = intp.CubicSpline(im_arr, like_arr, **intp_kwgs)
        # take value at im
        func = lambda im: res(im)

    elif like_type == 'exceedence' and method == 'linear':
        # use numerical differentitation to compute likelihood
        exceed_arr = np.copy(like_arr)
        like_arr = np.gradient(exceed_arr, im_arr, **diff_kwgs)
        func = lambda im: -np.interp(im, im_arr, like_arr, **intp_kwgs)

    elif like_type == 'exceedence' and method == 'cubic_spline':
        exceed_arr = np.copy(like_arr)
        if space == 'linear':
            res = intp.CubicSpline(im_arr, exceed_arr, **intp_kwgs)
            # take the derivative of res
            func = lambda im: -res.derivative()(im)
        elif space == 'log':
            res = intp.CubicSpline(np.log(im_arr), np.log(exceed_arr), **intp_kwgs)
            # take the derivative of res and transform back to linear space
            def func(im):
                log_f = res(np.log(im))
                log_fp = res.derivative()(np.log(im))
                like = -np.exp(log_f) / im * log_fp
                return like
        else:
            raise ValueError(f"Invalid space={space}")
            
    else:
        raise ValueError(f"Invalid method={method} or like_type={like_type}")
    
    return func
    

def user_fragility_curve(filepath=None, im_key='IM', ds_key='Slight'):

    assert filepath is not None, "Must provide filepath"

    fragility_df = pd.read_csv(filepath)
    
    func = lambda im: np.interp(im, fragility_df[im_key], fragility_df[ds_key])
    
    return func


def fragility_curve(median, dispersion):

    func = lambda im: stats.lognorm.cdf(im, dispersion, scale=median)

    return func