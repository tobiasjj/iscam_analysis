#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# ISCAM Analysis, provide functions to analyse and plot iSCAM data
# Copyright 2019,2020 Tobias Jachowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from .iscam_analysis import *
from scipy import stats
from scipy.special import fdtrc


def get_rad52_fracs(fit_params, verbose=False):
    fracs = []
    amp_idx = [0, -2, -1]
    amp_fct = [1, 11, 22]
    for idx, fct in zip(amp_idx, amp_fct):
        fracs.append(fit_params['areas'][idx] * fct)
    fracs = np.array(fracs)
    fracs = fracs / fracs.sum(axis=0)

    if verbose:
        print('  M,R,D:  {:.2f}±{:.2f},   {:.2f}±{:.2f},   {:.2f}±{:.2f}'.format(
                        *fracs.flatten()))
    return fracs


def get_weights(errors):
    e = errors
    w = 1 / e**2
    return w


def get_weighted_means(values, errors):
    v = values
    w = get_weights(errors)
    return np.sum(w*v, axis=0) / np.sum(w, axis=0)


def get_weighted_variances_of_weighted_means(values, errors, unbiased=True):
    """
    Calculate the un-/biased weighted estimator of the sample variance
    """
    # see https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
    # and https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    v = values
    e = errors
    w = get_weights(e)
    v_w = get_weighted_means(v, e)

    v_w_sum = np.sum(w*(v - v_w)**2, axis=0)
    w_sum = np.sum(w, axis=0)

    if unbiased:
        return w_sum / (w_sum**2 - np.sum(w**2, axis=0)) * v_w_sum
    else:
        return 1/w_sum * v_w_sum


def get_weighted_values(values, errors, verbose=False):
    v = values
    e = errors
    w = get_weights(e)

    # Calculate weighted mean and weighted SEM of values
    w_sum = np.sum(w, axis=0)  # weighted 1 / SEM²
    v_w = np.sum(v*w, axis=0) / w_sum  # weighted mean
    e_w = np.sqrt(1 / w_sum)  # weighted SEM
    if verbose:
        print('Weighted means and errors:')
        print(np.array([v_w, e_w]).T)
    return v_w, e_w


def welch_anova_np(args, var_equal=False):
    """
    args : array like of array likes
        A list of groups (lists of floats) which should be compared.
    var_equal : boolean
        The groups share a common variance.
    """
    # Define Welch's ANOVA, which is robust against unequal variances
    # see https://statisticsbyjim.com/anova/welchs-anova-compared-to-classic-one-way-anova/
    # https://stackoverflow.com/questions/50964427/welchs-anova-in-python
    # https://github.com/scipy/scipy/issues/11122
    args = [np.asarray(arg, dtype=float) for arg in args]
    k = len(args)
    ni = np.array([len(arg) for arg in args])
    mi = np.array([np.mean(arg) for arg in args])
    vi = np.array([np.var(arg, ddof=1) for arg in args])
    wi = ni/vi

    tmp = np.sum((1 - wi / np.sum(wi))**2 / (ni - 1))
    tmp /= (k**2 - 1)

    dfbn = k - 1
    dfwn = 1 / (3 * tmp)

    m = np.sum(mi*wi) / np.sum(wi)
    f = np.sum(wi * (mi - m)**2) / (dfbn * (1 + 2 * (dfbn - 1) * tmp))
    prob = fdtrc(dfbn, dfwn, f)
    return stats.stats.F_onewayResult(f, prob)
