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
import fnmatch
import h5py
import hashlib
import math
import numpy as np
import os
import sys
import warnings

from lmfit.models import GaussianModel, StepModel
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.special import erf


def get_filelist(path, pattern=None, recursive=False):
    """
    Example: get_filelist('/path/to/dir', pattern='*.txt')
    """
    pattern = '*' if pattern is None else pattern
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
        if not recursive:
            break
    return result


def get_iscam_h5py_filelist(path, pattern='*.h5', recursive=False):
    return get_filelist(path, pattern=pattern, recursive=recursive)


def get_iscam_intensities(h5py_filename):
    """
    Returns the raw intensity values of iscam events.
    """
    with h5py.File(h5py_filename, 'r') as f:
        # Column C contains the raw intensity values
        intensities = np.array(f['C'])
        # Column G contains indices of proper events
        idx = np.array(f['G'])
    return intensities[idx]


def get_iscam_mws(intensities, mw_intensity_line_pars=None):
    """
    Calculate the molecular weights of the intensities with the line parameters
    `mw_intensity_line_pars`

    Parameters
    ----------
    intensities : np.ndarray
        inensities of an iscam measurements
    mw_intensity_line_pars : array like
        [slope, intercept] of the mw to intensities linear function.
    """
    mw_intensity_line_pars = [1, 0] if mw_intensity_line_pars is None \
        else mw_intensity_line_pars
    slope, intercept = mw_intensity_line_pars
    mws = (intensities - intercept) / slope
    return mws


def create_iscam_dataset(filelist, name=None, description=None,
                         mw_intensity_line_pars=None, **kwargs):
    """
    Parameters
    ----------
    filelist : list
        A list of files to be loaded into the dataset, as returned by the
        function `get_iscam_h5py_filelist()` or the function `get_filelist()`.
    name : str
        A brief name describing the dataset.
    description : str
        A thourough description describing the dataset.
    mw_intensity_line_pars : array like of floats
        The slope and the intercept of the mw intensity relation [slope,
        intercept]
    **kwargs
    Additional keyword arguments stored in the information of the dataset.
    """
    dataset = {}
    dataset['data'] = []
    # name, description, list of filenames
    dataset['info'] = {}
    dataset['info']['name'] = name
    dataset['info']['description'] = description
    dataset['info']['columns'] = []
    dataset['info']['filelist'] = filelist
    dataset['info']['mw_intensity_line_pars'] = mw_intensity_line_pars
    for key, value in kwargs.items():
        dataset['info'][key] = value
    for file in filelist:
        ints = get_iscam_intensities(file)
        mws = get_iscam_mws(ints, mw_intensity_line_pars=mw_intensity_line_pars)
        dataset['data'].append(mws)
        column = os.path.basename(file)
        column = os.path.splitext(column)[0]
        dataset['info']['columns'].append(column)
    return dataset


def _fwhm(var):
    fwhm = math.sqrt(var) * (2 * math.sqrt(2 * math.log(2)))
    return fwhm


def _var(fwhm):
    var = (_sigma(fwhm))**2
    return var


def _sigma(fwhm):
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return sigma


def gauss_fwhm(x, amp=1, mu=0, fwhm=None, y0=0):
    if fwhm is None:
        # per default: sigma = 1, i.e. var = 1
        fwhm = _fwhm(1)
    sigma = _sigma(fwhm)

    # per default: amp = 1 / (sigma * sqrt(2*pi))
    amp = amp / (sigma * math.sqrt(2*math.pi))

    y = amp * np.exp(- 1/2 * ((x - mu) / sigma)**2)
    return y + y0


def gauss_var(x, amp=None, mu=0, var=1, y0=0):
    fwhm = _fwhm(var)
    return gauss_fwhm(x, amp=amp, mu=mu, fwhm=fwhm, y0=y0)

gauss = gauss_fwhm

def gaussian(x, amplitude, center, sigma):
    return amplitude / (sigma*math.sqrt(2*math.pi)) * np.exp(-(x-center)**2
                                                             / (2 * sigma**2))


def gaussian_cdf(x, amplitude, center, sigma):
    return amplitude / 2.0 * (1 + erf((x - center) / (sigma * math.sqrt(2))))


def make_model(centers, center_pm=None, amplitude=None, fwhm=None, cdf=False,
               **kwargs):
    """
    Build model to be fitted consisting of len(centers) gaussians.

    Parameters
    ----------
    centers : array like of type float
        Initial center positions of the gaussians to build the model from.
    center_pm : float
        The centers can by varied within the initial center position +/- the
        center_pm value. Defaults to 10 kDa.
    amplitude : float
        The initial value of the amplitude (area) of the gaussians. Defaults to
        1 / len(centers).
    fwhm : float
        Initial full width at half maximum of the gaussians in kDa. Defaults to
        10 kDa. Remark: 'fwhm = 2.3548 * sigma'.
    cdf : bool
        Build model for cumulative distribution function.
    **kwargs : dict
        Keyword arguments to adjust the model creation. Possible parameters:
        'sigma_center_line_pars' : array like
            containing [slope, intercept] of the linear realation of sigma to
            center. If it is set, the sigma of the gaussians are constrained to
            the expression 'slope * center + intercept'.

    Returns
    -------
    lmfit.model.CompositeModel
    """
    center_pm = 10 if center_pm is None else center_pm
    amplitude = 1/len(centers) if amplitude is None else amplitude  # prob
    fwhm = 10 if fwhm is None else fwhm
    sigma = fwhm/2.3548
    sigma_center_line_pars = kwargs.pop('sigma_center_line_pars', None)
    for i, center in enumerate(centers):
        prefix = 'p{}_'.format(i + 1)
        # Decide on Model
        if cdf: mod = StepModel(prefix=prefix, form='erf')
        else: mod = GaussianModel(prefix=prefix)
        # Create Composite Model
        if i == 0: model = mod
        else: model += mod
        model.set_param_hint('{}amplitude'.format(prefix),
                             value=amplitude,
                             min=0, max=1)
        model.set_param_hint('{}center'.format(prefix),
                             value=center,
                             min=max(0,center-center_pm),
                             max=center+center_pm)
        if sigma_center_line_pars is None:
            kwargs_sigma = {'value': sigma, 'min': 0, 'max': 10*sigma}
        else:
            expr = '{}center * {} + {}'.format(prefix, *sigma_center_line_pars)
            kwargs_sigma = {'expr': expr}
        model.set_param_hint('{}sigma'.format(prefix), **kwargs_sigma)
    return model


def fit_iscam(dataset, model, mw_range=None, bins=None, cdf=False,
              use_x_means=False, use_y_weights=False, add_fit=True):
    """
    Parameters
    ----------
    dataset : dataset
        A dataset containing the data to be fitted.
    mode : CompositeModel
        A model to fit the data to.
    mw_range : tuple of floats
        (min, max) range to be used for the histogram calculation. Defaults to
        (0, 1000) kDa.
    bins : int
        Number of bins to bin the data for the histogram. Defaults to 100.
    cdf : bool
        Fit the model to the cumulative distribution function of the histogram,
        instead the histogram itself.
    use_x_means : bool
        Use means of x values of bins, instead of centers of bins.
    use_y_weights : bool
        Weight the y values by 1 / N, where N is the number of datapoints for
        each individual bin.
    add_fit : bool
        Add the fit to the dataset.

    Returns
    -------
    lmfit.ModelResult

    Notes
    -----
    One can refit the returned result with changed paramters:
    result.params['g2_sigma'].value /= 2
    result.params['g3_sigma'].set(8, vary=False)
    result.params['g3_amplitude'].set(100, vary=True)
    result.fit()
    """
    mw_range = (0, 1000) if mw_range is None else mw_range
    bins = 100 if bins is None else bins

    data = get_data(dataset)
    try:
        data = np.concatenate(data)
    except:
        pass

    if cdf:
        # Calculate CDF of data
        x_cdf = data
        x_cdf.sort()
        N = len(x_cdf)
        y_cdf = np.linspace(1/N, 1, num=N)
        x = x_cdf
        y = y_cdf
    else:
        # Calculate bins and histogram from data
        y_bin, edges = np.histogram(data, bins=bins, range=mw_range,
                                    density=True)
        if use_x_means:
            # Get all elements of individual bins (l <= element < r) and
            # calculate the mean
            x_bin = np.array([np.mean(data[np.logical_and(data >= l,
                                                          data < r).nonzero()])
                              for l, r in zip(edges[:-1], edges[1:])])
        else:
            x_bin = (edges[:-1] + edges[1:]) / 2
        x = x_bin
        y = y_bin

    # Calculate weights / uncertainties of data
    if use_y_weights:
        # get number of datapoints for each bin
        bin_idx = np.digitize(data, edges)
        n_bin = np.bincount(bin_idx, minlength=len(y_bin))
        # Counting statistics (Poisson distribution)
        # -> variance = n, uncertainty = sqrt(n)
        # lmfit weights by multiplying (data - model) * weight
        # and expects residuals to "have been divided by the
        # true measurement uncertainty (data - model) / sigma"
        weights_bin = 1 / np.sqrt(n_bin)
    else:
        weights_bin = None

    # Get parameters of the model to be fitted
    mod_params = model.make_params()

    # Do the fitting
    result = model.fit(y, mod_params, x=x, weights=weights_bin,
                       nan_policy='omit')

    if add_fit:
        _add_fit(dataset, result,
                 model_params=mod_params,
                 mw_range=mw_range,
                 bins=bins)
    return result


def get_key(model_params, mw_range, bins):
    """
    Generate a hash key based on the model parameters used and the fited
    mw_range and number of bins.
    """
    hasher = hashlib.md5()
    hasher.update(bytes(str(model_params), 'ASCII'))
    hasher.update(bytes(str(mw_range), 'ASCII'))
    hasher.update(bytes(str(bins), 'ASCII'))
    key = hasher.hexdigest()
    return key


def _add_fit(dataset, result, **kwargs):
    if 'results' not in dataset.keys():
        dataset['results'] = {}
    hash_key = get_key(**kwargs)
    dataset['results'][hash_key] = {
        'settings': kwargs,
        'fit_report': result.fit_report(),
        'chisqr': result.chisqr,
        'redchi': result.redchi,
        'aic': result.aic,
        'bic': result.bic,
        'result_params': result.params,
        'fit_params': get_fit_params(result.params)
    }


def get_best_result(dataset, number_of_gaussians=None):
    aic = sys.float_info.max
    result_key = None
    result_params = None
    fit_params = None
    try:
        for key, result in dataset['results'].items():
            if (number_of_gaussians is None or number_of_gaussians
                    == len(result['fit_params']['centers'])):
                if result['aic'] < aic:
                    aic = result['aic']
                    result_key = key
                    result_params = result['result_params']
                    fit_params = result['fit_params']
    except KeyError:
        message = 'Dataset \'{}\' has no results!'.format(
                                                    dataset['info']['name'])
        warnings.warn(message)
    return result_key, result_params, fit_params


def plot_iscam_fit(dataset, mw_range, bins, fit_params=None, centers=None,
                   cdf_protomers=False, cdf_monomers=False, cdf_fit=False,
                   components=False, plot_range=None, mw=None,
                   labeled_xticks=None, figpath=None):
    """
    Parameters
    ----------
    data : np.ndarray or list of np.ndarray of type float
        A list with arrays containing the events
    mw_range : tuple of floats
        (min, max) range to be used for histogram calculation. Defaults to (0,
        1000) kDa.
    bins : int or str
        Number of bins to bin the data for the histogram or str to describe the
        method to determine the number of bins (see `np.histogram()`).
    fit_params : dict
        The fit_params from the result from `fit_iscam()` as returned by the
        function `get_fit_params()`. If fit_params is None, the best fit from
        dataset is chosen (`get_best_result()`).
    centers : bool or list of float
        Plot vertical black lines at fitted or given positions.
    cdf_protomers : bool
        Plot CDF of protomers.
    cdf_monomers : bool
        Plot CDF of monomers.
    cdf_fit : bool
        Plot CDF of fit.
    components : bool
        Plot individual fitted guassians.
    plot_range : tuple of floats
        (min, max) range to be used for xlim for plotting. Defaults to mw_range.
    mw : float
        The molecular weight to be used to plot the x_ticks. Defaults to 25.
    labeled_xticks : list
        A list of number of multiples of mw, where the ticks should be labeled
        with the molecular weight. Defaults to [1, 10, 100].
    figpath : str
        Absolute filename where to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the histogram and the fit.
    """
    mw_range = (0, 1000) if mw_range is None else mw_range
    bins = 100 if bins is None else bins
    plot_range = mw_range if plot_range is None else plot_range
    mw = 25 if mw is None else mw
    labeled_xticks = [1, 10, 100] if labeled_xticks is None else labeled_xticks
    data = get_data(dataset)
    try:
        data = np.concatenate(data)
    except:
        pass

    # Calculate cumsum of experimental data
    x_cdf = data
    N = len(x_cdf)
    y_cdf = np.linspace(1/N, 1, N)

    # Weight cumsum by mw (i.e. number of monomers) per event
    x_cdf.sort()
    y_cdf_mw = np.cumsum(x_cdf) / x_cdf.sum()

    # Calculate bins and histogram of experimental data
    y_bin, edges = np.histogram(data, bins=bins, range=mw_range, density=True)
    x_bin = (edges[0:-1] + edges[1:]) / 2
    N_bin = len(x_bin)

    # Plotting ...
    fig_fit, ax_hist = plt.subplots()
    ax_cdf = ax_hist.twinx()
    ax_cdf.spines['right'].set_visible(True)

    # Plot histogram of data
    width = (max(mw_range) - min(mw_range)) / N_bin * 0.85
    ax_hist.bar(x_bin, y_bin, width=width, antialiased=True, color='orange')

    # Plot CDF of experimental data
    if cdf_protomers:
        ax_cdf.plot(x_cdf, y_cdf, linewidth=0.5, color='cyan')
    if cdf_monomers:
        ax_cdf.plot(x_cdf, y_cdf_mw, linewidth=0.5, color='magenta')

    # Plot histogram and CDF of fit
    if fit_params is None:
        _, _, fit_params = get_best_result(dataset)
    if fit_params is not None:
        names, values = get_fit_values(fit_params)
        y_bin_fit = np.zeros(len(x_bin))
        y_cdf_fit = np.zeros(len(x_cdf))
        for params in values[:, :, 0].T:
            _y_bin_fit = gaussian(x_bin, *params)
            y_bin_fit += _y_bin_fit
            y_cdf_fit += gaussian_cdf(x_cdf, *params)
            if components:
                ax_hist.plot(x_bin, _y_bin_fit, linestyle='--', linewidth=0.5,
                             color='black')
        if not components:
            ax_hist.plot(x_bin, y_bin_fit, linestyle='--', linewidth=0.5,
                         color='black')
        if cdf_fit:
            ax_cdf.plot(x_cdf, y_cdf_fit, linestyle=':', linewidth=0.5,
                        color='black')

    # Plot vertical lines at center positions
    centers = [] if (centers is None or not centers) else centers
    if isinstance(centers, bool) and fit_params is not None:
        centers = fit_params['centers'][:,0]
    for c in centers:
        ax_hist.axvline(c, linestyle='--', linewidth=0.5, color='black')
        ax_hist.text(c, (y_bin.max() - y_bin.min()) * 0.97 + y_bin.min(),
                     '{:.0f}'.format(c))

    # Formatting ...
    xticks = [mw*tick for tick in labeled_xticks]
    ax_hist.xaxis.set_ticks(xticks)
    ax_hist.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax_hist.xaxis.set_ticklabels(["1", "", "", "10", "", "", "100", ""])
    ax_hist.xaxis.set_minor_locator(MultipleLocator(mw))
    #ax_cdf.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax_hist.yaxis.set_ticks([])
    ax_cdf.yaxis.set_ticks([0,1])
    xlim = (plot_range[0] - 10, plot_range[1] + 10)
    ax_hist.set_xlim(xlim)
    ax_cdf.set_xlim(xlim)

    ax_hist.set_xlabel('Molecular weight (kDa)')
    ax_hist.set_ylabel('$p_{events}$')
    #ax_cdf.set_ylabel('$\sum p$')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.yaxis.labelpad = -7

    fig_fit.show()

    if figpath is not None:
        fig_fit.savefig(figpath)

    return fig_fit, ax_hist, ax_cdf


def fit_and_plot_iscam(dataset, centers_init, center_pm=None, amplitude=None,
                       fwhm=None, cdf=False, sigma_center_line_pars=None,
                       mw_range=None, bins=None, use_x_means=False,
                       use_y_weights=False, add_fit=True, plot_range=None,
                       return_fig=True, **kwargs):
    model = make_model(centers_init, center_pm=center_pm, amplitude=amplitude,
                       fwhm=fwhm, cdf=cdf,
                       sigma_center_line_pars=sigma_center_line_pars)
    result = fit_iscam(dataset, model, mw_range=mw_range, bins=bins, cdf=cdf,
                       use_x_means=use_x_means, use_y_weights=use_y_weights,
                       add_fit=add_fit)
    fit_params = get_fit_params(result.params)
    fig, ax_hist, ax_cdf = plot_iscam_fit(dataset, mw_range, bins,
                                          fit_params=fit_params,
                                          plot_range=plot_range, **kwargs)
    fig_ax = (fig, ax_hist, ax_cdf)

    if return_fig:
        return result, fig_ax
    return result


def get_data(dataset):
    return dataset['data']


def get_info(dataset):
    return dataset['info']


def get_result(dataset):
    return dataset['result']


def get_number_of_events_and_mw(dataset, mw_monomer=None):
    """
    Calculate number of events and monomers
    """
    data = get_data(dataset)
    # determine the number of events of the data
    mw_monomer = 1 if mw_monomer is None else mw_monomer
    number_of_events = []
    number_of_monomers = []
    for d in data:
        number_of_events.append(len(d))
        number_of_monomers.append(d.sum() / mw_monomer)
    return np.array(number_of_events), np.array(number_of_monomers)


def get_values(result_params, key):
    values = []
    for p_key, param in result_params.items():
        if key in p_key:
            values.append([param.value, param.stderr])
    return np.array(values)


def get_areas(result_params):
    return get_values(result_params, 'amplitude')


def get_centers(result_params):
    return get_values(result_params, 'center')


def get_sigmas(result_params):
    return get_values(result_params, 'sigma')


def get_fit_params(result_params, verbose=False):
    """
    Return dictionary of fitted parameters. Each parameter consists of the
    fitted value and the corresponding estimated standard error.
    """
    params = {}
    for keys, key in zip(['areas', 'centers', 'sigmas'],
                         ['amplitude', 'center', 'sigma']):
        params[keys] = get_values(result_params, key)
    area = params['areas']
    cent = params['centers']
    sigm = params['sigmas']
    if verbose:
        values = '{:6.2f}±{:.2f}'
        values += ', {:6.2f}±{:.2f}' * max(0, (len(cent) - 1))
        print('  Areas: ' + values.format(*area.flatten()))
        print('Centers: ' + values.format(*cent.flatten()))
        print(' Sigmas: ' + values.format(*sigm.flatten()))
    return params


def get_fit_values(fit_params):
    values = []
    names = ['areas', 'centers', 'sigmas']
    for key in names:
        values.append(fit_params[key])
    return names, np.array(values)
