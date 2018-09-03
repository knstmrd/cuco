import pandas as pd
import numpy as np
from scipy import stats
import librosa as lr
from scipy.signal import filtfilt, argrelmax, find_peaks, correlate


def autocorrelation(array, samplerate=220, max_len_autocorr=220):
    try:
        autocorr = correlate(array, array)
        autocorr = autocorr[autocorr.shape[0]//2:]
        peaks = find_peaks(autocorr[:max_len_autocorr])
        if len(peaks[0]) == 0:
            peakpos = max_len_autocorr
            peakval = 1
        else:
            peakpos = peaks[0][0]
            peakval = autocorr[peaks[0][0]] / autocorr[0]
        return {'autocorr_peak_position': peakpos / samplerate,
                'autocorr_peak_value_normalized': peakval,
                'autocorr_ZCR': np.sum(lr.core.zero_crossings(autocorr[:max_len_autocorr])) / max_len_autocorr}
    except ValueError as e:
        return {'autocorr_peak_position': max_len_autocorr / samplerate,
                'autocorr_peak_value_normalized': 1,
                'autocorr_ZCR': 0}


def get_features(array, samplerate=220, max_len_autocorr=220):
    argmin = np.argmin(array)
    argmax = np.argmax(array)
    std = np.std(array)
    rms = np.sqrt(np.mean(array**2))
    acr = autocorrelation(array)
    zcr = np.sum(lr.core.zero_crossings(array)) / array.shape[0]
    return [array[argmin], array[argmax], np.mean(array), std,
            np.percentile(array, 10), np.percentile(array, 25),
            np.percentile(array, 50), np.percentile(array, 75), np.percentile(array, 90),
            stats.skew(array), stats.kurtosis(array),
            rms, rms / std,
            array[argmax] / array[argmin],
            acr['autocorr_peak_position'],
            acr['autocorr_peak_value_normalized'],
            acr['autocorr_ZCR']]


def get_features_with_derivative(array, samplerate=220, max_len_autocorr=220):
	feat_basic = get_features(array, samplerate, max_len_autocorr)
	feat_deriv = get_features(np.gradient(array), samplerate, max_len_autocorr)
	return feat_basic + feat_deriv