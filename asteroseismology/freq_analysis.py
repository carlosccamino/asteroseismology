"""
Author: Carlos Casimiro Camino Mesa
Date: 11-11-2025

Last Update: 07-12-2025

Module to analyze frequencies and stellar oscillations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools as it
import os

def to_microHz(f:float)->float:
    """
    Function to convert c/d to microHz

    Parameters
    ----------

    f: float
        Frequency in c/d

    Output
    ------

    f_muHz: float
        Frequency in microHz
    """

    return f*1000000/(24*3600)

def to_cd(f:float)->float:
    """
    Function to convert microHz to c/d

    Parameters
    ----------

    f: float
        Frequency in microHz

    Output
    ------

    f_cd: float
        Frequency in c/d
    """

    return 24*3600*f/1000000

def anti_aliasing(frequency_sample:pd.DataFrame, window_peaks:list, max_harmonic:int=3, save_file:bool=False, filename:str='test')-> pd.DataFrame:
    """
    Returns frequencies that are neither Nyquist nor window function aliased frequencies.

    Parameters
    ----------

    frequency_sample: pd.DataFrame
        Frequency Dataframe, preferably with amplitudes and frequency error estimation. Format must be
        Freqs, Amps, Freqs_error

    window_peaks: list
        List with all sources of aliasing, i.e. Nyquist frequency, window function 
        frequencies, known gap spacing frequencies
    
    max_harmonic: int
        Maximum integer to look for when applying f_alias = |f_real +/- f_wp|, 
        being f_wp a frequency from the window peaks

    save_file: bool
        If true, a folder 'real_freqs' will be generated at your working directory containing a file
        where columns are Freqs, Amps, Freq_err, Alias. If alias frequency, harmonic combination is shown,
        else, column is empty.
    
    file_name: str (mandatory if save_file)
        Name of the file where freqs will be stored
    
    Output
    ------

    real_freqs: pd.DataFrame
        DataFrame with the same format as the file output
    """

    real_frequencies = []
    all_frequencies = []
    columns = frequency_sample.columns
    frequency_sample.sort_values(by=columns[1], ascending=True, inplace=True)
    
    for row in frequency_sample.itertuples(index=False):
        is_alias = False
        f_peak = row[0]
        amp_peak = row[1]
        error_f_peak = row[2]
        combination = ''
        
        # === 1. DIRECT MATCH CHECK (vs. SWF peaks) ===
        # If the peak matches a window frequency, it's highly likely a sampling artifact.
        for i, f_w in enumerate(window_peaks):
            for n in range(1,max_harmonic+1):
                if abs(f_peak - n*f_w) < error_f_peak:
                    is_alias = True
                    combination = f"{n}*f_w{i}"
                    break
        
        if is_alias:
            all_frequencies.append((f_peak, amp_peak, error_f_peak, combination))
            continue
        
        # === 2. INVERSE ALIAS CHECK (vs. already accepted REAL peaks) ===
        # Is this peak (f_peak) an alias of a REAL frequency (f_real) already found?
        for i, (f_real, amp_real, error_f_real) in enumerate(real_frequencies):
            
            
            # We only check against the window frequencies
            for j, f_w in enumerate(window_peaks):
                for n in range(1, max_harmonic + 1):
                    # Alias formulas: f_alias = |f_real ± n*f_w|
                    alias_sum = abs(f_real + n * f_w)
                    alias_sub = abs(f_real - n * f_w)
                    
                    # Dynamic Tolerance Check: Does f_peak fall within twice the error of the theoretical alias?
                    if abs(f_peak - alias_sum) < error_f_peak:
                        
                        # Since data is sorted by amplitude, amp_peak <= amp_real,
                        # confirming this peak is a weaker "daughter" peak.
                        is_alias = True
                        combination = f"f_{i}+{n}*f_w{j}"
                        all_frequencies.append((f_peak, amp_peak, error_f_peak, combination))
                        break
                    elif abs(f_peak - alias_sub) < error_f_peak:
                        is_alias = True
                        combination = f"f_{i}-{n}*f_w{j}"
                        all_frequencies.append((f_peak, amp_peak, error_f_peak, combination))
                        break

                if is_alias:
                    break

            if is_alias:
                break
        
        # === 3. ACCEPTANCE ===
        # If the peak fails all alias checks, it's accepted as a candidate for a real physical signal.
        if not is_alias:
            real_frequencies.append((f_peak, amp_peak, error_f_peak))
            all_frequencies.append((f_peak, amp_peak, error_f_peak, combination))
    if save_file:
        os.makedirs('real_freqs', exist_ok=True)
        file_df = pd.DataFrame(data = all_frequencies, columns=['Freqs', 'Amps', 'Freq_err', 'Alias'])
        file_df.sort_values(by='Amps', ascending=False, inplace=True)
        file_df.to_csv('./real_freqs/'+filename
                       , index=False, header=True)
            
    return all_frequencies

def harmonic_check(freqs:pd.DataFrame, n:int, freqs_to_combine:int) -> pd.DataFrame:
    """
    Function to find the harmonics 

    Parameters
    ----------

    freqs: pd.DataFrame
        Frequencies to evaluate in a DataFrame containing an amplitude column.
    
    n: int
        Maximum harmonic to find, from -n..., 0, ..., n
    
    freqs_to_combine: int
        Determines how many frequencies will be used to compute the harmonics.
        For instance, if freqs_to_combine = 3:

        f = n_i·f_1+ n_j·f_2+n_k·f_3 with n_i,j,k from -n to n.

    Output
    ------

    Dataframe containing the original frequencies and the harmonic combination found.

    """

    # All n values
    n_values = range(-n, n+1)

    # Getting the frequencies that will serve as base of the harmonics
    columns = freqs.columns
    freqs_sorted = freqs.sort_values(by=columns[1], ascending=False)
    f_base = freqs_sorted.iloc[:freqs_to_combine, 0]


