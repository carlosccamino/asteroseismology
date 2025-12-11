"""
Author: Carlos Casimiro Camino Mesa
Date: 11-11-2025

Last Update: 11-12-2025

Module to analyze frequencies and stellar oscillations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools as it
import os
from intervaltree import IntervalTree

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
        being f_wp the p-th frequency from the window peaks

    save_file: bool
        If true, a folder 'real_freqs' will be generated at your working directory containing a file
        where columns are Freqs, Amps, Freq_err, Alias. If alias frequency, harmonic combination is shown,
        else, column is empty.
    
    file_name: str (mandatory if save_file)
        Name of the file where freqs will be stored
    
    Output
    ------

    real_freqs: pd.DataFrame
        DataFrame with the same format as the file output and a Alias column where the combinations are written as n_i*f_i+/-n_j*f_wj,
        being i and j the index of the i-th and j-th real frequency and window frequency, respectively
    """

    real_frequencies = []
    all_combis = []
    columns = frequency_sample.columns
    frequency_sample.sort_values(by=columns[1], ascending=False, inplace=True, ignore_index=True)
    file_df = frequency_sample.copy()
    
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
            all_combis.append(combination)
            continue
        
        # === 2. INVERSE ALIAS CHECK (vs. already accepted REAL peaks) ===
        # Is this peak (f_peak) an alias of a REAL frequency (f_real) already found?
        for i, f_real in enumerate(real_frequencies):
            
            
            # We only check against the window frequencies
            for j, f_w in enumerate(window_peaks):
                for n in range(1, max_harmonic + 1):
                    # Alias formulas: f_alias = |f_real ± n*f_w|
                    alias_sum = abs(f_real + n * f_w)
                    alias_sub = abs(f_real - n * f_w)
                    
                    # Dynamic Tolerance Check: Does f_peak fall within twice the error of the theoretical alias?
                    if abs(f_peak - alias_sum) <= error_f_peak:
                        
                        # Since data is sorted by amplitude, amp_peak <= amp_real,
                        # confirming this peak is a weaker "daughter" peak.
                        is_alias = True
                        combination = f"f_{i}+{n}*f_w{j}"
                        all_combis.append(combination)
                        break
                    elif abs(f_peak - alias_sub) <= error_f_peak:
                        is_alias = True
                        combination = f"f_{i}-{n}*f_w{j}"
                        all_combis.append(combination)
                        break

                if is_alias:
                    break

            if is_alias:
                break
        
        # === 3. ACCEPTANCE ===
        # If the peak fails all alias checks, it's accepted as a candidate for a real physical signal.
        if not is_alias:
            real_frequencies.append(f_peak)
            all_combis.append(np.nan)

    file_df['Alias'] = all_combis
    #file_df.sort_values(by=, ascending=False, inplace=True)
    if save_file:
        os.makedirs('real_freqs', exist_ok=True)
        file_df.to_csv('./real_freqs/'+filename
                       , index=False, header=True)
            
    return file_df


def harmonics(freqs:pd.DataFrame, n:int, freqs_to_combine:int, err:float, f_col:int=0, amp_col:int=1) -> pd.DataFrame:
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

    err: float
        Tolerance to consider a harmonic and a frequency the same frequency. Typically Rayleigh frequency.

    f_col: int
        Frequency column. Consider 0 as first column. Default = 0

    amp_col: int
        Amplitude column. Consider 0 as first column. Default = 1

    Output
    ------

    Dataframe containing the original frequencies, harmonics and the harmonic combination found.

    Example
    -------
        For instance, if freqs_to_combine = 3:

        f = n_i·f_0 + n_j·f_2 with n_i,j from -n to n.

        OR

        f = n_i·f_0+ n_j·f_1
    """

    # 1. Sorting the frequencies per amplitude
    columns = freqs.columns
    freqs_sorted = freqs.sort_values(by=columns[amp_col], ascending=False).reset_index(drop=True)
    structure = freqs_sorted.copy()
    fre = np.array(structure.iloc[:,f_col])

    # 2. Calculate all the possible combinations (including harmonics)
    # 2.1 In case number of frequencies is higher than the actual list, updating freqs_to_combine
    if freqs_to_combine > len(fre):
        freqs_to_combine = len(fre)
    product = list(it.product(range(1,freqs_to_combine+1),range(n,-(n+1),-1)))
    combination = list(it.combinations(product,2)) #Pairs of combination

    # 3. List containing the results [(frequency number,combination)]
    possible_combinations = []

    for pair in combination:
        #We avoid pairs of the same frequency and negative values of the combination to not repeat opperations
        if ( ( (pair[0][0] != pair[1][0]) or (pair[1][1] == 0) ) and ( (pair[0][1]*fre[pair[0][0]-1]+pair[1][1]*fre[pair[1][0]-1]) > 0 ) ):
            differencies = abs(fre - (pair[0][1]*fre[pair[0][0]-1]+pair[1][1]*fre[pair[1][0]-1])) #Differencies with all the frequencies
            differencies_under_tol = np.where(differencies <= err)
            if differencies_under_tol[0].size > 0:
                fre_id = int(differencies_under_tol[0][0]+1) #Just the first coincidence
                possible_combinations = possible_combinations+[(fre_id,)+pair]

    possible_combinations = sorted(possible_combinations) #We order the structure

    # 4. Now, complete the Combinations columns
    structure.insert(0,'ID', ['F'+str(i) for i in range(1,len(fre)+1)])
    f = [(0, (len(fre)+2, 0), (len(fre)+2, 0))] #We initialize a variable to allow to select the combination involving the first frequencies
    comb = []
    for m in possible_combinations:
        if ( m[1][1] == 0 and m[2][1] != 1 and m[2][0] != m[0] ):
            comb = str(m[2][1])+'*F'+str(m[2][0])
        elif ( m[2][1] == 0 and m[1][1] != 1 and m[1][0] != m[0] ):
            comb = str(m[1][1])+'*F'+str(m[1][0])
        elif ( m[1][0] < m[0] and m[2][0] < m[0] ):
            if m[2][1] > 0:
                comb = str(m[1][1])+'*F'+str(m[1][0])+'+'+str(m[2][1])+'*F'+str(m[2][0])
            else:
                comb = str(m[1][1])+'*F'+str(m[1][0])+str(m[2][1])+'*F'+str(m[2][0])

        # 4.1 Here we select the lowest combinations
        if ( comb != []):
            if ( m[0] != f[0]): #This is to add a combination for the next frequency
                structure.loc[m[0]-1, 'Combinations'] = comb
                f = m
            elif (m[0] == f[0] and m[1][0]+m[2][0] < f[1][0]+f[2][0]): #This is to select the lowest frequency for the combination
                structure.loc[m[0]-1, 'Combinations'] = comb
                f = m

        comb = [] # 4.2 Re-initialize this structure

    return pd.DataFrame(data=structure)


def freq_resolver(freqs:pd.DataFrame, err:float, f_col:int=0, amp_col:int=1, type:str='close-open'):
    """
    This function evaluates if there are frequencies closer than err (e.g. the Rayleigh frequency). It will take the highest amplitude's as the real value.

    Parameters
    ----------

    freqs: pd.DataFrame
        DataFrame containing the frequencies and amplitudes

    err: float
        Tolerance value. Resolving power. For frequencies closer than this value, the one with highest amplitude will be considered the real one

    f_col: int
        Index of the column containing frequencies.

    amp_col: int
        Index of the column containing amplitudes.

    type: str
        Type of error interval

            'close' = [-err, err]

            'open' = (-err, err)

            'close-open' = [-err, err)

            'open-close' = (-err, err]

    Output
    ------

    Frequency dataframe with only frequencies that are spaced more than err.
    """

    # 1. Extracting frequencies and amplitudes values
    columns = freqs.columns
    fs = freqs[columns[f_col]].values
    amps = freqs[columns[amp_col]].values
    err0 = err
    eps = 10**-8

    limits = {
        'close': [err, err+eps],
        'open' : [err-eps, err-eps],
        'close-open': [err, err],
        'open-close': [err-eps, err+eps]
    }
    

    #2. We create an interval tree to consider the overalps within err
    tree = IntervalTree()

    #3. We defined the resolved frequencies
    resolved = []

    # 4. Loop over the frequencies
    for i, f in enumerate(fs):

        # 4.1 Check if the frequency is within any tolerance range in the tree
        overlaps = tree.overlap(f-err/2, f+err/2)

        # 4.2 Adding to the tree in case there is no overlaps
        if not overlaps:
            tree.addi(f-limits[type][0]/2, f+limits[type][1]/2, i) #begin, end, index
            resolved.append(i)
            continue

        # 4.3 If the frequency does overlaps, comparing amplitudes
        best = i
        for iv in overlaps:
            j = iv.data
            if amps[j]>amps[i]:
                best = j

        # 4.4 If the current one is still the best, we replace and remove the rest that overlaps the current one
        if best == i:
            for iv in overlaps:
                tree.remove(iv)
                resolved.remove(iv.data)
            
            # 4.4.1 Inserting the current one
            tree.addi(f-limits[type][0]/2, f+limits[type][1]/2, i)
            resolved.append(i)

    return freqs.iloc[resolved].reset_index(drop=True)

def window_function(t:np.ndarray, f_min:float, f_max:float, w:np.ndarray=None, resol:int=2000):
    """
    Calculate the normalised window function for f in [f_min, f_max] for a given time series.

    W(f) = | sum_j w_j * exp(-2π i f t_j) |^2

    Parameters
    ----------

    t: np.ndarray
        Time array.

    f_min: float
        Minimum frecuency for which the window function will be computed

    f_max: float
        Minimum frecuency for which the window function will be computed

    w: np.ndarray
        Weights of the sum components. All 1 by default

    resol: int
        Resolution in frequency, i.e. the how many frequency to compute between f_min and f_max

    Output
    ------

    W(f) in a np.ndarray format.
    """

    # 1. Assigning weights if they are not
    if w is None:
        w = np.ones_like(t)
    
    # 2. Defining the frequency span interval
    freqs = np.linspace(f_min, f_max, resol)

    # 3. Creating the window function array as a complex vector
    W = np.zeros_like(freqs, dtype=np.complex128)

    # 4. Computing W for every frequency in blocks to not burn out RAM
    block = 2000
    for i in range(0, len(freqs), block):
        f_block = freqs[i:i+block]

        # 4.1 Computing the exponential
        exponent = -2j*np.pi*np.outer(f_block, t) # Outer of a ^ b (column vectors) is the same as the matritial product of a·b^T. 

        # 4.2 Computing W with the weights as a matrix product
        W[i:i+block] = np.exp(exponent) @ w # 

    return np.abs(W)**2/(np.max(np.abs(W)**2))
