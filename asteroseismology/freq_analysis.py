"""
Author: Carlos Casimiro Camino Mesa
Date: 11-11-2025

Last Update: 18-12-2025

Module to analyze frequencies and stellar oscillations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import itertools as it
import os
from intervaltree import IntervalTree
from astropy.timeseries import LombScargle

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


def harmonics(
    df: pd.DataFrame,
    freq_col: int,
    amp_col: int,
    harmonic_order: int,
    n_independent: int,
    tol: float
) -> pd.DataFrame:
    """
    Identifies independent frequencies and detects harmonic, subharmonic, and
    linear combinations using a vectorized, amplitude-prioritized approach.

    This function processes frequencies in descending order of amplitude to ensure
    a strict directed acyclic dependency graph (frequencies can only depend on
    those with higher amplitude). It employs a "Generator Basis" logic where only
    the first `n_independent` fundamental frequencies are allowed to form linear
    combinations to explain subsequent signals.

    Algorithm Logic
    ---------------
    1. **Sorting**: Frequencies are sorted by amplitude (descending).
    2. **Generator Basis**: A dynamic list of "parent" frequencies is maintained.
       This list is strictly limited to a maximum size of `n_independent`.
    3. **Vectorized Matching**: For each frequency `f`:
       - It is tested against all possible harmonics, subharmonics, and linear
         combinations of the current Generator Basis using vectorized NumPy
         operations for high performance.
       - **Harmonics**: Checks if `f ~= n * base` (where n is integer).
       - **Subharmonics**: Checks if `f ~= base / n` (where n is integer).
       - **Linear Combinations**: Checks if `f ~= n1 * base_i + n2 * base_j`.
    4. **Classification**:
       - If a match is found within `tol`, the relationship is recorded.
       - If no match is found, `f` is marked as 'independent'.
       - **Crucial Step**: If `f` is independent AND the Generator Basis is not
         yet full (size < `n_independent`), `f` is added to the basis. Otherwise,
         it remains independent but is not used to explain future frequencies.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing frequency and amplitude data.
    freq_col : int
        Zero-based column index for frequency values (Hz).
    amp_col : int
        Zero-based column index for amplitude values.
    harmonic_order : int
        Maximum integer order for harmonics, subharmonics, and linear coefficients.
        (e.g., if 2, checks ±1, ±2).
    n_independent : int
        Maximum size of the Generator Basis. Only the first `n_independent`
        unexplained frequencies will be used as parents for combinations.
    tol : float
        Absolute frequency tolerance for matching (e.g., 1e-3 for ±0.001 Hz).

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with two new columns prepended/appended:
        - 'ID': Unique identifier based on amplitude rank (F1, F2, ...).
        - 'linear_combination': String describing the relationship (e.g.,
          "2·F1", "1·F1+2·F3") or "independent".

    Notes
    -----
    - **Performance**: This implementation uses NumPy broadcasting and vectorization
      to evaluate thousands of potential combinations simultaneously, making it
      significantly faster than iterative approaches for large datasets.
    - **Precedence**: When multiple explanations are possible, the algorithm
      prioritizes matches involving higher-amplitude parents (lower ID indices).
    """

    out = df.copy()
    n = len(out)

    # Convert columns to numpy arrays for speed
    freqs = out.iloc[:, freq_col].to_numpy(dtype=float)
    amps = out.iloc[:, amp_col].to_numpy(dtype=float)

    # --------------------------------------------------
    # 1. Sort by Amplitude (Descending)
    # --------------------------------------------------
    order = np.argsort(-amps)
    freqs_sorted = freqs[order]
    
    # Create ordered IDs (F1, F2, ...)
    ids_ordered = np.array([f"F{i+1}" for i in range(n)], dtype=object)
    
    # Result container aligned with sorted arrays
    results_ordered = np.full(n, "", dtype=object)

    # --------------------------------------------------
    # 2. Pre-calculate Vectorized Coefficients
    # --------------------------------------------------
    # Coefficients: [-N, ..., -1, 1, ..., N]
    coeffs = np.r_[np.arange(-harmonic_order, 0), np.arange(1, harmonic_order + 1)]
    # Divisors for subharmonics: [2, ..., N]
    divisors = np.arange(2, harmonic_order + 1)

    # --------------------------------------------------
    # 3. Main Loop (Iterative Row, Vectorized Check)
    # --------------------------------------------------
    basis_vals = [] # Stores actual frequency values of the basis
    basis_ids = []  # Stores corresponding IDs (F1, F3, etc.)

    for i in range(n):
        target_f = freqs_sorted[i]
        
        # If basis is empty, current freq is automatically independent
        n_base = len(basis_vals)
        if n_base == 0:
            basis_vals.append(target_f)
            basis_ids.append(ids_ordered[i])
            results_ordered[i] = "independent"
            continue

        # Convert basis to array for broadcasting operations
        base_arr = np.array(basis_vals) # Shape: (n_base,)
        
        explained = False
        explanation_str = ""

        # --- A. Harmonic Check (Vectorized) ---
        # Matrix shape: (n_coeffs, n_base)
        harmonics_mat = coeffs[:, None] * base_arr[None, :]
        diff_h = np.abs(harmonics_mat - target_f)
        matches_h = np.argwhere(diff_h <= tol)
        
        if matches_h.size > 0:
            # Sort by base index (column 1) to prefer higher amplitude parent
            best_match = matches_h[np.argsort(matches_h[:, 1])][0]
            c_idx, b_idx = best_match
            explanation_str = f"{coeffs[c_idx]}·{basis_ids[b_idx]}"
            explained = True

        # --- B. Subharmonic Check (Vectorized) ---
        if not explained and divisors.size > 0:
            # Matrix shape: (n_base, n_divisors)
            sub_mat = base_arr[:, None] / divisors[None, :]
            diff_s = np.abs(sub_mat - target_f)
            matches_s = np.argwhere(diff_s <= tol)

            if matches_s.size > 0:
                # Sort by base index (row 0) to prefer higher amplitude parent
                best_match = matches_s[np.argsort(matches_s[:, 0])][0]
                b_idx, d_idx = best_match
                explanation_str = f"1/{divisors[d_idx]}·{basis_ids[b_idx]}"
                explained = True

        # --- C. Linear Combination Check (Vectorized Advanced) ---
        if not explained and n_base >= 2:
            # 1. Flatten all possible terms (c * base)
            # This allows us to use a single broadcasted sum matrix
            terms = (base_arr[:, None] * coeffs[None, :]).ravel()
            
            # Map flattened indices back to original basis/coeff indices
            term_base_idx = np.repeat(np.arange(n_base), len(coeffs))
            term_coeff_idx = np.tile(np.arange(len(coeffs)), n_base)

            # 2. Create Sum Matrix (Broadcasting all vs all)
            sum_matrix = terms[:, None] + terms[None, :]
            
            # 3. Find matches
            diff_c = np.abs(sum_matrix - target_f)
            matches_c = np.argwhere(diff_c <= tol)

            if matches_c.size > 0:
                # Indices in the flattened 'terms' array
                idx_flat_x = matches_c[:, 0]
                idx_flat_y = matches_c[:, 1]
                
                # Convert to basis indices to filter invalid pairs
                real_base_i = term_base_idx[idx_flat_x]
                real_base_j = term_base_idx[idx_flat_y]
                
                # Filter: strictly enforce i < j to avoid self-combinations 
                # or duplicates, ensuring amplitude hierarchy.
                valid_mask = real_base_i < real_base_j
                
                if np.any(valid_mask):
                    valid_matches = np.where(valid_mask)[0]
                    
                    # Sort logic: Prefer lowest index for base_j, then base_i
                    sort_idx = np.lexsort((
                        real_base_i[valid_matches], 
                        real_base_j[valid_matches]
                    ))
                    best_match_idx = valid_matches[sort_idx[0]]
                    
                    # Retrieve final components
                    final_x = idx_flat_x[best_match_idx]
                    final_y = idx_flat_y[best_match_idx]
                    
                    c1 = coeffs[term_coeff_idx[final_x]]
                    id1 = basis_ids[term_base_idx[final_x]]
                    
                    c2 = coeffs[term_coeff_idx[final_y]]
                    id2 = basis_ids[term_base_idx[final_y]]
                    
                    sign = "+" if c2 > 0 else ""
                    explanation_str = f"{c1}·{id1}{sign}{c2}·{id2}"
                    explained = True

        # --- D. Final Decision & Basis Update ---
        if explained:
            results_ordered[i] = explanation_str
        else:
            results_ordered[i] = "independent"
            # CRITICAL LOGIC: Only add to basis if we haven't reached the limit
            if n_base < n_independent:
                basis_vals.append(target_f)
                basis_ids.append(ids_ordered[i])

    # --------------------------------------------------
    # 4. Reconstruct DataFrame
    # --------------------------------------------------
    final_ids = np.empty(n, dtype=object)
    final_comb = np.empty(n, dtype=object)
    
    # Map sorted results back to original DataFrame order
    final_ids[order] = ids_ordered
    final_comb[order] = results_ordered
    
    out.insert(0, "ID", final_ids)
    out["linear_combination"] = final_comb

    return out

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

def window_function(t:np.ndarray, f_min:float, f_max:float, w:np.ndarray=None, resol:int=2000, method:str='Fourier', **ls_kwargs):
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
        Maximum frecuency for which the window function will be computed

    w: np.ndarray
        Weights of the sum components. All 1 by default

    resol: int
        - "Fourier": Resolution in frequency, i.e. the how many frequency to compute between f_min and f_max
        - "LombScargle": Oversampling factor

    method: str
        Method to compute the window function.
            - "Fourier": Classical theoretical FT of the window function.
            - "LombScargle": Compute the LombScargle periodogram of the window function. LombScargle from astropy.timeseries
    
    ls_kwargs: any
        kwargs for astropy.timeseries.LombScargle periodogram.

    Output
    ------

    W(f) in a np.ndarray format with 2 columns (N,2) [freq, W(f)]. Output is normalized by the maximum value
    """
    if method == 'Fourier':
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

        # 4.3 Returning the output
        window_normalized = np.abs(W)**2/(np.max(np.abs(W)**2))
        window_matrix_normalized = np.column_stack((freqs ,window_normalized))

    elif method == 'LombScargle':

        # 1. Creating the array with all fluxes set to 1
        flux = np.ones_like(t)

        # 2. Computing LombScargle periodogram without centering data nor fitting mean
        ls = LombScargle(t=t, y=flux, center_data=False, fit_mean=False, **ls_kwargs)

        # 2.1 Getting the amplitude and frequencies
        freqs, window = ls.autopower(method='fast', maximum_frequency=f_max, samples_per_peak=resol)
        window_matrix_normalized = np.column_stack((freqs, window))
        
    else:
        raise ValueError(f"Unknown method: {method}")

    return window_matrix_normalized


def window_check(freqs:pd.DataFrame, f_col:int, amp_col:int, window_function:np.ndarray, max_n:int = 1, tol:float=0) -> pd.DataFrame:
    """
    This function evaluates the alias frequencies produced by the convolution of the window function and the real signal.

    An alias is identified if:

            |f_alias - f_real| = n·f_wf +/- tol  
    
    where f_wf are the frequencies of the window function. This function also evaluates the amplitudes compared to the 
    
    offered by window function, i.e. amplitudes of peaks suspected to be aliased frequencies will be checked if their 
    
    amplitudes are a factor of the window function power computed for f_wf.

    Parameters
    ----------

    freqs: pd.DataFrame
        DataFrame containing (Frequencies, Amplitudes) from the sample
    
    f_col: int
        Index column of the frequencies

    amp_col: int
        Index column of the amplitudes

    window_function: np.ndarray
        Array of shape (N,2) containing (Frequencies, W(f)) from the computed window function

    max_n: int
        Max integer to evaluate the spurious frequencies in the window function. Usually this value is 1, more than this value is

        not implemented yet.

    tol: float
        Tolerance or Error to consider a frequency an alias

    Output
    ------

        DataFrame with all the frequencies and an extra column indicating the combination 
        found for an alias frequency.
    """

    # 1. Sorting the dataframe per amplitude
    columns = freqs.columns
    freqs_df_sorted = freqs.sort_values(by=columns[amp_col], ascending=False, ignore_index=True)
    freqs_matrix = freqs.iloc[:,[f_col,amp_col]].to_numpy()

    # 2. Finding the lowest amplitude detected in the real sample to see the lowest limit.
    freqs_sorted = freqs_matrix[np.argsort(freqs_matrix[:,1])][::-1]
    lowest_amp = freqs_sorted[:,1].min()
    freqs_list = freqs_sorted[:,0].copy()

    # 3. We first check if there are any observed frequency in the window function

    # 3.1 Creating a initial_check_matrix
    window_sorted = window_function[np.argsort(window_function[:,1])][::-1]
    window_list = window_sorted[:,0].copy()
    initial_check_matrix = freqs_list[:,None] - window_list[None,:] # f_real - f_wf

    # 3.2 Saving indexes of the observed frequencies that matched
    freqs_in_window_idx = np.argwhere(np.abs(initial_check_matrix) <= tol)

    # 4. We need to compute all the possible combinations of f_real +/- n·f_wf

    # 4.1 All n possible values (-n, -n+1,...0,...,n)
    n = np.arange(-max_n, max_n+1, 1)
    # 4.1.1 Discarding the n=0
    n = n[n!=0]
    # 4.2 Create a matrix win 2n+1 columns consisting of f_wf_i*n_j (f_wf x n)
    delta_matrix = np.outer(window_sorted[:,0], n)    
    
    # 5. We need to compute all the possible differences between all the frequencies

    # 5.1 Compute a matrix of f_i-f_j with both f_i and f_j sorted by amplitude (f_real x f_real)
    diffs_matrix = freqs_list[:,None] - freqs_list[None,:] # Broadcasting to be able to substract

    # 5.2 We need to compare which elements of |diffs_matrix - delta_matrix| <= tol (sólo nos quedamos con el primero)
    # 5.2.1 We expand dimensions to be able to substract (broadcasting)
    diffs_exp = diffs_matrix[:, :, None, None]       # (f_real, f_real, 1, 1)
    delta_exp = delta_matrix[None, None, :, :]       # (1, 1, f_wf, n)

    # 5.2.2 We take the boolean array where the match happens
    matches_id = np.abs(delta_exp-diffs_exp)<=tol

    # 5.2.3 We only select such that an alias is combination of stronger frequencies in amplitude (j<i)
    N = freqs_sorted.shape[0]
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    mask_hierarchy = j_idx < i_idx
    mask_hierarchy = mask_hierarchy[:, :, None, None]   # broadcasting

    matches_id &= mask_hierarchy

    # 5.2.4 Selecting the index where this happens
    alias_candidates_idx = np.argwhere(matches_id) # (f_real, f_real, f_wf, n)

    # 6. Creating the output as a dataframe
    combinations = []
    labels=[]
    teo_amp=[]
    for freq_idx in range(N):
        # If frequency was in window function, it will have a zero in initial_check_matrix at that row
        if np.isin(freq_idx, freqs_in_window_idx[:,0]):
            # We need to take the index of the FW, which is the column index
            match_idx = freqs_in_window_idx[freqs_in_window_idx[:,0] == freq_idx]
            wf_idx = match_idx[0, 1]
            combinations.append(f"FW_{wf_idx}")
            amp_wf = window_sorted[wf_idx][1]
            teo_amp.append(amp_wf)

        #If frequency is an alias due to the window function
        elif np.isin(freq_idx, alias_candidates_idx[:,0]):
            matches = alias_candidates_idx[alias_candidates_idx[:,0] == freq_idx]
            main_f_idx = matches[0,1]
            wf_idx = matches[0,2]
            amp_wf = window_sorted[wf_idx][1]
            amp_f = freqs_sorted[main_f_idx][1]
            if amp_f*amp_wf>lowest_amp:
                sign = "+" if matches[0,3] == 1 else "-"
                combinations.append(f"f{main_f_idx+1}{sign}FW{wf_idx}")
                teo_amp.append(amp_f*amp_wf)
            else:
                combinations.append(np.nan)
                teo_amp.append(np.nan)                

        #Real freq
        else:
            combinations.append(np.nan)
            teo_amp.append(np.nan)

        labels.append(f"f{freq_idx+1}")

    # 7. Updating the DataFrame
    freqs_df_sorted['Window_Alias'] = combinations
    freqs_df_sorted['Theoretical Alias_Amp'] = teo_amp
    if 'f_ID' not in freqs_df_sorted.columns:
        freqs_df_sorted.insert(loc=0, column='f_ID', value=labels)


    return freqs_df_sorted
