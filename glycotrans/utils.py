# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 11:43:07 2025

@author: ejasalth
"""

# Helper functions to run GlycoTrans (Abtheen, E.A., et al., Transformer-based Deep Learning for Glycan Structure Inference from Tandem Mass Spectrometry. bioRxiv, 2025: p. 2025.07.02.662857)
# Some of the functions have been adapted from CandyCrunch (Urban, J., Jin, C., Thomsson, K.A. et al. Predicting glycan structure from tandem mass spectrometry via deep learning. Nat Methods 21, 1206–1215 (2024))

import os
import sys
import numpy as np
import pandas as pd
import pyopenms
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from math import ceil
from transformers import BertConfig,  BertForSequenceClassification, BartForConditionalGeneration
from glycotrans.MSTokenizer import GlycoBertTokenizer, GlycoBartTokenizer
import time
import ast
from collections import defaultdict, Counter
from candycrunch.prediction import *
from glycowork.motif.processing import get_lib
from glycowork.motif.tokenization import (composition_to_mass,
                                          glycan_to_composition,
                                          glycan_to_mass, mapping_file,
                                          mz_to_composition)
from glycowork.glycan_data.loader import df_glycan
from glycowork.motif.draw import GlycoDraw
from candycrunch.analysis import *

glycan_path = os.path.join(os.path.dirname(__file__), "glycan_classes.pkl")
with open(glycan_path, "rb") as f:
    glycans_list = pickle.load(f)
glycans_list = pickle.load(open(glycan_path, 'rb'))

vocab_path_glycobert = os.path.join(os.path.dirname(__file__), "vocab_glycobert.json") 
vocab_path_glycobart = os.path.join(os.path.dirname(__file__), "vocab_glycobart.json")

  
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Functions to load mzML, mzXML, excel files
def open_mzML_pyopenms(filepath, num_peaks=1000, ms_level=2, intensity=True):
    highest_i_dict = defaultdict(dict)
    rts, intensities, reducing_mass, precursor_charges, scan_numbers = [], [], [], [], []

    # Create an MSExperiment object using PyOpenMS
    exp = pyopenms.MSExperiment()

    # Load the mzML file into the MSExperiment object
    pyopenms.MzMLFile().load(filepath, exp)

    for spectrum in exp:
        if spectrum.getMSLevel() == ms_level:
            mz_i_dict = {}
            mz_array = spectrum.get_peaks()[0]
            intensity_array = spectrum.get_peaks()[1]
            num_peaks_to_extract = min(num_peaks, len(mz_array))
            for mz, i in zip(mz_array[:num_peaks_to_extract], intensity_array[:num_peaks_to_extract]):
                mz_i_dict[mz] = i
            if mz_i_dict:
                key = f"{spectrum.getNativeID()}_{spectrum.getPrecursors()[0].getMZ()}"
                highest_i_dict[key] = mz_i_dict
                reducing_mass.append(float(key.split('_')[-1]))
                ret_time = spectrum.getRT()
                rts.append(ret_time / 60)
                if intensity:
                    inty = spectrum.getPrecursors()[0].getIntensity()
                    intensities.append(inty)
                precursor_charge = spectrum.getPrecursors()[0].getCharge()
                precursor_charges.append(precursor_charge)
                scan_numbers.append(spectrum.getNativeID())

    df_out = pd.DataFrame({
        'scan_number': scan_numbers,
        'reducing_mass': reducing_mass,
        'charge': precursor_charges,
        'peak_d': list(highest_i_dict.values()),
        'RT': rts,
    })

    if intensity:
        df_out['intensity'] = intensities

    scan_row1 = df_out['scan_number'].iloc[0]
    if isinstance(scan_row1, str) and 'scan=' in scan_row1:
        # Extract scan number from string as integer
        df_out['scan_number'] = df_out['scan_number'].apply(lambda x: x.split('=')[-1]).astype(int)
    elif isinstance(scan_row1, (int, float)):
        # Already numeric, just ensure integer type
        df_out['scan_number'] = df_out['scan_number'].astype(int)

    return df_out


def open_mzXML_pyopenms(filepath, num_peaks=1000, ms_level=2, intensity=True):
    highest_i_dict = defaultdict(dict)
    rts, intensities, reducing_mass, precursor_charges, scan_numbers = [], [], [], [], []

    # Create an MSExperiment object using PyOpenMS
    exp = pyopenms.MSExperiment()

    # Load the mzXML file into the MSExperiment object
    pyopenms.MzXMLFile().load(filepath, exp)

    for spectrum in exp:
        if spectrum.getMSLevel() == ms_level:
            mz_i_dict = {}
            mz_array = spectrum.get_peaks()[0]
            intensity_array = spectrum.get_peaks()[1]
            num_peaks_to_extract = min(num_peaks, len(mz_array))
            for mz, i in zip(mz_array[:num_peaks_to_extract], intensity_array[:num_peaks_to_extract]):
                mz_i_dict[mz] = i
            if mz_i_dict:
                key = f"{spectrum.getNativeID()}_{spectrum.getPrecursors()[0].getMZ()}"
                highest_i_dict[key] = mz_i_dict
                reducing_mass.append(float(key.split('_')[-1]))
                ret_time = spectrum.getRT()
                rts.append(ret_time/60)
                if intensity:
                    inty = spectrum.getPrecursors()[0].getIntensity()
                    intensities.append(inty)
                precursor_charge = spectrum.getPrecursors()[0].getCharge()
                precursor_charges.append(precursor_charge)
                scan_numbers.append(spectrum.getNativeID())

    df_out = pd.DataFrame({
        'scan_number': scan_numbers,
        'reducing_mass': reducing_mass,
        'charge' : precursor_charges,
        'peak_d': list(highest_i_dict.values()),
        'RT': rts,
    })

    if intensity:
        df_out['intensity'] = intensities

    scan_row1 = df_out['scan_number'].iloc[0]
    if isinstance(scan_row1, str) and 'scan=' in scan_row1:
        # Extract scan number from string as integer
        df_out['scan_number'] = df_out['scan_number'].apply(lambda x: x.split('=')[-1]).astype(int)
    elif isinstance(scan_row1, (int, float)):
        # Already numeric, just ensure integer type
        df_out['scan_number'] = df_out['scan_number'].astype(int)

    return df_out

def load_file(filepath, intensity=True):
    if filepath.endswith(".mzML"):
        return open_mzML_pyopenms(filepath, intensity=True)
    elif filepath.endswith(".mzXML"):
        return open_mzXML_pyopenms(filepath, intensity=True)
    else:
        raise ValueError("Unsupported file format. Supported formats: .mzML, .mzXML")
        
        
# Function to generate bin edges using constant bin size
def bin_linear(min_val, max_val, size):
    bin_edges = np.arange(min_val, max_val+size, size)

    return bin_edges


# Function to normalize RT values
def normalize_rt(group):
    # Convert RT to numeric, setting non-convertible values to NaN.
    # This is to allow RTs to be unspecified.
    rt_numeric = pd.to_numeric(group['RT'], errors='coerce')

    # Perform normalization
    max_rt = max(rt_numeric.max(), 30)
    normalized_values = rt_numeric / max_rt
    #group['RT'].loc[rt_numeric.notna()] = normalized_values
    group.loc[rt_numeric.notna(), "RT"] = normalized_values

    return group


# Function to bin RT
def rt_binning(rt_value, RT_bin_edges):
    if isinstance(rt_value, (int, float)):
        # Bin the numeric value
        binned_index = np.digitize([rt_value], RT_bin_edges)[0]
        return f'rt{binned_index}'
    else:
        # Return string value, for example unspecified
        return str(rt_value)


# Function to bin MZ
def mz_binning(mz_values, bin_edges):
    binned_indices = np.digitize(mz_values, bin_edges)
    return ['mz' + str(index) for index in binned_indices]


# Function to bin peaks
def peak_binning(peak_values, bin_edges):
    binned_indices = np.digitize(peak_values, bin_edges)
    return ['pk' + str(index) for index in binned_indices]


# Function to process, normalize, sort the dictionary, and bin both keys and values
def process_peak_d(dict_string, mz_bin_edges, peak_bin_edges, threshold):
    try:
        # convert peak_d to dictionary
        #dict_data = ast.literal_eval(dict_string)
        dict_data = dict_string
    except ValueError:
        return [], []

    # Normalize and threshold peak intensity
    total = sum(dict_data.values())
    normalized_dict = {k: v / total for k, v in dict_data.items() if (v / total) > threshold}

    # Sort and bin the peaks
    sorted_dict = dict(sorted(normalized_dict.items(), key=lambda item: item[1], reverse=True))
    mzs = list(sorted_dict.keys())
    peaks = list(sorted_dict.values())
    binned_mzs = mz_binning(mzs, mz_bin_edges)
    binned_peaks = peak_binning(peaks, peak_bin_edges)

    return binned_mzs, binned_peaks

# Function to process data
def process_data(df, RT_bin_edges, mz_bin_edges, peak_bin_edges, threshold):

    # Normalize RTs from each file
    df = df.groupby('filename', group_keys=False).apply(normalize_rt)

    # Get RT bin indices
    df['binned_RT'] = df['RT'].apply(lambda x: rt_binning(x, RT_bin_edges))

    # Get m/z bin index for precursor mass
    df['binned_mass'] = df['reducing_mass'].apply(lambda x: 'mz' + str(np.digitize(x, mz_bin_edges)))

    # Process peak_d column
    df['processed_peak_d'] = df['peak_d'].apply(lambda x: process_peak_d(x, mz_bin_edges, peak_bin_edges, threshold))

    # Get m/z and peak bin indices
    df['binned_mz'] = df['processed_peak_d'].apply(lambda x: x[0])
    df['binned_peak'] = df['processed_peak_d'].apply(lambda x: x[1])

    return df

# Function to generate corpus
def generate_corpus_mz(df):

    # Construct sentences
    corpus = pd.DataFrame()
    corpus['sentence'] = df[['lc', 'mode', 'ionization', 'modification', 'trap', 'fragmentation', 'glycan_type', 'binned_RT', 'binned_mass']].agg(' '.join, axis=1)
    corpus['sentence'] += ' ' + df['binned_mz'].apply(' '.join)

    return corpus['sentence'].tolist()


def calculate_file_comps_modified(masses_to_check,mass_threshold,df_in,mass_tag,mode='negative',
                         max_charge = 2, mass_value='monoisotopic',sample_prep='underivatized',
                         filter_out = None):
    mode_dict = {'negative':1.0078,'positive':3.0234}
    mass_modifier = mode_dict[mode]
    if mass_tag:
        mass_modifier+= mass_tag
    all_comps = [x for x in df_in.groupby('comp_str').first()['Composition']]
    comps_in = [x for x in copy.deepcopy(all_comps)]
    comp_masses = np.array([composition_to_mass(x, mass_value = mass_value, sample_prep = sample_prep)+mass_modifier for x in comps_in])
    comps_out = [(None,0) for x in masses_to_check]
    comps_in_copy = [x for x in copy.deepcopy(all_comps)]
    for charge in range(1,max_charge+1):
        chunked_calc_comps =[]
        charged_comp_masses = (comp_masses - (charge-1)*1.0078) / charge
        for mz_chunk in np.array_split(masses_to_check,max(1,len(masses_to_check)//1000)):
            compositions_out = mz_to_comp_vect(mz_chunk, charged_comp_masses,comps_in_copy,mass_threshold)
            chunked_calc_comps.extend(compositions_out)
        comps_out = [(y,charge) if (not x[0] and y) else x for x,y in zip(comps_out,chunked_calc_comps)]
    return comps_out


def create_struct_map_modified(df_glycan,glycan_class,filter_out=None,phylo_level = 'Kingdom',phylo_filter= 'Animalia'):
    processed_df_use = df_glycan[df_glycan[f"{phylo_level}"].apply(lambda x: phylo_filter in x)&(df_glycan['glycan_type']== glycan_class)]
    if filter_out:
        processed_df_use = processed_df_use.iloc[[i for i,x in enumerate(processed_df_use.Composition) if not filter_out.intersection(x)]]
    processed_df_use = processed_df_use.assign(comp_str=[comp_to_str_comp(x) for x in processed_df_use.Composition])
    processed_df_use = processed_df_use.assign(ref_counts=processed_df_use.loc[:,'ref'].map(len)+processed_df_use.loc[:,'tissue_ref'].map(len)+processed_df_use.loc[:,'disease_ref'].map(len))
    processed_df_use = processed_df_use[~(processed_df_use['glycan'].str.contains("}"))]
    processed_df_use = processed_df_use.sort_values('ref_counts',ascending=False)
    processed_df_use = processed_df_use.assign(topology= [structure_to_basic(x) for x in processed_df_use['glycan']])
    processed_df_use = processed_df_use.assign(glycan=[x.replace('-ol', '').replace('1Cer', '') for x in processed_df_use.glycan])
    small_comps = [x for x in processed_df_use['comp_str'] if x if sum([int(p[-1]) for p in x.split('$')])<6]
    small_comps = processed_df_use[processed_df_use['comp_str'].isin(small_comps)]
    df_use_unq_topos = small_comps.groupby('topology').first().groupby('comp_str').agg(list)
    topology_map = dict(zip(df_use_unq_topos.index,df_use_unq_topos.glycan))
    df_use_unq_comps = processed_df_use.groupby('comp_str').first()
    common_struct_map = dict(zip(df_use_unq_comps.index,df_use_unq_comps.glycan))
    return common_struct_map,processed_df_use,topology_map


def assign_candidate_structures_modified(df_in,df_glycan_in,comp_struct_map,topo_struct_map,mass_tolerance,mode,mass_tag):
    red_masses = np.array(df_in.reducing_mass)
    max_charge = df_in['charge'].max()
    comps_out = calculate_file_comps_modified(red_masses,mass_tolerance,df_glycan_in,mass_tag,mode=mode, max_charge = max_charge)
    df_in['composition'] = [x[0] for x in comps_out]
    df_in['charge'] = [x[1] if x[0] else None for x in comps_out]
    # This is so that I can explode the df by composition and candidate structure (definitely a more concise way of doing this)
    candidate_data = []
    for matched_comps_str,matched_comps in [([comp_to_str_comp(y) for y in x],x) if x else (x,x) for x in df_in.composition]:
        if not matched_comps:
            candidate_data.append(([None], [None]))
        else:
            structures = [s for comp_str in matched_comps_str for s in topo_struct_map.get(comp_str, [comp_struct_map[comp_str]])]
            compositions = [comp for comp,comp_str in zip(matched_comps,matched_comps_str) for _ in topo_struct_map.get(comp_str, [comp_struct_map[comp_str]])]
            candidate_data.append((structures, compositions))
    df_in['candidate_structure'], df_in['composition'] = zip(*candidate_data)
    #df_in = df_in.explode(['composition','candidate_structure']).reset_index(names='spec_id')
    #df_in['compositional_vector'] = [comp_to_input_vect(x) if x else None for x in df_in.composition]
    df_in = df_in[df_in['composition'].apply(lambda x: isinstance(x, list) and any(item is not None for item in x))]
    df_in = df_in.copy()
    df_in.drop(['composition', 'candidate_structure'], axis = 1, inplace = True)
    return df_in


def calculate_annotation_scores_bert(df_glycobert, multiplier, mass_tag, mass_tolerance):
    # Initialize a column to store annotation scores
    df_glycobert['annotation_scores'] = None

    # Iterate through each row in the dataframe
    for idx, row in df_glycobert.iterrows():
        try:
            # Extract the list of predicted glycans (tuples) and the charge for the current row
            predicted_glycans = row['predictions']  # List of tuples (glycan, confidence)
            row_charge = row['charge'][0]
            peak_d = row['peak_d']  # Assuming you have a column with peak data

            # Initialize a list to store scores for each glycan
            glycan_scores = []

            for glycan, confidence in predicted_glycans:
                if '][GlcNAc(b1-4)]' in glycan or '{' in glycan:
                    continue  # Skip invalid glycans

                # Deisotope the peaks for the current glycan
                rounded_mass_rows = [np.round(y, 1) for y in deisotope_ms2(peak_d, int(abs(row_charge)), 0.05)][:15]
                unq_rounded_masses = set(rounded_mass_rows)

                # Use CandyCrumbs to calculate scores
                cc_out = CandyCrumbs(
                    glycan, unq_rounded_masses, mass_tolerance,
                    simplify=False, charge=int(multiplier * abs(row_charge)),
                    disable_global_mods=True, disable_X_cross_rings=True,
                    max_cleavages=2, mass_tag=mass_tag
                )

                # Score the top fragment masses
                tester_mass_scores = score_top_frag_masses(cc_out)

                # Calculate the total score for the glycan
                glycan_score = sum([tester_mass_scores.get(mass, 0) for mass in rounded_mass_rows])

                # Append the result as a tuple
                glycan_scores.append(((glycan, confidence), glycan_score))

            # Store the scores in the dataframe
            df_glycobert.at[idx, 'annotation_scores'] = glycan_scores

        except (IndexError, KeyError, AttributeError) as e:
            # Handle exceptions gracefully
            print(f"Error processing row {idx}: {e}")
            df_glycobert.at[idx, 'annotation_scores'] = None

    return df_glycobert


def deduplicate_predictions_bert(df, mz_diff = 0.5, rt_diff = 1.0):
    """removes/unifies duplicate predictions\n
   | Arguments:
   | :-
   | df (dataframe): df_out generated within wrap_inference
   | mz_diff (float): mass tolerance for assigning spectra to the same peak; default:0.5
   | rt_diff (float): retention time tolerance (in minutes) for assigning spectra to the same peak; default:1.0\n
   | Returns:
   | :-
   | Returns a deduplicated dataframe
   """
    # Sort by index and 'RT'
    df.set_index('reducing_mass', inplace=True)
    df.sort_values(by = 'RT', inplace = True)
    df.sort_index(inplace = True)
    max_conf_rows = []
    # Loop through the DataFrame to find duplicates
    for idx, row in df.iterrows():
        # Set a mask for close enough index values and RT values
        mask = (np.abs(df.index - idx) < mz_diff) & (np.abs(df['RT_original'] - row['RT_original']) < rt_diff)
        # Filter DataFrame based on mask
        sub_df = df[mask]
        # Get the first prediction from the tuple
        first_pred = row['predictions'][0][0] if row['predictions'] else None
        if first_pred is None:
          max_conf_rows.append(row)
          continue
        # Filter sub_df based on the first prediction value
        pred_mask = sub_df['predictions'].apply(lambda x: x[0][0] if x else None) == first_pred
        # Choose the row with the max confidence for this prediction
        max_conf_row = sub_df.loc[pred_mask].iloc[np.argmax([p[0][1] for p in sub_df.loc[pred_mask, 'predictions']])]
        if 'rel_abundance' in df.columns:
          # Sum 'rel_abundance' for this subset
          summed_abundance = np.nansum(sub_df.loc[pred_mask, 'rel_abundance'])
          # Update 'rel_abundance'
          max_conf_row['rel_abundance'] = summed_abundance
        # Store in max_conf_rows
        max_conf_rows.append(max_conf_row)
    dedup_df = pd.DataFrame(max_conf_rows,columns = df.columns)
    dedup_df = dedup_df.astype(dict(df.dtypes))
    # Drop duplicate rows based on index and 'predictions'
    dedup_df = dedup_df[~dedup_df.index.duplicated(keep='first')]
    # dedup_df.drop_duplicates(subset=['RT'], keep = 'first', inplace = True)
    #dedup_df.reset_index(inplace=True)
    return dedup_df


def combine_charge_states_bert(df_out):
    """looks for several charges at the same RT with the same top prediction and combines their relative abundances\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference\n
    | Returns:
    | :-
    | Returns prediction dataframe where the singly-charged state now carries the sum of abundances
    """
    df_out['top_pred'] = [k[0][0] if len(k) > 0 else np.nan for k in df_out.predictions]
    repeated_top_pred = df_out['top_pred'].value_counts()
    repeated_top_pred = repeated_top_pred[repeated_top_pred > 1].index.tolist()
    filtered_top_pred = []
    for pred in repeated_top_pred:
        charge_values = df_out[df_out['top_pred'] == pred]['charge']
        if abs(charge_values.max() - charge_values.min()) >= 1:
            filtered_top_pred.append(pred)
    df_filtered = df_out[df_out['top_pred'].isin(filtered_top_pred)].copy()
    for pred in filtered_top_pred:
        idx = df_filtered.index[len(df_filtered) - 1 - df_filtered.top_pred.values.tolist()[::-1].index(pred)]
        idx_rt = df_filtered.loc[idx, 'RT_original']
        for k, row in df_filtered.iloc[:df_filtered.index.get_loc(idx)][::-1].iterrows():
            if row['top_pred'] == pred and abs(row['RT_original'] -idx_rt) < 1:
                df_filtered.at[idx, 'rel_abundance'] += row['rel_abundance']
                df_filtered.drop(k, inplace = True)
    df_out = pd.concat([df_out[~df_out['top_pred'].isin(filtered_top_pred)], df_filtered]).sort_index()
    df_out.drop(['top_pred'], axis = 1 , inplace = True)
    #df_out.reset_index(inplace=True)
    return df_out


def calculate_annotation_scores_bart(df_glycobart, multiplier, mass_tag, MS2_tolerance):
    # Initialize a column to store annotation scores
    df_glycobart['annotation_scores'] = None

    # Iterate through each row in the dataframe
    for idx, row in df_glycobart.iterrows():
        try:
            # Extract the list of predicted glycans (tuples) and the charge for the current row
            predicted_glycans = row['predictions']  # List of glycans
            row_charge = row['charge'][0]
            peak_d = row['peak_d']  # Assuming you have a column with peak data

            # Initialize a list to store scores for each glycan
            glycan_scores = []

            for glycan in predicted_glycans:
                if '][GlcNAc(b1-4)]' in glycan or '{' in glycan:
                    continue  # Skip invalid glycans

                # Deisotope the peaks for the current glycan
                rounded_mass_rows = [np.round(y, 1) for y in deisotope_ms2(peak_d, int(abs(row_charge)), 0.05)][:15]
                unq_rounded_masses = set(rounded_mass_rows)

                # Use CandyCrumbs to calculate scores
                cc_out = CandyCrumbs(
                    glycan, unq_rounded_masses, MS2_tolerance,
                    simplify=False, charge=int(multiplier * abs(row_charge)),
                    disable_global_mods=True, disable_X_cross_rings=True,
                    max_cleavages=2, mass_tag=mass_tag
                )

                # Score the top fragment masses
                tester_mass_scores = score_top_frag_masses(cc_out)

                # Calculate the total score for the glycan
                glycan_score = sum([tester_mass_scores.get(mass, 0) for mass in rounded_mass_rows])

                # Append the result as a tuple
                glycan_scores.append((glycan, glycan_score))

            # Store the scores in the dataframe
            df_glycobart.at[idx, 'annotation_scores'] = glycan_scores

        except (IndexError, KeyError, AttributeError) as e:
            # Handle exceptions gracefully
            print(f"Error processing row {idx}: {e}")
            df_glycobart.at[idx, 'annotation_scores'] = None

    return df_glycobart


def deduplicate_predictions_bart(df, mz_diff=0.5, rt_diff=1.0):
    df = df.copy()
    df.set_index('reducing_mass', inplace=True)
    df.sort_values(by='RT', inplace=True)
    df.sort_index(inplace=True)
    dedup_rows = []
    visited = set()
    for idx, row in df.iterrows():
        if idx in visited:
            continue
        first_glycan = row['predictions'][0] if row['predictions'] else None
        mask = (
            (abs(df.index - idx) < mz_diff) &
            (abs(df['RT_original'] - row['RT_original']) < rt_diff) &
            (df['predictions'].apply(lambda x: x[0] if x else None) == first_glycan)
        )
        group = df[mask]
        visited.update(group.index)
        # Pick the first row as representative
        rep_row = group.iloc[0].copy()
        if 'rel_abundance' in df.columns:
            rep_row['rel_abundance'] = group['rel_abundance'].sum()
        dedup_rows.append(rep_row)
    dedup_df = pd.DataFrame(dedup_rows, columns=df.columns)
    dedup_df = dedup_df.astype(dict(df.dtypes))
    dedup_df = dedup_df[~dedup_df.index.duplicated(keep='first')]
    return dedup_df


def combine_charge_states_bart(df_out):
    """looks for several charges at the same RT with the same top prediction and combines their relative abundances\n
    | Arguments:
    | :-
    | df_out (dataframe): prediction dataframe generated within wrap_inference\n
    | Returns:
    | :-
    | Returns prediction dataframe where the singly-charged state now carries the sum of abundances
    """
    df_out['top_pred'] = [k[0] if len(k) > 0 else np.nan for k in df_out.predictions]
    repeated_top_pred = df_out['top_pred'].value_counts()
    repeated_top_pred = repeated_top_pred[repeated_top_pred > 1].index.tolist()
    filtered_top_pred = []
    for pred in repeated_top_pred:
        charge_values = df_out[df_out['top_pred'] == pred]['charge']
        if abs(charge_values.max() - charge_values.min()) >= 1:
            filtered_top_pred.append(pred)
    df_filtered = df_out[df_out['top_pred'].isin(filtered_top_pred)].copy()
    for pred in filtered_top_pred:
        idx = df_filtered.index[len(df_filtered) - 1 - df_filtered.top_pred.values.tolist()[::-1].index(pred)]
        idx_rt = df_filtered.loc[idx, 'RT_original']
        for k, row in df_filtered.iloc[:df_filtered.index.get_loc(idx)][::-1].iterrows():
            if row['top_pred'] == pred and abs(row['RT_original'] -idx_rt) < 1:
                df_filtered.at[idx, 'rel_abundance'] += row['rel_abundance']
                df_filtered.drop(k, inplace = True)
    df_out = pd.concat([df_out[~df_out['top_pred'].isin(filtered_top_pred)], df_filtered]).sort_index()
    df_out.drop(['top_pred'], axis = 1 , inplace = True)
    #df_out.reset_index(inplace=True)
    return df_out


def enforce_class_modified(glycan, glycan_class):
  """given a glycan and glycan class, determines whether glycan is from this class\n
  | Arguments:
  | :-
  | glycan (string): glycan in IUPAC-condensed nomenclature
  | glycan_class (string): glycan class in form of "O", "N", "free", or "lipid"
  | Returns:
  | :-
  | Returns True if glycan is in glycan class and False if not
  """
  pools = {
    'O': ['GalNAc', 'GalNAcOS', 'GalNAc4S', 'GalNAc6S', 'Man', 'Fuc', 'Gal', 'GlcNAc', 'GlcNAcOS', 'GlcNAc6S'],
    'N': ['GlcNAc(b1-4)GlcNAc', '[Fuc(a1-6)]GlcNAc'],
    'free': ['Glc', 'GlcOS', 'Glc3S', 'GlcNAc', 'GlcNAcOS', 'Gal', 'GalOS', 'Gal3S', 'Ins'],
    'lipid': ['Glc', 'GlcOS', 'Glc3S', 'GlcNAc', 'GlcNAcOS', 'Gal', 'GalOS', 'Gal3S', 'Ins'],
    }
  glycan = glycan[:-3] if glycan.endswith('-ol') else glycan
  pool = pools.get(glycan_class, [])
  truth = any([glycan.endswith(k) for k in pool])
  if truth and glycan_class in {'free', 'lipid', 'O'}:
    truth = not any(glycan.endswith(k) for k in ['GlcNAc(b1-4)GlcNAc', '[Fuc(a1-6)]GlcNAc'])

  return truth


def mass_check_modified_ppm(mass, glycan, mode = 'negative', modification = 'underivatized', mass_tag = 0,
               double_thresh = 900, triple_thresh = 1500, quadruple_thresh = 3500, mass_thresh = 10):

    adduct_mass_dict_charge_1 = {'M+H': 1.00728, 'M+NH4': 18.03383, 'M+Na': 22.98922, 'M+K': 38.96316, 'M+2Na-H': 44.97116, 'M+2K-H': 76.91904, 'M+H-H20': -17.00273, 'M-H20-H': -19.01894, 'M-H': -1.00728, 'M+NA-2H': 20.97467, 'M+K-2H': 36.94861 }
    adduct_mass_dict_charge_2 = {'M+2H': 1.00728, 'M+H+NH4': 9.52055, 'M+H+Na': 11.99825, 'M+H+K': 19.98522, 'M+2Na': 22.98922, 'M+2K': 38.96316, 'M-2H': -1.00728, 'M+Na-3H': 9.98369 }
    adduct_mass_dict_charge_3 = {'M+3H': 1.00728, 'M-3H': -1.00728}
    adduct_mass_dict_charge_4 = {'M+4H': 1.00728}
    modification_mass_dict_glycobart = {'reduced': 20.02621, '2AA': 139.06333, '2AB': 138.07931, 'permethylated': 46.04186, 'permethylated and reduced': 47.04969, 'spacer': 90.07931 }

    try:
        mz = glycan_to_mass(glycan, sample_prep= 'underivatized') if isinstance(glycan, str) else glycan
        #print('mz:', mz)
    except:
        return False

    if not modification == 'underivatized':
        mz = mz - 18.01056 # remove mass of H2O
        #print('mz - water:', mz)
        mz += modification_mass_dict_glycobart.get(modification, mass_tag)
        #print('mz + modification:', mz)
    thresholds = [double_thresh, triple_thresh, quadruple_thresh]
    #print('threshold:', thresholds)

    #adduct_list_charge1 = ['M+H', 'M+NH4', 'M+Na', 'M+K', 'M+2Na-H', 'M+2K-H', 'M+H-H20'] if mode == 'positive' else ['M-H20-H', 'M-H', 'M+NA-2H', 'M+K-2H']   #all adducts included
    adduct_list_charge1 = ['M+H', 'M+NH4', 'M+Na', 'M+K', 'M+2Na-H', 'M+2K-H', 'M+H-H20'] if mode == 'positive' else ['M-H']  #only 'H' adducts
    #print('adduct_list_charge1:', adduct_list_charge1)
    if mode == 'positive':
       og_list = [mz] + [mz + adduct_mass_dict_charge_1.get(adduct, 999) for adduct in adduct_list_charge1] + [(2*mz + 22.98922), (2*mz + 1.00728)]
    else:
       og_list = [mz] + [mz + adduct_mass_dict_charge_1.get(adduct, 999) for adduct in adduct_list_charge1]
    #print('og_list:', og_list)

    #adduct_list_charge2 = ['M+2H', 'M+H+NH4', 'M+H+Na', 'M+H+K', 'M+2Na', 'M+2K'] if mode == 'positive' else ['M-2H', 'M+Na-3H']   #all adducts included
    adduct_list_charge2 = ['M+2H', 'M+H+NH4', 'M+H+Na', 'M+H+K', 'M+2Na', 'M+2K'] if mode == 'positive' else ['M-2H']   #only 'H' adducts
    adduct_mass_charge2 = [adduct_mass_dict_charge_2.get(adduct, 999) for adduct in adduct_list_charge2]

    adduct_list_charge3 = ['M+3H'] if mode == 'positive' else ['M-3H']
    adduct_mass_charge3 = [adduct_mass_dict_charge_3.get(adduct, 999) for adduct in adduct_list_charge3]

    adduct_list_charge4 = ['M+4H'] if mode == 'positive' else []
    adduct_mass_charge4 = [adduct_mass_dict_charge_4.get(adduct, 999) for adduct in adduct_list_charge4]

    adduct_masses = [adduct_mass_charge2, adduct_mass_charge3, adduct_mass_charge4 ]

    mz_list = og_list + [
        (m / z + adduct_adjust)
        for z, threshold, adduct_list in zip([2, 3, 4], thresholds, adduct_masses)
        for adduct_adjust in adduct_list
        for m in og_list
    ]
    #print('mz_list:', mz_list)
    return [m for m in mz_list if abs(mass - m) * 1e6 / mass < mass_thresh]


def mass_check_modified(mass, glycan, mode = 'negative', modification = 'underivatized', mass_tag = 0,
               double_thresh = 900, triple_thresh = 1500, quadruple_thresh = 3500, mass_thresh = 0.5):

    adduct_mass_dict_charge_1 = {'M+H': 1.00728, 'M+NH4': 18.03383, 'M+Na': 22.98922, 'M+K': 38.96316, 'M+2Na-H': 44.97116, 'M+2K-H': 76.91904, 'M+H-H20': -17.00273, 'M-H20-H': -19.01894, 'M-H': -1.00728, 'M+NA-2H': 20.97467, 'M+K-2H': 36.94861 }
    adduct_mass_dict_charge_2 = {'M+2H': 1.00728, 'M+H+NH4': 9.52055, 'M+H+Na': 11.99825, 'M+H+K': 19.98522, 'M+2Na': 22.98922, 'M+2K': 38.96316, 'M-2H': -1.00728, 'M+Na-3H': 9.98369 }
    adduct_mass_dict_charge_3 = {'M+3H': 1.00728, 'M-3H': -1.00728}
    adduct_mass_dict_charge_4 = {'M+4H': 1.00728}
    modification_mass_dict_glycobart = {'reduced': 20.02621, '2AA': 139.06333, '2AB': 138.07931, 'permethylated': 46.04186, 'permethylated and reduced': 47.04969, 'spacer': 90.07931 }

    try:
        mz = glycan_to_mass(glycan, sample_prep= 'underivatized') if isinstance(glycan, str) else glycan
        #print('mz:', mz)
    except:
        return False

    if not modification == 'underivatized':
        mz = mz - 18.01056 # remove mass of H2O
        #print('mz - water:', mz)
        mz += modification_mass_dict_glycobart.get(modification, mass_tag)
        #print('mz + modification:', mz)
    thresholds = [double_thresh, triple_thresh, quadruple_thresh]
    #print('threshold:', thresholds)

    #adduct_list_charge1 = ['M+H', 'M+NH4', 'M+Na', 'M+K', 'M+2Na-H', 'M+2K-H', 'M+H-H20'] if mode == 'positive' else ['M-H20-H', 'M-H', 'M+NA-2H', 'M+K-2H']   #all adducts included
    adduct_list_charge1 = ['M+H'] if mode == 'positive' else ['M-H']  #only 'H' adducts
    #print('adduct_list_charge1:', adduct_list_charge1)
    if mode == 'positive':
       og_list = [mz] + [mz + adduct_mass_dict_charge_1.get(adduct, 999) for adduct in adduct_list_charge1] + [(2*mz + 22.98922), (2*mz + 1.00728)]
    else:
       og_list = [mz] + [mz + adduct_mass_dict_charge_1.get(adduct, 999) for adduct in adduct_list_charge1]
    #print('og_list:', og_list)

    #adduct_list_charge2 = ['M+2H', 'M+H+NH4', 'M+H+Na', 'M+H+K', 'M+2Na', 'M+2K'] if mode == 'positive' else ['M-2H', 'M+Na-3H']   #all adducts included
    adduct_list_charge2 = ['M+2H'] if mode == 'positive' else ['M-2H']   #only 'H' adducts
    adduct_mass_charge2 = [adduct_mass_dict_charge_2.get(adduct, 999) for adduct in adduct_list_charge2]

    adduct_list_charge3 = ['M+3H'] if mode == 'positive' else ['M-3H']
    adduct_mass_charge3 = [adduct_mass_dict_charge_3.get(adduct, 999) for adduct in adduct_list_charge3]

    adduct_list_charge4 = ['M+4H'] if mode == 'positive' else []
    adduct_mass_charge4 = [adduct_mass_dict_charge_4.get(adduct, 999) for adduct in adduct_list_charge4]

    adduct_masses = [adduct_mass_charge2]
    #print('adduct mass:', adduct_masses)
    mz_list = og_list + [
        (m / z + adduct_adjust) for z, threshold, adduct_list in zip([2, 3, 4], thresholds, adduct_masses) for adduct_adjust in adduct_list
        #for m in og_list if m > threshold
        for m in og_list
    ]
    #print('mz_list:', mz_list)
    return [m for m in mz_list if abs(mass - m) < mass_thresh]


def get_top_fragments(row, threshold=0.8):
    peak_d = row['peak_d']  # Assuming 'peak_d' is the column containing the peak_d dictionary
    sorted_items = sorted(peak_d.items(), key=lambda x: x[1], reverse=True)

    total_values = sum(peak_d.values())
    cumulative_sum = 0
    top_keys = []

    for key, value in sorted_items:
        cumulative_sum += value
        key = round(key, 4)
        top_keys.append(key)
        if cumulative_sum / total_values >= threshold:
            break

    return top_keys


def domain_filter_modified_bert(df_out, glycan_class='N', mode='negative', modification='reduced',
                                    MS2_tolerance=0.5, filter_diag_ion=['Neu5Gc', 'Kdn', 'S'], min_ion_match = 1):
    multiplier = -1 if mode == 'negative' else 1
    df_out['ion_match_score'] = [[] for _ in range(len(df_out))]
    df_out['diagnostic_ions'] = [[] for _ in range(len(df_out))]
    df_out['evidence_level'] = [[] for _ in range(len(df_out))]
    m_fuc = 146.0579

    for k in range(len(df_out)):

        keep = []
        current_preds = df_out['predictions'].iloc[k] if len(df_out['predictions'].iloc[k]) > 0 else [''.join(list(df_out['composition'].iloc[k].keys()))]
        to_append = len(df_out['predictions'].iloc[k]) > 0
        match_list = []
        diagnostic_ions = []  # Initialize evidence_levels as a list
        evidence_levels = []
        #charge = df_out['charge'].iloc[k]

        for i, m in enumerate(current_preds):
            m = m[0]
            truth_comp = []  # Must be all True for passing postprocessing
            truth_diag = []   # diagnostic ion True values
            composition = glycan_to_composition(m)

            # Check if any of the filter_diag_ion values are present in the composition dict keys
            if any(filter_key in composition.keys() for filter_key in filter_diag_ion):
                truth_comp.append(False)
                continue

            charge = round(composition_to_mass(composition) / df_out.index[k]) * multiplier
            c = abs(charge)
            addy = charge * multiplier - 1
            assumed_mass = df_out.index[k] * c + charge
            cmasses = np.array(combinatorics(composition))
            top_fragments = np.array(df_out['top_fragments'].iloc[k][:20])
            match_count = 0
            found_match = False
            for top_fragment in top_fragments:
                for cmass in cmasses:
                    if mass_check_modified(top_fragment, cmass, mode = mode, modification = modification, mass_thresh=MS2_tolerance):
                        match_count += 1
                        break
            if match_count >= min_ion_match:
                found_match = True

            match_score = match_count / len(top_fragments)
            match_list.append(match_score)

            if found_match:
                truth_comp.append(True)
            else:
                truth_comp.append(False)

            # Check diagnostic ions
            if 'Neu5Ac' in m:
                truth_diag.append(any([abs(mass_dict['Neu5Ac'] + (1.0078 * multiplier) - j) < 1 or
                                       abs(assumed_mass - mass_dict['Neu5Ac'] - j) < 1 or
                                       abs(df_out.index.tolist()[k] - ((mass_dict['Neu5Ac'] - addy) / c) - j) < 1 or
                                       abs(mass_dict['Neu5Ac'] - mass_dict['H2O'] + (1.0078 * multiplier) - j) < 1 or
                                       abs(assumed_mass - (mass_dict['Neu5Ac'] - mass_dict['H2O']) - j) < 1 or
                                       abs(df_out.index.tolist()[k] - ((mass_dict['Neu5Ac'] - mass_dict['H2O'] - addy) / c) - j) < 1
                                       for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))
            if 'Neu5Gc' in m:
                truth_diag.append(any([abs(mass_dict['Neu5Gc'] + (1.0078 * multiplier) - j) < 1 or abs(assumed_mass - mass_dict['Neu5Gc'] - j) < 1 or
                                       abs(df_out.index.tolist()[k] - ((mass_dict['Neu5Gc'] - addy) / c) - j) < 1
                                       for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))

            if 'GlcNAc' in m:
                truth_diag.append(any([abs(mass_dict['HexNAc'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((mass_dict['HexNAc'] - addy) / c) - j) < 1 or
                                      abs(2 * mass_dict['HexNAc'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((2 * mass_dict['HexNAc'] - addy) / c) - j) < 1 or
                                      abs(mass_dict['HexNAc'] + mass_dict['Hex'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((mass_dict['HexNAc'] + mass_dict['Hex'] - addy) / c) - j) < 1 or
                                      abs(2 * mass_dict['HexNAc'] + mass_dict['Hex'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((2 * mass_dict['HexNAc'] + mass_dict['Hex'] - addy) / c) - j) < 1
                                      for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))

            if 'Fuc' in m:
                truth_diag.append(any([abs(m_fuc + (1.0078 * multiplier) - j) < 1 or abs(mass_dict['HexNAc'] + m_fuc + (1.0078 * multiplier) - j) < 1 or
                                      abs(2*mass_dict['HexNAc'] + m_fuc + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((m_fuc - addy) / c) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((mass_dict['HexNAc'] + m_fuc - addy) / c) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((2*mass_dict['HexNAc'] + m_fuc - addy) / c) - j) < 1
                                       for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))

            if all(truth_comp):

                # Append evidence level to the list based on match_score
                if all(truth_diag):
                    diagnostic_ion = "all diagnostic ions present"
                    evidence_level = "strong"
                elif not any(truth_diag):
                    diagnostic_ion = "no diagnostic ion present"
                    evidence_level = "weak"
                else:
                    diagnostic_ion = "some diagnostic ions present"
                    evidence_level = "moderate"

                if evidence_level in ["strong", "moderate"]:
                    if to_append:
                        keep.append(current_preds[i])
                    evidence_levels.append(evidence_level)
                    diagnostic_ions.append(diagnostic_ion)
                    match_list.append(match_score)
            else:
                if not to_append:
                    keep.append('remove')
                    diagnostic_ions.append("NA")
                    evidence_levels.append("NA")
                    match_list.append(0)

        # Use .loc instead of .at to assign values
        df_out.iat[k, 0] = keep
        df_out.iat[k, -3] = match_list
        df_out.iat[k, -2] = diagnostic_ions
        df_out.iat[k, -1] = evidence_levels

    return df_out[df_out['predictions'].apply(lambda x: 'remove' not in x[:1])]


def domain_filter_modified_bart(df_out, glycan_class='N', mode='negative', modification='reduced',
                                    MS2_tolerance=0.5, filter_diag_ion=['Neu5Gc', 'Kdn', 'S'], min_ion_match = 1):
    multiplier = -1 if mode == 'negative' else 1
    df_out['ion_match_score'] = [[] for _ in range(len(df_out))]
    df_out['diagnostic_ions'] = [[] for _ in range(len(df_out))]
    df_out['evidence_level'] = [[] for _ in range(len(df_out))]
    m_fuc = 146.0579

    for k in range(len(df_out)):

        keep = []
        current_preds = df_out['predictions'].iloc[k] if len(df_out['predictions'].iloc[k]) > 0 else [''.join(list(df_out['composition'].iloc[k].keys()))]
        to_append = len(df_out['predictions'].iloc[k]) > 0
        match_list = []
        diagnostic_ions = []  # Initialize evidence_levels as a list
        evidence_levels = []
        #charge = df_out['charge'].iloc[k]

        for i, m in enumerate(current_preds):
            #m = m[0]
            truth_comp = []  # Must be all True for passing postprocessing
            truth_diag = []   # diagnostic ion True values
            composition = glycan_to_composition(m)

            # Check if any of the filter_diag_ion values are present in the composition dict keys
            if any(filter_key in composition.keys() for filter_key in filter_diag_ion):
                truth_comp.append(False)
                continue

            charge = round(composition_to_mass(composition) / df_out.index[k]) * multiplier
            c = abs(charge)
            addy = charge * multiplier - 1
            assumed_mass = df_out.index[k] * c + charge
            cmasses = np.array(combinatorics(composition))
            top_fragments = np.array(df_out['top_fragments'].iloc[k][:20])
            match_count = 0
            found_match = False
            for top_fragment in top_fragments:
                for cmass in cmasses:
                    if mass_check_modified(top_fragment, cmass, mode = mode, modification = modification, mass_thresh=MS2_tolerance):
                        match_count += 1
                        break
            if match_count >= min_ion_match:
                found_match = True

            match_score = match_count / len(top_fragments)
            match_list.append(match_score)

            if found_match:
                truth_comp.append(True)
            else:
                truth_comp.append(False)

            # Check diagnostic ions
            if 'Neu5Ac' in m:
                truth_diag.append(any([abs(mass_dict['Neu5Ac'] + (1.0078 * multiplier) - j) < 1 or
                                       abs(assumed_mass - mass_dict['Neu5Ac'] - j) < 1 or
                                       abs(df_out.index.tolist()[k] - ((mass_dict['Neu5Ac'] - addy) / c) - j) < 1 or
                                       abs(mass_dict['Neu5Ac'] - mass_dict['H2O'] + (1.0078 * multiplier) - j) < 1 or
                                       abs(assumed_mass - (mass_dict['Neu5Ac'] - mass_dict['H2O']) - j) < 1 or
                                       abs(df_out.index.tolist()[k] - ((mass_dict['Neu5Ac'] - mass_dict['H2O'] - addy) / c) - j) < 1
                                       for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))
            if 'Neu5Gc' in m:
                truth_diag.append(any([abs(mass_dict['Neu5Gc'] + (1.0078 * multiplier) - j) < 1 or abs(assumed_mass - mass_dict['Neu5Gc'] - j) < 1 or
                                       abs(df_out.index.tolist()[k] - ((mass_dict['Neu5Gc'] - addy) / c) - j) < 1
                                       for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))

            if 'GlcNAc' in m:
                truth_diag.append(any([abs(mass_dict['HexNAc'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((mass_dict['HexNAc'] - addy) / c) - j) < 1 or
                                      abs(2 * mass_dict['HexNAc'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((2 * mass_dict['HexNAc'] - addy) / c) - j) < 1 or
                                      abs(mass_dict['HexNAc'] + mass_dict['Hex'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((mass_dict['HexNAc'] + mass_dict['Hex'] - addy) / c) - j) < 1 or
                                      abs(2 * mass_dict['HexNAc'] + mass_dict['Hex'] + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((2 * mass_dict['HexNAc'] + mass_dict['Hex'] - addy) / c) - j) < 1
                                      for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))

            if 'Fuc' in m:
                truth_diag.append(any([abs(m_fuc + (1.0078 * multiplier) - j) < 1 or abs(mass_dict['HexNAc'] + m_fuc + (1.0078 * multiplier) - j) < 1 or
                                      abs(2*mass_dict['HexNAc'] + m_fuc + (1.0078 * multiplier) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((m_fuc - addy) / c) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((mass_dict['HexNAc'] + m_fuc - addy) / c) - j) < 1 or
                                      abs(df_out.index.tolist()[k] - ((2*mass_dict['HexNAc'] + m_fuc - addy) / c) - j) < 1
                                       for j in df_out.top_fragments.values.tolist()[k][:20] if isinstance(j, float)]))

            if all(truth_comp):

                # Append evidence level to the list based on match_score
                if all(truth_diag):
                    diagnostic_ion = "all diagnostic ions present"
                    evidence_level = "strong"
                elif not any(truth_diag):
                    diagnostic_ion = "no diagnostic ion present"
                    evidence_level = "weak"
                else:
                    diagnostic_ion = "some diagnostic ions present"
                    evidence_level = "moderate"

                if evidence_level in ["strong", "moderate"]:
                    if to_append:
                        keep.append(current_preds[i])
                    evidence_levels.append(evidence_level)
                    diagnostic_ions.append(diagnostic_ion)
                    match_list.append(match_score)
            else:
                if not to_append:
                    keep.append('remove')
                    diagnostic_ions.append("NA")
                    evidence_levels.append("NA")
                    match_list.append(0)

        # Use .loc instead of .at to assign values
        df_out.iat[k, 0] = keep
        df_out.iat[k, -3] = match_list
        df_out.iat[k, -2] = diagnostic_ions
        df_out.iat[k, -1] = evidence_levels

    return df_out[df_out['predictions'].apply(lambda x: 'remove' not in x[:1])]


def glycobert_inference(filepath, vocab_path = vocab_path_glycobert, modelDir = 'CABSEL/glycobert', batch_size=256, filename='unspecified', lc='PGC', mode='negative',
                           modification='reduced', glycan_type='N', trap='linear', ionization='other_ion', fragmentation='CID',
                           taxonomy_level='Class', taxonomy_filter = 'Mammalia', df_use = None, mass_tolerance = 0.5, mass_tag = None,
                           filter_out = {'Ac','Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, glycan_pkl = glycan_path, device = device):

    # Start timing
    total_start_time = time.time()

    # Load glycans
    glycan_path = glycan_pkl
    glycans = pickle.load(open(glycan_path, 'rb'))
    device = device
    
    # Settings for binning
    threshold = 0.001  # peak intensity thresholding
    minMZ = 39.714  # minimum m/z
    maxMZ = 3000  # maximum m/z
    sizeMZ = 0.3  # m/z bin size
    mz_bin_edges = bin_linear(minMZ, maxMZ, sizeMZ)  # use linear binning
    minI = 0
    maxI = 1
    sizeI = 0.001  # peak intensity bin size
    peak_bin_edges = bin_linear(minI, maxI, sizeI)
    minRT = 0
    maxRT = 1
    sizeRT = 0.01  # relative retention time bin size
    RT_bin_edges = bin_linear(minRT, maxRT, sizeRT)

    df = load_file(filepath, intensity=True)

    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type==glycan_type])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)]

    common_structure_map,df_use,topo_struct_map = create_struct_map_modified(df_use,glycan_type,filter_out = filter_out,phylo_level = taxonomy_level,phylo_filter= taxonomy_filter)
    df = assign_candidate_structures_modified(df,df_use,common_structure_map,topo_struct_map,mass_tolerance,mode,mass_tag)

    # Update metadata to the DataFrame
    df['RT_original'] = df['RT']
    df['filename'] = filename
    df['lc'] = lc
    df['mode'] = mode
    df['ionization'] = ionization
    df['modification'] = modification
    df['trap'] = trap
    df['fragmentation'] = fragmentation
    df['glycan_type'] = glycan_type

    # Process the DataFrame
    df = process_data(df, RT_bin_edges, mz_bin_edges, peak_bin_edges, threshold)

    # Construct sentences
    sentences = generate_corpus_mz(df)

    # Use the vocabulary to create a tokenizer
    tokenizer = GlycoBertTokenizer.load_vocabulary(path=vocab_path)

    # Tokenize the sentences
    sentences_tensor = tokenizer.encode(sentences)
    
    # Load the trained model                                                          
    #config = BertConfig.from_pretrained(modelDir)
    model = BertForSequenceClassification.from_pretrained(modelDir).to(device)

    # Load input IDs and attention masks
    input_ids = sentences_tensor["token_ids"]
    attention_masks = sentences_tensor["attention_mask"]

    # Initialize lists to store results
    all_predictions = []
    n_batches = ceil(len(sentences) / batch_size)  # Use ceiling function to round up

    # Iterate through batches
    for i in range(n_batches):
        # Get the batches
        start = i * batch_size
        end = min(start + batch_size, len(sentences))  # Ensure the end index does not exceed the number of samples

        batch_input_ids = input_ids[start:end].to(device)
        batch_attention_masks = attention_masks[start:end].to(device)

        # Predict the labels
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks
            )

        # Process the model output
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        topk_probabilities, topk_indices = torch.topk(probabilities, 25, dim=1)

        # Compute the top 25 glycans and their scores for each spectrum in the batch
        topk_glycans = []
        topk_scores = []
        for i in range(logits.shape[0]):
            row_labels = [glycans[j] for j in topk_indices[i].tolist()]
            row_scores = topk_probabilities[i].tolist()
            topk_glycans.append(row_labels)
            topk_scores.append(row_scores)

        all_predictions.extend([(glycan, score) for glycan, score in zip(topk_glycan, topk_score)] for topk_glycan,
                               topk_score in zip(topk_glycans, topk_scores))

    # Assign predictions to the DataFrame
    df['predictions'] = all_predictions

    # Total time
    print(f"Total inference time: {time.time() - total_start_time:.2f} seconds")

    return df

def filter_glycans_glycobert(df_out, glycan_class='N', modification='reduced', mode='negative', pred_thresh = 0.01,
                   frag_threshold = 1, MS1_ppm = 10, MS2_tolerance=0.5, annotation_thresh = 3,
                   filter_diag_ion=['Neu5Gc', 'Kdn', 'S'], min_ion_match = 0, taxonomy_level='Class', taxonomy_filter = 'Mammalia', 
                   df_use = None, glycan_pkl = glycan_path):

    # Start timing
    total_start_time = time.time()
    
    glycan_path = glycan_pkl
    glycans_list = pickle.load(open(glycan_path, 'rb'))
    libr = get_lib(glycans_list)

    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type==glycan_class])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)]
    glycan_df_use = df_use['glycan'].tolist()

    #reduced = 1.0078 if modification == 'reduced' else 0
    multiplier = -1 if mode == 'negative' else 1

    columns = ['predictions', 'intensity', 'reducing_mass', 'peak_d', 'RT', 'RT_original', 'scan_number']
    df_filtered = df_out[columns]
    df_filtered.set_index('reducing_mass', inplace=True)

    # Filter glycans using model confidence
    df_filtered['predictions'] = [
        [(glycan, round(score, 4)) for glycan, score in preds if
         score > pred_thresh]
        for preds in df_filtered.predictions
    ]


    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]
    
    # Filter glycans using user-deifned glycan class
    df_filtered['predictions'] = [
        [(glycan, round(score, 4)) for glycan, score in preds if
         enforce_class_modified(glycan, glycan_class)]
        for preds in df_filtered.predictions
    ]


    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]

    # Filter glycans using mass-check
    df_filtered['predictions'] = [
        [
            (glycan, round(score, 4)) for glycan, score in preds if
            mass_check_modified_ppm(mass, glycan, modification=modification, mode=mode, mass_thresh = MS1_ppm)
        ][:5]
        for preds, mass in zip(df_filtered['predictions'], df_filtered.index)
    ]

    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]


    # Compute glycan composition
    df_filtered['composition'] = [[glycan_to_composition(g[0]) for g in predictions] if predictions else np.nan
                                  for predictions in df_filtered.predictions]

    df_filtered.dropna(subset=['composition'], inplace=True)

    # Compute precursor ion charge
    df_filtered['charge'] = [
        list(set([round(composition_to_mass(comp) / idx) * multiplier for comp in composition]))
        for composition, idx in zip(df_filtered['composition'], df_filtered.index)
    ]

    # Compute top fragments of the spectra
    df_filtered['top_fragments'] = df_filtered.apply(lambda row: get_top_fragments(row, threshold=frag_threshold), axis=1)

    # Filter glycans using domain-filter
    df_filtered = domain_filter_modified_bert(df_filtered, glycan_class, mode=mode,
                                         modification=modification, MS2_tolerance=MS2_tolerance, filter_diag_ion = filter_diag_ion, min_ion_match =min_ion_match)

    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]


    df_filtered.reset_index(inplace=True)

    cols = ['scan_number', 'predictions', 'intensity', 'composition', 'charge', 'reducing_mass', 'RT', 'RT_original', 'peak_d', 'top_fragments', 'ion_match_score', 'diagnostic_ions', 'evidence_level']
    df_filtered = df_filtered[cols]

    intensity = 'intensity' in df_filtered .columns and not (df_filtered['intensity'] == 0).all() and not df_filtered['intensity'].isnull().all()
    if intensity:
        df_filtered.loc[df_filtered['intensity'].isnull(),'intensity']=0
    else:
        df_filtered['intensity'] = [0]*len(df_filtered)
    df_filtered['rel_abundance'] = df_filtered['intensity']

    df_filtered = calculate_annotation_scores_bert(df_filtered, multiplier = -1, mass_tag = None, mass_tolerance = 0.5)

    df_filtered['filtered_predictions'] = df_filtered['annotation_scores'].apply(
        lambda scores: [(glycan_conf, annotation) for glycan_conf, annotation in scores if annotation >= annotation_thresh]
        if scores else [])

    df_filtered['prediction_in_df_use'] = df_filtered['filtered_predictions'].apply(lambda x: [(glycan_conf, annotation) for glycan_conf, annotation in x if glycan_conf[0] in glycan_df_use])
    df_filtered = df_filtered[df_filtered['prediction_in_df_use'].apply(lambda x: len(x) > 0)]
    df_filtered['annotation_score_top1'] = df_filtered['prediction_in_df_use'].apply(lambda scores: scores[0][1] if scores else 0)
    df_filtered['prediction_top1'] = df_filtered['prediction_in_df_use'].apply(lambda x: x[0][0][0])

    df_filtered['charge'] = df_filtered['charge'].apply(lambda x: x[0])
    df_filtered['predictions_glycobert'] = df_filtered['prediction_in_df_use'].apply(lambda x: [g[0][0] for g in x])
    df_filtered['predictions'] = df_filtered['prediction_in_df_use'].apply(lambda x: [g[0] for g in x])
    df_before_deduplication = df_filtered.copy(deep=True)
    df_filtered = deduplicate_predictions_bert(df_filtered, mz_diff = 0.5, rt_diff = 1.0)
    df_filtered = combine_charge_states_bert(df_filtered)

    # Total time
    print(f"Total postprocessing time: {time.time() - total_start_time:.2f} seconds")
    
    return df_filtered, df_before_deduplication


def glycobart_inference(filepath, vocab_path  = vocab_path_glycobart, modelDir = 'CABSEL/glycobart', batch_size=256, filename='unspecified', lc='PGC', mode='negative',
                        modification='reduced', glycan_type='N', trap='linear', ionization='other_ion', fragmentation='CID',
                        taxonomy_level='Class', taxonomy_filter = 'Mammalia', df_use = None, num_beam = 32, num_return = 32,
                        filter_out = {'Ac','Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'}, glycan_pkl = glycan_path,
                        mass_tolerance = 0.5, mass_tag = None, device = device):

    # Start timing
    total_start_time = time.time()

    # Load glycans
    glycan_path = glycan_pkl
    glycans = pickle.load(open(glycan_path, 'rb'))
    device = device
    
    # Settings for binning
    threshold = 0.001  # peak intensity thresholding
    minMZ = 39.714  # minimum m/z
    maxMZ = 3000  # maximum m/z
    sizeMZ = 0.3  # m/z bin size
    mz_bin_edges = bin_linear(minMZ, maxMZ, sizeMZ)  # use linear binning
    minI = 0
    maxI = 1
    sizeI = 0.001  # peak intensity bin size
    peak_bin_edges = bin_linear(minI, maxI, sizeI)
    minRT = 0
    maxRT = 1
    sizeRT = 0.01  # relative retention time bin size
    RT_bin_edges = bin_linear(minRT, maxRT, sizeRT)

    df = load_file(filepath, intensity=True)

    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type==glycan_type])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)]

    common_structure_map,df_use,topo_struct_map = create_struct_map_modified(df_use,glycan_type,filter_out = filter_out,phylo_level = taxonomy_level,phylo_filter= taxonomy_filter)
    df = assign_candidate_structures_modified(df,df_use,common_structure_map,topo_struct_map,mass_tolerance,mode,mass_tag)

    # Update metadata to the DataFrame
    df['RT_original'] = df['RT']
    df['filename'] = filename
    df['lc'] = lc
    df['mode'] = mode
    df['ionization'] = ionization
    df['modification'] = modification
    df['trap'] = trap
    df['fragmentation'] = fragmentation
    df['glycan_type'] = glycan_type

    # Process the DataFrame
    df = process_data(df, RT_bin_edges, mz_bin_edges, peak_bin_edges, threshold)

    # construct sentences
    sentences = generate_corpus_mz(df)
  
    # Use the vocabulary to create a tokenizer
    tokenizer = GlycoBartTokenizer.load_vocabulary(path=vocab_path)

    # tokenize the sentences
    sentences_tensor = tokenizer.encode(sentences)

    # Load the trained model
    # config = BartConfig.from_pretrained(modelDir)
    model = BartForConditionalGeneration.from_pretrained(modelDir).to(device)

    # DataLoader for testing
    test_data = TensorDataset(sentences_tensor["token_ids"], sentences_tensor["attention_mask"])
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Evaluation
    predictions = []

    with torch.no_grad():
        for batch_input_ids, batch_attention_masks in test_loader:
            batch_predictions = []
            for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
                input_ids = input_ids.unsqueeze(0).to(device)  # Unsqueeze to add batch dimension
                attention_mask = attention_mask.unsqueeze(0).to(device)  # Unsqueeze to add batch dimension

                # Generate predictions
                predicted_ids = model.generate(input_ids, attention_mask=attention_mask, num_beams=num_beam,
                                               num_return_sequences=num_return, max_length=200, early_stopping=True)
                predicted_texts = [tokenizer.decode(token_id, skip_special_tokens=True) for token_id in predicted_ids]
                batch_predictions.append(predicted_texts)

            predictions.extend(batch_predictions)

    df['predictions'] = predictions
    
    # Total time
    print(f"Total inference time: {time.time() - total_start_time:.2f} seconds")

    return df


def filter_glycans_glycobart(df_out, glycan_class='N', modification='reduced', mode='negative',
                   frag_threshold = 1, MS1_ppm = 10, MS2_tolerance=0.5, annotation_thresh = 3, mass_tag = None,
                   filter_diag_ion=['Neu5Gc', 'Kdn', 'S'], min_ion_match = 0, taxonomy_level='Class', taxonomy_filter = 'Mammalia', 
                   df_use = None, glycan_pkl = glycan_path):
    # Start timing
    total_start_time = time.time()

    glycan_path = glycan_pkl
    glycans_list = pickle.load(open(glycan_path, 'rb'))
    libr = get_lib(glycans_list)

    if df_use is None:
        df_use = copy.deepcopy(df_glycan[df_glycan.glycan_type==glycan_class])
        df_use = df_use[df_use[taxonomy_level].apply(lambda x: taxonomy_filter in x)]
    glycan_df_use = df_use['glycan'].tolist()

    multiplier = -1 if mode == 'negative' else 1

    columns = ['predictions', 'intensity', 'reducing_mass', 'peak_d', 'RT', 'RT_original', 'scan_number']
    df_filtered = df_out[columns]
    df_filtered.set_index('reducing_mass', inplace=True)

    df_filtered['predictions'] = df_filtered['predictions'].apply(
        lambda x: [convert_to_iupac(text) for text in x])
    df_filtered['predictions'] = df_filtered['predictions'].apply(
        lambda x: [text for text in x if text])

    df_filtered['sorted_glycans'] = df_filtered['predictions'].apply(lambda x: sort_glycans_by_topology(x))

    df_filtered['predictions'] = df_filtered['sorted_glycans'].apply(lambda x: x['sorted_glycans'])

    df_filtered['predictions'] = [
        [glycan for glycan in preds if
         enforce_class_modified(glycan, glycan_class)]
        for preds in df_filtered.predictions
    ]

    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]

    # Filter glycans using mass-check
    df_filtered['predictions'] = [
        [glycan for glycan in preds if
         mass_check_modified_ppm(mass, glycan, modification=modification, mode=mode,mass_thresh=MS1_ppm)
        ][:5]
        for preds, mass in zip(df_filtered['predictions'], df_filtered.index)
    ]


    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]

    # Compute glycan composition
    df_filtered['composition'] = [[glycan_to_composition(g) for g in predictions] if predictions else np.nan
                                  for predictions in df_filtered.predictions]

    df_filtered.dropna(subset=['composition'], inplace=True)

    # Compute precursor ion charge
    df_filtered['charge'] = [
        list(set([round(composition_to_mass(comp) / idx) * multiplier for comp in composition]))
        for composition, idx in zip(df_filtered['composition'], df_filtered.index)
    ]

    # Compute top fragments of the spectra
    df_filtered['top_fragments'] = df_filtered.apply(lambda row: get_top_fragments(row, threshold=frag_threshold), axis=1)

    # Filter glycans using domain-filter
    df_filtered = domain_filter_modified_bart(df_filtered, glycan_class, mode=mode,
                                         modification=modification, MS2_tolerance=MS2_tolerance, filter_diag_ion = filter_diag_ion, min_ion_match =min_ion_match)

    df_filtered = df_filtered[df_filtered['predictions'].apply(lambda x: len(x) > 0)]

    df_filtered.reset_index(inplace=True)

    cols = ['scan_number', 'predictions', 'intensity', 'composition', 'charge', 'reducing_mass', 'RT', 'RT_original', 'peak_d', 'top_fragments', 'ion_match_score', 'diagnostic_ions', 'evidence_level']
    df_filtered = df_filtered[cols]


    #df_filtered = process_glycan_predictions_bart(df_filtered, glycan_df_use, modification = modification, mode = mode, mass_tag=None, MS2_tolerance=MS2_tolerance, min_annotation_score=min_annotation_score)
    #print('after process glycan predictions:', len(df_filtered))


    intensity_exists = ('intensity' in df_filtered.columns and not (df_filtered['intensity'] == 0).all() and not df_filtered['intensity'].isnull().all())
    
    if intensity_exists:
        df_filtered.loc[df_filtered['intensity'].isnull(), 'intensity'] = 0
    else:
        df_filtered['intensity'] = [0] * len(df_filtered)

    df_filtered['rel_abundance'] = df_filtered['intensity']

    # Calculate annotation scores
    df_filtered = calculate_annotation_scores_bart(df_filtered, multiplier=multiplier, mass_tag=mass_tag, MS2_tolerance=MS2_tolerance)

    # Filter predictions based on annotation scores
    df_filtered['filtered_predictions'] = df_filtered['annotation_scores'].apply(
        lambda scores: [(glycan, annotation) for glycan, annotation in scores
                       if annotation >= annotation_thresh] if scores else []
    )

    # Filter predictions based on glycan_df_use
    df_filtered['prediction_in_df_use'] = df_filtered['filtered_predictions'].apply(
        lambda x: [(glycan, annotation) for glycan, annotation in x
                  if glycan in glycan_df_use]
    )

    # Keep only rows with valid predictions
    df_filtered = df_filtered[df_filtered['prediction_in_df_use'].apply(lambda x: len(x) > 0)]

    # Extract top predictions and scores
    df_filtered['annotation_score_top1'] = df_filtered['prediction_in_df_use'].apply(lambda x: x[0] if x else 0)
    
    df_filtered['prediction_top1'] = df_filtered['prediction_in_df_use'].apply(lambda x: x[0][0])

    df_filtered['charge'] = df_filtered['charge'].apply(lambda x: x[0])

    df_filtered['predictions_glycobart'] = df_filtered['prediction_in_df_use'].apply(lambda x: [g[0] for g in x])
    
    df_filtered['predictions'] = df_filtered['prediction_in_df_use'].apply(lambda x: [g[0] for g in x])
    
    df_before_deduplication = df_filtered.copy(deep=True)
    df_filtered = deduplicate_predictions_bart(df_filtered)
    
    df_filtered = combine_charge_states_bart(df_filtered)
    # Total time
    print(f"Total postprocessing time: {time.time() - total_start_time:.2f} seconds")

    return df_filtered, df_before_deduplication


def glycan_to_topology(glycan_string):
    # Define a regular expression pattern for anything inside parentheses
    inside_parentheses_pattern = re.compile(r'\(([^)]+)\)')

    # Function to convert anything inside parentheses to the format (??-?)
    def convert_inside_parentheses(match):
        inside_text = match.group(1)
        return f'(??-?)'

    # Apply the conversion function to anything inside parentheses in the glycan string
    modified_glycan = inside_parentheses_pattern.sub(convert_inside_parentheses, glycan_string)

    return modified_glycan


def sort_glycans_by_topology(glycan_list):
    """
    Sort glycans based on their topology (ignoring linkages), keeping only
    the first glycan from each topology group.

    Parameters:
    -----------
    glycan_list : list
        List of glycan structures
    glycan_to_topology : function
        Function that converts glycan structure to topology by replacing linkages with '?'

    Returns:
    --------
    dict
        Dictionary containing:
        - 'sorted_glycans': List of representative glycans (one per topology),
                           sorted by topology frequency
        - 'topology_scores': Dictionary of topology scores (normalized frequency)
        - 'topology_groups': Dictionary grouping glycans by their topology
    """
    # Create topology mapping
    topology_dict = {}
    for glycan in glycan_list:
        topology = glycan_to_topology(glycan)
        if topology not in topology_dict:
            topology_dict[topology] = []
        topology_dict[topology].append(glycan)

    # Count frequency of each topology
    topology_counts = {
        topology: len(glycans)
        for topology, glycans in topology_dict.items()
    }

    # Calculate topology scores (normalized frequency)
    total_glycans = len(glycan_list)
    topology_scores = {
        topology: count
        for topology, count in topology_counts.items()
    }

    # Sort topologies by frequency and take first glycan from each group
    sorted_glycans = [
        topology_dict[topology][0]  # Take only the first glycan from each topology group
        for topology in sorted(topology_dict.keys(),
                             key=lambda x: topology_counts[x],
                             reverse=True)
    ]

    return {
        'sorted_glycans': sorted_glycans,
        'topology_scores': topology_scores,
        'topology_groups': topology_dict
    }


def convert_to_antenna(text):
    # Split text into words
    words = text.split()

    # Function to check if a word is a linkage
    def is_linkage(word):
        return (word[0] in ['a', 'b', '.'] or word[0].isdigit() or
                word[-1] in ['a', 'b', '.'] or word[-1].isdigit())

    # Add parentheses to linkages
    reformatted_words = []
    for word in words:
        if is_linkage(word):
            reformatted_words.append("(" + word + ")")
        else:
            reformatted_words.append(word)

    # Construct antennae
    antenna = reformatted_words[0]
    for idx in range(1, len(reformatted_words)):
        current_word = reformatted_words[idx]
        prev_word = reformatted_words[idx - 1]

        # If previous word and current word are not linkages, then this is a start of new antenna
        if not prev_word.endswith(')') and not current_word.startswith('('):
            antenna += '. '

        antenna += current_word

    # Adding period
    antenna += '.'

    return antenna


def dict_to_iupac(d):
    # Base case: if d is not a dictionary, return d
    if not isinstance(d, dict) or not d:
        return ""

    # Recursive case: process the inner dictionaries
    parts = []
    for key, value in d.items():
        parts.append(dict_to_iupac(value) + key)

    # Join the parts together with brackets if there are multiple parts
    if len(parts) > 1:
        return parts[0] + "[" + "][".join(parts[1:]) + "]"
    else:
        return "".join(parts)


# Convert glycan antennae to IUPAC
def add_path_to_tree(tree, path):
    if not path:
        return
    if path[0] not in tree:
        tree[path[0]] = {}
    add_path_to_tree(tree[path[0]], path[1:])


def build_glycan_tree(paths):
    tree = {}
    # Sort paths from longest to shortest
    sorted_paths = sorted(paths, key=len, reverse=True)
    for path in sorted_paths:
        reversed_path = list(reversed(path))
        add_path_to_tree(tree, reversed_path)
    return tree


def antennae_to_paths(antennae):
    sentences = antennae.replace(')', ') ')
    sentences = sentences[:-1]
    paths = [sentence.split() for sentence in sentences.split('. ') if sentence]

    return paths


def antennae_to_iupac(antennae):
    paths = antennae_to_paths(antennae)
    tree = build_glycan_tree(paths)
    iupac = dict_to_iupac(tree)

    return iupac


def convert_to_iupac(x):
    try:
        return antennae_to_iupac(convert_to_antenna(x))
    except IndexError as e:
        pass


def remove_glycans(df, column_name='predictions'):
    """
    Process the glycans in the specified column of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing glycan data.
        column_name (str): The name of the column to process. Default is 'predictions'.

    Returns:
        pd.DataFrame: The DataFrame with processed glycan data.
    """
    for index, row in df.iterrows():
        glycans = row[column_name]

        if isinstance(glycans, list):
            # Process each glycan in the list
            processed_glycans = [
                glycan.replace('.', '?')
                for glycan in glycans
                if not (glycan.startswith('[') or glycan.startswith('(') or glycan.endswith(']') or ')(' in glycan or '[(' in glycan)
            ]
            df.at[index, column_name] = processed_glycans
        else:
            # Process the single glycan string
            glycan = glycans.replace('.', '?')
            if not (glycan.startswith('[') or glycan.startswith('(') or glycan.endswith(']') or ')(' in glycan or '[(' in glycan):
                df.at[index, column_name] = glycan
            else:
                df.at[index, column_name] = None  # Remove if it starts with '[', '(', ends with ']', or contains ')(' or '[('

    return df
