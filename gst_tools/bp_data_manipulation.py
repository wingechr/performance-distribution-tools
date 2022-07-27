import re
import os
import logging

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from shortcountrynames import to_name
from general_data_manipulation import *

# ======================
# Functions for bp data calculation and manipulation
# ======================

def calculate_energy_use(renamed_bp):
    # Primary energy consumption
    energy_use = renamed_bp['primary_ej']
    new_df = renamed_bp.copy()

    new_df['energy_use_ej'] = energy_use

    logging.debug('Added primary energy consumption. Unit: Exajoules')
    return new_df

def calculate_ff_share(renamed_bp):
    total_consumption = renamed_bp['primary_ej']
    ff_cons = renamed_bp['coalcons_ej'] + renamed_bp['gascons_ej'] + renamed_bp['oilcons_ej']
    
    new_df = renamed_bp.copy()
    new_df['ff_cons_share_%'] = (ff_cons/total_consumption)*100

    logging.debug('Added share of fossil fuel energy consumed. Unit: %')
    return new_df

def calculate_ren_elec_share(renamed_bp):
    # Share of renewable energy in electricy
    renewables = renamed_bp['electbyfuel_ren_power'] + renamed_bp['electbyfuel_hydro']

    total = renamed_bp['electbyfuel_total']
    
    share = (renewables/total)*100

    new_df = renamed_bp.copy()
    new_df['ren_elec_share_%'] = share

    logging.debug('Added share of renewables in electricity generated. Unit: %')
    return new_df

def define_bp_proc_fname(proc_data):
    source = 'bp_stat_rev_world_energy_2022'
    variable = proc_data['variable'][0]

    proc_fname = source + '_' + variable.replace(' ', '_').lower() + '.csv'

    return proc_fname

def filter_bp(renamed_bp, energy_variable, countries, start_year):
    if energy_variable != 1 and energy_variable != 2 and energy_variable != 3:
        raise ValueError('Error. Please provide a valid energy variable (either 1, 2 or 3).')
    else:
        # Calculate the variables, add them to the dataframe, and pivot the dataframe to only include this variable along with the unit column.
        if energy_variable == 1:
            filtered = calculate_ren_elec_share(renamed_bp)
            filtered = filtered[['country','ren_elec_share_%', 'year']]
            filtered = filtered.pivot(index='country', columns='year', values='ren_elec_share_%').reset_index().rename_axis(None, axis=1)
            filtered['variable'] = ['Share of renewable power']*len(filtered)
            filtered['unit'] = ['%']*len(filtered)
        elif energy_variable == 2:
            filtered = calculate_ff_share(renamed_bp)
            filtered = filtered[['country', 'ff_cons_share_%', 'year']]
            filtered = filtered.pivot(index='country', columns='year', values='ff_cons_share_%').reset_index().rename_axis(None, axis=1)
            filtered['variable'] = ['Share of fossil primary energy consumed']*len(filtered)
            filtered['unit'] = ['%']*len(filtered)
        elif energy_variable == 3:
            filtered = calculate_energy_use(renamed_bp)
            filtered = filtered[['country','energy_use_ej', 'year']]
            filtered = filtered.pivot(index='country', columns='year', values='energy_use_ej').reset_index().rename_axis(None, axis=1)
            filtered['variable'] = ['Energy use']*len(filtered)
            filtered['unit'] = ['EJ']*len(filtered)
        
        # Filter by countries
        filtered = filtered.loc[filtered['country'].isin(countries)]
        
        if len(filtered.index) == 0:
            raise ValueError('There is no data for the countries specified. Please, select different countries.')
        else:
            # Tell the user if any of the needed countries are missing and, if yes, which ones:
            missing_countries = list(set(countries) - set(filtered['country'].unique()))
            if missing_countries:
                logging.info('Not all countries requested were available in the raw data. You are missing the following:')
                for country in missing_countries:
                    logging.info('   ' + to_name(country))
                logging.info('---------')

            # Reduce to only required years
            filtered = change_first_year(filtered, start_year)
            filtered = check_column_order(filtered)

            # Check
            logging.debug('These are the 10 first rows of the processed data:')
            logging.debug(filtered.head(10))

            return filtered