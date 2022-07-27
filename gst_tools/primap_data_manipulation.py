import re
import os
import logging

import pandas as pd
import numpy as np
from shortcountrynames import to_name

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from general_data_manipulation import *

# ======================
# Functions for Primap data manipulation and calculation
# ======================

def convert_from_Gg_to_Mt(proc_data):
    conv_df = proc_data.copy()
    org_unit = proc_data['unit'].unique()[0]

    gas_unit = re.search('Gg (.+?) / yr', org_unit).group(1)

    desired_unit = 'Mt' + gas_unit

    conversion_factor = 1/1000

    logging.debug('*******************')
    logging.debug('Converting unit from "' + org_unit + '" to "' + desired_unit + 
          '" using a conversion factor of ' + str(conversion_factor))
    logging.debug('*******************')

    conv_df['unit'] = desired_unit
    
    # Convert the data
    conv_df, other_cols = set_non_year_cols_as_index(conv_df)
    conv_df = conv_df * conversion_factor
    conv_df = conv_df.reset_index()

    return conv_df

def define_primap_proc_fname(primap_filtered, gas_dict, sect_dict):
    
    source = primap_filtered['source'][0]
    source_scenario = source.lower().replace('hist', primap_filtered['scenario'][0].lower())
    gas = gas_dict[primap_filtered['gas'][0]]
    sector = sect_dict[primap_filtered['category'][0]]
    
    proc_fname =  source_scenario + '_' + gas + '_' + sector + '.csv'

    return proc_fname

def define_primap_variable_name(sector_code, gas, dict_gas_var, primap_sectors='gst_tools/primap_sectors.csv'):
    sector_codes = pd.read_csv(primap_sectors)
    sector_codes.set_index('code', inplace=True)

    gas_name = dict_gas_var[gas]

    if sector_code in ['M.0.EL', '1.B.3', '2.G', '2.H', '1.B.1', '1.B', '1.B.2']:
        variable_name = sector_codes.loc[sector_code]['1'] + ' ' + gas_name + ' ' + sector_codes.loc[sector_code]['3'] + ' ' + sector_codes.loc[sector_code]['4']
    elif sector_code in ['1', '5']:
        variable_name = sector_codes.loc[sector_code]['1'] + ' ' + gas_name + ' ' + sector_codes.loc[sector_code]['3']
    elif sector_code in ['2.D', '2.F', '4', '1.C', '1.A', '3.A', '2', '2.A', '2.B', '2.C', '2.E', 'M.AG', 'M.AG.ELV']:
        variable_name = gas_name + ' ' + sector_codes.loc[sector_code]['2'] + ' ' + sector_codes.loc[sector_code]['3']
    else:
        raise ValueError('The sector code in the dataset ("category" column) is wrong. Please check the dataset.')

    return variable_name

def filter_primap(renamed_data, gas, sector, scenario, countries, start_year, dict_gas, primap_sectors='gst_tools/primap_sectors.csv'):
    # Reduce the dataset to only the desired variables
    proc_primap = renamed_data.loc[(renamed_data['gas'] == gas) &
                        (renamed_data['scenario'] == scenario) &
                        (renamed_data['category'] == sector)
                        ]

    if len(proc_primap.index) == 0:
        raise ValueError('There is no data for the gas, the sector, and the data source specified. Please, provide new parameters.')

    else:
        # Reduce the countries or regions to only those desired
        proc_primap = proc_primap.loc[proc_primap['country'].isin(countries)]

        if len(proc_primap.index) == 0:
            raise ValueError('There is no data for the countries specified. Please, select different countries.')
        else:
            # Tell the user if any of the needed countries are missing and, if yes, which ones:
            missing_countries = list(set(countries) - set(proc_primap['country'].unique()))
            if missing_countries:
                logging.info('Not all countries requested were available in the raw data. You are missing the following:')
                for country in missing_countries:
                    logging.info('   ' + to_name(country))
                logging.info('---------')

            # Reduce to only the required years
            proc_primap = change_first_year(proc_primap, start_year)

            proc_primap['variable'] = define_primap_variable_name(sector, gas, dict_gas, primap_sectors=primap_sectors)

            proc_primap = check_column_order(proc_primap)

            # Check
            logging.debug('These are the first ten rows of the processed data:')
            logging.debug(proc_primap.head(10))

            return proc_primap