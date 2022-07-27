import re
#import sys
import os
import logging

import pandas as pd
import numpy as np
from shortcountrynames import to_name

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ======================

def load_data(folder, fname):
    file_path = os.path.join(folder, fname)
    logging.debug('Reading ' + file_path)
    raw_data = pd.read_csv(file_path)
    return raw_data

def rename_primap(raw_primap):
    # LTN
    raw_data_renamed = raw_primap.rename(columns={'scenario (PRIMAP-hist)': 'scenario',
                         'area (ISO3)': 'country',
                         'category (IPCC2006_PRIMAP)': 'category',
                         'entity': 'gas'}, inplace=False)
    raw_data_renamed['country'].replace({'EU27BX': 'EUU'}, inplace=True)
    return raw_data_renamed

def rename_bp(raw_bp):
    raw_data_renamed = raw_bp.rename(columns={
                        'ISO3166_alpha3': 'country',
                        'Year': 'year'}, inplace=False)
    raw_data_renamed = raw_data_renamed.astype({'year': str})  
    return raw_data_renamed

def calculate_energy_use(renamed_bp):
    # WARNING: Only to be applied to the renamed dataset to which no additional variable has been added.
    # Primary energy consumption
    energy_use = renamed_bp['primary_ej']
    new_df = renamed_bp.copy()

    new_df['energy_use_ej'] = energy_use

    logging.debug('Added primary energy consumption. Unit: Exajoules')
    return new_df

def calculate_ren_elec_share(renamed_bp):
    # Share of renewable energy in electricy
    renewables = renamed_bp['electbyfuel_ren_power'] + renamed_bp['electbyfuel_hydro']

    # What about the other columns that refer to renewable power?
    # Namely: ren_power_twh and ren_power_twh_net
    # ren_power_twh appears to be equal to electbyfuel_ren_power
    # ren_power_twh_net is a bit smaller

    total = renamed_bp['electbyfuel_total']
    
    share = (renewables/total)*100

    new_df = renamed_bp.copy()
    new_df['ren_elec_share_%'] = share

    logging.debug('Added share of renewables in electricity generated. Unit: %')
    return new_df

def calculate_ff_share(renamed_bp):
    total_consumption = renamed_bp['primary_ej']
    ff_cons = renamed_bp['coalcons_ej'] + renamed_bp['gascons_ej'] + renamed_bp['oilcons_ej']
    
    new_df = renamed_bp.copy()
    new_df['ff_cons_share_%'] = (ff_cons/total_consumption)*100

    logging.debug('Added share of fossil fuel energy consumed. Unit: %')
    return new_df

def filter_bp(renamed_bp, energy_variable, countries, start_year):
    if energy_variable != 1 and energy_variable != 2 and energy_variable != 3:
        raise ValueError('Error. Please provide a valid energy variable (either 1, 2 or 3).')
    else:
        # Calculate the variables, add them to the dataframe, and pivot the dataframe to only include this variable along with the
        # unit column.
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

            # make sure 'variable' contains all necessary information
            #proc_primap['variable'] = define_primap_variable_name(sector, gas, dict_gas, primap_sectors=primap_sectors)

            # label the source
            #proc_primap['source'] = source_name

            filtered = check_column_order(filtered)

            # Check
            logging.debug('These are the 10 first rows of the processed data:')
            logging.debug(filtered.head(10))

            return filtered


def change_first_year(df, new_start_year):

    """
    Reduces a dataframe to start in a later, specified year than the original data.
    Can be useful for reducing the size of the data to be handled or for performing analysis over a reduced timeframe.
    """

    # reduce the number of years (keeps things lighter and faster)
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
    other_cols = list(set(df.columns) - set(year_cols))

    # check the current start year
    if not year_cols:
        raise ValueError("no years left")
        
    cur_start_year = min(year_cols)
    last_year = max(year_cols)

    if int(cur_start_year) > int(new_start_year):

        new_start_year = cur_start_year

    else:
        # set other columns as index
        if other_cols:
            df = df.set_index(other_cols)

        # identify which years to keep
        years_to_keep = np.arange(int(new_start_year), (int(last_year) + 1), 1)
        years_keep_str = list(map(str, years_to_keep))

        # remove extra years
        df = df.loc[:, years_keep_str]

        # return other columns
        if other_cols:
            df = df.reset_index()

        # TODO - modify so that the output index is the same as the input!

    # check formatting
    df = check_column_order(df)

    # tell the user what happened
    logging.debug('First year of data available is now ' + str(new_start_year))
    logging.debug('Last year of data available is ' + str(last_year))

    return df

def check_column_order(df):

    """
    For ease of processing and plotting, it's best to have the columns with metadata all together and then the
    years all in the correct order. As this is a common check / priority, this is a general function for it.
    """

    # get all column names
    df_columns = df.columns

    # get the year columns
    year_columns = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]

    # if there aren't year columns, check for other numeric columns
    if year_columns:
        # make sure the years are sorted (makes them an integer in case they aren't already)
        ordered_year_columns = sorted(year_columns, key=int)

    else:
        year_columns = [y for y in df[df.columns] if isinstance(y, (int, float))]
        ordered_year_columns = sorted(year_columns)

    # get the columns that are not years
    metadata_cols = list(set(df_columns) - set(year_columns))

    # set the new column order with metadata first in alphabetical order
    new_column_order = sorted(metadata_cols) + ordered_year_columns

    # then create a new dataframe with that order
    reordered_df = df[new_column_order]

    return reordered_df

def set_non_year_cols_as_index(df):

    """ 
    Identifies all metadata (not year columns) in the dataframe and sets as index. 
    This enables the user to then manipulate the data knowing it is all numeric. 
    The function also returns the column headings of the metadata, in case needed.   
    """

    # get year columns
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]

    # get other columns
    other_cols = sorted(list(set(df.columns) - set(year_cols)))
    
    # set other columns as index
    df = df.set_index(other_cols)
    
    # make sure that the years are in the right (numeric) order
    if year_cols:
        order_year_columns = sorted(year_cols, key=int)
        df = df[order_year_columns]
        
    return df, other_cols

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

def calculate_diff_since_yearX(df_abs, yearX):

    """
    Calculates in absolute and relative terms the difference between data in all years compared to the specified year.
    For example, % difference relative to 1990 in all years.
    """

    print('Calculating difference compared to ' + yearX)

    # first, check that the desired year is in the data!
    if yearX not in df_abs.columns:
        print('The year you have selected for relative calculations ('
              + str(yearX) + ') is not available, please try again.')
        return

    # calculate all columns relative to the chosen year, first in absolute terms, then %
    df_abs_diff = df_abs.subtract(df_abs[yearX], axis='index')
    # print(df_abs_diff)
    df_perc_diff = 100 * df_abs_diff.divide(df_abs[yearX], axis='index')
    # print(df_perc_diff)

    return df_abs_diff, df_perc_diff

def set_countries_as_index(df):

        """
        Identifies all metadata (not year columns) in the dataframe and sets as index.
        This enables the user to then manipulate the data knowing it is all numeric.
        The function also returns the column headings of the metadata, in case needed.
        """

        # if years labelled as YNNNN, switch to NNNN
        for col in df.columns:
            if col.startswith('Y'):
                if len(col) == 5:
                    df = df.rename(columns={col: col[1:]})

        # get year columns
        year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]

        # get other columns
        other_cols = list(set(df.columns) - set(year_cols))

        # set country columns as index
        df = df.set_index(['country'])

        # drop other columns
        # remaining_cols = list(set(other_cols) - set(['country']))

        # make sure that the years are in the right (numeric) order
        if year_cols:
            order_year_columns = sorted(year_cols, key=int)
            df = df[order_year_columns]

        return df

def filter_primap(renamed_data, gas, sector, scenario, countries, start_year, dict_gas, primap_sectors='gst_tools/primap_sectors.csv'):
    # LOU Reduce to only the desired variables
    proc_primap = renamed_data.loc[(renamed_data['gas'] == gas) &
                        (renamed_data['scenario'] == scenario) &
                        (renamed_data['category'] == sector)
                        ]
    
    if len(proc_primap.index) == 0:
        raise ValueError('There is no data for the gas, the sector, and the data source specified. Please, provide new parameters.')

    else:
    # LOU Reduce the countries or regions to only those desired
        proc_primap = proc_primap.loc[proc_primap['country'].isin(countries)]

        if len(proc_primap.index) == 0:
            raise ValueError('There is no data for the countries specified. Please, select different countries.')
        else:
        # LOU Tell the user if any of the needed countries are missing and, if yes, which ones:
            missing_countries = list(set(countries) - set(proc_primap['country'].unique()))
            if missing_countries:
                logging.info('Not all countries requested were available in the raw data. You are missing the following:')
                for country in missing_countries:
                    logging.info('   ' + to_name(country))
                logging.info('---------')

            # LOU Reduce to only required years
            proc_primap = change_first_year(proc_primap, start_year)

            # rename columns to follow conventions
            #proc_primap = proc_primap.rename(columns={'entity': 'variable'})

            # make sure 'variable' contains all necessary information
            proc_primap['variable'] = define_primap_variable_name(sector, gas, dict_gas, primap_sectors=primap_sectors)

            # label the source
            #proc_primap['source'] = source_name

            proc_primap = check_column_order(proc_primap)

            # EPO Check
            logging.debug('These are the 10 first rows of the processed data:')
            logging.debug(proc_primap.head(10))

            return proc_primap

def define_primap_proc_fname(primap_filtered, gas_dict, sect_dict):
    
    source = primap_filtered['source'][0]
    source_scenario = source.lower().replace('hist', primap_filtered['scenario'][0].lower())
    gas = gas_dict[primap_filtered['gas'][0]]
    sector = sect_dict[primap_filtered['category'][0]]
    
    proc_fname =  source_scenario + '_' + gas + '_' + sector + '.csv'

    return proc_fname

def define_bp_proc_fname(proc_data):
    source = 'bp_stat_rev_world_energy_2022'
    variable = proc_data['variable'][0]

    proc_fname = source + '_' + variable.replace(' ', '_').lower() + '.csv'

    return proc_fname

def define_plot_name(type, variable, year_of_interest, baseline_year, output_folder):
    if type != 1 and type != 2 and type != 3 and type != 4:
        print('Error: Please provide a valid plot type (either 1, 2, 3 or 4).')
        return
    else:
        if type == 1:
            type_text = 'Distribution in ' + str(year_of_interest)
        elif type == 2:
            type_text = 'Change since ' + str(baseline_year)
        elif type == 3:
            type_text = 'Average annual change'
        elif type == 4:
            type_text = 'Year of peaking'
        #plot_name = type_text + variable.lower()
        fname = output_folder + (type_text.lower() + ' ' + variable.lower()).replace(' ', '_') + '.png'

        return type_text, fname


def write_to_file(proc_data, proc_folder, proc_fname):
    # First ensure that years, unit, 'country', and variable are all in data. If they are can proceed to print data
    # WARNING What about other things to check?
    if 'country' not in proc_data.columns or 'unit' not in proc_data.columns:
        raise ValueError('Missing required information! Please check your input data and processing!')
    else:
        fullfname_out = os.path.join(proc_folder, proc_fname)

    # LOU Check folder exists
    if not os.path.exists(proc_folder):
        os.makedirs(proc_folder)

    # LOU Write to csv in proc data folder
    proc_data.to_csv(fullfname_out, index=False)

    # celebrate success 
    logging.debug('Processed data written to file! - ' + fullfname_out)

def rearrange_wb_data(input_folder, wb_dset_fname):
    dset = pd.read_csv(os.path.join(input_folder, wb_dset_fname), header=2)
    dset.rename(columns={'Country Code':'country',
                         'Indicator Name':'variable'}, inplace=True)
    return dset

def ensure_common_years(df1, df2):

    """
    Removes any years from either dataframe that are not in both dataframes.
    """

    def put_other_cols_as_index(df):
        year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
        other_cols = list(set(df.columns) - set(year_cols))
        df = df.set_index(other_cols)
        return df

    df1 = put_other_cols_as_index(df1)
    df2 = put_other_cols_as_index(df2)

    # find the same years
    df1_cols = df1.columns
    df2_cols = df2.columns
    common_cols = list(set(df1_cols).intersection(df2_cols))

    # reset matrices
    df1 = df1.loc[:, sorted(common_cols, key=int)]
    df2 = df2.loc[:, sorted(common_cols, key=int)]

    # return other columns
    df1 = df1.reset_index()
    df2 = df2.reset_index()

    return df1, df2

def ensure_common_countries(df1, df2):

    """
    Removes any countries from either dataframe that are not in both dataframes.
    """

    # find the same countries
    df1_countries = df1['country'].unique()
    df2_countries = df2['country'].unique()
    common_countries = list(set(df1_countries).intersection(df2_countries))

    logging.debug('Common countries are: ')
    logging.debug(common_countries)
    # TODO - spit out list of countries not found!

    # reset matrices
    df1 = df1.loc[df1['country'].isin(common_countries), :]
    df2 = df2.loc[df2['country'].isin(common_countries), :]

    return df1, df2

def verify_data_format(df):

    """
    To work for the gst_tools, all data read in needs to contain one and only one variable and
    the unit must be both specified and the same for all variables. The set of countries must be unique.
    All of these need to be in correctly labelled columns.
    It's also fairly pointless if there are no years in the data.
    This function checks for these aspects and tells the user what's wrong if it doesn't work.
    """

    verified = True

    # First, check for the right columns
    columns_required = ['variable', 'unit', 'country']
    column_check = all(elem in df.columns for elem in columns_required)
    if column_check:
        verified = True
    else:
        logging.warning('Missing columns in dataframe! Columns missing are:')
        logging.warning(set(columns_required) - set(list(df.columns)))
        verified = False
        return verified

    # check the uniqueness of the data
    if len(df['variable'].unique()) != 1:
        logging.warning('WARNING: the "variable" is non-unique! Please check your input data!')
        verified = False
        return verified

    if len(df['unit'].unique()) != 1:
        logging.warning('WARNING: the "units" are non-unique! Please check your input data!')
        verified = False
        return verified

    # check that no countries are repeated
    if len(df['country'].unique()) != len(df['country']):
        logging.warning('WARNING: Some countries appear to be repeated! Please check your input data!')
        verified = False
        return verified

    # make sure that there are some year columns
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
    if len(year_cols) == 0:
        logging.warning("WARNING: there don't appear to be any year columns! Please check your input data!")
        verified = False
        return verified

    # if all the checks are passed, return True
    return verified

def prep_df_for_division(df):
    
    df = df.set_index('country')
    
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
    other_cols = list(set(df.columns) - set(year_cols))
    
    df = df.drop(other_cols, axis='columns')
    
    return df

def normalise(proc_data, normalising_dset, per_capita_or_per_usd):
    var1, var2 = ensure_common_years(proc_data, normalising_dset)
    var1, var2 = ensure_common_countries(var1, var2)
    
    if per_capita_or_per_usd == 'per capita':
        var2['unit'] = ['pers']*len(var2)
    elif per_capita_or_per_usd == 'per USD':
        var2['unit'] = ['usd']*len(var2)
    else:
        raise ValueError('Error: Please select either "per capita" or "per USD" for normalisation')
    
    check1 = verify_data_format(var1)
    check2 = verify_data_format(var2)

    if not check1 or not check2:
        raise ValueError('One of the dataframes is not correct! Please check and try again!')
    else:
        # LOU Get metadata for later use and checking
        var1_name = var1['variable'].unique()[0]
        #var2_name = var2['variable'].unique()[0]

        var1_unit  = var1['unit'].unique()[0]
        var2_unit = var2['unit'].unique()[0]

        var1 = prep_df_for_division(var1)
        var2 = prep_df_for_division(var2)

        data_normalised = var1 / var2

        # LOU Generate new metadata
        new_variable_name = var1_name + ' ' + per_capita_or_per_usd
#with open('gst_tools/name_relative_variable.txt', 'w') as f:
 #   f.write(new_variable_name)
        data_normalised['variable'] = new_variable_name

        # LOU Automatically generate the unit 
        data_normalised['unit'] = var1_unit + '/' + var2_unit
    
        data_normalised = data_normalised.reset_index()

        # LOU Reorganise dataframe
        data_normalised = check_column_order(data_normalised)

        return data_normalised

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
    
    # LOU Convert the data
    conv_df, other_cols = set_non_year_cols_as_index(conv_df)
    conv_df = conv_df * conversion_factor
    conv_df = conv_df.reset_index()

    return conv_df

def convert_norm(normalised_dset, data, per_capita_or_per_usd):
    conv_df = normalised_dset.copy()
    org_unit = normalised_dset['unit'].unique()[0]

    if data != 'emissions' and data != 'energy':
        raise ValueError('Error. Please provide a valid data type (either "emissions" or "energy".')
    else:
        if data == 'emissions':
            gas_unit = re.search('Mt(\w+)', org_unit).group(1)

            desired_unit = 't' + gas_unit + ' ' + per_capita_or_per_usd

            conversion_factor = 1000000
            if per_capita_or_per_usd == 'per USD':
                conversion_factor = 1000000 *(10**3)
                desired_unit = 'kg' + gas_unit + ' ' + per_capita_or_per_usd
        elif data == 'energy':
            desired_unit = 'J' + ' ' + per_capita_or_per_usd
            conversion_factor = 10**9
            if per_capita_or_per_usd == 'per USD':
                conversion_factor = 10**12
                desired_unit = 'mJ' + ' ' + per_capita_or_per_usd

        logging.debug('*******************')
        logging.debug('Converting unit from "' + org_unit + '" to "' + desired_unit + 
            '" using a conversion factor of ' + str(conversion_factor))
        logging.debug('*******************')

        conv_df['unit'] = desired_unit
        
        # LOU Convert the data
        conv_df, other_cols = set_non_year_cols_as_index(conv_df)
        conv_df = conv_df * conversion_factor
        conv_df = conv_df.reset_index()

        return conv_df

def prepare_for_plotting(final_dset, plot_type):
    if not verify_data_format(final_dset):
        raise ValueError('WARNING: The data is not correctly formatted! Please check before continuing!')
    # LOU Extract the key information
    else:
        variable = final_dset['variable'].unique()[0]
        unit = final_dset['unit'].unique()[0]

        # LOU Tidy up for next steps
        data_years = set_countries_as_index(final_dset)
        data_years = data_years.dropna(axis=1, how='all')
        data_years = data_years.dropna(axis=0, how='any')

        if plot_type == 4:
            year_max = data_years.idxmax(axis=1)
            year_max = pd.to_numeric(year_max)
            year_max.name = 'peak year'

            start_year = min(list(map(int, data_years.columns)))
            end_year = max(list(map(int, data_years.columns)))

            return year_max, start_year, end_year, data_years, variable, unit

        else:
            return data_years, variable, unit

def calculate_trends(df, num_years_trend=5):

    """
    Calculates the annual percentage change and also the rolling average over the specified number of years.
    """

    # disp average used for trend
    print('Averaging trend over ' + str(num_years_trend) + ' years.')

    # calculate annual % changes
    df_perc_change = df.pct_change(axis='columns') * 100
    new_unit = '%'

    # average over a window
    df_rolling_average = df_perc_change.rolling(window=num_years_trend, axis='columns').mean()

    return df_perc_change, df_rolling_average, new_unit

def get_uba_colours():

    # UBA dict from Annika Guenther
    uba_colours = {}
    uba_colours['uba_bright_green'] = [xx / 255 for xx in (94, 173, 53)]
    uba_colours['uba_dark_green'] = [xx / 255 for xx in (0, 118, 38)]
    uba_colours['uba_bright_blue'] = [xx / 255 for xx in (0, 155, 213)]
    uba_colours['uba_dark_blue'] = [xx / 255 for xx in (0, 95, 133)]
    uba_colours['uba_bright_orange'] = [xx / 255 for xx in (250, 187, 0)]
    uba_colours['uba_dark_orange'] = [xx / 255 for xx in (215, 132, 0)]
    uba_colours['uba_bright_pink'] = [xx / 255 for xx in (206, 31, 94)]
    uba_colours['uba_dark_pink'] = [xx / 255 for xx in (131, 5, 60)]
    uba_colours['uba_bright_purple'] = [xx / 255 for xx in (157, 87, 154)]
    uba_colours['uba_dark_purple'] = [xx / 255 for xx in (98, 47, 99)]
    uba_colours['uba_bright_grey'] = [xx / 255 for xx in (240, 241, 241)]
    uba_colours['uba_dark_grey'] = [xx / 255 for xx in (75, 75, 55)]

    return uba_colours

def set_uba_palette():

    uba_palette = [
                [xx / 255 for xx in (0, 118, 38)],
                [xx / 255 for xx in (18, 93, 134)],
                [xx / 255 for xx in (98, 47, 99)],
                [xx / 255 for xx in (215, 132, 0)],
                [xx / 255 for xx in (131, 5, 60)],
                [xx / 255 for xx in (97, 185, 49)],
                [xx / 255 for xx in (0, 155, 213)],
                [xx / 255 for xx in (157, 87, 154)],
                [xx / 255 for xx in (250, 187, 0)],
                [xx / 255 for xx in (206, 31, 94)],
                [xx / 255 for xx in (240, 241, 241)],
                [xx / 255 for xx in (75, 75, 55)]
                  ]

    return uba_palette

def eliminate_outliers(series, ktuk=3):
    # Outliers - in some cases, the date contains extreme outliers. These make for an unreadable
    # plot and in most cases arise from exceptional circumstances. These outliers are therefore removed
    # from the plots and the removal signalled to the user.
    # Example: Equatorial Guinea's emissions rose dramatically in the mid-90s due to the discovery of
    # oil. So much so, that the current emissions relative to 1990 are over 6000% higher. Including these
    # emissions in the plots would render a useless graph so we remove this country from the overview.

    # Use Tukey's fences and the interquartile range to set the bounds of the data
    # https://en.wikipedia.org/wiki/Outlier
    # For reference: kTUk default is set to 3 (above)
    # k = 1.5 -> outlier; k = 3 -> far out
    # TODO - get full and proper reference for this!!!

    logging.debug('-----------')
    logging.debug('Identifying and removing outliers')

    # calculate limits
    q75, q25 = np.percentile(series, [75, 25])
    iqr = q75 - q25
    tukey_min = q25 - ktuk * iqr
    tukey_max = q75 + ktuk * iqr
    # for testing:
    # logging.debug('tukey_min is ' + str(tukey_min))
    # logging.debug('tukey_max is ' + str(tukey_max))

    # Tell the user what the outliers are:
    lower_outliers = series[series < tukey_min]
    logging.debug('lower outliers are:')
    logging.debug(lower_outliers)
    upper_outliers = series[series > tukey_max]
    logging.debug('upper outliers are: ')
    logging.debug(upper_outliers)
    logging.debug('---')

    noutliers = len(lower_outliers) + len(upper_outliers)

    # actually remove the outliers
    series = series[(series > tukey_min) & (series < tukey_max)]

    return series, noutliers

def get_plot_stats(series):
    # get some basic info about the data to use for setting styles, calculating bin sizes, and annotating plot
    maximum = int(max(series))
    minimum = int(min(series))
    mean = np.mean(series)
    median = np.median(series)
    npts = len(series)

    # Use data metrics to determine which approach to use for bins.
    if (minimum < 0) & (maximum > 0):

        # If both positive and negative, bins should be symmetric around 0!
        # What's the range of data?
        full_range = np.ceil(maximum - minimum)

        # Freedman–Diaconis rule
        # (need to recalculate IQR)
        q75, q25 = np.percentile(series, [75, 25])

        iqr = q75 - q25

        if (int(2 * (iqr) / (npts ** (1 / 3)))) != 0:
            bin_width = int(2 * (iqr) / (npts ** (1 / 3)))
        else:
            bin_width = 1

        # or the simple 'excel' rule:
        # bin_width = int(full_range / np.ceil(npts**(0.5)))

        # for nbins, need to take into account asymmetric distribution around 0
        nbins = int(np.ceil(2 * max([abs(minimum), abs(maximum)])) / bin_width)
        if not (nbins / 2).is_integer():
            nbins = nbins + 1

        # determine bin edges
        bins_calc = range(int((0 - (1 + nbins / 2) * bin_width)), int((0 + (1 + nbins / 2) * bin_width)), bin_width)
        logging.debug('bins set to ' + str(bins_calc))

    else:
        if maximum < 25:

            bin_width = 1

            # or the simple 'excel' rule:
            # bin_width = int(full_range / np.ceil(npts**(0.5)))

            # for nbins, need to take into account asymmetric distribution around 0
            nbins = np.ceil(abs(maximum))

            # determine bin edges
            bins_calc = range(0, int(1 + nbins), bin_width)
            logging.debug('bins set to ' + str(bins_calc))

        else:
            # use inbuilt Freedman-Diaconis
            # ? TODO - modify to ensure integers? or replicate above?
            bins_calc = 'fd'
    return maximum, minimum, mean, median, npts, bins_calc

def plot(series, bins_calc, colour):
    fig, axs = plt.subplots()

    # make histogram
    sns.distplot(series,
                 kde=False,
                 bins=bins_calc,
                 hist_kws=dict(alpha=0.75),
                 color=colour)
                 #rug=False,
                 #rug_kws={"color": "rebeccapurple", "alpha": 0.7, "linewidth": 0.4, "height": 0.03})

    return fig, axs

def make_symmetric_around_zero(axs, series, xmin, xmax):
    # reset xmin or xmax
    if np.absolute(xmax) > np.absolute(xmin):
        plt.xlim(-xmax, xmax)
    else:
        plt.xlim(xmin, -xmin)

    # and add a line at 0
    axs.axvline(linewidth=1, color='k')

    # and annotate with the number of countries either side of the line
    # ARROWS!

    nbelow = len(series[series < 0])
    nabove = len(series[series > 0])

    axs.annotate(str(nbelow) + ' countries',
                xytext=(0.31, 1.0), xycoords=axs.transAxes,
                fontsize=9, color='black',
                xy=(0.15, 1.01),
                arrowprops=dict(arrowstyle="-|>", color='black'),
                bbox=dict(facecolor='white', edgecolor='grey', alpha=0.75)
                )
    axs.annotate(str(nabove) + ' countries',
                xytext=(0.54, 1.0), xycoords=axs.transAxes,
                fontsize=9, color='black',
                xy=(0.85, 1.01),
                arrowprops=dict(arrowstyle="-|>", color='black'),
                bbox=dict(facecolor='white', edgecolor='grey', alpha=0.75)
                )    

def add_selected_country(axs, xmin, xmax, selected_country, country_value, unit, uba_colours):

    if (country_value > xmin) & (country_value < xmax):
        # indicate it on the plot
        axs.axvline(x=country_value, ymax=0.9, linewidth=1.5, color=uba_colours['uba_dark_purple'])

        # annotate with country name
        ymin, ymax = axs.get_ylim()
        ypos = 0.65 * ymax
        axs.annotate((to_name(selected_country) + ' ' + "\n{:.2g}".format(country_value)) + ' ' + unit,
                    xy=(country_value, ypos), xycoords='data',
                    fontsize=9, color=uba_colours['uba_dark_purple'],
                    bbox=dict(facecolor='white', edgecolor=uba_colours['uba_dark_purple'], alpha=0.75)
                    )

    else:
        axs.annotate((to_name(selected_country) + ' ' + "\n{:.2g}".format(country_value)) + ' ' + unit,
                    xy=(.75, .65), xycoords=axs.transAxes,
                    fontsize=9, color=uba_colours['uba_dark_purple'],
                    bbox=dict(facecolor='white', edgecolor=uba_colours['uba_dark_purple'], alpha=0.75)
                    )

def annotate_plot(axs, xlabel, title, unit, sourcename, maximum, minimum, mean, median, npts, remove_outliers, noutliers):
    axs.annotate(("Data source: \n " + sourcename + "\n"
                  "\n Maximum  = {:.2f}".format(maximum) +
                  "\n Minimum   = {:.2f}".format(minimum) +
                  "\n Mean        = {:.2f}".format(mean) +
                  "\n Median     = {:.2f}".format(median) +
                  "\n Number of \n countries  = {:.0f}".format(npts)
                  ),
                 xy=(1.05, 0.6), xycoords=axs.transAxes,
                 fontsize=9, color='black',
                 bbox=dict(facecolor='white', edgecolor='grey', alpha=0.75))

    # if some countries were removed, indicate on the plot
    if remove_outliers:
        if noutliers == 1:
            sing_or_plur = ' outlier'
        else:
            sing_or_plur = ' outliers'
        axs.annotate(('  ' + str(noutliers) + sing_or_plur + ' not shown'),
                     xy=(1.05, 0.53), xycoords=axs.transAxes,
                     fontsize=8, color='black')

    # label axes and add title
    axs.set_xlabel((xlabel + ' \n(' + unit + ')'), fontsize=12)
    axs.set_ylabel('Number of countries', fontsize=12)
    axs.set_title((title + "\n"), fontweight='bold')


def plot_facet_grid_countries(df, variable, value, main_title='', plot_name='', save_plot=False):

    """
    plot a facet grid of variables for a range of countries. Can be used to, e.g. assess
    which countries have emissions that have peaked, and which not.
    """

    uba_palette = set_uba_palette()
    sns.set_palette(uba_palette)
    sns.set(style="darkgrid", context="paper")
    uba_colours = get_uba_colours()
    sns.set(font="Calibri")

    # First, get some idea of the data so that it's easier to make clean plots
    ranges = df.max(axis=1) - df.min(axis=1)
    check = (ranges.max() - ranges.min()) / ranges.min()
    if abs(check) < 8:
        yshare = True
    else:
        yshare = False

    # set up the df for plotting
    year_cols = df.columns
    dftomelt = df.reset_index()
    dftomelt['country'] = dftomelt['country'].apply(to_name)
    dfmelt = pd.melt(dftomelt, id_vars=['country'],
                     value_vars=year_cols, var_name=variable, value_name=value)

    # set up the grid
    grid = sns.FacetGrid(dfmelt, col='country', palette="tab20c", sharey=yshare,
                         col_wrap=4, aspect=1)

    # make the actual plots
    grid.map(sns.lineplot, variable, value, color=uba_colours['uba_dark_purple'])

    # Give subplots nice titles
    grid.set_titles(col_template='{col_name}')

    # tidy up a bit
    for ax in grid.axes.flat:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, prune="both"))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4, prune="both"))
        ax.axhline(0, color='k')
    if yshare:
        grid.fig.subplots_adjust(hspace=.15, wspace=.1, top=.95)
    else:
        grid.fig.subplots_adjust(hspace=.15, wspace=.25, top=.95)

    # give the whole plot a title
    grid.fig.suptitle(main_title, fontweight='bold', fontsize=15)

    if save_plot:
        filepath = os.path.join('output', 'plots')
        # grid.map(horiz_zero_line)
        fname = ('facetgrid-' + plot_name + '-' + value + '.pdf')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = os.path.join(filepath, fname)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

def peaking_barplot(summary_data, variable, max_year, save_plot=False):

    uba_palette = set_uba_palette()
    sns.set_palette(uba_palette)
    sns.set(style="darkgrid", context="paper")
    sns.set(font="Calibri")

    # make histogram
    fig, ax = plt.subplots()

    splot= sns.barplot(x=summary_data['category'], y=summary_data['count'],
                       alpha=0.85, palette=uba_palette)

    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.0f'),
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center',
                  xytext=(0, 10), textcoords='offset points')

    plt.tight_layout
    plt.xlabel('')
    plt.ylabel('number of countries')
    plt.title("Status of " + variable + "\nin " + max_year)

    if save_plot:
        filepath = os.path.join('output', 'plots')
        fname = ('peaking-categories-' + variable + '.png')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = os.path.join(filepath, fname)
        plt.savefig(filename, format='png', dpi=600, bbox_inches='tight')
        plt.close()


def make_histogram_peaking(df, var, unit_, start_year, end_year, save_plot=False):

    """
    This function is specifically written to plot the peaking year of a variable for a range
    of countries.
    """

    uba_palette = set_uba_palette()
    sns.set_palette(uba_palette)
    sns.set(style="darkgrid", context="paper")
    sns.set(font="Calibri")

    # Check the data - needs to not be, for example, all zeros
    if len(df.unique()) == 1:
        print('---------')
        print('All values in the series are the same! Exiting plotting routine for ' + str(var))
        print('---------')
        return

    # set a style
    sns.set(style="darkgrid")

    # STATS
    # get some basic info about the data to use for setting styles, calculating bin sizes, and annotating plot
    maximum = int(max(df))
    minimum = int(min(df))
    mean = np.mean(df)
    median = np.median(df)
    npts = len(df)

    # determine bin edges - annual!
    bin_width = 1
    bins_calc = range((start_year - 1), (end_year + 2), bin_width)

    # --------------
    # MAKE THE PLOT

    # set up the figure
    fig, axs = plt.subplots()

    uba_colours = get_uba_colours()

    # make histogram
    N, bins, patches = axs.hist(df, bins=bins_calc,
                                edgecolor='white', linewidth=1)

    for i in range(0, len(patches)):
        patches[i].set_facecolor(uba_colours['uba_dark_purple'])
    patches[-1].set_facecolor(uba_colours['uba_bright_orange'])
    patches[-1].set_alpha(0.5)

    # Dynamically set x axis range to make symmetric abut 0
    if minimum < 0:
        # get and reset xmin or xmax
        xmin, xmax = axs.get_xlim()
        if np.absolute(xmax) > np.absolute(xmin):
            plt.xlim(-xmax, xmax)
        else:
            plt.xlim(xmin, -xmin)

        # and add a line at 0
        axs.axvline(linewidth=1, color='k')

    # number of countries in the last bin
    nlast = N[-1]

    # Annotate the plot with stats
    axs.annotate(("{:.0f} countries, ".format(npts) +
                  "\nof which {:.0f} have ".format(nlast) +
                  "\nnot yet reached a maximum (orange)"),
                 xy=(0.03, 0.82), xycoords=axs.transAxes,
                 fontsize=10, color='black',
                 bbox=dict(facecolor='white', edgecolor='grey', alpha=0.75))

    # label axes and add title
    axs.set_xlabel('year')
    axs.set_ylabel('number of countries')
    axs.set_title(('year when ' + var + ' peaked'), fontweight='bold')

    # save to file
    if save_plot:
        filepath = os.path.join('output', 'plots')
        fname = ('basic_histogram-peaking-since-' + str(start_year) + '-' + var + '.png')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = os.path.join(filepath, fname)
        plt.savefig(filename, format='png', dpi=450, bbox_inches='tight')
        plt.close()

    # show the plot
    plt.show()

def make_histogram(df, year, unit, plot_type=0, xlabel='', variable_title='', 
                sourcename='unspecified', save_plot=True, filepath='',
                remove_outliers=True, ktuk=3,
                plot_name='', selected_country='', dpi=600):

    """
    This is based on the make_simple_histogram function but caters to data that
    contains both positive and negative values. For the GST, it's important to be
    able to see whether or not trends etc. are positive or negative and a symmetric
    binning approach is needed.

    To calculate the bin sizes, we use a couple of conditional rules based on the data
    available, including the max and min of the data and the number of data points.
    For most plots we are expecting around 200 countries, but could also be a few regions.

    TODO - 'df' is actually a series -> better name? (NO MORE)
    TODO - edit selected country option to deal with ISO codes or names.
    """

    # announce the plot..
    logging.debug('---------')
    logging.debug('Making plot for: ' + str(plot_name))
    logging.debug('---------')

    if plot_type == 1 or plot_type == 2:
        series = df[str(year)]
    else:
        series = df.iloc[:,-1]

    # Check the data - needs to not be, for example, all zeros
    if len(series.unique()) == 1:
        raise ValueError('All values in the series are the same! Exiting plotting routine for ' + str(plot_name))                

    # get the value here in case it's excluded as an outlier
    if selected_country:
        # get value of that country
        country_value = series[selected_country]

    # set a style
    # attempting to modify to UBA grid style but didn't work.
    #sns.set_style('darkgrid', {'xtick.color': '.95', 'grid.color': '.5'})
    sns.set(style='darkgrid')
    sns.set_palette(set_uba_palette())
    uba_colours = get_uba_colours()
    sns.set(font="Calibri")

    if remove_outliers:
        series, noutliers = eliminate_outliers(series, ktuk=ktuk)
    else:
        noutliers = 0

    # STATS
    # get some basic info about the data to use for setting styles, calculating bin sizes, and annotating plot
    maximum, minimum, mean, median, npts, bins_calc = get_plot_stats(series)

    # --------------
    # MAKE THE PLOT

    # set up the figure
    fig, axs = plot(series, bins_calc, uba_colours['uba_dark_green'])
    fig.patch.set_facecolor('white')
    xmin, xmax = axs.get_xlim()

    if minimum < 0:
        make_symmetric_around_zero(axs, series, xmin, xmax)

    # If a country is selected for highlighting, then indicate it on the plot!
    if selected_country:
        add_selected_country(axs, xmin, xmax, selected_country, country_value, unit, uba_colours)

    # Annotate the plot with stats
    title = variable_title + ' in ' + str(year)

    annotate_plot(axs, xlabel, title, unit, sourcename, maximum, minimum, mean, median, npts, remove_outliers, noutliers)

    # save to file
    
    
    
    if save_plot:
        if selected_country:
            filepath = filepath.replace('/', '/' + to_name(selected_country))

        #if not os.path.exists(filepath):
        #    os.makedirs(filepath)
        #filename = os.path.join(filepath, fname)
        plt.savefig(filepath, format='png', dpi=dpi, bbox_inches='tight')
        plt.close()

    # show the plot
    return plt