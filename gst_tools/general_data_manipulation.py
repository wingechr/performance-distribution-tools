import re
import os
import logging

import pandas as pd
import numpy as np
from shortcountrynames import to_name

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Functions for data loading and manipulation
# ======================

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

def convert_norm(normalised_dset, data, per_capita_or_per_usd):
    conv_df = normalised_dset.copy()
    org_unit = normalised_dset['unit'].unique()[0]

    if data != 'emissions' and data != 'energy':
        raise ValueError('Error. Please provide a valid data type (either "emissions" or "energy".)')
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
        
        # Convert the data
        conv_df, other_cols = set_non_year_cols_as_index(conv_df)
        conv_df = conv_df * conversion_factor
        conv_df = conv_df.reset_index()

        return conv_df

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

def load_data(folder, fname):
    file_path = os.path.join(folder, fname)
    logging.debug('Reading ' + file_path)
    raw_data = pd.read_csv(file_path)
    return raw_data

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

def prep_df_for_division(df):
    
    df = df.set_index('country')
    
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
    other_cols = list(set(df.columns) - set(year_cols))
    
    df = df.drop(other_cols, axis='columns')
    
    return df

def rearrange_wb_data(input_folder, wb_dset_fname):
    dset = pd.read_csv(os.path.join(input_folder, wb_dset_fname), header=2)
    dset.rename(columns={'Country Code':'country',
                         'Indicator Name':'variable'}, inplace=True)
    return dset

def rename_bp(raw_bp):
    raw_data_renamed = raw_bp.rename(columns={
                        'ISO3166_alpha3': 'country',
                        'Year': 'year'}, inplace=False)
    raw_data_renamed = raw_data_renamed.astype({'year': str})  
    return raw_data_renamed

def rename_primap(raw_primap):
    raw_data_renamed = raw_primap.rename(columns={'scenario (PRIMAP-hist)': 'scenario',
                         'area (ISO3)': 'country',
                         'category (IPCC2006_PRIMAP)': 'category',
                         'entity': 'gas'}, inplace=False)
    raw_data_renamed['country'].replace({'EU27BX': 'EUU'}, inplace=True)
    return raw_data_renamed

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