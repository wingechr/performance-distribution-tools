# Project and Title

# Author(s): Louise Jeffery
# Contact: louise.jeffery@pik-potsdam.de; l.jeffery@newlcimate.org
# Date: 2019

# Copyright License:
# 

# Purpose:
# 

# =====================================================

import re
import sys

import pandas as pd
import numpy as np

# ======================


def set_non_year_cols_as_index(df):

    """ 
    Identifies all metadata (not year columns) in the dataframe and sets as index. 
    This enables the user to then manipulate the data knowing it is all numeric. 
    The function also returns the column headings of the metadata, in case needed.   
    """

    # get year columns
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]

    # get other columns
    other_cols = list(set(df.columns) - set(year_cols))
    
    # set other columns as index
    df = df.set_index(other_cols)
    
    # make sure that the years are in the right (numeric) order
    if year_cols:
        order_year_columns = sorted(year_cols, key=int)
        df = df[order_year_columns]
        
    return df, other_cols


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


def calculate_trends(df, num_years_trend=10):

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


def change_first_year(df, new_start_year):

    """
    Reduces a dataframe to start in a later, specified year than the original data.
    Can be useful for reducing the size of the data to be handled or for performing analysis over a reduced timeframe.
    """

    # reduce the number of years (keeps things lighter and faster)
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
    other_cols = list(set(df.columns) - set(year_cols))

    # check the current start year
    cur_start_year = min(year_cols)
    last_year = max(year_cols)

    if int(cur_start_year) > new_start_year:

        new_start_year = cur_start_year

    else:
        # set other columns as index
        if other_cols:
            df = df.set_index(other_cols)

        # identify which years to keep
        years_to_keep = np.arange(new_start_year, (int(last_year) + 1), 1)
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
    print('First year of data available is now ' + str(new_start_year))
    print('Last year of data available is ' + str(last_year))

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

    print('Common countries are: ')
    print(common_countries)
    # TODO - spit out list of countries not found!

    # reset matrices
    df1 = df1.loc[df1['country'].isin(common_countries), :]
    df2 = df2.loc[df2['country'].isin(common_countries), :]

    return df1, df2


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
        print('Missing columns in dataframe! Columns missing are:')
        print(set(columns_required) - set(list(df.columns)))
        verified = False
        return verified

    # check the uniqueness of the data
    if len(df['variable'].unique()) != 1:
        print('WARNING: the "variable" is non-unique! Please check your input data!')
        verified = False
        return verified

    if len(df['unit'].unique()) != 1:
        print('WARNING: the "units" are non-unique! Please check your input data!')
        verified = False
        return verified

    # check that no countries are repeated
    if len(df['country'].unique()) != len(df['country']):
        print('WARNING: Some countries appear to be repeated! Please check your input data!')
        verified = False
        return verified

    # make sure that there are some year columns
    year_cols = [y for y in df[df.columns] if (re.match(r"[0-9]{4,7}$", str(y)) is not None)]
    if len(year_cols) == 0:
        print("WARNING: there don't appear to be any year columns! Please check your input data!")
        verified = False
        return verified

    # if all the checks are passed, return True
    return verified


def make_uba_color_dict():

    """
    Define a dict with the UBA colours contained so it can be used for the plots.
    """

    # Thank you to Annika Guenther for the numbers!

    uba_colours = dict
    uba_colours['uba_green'] = [xx / 255 for xx in (0, 118, 38)]  # 'darkgreen'
    uba_colours['uba_2'] = [xx / 255 for xx in (97, 185, 49)]  # 'mediumspringgreen'
    uba_colours['uba_3'] = [xx / 255 for xx in (0, 155, 213)]  # 'dodgerblue'
    uba_colours['uba_4'] = [xx / 255 for xx in (18, 93, 134)]  # 'navy'
    uba_colours['uba_5'] = [xx / 255 for xx in (250, 187, 0)]  # 'deeppink'
    uba_colours['uba_6'] = [xx / 255 for xx in (131, 5, 60)]  # 'orange'
    uba_colours['uba_7'] = [xx / 255 for xx in (206, 31, 94)]  # 'purple'
    uba_colours['uba_8'] = [xx / 255 for xx in (215, 132, 0)]  # 'magenta'
    uba_colours['uba_9'] = [xx / 255 for xx in (157, 87, 154)]  # 'magenta'
    uba_colours['uba_10'] = [xx / 255 for xx in (98, 47, 99)]  # 'magenta'

    return uba_colours


def convert_ISO2_to_ISO3():

    """
    convert a country 2-letter ISO code to a 3-letter ISO code.
    """

    country_code_file = "country_codes.csv"
    country_codes = pd.read_csv(country_code_file)




def convert_ISO3_to_name():

    """
    convert a 3-letter ISO code to a full country name
    """

# EPO:
def get_primap_variable_and_and_file_name(gas_name, raw_sector, raw_scenario, dset_version):
    
    sector_codes = pd.read_csv('primap_sectors.csv')
    sector_codes.set_index('code', inplace=True)

    source_name = 'PRIMAP-' + raw_scenario.lower() + '_v' + dset_version
    #source_name_display = 'PRIMAP-' + raw_scenario.lower()

    if raw_sector in ['M.0.EL', '1.B.3', '2.G', '2.H']:
        variable_name_to_display = sector_codes.loc[raw_sector]['1'] + ' ' + gas_name + ' ' + sector_codes.loc[raw_sector]['3'] + ' ' + sector_codes.loc[raw_sector]['4']
    elif raw_sector in ['1', '1.B', '5', '1.B.2']:
        variable_name_to_display = sector_codes.loc[raw_sector]['1'] + ' ' + gas_name + ' ' + sector_codes.loc[raw_sector]['3']
    elif raw_sector in ['2.D', '2.F', '4', '1.C', '1.A', '3.A', '1.B.1', '2', '2.A', '2.B', '2.C', '2.E', 'M.AG', 'M.AG.ELV']:
        variable_name_to_display = gas_name + ' ' + sector_codes.loc[raw_sector]['2'] + ' ' + sector_codes.loc[raw_sector]['3']
    else:
        print('The sector code introduced is incorrect. Please include a valid sector code.')

    # Output file
    proc_data_fname = source_name + '_' + variable_name_to_display.replace(' ', '_') + '.csv'

    return variable_name_to_display, proc_data_fname, source_name

# EPO
def define_dataset(population_fname, gdp_fname, other_fname, population_dset_name, gdp_dset_name, other_dataset_name, pop, gdp):
    if pop == True and gdp == False:
        raw_other_dataset = population_fname
        dset_name = population_dset_name
        return raw_other_dataset, dset_name
    elif gdp == True and pop == False:
    # Change here the name of the GDP dataset
        raw_other_dataset = gdp_fname
        dset_name = gdp_dset_name
        return raw_other_dataset, dset_name
    elif gdp == False and pop == False:
    # Change here the name of the other dataset
        raw_other_dataset = other_fname
        dset_name = other_dataset_name
        return raw_other_dataset, dset_name
    else:
        print('Error. Please check the options selected to define the secondary dataset (only one can be true).')

# moved by EPO
def convert_to_kt(new_df, population, gdp, other_unit='None'):
    if population == True:
        desired_unit = 'ktCO2/capita'
    elif gdp == True:
        desired_unit = 'ktCO2/USD'
    else:
        desired_unit = 'ktCO2/' + other_unit
    
    
    conversion_factor = 1000
    # For reference: 
    # * 1 Gg / Thousand Pers = 1 t / person

    conv_df = new_df.copy()
    org_unit = new_df['unit'].unique()

    print('*******************')
    print('Converting unit from "' + org_unit + '" to "' + desired_unit + 
          '" using a conversion factor of ' + str(conversion_factor))
    print('*******************')

    conv_df['unit'] = desired_unit
    
    # convert the data
    conv_df, other_cols = set_non_year_cols_as_index(conv_df)
    conv_df = conv_df * conversion_factor
    conv_df = conv_df.reset_index()

    return conv_df
