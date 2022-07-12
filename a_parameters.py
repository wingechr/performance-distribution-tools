#   This is the parameters file. To run any of the scripts in this project, you must first
#   specify here the variable, the gas, and the years that you want to plot.

from countrygroups import UNFCCC, EUROPEAN_UNION, ANNEX_ONE, NON_ANNEX_ONE

####        1. PROCESSING EMISSIONS DATASET        ####

# Include information below about the emissions dataset. The emissions dataset will be filtered to include
# the specified data. The processed dataset is ready for plotting in step 3.

# Include extrapolated data? Yes = True, No = False
include_extrapolated_data = True

# Please check whether the dataset has been updated on the PRIMAP website
#                   and update the file name here if that's the case.
if include_extrapolated_data:
    # Change here the name of the file with extrapolated data
    raw_data_file = 'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_20-Sep_2021.csv'
else:
    # Change here the name of the file without extrapolated data
    raw_data_file = 'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_no_extrap_20-Sep_2021.csv'

# Change here the version of the dataset
version = '2.3.1'

# Select here the gas to plot (for more info, see the gas_names dictionary below)
raw_entity = 'CO2'

# Select here the sector to plot (for more info, see the sector_names dictionary below)
raw_sector = 'M.0.EL'

# Select here the type of data that you want to plot.
    # Options are 'HISTCR' (country-reported) and 'HISTTP' (third-party).
raw_scenario = 'HISTCR'

# Select here the countries. See the options above (line 4).
needed_countries = UNFCCC

# Select here the years that you want to plot
years_of_interest = ['1990', '2005', '2016']

# The first year of the processed emissions dataset is set to be the first year plotted.
    # If you want to change the first year in the emissions dataset, please
    # overwrite "int(years_of_interest[0])" with the desired year.
start_year = int(years_of_interest[0])

# Do you want to save the plots as files? Yes = True, No = False
save_opt = True

####        2. PROCESSING SECONDARY DATASET     ####

# Here include data about a secondary dataset, in case you want to plot a relative indicator (e.g., per capita)

# Plotting absolute or relative emissions data? (In case you want to plot a relative indicator, set to False.)
absolute = False

# Select here whether the secondary dataset contains population data, GDP data, or none
population = True
gdp = False

# Change here the name of the population dataset and update its name
population_fname = 'API_SP.POP.TOTL_DS2_en_csv_v2_4218816.csv'
population_dset_name = 'Population_World_Bank_250522'

# Change here the name of the GDP dataset and update its name
gdp_fname = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4150784.csv'
gdp_dset_name = 'GDP_World_Bank_250522'

# Change here the name of the secondary dataset (when population and GDP are both false)
    # the name of the dataset, and the unit name
other_fname = ''
other_dset_name = ''
other_dset_unit = ''

# The program will divide the emissions dataset by the secondary dataset. Do you want to convert the
# resulting dataset from t[gas] to kt[gas]?
convert_unit = False


####        I. Dictionaries of sectors and gases        ####

# These are the different options that you can choose for the name of the gas
# and for the sector code. The different options are in the column on the left.

sector_names = {
   'M.0.EL':   'Total (excl.LULUCF)',
   '1':        'Energy',
   '1.A':      'Fuel combustion',
   '1.B':      'Fugitive',
   '1.B.1':    'Solid fuel',
   '1.B.2':    'Oil and gas',
   '1.B.3':    'other_from_energy_prod',
   '1.C':      'CO2_transport_and_storage',
   '2':        'IPPU',
   '2.A':      'mineral_industry',
   '2.B':      'chemical_industry',
   '2.C':      'metal_industry',
   '2.D':      'non-energy_products_from_fuels_and_solvents',
   '2.E':      'electronics_industry',
   '2.F':      'product_use_substitutes_for_ozone_depl_subs',
   '2.G':      'other_product_manufacture_and_use',
   '2.H':      'other_IPPU',
   'M.AG':     'Agriculture, sum of IPC3A and IPCMAGELV',
   '3.A':      'livestock',
   'M.AG.ELV': 'agriculture_excl_livestock',
   '4':        'waste',
   '5':        'other'
   }

gas_names = {
   'CH4':                      'CH4',
   'CO2':                      'CO2',
   'N2O':                      'N2O',
   'HFCS (SARGWP100)':         'HFCs (SAR)',
   'HFCS (AR4GWP100)':         'HFCs (AR4)',
   'PFCS (SARGWP100)':         'PFCs (SAR)',
   'PFCS (AR4GWP100)':         'PFCs (AR4)',
   'SF6':                      'SF6',
   'NF3':                      'NF3',
   'FGASES (SARGWP100)':       'F-gases (SAR)',
   'FGASES (AR4GWP100)':       'F-gases (AR4)',
   'KYOTOGHG (SARGWP100)':     'Kyoto GHGs (SAR)',
   'KYOTOGHGAR4 (AR4GWP100)':  'Kyoto GHGs (AR4)'
   }

#new_source_name = 'PRIMAP-' + raw_scenario + '_v2.3.1'

# Defining the varible name to be displayed on the plot

#gas_name = gas_names[raw_entity]
#data_source_to_display = 'PRIMAP-' + raw_scenario.lower()

#if raw_sector in ['M.0.EL', '1.B.3', '2.G', '2.H']:
#    variable_name_to_display = sector_codes.loc[raw_sector]['1'] + ' ' + gas_name + ' ' + sector_codes.loc[raw_sector]['3'] + ' ' + sector_codes.loc[raw_sector]['4']
#elif raw_sector in ['1', '1.B', '5', '1.B.2']:
#    variable_name_to_display = sector_codes.loc[raw_sector]['1'] + ' ' + gas_name + ' ' + sector_codes.loc[raw_sector]['3']
#elif raw_sector in ['2.D', '2.F', '4', '1.C', '1.A', '3.A', '1.B.1', '2', '2.A', '2.B', '2.C', '2.E', 'M.AG', 'M.AG.ELV']:
#    variable_name_to_display = gas_name + ' ' + sector_codes.loc[raw_sector]['2'] + ' ' + sector_codes.loc[raw_sector]['3']
#else:
#    print('The sector code introduced is incorrect. Please include a valid sector code.')


#file_variable_name = variable_name_to_display.replace(' ', '_')

#proc_data_file = new_source_name + '_' + file_variable_name + '.csv'


# Selecting the text to be displayed on the plots


## make_collective_progress_plots.ipynb

#input_file =
#sector_name = sector_names[raw_sector]

# Table with sector codes and names
#sector_codes = pd.read_csv('primap_sectors.csv')
#sector_codes.set_index('code', inplace=True)



## assess_peaking_emissions

#datafile_name
#var_name_for_plots
#peak_since = 
#nyears before end of data series by which peak should've occurred
#n_trend_years (emissions trend average)
#decrease_threshold
#region_of_interest (EU?)
