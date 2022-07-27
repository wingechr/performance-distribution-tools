# These are the default folders for the raw data (input_folder/), for the processed data (proc-data/), and for the plots (output/)

input_folder      = 'input-data/'
proc_data_folder  = 'proc-data/'
output_folder     = 'output/'

# These are the raw data files

# 1. Primap emissions data with extrapolation
primap_extrap_fname = 'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_20-Sep_2021.csv'
primap_source = 'PRIMAP-hist v2.3.1'

# 2. Energy data from bp

# Data must be in "panel" format
bp_world_energy_panel_fname = 'bp-stats-review-2022-consolidated-dataset-panel-format.csv'
bp_source = 'bp Statistical Review\n of World Energy 2022'

# 3. Sectoral emissions data from the IPCC's AR6

ipcc_ar6 = 'essd_ghg_data_gwp100.csv'
ipcc_source = 'Sixth Assessment Report - IPCC WG3'

# 4. Population and GDP data from the World Bank

# Population dataset
wb_population_fname = 'API_SP.POP.TOTL_DS2_en_csv_v2_4218816.csv'

# GDP dataset
wb_gdp_fname = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4150784.csv'


# These are the dictionaries that are used to generate the Primap file names

gas_names_fname = {
   'CH4':                      'CH4',
   'CO2':                      'CO2',
   'N2O':                      'N2O',
   'HFCS (SARGWP100)':         'HFCs_SAR',
   'HFCS (AR4GWP100)':         'HFCs_AR4',
   'PFCS (SARGWP100)':         'PFCs_SAR',
   'PFCS (AR4GWP100)':         'PFCs_AR4',
   'SF6':                      'SF6',
   'NF3':                      'NF3',
   'FGASES (SARGWP100)':       'F-gases_SAR',
   'FGASES (AR4GWP100)':       'F-gases_AR4',
   'KYOTOGHG (SARGWP100)':     'Kyoto_GHGs_SAR',
   'KYOTOGHG (AR4GWP100)':     'Kyoto_GHGs_AR4'
   }

sector_names_fname = {
   'M.0.EL':      'total_excl_LULUCF',
   '1':           'energy',
   '1.A':         'fuel_combustion',
   '1.B':         'fugitive',
   '1.B.1':       'solid_fuel',
   '1.B.2':       'oil_and_gas',
   '1.B.3':       'other_from_energy_prod',
   '1.C':         'CO2_transport_and_storage',
   '2':           'IPPU',
   '2.A':         'mineral_industry',
   '2.B':         'chemical_industry',
   '2.C':         'metal_industry',
   '2.D':         'non-energy_products_from_fuels_and_solvents',
   '2.E':         'electronics_industry',
   '2.F':         'product_use_substitutes_for_ozone_depl_subs',
   '2.G':         'other_product_manufacture_and_use',
   '2.H':         'other_IPPU',
   'M.AG':        'agriculture',
   '3.A':         'livestock',
   'M.AG.ELV':    'agriculture_excl_livestock',
   '4':           'waste',
   '5':           'other'
      }


# This is the dictionary that is used to generate the Primap variable names

SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
SUP = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')

gas_names_variable = {
   'CH4':                      'CH4'.translate(SUB),
   'CO2':                      'CO2'.translate(SUB),
   'N2O':                      'N2O'.translate(SUB),
   'HFCS (SARGWP100)':         'HFC',
   'HFCS (AR4GWP100)':         'HFC',
   'PFCS (SARGWP100)':         'PFC',
   'PFCS (AR4GWP100)':         'PFC',
   'SF6':                      'SF6'.translate(SUB),
   'NF3':                      'NF3'.translate(SUB),
   'FGASES (SARGWP100)':       'F-gas',
   'FGASES (AR4GWP100)':       'F-gas',
   'KYOTOGHG (SARGWP100)':     'GHG',
   'KYOTOGHG (AR4GWP100)':     'GHG'
      }