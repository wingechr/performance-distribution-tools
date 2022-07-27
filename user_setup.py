# %% [markdown]
# ### 1. Please update the names of any of the datasets that will be used for plotting:

# %% [markdown]
# #### 1a. Emissions data from Primap

# %%
# Primap emissions data without extrapolation
primap_extrap_fname = 'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_20-Sep_2021.csv'

# Primap emissions data with extrapolation
primap_no_extrap_fname = 'Guetschow-et-al-2021-PRIMAP-hist_v2.3.1_no_extrap_20-Sep_2021.csv'

primap_source = 'PRIMAP-hist v2.3.1'

# %% [markdown]
# #### 1b. Energy data from bp

# %%
# Data must be in "panel" format
bp_world_energy_panel_fname = 'bp-stats-review-2022-consolidated-dataset-panel-format.csv'

bp_source = 'bp Statistical Review\n of World Energy 2022'

# %% [markdown]
# #### 1c. Population and GDP data from the World Bank

# %%
# Population dataset
wb_population_fname = 'API_SP.POP.TOTL_DS2_en_csv_v2_4218816.csv'

# GDP dataset
wb_gdp_fname = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4150784.csv'

# %% [markdown]
# ### 2. Please select the data that you want to plot:

# %%
# Select country groups from this list. You can input the group itself or countries within the groups using ISO 3166 alpha-3 country codes.
# For example:
    # countries = UNFCCC (includes all countries in the UNFCCC)
    # countries = ['EGY', 'GRC', 'EUU', 'COL'] (includes Egypt, Greece, European Union, and Colombia)
from countrygroups import UNFCCC, EUROPEAN_UNION, ANNEX_ONE, NON_ANNEX_ONE

countries = UNFCCC

# Select the years that you want to plot:
#years_of_interest = ['1990', '2005', '2016']
year_of_interest = '2016'

# In case of plotting difference from a baseline year, specify the baseline year here:
baseline_year = '1990'

# Select the data that you want to plot.
#   Options are:    'emissions' (uses Primap-hist data)
#                   'energy' (uses bp data)
data = 'emissions'

# %%
# If plotting emissions data, please select whether you want to include extrapolated data.
extrapol = True

# %%
# If plotting Primap emissions data, please select the type of gas that you want to plot.
# These are the options:
#   'CH4'
#   'CO2'
#   'N2O'
#   'HFCS (SARGWP100)' (as defined in AR2)
#   'HFCS (AR4GWP100)' (as defined in AR4)
#   'PFCS (SARGWP100)' (as defined in AR2)
#   'PFCS (AR4GWP100)' (as defined in AR4)
#   'SF6'
#   'NF3'
#   'FGASES (SARGWP100)' (as defined in AR2)
#   'FGASES (AR4GWP100)' (as defined in AR4)
#   'KYOTOGHG (SARGWP100)' (as defined in AR2)
#   'KYOTOGHG (AR4GWP100)' (as defined in AR4)
   
primap_gas = 'KYOTOGHG (AR4GWP100)'

# %%
# If plotting Primap emissions data, please select the sector that you want to plot.
# These are the available sectors:
# 'M.0.EL'      (Total excluding LULUCF)
# '1'           (Energy)
# '1.A'         (Fuel combustion)
# '1.B'         (Fugitive emissions from energy production)
# '1.B.1'       (Fugitive emissions from solid fuels)
# '1.B.2'       (Fugitive emissions from oil and gas)
# '1.B.3'       (Other emissions from energy production)
# '1.C'         (NO DATA - CO2 transport and storage)
# '2'           (IPPU)
# '2.A'         (Mineral industry)
# '2.B'         (Chemical industry)
# '2.C'         (Metal industry)
# '2.D'         (Non-energy products from fuels and solvents)
# '2.E'         (Electronics industry)
# '2.F'         (NO DATA - Emissions from the use of substitutes of ozone-depleting substances)
# '2.G'         (Other emissions from product manufacture and use)
# '2.H'         (Other emissions from IPPU)
# 'M.AG'        (Agriculture)
# '3.A'         (Livestock)
# 'M.AG.ELV'    (Agriculture excluding livestock)
# '4'           (Waste)
# '5'           (Other)
   
primap_sector = 'M.0.EL'

# %%
# If plotting Primap emissions data, plese select the data scenario that you want to plot.
# These are the options:
# 'HISTCR' (country-reported data)
# 'HISTTP' (third-party data)

primap_scenario = 'HISTCR'

# %%
# If plotting energy data, please select the variable that you want to plot
# Options are:      1: Share of renewables in electricity
#                   2: Share of fossil fuels in primary energy consumed
#                   3: Energy use
energy_variable = 3

# %%
# Select the data type that you want to plot:
# Options are:      'absolute'
#                   'per capita'
#                   'per USD'
# WARNING:  Per capita and per USD measures do not apply to the following variables: share of renewables in electricity and share of fossil
#           fuels in TPES.
data_type = 'absolute'

# %%
# IN PROGRESS
# Select the type of plot that you want to create:
# Options are:      1: Distribution of variable in specified year.
#                   2: Change of variable since specified year.
#                   3: Rolling average trend
#                   4: Year of peaking

plot_type = 4


