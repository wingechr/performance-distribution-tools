input_folder = 'input-data/'
output_folder = 'output/'
proc_data_folder = 'proc-data/'

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

gas_names_variable = {
   'CH4':                      'CH4',
   'CO2':                      'CO2',
   'N2O':                      'N2O',
   'HFCS (SARGWP100)':         'HFC_SAR',
   'HFCS (AR4GWP100)':         'HFC_AR4',
   'PFCS (SARGWP100)':         'PFC_SAR',
   'PFCS (AR4GWP100)':         'PFC_AR4',
   'SF6':                      'SF6',
   'NF3':                      'NF3',
   'FGASES (SARGWP100)':       'F-gas_SAR',
   'FGASES (AR4GWP100)':       'F-gas_AR4',
   'KYOTOGHG (SARGWP100)':     'GHG_SAR',
   'KYOTOGHG (AR4GWP100)':     'GHG_AR4'
   }

sector_names_fname = {
   'M.0.EL':   'total_excl_LULUCF',
   '1':        'energy',
   '1.A':      'fuel_combustion',
   '1.B':      'fugitive',
   '1.B.1':    'solid_fuel',
   '1.B.2':    'oil_and_gas',
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
   'M.AG':     'agriculture',
   '3.A':      'livestock',
   'M.AG.ELV': 'agriculture_excl_livestock',
   '4':        'waste',
   '5':        'other'
   }