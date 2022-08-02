def peaking_barplot(summary_data, variable, max_year, save_plot=True):

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

    return plt

def plot_facet_grid_countries(series, variable, value, main_title='', plot_name='', save_plot=True, filepath=''):

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
    ranges = series.max(axis=1) - series.min(axis=1)
    check = (ranges.max() - ranges.min()) / ranges.min()
    if abs(check) < 8:
        yshare = True
    else:
        yshare = False

    # set up the df for plotting
    year_cols = series.columns
    dftomelt = series.reset_index()
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
    
    return plt


#Box 1

# Interrogate the data with different conditions (peak year, decreasing rate) to categorise countries  
nyears = 5
n_trend_years = 5
decrease_threshold = -1.5
# set up a dataframe to store analysis results
peaking_assessment = pd.DataFrame()
peaking_assessment['max year'] = year_max
  

# Identify countries that have 'peaked' before the time period chosen by the user above (n years ago)
peaking_assessment['max_reached'] = peaking_assessment['max year'].apply(
                                                                lambda x : (x < (end_year - nyears)))

# Identify countries with decreasing emissions trends
recent_trends, recent_trends_rolling, unit = utils.calculate_trends(dset_to_plot, num_years_trend=n_trend_years)
peaking_assessment['trend'] = recent_trends_rolling[str(end_year)]

peaking_assessment['decreasing'] = peaking_assessment['trend'].apply(lambda x: (x < 0))
peaking_assessment['strongly_decreasing'] = peaking_assessment['trend'].apply(lambda x: (x < decrease_threshold))
peaking_assessment['stable'] = peaking_assessment['trend'].apply(lambda x: (decrease_threshold < x < 0.5))


# Use multi-criteria to define different regimes
# First, the clear cases...
peaking_assessment['peaked']     = peaking_assessment['max_reached'] & \
                                   peaking_assessment['strongly_decreasing']

peaking_assessment['stabilised'] = (peaking_assessment['max_reached'] & \
                                    peaking_assessment['stable']) | \
                                   ((peaking_assessment['max_reached']==False) & \
                                    (peaking_assessment['strongly_decreasing']))

peaking_assessment['not_peaked'] = (peaking_assessment['stabilised']==False) & \
                                   (peaking_assessment['peaked']==False)


# Get stats of shares
share_peaked     = peaking_assessment['peaked'].value_counts(normalize=True) 
share_stabilised = peaking_assessment['stabilised'].value_counts(normalize=True)
share_increasing = peaking_assessment['not_peaked'].value_counts(normalize=True)
check_total = share_peaked[True] + share_stabilised[True] + share_increasing[True]


# and print to screen...
print('')
print('Share of countries peaked is {:.1f}%'.format(share_peaked[True]*100))
print('Share of countries stabilised is {:.1f}%'.format(share_stabilised[True]*100))
print('Share of countries not peaked or stabilised is {:.1f}%'.format(share_increasing[True]*100))
print('Total is: {:.1f}%'.format(check_total*100))
print('')











#Box 2

# Make a plot to summarise how many countries fall into each group.

# count the number of countries in each category
number_peaked     = peaking_assessment['peaked'].value_counts(normalize=False) 
number_stabilised = peaking_assessment['stabilised'].value_counts(normalize=False)
number_increasing = peaking_assessment['not_peaked'].value_counts(normalize=False)
total_countries = number_peaked[True] + number_stabilised[True] + number_increasing[True]

# define new dataframe
summary_data = pd.DataFrame({'category': ['peaked', 'stabilised', 'not peaked'],
                             'count': [number_peaked[True],  number_stabilised[True], number_increasing[True]]})

# make the plot
utils.peaking_barplot(summary_data, variable, str(end_year), save_plot=True)









#Box 3

# And make some plots to view and check the results. 


"""
These functions will plot facet grids of the emissions trends and absolute emissions
for all countries in each category. 

Note that this can take quite some time to run! 
"""


# 1. Peaking
# prep data for plotting 
peaked_country_trends = recent_trends_rolling[peaking_assessment['peaked']]
peaked_country_abs = dset_to_plot[peaking_assessment['peaked']] 

# make the plots
utils.plot_facet_grid_countries(peaked_country_trends, 'year', '% change', 
                          main_title='Countries with peaked emissions', 
                          plot_name=('peaked-' + variable), save_plot=True)

utils.plot_facet_grid_countries(peaked_country_abs, 'year', 'emissions', 
                          main_title='Absolute emissions for peaked countries', 
                          plot_name=('peaked-' + variable), save_plot=True)

    
# 2. Stabilised
stab_trends = recent_trends_rolling[peaking_assessment['stabilised']]
stab_abs    = dset_to_plot[peaking_assessment['stabilised']] 

utils.plot_facet_grid_countries(stab_trends, 'year', '% change', 
                         main_title='Trends in countries with stabilised emissions', 
                         plot_name=('stabilised-' + variable), save_plot=True)
utils.plot_facet_grid_countries(stab_abs, 'year', 'emissions', 
                         main_title='Emissions in countries with stabilised emissions', 
                         plot_name=('stabilised-' + variable), save_plot=True)

# 3. still increasing
not_peaked_trends = recent_trends_rolling[peaking_assessment['not_peaked']]
not_peaked_abs    = dset_to_plot[peaking_assessment['not_peaked']] 

utils.plot_facet_grid_countries(not_peaked_trends, 'year', '% change', 
                         main_title='Trends in countries with emissions that have not yet peaked',
                         plot_name=('not-peaked-' + variable),
                         save_plot=True)
utils.plot_facet_grid_countries(not_peaked_abs, 'year', 'emissions', 
                         main_title='Emissions in countries with emissions that have not yet peaked', 
                         plot_name=('not-peaked-' + variable),
                         save_plot=True)