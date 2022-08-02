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

def prepare_for_plotting(final_dset, plot_type):
    if not verify_data_format(final_dset):
        raise ValueError('WARNING: The data is not correctly formatted! Please check before continuing!')
    # Extract the key information
    else:
        variable = final_dset['variable'].unique()[0]
        unit = final_dset['unit'].unique()[0]

        # Tidy up for next steps
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