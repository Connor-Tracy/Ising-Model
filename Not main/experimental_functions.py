"""
Experimental functions: without prerequisite functions.
"""

# Note the key function I am yet to get right is the autocorrelation time.
# The attempts at and tests for this are in a separate self-contained script
# called autocorr_not_working.py.


"""
_________________________________________________________________________

"""

from sklearn.linear_model import LinearRegression as linreg
# Trying to estimate critical exponents via linear regression.
def crit_exp_est(data): # Assuming already discarded first 5*autocorrtime
    #Just one dimensional so each data point is a list of one element
    xdata = list(map(lambda x:[x], data[0]))
    ydata = data[1]
    plt.loglog(data[0], data[1]) # Checking data grows by a power law
    plt.show()
    reg = linreg().fit(np.log10(xdata), np.log10(ydata))
    return reg.coef_

# vv For output of parallel_temps vv
test000 = False
if test000:
    #print(data)
    print(crit_exp_est([data[0],data[2]]))
    ##print(crit_exp_est([data[0],data[2]]))


"""
_________________________________________________________________________

"""


# Change to be able to accept and plot multiple observables
## Plots a function against a changing parameter, the parameter in index^th place in its args.
def func_vs_param(func, defaults, index, pmin, pmax, num_pts,
                  xlabel='Some Parameter', ylabel='Some Observable',
                  pt=1, middled=True, plotted=True):

    if middled:
        pts = moremiddle(1, pmin, pmax, num_pts, pt)[1]
    else:
        pts = np.linspace(pmin, pmax, num_pts)
        
    defaults = list(defaults)
    args = []
    for point in pts:
        b = defaults[:]
        b.insert(index, point)
        args.append(b)
        
    results = pooling(func, args)
    results = list( map(list, zip(*results)) )
    if plotted:
        fig, axs = plt.subplots(1, 1, squeeze=False)
        plotting(axs, pts, results[0], ylabel, (0,0), pt, xlabel)
        #plt.tight_layout(h_pad=1)
        plt.show()
    return pts, results
# Assumes the parameter to vary is ommitted from defaults.
# index is such that index=1 maps (1,2,3) |--> (1,para,2,3).


"""
_________________________________________________________________________

"""

# Plotting the analytic solutions alongside simulation results doesn't work well.

# Need to do hline for the zero starting from temp 1 in mag plot





"""
Notes for future self___________________________________________________________________
"""


# Test for sanity check of parallel_temps output; only use if num_temps small (=<3)
test = False
if test:
    if __name__ == '__main__':
        x = parallel_temps(steps = 100, temp_min=0.5, num_temps = 2, middled=False,
                            skips=0.1, eng=True, susc=True, magabs=True)
        for i in x:
            print(i)

"""
Remember when I do error bars that the skips parameter will
reduce the number of steps in the mean for each run's value.
"""































































































# There are no Easter eggs here
