import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rand
import time # Will time it verus dimensions/steps
import multiprocessing as mp
import time
# Won't use emcee module as that is best for periodic data since using fft.


# Autocorrelation time graveyard. Cannot get it to work as expected so
# here is what I have and some rough tests.

# A self-contained introduction to the Ising model and an explanation of
# my autocorrelation time issues is at the very end.



"""
_________________________________________________________________________
Requisite functions
_________________________________________________________________________

"""

# Calcs change in configuration energy from changing flip at site (i,j)
def energy_delta(config, i, j, interact_J, magnet_H):
    n, m = len(config), len(config[0])
    return 2*config[i][j]*( magnet_H + interact_J *
                            ( config[(i+1)%n][j] + config[(i-1)%n][j]
                              + config[i][(j+1)%m ] + config[i][(j-1)%m] ) )

# Single flip at a random site for one Markov step
def update_single(config, temp, interact_J, magnet_H):
    i = rand.randint(len(config))
    j = rand.randint(len(config[0]))
    dE = energy_delta(config,i,j,interact_J, magnet_H)
    if dE < 0: # Saves expensive exp computation - always change if lower energy
        config[i,j] *= -1
        return config
    elif rand.random() < np.exp( -dE / temp):
        config[i,j] *= -1
        return config
    else:
        return config

# Simulates the model (once) and returns the sequence of magnetizations
def mag_simulation(dims=(20,20), steps=10**4, absolute=True,
                   temp=3, interact_J=1, magnet_H=0, perc=False, sweeps=1):
    config = rand.choice([-1,1],size=dims)
    t = 0
    mags = []
    while t < steps:
        for i in range(sweeps):
            config = update_single(config, temp, interact_J, magnet_H)
        if absolute:
            mags.append(abs(sum(sum(config))))
        else:
            mags.append(sum(sum(config)))
        t += 1
        if perc:
            if t % (steps/100) == 0:
                print(round(100*t/steps,2), '% done')
    print('Done sim of', steps, 'steps') # Temp for testing
    return np.divide(mags, dims[0]*dims[1])



"""
________________________________________________________________________

Attempts at autocorrelation time functions and rough-and-ready tests.
________________________________________________________________________

"""


"""
The motivation to compare autocorr to timeseries length was hoping to
have a plot matching my understanding of the theory, so I can determine
the correct window at which to cut off the sum in the naive approach.
"""



# Naive sum approach. Most online sources give more or less this solution
def autocorr_time(x, window):
    x -= np.mean(x)
    vals = (np.correlate(x, x, mode='full')[-len(x):])[:window] # First window cross corrs from 0^th
    vals /= vals[0] * np.arange(window,0,-1) # Because vals[0] is var*n(?)
    #print(min(vals),max(vals))
    #plt.scatter(range(len(vals)), len(x)*vals, s=0.2) # len(x) is test
    #plt.show()
    return 0.5 + np.sum(vals, dtype=np.float64)


test00 = False
if test00:
    for i in range(10):
        temp = 3
        y = mag_simulation(dims=(50,50), steps=5000, absolute=True,
                               temp=temp, perc=False, sweeps=1)

        auts = autocorr_time(y,len(y))
        print(auts)
    #plt.scatter(100, auts, label='Temp is {}'.format(temp), lw=0.5)
    #plt.legend()
    #plt.show()




test1 = False
if test1:
    temp = 3
    lens = list(np.linspace(50,3050,600))
    y = [mag_simulation(dims=(50,50), steps=i, absolute=True,
                       temp=temp, perc=False, sweeps=1)
         for i in lens]
    # Closer to 2.27 and larger dims, the higher autocorrtime in theory.
    y = [autocorr_time(j,len(j)) for j in y]
    plt.scatter(lens,y, label='Temp is {}'.format(temp), lw=0.5)
    plt.legend()
    plt.show()
# It should plato (see below) but it doesn't, it's still
# linear even the log-log plot. Does it not automatically
# choose an approriate window in which to sum up to?

# Maybe I just don't understand what it's supposed to do?
# Even if this is correct, it is very different to the behaviour of
# the naive function which is much much worse behaved.



######################


def actime(x):
    x -= np.mean(x)
    var = x.var()
    length = len(x)

    acf_array = []
    for t in range(length):
        temp = np.mean(x[:(length-t)]*x[t:])/var
        if t<2:
            print('start',temp)
        if t>length-2:
            print('end',temp)
        acf_array.append(temp)
    #print(min(acf_array),max(acf_array))
    #plt.scatter(range(len(acf_array)), acf_array, s=0.2) # len(x) is test
    #plt.show()
    return 0.5 + np.sum(acf_array, dtype=np.float64)


# Autocorrelation time estimates for each method on same data
test11 = False
if test11:
    st = time.time()
    y = []
    for i in range(4):
        temp = 3
        y.append(mag_simulation(dims=(30,30), steps=10**5, absolute=True,
                               temp=temp, perc=False, sweeps=250))

    auts = [autocorr_time(j,len(j)) for j in y]
    auts2 = [actime(i) for i in y]
    print([(auts[k],auts2[k]) for k in range(4)])
    st2 = time.time()
    print('time taken:', st2-st)



######################



"""
Here are some things I was using in a rough-and-ready way to try to
understand what was going on.
"""

# Quick plot of autocorrelation time against length of simulation
# Should become linear after discarding some initial section.
# It looks like it is becoming linear, but then when you scale it more,
# it instead is way-off. Same for the log-log plot of it.
# Although this was 2 weeks ago so I can't quite remember how much
# of the above is true.

test22 = False
if test22:
    aut_times = []
    num = []
    for i in range(1,20):
        num.append(5*10**6*i/20)
        sim = mag_simulation(dims=(20,20), steps=num[i-1], absolute=True perc=False)
        aut_times.append(autocorr_time(sim))
    plt.scatter(num,aut_times)
    plt.show()



######################

# Function to generate a sample autocorrtime for one run of magnetisations.
def autocorr_sim(steps, temp=0.3, dims=(20,20)):                                                
    sim = mag_simulation(dims, steps, absolute=True, temp=temp, perc=False)
    return autocorr_time(sim)


# Plotting autocorrelationtime against vector length, again.
# Basically the same as the short bit above in green except using
# multiprocessing instead of a for loop.
testing2 = False
if testing2:
    if __name__ == '__main__':
        print('Started', datetime.datetime.now())
        points = 20
        maxsize = 5000
        runs = 25
        
        args = [ [maxsize*i/points, 2]
                 for i in range(1,points+1) ]
        repeats = []
        for i in range(runs):      
            repeats.append(pooling(autocorr_sim, args))
        aut_times = np.mean(repeats, axis=0)

        plt.plot(list( map(list, zip(*args)) )[0], aut_times, linewidth=0.5)

        #plt.plot(xvals, [61 + 5.01*x/11 for x in xvals])
        #^^my attempt at fitting a linear line to it once,
        #which was way off when scaled
        
        plt.xlabel('Length of simulation')
        plt.ylabel('Autocorrelation Time of Simulation Magnetization')
        avg = np.mean(aut_times[len(aut_times)//2:])
        plt.axhline(avg, label=f'Last 50% mean {avg}')
        plt.legend()
        print('Ended', datetime.datetime.now())
        plt.show()

######################


# Less about whether it works and more just how scalable it is.

# Plots *execution time* of autocorrtime function against
# vector length to judge how much can be realistically fed into it.
testing3 = False
if testing3:
    min_size = 100
    max_size = 10**5
    points = 500
    
    def test1func(series, i):
                start = time.time()
                a = autocorr_time(series[:i])
                end = time.time()
                #print( len(xvals) * 100 / plot_points, '% Done')
                return end-start
            
    if __name__ == '__main__':
        print('Started', datetime.datetime.now())
        data = mag_simulation(steps = max_size, perc=False)
        pts = [ i for i in range(min_size,max_size+1)
                 if i%(math.floor((max_size-min_size)/points)) == 0 ]
        args = [ (data, i) for i in pts ]
        results = pooling(test1func, args)
        plt.scatter(pts, results, linestyle="None", s=2)
        plt.xlabel('Input Vector Length')
        plt.ylabel('Execution time of Autocorrelation Time Function')
        print('Ended', datetime.datetime.now())
        plt.show()







#########
# An old attempt I found hidden away. Possibly a close variant of something found online.

def autocorr_func(data, t):
    crosscorr = np.correlate(data[:-t], data[t:])
    crosscorr = np.divide(crosscorr, len(data)-t)
    data_var = np.var(data, dtype=np.float64)
    return np.divide(crosscorr - np.mean(data)**2,data_var)

def autocorr_time(data):
    return 0.5 + np.sum([autocorr_func(data,t) for t in range(1,len(data))], dtype=np.float64)











"""
________________________________________________________________________

Explanation of theory
________________________________________________________________________

"""




"""
 mag_simulation uses energy_delta and update_single to generate
 a timeseries corresponding to the value of the magnetization of
 the configurations at each step in one Markov Chain Monte Carlo
 simulation run. The theory behind what that means is below.

 The square-lattice Ising model consists of a regular (square) lattice
 with a value of +/-1 assigned to each vertex. A configuration is a list
 of lists corresponding to an nxm rectangular lattice of these +/-1 values.

 The magnetization is just the sum of all the values of a configuration,
 and dividing by the number of sites gives the specific/per spin mag.

 The energy of a confguration is
 magnet_H + interact_J * (the product of all adjacent sites).
 So the change of energy from one flip is only dependent on its neighbours.
 
 Given a configuration, the nexte configuration is borne from the previous
 one by considering a site at random and flipping its value/sign with a
 certain probability which favours configurations with a lower total energy.
 Repeated updates result in a sequence (Markov chain) of configurations,
 and the magnetizations of each step forms a new MC. It is this MC of
 magnetizations we are interested in.
 
 Since this update step leaves each next configuraiion very similar
 to the previous configuration (differs by zero or one sites), the
 magnetizations have high autocorrelation, that is there is a strong
 correlation between configurations a small number of time steps apart.

 The autocorrelation function (acf) is the correlation of the timeseries
 with itself shifted by t places.
 The integrated autocorrelation time is 0.5 + sum_t(acf(t)) and is
 the the number of iterations, on average, before the next configuration
 is independent. Hence we skip 'sweeps'=acorrtime configurations between
 each recorded step to get an indpendent timeseries. Moreover, we can
 'couple' from the initial random ('hot') configuration by discarding the
 first 5*acorrtime steps or so.

 The acf can be negative since it is just a correlation, which is naturally
 between -1 and +1. But this leads to some bad acorrtimes.

 Moreover, the autocorrelation function does not asymptotically tend to
 zero over time. Instead it decreases to noise; fluctuating about some
 non-negative value. The solution to this is to only take the sum over
 a finite 'window', such that the acf has decayed as much as it can,
 since sufficiently far-away configurations are close to independent.
 Unfortunately the formula for determining the best window to take depends
 on acorrtime itself.

 To get unbiased estimators of acorrtime there are different formulae
 out there but the 0.5 + sum is the basic naive way which should be fine.

 I tried plotting acorrtime function against vector length taking
 the first x entries of the same vector each time but it did not grow
 linearly at any scale (which was expected from the noise) so that
 suggests I wasn't able to write a correct function for acorrtime.
 However I have tried many things found online and nothing has worked
 as expected, so I worry this indicates some flaw in my samples somehow.

 I have recenty stumbled upon one comparison of different methods of
 estimating acorrtime which could be a good way to get a quick survey.
 https://stats.stackexchange.com/questions/353112/competing-methods
 -for-estimating-autocorrelation-time (NB broken into two lines).
             
""" 








