import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import roll as roll
from numpy import random as rand
import time
import multiprocessing as mp
import math
import time
import datetime
import winsound


# See the LaTeX PDF for my report on the Ising model accompanying this script.


# The user functions are at the end, followed by an example of using
# parallel_temps, the main function of this script. This function will plot
# estimates for any of 4 observables for each of a range of temperatures.
# Also included is plot_ising2D to animate a 2D lattice over time.





# This script contains
# - Fundamental functions and simple Ising animation function
# - Plotting observables' simulations against temperature

# This program does not contain
# - Specialised functions to return magnetisation or energy against
#       temperature to be plotted, which were made redundant by
#       parallel_temps
# - Checkerboard (parallel) updates since too much overhead and project
#       is to plot against temperature which parallelizes better. 
#    = Functions using checkerboard updates for speed such as plotting
#           and magnetization samples.
# These redundant but otherwise useful functions or unfinished starts to
# them are contained in the redundant_code.py script.

"""
_____________________________________________________________________________

      v v v v v      Fundamental and Helper Functions      v v v v v
_____________________________________________________________________________

"""


# Calcs change in configuration energy from changing flip at site (i,j).
def energy_delta(config, i, j, magnet_H=0, interact_J=1):
    n, m = len(config), len(config[0])
    return 2*config[i][j]*( magnet_H + interact_J *
                            ( config[(i+1)%n][j] + config[(i-1)%n][j]
                              + config[i][(j+1)%m ] + config[i][(j-1)%m] ) )


# Single flip at a random site for one Markov step.
# Input temp is *not* ratio of T_c
def update_single(config, temp=3, magnet_H=0, interact_J=1):
    i = rand.randint(len(config))
    j = rand.randint(len(config[0]))
    dE = energy_delta(config, i, j, magnet_H, interact_J)
    if dE < 0: # Saves expensive exp computation
        config_next = np.copy(config) # Is this copying actually necessary?
        config_next[i,j] *= -1
        return config_next
    elif np.exp( -dE / temp) > rand.random():
        config_next = np.copy(config)
        config_next[i,j] *= -1
        return config_next
    else:
        return config




# Returns specific magnetization (and mag susceptibility) from a single simulation run, not plotting.
# Input temp *is* ratio of T_c. Steps is not multiplied by number of sites.
def magnetization2D(dims=(10,10), magnet_H=0, interact_J=1, steps = 10000, temp=1.2,
                         susc=True, skips=0.5, sweeps=1): # Hot (random) start
    # Skips is ratio to discard to reach thermalisation/burn-in as required.
    # Sweeps are number discarded between independent samples; use autocorrtime
    temp_crit = 2 * interact_J / math.log(1+math.sqrt(2))
    temp = temp*temp_crit
    config = rand.choice([-1,1],size=dims)
    t = 0
    mag_terms, susc_terms, results = [], [], []
    if sweeps > 0.1*(1-skips)*steps:
        sweeps = 1
    
    while t<steps*skips:
        config = update_single(config, temp, magnet_H, interact_J)
        t += 1
    while t<steps:
        for i in range(sweeps-1):
            config = update_single(config, temp, magnet_H, interact_J)
        t += 1
        mag = sum(sum(config))
        mag_terms.append(mag) # Avoiding overflow if very many steps taken
        if susc:
            susc_terms.append(mag**2) # Avoiding overflow error

    mag_mean = np.mean(mag_terms)
    results.append( mag_mean / (dims[0]*dims[1]) )
    if susc: # Chi = ( <M^2> - <M>^2 ) / (temp * num. sites)
        mag_squ_mean = np.mean(susc_terms)
        susc_res = np.divide( mag_squ_mean - mag_mean**2 , temp*dims[0]*dims[1] )
        results.append(susc_res)
    return results




# Generates evenly-spaced temperatures at which we run comparason simulations
# Input temps are ratio of T_c
def temp_vecs(interact_J=1, temp_min=0.5, temp_max=1.5, num_temps=120):
    temp_crit = 2 * interact_J / math.log(1+math.sqrt(2))
    tempscaled = np.linspace(temp_min, temp_max, num=num_temps, endpoint=True)
    temp = tempscaled * temp_crit
    return temp, tempscaled


# Generates temp list concntrated close to pt.
# Input temps are ratios of T_c
def moremiddle(interact_J=1, temp_min=0.5, temp_max=1.5, num_temps=120, pt=1):
    temp_crit = 2 * interact_J / math.log(1+math.sqrt(2))
    window = min( (temp_max - temp_min)/10, (pt-temp_min)/5, (temp_max-pt)/5 )
    left = (pt-window/2) - temp_min
    right = temp_max - (pt+window) #l/(l+r) shares the non-middle points evenly
    space1 = np.linspace(temp_min, pt-window/2,
                         num=np.ceil(num_temps*0.8*left/(left+right)), endpoint=True)
    space2 = np.linspace(pt-window/2, pt+window/2,
                         num=np.ceil(num_temps*0.2), endpoint=False)
    space3 = np.linspace(pt+window/2, temp_max,
                         num=np.ceil(num_temps*0.8*right/(left+right)), endpoint=True)
    # Above, notice numtemps = numtemps*(p+(1-p)*(l+r)/(l+r)) ~= len(linspace)
    tempscaled = np.array(sorted(set([*space1,*space2,*space3])))
    temp = tempscaled*temp_crit
    return temp, tempscaled



#Calculates the Hamiltonian energy of a given configuration (for energy later).
def hamiltonian(config, magnet_H=0, interact_J=1):
        potential = np.sum(config *
                           ( roll(config,1,0) + roll(config,-1,0)
                             + roll(config,1,1) + roll(config,-1,1) ), dtype=np.float64)/4
        return -interact_J*potential - magnet_H*np.sum(config, dtype=np.float64)
# Credit for awesome rolling idea came from https://stanczakdominik.github.io/posts/parallelizable-numpy-implementation-of-2d-ising-model/
# We use the more efficient energy_deta functionto calculate change of Hamiltonian in probabilities.




# Returns a 'steps'-steps Monte Carlo simulation of energy (and heat capacity) at temp.
# Input temp *is* ratio of T_c. Steps is not multiplied by number of sites.
def energy2D(dims=(10,10), magnet_H=0, interact_J=1, steps=10000, temp=1.2, heat=True, skips=0.2, sweeps=10):
    # Skips is for thermalization before taking values for average
    temp_crit = 2 * interact_J / math.log(1+math.sqrt(2))
    temp = temp*temp_crit
    config = rand.choice([-1,1], size=dims)
    t = 0
    eng_terms, heat_terms, results = [], [], []
    if sweeps > (1-skips)*steps/100: # At least 100 indep. samples
        sweeps = 1
    
    while t<steps*skips: # Skips the first skips% to couple and give converged average
        config = update_single(config, temp, magnet_H, interact_J)
        t += 1
    while t<steps:
        for i in range(sweeps-1):
            config = update_single(config, temp, magnet_H, interact_J)
        t += 1
        eng = hamiltonian(config, magnet_H, interact_J)
        eng_terms.append(eng)
        if heat:
            heat_terms.append(eng**2) # Avoiding overflow error

    eng_mean_spec = np.mean(eng_terms) / (dims[0]*dims[1])
    results.append(eng_mean_spec)
    if heat: # C_V = ( <E^2> - <E>^2 ) / (num. sites * temp / beta)
        eng_squ_mean = np.mean(heat_terms) / (dims[0]*dims[1])
        heat_res = np.divide(eng_squ_mean -
                             dims[0]*dims[1]*eng_mean_spec**2, temp**2)
        results.append(heat_res)
    return results



# Helper function for parallelizing a looped function
def pooling(func,args):
    pool = mp.Pool(mp.cpu_count())
    res = pool.starmap(func,args) # Multiple arguments and results in order
    pool.close()
    pool.join()
    return res



# Helper function for subplots in parallel_temps
def plotting(axs, tempscaled, y, ylabel, sub_ind, pt=1,
             xlabel='Temperature kT/T_c', focus=True, label='No Label'):
    axs[sub_ind].scatter(tempscaled, y, s=1.2, label=label)
    if xlabel != None:
        axs[sub_ind].set_xlabel(xlabel)
    if ylabel != None:
        axs[sub_ind].set_ylabel(ylabel)
    if focus:
        axs[sub_ind].axvline(pt, color='r', label='Line at T_c', lw=0.3, ls='-')


# C_V = (2k/pi)(2J/kt)^2[-ln(1-T/T_c)+ln(kT_c/2J)-1-pi/4]/N # Assuming k=1?
# Returns heat capacity for a given temp t from analytical solution
def c(temp_crit=2.269, interact_J=1, t=1.2):
    res = (2/np.pi) * ( -np.log(1-t/temp_crit) + np.log(temp_crit/(2*interact_J)) -1 -np.pi/4 )
    res *= ((2*interact_J/temp_crit)**2)
    return res

# Quick functon returns magnetization for given temp from analytical solution
m = lambda interact_J, temp: (1 - ( np.sinh(2*interact_J/temp) )**(-4)) **(1/8)

# Returns magnetic susceptibility for a given temp t from analytical solution
def s(dims=(10,10), magnet_H=0, interact_J=1, t=1.2):
    res = (1/t)*dims[0]*dims[1]
    res /= np.cosh( magnet_H + 2 * len(dims) * interact_J * m(interact_J,t))**2
    res -= 2 * len(dims) * interact_J / t
    return res




"""
_____________________________________________________________________________

              v v v v v         User Functions       v v v v v         
_____________________________________________________________________________

"""






# Animation of the Ising lattice spins in 2D. Fun, generic and basic.
# Input temp is *not* ratio of T_c
def plot_ising2D(dims=(10,10), magnet_H=0, interact_J=1, steps=10000, temp=1.2, sweeps=10): # Hot start
    config = rand.choice([-1,1],size=dims)
    plt.ion()
    plt.show()
    image = plt.imshow(config, cmap='gray', vmin=-1, vmax=1, interpolation='none')
    t = 0
    if t % 50 == 0:
        image.set_data(config)
        plt.draw()
    for i in range(sweeps-1):
        config = update_single(config, temp, magnet_H, interact_J)
    plt.pause(.0001)
    t += 1



""" This script has build up to parallel_temps """
# Function to return/plot energy, heat capacity, magnetization, magnetic
# susceptibility versus temperature

# Only plots heat or susc if eng or mag activated respectively, since related.
# Steps will be multiplied by dims[0]*dims[1] and temps are ratios of T_c.
def parallel_temps(dims = (10,10), magnet_H = 0, interact_J = 1, steps = 125,
                   temp_min = 0.5, temp_max = 1.5, num_temps = 120,
                   eng = True, heat = True, mag = True, susc = True,
                   plotted = True, timed = True, skips=0.2, sweeps=10,
                   middled = False, pt=1, magabs=True, expon=False, alert=True):
    num_temps = mp.cpu_count()*np.floor(num_temps/mp.cpu_count()) # Make most use of parallelising to all cores
    if eng == False:
        heat = False
    if mag == False:
        susc = False
    if middled:
        temps, tempscaled = moremiddle(interact_J, temp_min, temp_max, num_temps, pt)
    else:
        temps, tempscaled = temp_vecs(interact_J, temp_min, temp_max, num_temps)
    temp_crit = 2 * interact_J / math.log(1+math.sqrt(2))
    lowtemps = [temp for temp in temps if temp < temp_crit] # For analytical solution plots
    eps = (temp_crit-max(lowtemps))/4
    for i in range(8): # Get temps close to T_c to show asymptotes
        lowtemps.append(temp_crit-eps/4**i)
    manylowtemps = np.hstack([np.linspace(lowtemps[i], lowtemps[i+1], 10) for i in range(len(lowtemps)-1)])
    manylowtemps = list(set(manylowtemps))
    manylowtemps.sort()
    steps = steps*dims[0]*dims[1] # Time for convergence at least grows with number of points
    results = []

    if plotted:
        fig, axs = plt.subplots(2, 2, squeeze=False)
        fig.suptitle(f'{dims[0]}x{dims[1]} Lattice; {int(steps*(1-skips)*sweeps)} Total Steps, {int(steps*(1-skips))} Indep. Steps, {100*skips}% Burn-in, Sweeps of {sweeps}')
        index_refs = [ (0,0), (0,1), (1,0), (1,1) ]
    ############
    if eng:
        if timed:
            start = time.time()
        tempseng, tempscaledeng = temp_vecs(interact_J, temp_min, temp_max, num_temps)
        args = [(dims, magnet_H, interact_J, steps, temp, heat, skips, sweeps) for temp in tempscaledeng]
        eng_results = pooling(energy2D, args)
        eng_results = list( map(list, zip(*eng_results)) )
        results.extend(np.array(eng_results))
        if plotted:
            plotting(axs, tempscaledeng, eng_results[0], 'Energy Density',
                     index_refs[0], pt, focus=False, label='Simulated')
            axs[index_refs[0]].legend()
            if heat:
                plotting(axs, tempscaled, eng_results[1], 'Specific Heat Capacity',
                         index_refs[1], pt, focus=False, label='Simulated')
                maxheat = max(eng_results[1])
                analyt_heat = [(temp,c(temp_crit, interact_J, temp)) for temp in manylowtemps if -0.1<c(temp_crit, interact_J, temp)<maxheat*1.1]
                analyt_heat = list(zip(*analyt_heat))
                axs[index_refs[1]].plot(np.divide(analyt_heat[0],temp_crit), analyt_heat[1], color='g',
                                                   label='Analytic', linewidth=0.75)
                axs[index_refs[1]].hlines(0,1,temp_max, color='g', linewidth=0.75)
                axs[index_refs[1]].vlines(1,0,maxheat*1.1, colors='g', linewidth=0.75)
                axs[index_refs[1]].legend()
        if timed:
            end = time.time()
            print('Energy time taken:', end - start)
    ############
    if mag:
        if timed:
            start = time.time()
        args = [(dims, magnet_H, interact_J, steps, temp, susc, skips, sweeps) for temp in tempscaled]        
        mag_results = pooling(magnetization2D, args)
        mag_results = np.array(list( map(list, zip(*mag_results)) ))
        if magabs:
            if susc: mag_results[0] = abs(mag_results[0])
            else: mag_results = abs(mag_results)
        results.extend(mag_results)
        if plotted:
            plotting(axs, tempscaled, mag_results[0], 'Specific Mag.', index_refs[eng + heat], pt,
                     focus=False, label='Simulated')
            maxmag = max(mag_results[0])
            analyt_mag = [(temp,m(interact_J,temp)) for temp in manylowtemps if m(interact_J,temp)<maxmag]
            analyt_mag = list(zip(*analyt_mag))
            axs[index_refs[eng + heat]].plot(np.divide(analyt_mag[0],temp_crit), analyt_mag[1], color='g',
                                             label='Analytic', linewidth=0.75)
            axs[index_refs[eng + heat]].hlines(0,1,temp_max, color='g', linewidth=0.75)
            axs[index_refs[eng + heat]].vlines(1,0,analyt_mag[1][-1], colors='g', linewidth=0.75)
            axs[index_refs[eng + heat]].legend()
            
            if susc:
                plotting(axs, tempscaled, mag_results[1], 'Specific Mag. Susc.',
                         index_refs[eng + heat + mag], pt, focus=False, label='Simulated')
                maxsusc = max(mag_results[1])
                analyt_susc = [(temp,s(dims, magnet_H, interact_J, temp)) for temp in manylowtemps if s(dims, magnet_H, interact_J, temp)<maxsusc]
                analyt_susc = list(zip(*analyt_susc))
                axs[index_refs[eng + heat + mag]].plot(np.divide(analyt_susc[0],temp_crit), analyt_susc[1],
                                                       color='g', label='Analytic', linewidth=0.75)
                axs[index_refs[eng + heat + mag]].hlines(0,1,temp_max, color='g', linewidth=0.75)
                axs[index_refs[eng + heat + mag]].vlines(1,0,maxsusc*1.1, colors='g', linewidth=0.75)
                axs[index_refs[eng + heat + mag]].legend()
    ############                  
        if timed:
            end = time.time()
            print('Magnetization time taken:', end - start)
    if alert:
        for i in range(2): # (Hopefully non-threatening) Alert when finished.
            winsound.Beep(500,500)
    if plotted:
        number = eng + eng*heat + mag + mag*susc
        for i in range(number,4): # Only plotting observables asked for
            fig.delaxes(axs[index_refs[i]])
        plt.tight_layout(h_pad=1)
        plt.show()
    if expon: # \propto|T-T_c| not T/T_c which is only useful for plots.
        return [np.abs(temps-temp_crit), *results]
    return [tempscaled,*results]





"""
Example of parallel_temps plotting all four observables
"""


# Example plotting of simulations for energy, heat capacity, magnetization and
# magnetic susceptibility against temperature over range 0.5*T_C to 1.5*T_C, on
# a 10x10 square lattice, with 10*10*125*(1-0.2)=10,000 independent-ish samples,
# for each of 120 temperatures, for each of the above observables.
# This may take a while! The pre-written example below took my laptop 13 minutes.
# Reduce dims or num_temps to speed up the simulation. Only converges at start
# if enough steps. The analytic solution in green applies in the thermodynamic
# limit as the number of sites dims[0]*dims[1] tends to infinity.

run_example = True
if run_example:
    if __name__ == '__main__': # par_temps will multiply steps by N.
        data = parallel_temps(dims = (10,10), magnet_H = 0, interact_J = 1,
                steps = 125, temp_min = 0.5, temp_max = 1.5,
                num_temps = 120, # Most small numbers divide 120 for number of cores
                eng = True, heat = True, mag = True, susc = True,
                plotted = True, timed = True, skips=0.2, sweeps=10,
                middled = True, pt=1, magabs=True, expon=True, alert=True)


