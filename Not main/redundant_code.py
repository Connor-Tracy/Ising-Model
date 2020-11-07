# This program contains otherwise useful functions which
# have been made redundant by more general functions later.
# Rather than delete my work, I would rather archive them here.


# - A function to return sequence of magnetizations of a MCMC.
#
# - Specialised functions to return magnetisation
#       or energy against temperature to be plotted,
#       which were made redundant by parallel_temps
#
# - Checkerboard (parallel) updates since too much
#       overhead and project wants plots against
#       temperature which needs parallels elsewhere.
#
# - Functions depending on checkerboard updates
#   = Plotting using checkerboard for speed gains.
#   = Generating magnetization samples using checkerboard



# Some of these functions were abandoned before being fully completed.



"""
# - Specialised functions to return magnetisation
#       or energy against temperature to be plotted,
#       which were made redundant by parallel_temps
"""


# Not parallelized so not called on later
# Made redundant by parallel_temps. Serial so likely slower anyway
def mag_vs_temp(dims=(10,10), magnet_H=0,
                interact_J=1, steps = 50000,
                temp_min=0.3, temp_max=2, num_temps=200,
                middled = False): #Absolute magnetization

    if middled:
        temps, tempscaled = middlemore(interact_J, temp_min, temp_max, num_temps)
    else:
        temps, tempscaled = temp_vecs(interact_J, temp_min, temp_max, num_temps)
    mag_results = []
    for temp in temps:
        mag_results.append(abs(magnetization2D(
            dims, magnet_H, interact_J, steps, temp, susc=False)))
    return [tempscaled, mag_results]


# Serially generates samples of energy over a randge of temperatures.
# Made redundant by parallel_temps. Serial so likely slower anyway
def energy_vs_temp(dims=(10,10), magnet_H=0,
                   interact_J=1, steps = 20000,
                   temp_min=0.3, temp_max=2, num_temps=50,
                   middled = True):
    
    if middled:
        temps, tempscaled = middlemore(interact_J, temp_min, temp_max, num_temps)
    else:
        temps, tempscaled = temp_vecs(interact_J, temp_min, temp_max, num_temps)
    eng_results = []
    for temp in temps: #This can be parallelised using multiprocessing.pool
        eng_results.append( energy2D(dims, magnet_H,
                                     interact_J, steps, temp, heat=False) )
    return [tempscaled,eng_results]







"""
# - Checkerboard (parallel) updates since too much
#       overhead and project wants plots against
#       temperature which needs parallels elsewhere.
"""




# Does (#Cores) update steps only on sites of one colour, typewriter ordering
# Not needed as decided against checkerboard
def update(config, temp, interact_J, magnet_H, index):
    changes = []
    i, j = index[0], index[1]
    cost = energy_delta(config, i, j, interact_J, magnet_H)
    if cost < 0: # Saves expensive exp computation
        return (i,j)
    elif np.exp( np.divide(-cost,temp) ) > rand.random():
        return(i,j)


# Fast: Typewriter and checkerboard. Forces even side lengths due to PBC.
# Put in error exception if odd sides.
# Not needed as decided against checkerboard
def update_parallel(config, temp, interact_J, magnet_H, sweeps, count): 
    black_inds = [(i,j) for i in range(len(config))
                  for j in range(len(config[0])) if (i+j)%2==0]
    white_inds = [(i,j) for i in range(len(config))
                  for j in range(len(config[0])) if (i+j)%2==1]
    cores = mp.cpu_count()

    while count < sweeps:
        count += 1
        indices = []
        
        if count%2 == 0:
            subspace = black_inds
            for x in range(count*cores, (count+1)*cores):
                indices.append(subspace[x%len(subspace)])
            args = [(config, temp, interact_J, magnet_H, index)
                    for index in indices]
            pool = mp.Pool(cores)
            changes = pool.starmap(update, args)
            changes = list(filter(None, changes)) 
            pool.close()
            pool.join()

        else:
            subspace = white_inds
            for x in range(count*cores, (count+1)*cores):
                indices.append(subspace[x%len(subspace)])
            args = [(config, temp, interact_J, magnet_H, index)
                    for index in indices]
            pool = mp.Pool(cores)
            changes = pool.starmap(update, args)
            changes = list(filter(None, changes))
            pool.close()
            pool.join()
        for i,j in changes:
                config[i,j] *= -1
    return config






"""
# - Functions depending on checkerboard updates
#   = Plotting using checkerboard for speed gains.
#   = Generating magnetization samples using checkerboard
"""



# Not quite finalised wrt sweeps.
# Not needed as decided against checkerboard
def plot_ising2D_parallel(dims=(10,10), temp=3, interact_J=1,
                          magnet_H=0, sweeps=5): # Hot start
    config = rand.choice([-1,1],size=dims)
    t = 0
    while t<20:
        plt.ion()
        plt.show()
        image = plt.imshow(config, cmap='gray', vmin=-1, vmax=1, interpolation='none')
        if t%5 ==0:
            image.set_data(config)
            plt.draw()
        config = update_parallel(config, temp, interact_J, magnet_H, sweeps, t)
        plt.pause(0.0001)
        t += 1







# Not quite finalised.
# Not needed as decided against checkerboard
def magnetization2D_checkerboard(dims=(10,10), magnet_H=0,
                         interact_J=1, steps = 50000, sweeps = 5, temp=2,
                         susc=True, skips=0.7): # Hot (random) start

    config = rand.choice([-1,1],size=dims)
    t = 0
    magsum = 0
    susc_sum = 0
    used_steps_N = (1-skips)*steps*dims[0]*dims[1]
    results = []
    cores = mp.cpu_count()
    
    while t<steps*skips:
        config = update_parallel(config, temp, interact_J, magnet_H, sweeps, count = t)
        t += cores*sweeps
    while t<steps:
        config = update_parallel(config, temp, interact_J, magnet_H, sweeps, count = t)
        t += cores*sweeps
        mag = sum(sum(config))
        magsum += mag
        if susc:
            susc_sum += mag**2

    mag_exp = np.divide(magsum, (1-skips)*steps)
    results.append(np.divide(mag_exp,dims[0]*dims[1]))
    if susc: # Chi = ( <M^2> - <M>^2 ) / (num. sites * temp)
        mag_squ_exp = np.divide(susc_sum, (1-skips)*steps)
        susc_res = np.divide(mag_squ_exp - mag_exp**2,temp*dims[0]*dims[1])
        results.append(susc_res)
    return results




# Returns sequence of magnetizations for steps/sweeps no. of 'independent' steps. Basic.
def mag_simulation(dims=(20,20), steps=10**4, absolute=True,
                   temp=0.3, interact_J=1, magnet_H=0, perc=False, sweeps=1):
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
                print(100*t/steps, '% done')
    return np.divide(mags, dims[0]*dims[1])












"""
Attempt to create an autocorrelation time function, realising it was
very slow for largish but still necessary numbers so testing how long
it took and trying to fit this to a power law unsucessfully.

Instead I used numpy.correlate and summed the resulting vector.
The same testing showed that instead of 7000 vector length before
computation time reached 20 seconds, the build-in funtion could do 180k(!)
"""




# Plot how long a function takes wrt steps.
#UNTESTED
def expensive(func, args, max_steps, steps_index):
    return func_vs_param(time_func, args, steps_index,
                         1, max_steps, max_steps,
                         xlabel='Number of Steps', ylabel='Time for Execution',
                         plotted=True)


# Plot time of energy and magnetization sims (forgot expensive existed)
def plots(xs, y11, y12, y21, y22):
	fig, axs = plt.subplots(2,2)
	fig.suptitle('Time of execution (s) vs 100s of steps')
	axs[0][0].plot(xs, y11, label = 'four cores energy', linestyle='-', marker='o')
	axs[0][0].plot(xs, y12, label = 'one core energy', linestyle='-', marker='o')
	axs[0][0].legend()
	ratiosenergy = [y12[i]/y11[i] for i in range(len(xs))]
	axs[0][1].plot(xs, ratiosenergy, label = 'Energy time ratio 1 to 4 cores', linestyle='-', marker='o')
	axs[0][1].legend()
	axs[1][0].plot(xs, y21, label = 'four cores mag.', linestyle='-', marker='o')
	axs[1][0].plot(xs, y22, label = 'one core mag.', linestyle='-', marker='o')
	axs[1][0].legend()
	ratiosmags = [y22[i]/y21[i] for i in range(len(xs))]
	axs[1][1].plot(xs, ratiosmags, label = 'Mag time ratio 1 to 4 cores', linestyle='-', marker='o')
	axs[1][1].legend()
	
	plt.show()

plots(steppies, fourcoresenergy, onecoreenergy, fourcoremags, onecoremags)






# Analysing the autocorrelation efficiency
# Not sure why he shell doesn't finish, not even using parallel here.
if __name__ == '__main__':
    print('Started', datetime.datetime.now())
    unit_size = 10**2
    min_size = 1
    max_size = 750
    plot_points = 10
    data = [random.uniform(-2,2) for i in range( max_size*unit_size )]
    args = []

    xvals = []
    times = []
    start = time.time()
    for i in range( min_size*unit_size, max_size*unit_size ):
        if i%(math.floor( unit_size * (max_size-min_size) / plot_points )) == 0:
            xvals.append(i)
            start = time.time()
            a = autocorr_time(data[:i])
            end = time.time()
            print( len(xvals) * 100 / plot_points, '% Done')
            times.append(end-start)
    logx = np.log10(xvals)
    logy = np.log10(times)
    text = ("The log-log trend is clearly linear\n"
            "beyond initial overhead time.\n"
            "Hence, this is a power law.\n"
            "This eyeballed trendline is\n"
            "log(y) = -7.3 + 2.15log(x).\n"
            "So 200k vector length would take\n"
            "200M seconds, which is 6.333 years!\n"
            "My line is wrong and it is not exponential\n"
            "since it is possible to do 200k.")
    plt.scatter(logx, logy, linestyle="None", s=2)
    plt.xlabel('Log_10 of Input Vector Length')
    plt.ylabel('Log_10 of Execution time of Autocorrelation Time Function')
    #plt.plot(logx, [-7.3 + 2.15*x for x in logx])
    #plt.annotate(text, xy=(0.05, 0.95),  xycoords='axes fraction',
     #       horizontalalignment='left', verticalalignment='top',
      #      wrap=True)
    print('Ended', datetime.datetime.now())
    plt.show()

