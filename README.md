# Ising-Model
2020 voluntary Summer project simulating the Ising model using Markov Chain Monte Carlo and parallel programming.
This project was not funded and I completed it for my own satisfaction/personal development (and it was rewarding!).

Kindly supervised by Dr D Proment I created a Python script to estimate the expectation value of observables on the
Ising model on a regular rectangular using Makov Chain Monte Carlo simulations. This is complemented by a LaTeX report
on the theory of the Ising model in two dimensions, with proofs in the appendix.

The main function parallel_temps will estimate the expectation value of your choice of 4 observables
over a range of temperatures and then return/plot these estimates. Comparison with the theoretical solutions in the
thermodynamic limit are also plotted. My results match the wider literature in form. Due to time constraints, some
desirable features have not been implemented such as critical exponent estimation, error bars, multidimensional
simulations, automatic autocorrelation time estimation and implementation. However, I plan to slowly work on this
over time and so they may be added at a later date. The parallel_temps function is optimized in several ways but
in particular it is written with parallel programming in mind to take advantage of additional CPU cores.

For an example output, please refer to bigsim.png or example.png.
For an example use of the parallel_temps function, see the bottom of the ising2d.py script.

I am extremely grateful to Dr D Proment for his guidance and I look forward to working with him going forwards.
The method and code of this project will translate naturally to general graphs: at a high level a graph may be
considered as a list of vertices together with a list of connecting edges. Then it is possible to accommodate general
graphs into this project’s pre-existing simulation script and thereby attempt to approach open questions numerically
in more general settings. This will be the basis of a planned research paper between this project’s supervisor Dr D
Proment and myself which we hope to publish as co-authors if we are able to identify and work towards a better
understanding of an open problem. One interesting application could be to better understand how the Ising model
applies to graphs of non-integral dimension.
