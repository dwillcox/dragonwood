"""
Module for calculating probabilities for the card game DragonWood.

Copyright (c) 2016, Donald E. Willcox
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib.pyplot as plt
import numpy as np

def die_outcomes():
    # Return the possible die outcomes
    return range(1, 5) # [1, 4]

def die_multiplicity(x):
    # Given outcome x, return its multiplicity
    if x == 1 or x == 4:
        return 1
    elif x == 2 or x == 3:
        return 2
    else:
        return 0

def die_mult_sum():
    # Return sum of multiplicities of die outcomes
    msum = 0
    for x in die_outcomes():
        msum += die_multiplicity(x)
    return msum

def die_probability(x):
    # Return the probability of die outcome being x
    return float(die_multiplicity(x))/float(die_mult_sum())
        
def die_expectation():
    # Return the expectation value for a single die
    e = 0.0
    for x in die_outcomes():
        e += die_probability(x)
    return e

def sum_outcomes(N):
    # Return the possible sum outcomes for N die
    return range(N, 4*N+1) # [N, 4*N]

def sum_expectation(N):
    # Given N die, return the sum expectation value
    return N * die_expectation()

def sum_multiplicity(N, x):
    # If x is not in the possible outcomes, return 0.
    if not x in sum_outcomes(N):
        return 0
    # Given sum outcome x, determine its multiplicity.
    if N == 1:
        # Return single die multiplicity for x
        return die_multiplicity(x)
    gsum = 0
    # Recurse
    for y in die_outcomes():
        gsum += die_multiplicity(y) * sum_multiplicity(N-1, x-y)
    return gsum

def sum_mult_sum(N):
    # Return the sum of multiplicities for a sum on N die
    msum = 0
    for x in sum_outcomes(N):
        msum += sum_multiplicity(N, x)
    return msum

def sum_probability(N, x):
    # Given outcome x for a sum on N die, return the probability
    return float(sum_multiplicity(N, x))/float(sum_mult_sum(N))

def sum_cdf(N, y):
    # Return cumulative distribution function for sum of N die being y or less.
    cdf = 0.0
    for x in sum_outcomes(N):
        if x <= y:
            cdf += sum_probability(N, x)
    return cdf

def sum_gt_thresh_prob(N, T):
    # Return probability sum on N die will be greater than or equal to T
    gtp = 0.0
    for x in sum_outcomes(N):
        if x >= T:
            gtp += sum_probability(N, x)
    return gtp

def get_num_die_thresh(T, CL):
    # Given a threshold T and confidence level CL, return
    # the least number of dice for which the probability
    # that their sum will be greater than or equal to T
    # is at least CL.
    if CL > 1.0 or CL < 0.0 or T < 1.0:
        return 0
    ndie = 1
    while True:
        if sum_gt_thresh_prob(ndie, T) >= CL:
            return ndie
        else:
            ndie += 1

def plot_prob_vs_numdie(T):
    # Given a threshold T, plot the probability of rolling
    # greater than or equal to T vs. number of cards used.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    numdie = np.arange(1, 7)
    pthrow = np.array([sum_gt_thresh_prob(n, T) for n in numdie])
    ax.plot(numdie, pthrow, linestyle='None', marker='o', color='green')
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Number of Die')
    ax.set_ylabel('$\mathrm{Probability \ Sum \\geq ' + '{}'.format(T) + '}$')
    plt.tight_layout()
    plt.show()

def plot_prob_vs_sum(N):
    # Given the number of die N, plot the probability
    # of each possible sum outcome.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sout = sum_outcomes(N)
    pout = np.array([sum_probability(N, s) for s in sout])
    ax.bar(sout, pout, align='center', color='green')
    ax.set_xticks(sout, ['{}'.format(s) for s in sout])
    ax.set_xlim([sout[0]-1, sout[-1]+1])
    ax.set_ylim([0.0, 1.0])    
    ax.set_xlabel('Sum Outcomes')
    ax.set_ylabel('{} Die Sum Probability Distribution'.format(N))
    plt.tight_layout()
    plt.show()

def plot_geprob_vs_thresh(N):
    # Given the number of die N, plot the probability
    # of rolling >= each possible threshold.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sout = sum_outcomes(N)
    pout = np.array([sum_gt_thresh_prob(N, s) for s in sout])
    ax.bar(sout, pout, align='center', color='green')
    ax.set_xticks(sout, ['{}'.format(s) for s in sout])    
    ax.set_xlim([sout[0]-1, sout[-1]+1])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Sum Threshold')
    ax.set_ylabel('{} Die G.E. Threshold Probability Distribution'.format(N))
    plt.tight_layout()
    plt.show()

    
