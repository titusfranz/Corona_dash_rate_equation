from scipy.integrate import odeint
import numpy as np
from lmfit import Model, Parameters


def logistic_ode(y, time, params):
    infected, recovered = y      # unpack current values of y
    infection_rate, recover_rate = params  # unpack parameters
    return [infection_rate*infected * (1 - infected - recovered) - recover_rate*infected,
            recover_rate*infected]


def logistic_function(time,
                      infection_rate=0.1, recover_rate=0.5, mortality=0.007, infected_0=0.001):
    y0 = [infected_0, 0]
    params = [infection_rate, recover_rate]
    y = odeint(logistic_ode, y0, time, args=(params,))
    infected = y[:, 0]
    deaths = mortality * np.array(y[:, 1])
    recovered = np.array(y[:, 1]) - deaths
    return infected, recovered, deaths


def infected_function(time, infection_rate=0.1, recover_rate=0.5, mortality=0.007, infected_0=0.001):
    return logistic_function(time, infection_rate, recover_rate, mortality, infected_0)[0]


LogisticModel = Model(infected_function)


