__author__ = 'gpfinley'

import matplotlib
from matplotlib import pyplot

costs = [float(line) for line in open('costs.txt')]

pyplot.plot(range(len(costs)), costs)

pyplot.show()



dev = open('dev_log.txt')
dev_costs = []
for line in dev:
    try:
        dev_costs.append(float(line))
    except:
        pass

pyplot.plot(range(len(dev_costs)), dev_costs)
pyplot.show()