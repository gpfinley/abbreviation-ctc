__author__ = 'gpfinley'

import matplotlib
from matplotlib import pyplot

costs = [float(line) for line in open('costs.txt')]



pyplot.plot(range(len(costs)), costs)

pyplot.show()