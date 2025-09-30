# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#plt.plot([2,3,4,5], linewidth=1 , c='b')
#plt.show()

#plt.plot([1,2,3], [4,5,6], linewidth=1, c = 'g')
#plt.show()

#Title
plt.title("Graph tiltle", fontsize = 20, fontweight='bold', loc = 'right', color = 'blue')
#plt.plot([1,2,3], [4,5,6], linewidth=1, c = 'g', linestyle = '--')
#plt.plot([1,2,3], [4,5,6], linewidth=1, c = 'g', linestyle = '-.')
plt.plot([1,2,3], [4,5,6], linewidth=1, c = 'g', linestyle = ':')
plt.plot([1,2,3], [1,4,9])
plt.grid(linestyle = ':')
plt.xlabel('Sequence', fontsize = 12, fontweight='bold')
plt.ylabel('Time(secs)', fontsize = 12, fontweight='bold')
plt.legend(['Mouse', 'Cat'])

plt.xlim([0, 4])      # X축의 범위: [xmin, xmax]
plt.ylim([0, 10])     # Y축의 범위: [ymin, ymax]

plt.show()
