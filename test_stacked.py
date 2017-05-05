import matplotlib.pyplot as plt
import numpy as np

y1 = np.random.random(10)*5
y2 = np.random.random(10)
x = np.arange(10)

plt.bar(x, y1, color="b")
plt.bar(x, y2, bottom=y1, color="r")
plt.show()
