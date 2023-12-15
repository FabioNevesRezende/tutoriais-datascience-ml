import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(10)
x1 = np.arange(10)
y1 = np.random.normal(size=10)

x2 = np.arange(10)
y2 = 1 / np.exp(np.random.normal(size=10) )


# plotar os 2 gráficos
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,3))

# gráfico 1
ax[0].plot(x1, y1)
ax[0].set_title("Gráfico 1")
ax[0].set_xlabel("x")
ax[0].set_ylabel("y1")

# gráfico 2
ax[1].plot(x1, y2)
ax[1].set_title("Gráfico 2")
ax[1].set_xlabel("x")
ax[1].set_ylabel("y2")

fig.tight_layout();
plt.show()