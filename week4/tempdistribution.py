# numpy-accerelated version (gif-output)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy

T1 = np.full((31, 31), 300)
T2 = np.full((31, 31), 300)
T1[15][15] = 350

dx = 1.0
dy = 1.0
dt = 0.1
alpha = 1.0
ux = 0.3
uy = 0.0

c1x = (alpha * dt) / (dx * dx)
c1y = (alpha * dt) / (dy * dy)
c2x = (ux * dt) / dx
c2y = (uy * dt) / dy

results = []

for t in range(0, 1001):
        if t % 10 == 0:
                print('step = ' + str(t))
                results.append(copy.deepcopy(T1))

        if t == 501:
                ux = 0.1
                uy = 0.2
                c2x = (ux * dt) / dx
                c2y = (uy * dt) / dy
        
        T2 = T1 + c1x * (np.roll(T1, 1, axis=1) - 2 * T1 + np.roll(T1, -1, axis=1)) \
                + c1y * (np.roll(T1, 1, axis=0) - 2 * T1 + np.roll(T1, -1, axis=0)) \
                - c2x * (np.roll(T1, 1, axis=1) - np.roll(T1,-1,axis=1)) \
                - c2y * (np.roll(T1, 1, axis=0) - np.roll(T1,-1,axis=0))        

        T2[15][15] = 350
        for y in range(31):
                T2[y][0] = 300
                T2[y][30] = 300
        for x in range(31):
                T2[0][x] = 300
                T2[30][x] = 300

        T1 = T2.copy()

# gif output
levels = np.linspace(300, 350, 100)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)

def update(frame):
    ax.clear()
    ax.set_title(f"Time: {frame}")
    ax.contourf(results[frame], levels, cmap="hot")

anim = animation.FuncAnimation(fig, update, frames=len(results), interval=10)

anim.save("thermal.gif", writer="pillow")
plt.show()