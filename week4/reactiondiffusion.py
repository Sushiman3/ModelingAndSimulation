# numpy-accerelated version (gif-output)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

u1=np.full((201,201),1.0)
v1=np.full((201,201),0.0)
u1[90:111,90:111]=0.5
v1[90:111,90:111]=0.25

u2 = u1.copy()
v2 = v1.copy()

dx = 0.01
dy = 0.01
dt = 1.0
du = 2 * 10**-5
dv = 1 * 10**-5
f = 0.022
k = 0.051
c1x = dt/(dx*dx)
c1y = dt/(dy*dy)
results = []

def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
        4 * Z
    )

for t in range(100001):
    if t % 100 == 0:
        print(f"Time: {int(t/100)}")
        results.append(u1.copy())

    Lu = laplacian(u1)
    Lv = laplacian(v1)

    u2 = u1 + du * c1x * Lu - u1 * v1 ** 2 + f * (1 - u1)
    v2 = v1 + dv * c1x * Lv + u1 * v1 ** 2 - (f + k) * v1

    u1 = u2
    v1 = v2

#gif output

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)

def update(frame):
  ax.clear()
  im = ax.imshow(results[frame], cmap="Blues", vmin=np.min(u1), vmax=np.max(u1))
  ax.set_title(f"Time: {frame}")
  return im

anim = animation.FuncAnimation(fig, update, frames=len(results), interval=100)

anim.save("thermal.gif", writer="pillow")
plt.show()
