import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# start of script

def unit_pyramid(size):
    d = size / 2
    h = size * 1.2  # Höhe etwas mehr für bessere Proportionen
    return np.array([
        [-d, -d, 0],
        [ d, -d, 0],
        [ d,  d, 0],
        [-d,  d, 0],
        [ 0,  0, h]
    ])

def get_faces(_):
    return [
        [0, 1, 2, 3],  # Basis
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ]

# Pyramide zuerst definieren
pyramid_size = 0.5
pyramid_verts = unit_pyramid(pyramid_size)
pyramid_faces = get_faces(pyramid_verts)

# Punkte auf der Kurve
customPoints = np.array([
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)],
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)],
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)],
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)]
    ])

#Start- und Endpunkt
p1 = np.array([1, 1, 1])

# Alle Punkte auf der Kurve, mit gleichem Start- und Endpunkt p1
points = np.vstack([
    p1,
    customPoints,
    p1
])
x, y, z = points.T

# Spline & Ableitungen
tck, u = splprep([x, y, z], s=0, k=2, per=True)
u_fine = np.linspace(0, 1, 300)
x_fine, y_fine, z_fine = splev(u_fine, tck)
dx, dy, dz = splev(u_fine, tck, der=1)

T = np.vstack((dx, dy, dz)).T
T = T / np.linalg.norm(T, axis=1)[:, None]

# Up-Vektor verfolgen
up = np.array([0, 0, 1])
Ups = []

for i in range(len(T)):
    if i == 0:
        Ups.append(up)
        continue

    prev_T = T[i - 1]
    cur_T = T[i]
    angle = np.arccos(np.clip(np.dot(prev_T, cur_T), -1, 1))
    axis = np.cross(prev_T, cur_T)

    if np.linalg.norm(axis) < 1e-6:
        Ups.append(Ups[-1])
        continue

    axis = axis / np.linalg.norm(axis)
    theta = angle * 2.0
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    new_up = R @ Ups[-1]
    new_up = (0.95 * new_up + 0.05 * up)
    Ups.append(new_up / np.linalg.norm(new_up))

Ups = np.array(Ups)
B = np.cross(T, Ups)
N = np.cross(B, T)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_fine, y_fine, z_fine, 'b-', label='Spline-Kurve')
ax.plot(x, y, z, 'ro', label='Stützpunkte')

# Initiale leere Pyramide
pyramid_poly = Poly3DCollection([], facecolors='orange', edgecolors='k', alpha=0.9)
ax.add_collection3d(pyramid_poly)

# Achsgrenzen
ax.set_xlim(np.min(x_fine)-1, np.max(x_fine)+1)
ax.set_ylim(np.min(y_fine)-1, np.max(y_fine)+1)
ax.set_zlim(np.min(z_fine)-1, np.max(z_fine)+1)
ax.set_title("Pyramide entlang Kurve")
ax.legend()

# Animation
def update(frame):
    pos = np.array([x_fine[frame], y_fine[frame], z_fine[frame]])
    T_vec = T[frame]
    N_vec = N[frame]
    B_vec = B[frame]
    rot = np.column_stack((N_vec, B_vec, T_vec))
    transformed_verts = (pyramid_verts @ rot.T) + pos
    transformed_faces = [[transformed_verts[i] for i in face] for face in pyramid_faces]
    pyramid_poly.set_verts(transformed_faces)
    return pyramid_poly,

#ani = FuncAnimation(fig, update, frames=len(u_fine), interval=20, blit=False)
ani = FuncAnimation(fig, update, frames=len(u_fine) - 1, interval=20, blit=False)

plt.show()
