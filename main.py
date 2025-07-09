import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

def unit_pyramid(size):
    d = size / 2
    h = size * 1.2
    return np.array([
        [-d, -d, 0],
        [ d, -d, 0],
        [ d,  d, 0],
        [-d,  d, 0],
        [ 0,  0, h]
    ])

def get_faces_pyramid(_):
    return [
        [0, 1, 2, 3],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4]
    ]

def get_faces_cube(_):
    return [
        [0,1,2,3],
        [4,5,6,7],
        [0,1,5,4],
        [1,2,6,5],
        [2,3,7,6],
        [3,0,4,7]
    ]

# Pyramiden-Größe und Würfel-Größe
pyramid_size = 0.5
cube_size = pyramid_size  # Würfelseitenlänge = Basis der Pyramide

pyramid_verts = unit_pyramid(pyramid_size)
pyramid_faces = get_faces_pyramid(pyramid_verts)

# Würfel-Verts definieren (Seitenlänge = cube_size)
d = cube_size / 2
cube_verts = np.array([
    [-d, -d, -d],
    [ d, -d, -d],
    [ d,  d, -d],
    [-d,  d, -d],
    [-d, -d,  d],
    [ d, -d,  d],
    [ d,  d,  d],
    [-d,  d,  d]
])
cube_faces = get_faces_cube(cube_verts)

# Kurvenpunkte (Beispielpunkte, du kannst deine eigenen nutzen)
customPoints = np.array([
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)],
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)],
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)],
    [np.random.randint(1, 6), np.random.randint(1, 6), np.random.randint(1, 6)]
])

p1 = np.array([1, 1, 1])
p2 = np.array([3, 3, 1])

points = np.vstack([
    p1, p2,
    customPoints,
    p1
])
x, y, z = points.T

# Spline durch Punkte
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

# Energieerhaltung für Geschwindigkeit
g = 9.81
z_max = np.max(z_fine)

v_eff = np.sqrt(2 * g * (z_max - z_fine))
v_eff = np.clip(v_eff, 0.3, None)
v_eff = v_eff / np.max(v_eff)

ds = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2 + np.diff(z_fine)**2)
dt = ds / v_eff[:-1]
t_cumulative = np.insert(np.cumsum(dt), 0, 0)
t_cumulative = t_cumulative / t_cumulative[-1]

time_to_u = interp1d(t_cumulative, np.linspace(0, 1, len(t_cumulative)), kind='linear')

n_frames = 300
u_anim = time_to_u(np.linspace(0, 1, n_frames))
x_anim, y_anim, z_anim = splev(u_anim, tck)
T_anim = np.vstack(splev(u_anim, tck, der=1)).T
T_anim = T_anim / np.linalg.norm(T_anim, axis=1)[:, None]

u_vals = np.linspace(0, 1, len(Ups))
Ups_interp = np.stack([
    interp1d(u_vals, Ups[:, i], kind='linear')(u_anim)
    for i in range(3)
], axis=1)

B_anim = np.cross(T_anim, Ups_interp)
N_anim = np.cross(B_anim, T_anim)

# Kurvenlänge zur Positionierung der Waggons
ds_fine = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2 + np.diff(z_fine)**2)
s_fine = np.insert(np.cumsum(ds_fine), 0, 0)
s_total = s_fine[-1]

s_to_u = interp1d(s_fine, u_fine, kind='linear')
u_to_s = interp1d(u_fine, s_fine, kind='linear')

# Würfelanzahl und Abstände einstellen
num_cubes = 5
first_cube_offset = 0.5  # Abstand Pyramide -> erster Würfel (in Einheiten entlang Kurve)
cube_spacing = 1.0       # Abstand Würfel-Würfel (in Einheiten)

# Animation Setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_fine, y_fine, z_fine, 'b-', label='Spline-Kurve')
ax.plot(x, y, z, 'ro', label='Stützpunkte')

pyramid_poly = Poly3DCollection([], facecolors='orange', edgecolors='k', alpha=0.9)
ax.add_collection3d(pyramid_poly)

cube_polys = []
for _ in range(num_cubes):
    cube_poly = Poly3DCollection([], facecolors='cyan', edgecolors='k', alpha=0.9)
    ax.add_collection3d(cube_poly)
    cube_polys.append(cube_poly)

ax.set_xlim(np.min(x_fine)-1, np.max(x_fine)+1)
ax.set_ylim(np.min(y_fine)-1, np.max(y_fine)+1)
ax.set_zlim(np.min(z_fine)-1, np.max(z_fine)+1)
ax.set_title("Pyramide mit angehängten Würfeln (gleicher Abstand & Größe)")
ax.legend()

def update(frame):
    u_now = u_anim[frame]
    s_now = float(u_to_s(u_now))

    # Pyramide transformieren
    pos = np.array([x_anim[frame], y_anim[frame], z_anim[frame]])
    T_vec = T_anim[frame]
    N_vec = N_anim[frame]
    B_vec = B_anim[frame]
    rot = np.column_stack((N_vec, B_vec, T_vec))
    verts_pyr = (pyramid_verts @ rot.T) + pos
    faces_pyr = [[verts_pyr[i] for i in face] for face in pyramid_faces]
    pyramid_poly.set_verts(faces_pyr)

    # Würfel transformieren (mit Abstand und Offset)
    for i, cube_poly in enumerate(cube_polys):
        dist = s_now - first_cube_offset - i * cube_spacing
        dist = dist % s_total
        u_c = float(s_to_u(dist))
        pos_c = np.array(splev(u_c, tck))
        T_c = np.array(splev(u_c, tck, der=1))
        T_c /= np.linalg.norm(T_c)
        Ups_c = np.stack([interp1d(u_vals, Ups[:, d], kind='linear')(u_c) for d in range(3)])
        B_c = np.cross(T_c, Ups_c)
        N_c = np.cross(B_c, T_c)
        rot_c = np.column_stack((N_c, B_c, T_c))
        verts_c = (cube_verts @ rot_c.T) + pos_c
        faces_c = [[verts_c[i] for i in face] for face in cube_faces]
        cube_poly.set_verts(faces_c)

    return [pyramid_poly] + cube_polys

ani = FuncAnimation(fig, update, frames=n_frames, interval=20, blit=False)
plt.show()
