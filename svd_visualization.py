import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import quaternion
from coords_and_svd import n, coords_world, S, Vh

# Play with these booleans:
scale_basis_by_significance = True

show_zeroth_basis = True
show_first_basis_possiblities = False
show_first_basis = True
show_second_basis = True

# We're handling the bases collectively, bools above are mostly for convinience of experimenter
show_basis = [show_zeroth_basis, show_first_basis, show_second_basis]

rot_amount = 0.025

# Generate random rotation as per https://stackoverflow.com/a/44031492
u,v,w = np.random.uniform(0,1,size=3)
full_rotation_quat = np.quaternion(np.sqrt(1-u) * np.sin(2 * np.pi * v), np.sqrt(1-u) * np.cos(2 * np.pi * v), np.sqrt(u) * np.sin(2 * np.pi * w), np.sqrt(u) * np.cos(2 * np.pi * w)) 
rotation_quat = quaternion.slerp(np.quaternion(1,0,0,0), full_rotation_quat, 0, 1, rot_amount)

#viewport rotation:
viewport_quat = quaternion.from_euler_angles([0,np.pi/4,np.pi/4])

#basis vecs:
if scale_basis_by_significance:
    S /= 5 # To fit better
else:
    S = np.asarray([100, 100, 100])

basis_vecs = np.diag(S)
first_basis_possibilities = np.asarray([[0, np.sin(th), np.cos(th)] for th in np.arange(0, 2*np.pi, 0.1*np.pi)]) * 50


lim = np.max(np.abs(coords_world)) * 1.1

#create plot:
fig = plt.figure()
ax = fig.add_subplot()
ax.axis('off')

#plot pointss (zero because draw logic is in update)
scatter = ax.scatter(np.zeros(n), np.zeros(n), color='black')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

# All basis plots start out as zero length, so we can plot them all from the start and only update once needed
basis_plots = [
    ax.plot([0,0], [0,0], c)[0] for c in ['r', 'g', 'b']
]

if show_first_basis_possiblities:
    first_basis_possible_plots = [
        ax.plot([0,0], [0,0], 'g')[0] for _ in range(len(first_basis_possibilities))
    ]

    
def update(frame):
    global coords_world, basis_vecs, first_basis_possibilities
    coords_world = quaternion.rotate_vectors(rotation_quat, coords_world)
    coords_viewport = quaternion.rotate_vectors(viewport_quat, coords_world)
    
    updates = [scatter]

    basis_vecs = quaternion.rotate_vectors(rotation_quat, basis_vecs)
    basis_vecs_viewport = quaternion.rotate_vectors(viewport_quat, basis_vecs)

    first_basis_possibilities = quaternion.rotate_vectors(rotation_quat, first_basis_possibilities)
    first_basis_possibilities_viewport = quaternion.rotate_vectors(viewport_quat, first_basis_possibilities)

    for i in range(3):
        if show_basis[i]:
            basis_plots[i].set_data([0, basis_vecs_viewport[i,0]], [0, basis_vecs_viewport[i,1]])
            updates.append(basis_plots[i])
    
    if show_first_basis_possiblities:
        for i in range(len(first_basis_possibilities)):
            first_basis_possible_plots[i].set_data([0, first_basis_possibilities_viewport[i,0]], [0, first_basis_possibilities_viewport[i,1]])
            updates.append(first_basis_possible_plots[i])
     
    scatter.set_offsets(coords_viewport[:,:2])
    
    return updates

ani = animation.FuncAnimation(fig, func=update, interval=50, frames=1, cache_frame_data=False)

#writer = animation.PillowWriter(fps=10,
#                                metadata=dict(artist='https://github.com/Threadnaught'),
#                                bitrate=1800)
#ani.save('gifs/s5.gif', writer=writer)

plt.show()
