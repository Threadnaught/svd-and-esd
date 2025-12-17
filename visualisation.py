import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import quaternion

np.random.seed(5)

n=200
dims=[100,25,50]
rot_amount = 0.025

# Play with these booleans:
add_basis = True
constrain_first_axis = True
constrain_second_axis = True
scale_basis_by_significance = True

coords_world = np.transpose([np.random.normal(scale=x,size=n) for x in dims])

# Compute the SVD:
U, S, Vh = np.linalg.svd(coords_world)

# Pre-align our coordinate system with the computed SVD.
# This is a little cheaty but since we're going to be rotating them all over the place it doesn't REALLY matter
coords_world = np.matmul(coords_world, Vh)

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


x_basis = quaternion.rotate_vectors(viewport_quat, [1,0,0]) * S[0]
y_basis = quaternion.rotate_vectors(viewport_quat, [0,1,0]) * S[1]
z_basis = quaternion.rotate_vectors(viewport_quat, [0,0,1]) * S[2]

lim = np.max(np.abs(coords_world)) * 1.1

#create plot:
fig = plt.figure()
ax = fig.add_subplot()
ax.axis('off')

#plot pointss (zero because draw logic is in update)
scatter = ax.scatter(np.zeros(n), np.zeros(n))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

# Add the basis vectors:
if add_basis:
    x_pl = ax.plot([0,x_basis[0]], [0,x_basis[1]], 'r')
    y_pl = ax.plot([0,y_basis[0]], [0,y_basis[1]], 'g')
    z_pl = ax.plot([0,z_basis[0]], [0,z_basis[1]], 'b')

if constrain_first_axis:
    # Kill all rotation except around the 0th axis
    unconstrained_rot_vector = quaternion.as_rotation_vector(rotation_quat)
    rotation_quat = quaternion.from_rotation_vector([unconstrained_rot_vector[0], 0, 0])

if constrain_second_axis:
    # Kill all rotation except around the 1st axis
    unconstrained_rot_vector = quaternion.as_rotation_vector(rotation_quat)
    rotation_quat = quaternion.from_rotation_vector([0, unconstrained_rot_vector[1], 0])

def update(frame):
    global coords_world
    coords_world = quaternion.rotate_vectors(rotation_quat, coords_world)
    coords_viewport = quaternion.rotate_vectors(viewport_quat, coords_world)
     
    scatter.set_offsets(coords_viewport[:,:2])
    
    return (scatter)

ani = animation.FuncAnimation(fig, func=update, interval=50, frames=1, cache_frame_data=False)

#writer = animation.PillowWriter(fps=10,
#                                metadata=dict(artist='https://github.com/Threadnaught'),
#                                bitrate=1800)
#ani.save('gifs/s5.gif', writer=writer)

plt.show()
