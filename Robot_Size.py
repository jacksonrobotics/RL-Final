import pybullet as p
import os
import numpy as np
directory = os.getcwd()  # Gets the current file directory

# Source: https://github.com/bulletphysics/bullet3/issues/2328

urdf_root=os.path.join(directory, 'V000/urdf', 'V000.urdf')

p.connect(p.DIRECT)
idx = p.loadURDF(urdf_root, basePosition=[0,0,0], baseOrientation=[0,0,0,1], useFixedBase=True, globalScaling=1.0)
boundaries = p.getAABB(idx)
lwh = np.array(boundaries[1])-np.array(boundaries[0])
print(lwh)
p.disconnect()