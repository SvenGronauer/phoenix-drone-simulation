r"""Define interface:

Notes:
    - real world logs have:     xyz, xyz_dot, rpy, rpy_dot
    - simulation has has obs:   xyz, quat, xyz_dot, rpy_dot
"""
xyz_real_slice = slice(0, 3)
xyz_dot_real_slice = slice(3, 6)
rpy_real_slice = slice(6, 9)
rpy_dot_real_slice = slice(9, 12)


xyz_sim_slice = slice(0, 3)
quaternion_sim_slice = slice(3, 7)
xyz_dot_sim_slice = slice(7, 10)
rpy_dot_sim_slice = slice(10, 13)
