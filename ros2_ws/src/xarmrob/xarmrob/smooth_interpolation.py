#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:07:24 2021

@author: pgadamczyk
"""

# Minimum Jerk Trajectory
# To use: 
    # import smooth_interpolation as smooth
# call the function(s): 
    # smooth.constant_velocity_interpolation(pos_init, pos_end, endpoint_speed, command_frequency)
    # smooth.minimum_jerk_interpolation(pos_init, pos_end, endpoint_speed, command_frequency)

import numpy as np

# Constant-Velocity Interpolation
def constant_velocity_interpolation(pos_init, pos_end, endpoint_speed, command_frequency):
    
    displacement = pos_end-pos_init
    distance = np.linalg.norm(displacement)
    duration_nom = distance / endpoint_speed 
    
    nsteps = np.ceil( duration_nom * command_frequency)
    duration_spec = nsteps / command_frequency
    t_rel = np.arange(nsteps)/nsteps
    t = t_rel*duration_spec
    
    # Code 
    disp_traj = np.linspace(pos_init, pos_end, len(t))
    
    return t,disp_traj

# Minimum-Jerk Interpolation
def minimum_jerk_interpolation(pos_init, pos_end, endpoint_speed, command_frequency):
    
    displacement = pos_end-pos_init
    distance = np.linalg.norm(displacement)
    duration_nom = distance / endpoint_speed 
    
    nsteps = np.ceil( duration_nom * command_frequency)
    duration_spec = nsteps / command_frequency
    t_rel = np.arange(nsteps)/nsteps
    t = t_rel*duration_spec
    
    min_jerk_traj = (10*t_rel**3 - 15*t_rel**4 + 6*t_rel**5)
    # Scale the displacement trajectory
    # Code 
    disp_traj = np.column_stack( [p_i+disp*min_jerk_traj for p_i,disp in zip(pos_init,displacement) ])  
    
    return t,disp_traj


if __name__ == '__main__' :
    # parameters
    endpoint_speed = 0.1
    command_frequency = 30
    pos_init = np.array([0.1, 0.2, 0.1])
    pos_end = np.array([0.3, -0.2, 0.2])
    
    from matplotlib import pyplot as plt
    
    # Plot the Minimum-Jerk Trajectory
    t,disp_traj_minjerk = minimum_jerk_interpolation(pos_init, pos_end, endpoint_speed, command_frequency)
    f = plt.figure()
    plt.plot(t,disp_traj_minjerk)
    plt.legend(('x','y','z'))
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Minimum-Jerk Trajectory')

    # Plot a Linear Spacing Trajectory for contrast
    t, disp_traj_constvel = constant_velocity_interpolation(pos_init, pos_end, endpoint_speed, command_frequency)
    f2 = plt.figure()
    plt.plot(t, disp_traj_constvel)
    plt.legend(('x','y','z'))
    plt.xlabel('Time (s)')
    plt.ylabel('Position')    
    plt.title('Constant-Velocity Trajectory')

    # Do a simpler example for plotting the spacing of points
    t, disp_traj = minimum_jerk_interpolation(np.asarray([0.,0.,0.]),np.asarray([1.,0.,0.]),1,20)
    plt.figure(); plt.plot(disp_traj[:,0],disp_traj[:,2], '-o')
    t, disp_traj = constant_velocity_interpolation(np.asarray([0.,0.,0.]),np.asarray([1.,0.,0.]),1,20)
    plt.figure(); plt.plot(disp_traj[:,0],disp_traj[:,2], '-o')