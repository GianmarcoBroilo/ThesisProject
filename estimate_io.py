## Context
"""
This code is used to estimate the covariance matrix that will be used as a new "a priori covariance matrix" in the
estimation.py script. This model simulates 3D cartesian position observables every 2 days on both Callisto
with accuracy that is the same as the a priori uncertainty.
"""
#%%


# Load standard modules
import numpy as np
from numpy import linalg as lalg
from matplotlib import pyplot as plt
import datetime
# Load tudatpy modules
#from tudatpy.util import result2array

import sys

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.astro import time_conversion
from sklearn.preprocessing import normalize



# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs START: 2023-04-01 END: 2025-04-01
calendar_start = datetime.datetime(2020,6,1)
simulation_start_epoch = time_conversion.calendar_date_to_julian_day_since_epoch(calendar_start)*constants.JULIAN_DAY
simulation_end_epoch = simulation_start_epoch +  2*constants.JULIAN_YEAR


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Io", "Jupiter","Sun","Saturn","Europa"]

time_step = 1500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create tabulated settings fo Io
original_io_ephemeris_settings = body_settings.get("Io").ephemeris_settings
body_settings.get("Io").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_io_ephemeris_settings,initial_time, final_time, time_step)


# Rotation model
body_settings.get("Io").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter",
                                                                                               global_frame_orientation,"Io_Fixed")
# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


"""
Propagation setup
"""

# Define bodies that are propagated
bodies_to_propagate = ["Io"]

# Define central bodies of propagation
central_bodies = ["Jupiter"]

# Create the acceleration model
acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        8,0,2,2)],
    Sun = [propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()],
    Europa = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        2,2,2,2)]
)

acceleration_settings = {"Io": acceleration_settings_io}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,acceleration_settings,bodies_to_propagate,central_bodies
)

# Define the initial state
"""
The initial state of Io that will be propagated is now defined 
"""

# Set the initial state of Io
initial_state = propagation.get_initial_state_of_bodies(
    bodies_to_propagate=bodies_to_propagate,
    central_bodies=central_bodies,
    body_system=bodies,
    initial_time=simulation_start_epoch,
)

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    termination_condition
)
"""
Propagate the dynamics of Io and extract state transition and sensitivity matrices
"""
#Setup paramaters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)

link_ends_stellar = dict()
link_ends_stellar[observation.receiver] = ("Earth", "")
link_ends_stellar[observation.transmitter] = ("Io", "")
link_ends_vlbi = dict()
link_ends_vlbi[observation.receiver] = ("Earth", "")
link_ends_vlbi[observation.transmitter] = ("Io", "")

parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(link_ends_stellar,observation.angular_position_type))
# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# Create numerical integrator settings.
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    simulation_start_epoch, 1800.0, propagation_setup.integrator.rkf_78, 1800.0, 1800.0, 1.0, 1.0
)
# Create the variational equation solver and propagate the dynamics
variational_equations_solver = numerical_simulation.SingleArcVariationalSimulator(
    bodies, integrator_settings, propagator_settings, parameters_to_estimate, integrate_on_creation=True)

# Extract the resulting state history, state transition matrix history, and sensitivity matrix history
states = variational_equations_solver.state_history
state_transition_matrix = variational_equations_solver.state_transition_matrix_history
sensitivity_matrix = variational_equations_solver.sensitivity_matrix_history

""""
Define the a priori covariance of Io
"""
#%%
#15km RSW position 0.15,1.15,0.75m/s RSW velocity
rotation_rsw_to_inertial_dict_io = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_io[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][:6]).reshape(3,3)
uncertainties_rsw_io = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_io,[15e3,15e3,15e3])
uncertainties_rsw_velocity_io = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_io,[0.15,1.15,0.75])
covariance_position_initial_io = lalg.multi_dot([rotation_rsw_to_inertial_dict_io[simulation_start_epoch],uncertainties_rsw_io**2,rotation_rsw_to_inertial_dict_io[simulation_start_epoch].T])
covariance_velocity_initial_io = lalg.multi_dot([rotation_rsw_to_inertial_dict_io[simulation_start_epoch],uncertainties_rsw_velocity_io**2,rotation_rsw_to_inertial_dict_io[simulation_start_epoch].T])

""""
Define global a priori covariance 
"""
covariance_a_priori = np.block([
    [covariance_position_initial_io, np.zeros((3,3))],
    [np.zeros((3,3)),covariance_velocity_initial_io]
])
covariance_a_priori2 = np.genfromtxt('/Users/gianmarcobroilo/Desktop/ThesisResults/vlbi-corrected/IO/covariance_matrix_io_best_200.dat')

bias = np.zeros((2,2))
np.fill_diagonal(bias,[1.93925472e-8**2,1.93925472e-8**2])

covariance_apriori_biased = np.block([
    [covariance_a_priori2,np.zeros((6,2))],
    [np.zeros((2,6)),bias]
])

covariance_a_priori_inverse = np.linalg.inv(covariance_apriori_biased)


""""
Observation Setup
"""
#%%

# Define the uplink/downlink link ends types
link_ends_io = dict()
link_ends_io[observation.observed_body] = ("Io","")


# Create observation settings for each link/observable
observation_settings_list_pos = observation.cartesian_position(link_ends_io)
bias_stellar = observation.absolute_bias(np.array([1.45444104e-8,1.45444104e-8]))

observation_settings_list_stellar = observation.angular_position(link_ends_stellar, bias_settings = bias_stellar)


# Define the observations for Callisto
observations_position = np.arange(simulation_start_epoch,simulation_end_epoch, 2*constants.JULIAN_DAY)
stellar_occ = datetime.datetime(2021,4,2)
stellar_occ = time_conversion.calendar_date_to_julian_day_since_epoch(stellar_occ)*constants.JULIAN_DAY
observation_times_io = np.array([stellar_occ])


observation_simulation_settings_pos = observation.tabulated_simulation_settings(
    observation.position_observable_type,
    link_ends_io,
    observations_position,
    reference_link_end_type = observation.observed_body
)
observation_simulation_settings_stellar = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_stellar,
    observation_times_io
)


# Add noise level of 15km to position observable
noise_level_io = 200e3
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_pos],
    noise_level_io,
    observation.position_observable_type
)

noise_level_stellar =  4.8481368e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_stellar],
    noise_level_stellar,
    observation.angular_position_type
)


""""
Estimation setup
"""
#%%
# Collect all settings required to simulate the observations
observation_settings_list = []
observation_settings_list.append(observation_settings_list_stellar)
#observation_settings_list.append(observation_settings_list_pos)


observation_simulation_settings = []
observation_simulation_settings.append(observation_simulation_settings_stellar)
#observation_simulation_settings.append(observation_simulation_settings_pos)


# Create the estimation object for Callisto and Jupiter
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    integrator_settings,
    propagator_settings)

# Simulate required observation on Callisto and Jupiter
simulated_observations = estimation.simulate_observations(
    observation_simulation_settings,
    estimator.observation_simulators,
    bodies)

# Collect all inputs for the inversion in a POD
truth_parameters = parameters_to_estimate.parameter_vector
pod_input = estimation.PodInput(
    simulated_observations, parameters_to_estimate.parameter_set_size, inverse_apriori_covariance=covariance_a_priori_inverse)

pod_input.define_estimation_settings(
    reintegrate_variational_equations=False)

# Setup the weight matrix W with weights for Callisto
weights_position_io = noise_level_io ** -2
weights_stellar = noise_level_stellar ** -2

#pod_input.set_constant_weight_for_observable_and_link_ends(observation.position_observable_type,link_ends_io,weights_position_io)
pod_input.set_constant_weight_for_observable_and_link_ends(observation.angular_position_type,link_ends_stellar,weights_stellar)


""""
Run the estimation
"""
# Perform estimation (this also prints the residuals and partials)
convergence = estimation.estimation_convergence_checker(1)
pod_output = estimator.perform_estimation(pod_input, convergence_checker=convergence)



""""
Post process the results 
"""
# Plot the correlation between the outputs
plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.title("Correlation between the outputs")
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
""""
Propagate the covariance matrix for prediction
"""
state_transition = estimator.state_transition_interface
cov_initial =  pod_output.covariance

# Create covariance dictionaries
propagated_times = dict()
propagated_covariance_rsw_dict = dict()
propagated_covariance_rsw_dict_io = dict()
propagated_formal_errors_rsw_dict_io = dict()


time = np.arange(simulation_start_epoch, simulation_end_epoch, 86400)
time = np.ndarray.tolist(time)

propagation = estimation.propagate_covariance_split_output(initial_covariance= cov_initial,state_transition_interface = state_transition,output_times = time)

propagated_times = propagation[0]
propagated_times = [int(num) for num in propagated_times]
propagated_covariance = propagation[1]

propagated_covariance_dict = {propagated_times[i]: propagated_covariance[i] for i in range(len(propagated_times))}

for epoch in list(propagated_covariance_dict):
    propagated_covariance_rsw_dict_io[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_io[epoch].T,propagated_covariance_dict[epoch][:3,:3],rotation_rsw_to_inertial_dict_io[epoch]])
    propagated_formal_errors_rsw_dict_io[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_io[epoch]))


# times are equal to epochs in state history
time_io = np.array(list(propagated_times))
values_io = np.vstack(propagated_formal_errors_rsw_dict_io.values())
ti = time_io/31536000
#%%
""""
Plot the propagated uncertainties  
"""

plt.figure(figsize=(9,5))
plt.plot(ti,values_io[:,0], label = 'R', color = 'red')
plt.plot(ti,values_io[:,1], label = 'S', color = 'green')
plt.plot(ti,values_io[:,2], label = 'W', color = 'blue')
plt.axvline(x = observation_times_io/31536000,label = 'occ',color="magenta")
plt.ylim([10e2,10e4])
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Io")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time [years after J2000]')
plt.legend()
plt.show()

#%%
""""
Export Covariance Matrix to use as input 
"""
covariance_matrix = np.savetxt("/Users/gianmarcobroilo/Desktop/ThesisResults/vlbi-corrected/IO/covariance_matrix_io_best_200.dat",pod_output.covariance)

#%%
""""
Propagate RA and DEC of Io
"""
da_dr = dict()
dd_dr = dict()
Ta_dict = dict()
Td_dict = dict()
T = dict()
propagated_icrf_cal = dict()
formal_errors_icrf_cal = dict()
for epoch in list(propagated_covariance_dict):
    Ta_dict[epoch] = np.array([-states[epoch][1],states[epoch][0],0]).reshape(1,3)
    Td_dict[epoch] = np.array([-states[epoch][0]*states[epoch][2],-states[epoch][1]*states[epoch][2],states[epoch][0]**2+states[epoch][1]**2]).reshape(1,3)
    da_dr[epoch] = 1/(states[epoch][0]**2 + states[epoch][1]**2)*Ta_dict[epoch]
    dd_dr[epoch] = 1/(np.linalg.norm(states[epoch][0:2])**2*np.sqrt(states[epoch][0]**2+states[epoch][1]**2))*Td_dict[epoch]
    T[epoch] = np.vstack((da_dr[epoch],dd_dr[epoch]))
    propagated_icrf_cal[epoch] = lalg.multi_dot([T[epoch],propagated_covariance_dict[epoch][:3,:3],T[epoch].T])
    formal_errors_icrf_cal[epoch] = np.sqrt(np.diag(propagated_icrf_cal[epoch]))

values_icrf = np.vstack(formal_errors_icrf_cal.values())
alpha = values_icrf[:,0]
dec = values_icrf[:,1]

fig, axs = plt.subplots(2,figsize=(12, 6))
fig.suptitle('Propagated uncertainties in Right Ascension and Declination of Io')


axs[0].plot(ti,alpha, color = 'black')
axs[0].set_ylabel('Right Ascension [rad]')
axs[0].set_yscale("log")
axs[0].axvline(x = observation_times_io/31536000,color="magenta")
axs[1].plot(time_io/31536000,dec, color = 'black')
axs[1].set_ylabel('Declination [rad]')
axs[1].set_xlabel('Time [years after J2000]')
axs[1].set_yscale("log")
axs[1].axvline(x = observation_times_io/31536000,color="magenta")
plt.show()
