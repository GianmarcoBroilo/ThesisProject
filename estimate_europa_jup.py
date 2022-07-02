
## Context
"""
INPUT
Data type: stellar occultation of Europa, VLBI Jupiter
A priori cov: uncertainties in the state of Europa and Jupiter in RSW
parameters: initial state of Europa, initial state of Jupiter bias for stellar occultation and VLBI 
OUTPUT
cov: uncertainty and correlation of estimated parameters for both Europa and Jupiter
"""
#%%


# Load standard modules
import numpy as np
from numpy import linalg as lalg
from matplotlib import pyplot as plt
import datetime
# Load tudatpy modules
from tudatpy.util import result2array
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



# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs START: 2019-06-01 END: 2021-06-01
calendar_start = datetime.datetime(2019,6,1)
simulation_start_epoch = time_conversion.calendar_date_to_julian_day_since_epoch(calendar_start)*constants.JULIAN_DAY
simulation_end_epoch = simulation_start_epoch +  2*constants.JULIAN_YEAR


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Io", "Jupiter","Sun","Saturn","Europa"]

time_step = 3500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create tabulated settings fo Io and Jupiter
original_europa_ephemeris_settings = body_settings.get("Europa").ephemeris_settings
body_settings.get("Europa").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_europa_ephemeris_settings,initial_time, final_time, time_step
)
original_jupiter_ephemeris_settings = body_settings.get("Jupiter").ephemeris_settings
body_settings.get("Jupiter").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_jupiter_ephemeris_settings,initial_time, final_time, time_step
)

# Rotation model
body_settings.get("Europa").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter",
                                                                                               global_frame_orientation,"Europa_Fixed")
# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


"""
Propagation setup
"""

# Define bodies that are propagated
bodies_to_propagate = ["Europa","Jupiter"]

# Define central bodies of propagation
central_bodies = []
for body_name in bodies_to_propagate:
    if body_name == "Europa":
        central_bodies.append("Jupiter")
    elif body_name == "Jupiter":
        central_bodies.append("Sun")

### Create the acceleration model
acceleration_settings_europa = dict(
    Jupiter = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        8,0,2,2)],
    Sun = [propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()],
    Io = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        2,2,2,2)],
    Ganymede = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        2,2,2,2)]
)
acceleration_settings_jup = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings = {"Europa": acceleration_settings_europa, "Jupiter": acceleration_settings_jup}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,acceleration_settings,bodies_to_propagate,central_bodies
)
### Define the initial state
"""
The initial state of Europa and Jupiter that will be propagated is now defined. 
"""

# Set the initial state of Europa
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
Propagate the dynamics of Jupiter and Io and extract state transition and sensitivity matrices
"""
#Setup paramaters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)


link_ends_europa = dict()
link_ends_europa[observation.receiver] = ("Earth", "")
link_ends_europa[observation.transmitter] = ("Europa", "")
link_ends_jup = dict()
link_ends_jup[observation.receiver] = ("Earth", "")
link_ends_jup[observation.transmitter] = ("Jupiter", "")

bias_stellar = observation.absolute_bias(np.array([1.93925472e-8,1.93925472e-8]))
bias_vlbi = observation.absolute_bias(np.array([0.5e-9,0.5e-9]))

parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(link_ends_europa,observation.angular_position_type))
parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(link_ends_jup,observation.angular_position_type))

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
Define the a priori covariance of Europa
"""
#%%
#15km RSW position 0.15,1.15,0.75m/s RSW velocity
rotation_rsw_to_inertial_dict_eu = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_eu[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][:6]).reshape(3,3)
uncertainties_rsw_eu = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_eu,[15e3,15e3,15e3])
uncertainties_rsw_velocity_eu = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_eu,[0.15,1.15,0.75])
covariance_position_initial_eu = lalg.multi_dot([rotation_rsw_to_inertial_dict_eu[simulation_start_epoch],uncertainties_rsw_eu**2,rotation_rsw_to_inertial_dict_eu[simulation_start_epoch].T])
covariance_velocity_initial_eu = lalg.multi_dot([rotation_rsw_to_inertial_dict_eu[simulation_start_epoch],uncertainties_rsw_velocity_eu**2,rotation_rsw_to_inertial_dict_eu[simulation_start_epoch].T])

""""
Define the a priori covariance of Jupiter 
"""
# 1km RSW position 0.1m/s RSW velocity
rotation_rsw_to_inertial_dict_jup = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_jup[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][6:12]).reshape(3,3)
uncertainties_rsw_jup = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_jup,[1e3,1e3,1e3])
uncertainties_rsw_velocity_jup = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_jup,[0.1,0.1,0.1])
covariance_position_initial_jup = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[simulation_start_epoch],uncertainties_rsw_jup**2,rotation_rsw_to_inertial_dict_jup[simulation_start_epoch].T])
covariance_velocity_initial_jup = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[simulation_start_epoch],uncertainties_rsw_velocity_jup**2,rotation_rsw_to_inertial_dict_jup[simulation_start_epoch].T])

""""
Define global a priori covariance 
"""
covariance_a_priori = np.block([
    [covariance_position_initial_eu, np.zeros((3,3)), np.zeros((3,3)),np.zeros((3,3))],
    [np.zeros((3,3)),covariance_velocity_initial_eu,np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)),np.zeros((3,3)), covariance_position_initial_jup, np.zeros((3,3))],
    [np.zeros((3,3)),np.zeros((3,3)), np.zeros((3,3)), covariance_velocity_initial_jup],
])

covariance_a_priori2 = np.genfromtxt('/Users/gianmarcobroilo/Desktop/ThesisResults/vlbi-corrected/IO/output_covariance_io_jup_nominal_15.dat')

bias_matrix = np.zeros((4,4))
np.fill_diagonal(bias_matrix,[1.93925472e-8**2,1.93925472e-8**2,0.5e-9**2,0.5e-9**2])

covariance_a_priori_bias = np.block([
    [covariance_a_priori2,np.zeros((12,4))],
    [np.zeros((4,12)),bias_matrix]
])

covariance_a_priori_inverse = np.linalg.inv(covariance_a_priori_bias)

""""
Observation Setup
"""
#%%

# Define the uplink link ends types


# Create observation settings for each link/observable
observation_settings_list_europa = observation.angular_position(link_ends_europa, bias_settings = bias_stellar)
observation_settings_list_jup = observation.angular_position(link_ends_jup, bias_settings = bias_vlbi)

# Define the observations for Europa
stellar_occ = datetime.datetime(2020,6,22)
stellar_occ = time_conversion.calendar_date_to_julian_day_since_epoch(stellar_occ)*constants.JULIAN_DAY
observation_times_europa = np.array([stellar_occ])

observation_simulation_settings_europa = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_europa,
    observation_times_europa
)

# Define the observations for Jupiter VLBI

observation_times_jup = np.arange(simulation_start_epoch,simulation_end_epoch,53.4*constants.JULIAN_DAY)
observation_simulation_settings_jup = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_jup,
    observation_times_jup
)


# Add noise levels of roughly 1mas to Io
noise_level_europa =  4.8481368e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_europa],
    noise_level_europa,
    observation.angular_position_type
)

# Add noise levels of roughly 05 nrad to Jupiter
noise_level_jup = 0.5e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_jup],
    noise_level_jup,
    observation.angular_position_type
)


""""
Estimation setup
"""
#%%
observation_settings_list = []
observation_settings_list.append(observation_settings_list_europa)
observation_settings_list.append(observation_settings_list_jup)
observation_simulation_settings = []
observation_simulation_settings.append(observation_simulation_settings_europa)
observation_simulation_settings.append(observation_simulation_settings_jup)

# Create the estimation object for Europa and Jupiter
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    integrator_settings,
    propagator_settings)

# Simulate required observation on Europa and Jupiter
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

# Setup the weight matrix W
weights_position_jup = noise_level_jup ** -2
weights_position_europa = noise_level_europa ** -2

pod_input.set_constant_weight_for_observable_and_link_ends(observation.angular_position_type,link_ends_jup,weights_position_jup)
pod_input.set_constant_weight_for_observable_and_link_ends(observation.angular_position_type,link_ends_europa,weights_position_europa)

""""
Run the estimation
"""
# Perform estimation (this also prints the residuals and partials)
convergence = estimation.estimation_convergence_checker(1)
pod_output = estimator.perform_estimation(pod_input, convergence_checker=convergence)



""""
Post process the results 
"""
plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.title("Correlation between the outputs")
plt.colorbar()
plt.tight_layout()
plt.show()

observation_times = np.array(simulated_observations.concatenated_times)

""""
Propagate the covariance matrix for prediction
"""
#%%

""""
Propagate the covariance matrix for prediction
"""
state_transition = estimator.state_transition_interface
cov_initial = pod_output.covariance

# Create covariance dictionaries
propagated_times = dict()
propagated_covariance_rsw_dict = dict()
propagated_covariance_rsw_dict_eu = dict()
propagated_covariance_rsw_dict_jup = dict()
propagated_formal_errors_rsw_dict_eu = dict()
propagated_formal_errors_rsw_dict_jup = dict()

time = np.arange(simulation_start_epoch, simulation_end_epoch, 86400)
time = np.ndarray.tolist(time)

propagation = estimation.propagate_covariance_split_output(initial_covariance= cov_initial,state_transition_interface = state_transition,output_times = time)

propagated_times = propagation[0]
propagated_times = [int(num) for num in propagated_times]
propagated_covariance = propagation[1]

propagated_covariance_dict = {propagated_times[i]: propagated_covariance[i] for i in range(len(propagated_times))}

for epoch in list(propagated_covariance_dict):
    propagated_covariance_rsw_dict_eu[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_eu[epoch].T,propagated_covariance_dict[epoch][:3,:3],rotation_rsw_to_inertial_dict_eu[epoch]])
    propagated_formal_errors_rsw_dict_eu[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_eu[epoch]))
    propagated_covariance_rsw_dict_jup[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[epoch].T,propagated_covariance_dict[epoch][6:9,6:9],rotation_rsw_to_inertial_dict_jup[epoch]])
    propagated_formal_errors_rsw_dict_jup[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_jup[epoch]))


# times are equal to epochs in state history
time_europa = np.array(list(propagated_times))
time_jup = np.array(list(propagated_times))
values_europa = np.vstack(propagated_formal_errors_rsw_dict_europa.values())
values_jup = np.vstack(propagated_formal_errors_rsw_dict_jup.values())
te = time_europa/31536000
tj = time_jup/31536000
#%%
""""
Plot the propagated uncertainties  
"""

plt.figure(figsize=(9,5))
plt.plot(te,values_europa[:,0], label = 'R', color = 'red')
plt.plot(te,values_europa[:,1], label = 'S', color = 'green')
plt.plot(te,values_europa[:,2], label = 'W', color = 'blue')
plt.plot(observation_times_europa/31536000, 100,'o')
plt.ylim([10e1,10e4])
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Europa")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time [years after J2000]')
plt.legend()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(tj,values_jup[:,0], label = 'R', color = 'salmon')
plt.plot(tj,values_jup[:,1], label = 'S', color = 'orange')
plt.plot(tj,values_jup[:,2], label = 'W', color = 'cornflowerblue')
plt.yscale("log")
plt.ylim([10e1,10e4])
plt.grid(True, which="both", ls="--")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Jupiter")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time [years after J2000]')
plt.legend()
plt.show()
#%%
""""
Export Covariance Matrix to use as input 
"""

covariance_matrix = np.savetxt("/Users/gianmarcobroilo/Desktop/ThesisResults/vlbi-corrected/IO/output_covariance_io_jup_nominal.dat",pod_output.covariance)


#%%

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
fig.suptitle('Propagated uncertainties in Right Ascension and Declination of Europa')


axs[0].plot(te,alpha, color = 'black')
axs[0].set_ylabel('Right Ascension [rad]')
axs[0].set_yscale("log")
axs[0].axvline(x = observation_times_europa/31536000,color="magenta")
axs[1].plot(te,dec, color = 'black')
axs[1].set_ylabel('Declination [rad]')
axs[1].axvline(x = observation_times_europa/31536000,color="magenta")
axs[1].set_xlabel('Time [years after J2000]')
axs[1].set_yscale("log")
plt.show()
