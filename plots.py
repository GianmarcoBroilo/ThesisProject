## Context
"""
This code performs a covariance analysis of the state of Callisto and Jupiter. The covariance matrix will be propagated
to see the behavior of the uncertainties of the two bodies in the RSW reference frame. The simulated observables are for
Callisto a predicted stellar occultation in 2024 (using SORA) and for Jupiter a combination of VLBI and range observables
simulated once every JUNO orbit, that is 53.4 days.
INPUT
Data type: stellar occultation of Callisto, VLBI and RANGE of Jupiter
A priori cov: uncertainties in the state of Callisto and Jupiter in RSW frame
parameters: initial state of Callisto, initial state of Jupiter
OUTPUT
cov: uncertainty and correlation of estimated parameters for both Callisto and Jupiter
"""
#%%


# Load standard modules
import numpy as np
from numpy import linalg as lalg
from matplotlib import pyplot as plt
import datetime
# Load tudatpy modules

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import  estimation_setup
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
from tudatpy.kernel.astro import time_conversion



# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs START: 2023-01-01 END: 2027-01-01
calendar_start = datetime.datetime(2024,1,1)
simulation_start_epoch = time_conversion.calendar_date_to_julian_day_since_epoch(calendar_start)*constants.JULIAN_DAY
simulation_end_epoch = simulation_start_epoch + 30*constants.JULIAN_DAY


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Callisto", "Jupiter","Sun","Saturn","Ganymede"]

time_step = 1500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "ECLIPJ2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)


# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


"""
Propagation setup
"""

# Define bodies that are propagated
bodies_to_propagate = ["Callisto","Jupiter"]

# Define central bodies of propagation
central_bodies = []
for body_name in bodies_to_propagate:
    if body_name == "Callisto":
        central_bodies.append("Jupiter")
    elif body_name == "Jupiter":
        central_bodies.append("Sun")



# Create the acceleration model
acceleration_settings_cal = dict(
    Jupiter = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        8,0,2,2)],
    Sun = [propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_jup = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings = {"Callisto": acceleration_settings_cal, "Jupiter": acceleration_settings_jup}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,acceleration_settings,bodies_to_propagate,central_bodies
)

# Define the initial state
"""
The initial state of Callisto and Jupiter that will be propagated is now defined 
"""

# Set the initial state of Io and Jupiter
initial_state = propagation.get_initial_state_of_bodies(
    bodies_to_propagate=bodies_to_propagate,
    central_bodies=central_bodies,
    body_system=bodies,
    initial_time=simulation_start_epoch,
)

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

#create dependent variables
dependent_variables_to_save = [
    propagation_setup.dependent_variable.relative_position("Callisto","Earth")
]

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    termination_condition,
    output_variables = dependent_variables_to_save
)

# Define the observations for Callisto stellar occultation
stellar_occ = datetime.datetime(2024,1,15)
stellar_occ = time_conversion.calendar_date_to_julian_day_since_epoch(stellar_occ)*constants.JULIAN_DAY

state_callisto = propagation.get_state_of_bodies(
    bodies_to_propagate,
    central_bodies,
    bodies,
    stellar_occ

)
"""
Propagate the dynamics of Jupiter and Callisto
"""
#Setup paramaters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)

# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# Create numerical integrator settings.
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    simulation_start_epoch, 900.0, propagation_setup.integrator.rkf_78, 900.0, 900.0, 1.0, 1.0
)
# Instantiate the dynamics simulator.
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings, print_dependent_variable_data=True)

# Extract the resulting state history
from tudatpy.util import result2array
states =  dynamics_simulator.state_history
dep_var = dynamics_simulator.dependent_variable_history
dep_var_array = result2array(dep_var)
states_array = result2array(states)

vector_moon_earth = [dep_var_array[1344,1],dep_var_array[1344,2]]
unit_vector_moon_earth = vector_moon_earth/np.linalg.norm(vector_moon_earth)

vector_jup_moon = [states_array[1344,1], states_array[1344,2]]
unit_vector_jup_moon = vector_jup_moon/np.linalg.norm(vector_jup_moon)

dot_product = np.dot(unit_vector_moon_earth,unit_vector_jup_moon)
angle = np.arccos([dot_product])
angle_deg = angle*180/np.pi

plt.figure(figsize=(7,7))

plt.title(f'Trajectory of Callisto w.r.t Jupiter')
plt.scatter(0, 0, zorder = 1, marker='.', label="Jupiter", color='darkorange',s=600,linewidth=1, edgecolor='k')
plt.scatter(state_callisto[1], state_callisto[2], zorder = 3, marker='.', label="Callisto", color='blue', s = 250,linewidth=1, edgecolor='k')
plt.plot(states_array[:, 1], states_array[:, 2], zorder = 2, label="orbit", color = 'grey', linestyle='-.')
plt.arrow(state_callisto[1], state_callisto[2], vector_moon_earth[0]/1e3, vector_moon_earth[1]/1e3, head_width=1e8, color='r')
plt.arrow(state_callisto[1], state_callisto[2], vector_jup_moon[0], vector_jup_moon[1], head_width=1e8, color='r')
plt.text(2.1e9,0.1e9,"to Earth",color = 'red')
plt.legend()
plt.xlim([-3e9,3e9])
plt.ylim([-3e9,3e9])
plt.grid(True, which="both", ls="--")
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.show()

