## Context
"""
This code performs a transformation of coordinates from RSW reference frame to Right ascension and Declination. The required parameters
as input are the following.
INPUT: matrix T that is the Jacobian of partials of the from d(alpha)/dx,dy,dz and d(delta)/dx,dy,dz. The partials are computed in MATLAB
      the propagated covariance matrix at each epoch propagated_covariance_dict 

OUTPUT: propagated_icrf and formal_errors expressed in terms of RA and Dec. The formal errors contain the uncertainty expressed in terms of radians. 
        The uncertainty are then expressed in milliarcseconds
        
"""


""""
Propagate RA and DEC of Jupiter  
"""


da_dr = dict()
dd_dr = dict()
Ta_dict = dict()
Td_dict = dict()
T = dict()
propagated_icrf_cal = dict()
formal_errors_icrf_cal = dict()
for epoch in list(propagated_covariance_dict):
    Ta_dict[epoch] = np.array([-states[epoch][3],states[epoch][2],0]).reshape(1,3)
    Td_dict[epoch] = np.array([-states[epoch][2]*states[epoch][4],-states[epoch][3]*states[epoch][4],states[epoch][2]**2+states[epoch][3]**2]).reshape(1,3)
    da_dr[epoch] = 1/(states[epoch][2]**2 + states[epoch][3]**2)*Ta_dict[epoch]
    dd_dr[epoch] = 1/(np.linalg.norm(states[epoch][2:4])**2*np.sqrt(states[epoch][2]**2+states[epoch][3]**2))*Td_dict[epoch]
    T[epoch] = np.vstack((da_dr[epoch],dd_dr[epoch]))
    propagated_icrf_cal[epoch] = lalg.multi_dot([T[epoch],propagated_covariance_dict[epoch][:3,:3],T[epoch].T])
    formal_errors_icrf_cal[epoch] = np.sqrt(np.diag(propagated_icrf_cal[epoch]))

values_icrf = np.vstack(formal_errors_icrf_cal.values())
alpha = values_icrf[:,0]
dec = values_icrf[:,1]


fig, axs = plt.subplots(2,figsize=(12, 6))
fig.suptitle('Propagated uncertainties in Right Ascension and Declination of Callisto')


axs[0].plot(tc,alpha, color = 'black')
axs[0].set_ylabel('Right Ascension [rad]')
axs[0].set_yscale("log")
axs[0].plot(observation_times_cal/31536000, 10e-2,'o')
axs[1].plot(tc,dec, color = 'black')
axs[1].plot(observation_times_cal/31536000, 10e-3,'o')
axs[1].set_ylabel('Declination [rad]')
axs[1].set_xlabel('Time [years after J2000]')
axs[1].set_yscale("log")
plt.show()

