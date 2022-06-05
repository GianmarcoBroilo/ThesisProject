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

T = np.block([
    [-8.34313652508797e-14,-1.73092383777687e-14,1.34742402731453e-12],
    [-2.74811677043994e-13,1.32460556046833e-12,0]
])

propagated_icrf_jup = dict()
formal_errors_icrf_jup = dict()
for epoch in list(propagated_covariance_dict):
    propagated_icrf_jup[epoch] = lalg.multi_dot([T,propagated_covariance_dict[epoch][6:9,6:9],T.T])
    formal_errors_icrf_jup[epoch] = np.sqrt(np.diag(propagated_icrf_jup[epoch]))

values_icrf = np.vstack(formal_errors_icrf_jup.values())
alpha = values_icrf[:,0]
dec = values_icrf[:,1]

alpha_marcsec = alpha*206264806.71915
delta_marcsec = dec*206264806.71915

fig, axs = plt.subplots(2,figsize=(12, 6))
fig.suptitle('Propagated uncertainties in Right Ascension and Declination of Jupiter')


axs[0].plot(tj,alpha_marcsec,'o', color = 'black')
axs[0].set_ylabel('Right Ascension [mas]')

axs[1].plot(time_cal/31536000,delta_marcsec,'o', color = 'black')
axs[1].set_ylabel('Declination [mas]')
axs[1].set_xlabel('Time [years after J2000]')
plt.show()
