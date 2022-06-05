from sora import Body, Star, LightCurve, Occultation, Observer
from sora.prediction import prediction
from sora.extra import draw_ellipse

import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time


""""
DEFINE THE BODY 
"""
# Define Callisto
callisto = Body(name='Callisto', ephem='horizons')

# Define the occultation
occ = Occultation(star='02 15 25.69246 +12 24 11.11970', body=callisto, time='2024-01-15 06:12:15.620')

# Define an Observeve
obs = Observer(name='San Diego, USA', lat='32 43 29', lon='-117 09 41', height=42)


# Create a generic light curve based on the Closest approach time
lightcurve = LightCurve(name='San Diego, USA', initial_time='2024-01-15 05:12', end_time='2024-01-15 07:12',immersion='2024-01-15 06:17:43.1',immersion_err=0.7,
                    emersion ='2024-01-15 06:24:38.5',emersion_err=0.7)


# Now we can add this observer and its light curve to our Occultation Object.
occ.chords.add_chord(name='San Diego, USA', observer=obs, lightcurve=lightcurve)
chord = occ.chords['San Diego, USA']

# Let's visualize the occultation geometry with respect to Callisto
import matplotlib.image as mpimg
img = mpimg.imread('/Users/gianmarcobroilo/Desktop/Tudat/SORA/callisto.png')
imgplot = plt.imshow(img,extent=[-3710.3, 3710.3, -2410.3, 2410.3])

draw_ellipse(equatorial_radius=2410.3)
#occ.chords.plot_chords()
chord.plot_chord(segment='positive', color='blue',label='San Diego,CA')
chord.plot_chord(segment='error', color='red')

plt.title('San Diego, USA')
plt.axis('square')
plt.xlim(-3000, +3000)
plt.ylim(-3000, +3000)
plt.legend()
plt.show()

# San Diego will observe a postive chord (assuming tha the JPL ephemeris is perfect)
# We can determine the exact expected times for this stations

occ.chords.get_theoretical_times(equatorial_radius=2410.3, step=0.1)

# The expected times
ti = '2024-01-15 06:17:43.1' # expected immersion time
te = '2024-01-15 06:24:38.5' # expected emersion tine
occ.body.apparent_magnitude(time='2024-01-15 06:17')

# Calculate the magnitude drop of the expected light curve

print('Occulted Star magnitude', occ.star.mag['V'])
print('Occulting object magnitude', occ.body.apparent_magnitude(time='2024-01-15 06:17'))

lightcurve.calc_magnitude_drop(mag_star=occ.star.mag['V'], mag_obj=occ.body.apparent_magnitude(time='2024-01-15 06:17'))

print(lightcurve.mag_drop)
print(lightcurve.bottom_flux)

# Generate a generic array for the times and fluxes
tinit = (Time('2024-01-15 06:17') - Time('2024-01-15 00:00')).sec - 30*60 # time in seconds relative to 00:00 UTC
tend  = (Time('2024-01-15 06:24') - Time('2024-01-15 00:00')).sec + 30*60 # time in seconds relative to 00:00 UTC
dt = 0.3 # Temporal resolution

times = np.arange(tinit, tend, dt)
fluxes = np.ones(len(times))

# Add arrays to LightCurve object
lightcurve.set_flux(time=times, flux=fluxes, exptime=dt, tref=Time('2024-01-15 00:00'))

# Create the modelled light curve
lightcurve.occ_model(immersion_time=(Time('2024-01-15 06:17:43.07') - Time('2024-01-15 00:00')).sec,
                     emersion_time=(Time('2024-01-15 06:24:38.55') - Time('2024-01-15 00:00')).sec,
                     opacity=1.0, flux_max=1.0, flux_min=0.926, mask=np.repeat(True,len(times)))

# Plot the modelled light curve
plt.figure(figsize=[15, 5])
plt.title('Light Curve')
plt.xlabel('Time [s]')
plt.ylabel('Relative Flux')
plt.plot(lightcurve.time, lightcurve.model, 'r.-',label='Model')
plt.legend()
plt.show()


# Add noise to the data
sigma = 0.02
noise = np.random.normal(0, sigma, len(lightcurve.model))

plt.figure(figsize=[15, 5])
plt.title('Occultation by Callisto on 2024-01-15')
plt.xlabel('Time [s]')
plt.ylabel('Relative Flux')
plt.plot(lightcurve.time, lightcurve.model + noise, 'k.-',label='Observation')
plt.legend()
plt.show()

immersion_time=(Time('2024-01-15 06:17:43.07') - Time('2024-01-15 00:00')).sec
emersion_time=(Time('2024-01-15 06:24:38.55') - Time('2024-01-15 00:00')).sec
opacity=1.0
mask=np.repeat(True,len(times))

plt.plot(lightcurve.time,lightcurve.model + noise,'k.-',zorder=1,label='Observation')
plt.plot(lightcurve.time, lightcurve.model, 'b.-',label='Model')
plt.axvline(immersion_time,color='r',linestyle='-',label='Immersion')
plt.axvline(emersion_time,color='r',linestyle='--',label='Emersion')
plt.title('Occultation by Callisto on 2024-01-15')
plt.xlabel('Time [s]')
plt.ylabel('Relative Flux')
#plt.xlim(22200,23500)
plt.legend()
plt.show()

lightcurve.plot_lc()
lightcurve.plot_model()
plt.xlim(22663,22663.5)
plt.ylim(0.925,1.025)

plt.show()
