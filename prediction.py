from sora import Body
from sora.ephem import EphemKernel
from sora.prediction import prediction
from sora import EphemHorizons

europa = Body(name='Europa', ephem = 'horizons')
ephem = EphemHorizons('Europa', spkid= '502')
europa.ephem = ephem

pred = prediction(body=europa, time_beg='2030-03-01',time_end='2030-06-01',mag_lim=12)

pred.pprint_all()
