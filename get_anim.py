import interlo as iso
import sys
from astropy import units as u

stars = iso.Starset(num_stars=3)
stars.get_isos(v_eject=15)
stars.integrate()