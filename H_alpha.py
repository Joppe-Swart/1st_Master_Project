import matplotlib
import numpy as np;
import matplotlib.pyplot as plt;
from astropy.io import fits;
import astropy.units as u;
from astropy.visualization import quantity_support
quantity_support()
import astropy.constants as const;
from astropy.cosmology import FlatLambdaCDM;
cosmo = FlatLambdaCDM(H0=70, Om0=0.3);

#filename = 'COSMOS_Ka_proposal_sources.fits' #input('Enter catalog filename: ');
#catalog = fits.open(filename);
#data = catalog[1].data;


FHa = 1.2289365257352287*10**(-16)*u.erg/u.s/(u.cm**2)*3.272975
FHb = 5.3215241146483076*10**(-17)*u.erg/u.s/(u.cm**2)*2.4367588
Ebv = 0.77*3*np.log10((FHa/FHb)*1/2.86)
Rv = 3.1
z = 1.44
x = 1/0.6563
y = x -1.82
a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4  + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
b = 1.413388*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
#b = -0.527*x**(1.61)
#a = 0.574*x**(1.61)
Av = Ebv*Rv#7.23*np.log10(FHa/FHb*1/2.87)
Alambda = Av*(a + b/2.87)
FHa0 = FHa*10**(0.4*Alambda)
D_l = cosmo.luminosity_distance(z)
LHa = (FHa0*4*np.pi*D_l**2).to(u.erg/u.s)
SFR = 7.9*10**(-42)*LHa
print(Av,Alambda)
print(FHa/FHb)
print(x,a, b)
print(SFR)
