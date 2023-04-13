from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
f = fits.open('3GHz_COSMOS_Whole_field.pb.fits')#input('Enter pb_corr.fits: '))
hdr = f[0].header

w = WCS(hdr)
if f[0].data.ndim == 4:
	f[0].data = f[0].data[0,0]
	w = w.dropaxis(3)
	w = w.dropaxis(2)
sky = w.wcs_world2pix(input('RA: '), input('DEC: '), 0)
print('The pixel coordinates of the source are: ', sky)
pbcor = f[0].data[int(sky[0]), int(sky[1])]
print(pbcor)  
f.close()


