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
restfreq = 1.4#*u.GHz

from matplotlib import cm
from matplotlib.colors import ListedColormap

palette = cm.get_cmap("Vega20", 20)

plt.style.use('science')

# Import the file
filename = 'CXS-2_Catalog.fits' #input('Enter catalog filename: ');
catalog = fits.open(filename);
data = catalog[1].data;

# Make the empty lists we want to put the data in
frequency = [];
flux = [];
E_flux = [];
alpha = [];
E_alpha = [];



# Order to merge CXS + master + COSMOS L + GMRT610 + GMRT325 + 9 GHZ + 10 GHz 11 GHZ + 2 GHZ + 4 GHz
# Specify the conditions when to add a flux or a frequency. 
if 'Peak_Flux_325MHz' in catalog[1].columns.names:
	flux325MHz = data.field('Peak_Flux_325MHz')*u.Jy;
	E_flux325MHz = np.sqrt(data.field('E_Peak_Flux_325MHz')**2 + (0.05*data.field('Peak_Flux_325MHz')**2))*u.Jy;
	E_flux.append(float(E_flux325MHz.value));
	flux.append(float(flux325MHz.value));
	frequency.append(0.325);
	
if 'Peak_Flux_610MHz' in catalog[1].columns.names:
	flux610MHz = data.field('Peak_Flux_610MHz')*u.Jy;
	E_flux610MHz = np.sqrt(data.field('E_Peak_Flux610MHz')**2 + (0.05*data.field('Peak_Flux_610MHz')**2))*u.Jy;
	E_flux.append(float(E_flux610MHz.value));
	flux.append(float(flux610MHz.value));
	frequency.append(0.610);


if 'Total_Flux_1.4GHz' in catalog[1].columns.names:
	flux1_4 = data.field('Total_Flux_1.4GHz')*u.Jy;
	E_flux1_4 = np.sqrt(data.field('E_Total_Flux_1.4GHz')**2 + (0.05*data.field('Total_Flux_1.4GHz')**2))*u.Jy;
	E_flux.append(float(E_flux1_4.value));
	flux.append(float(flux1_4.value));
	frequency.append(1.4);
	
#if 'Peak_Flux_2.5GHz' in catalog2[1].columns.names:
#	flux2_5 = data2.field('Peak_Flux_2.5GHz')/0.916*u.Jy;
#	E_flux2_5 = data2.field('E_Peak_Flux_2.5GHz')/0.916*u.Jy;
#	E_flux.append(float(E_flux2_5.value));
#3	flux.append(float(flux2_5.value));
#	frequency.append(2.5);
#
if 'Peak_Flux_3GHz' in catalog[1].columns.names:
	flux3 = (data.field('Peak_Flux_3GHz')*u.uJy).to(u.Jy);
	E_flux3 = (np.sqrt(data.field('E_Peak_Flux_3GHz')**2 + (0.05*data.field('Peak_Flux_3GHz')**2))*u.uJy).to(u.Jy);
	E_flux.append(float(E_flux3.value));
	flux.append(float(flux3.value));
	frequency.append(3);
	
#if 'Peak_Flux_3.5GHz' in catalog2[1].columns.names:
#	flux2_3_5 = data2.field('Peak_Flux_3.5GHz')/0.825*u.Jy;
#	E_flux2_3_5 = data2.field('E_Peak_Flux_3.5GHz')/0.825*u.Jy;
#	E_flux.append(float(E_flux2_3_5.value));
#	flux.append(float(flux2_3_5.value));
#	frequency.append(3.5);
	
#if 'Peak_Flux_9GHz' in catalog[1].columns.names:
#	flux9 = data.field('Peak_Flux_9GHz')/0.556*u.Jy;
#	E_flux9 = data.field('E_Peak_Flux_9GHz')/0.556*u.Jy;
#	E_flux.append(float(E_flux9.value));
#	flux.append(float(flux9.value))
#	frequency.append(9);
#
	
if 'Peak_Flux_10GHz' in catalog[1].columns.names and np.abs(data.field('Peak_Flux_10GHz')) != 99.0:
	flux10 = (data.field('Peak_Flux_10GHz')*u.uJy).to(u.Jy);
	E_flux10 = (np.sqrt(data.field('E_Peak_Flux_10GHz')**2 + (0.05*data.field('Peak_Flux_10GHz')**2))*u.uJy).to(u.Jy);
	flux.append(float(flux10.value));
	E_flux.append(float(E_flux10.value));
	frequency.append(10);
	
#if 'Peak_Flux_11GHz' in catalog[1].columns.names:
#	flux11 = data.field('Peak_Flux_11GHz')/0.403*u.Jy;
#	E_flux11 = data.field('E_Peak_Flux_11GHz')/0.403*u.Jy;
#	E_flux.append(float(E_flux11.value));
#	flux.append(float(flux11.value));
#	frequency.append(11);

if 'Peak_Flux_22GHz' in catalog[1].columns.names:
	flux22 = data.field('Peak_Flux_22GHz')*u.Jy;
	E_flux22 = (np.sqrt(data.field('E_Peak_Flux_22GHz')**2 + (0.05*data.field('Peak_Flux_22GHz')**2))*u.Jy);
	flux.append(float(flux22.value));
	E_flux.append(float(E_flux22.value));
	frequency.append(22);

flux = ((np.array(flux)*u.Jy).to(u.uJy)).value
frequency_obs = (np.array(frequency)*u.GHz).value
E_flux = ((np.array(E_flux)*u.Jy).to(u.uJy)).value
frequency = np.linspace(1.4, 22, num=30)


# Determine  the spectral index and uncertainty (325-610, 610-1.4,  1.4-3, 3-10, 10-22)
#for i in range(len(flux)-1):
#	alpha.append(np.log10(float(flux[i])/float(flux[i+1]))/np.log10(float(frequency[i])/float(frequency[i+1])));
#	E_alpha.append(np.sqrt((1/(float(flux[i])*np.log(10)*np.log10(float(frequency[i])/float(frequency[i+1]))))**2*float(E_flux[i])**2+(1/(float(flux[i+1])*np.log(10)*np.log10(float(frequency[i])/float(frequency[i+1]))))**2*float(E_flux[i+1])**2));


# Fitting the Different models
import scipy.stats as stats
import emcee
import corner

#We start with a simple powerlaw S_nu = S*nu**(-alpha)
#First we use a Chi^2 test to find the starting points for MCMC
#def Chi_pl(S_pl, a_pl):
 #    Chi_square = np.empty(shape=[0,3]); # Make an empty 4D array shape = [Chi^2, a, b, c]
  #   # Loop throug all the possible values
   #  for j in range(len(S_pl)):
    #     for i in range(len(a_pl)):
     #    	#for l in range(len(c)):
      #   		Square = 0
       #  		for n in range(len(flux)):
        #        		model_pl = S_pl[j]*(frequency[n])**(-a_pl[i]);
         #               	Square += ((flux[n]-model_pl)/(E_flux[n]))**2;
          #      	Chi_square = np.append(Chi_square, [[Square, S_pl[j], a_pl[i]]], axis=0);
     #return Chi_square

#Define the range you expect your parameters to be in between
#S_pl = np.linspace(20, 200, 10);
#a_pl = np.linspace(0.5,0.6, 10);
#c = np.linspace(0, 1, 20);

#Define the 5D matrix so it is more accesible
#Chi2_pl = Chi_pl(S_pl, a_pl);

#Loop over all the values to find the matrix element with the smallest Chi^2 and their parameters

#for i in range(len(Chi2_pl[:,0])):
#	if Chi2_pl[i,0] == min(Chi2_pl[:,0]):#
#		Chi2_pl_min = Chi2_pl[i];
        	#print(r'Smallest Chi2_pl($chi^2, S_{pl}, \alpha$) = ', Chi2_pl[i])
        	     	        
# Define the parameters needed for mcmc
# Define the best parameters from Chi2-test
S_pl_guess = 200#Chi2_pl_min[1]
a_pl_guess = 0.5#Chi2_pl_min[2]
#c_guess = Chi_min[3]

x = frequency_obs # Lijst met frequenties in GHz
y_obs = flux # data punten 
dy = E_flux # error in de data

def lnL_pl(theta, x, y, yerr):
    S,a= theta
    model_pl = S*(x/restfreq)**(-a)#*u.uJy
    inv_sigma2 = 1.0/(np.power(yerr,2))
    
    return -0.5*(np.sum((y-model_pl)**2*inv_sigma2))

def lnprior_pl(theta):
	S, a = theta
   	if 0 < S < 500 and 0. < a < 1:
        	return 0.0
    	return -np.inf

def lnprob_pl(theta, x, y, yerr):
    lp_pl = lnprior_pl(theta)
    if not np.isfinite(lp_pl):
        return -np.inf
    return lp_pl + lnL_pl(theta, x, y, yerr)
    
ndim_pl, nwalkers = 2, 500
theta_pl_guess = np.array([S_pl_guess, a_pl_guess])
pos_pl = [theta_pl_guess + 1e-4*np.random.randn(ndim_pl) for i in range(nwalkers)]
sampler_pl = emcee.EnsembleSampler(nwalkers, ndim_pl, lnprob_pl, args=(x, y_obs, dy))
tmp = sampler_pl.run_mcmc(pos_pl, 650)

fig, axes = plt.subplots(ncols=1, nrows=2)
fig.set_size_inches(12,12)
axes[0].plot(sampler_pl.chain[:, :, 0].transpose(), color='black', alpha=0.3)
axes[0].set_ylabel(r'$S_{pl}$')
axes[0].axvline(150, ls='dashed', color='red')
axes[1].plot(sampler_pl.chain[:, :, 1].transpose(), color='black', alpha=0.3)
axes[1].set_ylabel(r'$\alpha$')
axes[1].axvline(150, ls='dashed', color='red')
#axes[2].plot(sampler.chain[:, :, 2].transpose(), color='black', alpha=0.3)
#axes[2].set_ylabel('$c$')
#axes[2].axvline(400, ls='dashed', color='red')
fig.savefig('chain_pl.pdf', format='pdf')
plt.close()

samples_pl = sampler_pl.chain[:, 150:, :].reshape((-1, 2))
#print(samples.shape)

fig = corner.corner(samples_pl, labels=[r"$S_{pl}$", r"$\alpha$"], quantiles=[0.16, 0.50, 0.84], show_titles=True)#truths=[S_pl_guess, a_pl_guess],)
fig.savefig('corner_pl.pdf', format='pdf')
plt.close()

median_S_pl = np.percentile(samples_pl[:, 0], 50.0)
median_a_pl = np.percentile(samples_pl[:, 1], 50.0)

p16_S_pl = np.percentile(samples_pl[:, 0], 16)
p16_a_pl = np.percentile(samples_pl[:, 1], 16)

p84_S_pl = np.percentile(samples_pl[:, 0], 84)
p84_a_pl = np.percentile(samples_pl[:, 1], 84)

sigma_S_pl = 0.5*(p84_S_pl-p16_S_pl)
sigma_a_pl = 0.5*(p84_a_pl-p16_a_pl)

#print('chi square powerlaw = ', Chi_pl([median_S_pl], [median_a_pl])/(len(frequency)-2))


MCMC_pl=np.empty(shape=[len(samples_pl[:,0]), 0])
for i in range(len(frequency)):
	MCMC_pl = np.append(MCMC_pl, (samples_pl[:,0]*np.power(frequency[i]/restfreq, -samples_pl[:,1])).reshape(len(samples_pl[:,0]), 1), axis = 1)

pl_16 = []
pl_84 = []
for i in range(len(frequency)):
	pl_16.append(np.percentile(np.sort(MCMC_pl[:,i]),16))
	pl_84.append(np.percentile(np.sort(MCMC_pl[:,i]),84))
	
pl_16 = np.array(pl_16)*u.uJy
pl_84 = np.array(pl_84)*u.uJy
	
def red_chi_pl(S, a):
	chi_pl = 0
	model = (S*(frequency_obs/restfreq)**(-a))#*u.uJy
	for i in range(len(frequency_obs)):
		chi_pl += ((flux[i]-model[i])/E_flux[i])**2
	red_chi_pl = chi_pl/(len(frequency_obs)-2)
	return red_chi_pl



#----------------------------------------------------------------------------------------------------------------------------------------------	
#Next up is the non thermal synchrotron and thermal free free emission 
#First we use a Chi^2 test to find the starting points for MCMC

#def Chi_th(S_nu0, f_th, a_NT):
#     Chi_square = np.empty(shape=[0,4]); # Make an empty 4D array shape = [Chi^2, S_nu0, f_th, a_FF]
     # Loop throug all the possible values
#     for j in range(len(S_nu0)):
#         for i in range(len(f_th)):
#         	for l in range(len(a_NT)):
#         		Square = 0
#         		for n in range(len(flux)):
#                		model_th = ((1-f_th[i])*S_nu0[j]*(frequency[n]/restfreq)**(-a_NT[l]) + f_th[i]*S_nu0[j]*(frequency[n]/restfreq)**(-0.1))*u.uJy
#                        	Square += (((flux[n]-model_th)/(E_flux[n]))**2).value;
#                	Chi_square = np.append(Chi_square, [[Square, S_nu0[j], f_th[i], a_NT[l]]], axis=0);
#     return Chi_square

#Define the range you expect your parameters to be in between
#S_nu0 = np.linspace(20, 200, 10);
#f_th = np.linspace(0,1, 10);
#a_NT = np.linspace(0, 1, 10);

#Define the 5D matrix so it is more accesible
#Chi2_th = Chi_th(S_nu0, f_th, a_NT);

#Loop over all the values to find the matrix element with the smallest Chi^2 and their parameters

#for i in range(len(Chi2_th[:,0])):
#	if Chi2_th[i,0] == min(Chi2_th[:,0]):
#		Chi2_th_min = Chi2_th[i];
        	#print(r'Smallest Chi2_th = ', Chi2_th[i])
        	     	        
# Define the parameters needed for mcmc
# Define the best parameters from Chi2-test
S_nu0_guess = 200 #Chi2_th_min[1]
f_th_guess = 0 #Chi2_th_min[2]
a_NT_guess = 0.5 #Chi2_th_min[3]

x = frequency_obs # Lijst met frequenties in GHz
y_obs = flux # data punten 
dy = E_flux # error in de data

def lnL_th(theta, x, y, yerr):
    S, f, a= theta
    model_th = ((1-f)*S*(x/restfreq)**(-a)+f*S*(x/restfreq)**(-0.1))#*u.uJy
    inv_sigma2 = 1.0/(np.power(yerr,2))
    
    return -0.5*(np.sum((y-model_th)**2*inv_sigma2))

def lnprior_th(theta):
	S, f, a = theta
    	if 0 < S < 500 and -0.5 < f < 1 and 0. < a < 1:
        	return 0.0
    	return -np.inf

def lnprob_th(theta, x, y, yerr):
    lp_th = lnprior_th(theta)
    if not np.isfinite(lp_th):
        return -np.inf
    return lp_th + lnL_th(theta, x, y, yerr)
    
ndim_th, nwalkers = 3, 500
theta_th_guess = np.array([S_nu0_guess, f_th_guess, a_NT_guess])
pos_th = [theta_th_guess + 1e-4*np.random.randn(ndim_th) for i in range(nwalkers)]
sampler_th = emcee.EnsembleSampler(nwalkers, ndim_th, lnprob_th, args=(x, y_obs, dy))
tmp = sampler_th.run_mcmc(pos_th, 700)

fig, axes = plt.subplots(ncols=1, nrows=3)
fig.set_size_inches(12,12)
axes[0].plot(sampler_th.chain[:, :, 0].transpose(), color='black', alpha=0.3)
axes[0].set_ylabel(r'$S_{\nu_0}$')
axes[0].axvline(200, ls='dashed', color='red')
axes[1].plot(sampler_th.chain[:, :, 1].transpose(), color='black', alpha=0.3)
axes[1].set_ylabel(r'$f^{th}$')
axes[1].axvline(200, ls='dashed', color='red')
axes[2].plot(sampler_th.chain[:, :, 2].transpose(), color='black', alpha=0.3)
axes[2].set_ylabel(r'$\alpha^{NT}$')
axes[2].axvline(200, ls='dashed', color='red')
fig.savefig('chain_th.pdf', format='pdf')
plt.close()

samples_th = sampler_th.chain[:, 200:, :].reshape((-1, 3))


fig = corner.corner(samples_th, labels=[r"$S_{\nu_0}$", r"$f^{th}$", r"$\alpha^{NT}$"],quantiles=[0.16, 0.50, 0.84], show_titles=True)#truths=[S_nu0_guess, f_th_guess, a_NT_guess])
fig.savefig('corner_th.pdf', format='pdf')
plt.close()

median_S_nu0 = np.percentile(samples_th[:, 0], 50.0)
median_f_th = np.percentile(samples_th[:, 1], 50.0)
median_a_NT = np.percentile(samples_th[:, 2], 50.0)

p16_S_nu0 = np.percentile(samples_th[:, 0], 16)
p16_f_th = np.percentile(samples_th[:, 1], 16)
p16_a_NT = np.percentile(samples_th[:, 2], 16)

p84_S_nu0 = np.percentile(samples_th[:, 0], 84)
p84_f_th = np.percentile(samples_th[:, 1], 84)
p84_a_NT = np.percentile(samples_th[:, 2], 84)

sigma_S_nu0 = 0.5*(p84_S_nu0-p16_S_nu0)
sigma_f_th = 0.5*(p84_f_th-p16_f_th)
sigma_a_NT = 0.5*(p84_a_NT-p16_a_NT)

	
MCMC_th=np.empty(shape=[len(samples_th[:,0]), 0])
for i in range(len(frequency)):
	MCMC_th = np.append(MCMC_th, ((1-samples_th[:,1])*samples_th[:,0]*np.power(frequency[i]/restfreq,-samples_th[:,2])+samples_th[:,1]*samples_th[:,0]*np.power(frequency[i]/restfreq,-0.1)).reshape(len(samples_th[:,0]), 1), axis = 1)

th_16 = []
th_84 = []
for i in range(len(frequency)):
	th_16.append(np.percentile(np.sort(MCMC_th[:,i]),16))
	th_84.append(np.percentile(np.sort(MCMC_th[:,i]),84))
th_16 = np.array(th_16)*u.uJy
th_84 = np.array(th_84)*u.uJy

def red_chi_th(S, f, a):
	chi_th = 0
	model = ((1-f)*S*(frequency_obs/restfreq)**(-a)+f*S*(frequency_obs/restfreq)**(-0.1))#*u.uJy
	for i in range(len(frequency_obs)):
		chi_th += ((flux[i]-model[i])/E_flux[i])**2
	red_chi_th = chi_th/(len(frequency_obs)-3)
	return red_chi_th


#-----------------------------------------------------------------------------------------------------------------------------------------
#Synchrotron aging Model

S_nu_SA_guess = 200
a_SA_guess = 0.5
nu_b_SA_guess = 15


x = frequency_obs
y_obs = flux
dy = E_flux

def lnL_SA(theta, x, y, yerr):
    S_nu, a, nu_b = theta
    model_SA = S_nu*(((x/restfreq)**(-a))/(1+(x/(nu_b))**(0.5)))#*u.uJy
    inv_sigma2 = 1.0/(np.power(yerr,2))
    
    return -0.5*(np.sum((y-model_SA)**2*inv_sigma2))

def lnprior_SA(theta):
	S_nu, a, nu_b = theta	
	if 0 < S_nu < 500 and 0 < a < 1 and 1 < nu_b < 22:
        	return 0.0
    	return -np.inf

def lnprob_SA(theta, x, y, yerr):
    lp_SA = lnprior_SA(theta)
    if not np.isfinite(lp_SA):
        return -np.inf
    return lp_SA + lnL_SA(theta, x, y, yerr)
   
ndim_SA, nwalkers = 3, 500
theta_SA_guess = np.array([S_nu_SA_guess, a_SA_guess, nu_b_SA_guess])
pos_SA = [theta_SA_guess + 1e-04*np.random.randn(ndim_SA) for i in range(nwalkers)]
sampler_SA = emcee.EnsembleSampler(nwalkers, ndim_SA, lnprob_SA, args=(x, y_obs, dy))
tmp = sampler_SA.run_mcmc(pos_SA, 600)

fig, axes = plt.subplots(ncols=1, nrows=3)
fig.set_size_inches(12,12)
axes[0].plot(sampler_SA.chain[:, :, 0].transpose(), color='black', alpha=0.3)
axes[0].set_ylabel(r'$S_{\nu}$')
axes[0].axvline(100, ls='dashed', color='red')
axes[1].plot(sampler_SA.chain[:, :, 1].transpose(), color='black', alpha=0.3)
axes[1].set_ylabel(r'$\alpha$')
axes[1].axvline(100, ls='dashed', color='red')
axes[2].plot(sampler_SA.chain[:, :, 2].transpose(), color='black', alpha=0.3)
axes[2].set_ylabel(r'$\nu_b$')
axes[2].axvline(100, ls='dashed', color='red')
fig.savefig('chain_SA.pdf', format='pdf')
plt.close()

samples_SA = sampler_SA.chain[:, 100:, :].reshape((-1, 3))


fig = corner.corner(samples_SA, labels=[r"$S_{\nu}$", r"$\alpha$", r"$\nu_b$"],quantiles=[0.16, 0.50, 0.84], show_titles=True)#truths=[S_nu0_guess, f_th_guess, a_NT_guess])
fig.savefig('corner_SA.pdf', format='pdf')
plt.close()

median_S_nu_SA = np.percentile(samples_SA[:, 0], 50.0)
median_a_SA = np.percentile(samples_SA[:, 1], 50.0)
median_nu_b_SA = np.percentile(samples_SA[:, 2], 50.0)

p16_S_nu_SA = np.percentile(samples_SA[:, 0], 16)
p16_a_SA = np.percentile(samples_SA[:, 1], 16)
p16_nu_b_SA = np.percentile(samples_SA[:, 2], 16)

p84_S_nu_SA = np.percentile(samples_SA[:, 0], 84)
p84_a_SA = np.percentile(samples_SA[:, 1], 84)
p84_nu_b_SA = np.percentile(samples_SA[:, 2], 84)


sigma_S_nu_SA = 0.5*(p84_S_nu_SA-p16_S_nu_SA)
sigma_a_SA = 0.5*(p84_a_SA-p16_a_SA)
sigma_nu_b_SA = 0.5*(p84_nu_b_SA-p16_nu_b_SA)

MCMC_SA=np.empty(shape=[len(samples_SA[:,0]), 0])
for i in range(len(frequency)):
	MCMC_SA = np.append(MCMC_SA, (samples_SA[:,0]*((frequency[i]/restfreq)**(-samples_SA[:,1]))/(1+(frequency[i]/(samples_SA[:,2]))**(0.5))).reshape(len(samples_SA[:,0]), 1), axis = 1)

SA_16 = []
SA_84 = []
for i in range(len(frequency)):
	SA_16.append(np.percentile(np.sort(MCMC_SA[:,i]),16))
	SA_84.append(np.percentile(np.sort(MCMC_SA[:,i]),84))
SA_16 = np.array(SA_16)*u.uJy
SA_84 = np.array(SA_84)*u.uJy



def red_chi_SA(S, a, nu_b):
	chi_SA = 0
	model = S*(((frequency_obs/restfreq)**(-a))/(1+(frequency_obs/(nu_b))**(0.5)))#*u.uJy
	for i in range(len(frequency_obs)):
		chi_SA += ((flux[i]-model[i])/E_flux[i])**2
	red_chi_SA = chi_SA/(len(frequency_obs)-3)
	return red_chi_SA



#------------------------------------------------------------------------------------------------------------------------------------------
#Synchrotron aging with FF


S_nu_SA_FF_guess = 200
a_SA_FF_guess = 0.5
nu_b_SA_FF_guess = 12
f_th_SA_FF_guess = 0.1

x = frequency_obs
y_obs = flux
dy = E_flux

def lnL_SA_FF(theta, x, y, yerr):
    S_nu, f, a = theta
    model_SA_FF = (f*S_nu*(x/restfreq)**(-0.1) + (1-f)*S_nu*(((x/restfreq)**(-a))/(1+(x/(nu_b_SA_FF_guess))**(0.5))))#*u.uJy
    inv_sigma2 = 1.0/(np.power(yerr,2))
    
    return -0.5*(np.sum((y-model_SA_FF)**2*inv_sigma2))

def lnprior_SA_FF(theta):
	S_nu, f, a= theta
    	if 0 < S_nu < 500 and -0.5 < f < 1 and 0 < a < 1: #and 1 < nu_b < 22:
        	return 0.0
    	return -np.inf

def lnprob_SA_FF(theta, x, y, yerr):
    lp_SA_FF = lnprior_SA_FF(theta)
    if not np.isfinite(lp_SA_FF):
        return -np.inf
    return lp_SA_FF + lnL_SA_FF(theta, x, y, yerr)
    
ndim_SA_FF, nwalkers = 3, 500
theta_SA_FF_guess = np.array([S_nu_SA_FF_guess, f_th_SA_FF_guess, a_SA_FF_guess])
pos_SA_FF = [theta_SA_FF_guess + 1e-4*np.random.randn(ndim_SA_FF) for i in range(nwalkers)]
sampler_SA_FF = emcee.EnsembleSampler(nwalkers, ndim_SA_FF, lnprob_SA_FF, args=(x, y_obs, dy))
tmp = sampler_SA_FF.run_mcmc(pos_SA_FF, 600)

fig, axes = plt.subplots(ncols=1, nrows=3)
fig.set_size_inches(12,12)
axes[0].plot(sampler_SA_FF.chain[:, :, 0].transpose(), color='black', alpha=0.3)
axes[0].set_ylabel(r'$S_{\nu}$')
axes[0].axvline(100, ls='dashed', color='red')
axes[1].plot(sampler_SA_FF.chain[:, :, 1].transpose(), color='black', alpha=0.3)
axes[1].set_ylabel(r'$f_{th}$')
axes[1].axvline(100, ls='dashed', color='red')
axes[2].plot(sampler_SA_FF.chain[:, :, 2].transpose(), color='black', alpha=0.3)
axes[2].set_ylabel(r'$\alpha$')
axes[2].axvline(100, ls='dashed', color='red')
#axes[3].plot(sampler_SA_FF.chain[:, :, 3].transpose(), color='black', alpha=0.3)
#axes[3].set_ylabel(r'$\nu_b$')
#axes[3].axvline(100, ls='dashed', color='red')
fig.savefig('chain_SA_FF.pdf', format='pdf')
plt.close()

samples_SA_FF = sampler_SA_FF.chain[:, 100:, :].reshape((-1, 3))


fig = corner.corner(samples_SA_FF, labels=[r"$S_{\nu}$", r"$f_{th}$", r"$\alpha$"],quantiles=[0.16, 0.50, 0.84], show_titles=True)#truths=[S_nu0_guess, f_th_guess, a_NT_guess])
fig.savefig('corner_SA_FF.pdf', format='pdf')
plt.close()

median_S_nu_SA_FF = np.percentile(samples_SA_FF[:, 0], 50.0)
median_a_SA_FF = np.percentile(samples_SA_FF[:, 2], 50.0)
median_f_th_SA_FF = np.percentile(samples_SA_FF[:, 1], 50.0)
#median_nu_b_SA_FF = np.percentile(samples_SA_FF[:, 3], 50.0)

p16_S_nu_SA_FF = np.percentile(samples_SA_FF[:, 0], 16)
p16_a_SA_FF = np.percentile(samples_SA_FF[:, 2], 16)
p16_f_th_SA_FF = np.percentile(samples_SA[:, 1], 16)
#p16_nu_b_SA_FF = np.percentile(samples_SA_FF[:, 3], 16)

p84_S_nu_SA_FF = np.percentile(samples_SA_FF[:, 0], 84)
p84_a_SA_FF = np.percentile(samples_SA_FF[:, 2], 84)
p84_f_th_SA_FF = np.percentile(samples_SA_FF[:, 1], 84)
#p84_nu_b_SA_FF = np.percentile(samples_SA_FF[:, 3], 84)


sigma_S_nu_SA_FF = 0.5*(p84_S_nu_SA_FF-p16_S_nu_SA_FF)
sigma_a_SA_FF = 0.5*(p84_a_SA_FF-p16_a_SA_FF)
sigma_f_th_SA_FF = 0.5*(p84_f_th_SA_FF-p16_f_th_SA_FF)
#sigma_nu_b_SA_FF = 0.5*(p84_nu_b_SA_FF-p16_nu_b_SA_FF)

MCMC_SA_FF=np.empty(shape=[len(samples_SA_FF[:,0]), 0])
for i in range(len(frequency)):
	MCMC_SA_FF = np.append(MCMC_SA_FF, (samples_SA_FF[:,0]*samples_SA_FF[:,1]*((frequency[i]/restfreq)**(-0.1))+(1-samples_SA_FF[:,1])*samples_SA_FF[:,0]*((frequency[i]/restfreq)**(-samples_SA_FF[:,2]))/(1+(frequency[i]/nu_b_SA_FF_guess)**(0.5))).reshape(len(samples_SA_FF[:,0]), 1), axis = 1)

SA_FF_16 = []
SA_FF_84 = []
for i in range(len(frequency)):
	SA_FF_16.append(np.percentile(np.sort(MCMC_SA_FF[:,i]),16))
	SA_FF_84.append(np.percentile(np.sort(MCMC_SA_FF[:,i]),84))
SA_FF_16 = np.array(SA_FF_16)*u.uJy
SA_FF_84 = np.array(SA_FF_84)*u.uJy

def red_chi_SA_FF(S, f, a):
	chi_SA_FF = 0
	model = (f*S*(frequency_obs/restfreq)**(-0.1) + (1-f)*S*(((frequency_obs/restfreq)**(-a))/(1+(frequency_obs/(nu_b_SA_FF_guess))**(0.5))))#*u.uJy
	for i in range(len(frequency_obs)):
		chi_SA_FF += ((flux[i]-model[i])/E_flux[i])**2
	red_chi_SA_FF = chi_SA_FF/(len(frequency_obs)-3)
	return red_chi_SA_FF


#---------------------------------------------------------------------------------------------------------------------------------------
#Synchrotron aging + Free-free with an exponential cutoff


S_nu_SA_FF_CO_guess = 200
a_SA_FF_CO_guess = 0.5
nu_b_SA_FF_CO_guess = 14
f_th_SA_FF_CO_guess = 0.1

x = frequency_obs
y_obs = flux
dy = E_flux

def lnL_SA_FF_CO(theta, x, y, yerr):
    S_nu, f, a = theta
    model_SA_FF_CO = (f*S_nu*(x/restfreq)**(-0.1) + (1-f)*S_nu*(x/restfreq)**(-a)*np.exp(-x/(nu_b_SA_FF_CO_guess)))#*u.uJy
    inv_sigma2 = 1.0/(np.power(yerr,2))
    
    return -0.5*(np.sum((y-model_SA_FF_CO)**2*inv_sigma2))

def lnprior_SA_FF_CO(theta):
	S_nu, f, a = theta
    	if 0 < S_nu < 500 and -0.5 < f < 1 and 0 < a < 1: #and 1 < nu_b < 22:
        	return 0.0
    	return -np.inf

def lnprob_SA_FF_CO(theta, x, y, yerr):
    lp_SA_FF_CO = lnprior_SA_FF_CO(theta)
    if not np.isfinite(lp_SA_FF_CO):
        return -np.inf
    return lp_SA_FF_CO + lnL_SA_FF_CO(theta, x, y, yerr)
    
ndim_SA_FF_CO, nwalkers = 3, 500
theta_SA_FF_CO_guess = np.array([S_nu_SA_FF_CO_guess, f_th_SA_FF_CO_guess, a_SA_FF_CO_guess])
pos_SA_FF_CO = [theta_SA_FF_CO_guess + 1e-4*np.random.randn(ndim_SA_FF_CO) for i in range(nwalkers)]
sampler_SA_FF_CO = emcee.EnsembleSampler(nwalkers, ndim_SA_FF_CO, lnprob_SA_FF_CO, args=(x, y_obs, dy))
tmp = sampler_SA_FF_CO.run_mcmc(pos_SA_FF_CO, 680)

fig, axes = plt.subplots(ncols=1, nrows=3)
fig.set_size_inches(12,12)
axes[0].plot(sampler_SA_FF_CO.chain[:, :, 0].transpose(), color='black', alpha=0.3)
axes[0].set_ylabel(r'$S_{\nu}$')
axes[0].axvline(180, ls='dashed', color='red')
axes[1].plot(sampler_SA_FF_CO.chain[:, :, 1].transpose(), color='black', alpha=0.3)
axes[1].set_ylabel(r'$f_{th}$')
axes[1].axvline(180, ls='dashed', color='red')
axes[2].plot(sampler_SA_FF_CO.chain[:, :, 2].transpose(), color='black', alpha=0.3)
axes[2].set_ylabel(r'$\alpha$')
axes[2].axvline(180, ls='dashed', color='red')
#axes[3].plot(sampler_SA_FF_CO.chain[:, :, 3].transpose(), color='black', alpha=0.3)
#axes[3].set_ylabel(r'$\nu_b$')
#axes[3].axvline(180, ls='dashed', color='red')
fig.savefig('chain_SA_FF_CO.pdf', format='pdf')
plt.close()

samples_SA_FF_CO = sampler_SA_FF_CO.chain[:, 180:, :].reshape((-1, 3))


fig = corner.corner(samples_SA_FF_CO, labels=[r"$S_{\nu}$", r"$f_{th}$", r"$\alpha$"],quantiles=[0.16, 0.50, 0.84], show_titles=True)#truths=[S_nu0_guess, f_th_guess, a_NT_guess])
fig.savefig('corner_SA_FF_CO.pdf', format='pdf')
plt.close()

median_S_nu_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 0], 50.0)
median_a_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 2], 50.0)
median_f_th_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 1], 50.0)
#median_nu_b_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 3], 50.0)

p16_S_nu_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 0], 16)
p16_a_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 2], 16)
p16_f_th_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 1], 16)
#p16_nu_b_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 3], 16)

p84_S_nu_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 0], 84)
p84_a_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 2], 84)
p84_f_th_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 1], 84)
#p84_nu_b_SA_FF_CO = np.percentile(samples_SA_FF_CO[:, 3], 84)


sigma_S_nu_SA_FF_CO = 0.5*(p84_S_nu_SA_FF_CO-p16_S_nu_SA_FF_CO)
sigma_a_SA_FF_CO = 0.5*(p84_a_SA_FF_CO-p16_a_SA_FF_CO)
sigma_f_th_SA_FF_CO = 0.5*(p84_f_th_SA_FF_CO-p16_f_th_SA_FF_CO)
#sigma_nu_b_SA_FF_CO = 0.5*(p84_nu_b_SA_FF_CO-p16_nu_b_SA_FF_CO)

MCMC_SA_FF_CO=np.empty(shape=[len(samples_SA_FF_CO[:,0]), 0])
for i in range(len(frequency)):
	MCMC_SA_FF_CO = np.append(MCMC_SA_FF_CO, (samples_SA_FF_CO[:,0]*samples_SA_FF_CO[:,1]*((frequency[i]/restfreq)**(-0.1))+(1-samples_SA_FF_CO[:,1])*samples_SA_FF_CO[:,0]*(frequency[i]/restfreq)**(-samples_SA_FF_CO[:,2])*np.exp(-frequency[i]/(nu_b_SA_FF_CO_guess))).reshape(len(samples_SA_FF_CO[:,0]), 1), axis = 1)

SA_FF_CO_16 = []
SA_FF_CO_84 = []
for i in range(len(frequency)):
	SA_FF_CO_16.append(np.percentile(np.sort(MCMC_SA_FF_CO[:,i]),16))
	SA_FF_CO_84.append(np.percentile(np.sort(MCMC_SA_FF_CO[:,i]),84))
SA_FF_CO_16 = np.array(SA_FF_CO_16)*u.uJy
SA_FF_CO_84 = np.array(SA_FF_CO_84)*u.uJy

def red_chi_SA_FF_CO(S, f, a):
	chi_SA_FF_CO = 0
	model = (f*S*(frequency_obs/restfreq)**(-0.1) + (1-f)*S*(frequency_obs/restfreq)**(-a)*np.exp(-frequency_obs/(nu_b_SA_FF_CO_guess)))#*u.uJy
	for i in range(len(frequency_obs)):
		chi_SA_FF_CO += ((flux[i]-model[i])/E_flux[i])**2
	red_chi_SA_FF_CO = chi_SA_FF_CO/(len(frequency_obs)-3)
	return red_chi_SA_FF_CO

#------------------------------------------------------------------------------------------------------------------------------------
#print chi^2 for the different models

chi_pl = red_chi_pl(median_S_pl, median_a_pl)
chi_th = red_chi_th(median_S_nu0, median_f_th, median_a_NT)
chi_SA = red_chi_SA(median_S_nu_SA, median_a_SA, median_nu_b_SA)
chi_SA_FF = red_chi_SA_FF(median_S_nu_SA_FF, median_f_th_SA_FF, median_a_SA_FF)
chi_SA_FF_CO = red_chi_SA_FF_CO(median_S_nu_SA_FF_CO, median_f_th_SA_FF_CO, median_a_SA_FF_CO)


#------------------------------------------------------------------------------------------------------------------------------------

flux = flux*u.uJy
E_flux = E_flux*u.uJy
frequency_obs = frequency_obs*u.GHz
restfreq = restfreq*u.GHz
frequency = frequency*u.GHz

#Plot the figure 

axis=[1.4,3,10, 22]	

plt.figure(figsize=(10,8));
#frequency1 = frequency

#frequency = np.linspace(1.4, 22,100)*u.GHz
#The data points
plt.errorbar(frequency_obs, flux, yerr= E_flux, fmt='.k', markersize='11')#,label='nterms=1'

#All diffetrent lines from mcmc
#for i in range(len(samples_pl[:,0])):
#	plt.plot(frequency, samples_pl[i,0]*np.power(frequency, -samples_pl[i,1]), '-r', alpha=0.05, linewidth=0.1)

#Power law
plt.plot(frequency, (median_S_pl*np.power(frequency/restfreq, -median_a_pl))*u.uJy, '-',color=palette(0), label='Power law: $\chi^2_{red}$ = %.2f' % chi_pl)
plt.fill_between(frequency, pl_16, pl_84, facecolor = palette(1), alpha = 0.3, edgecolor= palette(1),interpolate = True);

#Free-free + Synchrotron
plt.plot(frequency, ((1-median_f_th)*median_S_nu0*np.power(frequency/restfreq,-median_a_NT)+median_f_th*median_S_nu0*np.power(frequency/restfreq,-0.1))*u.uJy, '-',color=palette(2), label = 'FF + S: $\chi^2_{red}$ = %.2f' % chi_th)
plt.plot(frequency, (median_f_th*median_S_nu0*np.power(frequency/restfreq,-0.1))*u.uJy, '-.',color=palette(2))#, label='Free-free emission')
plt.plot(frequency, ((1-median_f_th)*median_S_nu0*np.power(frequency/restfreq,-median_a_NT))*u.uJy, '--',color=palette(2))#, label='Synchrotron emission')
plt.fill_between(frequency, th_16, th_84, facecolor = palette(3), alpha = 0.3, edgecolor= palette(3),interpolate = True);

#Syncrotron aging
plt.plot(frequency, (median_S_nu_SA*((frequency/restfreq)**(-median_a_SA))/(1+(frequency/(median_nu_b_SA*u.GHz))**0.5))*u.uJy, '-',color=palette(4), label = r'SA: $\chi^2_{red}$ = %.2f' % chi_SA)
plt.fill_between(frequency, SA_16, SA_84, facecolor = palette(5), alpha = 0.3, edgecolor= palette(5),interpolate = True);

#Synchrotron aging + Free-free
plt.plot(frequency, (median_f_th_SA_FF*median_S_nu_SA_FF*(frequency/restfreq)**(-0.1)+(1-median_f_th_SA_FF)*median_S_nu_SA_FF*((frequency/restfreq)**(-median_a_SA_FF))/(1+(frequency/(nu_b_SA_FF_guess*u.GHz))**0.5))*u.uJy, '-',color=palette(6), label= r'SA + FF: $\chi^2_{red}$ = %.2f' % chi_SA_FF)
plt.fill_between(frequency, SA_FF_16, SA_FF_84, facecolor=palette(7), alpha=0.3, edgecolor=palette(7), interpolate = True);
plt.plot(frequency, (median_f_th_SA_FF*median_S_nu_SA_FF*(frequency/restfreq)**(-0.1)*u.uJy), '-.',color=palette(6));
plt.plot(frequency, (1-median_f_th_SA_FF)*median_S_nu_SA_FF*((frequency/restfreq)**(-median_a_SA_FF))/(1+(frequency/(nu_b_SA_FF_guess*u.GHz))**0.5)*u.uJy, '--',color=palette(6));

#Synchrotron aging +free free with exponential cutoff
plt.plot(frequency, (median_f_th_SA_FF_CO*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-0.1)+(1-median_f_th_SA_FF_CO)*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-median_a_SA_FF_CO)*np.exp(-frequency/(nu_b_SA_FF_CO_guess*u.GHz)))*u.uJy, '-',color=palette(8), label= r'SA + FF,  exp: $\chi^2_{red}$ = %.2f' % chi_SA_FF_CO)
plt.fill_between(frequency, SA_FF_CO_16, SA_FF_CO_84, facecolor= palette(9), alpha=0.3, edgecolor = palette(9), interpolate = True);
plt.plot(frequency, (median_f_th_SA_FF_CO*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-0.1))*u.uJy, '-.',color=palette(8));
plt.plot(frequency, ((1-median_f_th_SA_FF_CO)*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-median_a_SA_FF_CO)*np.exp(-frequency/(nu_b_SA_FF_CO_guess*u.GHz)))*u.uJy, '--',color = palette(8));

# The spectralindices between data points
#for i in range(len(flux)-1):
#	plt.text((frequency[i]+frequency[i+1])/2, (flux[i]+flux[i+1])/2, r'$\alpha^{%s \mathrm{GHz}}_{%s \mathrm{GHZ}} = {%.2f}\pm %.2f$' % (frequency[i], frequency[i+1], alpha[i], E_alpha[i]));

#Some figure settings
plt.title('CXS-2 (z=0.679)', fontsize=22);
plt.xlabel(r'$\nu_{obs}$ (GHz)', fontsize=22);
plt.ylabel(r'Flux Density ($\mu$Jy)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis,axis, fontsize=20)
plt.savefig('CXS-2_Flux_Density.pdf',  dpi=500, clobber=True);
#plt.show();

#-------------------------------------------------------------------------------------------------------------------------------------------

#Calculating the ff SFR Luminosity

SFR = 165*const.M_sun/u.year
z = 0.679
def L_ff(SFR):
	L_ff = ((SFR)*1/(4.3*10**(-28)*(frequency*(1+z)/u.GHz)**(0.1))*u.erg/u.s/u.Hz).to(u.W/u.Hz)
	return L_ff
	
# convert flux to luminosity
def Lumos(Flux, z=0.679):
	D_l = cosmo.luminosity_distance(z)
	Luminosity = (Flux*(4*np.pi*((D_l)**2/(1+z)))).to(u.W/u.Hz)
	return Luminosity
	
#------------------------------------------------------------------------------------------------------------------------------------------

#Plot the luminosities

axis1=[1.4*(1+z),3*(1+z),10*(1+z),22*(1+z)]

plt.figure(figsize=(10,8));

#The data points
plt.errorbar(frequency_obs*(1+z), Lumos(flux), yerr= Lumos(E_flux), fmt='.k', markersize='11')

#Expected free free luminosity
plt.plot(frequency*(1+z), L_ff(197), '-',color=palette(12), label='Expected Luminosity')
plt.fill_between(frequency*(1+z), L_ff(197-19), L_ff(197+19), facecolor=palette(13), alpha = 0.3, edgecolor=palette(13))

#All mcmc results for the power law
#for i in range(len(samples_pl[:,0])):
#	plt.plot(frequency, samples_pl[i,0]*np.power(frequency, -samples_pl[i,1]), '-r', alpha=0.05, linewidth=0.1)

#PLot the power law
plt.plot(frequency*(1+z), Lumos(median_S_pl*np.power(frequency/restfreq, -median_a_pl)*u.uJy), '-', color= palette(0), label='Power law: $\chi^2_{red}$ = %.2f' % chi_pl)
plt.fill_between(frequency*(1+z), Lumos(pl_16), Lumos(pl_84), facecolor = palette(1), alpha = 0.3, edgecolor= palette(1),interpolate = True);

#plot Synchrotron + free-free
plt.plot(frequency*(1+z), Lumos(((1-median_f_th)*median_S_nu0*np.power(frequency/restfreq,-median_a_NT)+median_f_th*median_S_nu0*np.power(frequency/restfreq,-0.1))*u.uJy), '-', color=palette(2),label = 'FF + S: $\chi^2_{red}$ = %.2f' % chi_th)
plt.plot(frequency*(1+z), Lumos((median_f_th*median_S_nu0*np.power(frequency/restfreq,-0.1))*u.uJy), '-.', color=palette(2))#,label='Free-free emission')
plt.plot(frequency*(1+z), Lumos(((1-median_f_th)*median_S_nu0*np.power(frequency/restfreq,-median_a_NT))*u.uJy), '--', color=palette(2))#,label='Synchrotron emission')
plt.fill_between(frequency*(1+z), Lumos(th_16), Lumos(th_84), facecolor = palette(3), alpha = 0.3, edgecolor= palette(3),interpolate = True);

#Synchrotron aging
plt.plot(frequency*(1+z), Lumos(((median_S_nu_SA*((frequency/restfreq)**(-median_a_SA))/(1+(frequency/(median_nu_b_SA*u.GHz))**0.5))*u.uJy)), '-',color=palette(4), label = r'SA: $\chi^2_{red}$ = %.2f' % chi_SA)
plt.fill_between(frequency*(1+z), Lumos(SA_16), Lumos(SA_84), facecolor = palette(5), alpha = 0.3, edgecolor= palette(5),interpolate = True);

#Synchrotron aging + free-free
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF*median_S_nu_SA_FF*(frequency/restfreq)**(-0.1)+(1-median_f_th_SA_FF)*median_S_nu_SA_FF*((frequency/restfreq)**(-median_a_SA_FF))/(1+(frequency/(nu_b_SA_FF_guess*u.GHz))**0.5))*u.uJy), '-', color = palette(6),label= r'SA + FF: $\chi^2_{red}$ = %.2f' % chi_SA_FF)
plt.fill_between(frequency*(1+z), Lumos(SA_FF_16), Lumos(SA_FF_84), facecolor= palette(7), alpha=0.3, edgecolor = palette(7), interpolate = True);
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF*median_S_nu_SA_FF*(frequency/restfreq)**(-0.1)*u.uJy)),'-.',color = palette(6));
plt.plot(frequency*(1+z), Lumos((1-median_f_th_SA_FF)*median_S_nu_SA_FF*((frequency/restfreq)**(-median_a_SA_FF))/(1+(frequency/(nu_b_SA_FF_guess*u.GHz))**0.5)*u.uJy), '--',color = palette(6));

#Synchrotron aging +free free with exponential cutoff
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF_CO*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-0.1)+(1-median_f_th_SA_FF_CO)*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-median_a_SA_FF_CO)*np.exp(-frequency/(nu_b_SA_FF_CO_guess*u.GHz)))*u.uJy), '-',color=palette(8), label= r'SA + FF,  exp: $\chi^2_{red}$ = %.2f' % chi_SA_FF_CO)
plt.fill_between(frequency*(1+z), Lumos(SA_FF_CO_16), Lumos(SA_FF_CO_84), facecolor= palette(9), alpha=0.3, edgecolor = palette(9), interpolate = True);
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF_CO*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-0.1))*u.uJy), '-.',color=palette(8));
plt.plot(frequency*(1+z), Lumos(((1-median_f_th_SA_FF_CO)*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-median_a_SA_FF_CO)*np.exp(-frequency/(nu_b_SA_FF_CO_guess*u.GHz)))*u.uJy), '--',color=palette(8));

#The spectral indices between the data points
#for i in range(len(flux)-1):
#	plt.text((frequency[i]+frequency[i+1])/2, (flux[i]+flux[i+1])/2, r'$\alpha^{%s \mathrm{GHz}}_{%s \mathrm{GHZ}} = {%.2f}\pm %.2f$' % (frequency[i], frequency[i+1], alpha[i], E_alpha[i]));

#Some logistics of the figure
plt.title('CXS-2 (z=0.679)', fontsize=22);
plt.xlabel(r'$\nu_{rest}$ (GHz)', fontsize = 22);
plt.ylabel(r'Luminosity (W Hz$^{-1}$)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis1,axis1, fontsize=20)
plt.savefig('CXS-2_Luminosity.pdf',  dpi=500, clobber=True);
#plt.show();

#--------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,8));
#The data points
plt.errorbar(frequency_obs*(1+z), Lumos(flux), yerr= Lumos(E_flux), fmt='.k', markersize='11')

#Expected free free luminosity
plt.plot(frequency*(1+z), L_ff(197), '-',color=palette(12), label='Expected Luminosity')
plt.fill_between(frequency*(1+z), L_ff(197-19), L_ff(197+19), facecolor=palette(13), alpha = 0.3, edgecolor=palette(13))


#PLot the power law
plt.plot(frequency*(1+z), Lumos(median_S_pl*np.power(frequency/restfreq, -median_a_pl)*u.uJy), '-', color= palette(0), label='Power law: $\chi^2_{red}$ = %.2f' % chi_pl)
plt.fill_between(frequency*(1+z), Lumos(pl_16), Lumos(pl_84), facecolor = palette(1), alpha = 0.3, edgecolor= palette(1),interpolate = True);

#Some logistics of the figure
#plt.title('CXS-2 (z=1.44)', fontsize=15);
plt.xlabel(r'$\nu_{rest}$ (GHz)', fontsize = 22);
plt.ylabel(r'Luminosity (W Hz$^{-1}$)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis1,axis1,fontsize=20)
plt.savefig('CXS-2_pl.pdf',  dpi=500, clobber=True);
#plt.show();

#--------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,8));

#The data points
plt.errorbar(frequency_obs*(1+z), Lumos(flux), yerr= Lumos(E_flux), fmt='.k', markersize='11')

#Expected free free luminosity
plt.plot(frequency*(1+z), L_ff(197), '-',color=palette(12), label='Expected Luminosity')
plt.fill_between(frequency*(1+z), L_ff(197-19), L_ff(197+19), facecolor=palette(13), alpha = 0.3, edgecolor=palette(13))

#plot Synchrotron + free-free
plt.plot(frequency*(1+z), Lumos(((1-median_f_th)*median_S_nu0*np.power(frequency/restfreq,-median_a_NT)+median_f_th*median_S_nu0*np.power(frequency/restfreq,-0.1))*u.uJy), '-', color=palette(2),label = 'FF + S: $\chi^2_{red}$ = %.2f' % chi_th)
plt.plot(frequency*(1+z), Lumos((median_f_th*median_S_nu0*np.power(frequency/restfreq,-0.1))*u.uJy), '-.', color=palette(2))#,label='Free-free emission')
plt.plot(frequency*(1+z), Lumos(((1-median_f_th)*median_S_nu0*np.power(frequency/restfreq,-median_a_NT))*u.uJy), '--', color=palette(2))#,label='Synchrotron emission')
plt.fill_between(frequency*(1+z), Lumos(th_16), Lumos(th_84), facecolor = palette(3), alpha = 0.3, edgecolor= palette(3),interpolate = True);


#plt.title('CXS-2 (z=1.44)', fontsize=15);
plt.xlabel(r'$\nu_{rest}$ (GHz)', fontsize = 22);
plt.ylabel(r'Luminosity (W Hz$^{-1}$)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis1,axis1,fontsize=20)
plt.savefig('CXS-2_FF+S.pdf',  dpi=500, clobber=True);
#plt.show();

#------------------------------------------------------------------------------------------------------

plt.figure(figsize=(10,8));

#The data points
plt.errorbar(frequency_obs*(1+z), Lumos(flux), yerr= Lumos(E_flux), fmt='.k', markersize='11')

#Expected free free luminosity
plt.plot(frequency*(1+z), L_ff(197), '-',color=palette(12), label='Expected Luminosity')
plt.fill_between(frequency*(1+z), L_ff(197-19), L_ff(197+19), facecolor=palette(13), alpha = 0.3, edgecolor=palette(13))

#Synchrotron aging
plt.plot(frequency*(1+z), Lumos(((median_S_nu_SA*((frequency/restfreq)**(-median_a_SA))/(1+(frequency/(median_nu_b_SA*u.GHz))**0.5))*u.uJy)), '-',color=palette(4), label = r'SA: $\chi^2_{red}$ = %.2f' % chi_SA)
plt.fill_between(frequency*(1+z), Lumos(SA_16), Lumos(SA_84), facecolor = palette(5), alpha = 0.3, edgecolor= palette(5),interpolate = True);


#Some logistics of the figure
#plt.title('CXS-2 (z=1.44)', fontsize=15);
plt.xlabel(r'$\nu_{rest}$ (GHz)', fontsize = 22);
plt.ylabel(r'Luminosity (W Hz$^{-1}$)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis1,axis1,fontsize=20)
plt.savefig('CXS-2_SA.pdf',  dpi=500, clobber=True);
#plt.show();

#---------------------------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,8));

#The data points
plt.errorbar(frequency_obs*(1+z), Lumos(flux), yerr= Lumos(E_flux), fmt='.k', markersize='11')

#Expected free free luminosity
plt.plot(frequency*(1+z), L_ff(197), '-',color=palette(12), label='Expected Luminosity')
plt.fill_between(frequency*(1+z), L_ff(197-19), L_ff(197+19), facecolor=palette(13), alpha = 0.3, edgecolor=palette(13))

#Synchrotron aging + free-free
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF*median_S_nu_SA_FF*(frequency/restfreq)**(-0.1)+(1-median_f_th_SA_FF)*median_S_nu_SA_FF*((frequency/restfreq)**(-median_a_SA_FF))/(1+(frequency/(nu_b_SA_FF_guess*u.GHz))**0.5))*u.uJy), '-', color = palette(6),label= r'SA + FF: $\chi^2_{red}$ = %.2f' % chi_SA_FF)
plt.fill_between(frequency*(1+z), Lumos(SA_FF_16), Lumos(SA_FF_84), facecolor= palette(7), alpha=0.3, edgecolor = palette(7), interpolate = True);
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF*median_S_nu_SA_FF*(frequency/restfreq)**(-0.1)*u.uJy)),'-.',color = palette(6));
plt.plot(frequency*(1+z), Lumos((1-median_f_th_SA_FF)*median_S_nu_SA_FF*((frequency/restfreq)**(-median_a_SA_FF))/(1+(frequency/(nu_b_SA_FF_guess*u.GHz))**0.5)*u.uJy), '--',color = palette(6));



#Some logistics of the figure
#plt.title('CXS-2 (z=1.44)', fontsize=15);
plt.xlabel(r'$\nu_{rest}$ (GHz)', fontsize = 22);
plt.ylabel(r'Luminosity (W Hz$^{-1}$)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis1,axis1,fontsize=20)
plt.savefig('CXS-2_SA+FF.pdf',  dpi=500, clobber=True);
#plt.show();

#---------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,8));

#The data points
plt.errorbar(frequency_obs*(1+z), Lumos(flux), yerr= Lumos(E_flux), fmt='.k', markersize='11')

#Expected free free luminosity
plt.plot(frequency*(1+z), L_ff(197), '-',color=palette(12), label='Expected Luminosity')
plt.fill_between(frequency*(1+z), L_ff(197-19), L_ff(197+19), facecolor=palette(13), alpha = 0.3, edgecolor=palette(13))

#Synchrotron aging +free free with exponential cutoff
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF_CO*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-0.1)+(1-median_f_th_SA_FF_CO)*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-median_a_SA_FF_CO)*np.exp(-frequency/(nu_b_SA_FF_CO_guess*u.GHz)))*u.uJy), '-',color=palette(8), label= r'SA + FF,  exp: $\chi^2_{red}$ = %.2f' % chi_SA_FF_CO)
plt.fill_between(frequency*(1+z), Lumos(SA_FF_CO_16), Lumos(SA_FF_CO_84), facecolor= palette(9), alpha=0.3, edgecolor = palette(9), interpolate = True);
plt.plot(frequency*(1+z), Lumos((median_f_th_SA_FF_CO*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-0.1))*u.uJy), '-.',color=palette(8));
plt.plot(frequency*(1+z), Lumos(((1-median_f_th_SA_FF_CO)*median_S_nu_SA_FF_CO*(frequency/restfreq)**(-median_a_SA_FF_CO)*np.exp(-frequency/(nu_b_SA_FF_CO_guess*u.GHz)))*u.uJy), '--',color=palette(8));


#Some logistics of the figure
#plt.title('CXS-2 (z=1.44)', fontsize=15);
plt.xlabel(r'$\nu_{rest}$ (GHz)', fontsize = 22);
plt.ylabel(r'Luminosity (W Hz$^{-1}$)', fontsize=22);
plt.xscale('log');
plt.yscale('log');
plt.legend(fontsize=22,frameon=True, loc='upper right')
plt.grid()
plt.yticks(fontsize=20)
plt.xticks(axis1,axis1,fontsize=20)
plt.savefig('CXS-2_SA+FF+exp.pdf',  dpi=500, clobber=True);
plt.show();



