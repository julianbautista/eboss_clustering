import galaxy_bao_fitter
import numpy as np
import pylab as plt
import copy
        
def get_cosmo(z=0.75, name='challenge'):
    cosmo = galaxy_bao_fitter.Cosmo(z=z, name=name)
    return cosmo

def test(cosmo, 
         pars_to_test= {'at': [0.95, 1.05, 11], 
                        'ap': [0.95, 1.05, 11], 
                        'beta': [0.25, 0.45, 11], 
                        'sigma_rec': [0, 50, 20],
                        'sigma_s': [0, 6, 11]},
         pars_center = {'ap': 1.0, 'at': 1.0, 
                        'bias': 1.0, 'beta': 0.6, 
                        'sigma_par': 10., 'sigma_per': 6., 
                        'sigma_s': 4., 'sigma_rec': 0.},
         rmin=1., rmax=200., scale_r=2, ell_max=4, 
         decoupled=False, no_peak=False, 
         figsize=(12, 5)):

        r = np.linspace(rmin, rmax, 2000)
        nell = ell_max//2+1
        lss = ['-', '--', ':', '-.']

        if 'aiso' in pars_to_test:
            pars = copy.deepcopy(pars_center)
            fig, ax = plt.subplots(nrows=1, ncols=nell, figsize=figsize)
            xs = pars_to_test['aiso']
            xs = np.linspace(xs[0], xs[1], xs[2])
            colors = plt.cm.jet(np.linspace(0, 1, len(aisos)))

            for i, ap in enumerate(aisos):
                pars['at'] = ap
                pars['ap'] = ap
                aiso = ap
                xi_mult = cosmo.get_xi_multipoles(r, pars, ell_max=ell_max, 
                                decoupled=decoupled, no_peak=no_peak)
                for j in range(nell):
                    ax[j].plot(r, xi_mult[j]*r**scale_r, 
                               color=colors[i], lw=1, 
                               label=r'$\alpha_{\rm iso} = %.2f$'%aiso)
                    ylabel = r'$\xi_{%d}$'%(j*2)
                    if scale_r:
                        ylabel+= r'$r^%d [h^{-%d} \mathrm{Mpc}^{%d}]$'%\
                                   (scale_r, scale_r, scale_r)
                    ax[j].set_ylabel(ylabel)
            ax[-1].set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
            ax[0].legend(loc=0, fontsize=10)
            plt.tight_layout()

        if 'epsilon' in pars_to_test: 
            pars = copy.deepcopy(pars_center)
            fig, ax = plt.subplots(nrows=1, ncols=nell, figsize=figsize)
            xs = pars_to_test['epsilon']
            xs = np.linspace(xs[0], xs[1], xs[2])
            colors = plt.cm.jet(np.linspace(0, 1, len(epsilons)))

            for i, ap in enumerate(pars_to_test['epsilon']):
                pars['at'] = 1./np.sqrt(ap)
                pars['ap'] = ap
                epsilon = (ap*np.sqrt(ap))**(1./3)-1
                xi_mult = cosmo.get_xi_multipoles(r, pars, ell_max=ell_max,
                                decoupled=decoupled, no_peak=no_peak)
                for j in range(nell):
                    ax[j].plot(r, xi_mult[j]*r**scale_r, 
                               color=colors[i], 
                               label=r'$\epsilon = %.3f$'%epsilon)
                    ylabel = r'$\xi_{%d}$'%(j*2)
                    if scale_r:
                        ylabel+= r'$r^%d [h^{-%d} \mathrm{Mpc}^{%d}]$'%\
                                   (scale_r, scale_r, scale_r)
                    ax[j].set_ylabel(ylabel)
            ax[-1].set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
            ax[0].legend(loc=0, fontsize=10)
            plt.tight_layout()

        for par in pars_to_test:
            if par == 'aiso' or par == 'epsilon': continue
            pars = copy.deepcopy(pars_center)
            xs = pars_to_test[par]
            values = np.linspace(xs[0], xs[1], xs[2])
            fig, ax = plt.subplots(nrows=1, ncols=nell, figsize=figsize)
            colors = plt.cm.jet(np.linspace(0, 1, len(values)))
            for i, val in enumerate(values):
                pars[par] = val
                xi_mult = cosmo.get_xi_multipoles(r, pars, ell_max=ell_max,
                                decoupled=decoupled, no_peak=no_peak)
                for j in range(nell):
                    ax[j].plot(r, xi_mult[j]*r**scale_r, 
                               color=colors[i], 
                               label=f'{par} = {val:.3f}')
                    ylabel = r'$\xi_{%d}$'%(j*2)
                    if scale_r:
                        ylabel+= r'$r^%d [h^{-%d} \mathrm{Mpc}^{%d}]$'%\
                                   (scale_r, scale_r, scale_r)
                    ax[j].set_ylabel(ylabel)
            ax[-1].set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
            ax[-1].legend(loc=0, fontsize=10)
            plt.tight_layout() 

def test_xu2012():
    cosmo = get_cosmo(z=0.35)
    test(  cosmo, 
           pars_to_test = {'epsilon': [1.0201, 0.9799, 10]}, 
           pars_center =  {'ap': 1.0, 'at': 1.0, 'bias': 2.2, 'beta': 0.35, 
                           'sigma_par': 0.0, 'sigma_per': .0, 'sigma_s': 0, 
                           'sigma_rec': 0.0})

