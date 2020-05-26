import galaxy_bao_fitter
import numpy as np
import pylab as plt
import copy
        
def test(pk_file='pk_camb_z0.698_challenge.txt', 
         pars_to_test= {'at': [0.95, 1.05, 10], 
                        'ap': [0.95, 1.05, 10], 
                        'bias': [1., 3., 10], 
                        'beta': [0., 1., 10], 
                        'sigma_rec': [0, 20, 10],
                        'sigma_s': [0, 6, 10]},
         pars_center = {'ap': 1.0, 'at': 1.0, 
                        'bias': 2.3, 'beta': 0.35, 
                        'sigma_par': 10., 'sigma_per': 6., 
                        'sigma_s': 4., 'sigma_rec': 0.},
         rmin=1., rmax=200., scale_r=2, ell_max=4, 
         decouple_peak=False, no_peak=False, 
         figsize=(12, 5), saveroot=None):

    cosmo = galaxy_bao_fitter.PowerSpectrum(pk_file)
    options = {'ell_max': ell_max, 'decouple_peak': decouple_peak, 'no_peak': no_peak}

    labels = {'aiso': r'$\alpha_{\rm iso}$', 
              'epsilon': r'$\epsilon$',
              'at': r'$\alpha_\perp$', 
              'ap': r'$\alpha_\parallel$',
              'beta': r'$\beta$', 'bias': r'$b$', 
              'sigma_rec': r'$\Sigma_{\rm rec}$', 
              'sigma_par': r'$\Sigma_{\parallel}$',
              'sigma_per': r'$\Sigma_{\perp}$',
              'sigma_s': r'$\Sigma_{\rm s}$'}

    title = ', '.join( [labels[par]+f' = {pars_center[par]:.1f}' for par in pars_center])
    r = np.linspace(rmin, rmax, 2000)
    nell = ell_max//2+1
    lss = ['-', '--', ':', '-.']

    for par in pars_to_test:
        pars = copy.deepcopy(pars_center)
        xs = pars_to_test[par]
        values = np.linspace(xs[0], xs[1], xs[2])
        fig, ax = plt.subplots(nrows=1, ncols=nell, figsize=figsize)
        colors = plt.cm.jet(np.linspace(0, 1, len(values)))
        for i, val in enumerate(values):
            pars[par] = val
            xi_mult = cosmo.get_xi_multipoles(r, pars, options)
            for j in range(nell):
                ax[j].plot(r, xi_mult[j]*r**scale_r, 
                           color=colors[i], lw=1,
                           label=labels[par]+f' = {val:.2f}')
                ylabel = r'$\xi_{%d}$'%(j*2)
                if scale_r:
                    ylabel+= r'$r^%d \ [h^{-%d}  \mathrm{Mpc}^{%d}]$'%\
                               (scale_r, scale_r, scale_r)
                ax[j].set_ylabel(ylabel)
                ax[j].set_xlabel(r'$r \ [h^{-1} \mathrm{Mpc}]$')
        ax[0].legend(loc=0, fontsize=10, ncol=2)
        plt.suptitle(title)
        plt.tight_layout() 
        fig.subplots_adjust(top=0.9)
        if not saveroot is None:
            plt.savefig(saveroot+f'{par}.pdf')

    plt.show()

def test_xu2013():
    ''' This function reproduces the figure 1 of Xu et al. 2013 
        https://ui.adsabs.harvard.edu/abs/2013MNRAS.431.2834X/abstract
    '''

    #-- Linear 
    test(   pk_file='data/pk_camb_z0.35_xu2013.txt', 
            pars_to_test = {'epsilon': [-0.01, 0.01, 3]}, 
            pars_center =  {'aiso': 1.0, 'epsilon': 0., 'bias': 2.2, 'beta': 0.35, 
                           'sigma_par': 0.0, 'sigma_per': .0, 'sigma_s': 0, 
                           'sigma_rec': 0.0},
            decouple_peak=False)

    #-- With damping
    test(   pk_file='data/pk_camb_z0.35_xu2013.txt', 
            pars_to_test = {'epsilon': [-0.01, 0.01, 3]}, 
            pars_center =  {'aiso': 1.0, 'epsilon': 0., 'bias': 2.2, 'beta': 0.35, 
                           'sigma_par': 10.0, 'sigma_per': 6.0, 'sigma_s': 4., 
                           'sigma_rec': 0.0}, 
            decouple_peak=False)

def test_window_function():
    ''' Do not use: this is currently broken '''

    cosmo = Cosmo(z=0.75) 
    cosmo.read_window_function('window_test.txt') 

    pars_center = {'ap': 1.0, 'at': 1.0,
                   'bias': 2.3, 'beta': 0.35,
                   'sigma_par': 10., 'sigma_per': 6.,
                   'sigma_s': 4., 'sigma_rec': 0.}
    
    kout = np.linspace(0.01, 0.3, 300)

    pk_mult0 = cosmo.get_pk_multipoles(kout, pars_center, apply_window=False)
    pk_mult1 = cosmo.get_pk_multipoles(kout, pars_center, apply_window=True )
    for i in range(len(pk_mult0)):
        ell = 2*i
        plt.plot(kout, pk_mult0[i]*kout, f'C{i}--', label=r'$\ell = {ell}$ Without Window'.format(ell=ell))
        plt.plot(kout, pk_mult1[i]*kout, f'C{i}-', label=r'$\ell = {ell}$ With Window'.format(ell=ell))
    plt.legend()
    plt.ylabel(r'$k P_{\ell}(k)$')

