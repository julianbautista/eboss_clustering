from ebosscat import Catalog
import systematics 
import numpy as N
import pylab as P



def syst_fiber_dep(cat, ran, fit_index=N.array([0])):

    allra = cat.RA
    alldec = cat.DEC

    cat.cut((cat.Z>=0.6)&(cat.Z<=1.0)&(cat.IMATCH==1))
    ran.cut((ran.Z>=0.6)&(ran.Z<=1.0))
    ngals = cat.size


    fluxmag = 4
    fluxfield = 'FIBER2FLUX'
    flux = 22.5-2.5*N.log10(cat.__dict__[fluxfield][:, fluxmag])
    
    #fluxfield = 'MODELMAG'
    #flux = cat.__dict__[fluxfield][:, fluxmag]
    
    flux -= cat.EXTINCTION[:, fluxmag]
    
    #fluxfield = 'Z'
    #oflux = cat.__dict__[fluxfield]

    flux = flux[(cat.Z>=0.6)&(cat.Z<=1.0)&(cat.IMATCH==1)]
    

    #-- computing magnitude/z bins
    nbins = 4
    fluxbins = N.array([N.percentile(flux, i*(98./nbins)+1) \
                        for i in range(nbins+1)])
    centers = N.array([N.median(\
                        flux[(flux>fluxbins[i])&(flux<=fluxbins[i+1])])\
                        for i in range(nbins)])
    print('Fluxbins:', fluxbins)
    

    ms = list()
    
    for i in range(nbins):
        m = systematics.MultiFit()

        #-- read data
        wd = (flux > fluxbins[i]) & (flux <= fluxbins[i+1])
        m.read_data(cat.RA[wd], \
                    cat.DEC[wd], \
                    cat.get_weights(cp=1, fkp=1, noz=1, syst=0)[wd])

        #-- read randoms
        m.read_randoms(ran.RA,  \
                       ran.DEC, \
                       ran.get_weights(fkp=1, cp=0, noz=0, syst=0))

        m.prepare_new()
        print('Before fit: Bin', i, 'Chi2 = ', m.get_chi2())
        m.fit_pars(fit_index=fit_index)
        print('After fit: Bin', i, 'Chi2 = ', m.get_chi2(m.pars))
        print('Best fit parameters :', m.pars)
        #m.plot_overdensity(ylim=[0.5, 1.5])
        ms.append(m)

    
    pars = N.array([m.pars for m in ms])
    nsyst = pars.shape[1]-1

    #-- fit linear functions for the dependency
    npoly = 1
    coeff = N.zeros((nsyst+1, npoly+1))
    pars_new = N.zeros((nsyst+1, ngals))

    for i in range(nsyst+1):
        coeff[i] = N.polyfit(centers, pars[:, i], npoly)
        pars_new[i] = N.polyval(coeff[i], flux)
   
    #-- get systematic weights for all galaxies
    syst_data = N.zeros((allra.size, fit_index.size))
    for i in range(nsyst):
        syst_data[:, fit_index[i]] = m.get_map_values(fit_index[i],\
                                         allra, alldec)

    for i in range(ngals):
        weights = 1./m.get_model_weights(pars_new[:, i], syst_data) 

    return centers, ms, weights, pars_new 

       

def test_prepare_new(cat, ran):

    zmin = 0.6
    zmax = 1.0

    wd = (cat.Z>zmin)&(cat.Z<zmax)&((cat.IMATCH==1)|(cat.IMATCH==2))
    wr = (ran.Z>zmin)&(ran.Z<zmax)

    m = systematics.MultiFit()
    m.read_data(cat.RA[wd], cat.DEC[wd], \
                cat.get_weights(cp=1, fkp=1, noz=0, syst=0)[wd])
    m.read_randoms(ran.RA[wr], ran.DEC[wr], \
                   ran.get_weights(cp=0, fkp=1, noz=0, syst=0)[wr])
    m.prepare_new()
    print('Chi2 before fit:', m.get_chi2())
    m.fit_pars()
    m.plot_overdensity(ylim=[0.5, 1.4])
    print('Chi2 after fit:', m.get_chi2(m.pars))
    return m

