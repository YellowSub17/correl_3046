




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from matplotlib.colors import Normalize






def plot_2d(im, **new_kwargs):
    kwargs ={   'extent':None,
                'fig':None,
                'axes':None,
                'log':False,
                'cmap':'viridis',
                'cb':True,
                'vminmax':(None, None),
                'xlabel':'',
                'ylabel':'',
                'title':'',
                'origin':'lower',
                'ticks':True,
                'norm':False,
                'subtmean':False,
                'blur':False,
                'save':''
            }

    kwargs.update(new_kwargs)


    # Make figure if none given
    if kwargs['fig'] is None:
        kwargs['fig'] = plt.figure()
        kwargs['axes'] = plt.gca()

    #Preprocessing: log intensity scale
    if kwargs['log']:
        im = np.log10(np.abs(im)+1)

    if kwargs['norm']:
        im -= im.min()
        im *= 1/im.max()

    if kwargs['subtmean']:
        means = im.mean(axis=1)
        xx, yy = np.meshgrid(np.ones(im.shape[1]), means)
        im -= yy

    if kwargs['blur']:
        im = gaussian_filter(im,  sigma=kwargs['blur'])





    kwargs['axes'].imshow(im, origin=kwargs['origin'], extent=kwargs['extent'], aspect='auto', cmap=kwargs['cmap'])

    # Postprocessing: colorscale limits
    vmin, vmax = kwargs['vminmax']
    if vmin is None:
        vmin = im.min()
    if vmax is None:
        vmax = im.max()


    if vmax==vmin:
        kwargs['cb']=False

    # Add colorbar
    if kwargs['cb']:
        norm = Normalize(vmin, vmax, clip=False)
        sm = ScalarMappable(cmap=kwargs['cmap'], norm=norm)
        sm._A = []
        kwargs['fig'].colorbar(sm, ax=kwargs['axes'])


    # Set colorlimits
    for axes_im in kwargs['axes'].get_images():
        axes_im.set_clim(vmin,vmax)

    kwargs['axes'].set_title(kwargs['title'])
    kwargs['axes'].set_xlabel(kwargs['xlabel'])
    kwargs['axes'].set_ylabel(kwargs['ylabel'])

    if kwargs['save'] is not None:
        plt.savefig(kwargs['save'])



