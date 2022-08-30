from corr_utils import polar_angular_correlation, to_polar




import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import h5py

from extra_geom import AGIPD_1MGeometry
from extra_data import open_run, stack_detector_data
import extra_data



# geomfilename = '../agipd_2304_vj_opt_v4.geom'
# geomfilename = '../agipd.geom'
# geomfilename = '../agipd_2819_v2.geom'
# geom = AGIPD_1MGeometry.from_crystfel_geom(geomfilename)
# geom.inspect()


geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
    (-525, 625),
    (-550, -10),
    (520, -160),
    (542.5, 475),
    ])

# h5filename = '../RAW-R0101-AGIPG15-S00009.h5'
# run = extra_data.H5File(h5filename, inc_suspect_trains=True)


# run = open_run(proposal=2819, run=101)

run = open_run(proposal=700000, run=5)
# run.info()




sel = run.select('*/DET/*', 'image.data')

train_id, train_data = sel.train_from_index(60)

stacked = stack_detector_data(train_data, 'image.data')
stacked_pulse = stacked[40][0]








res, center = geom.position_modules(stacked_pulse)



mask = np.zeros(res.shape)
mask[np.where(res!=0)] = 1

fig, axes = plt.subplots(1,2)
axes[0].imshow(res, origin='lower', vmin=0, vmax=10000)
axes[0].title('data')
axes[1].imshow(mask)
axes[1].title('mask')





im_unwrap = to_polar(res, 500, center[0], center[1])
mask_unwrap = to_polar(mask, 500, center[0], center[1])


# im_unwrap = im_unwrap[83:,:]
# mask_unwrap = mask_unwrap[83:,:]



fig, axes = plt.subplots(1,2)
axes[0].imshow(im_unwrap, origin='lower')
axes[1].imshow(mask_unwrap, origin='lower')


im_corr = polar_angular_correlation(im_unwrap)
mask_corr = polar_angular_correlation(mask_unwrap)





fig, axes = plt.subplots(1,2)
axes[0].imshow(im_corr, origin='lower')
axes[1].imshow(mask_corr, origin='lower')





im_corr_mask_div = im_corr[:]
im_corr_mask_div[np.where(mask_corr !=0)] *= 1/ mask_corr[np.where(mask_corr !=0)]

plt.figure()
plt.imshow(im_corr_mask_div, origin='lower')













plt.show()
