from corr_utils import polar_angular_correlation, to_polar, circle_center




import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

import h5py

from extra_geom import AGIPD_1MGeometry
from extra_data import open_run, stack_detector_data
import extra_data



# geomfilename = '../agipd_2819_v2.geom'
# geom = AGIPD_1MGeometry.from_crystfel_geom(geomfilename)


geom = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
    (-525, 625),
    (-550, -10),
    (520, -160),
    (542.5, 475),
    ])

# geom.inspect()



# run = open_run(proposal=2819, run=101)
run = open_run(proposal=700000, run=5)
# run.info()




sel = run.select('*/DET/*', 'image.data')

# train_id, train_data = sel.train_from_index(60)
# stacked = stack_detector_data(train_data, 'image.data')
# stacked_pulse = stacked[10]

train_id, train_data = run.select('*/DET/*', 'image.data').train_from_index(60)
stacked = stack_detector_data(train_data, 'image.data')
stacked_pulse = stacked[10][0]








res, center = geom.position_modules(stacked_pulse)

print(center)


mask = np.zeros(res.shape)
mask[np.where(res!=0)] = 1

fig, axes = plt.subplots(1,2)
axes[0].imshow(res, origin='lower', vmin=0, vmax=10000)
axes[0].set_title('data')
axes[1].imshow(mask)
axes[1].set_title('mask')


center = circle_center(
        (666.917, 721.986),
        (375.965, 610.029),
        (587.498, 477.084),
        )

print(center)

dx = 0#-5
dy = 0#-8


im_unwrap = to_polar(res, 500, center[1]+dx, center[0]+dy)
mask_unwrap = to_polar(mask, 500, center[1]+dx, center[0]+dy)




im_unwrap = im_unwrap[50:,:]
mask_unwrap = mask_unwrap[50:,:]



fig, axes = plt.subplots(1,2)
axes[0].imshow(im_unwrap, origin='lower')
axes[0].set_title('unwrapped image')
axes[1].imshow(mask_unwrap, origin='lower')
axes[1].set_title('unwrapped mask')


im_corr = polar_angular_correlation(im_unwrap)
mask_corr = polar_angular_correlation(mask_unwrap)





fig, axes = plt.subplots(1,2)
axes[0].imshow(im_corr, origin='lower')
axes[0].set_title('image corrlation')
axes[1].imshow(mask_corr, origin='lower')
axes[1].set_title('mask corrlation')





im_corr_mask_div = im_corr[:]
im_corr_mask_div[np.where(mask_corr !=0)] *= 1/ mask_corr[np.where(mask_corr !=0)]

plt.figure()
# plt.imshow(im_corr_mask_div, origin='lower', vmin=0, vmax=1)
plt.imshow(im_corr_mask_div, origin='lower')
plt.title('corr mask div')
plt.colorbar()













plt.show()
