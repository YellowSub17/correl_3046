from corr_utils import polar_angular_correlation, to_polar, circle_center,mask_correction
from plot_utils import plot_2d, plot_1d





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
print(stacked.shape)
stacked_pulse = stacked[20][0]








res, center = geom.position_modules(stacked_pulse)



mask = np.zeros(res.shape)
mask[np.where(res!=0)] = 1

# fig, axes = plt.subplots(1,2)
# axes[0].imshow(res, origin='lower', vmin=0, vmax=10000)
# axes[0].set_title('data')
# axes[1].imshow(mask)
# axes[1].set_title('mask')


# plot_2d(res, fig=fig, axes=axes[0], vminmax=(5e3, 6e3))
# plot_2d(mask, fig=fig, axes=axes[1], vminmax=(0, 1))


center = circle_center(
        # (666.917, 721.986),
        # (375.965, 610.029),
        # (587.498, 477.084),

        (276.134, 287.016),
        (410.735, 1034.42),
        (955.738, 562.901),
        )




im_unwrap = to_polar(res, 500, center[1], center[0])
mask_unwrap = to_polar(mask, 500, center[1], center[0])




im_unwrap = im_unwrap[50:,:]
mask_unwrap = mask_unwrap[50:,:]



fig, axes = plt.subplots(1,2)
plot_2d(im_unwrap, fig=fig, axes=axes[0])
plot_2d(mask_unwrap, fig=fig, axes=axes[1])


# axes[0].imshow(im_unwrap, origin='lower')
# axes[0].set_title('unwrapped image')
# axes[1].imshow(mask_unwrap, origin='lower')
# axes[1].set_title('unwrapped mask')


im_corr = polar_angular_correlation(im_unwrap)
mask_corr = polar_angular_correlation(mask_unwrap)





fig, axes = plt.subplots(1,2)
plot_2d(im_corr, fig=fig, axes=axes[0])
plot_2d(mask_corr, fig=fig, axes=axes[1])


# axes[0].imshow(im_corr, origin='lower')
# axes[0].set_title('image corrlation')
# axes[1].imshow(mask_corr, origin='lower')
# axes[1].set_title('mask corrlation')




im_corr_mask_corrected = mask_correction(im_corr, mask_corr)


# im_corr_mask_div = im_corr[:]
# im_corr_mask_div[np.where(mask_corr !=0)] *= 1/ mask_corr[np.where(mask_corr !=0)]

plot_2d(im_corr_mask_corrected, subtmean=True)
plot_2d(im_corr_mask_corrected, blur=4, subtmean=False)


# plt.figure()
# # plt.imshow(im_corr_mask_div, origin='lower', vmin=0, vmax=1)
# plt.imshow(im_corr_mask_div, origin='lower')
# plt.title('corr mask div')
# plt.colorbar()


# plt.figure()
# plt.plot(im_corr_mask_corrected[100:120, :].sum(axis=0))
# plt.plot(mask_corr[109:116, :].sum(axis=0))

fig, axes = plt.subplots(1,1)
plot_1d(im_corr_mask_corrected[100:120, :].sum(axis=0), norm=True, fig=fig, axes=axes)
plot_1d(mask_corr[100:120, :].sum(axis=0), norm=True, fig=fig, axes=axes, color='blue')
















plt.show()
