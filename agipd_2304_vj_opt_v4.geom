; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; Optimized panel offsets can be found at the end of the file
; AGIPD-1M geometry file written by geoAssembler 0.2.0
; You may need to edit this file to add:
; - data and mask locations in the file
; - mask_good & mask_bad values to interpret the mask
; - adu_per_eV & photon_energy
; - clen (detector distance)
;
; See: http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html

;data = /data/data
;mask = /mask

data = /entry_1/instrument_1/detector_1/data
mask = /entry_1/instrument_1/detector_1/mask


mask_good = 0x0000
mask_bad = 0xffff

adu_per_eV = 0.0075  ; no idea
clen = 0.16975
photon_energy = 9300 ;
;photon_energy = 3000000 ;
dim0 = %
dim1 = ss
dim2 = fs
res = 5000 ; 200 um pixels
rigid_group_q0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7,p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7,p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6,p3a0,p3a1,p3a2,p3a3,p3a4,p3a7
rigid_group_q1 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7,p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7,p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7,p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_q2 = p8a0,p8a1,p8a2,p8a3,p8a7,p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a7,p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7,p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_q3 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7,p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7,p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6,p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_p0 = p0a0,p0a1,p0a2,p0a3,p0a4,p0a5,p0a6,p0a7
rigid_group_p1 = p1a0,p1a1,p1a2,p1a3,p1a4,p1a5,p1a6,p1a7
rigid_group_p2 = p2a0,p2a1,p2a2,p2a3,p2a4,p2a5,p2a6
rigid_group_p3 = p3a0,p3a1,p3a2,p3a3,p3a4,p3a7
rigid_group_p4 = p4a0,p4a1,p4a2,p4a3,p4a4,p4a5,p4a6,p4a7
rigid_group_p5 = p5a0,p5a1,p5a2,p5a3,p5a4,p5a5,p5a6,p5a7
rigid_group_p6 = p6a0,p6a1,p6a2,p6a3,p6a4,p6a5,p6a6,p6a7
rigid_group_p7 = p7a0,p7a1,p7a2,p7a3,p7a4,p7a5,p7a6,p7a7
rigid_group_p8 = p8a0,p8a1,p8a2,p8a3,p8a7
rigid_group_p9 = p9a0,p9a1,p9a2,p9a3,p9a4,p9a5,p9a7
rigid_group_p10 = p10a0,p10a1,p10a2,p10a3,p10a4,p10a5,p10a6,p10a7
rigid_group_p11 = p11a0,p11a1,p11a2,p11a3,p11a4,p11a5,p11a6,p11a7
rigid_group_p12 = p12a0,p12a1,p12a2,p12a3,p12a4,p12a5,p12a6,p12a7
rigid_group_p13 = p13a0,p13a1,p13a2,p13a3,p13a4,p13a5,p13a6,p13a7
rigid_group_p14 = p14a0,p14a1,p14a2,p14a3,p14a4,p14a5,p14a6
rigid_group_p15 = p15a0,p15a1,p15a2,p15a3,p15a4,p15a5,p15a6,p15a7

rigid_group_collection_quadrants = q0,q1,q2,q3
rigid_group_collection_asics = p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15


;p0a4/dim1 = 0
;p0a4/dim2 = ss
;p0a4/dim3 = fs
p0a4/min_fs = 0
p0a4/min_ss = 256
p0a4/max_fs = 127
p0a4/max_ss = 319

;p0a1/dim1 = 0
;p0a1/dim2 = ss
;p0a1/dim3 = fs
p0a1/min_fs = 0
p0a1/min_ss = 64
p0a1/max_fs = 127
p0a1/max_ss = 127

;p0a2/dim1 = 0
;p0a2/dim2 = ss
;p0a2/dim3 = fs
p0a2/min_fs = 0
p0a2/min_ss = 128
p0a2/max_fs = 127
p0a2/max_ss = 191

;p0a0/dim1 = 0
;p0a0/dim2 = ss
;p0a0/dim3 = fs
p0a0/min_fs = 0
p0a0/min_ss = 0
p0a0/max_fs = 127
p0a0/max_ss = 63

;p0a3/dim1 = 0
;p0a3/dim2 = ss
;p0a3/dim3 = fs
p0a3/min_fs = 0
p0a3/min_ss = 192
p0a3/max_fs = 127
p0a3/max_ss = 255



;p0a5/dim1 = 0
;p0a5/dim2 = ss
;p0a5/dim3 = fs
p0a5/min_fs = 0
p0a5/min_ss = 320
p0a5/max_fs = 127
p0a5/max_ss = 383

;p0a6/dim1 = 0
;p0a6/dim2 = ss
;p0a6/dim3 = fs
p0a6/min_fs = 0
p0a6/min_ss = 384
p0a6/max_fs = 127
p0a6/max_ss = 447

;p0a7/dim1 = 0
;p0a7/dim2 = ss
;p0a7/dim3 = fs
p0a7/min_fs = 0
p0a7/min_ss = 448
p0a7/max_fs = 127
p0a7/max_ss = 511

;p1a0/dim1 = 1
;p1a0/dim2 = ss
;p1a0/dim3 = fs
p1a0/min_fs = 0
p1a0/min_ss = 512
p1a0/max_fs = 127
p1a0/max_ss = 575

;p1a1/dim1 = 1
;p1a1/dim2 = ss
;p1a1/dim3 = fs
p1a1/min_fs = 0
p1a1/min_ss = 576
p1a1/max_fs = 127
p1a1/max_ss = 639

;p1a2/dim1 = 1
;p1a2/dim2 = ss
;p1a2/dim3 = fs
p1a2/min_fs = 0
p1a2/min_ss = 640
p1a2/max_fs = 127
p1a2/max_ss = 703

;p1a3/dim1 = 1
;p1a3/dim2 = ss
;p1a3/dim3 = fs
p1a3/min_fs = 0
p1a3/min_ss = 704
p1a3/max_fs = 127
p1a3/max_ss = 767

;p1a4/dim1 = 1
;p1a4/dim2 = ss
;p1a4/dim3 = fs
p1a4/min_fs = 0
p1a4/min_ss = 768
p1a4/max_fs = 127
p1a4/max_ss = 831

;p1a5/dim1 = 1
;p1a5/dim2 = ss
;p1a5/dim3 = fs
p1a5/min_fs = 0
p1a5/min_ss = 832
p1a5/max_fs = 127
p1a5/max_ss = 895

;p1a6/dim1 = 1
;p1a6/dim2 = ss
;p1a6/dim3 = fs
p1a6/min_fs = 0
p1a6/min_ss = 896
p1a6/max_fs = 127
p1a6/max_ss = 959

;p1a7/dim1 = 1
;p1a7/dim2 = ss
;p1a7/dim3 = fs
p1a7/min_fs = 0
p1a7/min_ss = 960
p1a7/max_fs = 127
p1a7/max_ss = 1023

;p2a0/dim1 = 2
;p2a0/dim2 = ss
;p2a0/dim3 = fs
p2a0/min_fs = 0
p2a0/min_ss = 1024
p2a0/max_fs = 127
p2a0/max_ss = 1087

;p2a1/dim1 = 2
;p2a1/dim2 = ss
;p2a1/dim3 = fs
p2a1/min_fs = 0
p2a1/min_ss = 1088
p2a1/max_fs = 127
p2a1/max_ss = 1151

;p2a2/dim1 = 2
;p2a2/dim2 = ss
;p2a2/dim3 = fs
p2a2/min_fs = 0
p2a2/min_ss = 1152
p2a2/max_fs = 127
p2a2/max_ss = 1215

;p2a3/dim1 = 2
;p2a3/dim2 = ss
;p2a3/dim3 = fs
p2a3/min_fs = 0
p2a3/min_ss = 1216
p2a3/max_fs = 127
p2a3/max_ss = 1279

;p2a4/dim1 = 2
;p2a4/dim2 = ss
;p2a4/dim3 = fs
p2a4/min_fs = 0
p2a4/min_ss = 1280
p2a4/max_fs = 127
p2a4/max_ss = 1343

;p2a5/dim1 = 2
;p2a5/dim2 = ss
;p2a5/dim3 = fs
p2a5/min_fs = 0
p2a5/min_ss = 1344
p2a5/max_fs = 127
p2a5/max_ss = 1407

;p2a6/dim1 = 2
;p2a6/dim2 = ss
;p2a6/dim3 = fs
p2a6/min_fs = 0
p2a6/min_ss = 1408
p2a6/max_fs = 127
p2a6/max_ss = 1471


;p3a0/dim1 = 3
;p3a0/dim2 = ss
;p3a0/dim3 = fs
p3a0/min_fs = 0
p3a0/min_ss = 1536
p3a0/max_fs = 127
p3a0/max_ss = 1599

;p3a1/dim1 = 3
;p3a1/dim2 = ss
;p3a1/dim3 = fs
p3a1/min_fs = 0
p3a1/min_ss = 1600
p3a1/max_fs = 127
p3a1/max_ss = 1663

;p3a2/dim1 = 3
;p3a2/dim2 = ss
;p3a2/dim3 = fs
p3a2/min_fs = 0
p3a2/min_ss = 1664
p3a2/max_fs = 127
p3a2/max_ss = 1727

;p3a3/dim1 = 3
;p3a3/dim2 = ss
;p3a3/dim3 = fs
p3a3/min_fs = 0
p3a3/min_ss = 1728
p3a3/max_fs = 127
p3a3/max_ss = 1791

;p3a4/dim1 = 3
;p3a4/dim2 = ss
;p3a4/dim3 = fs
p3a4/min_fs = 0
p3a4/min_ss = 1792
p3a4/max_fs = 127
p3a4/max_ss = 1855


;p3a7/dim1 = 3
;p3a7/dim2 = ss
;p3a7/dim3 = fs
p3a7/min_fs = 0
p3a7/min_ss = 1984
p3a7/max_fs = 127
p3a7/max_ss = 2047

;p4a0/dim1 = 4
;p4a0/dim2 = ss
;p4a0/dim3 = fs
p4a0/min_fs = 0
p4a0/min_ss = 2048
p4a0/max_fs = 127
p4a0/max_ss = 2111

;p4a1/dim1 = 4
;p4a1/dim2 = ss
;p4a1/dim3 = fs
p4a1/min_fs = 0
p4a1/min_ss = 2112
p4a1/max_fs = 127
p4a1/max_ss = 2175

;p4a2/dim1 = 4
;p4a2/dim2 = ss
;p4a2/dim3 = fs
p4a2/min_fs = 0
p4a2/min_ss = 2176
p4a2/max_fs = 127
p4a2/max_ss = 2239

;p4a3/dim1 = 4
;p4a3/dim2 = ss
;p4a3/dim3 = fs
p4a3/min_fs = 0
p4a3/min_ss = 2240
p4a3/max_fs = 127
p4a3/max_ss = 2303

;p4a4/dim1 = 4
;p4a4/dim2 = ss
;p4a4/dim3 = fs
p4a4/min_fs = 0
p4a4/min_ss = 2304
p4a4/max_fs = 127
p4a4/max_ss = 2367

;p4a5/dim1 = 4
;p4a5/dim2 = ss
;p4a5/dim3 = fs
p4a5/min_fs = 0
p4a5/min_ss = 2368
p4a5/max_fs = 127
p4a5/max_ss = 2431

;p4a6/dim1 = 4
;p4a6/dim2 = ss
;p4a6/dim3 = fs
p4a6/min_fs = 0
p4a6/min_ss = 2432
p4a6/max_fs = 127
p4a6/max_ss = 2495

;p4a7/dim1 = 4
;p4a7/dim2 = ss
;p4a7/dim3 = fs
p4a7/min_fs = 0
p4a7/min_ss = 2496
p4a7/max_fs = 127
p4a7/max_ss = 2559

;p5a0/dim1 = 5
;p5a0/dim2 = ss
;p5a0/dim3 = fs
p5a0/min_fs = 0
p5a0/min_ss = 2560
p5a0/max_fs = 127
p5a0/max_ss = 2623

;p5a1/dim1 = 5
;p5a1/dim2 = ss
;p5a1/dim3 = fs
p5a1/min_fs = 0
p5a1/min_ss = 2624
p5a1/max_fs = 127
p5a1/max_ss = 2687

;p5a2/dim1 = 5
;p5a2/dim2 = ss
;p5a2/dim3 = fs
p5a2/min_fs = 0
p5a2/min_ss = 2688
p5a2/max_fs = 127
p5a2/max_ss = 2751

;p5a3/dim1 = 5
;p5a3/dim2 = ss
;p5a3/dim3 = fs
p5a3/min_fs = 0
p5a3/min_ss = 2752
p5a3/max_fs = 127
p5a3/max_ss = 2815

;p5a4/dim1 = 5
;p5a4/dim2 = ss
;p5a4/dim3 = fs
p5a4/min_fs = 0
p5a4/min_ss = 2816
p5a4/max_fs = 127
p5a4/max_ss = 2879

;p5a5/dim1 = 5
;p5a5/dim2 = ss
;p5a5/dim3 = fs
p5a5/min_fs = 0
p5a5/min_ss = 2880
p5a5/max_fs = 127
p5a5/max_ss = 2943

;p5a6/dim1 = 5
;p5a6/dim2 = ss
;p5a6/dim3 = fs
p5a6/min_fs = 0
p5a6/min_ss = 2944
p5a6/max_fs = 127
p5a6/max_ss = 3007

;p5a7/dim1 = 5
;p5a7/dim2 = ss
;p5a7/dim3 = fs
p5a7/min_fs = 0
p5a7/min_ss = 3008
p5a7/max_fs = 127
p5a7/max_ss = 3071

;p6a0/dim1 = 6
;p6a0/dim2 = ss
;p6a0/dim3 = fs
p6a0/min_fs = 0
p6a0/min_ss = 3072
p6a0/max_fs = 127
p6a0/max_ss = 3135

;p6a1/dim1 = 6
;p6a1/dim2 = ss
;p6a1/dim3 = fs
p6a1/min_fs = 0
p6a1/min_ss = 3136
p6a1/max_fs = 127
p6a1/max_ss = 3199

;p6a2/dim1 = 6
;p6a2/dim2 = ss
;p6a2/dim3 = fs
p6a2/min_fs = 0
p6a2/min_ss = 3200
p6a2/max_fs = 127
p6a2/max_ss = 3263

;p6a3/dim1 = 6
;p6a3/dim2 = ss
;p6a3/dim3 = fs
p6a3/min_fs = 0
p6a3/min_ss = 3264
p6a3/max_fs = 127
p6a3/max_ss = 3327

;p6a4/dim1 = 6
;p6a4/dim2 = ss
;p6a4/dim3 = fs
p6a4/min_fs = 0
p6a4/min_ss = 3328
p6a4/max_fs = 127
p6a4/max_ss = 3391

;p6a5/dim1 = 6
;p6a5/dim2 = ss
;p6a5/dim3 = fs
p6a5/min_fs = 0
p6a5/min_ss = 3392
p6a5/max_fs = 127
p6a5/max_ss = 3455

;p6a6/dim1 = 6
;p6a6/dim2 = ss
;p6a6/dim3 = fs
p6a6/min_fs = 0
p6a6/min_ss = 3456
p6a6/max_fs = 127
p6a6/max_ss = 3519

;p6a7/dim1 = 6
;p6a7/dim2 = ss
;p6a7/dim3 = fs
p6a7/min_fs = 0
p6a7/min_ss = 3520
p6a7/max_fs = 127
p6a7/max_ss = 3583

;p7a0/dim1 = 7
;p7a0/dim2 = ss
;p7a0/dim3 = fs
p7a0/min_fs = 0
p7a0/min_ss = 3584
p7a0/max_fs = 127
p7a0/max_ss = 3647

;p7a1/dim1 = 7
;p7a1/dim2 = ss
;p7a1/dim3 = fs
p7a1/min_fs = 0
p7a1/min_ss = 3648
p7a1/max_fs = 127
p7a1/max_ss = 3711

;p7a2/dim1 = 7
;p7a2/dim2 = ss
;p7a2/dim3 = fs
p7a2/min_fs = 0
p7a2/min_ss = 3712
p7a2/max_fs = 127
p7a2/max_ss = 3775

;p7a3/dim1 = 7
;p7a3/dim2 = ss
;p7a3/dim3 = fs
p7a3/min_fs = 0
p7a3/min_ss = 3776
p7a3/max_fs = 127
p7a3/max_ss = 3839

;p7a4/dim1 = 7
;p7a4/dim2 = ss
;p7a4/dim3 = fs
p7a4/min_fs = 0
p7a4/min_ss = 3840
p7a4/max_fs = 127
p7a4/max_ss = 3903

;p7a5/dim1 = 7
;p7a5/dim2 = ss
;p7a5/dim3 = fs
p7a5/min_fs = 0
p7a5/min_ss = 3904
p7a5/max_fs = 127
p7a5/max_ss = 3967

;p7a6/dim1 = 7
;p7a6/dim2 = ss
;p7a6/dim3 = fs
p7a6/min_fs = 0
p7a6/min_ss = 3968
p7a6/max_fs = 127
p7a6/max_ss = 4031

;p7a7/dim1 = 7
;p7a7/dim2 = ss
;p7a7/dim3 = fs
p7a7/min_fs = 0
p7a7/min_ss = 4032
p7a7/max_fs = 127
p7a7/max_ss = 4095

;p8a0/dim1 = 8
;p8a0/dim2 = ss
;p8a0/dim3 = fs
p8a0/min_fs = 0
p8a0/min_ss = 4096
p8a0/max_fs = 127
p8a0/max_ss = 4159

;p8a1/dim1 = 8
;p8a1/dim2 = ss
;p8a1/dim3 = fs
p8a1/min_fs = 0
p8a1/min_ss = 4160
p8a1/max_fs = 127
p8a1/max_ss = 4223

;p8a2/dim1 = 8
;p8a2/dim2 = ss
;p8a2/dim3 = fs
p8a2/min_fs = 0
p8a2/min_ss = 4224
p8a2/max_fs = 127
p8a2/max_ss = 4287

;p8a3/dim1 = 8
;p8a3/dim2 = ss
;p8a3/dim3 = fs
p8a3/min_fs = 0
p8a3/min_ss = 4288
p8a3/max_fs = 127
p8a3/max_ss = 4351





;p8a7/dim1 = 8
;p8a7/dim2 = ss
;p8a7/dim3 = fs
p8a7/min_fs = 0
p8a7/min_ss = 4544
p8a7/max_fs = 127
p8a7/max_ss = 4607

;p9a0/dim1 = 9
;p9a0/dim2 = ss
;p9a0/dim3 = fs
p9a0/min_fs = 0
p9a0/min_ss = 4608
p9a0/max_fs = 127
p9a0/max_ss = 4671

;p9a1/dim1 = 9
;p9a1/dim2 = ss
;p9a1/dim3 = fs
p9a1/min_fs = 0
p9a1/min_ss = 4672
p9a1/max_fs = 127
p9a1/max_ss = 4735

;p9a2/dim1 = 9
;p9a2/dim2 = ss
;p9a2/dim3 = fs
p9a2/min_fs = 0
p9a2/min_ss = 4736
p9a2/max_fs = 127
p9a2/max_ss = 4799

;p9a3/dim1 = 9
;p9a3/dim2 = ss
;p9a3/dim3 = fs
p9a3/min_fs = 0
p9a3/min_ss = 4800
p9a3/max_fs = 127
p9a3/max_ss = 4863

;p9a4/dim1 = 9
;p9a4/dim2 = ss
;p9a4/dim3 = fs
p9a4/min_fs = 0
p9a4/min_ss = 4864
p9a4/max_fs = 127
p9a4/max_ss = 4927

;p9a5/dim1 = 9
;p9a5/dim2 = ss
;p9a5/dim3 = fs
p9a5/min_fs = 0
p9a5/min_ss = 4928
p9a5/max_fs = 127
p9a5/max_ss = 4991


;p9a7/dim1 = 9
;p9a7/dim2 = ss
;p9a7/dim3 = fs
p9a7/min_fs = 0
p9a7/min_ss = 5056
p9a7/max_fs = 127
p9a7/max_ss = 5119

;p10a0/dim1 = 10
;p10a0/dim2 = ss
;p10a0/dim3 = fs
p10a0/min_fs = 0
p10a0/min_ss = 5120
p10a0/max_fs = 127
p10a0/max_ss = 5183

;p10a1/dim1 = 10
;p10a1/dim2 = ss
;p10a1/dim3 = fs
p10a1/min_fs = 0
p10a1/min_ss = 5184
p10a1/max_fs = 127
p10a1/max_ss = 5247

;p10a2/dim1 = 10
;p10a2/dim2 = ss
;p10a2/dim3 = fs
p10a2/min_fs = 0
p10a2/min_ss = 5248
p10a2/max_fs = 127
p10a2/max_ss = 5311

;p10a3/dim1 = 10
;p10a3/dim2 = ss
;p10a3/dim3 = fs
p10a3/min_fs = 0
p10a3/min_ss = 5312
p10a3/max_fs = 127
p10a3/max_ss = 5375

;p10a4/dim1 = 10
;p10a4/dim2 = ss
;p10a4/dim3 = fs
p10a4/min_fs = 0
p10a4/min_ss = 5376
p10a4/max_fs = 127
p10a4/max_ss = 5439

;p10a5/dim1 = 10
;p10a5/dim2 = ss
;p10a5/dim3 = fs
p10a5/min_fs = 0
p10a5/min_ss = 5440
p10a5/max_fs = 127
p10a5/max_ss = 5503

;p10a6/dim1 = 10
;p10a6/dim2 = ss
;p10a6/dim3 = fs
p10a6/min_fs = 0
p10a6/min_ss = 5504
p10a6/max_fs = 127
p10a6/max_ss = 5567

;p10a7/dim1 = 10
;p10a7/dim2 = ss
;p10a7/dim3 = fs
p10a7/min_fs = 0
p10a7/min_ss = 5568
p10a7/max_fs = 127
p10a7/max_ss = 5631

;p11a0/dim1 = 11
;p11a0/dim2 = ss
;p11a0/dim3 = fs
p11a0/min_fs = 0
p11a0/min_ss = 5632
p11a0/max_fs = 127
p11a0/max_ss = 5695

;p11a1/dim1 = 11
;p11a1/dim2 = ss
;p11a1/dim3 = fs
p11a1/min_fs = 0
p11a1/min_ss = 5696
p11a1/max_fs = 127
p11a1/max_ss = 5759

;p11a2/dim1 = 11
;p11a2/dim2 = ss
;p11a2/dim3 = fs
p11a2/min_fs = 0
p11a2/min_ss = 5760
p11a2/max_fs = 127
p11a2/max_ss = 5823

;p11a3/dim1 = 11
;p11a3/dim2 = ss
;p11a3/dim3 = fs
p11a3/min_fs = 0
p11a3/min_ss = 5824
p11a3/max_fs = 127
p11a3/max_ss = 5887

;p11a4/dim1 = 11
;p11a4/dim2 = ss
;p11a4/dim3 = fs
p11a4/min_fs = 0
p11a4/min_ss = 5888
p11a4/max_fs = 127
p11a4/max_ss = 5951

;p11a5/dim1 = 11
;p11a5/dim2 = ss
;p11a5/dim3 = fs
p11a5/min_fs = 0
p11a5/min_ss = 5952
p11a5/max_fs = 127
p11a5/max_ss = 6015

;p11a6/dim1 = 11
;p11a6/dim2 = ss
;p11a6/dim3 = fs
p11a6/min_fs = 0
p11a6/min_ss = 6016
p11a6/max_fs = 127
p11a6/max_ss = 6079

;p11a7/dim1 = 11
;p11a7/dim2 = ss
;p11a7/dim3 = fs
p11a7/min_fs = 0
p11a7/min_ss = 6080
p11a7/max_fs = 127
p11a7/max_ss = 6143

;p12a0/dim1 = 12
;p12a0/dim2 = ss
;p12a0/dim3 = fs
p12a0/min_fs = 0
p12a0/min_ss = 6144
p12a0/max_fs = 127
p12a0/max_ss = 6207

;p12a1/dim1 = 12
;p12a1/dim2 = ss
;p12a1/dim3 = fs
p12a1/min_fs = 0
p12a1/min_ss = 6208
p12a1/max_fs = 127
p12a1/max_ss = 6271

;p12a2/dim1 = 12
;p12a2/dim2 = ss
;p12a2/dim3 = fs
p12a2/min_fs = 0
p12a2/min_ss = 6272
p12a2/max_fs = 127
p12a2/max_ss = 6335

;p12a3/dim1 = 12
;p12a3/dim2 = ss
;p12a3/dim3 = fs
p12a3/min_fs = 0
p12a3/min_ss = 6336
p12a3/max_fs = 127
p12a3/max_ss = 6399

;p12a4/dim1 = 12
;p12a4/dim2 = ss
;p12a4/dim3 = fs
p12a4/min_fs = 0
p12a4/min_ss = 6400
p12a4/max_fs = 127
p12a4/max_ss = 6463

;p12a5/dim1 = 12
;p12a5/dim2 = ss
;p12a5/dim3 = fs
p12a5/min_fs = 0
p12a5/min_ss = 6464
p12a5/max_fs = 127
p12a5/max_ss = 6527

;p12a6/dim1 = 12
;p12a6/dim2 = ss
;p12a6/dim3 = fs
p12a6/min_fs = 0
p12a6/min_ss = 6528
p12a6/max_fs = 127
p12a6/max_ss = 6591

;p12a7/dim1 = 12
;p12a7/dim2 = ss
;p12a7/dim3 = fs
p12a7/min_fs = 0
p12a7/min_ss = 6592
p12a7/max_fs = 127
p12a7/max_ss = 6655

;p13a0/dim1 = 13
;p13a0/dim2 = ss
;p13a0/dim3 = fs
p13a0/min_fs = 0
p13a0/min_ss = 6656
p13a0/max_fs = 127
p13a0/max_ss = 6719

;p13a1/dim1 = 13
;p13a1/dim2 = ss
;p13a1/dim3 = fs
p13a1/min_fs = 0
p13a1/min_ss = 6720
p13a1/max_fs = 127
p13a1/max_ss = 6783

;p13a2/dim1 = 13
;p13a2/dim2 = ss
;p13a2/dim3 = fs
p13a2/min_fs = 0
p13a2/min_ss = 6784
p13a2/max_fs = 127
p13a2/max_ss = 6847

;p13a3/dim1 = 13
;p13a3/dim2 = ss
;p13a3/dim3 = fs
p13a3/min_fs = 0
p13a3/min_ss = 6848
p13a3/max_fs = 127
p13a3/max_ss = 6911

;p13a4/dim1 = 13
;p13a4/dim2 = ss
;p13a4/dim3 = fs
p13a4/min_fs = 0
p13a4/min_ss = 6912
p13a4/max_fs = 127
p13a4/max_ss = 6975

;p13a5/dim1 = 13
;p13a5/dim2 = ss
;p13a5/dim3 = fs
p13a5/min_fs = 0
p13a5/min_ss = 6976
p13a5/max_fs = 127
p13a5/max_ss = 7039

;p13a6/dim1 = 13
;p13a6/dim2 = ss
;p13a6/dim3 = fs
p13a6/min_fs = 0
p13a6/min_ss = 7040
p13a6/max_fs = 127
p13a6/max_ss = 7103

;p13a7/dim1 = 13
;p13a7/dim2 = ss
;p13a7/dim3 = fs
p13a7/min_fs = 0
p13a7/min_ss = 7104
p13a7/max_fs = 127
p13a7/max_ss = 7167

;p14a0/dim1 = 14
;p14a0/dim2 = ss
;p14a0/dim3 = fs
p14a0/min_fs = 0
p14a0/min_ss = 7168
p14a0/max_fs = 127
p14a0/max_ss = 7231

;p14a1/dim1 = 14
;p14a1/dim2 = ss
;p14a1/dim3 = fs
p14a1/min_fs = 0
p14a1/min_ss = 7232
p14a1/max_fs = 127
p14a1/max_ss = 7295

;p14a2/dim1 = 14
;p14a2/dim2 = ss
;p14a2/dim3 = fs
p14a2/min_fs = 0
p14a2/min_ss = 7296
p14a2/max_fs = 127
p14a2/max_ss = 7359

;p14a3/dim1 = 14
;p14a3/dim2 = ss
;p14a3/dim3 = fs
p14a3/min_fs = 0
p14a3/min_ss = 7360
p14a3/max_fs = 127
p14a3/max_ss = 7423

;p14a4/dim1 = 14
;p14a4/dim2 = ss
;p14a4/dim3 = fs
p14a4/min_fs = 0
p14a4/min_ss = 7424
p14a4/max_fs = 127
p14a4/max_ss = 7487

;p14a5/dim1 = 14
;p14a5/dim2 = ss
;p14a5/dim3 = fs
p14a5/min_fs = 0
p14a5/min_ss = 7488
p14a5/max_fs = 127
p14a5/max_ss = 7551

;p14a6/dim1 = 14
;p14a6/dim2 = ss
;p14a6/dim3 = fs
p14a6/min_fs = 0
p14a6/min_ss = 7552
p14a6/max_fs = 127
p14a6/max_ss = 7615


;p15a0/dim1 = 15
;p15a0/dim2 = ss
;p15a0/dim3 = fs
p15a0/min_fs = 0
p15a0/min_ss = 7680
p15a0/max_fs = 127
p15a0/max_ss = 7743

;p15a1/dim1 = 15
;p15a1/dim2 = ss
;p15a1/dim3 = fs
p15a1/min_fs = 0
p15a1/min_ss = 7744
p15a1/max_fs = 127
p15a1/max_ss = 7807

;p15a2/dim1 = 15
;p15a2/dim2 = ss
;p15a2/dim3 = fs
p15a2/min_fs = 0
p15a2/min_ss = 7808
p15a2/max_fs = 127
p15a2/max_ss = 7871

;p15a3/dim1 = 15
;p15a3/dim2 = ss
;p15a3/dim3 = fs
p15a3/min_fs = 0
p15a3/min_ss = 7872
p15a3/max_fs = 127
p15a3/max_ss = 7935

;p15a4/dim1 = 15
;p15a4/dim2 = ss
;p15a4/dim3 = fs
p15a4/min_fs = 0
p15a4/min_ss = 7936
p15a4/max_fs = 127
p15a4/max_ss = 7999

;p15a5/dim1 = 15
;p15a5/dim2 = ss
;p15a5/dim3 = fs
p15a5/min_fs = 0
p15a5/min_ss = 8000
p15a5/max_fs = 127
p15a5/max_ss = 8063

;p15a6/dim1 = 15
;p15a6/dim2 = ss
;p15a6/dim3 = fs
p15a6/min_fs = 0
p15a6/min_ss = 8064
p15a6/max_fs = 127
p15a6/max_ss = 8127

;p15a7/dim1 = 15
;p15a7/dim2 = ss
;p15a7/dim3 = fs
p15a7/min_fs = 0
p15a7/min_ss = 8128
p15a7/max_fs = 127
p15a7/max_ss = 8191

p0a0/fs = -0.005061x -0.999983y
p0a0/ss = +0.999983x -0.005061y
p0a0/corner_x = -516.4836744702812
p0a0/corner_y = 636.0854003962355
p0a1/fs = -0.005061x -0.999983y
p0a1/ss = +0.999983x -0.005061y
p0a1/corner_x = -450.48467447028105
p0a1/corner_y = 635.7294003962355
p0a2/fs = -0.005061x -0.999983y
p0a2/ss = +0.999983x -0.005061y
p0a2/corner_x = -384.49167447028105
p0a2/corner_y = 635.3724003962355
p0a3/fs = -0.005061x -0.999983y
p0a3/ss = +0.999983x -0.005061y
p0a3/corner_x = -318.4906744702811
p0a3/corner_y = 635.0174003962355
p0a4/fs = -0.005061x -0.999983y
p0a4/ss = +0.999983x -0.005061y
p0a4/corner_x = -252.49167447028103
p0a4/corner_y = 634.6594003962355
p0a5/fs = -0.005061x -0.999983y
p0a5/ss = +0.999983x -0.005061y
p0a5/corner_x = -186.48967447028107
p0a5/corner_y = 634.4124003962355
p0a6/fs = -0.005061x -0.999983y
p0a6/ss = +0.999983x -0.005061y
p0a6/corner_x = -120.49067447028106
p0a6/corner_y = 634.0814003962355
p0a7/fs = -0.005061x -0.999983y
p0a7/ss = +0.999983x -0.005061y
p0a7/corner_x = -54.492074470281075
p0a7/corner_y = 633.7454003962355
p1a0/fs = -0.002979x -0.999997y
p1a0/ss = +0.999997x -0.002979y
p1a0/corner_x = -516.8346744702812
p1a0/corner_y = 477.5424003962354
p1a1/fs = -0.002979x -0.999997y
p1a1/ss = +0.999997x -0.002979y
p1a1/corner_x = -450.83767447028106
p1a1/corner_y = 477.3584003962354
p1a2/fs = -0.002979x -0.999997y
p1a2/ss = +0.999997x -0.002979y
p1a2/corner_x = -384.84067447028104
p1a2/corner_y = 477.1724003962354
p1a3/fs = -0.002979x -0.999997y
p1a3/ss = +0.999997x -0.002979y
p1a3/corner_x = -318.8436744702811
p1a3/corner_y = 476.9894003962354
p1a4/fs = -0.002979x -0.999997y
p1a4/ss = +0.999997x -0.002979y
p1a4/corner_x = -252.84467447028103
p1a4/corner_y = 476.8064003962354
p1a5/fs = -0.002979x -0.999997y
p1a5/ss = +0.999997x -0.002979y
p1a5/corner_x = -186.84867447028105
p1a5/corner_y = 476.6184003962354
p1a6/fs = -0.002979x -0.999997y
p1a6/ss = +0.999997x -0.002979y
p1a6/corner_x = -120.83467447028107
p1a6/corner_y = 476.4334003962354
p1a7/fs = -0.002979x -0.999997y
p1a7/ss = +0.999997x -0.002979y
p1a7/corner_x = -54.832574470281074
p1a7/corner_y = 476.2494003962354
p2a0/fs = -0.005902x -0.999985y
p2a0/ss = +0.999985x -0.005902y
p2a0/corner_x = -515.7476744702813
p2a0/corner_y = 323.75240039623543
p2a1/fs = -0.005902x -0.999985y
p2a1/ss = +0.999985x -0.005902y
p2a1/corner_x = -449.75067447028107
p2a1/corner_y = 323.3554003962354
p2a2/fs = -0.005902x -0.999985y
p2a2/ss = +0.999985x -0.005902y
p2a2/corner_x = -383.75167447028105
p2a2/corner_y = 322.9584003962354
p2a3/fs = -0.005902x -0.999985y
p2a3/ss = +0.999985x -0.005902y
p2a3/corner_x = -317.75367447028106
p2a3/corner_y = 322.5594003962354
p2a4/fs = -0.005902x -0.999985y
p2a4/ss = +0.999985x -0.005902y
p2a4/corner_x = -251.75567447028104
p2a4/corner_y = 322.1594003962354
p2a5/fs = -0.005902x -0.999985y
p2a5/ss = +0.999985x -0.005902y
p2a5/corner_x = -185.75967447028106
p2a5/corner_y = 321.76440039623543
p2a6/fs = -0.005902x -0.999985y
p2a6/ss = +0.999985x -0.005902y
p2a6/corner_x = -119.74567447028107
p2a6/corner_y = 321.3634003962354

p3a0/fs = +0.001827x -0.999997y
p3a0/ss = +0.999997x +0.001827y
p3a0/corner_x = -516.5986744702813
p3a0/corner_y = 162.4604003962353
p3a1/fs = +0.001827x -0.999997y
p3a1/ss = +0.999997x +0.001827y
p3a1/corner_x = -450.6036744702811
p3a1/corner_y = 162.5864003962353
p3a2/fs = +0.001827x -0.999997y
p3a2/ss = +0.999997x +0.001827y
p3a2/corner_x = -384.60467447028105
p3a2/corner_y = 162.7074003962353
p3a3/fs = +0.001827x -0.999997y
p3a3/ss = +0.999997x +0.001827y
p3a3/corner_x = -318.60667447028106
p3a3/corner_y = 162.8294003962353
p3a4/fs = +0.001827x -0.999997y
p3a4/ss = +0.999997x +0.001827y
p3a4/corner_x = -252.60867447028104
p3a4/corner_y = 162.9494003962353


p3a7/fs = +0.001827x -0.999997y
p3a7/ss = +0.999997x +0.001827y
p3a7/corner_x = -54.59127447028108
p3a7/corner_y = 163.3184003962353
p4a0/fs = +0.001868x -0.999999y
p4a0/ss = +0.999999x +0.001868y
p4a0/corner_x = -557.4698998326617
p4a0/corner_y = -10.729882473625876
p4a1/fs = +0.001868x -0.999999y
p4a1/ss = +0.999999x +0.001868y
p4a1/corner_x = -491.4748998326619
p4a1/corner_y = -10.598192473625877
p4a2/fs = +0.001868x -0.999999y
p4a2/ss = +0.999999x +0.001868y
p4a2/corner_x = -425.4778998326619
p4a2/corner_y = -10.466602473625878
p4a3/fs = +0.001868x -0.999999y
p4a3/ss = +0.999999x +0.001868y
p4a3/corner_x = -359.48189983266184
p4a3/corner_y = -10.335022473625878
p4a4/fs = +0.001868x -0.999999y
p4a4/ss = +0.999999x +0.001868y
p4a4/corner_x = -293.4828998326619
p4a4/corner_y = -10.203422473625876
p4a5/fs = +0.001868x -0.999999y
p4a5/ss = +0.999999x +0.001868y
p4a5/corner_x = -227.48389983266173
p4a5/corner_y = -10.071852473625876
p4a6/fs = +0.001868x -0.999999y
p4a6/ss = +0.999999x +0.001868y
p4a6/corner_x = -161.48589983266172
p4a6/corner_y = -9.940172473625877
p4a7/fs = +0.001868x -0.999999y
p4a7/ss = +0.999999x +0.001868y
p4a7/corner_x = -95.48929983266177
p4a7/corner_y = -9.808682473625877
p5a0/fs = +0.004620x -0.999988y
p5a0/ss = +0.999988x +0.004620y
p5a0/corner_x = -557.5018998326617
p5a0/corner_y = -170.34252247362588
p5a1/fs = +0.004620x -0.999988y
p5a1/ss = +0.999988x +0.004620y
p5a1/corner_x = -491.50589983266184
p5a1/corner_y = -170.04252247362587
p5a2/fs = +0.004620x -0.999988y
p5a2/ss = +0.999988x +0.004620y
p5a2/corner_x = -425.50989983266186
p5a2/corner_y = -169.74452247362586
p5a3/fs = +0.004620x -0.999988y
p5a3/ss = +0.999988x +0.004620y
p5a3/corner_x = -359.51489983266185
p5a3/corner_y = -169.44852247362587
p5a4/fs = +0.004620x -0.999988y
p5a4/ss = +0.999988x +0.004620y
p5a4/corner_x = -293.5158998326619
p5a4/corner_y = -169.14852247362586
p5a5/fs = +0.004620x -0.999988y
p5a5/ss = +0.999988x +0.004620y
p5a5/corner_x = -227.50189983266173
p5a5/corner_y = -168.85452247362588
p5a6/fs = +0.004620x -0.999988y
p5a6/ss = +0.999988x +0.004620y
p5a6/corner_x = -161.50189983266173
p5a6/corner_y = -168.55652247362588
p5a7/fs = +0.004620x -0.999988y
p5a7/ss = +0.999988x +0.004620y
p5a7/corner_x = -95.50149983266178
p5a7/corner_y = -168.25752247362587
p6a0/fs = +0.000168x -1.000001y
p6a0/ss = +1.000001x +0.000168y
p6a0/corner_x = -557.3348998326617
p6a0/corner_y = -325.0825224736259
p6a1/fs = +0.000168x -1.000001y
p6a1/ss = +1.000001x +0.000168y
p6a1/corner_x = -491.33689983266186
p6a1/corner_y = -325.0755224736259
p6a2/fs = +0.000168x -1.000001y
p6a2/ss = +1.000001x +0.000168y
p6a2/corner_x = -425.33989983266184
p6a2/corner_y = -325.0775224736259
p6a3/fs = +0.000168x -1.000001y
p6a3/ss = +1.000001x +0.000168y
p6a3/corner_x = -359.34189983266185
p6a3/corner_y = -325.0745224736259
p6a4/fs = +0.000168x -1.000001y
p6a4/ss = +1.000001x +0.000168y
p6a4/corner_x = -293.34689983266185
p6a4/corner_y = -325.0715224736259
p6a5/fs = +0.000168x -1.000001y
p6a5/ss = +1.000001x +0.000168y
p6a5/corner_x = -227.34689983266173
p6a5/corner_y = -325.0685224736259
p6a6/fs = +0.000168x -1.000001y
p6a6/ss = +1.000001x +0.000168y
p6a6/corner_x = -161.34989983266172
p6a6/corner_y = -325.06652247362587
p6a7/fs = +0.000168x -1.000001y
p6a7/ss = +1.000001x +0.000168y
p6a7/corner_x = -95.34979983266177
p6a7/corner_y = -325.0645224736259
p7a0/fs = +0.003748x -0.999992y
p7a0/ss = +0.999992x +0.003748y
p7a0/corner_x = -556.4758998326616
p7a0/corner_y = -483.2115224736259
p7a1/fs = +0.003748x -0.999992y
p7a1/ss = +0.999992x +0.003748y
p7a1/corner_x = -490.4828998326619
p7a1/corner_y = -482.9665224736259
p7a2/fs = +0.003748x -0.999992y
p7a2/ss = +0.999992x +0.003748y
p7a2/corner_x = -424.48789983266187
p7a2/corner_y = -482.7245224736259
p7a3/fs = +0.003748x -0.999992y
p7a3/ss = +0.999992x +0.003748y
p7a3/corner_x = -358.4898998326619
p7a3/corner_y = -482.4815224736259
p7a4/fs = +0.003748x -0.999992y
p7a4/ss = +0.999992x +0.003748y
p7a4/corner_x = -292.49389983266184
p7a4/corner_y = -482.2345224736259
p7a5/fs = +0.003748x -0.999992y
p7a5/ss = +0.999992x +0.003748y
p7a5/corner_x = -226.4968998326617
p7a5/corner_y = -481.9915224736259
p7a6/fs = +0.003748x -0.999992y
p7a6/ss = +0.999992x +0.003748y
p7a6/corner_x = -160.49789983266172
p7a6/corner_y = -481.7455224736259
p7a7/fs = +0.003748x -0.999992y
p7a7/ss = +0.999992x +0.003748y
p7a7/corner_x = -94.50259983266177
p7a7/corner_y = -481.5005224736259
p8a0/fs = -0.001859x +0.999998y
p8a0/ss = -0.999998x -0.001859y
p8a0/corner_x = 514.0488340685429
p8a0/corner_y = -171.08025482550443
p8a1/fs = -0.001859x +0.999998y
p8a1/ss = -0.999998x -0.001859y
p8a1/corner_x = 448.0508340685429
p8a1/corner_y = -171.20725482550444
p8a2/fs = -0.001859x +0.999998y
p8a2/ss = -0.999998x -0.001859y
p8a2/corner_x = 382.05183406854286
p8a2/corner_y = -171.33525482550442
p8a3/fs = -0.001859x +0.999998y
p8a3/ss = -0.999998x -0.001859y
p8a3/corner_x = 316.0558340685429
p8a3/corner_y = -171.46625482550442



p8a7/fs = -0.001859x +0.999998y
p8a7/ss = -0.999998x -0.001859y
p8a7/corner_x = 52.06433406854294
p8a7/corner_y = -171.97425482550443
p9a0/fs = -0.001991x +0.999998y
p9a0/ss = -0.999998x -0.001991y
p9a0/corner_x = 514.3618340685429
p9a0/corner_y = -328.2352548255044
p9a1/fs = -0.001991x +0.999998y
p9a1/ss = -0.999998x -0.001991y
p9a1/corner_x = 448.36483406854285
p9a1/corner_y = -328.3702548255044
p9a2/fs = -0.001991x +0.999998y
p9a2/ss = -0.999998x -0.001991y
p9a2/corner_x = 382.36883406854287
p9a2/corner_y = -328.5062548255044
p9a3/fs = -0.001991x +0.999998y
p9a3/ss = -0.999998x -0.001991y
p9a3/corner_x = 316.37183406854285
p9a3/corner_y = -328.6392548255044
p9a4/fs = -0.001991x +0.999998y
p9a4/ss = -0.999998x -0.001991y
p9a4/corner_x = 250.37383406854295
p9a4/corner_y = -328.7752548255044
p9a5/fs = -0.001991x +0.999998y
p9a5/ss = -0.999998x -0.001991y
p9a5/corner_x = 184.37683406854296
p9a5/corner_y = -328.9102548255044

p9a7/fs = -0.001991x +0.999998y
p9a7/ss = -0.999998x -0.001991y
p9a7/corner_x = 52.357234068542944
p9a7/corner_y = -329.1802548255044
p10a0/fs = -0.003377x +0.999992y
p10a0/ss = -0.999992x -0.003377y
p10a0/corner_x = 514.0268340685428
p10a0/corner_y = -484.9192548255044
p10a1/fs = -0.003377x +0.999992y
p10a1/ss = -0.999992x -0.003377y
p10a1/corner_x = 448.0318340685429
p10a1/corner_y = -485.15225482550443
p10a2/fs = -0.003377x +0.999992y
p10a2/ss = -0.999992x -0.003377y
p10a2/corner_x = 382.0338340685429
p10a2/corner_y = -485.3832548255044
p10a3/fs = -0.003377x +0.999992y
p10a3/ss = -0.999992x -0.003377y
p10a3/corner_x = 316.03783406854285
p10a3/corner_y = -485.6152548255044
p10a4/fs = -0.003377x +0.999992y
p10a4/ss = -0.999992x -0.003377y
p10a4/corner_x = 250.03983406854294
p10a4/corner_y = -485.8472548255044
p10a5/fs = -0.003377x +0.999992y
p10a5/ss = -0.999992x -0.003377y
p10a5/corner_x = 184.04283406854296
p10a5/corner_y = -486.0792548255044
p10a6/fs = -0.003377x +0.999992y
p10a6/ss = -0.999992x -0.003377y
p10a6/corner_x = 118.04783406854293
p10a6/corner_y = -486.3112548255044
p10a7/fs = -0.003377x +0.999992y
p10a7/ss = -0.999992x -0.003377y
p10a7/corner_x = 52.04993406854294
p10a7/corner_y = -486.5432548255044
p11a0/fs = -0.005792x +0.999982y
p11a0/ss = -0.999982x -0.005792y
p11a0/corner_x = 514.4148340685429
p11a0/corner_y = -640.8602548255045
p11a1/fs = -0.005792x +0.999982y
p11a1/ss = -0.999982x -0.005792y
p11a1/corner_x = 448.41483406854286
p11a1/corner_y = -641.2472548255046
p11a2/fs = -0.005792x +0.999982y
p11a2/ss = -0.999982x -0.005792y
p11a2/corner_x = 382.4188340685429
p11a2/corner_y = -641.6372548255046
p11a3/fs = -0.005792x +0.999982y
p11a3/ss = -0.999982x -0.005792y
p11a3/corner_x = 316.4228340685429
p11a3/corner_y = -642.0252548255045
p11a4/fs = -0.005792x +0.999982y
p11a4/ss = -0.999982x -0.005792y
p11a4/corner_x = 250.42783406854295
p11a4/corner_y = -642.4112548255046
p11a5/fs = -0.005792x +0.999982y
p11a5/ss = -0.999982x -0.005792y
p11a5/corner_x = 184.43183406854294
p11a5/corner_y = -642.8052548255046
p11a6/fs = -0.005792x +0.999982y
p11a6/ss = -0.999982x -0.005792y
p11a6/corner_x = 118.43483406854293
p11a6/corner_y = -643.1902548255046
p11a7/fs = -0.005792x +0.999982y
p11a7/ss = -0.999982x -0.005792y
p11a7/corner_x = 52.43653406854294
p11a7/corner_y = -643.5802548255045
p12a0/fs = -0.001548x +0.999998y
p12a0/ss = -0.999998x -0.001548y
p12a0/corner_x = 552.8893045797711
p12a0/corner_y = 464.9670025408906
p12a1/fs = -0.001548x +0.999998y
p12a1/ss = -0.999998x -0.001548y
p12a1/corner_x = 486.890304579771
p12a1/corner_y = 464.8150025408906
p12a2/fs = -0.001548x +0.999998y
p12a2/ss = -0.999998x -0.001548y
p12a2/corner_x = 420.895304579771
p12a2/corner_y = 464.66700254089056
p12a3/fs = -0.001548x +0.999998y
p12a3/ss = -0.999998x -0.001548y
p12a3/corner_x = 354.891304579771
p12a3/corner_y = 464.66000254089056
p12a4/fs = -0.001548x +0.999998y
p12a4/ss = -0.999998x -0.001548y
p12a4/corner_x = 288.890304579771
p12a4/corner_y = 464.5560025408906
p12a5/fs = -0.001548x +0.999998y
p12a5/ss = -0.999998x -0.001548y
p12a5/corner_x = 222.88930457977096
p12a5/corner_y = 464.4540025408906
p12a6/fs = -0.001548x +0.999998y
p12a6/ss = -0.999998x -0.001548y
p12a6/corner_x = 156.88830457977096
p12a6/corner_y = 464.3490025408906
p12a7/fs = -0.001548x +0.999998y
p12a7/ss = -0.999998x -0.001548y
p12a7/corner_x = 90.88910457977093
p12a7/corner_y = 464.24900254089056
p13a0/fs = +0.000166x +1.000002y
p13a0/ss = -1.000002x +0.000166y
p13a0/corner_x = 552.123304579771
p13a0/corner_y = 308.7740025408906
p13a1/fs = +0.000166x +1.000002y
p13a1/ss = -1.000002x +0.000166y
p13a1/corner_x = 486.12330457977095
p13a1/corner_y = 308.7710025408906
p13a2/fs = +0.000166x +1.000002y
p13a2/ss = -1.000002x +0.000166y
p13a2/corner_x = 420.129304579771
p13a2/corner_y = 308.7710025408906
p13a3/fs = +0.000166x +1.000002y
p13a3/ss = -1.000002x +0.000166y
p13a3/corner_x = 354.13230457977096
p13a3/corner_y = 308.7720025408906
p13a4/fs = +0.000166x +1.000002y
p13a4/ss = -1.000002x +0.000166y
p13a4/corner_x = 288.13430457977097
p13a4/corner_y = 308.76800254089056
p13a5/fs = +0.000166x +1.000002y
p13a5/ss = -1.000002x +0.000166y
p13a5/corner_x = 222.12030457977096
p13a5/corner_y = 308.7690025408906
p13a6/fs = +0.000166x +1.000002y
p13a6/ss = -1.000002x +0.000166y
p13a6/corner_x = 156.11830457977098
p13a6/corner_y = 308.7700025408906
p13a7/fs = +0.000166x +1.000002y
p13a7/ss = -1.000002x +0.000166y
p13a7/corner_x = 90.11980457977093
p13a7/corner_y = 308.7690025408906
p14a0/fs = -0.004249x +0.999992y
p14a0/ss = -0.999992x -0.004249y
p14a0/corner_x = 553.796304579771
p14a0/corner_y = 153.7270025408907
p14a1/fs = -0.004249x +0.999992y
p14a1/ss = -0.999992x -0.004249y
p14a1/corner_x = 487.79830457977096
p14a1/corner_y = 153.4440025408907
p14a2/fs = -0.004249x +0.999992y
p14a2/ss = -0.999992x -0.004249y
p14a2/corner_x = 421.804304579771
p14a2/corner_y = 153.1590025408907
p14a3/fs = -0.004249x +0.999992y
p14a3/ss = -0.999992x -0.004249y
p14a3/corner_x = 355.80530457977096
p14a3/corner_y = 152.8760025408907
p14a4/fs = -0.004249x +0.999992y
p14a4/ss = -0.999992x -0.004249y
p14a4/corner_x = 289.80730457977097
p14a4/corner_y = 152.5930025408907
p14a5/fs = -0.004249x +0.999992y
p14a5/ss = -0.999992x -0.004249y
p14a5/corner_x = 223.79430457977097
p14a5/corner_y = 152.3100025408907
p14a6/fs = -0.004249x +0.999992y
p14a6/ss = -0.999992x -0.004249y
p14a6/corner_x = 157.79430457977097
p14a6/corner_y = 152.0260025408907

p15a0/fs = -0.001372x +1.000003y
p15a0/ss = -1.000003x -0.001372y
p15a0/corner_x = 554.122304579771
p15a0/corner_y = -4.1768974591093295
p15a1/fs = -0.001372x +1.000003y
p15a1/ss = -1.000003x -0.001372y
p15a1/corner_x = 488.12530457977095
p15a1/corner_y = -4.2736774591093285
p15a2/fs = -0.001372x +1.000003y
p15a2/ss = -1.000003x -0.001372y
p15a2/corner_x = 422.129304579771
p15a2/corner_y = -4.370237459109329
p15a3/fs = -0.001372x +1.000003y
p15a3/ss = -1.000003x -0.001372y
p15a3/corner_x = 356.128304579771
p15a3/corner_y = -4.466837459109329
p15a4/fs = -0.001372x +1.000003y
p15a4/ss = -1.000003x -0.001372y
p15a4/corner_x = 290.129304579771
p15a4/corner_y = -4.563407459109329
p15a5/fs = -0.001372x +1.000003y
p15a5/ss = -1.000003x -0.001372y
p15a5/corner_x = 224.11330457977098
p15a5/corner_y = -4.660107459109329
p15a6/fs = -0.001372x +1.000003y
p15a6/ss = -1.000003x -0.001372y
p15a6/corner_x = 158.11230457977098
p15a6/corner_y = -4.75670745910933
p15a7/fs = -0.001372x +1.000003y
p15a7/ss = -1.000003x -0.001372y
p15a7/corner_x = 92.11190457977094
p15a7/corner_y = -4.85336745910933















p0a0/coffset = 0.000409
p0a1/coffset = 0.000409
p0a2/coffset = 0.000409
p0a3/coffset = 0.000409
p0a4/coffset = 0.000409
p0a5/coffset = 0.000409
p0a6/coffset = 0.000409
p0a7/coffset = 0.000409
p1a0/coffset = 0.000200
p1a1/coffset = 0.000200
p1a2/coffset = 0.000200
p1a3/coffset = 0.000200
p1a4/coffset = 0.000200
p1a5/coffset = 0.000200
p1a6/coffset = 0.000200
p1a7/coffset = 0.000200
p2a0/coffset = -0.000147
p2a1/coffset = -0.000147
p2a2/coffset = -0.000147
p2a3/coffset = -0.000147
p2a4/coffset = -0.000147
p2a5/coffset = -0.000147
p2a6/coffset = -0.000147

p3a0/coffset = -0.000102
p3a1/coffset = -0.000102
p3a2/coffset = -0.000102
p3a3/coffset = -0.000102
p3a4/coffset = -0.000102

p3a7/coffset = -0.000102
p4a0/coffset = -0.000022
p4a1/coffset = -0.000022
p4a2/coffset = -0.000022
p4a3/coffset = -0.000022
p4a4/coffset = -0.000022
p4a5/coffset = -0.000022
p4a6/coffset = -0.000022
p4a7/coffset = -0.000022
p5a0/coffset = -0.000063
p5a1/coffset = -0.000063
p5a2/coffset = -0.000063
p5a3/coffset = -0.000063
p5a4/coffset = -0.000063
p5a5/coffset = -0.000063
p5a6/coffset = -0.000063
p5a7/coffset = -0.000063
p6a0/coffset = 0.000009
p6a1/coffset = 0.000009
p6a2/coffset = 0.000009
p6a3/coffset = 0.000009
p6a4/coffset = 0.000009
p6a5/coffset = 0.000009
p6a6/coffset = 0.000009
p6a7/coffset = 0.000009
p7a0/coffset = 0.000225
p7a1/coffset = 0.000225
p7a2/coffset = 0.000225
p7a3/coffset = 0.000225
p7a4/coffset = 0.000225
p7a5/coffset = 0.000225
p7a6/coffset = 0.000225
p7a7/coffset = 0.000225
p8a0/coffset = -0.000175
p8a1/coffset = -0.000175
p8a2/coffset = -0.000175
p8a3/coffset = -0.000175



p8a7/coffset = -0.000175
p9a0/coffset = 0.000045
p9a1/coffset = 0.000045
p9a2/coffset = 0.000045
p9a3/coffset = 0.000045
p9a4/coffset = 0.000045
p9a5/coffset = 0.000045

p9a7/coffset = 0.000045
p10a0/coffset = 0.000137
p10a1/coffset = 0.000137
p10a2/coffset = 0.000137
p10a3/coffset = 0.000137
p10a4/coffset = 0.000137
p10a5/coffset = 0.000137
p10a6/coffset = 0.000137
p10a7/coffset = 0.000137
p11a0/coffset = 0.000023
p11a1/coffset = 0.000023
p11a2/coffset = 0.000023
p11a3/coffset = 0.000023
p11a4/coffset = 0.000023
p11a5/coffset = 0.000023
p11a6/coffset = 0.000023
p11a7/coffset = 0.000023
p12a0/coffset = -0.000122
p12a1/coffset = -0.000122
p12a2/coffset = -0.000122
p12a3/coffset = -0.000122
p12a4/coffset = -0.000122
p12a5/coffset = -0.000122
p12a6/coffset = -0.000122
p12a7/coffset = -0.000122
p13a0/coffset = 0.000058
p13a1/coffset = 0.000058
p13a2/coffset = 0.000058
p13a3/coffset = 0.000058
p13a4/coffset = 0.000058
p13a5/coffset = 0.000058
p13a6/coffset = 0.000058
p13a7/coffset = 0.000058
p14a0/coffset = -0.000071
p14a1/coffset = -0.000071
p14a2/coffset = -0.000071
p14a3/coffset = -0.000071
p14a4/coffset = -0.000071
p14a5/coffset = -0.000071
p14a6/coffset = -0.000071

p15a0/coffset = -0.000453
p15a1/coffset = -0.000453
p15a2/coffset = -0.000453
p15a3/coffset = -0.000453
p15a4/coffset = -0.000453
p15a5/coffset = -0.000453
p15a6/coffset = -0.000453
p15a7/coffset = -0.000453
