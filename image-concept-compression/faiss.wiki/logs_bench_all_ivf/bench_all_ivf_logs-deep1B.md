# Detailed logs for dataset deep1B

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fsr` </summary>
Index size 16700933580

 code_size 8

 log filename: autotune.dbdeep1B.OPQ16_64_IVF1048576_HNSW32_PQ16x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2093 0.4282 0.5864      0.01066      101203470    29
nprobe=1,quantizer_efSearch=4            0.1049 0.1905 0.2272      0.00239       12906801    126
nprobe=2,quantizer_efSearch=4            0.1280 0.2448 0.3071      0.00281       25640861    107
nprobe=4,quantizer_efSearch=4            0.1447 0.2905 0.3838      0.00346       51291060    87
nprobe=2,quantizer_efSearch=8            0.1540 0.2872 0.3583      0.00379       25678350    80
nprobe=4,quantizer_efSearch=8            0.1762 0.3485 0.4576      0.00449       51219841    67
nprobe=8,quantizer_efSearch=4            0.1871 0.3835 0.5243      0.00546      102523055    55
nprobe=8,quantizer_efSearch=8            0.1912 0.3946 0.5402      0.00584      102385112    52
nprobe=8,quantizer_efSearch=16           0.2038 0.4188 0.5740      0.00757      101707695    40
nprobe=16,quantizer_efSearch=8           0.2072 0.4370 0.6295      0.00902      203598546    34
nprobe=16,quantizer_efSearch=16          0.2156 0.4542 0.6511      0.01003      202890491    30
nprobe=16,quantizer_efSearch=32          0.2219 0.4659 0.6669      0.01310      201816097    23
nprobe=32,quantizer_efSearch=16          0.2247 0.4797 0.7145      0.01497      403431234    21
nprobe=32,quantizer_efSearch=32          0.2291 0.4902 0.7284      0.01741      401487186    18
nprobe=32,quantizer_efSearch=64          0.2307 0.4943 0.7335      0.02288      399786080    14
nprobe=64,quantizer_efSearch=32          0.2325 0.5018 0.7629      0.03031      797293909    10
nprobe=64,quantizer_efSearch=64          0.2343 0.5059 0.7707      0.03398      793241651    9
nprobe=64,quantizer_efSearch=128         0.2348 0.5067 0.7724      0.04226      790708818    8
nprobe=128,quantizer_efSearch=64         0.2355 0.5121 0.7905      0.05664     1572309882    6
nprobe=128,quantizer_efSearch=128        0.2365 0.5137 0.7930      0.06360     1565324483    5
nprobe=256,quantizer_efSearch=256        0.2366 0.5161 0.8029      0.12556     3079908245    3
```

</details>
<details><summary>`OPQ16_64,IVF4194304_HNSW32,PQ16x4fsr` </summary>
Index size 18810180044

 code_size 8

 log filename: autotune.dbdeep1B.OPQ16_64_IVF4194304_HNSW32_PQ16x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2369 0.5102 0.7677      0.07945      884997726    4
nprobe=1,quantizer_efSearch=4            0.0925 0.1552 0.1800      0.00282        3730486    107
nprobe=2,quantizer_efSearch=4            0.1133 0.2016 0.2436      0.00310        7437656    97
nprobe=4,quantizer_efSearch=4            0.1271 0.2433 0.3060      0.00371       14744968    81
nprobe=2,quantizer_efSearch=8            0.1448 0.2586 0.3066      0.00440        7410325    69
nprobe=4,quantizer_efSearch=8            0.1627 0.3133 0.3878      0.00504       14728160    60
nprobe=8,quantizer_efSearch=4            0.1707 0.3459 0.4486      0.00548       29368624    55
nprobe=8,quantizer_efSearch=8            0.1789 0.3600 0.4670      0.00594       29303423    51
nprobe=16,quantizer_efSearch=4           0.1811 0.3806 0.5165      0.00717       58408296    42
nprobe=8,quantizer_efSearch=16           0.1992 0.3962 0.5123      0.00813       29022125    37
nprobe=16,quantizer_efSearch=8           0.2000 0.4163 0.5678      0.00837       58213805    36
nprobe=16,quantizer_efSearch=16          0.2107 0.4353 0.5935      0.00978       57914364    31
nprobe=32,quantizer_efSearch=16          0.2223 0.4696 0.6622      0.01349      115103620    23
nprobe=32,quantizer_efSearch=32          0.2287 0.4795 0.6788      0.01682      114286644    18
nprobe=32,quantizer_efSearch=64          0.2337 0.4902 0.6905      0.02417      113441280    13
nprobe=64,quantizer_efSearch=32          0.2348 0.5017 0.7339      0.02818      226855853    11
nprobe=64,quantizer_efSearch=64          0.2411 0.5119 0.7486      0.03252      225048667    10
nprobe=64,quantizer_efSearch=128         0.2430 0.5151 0.7532      0.04510      223709427    7
nprobe=128,quantizer_efSearch=64         0.2435 0.5240 0.7810      0.05293      446218350    6
nprobe=128,quantizer_efSearch=128        0.2464 0.5289 0.7892      0.06159      442903236    5
nprobe=128,quantizer_efSearch=256        0.2476 0.5308 0.7921      0.08486      440910860    4
nprobe=256,quantizer_efSearch=256        0.2488 0.5372 0.8101      0.12569      869803179    3
nprobe=256,quantizer_efSearch=512        0.2495 0.5379 0.8118      0.16238      866813065    2
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 17769449232

 code_size 8

 log filename: autotune.dbdeep1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2281 0.4671 0.6654      0.02543      124381784    12
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.0626 0.1055 0.1206      0.00203        5034581    148
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.0942 0.1665 0.1983      0.00235        8709745    128
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.0994 0.1664 0.1890      0.00263        6633508    115
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1032 0.1743 0.1986      0.00274        6631420    110
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1046 0.1773 0.2030      0.00295        6619072    102
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1251 0.2199 0.2604      0.00298       10241156    101
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1375 0.2568 0.3148      0.00343       17421926    88
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1463 0.2737 0.3381      0.00365       17440231    83
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1507 0.2817 0.3489      0.00399       17435305    76
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1510 0.2849 0.3523      0.00447       17407676    68
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1547 0.2880 0.3537      0.00458       20341843    66
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1651 0.3084 0.3787      0.00484       20342022    62
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1723 0.3225 0.3975      0.00592       20329194    51
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1856 0.3623 0.4685      0.00620       34639947    49
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1886 0.3702 0.4794      0.00682       34594862    45
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1914 0.3677 0.4702      0.00811       40206262    37
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2009 0.4061 0.5553      0.00931       62947536    33
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2107 0.4168 0.5610      0.01004       68598860    30
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2172 0.4363 0.5964      0.01180       68506057    26
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2177 0.4362 0.5983      0.01358       68454056    23
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2198 0.4435 0.6017      0.01432       79107236    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2262 0.4644 0.6576      0.01483      124793573    21
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2280 0.4668 0.6650      0.01991      124514187    16
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2299 0.4713 0.6604      0.02348      154637393    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2340 0.4858 0.7128      0.02694      235523785    12
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2359 0.4869 0.6906      0.02711      155797829    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2366 0.4960 0.7207      0.02901      245670999    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2402 0.5032 0.7375      0.03100      245818769    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2412 0.5094 0.7449      0.03653      265795202    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2426 0.5110 0.7496      0.03928      266027274    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2440 0.5144 0.7752      0.05155      463525665    6
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2459 0.5230 0.7881      0.06345      483632994    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2463 0.5238 0.7871      0.06525      523299145    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2471 0.5249 0.7909      0.07262      522962641    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2473 0.5251 0.7914      0.08615      600273855    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.2474 0.5255 0.7911      0.08410      523317518    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2476 0.5255 0.7916      0.09961      602649728    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2480 0.5323 0.8119      0.12447      949358278    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2481 0.5320 0.8126      0.12476     1028846451    3
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2482 0.5324 0.8131      0.15941     1028124992    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2488 0.5340 0.8205      0.19109     1783791099    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2489 0.5343 0.8225      0.19899     1783927828    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2493 0.5343 0.8212      0.20132     1857940910    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2494 0.5349 0.8242      0.24274     1861227369    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2498 0.5366 0.8299      0.67446     6577274410    1
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.2499 0.5379 0.8314      0.77210     6645735962    1
```

</details>
<details><summary>`OPQ8_64,IVF1048576_HNSW32,PQ8` </summary>
Index size 16562276272

 code_size 8

 log filename: autotune.dbdeep1B.OPQ8_64_IVF1048576_HNSW32_PQ8.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2388 0.5138 0.7539      0.07391      398463171    5
nprobe=1,quantizer_efSearch=4,ht=18      0.0573 0.0825 0.0905      0.00315       12879549    96
nprobe=1,quantizer_efSearch=4,ht=20      0.0695 0.1005 0.1104      0.00317       12879549    95
nprobe=1,quantizer_efSearch=8,ht=24      0.1097 0.1721 0.1877      0.00423       12886827    71
nprobe=2,quantizer_efSearch=4,ht=26      0.1270 0.2248 0.2553      0.00443       25672122    68
nprobe=2,quantizer_efSearch=8,ht=26      0.1475 0.2588 0.2925      0.00538       25675537    56
nprobe=2,quantizer_efSearch=8,ht=30      0.1535 0.2882 0.3413      0.00632       25675537    48
nprobe=4,quantizer_efSearch=8,ht=24      0.1591 0.2792 0.3132      0.00698       51282311    43
nprobe=4,quantizer_efSearch=8,ht=26      0.1696 0.3186 0.3719      0.00747       51282311    41
nprobe=4,quantizer_efSearch=16,ht=24     0.1720 0.2957 0.3303      0.00870       50968984    35
nprobe=4,quantizer_efSearch=8,ht=30      0.1763 0.3541 0.4380      0.00900       51282311    34
nprobe=8,quantizer_efSearch=4,ht=64      0.1866 0.3972 0.5343      0.01142      102513270    27
nprobe=4,quantizer_efSearch=16,ht=32     0.1899 0.3816 0.4776      0.01144       50968984    27
nprobe=8,quantizer_efSearch=16,ht=64     0.2070 0.4363 0.5874      0.01347      101738359    23
nprobe=8,quantizer_efSearch=32,ht=32     0.2115 0.4422 0.5881      0.02022      101176596    15
nprobe=16,quantizer_efSearch=16,ht=26    0.2147 0.4292 0.5412      0.02076      202735255    15
nprobe=16,quantizer_efSearch=32,ht=28    0.2244 0.4653 0.6166      0.02582      201618868    12
nprobe=16,quantizer_efSearch=32,ht=32    0.2273 0.4835 0.6749      0.03169      201618868    10
nprobe=32,quantizer_efSearch=16,ht=30    0.2304 0.4948 0.7070      0.04568      403275809    7
nprobe=32,quantizer_efSearch=64,ht=28    0.2353 0.4950 0.6866      0.04783      399552868    7
nprobe=32,quantizer_efSearch=64,ht=64    0.2384 0.5136 0.7534      0.04325      399552868    7
nprobe=32,quantizer_efSearch=256,ht=64   0.2388 0.5138 0.7539      0.07433      398463171    5
nprobe=64,quantizer_efSearch=256,ht=64   0.2411 0.5293 0.7944      0.10515      789122478    3
nprobe=128,quantizer_efSearch=256,ht=64  0.2428 0.5366 0.8179      0.15498     1560506983    2
nprobe=256,quantizer_efSearch=256,ht=64  0.2433 0.5400 0.8310      0.27457     3076755088    2
nprobe=1024,quantizer_efSearch=512,ht=30 0.2435 0.5368 0.8102      1.25098    11828894646    1
```

</details>
<details><summary>`OPQ8_64,IVF4194304_HNSW32,PQ8` </summary>
Index size 18248822192

 code_size 8

 log filename: autotune.dbdeep1B.OPQ8_64_IVF4194304_HNSW32_PQ8.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1810 0.3445 0.3938      0.01004       14581208    30
nprobe=1,quantizer_efSearch=4,ht=18      0.0464 0.0663 0.0734      0.00352        3718239    86
nprobe=1,quantizer_efSearch=4,ht=20      0.0566 0.0812 0.0887      0.00363        3718239    83
nprobe=1,quantizer_efSearch=4,ht=32      0.0929 0.1611 0.1769      0.00415        3718239    73
nprobe=2,quantizer_efSearch=4,ht=26      0.1026 0.1761 0.1926      0.00448        7391847    67
nprobe=2,quantizer_efSearch=4,ht=28      0.1084 0.1957 0.2169      0.00466        7391847    65
nprobe=2,quantizer_efSearch=4,ht=30      0.1110 0.2063 0.2310      0.00488        7391847    62
nprobe=2,quantizer_efSearch=8,ht=26      0.1334 0.2216 0.2399      0.00564        7357868    54
nprobe=2,quantizer_efSearch=8,ht=30      0.1435 0.2583 0.2865      0.00612        7357868    50
nprobe=2,quantizer_efSearch=8,ht=32      0.1441 0.2640 0.2964      0.00633        7357868    48
nprobe=4,quantizer_efSearch=8,ht=26      0.1556 0.2731 0.3026      0.00749       14683196    41
nprobe=4,quantizer_efSearch=8,ht=30      0.1651 0.3168 0.3648      0.00810       14683196    38
nprobe=8,quantizer_efSearch=4,ht=64      0.1749 0.3638 0.4526      0.00847       29295473    36
nprobe=4,quantizer_efSearch=16,ht=28     0.1777 0.3286 0.3681      0.00995       14581208    31
nprobe=4,quantizer_efSearch=16,ht=30     0.1810 0.3445 0.3938      0.01027       14581208    30
nprobe=4,quantizer_efSearch=16,ht=32     0.1821 0.3508 0.4080      0.01056       14581208    29
nprobe=8,quantizer_efSearch=16,ht=64     0.2027 0.4131 0.5133      0.01110       28986334    28
nprobe=8,quantizer_efSearch=32,ht=32     0.2121 0.4233 0.5194      0.01821       28752419    17
nprobe=32,quantizer_efSearch=8,ht=64     0.2143 0.4593 0.6272      0.02132      115559233    15
nprobe=16,quantizer_efSearch=16,ht=32    0.2173 0.4548 0.5896      0.02169       57804833    14
nprobe=16,quantizer_efSearch=32,ht=28    0.2231 0.4456 0.5477      0.02406       57300601    13
nprobe=16,quantizer_efSearch=32,ht=32    0.2286 0.4714 0.6106      0.02566       57300601    12
nprobe=32,quantizer_efSearch=64,ht=64    0.2421 0.5156 0.7029      0.03383      113324421    9
nprobe=64,quantizer_efSearch=32,ht=64    0.2444 0.5265 0.7485      0.04298      226590096    7
nprobe=64,quantizer_efSearch=64,ht=30    0.2476 0.5273 0.7278      0.07348      224833938    5
nprobe=128,quantizer_efSearch=64,ht=64   0.2521 0.5476 0.8006      0.07772      445714400    4
nprobe=128,quantizer_efSearch=256,ht=64  0.2545 0.5544 0.8110      0.11219      440303122    3
nprobe=256,quantizer_efSearch=256,ht=64  0.2573 0.5627 0.8350      0.16354      868728932    2
nprobe=512,quantizer_efSearch=256,ht=32  0.2578 0.5640 0.8378      0.46593     1712044578    1
nprobe=1024,quantizer_efSearch=512,ht=64 0.2579 0.5684 0.8547      0.52411     3343122347    1
nprobe=2048,quantizer_efSearch=512,ht=64 0.2580 0.5687 0.8556      0.94667     6543183014    1
```

</details>
<details><summary>`OPQ8_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ8` </summary>
Index size 17208589300

 code_size 8

 log filename: autotune.dbdeep1B.OPQ8_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ8.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.0112 0.0167 0.0226      0.18327      443114272    2
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=20       0.0601 0.0826 0.0907      0.00355        8697231    85
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=32       0.0618 0.1032 0.1149      0.00333        5038092    91
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=32       0.0831 0.1426 0.1599      0.00355        5130514    85
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=24      0.0869 0.1292 0.1396      0.00408        6621393    74
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=28       0.1197 0.2042 0.2282      0.00399       10234767    76
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=28      0.1210 0.2213 0.2546      0.00579       15788713    52
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=64      0.1677 0.3409 0.4246      0.00808       31611057    38
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64       0.1710 0.3303 0.3918      0.00709       20314580    43
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=32      0.1728 0.3296 0.3895      0.01074       20290646    28
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64      0.1784 0.3421 0.4026      0.01132       36672319    27
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=64     0.1879 0.3594 0.4263      0.01397       36517710    22
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64      0.2068 0.4342 0.5671      0.01065       62944827    29
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=64      0.2226 0.4636 0.6039      0.01280       68499057    24
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=64    0.2428 0.5134 0.7006      0.02384      135020213    13
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=32    0.2431 0.5116 0.6948      0.04778      155686224    7
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64    0.2467 0.5289 0.7400      0.07299      463698822    5
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=64    0.2526 0.5427 0.7669      0.07608      265257100    4
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=64   0.2533 0.5446 0.7690      0.12866      385344965    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=30    0.2547 0.5355 0.7343      0.14357      535735784    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64    0.2553 0.5582 0.8170      0.14365      891816883    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=64    0.2557 0.5607 0.8281      0.28191     1728166029    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=32  0.2584 0.5638 0.8244      0.37681     1027146153    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=512,ht=64  0.2595 0.5706 0.8486      0.57102     2330274958    1
nprobe=2048,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=64 0.2596 0.5728 0.8568      1.24676     6639091765    1
```

</details>

## Code sizes in [9, 16]

<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16` </summary>
Index size 24562276272

 code_size 16

 log filename: autotune.dbdeep1B.OPQ16_64_IVF1048576_HNSW32_PQ16.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3816 0.7095 0.8013      0.06670      403759666    5
nprobe=1,quantizer_efSearch=4,ht=24       0.0293 0.0377 0.0408      0.00345       12930231    88
nprobe=1,quantizer_efSearch=4,ht=28       0.0391 0.0510 0.0544      0.00343       12930231    88
nprobe=1,quantizer_efSearch=4,ht=40       0.0851 0.1101 0.1147      0.00350       12930231    86
nprobe=1,quantizer_efSearch=4,ht=54       0.1488 0.2111 0.2178      0.00389       12930231    78
nprobe=2,quantizer_efSearch=4,ht=50       0.1741 0.2491 0.2575      0.00501       25731855    60
nprobe=1,quantizer_efSearch=8,ht=128      0.1788 0.2626 0.2717      0.00551       12873822    55
nprobe=2,quantizer_efSearch=8,ht=48       0.1906 0.2682 0.2767      0.00587       25663457    52
nprobe=4,quantizer_efSearch=4,ht=48       0.1952 0.2837 0.2953      0.00764       51451279    40
nprobe=4,quantizer_efSearch=8,ht=46       0.2144 0.2978 0.3088      0.00850       51230513    36
nprobe=8,quantizer_efSearch=4,ht=48       0.2549 0.3811 0.3984      0.01379      102546559    22
nprobe=4,quantizer_efSearch=16,ht=50      0.2620 0.3870 0.4025      0.01070       50961355    29
nprobe=4,quantizer_efSearch=16,ht=52      0.2737 0.4202 0.4375      0.01101       50961355    28
nprobe=4,quantizer_efSearch=16,ht=54      0.2825 0.4444 0.4645      0.01142       50961355    27
nprobe=4,quantizer_efSearch=32,ht=56      0.2940 0.4655 0.4892      0.01509       50751187    20
nprobe=8,quantizer_efSearch=16,ht=56      0.3273 0.5524 0.5905      0.01828      101731524    17
nprobe=16,quantizer_efSearch=16,ht=54     0.3530 0.6063 0.6568      0.02934      202964440    11
nprobe=16,quantizer_efSearch=16,ht=62     0.3614 0.6470 0.7175      0.03743      202964440    9
nprobe=16,quantizer_efSearch=64,ht=54     0.3640 0.6242 0.6760      0.03821      201169147    8
nprobe=16,quantizer_efSearch=64,ht=60     0.3730 0.6625 0.7331      0.04381      201169147    7
nprobe=16,quantizer_efSearch=64,ht=64     0.3732 0.6669 0.7402      0.04826      201169147    7
nprobe=32,quantizer_efSearch=32,ht=58     0.3874 0.7148 0.8032      0.06212      401983663    5
nprobe=32,quantizer_efSearch=128,ht=60    0.3911 0.7281 0.8218      0.08046      399312003    4
nprobe=64,quantizer_efSearch=32,ht=54     0.3928 0.7171 0.7982      0.10055      797619832    3
nprobe=64,quantizer_efSearch=32,ht=56     0.3970 0.7387 0.8353      0.10236      797619832    3
nprobe=64,quantizer_efSearch=64,ht=56     0.4005 0.7470 0.8442      0.10682      793579299    3
nprobe=64,quantizer_efSearch=256,ht=58    0.4048 0.7628 0.8715      0.13390      790113830    3
nprobe=128,quantizer_efSearch=128,ht=60   0.4108 0.7905 0.9236      0.20853     1566219260    2
nprobe=128,quantizer_efSearch=512,ht=128  0.4118 0.7961 0.9344      0.18556     1561264398    2
nprobe=256,quantizer_efSearch=256,ht=128  0.4173 0.8100 0.9585      0.33590     3081955280    1
nprobe=256,quantizer_efSearch=512,ht=128  0.4174 0.8103 0.9589      0.34712     3076901449    1
nprobe=512,quantizer_efSearch=512,ht=128  0.4188 0.8159 0.9708      0.65806     6045330311    1
nprobe=1024,quantizer_efSearch=512,ht=58  0.4190 0.8086 0.9541      1.37595    11851045733    1
nprobe=2048,quantizer_efSearch=512,ht=128 0.4202 0.8206 0.9807      2.27111    23089656214    1
```

</details>
<details><summary>`OPQ16_64,IVF4194304_HNSW32,PQ16` </summary>
Index size 26248822192

 code_size 16

 log filename: autotune.dbdeep1B.OPQ16_64_IVF4194304_HNSW32_PQ16.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.0019 0.0034 0.0054      0.00485        3689000    62
nprobe=1,quantizer_efSearch=4,ht=2        0.0009 0.0016 0.0028      0.00387        3695041    78
nprobe=1,quantizer_efSearch=4,ht=16       0.0100 0.0140 0.0173      0.00401        3695041    75
nprobe=1,quantizer_efSearch=4,ht=40       0.0665 0.0833 0.0877      0.00390        3695041    77
nprobe=1,quantizer_efSearch=4,ht=54       0.1209 0.1628 0.1680      0.00409        3695041    74
nprobe=2,quantizer_efSearch=4,ht=50       0.1422 0.1938 0.2001      0.00518        7379401    58
nprobe=1,quantizer_efSearch=8,ht=128      0.1574 0.2178 0.2247      0.00530        3689000    57
nprobe=2,quantizer_efSearch=8,ht=48       0.1630 0.2188 0.2260      0.00633        7361373    48
nprobe=4,quantizer_efSearch=8,ht=46       0.1767 0.2359 0.2454      0.00871       14705125    35
nprobe=4,quantizer_efSearch=16,ht=54      0.2576 0.3772 0.3902      0.01128       14610262    27
nprobe=4,quantizer_efSearch=32,ht=56      0.2668 0.3992 0.4134      0.01534       14492424    20
nprobe=8,quantizer_efSearch=16,ht=56      0.3033 0.4767 0.4991      0.01673       29079080    18
nprobe=16,quantizer_efSearch=16,ht=54     0.3274 0.5288 0.5589      0.02661       57977826    12
nprobe=16,quantizer_efSearch=16,ht=62     0.3415 0.5813 0.6256      0.02874       57977826    11
nprobe=16,quantizer_efSearch=64,ht=60     0.3517 0.6009 0.6444      0.03901       57107753    8
nprobe=32,quantizer_efSearch=32,ht=64     0.3742 0.6673 0.7340      0.05566      114469634    6
nprobe=32,quantizer_efSearch=128,ht=128   0.3833 0.6824 0.7509      0.05437      113095530    6
nprobe=128,quantizer_efSearch=64,ht=128   0.4113 0.7724 0.8839      0.10984      447067162    3
nprobe=128,quantizer_efSearch=512,ht=128  0.4161 0.7827 0.8964      0.16426      440685023    2
nprobe=256,quantizer_efSearch=256,ht=128  0.4230 0.8047 0.9340      0.16487      871372577    2
nprobe=512,quantizer_efSearch=512,ht=128  0.4260 0.8186 0.9604      0.32995     1707834255    1
nprobe=2048,quantizer_efSearch=512,ht=128 0.4286 0.8274 0.9776      1.29944     6566783542    1
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16` </summary>
Index size 25208587252

 code_size 16

 log filename: autotune.dbdeep1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.1734 0.2378 0.2449      0.00566       10256421    54
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=2         0.0009 0.0015 0.0024      0.00314        5103041    96
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=26        0.0227 0.0300 0.0342      0.00321        5128875    94
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=42        0.0801 0.1021 0.1066      0.00447        9558520    68
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=42       0.0921 0.1186 0.1249      0.00483        8729019    63
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=52        0.1302 0.1737 0.1792      0.00468        9543489    65
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=128      0.1564 0.2174 0.2240      0.00517        9546781    59
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=52       0.1734 0.2378 0.2449      0.00562       10251023    54
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=56        0.2157 0.3183 0.3306      0.00821       17496299    37
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=128      0.2751 0.4528 0.4820      0.01586       60248721    19
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128     0.3191 0.5143 0.5439      0.02051       51076044    15
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56       0.3326 0.5399 0.5706      0.02796       68725300    11
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=54      0.3414 0.5414 0.5692      0.03204       79347985    10
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=128    0.3552 0.6019 0.6478      0.03635       79303644    9
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=60     0.3554 0.5991 0.6422      0.04588       99501661    7
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=64     0.3579 0.6059 0.6517      0.05187      139326929    6
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=64      0.3723 0.6427 0.6985      0.07074      195810331    5
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=60     0.3856 0.6708 0.7362      0.07438      156003642    5
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=60     0.3864 0.6723 0.7374      0.07649      195244560    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128     0.4057 0.7382 0.8289      0.08078      304420461    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=62   0.4184 0.7807 0.8896      0.25279      755228989    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=60    0.4205 0.7883 0.9036      0.34983     1017944510    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=58    0.4221 0.7885 0.9005      0.36740      950742887    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=128  0.4292 0.8195 0.9609      0.52770     2315572857    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=512,ht=62   0.4294 0.8181 0.9556      0.93246     2335525212    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=62   0.4315 0.8258 0.9693      1.28628     3492944134    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=128 0.4331 0.8316 0.9822      2.56005     7126246083    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576_HNSW32,PQ32x4fsr` </summary>
Index size 24831365580

 code_size 16

 log filename: autotune.dbdeep1B.OPQ32_64_IVF1048576_HNSW32_PQ32x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3953 0.7776 0.9486      0.15688     3093988406    2
nprobe=2,quantizer_efSearch=4            0.1910 0.3019 0.3203      0.00231       25690407    131
nprobe=2,quantizer_efSearch=8            0.2234 0.3490 0.3715      0.00300       25674784    100
nprobe=4,quantizer_efSearch=8            0.2634 0.4406 0.4787      0.00403       51268614    75
nprobe=4,quantizer_efSearch=16           0.2775 0.4648 0.5036      0.00519       50939530    58
nprobe=8,quantizer_efSearch=4            0.2863 0.5042 0.5614      0.00558      102527558    54
nprobe=8,quantizer_efSearch=8            0.2959 0.5187 0.5783      0.00574      102370034    53
nprobe=8,quantizer_efSearch=16           0.3153 0.5559 0.6182      0.00691      101789984    44
nprobe=8,quantizer_efSearch=32           0.3215 0.5669 0.6296      0.00935      101271413    33
nprobe=16,quantizer_efSearch=8           0.3281 0.6060 0.6934      0.00929      203753026    33
nprobe=16,quantizer_efSearch=32          0.3481 0.6410 0.7335      0.01207      201965679    25
nprobe=16,quantizer_efSearch=64          0.3517 0.6457 0.7381      0.01614      201266774    19
nprobe=32,quantizer_efSearch=16          0.3584 0.6833 0.7993      0.02174      403883441    14
nprobe=32,quantizer_efSearch=32          0.3674 0.6987 0.8180      0.02332      401887439    13
nprobe=32,quantizer_efSearch=64          0.3730 0.7058 0.8253      0.02776      400159073    11
nprobe=32,quantizer_efSearch=128         0.3733 0.7059 0.8251      0.03699      399295762    9
nprobe=64,quantizer_efSearch=32          0.3786 0.7319 0.8743      0.04367      797772620    7
nprobe=64,quantizer_efSearch=64          0.3834 0.7400 0.8843      0.04429      793787357    7
nprobe=64,quantizer_efSearch=128         0.3837 0.7413 0.8862      0.05048      791233013    6
nprobe=64,quantizer_efSearch=256         0.3841 0.7421 0.8871      0.06979      790161690    5
nprobe=128,quantizer_efSearch=128        0.3906 0.7663 0.9258      0.10492     1566432668    3
nprobe=128,quantizer_efSearch=256        0.3914 0.7665 0.9263      0.10836     1562796687    3
nprobe=128,quantizer_efSearch=512        0.3915 0.7668 0.9264      0.15025     1561482680    2
nprobe=256,quantizer_efSearch=64         0.3928 0.7718 0.9405      0.17284     3109987996    2
nprobe=256,quantizer_efSearch=128        0.3953 0.7776 0.9486      0.17742     3093988406    2
nprobe=256,quantizer_efSearch=256        0.3959 0.7799 0.9519      0.17899     3082262469    2
nprobe=256,quantizer_efSearch=512        0.3960 0.7800 0.9522      0.21794     3077163891    2
nprobe=512,quantizer_efSearch=512        0.3961 0.7865 0.9636      0.37453     6045744752    1
nprobe=1024,quantizer_efSearch=512       0.3977 0.7868 0.9692      0.61207    11851251368    1
```

</details>
<details><summary>`OPQ32_64,IVF4194304_HNSW32,PQ32x4fsr` </summary>
Index size 27337870284

 code_size 16

 log filename: autotune.dbdeep1B.OPQ32_64_IVF4194304_HNSW32_PQ32x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4012 0.7808 0.9238      0.16255      877151433    2
nprobe=2,quantizer_efSearch=4            0.1575 0.2411 0.2523      0.00288        7413063    105
nprobe=4,quantizer_efSearch=4            0.1856 0.2989 0.3169      0.00334       14732621    90
nprobe=2,quantizer_efSearch=8            0.1961 0.2949 0.3075      0.00369        7402754    82
nprobe=4,quantizer_efSearch=8            0.2327 0.3723 0.3935      0.00434       14738509    70
nprobe=8,quantizer_efSearch=8            0.2671 0.4477 0.4832      0.00578       29358158    52
nprobe=8,quantizer_efSearch=16           0.2948 0.4892 0.5265      0.00748       29075185    41
nprobe=16,quantizer_efSearch=8           0.3120 0.5499 0.6060      0.00902       58322940    34
nprobe=16,quantizer_efSearch=32          0.3367 0.5916 0.6485      0.01291       57446291    24
nprobe=16,quantizer_efSearch=64          0.3401 0.5969 0.6540      0.01879       57110288    16
nprobe=32,quantizer_efSearch=16          0.3495 0.6376 0.7141      0.02042      115278616    15
nprobe=32,quantizer_efSearch=32          0.3581 0.6568 0.7357      0.02220      114421054    14
nprobe=32,quantizer_efSearch=64          0.3661 0.6676 0.7479      0.02622      113595889    12
nprobe=64,quantizer_efSearch=32          0.3752 0.7083 0.8089      0.03817      227249290    8
nprobe=64,quantizer_efSearch=64          0.3839 0.7224 0.8247      0.04224      225494017    8
nprobe=64,quantizer_efSearch=128         0.3865 0.7271 0.8298      0.04936      224141774    7
nprobe=64,quantizer_efSearch=256         0.3881 0.7291 0.8320      0.07009      223455788    5
nprobe=128,quantizer_efSearch=64         0.3937 0.7539 0.8786      0.08902      447004879    4
nprobe=128,quantizer_efSearch=128        0.3968 0.7611 0.8863      0.09659      443589450    4
nprobe=128,quantizer_efSearch=256        0.4000 0.7652 0.8907      0.12212      441518749    3
nprobe=128,quantizer_efSearch=512        0.4006 0.7661 0.8913      0.15878      440610483    2
nprobe=256,quantizer_efSearch=128        0.4012 0.7808 0.9238      0.16267      877151433    2
nprobe=256,quantizer_efSearch=256        0.4053 0.7851 0.9288      0.18262      871239418    2
nprobe=256,quantizer_efSearch=512        0.4063 0.7859 0.9299      0.23912      868186850    2
nprobe=512,quantizer_efSearch=256        0.4064 0.7981 0.9538      0.33423     1717315753    1
nprobe=512,quantizer_efSearch=512        0.4077 0.8003 0.9557      0.36551     1707442763    1
nprobe=1024,quantizer_efSearch=512       0.4085 0.8022 0.9673      0.62543     3353028750    1
nprobe=2048,quantizer_efSearch=512       0.4097 0.8049 0.9714      1.20607     6564981560    1
```

</details>
<details><summary>`OPQ32_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 26296864272

 code_size 16

 log filename: autotune.dbdeep1B.OPQ32_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ32x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.4077 0.7800 0.9271      0.15293     1030508238    2
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1436 0.2041 0.2127      0.00188        8749822    160
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.1470 0.2013 0.2091      0.00218        6649664    138
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1844 0.2721 0.2842      0.00280       17358932    108
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2049 0.3123 0.3273      0.00307       17490118    98
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2184 0.3334 0.3495      0.00308       17475269    98
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.2245 0.3453 0.3628      0.00338       17480704    89
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.2275 0.3494 0.3671      0.00374       17466283    81
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2281 0.3454 0.3616      0.00406       20393069    74
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=2      0.2283 0.3503 0.3683      0.00427       17456270    71
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.2444 0.3785 0.3997      0.00518       34560699    58
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.2587 0.4152 0.4456      0.00553       31721175    55
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2794 0.4546 0.4855      0.00607       34714651    50
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.2870 0.4641 0.4966      0.00644       34688995    47
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3153 0.5399 0.5919      0.00919       63071433    33
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3186 0.5365 0.5831      0.01010       68703484    30
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3359 0.5767 0.6310      0.01114       68553589    27
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.3370 0.5798 0.6348      0.01238       68503981    25
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3397 0.5792 0.6330      0.01288       79320935    24
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3430 0.5900 0.6452      0.01368       79252680    22
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.3447 0.5930 0.6489      0.01495       79271203    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3561 0.6330 0.7074      0.01936      124948215    16
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.3594 0.6418 0.7189      0.02231      124737872    14
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3742 0.6667 0.7460      0.02575      155149426    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3791 0.6917 0.7867      0.04122      246700672    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3837 0.7094 0.8118      0.04035      246168875    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.3881 0.7197 0.8228      0.04725      304249447    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3906 0.7232 0.8272      0.04543      266559944    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128    0.3921 0.7259 0.8299      0.05739      384551537    6
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128   0.3924 0.7256 0.8304      0.06856      386111159    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3941 0.7476 0.8730      0.08312      463942036    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3993 0.7577 0.8859      0.08200      483611951    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3997 0.7584 0.8868      0.08843      523290761    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.4011 0.7608 0.8901      0.08491      522146581    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.4013 0.7610 0.8903      0.09913      523926366    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4039 0.7742 0.9150      0.15287      951420975    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.4086 0.7813 0.9283      0.16299      950733628    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4116 0.7931 0.9530      0.33685     1784471648    1
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.4125 0.7945 0.9551      0.33298     1863928255    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256  0.4127 0.7947 0.9559      0.40241     2020120284    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.4139 0.8016 0.9686      0.62362     3647124275    1
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.4142 0.8013 0.9683      0.65286     3491407736    1
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF1048576_HNSW32,PQ48x4fs` </summary>
Index size 33095809669

 code_size 24

 log filename: autotune.dbdeep1B.IVF1048576_HNSW32_PQ48x4fs.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3460 0.6999 0.9149      0.12036     3119478909    3
nprobe=2,quantizer_efSearch=4            0.1695 0.2908 0.3298      0.00381       25950053    79
nprobe=4,quantizer_efSearch=4            0.1990 0.3548 0.4138      0.00483       51884440    63
nprobe=4,quantizer_efSearch=8            0.2342 0.4213 0.4898      0.00616       51844234    49
nprobe=8,quantizer_efSearch=4            0.2559 0.4730 0.5668      0.00757      103761569    40
nprobe=8,quantizer_efSearch=8            0.2627 0.4886 0.5856      0.00809      103529502    38
nprobe=8,quantizer_efSearch=16           0.2842 0.5248 0.6256      0.01035      102900798    29
nprobe=16,quantizer_efSearch=8           0.2928 0.5629 0.6925      0.01244      205996653    25
nprobe=16,quantizer_efSearch=16          0.3042 0.5816 0.7153      0.01390      205221557    22
nprobe=16,quantizer_efSearch=32          0.3122 0.5973 0.7347      0.01787      204016835    17
nprobe=32,quantizer_efSearch=16          0.3165 0.6275 0.7901      0.02173      407756575    14
nprobe=32,quantizer_efSearch=32          0.3245 0.6417 0.8077      0.02466      405666679    13
nprobe=32,quantizer_efSearch=64          0.3284 0.6484 0.8150      0.03186      403873445    10
nprobe=64,quantizer_efSearch=32          0.3346 0.6672 0.8549      0.03946      805371011    8
nprobe=64,quantizer_efSearch=64          0.3380 0.6742 0.8642      0.04496      801096334    7
nprobe=64,quantizer_efSearch=128         0.3388 0.6761 0.8665      0.05736      798465438    6
nprobe=128,quantizer_efSearch=64         0.3427 0.6876 0.8933      0.07195     1587179638    5
nprobe=128,quantizer_efSearch=128        0.3438 0.6917 0.8981      0.08171     1579957016    4
nprobe=128,quantizer_efSearch=256        0.3444 0.6931 0.8997      0.10424     1576089259    3
nprobe=256,quantizer_efSearch=128        0.3460 0.6999 0.9149      0.12424     3119478909    3
nprobe=256,quantizer_efSearch=256        0.3465 0.7015 0.9175      0.14162     3107069811    3
nprobe=512,quantizer_efSearch=128        0.3472 0.7035 0.9229      0.18308     6139170340    2
nprobe=512,quantizer_efSearch=256        0.3479 0.7053 0.9261      0.20181     6112281816    2
nprobe=512,quantizer_efSearch=512        0.3485 0.7057 0.9268      0.23533     6091700195    2
nprobe=1024,quantizer_efSearch=256       0.3493 0.7076 0.9306      0.31666    11970962907    1
nprobe=1024,quantizer_efSearch=512       0.3497 0.7081 0.9320      0.39000    11934042346    1
nprobe=2048,quantizer_efSearch=512       0.3503 0.7092 0.9335      0.56911    23243061508    1
```

</details>
<details><summary>`IVF4194304_HNSW32,PQ48x4fs` </summary>
Index size 36402119045

 code_size 24

 log filename: autotune.dbdeep1B.IVF4194304_HNSW32_PQ48x4fs.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3484 0.6989 0.8984      0.07550      898342557    4
nprobe=1,quantizer_efSearch=4            0.1133 0.1790 0.1937      0.00362        3806547    83
nprobe=2,quantizer_efSearch=4            0.1420 0.2322 0.2573      0.00386        7586138    78
nprobe=4,quantizer_efSearch=4            0.1608 0.2814 0.3214      0.00427       15103091    71
nprobe=2,quantizer_efSearch=8            0.1779 0.2898 0.3180      0.00536        7587082    56
nprobe=4,quantizer_efSearch=8            0.2083 0.3613 0.4077      0.00579       15129498    52
nprobe=8,quantizer_efSearch=4            0.2195 0.4079 0.4724      0.00600       30159444    50
nprobe=8,quantizer_efSearch=8            0.2294 0.4250 0.4910      0.00653       30114129    46
nprobe=16,quantizer_efSearch=4           0.2436 0.4627 0.5475      0.00770       60026051    39
nprobe=8,quantizer_efSearch=16           0.2572 0.4690 0.5398      0.00927       29862499    33
nprobe=16,quantizer_efSearch=8           0.2742 0.5144 0.6089      0.00901       59846556    34
nprobe=16,quantizer_efSearch=16          0.2854 0.5350 0.6323      0.01067       59553820    29
nprobe=32,quantizer_efSearch=16          0.3027 0.5892 0.7165      0.01444      118267499    21
nprobe=64,quantizer_efSearch=16          0.3117 0.6144 0.7634      0.02047      234810917    15
nprobe=64,quantizer_efSearch=32          0.3254 0.6448 0.8000      0.02420      233050865    13
nprobe=64,quantizer_efSearch=64          0.3324 0.6568 0.8149      0.03202      231208384    10
nprobe=128,quantizer_efSearch=64         0.3397 0.6775 0.8600      0.04344      458221352    7
nprobe=128,quantizer_efSearch=128        0.3453 0.6853 0.8708      0.05787      454653297    6
nprobe=256,quantizer_efSearch=128        0.3484 0.6989 0.8984      0.08005      898342557    4
nprobe=256,quantizer_efSearch=256        0.3508 0.7025 0.9028      0.10565      892081090    3
nprobe=512,quantizer_efSearch=256        0.3525 0.7074 0.9160      0.14382     1757887559    3
nprobe=512,quantizer_efSearch=512        0.3537 0.7103 0.9200      0.19728     1747155256    2
nprobe=1024,quantizer_efSearch=512       0.3546 0.7132 0.9289      0.27219     3430146471    2
nprobe=2048,quantizer_efSearch=512       0.3554 0.7123 0.9296      0.39922     6711776815    1
```

</details>
<details><summary>`IVF4194304(IVF1024,PQ48x4fs,RFlat),PQ48x4fs` </summary>
Index size 35394384073

 code_size 24

 log filename: autotune.dbdeep1B.IVF4194304_IVF1024_PQ48x4fs_RFlat__PQ48x4fs.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3563 0.7143 0.9322      0.44048     7271268514    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.0832 0.1280 0.1383      0.00241        5226431    125
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1288 0.2053 0.2264      0.00272        8920702    111
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1323 0.2055 0.2210      0.00375        6742512    81
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1840 0.3183 0.3546      0.00420       17830235    72
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1926 0.3352 0.3782      0.00497       17846559    61
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1945 0.3389 0.3816      0.00556       17824751    54
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2041 0.3516 0.3906      0.00645       20823465    47
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2141 0.3702 0.4133      0.00671       20861322    45
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2171 0.3870 0.4352      0.00725       35477714    42
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.2192 0.3806 0.4275      0.00789       20824212    39
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2428 0.4427 0.5098      0.00806       35527539    38
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2465 0.4496 0.5182      0.00869       35487213    35
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2678 0.5080 0.6062      0.01217      122203504    25
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2698 0.5104 0.6028      0.01110       64531065    28
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2798 0.5398 0.6536      0.01276      122325559    24
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2831 0.5226 0.6134      0.01250       70260797    24
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2919 0.5438 0.6399      0.01432       70167593    21
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.2922 0.5442 0.6410      0.01924       70089950    16
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3009 0.5611 0.6587      0.01816       81015378    17
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.3016 0.5616 0.6595      0.02058       80892300    15
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3078 0.5905 0.7159      0.01685      127771242    18
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3087 0.5919 0.7225      0.02066      241665755    15
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3290 0.6403 0.7926      0.02555      252276603    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3317 0.6483 0.8022      0.03019      271754290    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3346 0.6565 0.8146      0.03309      272339624    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3410 0.6751 0.8554      0.04218      475056200    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3440 0.6860 0.8697      0.05924      536113497    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3450 0.6844 0.8765      0.05915      914575457    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3486 0.6946 0.8914      0.06218      932534866    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3490 0.6988 0.8949      0.07524      972677255    4
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.3508 0.6996 0.8985      0.11761      931848578    3
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.3513 0.7034 0.9033      0.13556      972014548    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3523 0.7031 0.9104      0.10881     1790381956    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3549 0.7109 0.9196      0.15804     1903051515    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3559 0.7141 0.9290      0.21058     3570177813    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.3564 0.7144 0.9323      0.36379     6959043049    1
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.3565 0.7148 0.9326      0.51673     6793770656    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512 0.3567 0.7150 0.9328      0.50774     7261905822    1
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=512 0.3568 0.7152 0.9329      0.60925     7260586205    1
```

</details>
<details><summary>`OPQ32_128,IVF1048576_HNSW32,PQ32` </summary>
Index size 40830801840

 code_size 32

 log filename: autotune.dbdeep1B.OPQ32_128_IVF1048576_HNSW32_PQ32.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.5057 0.6532 0.6568      0.06805      101902386    5
nprobe=1,quantizer_efSearch=4,ht=74       0.0565 0.0666 0.0687      0.00413       12964355    73
nprobe=1,quantizer_efSearch=4,ht=118      0.2012 0.2424 0.2447      0.00489       12964355    62
nprobe=1,quantizer_efSearch=8,ht=116      0.2306 0.2766 0.2789      0.00578       13009964    52
nprobe=2,quantizer_efSearch=4,ht=114      0.2622 0.3208 0.3232      0.00671       25946847    45
nprobe=2,quantizer_efSearch=4,ht=122      0.2679 0.3297 0.3322      0.00757       25946847    40
nprobe=2,quantizer_efSearch=4,ht=126      0.2686 0.3304 0.3330      0.00813       25946847    37
nprobe=2,quantizer_efSearch=8,ht=120      0.3132 0.3836 0.3860      0.00829       25946053    37
nprobe=2,quantizer_efSearch=32,ht=110     0.3145 0.3791 0.3815      0.01244       25672727    25
nprobe=4,quantizer_efSearch=16,ht=110     0.3946 0.4862 0.4892      0.01300       51489491    24
nprobe=4,quantizer_efSearch=8,ht=122      0.3959 0.4977 0.5007      0.01356       51839610    23
nprobe=4,quantizer_efSearch=32,ht=110     0.4008 0.4930 0.4960      0.01650       51253500    19
nprobe=8,quantizer_efSearch=4,ht=108      0.4127 0.5157 0.5189      0.01835      103761767    17
nprobe=4,quantizer_efSearch=32,ht=126     0.4249 0.5329 0.5358      0.01966       51253500    16
nprobe=8,quantizer_efSearch=4,ht=120      0.4491 0.5774 0.5810      0.02158      103761767    14
nprobe=8,quantizer_efSearch=32,ht=256     0.5039 0.6518 0.6554      0.02522      102338938    12
nprobe=8,quantizer_efSearch=64,ht=124     0.5046 0.6521 0.6558      0.03536      102051247    9
nprobe=16,quantizer_efSearch=8,ht=114     0.5260 0.6890 0.6936      0.03652      205973223    9
nprobe=16,quantizer_efSearch=64,ht=116    0.5620 0.7433 0.7480      0.04775      203309063    7
nprobe=16,quantizer_efSearch=32,ht=128    0.5652 0.7549 0.7598      0.05224      204035888    6
nprobe=16,quantizer_efSearch=128,ht=122   0.5677 0.7561 0.7609      0.06335      202962215    5
nprobe=32,quantizer_efSearch=16,ht=256    0.5992 0.8198 0.8256      0.07119      407764118    5
nprobe=32,quantizer_efSearch=32,ht=256    0.6112 0.8375 0.8435      0.07154      405657392    5
nprobe=64,quantizer_efSearch=16,ht=118    0.6178 0.8508 0.8574      0.14172      809977547    3
nprobe=64,quantizer_efSearch=128,ht=112   0.6261 0.8524 0.8581      0.14722      798468515    3
nprobe=64,quantizer_efSearch=64,ht=118    0.6427 0.8902 0.8970      0.14561      801110584    3
nprobe=128,quantizer_efSearch=32,ht=120   0.6516 0.9126 0.9207      0.28508     1596676099    2
nprobe=128,quantizer_efSearch=512,ht=116  0.6640 0.9238 0.9317      0.33721     1574649129    1
nprobe=256,quantizer_efSearch=128,ht=114  0.6654 0.9290 0.9370      0.51084     3119508684    1
nprobe=256,quantizer_efSearch=64,ht=122   0.6702 0.9506 0.9602      0.57428     3136440510    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6774 0.9654 0.9751      1.07049     6139259109    1
nprobe=1024,quantizer_efSearch=128,ht=124 0.6806 0.9739 0.9844      2.23601    11895863509    1
nprobe=1024,quantizer_efSearch=128,ht=128 0.6817 0.9759 0.9863      2.49482    11895863509    1
nprobe=2048,quantizer_efSearch=256,ht=124 0.6843 0.9810 0.9918      4.39620    22889464385    1
nprobe=2048,quantizer_efSearch=512,ht=256 0.6866 0.9862 0.9972      3.81222    23243145894    1
```

</details>
<details><summary>`OPQ32_128,IVF4194304_HNSW32,PQ32` </summary>
Index size 43322654128

 code_size 32

 log filename: autotune.dbdeep1B.OPQ32_128_IVF4194304_HNSW32_PQ32.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3425 0.4138 0.4167      0.02478       30148046    13
nprobe=1,quantizer_efSearch=4,ht=116      0.1640 0.1894 0.1910      0.00656        3803080    46
nprobe=1,quantizer_efSearch=4,ht=118      0.1655 0.1912 0.1929      0.00666        3803080    46
nprobe=2,quantizer_efSearch=4,ht=126      0.2177 0.2574 0.2590      0.00986        7584844    31
nprobe=1,quantizer_efSearch=16,ht=256     0.2181 0.2506 0.2525      0.01119        3757863    27
nprobe=2,quantizer_efSearch=8,ht=120      0.2691 0.3157 0.3175      0.01135        7584823    27
nprobe=4,quantizer_efSearch=8,ht=122      0.3378 0.4081 0.4106      0.01694       15122435    18
nprobe=4,quantizer_efSearch=16,ht=110     0.3384 0.4008 0.4033      0.01921       15012589    16
nprobe=8,quantizer_efSearch=4,ht=108      0.3425 0.4138 0.4167      0.02474       30148046    13
nprobe=4,quantizer_efSearch=32,ht=110     0.3477 0.4118 0.4143      0.02532       14886236    12
nprobe=8,quantizer_efSearch=4,ht=120      0.3797 0.4720 0.4750      0.02646       30148046    12
nprobe=8,quantizer_efSearch=32,ht=256     0.4522 0.5621 0.5652      0.02794       29616055    11
nprobe=16,quantizer_efSearch=8,ht=114     0.4695 0.5901 0.5939      0.04748       59845976    7
nprobe=32,quantizer_efSearch=8,ht=256     0.5200 0.6848 0.6893      0.05440      119044282    6
nprobe=32,quantizer_efSearch=32,ht=256    0.5717 0.7568 0.7614      0.06233      117442981    5
nprobe=64,quantizer_efSearch=16,ht=118    0.5802 0.7731 0.7783      0.16200      234805247    2
nprobe=64,quantizer_efSearch=64,ht=118    0.6142 0.8255 0.8310      0.17413      231204306    2
nprobe=128,quantizer_efSearch=32,ht=120   0.6287 0.8593 0.8657      0.31089      462043059    1
nprobe=128,quantizer_efSearch=512,ht=116  0.6440 0.8784 0.8851      0.41245      451390884    1
nprobe=256,quantizer_efSearch=128,ht=114  0.6556 0.8940 0.9007      0.57533      898360294    1
nprobe=256,quantizer_efSearch=64,ht=122   0.6591 0.9188 0.9265      0.57710      905931212    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6752 0.9458 0.9540      1.11862     1771189272    1
nprobe=1024,quantizer_efSearch=128,ht=124 0.6805 0.9614 0.9699      2.13891     3448270396    1
nprobe=1024,quantizer_efSearch=128,ht=128 0.6814 0.9641 0.9725      2.02003     3448270396    1
nprobe=2048,quantizer_efSearch=512,ht=256 0.6911 0.9838 0.9932      2.53783     6711763693    1
```

</details>
<details><summary>`OPQ32_128,IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ32` </summary>
Index size 42350050804

 code_size 32

 log filename: autotune.dbdeep1B.OPQ32_128_IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ32.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2995 0.3703 0.3729      0.02815       30723570    11
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=112      0.1501 0.1733 0.1751      0.00607        5230803    50
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=122       0.1848 0.2133 0.2152      0.00754        6758248    40
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=128       0.1904 0.2203 0.2223      0.00792        6741742    38
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=114       0.1960 0.2249 0.2269      0.01011        9742610    30
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=116       0.3090 0.3694 0.3720      0.01551       17829762    20
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=110       0.3316 0.3912 0.3938      0.01836       20862687    17
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=120      0.3546 0.4259 0.4285      0.02002       20839702    15
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=122      0.3614 0.4325 0.4346      0.03182       37605210    10
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=124      0.3844 0.4619 0.4643      0.04503       58727422    7
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=256     0.4228 0.5364 0.5399      0.03318       61370376    10
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=124      0.4412 0.5435 0.5467      0.04320       52300481    7
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=114      0.4625 0.5812 0.5849      0.04956       64552353    7
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=126      0.5088 0.6494 0.6532      0.05959       70198146    6
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=126     0.5776 0.7601 0.7646      0.11042      138716878    3
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=118    0.5809 0.7584 0.7627      0.15346      200970748    2
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=112     0.5833 0.7634 0.7680      0.18583      252200750    2
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=126     0.6126 0.8255 0.8310      0.20763      252200750    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=114    0.6263 0.8400 0.8458      0.32749      475463452    1
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=116    0.6385 0.8641 0.8701      0.32294      474075021    1
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128   0.6496 0.8914 0.8982      0.39445      474711311    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=256  0.6791 0.9482 0.9562      0.47450     1052131770    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=124   0.6873 0.9641 0.9725      1.26783     1905966947    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=512,ht=124   0.6874 0.9642 0.9725      1.39580     2376741398    1
nprobe=1024,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=120 0.6914 0.9715 0.9802      3.40805     3565693173    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=118  0.6917 0.9679 0.9765      4.65060     6792766116    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=256 0.6972 0.9864 0.9963      5.17765     7259238969    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64x4fsr` </summary>
Index size 41360744396

 code_size 32

 log filename: autotune.dbdeep1B.OPQ64_128_IVF1048576_HNSW32_PQ64x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6137 0.9435 0.9719      0.34374     3119499595    1
nprobe=1,quantizer_efSearch=4            0.1988 0.2444 0.2480      0.00373       12960497    81
nprobe=2,quantizer_efSearch=4            0.2579 0.3301 0.3351      0.00479       25932114    63
nprobe=2,quantizer_efSearch=8            0.2998 0.3822 0.3874      0.00650       25937381    47
nprobe=4,quantizer_efSearch=4            0.3129 0.4166 0.4239      0.00695       51883889    44
nprobe=4,quantizer_efSearch=8            0.3687 0.4932 0.5010      0.00902       51849616    34
nprobe=4,quantizer_efSearch=16           0.3909 0.5209 0.5289      0.01171       51491853    26
nprobe=8,quantizer_efSearch=4            0.4150 0.5752 0.5853      0.01237      103766153    25
nprobe=8,quantizer_efSearch=8            0.4293 0.5930 0.6034      0.01298      103567695    24
nprobe=8,quantizer_efSearch=16           0.4571 0.6320 0.6429      0.01605      102907727    19
nprobe=8,quantizer_efSearch=32           0.4659 0.6439 0.6549      0.02081      102338716    15
nprobe=16,quantizer_efSearch=8           0.4905 0.7034 0.7174      0.02486      205992416    13
nprobe=16,quantizer_efSearch=16          0.5050 0.7270 0.7411      0.02608      205258799    12
nprobe=16,quantizer_efSearch=32          0.5184 0.7452 0.7595      0.03297      204043249    10
nprobe=16,quantizer_efSearch=64          0.5216 0.7490 0.7633      0.03981      203310433    8
nprobe=32,quantizer_efSearch=8           0.5229 0.7611 0.7777      0.04654      409678792    7
nprobe=32,quantizer_efSearch=16          0.5517 0.8080 0.8249      0.04756      407758133    7
nprobe=32,quantizer_efSearch=32          0.5634 0.8256 0.8426      0.04960      405665720    7
nprobe=32,quantizer_efSearch=64          0.5672 0.8326 0.8497      0.05804      403882063    6
nprobe=64,quantizer_efSearch=32          0.5802 0.8769 0.8982      0.09683      805381823    4
nprobe=64,quantizer_efSearch=64          0.5877 0.8867 0.9083      0.10054      801109414    3
nprobe=64,quantizer_efSearch=128         0.5889 0.8895 0.9107      0.10846      798466112    3
nprobe=64,quantizer_efSearch=256         0.5896 0.8902 0.9113      0.14440      797355029    3
nprobe=128,quantizer_efSearch=32         0.5939 0.9031 0.9270      0.17693     1596677914    2
nprobe=128,quantizer_efSearch=64         0.6042 0.9212 0.9453      0.17667     1587179418    2
nprobe=128,quantizer_efSearch=128        0.6068 0.9259 0.9500      0.18481     1579951693    2
nprobe=128,quantizer_efSearch=256        0.6085 0.9275 0.9516      0.20880     1576093606    2
nprobe=128,quantizer_efSearch=512        0.6090 0.9275 0.9515      0.26685     1574642339    2
nprobe=256,quantizer_efSearch=128        0.6137 0.9435 0.9719      0.35595     3119499595    1
nprobe=256,quantizer_efSearch=256        0.6147 0.9459 0.9742      0.36358     3107048964    1
nprobe=256,quantizer_efSearch=512        0.6150 0.9463 0.9748      0.40504     3101530608    1
nprobe=512,quantizer_efSearch=128        0.6155 0.9543 0.9816      0.68237     6139103174    1
nprobe=512,quantizer_efSearch=256        0.6191 0.9579 0.9858      0.68198     6112340840    1
nprobe=1024,quantizer_efSearch=256       0.6207 0.9616 0.9913      1.29942    11970941111    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304_HNSW32,PQ64x4fsr` </summary>
Index size 45466757580

 code_size 32

 log filename: autotune.dbdeep1B.OPQ64_128_IVF4194304_HNSW32_PQ64x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6171 0.9275 0.9487      0.32660      898363355    1
nprobe=2,quantizer_efSearch=4            0.2098 0.2579 0.2612      0.00567        7589500    53
nprobe=4,quantizer_efSearch=4            0.2533 0.3255 0.3295      0.00703       15126035    43
nprobe=2,quantizer_efSearch=8            0.2566 0.3179 0.3213      0.00781        7581220    39
nprobe=4,quantizer_efSearch=8            0.3191 0.4096 0.4136      0.00838       15129707    36
nprobe=8,quantizer_efSearch=4            0.3563 0.4769 0.4822      0.01031       30170477    30
nprobe=8,quantizer_efSearch=8            0.3709 0.4958 0.5011      0.01089       30124812    28
nprobe=8,quantizer_efSearch=16           0.4072 0.5431 0.5485      0.01400       29854674    22
nprobe=8,quantizer_efSearch=32           0.4190 0.5595 0.5649      0.01998       29621157    16
nprobe=16,quantizer_efSearch=8           0.4464 0.6142 0.6218      0.02284       59849437    14
nprobe=16,quantizer_efSearch=16          0.4650 0.6395 0.6472      0.02507       59548902    12
nprobe=16,quantizer_efSearch=32          0.4803 0.6598 0.6673      0.03276       59002455    10
nprobe=16,quantizer_efSearch=64          0.4874 0.6696 0.6770      0.04628       58647839    7
nprobe=32,quantizer_efSearch=16          0.5135 0.7276 0.7385      0.04305      118262424    7
nprobe=32,quantizer_efSearch=32          0.5288 0.7499 0.7608      0.04835      117446341    7
nprobe=32,quantizer_efSearch=64          0.5393 0.7641 0.7750      0.05765      116534724    6
nprobe=64,quantizer_efSearch=32          0.5601 0.8185 0.8326      0.09079      233046652    4
nprobe=64,quantizer_efSearch=64          0.5710 0.8332 0.8479      0.10247      231205372    3
nprobe=64,quantizer_efSearch=128         0.5751 0.8396 0.8545      0.12357      229752421    3
nprobe=64,quantizer_efSearch=256         0.5760 0.8405 0.8555      0.15442      228991944    2
nprobe=128,quantizer_efSearch=32         0.5803 0.8587 0.8760      0.15912      462062145    2
nprobe=128,quantizer_efSearch=64         0.5926 0.8839 0.9015      0.16092      458186641    2
nprobe=128,quantizer_efSearch=128        0.5994 0.8945 0.9123      0.17878      454647023    2
nprobe=128,quantizer_efSearch=256        0.6007 0.8964 0.9140      0.22152      452407385    2
nprobe=256,quantizer_efSearch=128        0.6171 0.9275 0.9487      0.31823      898363355    1
nprobe=256,quantizer_efSearch=256        0.6183 0.9324 0.9540      0.34475      892078303    1
nprobe=512,quantizer_efSearch=128        0.6196 0.9414 0.9650      0.60932     1771276365    1
nprobe=512,quantizer_efSearch=256        0.6244 0.9491 0.9726      0.62666     1757916212    1
nprobe=512,quantizer_efSearch=512        0.6265 0.9530 0.9770      0.64172     1747195556    1
nprobe=1024,quantizer_efSearch=512       0.6291 0.9640 0.9883      1.18068     3430202662    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 44492430864

 code_size 32

 log filename: autotune.dbdeep1B.OPQ64_128_IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ64x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.6133 0.9342 0.9555      0.35239     1046922026    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1335 0.1601 0.1625      0.00351        5208092    86
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1953 0.2401 0.2437      0.00417        8908463    72
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2428 0.2979 0.3023      0.00566       10444814    53
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.2430 0.3101 0.3145      0.00595       16210958    51
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2940 0.3713 0.3767      0.00671       17823416    45
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2986 0.3788 0.3840      0.00696       17834794    44
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2988 0.3808 0.3861      0.00747       17822187    41
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3291 0.4164 0.4218      0.00981       20821615    31
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3326 0.4230 0.4283      0.01000       20809102    30
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3337 0.4261 0.4315      0.01119       20814608    27
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.3501 0.4594 0.4659      0.01187       32426818    26
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3710 0.4846 0.4909      0.01232       35414745    25
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3955 0.5198 0.5260      0.01306       35467472    23
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3956 0.5209 0.5274      0.01374       35485266    22
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4154 0.5454 0.5518      0.01719       41096877    18
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4272 0.5735 0.5818      0.02245       64424999    14
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.4461 0.6083 0.6176      0.02421       64490947    13
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4551 0.6109 0.6193      0.02610       70093167    12
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4740 0.6439 0.6533      0.02659       70133611    12
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.4741 0.6463 0.6562      0.03168       70060590    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4858 0.6638 0.6735      0.03517       81152513    9
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4862 0.6640 0.6738      0.03635       81143726    9
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.4863 0.6646 0.6744      0.03877       80958227    8
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5052 0.7000 0.7111      0.04223      127602924    8
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5218 0.7304 0.7434      0.04460      127663334    7
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5241 0.7275 0.7387      0.05932      159171759    6
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5414 0.7582 0.7703      0.05982      159027034    6
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5445 0.7646 0.7776      0.06359      159114895    5
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5457 0.7657 0.7787      0.07659      199621082    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5509 0.7971 0.8124      0.08809      241028352    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.5510 0.7976 0.8127      0.09030      241021572    4
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5579 0.8004 0.8144      0.08997      251673734    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5699 0.8279 0.8434      0.09247      251175816    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5763 0.8373 0.8530      0.09960      271621094    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5775 0.8392 0.8547      0.11716      311990485    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.5776 0.8399 0.8553      0.14732      392845761    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.5922 0.8798 0.8984      0.16268      474374557    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5987 0.8904 0.9086      0.16571      495181778    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.5994 0.8910 0.9098      0.17244      495068519    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6011 0.8945 0.9135      0.19946      616666378    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.6090 0.9158 0.9373      0.31999      929140440    1
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6111 0.9199 0.9413      0.31801      971987679    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6129 0.9321 0.9537      0.32171      969134916    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6132 0.9333 0.9548      0.33077     1052224267    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6133 0.9342 0.9555      0.34153     1052069360    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.6134 0.9342 0.9556      0.45389     1210174561    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6191 0.9455 0.9691      0.58845     1786145058    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6234 0.9531 0.9765      0.60917     1898963920    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6239 0.9533 0.9767      0.62311     1904055825    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.6268 0.9583 0.9839      1.11431     3487779485    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.6276 0.9619 0.9874      1.14640     3488398835    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6282 0.9634 0.9891      1.14396     3565416728    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.6287 0.9646 0.9899      1.19920     3722397521    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6328 0.9693 0.9947      2.08606     3497659555    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.6329 0.9696 0.9950      2.10397     3653711809    1
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF1048576_HNSW32,PQ128x4fsr` </summary>
Index size 74418362316

 code_size 64

 log filename: autotune.dbdeep1B.OPQ128_256_IVF1048576_HNSW32_PQ128x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5792 0.6520 0.6557      0.03975      102331613    8
nprobe=1,quantizer_efSearch=4            0.2254 0.2463 0.2486      0.00684       12982665    44
nprobe=2,quantizer_efSearch=4            0.3034 0.3328 0.3355      0.00908       25973738    34
nprobe=4,quantizer_efSearch=4            0.3796 0.4208 0.4243      0.01298       51935001    24
nprobe=4,quantizer_efSearch=8            0.4484 0.4972 0.5009      0.01630       51858501    19
nprobe=4,quantizer_efSearch=16           0.4750 0.5260 0.5298      0.02152       51479619    14
nprobe=8,quantizer_efSearch=4            0.5168 0.5809 0.5843      0.02532      103791053    12
nprobe=8,quantizer_efSearch=8            0.5316 0.5993 0.6028      0.02817      103560087    11
nprobe=8,quantizer_efSearch=16           0.5676 0.6399 0.6435      0.03250      102883876    10
nprobe=8,quantizer_efSearch=32           0.5792 0.6520 0.6557      0.04065      102331613    8
nprobe=16,quantizer_efSearch=8           0.6210 0.7137 0.7177      0.04913      205976683    7
nprobe=16,quantizer_efSearch=16          0.6403 0.7380 0.7421      0.05500      205206399    6
nprobe=16,quantizer_efSearch=32          0.6558 0.7564 0.7606      0.06045      204014655    5
nprobe=16,quantizer_efSearch=64          0.6589 0.7596 0.7637      0.07570      203299157    4
nprobe=32,quantizer_efSearch=8           0.6669 0.7736 0.7775      0.09174      409730605    4
nprobe=32,quantizer_efSearch=16          0.7044 0.8223 0.8262      0.09454      407760009    4
nprobe=32,quantizer_efSearch=32          0.7174 0.8395 0.8435      0.10009      405646656    3
nprobe=32,quantizer_efSearch=64          0.7237 0.8462 0.8501      0.11498      403879283    3
nprobe=32,quantizer_efSearch=128         0.7238 0.8464 0.8503      0.14328      402905008    3
nprobe=64,quantizer_efSearch=16          0.7342 0.8646 0.8692      0.17438      809873721    2
nprobe=64,quantizer_efSearch=32          0.7573 0.8945 0.8992      0.17300      805361945    2
nprobe=64,quantizer_efSearch=64          0.7644 0.9039 0.9086      0.18717      801093556    2
nprobe=64,quantizer_efSearch=128         0.7671 0.9067 0.9114      0.20504      798458261    2
nprobe=64,quantizer_efSearch=256         0.7683 0.9074 0.9121      0.25682      797348733    2
nprobe=128,quantizer_efSearch=32         0.7746 0.9230 0.9278      0.33528     1596660477    1
nprobe=128,quantizer_efSearch=128        0.7911 0.9456 0.9506      0.35152     1579953738    1
nprobe=128,quantizer_efSearch=256        0.7933 0.9471 0.9521      0.40195     1576089495    1
nprobe=256,quantizer_efSearch=256        0.8049 0.9696 0.9746      0.69834     3107051670    1
nprobe=256,quantizer_efSearch=512        0.8055 0.9702 0.9752      0.77255     3101530178    1
nprobe=512,quantizer_efSearch=128        0.8072 0.9772 0.9823      1.21674     6139278875    1
nprobe=512,quantizer_efSearch=256        0.8100 0.9812 0.9859      1.26710     6112258858    1
nprobe=1024,quantizer_efSearch=512       0.8136 0.9885 0.9932      2.40397     6001239146    1
```

</details>
<details><summary>`OPQ128_256,IVF4194304_HNSW32,PQ128x4fsr` </summary>
Index size 81724480972

 code_size 64

 log filename: autotune.dbdeep1B.OPQ128_256_IVF4194304_HNSW32_PQ128x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.8016 0.9441 0.9489      0.58016      898366496    1
nprobe=1,quantizer_efSearch=4            0.1799 0.1938 0.1954      0.00807        3801188    38
nprobe=2,quantizer_efSearch=8            0.2902 0.3177 0.3202      0.01113        7583791    27
nprobe=4,quantizer_efSearch=4            0.2947 0.3236 0.3266      0.01017       15110332    30
nprobe=4,quantizer_efSearch=8            0.3705 0.4087 0.4119      0.01328       15124238    23
nprobe=8,quantizer_efSearch=8            0.4442 0.4964 0.4997      0.02455       30106687    13
nprobe=8,quantizer_efSearch=16           0.4889 0.5445 0.5479      0.02970       29857250    11
nprobe=8,quantizer_efSearch=32           0.5047 0.5619 0.5653      0.03784       29622200    8
nprobe=16,quantizer_efSearch=8           0.5469 0.6179 0.6217      0.04347       59849026    7
nprobe=16,quantizer_efSearch=16          0.5689 0.6427 0.6464      0.04656       59547830    7
nprobe=16,quantizer_efSearch=32          0.5884 0.6636 0.6676      0.05299       59012067    6
nprobe=32,quantizer_efSearch=8           0.5951 0.6842 0.6884      0.07744      119005218    4
nprobe=32,quantizer_efSearch=16          0.6377 0.7335 0.7378      0.08600      118271309    4
nprobe=32,quantizer_efSearch=64          0.6722 0.7716 0.7758      0.10511      116532947    3
nprobe=64,quantizer_efSearch=32          0.7145 0.8294 0.8338      0.16476      233053368    2
nprobe=64,quantizer_efSearch=64          0.7268 0.8446 0.8490      0.17117      231206710    2
nprobe=64,quantizer_efSearch=128         0.7334 0.8508 0.8552      0.19687      229749543    2
nprobe=128,quantizer_efSearch=64         0.7648 0.8975 0.9027      0.29777      458213701    2
nprobe=128,quantizer_efSearch=128        0.7746 0.9076 0.9128      0.31824      454648493    1
nprobe=128,quantizer_efSearch=256        0.7774 0.9097 0.9148      0.38233      452403913    1
nprobe=128,quantizer_efSearch=512        0.7775 0.9099 0.9150      0.47997      451398103    1
nprobe=256,quantizer_efSearch=128        0.8016 0.9441 0.9489      0.57248      898366496    1
nprobe=256,quantizer_efSearch=256        0.8062 0.9493 0.9543      0.61516      892082455    1
nprobe=256,quantizer_efSearch=512        0.8081 0.9515 0.9564      0.70692      888749729    1
nprobe=512,quantizer_efSearch=256        0.8148 0.9671 0.9727      1.09333     1757910009    1
nprobe=512,quantizer_efSearch=512        0.8190 0.9716 0.9772      1.13924     1747178744    1
nprobe=1024,quantizer_efSearch=512       0.8222 0.9830 0.9887      2.09448     1725403705    1
```

</details>
<details><summary>`OPQ128_256,IVF4194304(IVF1024,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 80884265488

 code_size 64

 log filename: autotune.dbdeep1B.OPQ128_256_IVF4194304_IVF1024_PQ128x4fs_RFlat__PQ128x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.8037 0.9507 0.9560      0.61748     1052034524    1
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.2270 0.2449 0.2480      0.00790        8917307    38
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.2851 0.3122 0.3153      0.01137       16205055    27
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.3463 0.3778 0.3808      0.01376       17807928    22
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3495 0.3826 0.3859      0.01398       17812119    22
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3906 0.4279 0.4311      0.02082       20836453    15
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3908 0.4280 0.4312      0.02056       20845083    15
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3912 0.4288 0.4321      0.02247       20838195    14
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.4151 0.4626 0.4659      0.02750       32430156    11
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4673 0.5185 0.5216      0.02879       35480521    11
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4711 0.5246 0.5275      0.02943       35508336    11
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4963 0.5526 0.5556      0.03862       41093300    8
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.5381 0.6062 0.6100      0.04679       64500189    7
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.5459 0.6147 0.6186      0.05043       64538884    6
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5731 0.6456 0.6493      0.05521       70168444    6
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5780 0.6521 0.6559      0.05726       70171745    6
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.5781 0.6524 0.6563      0.06589       70165398    5
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5868 0.6630 0.6667      0.07590       81222695    4
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5931 0.6708 0.6745      0.07610       81218737    4
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.5993 0.6841 0.6876      0.08209      122086134    4
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.6059 0.6912 0.6951      0.08257      122093623    4
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.6432 0.7345 0.7382      0.08904      127581962    4
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6481 0.7409 0.7448      0.09221      127643171    4
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6638 0.7591 0.7626      0.10366      138586979    3
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6748 0.7738 0.7777      0.12239      158785560    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.6946 0.8037 0.8081      0.15715      241162143    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6974 0.8086 0.8130      0.15962      241098645    2
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7186 0.8352 0.8397      0.18169      251819118    2
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.7209 0.8397 0.8439      0.17641      251728354    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7290 0.8495 0.8537      0.17609      271198443    2
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7292 0.8496 0.8538      0.17351      271966233    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7302 0.8511 0.8553      0.20357      312500315    2
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7309 0.8518 0.8560      0.23241      386873577    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.7610 0.8896 0.8943      0.29531      474789343    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.7647 0.8942 0.8991      0.30671      473945423    1
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7747 0.9059 0.9106      0.32720      535973901    1
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7750 0.9087 0.9136      0.34003      535936737    1
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7756 0.9069 0.9116      0.37708      616672419    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.7872 0.9291 0.9344      0.55180      912294707    1
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.7988 0.9422 0.9473      0.55679      931730155    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.7990 0.9451 0.9504      0.56012      931527878    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8037 0.9507 0.9560      0.60814     1052024606    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.8039 0.9509 0.9562      0.68330     1210135686    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.8104 0.9637 0.9698      1.08438     1785999384    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.8149 0.9705 0.9766      1.04158     1824543913    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8153 0.9712 0.9773      1.09276     1898782330    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.8216 0.9831 0.9883      2.01908     1796387506    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.8230 0.9847 0.9899      2.07877     1874939780    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.8236 0.9837 0.9891      2.15003     2032216629    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.8240 0.9887 0.9950      3.92782     1874847841    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 72701067666

 code_size 64

 log filename: autotune.dbdeep1B.OPQ16_64_IVF1048576_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.6050 0.7276 0.7286      0.04532      797270200    7
k_factor_rf=1,nprobe=1,quantizer_efSearch=4      0.1859 0.2088 0.2096      0.00484       12909444    62
k_factor_rf=1,nprobe=2,quantizer_efSearch=4      0.2353 0.2649 0.2655      0.00519       25652045    58
k_factor_rf=1,nprobe=2,quantizer_efSearch=8      0.2790 0.3127 0.3136      0.00619       25673360    49
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.3557 0.3997 0.4007      0.00730      102515905    42
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.3668 0.4116 0.4126      0.00763      102391463    40
k_factor_rf=1,nprobe=8,quantizer_efSearch=16     0.3903 0.4385 0.4396      0.00953      101724724    32
k_factor_rf=2,nprobe=8,quantizer_efSearch=4      0.3909 0.4491 0.4498      0.00966      102515905    32
k_factor_rf=2,nprobe=8,quantizer_efSearch=16     0.4294 0.4935 0.4944      0.01184      101724724    26
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.4471 0.5140 0.5147      0.01280      203607308    24
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.4623 0.5314 0.5323      0.01387      202894050    22
k_factor_rf=2,nprobe=16,quantizer_efSearch=32    0.4755 0.5461 0.5470      0.01689      201805828    18
k_factor_rf=4,nprobe=16,quantizer_efSearch=8     0.4894 0.5746 0.5753      0.01774      203607308    17
k_factor_rf=4,nprobe=16,quantizer_efSearch=16    0.5060 0.5945 0.5953      0.01877      202894050    17
k_factor_rf=4,nprobe=32,quantizer_efSearch=8     0.5087 0.5972 0.5980      0.02227      405092889    14
k_factor_rf=4,nprobe=32,quantizer_efSearch=16    0.5368 0.6309 0.6318      0.02358      403422161    13
k_factor_rf=4,nprobe=32,quantizer_efSearch=32    0.5474 0.6427 0.6435      0.02610      401503910    12
k_factor_rf=8,nprobe=32,quantizer_efSearch=8     0.5499 0.6592 0.6603      0.03299      405092889    10
k_factor_rf=4,nprobe=64,quantizer_efSearch=32    0.5561 0.6525 0.6533      0.03346      797270200    9
k_factor_rf=8,nprobe=32,quantizer_efSearch=32    0.5920 0.7110 0.7121      0.03623      401503910    9
k_factor_rf=8,nprobe=32,quantizer_efSearch=64    0.5971 0.7164 0.7175      0.04145      399777703    8
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.6050 0.7276 0.7286      0.04466      797270200    7
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.6113 0.7357 0.7367      0.04971      793236768    7
k_factor_rf=16,nprobe=32,quantizer_efSearch=32   0.6209 0.7609 0.7625      0.05570      401503910    6
k_factor_rf=16,nprobe=64,quantizer_efSearch=16   0.6222 0.7648 0.7663      0.06234      801768386    5
k_factor_rf=16,nprobe=64,quantizer_efSearch=64   0.6490 0.7985 0.8000      0.06849      793236768    5
k_factor_rf=16,nprobe=64,quantizer_efSearch=128  0.6507 0.8000 0.8015      0.07844      790708757    4
k_factor_rf=16,nprobe=128,quantizer_efSearch=128 0.6590 0.8116 0.8130      0.09482     1565328163    4
k_factor_rf=32,nprobe=64,quantizer_efSearch=256  0.6742 0.8454 0.8476      0.13493      789680740    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.6923 0.8687 0.8704      0.16230     3091684534    2
k_factor_rf=64,nprobe=128,quantizer_efSearch=64  0.7044 0.8986 0.9016      0.18691     1572287091    2
k_factor_rf=64,nprobe=128,quantizer_efSearch=128 0.7067 0.9023 0.9053      0.19608     1565328163    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=64  0.7089 0.9032 0.9064      0.21618     3107957291    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=128 0.7151 0.9117 0.9148      0.22524     3091684534    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=512 0.7168 0.9145 0.9177      0.28534     3074957284    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=256 0.7176 0.9135 0.9166      0.27278     6059735726    2
```

</details>
<details><summary>`OPQ16_64,IVF4194304_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 74810331538

 code_size 64

 log filename: autotune.dbdeep1B.OPQ16_64_IVF4194304_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.5914 0.7152 0.7165      0.03404      226843960    9
k_factor_rf=1,nprobe=1,quantizer_efSearch=4      0.1551 0.1748 0.1757      0.00521        3723322    58
k_factor_rf=1,nprobe=2,quantizer_efSearch=4      0.2030 0.2283 0.2291      0.00543        7427430    56
k_factor_rf=1,nprobe=2,quantizer_efSearch=8      0.2570 0.2881 0.2889      0.00663        7402703    46
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.3245 0.3667 0.3674      0.00702       29360659    43
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.3408 0.3842 0.3849      0.00746       29295052    41
k_factor_rf=1,nprobe=16,quantizer_efSearch=4     0.3511 0.3913 0.3919      0.00795       58398177    38
k_factor_rf=2,nprobe=8,quantizer_efSearch=4      0.3558 0.4106 0.4112      0.00879       29360659    35
k_factor_rf=1,nprobe=8,quantizer_efSearch=16     0.3762 0.4239 0.4246      0.00960       29020523    32
k_factor_rf=2,nprobe=16,quantizer_efSearch=4     0.3875 0.4457 0.4462      0.00992       58398177    31
k_factor_rf=1,nprobe=32,quantizer_efSearch=8     0.3933 0.4377 0.4383      0.01071      115769980    29
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.4300 0.4939 0.4945      0.01108       58214225    28
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.4504 0.5175 0.5181      0.01243       57911009    25
k_factor_rf=4,nprobe=16,quantizer_efSearch=8     0.4614 0.5425 0.5436      0.01589       58214225    19
k_factor_rf=4,nprobe=16,quantizer_efSearch=16    0.4820 0.5682 0.5693      0.01685       57911009    18
k_factor_rf=2,nprobe=32,quantizer_efSearch=32    0.4852 0.5555 0.5560      0.01864      114285881    17
k_factor_rf=4,nprobe=32,quantizer_efSearch=16    0.5161 0.6098 0.6109      0.01972      115119907    16
k_factor_rf=4,nprobe=64,quantizer_efSearch=16    0.5301 0.6228 0.6237      0.02379      228592878    13
k_factor_rf=4,nprobe=64,quantizer_efSearch=32    0.5509 0.6488 0.6496      0.02706      226843960    12
k_factor_rf=8,nprobe=32,quantizer_efSearch=32    0.5622 0.6794 0.6806      0.03195      114285881    10
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.5914 0.7152 0.7165      0.03613      226843960    9
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.6040 0.7309 0.7322      0.04268      225048284    8
k_factor_rf=8,nprobe=64,quantizer_efSearch=128   0.6083 0.7367 0.7380      0.05589      223714238    6
k_factor_rf=16,nprobe=64,quantizer_efSearch=64   0.6291 0.7801 0.7815      0.06078      225048284    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=64  0.6482 0.8051 0.8065      0.06866      446213382    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=128 0.6547 0.8142 0.8156      0.08072      442897380    4
k_factor_rf=32,nprobe=128,quantizer_efSearch=64  0.6681 0.8464 0.8486      0.10171      446213382    3
k_factor_rf=32,nprobe=128,quantizer_efSearch=128 0.6742 0.8550 0.8571      0.11302      442897380    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.6830 0.8682 0.8701      0.13069      875478005    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=256 0.6878 0.8738 0.8758      0.15324      869808308    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=64  0.6897 0.8899 0.8927      0.18224      882695750    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=128 0.6995 0.9035 0.9064      0.18390      875478005    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=256 0.7046 0.9095 0.9125      0.20085      869808308    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=256 0.7136 0.9181 0.9209      0.23708     1714374034    2
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 73769607126

 code_size 64

 log filename: autotune.dbdeep1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.4720 0.5658 0.5669      0.02901       62855700    11
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1711 0.1921 0.1928      0.00503       10141010    60
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.2354 0.2608 0.2615      0.00567       30008003    53
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2460 0.2748 0.2755      0.00626       13151538    48
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2817 0.3153 0.3161      0.00655       20333211    46
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3328 0.3714 0.3721      0.00731       34650266    42
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=4      0.3359 0.3836 0.3842      0.01089       20333375    28
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3858 0.4303 0.4309      0.01223      119309387    25
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3994 0.4539 0.4544      0.01283      118309143    24
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4252 0.4734 0.4741      0.01692      135231161    18
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4587 0.5242 0.5246      0.01575      124748883    20
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4735 0.5410 0.5416      0.01929      135107443    16
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4915 0.5774 0.5782      0.02733      229891142    11
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.5346 0.6312 0.6322      0.03100      135195018    10
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5437 0.6396 0.6405      0.02909      246110518    11
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.5939 0.7172 0.7184      0.04058      245859622    8
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6196 0.7477 0.7488      0.08036      910379179    4
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.6450 0.7965 0.7977      0.08618      892486665    4
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.6749 0.8543 0.8562      0.14065      891328582    3
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.6763 0.8577 0.8599      0.15676      602649728    2
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.6767 0.8522 0.8541      0.16571     1730122502    2
k_factor_rf=64,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6903 0.8890 0.8919      0.18130      911331834    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7108 0.9138 0.9167      0.23291     1858089628    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.7158 0.9207 0.9238      0.36563     2008314960    1
```

</details>
<details><summary>`OPQ56_112,IVF1048576_HNSW32,PQ7+56` </summary>
Index size 71763785180

 code_size 63

 log filename: autotune.dbdeep1B.OPQ56_112_IVF1048576_HNSW32_PQ7+56.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.2724 0.3121 0.3132      0.01039              0    29
nprobe=1,quantizer_efSearch=4,ht=56,k_factor=1       0.2085 0.2364 0.2374      0.00895              0    34
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.2366 0.2685 0.2697      0.01041              0    29
nprobe=2,quantizer_efSearch=4,ht=56,k_factor=1       0.2724 0.3121 0.3132      0.01041              0    29
nprobe=2,quantizer_efSearch=8,ht=56,k_factor=1       0.3140 0.3614 0.3627      0.01197              0    26
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.3336 0.3861 0.3873      0.01319              0    23
nprobe=2,quantizer_efSearch=32,ht=56,k_factor=1      0.3347 0.3852 0.3866      0.01924              0    16
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.4355 0.5111 0.5125      0.01960              0    16
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=1       0.4484 0.5274 0.5288      0.02010              0    15
nprobe=8,quantizer_efSearch=32,ht=56,k_factor=1      0.4903 0.5785 0.5800      0.02731              0    11
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.5053 0.6033 0.6048      0.03172              0    10
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.5206 0.6224 0.6239      0.03333              0    10
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.5340 0.6407 0.6422      0.03787              0    8
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.5451 0.6663 0.6675      0.03935              0    8
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=4     0.5617 0.7001 0.7016      0.05066              0    6
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=4     0.5762 0.7188 0.7203      0.05296              0    6
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.5840 0.7201 0.7213      0.06052              0    5
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.5967 0.7358 0.7370      0.06405              0    5
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=2     0.6008 0.7416 0.7428      0.07220              0    5
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=4     0.6063 0.7642 0.7655      0.07095              0    5
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=4     0.6239 0.7875 0.7888      0.08211              0    4
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=8     0.6318 0.8104 0.8122      0.09693              0    4
nprobe=32,quantizer_efSearch=128,ht=56,k_factor=8    0.6367 0.8174 0.8192      0.11638              0    3
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=4     0.6474 0.8244 0.8258      0.13065              0    3
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=4    0.6497 0.8266 0.8280      0.14431              0    3
nprobe=64,quantizer_efSearch=32,ht=56,k_factor=8     0.6588 0.8537 0.8556      0.13922              0    3
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.6647 0.8636 0.8655      0.15317              0    3
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=8    0.6671 0.8659 0.8678      0.16290              0    2
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=16   0.6746 0.8895 0.8917      0.21747              0    2
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.6799 0.8884 0.8904      0.22505              0    2
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.6835 0.8926 0.8947      0.24134              0    2
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=8   0.6848 0.8941 0.8962      0.24756              0    2
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=16  0.6939 0.9224 0.9248      0.27935              0    2
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=16  0.6953 0.9240 0.9264      0.28709              0    2
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=32  0.6985 0.9388 0.9417      0.35822              0    1
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=32  0.6997 0.9404 0.9434      0.36568              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.7035 0.9391 0.9419      0.41648              0    1
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=16  0.7037 0.9396 0.9424      0.45879              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.7088 0.9585 0.9619      0.49461              0    1
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=64  0.7096 0.9649 0.9693      0.66385              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=64  0.7111 0.9672 0.9716      0.66892              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.7136 0.9677 0.9711      0.76153              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=64  0.7159 0.9774 0.9820      0.99154              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=64 0.7188 0.9829 0.9876      1.49782              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=64 0.7194 0.9844 0.9891      2.47300              0    1
```

</details>
<details><summary>`OPQ56_112,IVF4194304_HNSW32,PQ7+56` </summary>
Index size 74054310876

 code_size 63

 log filename: autotune.dbdeep1B.OPQ56_112_IVF4194304_HNSW32_PQ7+56.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.2085 0.2499 0.2511      0.00964              0    32
nprobe=1,quantizer_efSearch=4,ht=56,k_factor=1       0.1622 0.1903 0.1915      0.00845              0    36
nprobe=2,quantizer_efSearch=4,ht=56,k_factor=1       0.2085 0.2499 0.2511      0.00969              0    31
nprobe=2,quantizer_efSearch=8,ht=56,k_factor=1       0.2582 0.3088 0.3103      0.01126              0    27
nprobe=2,quantizer_efSearch=16,ht=56,k_factor=1      0.2785 0.3345 0.3360      0.01403              0    22
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.3507 0.4419 0.4435      0.01614              0    19
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=1       0.3659 0.4600 0.4616      0.01673              0    18
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=2       0.3744 0.4805 0.4822      0.02135              0    15
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.3926 0.5016 0.5032      0.02334              0    13
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.4329 0.5546 0.5563      0.02479              0    13
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.4494 0.5771 0.5790      0.02654              0    12
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=4      0.4527 0.6068 0.6099      0.03887              0    8
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.5044 0.6806 0.6829      0.04616              0    7
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=4     0.5140 0.7101 0.7138      0.05554              0    6
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=2     0.5304 0.7176 0.7199      0.05937              0    6
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=4     0.5394 0.7471 0.7509      0.06783              0    5
nprobe=64,quantizer_efSearch=32,ht=56,k_factor=2     0.5480 0.7474 0.7502      0.07479              0    5
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.5575 0.7622 0.7652      0.08063              0    4
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=4     0.5697 0.8038 0.8092      0.09094              0    4
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.5718 0.7880 0.7913      0.10400              0    3
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=4    0.5742 0.8101 0.8156      0.07488              0    5
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.5932 0.8482 0.8541      0.10147              0    3
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.5954 0.8684 0.8765      0.10862              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.6022 0.8788 0.8872      0.11696              0    3
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=8   0.6033 0.8800 0.8884      0.14207              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=32  0.6071 0.8978 0.9096      0.22522              0    2
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=32  0.6082 0.8994 0.9111      0.23154              0    2
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=8   0.6138 0.9027 0.9118      0.17585              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=8   0.6168 0.9071 0.9163      0.20415              0    2
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=8   0.6171 0.9094 0.9186      0.25348              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=8   0.6214 0.9176 0.9273      0.31832              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=8   0.6230 0.9212 0.9309      0.47447              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.6234 0.9350 0.9488      0.40260              0    1
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=32  0.6238 0.9372 0.9511      0.45470              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.6251 0.9392 0.9514      0.47521              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.6268 0.9432 0.9555      0.52336              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.6282 0.9498 0.9653      0.55767              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.6298 0.9541 0.9697      0.60857              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=16 0.6302 0.9504 0.9630      0.85391              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=32 0.6312 0.9573 0.9733      0.88583              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=64 0.6349 0.9671 0.9855      1.13297              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=64 0.6359 0.9693 0.9882      1.75881              0    1
```

</details>
<details><summary>`OPQ56_112,IVF4194304(IVF1024,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 73064810656

 code_size 63

 log filename: autotune.dbdeep1B.OPQ56_112_IVF4194304_IVF1024_PQ56x4fs_RFlat__PQ7+56.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                           0.6100 0.8673 0.8727      0.35879       44160066    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=56,k_factor=1       0.1215 0.1405 0.1418      0.00739        1510837    41
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2,ht=56,k_factor=1       0.1978 0.2350 0.2364      0.00886        3002232    34
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=56,k_factor=1       0.2807 0.3391 0.3405      0.01291        5946824    24
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=56,k_factor=1       0.2921 0.3550 0.3566      0.01086        2999301    28
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1       0.2923 0.3532 0.3546      0.01593       11661013    19
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=56,k_factor=2       0.3415 0.4288 0.4303      0.01795        3002232    17
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=56,k_factor=2     0.3950 0.5087 0.5105      0.02614        2994884    12
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=56,k_factor=1     0.4368 0.5538 0.5556      0.03443        5975724    9
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=2      0.4721 0.6166 0.6184      0.03526       11463270    9
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2     0.4838 0.6347 0.6365      0.04159       22174982    8
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=56,k_factor=4     0.4999 0.6741 0.6763      0.06366       22596112    5
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=56,k_factor=1    0.5131 0.6642 0.6661      0.05880       22415712    6
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=4     0.5776 0.8093 0.8135      0.10968       44160066    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8     0.5830 0.8307 0.8368      0.13151       43234565    3
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=56,k_factor=4    0.5991 0.8468 0.8521      0.18619       85535996    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=56,k_factor=4    0.6119 0.8707 0.8764      0.29277       85535996    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=56,k_factor=8    0.6187 0.9041 0.9126      0.27638       43465077    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=56,k_factor=8   0.6190 0.9048 0.9133      0.32197       44160066    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=56,k_factor=8   0.6212 0.9086 0.9173      0.32526       85535996    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32   0.6252 0.9286 0.9417      0.43697       85535996    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=8    0.6274 0.9196 0.9289      0.47566       85535996    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=16  0.6277 0.9291 0.9406      0.41085      166457685    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=32  0.6281 0.9362 0.9496      0.53507      637689892    1
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=64  0.6291 0.9394 0.9540      0.65680      324676858    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=32  0.6322 0.9469 0.9627      0.91857       44160066    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=56,k_factor=32  0.6360 0.9525 0.9684      0.70554       85535996    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=16  0.6368 0.9493 0.9620      0.85335       82916865    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=16 0.6378 0.9517 0.9646      1.01568      637689892    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=32 0.6404 0.9629 0.9798      1.11352      324676858    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=64 0.6406 0.9670 0.9865      1.85740      166457685    1
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=32 0.6416 0.9655 0.9829      1.81000      324676858    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=64 0.6427 0.9716 0.9914      1.84869      324676858    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64` </summary>
Index size 72830801840

 code_size 64

 log filename: autotune.dbdeep1B.OPQ64_128_IVF1048576_HNSW32_PQ64.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.8114 0.9184 0.9191      0.53830     1596686159    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0039 0.0045 0.0052      0.00618       12968331    49
nprobe=1,quantizer_efSearch=4,ht=232      0.2188 0.2322 0.2328      0.00704       12968331    43
nprobe=1,quantizer_efSearch=16,ht=228     0.2491 0.2651 0.2659      0.01095       12924519    28
nprobe=2,quantizer_efSearch=8,ht=230      0.3297 0.3538 0.3544      0.01211       25955912    25
nprobe=4,quantizer_efSearch=4,ht=226      0.3460 0.3688 0.3695      0.01721       51860390    18
nprobe=4,quantizer_efSearch=4,ht=228      0.3546 0.3789 0.3796      0.01748       51860390    18
nprobe=4,quantizer_efSearch=16,ht=236     0.4734 0.5103 0.5109      0.02330       51484848    13
nprobe=8,quantizer_efSearch=8,ht=242      0.5473 0.5947 0.5953      0.03970      103557453    8
nprobe=16,quantizer_efSearch=32,ht=224    0.5782 0.6254 0.6260      0.06637      204038152    5
nprobe=16,quantizer_efSearch=32,ht=232    0.6458 0.7077 0.7083      0.06989      204038152    5
nprobe=16,quantizer_efSearch=64,ht=234    0.6593 0.7237 0.7242      0.07904      203300320    4
nprobe=16,quantizer_efSearch=16,ht=246    0.6671 0.7360 0.7369      0.08152      205202461    4
nprobe=16,quantizer_efSearch=128,ht=236   0.6675 0.7342 0.7348      0.09460      202956357    4
nprobe=32,quantizer_efSearch=32,ht=230    0.6941 0.7653 0.7660      0.12312      405666148    3
nprobe=32,quantizer_efSearch=32,ht=234    0.7200 0.7986 0.7991      0.12772      405666148    3
nprobe=32,quantizer_efSearch=128,ht=240   0.7459 0.8322 0.8328      0.16101      402904790    2
nprobe=32,quantizer_efSearch=256,ht=250   0.7585 0.8481 0.8490      0.22011      402579923    2
nprobe=64,quantizer_efSearch=256,ht=236   0.7814 0.8754 0.8761      0.29355      797343755    2
nprobe=64,quantizer_efSearch=64,ht=250    0.8033 0.9059 0.9068      0.33202      801097186    1
nprobe=64,quantizer_efSearch=256,ht=246   0.8040 0.9058 0.9066      0.34085      797343755    1
nprobe=128,quantizer_efSearch=64,ht=236   0.8057 0.9077 0.9084      0.47761     1587169359    1
nprobe=128,quantizer_efSearch=64,ht=238   0.8126 0.9188 0.9194      0.48911     1587169359    1
nprobe=128,quantizer_efSearch=64,ht=250   0.8302 0.9429 0.9438      0.62782     1587169359    1
nprobe=128,quantizer_efSearch=256,ht=256  0.8389 0.9520 0.9529      0.75385     1576075192    1
nprobe=256,quantizer_efSearch=256,ht=240  0.8397 0.9550 0.9556      1.00057     3107041071    1
nprobe=256,quantizer_efSearch=64,ht=248   0.8417 0.9599 0.9607      1.13730     3136148904    1
nprobe=512,quantizer_efSearch=128,ht=240  0.8427 0.9623 0.9629      1.85759     6139480313    1
nprobe=512,quantizer_efSearch=128,ht=242  0.8477 0.9683 0.9689      1.92645     6139480313    1
nprobe=1024,quantizer_efSearch=256,ht=240 0.8503 0.9721 0.9727      3.62158    11971628367    1
nprobe=2048,quantizer_efSearch=512,ht=250 0.8664 0.9942 0.9951      8.75813    23243246391    1
nprobe=2048,quantizer_efSearch=512,ht=512 0.8682 0.9969 0.9977      9.38569    23243246391    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304_HNSW32,PQ64` </summary>
Index size 75322654128

 code_size 64

 log filename: autotune.dbdeep1B.OPQ64_128_IVF4194304_HNSW32_PQ64.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.7668 0.8660 0.8668      0.37788      462047941    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0019 0.0030 0.0038      0.00638        3801071    47
nprobe=1,quantizer_efSearch=4,ht=232      0.1728 0.1825 0.1833      0.00669        3801071    45
nprobe=2,quantizer_efSearch=8,ht=230      0.2725 0.2901 0.2907      0.01136        7588191    27
nprobe=4,quantizer_efSearch=16,ht=236     0.3941 0.4240 0.4248      0.02091       15013535    15
nprobe=8,quantizer_efSearch=8,ht=242      0.4525 0.4923 0.4931      0.03027       30110610    10
nprobe=16,quantizer_efSearch=32,ht=224    0.4889 0.5261 0.5268      0.05886       59012811    6
nprobe=16,quantizer_efSearch=16,ht=246    0.5839 0.6402 0.6410      0.05935       59543193    6
nprobe=16,quantizer_efSearch=128,ht=236   0.5875 0.6418 0.6426      0.09165       58434144    4
nprobe=32,quantizer_efSearch=8,ht=238     0.6002 0.6626 0.6635      0.09773      119031174    4
nprobe=32,quantizer_efSearch=32,ht=230    0.6142 0.6707 0.6714      0.10107      117453737    3
nprobe=32,quantizer_efSearch=32,ht=234    0.6438 0.7063 0.7072      0.10209      117453737    3
nprobe=32,quantizer_efSearch=16,ht=250    0.6623 0.7343 0.7352      0.10965      118271390    3
nprobe=32,quantizer_efSearch=128,ht=240   0.6858 0.7571 0.7579      0.13299      115977669    3
nprobe=32,quantizer_efSearch=256,ht=250   0.7003 0.7757 0.7766      0.17423      115729907    2
nprobe=64,quantizer_efSearch=64,ht=250    0.7515 0.8450 0.8459      0.20343      231205599    2
nprobe=64,quantizer_efSearch=256,ht=246   0.7558 0.8480 0.8488      0.24760      228992053    2
nprobe=128,quantizer_efSearch=64,ht=238   0.7688 0.8655 0.8664      0.35219      458203199    1
nprobe=128,quantizer_efSearch=64,ht=250   0.7918 0.8987 0.8996      0.38485      458203199    1
nprobe=128,quantizer_efSearch=256,ht=256  0.8064 0.9136 0.9145      0.45427      452395020    1
nprobe=256,quantizer_efSearch=256,ht=240  0.8184 0.9277 0.9285      0.71802      892068217    1
nprobe=512,quantizer_efSearch=128,ht=240  0.8248 0.9380 0.9388      1.32748     1771292631    1
nprobe=512,quantizer_efSearch=128,ht=242  0.8308 0.9465 0.9473      1.29946     1771292631    1
nprobe=1024,quantizer_efSearch=256,ht=240 0.8382 0.9556 0.9564      2.55429     3454280829    1
nprobe=2048,quantizer_efSearch=512,ht=512 0.8638 0.9933 0.9941      3.31723     6711838986    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ64` </summary>
Index size 74350047732

 code_size 64

 log filename: autotune.dbdeep1B.OPQ64_128_IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ64.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2431 0.2578 0.2585      0.11677      318401575    3
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=2         0.0002 0.0004 0.0009      0.00569        5233565    53
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=24        0.0010 0.0014 0.0022      0.00582        5236064    52
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=512       0.1701 0.1798 0.1806      0.00577        5234911    53
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=224      0.1727 0.1829 0.1836      0.00815        6739116    37
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=244       0.2068 0.2189 0.2196      0.00785        6742103    39
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=240      0.2263 0.2395 0.2402      0.01121        9743160    27
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=244      0.3541 0.3803 0.3812      0.01875       17827459    17
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=232       0.3794 0.4048 0.4057      0.02386       26495693    13
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=244      0.5130 0.5534 0.5544      0.04557       52152429    7
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=254     0.5661 0.6162 0.6171      0.07853       64528956    4
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=246     0.5959 0.6482 0.6491      0.07925       70174721    4
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=236    0.6654 0.7264 0.7273      0.14228      138573465    3
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=242   0.6924 0.7615 0.7624      0.17692      276722911    2
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=248    0.7020 0.7731 0.7740      0.18255      199708999    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=252     0.7542 0.8395 0.8405      0.22028      252011649    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=252     0.7635 0.8505 0.8514      0.22916      272222087    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=254    0.7652 0.8527 0.8536      0.27572      392287885    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=246    0.8025 0.9013 0.9023      0.41489      495168864    1
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=252    0.8075 0.9083 0.9092      0.43157      495168864    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=512  0.8423 0.9565 0.9574      0.62965     1210292555    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=244  0.8496 0.9637 0.9646      1.52021     1904281400    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=512  0.8673 0.9896 0.9905      1.78860     3570820390    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=512 0.8723 0.9962 0.9971      5.43312     7259582441    1
```

</details>
