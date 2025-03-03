# Detailed logs for dataset bigann10M

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF16384_HNSW32,PQ16x4fs` </summary>
Index size 170987980

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF16384_HNSW32_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0916 0.3115 0.5336      0.00124       28216986    242
nprobe=1,quantizer_efSearch=4            0.0674 0.1993 0.2893      0.00072        7144882    419
nprobe=1,quantizer_efSearch=8            0.0685 0.2054 0.2973      0.00086        7120023    348
nprobe=2,quantizer_efSearch=4            0.0802 0.2534 0.3974      0.00090       14207265    334
nprobe=2,quantizer_efSearch=8            0.0842 0.2642 0.4141      0.00105       14173391    288
nprobe=4,quantizer_efSearch=4            0.0854 0.2934 0.5002      0.00117       28252608    257
nprobe=4,quantizer_efSearch=8            0.0916 0.3115 0.5336      0.00127       28216986    238
nprobe=8,quantizer_efSearch=4            0.0948 0.3317 0.6148      0.00146       55881123    206
nprobe=16,quantizer_efSearch=8           0.0961 0.3558 0.6945      0.00203      109985019    148
nprobe=16,quantizer_efSearch=32          0.0972 0.3614 0.7049      0.00261      109720554    115
```

</details>
<details><summary>`OPQ16_64,IVF16384_HNSW32,PQ16x4fsr` </summary>
Index size 170980812

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF16384_HNSW32_PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2116 0.6015 0.8252      0.00519      109386257    58
nprobe=2,quantizer_efSearch=4            0.1468 0.3562 0.4157      0.00094       14153568    319
nprobe=2,quantizer_efSearch=8            0.1563 0.3768 0.4396      0.00110       14153085    274
nprobe=4,quantizer_efSearch=4            0.1707 0.4455 0.5498      0.00123       28142898    244
nprobe=4,quantizer_efSearch=8            0.1845 0.4745 0.5867      0.00140       28119627    215
nprobe=4,quantizer_efSearch=16           0.1855 0.4801 0.5924      0.00173       28086820    174
nprobe=8,quantizer_efSearch=4            0.1968 0.5342 0.6986      0.00183       55693164    164
nprobe=8,quantizer_efSearch=8            0.2004 0.5457 0.7128      0.00183       55650395    164
nprobe=16,quantizer_efSearch=4           0.2032 0.5711 0.7772      0.00260      109693761    116
nprobe=16,quantizer_efSearch=16          0.2112 0.5994 0.8216      0.00298      109492319    101
nprobe=16,quantizer_efSearch=32          0.2116 0.6015 0.8252      0.00378      109386257    80
nprobe=16,quantizer_efSearch=64          0.2117 0.6013 0.8258      0.00475      109349220    64
nprobe=32,quantizer_efSearch=16          0.2149 0.6243 0.8844      0.00450      215278421    67
nprobe=32,quantizer_efSearch=32          0.2153 0.6279 0.8910      0.00491      215012528    62
nprobe=32,quantizer_efSearch=64          0.2157 0.6287 0.8924      0.00590      214926764    52
nprobe=32,quantizer_efSearch=128         0.2159 0.6289 0.8931      0.00803      214934808    38
nprobe=32,quantizer_efSearch=256         0.2160 0.6289 0.8931      0.01386      214928382    22
nprobe=64,quantizer_efSearch=32          0.2163 0.6343 0.9236      0.01785      421058198    17
nprobe=64,quantizer_efSearch=64          0.2165 0.6353 0.9257      0.01839      420680098    17
nprobe=256,quantizer_efSearch=64         0.2169 0.6407 0.9455      0.07470     1604400264    5
nprobe=256,quantizer_efSearch=128        0.2172 0.6409 0.9464      0.07956     1601641485    4
nprobe=512,quantizer_efSearch=128        0.2178 0.6414 0.9475      0.13733     3127612272    3
nprobe=512,quantizer_efSearch=32         0.2180 0.6384 0.9397      0.12751     2974780302    3
nprobe=512,quantizer_efSearch=256        0.2181 0.6413 0.9479      0.14468     3122746400    3
nprobe=1024,quantizer_efSearch=64        0.2190 0.6413 0.9468      0.25081     5424923823    2
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fs` </summary>
Index size 334629580

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_HNSW32_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0951 0.2779 0.3777      0.00163        1745880    184
nprobe=2,quantizer_efSearch=4            0.0776 0.2054 0.2497      0.00100         873523    301
nprobe=4,quantizer_efSearch=4            0.0850 0.2512 0.3425      0.00117        1748468    256
nprobe=4,quantizer_efSearch=8            0.0951 0.2779 0.3777      0.00163        1745880    184
nprobe=8,quantizer_efSearch=4            0.1003 0.3215 0.4797      0.00179        3485253    168
nprobe=16,quantizer_efSearch=8           0.1087 0.3613 0.5939      0.00272        6948361    111
nprobe=16,quantizer_efSearch=32          0.1123 0.3714 0.6088      0.00498        6927118    61
nprobe=16,quantizer_efSearch=128         0.1125 0.3726 0.6101      0.01259        6920426    24
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fsr` </summary>
Index size 334564044

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_HNSW32_PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1680 0.3282 0.3506      0.00150        1743458    200
nprobe=2,quantizer_efSearch=4            0.1346 0.2411 0.2519      0.00113         868539    266
nprobe=4,quantizer_efSearch=4            0.1680 0.3282 0.3506      0.00143        1743458    210
nprobe=4,quantizer_efSearch=8            0.1803 0.3619 0.3864      0.00197        1740659    152
nprobe=8,quantizer_efSearch=4            0.2058 0.4475 0.4921      0.00225        3479181    134
nprobe=8,quantizer_efSearch=8            0.2090 0.4567 0.5028      0.00241        3474709    125
nprobe=16,quantizer_efSearch=4           0.2233 0.5174 0.5926      0.00343        6946330    88
nprobe=16,quantizer_efSearch=8           0.2296 0.5442 0.6265      0.00379        6940195    80
nprobe=16,quantizer_efSearch=16          0.2319 0.5566 0.6401      0.00446        6928317    68
nprobe=32,quantizer_efSearch=8           0.2392 0.5975 0.7166      0.00552       13799891    55
nprobe=32,quantizer_efSearch=16          0.2447 0.6194 0.7459      0.00634       13772154    48
nprobe=32,quantizer_efSearch=32          0.2464 0.6274 0.7568      0.00799       13748136    38
nprobe=32,quantizer_efSearch=64          0.2471 0.6293 0.7604      0.01041       13733396    29
nprobe=64,quantizer_efSearch=16          0.2499 0.6504 0.8159      0.01677       27306453    18
nprobe=64,quantizer_efSearch=32          0.2526 0.6636 0.8366      0.01786       27246441    17
nprobe=64,quantizer_efSearch=64          0.2538 0.6678 0.8440      0.01918       27211004    16
nprobe=128,quantizer_efSearch=32         0.2541 0.6828 0.8882      0.03221       53883992    10
nprobe=128,quantizer_efSearch=64         0.2553 0.6903 0.9016      0.03344       53759753    9
nprobe=256,quantizer_efSearch=512        0.2556 0.7024 0.9427      0.10861      105712732    3
nprobe=1024,quantizer_efSearch=64        0.2561 0.7031 0.9509      0.26029      387514370    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 269914384

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.0851 0.2377 0.3269      0.00131        2114794    230
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4   0.0526 0.1208 0.1337      0.00100        1143252    299
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1   0.0542 0.1210 0.1376      0.00095         616111    315
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0665 0.1521 0.1707      0.00103        1140794    292
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.0776 0.2004 0.2430      0.00110        1582415    273
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.0828 0.2195 0.2654      0.00115        2261887    261
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.0847 0.2401 0.3213      0.00128        3147990    236
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2   0.0851 0.2377 0.3269      0.00129        2114794    232
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.0874 0.2546 0.3470      0.00128        2463872    234
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.0892 0.2622 0.3579      0.00129        2459777    233
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0897 0.2631 0.3610      0.00141        2457693    213
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4   0.0929 0.2768 0.4117      0.00152        4251005    197
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.0968 0.2928 0.4354      0.00166        4919608    181
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.0994 0.3206 0.4797      0.00169        4887929    178
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=4 0.1015 0.3323 0.5413      0.00611        7697703    50
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 269935632

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.2511 0.6553 0.8255      0.01807       30102246    17
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2    0.1241 0.2189 0.2285      0.00115        1235137    262
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1396 0.2489 0.2588      0.00117        2258638    257
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1425 0.2600 0.2710      0.00134        2252551    225
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2    0.1448 0.2754 0.2919      0.00138        2130023    218
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1656 0.3204 0.3384      0.00144        3146543    209
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1688 0.3418 0.3666      0.00148        2456210    204
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1770 0.3611 0.3868      0.00162        3129374    186
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1777 0.3623 0.3886      0.00179        3124702    168
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8   0.1778 0.3628 0.3892      0.00202        3125725    149
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1803 0.3664 0.3931      0.00199        4452978    152
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1984 0.4318 0.4792      0.00211        4201273    142
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2077 0.4502 0.4982      0.00214        4878414    141
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.2118 0.4615 0.5120      0.00259        4863375    116
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2122 0.4583 0.5071      0.00256        6197564    118
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2156 0.4693 0.5197      0.00267        6186801    113
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2159 0.4695 0.5211      0.00301        6184005    100
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.2220 0.5095 0.5821      0.00349        9724535    87
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.2303 0.5449 0.6271      0.00462        8344310    65
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.2355 0.5576 0.6439      0.00466        9648069    65
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2365 0.5623 0.6481      0.00495       12269878    61
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.2395 0.5966 0.7168      0.00523       15257616    58
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.2406 0.5993 0.7224      0.00631       15234944    48
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2441 0.6165 0.7423      0.00703       16522657    44
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2464 0.6227 0.7513      0.00635       19127406    48
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2478 0.6287 0.7610      0.00687       19107123    44
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2480 0.6306 0.7631      0.00974       34494357    31
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2511 0.6553 0.8255      0.01617       30102246    19
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2545 0.6656 0.8434      0.01881       37810100    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.2547 0.6658 0.8434      0.01896       32599692    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.2552 0.6678 0.8476      0.02053       37740330    15
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32 0.2563 0.6794 0.8790      0.03148       60156946    11
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64 0.2570 0.6827 0.8835      0.03174       65214521    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32 0.2573 0.6903 0.9016      0.03589       59180613    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.2604 0.7036 0.9414      0.06858      116536655    5
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs` </summary>
Index size 203788492

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0945 0.3028 0.4723      0.00129        6910918    233
nprobe=1,quantizer_efSearch=4            0.0659 0.1753 0.2234      0.00066        1729584    457
nprobe=2,quantizer_efSearch=4            0.0823 0.2341 0.3299      0.00078        3467841    383
nprobe=4,quantizer_efSearch=4            0.0878 0.2825 0.4379      0.00099        6921159    304
nprobe=4,quantizer_efSearch=8            0.0945 0.3028 0.4723      0.00118        6910918    254
nprobe=8,quantizer_efSearch=4            0.0994 0.3306 0.5656      0.00137       13802022    219
nprobe=16,quantizer_efSearch=8           0.1036 0.3613 0.6584      0.00203       27365508    148
nprobe=16,quantizer_efSearch=32          0.1065 0.3676 0.6730      0.00280       27299348    108
nprobe=16,quantizer_efSearch=128         0.1070 0.3694 0.6752      0.00673       27283195    45
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fsr` </summary>
Index size 203807948

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2303 0.5994 0.7547      0.00478       27259915    63
nprobe=2,quantizer_efSearch=4            0.1463 0.3077 0.3378      0.00090        3458839    332
nprobe=2,quantizer_efSearch=8            0.1531 0.3261 0.3574      0.00112        3450374    268
nprobe=4,quantizer_efSearch=4            0.1739 0.3990 0.4563      0.00120        6920690    250
nprobe=8,quantizer_efSearch=4            0.2029 0.5050 0.6045      0.00180       13780111    167
nprobe=8,quantizer_efSearch=8            0.2077 0.5177 0.6206      0.00190       13771750    158
nprobe=8,quantizer_efSearch=16           0.2112 0.5275 0.6334      0.00231       13753397    130
nprobe=16,quantizer_efSearch=4           0.2145 0.5561 0.6940      0.00289       27373612    104
nprobe=16,quantizer_efSearch=8           0.2244 0.5842 0.7336      0.00305       27336514    99
nprobe=16,quantizer_efSearch=16          0.2284 0.5929 0.7463      0.00336       27297794    90
nprobe=16,quantizer_efSearch=32          0.2303 0.5994 0.7547      0.00407       27259915    74
nprobe=32,quantizer_efSearch=16          0.2325 0.6332 0.8325      0.00526       54096328    58
nprobe=32,quantizer_efSearch=32          0.2347 0.6398 0.8420      0.00580       54020377    52
nprobe=32,quantizer_efSearch=128         0.2348 0.6409 0.8439      0.00996       53978602    31
nprobe=64,quantizer_efSearch=16          0.2356 0.6564 0.8856      0.01640      106865622    19
nprobe=64,quantizer_efSearch=32          0.2385 0.6654 0.9021      0.01756      106657343    18
nprobe=64,quantizer_efSearch=64          0.2388 0.6667 0.9062      0.01847      106555932    17
nprobe=64,quantizer_efSearch=128         0.2391 0.6676 0.9072      0.02108      106525416    15
nprobe=64,quantizer_efSearch=256         0.2392 0.6677 0.9072      0.02727      106516136    12
nprobe=128,quantizer_efSearch=256        0.2398 0.6797 0.9393      0.04594      209391255    7
nprobe=256,quantizer_efSearch=128        0.2399 0.6812 0.9547      0.07195      410738664    5
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 187681296

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.0919 0.2881 0.4447      0.00147        7311105    205
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0698 0.1867 0.2382      0.00116        2413032    259
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8 0.0705 0.1869 0.2382      0.00117        2411792    258
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2  0.0710 0.1970 0.2758      0.00123        3691125    244
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4  0.0755 0.2116 0.2953      0.00125        3851485    241
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0807 0.2301 0.3274      0.00124        3830692    243
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4  0.0832 0.2368 0.3366      0.00126        3821463    239
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4  0.0839 0.2391 0.3406      0.00130        3817202    231
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8  0.0860 0.2456 0.3479      0.00125        4143036    240
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8 0.0866 0.2470 0.3510      0.00136        4139471    221
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0919 0.2881 0.4447      0.00142        7311218    212
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.0957 0.3001 0.4636      0.00142        7623792    212
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0959 0.3045 0.4740      0.00149        7596528    202
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0972 0.3069 0.4774      0.00168        8240691    179
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.0997 0.3369 0.5744      0.00173       14514166    174
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8 0.1013 0.3542 0.6477      0.00203       28149656    148
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.1029 0.3478 0.5938      0.00269       16387576    112
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 187689744

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1575 0.3313 0.3626      0.00244        4787562    123
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4    0.1136 0.2180 0.2306      0.00119        2078875    252
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1402 0.2918 0.3210      0.00125        3657931    241
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1486 0.3106 0.3401      0.00126        3823168    239
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1524 0.3187 0.3487      0.00127        4144213    236
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1567 0.3276 0.3592      0.00130        4140741    232
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1580 0.3304 0.3626      0.00143        4791696    211
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1753 0.3931 0.4447      0.00153        7662058    197
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1839 0.4309 0.4915      0.00160        7598566    188
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1855 0.4323 0.4920      0.00166        8245510    181
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1924 0.4629 0.5473      0.00204       14367078    148
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2017 0.4895 0.5831      0.00211       14232367    142
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2095 0.5186 0.6175      0.00213       14505413    142
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2101 0.5225 0.6239      0.00210       14481477    143
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2117 0.5289 0.6282      0.00218       15125095    138
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2127 0.5327 0.6348      0.00232       15100262    130
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2128 0.5329 0.6357      0.00258       15099406    117
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2132 0.5345 0.6374      0.00287       16382664    105
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2259 0.5865 0.7295      0.00315       28076453    96
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2295 0.6001 0.7496      0.00349       28655572    87
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2298 0.6004 0.7496      0.00383       28651433    79
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2300 0.6047 0.7557      0.00420       29909847    72
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2302 0.6047 0.7552      0.00457       29907050    66
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.2333 0.6230 0.8073      0.00509       56268143    59
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2376 0.6458 0.8421      0.00575       56744295    53
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.2378 0.6470 0.8447      0.00802       56675994    38
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.2408 0.6615 0.8845      0.01681      110910567    18
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2414 0.6625 0.8937      0.01772      108265247    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2423 0.6705 0.9043      0.01947      109389742    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2426 0.6715 0.9075      0.01902      111781617    16
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.2432 0.6822 0.9531      0.12635      820152972    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.2433 0.6842 0.9574      0.12812      820550632    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.2436 0.6842 0.9578      0.13034      825194027    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2437 0.6851 0.9586      0.14330      813868828    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2441 0.6829 0.9541      0.12886      809132126    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.2452 0.6869 0.9599      0.23526     1577908679    2
```

</details>
<details><summary>`OPQ16_64,IVF65536,PQ16x4fs` </summary>
Index size 185973271

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0964 0.3087 0.4808      0.00847        6878533    36
nprobe=2                                 0.0867 0.2498 0.3537      0.00804        3442200    38
nprobe=4                                 0.0964 0.3087 0.4808      0.00828        6878533    37
nprobe=8                                 0.1030 0.3484 0.5966      0.00885       13730100    34
```

</details>
<details><summary>`OPQ16_64,IVF65536,PQ16x4fsr` </summary>
Index size 185975831

 code_size 8

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_PQ16x4fsr.h.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.1890 0.4372 0.5007      0.00842        6882058    36
nprobe=2                                 max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.1571 0.3310 0.3633      0.00831        3442453    37
nprobe=4                                 max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.1890 0.4372 0.5007      0.00839        6882058    36
nprobe=8                                 max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.2146 0.5322 0.6394      0.00906       13732102    34
nprobe=16                                max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.2317 0.6033 0.7571      0.01030       27242792    30
nprobe=32                                max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.2361 0.6421 0.8456      0.01178       53966129    26
nprobe=64                                max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
max threads 32
0.2414 0.6666 0.9085      0.02248      106489555    14
```

</details>
<details><summary>`OPQ8_64,IVF16384_HNSW32,PQ8` </summary>
Index size 168884400

 code_size 8

 log filename: autotune.dbbigann10M.OPQ8_64_IVF16384_HNSW32_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1263 0.3269 0.4030      0.00311       26469871    97
nprobe=1,quantizer_efSearch=4,ht=28      0.1007 0.2149 0.2428      0.00200       13863784    150
nprobe=1,quantizer_efSearch=4,ht=64      0.1026 0.2277 0.2646      0.00208       13863784    144
nprobe=1,quantizer_efSearch=16,ht=28     0.1066 0.2307 0.2613      0.00250       13501273    120
nprobe=2,quantizer_efSearch=8,ht=26      0.1238 0.2977 0.3461      0.00245       26469871    123
nprobe=2,quantizer_efSearch=8,ht=30      0.1263 0.3269 0.4030      0.00309       26469871    97
nprobe=4,quantizer_efSearch=8,ht=22      0.1271 0.2809 0.3137      0.00298       50774696    101
nprobe=2,quantizer_efSearch=16,ht=64     0.1281 0.3333 0.4168      0.00301       26262130    100
nprobe=4,quantizer_efSearch=16,ht=22     0.1282 0.2840 0.3173      0.00341       50280238    88
nprobe=4,quantizer_efSearch=4,ht=64      0.1338 0.3688 0.4982      0.00319       51436059    95
nprobe=4,quantizer_efSearch=32,ht=64     0.1466 0.4121 0.5604      0.00439       50092223    69
nprobe=8,quantizer_efSearch=8,ht=28      0.1500 0.4379 0.6281      0.00684       97570365    44
nprobe=8,quantizer_efSearch=32,ht=30     0.1525 0.4538 0.6697      0.00835       96418495    36
nprobe=8,quantizer_efSearch=64,ht=32     0.1528 0.4568 0.6768      0.00947       96280965    32
nprobe=16,quantizer_efSearch=8,ht=28     0.1537 0.4622 0.7147      0.01234      186057079    25
nprobe=16,quantizer_efSearch=16,ht=32    0.1542 0.4777 0.7550      0.01382      185385760    22
nprobe=16,quantizer_efSearch=32,ht=30    0.1549 0.4785 0.7552      0.01441      184312287    21
nprobe=16,quantizer_efSearch=64,ht=32    0.1553 0.4813 0.7607      0.01571      183926565    20
nprobe=64,quantizer_efSearch=32,ht=64    0.1563 0.4910 0.8230      0.02507      667440398    12
```

</details>
<details><summary>`OPQ8_64,IVF262144_HNSW32,PQ8` </summary>
Index size 300647600

 code_size 8

 log filename: autotune.dbbigann10M.OPQ8_64_IVF262144_HNSW32_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2282 0.5130 0.5707      0.00884        6966900    34
nprobe=1,quantizer_efSearch=4,ht=28      0.0905 0.1300 0.1323      0.00242         436240    124
nprobe=4,quantizer_efSearch=4,ht=64      0.1702 0.3304 0.3478      0.00246        1750788    122
nprobe=4,quantizer_efSearch=8,ht=30      0.1812 0.3346 0.3483      0.00320        1746078    94
nprobe=8,quantizer_efSearch=4,ht=30      0.2071 0.4206 0.4489      0.00491        3491755    62
nprobe=8,quantizer_efSearch=32,ht=30     0.2185 0.4445 0.4746      0.00742        3469566    41
nprobe=16,quantizer_efSearch=4,ht=32     0.2282 0.5130 0.5707      0.00885        6966900    34
nprobe=16,quantizer_efSearch=8,ht=28     0.2308 0.4733 0.5075      0.00947        6954656    32
nprobe=16,quantizer_efSearch=16,ht=32    0.2437 0.5506 0.6135      0.00968        6940670    31
nprobe=16,quantizer_efSearch=64,ht=32    0.2453 0.5566 0.6207      0.01392        6925099    22
nprobe=64,quantizer_efSearch=64,ht=64    0.2698 0.6870 0.8426      0.01542       27243563    20
nprobe=64,quantizer_efSearch=128,ht=64   0.2699 0.6867 0.8426      0.01980       27225421    16
nprobe=64,quantizer_efSearch=256,ht=64   0.2702 0.6872 0.8430      0.03057       27219733    10
nprobe=512,quantizer_efSearch=32,ht=64   0.2723 0.7203 0.9258      0.06959      203445230    5
nprobe=4096,quantizer_efSearch=64,ht=64  0.2743 0.7312 0.9544      0.18432      519961113    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2755 0.7354 0.9698      0.16572      407627460    2
```

</details>
<details><summary>`OPQ8_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ8` </summary>
Index size 235869172

 code_size 8

 log filename: autotune.dbbigann10M.OPQ8_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                              0.2110 0.4583 0.5124      0.03377       28540195    9
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=20     0.0301 0.0394 0.0408      0.00243        1822436    124
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=28     0.0991 0.1381 0.1403      0.00249        1822558    121
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=28     0.1181 0.1780 0.1821      0.00253        1234032    119
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=30     0.1191 0.1834 0.1882      0.00254        1058865    119
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=32     0.1386 0.2257 0.2313      0.00263        1231371    114
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=64    0.1411 0.2338 0.2396      0.00264        1231051    114
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=30    0.1431 0.2281 0.2334      0.00281        1575487    107
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=32     0.1663 0.2959 0.3069      0.00287        3146096    105
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=32     0.1706 0.3064 0.3204      0.00295        2111479    102
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64    0.2063 0.4147 0.4441      0.00304        6233274    99
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=64     0.2279 0.4725 0.5131      0.00311        4874530    97
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=64    0.2368 0.5082 0.5624      0.00354        8438343    85
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64    0.2416 0.5609 0.6498      0.00711       14616109    43
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=30  0.2478 0.5382 0.5862      0.01154       12293260    27
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=30  0.2487 0.5373 0.5854      0.01315       27744468    23
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=64 0.2718 0.6487 0.7632      0.01774       34484330    17
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64  0.2781 0.7014 0.8825      0.02342       56766463    13
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=64 0.2851 0.7340 0.9499      0.08645      116322616    4
```

</details>
<details><summary>`OPQ8_64,IVF65536_HNSW32,PQ8` </summary>
Index size 195237552

 code_size 8

 log filename: autotune.dbbigann10M.OPQ8_64_IVF65536_HNSW32_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2402 0.5751 0.6835      0.00862       27399471    35
nprobe=2,quantizer_efSearch=4,ht=30      0.1572 0.2922 0.3126      0.00209        3465040    144
nprobe=4,quantizer_efSearch=4,ht=64      0.1946 0.4102 0.4562      0.00232        6927278    130
nprobe=4,quantizer_efSearch=8,ht=30      0.2036 0.4163 0.4565      0.00294        6908360    102
nprobe=4,quantizer_efSearch=32,ht=64     0.2091 0.4471 0.4977      0.00362        6888341    83
nprobe=8,quantizer_efSearch=8,ht=28      0.2213 0.4760 0.5269      0.00454       13782640    67
nprobe=8,quantizer_efSearch=4,ht=30      0.2253 0.5015 0.5691      0.00461       13799868    66
nprobe=8,quantizer_efSearch=32,ht=30     0.2333 0.5255 0.5969      0.00592       13744536    51
nprobe=8,quantizer_efSearch=64,ht=32     0.2350 0.5416 0.6238      0.00719       13740475    42
nprobe=16,quantizer_efSearch=4,ht=32     0.2402 0.5751 0.6835      0.00837       27399471    36
nprobe=16,quantizer_efSearch=8,ht=28     0.2404 0.5488 0.6247      0.00842       27354568    36
nprobe=16,quantizer_efSearch=16,ht=30    0.2493 0.5944 0.7015      0.00865       27318631    35
nprobe=16,quantizer_efSearch=16,ht=32    0.2507 0.6105 0.7312      0.00879       27318631    35
nprobe=16,quantizer_efSearch=32,ht=30    0.2508 0.5990 0.7063      0.00941       27283651    32
nprobe=16,quantizer_efSearch=64,ht=30    0.2509 0.5996 0.7071      0.01067       27271450    29
nprobe=16,quantizer_efSearch=64,ht=32    0.2522 0.6149 0.7372      0.01081       27271450    28
nprobe=32,quantizer_efSearch=128,ht=64   0.2634 0.6726 0.8474      0.01142       54013992    27
nprobe=64,quantizer_efSearch=32,ht=64    0.2665 0.6965 0.9057      0.01243      106748424    25
nprobe=64,quantizer_efSearch=64,ht=64    0.2674 0.6985 0.9088      0.01347      106639223    23
nprobe=64,quantizer_efSearch=128,ht=64   0.2675 0.6995 0.9097      0.01618      106601841    19
nprobe=64,quantizer_efSearch=256,ht=64   0.2676 0.6995 0.9098      0.02286      106594759    14
nprobe=256,quantizer_efSearch=64,ht=32   0.2677 0.7104 0.9417      0.12527      411562302    3
nprobe=256,quantizer_efSearch=512,ht=32  0.2679 0.7116 0.9453      0.15303      410656009    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2687 0.7168 0.9689      0.18132     1572442399    2
nprobe=4096,quantizer_efSearch=512,ht=64 0.2688 0.7169 0.9696      0.70323     5797608834    1
```

</details>
<details><summary>`OPQ8_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ8` </summary>
Index size 179110900

 code_size 8

 log filename: autotune.dbbigann10M.OPQ8_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                              0.0016 0.0027 0.0029      0.62680     1704362746    1
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=14    0.0087 0.0140 0.0150      0.00246        1832960    122
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8,ht=32    0.1210 0.2221 0.2306      0.00250        2412141    120
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=30     0.1424 0.2707 0.2850      0.00261        4169694    115
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=64     0.1475 0.2942 0.3176      0.00258        3666383    117
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=28    0.1526 0.2828 0.2980      0.00262        3818285    115
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=64    0.1639 0.3339 0.3597      0.00268        4791884    112
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=64     0.2030 0.4642 0.5296      0.00300       14373229    101
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=32     0.2238 0.5239 0.6005      0.00489       14521423    62
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8,ht=30    0.2239 0.5174 0.5846      0.00522       14469637    58
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=64   0.2246 0.5388 0.6254      0.00523       24109133    58
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=30   0.2251 0.5277 0.5965      0.00579       15099661    53
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=64   0.2431 0.6106 0.7453      0.00630       32556281    48
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=64  0.2519 0.6690 0.8484      0.01116       59187583    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=64  0.2558 0.6951 0.9109      0.01676      116869374    18
nprobe=256,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=64 0.2572 0.7095 0.9605      0.08353      413956197    4
```

</details>
<details><summary>`OPQ8_64,IVF65536,PQ8` </summary>
Index size 177400059

 code_size 8

 log filename: autotune.dbbigann10M.OPQ8_64_IVF65536_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0089 0.0149 0.0168      1.09843     3070480155    1
nprobe=1,ht=64                           0.1267 0.2298 0.2394      0.00915        1722158    33
nprobe=2,ht=64                           0.1699 0.3375 0.3606      0.00944        3444024    32
nprobe=4,ht=64                           0.2052 0.4506 0.4976      0.00969        6881313    31
nprobe=16,ht=64                          0.2490 0.6276 0.7561      0.01284       27240532    24
nprobe=32,ht=64                          0.2557 0.6783 0.8496      0.01407       53951127    22
nprobe=128,ht=64                         0.2605 0.7138 0.9498      0.03029      209227729    10
nprobe=1024,ht=64                        0.2606 0.7178 0.9741      0.16898     1569726785    2
```

</details>
<details><summary>`PCAR16,IVF16384_HNSW32,SQ4` </summary>
Index size 165715349

 code_size 8

 log filename: autotune.dbbigann10M.PCAR16_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0721 0.2603 0.4533      0.00299       33323842    101
nprobe=2,quantizer_efSearch=4            0.0666 0.2080 0.3231      0.00212       17080891    142
nprobe=2,quantizer_efSearch=8            0.0671 0.2180 0.3368      0.00221       16985442    136
nprobe=4,quantizer_efSearch=4            0.0709 0.2481 0.4283      0.00301       33443353    100
nprobe=4,quantizer_efSearch=8            0.0721 0.2603 0.4533      0.00302       33323842    100
nprobe=4,quantizer_efSearch=32           0.0725 0.2624 0.4581      0.00388       33214580    78
nprobe=8,quantizer_efSearch=4            0.0770 0.2921 0.5531      0.00653       65056674    46
nprobe=16,quantizer_efSearch=4           0.0789 0.3068 0.6165      0.00728      127046718    42
nprobe=16,quantizer_efSearch=16          0.0792 0.3154 0.6436      0.00746      126983273    41
nprobe=16,quantizer_efSearch=32          0.0793 0.3158 0.6448      0.00775      126873832    39
nprobe=32,quantizer_efSearch=32          0.0804 0.3256 0.6832      0.01303      248281991    24
nprobe=64,quantizer_efSearch=32          0.0810 0.3272 0.7016      0.02260      485004267    14
```

</details>
<details><summary>`PCAR16,IVF262144_HNSW32,SQ4` </summary>
Index size 250292629

 code_size 8

 log filename: autotune.dbbigann10M.PCAR16_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0661 0.2064 0.2813      0.00204        1725361    148
nprobe=2,quantizer_efSearch=4            0.0548 0.1489 0.1801      0.00180         866812    168
nprobe=4,quantizer_efSearch=4            0.0630 0.1958 0.2650      0.00186        1723953    161
nprobe=4,quantizer_efSearch=8            0.0661 0.2064 0.2813      0.00207        1725361    146
nprobe=8,quantizer_efSearch=4            0.0696 0.2434 0.3778      0.00208        3426843    144
nprobe=8,quantizer_efSearch=8            0.0705 0.2456 0.3840      0.00217        3426573    139
nprobe=16,quantizer_efSearch=4           0.0719 0.2715 0.4642      0.00230        6790935    131
nprobe=16,quantizer_efSearch=8           0.0742 0.2788 0.4832      0.00249        6795209    121
nprobe=16,quantizer_efSearch=16          0.0746 0.2829 0.4890      0.00273        6793606    110
nprobe=16,quantizer_efSearch=32          0.0749 0.2842 0.4916      0.00346        6789373    87
nprobe=32,quantizer_efSearch=16          0.0761 0.3021 0.5753      0.00374       13424193    81
nprobe=32,quantizer_efSearch=64          0.0767 0.3044 0.5808      0.00562       13407884    54
nprobe=32,quantizer_efSearch=128         0.0768 0.3043 0.5810      0.00832       13406671    37
nprobe=256,quantizer_efSearch=32         0.0769 0.3153 0.6820      0.01755      102719547    18
```

</details>
<details><summary>`PCAR16,IVF262144(IVF512,PQ8x4fs,RFlat),SQ4` </summary>
Index size 182167257

 code_size 8

 log filename: autotune.dbbigann10M.PCAR16_IVF262144_IVF512_PQ8x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.0697 0.2501 0.4066      0.00240        8351881    125
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.0491 0.1236 0.1454      0.00215        1063923    140
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.0611 0.1865 0.2454      0.00228        2116475    132
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.0642 0.1984 0.2643      0.00226        2458403    133
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.0658 0.2037 0.2706      0.00227        2439656    133
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.0697 0.2501 0.4066      0.00237        8351881    127
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.0700 0.2448 0.3689      0.00247        4156982    122
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.0705 0.2504 0.4068      0.00260        9659169    116
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0727 0.2674 0.4563      0.00266        7586433    113
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.0748 0.2789 0.4796      0.00280        8260582    108
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.0766 0.2942 0.5478      0.00383       15062974    79
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.0767 0.3026 0.6003      0.00603       29872527    50
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.0773 0.3008 0.5707      0.00502       18871596    60
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.0774 0.3110 0.6233      0.00535       29625603    57
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.0777 0.3123 0.6247      0.00704       32217791    43
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.0779 0.3124 0.6242      0.00750       37409711    41
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.0783 0.3163 0.6672      0.00939       55565085    32
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.0784 0.3173 0.6751      0.01235       57766033    25
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.0785 0.3183 0.6759      0.01887       62821117    16
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.0786 0.3138 0.6347      0.01680       47256443    18
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.0787 0.3190 0.6926      0.01795      108799016    17
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.0788 0.3202 0.6992      0.03668      212787545    9
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.0789 0.3202 0.6977      0.03356      225199485    9
```

</details>
<details><summary>`PCAR16,IVF65536_HNSW32,SQ4` </summary>
Index size 182631317

 code_size 8

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0754 0.3174 0.6974      0.05029      399529815    6
nprobe=2,quantizer_efSearch=4            0.0557 0.1862 0.2588      0.00174        3450570    173
nprobe=4,quantizer_efSearch=4            0.0630 0.2269 0.3589      0.00174        6855465    173
nprobe=4,quantizer_efSearch=8            0.0651 0.2363 0.3780      0.00197        6867601    153
nprobe=8,quantizer_efSearch=8            0.0700 0.2692 0.4823      0.00206       13609780    146
nprobe=8,quantizer_efSearch=4            0.0704 0.2652 0.4734      0.00202       13602452    149
nprobe=16,quantizer_efSearch=4           0.0729 0.2868 0.5512      0.00297       26884229    101
nprobe=16,quantizer_efSearch=8           0.0731 0.2943 0.5713      0.00294       26898323    103
nprobe=16,quantizer_efSearch=16          0.0733 0.2967 0.5775      0.00353       26889397    86
nprobe=16,quantizer_efSearch=32          0.0737 0.2972 0.5788      0.00356       26877755    85
nprobe=16,quantizer_efSearch=64          0.0740 0.2974 0.5793      0.00439       26877009    69
nprobe=32,quantizer_efSearch=64          0.0741 0.3105 0.6427      0.00627       52976699    48
nprobe=64,quantizer_efSearch=8           0.0742 0.3095 0.6507      0.00737      104284027    41
nprobe=64,quantizer_efSearch=16          0.0746 0.3148 0.6712      0.00715      104262703    42
nprobe=64,quantizer_efSearch=32          0.0751 0.3157 0.6761      0.00746      104156868    41
nprobe=64,quantizer_efSearch=64          0.0754 0.3160 0.6782      0.00815      104131886    37
nprobe=512,quantizer_efSearch=128        0.0755 0.3177 0.6982      0.05702      782342835    6
nprobe=2048,quantizer_efSearch=512       0.0756 0.3179 0.6995      0.24020     2989191830    2
```

</details>
<details><summary>`PCAR16,IVF65536(IVF256,PQ8x4fs,RFlat),SQ4` </summary>
Index size 165617369

 code_size 8

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_IVF256_PQ8x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0759 0.2896 0.5731      0.00728      108412628    42
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.0458 0.1366 0.1718      0.00240        3081821    125
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4   0.0470 0.1372 0.1729      0.00223        2090060    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.0573 0.1851 0.2625      0.00227        4816039    133
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0661 0.2303 0.3671      0.00240        7251115    126
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.0676 0.2336 0.3753      0.00246        7603630    122
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16  0.0678 0.2363 0.3807      0.00278        8210238    109
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.0687 0.2420 0.4169      0.00312       14444003    97
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.0736 0.2641 0.4706      0.00336       16350186    90
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.0748 0.2723 0.4875      0.00353       16300707    85
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0761 0.2831 0.5380      0.00468       27409314    65
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0791 0.2966 0.5778      0.00501       28336372    60
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.0796 0.2972 0.5782      0.00636       37253192    48
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.0804 0.3116 0.6447      0.00838       55777702    36
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.0808 0.3158 0.6802      0.01160      109869377    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.0809 0.3114 0.6633      0.01175      115678627    26
```

</details>
<details><summary>`PCAR16,IVF65536,SQ4` </summary>
Index size 164793824

 code_size 8

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0759 0.2951 0.5821      0.00798       26865185    38
nprobe=8                                 0.0711 0.2721 0.4922      0.00638       13600536    47
nprobe=16                                0.0759 0.2951 0.5821      0.00808       26865185    38
nprobe=32                                0.0782 0.3093 0.6454      0.01137       52948423    27
nprobe=64                                0.0794 0.3142 0.6812      0.01641      104069128    19
```

</details>

## Code sizes in [9, 16]

<details><summary>`OPQ16_64,IVF16384_HNSW32,PQ16` </summary>
Index size 248884400

 code_size 16

 log filename: autotune.dbbigann10M.OPQ16_64_IVF16384_HNSW32_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3296 0.7778 0.8990      0.03495      353095604    9
nprobe=2,quantizer_efSearch=4,ht=46      0.1585 0.2292 0.2312      0.00242       27127591    124
nprobe=1,quantizer_efSearch=4,ht=60      0.1635 0.2580 0.2640      0.00257       14053190    117
nprobe=1,quantizer_efSearch=8,ht=58      0.1714 0.2716 0.2774      0.00259       13743826    117
nprobe=1,quantizer_efSearch=16,ht=128    0.1733 0.2779 0.2846      0.00281       13599270    107
nprobe=2,quantizer_efSearch=4,ht=128     0.2136 0.3742 0.3872      0.00315       27127591    96
nprobe=2,quantizer_efSearch=16,ht=52     0.2196 0.3642 0.3703      0.00338       26333534    89
nprobe=2,quantizer_efSearch=16,ht=54     0.2262 0.3842 0.3926      0.00357       26333534    85
nprobe=4,quantizer_efSearch=4,ht=50      0.2310 0.4103 0.4188      0.00434       52072749    70
nprobe=4,quantizer_efSearch=8,ht=50      0.2475 0.4498 0.4596      0.00440       51163900    69
nprobe=4,quantizer_efSearch=16,ht=50     0.2517 0.4571 0.4667      0.00484       50597876    62
nprobe=4,quantizer_efSearch=8,ht=52      0.2589 0.4865 0.5017      0.00465       51163900    65
nprobe=4,quantizer_efSearch=8,ht=64      0.2682 0.5349 0.5657      0.00630       51163900    48
nprobe=8,quantizer_efSearch=8,ht=128     0.2981 0.6391 0.6939      0.00700       98231074    43
nprobe=8,quantizer_efSearch=16,ht=56     0.3031 0.6412 0.6903      0.00945       97379194    32
nprobe=8,quantizer_efSearch=16,ht=60     0.3044 0.6515 0.7063      0.01073       97379194    28
nprobe=8,quantizer_efSearch=32,ht=64     0.3047 0.6546 0.7112      0.01189       96889183    26
nprobe=16,quantizer_efSearch=16,ht=60    0.3189 0.7276 0.8128      0.01960      186508066    16
nprobe=16,quantizer_efSearch=64,ht=64    0.3203 0.7359 0.8247      0.02195      184891705    14
nprobe=32,quantizer_efSearch=16,ht=128   0.3286 0.7762 0.8960      0.02200      355059901    14
nprobe=32,quantizer_efSearch=32,ht=60    0.3296 0.7778 0.8990      0.03539      353095604    9
nprobe=32,quantizer_efSearch=256,ht=64   0.3302 0.7828 0.9067      0.04570      351697303    7
nprobe=64,quantizer_efSearch=32,ht=56    0.3321 0.7887 0.9233      0.06056      669697368    5
nprobe=64,quantizer_efSearch=128,ht=60   0.3333 0.8009 0.9462      0.07102      666005197    5
nprobe=64,quantizer_efSearch=512,ht=128  0.3337 0.8043 0.9517      0.07004      665769073    5
nprobe=128,quantizer_efSearch=256,ht=128 0.3346 0.8104 0.9736      0.10518     1258217184    3
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16` </summary>
Index size 380647600

 code_size 16

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_HNSW32_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.4180 0.7194 0.7303      0.02591       13773647    12
nprobe=2,quantizer_efSearch=4,ht=128      0.1887 0.2502 0.2509      0.00232         873889    130
nprobe=2,quantizer_efSearch=16,ht=62      0.2007 0.2651 0.2658      0.00401         868414    75
nprobe=4,quantizer_efSearch=8,ht=64       0.2629 0.3776 0.3797      0.00424        1745591    71
nprobe=8,quantizer_efSearch=8,ht=128      0.3220 0.4991 0.5033      0.00381        3482073    79
nprobe=8,quantizer_efSearch=16,ht=60      0.3240 0.4906 0.4943      0.00813        3471131    37
nprobe=8,quantizer_efSearch=32,ht=64      0.3310 0.5116 0.5155      0.00941        3465233    32
nprobe=16,quantizer_efSearch=4,ht=62      0.3541 0.5770 0.5827      0.01248        6962470    25
nprobe=32,quantizer_efSearch=16,ht=128    0.4164 0.7351 0.7483      0.01483       13801217    21
nprobe=32,quantizer_efSearch=32,ht=60     0.4180 0.7194 0.7303      0.02547       13773647    12
nprobe=128,quantizer_efSearch=16,ht=128   0.4493 0.8382 0.8666      0.03970       54084377    8
nprobe=128,quantizer_efSearch=256,ht=128  0.4672 0.8813 0.9164      0.05561       53776527    6
nprobe=512,quantizer_efSearch=512,ht=128  0.4783 0.9337 0.9807      0.18687      207905417    2
nprobe=4096,quantizer_efSearch=512,ht=128 0.4807 0.9434 0.9967      1.15121     1520724991    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16` </summary>
Index size 315867636

 code_size 16

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.4383 0.8054 0.8215      0.05025       47909247    6
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=40       0.0294 0.0382 0.0382      0.00233        2262788    129
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=50       0.0959 0.1177 0.1179      0.00240        1061001    125
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=64       0.1028 0.1327 0.1328      0.00246        1143594    122
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=58       0.1224 0.1568 0.1569      0.00253        1065789    119
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=128     0.1375 0.1770 0.1770      0.00271        3151910    111
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=128     0.2278 0.3326 0.3343      0.00265        2112107    114
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64       0.2498 0.3647 0.3663      0.00391        3140916    77
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=62     0.2608 0.3831 0.3848      0.00676        4457592    45
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=54      0.2778 0.3985 0.4008      0.00790        4872302    38
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=128    0.3746 0.6261 0.6330      0.00831       17486815    37
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.4216 0.7503 0.7650      0.01454       24301932    21
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=128   0.4220 0.7523 0.7666      0.01890       54999876    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128    0.4426 0.8289 0.8509      0.02523       32590459    12
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128   0.4447 0.8327 0.8550      0.02819       37712002    11
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=62    0.4470 0.8490 0.8754      0.09619       56793874    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60    0.4505 0.8596 0.8861      0.17732      109564975    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=60  0.4650 0.8950 0.9251      0.26061      147091742    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128   0.4721 0.9357 0.9804      0.16793      218624607    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=62   0.4726 0.9325 0.9767      0.70179      418990850    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=128 0.4745 0.9475 0.9991      1.06130     1598891396    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16` </summary>
Index size 275237552

 code_size 16

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_HNSW32_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.4379 0.8070 0.8356      0.02189       54080527    14
nprobe=2,quantizer_efSearch=4,ht=46       0.1145 0.1483 0.1494      0.00212        3467613    142
nprobe=2,quantizer_efSearch=4,ht=128      0.2291 0.3372 0.3399      0.00234        3467613    129
nprobe=2,quantizer_efSearch=16,ht=62      0.2396 0.3528 0.3553      0.00301        3448040    100
nprobe=4,quantizer_efSearch=8,ht=52       0.2657 0.3758 0.3787      0.00346        6908997    87
nprobe=8,quantizer_efSearch=8,ht=128      0.3628 0.6142 0.6257      0.00364       13788996    83
nprobe=8,quantizer_efSearch=16,ht=60      0.3653 0.6126 0.6225      0.00664       13759452    46
nprobe=8,quantizer_efSearch=32,ht=64      0.3701 0.6274 0.6385      0.00779       13748209    39
nprobe=32,quantizer_efSearch=16,ht=128    0.4363 0.8120 0.8455      0.01515       54163165    20
nprobe=32,quantizer_efSearch=32,ht=60     0.4379 0.8070 0.8356      0.02367       54080527    13
nprobe=32,quantizer_efSearch=256,ht=64    0.4399 0.8208 0.8540      0.03346       54040682    9
nprobe=128,quantizer_efSearch=16,ht=128   0.4546 0.8838 0.9299      0.04589      210605349    7
nprobe=64,quantizer_efSearch=128,ht=60    0.4566 0.8687 0.9059      0.04545      106646353    7
nprobe=64,quantizer_efSearch=128,ht=62    0.4571 0.8759 0.9164      0.04660      106646353    7
nprobe=128,quantizer_efSearch=256,ht=128  0.4638 0.9146 0.9657      0.04852      209608649    7
nprobe=512,quantizer_efSearch=512,ht=128  0.4686 0.9356 0.9946      0.16665      804112201    2
nprobe=2048,quantizer_efSearch=512,ht=128 0.4691 0.9379 0.9986      0.60695     3075555001    1
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16` </summary>
Index size 259110388

 code_size 16

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.4059 0.7100 0.7239      0.01360       37600061    23
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8,ht=44       0.0903 0.1148 0.1155      0.00261        4140198    116
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8,ht=54       0.2200 0.3018 0.3036      0.00263        4140028    114
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58       0.2385 0.3406 0.3429      0.00287        4139949    105
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=128       0.3044 0.4747 0.4791      0.00288        7282499    105
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=128      0.3124 0.4952 0.5000      0.00350        9523470    86
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56        0.3277 0.5182 0.5238      0.00623       14641162    49
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=58       0.3612 0.5996 0.6087      0.00661       15107390    46
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60       0.3627 0.6058 0.6148      0.00662       15140067    46
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=60       0.3631 0.6040 0.6134      0.00724       14476849    42
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=128    0.3711 0.6314 0.6426      0.00732       24059008    41
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128    0.4131 0.7378 0.7573      0.00844       28662438    36
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=64     0.4143 0.7395 0.7587      0.01468       32477171    21
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56      0.4260 0.7642 0.7810      0.02221       56679158    14
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=62      0.4277 0.7861 0.8106      0.02895       55038183    11
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.4323 0.8029 0.8269      0.04670      108069489    7
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=60     0.4554 0.8682 0.9024      0.04984      109314294    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=64    0.4648 0.9133 0.9614      0.08870      212408186    4
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=64    0.4651 0.9148 0.9634      0.11720      214744369    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=62     0.4657 0.9218 0.9730      0.16017      414560721    2
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=128  0.4690 0.9355 0.9949      0.17260      814090478    2
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=128 0.4698 0.9383 0.9991      1.78192     6032533461    1
```

</details>
<details><summary>`OPQ16_64,IVF65536,PQ16` </summary>
Index size 257400059

 code_size 16

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4684 0.9373 0.9950      0.15317      803413105    2
nprobe=1,ht=60                           0.1706 0.2335 0.2338      0.00948        1721696    32
nprobe=2,ht=128                          0.2453 0.3597 0.3622      0.00932        3442200    33
nprobe=8,ht=128                          0.3732 0.6330 0.6440      0.01065       13730100    29
nprobe=16,ht=54                          0.3904 0.6406 0.6496      0.01761       27253414    18
nprobe=16,ht=56                          0.4004 0.6837 0.6957      0.01770       27253414    17
nprobe=16,ht=60                          0.4126 0.7301 0.7474      0.01809       27253414    17
nprobe=32,ht=128                         0.4390 0.8284 0.8592      0.01909       53978616    16
nprobe=64,ht=128                         0.4561 0.8860 0.9275      0.02728      106507441    11
nprobe=256,ht=128                        0.4678 0.9328 0.9872      0.08278      410440345    4
nprobe=512,ht=128                        0.4684 0.9373 0.9950      0.15653      803413105    2
nprobe=1024,ht=128                       0.4686 0.9392 0.9981      0.29907     1570963471    2
```

</details>
<details><summary>`OPQ32_64,IVF16384_HNSW32,PQ32x4fs` </summary>
Index size 253021900

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF16384_HNSW32_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2157 0.4758 0.5638      0.00191       51303704    158
nprobe=2,quantizer_efSearch=4            0.1651 0.3405 0.3873      0.00119       27137262    252
nprobe=2,quantizer_efSearch=8            0.1795 0.3666 0.4183      0.00137       26547823    220
nprobe=4,quantizer_efSearch=4            0.1959 0.4310 0.5085      0.00140       52068403    214
nprobe=4,quantizer_efSearch=8            0.2157 0.4758 0.5638      0.00168       51303704    179
nprobe=8,quantizer_efSearch=4            0.2322 0.5493 0.6718      0.00185       98458386    163
nprobe=8,quantizer_efSearch=8            0.2379 0.5642 0.6924      0.00193       98281921    156
nprobe=8,quantizer_efSearch=16           0.2427 0.5760 0.7066      0.00225       97283573    134
nprobe=16,quantizer_efSearch=8           0.2514 0.6313 0.7975      0.00271      187096359    111
nprobe=16,quantizer_efSearch=16          0.2547 0.6417 0.8110      0.00307      186068408    98
nprobe=32,quantizer_efSearch=8           0.2582 0.6672 0.8605      0.00365      355355375    83
nprobe=32,quantizer_efSearch=32          0.2644 0.6902 0.8958      0.00439      352434508    69
nprobe=64,quantizer_efSearch=16          0.2662 0.7032 0.9254      0.00569      670490557    53
nprobe=64,quantizer_efSearch=32          0.2685 0.7114 0.9396      0.00633      667540013    48
nprobe=64,quantizer_efSearch=64          0.2692 0.7143 0.9441      0.00709      665307781    43
nprobe=128,quantizer_efSearch=64         0.2714 0.7242 0.9664      0.01118     1258495274    27
nprobe=128,quantizer_efSearch=128        0.2716 0.7246 0.9672      0.01224     1255604779    25
```

</details>
<details><summary>`OPQ32_64,IVF16384_HNSW32,PQ32x4fsr` </summary>
Index size 253020364

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF16384_HNSW32_PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4024 0.7944 0.8410      0.00566      109596708    54
nprobe=1,quantizer_efSearch=8            0.2065 0.3038 0.3091      0.00110        7120010    273
nprobe=2,quantizer_efSearch=4            0.2600 0.4180 0.4278      0.00119       14200946    252
nprobe=2,quantizer_efSearch=8            0.2708 0.4354 0.4455      0.00134       14154930    224
nprobe=2,quantizer_efSearch=16           0.2727 0.4385 0.4486      0.00167       14142680    180
nprobe=4,quantizer_efSearch=4            0.3101 0.5402 0.5591      0.00168       28222063    179
nprobe=4,quantizer_efSearch=16           0.3285 0.5762 0.5957      0.00225       28102361    134
nprobe=8,quantizer_efSearch=4            0.3629 0.6735 0.7040      0.00277       55882843    109
nprobe=8,quantizer_efSearch=8            0.3686 0.6865 0.7185      0.00284       55819078    106
nprobe=8,quantizer_efSearch=16           0.3743 0.6979 0.7304      0.00375       55716976    80
nprobe=8,quantizer_efSearch=32           0.3751 0.6991 0.7317      0.00393       55671071    77
nprobe=16,quantizer_efSearch=4           0.3835 0.7506 0.7926      0.00448      109951927    69
nprobe=16,quantizer_efSearch=16          0.4008 0.7918 0.8374      0.00478      109683968    63
nprobe=16,quantizer_efSearch=32          0.4024 0.7944 0.8410      0.00527      109596708    57
nprobe=16,quantizer_efSearch=64          0.4028 0.7953 0.8419      0.00627      109584673    48
nprobe=32,quantizer_efSearch=8           0.4088 0.8299 0.8881      0.01647      216097899    19
nprobe=32,quantizer_efSearch=16          0.4172 0.8493 0.9113      0.01742      215685554    18
nprobe=32,quantizer_efSearch=32          0.4182 0.8543 0.9170      0.01808      215416853    17
nprobe=32,quantizer_efSearch=64          0.4184 0.8554 0.9182      0.01874      215333205    17
nprobe=64,quantizer_efSearch=16          0.4219 0.8783 0.9510      0.03096      422586436    10
nprobe=64,quantizer_efSearch=32          0.4228 0.8866 0.9622      0.03273      421757971    10
nprobe=64,quantizer_efSearch=64          0.4239 0.8885 0.9639      0.03289      421381109    10
nprobe=64,quantizer_efSearch=128         0.4243 0.8887 0.9644      0.03532      421295403    9
nprobe=128,quantizer_efSearch=16         0.4264 0.8884 0.9660      0.07215      826469734    5
nprobe=128,quantizer_efSearch=64         0.4297 0.9025 0.9849      0.07083      822815856    5
nprobe=256,quantizer_efSearch=128        0.4307 0.9071 0.9951      0.13520     1603289065    3
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fs` </summary>
Index size 446494412

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF262144_HNSW32_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1799 0.3522 0.3810      0.00170        1745076    177
nprobe=2,quantizer_efSearch=4            0.1338 0.2383 0.2497      0.00105         873424    286
nprobe=4,quantizer_efSearch=4            0.1623 0.3185 0.3464      0.00122        1749135    246
nprobe=4,quantizer_efSearch=8            0.1799 0.3522 0.3810      0.00167        1745076    181
nprobe=8,quantizer_efSearch=4            0.2032 0.4368 0.4897      0.00184        3485496    164
nprobe=8,quantizer_efSearch=8            0.2075 0.4449 0.4986      0.00195        3480984    155
nprobe=16,quantizer_efSearch=4           0.2171 0.5119 0.5910      0.00228        6960463    132
nprobe=16,quantizer_efSearch=8           0.2296 0.5393 0.6238      0.00267        6949100    113
nprobe=16,quantizer_efSearch=16          0.2352 0.5494 0.6345      0.00322        6938936    94
nprobe=32,quantizer_efSearch=8           0.2389 0.5968 0.7146      0.00330       13829767    91
nprobe=32,quantizer_efSearch=16          0.2491 0.6217 0.7445      0.00411       13800614    73
nprobe=64,quantizer_efSearch=16          0.2566 0.6575 0.8158      0.00578       27355707    52
nprobe=64,quantizer_efSearch=32          0.2588 0.6706 0.8353      0.00699       27292294    44
nprobe=128,quantizer_efSearch=16         0.2596 0.6718 0.8539      0.00679       54073260    45
nprobe=128,quantizer_efSearch=32         0.2632 0.6941 0.8871      0.00835       53986347    37
nprobe=128,quantizer_efSearch=64         0.2643 0.7009 0.8990      0.01147       53855064    27
nprobe=256,quantizer_efSearch=32         0.2657 0.7042 0.9133      0.01188      106412779    26
nprobe=256,quantizer_efSearch=64         0.2682 0.7137 0.9330      0.01632      106218240    19
nprobe=256,quantizer_efSearch=128        0.2690 0.7188 0.9395      0.02264      105982600    14
nprobe=512,quantizer_efSearch=128        0.2697 0.7235 0.9558      0.03113      208414195    10
nprobe=1024,quantizer_efSearch=128       0.2704 0.7248 0.9624      0.04596      406245452    7
nprobe=2048,quantizer_efSearch=256       0.2706 0.7286 0.9692      0.09140      788050009    4
nprobe=4096,quantizer_efSearch=256       0.2707 0.7294 0.9694      0.11883     1341339711    3
nprobe=2048,quantizer_efSearch=512       0.2709 0.7296 0.9708      0.14269      797333270    3
nprobe=4096,quantizer_efSearch=512       0.2710 0.7290 0.9710      0.21320     1520734494    2
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fsr` </summary>
Index size 446432972

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF262144_HNSW32_PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3687 0.6357 0.6482      0.00700        6918122    43
nprobe=2,quantizer_efSearch=8            0.1939 0.2660 0.2677      0.00178         868374    169
nprobe=4,quantizer_efSearch=4            0.2315 0.3435 0.3466      0.00175        1744002    172
nprobe=4,quantizer_efSearch=8            0.2559 0.3794 0.3829      0.00217        1742199    139
nprobe=8,quantizer_efSearch=4            0.3025 0.4871 0.4935      0.00273        3479156    110
nprobe=8,quantizer_efSearch=8            0.3088 0.4977 0.5041      0.00282        3474819    107
nprobe=8,quantizer_efSearch=16           0.3155 0.5112 0.5179      0.00373        3465257    81
nprobe=16,quantizer_efSearch=4           0.3434 0.5840 0.5965      0.00426        6947898    71
nprobe=16,quantizer_efSearch=8           0.3625 0.6176 0.6300      0.00463        6938387    65
nprobe=16,quantizer_efSearch=16          0.3664 0.6276 0.6402      0.00518        6926850    58
nprobe=16,quantizer_efSearch=32          0.3687 0.6357 0.6482      0.00666        6918122    46
nprobe=16,quantizer_efSearch=64          0.3703 0.6371 0.6496      0.00943        6913623    32
nprobe=16,quantizer_efSearch=128         0.3705 0.6375 0.6500      0.01492        6911415    21
nprobe=32,quantizer_efSearch=32          0.4038 0.7406 0.7631      0.01788       13750126    17
nprobe=32,quantizer_efSearch=64          0.4053 0.7433 0.7661      0.02057       13737113    15
nprobe=64,quantizer_efSearch=16          0.4187 0.7939 0.8268      0.02961       27308257    11
nprobe=64,quantizer_efSearch=32          0.4260 0.8141 0.8475      0.03121       27246645    10
nprobe=64,quantizer_efSearch=64          0.4277 0.8199 0.8540      0.03282       27209705    10
nprobe=64,quantizer_efSearch=128         0.4286 0.8211 0.8554      0.03659       27193390    9
nprobe=64,quantizer_efSearch=256         0.4287 0.8211 0.8553      0.04623       27189846    7
nprobe=128,quantizer_efSearch=32         0.4375 0.8590 0.9018      0.06620       53890052    5
nprobe=128,quantizer_efSearch=64         0.4409 0.8700 0.9150      0.06932       53773785    5
nprobe=128,quantizer_efSearch=128        0.4425 0.8724 0.9177      0.07417       53710352    5
nprobe=256,quantizer_efSearch=64         0.4494 0.9015 0.9528      0.13267      106052091    3
nprobe=256,quantizer_efSearch=128        0.4495 0.9071 0.9592      0.14342      105830005    3
nprobe=256,quantizer_efSearch=256        0.4500 0.9084 0.9606      0.14636      105747985    3
nprobe=512,quantizer_efSearch=64         0.4509 0.9104 0.9676      0.24867      208120290    2
nprobe=512,quantizer_efSearch=256        0.4540 0.9218 0.9804      0.27005      207774347    2
nprobe=512,quantizer_efSearch=512        0.4544 0.9223 0.9808      0.28289      207657908    2
nprobe=1024,quantizer_efSearch=128       0.4555 0.9244 0.9858      0.48974      405776940    1
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 381872656

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2548 0.6425 0.7875      0.00602       33177167    50
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.0852 0.1342 0.1372      0.00099         615995    303
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.0971 0.1661 0.1718      0.00108        1065862    279
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1111 0.1927 0.1991      0.00111        1239401    271
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1376 0.2465 0.2581      0.00116        1577040    259
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1461 0.2590 0.2700      0.00120        2258233    250
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1577 0.2946 0.3121      0.00132        2475843    229
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1666 0.3249 0.3489      0.00132        2462880    227
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1697 0.3330 0.3600      0.00133        2458649    227
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1733 0.3379 0.3654      0.00143        2456995    211
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1813 0.3782 0.4168      0.00156        4251734    193
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.1918 0.4012 0.4410      0.00164        4920723    184
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2064 0.4395 0.4896      0.00169        4887431    178
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2099 0.4848 0.5622      0.00201        7735047    150
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2137 0.4861 0.5546      0.00213        8451765    141
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2253 0.5276 0.6100      0.00227        8384648    132
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2319 0.5424 0.6246      0.00278        9692928    108
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2324 0.5591 0.6622      0.00283       15483288    106
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2371 0.5522 0.6373      0.00294        9671208    103
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2404 0.5899 0.7064      0.00418       15295189    72
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2435 0.6064 0.7422      0.00412       29427199    73
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.2526 0.6362 0.7772      0.00462       30634671    65
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2571 0.6596 0.8170      0.00488       30173801    62
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2583 0.6643 0.8251      0.00582       30080741    52
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2595 0.6697 0.8310      0.00641       32724858    47
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2615 0.6853 0.8733      0.00698       56984538    43
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2621 0.6745 0.8397      0.00708       32635885    43
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.2624 0.6891 0.8896      0.00953      112090105    32
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.2678 0.7043 0.9134      0.01030      114224928    30
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2700 0.7139 0.9308      0.01355      111859935    23
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2714 0.7182 0.9403      0.01880      126988742    16
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.2716 0.7264 0.9634      0.03540      439128929    9
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 381920784

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4239 0.8045 0.8353      0.02920       30092706    11
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1745 0.2343 0.2356      0.00131        1233034    229
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1927 0.2603 0.2617      0.00133        2260874    227
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1986 0.2697 0.2712      0.00147        2254910    205
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2092 0.2996 0.3018      0.00172        2128162    174
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2355 0.3411 0.3436      0.00175        3146849    172
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2429 0.3551 0.3582      0.00177        2461646    170
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2573 0.3821 0.3855      0.00188        3132806    160
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2579 0.3825 0.3860      0.00205        3132272    147
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2589 0.3810 0.3841      0.00220        4468878    137
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2607 0.3876 0.3910      0.00228        4464567    132
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2941 0.4663 0.4725      0.00255        4207791    118
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2981 0.4639 0.4691      0.00250        4912430    120
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3141 0.5002 0.5065      0.00268        4873711    113
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3199 0.5085 0.5148      0.00306        6198143    99
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3231 0.5149 0.5213      0.00324        6192727    93
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3361 0.5672 0.5774      0.00450        7695188    67
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3604 0.6145 0.6260      0.00469        8350039    64
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3688 0.6329 0.6448      0.00486        9661203    62
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3690 0.6337 0.6456      0.00532        9658726    57
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3706 0.6377 0.6494      0.00571       12277142    53
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3716 0.6393 0.6510      0.00685       17446632    44
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.3718 0.6394 0.6511      0.00854       27674938    36
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3883 0.7027 0.7227      0.01520       15254084    20
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3923 0.7006 0.7181      0.01531       16695947    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4001 0.7318 0.7520      0.01598       16540300    19
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4037 0.7407 0.7614      0.01639       19142105    19
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4052 0.7424 0.7631      0.01815       24258065    17
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4065 0.7467 0.7686      0.01816       24291339    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4239 0.8045 0.8353      0.02842       30069509    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4286 0.8199 0.8520      0.03125       32617223    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4322 0.8219 0.8543      0.03031       37797850    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4421 0.8757 0.9191      0.07370       74414706    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4536 0.9077 0.9600      0.12573      116547027    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.4549 0.9248 0.9899      0.47187      435177709    1
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fs` </summary>
Index size 291947724

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_HNSW32_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2667 0.7269 0.9661      0.04675      410898003    7
nprobe=2,quantizer_efSearch=4            0.1522 0.3035 0.3362      0.00084        3464277    358
nprobe=2,quantizer_efSearch=8            0.1598 0.3209 0.3545      0.00105        3451399    287
nprobe=4,quantizer_efSearch=4            0.1836 0.4017 0.4566      0.00103        6921917    292
nprobe=4,quantizer_efSearch=8            0.1953 0.4303 0.4883      0.00123        6905851    244
nprobe=8,quantizer_efSearch=4            0.2154 0.5147 0.6050      0.00144       13796693    209
nprobe=8,quantizer_efSearch=8            0.2201 0.5275 0.6194      0.00150       13786633    200
nprobe=16,quantizer_efSearch=4           0.2297 0.5730 0.6968      0.00193       27397270    156
nprobe=16,quantizer_efSearch=8           0.2398 0.5999 0.7321      0.00207       27367349    146
nprobe=16,quantizer_efSearch=16          0.2424 0.6073 0.7429      0.00236       27334111    127
nprobe=32,quantizer_efSearch=8           0.2476 0.6448 0.8082      0.00276       54260418    109
nprobe=32,quantizer_efSearch=16          0.2537 0.6631 0.8319      0.00334       54158721    90
nprobe=32,quantizer_efSearch=32          0.2547 0.6692 0.8412      0.00354       54080794    85
nprobe=64,quantizer_efSearch=16          0.2615 0.6905 0.8863      0.00408      106987260    74
nprobe=64,quantizer_efSearch=32          0.2632 0.7012 0.9022      0.00468      106781765    65
nprobe=64,quantizer_efSearch=64          0.2649 0.7048 0.9069      0.00560      106685737    54
nprobe=128,quantizer_efSearch=64         0.2655 0.7186 0.9452      0.00767      209787385    40
nprobe=64,quantizer_efSearch=128         0.2656 0.7059 0.9083      0.00889      106642809    34
nprobe=128,quantizer_efSearch=128        0.2662 0.7190 0.9464      0.00975      209621256    31
nprobe=128,quantizer_efSearch=256        0.2664 0.7192 0.9468      0.01836      209596312    17
nprobe=512,quantizer_efSearch=128        0.2668 0.7257 0.9695      0.02198      805524231    14
nprobe=512,quantizer_efSearch=256        0.2669 0.7264 0.9709      0.02929      804458704    11
nprobe=1024,quantizer_efSearch=128       0.2670 0.7253 0.9711      0.03216     1560800717    10
nprobe=512,quantizer_efSearch=512        0.2671 0.7266 0.9715      0.04658      804187946    7
nprobe=2048,quantizer_efSearch=128       0.2672 0.7251 0.9712      0.04465     2676940137    7
nprobe=1024,quantizer_efSearch=256       0.2681 0.7267 0.9731      0.04768     1573715539    7
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fsr` </summary>
Index size 291933900

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_HNSW32_PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3986 0.7297 0.7601      0.00606       27270215    50
nprobe=2,quantizer_efSearch=4            0.2225 0.3334 0.3376      0.00114        3463743    264
nprobe=2,quantizer_efSearch=8            0.2355 0.3524 0.3568      0.00133        3450155    226
nprobe=4,quantizer_efSearch=4            0.2772 0.4487 0.4586      0.00161        6913228    187
nprobe=4,quantizer_efSearch=16           0.2988 0.4877 0.4979      0.00220        6887441    137
nprobe=8,quantizer_efSearch=8            0.3484 0.6070 0.6265      0.00278       13773319    109
nprobe=8,quantizer_efSearch=16           0.3535 0.6171 0.6370      0.00316       13748314    96
nprobe=8,quantizer_efSearch=32           0.3557 0.6201 0.6400      0.00397       13733787    76
nprobe=16,quantizer_efSearch=4           0.3757 0.6781 0.7053      0.00437       27366264    69
nprobe=16,quantizer_efSearch=8           0.3889 0.7135 0.7427      0.00448       27334585    67
nprobe=16,quantizer_efSearch=32          0.3986 0.7297 0.7601      0.00545       27270215    56
nprobe=16,quantizer_efSearch=64          0.3994 0.7317 0.7622      0.00679       27260448    45
nprobe=32,quantizer_efSearch=8           0.4069 0.7799 0.8214      0.01559       54193972    20
nprobe=32,quantizer_efSearch=16          0.4141 0.8017 0.8453      0.01588       54097337    19
nprobe=32,quantizer_efSearch=32          0.4172 0.8095 0.8539      0.01630       54020579    19
nprobe=32,quantizer_efSearch=64          0.4180 0.8126 0.8571      0.01773       53994433    17
nprobe=32,quantizer_efSearch=128         0.4184 0.8129 0.8574      0.02193       53986751    14
nprobe=32,quantizer_efSearch=256         0.4185 0.8130 0.8575      0.02608       53984788    12
nprobe=64,quantizer_efSearch=16          0.4269 0.8486 0.9030      0.02912      106858402    11
nprobe=64,quantizer_efSearch=32          0.4306 0.8620 0.9191      0.03019      106661751    10
nprobe=64,quantizer_efSearch=64          0.4321 0.8664 0.9238      0.03103      106572082    10
nprobe=64,quantizer_efSearch=128         0.4326 0.8674 0.9249      0.03576      106536375    9
nprobe=64,quantizer_efSearch=256         0.4327 0.8675 0.9249      0.03924      106529998    8
nprobe=128,quantizer_efSearch=64         0.4364 0.8944 0.9643      0.07181      209553536    5
nprobe=128,quantizer_efSearch=128        0.4366 0.8959 0.9657      0.06775      209408054    5
nprobe=128,quantizer_efSearch=256        0.4369 0.8959 0.9662      0.08079      209381781    4
nprobe=128,quantizer_efSearch=512        0.4370 0.8962 0.9665      0.09594      209375062    4
nprobe=256,quantizer_efSearch=128        0.4396 0.9081 0.9854      0.13589      410765553    3
nprobe=256,quantizer_efSearch=256        0.4405 0.9091 0.9865      0.13431      410577504    3
nprobe=512,quantizer_efSearch=256        0.4411 0.9129 0.9936      0.26027      803883003    2
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 275804176

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ32x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2352 0.5918 0.7230      0.00235       28103636    128
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1239 0.2210 0.2375      0.00119        2410818    253
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1588 0.3182 0.3510      0.00133        4144513    227
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1595 0.3199 0.3530      0.00142        4795618    212
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1596 0.3212 0.3544      0.00138        4136629    218
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1672 0.3610 0.4100      0.00149        7170107    202
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1811 0.4011 0.4551      0.00148        7315263    204
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1899 0.4178 0.4740      0.00150        7623642    201
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1942 0.4303 0.4878      0.00156        7595200    193
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1948 0.4324 0.4903      0.00171        8246105    176
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2168 0.5158 0.6042      0.00185       14525585    163
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2202 0.5260 0.6183      0.00189       14475712    159
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2208 0.5271 0.6168      0.00207       15150323    145
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2346 0.5869 0.7169      0.00223       28177777    135
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2352 0.5918 0.7230      0.00233       28103485    129
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2425 0.6098 0.7450      0.00275       28679554    110
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2477 0.6410 0.8029      0.00299       55190552    101
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.2481 0.6369 0.7916      0.00307       56447362    98
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2485 0.6445 0.8063      0.00420       55059720    72
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2544 0.6662 0.8361      0.00367       55503924    83
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2546 0.6659 0.8362      0.00412       55488052    73
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.2581 0.6775 0.8675      0.00401      110352124    75
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2625 0.6936 0.8920      0.00438      108491061    69
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2632 0.6961 0.8947      0.00487      108271630    62
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2640 0.7008 0.9026      0.00668      109549187    45
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2653 0.7044 0.9075      0.00696      109329148    44
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2661 0.7163 0.9419      0.00853      212782043    36
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2665 0.7189 0.9446      0.00833      212434570    37
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.2674 0.7225 0.9558      0.01121      425168344    27
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.2683 0.7269 0.9693      0.02439      831801159    13
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2684 0.7274 0.9749      0.03759     1582583692    8
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 275798032

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2371 0.3560 0.3603      0.00234        4787661    129
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=2    0.1539 0.2196 0.2211      0.00129        1914143    233
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.2110 0.3171 0.3215      0.00139        3659968    216
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2301 0.3460 0.3504      0.00142        4143146    212
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2304 0.3485 0.3529      0.00150        4792364    201
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2305 0.3445 0.3490      0.00145        3812162    207
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2351 0.3522 0.3565      0.00145        4143489    207
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2360 0.3529 0.3572      0.00147        4137957    205
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2365 0.3537 0.3580      0.00148        4136460    204
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.2366 0.3538 0.3581      0.00158        4136529    191
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2368 0.3555 0.3598      0.00159        4786366    189
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.2371 0.3560 0.3603      0.00182        4784168    165
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.2842 0.4556 0.4634      0.00198        8289868    152
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3028 0.4885 0.4991      0.00206        8232530    146
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.3035 0.4883 0.4988      0.00248        8230773    121
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3485 0.6023 0.6218      0.00274       14500517    110
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3509 0.6069 0.6263      0.00278       14474159    108
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3559 0.6195 0.6392      0.00304       15100617    99
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3564 0.6197 0.6395      0.00323       15096825    93
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3565 0.6208 0.6407      0.00350       16375017    86
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3571 0.6210 0.6408      0.00372       16374915    81
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.3652 0.6546 0.6795      0.00444       27887678    68
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3970 0.7263 0.7566      0.00489       28661100    62
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3985 0.7301 0.7616      0.00525       29917845    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3986 0.7305 0.7620      0.00609       32462393    50
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4095 0.7841 0.8234      0.01565       56222923    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4155 0.8052 0.8481      0.01558       55538688    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4186 0.8131 0.8567      0.01568       56735216    20
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4310 0.8519 0.9060      0.02870      110754154    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4328 0.8668 0.9236      0.03019      109362904    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4332 0.8674 0.9248      0.03110      111802485    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.4348 0.8801 0.9450      0.06812      211800065    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.4365 0.8892 0.9541      0.05980      222782324    6
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.4394 0.8980 0.9664      0.06353      219804928    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.4395 0.8978 0.9664      0.06574      219739043    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.4396 0.9104 0.9859      0.12394      416142943    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.4398 0.9104 0.9866      0.13773      416032470    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.4401 0.9107 0.9886      0.24215      809218067    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.4410 0.9096 0.9860      0.24698      819625176    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4422 0.9131 0.9911      0.24752      819949220    2
```

</details>
<details><summary>`OPQ32_64,IVF65536,PQ32x4fs` </summary>
Index size 274115863

 code_size 16

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2670 0.7273 0.9670      0.01636      410461813    19
nprobe=2                                 0.1619 0.3253 0.3589      0.00806        3440024    38
nprobe=4                                 0.1989 0.4381 0.4964      0.00817        6879154    37
nprobe=8                                 0.2272 0.5424 0.6354      0.00885       13727571    34
nprobe=16                                0.2451 0.6155 0.7524      0.00948       27253055    32
nprobe=32                                0.2560 0.6731 0.8460      0.01049       53976201    29
nprobe=64                                0.2655 0.7069 0.9102      0.01214      106505221    25
nprobe=128                               0.2670 0.7209 0.9484      0.01539      209341594    20
nprobe=512                               0.2679 0.7277 0.9727      0.02747      803462432    11
nprobe=1024                              0.2686 0.7274 0.9746      0.03532     1571027769    9
```

</details>
<details><summary>`PCAR16,IVF16384_HNSW32,SQ8` </summary>
Index size 245715349

 code_size 16

 log filename: autotune.dbbigann10M.PCAR16_IVF16384_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0783 0.2742 0.4561      0.00279       33341963    108
nprobe=2,quantizer_efSearch=4            0.0686 0.2155 0.3220      0.00193       17095536    156
nprobe=2,quantizer_efSearch=8            0.0705 0.2252 0.3379      0.00215       17020612    140
nprobe=4,quantizer_efSearch=4            0.0763 0.2593 0.4288      0.00340       33388366    89
nprobe=4,quantizer_efSearch=8            0.0783 0.2742 0.4561      0.00278       33341963    108
nprobe=4,quantizer_efSearch=16           0.0788 0.2752 0.4604      0.00305       33245400    99
nprobe=4,quantizer_efSearch=32           0.0789 0.2753 0.4612      0.00345       33213639    87
nprobe=8,quantizer_efSearch=8            0.0831 0.3103 0.5652      0.00626       65122567    48
nprobe=16,quantizer_efSearch=8           0.0846 0.3315 0.6435      0.01006      127127752    30
nprobe=16,quantizer_efSearch=16          0.0852 0.3346 0.6524      0.00655      126999233    46
nprobe=32,quantizer_efSearch=16          0.0859 0.3433 0.6904      0.01090      248538491    28
nprobe=32,quantizer_efSearch=128         0.0862 0.3442 0.6936      0.01423      248120276    22
nprobe=64,quantizer_efSearch=32          0.0867 0.3469 0.7122      0.01905      484995005    16
nprobe=128,quantizer_efSearch=32         0.0868 0.3475 0.7190      0.03685      947888016    9
```

</details>
<details><summary>`PCAR16,IVF262144_HNSW32,SQ8` </summary>
Index size 330292629

 code_size 16

 log filename: autotune.dbbigann10M.PCAR16_IVF262144_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0707 0.2179 0.2814      0.00204        1726176    148
nprobe=2,quantizer_efSearch=4            0.0573 0.1536 0.1786      0.00178         866493    169
nprobe=4,quantizer_efSearch=4            0.0675 0.2043 0.2636      0.00186        1722752    161
nprobe=8,quantizer_efSearch=4            0.0774 0.2599 0.3794      0.00205        3428122    147
nprobe=8,quantizer_efSearch=8            0.0783 0.2625 0.3852      0.00217        3427742    139
nprobe=16,quantizer_efSearch=4           0.0812 0.2893 0.4694      0.00228        6792681    132
nprobe=16,quantizer_efSearch=8           0.0832 0.2980 0.4876      0.00235        6795519    128
nprobe=16,quantizer_efSearch=16          0.0841 0.3013 0.4933      0.00259        6793243    116
nprobe=16,quantizer_efSearch=32          0.0849 0.3025 0.4961      0.00326        6789014    93
nprobe=32,quantizer_efSearch=8           0.0861 0.3176 0.5696      0.00295       13429170    102
nprobe=32,quantizer_efSearch=16          0.0870 0.3238 0.5842      0.00368       13423170    82
nprobe=32,quantizer_efSearch=32          0.0874 0.3265 0.5882      0.00406       13412217    75
nprobe=64,quantizer_efSearch=32          0.0885 0.3368 0.6508      0.00584       26477823    52
nprobe=64,quantizer_efSearch=64          0.0886 0.3377 0.6533      0.00667       26463103    45
```

</details>
<details><summary>`PCAR16,IVF262144(IVF512,PQ8x4fs,RFlat),SQ8` </summary>
Index size 262167257

 code_size 16

 log filename: autotune.dbbigann10M.PCAR16_IVF262144_IVF512_PQ8x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0775 0.2659 0.4092      0.00258        8350654    117
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8   0.0461 0.1073 0.1161      0.00217        1811739    139
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1    0.0519 0.1287 0.1454      0.00218        1063923    138
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0547 0.1479 0.1702      0.00222        1585218    136
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.0669 0.1972 0.2454      0.00225        2116232    134
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0701 0.2095 0.2643      0.00229        2458061    132
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0716 0.2144 0.2705      0.00230        2440662    131
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.0775 0.2659 0.4092      0.00252        8350786    120
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.0782 0.2660 0.4092      0.00274        9661059    110
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0805 0.2876 0.4587      0.00271        7587074    111
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8   0.0818 0.2981 0.4824      0.00296        8260357    102
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0830 0.3015 0.4880      0.00324        9582666    93
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.0847 0.3144 0.5556      0.00386       15065320    78
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.0867 0.3242 0.5797      0.00506       18872247    60
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.0871 0.3337 0.6362      0.00616       29628699    49
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16 0.0873 0.3370 0.6671      0.00978       56128427    31
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=16 0.0877 0.3367 0.6433      0.00935       29302058    33
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0882 0.3403 0.6850      0.01047       55567823    29
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0883 0.3451 0.7144      0.01635      108799016    19
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=16 0.0885 0.3432 0.7046      0.02241      105794119    14
```

</details>
<details><summary>`PCAR16,IVF65536_HNSW32,SQ8` </summary>
Index size 262631317

 code_size 16

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0757 0.2543 0.3810      0.00188        6870464    160
nprobe=1,quantizer_efSearch=4            0.0520 0.1433 0.1695      0.00170        1738814    177
nprobe=2,quantizer_efSearch=4            0.0659 0.1973 0.2593      0.00173        3453918    174
nprobe=4,quantizer_efSearch=4            0.0721 0.2430 0.3602      0.00177        6855676    170
nprobe=4,quantizer_efSearch=8            0.0757 0.2543 0.3810      0.00193        6870464    156
nprobe=8,quantizer_efSearch=8            0.0811 0.2925 0.4893      0.00202       13614227    149
nprobe=8,quantizer_efSearch=16           0.0812 0.2959 0.4949      0.00255       13610692    118
nprobe=8,quantizer_efSearch=4            0.0813 0.2892 0.4812      0.00209       13608585    144
nprobe=8,quantizer_efSearch=32           0.0816 0.2966 0.4966      0.00305       13607543    99
nprobe=16,quantizer_efSearch=4           0.0849 0.3109 0.5625      0.00315       26890246    96
nprobe=16,quantizer_efSearch=32          0.0853 0.3226 0.5896      0.00362       26879306    83
nprobe=16,quantizer_efSearch=64          0.0855 0.3229 0.5899      0.00474       26877161    64
nprobe=32,quantizer_efSearch=16          0.0880 0.3366 0.6587      0.00499       53009184    61
nprobe=32,quantizer_efSearch=32          0.0882 0.3365 0.6604      0.00534       52983329    57
nprobe=32,quantizer_efSearch=64          0.0884 0.3366 0.6606      0.00590       52976894    51
nprobe=64,quantizer_efSearch=16          0.0886 0.3430 0.6937      0.00918      104268418    33
nprobe=64,quantizer_efSearch=32          0.0889 0.3431 0.6986      0.00795      104160697    38
nprobe=64,quantizer_efSearch=64          0.0892 0.3435 0.7007      0.00921      104133203    33
```

</details>
<details><summary>`PCAR16,IVF65536(IVF256,PQ8x4fs,RFlat),SQ8` </summary>
Index size 245617369

 code_size 16

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_IVF256_PQ8x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.0840 0.3113 0.5889      0.01238      108412136    25
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16 0.0526 0.1437 0.1719      0.00221        3083957    136
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8  0.0546 0.1467 0.1756      0.00226        2428756    133
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4  0.0547 0.1447 0.1729      0.00225        2090308    134
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0662 0.1999 0.2644      0.00232        3829572    130
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8  0.0663 0.2028 0.2698      0.00233        4148167    129
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=16 0.0669 0.2040 0.2719      0.00243        4795974    124
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.0721 0.2356 0.3534      0.00238        8261938    126
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16 0.0769 0.2562 0.3830      0.00249        8226995    121
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8   0.0814 0.2910 0.4862      0.00280       14374710    107
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16 0.0832 0.2948 0.4929      0.00332       14977463    91
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4  0.0838 0.3059 0.5474      0.00358       27409410    84
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0867 0.3197 0.5796      0.00389       29744435    78
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.0868 0.3198 0.5796      0.00443       32292748    69
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0879 0.3345 0.6519      0.00508       54726339    60
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0881 0.3354 0.6529      0.00567       55994164    53
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32 0.0882 0.3367 0.6602      0.00617       55779962    49
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.0886 0.3440 0.6993      0.01062      109863795    29
```

</details>
<details><summary>`PCAR16,IVF65536,SQ8` </summary>
Index size 244793824

 code_size 16

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0848 0.3232 0.5912      0.00685       26865185    44
nprobe=8                                 0.0814 0.2973 0.4978      0.00597       13600536    51
nprobe=16                                0.0848 0.3232 0.5912      0.00694       26865185    44
nprobe=32                                0.0875 0.3375 0.6626      0.00853       52948423    36
nprobe=64                                0.0883 0.3441 0.7024      0.01197      104069128    26
```

</details>
<details><summary>`PCAR32,IVF16384_HNSW32,SQ4` </summary>
Index size 246772309

 code_size 16

 log filename: autotune.dbbigann10M.PCAR32_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2195 0.4892 0.5368      0.00474       40132436    64
nprobe=1,quantizer_efSearch=16           0.1451 0.2620 0.2716      0.00248       10515145    122
nprobe=2,quantizer_efSearch=4            0.1781 0.3479 0.3688      0.00315       20828875    96
nprobe=2,quantizer_efSearch=8            0.1874 0.3732 0.3955      0.00305       20619554    99
nprobe=2,quantizer_efSearch=16           0.1890 0.3777 0.4002      0.00334       20508132    90
nprobe=2,quantizer_efSearch=32           0.1895 0.3788 0.4012      0.00374       20466989    81
nprobe=4,quantizer_efSearch=8            0.2195 0.4892 0.5368      0.00451       40132436    67
nprobe=4,quantizer_efSearch=16           0.2221 0.4970 0.5450      0.00474       39907993    64
nprobe=4,quantizer_efSearch=32           0.2224 0.4984 0.5464      0.00534       39826426    57
nprobe=4,quantizer_efSearch=64           0.2225 0.4984 0.5464      0.00623       39812951    49
nprobe=8,quantizer_efSearch=8            0.2474 0.5925 0.6723      0.01176       77444348    26
nprobe=8,quantizer_efSearch=16           0.2533 0.6074 0.6901      0.00804       77079903    38
nprobe=8,quantizer_efSearch=32           0.2543 0.6097 0.6932      0.00820       76894982    37
nprobe=16,quantizer_efSearch=8           0.2702 0.6715 0.7905      0.01285      149066994    24
nprobe=16,quantizer_efSearch=32          0.2730 0.6835 0.8073      0.01330      148296866    23
nprobe=16,quantizer_efSearch=16          0.2731 0.6825 0.8043      0.01298      148737053    24
nprobe=32,quantizer_efSearch=8           0.2770 0.7108 0.8595      0.02450      287099064    13
nprobe=32,quantizer_efSearch=16          0.2822 0.7327 0.8919      0.02225      286429068    14
nprobe=32,quantizer_efSearch=64          0.2847 0.7365 0.8988      0.02427      285421050    13
nprobe=64,quantizer_efSearch=32          0.2886 0.7600 0.9478      0.04186      550223763    8
nprobe=64,quantizer_efSearch=64          0.2891 0.7621 0.9496      0.04142      549422766    8
nprobe=64,quantizer_efSearch=128         0.2892 0.7627 0.9499      0.04401      549029559    7
nprobe=128,quantizer_efSearch=32         0.2900 0.7673 0.9684      0.07853     1056886397    4
nprobe=128,quantizer_efSearch=128        0.2904 0.7712 0.9732      0.07918     1053722331    4
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ4` </summary>
Index size 347078229

 code_size 16

 log filename: autotune.dbbigann10M.PCAR32_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1796 0.3440 0.3584      0.00222        1744495    136
nprobe=1,quantizer_efSearch=4            0.0968 0.1479 0.1490      0.00182         436634    165
nprobe=2,quantizer_efSearch=4            0.1357 0.2278 0.2329      0.00187         874343    161
nprobe=4,quantizer_efSearch=4            0.1648 0.3132 0.3265      0.00197        1747074    153
nprobe=4,quantizer_efSearch=8            0.1796 0.3440 0.3584      0.00222        1744495    135
nprobe=8,quantizer_efSearch=8            0.2145 0.4460 0.4763      0.00231        3480865    130
nprobe=16,quantizer_efSearch=4           0.2346 0.5188 0.5731      0.00243        6945967    124
nprobe=16,quantizer_efSearch=8           0.2406 0.5389 0.5978      0.00253        6938576    119
nprobe=16,quantizer_efSearch=16          0.2418 0.5462 0.6056      0.00315        6930134    96
nprobe=32,quantizer_efSearch=4           0.2438 0.5701 0.6498      0.00347       13774094    87
nprobe=32,quantizer_efSearch=8           0.2531 0.6074 0.6967      0.00349       13780536    86
nprobe=32,quantizer_efSearch=16          0.2570 0.6260 0.7213      0.00397       13755409    76
nprobe=32,quantizer_efSearch=32          0.2592 0.6314 0.7279      0.00454       13734069    67
nprobe=64,quantizer_efSearch=16          0.2679 0.6741 0.8039      0.00849       27230886    36
nprobe=64,quantizer_efSearch=32          0.2706 0.6859 0.8203      0.00866       27172951    35
nprobe=64,quantizer_efSearch=64          0.2709 0.6887 0.8243      0.00861       27142182    35
nprobe=128,quantizer_efSearch=32         0.2750 0.7144 0.8819      0.00962       53678709    32
nprobe=128,quantizer_efSearch=64         0.2767 0.7224 0.8937      0.01125       53563737    27
nprobe=128,quantizer_efSearch=128        0.2774 0.7238 0.8964      0.01686       53507671    18
nprobe=256,quantizer_efSearch=64         0.2783 0.7369 0.9322      0.02020      105514929    15
nprobe=256,quantizer_efSearch=128        0.2790 0.7405 0.9375      0.02274      105294613    14
nprobe=512,quantizer_efSearch=64         0.2796 0.7439 0.9517      0.03445      206985120    9
nprobe=256,quantizer_efSearch=256        0.2797 0.7415 0.9391      0.02744      105219471    11
nprobe=512,quantizer_efSearch=128        0.2804 0.7492 0.9607      0.03721      206875453    9
nprobe=512,quantizer_efSearch=256        0.2812 0.7505 0.9639      0.04556      206511313    7
nprobe=1024,quantizer_efSearch=256       0.2814 0.7532 0.9737      0.11663      404665078    3
nprobe=1024,quantizer_efSearch=512       0.2815 0.7536 0.9745      0.12602      404081438    3
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ4` </summary>
Index size 280067993

 code_size 16

 log filename: autotune.dbbigann10M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2584 0.6286 0.7218      0.00647       33167773    47
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.0668 0.0978 0.0982      0.00222        1146242    136
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1045 0.1571 0.1582      0.00225        1823941    134
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.1088 0.1816 0.1844      0.00223        1062094    135
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1    0.1100 0.1834 0.1863      0.00226        1059955    133
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1365 0.2312 0.2349      0.00228        1582836    132
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1451 0.2454 0.2493      0.00235        2259577    128
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1561 0.2882 0.2970      0.00240        2476723    125
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1595 0.2984 0.3078      0.00239        2122325    126
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1691 0.3177 0.3286      0.00235        2469660    128
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1760 0.3311 0.3420      0.00238        2465107    126
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2003 0.4058 0.4285      0.00246        4910042    123
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2034 0.4170 0.4406      0.00249        4224468    121
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.2090 0.4404 0.4725      0.00273        8460724    110
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2270 0.5055 0.5506      0.00276        8406081    109
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2309 0.5188 0.5651      0.00317        9721447    95
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.2365 0.5343 0.5877      0.00332        8350644    91
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2486 0.5837 0.6607      0.00366       15311726    82
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2511 0.6037 0.6876      0.00436       15242396    69
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2552 0.6088 0.6890      0.00477       19199590    64
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2659 0.6682 0.7875      0.00629       30182335    48
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2689 0.6765 0.7967      0.00665       32735285    46
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2723 0.6896 0.8224      0.00812       32587620    37
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2730 0.6916 0.8241      0.00904       37754773    34
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2734 0.6944 0.8281      0.01096       37671141    28
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2778 0.7209 0.8816      0.01130       64493769    27
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2788 0.7290 0.8969      0.02079       74377485    15
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.2791 0.7300 0.8989      0.02484       64068917    13
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2814 0.7474 0.9393      0.02644      115991087    12
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.2823 0.7561 0.9620      0.04167      248875183    8
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.2826 0.7573 0.9671      0.05782      247604321    6
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ4` </summary>
Index size 266834005

 code_size 16

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2797 0.7502 0.9697      0.06986      408655495    5
nprobe=2,quantizer_efSearch=4            0.1620 0.3120 0.3278      0.00184        3479680    163
nprobe=4,quantizer_efSearch=4            0.1945 0.4039 0.4364      0.00183        6947985    164
nprobe=4,quantizer_efSearch=8            0.2060 0.4315 0.4673      0.00217        6941833    139
nprobe=4,quantizer_efSearch=16           0.2077 0.4356 0.4720      0.00237        6929613    127
nprobe=8,quantizer_efSearch=8            0.2303 0.5289 0.5936      0.00271       13804338    111
nprobe=8,quantizer_efSearch=16           0.2349 0.5381 0.6053      0.00299       13784187    101
nprobe=8,quantizer_efSearch=32           0.2360 0.5402 0.6074      0.00362       13772942    83
nprobe=16,quantizer_efSearch=4           0.2436 0.5866 0.6834      0.00376       27366312    80
nprobe=16,quantizer_efSearch=8           0.2517 0.6103 0.7141      0.00396       27338163    76
nprobe=16,quantizer_efSearch=16          0.2543 0.6177 0.7236      0.00415       27313485    73
nprobe=16,quantizer_efSearch=32          0.2555 0.6216 0.7286      0.00486       27287459    62
nprobe=32,quantizer_efSearch=8           0.2633 0.6616 0.7988      0.00626       54089396    48
nprobe=32,quantizer_efSearch=32          0.2688 0.6830 0.8291      0.00681       53945494    45
nprobe=32,quantizer_efSearch=64          0.2694 0.6847 0.8312      0.00813       53919917    37
nprobe=64,quantizer_efSearch=32          0.2754 0.7199 0.8998      0.01130      106394123    27
nprobe=64,quantizer_efSearch=64          0.2762 0.7223 0.9033      0.01233      106305538    25
nprobe=128,quantizer_efSearch=64         0.2788 0.7404 0.9459      0.02439      208928118    13
nprobe=128,quantizer_efSearch=128        0.2791 0.7414 0.9474      0.02437      208781527    13
nprobe=256,quantizer_efSearch=128        0.2797 0.7500 0.9689      0.04326      408867019    7
nprobe=256,quantizer_efSearch=256        0.2798 0.7500 0.9693      0.04480      408688265    7
nprobe=512,quantizer_efSearch=128        0.2802 0.7512 0.9779      0.09160      800329092    4
```

</details>
<details><summary>`PCAR32,IVF65536(IVF256,PQ16x4fs,RFlat),SQ4` </summary>
Index size 250115737

 code_size 16

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_IVF256_PQ16x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2478 0.6033 0.7021      0.00509       28125616    59
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2    0.1164 0.2020 0.2071      0.00231        1929718    131
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.1275 0.2229 0.2283      0.00228        3071477    132
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1293 0.2251 0.2305      0.00232        2422658    130
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1601 0.3149 0.3308      0.00241        3831787    125
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1653 0.3250 0.3415      0.00239        4154854    126
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2015 0.4286 0.4606      0.00249        8282614    121
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2031 0.4318 0.4650      0.00253        7636149    119
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.2037 0.4335 0.4667      0.00267        7628983    113
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2051 0.4387 0.4723      0.00303        9554051    100
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16   0.2056 0.4383 0.4718      0.00303        8263667    99
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2224 0.5059 0.5600      0.00340       14580495    89
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2310 0.5341 0.5959      0.00335       14513994    90
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.2338 0.5424 0.6058      0.00392       15125463    77
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=16   0.2340 0.5426 0.6060      0.00437       15122282    69
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2478 0.6033 0.7021      0.00521       28124818    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2522 0.6196 0.7208      0.00550       28710161    55
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2536 0.6230 0.7253      0.00559       28667877    54
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.2540 0.6254 0.7289      0.00672       29909891    45
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32  0.2541 0.6255 0.7291      0.00748       29912574    41
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2576 0.6498 0.7798      0.00853       55178408    36
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2587 0.6578 0.7927      0.00906       54942410    34
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2650 0.6776 0.8205      0.00879       55453220    35
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2653 0.6800 0.8232      0.00949       55376312    32
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2661 0.6848 0.8309      0.00975       56580610    31
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2719 0.7141 0.8906      0.01627      109478360    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2741 0.7239 0.9049      0.01645      111543643    19
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2745 0.7295 0.9258      0.02873      211714702    11
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2765 0.7403 0.9462      0.02854      211698659    11
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2770 0.7415 0.9480      0.03015      219144123    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.2772 0.7425 0.9491      0.03224      218997041    10
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.2775 0.7427 0.9519      0.05216      424971963    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2782 0.7477 0.9670      0.05357      415322072    6
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.2783 0.7499 0.9707      0.05973      413883052    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2787 0.7504 0.9791      0.10237      811142223    3
```

</details>
<details><summary>`PCAR32,IVF65536,SQ4` </summary>
Index size 248996512

 code_size 16

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2769 0.7526 0.9701      0.04635      408342557    7
nprobe=2                                 0.1686 0.3315 0.3468      0.00618        3459974    49
nprobe=8                                 0.2351 0.5448 0.6096      0.00902       13759195    34
nprobe=16                                0.2541 0.6288 0.7313      0.00892       27257214    34
nprobe=32                                0.2671 0.6886 0.8336      0.01115       53861966    27
nprobe=64                                0.2735 0.7270 0.9055      0.01611      106169467    19
nprobe=128                               0.2760 0.7448 0.9488      0.02548      208562864    12
nprobe=256                               0.2769 0.7526 0.9701      0.04614      408342557    7
nprobe=512                               0.2774 0.7537 0.9790      0.08295      798336663    4
nprobe=2048                              0.2775 0.7542 0.9822      0.28406     3048501660    2
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF16384_HNSW32,PQ64x4fs` </summary>
Index size 421225605

 code_size 32

 log filename: autotune.dbbigann10M.IVF16384_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3400 0.5730 0.6008      0.00247       28841811    122
nprobe=1,quantizer_efSearch=4            0.1926 0.2873 0.2958      0.00138        7304261    218
nprobe=1,quantizer_efSearch=8            0.2051 0.3039 0.3127      0.00160        7285355    188
nprobe=2,quantizer_efSearch=4            0.2574 0.4099 0.4262      0.00177       14538324    170
nprobe=2,quantizer_efSearch=8            0.2756 0.4374 0.4542      0.00199       14535232    151
nprobe=4,quantizer_efSearch=4            0.3167 0.5335 0.5596      0.00222       28836075    136
nprobe=4,quantizer_efSearch=8            0.3400 0.5730 0.6008      0.00246       28841811    123
nprobe=8,quantizer_efSearch=4            0.3793 0.6696 0.7089      0.00276       57062190    109
nprobe=8,quantizer_efSearch=8            0.3857 0.6854 0.7253      0.00287       57007254    105
nprobe=16,quantizer_efSearch=4           0.4073 0.7472 0.7973      0.00321      112345480    94
nprobe=16,quantizer_efSearch=8           0.4222 0.7782 0.8308      0.00346      112263620    87
nprobe=16,quantizer_efSearch=16          0.4264 0.7865 0.8402      0.00368      112084263    82
nprobe=32,quantizer_efSearch=8           0.4351 0.8305 0.8911      0.00442      220571042    68
nprobe=32,quantizer_efSearch=16          0.4436 0.8491 0.9130      0.00472      220131180    64
nprobe=32,quantizer_efSearch=32          0.4452 0.8540 0.9183      0.00548      219827319    56
nprobe=64,quantizer_efSearch=16          0.4526 0.8806 0.9525      0.00754      430925180    40
nprobe=64,quantizer_efSearch=32          0.4562 0.8897 0.9626      0.00797      429986590    38
nprobe=128,quantizer_efSearch=32         0.4578 0.9030 0.9810      0.01164      839057341    26
nprobe=128,quantizer_efSearch=64         0.4581 0.9062 0.9857      0.01266      837263562    24
nprobe=128,quantizer_efSearch=128        0.4583 0.9067 0.9862      0.01618      836665460    19
nprobe=256,quantizer_efSearch=32         0.4590 0.9071 0.9875      0.01619     1633261751    19
nprobe=256,quantizer_efSearch=128        0.4594 0.9123 0.9945      0.02120     1627139094    15
```

</details>
<details><summary>`IVF262144_HNSW32,PQ64x4fs` </summary>
Index size 737772677

 code_size 32

 log filename: autotune.dbbigann10M.IVF262144_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4299 0.8244 0.8823      0.01667      179654569    18
nprobe=1,quantizer_efSearch=4            0.1272 0.1646 0.1661      0.00166         434294    181
nprobe=2,quantizer_efSearch=4            0.1764 0.2466 0.2503      0.00180         870228    167
nprobe=4,quantizer_efSearch=4            0.2247 0.3409 0.3476      0.00209        1742913    144
nprobe=4,quantizer_efSearch=8            0.2497 0.3774 0.3847      0.00318        1736500    95
nprobe=8,quantizer_efSearch=4            0.2958 0.4807 0.4947      0.00296        3476331    102
nprobe=16,quantizer_efSearch=4           0.3344 0.5727 0.5957      0.00401        6945886    75
nprobe=16,quantizer_efSearch=8           0.3519 0.6069 0.6308      0.00460        6932966    66
nprobe=32,quantizer_efSearch=4           0.3570 0.6318 0.6627      0.00519       13796656    58
nprobe=32,quantizer_efSearch=8           0.3837 0.6852 0.7205      0.00602       13799553    50
nprobe=64,quantizer_efSearch=8           0.3996 0.7349 0.7769      0.00852       27343216    36
nprobe=128,quantizer_efSearch=8          0.4065 0.7550 0.8004      0.01241       53817660    25
nprobe=64,quantizer_efSearch=16          0.4181 0.7776 0.8233      0.01001       27314174    30
nprobe=64,quantizer_efSearch=32          0.4262 0.7984 0.8460      0.01272       27243971    24
nprobe=256,quantizer_efSearch=16         0.4289 0.8233 0.8801      0.02008      105391012    15
nprobe=128,quantizer_efSearch=32         0.4376 0.8429 0.9005      0.01686       53916124    18
nprobe=256,quantizer_efSearch=32         0.4436 0.8639 0.9279      0.02339      106327750    13
nprobe=512,quantizer_efSearch=32         0.4449 0.8696 0.9363      0.03235      203922074    10
nprobe=256,quantizer_efSearch=64         0.4523 0.8825 0.9496      0.02863      106141686    11
nprobe=512,quantizer_efSearch=64         0.4538 0.8930 0.9652      0.03876      208403510    8
nprobe=256,quantizer_efSearch=128        0.4558 0.8889 0.9569      0.03766      105899655    8
nprobe=512,quantizer_efSearch=128        0.4569 0.9019 0.9762      0.04827      208368222    7
nprobe=256,quantizer_efSearch=256        0.4570 0.8907 0.9589      0.05277      105800558    6
nprobe=1024,quantizer_efSearch=128       0.4592 0.9062 0.9835      0.07189      406561779    5
nprobe=2048,quantizer_efSearch=128       0.4594 0.9079 0.9855      0.08009      734905817    4
nprobe=4096,quantizer_efSearch=128       0.4595 0.9080 0.9856      0.07998      886739492    4
nprobe=1024,quantizer_efSearch=256       0.4599 0.9106 0.9886      0.08290      408045707    4
nprobe=2048,quantizer_efSearch=256       0.4608 0.9128 0.9922      0.10790      789526293    3
```

</details>
<details><summary>`IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 677837257

 code_size 32

 log filename: autotune.dbbigann10M.IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4149 0.7814 0.8267      0.00992       33200552    31
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1115 0.1450 0.1464      0.00111         616137    271
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1127 0.1465 0.1480      0.00117         616016    256
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1323 0.1828 0.1851      0.00120        1062662    251
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1566 0.2153 0.2180      0.00123        1233139    244
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1691 0.2334 0.2363      0.00137        1581032    220
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1805 0.2518 0.2550      0.00139        1580142    216
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2170 0.3256 0.3323      0.00154        2115464    196
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2183 0.3302 0.3373      0.00170        2108678    176
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2231 0.3304 0.3357      0.00163        2466704    184
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2387 0.3599 0.3672      0.00164        2461144    183
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2409 0.3652 0.3731      0.00175        2455443    172
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2419 0.3678 0.3754      0.00205        2451486    147
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2717 0.4345 0.4475      0.00197        4246229    153
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3049 0.4925 0.5075      0.00239        4882974    126
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3365 0.5749 0.5972      0.00313        8444910    96
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3488 0.6018 0.6254      0.00334        8361169    90
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3693 0.6636 0.6963      0.00384       15486006    79
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3801 0.6860 0.7225      0.00461       15257402    66
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3820 0.6897 0.7239      0.00465       16752080    65
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3945 0.7310 0.7737      0.00546       29464434    55
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4009 0.7253 0.7630      0.00645       19137693    47
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4023 0.7278 0.7653      0.00819       24327285    37
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4095 0.7693 0.8142      0.00646       30655758    47
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4212 0.7885 0.8360      0.00695       30095116    44
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4275 0.8027 0.8512      0.00875       32656516    35
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4283 0.8042 0.8527      0.01013       32567958    30
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4304 0.8232 0.8780      0.00790       58185047    38
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4349 0.8330 0.8896      0.01103       56842096    28
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4414 0.8468 0.9028      0.01349       65686172    23
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4475 0.8602 0.9192      0.01520       64374175    20
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4484 0.8728 0.9390      0.01963      114583417    16
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4532 0.8821 0.9504      0.01983      111637867    16
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.4539 0.8834 0.9512      0.02345      111431553    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4573 0.8933 0.9621      0.02523      126712398    12
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4575 0.8935 0.9616      0.02820      116332254    11
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4582 0.8943 0.9626      0.03074      126468951    10
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.4584 0.9068 0.9820      0.04572      249269339    7
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4607 0.9116 0.9892      0.04918      432137945    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.4613 0.9120 0.9909      0.06173      419004037    5
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4615 0.9132 0.9928      0.06279      428387995    5
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4630 0.9168 0.9972      0.10484      818613237    3
```

</details>
<details><summary>`IVF65536_HNSW32,PQ64x4fs` </summary>
Index size 484872837

 code_size 32

 log filename: autotune.dbbigann10M.IVF65536_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2886 0.4755 0.4924      0.00194        6878394    155
nprobe=1,quantizer_efSearch=4            0.1514 0.2165 0.2206      0.00113        1723009    266
nprobe=2,quantizer_efSearch=4            0.2086 0.3229 0.3315      0.00128        3451455    236
nprobe=4,quantizer_efSearch=4            0.2628 0.4346 0.4505      0.00153        6889961    196
nprobe=4,quantizer_efSearch=8            0.2886 0.4755 0.4924      0.00188        6878394    160
nprobe=8,quantizer_efSearch=4            0.3322 0.5836 0.6103      0.00209       13750196    144
nprobe=8,quantizer_efSearch=8            0.3400 0.5971 0.6248      0.00220       13740571    137
nprobe=16,quantizer_efSearch=4           0.3627 0.6660 0.7026      0.00271       27336004    111
nprobe=16,quantizer_efSearch=8           0.3788 0.6995 0.7394      0.00296       27304609    102
nprobe=16,quantizer_efSearch=16          0.3846 0.7140 0.7547      0.00333       27265659    91
nprobe=32,quantizer_efSearch=8           0.4039 0.7646 0.8143      0.00380       54166544    79
nprobe=64,quantizer_efSearch=8           0.4138 0.7978 0.8566      0.00486      106961084    62
nprobe=64,quantizer_efSearch=16          0.4273 0.8326 0.8973      0.00531      106858958    57
nprobe=64,quantizer_efSearch=32          0.4358 0.8496 0.9167      0.00782      106672864    39
nprobe=64,quantizer_efSearch=64          0.4366 0.8530 0.9204      0.00975      106568169    31
nprobe=128,quantizer_efSearch=32         0.4423 0.8783 0.9538      0.01027      210050578    30
nprobe=128,quantizer_efSearch=64         0.4445 0.8849 0.9617      0.01156      209656827    27
nprobe=256,quantizer_efSearch=32         0.4456 0.8892 0.9683      0.01510      412508129    20
nprobe=128,quantizer_efSearch=128        0.4457 0.8864 0.9634      0.01603      209503678    19
nprobe=256,quantizer_efSearch=64         0.4492 0.8987 0.9810      0.01532      411744995    20
nprobe=256,quantizer_efSearch=128        0.4498 0.9007 0.9838      0.01975      411076724    16
nprobe=256,quantizer_efSearch=256        0.4503 0.9010 0.9843      0.02728      410874476    11
```

</details>
<details><summary>`IVF65536(IVF256,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 470022601

 code_size 32

 log filename: autotune.dbbigann10M.IVF65536_IVF256_PQ64x4fs_RFlat__PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.4418 0.8771 0.9526      0.01069      217906723    29
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1649 0.2341 0.2384      0.00135        2407736    223
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2198 0.3329 0.3410      0.00143        3825039    210
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2226 0.3391 0.3476      0.00145        3813760    208
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2289 0.3504 0.3593      0.00149        4137900    201
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.2365 0.3847 0.3978      0.00165        7201109    182
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2750 0.4473 0.4614      0.00168        7653545    179
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2914 0.4773 0.4937      0.00171        7598682    176
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3120 0.5333 0.5577      0.00208       14360148    145
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3230 0.5568 0.5840      0.00208       14199262    145
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3296 0.5676 0.5931      0.00218       14638333    138
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3417 0.5976 0.6259      0.00224       14482580    134
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3433 0.6005 0.6290      0.00271       14435882    111
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.3454 0.6176 0.6523      0.00259       28362834    116
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.3722 0.6741 0.7114      0.00272       28552431    111
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3786 0.6963 0.7362      0.00306       28033744    99
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3795 0.6905 0.7298      0.00350       29118481    86
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.4034 0.7639 0.8153      0.00371       55062620    81
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4093 0.7737 0.8255      0.00485       56593828    62
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.4159 0.8024 0.8632      0.00499      108130473    61
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.4164 0.7920 0.8464      0.00532       55392943    57
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.4194 0.8002 0.8554      0.00636       56566791    48
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4287 0.8318 0.8960      0.00671      110778910    45
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4322 0.8410 0.9067      0.00704      108334759    43
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4339 0.8433 0.9084      0.00762      111783987    40
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4368 0.8540 0.9210      0.00805      109374742    38
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.4395 0.8696 0.9433      0.00948      211880100    32
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.4418 0.8771 0.9526      0.01080      217909562    28
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4421 0.8789 0.9544      0.01186      220191213    26
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.4462 0.8867 0.9637      0.01128      212500128    27
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.4469 0.8883 0.9656      0.01594      219632093    19
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4480 0.9000 0.9820      0.01650      427718735    19
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.4510 0.9029 0.9859      0.01884      416236135    16
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4511 0.9029 0.9861      0.01967      421226751    16
```

</details>
<details><summary>`IVF65536,PQ64x4fs` </summary>
Index size 467115216

 code_size 32

 log filename: autotune.dbbigann10M.IVF65536_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4505 0.9022 0.9857      0.02444      410218508    13
nprobe=2                                 0.2317 0.3550 0.3640      0.01157        3433862    26
nprobe=4                                 0.2962 0.4867 0.5037      0.01190        6851345    26
nprobe=8                                 0.3506 0.6162 0.6445      0.01309       13683056    23
nprobe=16                                0.3916 0.7250 0.7663      0.01348       27173023    23
nprobe=32                                0.4194 0.8011 0.8561      0.01413       53836692    22
nprobe=128                               0.4466 0.8880 0.9657      0.01919      209127406    16
nprobe=256                               0.4505 0.9022 0.9857      0.02424      410218508    13
```

</details>
<details><summary>`OPQ32_128,IVF16384_HNSW32,PQ32` </summary>
Index size 413177008

 code_size 32

 log filename: autotune.dbbigann10M.OPQ32_128_IVF16384_HNSW32_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.5294 0.9476 0.9959      0.52856     2744645080    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0018 0.0021 0.0021      0.00265       17316790    114
nprobe=1,quantizer_efSearch=4,ht=106      0.1917 0.2336 0.2336      0.00309       17316790    98
nprobe=1,quantizer_efSearch=16,ht=114     0.2240 0.2842 0.2843      0.00436       16661946    69
nprobe=2,quantizer_efSearch=4,ht=102      0.2482 0.3089 0.3090      0.00466       33579449    65
nprobe=2,quantizer_efSearch=16,ht=110     0.3122 0.4135 0.4141      0.00597       32308917    51
nprobe=2,quantizer_efSearch=64,ht=120     0.3202 0.4341 0.4350      0.00962       32107617    32
nprobe=4,quantizer_efSearch=32,ht=256     0.4033 0.5851 0.5881      0.01024       61728198    30
nprobe=8,quantizer_efSearch=32,ht=106     0.4410 0.6490 0.6520      0.01654      117734600    19
nprobe=8,quantizer_efSearch=16,ht=114     0.4560 0.7035 0.7098      0.01851      118572106    17
nprobe=8,quantizer_efSearch=32,ht=116     0.4598 0.7114 0.7181      0.02025      117734600    15
nprobe=8,quantizer_efSearch=64,ht=126     0.4609 0.7149 0.7223      0.02425      117416397    13
nprobe=16,quantizer_efSearch=64,ht=104    0.4717 0.7224 0.7284      0.03021      223256136    10
nprobe=16,quantizer_efSearch=64,ht=110    0.4926 0.7955 0.8071      0.03280      223256136    10
nprobe=16,quantizer_efSearch=16,ht=128    0.4958 0.8125 0.8283      0.04059      225710570    8
nprobe=16,quantizer_efSearch=64,ht=122    0.5004 0.8221 0.8386      0.04108      223256136    8
nprobe=32,quantizer_efSearch=32,ht=116    0.5192 0.8867 0.9103      0.06717      424035868    5
nprobe=32,quantizer_efSearch=512,ht=128   0.5203 0.8918 0.9169      0.10757      421823158    3
nprobe=64,quantizer_efSearch=256,ht=114   0.5245 0.9215 0.9554      0.13326      791376617    3
nprobe=64,quantizer_efSearch=64,ht=126    0.5268 0.9281 0.9643      0.14638      793158079    3
nprobe=128,quantizer_efSearch=64,ht=256   0.5292 0.9422 0.9852      0.17374     1482476000    2
nprobe=256,quantizer_efSearch=512,ht=120  0.5294 0.9476 0.9959      0.54039     2744645080    1
nprobe=256,quantizer_efSearch=512,ht=122  0.5297 0.9478 0.9964      0.55587     2744645080    1
nprobe=512,quantizer_efSearch=64,ht=122   0.5298 0.9467 0.9950      0.99187     4993338065    1
nprobe=512,quantizer_efSearch=64,ht=128   0.5300 0.9472 0.9957      1.02051     4993338065    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.5301 0.9489 0.9995      2.00289     9360159235    1
```

</details>
<details><summary>`OPQ32_128,IVF262144_HNSW32,PQ32` </summary>
Index size 607854768

 code_size 32

 log filename: autotune.dbbigann10M.OPQ32_128_IVF262144_HNSW32_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6708 0.9266 0.9287      0.41319      105775878    1
nprobe=1,quantizer_efSearch=4,ht=106      0.0984 0.1077 0.1077      0.00309         434287    98
nprobe=2,quantizer_efSearch=4,ht=102      0.1131 0.1279 0.1280      0.00425         871092    71
nprobe=1,quantizer_efSearch=16,ht=116     0.1476 0.1639 0.1639      0.00621         430838    49
nprobe=2,quantizer_efSearch=16,ht=110     0.1835 0.2089 0.2090      0.00749         865057    41
nprobe=4,quantizer_efSearch=8,ht=118      0.3059 0.3626 0.3629      0.00975        1736918    31
nprobe=4,quantizer_efSearch=32,ht=256     0.3298 0.3951 0.3954      0.01088        1730150    28
nprobe=8,quantizer_efSearch=16,ht=114     0.3788 0.4569 0.4574      0.01525        3462111    20
nprobe=8,quantizer_efSearch=16,ht=128     0.4195 0.5187 0.5192      0.01564        3462111    20
nprobe=16,quantizer_efSearch=64,ht=256    0.5112 0.6519 0.6527      0.02323        6903335    13
nprobe=32,quantizer_efSearch=8,ht=124     0.5467 0.7145 0.7154      0.04608       13801286    7
nprobe=64,quantizer_efSearch=64,ht=126    0.6232 0.8441 0.8458      0.10053       27202184    3
nprobe=64,quantizer_efSearch=128,ht=126   0.6236 0.8443 0.8459      0.10871       27184628    3
nprobe=128,quantizer_efSearch=512,ht=256  0.6605 0.9150 0.9174      0.14820       53691609    3
nprobe=256,quantizer_efSearch=32,ht=126   0.6614 0.9218 0.9243      0.34256      106331206    1
nprobe=256,quantizer_efSearch=512,ht=120  0.6708 0.9266 0.9287      0.41163      105775878    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6759 0.9404 0.9427      0.41329      105775878    1
nprobe=512,quantizer_efSearch=64,ht=128   0.6807 0.9600 0.9639      0.68960      208401368    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6859 0.9687 0.9727      1.43662      407517473    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.6907 0.9830 0.9877      1.43716      407517473    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6936 0.9913 0.9964      2.14944     1525757623    1
```

</details>
<details><summary>`OPQ32_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ32` </summary>
Index size 547540468

 code_size 32

 log filename: autotune.dbbigann10M.OPQ32_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ32.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.1827 0.2074 0.2075      0.00370        1055191    82
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=116       0.1193 0.1312 0.1312      0.00252         617429    119
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=256      0.1827 0.2074 0.2075      0.00357        1055173    85
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=128       0.2360 0.2697 0.2698      0.00411        2264242    73
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=122      0.2376 0.2715 0.2716      0.00496        3588635    61
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=256       0.3508 0.4281 0.4283      0.00573        3865216    53
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=118       0.3571 0.4283 0.4284      0.01235        4227500    25
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=124       0.4150 0.5076 0.5078      0.01394        4863109    22
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=126      0.4228 0.5163 0.5165      0.01348        6190617    23
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=128     0.4319 0.5277 0.5279      0.01869       13970373    17
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=256     0.5103 0.6599 0.6605      0.02636       14594770    12
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=128    0.5106 0.6477 0.6479      0.02667        9643402    12
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=116      0.5208 0.6647 0.6652      0.04720       15211971    7
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=126     0.5742 0.7522 0.7528      0.04806       16478154    7
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=256     0.5894 0.7899 0.7911      0.07256       28798394    5
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=116    0.5936 0.7756 0.7765      0.10150       32566381    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128    0.6314 0.8522 0.8534      0.10480       37723905    3
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=256  0.6652 0.9193 0.9211      0.16143       74469099    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=256   0.6786 0.9503 0.9524      0.20065      111419516    2
nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=256  0.6960 0.9926 0.9957      1.33627      810169892    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=256 0.6975 0.9964 0.9997      3.69165     1599503372    1
```

</details>
<details><summary>`OPQ32_128,IVF65536_HNSW32,PQ32` </summary>
Index size 452113072

 code_size 32

 log filename: autotune.dbbigann10M.OPQ32_128_IVF65536_HNSW32_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6732 0.9667 0.9694      0.33377      410821310    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0017 0.0021 0.0021      0.00247        1724419    122
nprobe=1,quantizer_efSearch=4,ht=68       0.0066 0.0079 0.0079      0.00242        1724419    125
nprobe=1,quantizer_efSearch=4,ht=106      0.1390 0.1582 0.1583      0.00245        1724419    123
nprobe=1,quantizer_efSearch=16,ht=114     0.1876 0.2163 0.2164      0.00364        1719185    83
nprobe=1,quantizer_efSearch=16,ht=116     0.1936 0.2237 0.2238      0.00342        1719185    88
nprobe=2,quantizer_efSearch=16,ht=110     0.2547 0.2981 0.2982      0.00468        3440849    65
nprobe=4,quantizer_efSearch=32,ht=256     0.4013 0.5017 0.5021      0.00596        6859641    51
nprobe=8,quantizer_efSearch=16,ht=114     0.4653 0.5855 0.5862      0.01192       13722458    26
nprobe=8,quantizer_efSearch=16,ht=128     0.4932 0.6362 0.6370      0.01221       13722458    25
nprobe=16,quantizer_efSearch=64,ht=256    0.5725 0.7630 0.7641      0.01653       27217875    19
nprobe=32,quantizer_efSearch=8,ht=124     0.5962 0.8104 0.8117      0.04050       54163754    8
nprobe=32,quantizer_efSearch=32,ht=116    0.5991 0.8077 0.8088      0.04076       53964962    8
nprobe=32,quantizer_efSearch=512,ht=128   0.6182 0.8516 0.8529      0.07834       53921737    4
nprobe=64,quantizer_efSearch=64,ht=126    0.6504 0.9173 0.9194      0.08799      106568800    4
nprobe=64,quantizer_efSearch=128,ht=126   0.6509 0.9182 0.9203      0.08570      106530488    4
nprobe=128,quantizer_efSearch=512,ht=256  0.6700 0.9624 0.9652      0.10967      209465316    3
nprobe=256,quantizer_efSearch=32,ht=126   0.6718 0.9655 0.9684      0.30625      412492329    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6765 0.9747 0.9777      0.33541      410821310    1
nprobe=512,quantizer_efSearch=64,ht=128   0.6790 0.9842 0.9872      0.61200      804811333    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6807 0.9864 0.9898      1.22574     1573924254    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.6824 0.9933 0.9969      1.27308     1573924254    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6827 0.9953 0.9993      2.18387     5827132151    1
```

</details>
<details><summary>`OPQ32_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ32` </summary>
Index size 437167604

 code_size 32

 log filename: autotune.dbbigann10M.OPQ32_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ32.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2413 0.2791 0.2792      0.00443        4792385    68
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=82        0.0226 0.0267 0.0267      0.00277        2410898    109
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=112       0.1645 0.1871 0.1872      0.00287        2079038    105
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=128      0.1858 0.2175 0.2176      0.00308        1912143    98
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=120      0.1995 0.2324 0.2325      0.00341        2406951    89
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=108      0.2188 0.2561 0.2562      0.00405        3648422    75
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=120       0.2645 0.3170 0.3171      0.00431        3648837    70
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=118       0.2732 0.3392 0.3394      0.00702        7029218    43
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=124       0.3406 0.4249 0.4251      0.00689        7114163    44
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=116      0.3660 0.4484 0.4486      0.00824       12133274    37
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=120     0.3766 0.4644 0.4646      0.00971       17253344    31
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=108      0.4039 0.4954 0.4956      0.01172       16365552    26
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=128     0.4897 0.6425 0.6429      0.01387       16332870    22
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=256     0.6012 0.8364 0.8374      0.02076       59850714    15
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=256    0.6417 0.9213 0.9228      0.04658      109167174    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=256   0.6632 0.9645 0.9669      0.08300      214524104    4
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=126   0.6688 0.9826 0.9853      0.34232      415839455    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=256   0.6701 0.9893 0.9925      0.57033     1584031336    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=126    0.6714 0.9903 0.9931      0.61156      809780969    1
nprobe=2048,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=126 0.6732 0.9938 0.9974      2.53143     3086259142    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=256 0.6737 0.9962 1.0000      2.80812     6041146499    1
```

</details>
<details><summary>`OPQ32_128,IVF65536,PQ32` </summary>
Index size 434275579

 code_size 32

 log filename: autotune.dbbigann10M.OPQ32_128_IVF65536_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6766 0.9843 0.9875      0.15055      410218683    3
nprobe=2,ht=256                          0.3035 0.3640 0.3642      0.01316        3433878    23
nprobe=4,ht=124                          0.4025 0.4995 0.4998      0.01644        6851361    19
nprobe=16,ht=256                         0.5698 0.7661 0.7671      0.02070       27173052    15
nprobe=64,ht=256                         0.6496 0.9222 0.9246      0.04849      106344727    7
nprobe=128,ht=256                        0.6696 0.9642 0.9673      0.08453      209127552    4
nprobe=256,ht=256                        0.6766 0.9843 0.9875      0.15716      410218683    2
nprobe=1024,ht=120                       0.6767 0.9762 0.9792      1.14920     1572049993    1
nprobe=1024,ht=122                       0.6778 0.9842 0.9875      1.15862     1572049993    1
nprobe=1024,ht=126                       0.6803 0.9923 0.9959      1.18915     1572049993    1
nprobe=4096,ht=256                       0.6810 0.9962 1.0000      2.12830     6032225864    1
```

</details>
<details><summary>`OPQ64_128,IVF16384_HNSW32,PQ64x4fs` </summary>
Index size 421258444

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF16384_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3188 0.5450 0.5722      0.00236       63020483    128
nprobe=2,quantizer_efSearch=4            0.2372 0.3777 0.3941      0.00177       33687468    170
nprobe=4,quantizer_efSearch=4            0.2927 0.4914 0.5167      0.00209       63882862    144
nprobe=4,quantizer_efSearch=8            0.3188 0.5450 0.5722      0.00236       63020483    128
nprobe=8,quantizer_efSearch=4            0.3507 0.6325 0.6709      0.00271      120240449    111
nprobe=8,quantizer_efSearch=8            0.3599 0.6517 0.6926      0.00283      119963214    106
nprobe=8,quantizer_efSearch=16           0.3745 0.6763 0.7175      0.00342      118578022    88
nprobe=16,quantizer_efSearch=16          0.4006 0.7705 0.8266      0.00443      225622277    68
nprobe=32,quantizer_efSearch=8           0.4050 0.7980 0.8617      0.00558      427362810    54
nprobe=32,quantizer_efSearch=32          0.4227 0.8446 0.9125      0.00734      424052288    41
nprobe=32,quantizer_efSearch=64          0.4228 0.8467 0.9149      0.00865      422438417    35
nprobe=64,quantizer_efSearch=16          0.4256 0.8599 0.9349      0.00948      798998459    32
nprobe=64,quantizer_efSearch=32          0.4320 0.8772 0.9572      0.01091      796668918    28
nprobe=64,quantizer_efSearch=64          0.4345 0.8821 0.9624      0.01119      793159915    27
nprobe=128,quantizer_efSearch=128        0.4346 0.8969 0.9847      0.01787     1478438435    17
nprobe=256,quantizer_efSearch=64         0.4365 0.9010 0.9920      0.02493     2756526968    13
nprobe=256,quantizer_efSearch=256        0.4367 0.9039 0.9954      0.03294     2745751603    10
nprobe=256,quantizer_efSearch=128        0.4370 0.9033 0.9947      0.02796     2750527308    11
```

</details>
<details><summary>`OPQ64_128,IVF16384_HNSW32,PQ64x4fsr` </summary>
Index size 421345484

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF16384_HNSW32_PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5793 0.8403 0.8439      0.01769      111843193    17
nprobe=1,quantizer_efSearch=8            0.2530 0.3077 0.3084      0.00170        7294315    177
nprobe=1,quantizer_efSearch=16           0.2558 0.3109 0.3116      0.00213        7284079    141
nprobe=2,quantizer_efSearch=4            0.3316 0.4269 0.4280      0.00210       14524815    143
nprobe=2,quantizer_efSearch=8            0.3513 0.4524 0.4535      0.00227       14526977    133
nprobe=2,quantizer_efSearch=16           0.3567 0.4590 0.4601      0.00267       14507868    113
nprobe=4,quantizer_efSearch=4            0.4141 0.5560 0.5577      0.00301       28823697    100
nprobe=4,quantizer_efSearch=8            0.4428 0.5953 0.5970      0.00312       28856917    97
nprobe=4,quantizer_efSearch=16           0.4506 0.6057 0.6074      0.00382       28798299    79
nprobe=4,quantizer_efSearch=32           0.4518 0.6066 0.6083      0.00442       28776070    68
nprobe=4,quantizer_efSearch=64           0.4519 0.6068 0.6085      0.00575       28771024    53
nprobe=8,quantizer_efSearch=8            0.5166 0.7224 0.7247      0.00519       57009602    58
nprobe=8,quantizer_efSearch=16           0.5252 0.7356 0.7379      0.00556       56879848    54
nprobe=8,quantizer_efSearch=32           0.5264 0.7363 0.7386      0.00654       56820272    46
nprobe=16,quantizer_efSearch=16          0.5759 0.8370 0.8405      0.01693      111981559    18
nprobe=16,quantizer_efSearch=32          0.5793 0.8403 0.8439      0.01749      111843193    18
nprobe=32,quantizer_efSearch=8           0.5986 0.8856 0.8896      0.03060      220416259    10
nprobe=32,quantizer_efSearch=16          0.6139 0.9109 0.9153      0.03106      219937519    10
nprobe=32,quantizer_efSearch=32          0.6158 0.9157 0.9201      0.03216      219623487    10
nprobe=64,quantizer_efSearch=16          0.6286 0.9472 0.9533      0.07221      430492278    5
nprobe=64,quantizer_efSearch=64          0.6323 0.9600 0.9666      0.07341      429127987    5
nprobe=128,quantizer_efSearch=256        0.6430 0.9800 0.9876      0.14255      835638723    3
nprobe=256,quantizer_efSearch=128        0.6443 0.9864 0.9962      0.22499     1625883891    2
nprobe=256,quantizer_efSearch=256        0.6444 0.9866 0.9964      0.25853     1625033546    2
nprobe=256,quantizer_efSearch=512        0.6446 0.9868 0.9964      0.27754     1624802817    2
nprobe=512,quantizer_efSearch=128        0.6454 0.9895 0.9988      0.48023     3165967426    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fs` </summary>
Index size 737833164

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF262144_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2451 0.3763 0.3834      0.00290        1736459    104
nprobe=2,quantizer_efSearch=4            0.1713 0.2467 0.2500      0.00177         869922    170
nprobe=4,quantizer_efSearch=4            0.2235 0.3425 0.3493      0.00199        1742452    152
nprobe=4,quantizer_efSearch=8            0.2451 0.3763 0.3834      0.00290        1736459    104
nprobe=8,quantizer_efSearch=4            0.2905 0.4786 0.4929      0.00281        3475902    107
nprobe=8,quantizer_efSearch=8            0.2977 0.4914 0.5057      0.00321        3471817    94
nprobe=16,quantizer_efSearch=4           0.3253 0.5728 0.5955      0.00351        6944619    86
nprobe=16,quantizer_efSearch=8           0.3427 0.6066 0.6315      0.00447        6934213    68
nprobe=32,quantizer_efSearch=4           0.3458 0.6322 0.6637      0.00432       13796036    70
nprobe=16,quantizer_efSearch=16          0.3490 0.6177 0.6431      0.00531        6922585    57
nprobe=32,quantizer_efSearch=8           0.3714 0.6870 0.7228      0.00543       13800447    56
nprobe=32,quantizer_efSearch=16          0.3822 0.7118 0.7507      0.00719       13769968    42
nprobe=64,quantizer_efSearch=8           0.3875 0.7361 0.7790      0.00675       27347445    45
nprobe=64,quantizer_efSearch=16          0.4014 0.7753 0.8246      0.00804       27316405    38
nprobe=128,quantizer_efSearch=16         0.4136 0.8097 0.8678      0.01081       54019616    28
nprobe=128,quantizer_efSearch=32         0.4232 0.8390 0.9013      0.01447       53921039    21
nprobe=256,quantizer_efSearch=32         0.4293 0.8584 0.9283      0.01566      106337908    20
nprobe=256,quantizer_efSearch=64         0.4338 0.8759 0.9501      0.02376      106153034    13
nprobe=512,quantizer_efSearch=64         0.4350 0.8861 0.9652      0.03021      208433355    10
nprobe=256,quantizer_efSearch=128        0.4354 0.8822 0.9573      0.03500      105911058    9
nprobe=1024,quantizer_efSearch=64        0.4362 0.8885 0.9695      0.04498      389772157    7
nprobe=512,quantizer_efSearch=128        0.4369 0.8941 0.9760      0.04393      208396827    7
nprobe=1024,quantizer_efSearch=128       0.4386 0.8984 0.9831      0.05982      406627023    6
nprobe=1024,quantizer_efSearch=256       0.4391 0.9016 0.9877      0.08127      408101551    4
nprobe=1024,quantizer_efSearch=512       0.4396 0.9024 0.9890      0.12701      407577731    3
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fsr` </summary>
Index size 737734860

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF262144_HNSW32_PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5001 0.6518 0.6530      0.01833        6897745    17
nprobe=2,quantizer_efSearch=4            0.2125 0.2504 0.2506      0.00237         868348    127
nprobe=4,quantizer_efSearch=4            0.2864 0.3479 0.3484      0.00347        1739198    87
nprobe=4,quantizer_efSearch=8            0.3162 0.3847 0.3852      0.00432        1733154    70
nprobe=8,quantizer_efSearch=4            0.3939 0.4935 0.4944      0.00546        3467767    55
nprobe=8,quantizer_efSearch=8            0.4024 0.5054 0.5063      0.00593        3463743    51
nprobe=8,quantizer_efSearch=16           0.4151 0.5214 0.5223      0.00731        3454997    42
nprobe=8,quantizer_efSearch=32           0.4193 0.5259 0.5268      0.01008        3450930    30
nprobe=16,quantizer_efSearch=4           0.4612 0.5962 0.5975      0.01596        6929402    19
nprobe=16,quantizer_efSearch=16          0.4942 0.6446 0.6459      0.01789        6908031    17
nprobe=16,quantizer_efSearch=32          0.5001 0.6518 0.6530      0.02110        6897745    15
nprobe=16,quantizer_efSearch=64          0.5021 0.6532 0.6544      0.02348        6892092    13
nprobe=32,quantizer_efSearch=8           0.5379 0.7233 0.7248      0.02836       13766756    11
nprobe=32,quantizer_efSearch=16          0.5544 0.7522 0.7537      0.03166       13740830    10
nprobe=32,quantizer_efSearch=32          0.5627 0.7640 0.7655      0.03457       13719415    9
nprobe=32,quantizer_efSearch=64          0.5652 0.7666 0.7681      0.03548       13705234    9
nprobe=32,quantizer_efSearch=128         0.5664 0.7681 0.7696      0.04056       13700381    8
nprobe=64,quantizer_efSearch=32          0.6041 0.8471 0.8491      0.06212       27188917    5
nprobe=64,quantizer_efSearch=64          0.6069 0.8519 0.8537      0.06717       27152025    5
nprobe=64,quantizer_efSearch=128         0.6079 0.8530 0.8548      0.08056       27137582    4
nprobe=64,quantizer_efSearch=256         0.6080 0.8532 0.8550      0.09810       27132637    4
nprobe=128,quantizer_efSearch=16         0.6121 0.8676 0.8698      0.13371       53893299    3
nprobe=128,quantizer_efSearch=32         0.6296 0.9018 0.9045      0.13297       53803438    3
nprobe=128,quantizer_efSearch=64         0.6347 0.9139 0.9165      0.13568       53681362    3
nprobe=128,quantizer_efSearch=128        0.6367 0.9156 0.9185      0.14606       53622955    3
nprobe=128,quantizer_efSearch=256        0.6372 0.9168 0.9197      0.15716       53603908    2
nprobe=256,quantizer_efSearch=256        0.6556 0.9592 0.9626      0.27918      105628877    2
nprobe=512,quantizer_efSearch=64         0.6558 0.9625 0.9678      0.49514      208036024    1
nprobe=512,quantizer_efSearch=256        0.6642 0.9764 0.9821      0.52815      207667284    1
nprobe=512,quantizer_efSearch=512        0.6643 0.9768 0.9826      0.54727      207549823    1
nprobe=1024,quantizer_efSearch=256       0.6662 0.9853 0.9910      0.97318      407546864    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 677872144

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.4008 0.7741 0.8237      0.00793       33083995    38
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1562 0.2222 0.2241      0.00133        1236266    226
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.1669 0.2383 0.2402      0.00152        1583777    198
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.1812 0.2593 0.2626      0.00153        1578536    197
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2147 0.3286 0.3354      0.00167        2114019    181
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.2254 0.3433 0.3494      0.00172        2467566    175
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2360 0.3642 0.3712      0.00179        2456686    168
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2386 0.3674 0.3743      0.00192        2451540    157
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.2400 0.3685 0.3755      0.00210        2450840    143
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.2689 0.4429 0.4549      0.00222        4240633    136
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.2833 0.4658 0.4783      0.00255        4909203    118
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2865 0.4688 0.4832      0.00249        4195760    121
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.2981 0.4915 0.5058      0.00270        4873236    111
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3161 0.5591 0.5809      0.00316        7700103    95
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3234 0.5687 0.5903      0.00327        8426749    92
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3398 0.6038 0.6276      0.00352        8348245    86
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3512 0.6207 0.6451      0.00429        9656997    70
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3520 0.6224 0.6475      0.00463        9641850    65
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3586 0.6614 0.6951      0.00469       15435906    64
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3700 0.6855 0.7208      0.00543       16704846    56
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3794 0.7251 0.7716      0.00702       29341797    43
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.3892 0.7256 0.7639      0.00719       19117654    42
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3960 0.7626 0.8119      0.00781       30534263    39
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4055 0.7850 0.8360      0.00866       30063740    35
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4112 0.7990 0.8508      0.01012       32616906    30
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4139 0.8156 0.8753      0.01170       57867593    26
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4159 0.8267 0.8896      0.01301       56790460    24
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.4236 0.8394 0.9008      0.01490       65380945    21
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4261 0.8532 0.9180      0.01635       64322596    19
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4319 0.8752 0.9501      0.02218      111549103    14
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.4337 0.8759 0.9484      0.02440      128916439    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.4357 0.8850 0.9604      0.02698      126557044    12
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.4358 0.8866 0.9622      0.03207      126455691    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4360 0.8854 0.9612      0.02905      116346013    11
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.4390 0.8930 0.9750      0.03206      223636113    10
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256   0.4391 0.8987 0.9815      0.04550      249109480    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4395 0.9027 0.9900      0.05952      418784970    6
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4396 0.9027 0.9900      0.07976      418541742    4
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.4397 0.9045 0.9920      0.22173      448298704    2
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 677845520

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.6064 0.8544 0.8563      0.06409       37402098    5
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1411 0.1571 0.1571      0.00166         791473    181
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1487 0.1663 0.1663      0.00180         788367    167
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1578 0.1767 0.1767      0.00151        1139032    199
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1579 0.1768 0.1768      0.00162        1139386    185
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1615 0.1805 0.1805      0.00178        1823523    169
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2030 0.2345 0.2345      0.00172        1231973    175
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2182 0.2531 0.2531      0.00176        1578399    171
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2249 0.2635 0.2636      0.00190        1574925    158
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2271 0.2629 0.2629      0.00200        2261042    150
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2347 0.2741 0.2741      0.00203        2257643    148
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2360 0.2758 0.2759      0.00228        2258463    132
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3063 0.3725 0.3728      0.00262        2449663    115
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3239 0.3925 0.3928      0.00329        3128940    92
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3275 0.3979 0.3982      0.00385        4457068    78
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3449 0.4281 0.4290      0.00446        3858966    68
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3872 0.4836 0.4844      0.00451        4194340    67
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4178 0.5258 0.5266      0.00503        6183712    60
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4185 0.5264 0.5272      0.00521        6183667    58
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4200 0.5287 0.5295      0.00613        8814846    49
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4207 0.5293 0.5301      0.00629        8810967    48
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4467 0.5787 0.5798      0.01462        7731687    21
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4885 0.6305 0.6318      0.01515        8332468    20
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4909 0.6398 0.6409      0.01541        9687323    20
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4982 0.6483 0.6496      0.01765        9643180    17
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5035 0.6532 0.6545      0.01853       12262920    17
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5050 0.6542 0.6555      0.01879       17456878    16
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.5055 0.6544 0.6557      0.02024       27718646    15
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5509 0.7477 0.7492      0.02836       16595096    11
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5587 0.7564 0.7579      0.02951       19199593    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5645 0.7659 0.7675      0.03287       19088006    10
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5648 0.7667 0.7683      0.03256       19087565    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5666 0.7683 0.7699      0.03011       24272880    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6039 0.8515 0.8534      0.06696       32531331    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6064 0.8544 0.8563      0.07184       37591535    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6395 0.9180 0.9205      0.12771       64234376    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6399 0.9187 0.9213      0.12812       74468096    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.6402 0.9189 0.9215      0.13609       94891743    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6531 0.9604 0.9641      0.25302      146931762    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6594 0.9771 0.9821      0.48030      230818514    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6596 0.9760 0.9808      0.44252      220467343    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6634 0.9873 0.9929      0.90309      431826156    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fs` </summary>
Index size 484935372

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4377 0.8991 0.9849      0.05481      410827311    6
nprobe=1,quantizer_efSearch=4            0.1514 0.2172 0.2210      0.00118        1723996    255
nprobe=2,quantizer_efSearch=4            0.2116 0.3242 0.3329      0.00134        3452229    224
nprobe=4,quantizer_efSearch=4            0.2656 0.4346 0.4517      0.00163        6891358    184
nprobe=4,quantizer_efSearch=8            0.2902 0.4738 0.4919      0.00187        6878787    161
nprobe=8,quantizer_efSearch=4            0.3319 0.5815 0.6082      0.00216       13752405    139
nprobe=8,quantizer_efSearch=8            0.3393 0.5959 0.6236      0.00240       13741887    126
nprobe=16,quantizer_efSearch=4           0.3627 0.6644 0.7010      0.00277       27341474    109
nprobe=16,quantizer_efSearch=8           0.3806 0.6997 0.7388      0.00302       27303711    100
nprobe=16,quantizer_efSearch=16          0.3869 0.7146 0.7545      0.00361       27266130    83
nprobe=32,quantizer_efSearch=8           0.3972 0.7613 0.8138      0.00401       54165593    75
nprobe=32,quantizer_efSearch=16          0.4071 0.7877 0.8414      0.00473       54040866    64
nprobe=64,quantizer_efSearch=16          0.4179 0.8326 0.8975      0.00590      106861542    51
nprobe=128,quantizer_efSearch=16         0.4233 0.8521 0.9257      0.00845      210454910    36
nprobe=64,quantizer_efSearch=32          0.4248 0.8483 0.9160      0.00756      106673620    40
nprobe=128,quantizer_efSearch=32         0.4321 0.8757 0.9536      0.00950      210050938    32
nprobe=128,quantizer_efSearch=64         0.4343 0.8827 0.9618      0.01140      209656344    27
nprobe=256,quantizer_efSearch=32         0.4350 0.8856 0.9683      0.01527      412516993    20
nprobe=128,quantizer_efSearch=128        0.4353 0.8843 0.9634      0.01476      209499936    21
nprobe=256,quantizer_efSearch=64         0.4373 0.8964 0.9813      0.01853      411747941    17
nprobe=256,quantizer_efSearch=128        0.4377 0.8982 0.9839      0.02330      411073277    13
nprobe=256,quantizer_efSearch=256        0.4378 0.8988 0.9845      0.03335      410872315    9
nprobe=512,quantizer_efSearch=128        0.4387 0.9018 0.9918      0.03485      805897029    9
nprobe=1024,quantizer_efSearch=128       0.4388 0.9020 0.9938      0.04411     1563256348    7
nprobe=512,quantizer_efSearch=512        0.4392 0.9031 0.9934      0.06007      804623064    6
nprobe=2048,quantizer_efSearch=256       0.4393 0.9034 0.9957      0.08679     3024767014    4
nprobe=1024,quantizer_efSearch=512       0.4397 0.9044 0.9966      0.09371     1573961864    4
nprobe=2048,quantizer_efSearch=512       0.4405 0.9050 0.9975      0.13620     3079822782    3
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fsr` </summary>
Index size 484985548

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_HNSW32_PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5558 0.7629 0.7645      0.01653       27193409    19
nprobe=2,quantizer_efSearch=4            0.2731 0.3321 0.3322      0.00184        3447698    164
nprobe=2,quantizer_efSearch=8            0.2931 0.3572 0.3573      0.00211        3442437    142
nprobe=2,quantizer_efSearch=16           0.2980 0.3626 0.3627      0.00266        3437660    113
nprobe=4,quantizer_efSearch=4            0.3540 0.4501 0.4510      0.00270        6882383    112
nprobe=4,quantizer_efSearch=16           0.3915 0.4995 0.5006      0.00360        6857866    84
nprobe=8,quantizer_efSearch=8            0.4719 0.6228 0.6242      0.00458       13721686    66
nprobe=8,quantizer_efSearch=16           0.4831 0.6386 0.6399      0.00510       13703691    59
nprobe=8,quantizer_efSearch=32           0.4852 0.6423 0.6435      0.00604       13689586    50
nprobe=8,quantizer_efSearch=64           0.4863 0.6430 0.6442      0.00808       13687430    38
nprobe=8,quantizer_efSearch=128          0.4866 0.6432 0.6444      0.01145       13686498    27
nprobe=16,quantizer_efSearch=8           0.5408 0.7392 0.7407      0.01549       27263796    20
nprobe=16,quantizer_efSearch=16          0.5500 0.7546 0.7562      0.01576       27228538    20
nprobe=16,quantizer_efSearch=32          0.5558 0.7629 0.7645      0.01613       27193409    19
nprobe=16,quantizer_efSearch=64          0.5561 0.7639 0.7655      0.01910       27184290    16
nprobe=16,quantizer_efSearch=128         0.5565 0.7645 0.7661      0.02148       27181067    15
nprobe=32,quantizer_efSearch=8           0.5766 0.8134 0.8159      0.02558       54075325    12
nprobe=32,quantizer_efSearch=16          0.5927 0.8411 0.8437      0.02661       53959066    12
nprobe=32,quantizer_efSearch=64          0.5998 0.8538 0.8562      0.03451       53863159    9
nprobe=64,quantizer_efSearch=64          0.6310 0.9189 0.9230      0.07405      106424220    5
nprobe=64,quantizer_efSearch=128         0.6312 0.9197 0.9237      0.07702      106389027    4
nprobe=64,quantizer_efSearch=256         0.6313 0.9198 0.9238      0.08515      106381997    4
nprobe=64,quantizer_efSearch=512         0.6314 0.9199 0.9239      0.10716      106379472    3
nprobe=128,quantizer_efSearch=32         0.6464 0.9515 0.9566      0.12641      209749246    3
nprobe=128,quantizer_efSearch=64         0.6489 0.9592 0.9647      0.12761      209377442    3
nprobe=128,quantizer_efSearch=256        0.6490 0.9609 0.9663      0.13822      209208028    3
nprobe=256,quantizer_efSearch=128        0.6508 0.9797 0.9862      0.25662      410598916    2
nprobe=256,quantizer_efSearch=256        0.6515 0.9805 0.9868      0.25784      410407367    2
nprobe=256,quantizer_efSearch=512        0.6516 0.9807 0.9870      0.27966      410364671    2
nprobe=512,quantizer_efSearch=512        0.6533 0.9884 0.9955      0.51286      803898145    1
nprobe=512,quantizer_efSearch=256        0.6538 0.9880 0.9950      0.50256      804141357    1
nprobe=1024,quantizer_efSearch=256       0.6551 0.9904 0.9978      0.88903     1574076381    1
```

</details>
<details><summary>`OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 470085136

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4306 0.8752 0.9520      0.01124      216541690    27
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1317 0.1851 0.1881      0.00138        1833855    217
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1662 0.2359 0.2396      0.00143        2410329    211
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1677 0.2474 0.2532      0.00153        3600213    196
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2045 0.3083 0.3149      0.00156        3830599    193
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2050 0.3095 0.3183      0.00159        3659073    189
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2197 0.3374 0.3462      0.00168        3811100    179
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2203 0.3380 0.3468      0.00167        3810698    180
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2287 0.3500 0.3587      0.00167        4132878    180
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2764 0.4456 0.4607      0.00203        7646293    148
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2865 0.4709 0.4889      0.00194        7587264    155
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2917 0.4757 0.4939      0.00217        7569146    139
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3123 0.5370 0.5624      0.00232       14316242    130
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3207 0.5597 0.5865      0.00236       14177443    128
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3212 0.5615 0.5881      0.00238       14157809    127
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3280 0.5694 0.5949      0.00237       14593200    127
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3399 0.5984 0.6260      0.00253       14464147    119
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3425 0.6007 0.6277      0.00257       14441041    117
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3457 0.6106 0.6382      0.00277       15083023    109
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3666 0.6742 0.7123      0.00301       28426932    100
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3773 0.6969 0.7363      0.00331       28026049    91
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3986 0.7627 0.8153      0.00404       55009125    75
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4054 0.7741 0.8254      0.00463       56307955    65
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4110 0.7921 0.8461      0.00515       55383620    59
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4142 0.8009 0.8551      0.00614       56562339    49
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4190 0.8312 0.8953      0.00657      110163281    46
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4221 0.8402 0.9068      0.00756      108245727    41
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.4240 0.8423 0.9083      0.00812      111148487    37
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4263 0.8532 0.9216      0.00901      109159953    34
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4266 0.8533 0.9222      0.00966      111651314    32
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4279 0.8674 0.9437      0.00974      211762477    31
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4306 0.8752 0.9520      0.01064      216538777    29
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4345 0.8839 0.9631      0.01114      212366428    27
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.4348 0.8845 0.9640      0.01380      212205531    22
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4356 0.8861 0.9657      0.01540      219622264    20
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4370 0.8952 0.9801      0.01611      424583347    19
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4377 0.8999 0.9856      0.01783      416036586    17
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4379 0.9002 0.9860      0.02197      421035930    14
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4382 0.9006 0.9863      0.02420      420820968    13
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.4390 0.9024 0.9921      0.02461      831283036    13
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4391 0.9033 0.9936      0.03079      814496836    10
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4393 0.9037 0.9941      0.03309      814264772    10
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4395 0.9048 0.9972      0.04224     1582935825    8
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4402 0.9055 0.9978      0.07870     3086506444    4
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.4403 0.9058 0.9981      0.11446     6157679911    3
```

</details>
<details><summary>`OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 470086160

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5549 0.7636 0.7657      0.02436       37485963    13
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4     0.1976 0.2316 0.2316      0.00162        2075398    186
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.1989 0.2323 0.2323      0.00173        3071438    174
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2048 0.2400 0.2400      0.00177        3063577    170
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.2259 0.2737 0.2739      0.00201        3567062    149
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2779 0.3378 0.3381      0.00196        3822807    153
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2851 0.3465 0.3467      0.00199        3809830    151
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2953 0.3592 0.3594      0.00205        4132084    147
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.2955 0.3593 0.3595      0.00225        4131028    134
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2972 0.3613 0.3615      0.00222        4783147    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.2985 0.3627 0.3629      0.00238        4785338    127
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3359 0.4246 0.4255      0.00288        7119438    105
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.3369 0.4258 0.4267      0.00312        7114192    97
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3883 0.4924 0.4935      0.00308        7567202    98
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3926 0.4978 0.4989      0.00328        8215372    92
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3948 0.5007 0.5018      0.00337        8206675    90
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4350 0.5771 0.5783      0.00464       14238862    65
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4443 0.5883 0.5895      0.00495       14155682    61
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4820 0.6397 0.6408      0.00513       15057008    59
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4821 0.6391 0.6404      0.00524       15057851    58
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4846 0.6424 0.6437      0.00558       16328607    54
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4894 0.6697 0.6715      0.01434       27999561    21
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5501 0.7575 0.7596      0.01535       28590912    20
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5544 0.7633 0.7654      0.01755       32410657    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5549 0.7637 0.7658      0.01709       32405266    18
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.5552 0.7641 0.7662      0.01835       29845516    17
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5690 0.8067 0.8095      0.02759       55324042    11
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5921 0.8373 0.8403      0.02663       55777526    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5957 0.8436 0.8466      0.02789       55372065    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5962 0.8440 0.8470      0.02773       55383110    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6022 0.8530 0.8561      0.02881       56551832    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6336 0.9196 0.9241      0.06815      116706908    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.6355 0.9382 0.9430      0.12055      213265952    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.6405 0.9408 0.9455      0.12346      211626973    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.6512 0.9620 0.9669      0.13562      214522574    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6514 0.9622 0.9671      0.13125      219620633    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6544 0.9787 0.9841      0.24705      414042584    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6556 0.9812 0.9864      0.24838      423974244    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6557 0.9823 0.9876      0.24946      415831883    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6565 0.9898 0.9959      0.46833      814267026    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6582 0.9914 0.9987      0.89374     1593753889    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.6583 0.9917 0.9985      0.89244     1579517717    1
```

</details>
<details><summary>`OPQ64_128,IVF65536,PQ64x4fs` </summary>
Index size 467182871

 code_size 32

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4380 0.8997 0.9858      0.02566      410218047    12
nprobe=64                                0.4279 0.8548 0.9233      0.01669      106344849    19
nprobe=128                               0.4361 0.8865 0.9658      0.01871      209127381    17
nprobe=256                               0.4380 0.8997 0.9858      0.02654      410218047    12
nprobe=512                               0.4394 0.9039 0.9943      0.03257      803574357    10
nprobe=1024                              0.4399 0.9047 0.9972      0.04765     1572050768    7
nprobe=2048                              0.4405 0.9056 0.9978      0.07536     3076343802    4
```

</details>
<details><summary>`PCAR16,IVF16384_HNSW32,SQfp16` </summary>
Index size 405715221

 code_size 32

 log filename: autotune.dbbigann10M.PCAR16_IVF16384_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0787 0.2732 0.4541      0.00287       33325374    105
nprobe=2,quantizer_efSearch=4            0.0694 0.2180 0.3236      0.00204       17079594    148
nprobe=2,quantizer_efSearch=8            0.0712 0.2248 0.3361      0.00218       17005749    138
nprobe=4,quantizer_efSearch=4            0.0757 0.2595 0.4297      0.00270       33414065    112
nprobe=4,quantizer_efSearch=8            0.0787 0.2732 0.4541      0.00281       33325374    107
nprobe=4,quantizer_efSearch=16           0.0794 0.2756 0.4605      0.00308       33243679    98
nprobe=4,quantizer_efSearch=32           0.0795 0.2759 0.4611      0.00355       33214605    85
nprobe=8,quantizer_efSearch=8            0.0829 0.3083 0.5645      0.00436       65099164    69
nprobe=8,quantizer_efSearch=4            0.0833 0.3065 0.5570      0.00429       65053557    70
nprobe=8,quantizer_efSearch=16           0.0837 0.3120 0.5728      0.00462       64982468    65
nprobe=8,quantizer_efSearch=32           0.0838 0.3122 0.5738      0.00510       64941947    59
nprobe=16,quantizer_efSearch=8           0.0853 0.3301 0.6423      0.00814      127125194    37
nprobe=16,quantizer_efSearch=32          0.0859 0.3348 0.6533      0.00877      126871631    35
nprobe=16,quantizer_efSearch=16          0.0861 0.3344 0.6515      0.00797      126975181    38
nprobe=32,quantizer_efSearch=16          0.0867 0.3438 0.6914      0.01434      248511188    21
nprobe=32,quantizer_efSearch=32          0.0872 0.3446 0.6937      0.01537      248275174    20
nprobe=64,quantizer_efSearch=64          0.0875 0.3481 0.7145      0.02702      484666443    12
nprobe=128,quantizer_efSearch=32         0.0876 0.3489 0.7191      0.05028      947780459    6
```

</details>
<details><summary>`PCAR16,IVF262144_HNSW32,SQfp16` </summary>
Index size 490292501

 code_size 32

 log filename: autotune.dbbigann10M.PCAR16_IVF262144_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0702 0.2170 0.2804      0.00208        1727513    145
nprobe=1,quantizer_efSearch=4            0.0424 0.1020 0.1103      0.00176         434833    171
nprobe=2,quantizer_efSearch=4            0.0571 0.1536 0.1780      0.00183         867958    165
nprobe=4,quantizer_efSearch=4            0.0676 0.2038 0.2629      0.00188        1725158    160
nprobe=4,quantizer_efSearch=8            0.0702 0.2170 0.2804      0.00208        1727513    145
nprobe=8,quantizer_efSearch=4            0.0772 0.2592 0.3772      0.00211        3428880    143
nprobe=8,quantizer_efSearch=8            0.0782 0.2624 0.3838      0.00220        3428764    137
nprobe=16,quantizer_efSearch=4           0.0805 0.2884 0.4672      0.00231        6793878    130
nprobe=16,quantizer_efSearch=8           0.0833 0.2979 0.4872      0.00243        6798037    124
nprobe=16,quantizer_efSearch=16          0.0841 0.3016 0.4934      0.00267        6795162    113
nprobe=32,quantizer_efSearch=8           0.0862 0.3176 0.5687      0.00304       13433466    99
nprobe=32,quantizer_efSearch=16          0.0869 0.3239 0.5842      0.00335       13426182    90
nprobe=64,quantizer_efSearch=32          0.0886 0.3378 0.6503      0.00578       26480098    52
```

</details>
<details><summary>`PCAR16,IVF262144(IVF512,PQ8x4fs,RFlat),SQfp16` </summary>
Index size 422167129

 code_size 32

 log filename: autotune.dbbigann10M.PCAR16_IVF262144_IVF512_PQ8x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0777 0.2658 0.4091      0.00260        8351790    116
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8   0.0460 0.1073 0.1161      0.00220        1811397    137
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1    0.0514 0.1287 0.1454      0.00221        1063847    136
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1   0.0536 0.1323 0.1497      0.00224        1055817    134
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0547 0.1479 0.1702      0.00224        1585481    134
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1    0.0587 0.1598 0.1960      0.00236        1943501    127
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0616 0.1792 0.2229      0.00228        2127354    132
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.0647 0.1888 0.2353      0.00229        2466912    132
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.0666 0.1971 0.2454      0.00230        2116376    131
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0700 0.2093 0.2643      0.00232        2458239    130
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0717 0.2146 0.2705      0.00239        2440442    126
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.0758 0.2441 0.3449      0.00243        4217641    124
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0776 0.2596 0.3690      0.00256        4157461    118
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.0777 0.2658 0.4091      0.00264        8352122    114
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.0785 0.2661 0.4091      0.00279        9660085    108
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0810 0.2878 0.4588      0.00287        7585579    105
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.0824 0.2933 0.4666      0.00306        9631216    99
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.0825 0.3003 0.5154      0.00354       15170818    85
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.0826 0.2938 0.4654      0.00346       12238164    87
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0835 0.3022 0.4887      0.00339        9581388    89
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.0853 0.3157 0.5562      0.00382       15065318    79
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.0872 0.3250 0.5798      0.00502       18874992    60
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.0880 0.3339 0.6373      0.00591       29626961    51
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.0882 0.3382 0.6483      0.00899       37201014    34
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=16 0.0885 0.3373 0.6439      0.00886       29304713    34
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0888 0.3403 0.6847      0.01038       55565018    29
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0889 0.3456 0.7141      0.01879      108788766    16
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=16 0.0891 0.3431 0.7049      0.02098      105775975    15
```

</details>
<details><summary>`PCAR16,IVF65536_HNSW32,SQfp16` </summary>
Index size 422631189

 code_size 32

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0750 0.2542 0.3806      0.00201        6869648    150
nprobe=2,quantizer_efSearch=4            0.0650 0.1968 0.2579      0.00179        3453574    168
nprobe=4,quantizer_efSearch=4            0.0715 0.2423 0.3595      0.00191        6857740    157
nprobe=4,quantizer_efSearch=8            0.0750 0.2542 0.3806      0.00206        6869648    146
nprobe=8,quantizer_efSearch=4            0.0807 0.2891 0.4812      0.00204       13605846    147
nprobe=16,quantizer_efSearch=4           0.0848 0.3108 0.5636      0.00266       26892941    113
nprobe=32,quantizer_efSearch=4           0.0868 0.3225 0.6152      0.00424       53000649    71
nprobe=32,quantizer_efSearch=8           0.0877 0.3334 0.6451      0.00445       53045336    68
nprobe=32,quantizer_efSearch=32          0.0879 0.3369 0.6611      0.00487       52980105    62
nprobe=32,quantizer_efSearch=64          0.0880 0.3370 0.6615      0.00585       52975961    52
nprobe=64,quantizer_efSearch=8           0.0883 0.3379 0.6748      0.00756      104286734    40
nprobe=64,quantizer_efSearch=32          0.0885 0.3433 0.6991      0.00805      104156739    38
nprobe=64,quantizer_efSearch=64          0.0888 0.3438 0.7009      0.00831      104131543    37
```

</details>
<details><summary>`PCAR16,IVF65536(IVF256,PQ8x4fs,RFlat),SQfp16` </summary>
Index size 405617241

 code_size 32

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_IVF256_PQ8x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.0841 0.3120 0.5890      0.00702      108412628    43
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0526 0.1431 0.1719      0.00236        3084513    128
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.0546 0.1461 0.1756      0.00238        2428427    127
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4    0.0547 0.1440 0.1729      0.00236        2090129    128
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.0641 0.1965 0.2628      0.00247        4815605    122
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.0660 0.1999 0.2644      0.00244        3829486    123
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.0716 0.2349 0.3536      0.00252        8264209    119
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.0756 0.2470 0.3664      0.00262        7275960    115
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.0799 0.2860 0.4727      0.00264       14432323    114
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.0807 0.2908 0.4862      0.00261       14373623    116
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0825 0.2949 0.4926      0.00314       14977976    96
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.0864 0.3204 0.5796      0.00364       29743717    83
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.0865 0.3205 0.5796      0.00414       32297719    73
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.0879 0.3350 0.6520      0.00503       54729835    60
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.0880 0.3366 0.6538      0.00578       58546202    52
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.0881 0.3364 0.6585      0.00715       54432615    42
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.0883 0.3372 0.6601      0.00572       55779962    53
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.0887 0.3399 0.6813      0.00795      108066530    38
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.0889 0.3442 0.6992      0.00973      109864157    31
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=32 0.0890 0.3464 0.7177      0.03502      207056156    9
```

</details>
<details><summary>`PCAR16,IVF65536,SQfp16` </summary>
Index size 404793696

 code_size 32

 log filename: autotune.dbbigann10M.PCAR16_IVF65536_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0761 0.2574 0.3868      0.00600        6861789    50
nprobe=2                                 0.0667 0.2057 0.2747      0.00540        3455634    56
nprobe=8                                 0.0812 0.2968 0.4983      0.00632       13600536    48
nprobe=16                                0.0850 0.3230 0.5918      0.00693       26865185    44
nprobe=32                                0.0880 0.3372 0.6622      0.00796       52948423    38
nprobe=64                                0.0890 0.3439 0.7019      0.01119      104069128    27
```

</details>
<details><summary>`PCAR32,IVF16384_HNSW32,SQ8` </summary>
Index size 406772309

 code_size 32

 log filename: autotune.dbbigann10M.PCAR32_IVF16384_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2354 0.4989 0.5366      0.00633       40134707    48
nprobe=1,quantizer_efSearch=4            0.1459 0.2491 0.2553      0.00277       10675785    109
nprobe=1,quantizer_efSearch=8            0.1536 0.2629 0.2696      0.00301       10552872    100
nprobe=1,quantizer_efSearch=16           0.1547 0.2649 0.2717      0.00341       10509742    88
nprobe=2,quantizer_efSearch=4            0.1890 0.3550 0.3707      0.00392       20870360    77
nprobe=2,quantizer_efSearch=8            0.1995 0.3787 0.3951      0.00413       20622376    73
nprobe=2,quantizer_efSearch=16           0.2019 0.3834 0.4001      0.00459       20509319    66
nprobe=2,quantizer_efSearch=32           0.2026 0.3844 0.4011      0.00534       20466708    57
nprobe=4,quantizer_efSearch=4            0.2229 0.4641 0.4972      0.00610       40410098    50
nprobe=4,quantizer_efSearch=8            0.2354 0.4989 0.5366      0.00632       40134707    48
nprobe=4,quantizer_efSearch=16           0.2378 0.5067 0.5449      0.00678       39893948    45
nprobe=4,quantizer_efSearch=32           0.2381 0.5083 0.5466      0.00752       39825696    40
nprobe=8,quantizer_efSearch=4            0.2633 0.5932 0.6556      0.01026       77605829    30
nprobe=8,quantizer_efSearch=8            0.2664 0.6063 0.6721      0.01036       77456166    29
nprobe=8,quantizer_efSearch=16           0.2719 0.6214 0.6908      0.01074       77052409    28
nprobe=8,quantizer_efSearch=32           0.2732 0.6241 0.6936      0.01151       76894915    27
nprobe=16,quantizer_efSearch=4           0.2815 0.6597 0.7495      0.01798      149121468    17
nprobe=16,quantizer_efSearch=8           0.2937 0.6903 0.7895      0.01818      149011254    17
nprobe=16,quantizer_efSearch=16          0.2976 0.7017 0.8051      0.01844      148705441    17
nprobe=16,quantizer_efSearch=32          0.2980 0.7043 0.8088      0.01922      148293915    16
nprobe=16,quantizer_efSearch=128         0.2981 0.7046 0.8087      0.02378      148163239    13
nprobe=32,quantizer_efSearch=8           0.3020 0.7318 0.8608      0.03287      287051913    10
nprobe=32,quantizer_efSearch=16          0.3081 0.7561 0.8947      0.03320      286366144    10
nprobe=32,quantizer_efSearch=32          0.3100 0.7601 0.9009      0.03374      285812567    9
nprobe=32,quantizer_efSearch=64          0.3102 0.7609 0.9016      0.03521      285410875    9
nprobe=64,quantizer_efSearch=16          0.3111 0.7786 0.9364      0.06080      551209608    5
nprobe=64,quantizer_efSearch=32          0.3146 0.7869 0.9507      0.06127      550267927    5
nprobe=64,quantizer_efSearch=64          0.3149 0.7881 0.9524      0.06235      549395039    5
nprobe=128,quantizer_efSearch=64         0.3158 0.7973 0.9761      0.11583     1055098207    3
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ8` </summary>
Index size 507078229

 code_size 32

 log filename: autotune.dbbigann10M.PCAR32_IVF262144_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3103 0.7875 0.9430      0.05486      105208943    6
nprobe=1,quantizer_efSearch=4            0.1047 0.1476 0.1482      0.00184         437080    163
nprobe=2,quantizer_efSearch=4            0.1472 0.2303 0.2320      0.00190         874434    158
nprobe=4,quantizer_efSearch=4            0.1815 0.3186 0.3248      0.00197        1747007    153
nprobe=4,quantizer_efSearch=8            0.1989 0.3511 0.3578      0.00220        1745044    137
nprobe=8,quantizer_efSearch=8            0.2376 0.4592 0.4762      0.00237        3481842    127
nprobe=16,quantizer_efSearch=4           0.2569 0.5371 0.5720      0.00262        6947382    115
nprobe=16,quantizer_efSearch=8           0.2651 0.5596 0.5972      0.00301        6938767    100
nprobe=16,quantizer_efSearch=16          0.2671 0.5667 0.6058      0.00305        6930194    99
nprobe=32,quantizer_efSearch=4           0.2690 0.5935 0.6492      0.00413       13777523    73
nprobe=32,quantizer_efSearch=8           0.2785 0.6345 0.6972      0.00401       13781689    75
nprobe=32,quantizer_efSearch=32          0.2860 0.6612 0.7291      0.00537       13735279    57
nprobe=32,quantizer_efSearch=64          0.2873 0.6626 0.7307      0.00725       13724187    42
nprobe=64,quantizer_efSearch=16          0.2942 0.7088 0.8056      0.00760       27231539    40
nprobe=64,quantizer_efSearch=32          0.2991 0.7216 0.8234      0.00752       27175318    40
nprobe=64,quantizer_efSearch=64          0.2998 0.7245 0.8269      0.00892       27144288    34
nprobe=64,quantizer_efSearch=128         0.3000 0.7255 0.8277      0.01224       27129251    25
nprobe=128,quantizer_efSearch=64         0.3069 0.7629 0.8973      0.01612       53566408    19
nprobe=128,quantizer_efSearch=128        0.3075 0.7648 0.8999      0.01919       53510617    16
nprobe=256,quantizer_efSearch=64         0.3085 0.7833 0.9362      0.02320      105520166    13
nprobe=256,quantizer_efSearch=128        0.3097 0.7862 0.9412      0.02794      105299500    11
nprobe=256,quantizer_efSearch=256        0.3104 0.7873 0.9427      0.03253      105224604    10
nprobe=512,quantizer_efSearch=128        0.3105 0.7968 0.9652      0.05137      206886199    6
nprobe=512,quantizer_efSearch=256        0.3109 0.7984 0.9684      0.06079      206521137    5
nprobe=1024,quantizer_efSearch=128       0.3111 0.7999 0.9747      0.09459      402900584    4
nprobe=1024,quantizer_efSearch=256       0.3117 0.8026 0.9795      0.11013      404683530    3
nprobe=2048,quantizer_efSearch=256       0.3118 0.8030 0.9836      0.17917      779676625    2
nprobe=2048,quantizer_efSearch=512       0.3119 0.8043 0.9862      0.23232      790413587    2
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ8` </summary>
Index size 440067993

 code_size 32

 log filename: autotune.dbbigann10M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.3096 0.7930 0.9618      0.08726      425212645    4
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.1000 0.1497 0.1506      0.00222        1246042    136
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1114 0.1578 0.1582      0.00222        1823956    135
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1429 0.2283 0.2302      0.00223        2265914    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1475 0.2328 0.2349      0.00223        1582986    135
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1742 0.3027 0.3078      0.00231        2122188    131
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1756 0.3051 0.3097      0.00239        4484306    126
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1847 0.3228 0.3286      0.00228        2469294    132
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1927 0.3359 0.3420      0.00241        2465107    125
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2229 0.4251 0.4405      0.00256        4224119    118
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.2255 0.4363 0.4523      0.00267        4206121    113
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.2290 0.4516 0.4723      0.00257        8461211    117
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2475 0.5189 0.5506      0.00292        8404221    103
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2527 0.5329 0.5652      0.00300        9724084    101
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2657 0.5610 0.5974      0.00327        9683313    92
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2658 0.5636 0.5999      0.00366       12296074    83
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2734 0.6056 0.6611      0.00444       15314243    68
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2774 0.6281 0.6881      0.00460       15242857    66
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2808 0.6316 0.6894      0.00487       19204710    62
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2863 0.6589 0.7230      0.00625       24311391    49
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2887 0.6653 0.7321      0.00617       19094931    49
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2899 0.6657 0.7333      0.00710       24264155    43
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2938 0.6991 0.7885      0.00721       30180197    42
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2967 0.7158 0.8120      0.00772       30023485    39
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3000 0.7239 0.8237      0.00822       32587987    37
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3011 0.7261 0.8254      0.00966       37734221    32
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3063 0.7574 0.8836      0.01393       64493769    22
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3079 0.7667 0.8996      0.01696       64165268    18
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.3080 0.7673 0.9021      0.02055       94703760    15
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.3083 0.7674 0.9021      0.02185       64068917    14
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.3090 0.7836 0.9373      0.02343      110958819    13
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3095 0.7877 0.9430      0.02802      126032414    11
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.3096 0.7876 0.9429      0.02858      146454355    11
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.3101 0.7882 0.9436      0.03594      115853789    9
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.3102 0.7975 0.9663      0.05035      248875183    6
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3110 0.7998 0.9712      0.04970      217338677    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.3115 0.8035 0.9809      0.08307      417071188    4
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ8` </summary>
Index size 426834005

 code_size 32

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3125 0.8001 0.9746      0.06387      408653394    5
nprobe=1,quantizer_efSearch=4            0.1355 0.2156 0.2178      0.00181        1738886    166
nprobe=4,quantizer_efSearch=4            0.2149 0.4189 0.4353      0.00189        6944443    160
nprobe=4,quantizer_efSearch=8            0.2257 0.4481 0.4667      0.00206        6942035    146
nprobe=4,quantizer_efSearch=16           0.2278 0.4525 0.4715      0.00238        6929692    127
nprobe=8,quantizer_efSearch=4            0.2534 0.5416 0.5805      0.00247       13817737    122
nprobe=8,quantizer_efSearch=8            0.2572 0.5533 0.5938      0.00248       13803237    121
nprobe=8,quantizer_efSearch=16           0.2616 0.5628 0.6054      0.00304       13784128    99
nprobe=8,quantizer_efSearch=32           0.2632 0.5650 0.6076      0.00333       13772308    91
nprobe=16,quantizer_efSearch=4           0.2735 0.6194 0.6835      0.00362       27361437    83
nprobe=16,quantizer_efSearch=8           0.2812 0.6432 0.7138      0.00361       27337271    84
nprobe=16,quantizer_efSearch=16          0.2843 0.6518 0.7235      0.00392       27313452    77
nprobe=16,quantizer_efSearch=32          0.2862 0.6564 0.7293      0.00440       27286447    69
nprobe=16,quantizer_efSearch=64          0.2866 0.6576 0.7304      0.00554       27280346    55
nprobe=32,quantizer_efSearch=32          0.3005 0.7249 0.8311      0.00842       53943563    36
nprobe=32,quantizer_efSearch=64          0.3010 0.7265 0.8327      0.00802       53919022    38
nprobe=32,quantizer_efSearch=128         0.3012 0.7271 0.8333      0.00997       53909472    31
nprobe=64,quantizer_efSearch=32          0.3065 0.7663 0.9026      0.01310      106395238    23
nprobe=64,quantizer_efSearch=64          0.3079 0.7692 0.9059      0.01353      106305124    23
nprobe=128,quantizer_efSearch=128        0.3113 0.7906 0.9517      0.02813      208780952    11
nprobe=256,quantizer_efSearch=64         0.3115 0.7973 0.9700      0.04061      409549109    8
nprobe=256,quantizer_efSearch=128        0.3126 0.7997 0.9740      0.04337      408868709    7
nprobe=512,quantizer_efSearch=256        0.3131 0.8029 0.9841      0.08445      799138136    4
nprobe=1024,quantizer_efSearch=128       0.3133 0.8024 0.9852      0.15613     1547560311    2
```

</details>
<details><summary>`PCAR32,IVF65536(IVF256,PQ16x4fs,RFlat),SQ8` </summary>
Index size 410115737

 code_size 32

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_IVF256_PQ16x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2816 0.6344 0.7030      0.00479       28125963    63
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1413 0.2259 0.2283      0.00224        3071193    134
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1432 0.2280 0.2304      0.00229        2423002    131
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1875 0.3341 0.3415      0.00238        4155526    126
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2140 0.4129 0.4283      0.00248        8298850    121
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2249 0.4425 0.4608      0.00248        8281597    122
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2269 0.4467 0.4652      0.00248        7636147    121
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2295 0.4529 0.4717      0.00267        8266489    113
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.2301 0.4533 0.4720      0.00298        8264963    101
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2504 0.5254 0.5603      0.00316       14579551    95
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2621 0.5548 0.5962      0.00317       14513356    95
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2642 0.5634 0.6061      0.00363       15123000    83
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.2643 0.5634 0.6063      0.00401       15121816    75
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2816 0.6344 0.7030      0.00487       28125985    62
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2855 0.6506 0.7217      0.00524       28710192    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2857 0.6528 0.7234      0.00593       32506254    51
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2871 0.6540 0.7262      0.00527       28669471    57
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.2875 0.6566 0.7298      0.00639       29908493    47
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.2876 0.6569 0.7300      0.00765       29912491    40
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2932 0.6961 0.7939      0.00847       54943634    36
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.2957 0.7093 0.8101      0.00929       59383663    33
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3004 0.7213 0.8273      0.00913       56660241    33
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3018 0.7252 0.8325      0.00974       56580610    31
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3019 0.7256 0.8331      0.01032       59108029    30
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.3023 0.7263 0.8338      0.01088       59093200    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3072 0.7537 0.8839      0.01435      108388666    21
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3079 0.7612 0.8942      0.01563      107898542    20
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3095 0.7703 0.9079      0.01555      111543643    20
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3096 0.7707 0.9079      0.01648      111450723    19
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.3121 0.7902 0.9515      0.03098      211600641    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.3127 0.7915 0.9536      0.03272      218997530    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3128 0.7910 0.9521      0.03280      219144123    10
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3129 0.7910 0.9561      0.05397      424944768    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3139 0.7978 0.9716      0.05204      420336838    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3143 0.8012 0.9839      0.09715      806409921    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3144 0.8022 0.9882      0.18416     1573051378    2
```

</details>
<details><summary>`PCAR32,IVF65536,SQ8` </summary>
Index size 408996512

 code_size 32

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3125 0.8010 0.9764      0.04415      408342557    7
nprobe=1                                 0.1424 0.2274 0.2299      0.00618        1730798    49
nprobe=2                                 0.1886 0.3390 0.3466      0.00654        3459974    46
nprobe=4                                 0.2293 0.4557 0.4742      0.00665        6919012    46
nprobe=8                                 0.2634 0.5662 0.6096      0.00715       13759195    42
nprobe=16                                0.2864 0.6591 0.7319      0.00901       27257214    34
nprobe=32                                0.3012 0.7284 0.8354      0.01197       53861966    26
nprobe=64                                0.3081 0.7715 0.9087      0.01534      106169467    20
nprobe=128                               0.3112 0.7917 0.9539      0.02251      208562864    14
nprobe=256                               0.3125 0.8010 0.9764      0.04210      408342557    8
nprobe=512                               0.3130 0.8034 0.9858      0.07996      798336663    4
nprobe=2048                              0.3131 0.8032 0.9892      0.27705     3048501660    2
```

</details>
<details><summary>`PCAR64,IVF16384_HNSW32,SQ4` </summary>
Index size 408886229

 code_size 32

 log filename: autotune.dbbigann10M.PCAR64_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5927 0.9842 0.9966      0.32757     2364460616    1
nprobe=1,quantizer_efSearch=16           0.2273 0.2877 0.2887      0.00358       13565883    84
nprobe=2,quantizer_efSearch=4            0.2916 0.3868 0.3881      0.00521       27105950    58
nprobe=2,quantizer_efSearch=8            0.3156 0.4205 0.4219      0.00516       26609142    59
nprobe=2,quantizer_efSearch=16           0.3184 0.4240 0.4254      0.00536       26342220    56
nprobe=2,quantizer_efSearch=32           0.3191 0.4243 0.4257      0.00601       26222939    50
nprobe=2,quantizer_efSearch=64           0.3194 0.4248 0.4262      0.00705       26202798    43
nprobe=4,quantizer_efSearch=4            0.3619 0.5099 0.5122      0.01094       52018312    28
nprobe=4,quantizer_efSearch=8            0.3966 0.5662 0.5686      0.00832       51346134    37
nprobe=4,quantizer_efSearch=16           0.4038 0.5766 0.5789      0.00850       50719736    36
nprobe=4,quantizer_efSearch=32           0.4053 0.5776 0.5799      0.00953       50458056    33
nprobe=4,quantizer_efSearch=64           0.4056 0.5781 0.5804      0.01047       50417066    29
nprobe=4,quantizer_efSearch=128          0.4057 0.5782 0.5805      0.01260       50397229    24
nprobe=8,quantizer_efSearch=4            0.4561 0.6755 0.6793      0.01642       98601100    19
nprobe=8,quantizer_efSearch=8            0.4645 0.6921 0.6960      0.01482       98182399    21
nprobe=8,quantizer_efSearch=16           0.4733 0.7091 0.7131      0.01525       97464986    20
nprobe=8,quantizer_efSearch=32           0.4744 0.7113 0.7153      0.01540       97033856    20
nprobe=8,quantizer_efSearch=64           0.4745 0.7115 0.7155      0.01635       96866648    19
nprobe=8,quantizer_efSearch=128          0.4746 0.7116 0.7156      0.01876       96832889    16
nprobe=16,quantizer_efSearch=4           0.4957 0.7592 0.7643      0.04366      187697976    7
nprobe=16,quantizer_efSearch=8           0.5151 0.8007 0.8064      0.02677      187295591    12
nprobe=16,quantizer_efSearch=16          0.5236 0.8174 0.8229      0.02590      186241898    12
nprobe=16,quantizer_efSearch=32          0.5268 0.8239 0.8295      0.02699      185232081    12
nprobe=16,quantizer_efSearch=64          0.5280 0.8251 0.8307      0.02773      184798108    11
nprobe=32,quantizer_efSearch=8           0.5417 0.8644 0.8714      0.05473      355369574    6
nprobe=32,quantizer_efSearch=64          0.5593 0.9050 0.9125      0.04816      351643447    7
nprobe=64,quantizer_efSearch=16          0.5702 0.9293 0.9386      0.08658      671307494    4
nprobe=64,quantizer_efSearch=32          0.5773 0.9457 0.9557      0.08655      669076683    4
nprobe=64,quantizer_efSearch=128         0.5783 0.9502 0.9602      0.09040      665298315    4
nprobe=64,quantizer_efSearch=256         0.5784 0.9504 0.9604      0.09624      665074841    4
nprobe=128,quantizer_efSearch=128        0.5882 0.9741 0.9855      0.16232     1257552277    2
nprobe=128,quantizer_efSearch=256        0.5883 0.9744 0.9858      0.16935     1256620517    2
nprobe=256,quantizer_efSearch=64         0.5916 0.9819 0.9939      0.30071     2375227540    1
nprobe=256,quantizer_efSearch=256        0.5926 0.9840 0.9964      0.30656     2365310880    1
nprobe=256,quantizer_efSearch=512        0.5927 0.9842 0.9966      0.32771     2364460616    1
nprobe=512,quantizer_efSearch=128        0.5929 0.9860 0.9988      0.57016     4450856358    1
nprobe=512,quantizer_efSearch=256        0.5930 0.9864 0.9992      0.57024     4446347541    1
nprobe=512,quantizer_efSearch=512        0.5931 0.9866 0.9994      0.58480     4441892696    1
nprobe=1024,quantizer_efSearch=256       0.5932 0.9866 0.9994      1.05856     8328786123    1
nprobe=1024,quantizer_efSearch=512       0.5934 0.9869 0.9997      1.09314     8328296746    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ4` </summary>
Index size 540649429

 code_size 32

 log filename: autotune.dbbigann10M.PCAR64_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5490 0.9413 0.9586      0.07483      105884255    5
nprobe=1,quantizer_efSearch=4            0.1380 0.1654 0.1655      0.00210         435113    144
nprobe=2,quantizer_efSearch=4            0.2007 0.2506 0.2515      0.00218         874940    138
nprobe=4,quantizer_efSearch=4            0.2613 0.3470 0.3492      0.00228        1751124    132
nprobe=8,quantizer_efSearch=4            0.3502 0.4915 0.4959      0.00280        3486418    108
nprobe=8,quantizer_efSearch=8            0.3563 0.5005 0.5049      0.00301        3482501    100
nprobe=16,quantizer_efSearch=4           0.3971 0.5885 0.5941      0.00335        6963502    90
nprobe=16,quantizer_efSearch=8           0.4183 0.6222 0.6279      0.00412        6953082    73
nprobe=16,quantizer_efSearch=16          0.4251 0.6344 0.6402      0.00472        6940696    64
nprobe=32,quantizer_efSearch=4           0.4284 0.6582 0.6654      0.00520       13828726    58
nprobe=32,quantizer_efSearch=8           0.4562 0.7130 0.7204      0.00546       13830232    55
nprobe=32,quantizer_efSearch=16          0.4713 0.7432 0.7508      0.00646       13801067    47
nprobe=32,quantizer_efSearch=32          0.4771 0.7538 0.7616      0.00818       13775374    37
nprobe=64,quantizer_efSearch=8           0.4802 0.7702 0.7796      0.00845       27381709    36
nprobe=64,quantizer_efSearch=16          0.4994 0.8155 0.8258      0.00936       27363964    33
nprobe=64,quantizer_efSearch=32          0.5080 0.8347 0.8458      0.01115       27295292    27
nprobe=64,quantizer_efSearch=64          0.5106 0.8409 0.8521      0.01447       27254195    21
nprobe=128,quantizer_efSearch=16         0.5155 0.8554 0.8682      0.01530       54083970    20
nprobe=128,quantizer_efSearch=32         0.5286 0.8867 0.9008      0.01680       53993280    18
nprobe=128,quantizer_efSearch=64         0.5329 0.8993 0.9138      0.02012       53871648    15
nprobe=128,quantizer_efSearch=128        0.5346 0.9022 0.9165      0.02749       53805250    11
nprobe=256,quantizer_efSearch=32         0.5382 0.9143 0.9304      0.03106      106426466    10
nprobe=256,quantizer_efSearch=64         0.5459 0.9340 0.9510      0.03129      106224747    10
nprobe=256,quantizer_efSearch=128        0.5484 0.9397 0.9569      0.03702      105992538    9
nprobe=256,quantizer_efSearch=256        0.5491 0.9415 0.9587      0.04903      105903355    7
nprobe=512,quantizer_efSearch=128        0.5530 0.9574 0.9766      0.06122      208445157    5
nprobe=512,quantizer_efSearch=256        0.5531 0.9598 0.9793      0.07440      208047584    5
nprobe=512,quantizer_efSearch=512        0.5536 0.9607 0.9802      0.09733      207916175    4
nprobe=1024,quantizer_efSearch=128       0.5546 0.9646 0.9850      0.10731      406284430    3
nprobe=1024,quantizer_efSearch=256       0.5560 0.9695 0.9900      0.12937      408022625    3
nprobe=1024,quantizer_efSearch=512       0.5568 0.9708 0.9915      0.16198      407477589    2
nprobe=2048,quantizer_efSearch=512       0.5578 0.9742 0.9959      0.32198      797415555    1
nprobe=4096,quantizer_efSearch=512       0.5582 0.9748 0.9968      0.46789     1520791903    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ4` </summary>
Index size 475871001

 code_size 32

 log filename: autotune.dbbigann10M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5591 0.9590 0.9746      0.10214      425664030    3
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.1174 0.1370 0.1370      0.00224        1824648    134
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1431 0.1699 0.1699      0.00226        1821060    133
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1432 0.1717 0.1717      0.00226        1138606    133
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1458 0.1741 0.1741      0.00229        1822563    131
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.1765 0.2152 0.2155      0.00231        2264474    130
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2057 0.2584 0.2589      0.00233        1578180    129
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2132 0.2657 0.2661      0.00234        2257946    129
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2474 0.3202 0.3209      0.00243        3147978    124
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2714 0.3629 0.3635      0.00242        2460163    125
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3402 0.4750 0.4768      0.00256        4206883    118
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3552 0.4929 0.4942      0.00266        4887568    113
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3622 0.5196 0.5212      0.00315        7797756    96
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4160 0.6115 0.6144      0.00342        8374846    88
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4271 0.6280 0.6311      0.00384        9683353    79
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4336 0.6389 0.6422      0.00415        9667564    73
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4341 0.6406 0.6440      0.00540        9651629    56
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4580 0.7085 0.7138      0.00588       15283421    52
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4642 0.7198 0.7254      0.00553       15247106    55
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4805 0.7451 0.7506      0.00667       19172283    46
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4811 0.7463 0.7518      0.00787       24336408    39
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4871 0.7597 0.7656      0.00911       19120806    33
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4881 0.7601 0.7660      0.00846       24305632    36
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.4970 0.7880 0.7948      0.00867       33135628    35
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5077 0.8188 0.8277      0.00847       30166076    36
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5172 0.8344 0.8438      0.00960       32718581    32
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5199 0.8419 0.8513      0.01323       32628259    23
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5216 0.8452 0.8546      0.01167       37788370    26
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5304 0.8694 0.8796      0.01651       65473608    19
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5435 0.9071 0.9193      0.02089       64353496    15
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.5447 0.9082 0.9203      0.02828       64328564    11
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5484 0.9207 0.9333      0.03107      129095738    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5542 0.9386 0.9519      0.03419      111554062    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5564 0.9441 0.9571      0.03537      126948560    9
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5583 0.9473 0.9609      0.03550      116490592    9
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.5585 0.9473 0.9610      0.04117      116464928    8
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5586 0.9524 0.9673      0.05030      223803220    7
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5588 0.9528 0.9677      0.06440      233793278    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5632 0.9639 0.9798      0.06652      219116084    5
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.5635 0.9648 0.9806      0.07992      218630900    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5657 0.9750 0.9924      0.11628      428803305    3
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.5662 0.9764 0.9939      0.16251      830214834    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5675 0.9800 0.9980      0.22146      818670569    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.5676 0.9802 0.9981      0.22028      817630913    2
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5678 0.9813 0.9994      0.42091     1580713896    1
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ4` </summary>
Index size 435239381

 code_size 32

 log filename: autotune.dbbigann10M.PCAR64_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5561 0.9652 0.9858      0.12836      410883994    3
nprobe=1,quantizer_efSearch=4            0.1781 0.2260 0.2268      0.00205        1728532    147
nprobe=2,quantizer_efSearch=4            0.2475 0.3353 0.3370      0.00212        3462384    142
nprobe=2,quantizer_efSearch=8            0.2615 0.3533 0.3552      0.00236        3451339    128
nprobe=4,quantizer_efSearch=4            0.3215 0.4555 0.4588      0.00278        6914088    108
nprobe=4,quantizer_efSearch=8            0.3411 0.4876 0.4913      0.00297        6903195    101
nprobe=4,quantizer_efSearch=16           0.3454 0.4925 0.4962      0.00336        6893366    90
nprobe=4,quantizer_efSearch=32           0.3472 0.4947 0.4984      0.00417        6887228    72
nprobe=8,quantizer_efSearch=4            0.4017 0.6062 0.6117      0.00445       13789760    68
nprobe=8,quantizer_efSearch=8            0.4099 0.6183 0.6239      0.00452       13779186    67
nprobe=8,quantizer_efSearch=16           0.4164 0.6285 0.6342      0.00494       13760777    61
nprobe=8,quantizer_efSearch=32           0.4192 0.6326 0.6383      0.00564       13746089    54
nprobe=8,quantizer_efSearch=64           0.4203 0.6338 0.6395      0.00695       13741687    44
nprobe=16,quantizer_efSearch=8           0.4639 0.7327 0.7406      0.00793       27364495    38
nprobe=16,quantizer_efSearch=16          0.4698 0.7427 0.7505      0.00813       27330939    37
nprobe=16,quantizer_efSearch=32          0.4728 0.7487 0.7566      0.00884       27298380    34
nprobe=16,quantizer_efSearch=64          0.4742 0.7505 0.7584      0.01015       27285460    30
nprobe=32,quantizer_efSearch=8           0.4932 0.8074 0.8191      0.01397       54239822    22
nprobe=32,quantizer_efSearch=16          0.5046 0.8293 0.8419      0.01432       54153210    21
nprobe=32,quantizer_efSearch=32          0.5088 0.8379 0.8505      0.01488       54075922    21
nprobe=32,quantizer_efSearch=64          0.5106 0.8415 0.8541      0.01624       54047695    19
nprobe=64,quantizer_efSearch=16          0.5257 0.8852 0.8996      0.02641      106971450    12
nprobe=64,quantizer_efSearch=32          0.5343 0.9018 0.9167      0.02695      106779323    12
nprobe=64,quantizer_efSearch=64          0.5361 0.9063 0.9212      0.02791      106685941    11
nprobe=64,quantizer_efSearch=128         0.5362 0.9071 0.9223      0.03075      106646149    10
nprobe=64,quantizer_efSearch=256         0.5363 0.9072 0.9224      0.03785      106637053    8
nprobe=128,quantizer_efSearch=32         0.5464 0.9357 0.9538      0.05094      210185660    6
nprobe=128,quantizer_efSearch=128        0.5494 0.9450 0.9638      0.05395      209624205    6
nprobe=128,quantizer_efSearch=256        0.5497 0.9455 0.9642      0.06101      209594026    5
nprobe=128,quantizer_efSearch=512        0.5498 0.9458 0.9645      0.08131      209588133    4
nprobe=256,quantizer_efSearch=64         0.5553 0.9613 0.9816      0.09810      411822577    4
nprobe=256,quantizer_efSearch=128        0.5557 0.9638 0.9844      0.10095      411132239    3
nprobe=256,quantizer_efSearch=256        0.5560 0.9649 0.9855      0.10581      410929086    3
nprobe=256,quantizer_efSearch=512        0.5561 0.9652 0.9858      0.12625      410883994    3
nprobe=512,quantizer_efSearch=256        0.5570 0.9709 0.9932      0.20020      804470477    2
nprobe=512,quantizer_efSearch=512        0.5573 0.9714 0.9937      0.21067      804198533    2
nprobe=1024,quantizer_efSearch=256       0.5574 0.9737 0.9963      0.36992     1573724692    1
nprobe=1024,quantizer_efSearch=512       0.5578 0.9743 0.9969      0.40323     1572405678    1
nprobe=2048,quantizer_efSearch=512       0.5579 0.9752 0.9978      0.73937     3075812377    1
```

</details>
<details><summary>`PCAR64,IVF65536(IVF256,PQ32x4fs,RFlat),SQ4` </summary>
Index size 419114265

 code_size 32

 log filename: autotune.dbbigann10M.PCAR64_IVF65536_IVF256_PQ32x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4623 0.7474 0.7589      0.02186      110610011    14
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1875 0.2377 0.2387      0.00234        2408325    129
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2598 0.3509 0.3532      0.00246        4147117    122
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2607 0.3527 0.3550      0.00249        4799870    121
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.2629 0.3565 0.3588      0.00260        4788718    116
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3454 0.4921 0.4958      0.00298        8240231    101
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.3469 0.4942 0.4983      0.00329        8231194    92
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3474 0.4954 0.4995      0.00346        9518876    87
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4145 0.6201 0.6263      0.00433       14476827    70
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.4208 0.6328 0.6389      0.00491       15097643    62
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4215 0.6340 0.6401      0.00493       16375848    61
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4586 0.7151 0.7237      0.00669       28166794    45
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4709 0.7391 0.7478      0.00728       29994804    42
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4712 0.7392 0.7479      0.00793       32556948    38
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4734 0.7485 0.7572      0.00773       29931506    39
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4739 0.7489 0.7576      0.00867       32487305    35
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.4748 0.7510 0.7597      0.00886       29917675    34
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.4751 0.7511 0.7598      0.00965       37576477    32
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4909 0.7994 0.8107      0.01307       55177238    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4933 0.8058 0.8176      0.01197       55053062    26
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5065 0.8349 0.8472      0.01227       55503822    25
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5073 0.8362 0.8484      0.01279       55480138    24
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5090 0.8430 0.8552      0.01333       56679253    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5091 0.8423 0.8548      0.01268       56707495    24
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5095 0.8428 0.8553      0.01339       59238072    23
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5252 0.8910 0.9063      0.02183      108493793    14
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5320 0.9094 0.9250      0.02372      111821740    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.5321 0.9095 0.9251      0.02467      116910153    13
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5368 0.9264 0.9442      0.04135      212156475    8
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5377 0.9245 0.9420      0.04215      219395070    8
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5453 0.9474 0.9661      0.04704      219822858    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5512 0.9648 0.9859      0.07961      416663005    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5515 0.9650 0.9861      0.08066      421661202    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5534 0.9728 0.9945      0.15273      815021543    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5536 0.9737 0.9954      0.15792      814158146    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5544 0.9770 0.9989      0.30288     1582770415    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5545 0.9775 0.9997      0.55541     3084303045    1
```

</details>
<details><summary>`PCAR64,IVF65536,SQ4` </summary>
Index size 417401888

 code_size 32

 log filename: autotune.dbbigann10M.PCAR64_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5516 0.9654 0.9876      0.07964      410455842    4
nprobe=8                                 0.4206 0.6355 0.6411      0.01110       13727271    28
nprobe=32                                0.5101 0.8437 0.8565      0.01883       53973665    16
nprobe=64                                0.5323 0.9088 0.9253      0.02786      106507883    11
nprobe=128                               0.5457 0.9477 0.9669      0.04507      209345098    7
nprobe=256                               0.5516 0.9654 0.9876      0.08011      410455842    4
nprobe=512                               0.5542 0.9723 0.9956      0.14628      803489414    3
nprobe=1024                              0.5548 0.9750 0.9987      0.27301     1571092329    2
nprobe=2048                              0.5551 0.9757 0.9997      0.56432     3073208140    1
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF16384_HNSW32,PQ128x4fs` </summary>
Index size 757699788

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF16384_HNSW32_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5621 0.9734 0.9974      0.07897     2744528545    4
nprobe=2,quantizer_efSearch=4            0.2747 0.3876 0.3911      0.00284       33661785    106
nprobe=2,quantizer_efSearch=8            0.3003 0.4255 0.4291      0.00364       32761903    83
nprobe=4,quantizer_efSearch=4            0.3409 0.5061 0.5117      0.00376       63935355    80
nprobe=4,quantizer_efSearch=8            0.3777 0.5649 0.5706      0.00407       63116859    74
nprobe=4,quantizer_efSearch=16           0.3883 0.5804 0.5861      0.00524       62143683    58
nprobe=8,quantizer_efSearch=4            0.4255 0.6617 0.6708      0.00519      120224380    58
nprobe=8,quantizer_efSearch=8            0.4379 0.6827 0.6923      0.00520      120010390    58
nprobe=8,quantizer_efSearch=16           0.4539 0.7079 0.7172      0.00669      118576704    45
nprobe=16,quantizer_efSearch=4           0.4594 0.7369 0.7493      0.00806      227370944    38
nprobe=16,quantizer_efSearch=8           0.4910 0.7948 0.8081      0.00626      227161536    49
nprobe=16,quantizer_efSearch=16          0.5008 0.8137 0.8273      0.00746      225678538    41
nprobe=32,quantizer_efSearch=8           0.5100 0.8477 0.8630      0.01157      427094192    26
nprobe=32,quantizer_efSearch=32          0.5346 0.8977 0.9143      0.01390      424030881    22
nprobe=32,quantizer_efSearch=64          0.5365 0.9001 0.9169      0.01524      422357081    20
nprobe=64,quantizer_efSearch=16          0.5398 0.9189 0.9379      0.01669      799000046    18
nprobe=64,quantizer_efSearch=32          0.5493 0.9398 0.9600      0.01939      796558944    16
nprobe=64,quantizer_efSearch=64          0.5517 0.9442 0.9643      0.02235      793106822    14
nprobe=64,quantizer_efSearch=128         0.5519 0.9443 0.9645      0.02553      791618462    12
nprobe=128,quantizer_efSearch=64         0.5581 0.9623 0.9853      0.02570     1482334060    12
nprobe=128,quantizer_efSearch=128        0.5589 0.9636 0.9867      0.03506     1478244084    9
nprobe=256,quantizer_efSearch=64         0.5608 0.9703 0.9942      0.04813     2755798092    7
nprobe=256,quantizer_efSearch=256        0.5621 0.9734 0.9973      0.05571     2745377849    6
nprobe=512,quantizer_efSearch=512        0.5633 0.9755 0.9997      0.10717     5078279677    3
```

</details>
<details><summary>`OPQ128_256,IVF16384_HNSW32,PQ128x4fsr` </summary>
Index size 757857484

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF16384_HNSW32_PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7286 0.8428 0.8428      0.03324      111847067    10
nprobe=1,quantizer_efSearch=8            0.2894 0.3107 0.3107      0.00275        7290163    110
nprobe=1,quantizer_efSearch=16           0.2915 0.3129 0.3129      0.00338        7278045    89
nprobe=2,quantizer_efSearch=4            0.3945 0.4310 0.4310      0.00371       14555709    81
nprobe=2,quantizer_efSearch=8            0.4177 0.4558 0.4558      0.00408       14536371    74
nprobe=2,quantizer_efSearch=16           0.4219 0.4605 0.4605      0.00471       14509578    64
nprobe=4,quantizer_efSearch=4            0.5020 0.5563 0.5563      0.00555       28834765    55
nprobe=4,quantizer_efSearch=8            0.5410 0.5995 0.5995      0.00589       28827848    51
nprobe=4,quantizer_efSearch=16           0.5479 0.6077 0.6077      0.00645       28782723    47
nprobe=4,quantizer_efSearch=32           0.5492 0.6089 0.6089      0.00739       28759218    41
nprobe=4,quantizer_efSearch=64           0.5494 0.6092 0.6092      0.00900       28754503    34
nprobe=8,quantizer_efSearch=8            0.6373 0.7230 0.7230      0.01672       56972919    18
nprobe=8,quantizer_efSearch=16           0.6461 0.7347 0.7347      0.01752       56879193    18
nprobe=8,quantizer_efSearch=32           0.6485 0.7372 0.7372      0.01903       56809500    16
nprobe=8,quantizer_efSearch=64           0.6490 0.7376 0.7376      0.02001       56802718    16
nprobe=8,quantizer_efSearch=128          0.6491 0.7377 0.7377      0.02307       56799047    14
nprobe=16,quantizer_efSearch=16          0.7267 0.8406 0.8406      0.03116      112004911    10
nprobe=16,quantizer_efSearch=32          0.7286 0.8428 0.8428      0.03386      111847067    9
nprobe=16,quantizer_efSearch=64          0.7292 0.8434 0.8434      0.03432      111822653    9
nprobe=16,quantizer_efSearch=128         0.7294 0.8436 0.8436      0.03692      111814017    9
nprobe=32,quantizer_efSearch=64          0.7843 0.9191 0.9191      0.07020      219475235    5
nprobe=32,quantizer_efSearch=128         0.7844 0.9192 0.9192      0.07444      219446886    5
nprobe=64,quantizer_efSearch=32          0.8113 0.9630 0.9631      0.12818      429487786    3
nprobe=64,quantizer_efSearch=64          0.8132 0.9647 0.9648      0.13106      429109079    3
nprobe=64,quantizer_efSearch=128         0.8137 0.9647 0.9648      0.13290      428990138    3
nprobe=64,quantizer_efSearch=512         0.8138 0.9646 0.9647      0.16441      428940461    2
nprobe=128,quantizer_efSearch=256        0.8265 0.9873 0.9873      0.25731      835697272    2
nprobe=256,quantizer_efSearch=64         0.8311 0.9953 0.9953      0.47486     1628791127    1
nprobe=256,quantizer_efSearch=128        0.8321 0.9964 0.9964      0.47448     1625872686    1
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fs` </summary>
Index size 1319993548

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF262144_HNSW32_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5282 0.8668 0.8842      0.02853      179735356    11
nprobe=2,quantizer_efSearch=4            0.1972 0.2501 0.2510      0.00362         871255    83
nprobe=4,quantizer_efSearch=4            0.2545 0.3438 0.3466      0.00401        1744127    75
nprobe=8,quantizer_efSearch=4            0.3454 0.4869 0.4922      0.00560        3477394    54
nprobe=8,quantizer_efSearch=8            0.3545 0.5002 0.5055      0.00669        3473150    45
nprobe=16,quantizer_efSearch=4           0.3965 0.5872 0.5944      0.00673        6946311    45
nprobe=32,quantizer_efSearch=4           0.4248 0.6535 0.6631      0.00829       13798908    37
nprobe=64,quantizer_efSearch=4           0.4403 0.6852 0.6962      0.01064       27251574    29
nprobe=32,quantizer_efSearch=8           0.4576 0.7126 0.7227      0.01032       13800995    30
nprobe=64,quantizer_efSearch=8           0.4821 0.7673 0.7802      0.01246       27348489    25
nprobe=64,quantizer_efSearch=16          0.5056 0.8108 0.8246      0.01588       27312430    19
nprobe=128,quantizer_efSearch=16         0.5230 0.8516 0.8679      0.01945       54016776    16
nprobe=256,quantizer_efSearch=16         0.5278 0.8652 0.8821      0.02607      105392926    12
nprobe=128,quantizer_efSearch=32         0.5372 0.8838 0.9012      0.02549       53916621    12
nprobe=256,quantizer_efSearch=32         0.5476 0.9090 0.9285      0.03192      106332176    10
nprobe=512,quantizer_efSearch=32         0.5485 0.9167 0.9369      0.04078      203918653    8
nprobe=256,quantizer_efSearch=64         0.5569 0.9297 0.9509      0.04469      106141184    7
nprobe=512,quantizer_efSearch=64         0.5607 0.9434 0.9662      0.05341      208405236    6
nprobe=1024,quantizer_efSearch=64        0.5618 0.9474 0.9705      0.06580      389744950    5
nprobe=512,quantizer_efSearch=128        0.5649 0.9536 0.9771      0.06816      208370655    5
nprobe=1024,quantizer_efSearch=128       0.5671 0.9602 0.9844      0.08439      406569802    4
nprobe=1024,quantizer_efSearch=256       0.5694 0.9649 0.9894      0.11864      408042964    3
nprobe=2048,quantizer_efSearch=256       0.5695 0.9676 0.9932      0.15309      789511298    2
nprobe=1024,quantizer_efSearch=512       0.5696 0.9659 0.9906      0.17750      407515187    2
nprobe=2048,quantizer_efSearch=512       0.5699 0.9701 0.9955      0.21750      797915507    2
nprobe=4096,quantizer_efSearch=512       0.5701 0.9704 0.9962      0.28181     1525643159    2
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fsr` </summary>
Index size 1319928012

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF262144_HNSW32_PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5937 0.6530 0.6530      0.03454        6898405    9
nprobe=2,quantizer_efSearch=4            0.2381 0.2520 0.2520      0.00518         867852    58
nprobe=4,quantizer_efSearch=4            0.3256 0.3486 0.3486      0.00655        1738767    46
nprobe=4,quantizer_efSearch=8            0.3621 0.3864 0.3864      0.00900        1733836    34
nprobe=4,quantizer_efSearch=16           0.3694 0.3940 0.3940      0.01321        1729533    23
nprobe=8,quantizer_efSearch=4            0.4579 0.4951 0.4951      0.01571        3469625    20
nprobe=8,quantizer_efSearch=16           0.4813 0.5220 0.5220      0.01872        3455689    17
nprobe=8,quantizer_efSearch=32           0.4861 0.5267 0.5267      0.02414        3451595    13
nprobe=16,quantizer_efSearch=8           0.5759 0.6333 0.6333      0.02902        6918353    11
nprobe=16,quantizer_efSearch=16          0.5867 0.6455 0.6455      0.03156        6909313    10
nprobe=16,quantizer_efSearch=32          0.5937 0.6530 0.6530      0.03396        6898405    9
nprobe=16,quantizer_efSearch=64          0.5952 0.6544 0.6544      0.04374        6893030    7
nprobe=16,quantizer_efSearch=128         0.5957 0.6548 0.6548      0.05969        6891864    6
nprobe=32,quantizer_efSearch=8           0.6480 0.7239 0.7239      0.06208       13767346    5
nprobe=32,quantizer_efSearch=16          0.6729 0.7543 0.7543      0.06329       13741409    5
nprobe=32,quantizer_efSearch=64          0.6858 0.7683 0.7683      0.08004       13706165    4
nprobe=32,quantizer_efSearch=128         0.6877 0.7697 0.7697      0.09955       13701475    4
nprobe=64,quantizer_efSearch=8           0.6947 0.7806 0.7806      0.12654       27277848    3
nprobe=64,quantizer_efSearch=16          0.7342 0.8276 0.8276      0.12771       27252469    3
nprobe=64,quantizer_efSearch=32          0.7522 0.8490 0.8490      0.13234       27189134    3
nprobe=64,quantizer_efSearch=64          0.7569 0.8541 0.8541      0.14520       27152735    3
nprobe=64,quantizer_efSearch=128         0.7585 0.8549 0.8549      0.15932       27138210    2
nprobe=128,quantizer_efSearch=16         0.7656 0.8703 0.8703      0.25309       53891700    2
nprobe=128,quantizer_efSearch=64         0.8035 0.9166 0.9166      0.22705       53681022    2
nprobe=128,quantizer_efSearch=256        0.8045 0.9196 0.9196      0.29174       53604692    2
nprobe=128,quantizer_efSearch=512        0.8047 0.9200 0.9200      0.35766       53599947    1
nprobe=256,quantizer_efSearch=64         0.8275 0.9537 0.9537      0.48463      105940757    1
nprobe=256,quantizer_efSearch=128        0.8329 0.9609 0.9609      0.50287      105718835    1
nprobe=256,quantizer_efSearch=256        0.8331 0.9624 0.9624      0.50133      105629928    1
nprobe=256,quantizer_efSearch=512        0.8333 0.9629 0.9629      0.55574      105608349    1
nprobe=512,quantizer_efSearch=128        0.8455 0.9791 0.9791      0.90704      208035065    1
nprobe=512,quantizer_efSearch=256        0.8464 0.9820 0.9820      0.89270      207664996    1
nprobe=1024,quantizer_efSearch=128       0.8469 0.9859 0.9859      1.74178      203764715    1
nprobe=1024,quantizer_efSearch=256       0.8479 0.9908 0.9908      1.77779      204645403    1
nprobe=1024,quantizer_efSearch=512       0.8495 0.9922 0.9922      1.67184      204404265    1
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 1269505552

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5626 0.9529 0.9764      0.06093      425041771    5
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1236 0.1478 0.1481      0.00186         615398    162
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1633 0.2033 0.2042      0.00198        1062011    152
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1644 0.2067 0.2076      0.00219        1055567    138
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1872 0.2357 0.2366      0.00214        1235419    141
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.2053 0.2680 0.2703      0.00233        1951726    129
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2062 0.2619 0.2630      0.00239        1577695    126
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.2087 0.2739 0.2766      0.00245        1936914    123
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2521 0.3352 0.3380      0.00253        2109769    119
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2532 0.3367 0.3395      0.00279        2107527    108
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2672 0.3583 0.3606      0.00263        2466508    114
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2779 0.3723 0.3752      0.00293        2452033    103
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2780 0.3713 0.3742      0.00270        2454508    112
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2788 0.3736 0.3759      0.00303        3146246    100
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3307 0.4661 0.4705      0.00316        4232483    96
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3519 0.4959 0.5005      0.00354        4901191    85
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3872 0.5687 0.5757      0.00419        7776585    72
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4121 0.6078 0.6153      0.00463        8422615    65
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4185 0.6236 0.6310      0.00512        8341120    59
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4550 0.7022 0.7123      0.00583       15421980    52
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4721 0.7341 0.7442      0.00703       16697733    43
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4879 0.7715 0.7853      0.00979       29319358    31
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5084 0.8138 0.8282      0.01083       30517428    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5131 0.8238 0.8383      0.01231       30022645    25
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5179 0.8290 0.8435      0.01271       33059069    24
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5210 0.8388 0.8532      0.01404       32585359    22
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5325 0.8692 0.8871      0.01332       57820872    23
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5355 0.8743 0.8925      0.01753       56728371    18
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5465 0.8956 0.9143      0.01907       65320452    16
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5545 0.9265 0.9470      0.02396      113747411    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5574 0.9308 0.9518      0.03040      111436401    10
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5592 0.9363 0.9572      0.03126      128875260    10
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5606 0.9433 0.9661      0.03891      218880701    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5630 0.9419 0.9634      0.03806      126476945    8
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5665 0.9550 0.9784      0.04074      223215626    8
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5672 0.9566 0.9799      0.04501      233310306    7
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.5690 0.9663 0.9907      0.06616      428414600    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5695 0.9678 0.9923      0.06824      437937202    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5696 0.9679 0.9925      0.07983      458233908    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5697 0.9670 0.9917      0.10235      418371115    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5703 0.9684 0.9934      0.08759      428071019    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5705 0.9686 0.9937      0.10132      448353786    3
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5707 0.9726 0.9984      0.16523      818078477    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5709 0.9728 0.9987      0.16803      837978052    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5711 0.9735 0.9994      0.19948     1619235711    2
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 1269564944

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.7401 0.8387 0.8387      0.12345       30009568    3
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1703 0.1768 0.1768      0.00274        1140926    110
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1705 0.1769 0.1769      0.00287        1139644    105
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1973 0.2066 0.2066      0.00331        1054131    91
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2333 0.2456 0.2456      0.00340        1230265    89
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2340 0.2456 0.2456      0.00373        1227296    81
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2485 0.2612 0.2612      0.00352        1578321    86
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2514 0.2635 0.2635      0.00367        1574359    82
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2586 0.2716 0.2716      0.00398        2261031    76
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2629 0.2755 0.2755      0.00401        2257847    75
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.3133 0.3350 0.3350      0.00490        2110692    62
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3521 0.3756 0.3756      0.00521        2451661    58
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3664 0.3905 0.3905      0.00562        3133172    54
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3681 0.3921 0.3921      0.00597        3129284    51
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3720 0.3960 0.3960      0.00637        4458010    48
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3741 0.3985 0.3985      0.00625        4455521    49
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4484 0.4843 0.4843      0.01521        4194893    20
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.4486 0.4844 0.4844      0.01575        4194995    20
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4682 0.5075 0.5075      0.01601        4876993    19
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.4821 0.5219 0.5219      0.01717        8819230    18
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4870 0.5275 0.5275      0.01818        6186561    17
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4897 0.5300 0.5300      0.01768        8795925    17
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.5295 0.5808 0.5808      0.02850        7709490    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.5335 0.5851 0.5851      0.02722        7682930    12
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5875 0.6470 0.6470      0.03087       12272058    10
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5907 0.6500 0.6500      0.03178        9635230    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5948 0.6541 0.6541      0.03053       12197824    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5961 0.6550 0.6550      0.03247       17358090    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6791 0.7595 0.7595      0.06981       16488807    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6865 0.7681 0.7681      0.06599       19087829    5
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.6877 0.7683 0.7683      0.06330       24069921    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6892 0.7703 0.7703      0.06510       24271631    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.7002 0.7906 0.7906      0.12960       28806435    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7386 0.8360 0.8360      0.12749       30134992    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7503 0.8501 0.8501      0.12310       32632837    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.7537 0.8540 0.8540      0.12980       37743797    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7562 0.8561 0.8561      0.12727       37285937    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.7824 0.8926 0.8926      0.23663       56701373    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.8035 0.9191 0.9191      0.24221       64341295    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.8052 0.9202 0.9202      0.24449       64231334    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.8053 0.9211 0.9211      0.26300       94893529    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.8354 0.9627 0.9627      0.47572      116026533    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.8359 0.9638 0.9638      0.50991      146935622    1
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.8360 0.9638 0.9638      0.49319      126517824    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.8468 0.9840 0.9840      0.90698      228958090    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8473 0.9833 0.9833      0.90911      228517349    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.8474 0.9820 0.9820      0.93779      218505655    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.8512 0.9934 0.9934      1.74663      225344977    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.8514 0.9937 0.9937      1.73996      245684237    1
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fs` </summary>
Index size 871078604

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF65536_HNSW32_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5633 0.9612 0.9860      0.07924      410834812    4
nprobe=2,quantizer_efSearch=4            0.2447 0.3304 0.3322      0.00250        3451799    121
nprobe=4,quantizer_efSearch=4            0.3141 0.4470 0.4513      0.00304        6892211    99
nprobe=4,quantizer_efSearch=8            0.3420 0.4869 0.4911      0.00387        6879782    78
nprobe=8,quantizer_efSearch=8            0.4118 0.6169 0.6245      0.00471       13744613    64
nprobe=16,quantizer_efSearch=4           0.4455 0.6916 0.7017      0.00532       27340392    57
nprobe=16,quantizer_efSearch=8           0.4671 0.7296 0.7404      0.00637       27303844    48
nprobe=32,quantizer_efSearch=8           0.4989 0.8009 0.8154      0.00712       54163736    43
nprobe=32,quantizer_efSearch=16          0.5109 0.8270 0.8422      0.00985       54040816    31
nprobe=64,quantizer_efSearch=8           0.5156 0.8401 0.8581      0.01163      106956611    26
nprobe=32,quantizer_efSearch=32          0.5171 0.8359 0.8515      0.01217       53966912    25
nprobe=64,quantizer_efSearch=16          0.5307 0.8797 0.8986      0.01255      106859015    24
nprobe=64,quantizer_efSearch=32          0.5399 0.8975 0.9172      0.01491      106673419    21
nprobe=128,quantizer_efSearch=16         0.5407 0.9055 0.9270      0.01688      210449963    18
nprobe=128,quantizer_efSearch=32         0.5536 0.9319 0.9547      0.01865      210054203    17
nprobe=128,quantizer_efSearch=64         0.5563 0.9401 0.9631      0.02535      209663353    12
nprobe=256,quantizer_efSearch=32         0.5573 0.9454 0.9696      0.02451      412502899    13
nprobe=256,quantizer_efSearch=64         0.5625 0.9580 0.9826      0.03035      411753674    10
nprobe=256,quantizer_efSearch=128        0.5631 0.9603 0.9851      0.03808      411081297    8
nprobe=512,quantizer_efSearch=128        0.5646 0.9673 0.9930      0.04986      805901665    7
nprobe=512,quantizer_efSearch=256        0.5647 0.9680 0.9936      0.06804      804901231    5
nprobe=2048,quantizer_efSearch=128       0.5649 0.9695 0.9955      0.08811     2700190541    4
nprobe=512,quantizer_efSearch=512        0.5650 0.9689 0.9945      0.08904      804632733    4
nprobe=1024,quantizer_efSearch=256       0.5654 0.9708 0.9967      0.08324     1575223290    4
nprobe=2048,quantizer_efSearch=256       0.5655 0.9709 0.9970      0.12512     3024933482    3
nprobe=1024,quantizer_efSearch=512       0.5661 0.9721 0.9980      0.12163     1573970120    3
nprobe=2048,quantizer_efSearch=512       0.5664 0.9729 0.9989      0.17046     3079828988    2
nprobe=4096,quantizer_efSearch=512       0.5665 0.9730 0.9991      0.25317     5827398870    2
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fsr` </summary>
Index size 871121612

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF65536_HNSW32_PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6813 0.7643 0.7643      0.03127       27193268    10
nprobe=2,quantizer_efSearch=4            0.3089 0.3330 0.3330      0.00364        3447071    83
nprobe=4,quantizer_efSearch=4            0.4144 0.4517 0.4517      0.00533        6878948    57
nprobe=4,quantizer_efSearch=8            0.4508 0.4919 0.4919      0.00613        6869542    49
nprobe=4,quantizer_efSearch=16           0.4587 0.5005 0.5005      0.00766        6858620    40
nprobe=4,quantizer_efSearch=32           0.4611 0.5031 0.5031      0.00947        6854017    32
nprobe=4,quantizer_efSearch=64           0.4614 0.5033 0.5033      0.01373        6852827    22
nprobe=8,quantizer_efSearch=4            0.5498 0.6092 0.6092      0.01549       13731643    20
nprobe=8,quantizer_efSearch=8            0.5651 0.6259 0.6259      0.01600       13723385    19
nprobe=8,quantizer_efSearch=16           0.5765 0.6397 0.6397      0.01961       13704931    16
nprobe=8,quantizer_efSearch=32           0.5803 0.6432 0.6432      0.01974       13689862    16
nprobe=8,quantizer_efSearch=64           0.5813 0.6440 0.6440      0.02332       13687517    13
nprobe=8,quantizer_efSearch=128          0.5817 0.6443 0.6443      0.02877       13686721    11
nprobe=16,quantizer_efSearch=4           0.6282 0.7025 0.7025      0.02562       27295513    12
nprobe=16,quantizer_efSearch=8           0.6608 0.7407 0.7407      0.02768       27261602    11
nprobe=16,quantizer_efSearch=16          0.6738 0.7564 0.7564      0.03008       27228429    10
nprobe=16,quantizer_efSearch=32          0.6813 0.7643 0.7643      0.03158       27193268    10
nprobe=16,quantizer_efSearch=64          0.6827 0.7655 0.7655      0.03463       27184655    9
nprobe=16,quantizer_efSearch=128         0.6837 0.7663 0.7663      0.04046       27181694    8
nprobe=16,quantizer_efSearch=256         0.6838 0.7664 0.7664      0.05392       27180669    6
nprobe=32,quantizer_efSearch=8           0.7183 0.8161 0.8161      0.06517       54079463    5
nprobe=32,quantizer_efSearch=64          0.7513 0.8562 0.8562      0.07357       53863486    5
nprobe=32,quantizer_efSearch=128         0.7515 0.8565 0.8565      0.07763       53855656    4
nprobe=32,quantizer_efSearch=512         0.7516 0.8566 0.8566      0.11611       53851927    3
nprobe=64,quantizer_efSearch=32          0.7999 0.9191 0.9191      0.11441      106519159    3
nprobe=64,quantizer_efSearch=64          0.8023 0.9230 0.9230      0.11392      106425371    3
nprobe=64,quantizer_efSearch=128         0.8032 0.9239 0.9239      0.12245      106389957    3
nprobe=64,quantizer_efSearch=256         0.8034 0.9241 0.9241      0.14482      106383107    3
nprobe=64,quantizer_efSearch=512         0.8035 0.9242 0.9242      0.17801      106380439    2
nprobe=128,quantizer_efSearch=32         0.8252 0.9565 0.9565      0.24839      209750303    2
nprobe=128,quantizer_efSearch=64         0.8311 0.9648 0.9648      0.24419      209380469    2
nprobe=128,quantizer_efSearch=128        0.8318 0.9663 0.9663      0.24992      209236446    2
nprobe=128,quantizer_efSearch=512        0.8322 0.9669 0.9669      0.28930      209202695    2
nprobe=256,quantizer_efSearch=64         0.8436 0.9838 0.9838      0.46131      411230695    1
nprobe=256,quantizer_efSearch=128        0.8443 0.9865 0.9865      0.47599      410604900    1
```

</details>
<details><summary>`OPQ128_256,IVF65536(IVF256,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 858652176

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF65536_IVF256_PQ128x4fs_RFlat__PQ128x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5561 0.9390 0.9612      0.01627      216405474    19
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1474 0.1871 0.1881      0.00174        1833271    172
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1677 0.2161 0.2171      0.00182        1913241    165
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1780 0.2289 0.2299      0.00192        2077781    157
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1982 0.2630 0.2650      0.00205        3599889    147
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.2027 0.2697 0.2717      0.00207        3578861    146
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2320 0.3097 0.3117      0.00205        3678450    147
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2378 0.3193 0.3212      0.00212        3660428    142
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2459 0.3300 0.3320      0.00224        3831648    134
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2549 0.3430 0.3449      0.00224        3818257    135
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2567 0.3450 0.3468      0.00227        3809147    133
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2928 0.4120 0.4161      0.00263        7195942    115
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2974 0.4204 0.4247      0.00268        7122295    112
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2988 0.4221 0.4264      0.00277        7113384    109
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3288 0.4652 0.4695      0.00277        7268593    109
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3367 0.4737 0.4775      0.00291        7646322    104
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3466 0.4886 0.4929      0.00298        7578149    101
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3467 0.4893 0.4936      0.00329        7567854    92
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3508 0.4965 0.5008      0.00358        8211272    84
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3801 0.5657 0.5726      0.00361       14328185    84
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3886 0.5799 0.5873      0.00377       14164416    80
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4057 0.6032 0.6103      0.00390       14610509    77
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4151 0.6201 0.6276      0.00403       14447391    75
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.4153 0.6210 0.6287      0.00409       14436133    74
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4218 0.6329 0.6403      0.00434       15069608    70
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4236 0.6560 0.6671      0.00488       28238497    62
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4590 0.7156 0.7271      0.00511       28442108    59
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4630 0.7251 0.7363      0.00526       28042509    57
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4716 0.7363 0.7479      0.00548       29012027    55
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4777 0.7487 0.7599      0.00579       28591801    52
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4819 0.7537 0.7650      0.00638       29860991    48
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4954 0.8016 0.8161      0.00613       54965234    49
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5075 0.8245 0.8395      0.00648       56316461    47
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5128 0.8316 0.8469      0.00796       55382086    38
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.5142 0.8331 0.8482      0.00910       60038387    33
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5186 0.8401 0.8559      0.00840       56556895    36
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5318 0.8848 0.9039      0.00901      110096459    34
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5348 0.8890 0.9086      0.01041      108167736    29
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5401 0.8991 0.9186      0.01034      111108525    30
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5431 0.9030 0.9226      0.01136      109196876    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5436 0.9040 0.9235      0.01541      111641911    20
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5469 0.9213 0.9431      0.01487      215834804    21
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5482 0.9236 0.9455      0.01479      211656839    21
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5561 0.9390 0.9612      0.01457      216401782    21
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5574 0.9431 0.9655      0.01555      212245683    20
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5634 0.9609 0.9851      0.02411      424498110    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5642 0.9634 0.9878      0.02633      415879676    12
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5654 0.9699 0.9953      0.03804      831091743    8
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5655 0.9705 0.9957      0.03942      809731004    8
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5660 0.9731 0.9990      0.07248     1582679535    5
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5665 0.9735 0.9993      0.11647     3088601594    3
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5666 0.9737 0.9999      0.18455     6156480968    2
```

</details>
<details><summary>`OPQ128_256,IVF65536(IVF256,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 858646032

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF65536_IVF256_PQ128x4fs_RFlat__PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5791 0.6398 0.6398      0.01633       16371485    19
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=1     0.1804 0.1912 0.1912      0.00287        1828241    105
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.2252 0.2389 0.2389      0.00308        3064578    98
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3002 0.3223 0.3223      0.00327        3646690    92
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3225 0.3467 0.3467      0.00329        3810376    92
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3340 0.3593 0.3593      0.00347        4132843    87
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3343 0.3591 0.3591      0.00397        4788080    76
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3373 0.3626 0.3626      0.00399        4783977    76
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3926 0.4265 0.4265      0.00505        7112705    60
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.4526 0.4932 0.4932      0.00542        7567199    56
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4604 0.5012 0.5012      0.00577        8211781    52
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4617 0.5027 0.5027      0.00718        9496300    42
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.4694 0.5155 0.5155      0.01549       14078059    20
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.5634 0.6236 0.6236      0.01607       14473034    19
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.5671 0.6280 0.6280      0.01817       14437041    17
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.5782 0.6399 0.6399      0.01849       15054003    17
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.5791 0.6398 0.6398      0.01820       16366732    17
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5820 0.6436 0.6436      0.01721       16331644    18
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6744 0.7569 0.7569      0.02698       28687874    12
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6832 0.7659 0.7659      0.02994       29845454    11
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7453 0.8468 0.8468      0.06500       55597999    5
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.7523 0.8560 0.8560      0.06892       56526590    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7525 0.8560 0.8560      0.06649       59094676    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7991 0.9238 0.9238      0.12964      116768412    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.8024 0.9250 0.9250      0.12635      117161713    3
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.8289 0.9659 0.9660      0.23721      220462129    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.8379 0.9837 0.9837      0.46735      415730528    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.8396 0.9877 0.9877      0.44233      415520772    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8400 0.9880 0.9880      0.43862      420821997    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.8423 0.9898 0.9900      0.85499      809490680    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.8424 0.9898 0.9901      0.86483      812607003    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.8453 0.9959 0.9961      0.86293      814274734    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.8458 0.9985 0.9988      1.69746      795364034    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.8459 0.9986 0.9989      1.51449      802227908    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.8471 0.9989 0.9992      1.71860      799483700    1
```

</details>
<details><summary>`OPQ128_256,IVF65536,PQ128x4fs` </summary>
Index size 853312791

 code_size 64

 log filename: autotune.dbbigann10M.OPQ128_256_IVF65536_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5637 0.9627 0.9874      0.04058      410218176    8
nprobe=8                                 0.4246 0.6375 0.6451      0.02649       13683068    12
nprobe=16                                0.4821 0.7558 0.7671      0.02320       27172971    13
nprobe=32                                0.5194 0.8416 0.8572      0.02507       53836429    12
nprobe=128                               0.5577 0.9443 0.9672      0.03816      209127593    8
nprobe=256                               0.5637 0.9627 0.9874      0.04037      410218176    8
nprobe=512                               0.5655 0.9705 0.9960      0.05278      803574139    6
nprobe=1024                              0.5658 0.9731 0.9990      0.06964     1572049623    5
nprobe=2048                              0.5662 0.9736 0.9996      0.11003     3076341515    3
nprobe=4096                              0.5663 0.9737 0.9999      0.19168     6032223498    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 830086614

 code_size 64

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.2464 0.2704 0.2704      0.00656       21642493    46
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2        0.1359 0.1465 0.1465      0.00206         793700    146
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1448 0.1556 0.1556      0.00198        1143190    152
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4        0.1581 0.1694 0.1694      0.00210        1140797    143
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4       0.1602 0.1716 0.1716      0.00229        1139697    131
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.1864 0.2038 0.2038      0.00233        1058323    129
k_factor_rf=2,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.2200 0.2423 0.2423      0.00244        1232619    123
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2201 0.2417 0.2417      0.00238        1232283    127
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.3067 0.3470 0.3470      0.00270        2464168    112
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3256 0.3689 0.3689      0.00362        3141452    83
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4        0.3658 0.4202 0.4202      0.00417        4251221    72
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.4049 0.4668 0.4668      0.00391        7376507    77
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4376 0.5056 0.5056      0.00406        4876702    74
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.4438 0.5056 0.5056      0.00459        8819261    66
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.4469 0.5155 0.5155      0.00458        6197295    66
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5227 0.6141 0.6141      0.00590        8353472    51
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5631 0.6686 0.6686      0.00646       16722670    47
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.6159 0.7248 0.7248      0.00676       30136911    45
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.6590 0.7942 0.7942      0.00969       32686744    31
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.6653 0.8119 0.8119      0.01106       30140624    28
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.6932 0.8501 0.8501      0.01435       60288437    21
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6943 0.8554 0.8554      0.01466       56916681    21
k_factor_rf=2,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.6989 0.8444 0.8444      0.02339      149183814    13
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.7273 0.8960 0.8960      0.02621      149378066    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.7461 0.9336 0.9336      0.02752      111745236    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256    0.7527 0.9420 0.9421      0.03412      147031835    9
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.7565 0.9484 0.9485      0.04075      253763106    8
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7582 0.9556 0.9556      0.04225      116483711    8
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.7612 0.9565 0.9566      0.05638      218624607    6
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7621 0.9658 0.9660      0.06893      214413718    5
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7748 0.9816 0.9818      0.08246      418557853    4
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7752 0.9823 0.9827      0.07965      428070860    4
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7814 0.9941 0.9947      0.12056      818263294    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7815 0.9942 0.9948      0.14537      817605169    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7816 0.9943 0.9949      0.17920      837587952    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.7829 0.9970 0.9977      0.19719     1634043444    2
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.7830 0.9970 0.9977      0.59884      817597200    1
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.7834 0.9977 0.9984      0.67134     1579406341    1
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256 0.7835 0.9978 0.9985      1.47465     1598020990    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 810018898

 code_size 62

 log filename: autotune.dbbigann10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR72_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.2389 0.2703 0.2704      0.00610       21643341    50
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=8        0.1242 0.1371 0.1371      0.00167        1825724    180
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.1284 0.1406 0.1406      0.00167         614451    180
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1415 0.1554 0.1554      0.00153        1143245    197
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1417 0.1556 0.1556      0.00154        1143114    196
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2        0.1434 0.1574 0.1574      0.00158         791910    190
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4        0.1540 0.1694 0.1694      0.00168        1141378    179
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4       0.1558 0.1716 0.1716      0.00184        1140475    164
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.1821 0.2037 0.2038      0.00194        1058250    155
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2141 0.2416 0.2417      0.00197        1232379    152
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2361 0.2666 0.2667      0.00208        2260729    145
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.2965 0.3469 0.3470      0.00223        2464056    135
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.3001 0.3578 0.3578      0.00303        7182544    99
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3129 0.3687 0.3689      0.00275        3140449    109
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.3829 0.4668 0.4668      0.00324        7376228    93
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4151 0.5054 0.5056      0.00363        4873274    83
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.4191 0.5055 0.5056      0.00401        8819679    75
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.4237 0.5153 0.5155      0.00402        6199798    75
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.4796 0.5964 0.5964      0.00448       12300631    67
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.4857 0.6092 0.6093      0.00525       28376978    58
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5208 0.6684 0.6686      0.00549       16728544    55
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5617 0.7247 0.7248      0.00610       30130323    50
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5964 0.7938 0.7942      0.00854       32687691    36
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.6017 0.8114 0.8119      0.00957       30140624    32
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6264 0.8549 0.8554      0.01367       56920914    22
k_factor_rf=2,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.6307 0.8442 0.8444      0.02130      149378066    15
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.6515 0.8955 0.8960      0.02530      149270539    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6660 0.9325 0.9336      0.02761      111700546    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.6661 0.9335 0.9346      0.03064      111540396    10
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256    0.6716 0.9409 0.9421      0.03253      147319540    10
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6764 0.9541 0.9556      0.03909      116442960    8
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.6787 0.9552 0.9566      0.07163      218624607    5
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6879 0.9797 0.9818      0.07887      418557853    4
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6928 0.9924 0.9947      0.11962      818263294    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6929 0.9925 0.9948      0.15336      817605169    2
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6930 0.9926 0.9949      0.15962      837587952    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.6939 0.9953 0.9977      0.20016     1634043444    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.6942 0.9960 0.9984      0.60588     1579406341    1
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256 0.6943 0.9961 0.9985      1.44107     1598020990    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 763962002

 code_size 64

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7758 0.9801 0.9805      0.05775      411802082    6
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4872 0.5651 0.5652      0.00270       13799895    112
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4992 0.5777 0.5778      0.00286       13788664    105
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.5380 0.6264 0.6265      0.00328       27398099    92
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5638 0.6578 0.6579      0.00326       27363666    92
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5707 0.6673 0.6674      0.00360       27330522    84
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5969 0.6998 0.6999      0.00403       54262637    75
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.6011 0.7169 0.7170      0.00457       27330522    66
k_factor_rf=1,nprobe=64,quantizer_efSearch=8      0.6066 0.7126 0.7127      0.00485      107093294    62
k_factor_rf=1,nprobe=32,quantizer_efSearch=32     0.6175 0.7254 0.7255      0.00510       54083148    59
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.6343 0.7614 0.7615      0.00501       54262637    60
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.6510 0.7823 0.7824      0.00529       54165417    57
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.6567 0.7895 0.7896      0.00577       54083148    52
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.6734 0.8148 0.8149      0.00627      106986654    48
k_factor_rf=4,nprobe=64,quantizer_efSearch=8      0.6785 0.8308 0.8310      0.00865      107093294    35
k_factor_rf=2,nprobe=128,quantizer_efSearch=32    0.6933 0.8399 0.8400      0.00858      210192791    35
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.7035 0.8645 0.8647      0.00860      106986654    35
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.7184 0.8845 0.8847      0.01072      106684368    28
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.7298 0.9013 0.9015      0.01284      210192791    24
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.7352 0.9088 0.9090      0.01515      209794454    20
k_factor_rf=4,nprobe=256,quantizer_efSearch=64    0.7401 0.9156 0.9157      0.01924      411802082    16
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.7546 0.9424 0.9426      0.02150      209794454    14
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.7564 0.9443 0.9445      0.02132      209638731    15
k_factor_rf=8,nprobe=512,quantizer_efSearch=64    0.7613 0.9538 0.9541      0.02880      804264873    11
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.7636 0.9565 0.9567      0.02733      411122556    12
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.7720 0.9733 0.9736      0.03560      411802082    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7734 0.9757 0.9760      0.03729      411122556    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7742 0.9767 0.9770      0.04353      410914732    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.7758 0.9801 0.9805      0.05633      411802082    6
k_factor_rf=16,nprobe=1024,quantizer_efSearch=128 0.7762 0.9804 0.9808      0.05706     1560564160    6
k_factor_rf=16,nprobe=512,quantizer_efSearch=256  0.7774 0.9812 0.9816      0.05556      804402630    6
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.7782 0.9847 0.9852      0.06311      804264873    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7822 0.9904 0.9909      0.07525      804402630    4
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.7825 0.9909 0.9914      0.08873      804114064    4
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7830 0.9926 0.9931      0.12750     1572274616    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=512  0.7838 0.9935 0.9940      0.14253      804114064    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7847 0.9956 0.9961      0.14178     1573595282    3
k_factor_rf=64,nprobe=2048,quantizer_efSearch=256 0.7850 0.9962 0.9967      0.18248     3016335230    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7851 0.9965 0.9970      0.16946     1572274616    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.7856 0.9973 0.9978      0.29250     5803220898    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCA72,SQ6)` </summary>
Index size 743905806

 code_size 62

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCA72_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6797 0.9776 0.9804      0.05519      411801993    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3617 0.4376 0.4377      0.00187        6921357    161
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3873 0.4724 0.4725      0.00216        6908222    139
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4527 0.5659 0.5660      0.00238       13797433    126
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4623 0.5798 0.5799      0.00237       13788843    127
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4938 0.6262 0.6263      0.00276       27389614    109
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5165 0.6586 0.6587      0.00296       27362687    102
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5213 0.6674 0.6675      0.00320       27329400    94
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5433 0.6998 0.6999      0.00362       54266465    83
k_factor_rf=1,nprobe=64,quantizer_efSearch=8      0.5534 0.7122 0.7123      0.00436      107100914    69
k_factor_rf=1,nprobe=32,quantizer_efSearch=32     0.5610 0.7251 0.7252      0.00458       54084288    66
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5713 0.7610 0.7613      0.00478       54266465    63
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5835 0.7810 0.7813      0.00565       54166413    54
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.5887 0.7890 0.7893      0.00570       54084288    53
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.6023 0.8133 0.8135      0.00601      106989566    50
k_factor_rf=4,nprobe=64,quantizer_efSearch=8      0.6081 0.8295 0.8303      0.00837      107100914    36
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6257 0.8627 0.8635      0.00922      106989566    33
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6468 0.9005 0.9013      0.01144      210188955    27
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.6513 0.9079 0.9089      0.01237      209794863    25
k_factor_rf=4,nprobe=256,quantizer_efSearch=64    0.6556 0.9146 0.9157      0.01581      411801993    19
k_factor_rf=8,nprobe=256,quantizer_efSearch=32    0.6644 0.9395 0.9409      0.02085      412563928    15
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6658 0.9408 0.9425      0.01883      209794863    16
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6668 0.9421 0.9439      0.02047      209639535    15
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.6720 0.9548 0.9566      0.02759      411126150    11
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6783 0.9709 0.9735      0.03446      411801993    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6794 0.9731 0.9758      0.03925      411126150    8
k_factor_rf=16,nprobe=512,quantizer_efSearch=64   0.6798 0.9739 0.9765      0.03941      804281547    8
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.6801 0.9741 0.9768      0.04245      410918023    8
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6810 0.9799 0.9828      0.05722      411126150    6
k_factor_rf=16,nprobe=1024,quantizer_efSearch=128 0.6819 0.9778 0.9806      0.06153     1560592334    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.6820 0.9823 0.9853      0.06948      804281547    5
k_factor_rf=16,nprobe=512,quantizer_efSearch=512  0.6825 0.9794 0.9822      0.06846      804118212    5
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.6832 0.9795 0.9823      0.07033     1573615290    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6846 0.9876 0.9907      0.07410      804409413    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.6851 0.9884 0.9915      0.08785      804118212    4
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.6868 0.9901 0.9932      0.12627     1572286505    3
k_factor_rf=64,nprobe=2048,quantizer_efSearch=256 0.6872 0.9933 0.9965      0.17982     3016457790    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.6876 0.9939 0.9971      0.17850     1572286505    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.6882 0.9947 0.9979      0.28177     5803591932    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 743895822

 code_size 62

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCAR72_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6819 0.9769 0.9800      0.05379      411801836    6
k_factor_rf=1,nprobe=1,quantizer_efSearch=8       0.2068 0.2357 0.2358      0.00196        1726919    154
k_factor_rf=1,nprobe=2,quantizer_efSearch=4       0.2818 0.3307 0.3308      0.00178        3466567    169
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.2969 0.3486 0.3487      0.00205        3454999    147
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3645 0.4377 0.4378      0.00191        6921619    158
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4556 0.5661 0.5662      0.00237       13798908    127
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4947 0.6258 0.6259      0.00284       27396946    106
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5174 0.6575 0.6576      0.00307       27364481    98
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5236 0.6672 0.6673      0.00335       27329780    90
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5443 0.6994 0.6995      0.00366       54264653    83
k_factor_rf=1,nprobe=64,quantizer_efSearch=8      0.5529 0.7116 0.7117      0.00433      107104330    70
k_factor_rf=1,nprobe=32,quantizer_efSearch=32     0.5640 0.7253 0.7254      0.00435       54081073    69
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5733 0.7607 0.7609      0.00469       54264653    64
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5878 0.7813 0.7815      0.00497       54165089    61
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.6062 0.8135 0.8137      0.00591      106991053    51
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.6189 0.8329 0.8331      0.00798      106683221    38
k_factor_rf=2,nprobe=128,quantizer_efSearch=32    0.6212 0.8392 0.8394      0.00820      210188041    37
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6288 0.8628 0.8636      0.00808      106991053    38
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.6384 0.8778 0.8787      0.01021      210601875    30
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6416 0.8839 0.8846      0.01078      106683221    28
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6491 0.9004 0.9013      0.01121      210188041    27
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.6540 0.9077 0.9088      0.01494      209792707    21
k_factor_rf=4,nprobe=256,quantizer_efSearch=64    0.6579 0.9142 0.9154      0.01683      411801836    18
k_factor_rf=8,nprobe=256,quantizer_efSearch=32    0.6667 0.9390 0.9410      0.02160      412566819    15
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6678 0.9402 0.9422      0.02020      209792707    15
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6686 0.9418 0.9438      0.02328      209637617    13
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.6741 0.9541 0.9564      0.02822      411122424    11
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6800 0.9701 0.9731      0.03446      411801836    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6809 0.9724 0.9755      0.03745      411122424    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.6815 0.9734 0.9765      0.04348      410915139    8
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.6819 0.9769 0.9800      0.05608      411801836    6
k_factor_rf=16,nprobe=512,quantizer_efSearch=256  0.6830 0.9779 0.9811      0.05731      804408825    6
k_factor_rf=16,nprobe=1024,quantizer_efSearch=128 0.6832 0.9773 0.9804      0.05829     1560590316    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.6835 0.9802 0.9834      0.06292      410915139    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.6840 0.9818 0.9849      0.06534      804309964    5
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.6841 0.9791 0.9821      0.06988     1573613056    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6860 0.9871 0.9904      0.07594      804408825    4
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.6865 0.9878 0.9911      0.09138      804117644    4
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.6871 0.9903 0.9937      0.13228     1560590316    3
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.6878 0.9895 0.9929      0.14154     1572287147    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.6882 0.9923 0.9957      0.14427     1573613056    3
k_factor_rf=64,nprobe=2048,quantizer_efSearch=256 0.6885 0.9928 0.9963      0.18092     3016300421    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.6889 0.9934 0.9968      0.17068     1572287147    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.6894 0.9941 0.9976      0.28463     5803006537    2
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 747853526

 code_size 64

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.4343 0.5031 0.5033      0.03916       17209268    8
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4        0.3995 0.4546 0.4547      0.00287        7287113    105
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.4115 0.4731 0.4732      0.00305       14074276    99
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4       0.4703 0.5443 0.5444      0.00328       14200095    92
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4930 0.5709 0.5710      0.00336       28310080    90
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.5253 0.6088 0.6089      0.00384       55871566    79
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5737 0.6693 0.6694      0.00461       29984539    66
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.5867 0.6963 0.6964      0.00479       28149551    63
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.5947 0.6966 0.6967      0.00568       55037798    53
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.6203 0.7283 0.7284      0.00624       56687602    49
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.6297 0.7417 0.7418      0.00664      215729164    46
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6462 0.7618 0.7619      0.00746      212665622    41
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.6504 0.7825 0.7826      0.00664      108075409    46
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64      0.6534 0.7862 0.7863      0.00782       59333169    39
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.6786 0.8297 0.8298      0.00902      108073749    34
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.6823 0.8346 0.8347      0.01023      216259868    30
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.6921 0.8464 0.8465      0.00950      110194977    32
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.6984 0.8544 0.8546      0.01169      113681696    26
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.7232 0.8935 0.8936      0.01504      211819519    20
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.7367 0.9100 0.9102      0.01512      214997170    20
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128    0.7374 0.9121 0.9123      0.01983      219832056    16
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.7412 0.9159 0.9161      0.02816      809925416    11
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.7427 0.9257 0.9258      0.02798      211816518    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.7618 0.9540 0.9543      0.02531      414218458    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128    0.7645 0.9582 0.9584      0.02731      421410011    11
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7658 0.9603 0.9606      0.04706     1582162115    7
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7784 0.9842 0.9846      0.05824      416022312    6
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.7814 0.9888 0.9893      0.06913      829048260    5
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7842 0.9941 0.9946      0.16284     3082835226    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7863 0.9977 0.9982      0.21804     6156981296    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.7866 0.9985 0.9990      1.01066     6032533461    1
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 727785810

 code_size 62

 log filename: autotune.dbbigann10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR72_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.4067 0.5026 0.5032      0.03970       17209268    8
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4       0.2040 0.2317 0.2318      0.00210        2081919    143
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16       0.2645 0.3087 0.3087      0.00219        4825120    138
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.2990 0.3501 0.3502      0.00244        4792662    124
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4        0.3772 0.4545 0.4547      0.00241        7286760    125
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2        0.3837 0.4682 0.4683      0.00257       14116962    117
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.3873 0.4731 0.4732      0.00267       14074279    113
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4580 0.5709 0.5710      0.00292       28310080    103
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.4738 0.5960 0.5961      0.00299       27928605    101
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4847 0.6088 0.6089      0.00347       55871174    87
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5274 0.6693 0.6694      0.00421       29985953    72
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.5372 0.6960 0.6964      0.00410       28149376    74
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.5437 0.6966 0.6967      0.00521       55036311    58
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.5664 0.7283 0.7284      0.00587       56690967    52
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.5727 0.7417 0.7418      0.00591      215726781    51
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.5740 0.7561 0.7564      0.00704       60086268    43
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5869 0.7617 0.7619      0.00734      212667636    41
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64      0.5916 0.7860 0.7863      0.00739       59329235    41
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.6206 0.8456 0.8465      0.00809      110193873    38
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.6255 0.8537 0.8546      0.01216      113675292    25
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.6443 0.8923 0.8936      0.01366      211819722    23
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.6533 0.9091 0.9102      0.01389      214982520    22
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128    0.6553 0.9112 0.9123      0.02026      219810156    15
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.6572 0.9150 0.9161      0.02579      809913121    12
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.6586 0.9237 0.9258      0.02677      211816518    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.6727 0.9520 0.9543      0.02568      414218458    12
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.6738 0.9577 0.9603      0.03529      219796140    9
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128    0.6753 0.9561 0.9584      0.02569      421383274    12
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6774 0.9583 0.9606      0.04329     1582162115    7
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6837 0.9813 0.9845      0.05863      416021933    6
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.6862 0.9860 0.9892      0.06560      829048260    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=128  0.6871 0.9892 0.9925      0.13082      814090486    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6886 0.9910 0.9945      0.16135     3082835226    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6896 0.9946 0.9981      0.21103     6156981296    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.6900 0.9954 0.9989      0.97459     6032533461    1
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 749986034

 code_size 56

 log filename: autotune.dbbigann10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR64_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.2304 0.2702 0.2704      0.00628       21649435    48
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1        0.1022 0.1134 0.1135      0.00154         619601    195
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1363 0.1545 0.1547      0.00148        1141067    203
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1364 0.1547 0.1549      0.00155        1141176    194
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2        0.1387 0.1569 0.1571      0.00160         791125    188
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4        0.1509 0.1714 0.1715      0.00164        1140081    184
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1        0.1669 0.1910 0.1912      0.00180        1060599    167
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1        0.1726 0.1987 0.1989      0.00179        1060046    168
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.1746 0.2010 0.2012      0.00183        1058598    165
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2053 0.2407 0.2409      0.00193        1231376    156
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2301 0.2699 0.2701      0.00203        2259790    148
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.2810 0.3451 0.3455      0.00227        2463006    133
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.2974 0.3654 0.3658      0.00282        3138479    107
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.3591 0.4568 0.4572      0.00329        7376264    92
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.3895 0.5032 0.5037      0.00366        4874302    82
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.3921 0.5007 0.5010      0.00420        8824530    72
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.3960 0.5123 0.5128      0.00410        6197520    74
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.4472 0.5868 0.5872      0.00463       12309793    65
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.4517 0.6097 0.6105      0.00494        8352435    61
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.4836 0.6572 0.6579      0.00554       16759456    55
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5003 0.6787 0.6792      0.00631       30169258    48
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5207 0.7389 0.7402      0.00866       19181805    35
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.5256 0.7501 0.7514      0.00937       19125345    33
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.5386 0.7616 0.7624      0.01313       30067102    23
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5414 0.7674 0.7682      0.00991       32690389    31
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5496 0.7983 0.7998      0.00942       30176705    32
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5688 0.8358 0.8376      0.01283       56986326    24
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.5814 0.8608 0.8626      0.02457      149780715    13
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6046 0.9147 0.9177      0.02484      111859935    13
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.6065 0.9182 0.9214      0.02847      111563714    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256    0.6083 0.9239 0.9270      0.03211      147453572    10
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6153 0.9444 0.9488      0.03885      116498965    8
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.6154 0.9446 0.9491      0.05807      116473050    6
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6189 0.9579 0.9639      0.06390      214598076    5
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6210 0.9611 0.9659      0.07176      418584525    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.6240 0.9710 0.9769      0.08474      228655838    4
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6275 0.9786 0.9851      0.12374      818713753    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6277 0.9791 0.9856      0.18237      837633242    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.6306 0.9868 0.9940      0.20434     1639913674    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256 0.6307 0.9874 0.9946      1.48495     1598018561    1
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 862020310

 code_size 64

 log filename: autotune.dbbigann10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fs_Refine_OPQ48_96_PQ48_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.2418 0.2716 0.2716      0.00723       21662747    42
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.1415 0.1549 0.1549      0.00212        1141359    142
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.1428 0.1571 0.1571      0.00207         791089    145
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4      0.1558 0.1717 0.1717      0.00227        1138714    133
k_factor_rf=2,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.2132 0.2415 0.2415      0.00248        1231834    121
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.2134 0.2422 0.2422      0.00237        1231609    127
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.2406 0.2701 0.2701      0.00258        2259600    117
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3020 0.3488 0.3489      0.00260        2462939    116
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.3168 0.3666 0.3667      0.00321        3138690    94
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.3552 0.4172 0.4173      0.00377        4251704    80
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.4055 0.4944 0.4944      0.00405        7375708    75
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.4256 0.5073 0.5076      0.00406        4874649    74
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.4359 0.5174 0.5174      0.00435        8819791    70
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.4384 0.5210 0.5210      0.00466        8817775    65
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5158 0.6296 0.6296      0.00491       12306837    62
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.5456 0.6901 0.6901      0.00587       28411088    52
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.5507 0.6864 0.6867      0.00579       16761029    52
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6358 0.8170 0.8170      0.00629       30175092    48
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.6420 0.8312 0.8317      0.01034       30068407    30
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.6537 0.8427 0.8428      0.00980       37770653    31
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6722 0.8845 0.8851      0.01210       56986326    25
k_factor_rf=2,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256   0.7031 0.9294 0.9300      0.02294      149780715    14
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7127 0.9471 0.9481      0.02414      111859935    13
k_factor_rf=1,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256   0.7208 0.9495 0.9497      0.02842      254583151    11
k_factor_rf=1,nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7288 0.9632 0.9634      0.04403      439128929    7
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.7313 0.9783 0.9792      0.04873      248939126    7
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.7319 0.9795 0.9807      0.06016      218635569    5
k_factor_rf=2,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7384 0.9871 0.9880      0.07914      818713753    4
k_factor_rf=2,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.7385 0.9873 0.9882      0.07306      838751474    5
k_factor_rf=2,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7390 0.9876 0.9885      0.10658     1621258677    3
k_factor_rf=4,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7397 0.9941 0.9952      0.10751      817630231    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7403 0.9964 0.9983      0.12580      818713753    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.7404 0.9966 0.9985      0.14304      817630231    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.7405 0.9968 0.9987      0.16506      837633242    2
k_factor_rf=16,nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.7406 0.9961 0.9978      0.15376     1577181702    2
k_factor_rf=8,nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7408 0.9976 0.9992      0.18672     1579400121    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.7409 0.9974 0.9993      0.21304     1639913674    2
k_factor_rf=16,nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.7410 0.9981 0.9999      0.21041     1598033569    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ16x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 683976338

 code_size 56

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_HNSW32_PQ16x4fs_Refine_OPQ48_96_PQ48_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7287 0.9735 0.9743      0.05224      411820510    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3638 0.4255 0.4255      0.00223        6918566    135
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3889 0.4560 0.4560      0.00237        6906379    127
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4563 0.5428 0.5428      0.00269       13794047    112
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4665 0.5546 0.5546      0.00264       13785796    114
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4942 0.5895 0.5895      0.00293       27394726    103
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5164 0.6180 0.6180      0.00314       27364783    96
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5234 0.6271 0.6271      0.00346       27331871    87
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5355 0.6416 0.6416      0.00367       54255925    82
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5635 0.6925 0.6925      0.00442       27331871    68
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5876 0.7252 0.7253      0.00481       54255925    63
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.6022 0.7459 0.7460      0.00542       54155866    56
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.6087 0.7548 0.7549      0.00691       54080025    44
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.6128 0.7614 0.7615      0.00680      106983195    45
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.6303 0.7969 0.7971      0.00873       54155866    35
k_factor_rf=4,nprobe=64,quantizer_efSearch=8      0.6306 0.7983 0.7985      0.00857      107092717    35
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6539 0.8322 0.8324      0.00945      106983195    32
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.6553 0.8358 0.8360      0.01117      210589994    28
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6661 0.8506 0.8508      0.01121      106686907    27
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.6761 0.8663 0.8665      0.01348      209789060    23
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.6827 0.8866 0.8868      0.01520      106781216    20
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.6858 0.8912 0.8914      0.01587      106686907    19
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.7013 0.9178 0.9181      0.01892      209789060    16
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.7022 0.9188 0.9192      0.02325      209621699    13
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.7070 0.9256 0.9259      0.02712      411147370    12
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.7144 0.9459 0.9465      0.03053      209789060    10
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7152 0.9470 0.9477      0.03349      209621699    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.7217 0.9569 0.9575      0.03584      411820510    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7230 0.9599 0.9605      0.04002      411147370    8
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7239 0.9606 0.9612      0.04557      410944278    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.7287 0.9735 0.9743      0.05293      411820510    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.7299 0.9764 0.9772      0.05866      411147370    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7308 0.9772 0.9780      0.06277      410944278    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7330 0.9809 0.9818      0.07541      804461637    4
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.7333 0.9813 0.9822      0.08848      804185172    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.7359 0.9887 0.9897      0.10812      805525322    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=512  0.7374 0.9907 0.9917      0.12635      804185172    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7376 0.9915 0.9925      0.13368     1573715520    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7382 0.9922 0.9932      0.17594     1572391657    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCA64,SQ6)` </summary>
Index size 683923630

 code_size 56

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCA64_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6191 0.9674 0.9740      0.05385      411824023    6
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.1889 0.2224 0.2225      0.00169        1727694    178
k_factor_rf=1,nprobe=2,quantizer_efSearch=4       0.2648 0.3226 0.3229      0.00178        3462684    169
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3364 0.4243 0.4246      0.00202        6918894    149
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3587 0.4554 0.4557      0.00227        6907037    133
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4166 0.5418 0.5421      0.00305       13795910    99
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4250 0.5543 0.5546      0.00260       13787249    116
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4457 0.5884 0.5887      0.00281       27395272    107
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.4645 0.6173 0.6176      0.00299       27367417    101
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.4693 0.6265 0.6268      0.00329       27336414    92
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.4773 0.6413 0.6416      0.00364       54255435    83
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.4941 0.6820 0.6826      0.00416       27367417    73
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.4994 0.6919 0.6925      0.00421       27336414    72
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5168 0.7246 0.7254      0.00496       54255435    61
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5289 0.7444 0.7453      0.00543       54158945    56
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.5343 0.7536 0.7545      0.00661       54079731    46
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.5359 0.7603 0.7612      0.00594      106989104    51
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.5479 0.7953 0.7966      0.00711       54158945    43
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.5536 0.8052 0.8066      0.00751       54079731    40
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.5654 0.8304 0.8322      0.00864      106989104    35
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.5686 0.8344 0.8359      0.01035      210602839    29
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.5797 0.8573 0.8589      0.01146      210183029    27
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.5830 0.8646 0.8663      0.01361      209791985    23
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.5853 0.8835 0.8865      0.01478      106781930    21
k_factor_rf=8,nprobe=128,quantizer_efSearch=16    0.5864 0.8814 0.8842      0.01629      210602839    19
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.5879 0.8880 0.8911      0.01742      106687745    18
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6016 0.9142 0.9179      0.01987      209791985    16
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.6070 0.9223 0.9258      0.02854      411151786    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6093 0.9413 0.9463      0.03195      209791985    10
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.6094 0.9424 0.9476      0.03192      209624662    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6142 0.9519 0.9572      0.03415      411824023    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6154 0.9549 0.9604      0.03820      411151786    8
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.6158 0.9556 0.9611      0.04310      410948733    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.6191 0.9674 0.9740      0.05767      411824023    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6203 0.9702 0.9771      0.05891      411151786    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.6208 0.9710 0.9778      0.06259      410948733    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6219 0.9747 0.9817      0.08170      804473525    4
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.6221 0.9751 0.9821      0.09007      804198499    4
k_factor_rf=64,nprobe=256,quantizer_efSearch=128  0.6223 0.9764 0.9838      0.09328      411151786    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=64   0.6226 0.9783 0.9854      0.10197      804432943    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.6237 0.9820 0.9896      0.10319      805542271    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.6249 0.9826 0.9903      0.12298     1560827599    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.6260 0.9848 0.9925      0.14295     1573753864    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.6264 0.9855 0.9932      0.16846     1572423497    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 683921582

 code_size 56

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCAR64_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6251 0.9689 0.9739      0.05258      411824384    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3398 0.4253 0.4254      0.00188        6922662    160
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3626 0.4560 0.4561      0.00216        6908182    140
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4267 0.5545 0.5546      0.00251       13787832    120
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4463 0.5888 0.5889      0.00266       27398180    113
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.4659 0.6176 0.6177      0.00298       27367662    101
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.4718 0.6268 0.6269      0.00320       27333636    94
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.4786 0.6414 0.6416      0.00356       54257878    85
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.4964 0.6823 0.6828      0.00395       27367662    77
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5025 0.6919 0.6924      0.00426       27333636    71
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5215 0.7246 0.7250      0.00470       54257878    64
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5337 0.7453 0.7457      0.00502       54160095    60
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.5383 0.7539 0.7544      0.00559       54080989    54
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.5402 0.7606 0.7611      0.00646      106988533    47
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.5510 0.7779 0.7783      0.00781      106688229    39
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.5565 0.8052 0.8065      0.00823       54080989    37
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.5709 0.8311 0.8322      0.00852      106988533    36
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.5740 0.8349 0.8361      0.00985      210607874    31
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.5825 0.8491 0.8504      0.01019      106688229    30
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.5889 0.8648 0.8662      0.01260      209792181    24
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.5909 0.8841 0.8863      0.01553      106784424    20
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.5938 0.8886 0.8908      0.01525      106688229    20
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6063 0.9146 0.9177      0.01919      209792181    16
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6065 0.9157 0.9190      0.02184      209624520    14
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.6113 0.9231 0.9257      0.02845      411150237    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6146 0.9420 0.9461      0.03020      209792181    10
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.6149 0.9431 0.9475      0.03092      209624520    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6205 0.9532 0.9571      0.03361      411824384    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6216 0.9561 0.9603      0.03722      411150237    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.6220 0.9568 0.9610      0.04215      410948032    8
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.6251 0.9689 0.9739      0.05347      411824384    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6259 0.9716 0.9770      0.05586      411150237    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.6264 0.9724 0.9777      0.06214      410948032    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6276 0.9762 0.9816      0.07998      804470283    4
k_factor_rf=64,nprobe=256,quantizer_efSearch=128  0.6283 0.9777 0.9837      0.09211      411150237    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.6299 0.9833 0.9895      0.10828      805535338    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.6304 0.9840 0.9902      0.12860     1560899774    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.6314 0.9860 0.9924      0.13755     1573750410    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.6318 0.9867 0.9931      0.17113     1572422159    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 772090258

 code_size 64

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_HNSW32_PQ32x4fs_Refine_OPQ48_96_PQ48_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7317 0.9706 0.9708      0.03047      804454332    10
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.3055 0.3546 0.3546      0.00223        3451615    135
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3839 0.4564 0.4565      0.00241        6919550    125
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.4096 0.4879 0.4880      0.00255        6905911    118
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4923 0.6047 0.6047      0.00279       13794908    108
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.5043 0.6193 0.6193      0.00284       13785437    106
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.5579 0.6966 0.6966      0.00316       27395378    96
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5824 0.7319 0.7319      0.00331       27366694    91
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5901 0.7426 0.7426      0.00359       27335379    84
k_factor_rf=1,nprobe=32,quantizer_efSearch=4      0.5953 0.7534 0.7536      0.00386       54246142    78
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.6326 0.8081 0.8083      0.00412       54254910    73
k_factor_rf=1,nprobe=32,quantizer_efSearch=32     0.6549 0.8408 0.8410      0.00505       54080988    60
k_factor_rf=1,nprobe=64,quantizer_efSearch=8      0.6584 0.8496 0.8497      0.00507      107085066    60
k_factor_rf=1,nprobe=64,quantizer_efSearch=16     0.6821 0.8857 0.8858      0.00533      106984672    57
k_factor_rf=1,nprobe=64,quantizer_efSearch=32     0.6925 0.9016 0.9017      0.00607      106783176    50
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.6990 0.9162 0.9165      0.00836      106686820    36
k_factor_rf=2,nprobe=128,quantizer_efSearch=32    0.7147 0.9473 0.9477      0.00910      210185041    33
k_factor_rf=1,nprobe=128,quantizer_efSearch=64    0.7172 0.9450 0.9451      0.00894      209789705    34
k_factor_rf=2,nprobe=128,quantizer_efSearch=64    0.7208 0.9561 0.9566      0.01028      209789705    30
k_factor_rf=2,nprobe=256,quantizer_efSearch=32    0.7219 0.9608 0.9613      0.01246      412571875    25
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.7222 0.9607 0.9614      0.01382      209789705    22
k_factor_rf=2,nprobe=256,quantizer_efSearch=64    0.7304 0.9740 0.9745      0.01365      411816385    23
k_factor_rf=4,nprobe=256,quantizer_efSearch=64    0.7319 0.9790 0.9798      0.01692      411816385    18
k_factor_rf=4,nprobe=512,quantizer_efSearch=64    0.7350 0.9848 0.9857      0.02245      804396992    14
k_factor_rf=2,nprobe=512,quantizer_efSearch=128   0.7357 0.9836 0.9841      0.02369      805516463    13
k_factor_rf=2,nprobe=512,quantizer_efSearch=256   0.7369 0.9851 0.9856      0.03454      804454332    9
k_factor_rf=4,nprobe=512,quantizer_efSearch=256   0.7383 0.9906 0.9915      0.03799      804454332    8
k_factor_rf=8,nprobe=512,quantizer_efSearch=256   0.7390 0.9922 0.9932      0.04703      804454332    7
k_factor_rf=8,nprobe=512,quantizer_efSearch=512   0.7393 0.9926 0.9936      0.05975      804183023    6
k_factor_rf=8,nprobe=1024,quantizer_efSearch=256  0.7404 0.9951 0.9961      0.05841     1573708696    6
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.7406 0.9960 0.9971      0.07268     1573708696    5
k_factor_rf=8,nprobe=2048,quantizer_efSearch=256  0.7409 0.9957 0.9967      0.08496     3016577038    4
k_factor_rf=8,nprobe=1024,quantizer_efSearch=512  0.7410 0.9958 0.9968      0.09235     1572385329    4
k_factor_rf=16,nprobe=1024,quantizer_efSearch=512 0.7412 0.9967 0.9978      0.10454     1572385329    3
k_factor_rf=16,nprobe=2048,quantizer_efSearch=512 0.7418 0.9977 0.9988      0.15284     3075754890    2
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 667790322

 code_size 56

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR64_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.3839 0.4976 0.4982      0.02815       17183491    11
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.1960 0.2274 0.2275      0.00209        2080702    144
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.2479 0.2943 0.2943      0.00227        4825527    133
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=2      0.2552 0.3073 0.3074      0.00234        3654077    129
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.3109 0.3857 0.3858      0.00233        7369136    129
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.3482 0.4367 0.4368      0.00232        7293175    130
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.3531 0.4472 0.4473      0.00266       14129759    114
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.4054 0.5193 0.5194      0.00292       14194624    103
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4153 0.5332 0.5333      0.00298       28342896    101
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4292 0.5596 0.5597      0.00298       27960291    101
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4293 0.5583 0.5584      0.00348       55947994    87
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4876 0.6670 0.6674      0.00427       28177069    71
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4877 0.6538 0.6539      0.00611      216322363    50
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4958 0.6671 0.6672      0.00622       56688390    49
k_factor_rf=4,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4975 0.6989 0.6996      0.00650       28175488    47
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4976 0.6966 0.6976      0.00732       54975681    41
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.5392 0.7504 0.7510      0.00749       59389306    41
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.5643 0.8136 0.8147      0.00970      110351379    31
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64     0.5686 0.8208 0.8220      0.01170      113833437    26
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.5750 0.8375 0.8389      0.01532      108259210    20
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5790 0.8487 0.8501      0.01381      211830631    22
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5889 0.8649 0.8663      0.01408      215113915    22
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5927 0.8954 0.8984      0.02575      108491210    12
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.5957 0.8976 0.9003      0.02549      211819289    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6096 0.9209 0.9235      0.02702      414250692    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6121 0.9243 0.9272      0.02629      421627942    12
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.6149 0.9442 0.9486      0.03367      219789030    9
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6190 0.9538 0.9589      0.04932      220215800    7
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=32  0.6196 0.9539 0.9579      0.07880      414240862    4
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6267 0.9740 0.9793      0.06687      831736441    5
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.6289 0.9775 0.9827      0.12952     3082857910    3
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.6313 0.9862 0.9924      0.20594     6175336699    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=64 0.6318 0.9876 0.9939      0.23299     1578315391    2
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 755951830

 code_size 64

 log filename: autotune.dbbigann10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ32x4fs_Refine_OPQ48_96_PQ48_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.7314 0.9714 0.9716      0.02301      810225623    14
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.3947 0.4685 0.4686      0.00284        7293152    106
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.4152 0.5023 0.5023      0.00297       14129756    102
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.4767 0.5827 0.5827      0.00318       14194693    95
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.5097 0.6275 0.6275      0.00342       28342896    88
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.5340 0.6638 0.6639      0.00359       27960734    84
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.5568 0.6983 0.6984      0.00400       55947851    75
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.5740 0.7215 0.7217      0.00486       28177595    62
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5917 0.7426 0.7426      0.00478       30007385    63
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.6330 0.8058 0.8060      0.00609       55045306    50
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.6570 0.8435 0.8437      0.00693       56688228    44
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6966 0.9111 0.9112      0.00813      216320938    37
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7141 0.9418 0.9419      0.01078      212781284    28
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7213 0.9602 0.9608      0.01617      215104354    19
k_factor_rf=1,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7255 0.9622 0.9624      0.01824      414245900    17
k_factor_rf=1,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7278 0.9667 0.9669      0.02849      809193442    11
k_factor_rf=1,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7314 0.9714 0.9716      0.02350      810236545    13
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7381 0.9913 0.9922      0.02948      810249009    11
k_factor_rf=2,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7393 0.9901 0.9906      0.03954     1582584042    8
k_factor_rf=4,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7407 0.9957 0.9966      0.04036     1582583122    8
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7414 0.9972 0.9982      0.05263     1582624173    6
k_factor_rf=8,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7419 0.9980 0.9990      0.07542     3084109541    4
k_factor_rf=16,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7421 0.9989 1.0000      0.08705     3084109541    4
```

</details>
<details><summary>`OPQ56_112,IVF262144(IVF512,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 839733536

 code_size 63

 log filename: autotune.dbbigann10M.OPQ56_112_IVF262144_IVF512_PQ56x4fs_RFlat__PQ7+56.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.3660 0.3999 0.3999      0.01600       10553518    19
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=56,k_factor=2        0.1646 0.1730 0.1730      0.00388         705398    78
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=1        0.1700 0.1788 0.1788      0.00393        1393788    77
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1       0.1706 0.1795 0.1795      0.00429        2727888    70
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=1       0.1720 0.1807 0.1807      0.00430        2727680    70
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1        0.2228 0.2389 0.2389      0.00455        1392156    67
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=56,k_factor=8        0.2445 0.2617 0.2617      0.00499         705312    61
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=4        0.2581 0.2758 0.2758      0.00502        1393116    60
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=56,k_factor=1        0.3404 0.3722 0.3722      0.00504         705685    60
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=1        0.3530 0.3868 0.3868      0.00527        1393878    57
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=56,k_factor=1       0.4818 0.5459 0.5459      0.00642         706761    47
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1      0.5616 0.6455 0.6455      0.00772        2729120    39
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=2      0.5630 0.6476 0.6476      0.01065        5360744    29
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=1      0.6111 0.7145 0.7145      0.01052        5348875    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=2       0.6167 0.7250 0.7250      0.01265        1393616    24
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=1       0.6451 0.7667 0.7667      0.01501        1395392    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=2     0.6480 0.7662 0.7662      0.01782       20765938    17
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=2     0.6481 0.7670 0.7670      0.01787       20804490    17
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=1      0.6915 0.8260 0.8261      0.02050        5360744    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1     0.7073 0.8511 0.8512      0.02687        2731333    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=4      0.7074 0.8528 0.8529      0.02562       10553518    12
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=2     0.7253 0.8823 0.8824      0.02946        2731333    11
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=2     0.7441 0.9073 0.9074      0.03236       10553518    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=2    0.7463 0.9097 0.9098      0.04633       41216679    7
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=2     0.7502 0.9196 0.9197      0.05071        5360744    6
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=56,k_factor=4    0.7610 0.9423 0.9424      0.06365       41216679    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=8    0.7737 0.9613 0.9614      0.07607       41216679    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=8    0.7750 0.9631 0.9632      0.08232       41216679    4
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=4    0.7763 0.9673 0.9674      0.10602       20588513    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=56,k_factor=8    0.7798 0.9743 0.9744      0.12031       41216679    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=16  0.7868 0.9893 0.9894      0.21993       20804490    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=16  0.7919 0.9971 0.9973      0.44057       41216679    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=32  0.7920 0.9982 0.9984      0.48529       20804490    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=64  0.7921 0.9991 0.9993      0.87827       20804490    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=56,k_factor=64 0.7923 0.9994 0.9996      2.31080       41216679    1
```

</details>
<details><summary>`OPQ56_112,IVF65536_HNSW32,PQ7+56` </summary>
Index size 758008924

 code_size 63

 log filename: autotune.dbbigann10M.OPQ56_112_IVF65536_HNSW32_PQ7+56.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.8028 0.9542 0.9542      0.06619              0    5
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.2216 0.2357 0.2357      0.00492              0    62
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.4419 0.4861 0.4861      0.00510              0    59
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.5328 0.5963 0.5963      0.00534              0    57
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.5998 0.6781 0.6781      0.00644              0    47
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.6282 0.7110 0.7110      0.00670              0    45
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.6407 0.7264 0.7264      0.00706              0    43
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.6481 0.7346 0.7346      0.00800              0    38
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=1      0.6751 0.7709 0.7709      0.00902              0    34
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=1     0.7022 0.8043 0.8043      0.01023              0    30
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=1     0.7047 0.8069 0.8069      0.01200              0    26
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.7150 0.8262 0.8262      0.01235              0    25
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.7226 0.8358 0.8358      0.01310              0    23
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=2     0.7252 0.8384 0.8384      0.01482              0    21
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=2     0.7478 0.8727 0.8727      0.01701              0    18
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.7654 0.8952 0.8952      0.01934              0    16
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=2    0.7660 0.8957 0.8957      0.02274              0    14
nprobe=64,quantizer_efSearch=256,ht=56,k_factor=2    0.7661 0.8958 0.8958      0.03068              0    10
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.7852 0.9215 0.9215      0.02914              0    11
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=4    0.7972 0.9427 0.9427      0.03283              0    10
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=4    0.8016 0.9488 0.9488      0.03461              0    9
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.8027 0.9503 0.9503      0.03736              0    9
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.8059 0.9580 0.9580      0.04555              0    7
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.8070 0.9597 0.9597      0.04843              0    7
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=4    0.8117 0.9636 0.9636      0.05415              0    6
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=8    0.8188 0.9768 0.9768      0.06485              0    5
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=8   0.8203 0.9795 0.9795      0.06860              0    5
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.8230 0.9844 0.9844      0.09510              0    4
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=16  0.8233 0.9850 0.9850      0.11923              0    3
nprobe=512,quantizer_efSearch=64,ht=56,k_factor=16   0.8241 0.9868 0.9868      0.13136              0    3
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.8271 0.9917 0.9917      0.14702              0    3
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.8276 0.9925 0.9925      0.16176              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.8278 0.9932 0.9932      0.18830              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.8282 0.9939 0.9939      0.20337              0    2
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=16 0.8295 0.9952 0.9952      0.26872              0    2
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=64 0.8298 0.9967 0.9967      0.37165              0    1
nprobe=2048,quantizer_efSearch=256,ht=56,k_factor=32 0.8300 0.9965 0.9965      0.43574              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=32 0.8309 0.9982 0.9982      0.48771              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=64 0.8311 0.9991 0.9991      0.90029              0    1
```

</details>
<details><summary>`OPQ56_112,IVF65536(IVF256,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 742766240

 code_size 63

 log filename: autotune.dbbigann10M.OPQ56_112_IVF65536_IVF256_PQ56x4fs_RFlat__PQ7+56.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.8207 0.9782 0.9782      0.11137        2639517    3
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1        0.5203 0.5753 0.5753      0.00532         684398    57
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=1        0.5515 0.6135 0.6135      0.00563         683874    54
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=56,k_factor=1      0.6185 0.6962 0.6962      0.00694        1341019    44
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1      0.6478 0.7315 0.7315      0.00723        1343843    42
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1       0.6623 0.7502 0.7502      0.00928         685157    33
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=1     0.7083 0.8075 0.8075      0.01344       10326100    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.7227 0.8312 0.8312      0.01279        1344123    24
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=2      0.7266 0.8377 0.8377      0.01335        2639517    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=2      0.7280 0.8383 0.8383      0.01455        5210236    21
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=1      0.7402 0.8499 0.8499      0.01613        2639517    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.7604 0.8824 0.8824      0.01896        1345814    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=2     0.7683 0.8952 0.8952      0.02155       10316562    14
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=2     0.7824 0.9133 0.9133      0.02914        2632144    11
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=4     0.7976 0.9408 0.9408      0.03450        2639517    9
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=8     0.8032 0.9517 0.9517      0.04540        5210236    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8     0.8102 0.9620 0.9620      0.04788        2615710    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56,k_factor=8   0.8103 0.9630 0.9630      0.07661       10340174    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56,k_factor=4    0.8144 0.9671 0.9671      0.07071       10340174    5
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8     0.8207 0.9782 0.9782      0.07461        2623118    5
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=16    0.8227 0.9829 0.9829      0.09456        2639517    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8     0.8234 0.9829 0.9829      0.13377        2639517    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=16    0.8294 0.9939 0.9939      0.14655        5155803    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=16   0.8298 0.9946 0.9946      0.22853        5210236    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=32  0.8316 0.9983 0.9983      0.28064       10340174    2
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56,k_factor=32  0.8320 0.9992 0.9992      0.52496       10340174    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=56,k_factor=64 0.8321 0.9999 0.9999      1.61651       10340174    1
```

</details>
<details><summary>`OPQ64_128,IVF16384_HNSW32,PQ64` </summary>
Index size 733177008

 code_size 64

 log filename: autotune.dbbigann10M.OPQ64_128_IVF16384_HNSW32_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3839 0.4280 0.4280      0.01247       32770295    25
nprobe=1,quantizer_efSearch=4,ht=204      0.1231 0.1281 0.1281      0.00390       17363303    77
nprobe=1,quantizer_efSearch=4,ht=252      0.2439 0.2674 0.2674      0.00700       17363303    43
nprobe=1,quantizer_efSearch=16,ht=234     0.2584 0.2815 0.2815      0.00622       16700492    49
nprobe=2,quantizer_efSearch=16,ht=226     0.3662 0.4038 0.4038      0.00931       32297304    33
nprobe=2,quantizer_efSearch=8,ht=254      0.3840 0.4282 0.4282      0.01232       32770295    25
nprobe=2,quantizer_efSearch=32,ht=242     0.3873 0.4328 0.4328      0.01212       32139930    25
nprobe=4,quantizer_efSearch=32,ht=228     0.4891 0.5585 0.5585      0.01660       61710055    19
nprobe=4,quantizer_efSearch=32,ht=244     0.5071 0.5863 0.5863      0.02211       61710055    14
nprobe=8,quantizer_efSearch=64,ht=222     0.5587 0.6500 0.6501      0.02821      117430975    11
nprobe=8,quantizer_efSearch=32,ht=244     0.6026 0.7209 0.7210      0.03849      117761253    8
nprobe=8,quantizer_efSearch=128,ht=256    0.6031 0.7219 0.7220      0.04586      117381824    7
nprobe=16,quantizer_efSearch=64,ht=232    0.6662 0.8206 0.8207      0.06077      223283907    5
nprobe=16,quantizer_efSearch=128,ht=242   0.6747 0.8365 0.8366      0.07402      223096910    5
nprobe=16,quantizer_efSearch=32,ht=246    0.6754 0.8378 0.8379      0.07244      223956958    5
nprobe=16,quantizer_efSearch=256,ht=246   0.6760 0.8382 0.8383      0.08603      223085830    4
nprobe=32,quantizer_efSearch=32,ht=224    0.6842 0.8518 0.8520      0.09729      424105557    4
nprobe=32,quantizer_efSearch=128,ht=512   0.7181 0.9165 0.9169      0.10364      421930519    3
nprobe=64,quantizer_efSearch=32,ht=234    0.7306 0.9460 0.9468      0.22072      796769509    2
nprobe=64,quantizer_efSearch=512,ht=256   0.7393 0.9637 0.9647      0.31401      791341519    1
nprobe=128,quantizer_efSearch=64,ht=246   0.7465 0.9829 0.9842      0.49830     1482624054    1
nprobe=512,quantizer_efSearch=512,ht=512  0.7507 0.9976 0.9997      1.26713     5078944888    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.7508 0.9978 1.0000      7.73224    28117405154    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64` </summary>
Index size 927854768

 code_size 64

 log filename: autotune.dbbigann10M.OPQ64_128_IVF262144_HNSW32_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.1511 0.1558 0.1558      1.37078      315419832    1
nprobe=1,quantizer_efSearch=4,ht=256      0.1623 0.1670 0.1670      0.00347         434146    87
nprobe=2,quantizer_efSearch=4,ht=230      0.1822 0.1881 0.1881      0.00490         870080    62
nprobe=2,quantizer_efSearch=8,ht=254      0.2583 0.2678 0.2678      0.00620         866952    49
nprobe=4,quantizer_efSearch=4,ht=238      0.3013 0.3145 0.3145      0.00824        1742198    37
nprobe=4,quantizer_efSearch=32,ht=244     0.3627 0.3781 0.3781      0.01415        1729855    22
nprobe=8,quantizer_efSearch=32,ht=244     0.4774 0.5026 0.5026      0.02074        3456751    15
nprobe=16,quantizer_efSearch=16,ht=256    0.5998 0.6425 0.6425      0.03218        6922061    10
nprobe=32,quantizer_efSearch=128,ht=512   0.7075 0.7673 0.7673      0.03926       13723893    8
nprobe=64,quantizer_efSearch=512,ht=256   0.7780 0.8495 0.8495      0.18577       27182037    2
nprobe=128,quantizer_efSearch=64,ht=246   0.8123 0.8907 0.8907      0.23611       53784663    2
nprobe=512,quantizer_efSearch=512,ht=512  0.8818 0.9813 0.9813      0.33804      207843784    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8932 0.9965 0.9965      2.16096     1525762559    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64` </summary>
Index size 867536372

 code_size 64

 log filename: autotune.dbbigann10M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2431 0.2510 0.2510      0.00681        3588492    45
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=230       0.1097 0.1119 0.1119      0.00269         615827    112
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=230       0.1231 0.1256 0.1256      0.00310        1826473    97
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=246      0.1638 0.1677 0.1677      0.00382        3156403    79
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=234      0.1694 0.1749 0.1749      0.00453        1054909    67
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=240     0.2431 0.2510 0.2510      0.00686        3582519    44
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256      0.2658 0.2755 0.2755      0.00837       11410562    36
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=512      0.5441 0.5812 0.5812      0.01064        7698419    29
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=256    0.6094 0.6524 0.6524      0.03722       17418054    9
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=254      0.6665 0.7214 0.7214      0.05829       15213860    6
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=242    0.6693 0.7209 0.7209      0.06209       34557947    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=238     0.6972 0.7512 0.7512      0.11073       30066163    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256     0.7791 0.8531 0.8531      0.12371       37723723    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=246   0.7908 0.8652 0.8652      0.24235       56705268    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=512    0.8677 0.9648 0.9648      0.27369      219162544    2
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=512 0.8944 0.9997 0.9997      3.79806     1599479632    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64` </summary>
Index size 772113072

 code_size 64

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_HNSW32_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3361 0.3543 0.3543      0.00524        3446896    58
nprobe=1,quantizer_efSearch=4,ht=204      0.0555 0.0570 0.0570      0.00254        1724767    118
nprobe=1,quantizer_efSearch=4,ht=252      0.2104 0.2194 0.2194      0.00291        1724767    104
nprobe=1,quantizer_efSearch=4,ht=256      0.2112 0.2202 0.2202      0.00298        1724767    101
nprobe=2,quantizer_efSearch=4,ht=230      0.2632 0.2755 0.2755      0.00439        3451965    69
nprobe=2,quantizer_efSearch=8,ht=254      0.3368 0.3550 0.3550      0.00542        3446896    56
nprobe=4,quantizer_efSearch=4,ht=238      0.4002 0.4240 0.4240      0.00814        6890146    37
nprobe=4,quantizer_efSearch=32,ht=244     0.4620 0.4914 0.4914      0.01064        6860258    29
nprobe=8,quantizer_efSearch=32,ht=244     0.5833 0.6292 0.6292      0.01778       13705717    17
nprobe=8,quantizer_efSearch=128,ht=256    0.5944 0.6417 0.6417      0.02417       13701496    13
nprobe=16,quantizer_efSearch=32,ht=246    0.6904 0.7520 0.7520      0.03336       27228534    9
nprobe=32,quantizer_efSearch=128,ht=512   0.7753 0.8543 0.8543      0.03442       53926180    9
nprobe=256,quantizer_efSearch=32,ht=512   0.8623 0.9702 0.9702      0.19192      412495437    2
nprobe=512,quantizer_efSearch=512,ht=512  0.8805 0.9950 0.9950      0.41481      804633202    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8838 0.9996 0.9996      2.86390     5827043514    1
```

</details>
<details><summary>`OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64` </summary>
Index size 757164532

 code_size 64

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ64.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.1479 0.1529 0.1529      0.22731      212929862    2
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=174       0.0047 0.0051 0.0051      0.00274        2410856    110
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=208       0.0649 0.0660 0.0660      0.00293        1833848    103
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=244       0.1811 0.1871 0.1871      0.00308        1830853    98
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=242      0.2043 0.2108 0.2108      0.00311        1912289    97
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=252       0.3024 0.3169 0.3169      0.00505        3659131    60
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=222       0.3068 0.3196 0.3196      0.00820        7586217    37
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=256     0.3089 0.3257 0.3257      0.00848       13790749    36
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=254     0.3439 0.3618 0.3618      0.00721        8647327    42
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=240      0.4213 0.4453 0.4453      0.01007       12147891    30
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=250     0.4679 0.4981 0.4981      0.01195       17189462    26
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=236      0.5529 0.5904 0.5904      0.01728       16336714    18
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=248      0.5867 0.6324 0.6324      0.01702       15080809    18
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=234      0.6177 0.6627 0.6627      0.03066       28024255    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=240     0.6791 0.7351 0.7351      0.03375       32391047    9
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=244     0.6868 0.7453 0.7453      0.03171       28599377    10
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=256    0.7020 0.7650 0.7650      0.03654       37535262    9
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=512      0.7804 0.8642 0.8642      0.08490      108050471    4
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=512    0.8281 0.9235 0.9235      0.07586      111648324    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=250    0.8568 0.9605 0.9605      0.24633      214684745    2
nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=512 0.8836 0.9990 0.9990      0.92238     1582629966    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=512 0.8843 1.0000 1.0000      3.76673     6041063250    1
```

</details>
<details><summary>`OPQ64_128,IVF65536,PQ64` </summary>
Index size 754275579

 code_size 64

 log filename: autotune.dbbigann10M.OPQ64_128_IVF65536_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4504 0.4765 0.4765      0.03357       13683071    9
nprobe=1,ht=162                          0.0013 0.0014 0.0014      0.01304        1716239    24
nprobe=2,ht=198                          0.0552 0.0567 0.0567      0.02047        3433875    15
nprobe=2,ht=256                          0.3463 0.3635 0.3635      0.01522        3433875    20
nprobe=4,ht=254                          0.4733 0.5030 0.5030      0.01985        6851357    16
nprobe=4,ht=256                          0.4734 0.5033 0.5033      0.01918        6851357    16
nprobe=8,ht=512                          0.5967 0.6451 0.6451      0.01787       13683071    17
nprobe=32,ht=512                         0.7763 0.8572 0.8572      0.03870       53836701    8
nprobe=64,ht=512                         0.8291 0.9246 0.9246      0.05903      106344849    6
nprobe=256,ht=512                        0.8753 0.9875 0.9875      0.19554      410218047    2
nprobe=512,ht=248                        0.8779 0.9898 0.9898      0.90862      803574357    1
nprobe=512,ht=252                        0.8800 0.9939 0.9939      0.92734      803574357    1
nprobe=1024,ht=248                       0.8803 0.9928 0.9928      1.80736     1572050768    1
nprobe=4096,ht=512                       0.8840 1.0000 1.0000      2.92107     6032225893    1
```

</details>
<details><summary>`PCAR128,IVF16384_HNSW32,SQ4` </summary>
Index size 733114069

 code_size 64

 log filename: autotune.dbbigann10M.PCAR128_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.8148 0.9972 0.9974      0.76277     2744591033    1
nprobe=1,quantizer_efSearch=16           0.2632 0.2891 0.2891      0.00738       16687481    41
nprobe=2,quantizer_efSearch=4            0.3483 0.3908 0.3908      0.01225       33758578    25
nprobe=2,quantizer_efSearch=8            0.3834 0.4294 0.4294      0.01138       32738666    27
nprobe=2,quantizer_efSearch=16           0.3879 0.4344 0.4344      0.01157       32311944    26
nprobe=2,quantizer_efSearch=32           0.3890 0.4354 0.4354      0.01231       32132577    25
nprobe=4,quantizer_efSearch=8            0.4978 0.5710 0.5711      0.01934       63010864    16
nprobe=4,quantizer_efSearch=16           0.5110 0.5861 0.5862      0.01959       62101216    16
nprobe=4,quantizer_efSearch=32           0.5126 0.5874 0.5875      0.02025       61718196    15
nprobe=4,quantizer_efSearch=64           0.5127 0.5874 0.5875      0.03570       61622359    9
nprobe=8,quantizer_efSearch=4            0.5733 0.6710 0.6711      0.03465      120363670    9
nprobe=8,quantizer_efSearch=16           0.6113 0.7174 0.7175      0.03516      118559510    9
nprobe=8,quantizer_efSearch=64           0.6161 0.7222 0.7223      0.03740      117421400    9
nprobe=16,quantizer_efSearch=8           0.6781 0.8062 0.8063      0.07557      227470313    4
nprobe=16,quantizer_efSearch=16          0.6946 0.8279 0.8280      0.06486      225707868    5
nprobe=16,quantizer_efSearch=32          0.7047 0.8387 0.8388      0.06540      223886178    5
nprobe=16,quantizer_efSearch=64          0.7052 0.8396 0.8397      0.06503      223241391    5
nprobe=32,quantizer_efSearch=16          0.7496 0.9017 0.9018      0.11900      426763419    3
nprobe=32,quantizer_efSearch=64          0.7613 0.9172 0.9173      0.11725      422375156    3
nprobe=64,quantizer_efSearch=64          0.7924 0.9645 0.9646      0.21799      793168951    2
nprobe=64,quantizer_efSearch=128         0.7927 0.9646 0.9647      0.21736      791667676    2
nprobe=64,quantizer_efSearch=256         0.7928 0.9647 0.9648      0.23396      791365282    2
nprobe=64,quantizer_efSearch=512         0.7929 0.9648 0.9649      0.24601      791287243    2
nprobe=128,quantizer_efSearch=64         0.8065 0.9853 0.9854      0.40836     1482524809    1
nprobe=128,quantizer_efSearch=128        0.8072 0.9867 0.9868      0.40288     1478325864    1
nprobe=128,quantizer_efSearch=256        0.8074 0.9869 0.9870      0.40922     1477135111    1
nprobe=128,quantizer_efSearch=512        0.8075 0.9870 0.9871      0.42121     1476831728    1
nprobe=256,quantizer_efSearch=256        0.8147 0.9971 0.9973      0.76034     2745450138    1
nprobe=256,quantizer_efSearch=512        0.8148 0.9972 0.9974      0.76464     2744591033    1
nprobe=512,quantizer_efSearch=256        0.8157 0.9994 0.9996      1.36595     5082791020    1
nprobe=512,quantizer_efSearch=512        0.8158 0.9995 0.9997      1.37281     5078394677    1
nprobe=1024,quantizer_efSearch=512       0.8159 0.9998 1.0000      2.53325     9360527916    1
```

</details>
<details><summary>`PCAR128,IVF262144_HNSW32,SQ4` </summary>
Index size 927791829

 code_size 64

 log filename: autotune.dbbigann10M.PCAR128_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7515 0.9589 0.9606      0.11368      105770794    3
nprobe=1,quantizer_efSearch=4            0.1537 0.1681 0.1681      0.00277         434557    109
nprobe=2,quantizer_efSearch=4            0.2246 0.2515 0.2517      0.00282         870966    107
nprobe=4,quantizer_efSearch=4            0.3019 0.3481 0.3486      0.00300        1743241    100
nprobe=4,quantizer_efSearch=8            0.3355 0.3837 0.3842      0.00381        1737465    79
nprobe=8,quantizer_efSearch=4            0.4212 0.4942 0.4950      0.00387        3478167    78
nprobe=8,quantizer_efSearch=8            0.4294 0.5045 0.5053      0.00424        3473591    71
nprobe=16,quantizer_efSearch=8           0.5256 0.6319 0.6329      0.00564        6935087    54
nprobe=32,quantizer_efSearch=4           0.5469 0.6643 0.6656      0.00770       13801699    39
nprobe=32,quantizer_efSearch=8           0.5895 0.7218 0.7232      0.00825       13802129    37
nprobe=32,quantizer_efSearch=16          0.6119 0.7511 0.7525      0.01025       13769872    30
nprobe=32,quantizer_efSearch=32          0.6224 0.7621 0.7635      0.01169       13744308    26
nprobe=64,quantizer_efSearch=8           0.6307 0.7782 0.7797      0.01340       27346269    23
nprobe=64,quantizer_efSearch=16          0.6640 0.8245 0.8260      0.01430       27314398    21
nprobe=64,quantizer_efSearch=32          0.6801 0.8453 0.8468      0.01779       27242381    17
nprobe=128,quantizer_efSearch=16         0.6927 0.8678 0.8693      0.02176       54015873    14
nprobe=128,quantizer_efSearch=32         0.7161 0.9006 0.9021      0.02737       53912295    12
nprobe=128,quantizer_efSearch=64         0.7242 0.9120 0.9136      0.03009       53780180    10
nprobe=128,quantizer_efSearch=128        0.7260 0.9143 0.9159      0.03838       53714719    8
nprobe=256,quantizer_efSearch=32         0.7324 0.9282 0.9298      0.04215      106322456    8
nprobe=256,quantizer_efSearch=64         0.7460 0.9495 0.9511      0.04626      106134187    7
nprobe=256,quantizer_efSearch=128        0.7505 0.9567 0.9584      0.05821      105890336    6
nprobe=256,quantizer_efSearch=256        0.7513 0.9584 0.9601      0.07682      105791427    5
nprobe=512,quantizer_efSearch=64         0.7553 0.9650 0.9666      0.08499      208390999    4
nprobe=512,quantizer_efSearch=128        0.7619 0.9761 0.9778      0.09603      208354749    4
nprobe=512,quantizer_efSearch=256        0.7636 0.9790 0.9807      0.11836      207947115    3
nprobe=1024,quantizer_efSearch=128       0.7659 0.9831 0.9848      0.20809      406546526    2
nprobe=1024,quantizer_efSearch=256       0.7682 0.9879 0.9896      0.21484      408021780    2
nprobe=1024,quantizer_efSearch=512       0.7688 0.9890 0.9907      0.25846      407496910    2
nprobe=2048,quantizer_efSearch=256       0.7704 0.9916 0.9933      0.32906      789480313    1
nprobe=2048,quantizer_efSearch=512       0.7713 0.9938 0.9955      0.40224      797880093    1
nprobe=4096,quantizer_efSearch=512       0.7716 0.9944 0.9961      0.67566     1525622614    1
```

</details>
<details><summary>`PCAR128,IVF262144(IVF512,PQ64x4fs,RFlat),SQ4` </summary>
Index size 867474457

 code_size 64

 log filename: autotune.dbbigann10M.PCAR128_IVF262144_IVF512_PQ64x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.7661 0.9737 0.9746      0.16396      425546607    2
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1342 0.1456 0.1456      0.00226         616550    133
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1366 0.1483 0.1483      0.00224         615370    135
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1735 0.1914 0.1914      0.00233        1063517    129
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2007 0.2233 0.2233      0.00231        1236610    130
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2128 0.2363 0.2363      0.00241        1583956    125
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2275 0.2543 0.2543      0.00247        1579114    122
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.2446 0.2780 0.2780      0.00260        1934363    116
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3021 0.3409 0.3409      0.00256        2471176    118
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3255 0.3706 0.3706      0.00263        2458096    114
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3289 0.3746 0.3746      0.00278        2453281    108
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3913 0.4521 0.4523      0.00291        4239873    103
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4134 0.4834 0.4838      0.00354        4196549    85
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4333 0.5064 0.5066      0.00332        4872951    91
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4496 0.5262 0.5266      0.00426        6187536    71
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4694 0.5547 0.5551      0.00459        7785795    66
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5041 0.5961 0.5965      0.00468        8435275    65
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5246 0.6248 0.6254      0.00473        8354460    64
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5427 0.6457 0.6463      0.00521        9663271    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5435 0.6482 0.6488      0.00828        9634557    37
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5445 0.6487 0.6494      0.00776        9639985    39
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5457 0.6484 0.6490      0.00664       12284410    46
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6001 0.7239 0.7245      0.00767       16720604    40
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6285 0.7626 0.7633      0.00924       19124843    33
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.6321 0.7677 0.7684      0.01456       19056808    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6351 0.7704 0.7711      0.01162       24250818    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6747 0.8344 0.8351      0.01275       30066780    24
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6871 0.8499 0.8506      0.01430       32619812    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6892 0.8528 0.8535      0.02335       32556282    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6927 0.8561 0.8568      0.02004       37706899    15
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7207 0.8986 0.8993      0.02452       65410725    13
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7337 0.9193 0.9200      0.03204       64338611    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7341 0.9199 0.9206      0.03204       64240235    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7343 0.9204 0.9211      0.03759       74391723    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7344 0.9207 0.9214      0.04020       94897532    8
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7515 0.9476 0.9483      0.05131      129007645    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.7537 0.9518 0.9525      0.05337      111424700    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7622 0.9620 0.9627      0.04890      116302656    7
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.7623 0.9621 0.9628      0.06658      116345843    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7629 0.9632 0.9639      0.06726      146940592    5
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7688 0.9751 0.9760      0.09567      223622584    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7731 0.9813 0.9821      0.11537      218513606    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.7737 0.9826 0.9834      0.12968      228519228    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.7778 0.9899 0.9908      0.18838      438377698    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7797 0.9925 0.9934      0.18506      428311094    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7825 0.9974 0.9984      0.32116      818396374    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.7833 0.9983 0.9993      0.60316     1620663603    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7834 0.9984 0.9994      0.61118     1581437586    1
```

</details>
<details><summary>`PCAR128,IVF65536_HNSW32,SQ4` </summary>
Index size 772050133

 code_size 64

 log filename: autotune.dbbigann10M.PCAR128_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7680 0.9851 0.9861      0.17360      410829623    2
nprobe=1,quantizer_efSearch=4            0.1967 0.2213 0.2215      0.00236        1724853    127
nprobe=2,quantizer_efSearch=4            0.2892 0.3323 0.3326      0.00261        3452952    116
nprobe=2,quantizer_efSearch=8            0.3104 0.3559 0.3562      0.00283        3446113    107
nprobe=2,quantizer_efSearch=16           0.3152 0.3614 0.3617      0.00343        3440959    88
nprobe=4,quantizer_efSearch=4            0.3839 0.4512 0.4518      0.00368        6893466    82
nprobe=4,quantizer_efSearch=8            0.4186 0.4908 0.4914      0.00399        6878867    76
nprobe=4,quantizer_efSearch=16           0.4262 0.4989 0.4995      0.00456        6865376    66
nprobe=4,quantizer_efSearch=32           0.4283 0.5013 0.5019      0.00553        6860086    55
nprobe=8,quantizer_efSearch=4            0.5053 0.6079 0.6087      0.00634       13753210    48
nprobe=8,quantizer_efSearch=8            0.5177 0.6230 0.6238      0.00644       13742621    47
nprobe=8,quantizer_efSearch=16           0.5299 0.6373 0.6381      0.00692       13722509    44
nprobe=8,quantizer_efSearch=32           0.5329 0.6411 0.6419      0.00790       13705600    38
nprobe=8,quantizer_efSearch=64           0.5332 0.6416 0.6424      0.00969       13703085    31
nprobe=16,quantizer_efSearch=4           0.5750 0.7012 0.7020      0.01088       27339527    28
nprobe=16,quantizer_efSearch=8           0.6037 0.7388 0.7396      0.01112       27303803    27
nprobe=16,quantizer_efSearch=16          0.6160 0.7536 0.7544      0.01150       27266593    27
nprobe=16,quantizer_efSearch=32          0.6226 0.7615 0.7623      0.01249       27228542    25
nprobe=16,quantizer_efSearch=64          0.6232 0.7625 0.7633      0.01429       27218548    21
nprobe=32,quantizer_efSearch=64          0.6827 0.8525 0.8534      0.02351       53935843    13
nprobe=32,quantizer_efSearch=128         0.6829 0.8529 0.8538      0.02677       53926037    12
nprobe=64,quantizer_efSearch=8           0.6840 0.8568 0.8577      0.03837      106958748    8
nprobe=64,quantizer_efSearch=16          0.7130 0.8974 0.8983      0.03813      106858049    8
nprobe=64,quantizer_efSearch=32          0.7259 0.9164 0.9173      0.03896      106673261    8
nprobe=64,quantizer_efSearch=64          0.7276 0.9199 0.9209      0.04051      106570910    8
nprobe=64,quantizer_efSearch=128         0.7286 0.9209 0.9219      0.04408      106532031    7
nprobe=64,quantizer_efSearch=256         0.7287 0.9210 0.9220      0.05199      106524831    6
nprobe=128,quantizer_efSearch=16         0.7305 0.9254 0.9264      0.07243      210453553    5
nprobe=128,quantizer_efSearch=32         0.7493 0.9539 0.9549      0.07315      210052473    5
nprobe=128,quantizer_efSearch=64         0.7533 0.9619 0.9629      0.07503      209658878    4
nprobe=128,quantizer_efSearch=128        0.7542 0.9635 0.9645      0.07865      209504532    4
nprobe=128,quantizer_efSearch=256        0.7543 0.9637 0.9647      0.08627      209477963    4
nprobe=128,quantizer_efSearch=512        0.7545 0.9641 0.9651      0.10761      209469114    3
nprobe=256,quantizer_efSearch=32         0.7583 0.9688 0.9698      0.14198      412505132    3
nprobe=256,quantizer_efSearch=64         0.7656 0.9815 0.9825      0.14496      411748641    3
nprobe=256,quantizer_efSearch=128        0.7671 0.9841 0.9851      0.14611      411079012    3
nprobe=256,quantizer_efSearch=256        0.7676 0.9847 0.9857      0.15209      410875673    2
nprobe=256,quantizer_efSearch=512        0.7680 0.9851 0.9861      0.17599      410829623    2
nprobe=512,quantizer_efSearch=128        0.7706 0.9920 0.9930      0.27703      805899127    2
nprobe=512,quantizer_efSearch=256        0.7712 0.9927 0.9937      0.28829      804892025    2
nprobe=512,quantizer_efSearch=512        0.7717 0.9936 0.9946      0.30846      804622275    1
nprobe=1024,quantizer_efSearch=128       0.7722 0.9942 0.9952      0.51887     1563300071    1
nprobe=1024,quantizer_efSearch=256       0.7733 0.9958 0.9968      0.54448     1575214083    1
nprobe=1024,quantizer_efSearch=512       0.7740 0.9971 0.9981      0.56757     1573955035    1
nprobe=2048,quantizer_efSearch=512       0.7745 0.9980 0.9990      1.06688     3079826647    1
nprobe=4096,quantizer_efSearch=512       0.7746 0.9982 0.9992      1.97913     5826917879    1
```

</details>
<details><summary>`PCAR128,IVF65536(IVF256,PQ64x4fs,RFlat),SQ4` </summary>
Index size 757103641

 code_size 64

 log filename: autotune.dbbigann10M.PCAR128_IVF65536_IVF256_PQ64x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.6145 0.7591 0.7603      0.03370      110400465    9
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.2064 0.2312 0.2315      0.00256        2076473    118
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2127 0.2382 0.2385      0.00261        2407475    115
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3036 0.3465 0.3469      0.00279        3809493    108
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3126 0.3571 0.3575      0.00266        4135727    113
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.3144 0.3591 0.3595      0.00299        4130845    101
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3159 0.3610 0.3614      0.00305        4787086    99
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.3173 0.3625 0.3629      0.00347        4781501    87
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3619 0.4221 0.4227      0.00374        7132455    81
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3976 0.4664 0.4670      0.00387        7276266    78
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4007 0.4709 0.4715      0.00367        7258858    82
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4187 0.4929 0.4935      0.00399        7567376    76
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4239 0.4989 0.4995      0.00405        8212889    75
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.4257 0.5010 0.5016      0.00532        8207548    57
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.4258 0.5011 0.5017      0.00512        8209285    59
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4267 0.5018 0.5024      0.00509        9496898    59
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4272 0.5023 0.5029      0.00529        9495410    57
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4934 0.5900 0.5907      0.00576       14593747    53
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5248 0.6319 0.6328      0.00631       15089743    48
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5278 0.6358 0.6367      0.00696       16367770    44
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5328 0.6420 0.6428      0.00701       16337287    43
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.5336 0.6429 0.6437      0.00713       16330642    43
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.5384 0.6523 0.6532      0.01002       28238903    30
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.6009 0.7344 0.7354      0.00995       28072011    31
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6228 0.7616 0.7626      0.01075       29898486    28
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6249 0.7651 0.7661      0.01096       29850397    28
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.6590 0.8147 0.8159      0.01815       55006214    17
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6668 0.8256 0.8268      0.01821       56291173    17
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6801 0.8459 0.8470      0.01847       55385011    17
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6863 0.8550 0.8561      0.01898       56560811    16
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.7200 0.9069 0.9082      0.03450      108247718    9
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7294 0.9223 0.9236      0.03609      111652792    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.7295 0.9221 0.9234      0.03712      111630133    9
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7297 0.9224 0.9237      0.03771      116775150    8
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=128  0.7298 0.9224 0.9237      0.04783      116770110    7
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.7419 0.9436 0.9450      0.06605      211757392    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.7554 0.9635 0.9649      0.06497      212364270    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.7557 0.9642 0.9656      0.06641      212210865    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7564 0.9658 0.9672      0.06800      219623760    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7688 0.9867 0.9881      0.12482      421041936    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7691 0.9866 0.9880      0.13181      420820395    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7725 0.9940 0.9955      0.24053      809958226    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7728 0.9944 0.9959      0.24224      814553034    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7729 0.9944 0.9959      0.24692      814267541    2
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.7744 0.9970 0.9985      0.47219     1579510997    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7748 0.9975 0.9990      0.46730     1582976700    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7753 0.9982 0.9997      0.87938     3086525978    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.7754 0.9985 1.0000      1.73670     6164833641    1
```

</details>
<details><summary>`PCAR128,IVF65536,SQ4` </summary>
Index size 754212640

 code_size 64

 log filename: autotune.dbbigann10M.PCAR128_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7738 0.9860 0.9875      0.13173      410218775    3
nprobe=1                                 0.2149 0.2390 0.2392      0.01253        1716246    24
nprobe=4                                 0.4308 0.5029 0.5040      0.01398        6851311    22
nprobe=8                                 0.5389 0.6440 0.6451      0.01556       13683067    20
nprobe=16                                0.6300 0.7659 0.7671      0.01949       27173044    16
nprobe=32                                0.6920 0.8560 0.8572      0.02813       53836614    11
nprobe=64                                0.7350 0.9234 0.9246      0.04287      106344840    7
nprobe=128                               0.7611 0.9660 0.9673      0.08336      209127551    4
nprobe=256                               0.7738 0.9860 0.9875      0.13120      410218775    3
nprobe=512                               0.7774 0.9945 0.9961      0.24416      803574263    2
nprobe=1024                              0.7793 0.9974 0.9991      0.46884     1572050712    1
nprobe=2048                              0.7797 0.9980 0.9997      0.90630     3076343122    1
nprobe=4096                              0.7798 0.9983 1.0000      1.73487     6032225918    1
```

</details>
<details><summary>`PCAR32,IVF16384_HNSW32,SQfp16` </summary>
Index size 726772053

 code_size 64

 log filename: autotune.dbbigann10M.PCAR32_IVF16384_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2347 0.5005 0.5371      0.00552       40132890    55
nprobe=1,quantizer_efSearch=16           0.1553 0.2661 0.2724      0.00260       10509233    116
nprobe=2,quantizer_efSearch=8            0.1999 0.3804 0.3964      0.00296       20628288    102
nprobe=2,quantizer_efSearch=16           0.2021 0.3851 0.4012      0.00332       20507336    91
nprobe=2,quantizer_efSearch=32           0.2028 0.3859 0.4020      0.00365       20470241    83
nprobe=4,quantizer_efSearch=4            0.2222 0.4624 0.4949      0.00455       40498310    66
nprobe=4,quantizer_efSearch=8            0.2347 0.5005 0.5371      0.00462       40132890    65
nprobe=4,quantizer_efSearch=16           0.2377 0.5093 0.5463      0.00494       39902665    61
nprobe=4,quantizer_efSearch=32           0.2378 0.5103 0.5474      0.00536       39831052    56
nprobe=8,quantizer_efSearch=8            0.2661 0.6086 0.6734      0.00783       77449412    39
nprobe=8,quantizer_efSearch=16           0.2719 0.6246 0.6922      0.00808       77065733    38
nprobe=8,quantizer_efSearch=32           0.2730 0.6269 0.6947      0.00872       76894289    35
nprobe=16,quantizer_efSearch=4           0.2804 0.6614 0.7494      0.01365      149151527    22
nprobe=16,quantizer_efSearch=8           0.2927 0.6923 0.7909      0.01384      149055455    22
nprobe=16,quantizer_efSearch=16          0.2967 0.7037 0.8063      0.01397      148743375    22
nprobe=16,quantizer_efSearch=64          0.2974 0.7064 0.8095      0.01586      148193810    19
nprobe=16,quantizer_efSearch=32          0.2975 0.7063 0.8096      0.01455      148330272    21
nprobe=16,quantizer_efSearch=128         0.2976 0.7064 0.8094      0.01815      148161289    17
nprobe=32,quantizer_efSearch=64          0.3099 0.7630 0.9020      0.02669      285416178    12
nprobe=64,quantizer_efSearch=32          0.3148 0.7887 0.9508      0.04921      550237812    7
nprobe=64,quantizer_efSearch=128         0.3149 0.7903 0.9523      0.05094      549021809    6
nprobe=128,quantizer_efSearch=32         0.3156 0.7970 0.9721      0.09068     1056853307    4
nprobe=128,quantizer_efSearch=128        0.3158 0.8001 0.9764      0.09320     1053686796    4
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQfp16` </summary>
Index size 827077973

 code_size 64

 log filename: autotune.dbbigann10M.PCAR32_IVF262144_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3104 0.7874 0.9428      0.05639      105207031    6
nprobe=1,quantizer_efSearch=4            0.1036 0.1476 0.1482      0.00185         437402    162
nprobe=2,quantizer_efSearch=4            0.1463 0.2299 0.2316      0.00193         874659    156
nprobe=4,quantizer_efSearch=4            0.1808 0.3192 0.3253      0.00203        1747275    148
nprobe=8,quantizer_efSearch=4            0.2318 0.4514 0.4674      0.00231        3485907    131
nprobe=8,quantizer_efSearch=8            0.2367 0.4598 0.4761      0.00237        3480413    127
nprobe=16,quantizer_efSearch=4           0.2555 0.5376 0.5721      0.00255        6948705    118
nprobe=16,quantizer_efSearch=8           0.2650 0.5603 0.5974      0.00286        6939287    105
nprobe=16,quantizer_efSearch=16          0.2668 0.5672 0.6060      0.00316        6930671    95
nprobe=32,quantizer_efSearch=4           0.2676 0.5944 0.6489      0.00364       13775913    83
nprobe=32,quantizer_efSearch=8           0.2785 0.6358 0.6982      0.00384       13782502    79
nprobe=32,quantizer_efSearch=16          0.2835 0.6549 0.7218      0.00431       13757223    70
nprobe=32,quantizer_efSearch=32          0.2863 0.6618 0.7288      0.00498       13735807    61
nprobe=32,quantizer_efSearch=64          0.2875 0.6636 0.7308      0.00693       13723893    44
nprobe=64,quantizer_efSearch=16          0.2943 0.7089 0.8052      0.00668       27230963    45
nprobe=64,quantizer_efSearch=32          0.2992 0.7210 0.8228      0.00773       27174649    39
nprobe=64,quantizer_efSearch=64          0.2999 0.7246 0.8270      0.00877       27143526    35
nprobe=128,quantizer_efSearch=32         0.3052 0.7550 0.8846      0.01250       53681422    24
nprobe=128,quantizer_efSearch=64         0.3069 0.7631 0.8973      0.01378       53566472    22
nprobe=128,quantizer_efSearch=128        0.3076 0.7645 0.8998      0.01729       53510077    18
nprobe=256,quantizer_efSearch=64         0.3084 0.7827 0.9358      0.02285      105520821    14
nprobe=256,quantizer_efSearch=128        0.3097 0.7859 0.9408      0.02688      105298402    12
nprobe=256,quantizer_efSearch=256        0.3104 0.7871 0.9425      0.03294      105222432    10
nprobe=512,quantizer_efSearch=256        0.3107 0.7983 0.9680      0.06066      206516026    5
nprobe=512,quantizer_efSearch=512        0.3108 0.7987 0.9689      0.07722      206402278    4
nprobe=1024,quantizer_efSearch=256       0.3114 0.8023 0.9792      0.10583      404674677    3
nprobe=1024,quantizer_efSearch=512       0.3115 0.8027 0.9803      0.13773      404089418    3
nprobe=2048,quantizer_efSearch=256       0.3116 0.8023 0.9832      0.18696      779654336    2
nprobe=2048,quantizer_efSearch=512       0.3118 0.8037 0.9859      0.24695      790394603    2
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 760067737

 code_size 64

 log filename: autotune.dbbigann10M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.3100 0.7935 0.9619      0.07556      425212645    4
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.0713 0.0980 0.0982      0.00218        1146242    138
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.1004 0.1498 0.1506      0.00222        1245963    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.1181 0.1826 0.1844      0.00222        1062179    135
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1    0.1191 0.1846 0.1863      0.00226        1060037    133
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1281 0.1965 0.1977      0.00226        1588175    133
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1387 0.2156 0.2178      0.00224        1237378    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1475 0.2328 0.2349      0.00226        1582840    133
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1709 0.2946 0.2998      0.00237        2126990    127
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1845 0.3225 0.3286      0.00236        2469310    127
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1927 0.3358 0.3420      0.00241        2464769    125
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2182 0.4144 0.4284      0.00254        4908652    119
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2227 0.4254 0.4405      0.00259        4224283    116
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.2286 0.4518 0.4724      0.00264        8461223    114
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2332 0.4552 0.4717      0.00289        6220136    104
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2521 0.5331 0.5652      0.00301        9718066    100
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.2625 0.5520 0.5879      0.00312        8351830    97
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2654 0.5612 0.5974      0.00321        9685166    94
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2674 0.5662 0.6037      0.00359        9664859    84
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2731 0.6054 0.6611      0.00404       15313787    75
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2774 0.6282 0.6882      0.00511       15241921    59
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2806 0.6313 0.6894      0.00523       19201761    58
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2886 0.6653 0.7322      0.00621       19088423    49
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2937 0.6987 0.7885      0.00677       30180497    45
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2969 0.7067 0.7979      0.00723       32745080    42
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3000 0.7235 0.8237      0.00890       32592383    34
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3011 0.7255 0.8254      0.00904       37754773    34
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.3026 0.7483 0.8751      0.01323       56585803    23
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3065 0.7569 0.8837      0.01392       64478484    22
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3081 0.7664 0.8994      0.01487       64182997    21
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.3085 0.7672 0.9020      0.02149       64001436    15
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.3086 0.7800 0.9309      0.02588      111595725    12
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3091 0.7839 0.9357      0.02589      126706367    12
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.3094 0.7838 0.9370      0.02567      110983930    12
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3100 0.7879 0.9428      0.02660      115991087    12
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.3106 0.7879 0.9434      0.03190      115820493    10
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3116 0.8004 0.9711      0.04984      217338677    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.3121 0.8041 0.9806      0.08297      417071188    4
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQfp16` </summary>
Index size 746833749

 code_size 64

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3125 0.8005 0.9748      0.06930      408651230    5
nprobe=1,quantizer_efSearch=4            0.1355 0.2161 0.2183      0.00189        1740307    159
nprobe=2,quantizer_efSearch=4            0.1793 0.3205 0.3278      0.00202        3481015    149
nprobe=4,quantizer_efSearch=4            0.2146 0.4196 0.4362      0.00215        6949321    140
nprobe=4,quantizer_efSearch=8            0.2257 0.4489 0.4671      0.00232        6942125    130
nprobe=8,quantizer_efSearch=8            0.2568 0.5538 0.5944      0.00246       13803950    122
nprobe=8,quantizer_efSearch=16           0.2611 0.5637 0.6057      0.00284       13784149    106
nprobe=8,quantizer_efSearch=32           0.2628 0.5657 0.6077      0.00338       13772842    89
nprobe=16,quantizer_efSearch=4           0.2732 0.6191 0.6837      0.00365       27360432    83
nprobe=16,quantizer_efSearch=8           0.2809 0.6441 0.7143      0.00364       27338530    83
nprobe=16,quantizer_efSearch=16          0.2837 0.6520 0.7235      0.00381       27313466    79
nprobe=16,quantizer_efSearch=32          0.2859 0.6567 0.7293      0.00464       27286888    65
nprobe=16,quantizer_efSearch=64          0.2864 0.6580 0.7306      0.00559       27280564    54
nprobe=32,quantizer_efSearch=8           0.2921 0.7001 0.8005      0.00654       54091230    46
nprobe=32,quantizer_efSearch=32          0.3000 0.7246 0.8311      0.00692       53943045    44
nprobe=32,quantizer_efSearch=64          0.3007 0.7263 0.8328      0.00788       53918384    39
nprobe=32,quantizer_efSearch=128         0.3009 0.7269 0.8335      0.01028       53909242    30
nprobe=64,quantizer_efSearch=16          0.3032 0.7588 0.8892      0.01184      106573542    26
nprobe=64,quantizer_efSearch=32          0.3060 0.7674 0.9032      0.01207      106395644    25
nprobe=64,quantizer_efSearch=64          0.3075 0.7701 0.9062      0.01354      106303021    23
nprobe=64,quantizer_efSearch=128         0.3076 0.7709 0.9074      0.01500      106268704    21
nprobe=128,quantizer_efSearch=32         0.3090 0.7846 0.9426      0.02261      209294780    14
nprobe=128,quantizer_efSearch=64         0.3108 0.7901 0.9506      0.02311      208923293    13
nprobe=128,quantizer_efSearch=128        0.3114 0.7915 0.9522      0.02503      208778709    13
nprobe=256,quantizer_efSearch=64         0.3116 0.7978 0.9703      0.04400      409551169    7
nprobe=256,quantizer_efSearch=128        0.3126 0.8000 0.9741      0.04615      408869920    7
nprobe=512,quantizer_efSearch=128        0.3131 0.8024 0.9836      0.08556      800319680    4
nprobe=1024,quantizer_efSearch=256       0.3132 0.8030 0.9869      0.17612     1562074528    2
```

</details>
<details><summary>`PCAR32,IVF65536(IVF256,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 730115481

 code_size 64

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_IVF256_PQ16x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2803 0.6345 0.7030      0.00387       28125466    78
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1416 0.2260 0.2283      0.00239        3070388    126
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1435 0.2281 0.2304      0.00245        2422822    123
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1807 0.3238 0.3309      0.00254        3831776    119
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1849 0.3336 0.3408      0.00252        4807246    119
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2246 0.4425 0.4608      0.00265        8282175    114
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2264 0.4468 0.4652      0.00269        7635949    112
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2289 0.4530 0.4717      0.00286        8267853    105
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2494 0.5258 0.5603      0.00283       14579894    106
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2604 0.5548 0.5962      0.00291       14513615    104
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2630 0.5638 0.6061      0.00334       15123698    90
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2803 0.6345 0.7030      0.00402       28125471    75
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2861 0.6540 0.7262      0.00446       28667103    68
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.2864 0.6564 0.7298      0.00592       29910678    51
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.2865 0.6567 0.7300      0.00644       29909709    47
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2902 0.6870 0.7810      0.00616       55177975    49
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2916 0.6958 0.7939      0.00663       54943770    46
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2988 0.7169 0.8219      0.00680       55456839    45
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2989 0.7211 0.8273      0.00738       56658584    41
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2992 0.7194 0.8247      0.00753       55377618    40
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3000 0.7253 0.8325      0.00817       56573925    37
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3001 0.7256 0.8331      0.00829       59108029    37
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.3005 0.7263 0.8338      0.00968       59087963    31
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3059 0.7536 0.8838      0.01173      108391119    26
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3063 0.7623 0.8943      0.01450      107898542    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3079 0.7708 0.9080      0.01350      111529608    23
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3080 0.7714 0.9079      0.01432      111459301    21
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3103 0.7878 0.9442      0.02232      212470290    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3115 0.7918 0.9522      0.02662      219144123    12
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3123 0.7990 0.9716      0.04369      415299068    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3124 0.7990 0.9714      0.04373      420336838    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3125 0.8012 0.9759      0.04496      419029083    7
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3128 0.8024 0.9838      0.07923      806409921    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3129 0.8033 0.9882      0.15605     1573051378    2
```

</details>
<details><summary>`PCAR32,IVF65536,SQfp16` </summary>
Index size 728996256

 code_size 64

 log filename: autotune.dbbigann10M.PCAR32_IVF65536_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3124 0.8010 0.9759      0.04669      408342557    7
nprobe=2                                 0.1881 0.3390 0.3466      0.00650        3459974    47
nprobe=4                                 0.2290 0.4554 0.4742      0.00670        6919012    45
nprobe=8                                 0.2631 0.5672 0.6096      0.00705       13759195    43
nprobe=16                                0.2865 0.6587 0.7319      0.00833       27257214    37
nprobe=32                                0.3010 0.7280 0.8354      0.01102       53861966    28
nprobe=64                                0.3078 0.7719 0.9087      0.01634      106169467    19
nprobe=128                               0.3113 0.7923 0.9539      0.02638      208562864    12
nprobe=256                               0.3124 0.8010 0.9759      0.04590      408342557    7
nprobe=512                               0.3128 0.8033 0.9858      0.08534      798336663    4
nprobe=2048                              0.3129 0.8034 0.9893      0.30813     3048501660    1
```

</details>
<details><summary>`PCAR64,IVF16384_HNSW32,SQ8` </summary>
Index size 728886229

 code_size 64

 log filename: autotune.dbbigann10M.PCAR64_IVF16384_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6388 0.9902 0.9966      0.27850     2364432340    2
nprobe=1,quantizer_efSearch=16           0.2392 0.2879 0.2882      0.00325       13563506    93
nprobe=2,quantizer_efSearch=16           0.3355 0.4245 0.4250      0.00506       26322795    60
nprobe=2,quantizer_efSearch=32           0.3357 0.4250 0.4255      0.00545       26221943    56
nprobe=2,quantizer_efSearch=64           0.3359 0.4254 0.4259      0.00643       26201270    47
nprobe=4,quantizer_efSearch=4            0.3805 0.5088 0.5096      0.00762       52013203    40
nprobe=4,quantizer_efSearch=8            0.4223 0.5679 0.5688      0.00713       51275167    43
nprobe=4,quantizer_efSearch=16           0.4308 0.5780 0.5789      0.00754       50711573    41
nprobe=8,quantizer_efSearch=4            0.4855 0.6776 0.6788      0.01306       98608305    23
nprobe=8,quantizer_efSearch=8            0.4959 0.6946 0.6959      0.01223       98184884    25
nprobe=8,quantizer_efSearch=16           0.5064 0.7120 0.7133      0.01272       97432675    24
nprobe=8,quantizer_efSearch=128          0.5065 0.7143 0.7156      0.01683       96831389    18
nprobe=16,quantizer_efSearch=16          0.5612 0.8212 0.8230      0.02195      186129608    14
nprobe=16,quantizer_efSearch=64          0.5643 0.8291 0.8309      0.02355      184796265    13
nprobe=32,quantizer_efSearch=8           0.5832 0.8688 0.8714      0.04010      355472386    8
nprobe=32,quantizer_efSearch=256         0.6035 0.9094 0.9127      0.04933      351194975    7
nprobe=64,quantizer_efSearch=32          0.6236 0.9511 0.9556      0.07789      669114958    4
nprobe=64,quantizer_efSearch=128         0.6247 0.9558 0.9602      0.08543      665307628    4
nprobe=128,quantizer_efSearch=128        0.6344 0.9800 0.9855      0.14085     1257572078    3
nprobe=128,quantizer_efSearch=256        0.6345 0.9804 0.9859      0.14631     1256637770    3
nprobe=256,quantizer_efSearch=128        0.6386 0.9895 0.9959      0.25839     2369153898    2
nprobe=256,quantizer_efSearch=512        0.6388 0.9902 0.9966      0.29202     2364432340    2
nprobe=512,quantizer_efSearch=128        0.6393 0.9919 0.9986      0.49501     4450430369    1
nprobe=512,quantizer_efSearch=256        0.6394 0.9924 0.9991      0.49741     4446225608    1
nprobe=512,quantizer_efSearch=512        0.6395 0.9927 0.9994      0.50180     4441797682    1
nprobe=1024,quantizer_efSearch=512       0.6397 0.9930 0.9997      0.93118     8327971149    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ8` </summary>
Index size 860649429

 code_size 64

 log filename: autotune.dbbigann10M.PCAR64_IVF262144_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6344 0.9557 0.9590      0.07295      105884314    5
nprobe=1,quantizer_efSearch=4            0.1476 0.1648 0.1649      0.00212         435677    142
nprobe=2,quantizer_efSearch=4            0.2171 0.2498 0.2500      0.00215         874724    140
nprobe=4,quantizer_efSearch=4            0.2879 0.3490 0.3493      0.00223        1751020    135
nprobe=8,quantizer_efSearch=8            0.3955 0.5021 0.5025      0.00278        3481692    108
nprobe=16,quantizer_efSearch=4           0.4496 0.5936 0.5942      0.00361        6962185    84
nprobe=16,quantizer_efSearch=8           0.4716 0.6273 0.6278      0.00359        6952570    84
nprobe=16,quantizer_efSearch=16          0.4802 0.6400 0.6405      0.00428        6940636    71
nprobe=32,quantizer_efSearch=4           0.4870 0.6655 0.6664      0.00493       13829611    61
nprobe=32,quantizer_efSearch=8           0.5183 0.7188 0.7197      0.00552       13828309    55
nprobe=32,quantizer_efSearch=16          0.5378 0.7502 0.7513      0.00628       13800630    48
nprobe=32,quantizer_efSearch=32          0.5441 0.7607 0.7619      0.00730       13775596    42
nprobe=64,quantizer_efSearch=8           0.5475 0.7775 0.7789      0.00860       27381251    35
nprobe=64,quantizer_efSearch=16          0.5744 0.8247 0.8264      0.00955       27363061    32
nprobe=64,quantizer_efSearch=32          0.5846 0.8445 0.8464      0.01160       27295427    26
nprobe=64,quantizer_efSearch=64          0.5874 0.8506 0.8525      0.01380       27253777    22
nprobe=128,quantizer_efSearch=32         0.6097 0.8989 0.9014      0.01855       53992080    17
nprobe=128,quantizer_efSearch=64         0.6152 0.9120 0.9145      0.02034       53869935    15
nprobe=128,quantizer_efSearch=128        0.6165 0.9144 0.9168      0.02594       53804254    12
nprobe=256,quantizer_efSearch=32         0.6223 0.9282 0.9310      0.03106      106429435    10
nprobe=256,quantizer_efSearch=64         0.6317 0.9485 0.9517      0.04107      106226405    8
nprobe=256,quantizer_efSearch=128        0.6334 0.9537 0.9571      0.03751      105992873    9
nprobe=256,quantizer_efSearch=256        0.6343 0.9557 0.9590      0.04711      105905124    7
nprobe=512,quantizer_efSearch=64         0.6361 0.9628 0.9666      0.05901      208454845    6
nprobe=512,quantizer_efSearch=256        0.6406 0.9755 0.9795      0.07680      208053651    4
nprobe=512,quantizer_efSearch=512        0.6411 0.9765 0.9806      0.11333      207920402    3
nprobe=1024,quantizer_efSearch=128       0.6424 0.9803 0.9851      0.13468      406298393    3
nprobe=1024,quantizer_efSearch=256       0.6440 0.9851 0.9900      0.15212      408037886    2
nprobe=1024,quantizer_efSearch=512       0.6448 0.9867 0.9917      0.16963      407488889    2
nprobe=2048,quantizer_efSearch=256       0.6449 0.9890 0.9941      0.25809      788125489    2
nprobe=2048,quantizer_efSearch=512       0.6458 0.9909 0.9961      0.36162      797438233    1
nprobe=4096,quantizer_efSearch=512       0.6461 0.9916 0.9970      0.56665     1520822781    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ8` </summary>
Index size 795871001

 code_size 64

 log filename: autotune.dbbigann10M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.6370 0.9706 0.9747      0.10536      425664030    3
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.2236 0.2587 0.2589      0.00219        1578380    137
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.2305 0.2659 0.2661      0.00219        2259846    137
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2877 0.3479 0.3482      0.00222        2462339    136
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2988 0.3633 0.3635      0.00224        2460353    134
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.3001 0.3652 0.3655      0.00234        2456100    128
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.3501 0.4338 0.4342      0.00240        4914398    126
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.3782 0.4765 0.4769      0.00251        4207521    120
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.3948 0.4939 0.4943      0.00269        4889965    112
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.4096 0.5173 0.5177      0.00303        6197592    100
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4304 0.5557 0.5561      0.00323        8443071    93
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4392 0.5776 0.5781      0.00367        7697837    82
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4769 0.6307 0.6311      0.00391        9685570    77
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4842 0.6417 0.6422      0.00401        9667767    75
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4926 0.6669 0.6674      0.00466       15461729    65
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.5094 0.6903 0.6909      0.00490       16731918    62
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.5166 0.7126 0.7139      0.00538       15284599    56
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.5235 0.7242 0.7255      0.00572       15245567    53
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5404 0.7495 0.7507      0.00622       19168911    49
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.5411 0.7507 0.7519      0.00757       24337478    40
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.5474 0.7645 0.7657      0.00742       19116747    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.5490 0.7649 0.7661      0.00810       24296750    38
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.5517 0.7809 0.7824      0.00820       30596415    37
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.5527 0.7899 0.7915      0.01004       28862121    30
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.5771 0.8336 0.8353      0.01037       30078330    29
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5838 0.8421 0.8439      0.00984       32723965    31
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5860 0.8496 0.8514      0.01124       32635442    27
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.5882 0.8537 0.8555      0.01443       37743521    21
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.6005 0.8775 0.8797      0.01819       65473608    17
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6137 0.9103 0.9128      0.01955       64482877    16
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6139 0.9102 0.9127      0.01986       74725615    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6175 0.9169 0.9194      0.02613       64299965    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.6176 0.9173 0.9198      0.02582       94799858    12
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.6180 0.9181 0.9205      0.03132       94991817    10
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6279 0.9450 0.9481      0.02691      111831937    12
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6336 0.9539 0.9572      0.03158      126883612    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.6349 0.9575 0.9609      0.03411      126598677    10
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6350 0.9577 0.9611      0.04083      116464928    8
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.6370 0.9636 0.9674      0.06770      223803220    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256   0.6411 0.9761 0.9804      0.07344      249551820    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6412 0.9757 0.9799      0.06903      219116084    5
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6413 0.9763 0.9803      0.06724      218655094    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6444 0.9834 0.9878      0.11441      438459545    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6449 0.9868 0.9914      0.13198      419236875    3
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6454 0.9876 0.9924      0.14297      418574633    3
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6469 0.9929 0.9981      0.25139      818670569    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6471 0.9931 0.9983      0.25062      837597776    2
nprobe=2048,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.6472 0.9931 0.9983      0.40745      817608659    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6474 0.9944 0.9997      0.53155     1597966441    1
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ8` </summary>
Index size 755239381

 code_size 64

 log filename: autotune.dbbigann10M.PCAR64_IVF65536_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6417 0.9815 0.9863      0.10571      410877007    3
nprobe=1,quantizer_efSearch=4            0.1955 0.2264 0.2266      0.00206        1729770    146
nprobe=2,quantizer_efSearch=4            0.2797 0.3371 0.3372      0.00223        3465944    135
nprobe=4,quantizer_efSearch=4            0.3639 0.4582 0.4585      0.00247        6918803    122
nprobe=4,quantizer_efSearch=8            0.3857 0.4912 0.4915      0.00263        6906981    115
nprobe=4,quantizer_efSearch=16           0.3899 0.4959 0.4962      0.00322        6892959    94
nprobe=4,quantizer_efSearch=32           0.3921 0.4981 0.4984      0.00384        6887064    79
nprobe=8,quantizer_efSearch=8            0.4670 0.6234 0.6241      0.00389       13782881    78
nprobe=8,quantizer_efSearch=16           0.4742 0.6333 0.6340      0.00425       13758959    71
nprobe=8,quantizer_efSearch=32           0.4782 0.6379 0.6386      0.00513       13745564    59
nprobe=8,quantizer_efSearch=64           0.4795 0.6392 0.6399      0.00624       13741414    49
nprobe=16,quantizer_efSearch=4           0.5068 0.7022 0.7030      0.00832       27391900    37
nprobe=16,quantizer_efSearch=8           0.5281 0.7383 0.7392      0.00674       27363832    45
nprobe=16,quantizer_efSearch=16          0.5344 0.7486 0.7495      0.00693       27330316    44
nprobe=16,quantizer_efSearch=32          0.5404 0.7561 0.7570      0.00771       27297997    39
nprobe=16,quantizer_efSearch=64          0.5421 0.7580 0.7589      0.00903       27285387    34
nprobe=32,quantizer_efSearch=16          0.5782 0.8392 0.8413      0.01156       54152011    26
nprobe=32,quantizer_efSearch=32          0.5836 0.8490 0.8509      0.01193       54074487    26
nprobe=32,quantizer_efSearch=64          0.5859 0.8525 0.8544      0.01330       54047033    23
nprobe=64,quantizer_efSearch=16          0.6030 0.8961 0.8991      0.02053      106968249    15
nprobe=64,quantizer_efSearch=32          0.6125 0.9142 0.9171      0.02231      106779345    14
nprobe=64,quantizer_efSearch=128         0.6156 0.9197 0.9227      0.02488      106643332    13
nprobe=64,quantizer_efSearch=256         0.6157 0.9198 0.9228      0.03221      106634810    10
nprobe=128,quantizer_efSearch=16         0.6163 0.9229 0.9267      0.03900      210551123    8
nprobe=128,quantizer_efSearch=32         0.6281 0.9505 0.9543      0.03933      210178913    8
nprobe=128,quantizer_efSearch=64         0.6320 0.9592 0.9632      0.04041      209784710    8
nprobe=128,quantizer_efSearch=256        0.6321 0.9605 0.9647      0.04923      209589994    7
nprobe=128,quantizer_efSearch=512        0.6322 0.9608 0.9650      0.07192      209584381    5
nprobe=256,quantizer_efSearch=32         0.6346 0.9639 0.9684      0.07211      412557172    5
nprobe=256,quantizer_efSearch=64         0.6404 0.9775 0.9821      0.07584      411820593    4
nprobe=256,quantizer_efSearch=128        0.6409 0.9801 0.9849      0.07721      411122235    4
nprobe=256,quantizer_efSearch=256        0.6416 0.9812 0.9860      0.08210      410920692    4
nprobe=256,quantizer_efSearch=512        0.6417 0.9815 0.9863      0.10242      410877007    3
nprobe=512,quantizer_efSearch=64         0.6426 0.9831 0.9881      0.13977      804361773    3
nprobe=512,quantizer_efSearch=128        0.6428 0.9869 0.9921      0.15392      805520991    2
nprobe=512,quantizer_efSearch=256        0.6435 0.9885 0.9937      0.15205      804456128    2
nprobe=512,quantizer_efSearch=512        0.6438 0.9891 0.9943      0.17357      804191334    2
nprobe=1024,quantizer_efSearch=256       0.6447 0.9913 0.9967      0.28435     1573722010    2
nprobe=1024,quantizer_efSearch=512       0.6451 0.9919 0.9973      0.31354     1572398273    1
nprobe=2048,quantizer_efSearch=512       0.6454 0.9928 0.9983      0.59094     3075815047    1
```

</details>
<details><summary>`PCAR64,IVF65536(IVF256,PQ32x4fs,RFlat),SQ8` </summary>
Index size 739114265

 code_size 64

 log filename: autotune.dbbigann10M.PCAR64_IVF65536_IVF256_PQ32x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5322 0.7575 0.7590      0.01907      110609270    16
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.1899 0.2196 0.2198      0.00236        3062685    128
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2023 0.2330 0.2332      0.00242        2411192    125
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.2072 0.2386 0.2388      0.00247        2408623    122
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2568 0.3080 0.3081      0.00246        3675244    122
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2759 0.3319 0.3320      0.00254        3832478    119
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2879 0.3462 0.3463      0.00249        3824050    121
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2950 0.3549 0.3550      0.00258        4800193    117
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3394 0.4263 0.4265      0.00283        7143506    107
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3903 0.4956 0.4958      0.00280        8241525    108
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3911 0.4964 0.4966      0.00318        9525291    95
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.3919 0.4981 0.4983      0.00371        8232223    81
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3929 0.4993 0.4995      0.00344        9519871    88
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4324 0.5658 0.5663      0.00388       15267772    78
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4477 0.5907 0.5912      0.00388       14210101    78
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4688 0.6257 0.6263      0.00513       14476395    59
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4771 0.6377 0.6382      0.00447       16385171    68
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.4779 0.6384 0.6389      0.00500       15100385    61
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4797 0.6396 0.6401      0.00459       16373503    66
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5066 0.6960 0.6968      0.00615       29077397    49
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5197 0.7228 0.7237      0.00608       28165961    50
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5373 0.7519 0.7528      0.00725       28669917    42
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5381 0.7541 0.7550      0.00676       28657308    45
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5410 0.7563 0.7572      0.00682       29931788    44
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.5420 0.7588 0.7597      0.00774       29915977    39
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5426 0.7589 0.7598      0.00850       37596655    36
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5599 0.7999 0.8017      0.01039       56457074    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5655 0.8158 0.8177      0.01049       55052347    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5876 0.8528 0.8549      0.01120       56704404    27
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6062 0.9035 0.9064      0.01896      108491179    16
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6139 0.9159 0.9188      0.01935      109543975    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6159 0.9213 0.9242      0.02014      109324832    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6162 0.9222 0.9251      0.02076      111821740    15
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6163 0.9217 0.9246      0.02177      111805044    14
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.6204 0.9405 0.9443      0.03550      212147818    9
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6304 0.9578 0.9620      0.03618      212813831    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.6310 0.9598 0.9639      0.03720      212439021    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6324 0.9621 0.9663      0.03914      219843374    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6411 0.9812 0.9860      0.06918      416663005    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6412 0.9814 0.9862      0.07035      421661202    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6416 0.9827 0.9875      0.07272      421060195    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6431 0.9893 0.9945      0.13436      810365997    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6439 0.9903 0.9955      0.13968      814158146    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6453 0.9937 0.9990      0.24815     1582770415    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6455 0.9944 0.9998      0.48401     3084303045    1
```

</details>
<details><summary>`PCAR64,IVF65536,SQ8` </summary>
Index size 737401888

 code_size 64

 log filename: autotune.dbbigann10M.PCAR64_IVF65536_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6399 0.9830 0.9877      0.06410      410455842    5
nprobe=8                                 0.4796 0.6405 0.6411      0.01028       13727271    30
nprobe=16                                0.5418 0.7600 0.7609      0.01231       27252002    25
nprobe=32                                0.5861 0.8548 0.8566      0.01601       53973665    19
nprobe=64                                0.6154 0.9226 0.9254      0.02277      106507883    14
nprobe=128                               0.6313 0.9631 0.9670      0.04304      209345098    7
nprobe=256                               0.6399 0.9830 0.9877      0.06507      410455842    5
nprobe=512                               0.6426 0.9906 0.9957      0.11796      803489414    3
nprobe=1024                              0.6437 0.9935 0.9988      0.21411     1571092329    2
nprobe=2048                              0.6440 0.9944 0.9998      0.40832     3073208140    1
```

</details>
