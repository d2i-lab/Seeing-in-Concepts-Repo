# Detailed logs for dataset deep100M

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fs` </summary>
Index size 2301974476

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0767 0.2190 0.3702      0.00268        5129476    113
nprobe=2,quantizer_efSearch=4            0.0584 0.1600 0.2473      0.00163        2572278    184
nprobe=4,quantizer_efSearch=4            0.0642 0.1854 0.3165      0.00195        5139932    154
nprobe=2,quantizer_efSearch=8            0.0686 0.1873 0.2844      0.00242        2571485    125
nprobe=4,quantizer_efSearch=8            0.0767 0.2190 0.3702      0.00262        5129476    115
nprobe=4,quantizer_efSearch=16           0.0828 0.2335 0.3908      0.00393        5100270    77
nprobe=32,quantizer_efSearch=16          0.0838 0.2541 0.5338      0.00732       40388393    41
nprobe=16,quantizer_efSearch=32          0.0862 0.2573 0.5150      0.00740       20192300    41
nprobe=16,quantizer_efSearch=128         0.0865 0.2588 0.5171      0.01977       20088496    16
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fsr` </summary>
Index size 2301986252

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2069 0.5383 0.8013      0.03111       79067830    10
nprobe=1,quantizer_efSearch=4            0.0884 0.1677 0.1836      0.00152        1294606    199
nprobe=1,quantizer_efSearch=16           0.1048 0.1955 0.2137      0.00370        1280301    82
nprobe=2,quantizer_efSearch=16           0.1313 0.2675 0.3046      0.00387        2552691    78
nprobe=4,quantizer_efSearch=16           0.1586 0.3447 0.4131      0.00418        5097364    72
nprobe=8,quantizer_efSearch=16           0.1778 0.4079 0.5234      0.00480       10178252    63
nprobe=16,quantizer_efSearch=16          0.1926 0.4641 0.6277      0.00594       20297388    51
nprobe=32,quantizer_efSearch=16          0.1990 0.5000 0.7140      0.00832       40350173    37
nprobe=32,quantizer_efSearch=32          0.2018 0.5091 0.7254      0.01023       40154848    30
nprobe=32,quantizer_efSearch=64          0.2030 0.5127 0.7300      0.01452       39983655    21
nprobe=64,quantizer_efSearch=64          0.2061 0.5363 0.7986      0.02322       79319936    13
nprobe=64,quantizer_efSearch=128         0.2069 0.5383 0.8013      0.03105       79067830    10
nprobe=128,quantizer_efSearch=64         0.2094 0.5491 0.8429      0.03724      157220871    9
nprobe=128,quantizer_efSearch=128        0.2095 0.5524 0.8492      0.04434      156523165    7
nprobe=128,quantizer_efSearch=256        0.2101 0.5532 0.8511      0.06032      156163277    5
nprobe=256,quantizer_efSearch=64         0.2104 0.5543 0.8600      0.08026      310766834    4
nprobe=256,quantizer_efSearch=128        0.2108 0.5589 0.8695      0.09114      309124625    4
nprobe=512,quantizer_efSearch=128        0.2128 0.5605 0.8773      0.15491      608352125    2
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 2042283280

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0601 0.1629 0.2464      0.00147        3302820    204
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2    0.0481 0.1182 0.1590      0.00116        2019295    260
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0508 0.1281 0.1730      0.00121        2019181    249
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.0530 0.1342 0.1816      0.00128        2017158    235
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.0562 0.1399 0.1882      0.00175        2739327    172
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0592 0.1475 0.1995      0.00144        2739076    209
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0601 0.1629 0.2464      0.00141        3302597    213
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2   0.0623 0.1679 0.2570      0.00176        3298282    171
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2    0.0644 0.1810 0.3027      0.00169        5855524    178
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0697 0.1868 0.2823      0.00169        4021769    178
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.0724 0.2014 0.3375      0.00186        6577332    162
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.0751 0.2131 0.3547      0.00200        6577242    150
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.0757 0.2220 0.4023      0.00242       11670929    124
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0787 0.2166 0.3668      0.00241        6563154    125
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.0816 0.2424 0.4426      0.00289       13036864    104
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.0842 0.2492 0.4546      0.00459       21105353    66
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.0876 0.2612 0.5446      0.01971      123494323    16
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 2042365200

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1818 0.4344 0.5755      0.00526       21771353    57
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.0956 0.1741 0.1921      0.00193        2023446    156
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.0967 0.1875 0.2108      0.00189        3303188    159
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1158 0.2338 0.2672      0.00266        3308937    113
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1171 0.2331 0.2651      0.00271        4026367    111
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1249 0.2494 0.2842      0.00283        4019067    107
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1312 0.2796 0.3329      0.00292        5870251    103
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1357 0.2919 0.3478      0.00267        5868787    113
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1442 0.3082 0.3676      0.00272        6573319    111
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1511 0.3264 0.3911      0.00310        6567218    97
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1527 0.3403 0.4179      0.00343       11664451    88
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1638 0.3720 0.4653      0.00364       11659562    83
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1715 0.3893 0.4880      0.00415       13027758    73
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1807 0.4137 0.5229      0.00539       15760167    56
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1812 0.4278 0.5634      0.00505       21805885    60
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1818 0.4344 0.5755      0.00543       21769807    56
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1894 0.4482 0.5921      0.00605       23150852    50
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1925 0.4577 0.6088      0.00601       23113391    50
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1938 0.4639 0.6307      0.00785       43244131    39
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2013 0.4897 0.6808      0.00824       43185020    37
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2049 0.5107 0.7158      0.01032       45653203    30
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2056 0.5149 0.7225      0.01236       50872758    25
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.2067 0.5101 0.7105      0.01555       61187686    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.2069 0.5100 0.7105      0.02438       81571287    13
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2090 0.5313 0.7751      0.02131       85172382    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2114 0.5369 0.7849      0.02259       85044546    14
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2138 0.5435 0.7981      0.03550      100000811    9
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2158 0.5547 0.8354      0.04361      177846282    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2161 0.5570 0.8441      0.04439      177352950    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2164 0.5572 0.8447      0.05324      197991466    6
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2165 0.5579 0.8463      0.05298      177068716    6
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=512 0.2166 0.5590 0.8472      0.12210      321266352    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.2171 0.5652 0.8686      0.14565      689555743    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=512 0.2175 0.5688 0.8848      0.26171     1356380390    2
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fs` </summary>
Index size 1775150540

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0751 0.2199 0.4139      0.00218       17977280    138
nprobe=2,quantizer_efSearch=4            0.0596 0.1703 0.3029      0.00134        8981109    225
nprobe=1,quantizer_efSearch=8            0.0615 0.1614 0.2540      0.00162        4478669    185
nprobe=4,quantizer_efSearch=4            0.0629 0.1885 0.3579      0.00178       17983555    169
nprobe=2,quantizer_efSearch=8            0.0710 0.1945 0.3437      0.00182        8964633    165
nprobe=4,quantizer_efSearch=8            0.0751 0.2199 0.4139      0.00220       17977280    137
nprobe=8,quantizer_efSearch=8            0.0766 0.2225 0.4594      0.00270       35899979    112
nprobe=4,quantizer_efSearch=16           0.0780 0.2271 0.4260      0.00287       17913280    105
nprobe=4,quantizer_efSearch=32           0.0802 0.2296 0.4286      0.00424       17890892    71
nprobe=16,quantizer_efSearch=32          0.0829 0.2396 0.5102      0.00560       71238203    54
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fsr` </summary>
Index size 1775226828

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1813 0.4477 0.6151      0.01270       35668231    24
nprobe=1,quantizer_efSearch=4            0.1030 0.2050 0.2407      0.00112        4481479    268
nprobe=1,quantizer_efSearch=16           0.1174 0.2326 0.2712      0.00245        4472010    123
nprobe=2,quantizer_efSearch=16           0.1447 0.3170 0.3869      0.00265        8936783    114
nprobe=4,quantizer_efSearch=16           0.1671 0.3896 0.5041      0.00302       17903131    100
nprobe=8,quantizer_efSearch=16           0.1801 0.4447 0.6104      0.00365       35774795    83
nprobe=8,quantizer_efSearch=32           0.1819 0.4474 0.6148      0.00511       35706405    59
nprobe=16,quantizer_efSearch=128         0.1941 0.4921 0.7124      0.01405       71122491    22
nprobe=32,quantizer_efSearch=128         0.1999 0.5199 0.7812      0.01587      141641151    19
nprobe=64,quantizer_efSearch=16          0.2012 0.5212 0.8058      0.01676      283279161    18
nprobe=64,quantizer_efSearch=256         0.2043 0.5326 0.8273      0.03476      281454727    9
nprobe=128,quantizer_efSearch=256        0.2047 0.5386 0.8521      0.05083      557440225    6
nprobe=256,quantizer_efSearch=128        0.2054 0.5413 0.8609      0.06787     1101983082    5
nprobe=256,quantizer_efSearch=256        0.2057 0.5417 0.8618      0.07881     1100298573    4
nprobe=256,quantizer_efSearch=512        0.2058 0.5418 0.8618      0.10343     1099817586    3
nprobe=512,quantizer_efSearch=256        0.2060 0.5411 0.8640      0.14734     2163264982    3
nprobe=1024,quantizer_efSearch=512       0.2064 0.5407 0.8632      0.28188     4225312024    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 1710408976

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.0600 0.1553 0.2472      0.00138        5214287    218
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1   0.0440 0.1172 0.1873      0.00120        4707549    250
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.0564 0.1468 0.2355      0.00122        5237151    246
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0594 0.1525 0.2428      0.00130        5221316    232
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0597 0.1544 0.2462      0.00125        5216801    241
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8   0.0621 0.1593 0.2508      0.00131        5898786    230
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8  0.0630 0.1616 0.2546      0.00148        5887722    203
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0673 0.1858 0.3306      0.00142        9726726    212
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.0720 0.1933 0.3377      0.00148       10398263    204
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8  0.0735 0.1972 0.3469      0.00172       10378398    175
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0748 0.2003 0.3501      0.00204       11697722    148
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.0764 0.2168 0.4020      0.00225       20719640    134
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.0767 0.2252 0.4515      0.00249       37452881    121
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16 0.0782 0.2235 0.4652      0.00382       74988792    79
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 1710356496

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.1995 0.5211 0.8026      0.02152      286442116    14
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2   0.1027 0.2056 0.2393      0.00164        4869378    184
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1070 0.2163 0.2499      0.00142        5219368    211
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1095 0.2214 0.2567      0.00136        5217473    220
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1184 0.2539 0.3048      0.00196       10419315    153
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1367 0.2972 0.3611      0.00160        9720660    189
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1536 0.3536 0.4542      0.00183       18788049    165
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1564 0.3640 0.4695      0.00182       18763439    165
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1690 0.4137 0.5669      0.00255       36766695    118
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1765 0.4337 0.5978      0.00299       37285920    101
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1784 0.4322 0.5915      0.00315       38647227    96
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1796 0.4406 0.6063      0.00334       38584681    90
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.1838 0.4545 0.6397      0.00433       74721374    70
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.1873 0.4740 0.6859      0.00410       73091617    74
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1913 0.4845 0.7022      0.00467       74201388    65
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32 0.1921 0.4898 0.7109      0.00716       76605487    42
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8   0.1939 0.4987 0.7479      0.00739      144148528    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.1988 0.5151 0.7750      0.00818      147407503    37
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.1991 0.5118 0.7690      0.00798      144975860    38
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.1995 0.5165 0.7787      0.01225      152375365    25
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.1996 0.5168 0.7790      0.01236      162753680    25
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2002 0.5266 0.8129      0.01751      288315003    18
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2011 0.5285 0.8182      0.02279      287600210    14
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.2016 0.5293 0.8201      0.02543      287467213    12
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32 0.2017 0.5291 0.8204      0.02892      287399681    11
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.2029 0.5339 0.8401      0.03686      565814224    9
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs` </summary>
Index size 1643796684

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0747 0.2105 0.4335      0.00280       71075208    107
nprobe=1,quantizer_efSearch=4            0.0544 0.1481 0.2622      0.00125       17752816    241
nprobe=1,quantizer_efSearch=8            0.0607 0.1612 0.2842      0.00147       17744841    205
nprobe=2,quantizer_efSearch=4            0.0610 0.1762 0.3400      0.00173       35499277    174
nprobe=1,quantizer_efSearch=16           0.0612 0.1639 0.2886      0.00174       17706624    173
nprobe=2,quantizer_efSearch=8            0.0689 0.1937 0.3691      0.00192       35510211    157
nprobe=2,quantizer_efSearch=16           0.0697 0.1971 0.3751      0.00229       35442368    131
nprobe=4,quantizer_efSearch=8            0.0747 0.2105 0.4335      0.00265       71075208    114
nprobe=4,quantizer_efSearch=16           0.0761 0.2153 0.4433      0.00299       70902784    101
nprobe=4,quantizer_efSearch=32           0.0769 0.2161 0.4450      0.00380       70873063    79
nprobe=16,quantizer_efSearch=32          0.0773 0.2244 0.4976      0.00592      281389206    51
nprobe=32,quantizer_efSearch=16          0.0774 0.2217 0.4928      0.00664      559152394    46
nprobe=16,quantizer_efSearch=128         0.0775 0.2251 0.4984      0.01021      281204390    30
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fsr` </summary>
Index size 1643820492

 code_size 8

 log filename: autotune.dbdeep100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1816 0.4569 0.6824      0.00799      141245554    38
nprobe=1,quantizer_efSearch=4            0.1072 0.2413 0.3023      0.00099       17725369    303
nprobe=1,quantizer_efSearch=16           0.1183 0.2613 0.3281      0.00159       17690953    189
nprobe=2,quantizer_efSearch=16           0.1476 0.3418 0.4540      0.00189       35443189    159
nprobe=4,quantizer_efSearch=16           0.1700 0.4123 0.5814      0.00237       70933537    127
nprobe=8,quantizer_efSearch=16           0.1796 0.4534 0.6779      0.00328      141459523    92
nprobe=8,quantizer_efSearch=32           0.1814 0.4563 0.6817      0.00398      141292122    76
nprobe=8,quantizer_efSearch=64           0.1815 0.4567 0.6823      0.00540      141258001    56
nprobe=16,quantizer_efSearch=64          0.1851 0.4829 0.7541      0.00705      281291274    43
nprobe=32,quantizer_efSearch=64          0.1866 0.4991 0.7992      0.00933      557679495    33
nprobe=64,quantizer_efSearch=16          0.1874 0.4988 0.8116      0.01906     1106974663    16
nprobe=64,quantizer_efSearch=64          0.1886 0.5050 0.8230      0.02018     1102278341    15
nprobe=64,quantizer_efSearch=256         0.1890 0.5050 0.8232      0.02835     1101479272    11
nprobe=128,quantizer_efSearch=64         0.1897 0.5073 0.8358      0.03452     2170337806    9
nprobe=128,quantizer_efSearch=256        0.1900 0.5070 0.8357      0.04241     2166546624    8
```

</details>
<details><summary>`OPQ8_64,IVF1048576_HNSW32,PQ8` </summary>
Index size 2162276272

 code_size 8

 log filename: autotune.dbdeep100M.OPQ8_64_IVF1048576_HNSW32_PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1918 0.4386 0.5552      0.01037       20382772    29
nprobe=1,quantizer_efSearch=4,ht=28      0.0872 0.1404 0.1501      0.00282        1288523    107
nprobe=4,quantizer_efSearch=4,ht=64      0.1413 0.2943 0.3431      0.00291        5127597    104
nprobe=4,quantizer_efSearch=8,ht=30      0.1604 0.3191 0.3600      0.00439        5124031    69
nprobe=4,quantizer_efSearch=16,ht=28     0.1614 0.3015 0.3321      0.00566        5098982    53
nprobe=8,quantizer_efSearch=4,ht=30      0.1751 0.3735 0.4421      0.00619       10238674    49
nprobe=8,quantizer_efSearch=32,ht=30     0.1921 0.4123 0.4836      0.01034       10120051    30
nprobe=16,quantizer_efSearch=8,ht=28     0.1948 0.4182 0.4970      0.01070       20339888    29
nprobe=16,quantizer_efSearch=16,ht=32    0.2064 0.4807 0.6108      0.01196       20274521    26
nprobe=16,quantizer_efSearch=32,ht=30    0.2070 0.4725 0.5838      0.01433       20168098    21
nprobe=16,quantizer_efSearch=64,ht=30    0.2081 0.4734 0.5853      0.01890       20101782    16
nprobe=16,quantizer_efSearch=64,ht=32    0.2111 0.4891 0.6215      0.01906       20101782    16
nprobe=64,quantizer_efSearch=32,ht=64    0.2268 0.5654 0.8013      0.02117       79646884    15
nprobe=64,quantizer_efSearch=64,ht=64    0.2285 0.5696 0.8073      0.02503       79247794    12
nprobe=64,quantizer_efSearch=128,ht=64   0.2296 0.5719 0.8101      0.03315       79005593    10
nprobe=256,quantizer_efSearch=64,ht=32   0.2313 0.5855 0.8509      0.14310      310430007    3
nprobe=256,quantizer_efSearch=512,ht=32  0.2340 0.5906 0.8613      0.20109      307160566    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2354 0.6014 0.9015      0.26545     1186131943    2
nprobe=4096,quantizer_efSearch=512,ht=64 0.2355 0.6013 0.9054      1.00949     4329040668    1
```

</details>
<details><summary>`OPQ8_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ8` </summary>
Index size 1902616052

 code_size 8

 log filename: autotune.dbdeep100M.OPQ8_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.2148 0.5039 0.6446      0.02080       51031874    15
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=30       0.0859 0.1479 0.1601      0.00253        2949920    119
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=30       0.1107 0.1946 0.2084      0.00302        5420561    100
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=28       0.1117 0.2014 0.2204      0.00326        5492604    92
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=24       0.1189 0.1863 0.2008      0.00368        6571743    82
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=28      0.1312 0.2255 0.2416      0.00391        4013002    77
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=64     0.1748 0.3595 0.4139      0.00474       10675335    64
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=30      0.1889 0.4030 0.4663      0.00723       15749277    42
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=64     0.1967 0.4374 0.5298      0.00982       15714586    31
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64      0.2101 0.5079 0.6775      0.01106       43164859    28
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=28     0.2137 0.4846 0.5964      0.02112       45635844    15
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=30     0.2148 0.5039 0.6446      0.02088       51031874    15
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=64     0.2288 0.5740 0.8044      0.02537       90032815    12
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=512,ht=64   0.2327 0.5940 0.8573      0.06417      320940000    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=64   0.2339 0.6052 0.8948      0.14341      688316846    3
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=64  0.2344 0.6073 0.9024      0.29770     1222319393    2
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=64 0.2345 0.6077 0.9060      3.18760     4538599209    1
```

</details>
<details><summary>`OPQ8_64,IVF262144_HNSW32,PQ8` </summary>
Index size 1740639408

 code_size 8

 log filename: autotune.dbdeep100M.OPQ8_64_IVF262144_HNSW32_PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1896 0.4717 0.6490      0.01010       71752711    30
nprobe=1,quantizer_efSearch=4,ht=28      0.0992 0.1825 0.1979      0.00248        4484970    121
nprobe=1,quantizer_efSearch=4,ht=64      0.1060 0.2119 0.2387      0.00244        4484970    124
nprobe=2,quantizer_efSearch=4,ht=30      0.1305 0.2707 0.3098      0.00247        8977058    122
nprobe=4,quantizer_efSearch=4,ht=64      0.1492 0.3432 0.4329      0.00276       17964820    109
nprobe=2,quantizer_efSearch=8,ht=30      0.1500 0.3077 0.3504      0.00291        8966889    103
nprobe=2,quantizer_efSearch=16,ht=30     0.1536 0.3142 0.3571      0.00368        8941214    82
nprobe=2,quantizer_efSearch=16,ht=64     0.1553 0.3332 0.3898      0.00365        8941214    83
nprobe=4,quantizer_efSearch=8,ht=30      0.1714 0.3784 0.4569      0.00463       17979084    65
nprobe=4,quantizer_efSearch=32,ht=64     0.1786 0.4098 0.5127      0.00846       17891421    36
nprobe=8,quantizer_efSearch=8,ht=28      0.1794 0.4104 0.5029      0.00571       35916064    53
nprobe=8,quantizer_efSearch=4,ht=30      0.1803 0.4258 0.5463      0.00564       35967203    54
nprobe=8,quantizer_efSearch=32,ht=30     0.1935 0.4550 0.5810      0.00801       35738512    38
nprobe=16,quantizer_efSearch=16,ht=30    0.2003 0.4958 0.6640      0.01234       71429521    25
nprobe=16,quantizer_efSearch=16,ht=32    0.2013 0.5065 0.6960      0.01111       71429521    28
nprobe=16,quantizer_efSearch=32,ht=30    0.2040 0.5017 0.6722      0.01191       71267118    26
nprobe=16,quantizer_efSearch=64,ht=30    0.2041 0.5020 0.6727      0.01456       71190634    21
nprobe=16,quantizer_efSearch=64,ht=32    0.2047 0.5120 0.7050      0.01488       71190634    21
nprobe=32,quantizer_efSearch=128,ht=64   0.2122 0.5464 0.7942      0.02830      141659611    11
nprobe=64,quantizer_efSearch=64,ht=64    0.2176 0.5644 0.8433      0.02008      281902917    15
nprobe=64,quantizer_efSearch=128,ht=64   0.2179 0.5649 0.8447      0.02521      281591800    12
nprobe=128,quantizer_efSearch=32,ht=32   0.2182 0.5653 0.8451      0.09450      560499127    4
nprobe=128,quantizer_efSearch=64,ht=30   0.2187 0.5580 0.8183      0.06881      559064051    5
nprobe=256,quantizer_efSearch=64,ht=32   0.2204 0.5705 0.8618      0.14061     1104431332    3
nprobe=256,quantizer_efSearch=512,ht=32  0.2214 0.5739 0.8679      0.17881     1099919992    2
nprobe=2048,quantizer_efSearch=256,ht=32 0.2215 0.5750 0.8753      1.03335     7959193149    1
```

</details>
<details><summary>`OPQ8_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ8` </summary>
Index size 1675862516

 code_size 8

 log filename: autotune.dbdeep100M.OPQ8_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.1837 0.4689 0.6425      0.01035       72733355    29
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=16       0.0237 0.0361 0.0410      0.00252        4885290    120
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=28       0.1179 0.2382 0.2686      0.00262        9444730    115
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=28      0.1261 0.2563 0.2915      0.00266        9393242    113
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=32       0.1325 0.2820 0.3319      0.00287        9394372    105
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=64      0.1329 0.2870 0.3418      0.00291        9388075    104
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=30      0.1439 0.3042 0.3509      0.00284        9719020    106
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=32       0.1494 0.3298 0.3999      0.00366       19482647    82
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64      0.1724 0.4153 0.5336      0.00400       38806509    75
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=64       0.1833 0.4601 0.6090      0.00421       37322887    72
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=64      0.1850 0.4624 0.6236      0.00567       73725457    53
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=32      0.1900 0.4680 0.6114      0.00878       46356309    35
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64      0.1907 0.4932 0.7071      0.01376      144147876    22
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=30    0.1978 0.5015 0.6760      0.01275       76648583    24
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=64     0.2032 0.5252 0.7596      0.01522      144207473    20
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=64   0.2099 0.5441 0.7923      0.01983      162778737    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64    0.2126 0.5618 0.8506      0.03379      564272017    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=32     0.2127 0.5597 0.8256      0.03930      287583017    8
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=64   0.2165 0.5738 0.8827      0.11237     1112740702    3
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=64  0.2167 0.5747 0.8895      0.16337     2204249688    2
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=64 0.2168 0.5751 0.8913      2.00329    16005721492    1
```

</details>
<details><summary>`OPQ8_64,IVF65536_HNSW32,PQ8` </summary>
Index size 1635229360

 code_size 8

 log filename: autotune.dbdeep100M.OPQ8_64_IVF65536_HNSW32_PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1218 0.2649 0.3191      0.00261       17746473    115
nprobe=1,quantizer_efSearch=4,ht=4       0.0035 0.0070 0.0083      0.00227       17751866    132
nprobe=1,quantizer_efSearch=4,ht=30      0.1124 0.2423 0.2880      0.00239       17751866    126
nprobe=1,quantizer_efSearch=4,ht=64      0.1132 0.2491 0.3028      0.00233       17751866    129
nprobe=1,quantizer_efSearch=8,ht=30      0.1206 0.2587 0.3073      0.00256       17746473    117
nprobe=1,quantizer_efSearch=8,ht=32      0.1218 0.2649 0.3191      0.00262       17746473    115
nprobe=2,quantizer_efSearch=8,ht=26      0.1412 0.2956 0.3439      0.00273       35538798    111
nprobe=2,quantizer_efSearch=8,ht=30      0.1503 0.3406 0.4260      0.00340       35538798    89
nprobe=2,quantizer_efSearch=16,ht=64     0.1557 0.3563 0.4570      0.00371       35473302    81
nprobe=4,quantizer_efSearch=4,ht=64      0.1562 0.3847 0.5263      0.00418       71059434    72
nprobe=4,quantizer_efSearch=32,ht=24     0.1600 0.3210 0.3695      0.00465       70888244    65
nprobe=4,quantizer_efSearch=16,ht=28     0.1756 0.4058 0.5216      0.00472       70942415    64
nprobe=4,quantizer_efSearch=32,ht=64     0.1794 0.4305 0.5889      0.00515       70888244    59
nprobe=8,quantizer_efSearch=8,ht=28      0.1801 0.4423 0.6004      0.00722      141811050    42
nprobe=8,quantizer_efSearch=32,ht=30     0.1880 0.4693 0.6623      0.00904      141321624    34
nprobe=8,quantizer_efSearch=64,ht=32     0.1898 0.4761 0.6842      0.01140      141266069    27
nprobe=16,quantizer_efSearch=8,ht=28     0.1902 0.4800 0.6779      0.01247      282484845    25
nprobe=16,quantizer_efSearch=16,ht=30    0.1950 0.5006 0.7352      0.01494      281947045    21
nprobe=16,quantizer_efSearch=32,ht=30    0.1959 0.5037 0.7402      0.01525      281503010    20
nprobe=16,quantizer_efSearch=64,ht=30    0.1961 0.5040 0.7417      0.01645      281358551    19
nprobe=16,quantizer_efSearch=16,ht=32    0.1962 0.5061 0.7586      0.01702      281947045    18
nprobe=16,quantizer_efSearch=64,ht=32    0.1973 0.5101 0.7652      0.01880      281358551    16
nprobe=32,quantizer_efSearch=128,ht=64   0.2006 0.5317 0.8227      0.02283      557524646    14
nprobe=64,quantizer_efSearch=64,ht=64    0.2030 0.5404 0.8487      0.04073     1102568290    8
nprobe=64,quantizer_efSearch=128,ht=64   0.2032 0.5405 0.8492      0.03840     1101917810    8
nprobe=64,quantizer_efSearch=128,ht=32   0.2039 0.5372 0.8412      0.06680     1101917810    5
```

</details>

## Code sizes in [9, 16]

<details><summary>`IVF1048576_HNSW32,PQ16` </summary>
Index size 3096502121

 code_size 16

 log filename: autotune.dbdeep100M.IVF1048576_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3626 0.6879 0.7423      0.05799       40858958    6
nprobe=2,quantizer_efSearch=4,ht=128      0.1719 0.2592 0.2672      0.00346        2613060    87
nprobe=2,quantizer_efSearch=16,ht=54      0.1825 0.2556 0.2625      0.00749        2596542    41
nprobe=2,quantizer_efSearch=16,ht=62      0.2011 0.3039 0.3121      0.00811        2596542    37
nprobe=4,quantizer_efSearch=8,ht=52       0.2083 0.2992 0.3090      0.00870        5219482    35
nprobe=8,quantizer_efSearch=8,ht=128      0.2837 0.4915 0.5186      0.00899       10432267    34
nprobe=8,quantizer_efSearch=16,ht=56      0.2894 0.4661 0.4847      0.01612       10365233    19
nprobe=8,quantizer_efSearch=16,ht=60      0.2977 0.5052 0.5283      0.01597       10365233    19
nprobe=8,quantizer_efSearch=32,ht=64      0.3036 0.5245 0.5513      0.01937       10303746    16
nprobe=16,quantizer_efSearch=4,ht=62      0.3088 0.5538 0.5901      0.02829       20822419    11
nprobe=16,quantizer_efSearch=16,ht=60     0.3335 0.5970 0.6343      0.02841       20674663    11
nprobe=32,quantizer_efSearch=16,ht=128    0.3586 0.6907 0.7536      0.02926       41085015    11
nprobe=32,quantizer_efSearch=32,ht=60     0.3626 0.6879 0.7423      0.05696       40858958    6
nprobe=32,quantizer_efSearch=256,ht=64    0.3669 0.7054 0.7690      0.08856       40531182    4
nprobe=128,quantizer_efSearch=16,ht=128   0.3755 0.7588 0.8550      0.09896      161335076    4
nprobe=64,quantizer_efSearch=128,ht=62    0.3830 0.7574 0.8410      0.11866       80393787    3
nprobe=128,quantizer_efSearch=256,ht=128  0.3925 0.8055 0.9149      0.13173      158667996    3
nprobe=256,quantizer_efSearch=128,ht=64   0.3975 0.8255 0.9466      0.41379      314089781    1
nprobe=512,quantizer_efSearch=512,ht=128  0.4015 0.8421 0.9763      0.43026      613007487    1
nprobe=2048,quantizer_efSearch=256,ht=128 0.4023 0.8463 0.9865      1.40604     2303221852    1
nprobe=4096,quantizer_efSearch=512,ht=128 0.4028 0.8477 0.9900      2.77128     4389666207    1
```

</details>
<details><summary>`IVF1048576_HNSW32,PQ32x4fs` </summary>
Index size 3367979909

 code_size 16

 log filename: autotune.dbdeep100M.IVF1048576_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1788 0.4428 0.6664      0.00629       41300724    48
nprobe=8,quantizer_efSearch=4            0.1539 0.3632 0.4932      0.00322       10464151    94
nprobe=8,quantizer_efSearch=8            0.1586 0.3722 0.5060      0.00342       10439918    88
nprobe=16,quantizer_efSearch=4           0.1647 0.3977 0.5720      0.00417       20828864    72
nprobe=16,quantizer_efSearch=8           0.1739 0.4238 0.6096      0.00512       20760567    59
nprobe=32,quantizer_efSearch=8           0.1788 0.4428 0.6664      0.00665       41300724    46
nprobe=16,quantizer_efSearch=16          0.1797 0.4363 0.6268      0.00609       20677175    50
nprobe=64,quantizer_efSearch=8           0.1806 0.4489 0.6964      0.00850       81854799    36
nprobe=32,quantizer_efSearch=16          0.1866 0.4642 0.6999      0.00782       41089513    39
nprobe=64,quantizer_efSearch=16          0.1914 0.4741 0.7430      0.01036       81619635    29
nprobe=32,quantizer_efSearch=32          0.1916 0.4746 0.7139      0.00999       40860096    31
nprobe=64,quantizer_efSearch=32          0.1953 0.4873 0.7648      0.01258       81129627    24
nprobe=64,quantizer_efSearch=64          0.1976 0.4917 0.7735      0.01771       80677090    17
```

</details>
<details><summary>`IVF1048576(IVF1024,PQ48x4fs,RFlat),PQ16` </summary>
Index size 2845499309

 code_size 16

 log filename: autotune.dbdeep100M.IVF1048576_IVF1024_PQ48x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.3875 0.7499 0.8277      0.19150      165886038    2
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=12        0.0034 0.0073 0.0094      0.00274        1669606    110
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=28        0.0174 0.0242 0.0268      0.00285        2047358    106
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=52        0.1148 0.1501 0.1534      0.00322        2756139    94
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=54       0.1293 0.1717 0.1753      0.00335        2749876    90
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=62       0.1362 0.1889 0.1932      0.00327        2040840    92
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=54      0.1369 0.1815 0.1852      0.00436        6948818    69
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=62        0.1432 0.2084 0.2144      0.00454        2974444    67
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=64       0.1532 0.2145 0.2186      0.00436        6948606    69
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=60        0.1993 0.2886 0.2961      0.00512        5460163    59
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60      0.2084 0.3021 0.3100      0.00656        8230533    46
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128     0.2123 0.3131 0.3217      0.00892       24238548    34
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64       0.2281 0.3488 0.3584      0.00904       10817065    34
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=128      0.2987 0.5051 0.5282      0.01084       21289798    28
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128      0.3120 0.5287 0.5550      0.01135       21256316    27
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=64     0.3470 0.6215 0.6623      0.04349       25988085    7
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=58      0.3595 0.6508 0.6948      0.05688       46181526    6
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=58     0.3662 0.6673 0.7136      0.06878       46072161    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128    0.3904 0.7699 0.8519      0.07018      163670093    5
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62    0.3910 0.7625 0.8412      0.15046      245080639    2
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=64    0.3920 0.7688 0.8498      0.18688      245078976    2
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62   0.4010 0.8046 0.9014      0.27090      322799600    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128   0.4091 0.8444 0.9741      0.39408      692268203    1
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=64  0.4098 0.8474 0.9799      1.64592     1232412014    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=64   0.4099 0.8495 0.9844      3.03613     2478551610    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=128 0.4101 0.8512 0.9899      5.59181     4649352810    1
srun: error: learnfair4131: task 0: Bus error (core dumped)
srun: Terminating job step 33564459.0
```

</details>
<details><summary>`IVF1048576(IVF1024,PQ48x4fs,RFlat),PQ32x4fs` </summary>
Index size 3116558793

 code_size 16

 log filename: autotune.dbdeep100M.IVF1048576_IVF1024_PQ48x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.1483 0.3268 0.4312      0.00256       11804051    118
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1    0.0815 0.1463 0.1629      0.00140        1674022    214
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1    0.0826 0.1501 0.1665      0.00144        1669558    209
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0932 0.1714 0.1893      0.00149        2047221    202
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.0946 0.1745 0.1930      0.00157        2041418    192
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1    0.0952 0.1884 0.2246      0.00159        2961076    189
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.0992 0.1833 0.2027      0.00175        2755320    172
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2    0.0993 0.1958 0.2268      0.00163        3349031    184
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.1123 0.2269 0.2684      0.00171        3335182    176
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2    0.1173 0.2510 0.3088      0.00191        5941267    158
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2    0.1261 0.2754 0.3446      0.00195        5927747    154
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.1284 0.2835 0.3592      0.00204        5922805    148
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1399 0.3062 0.3821      0.00212        6636441    142
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1439 0.3166 0.3997      0.00234        6622761    129
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.1483 0.3268 0.4312      0.00264       11804710    114
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1596 0.3563 0.4793      0.00268       11773073    112
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1690 0.3864 0.5216      0.00331       13137024    91
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1701 0.3837 0.5148      0.00383       15917898    79
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.1707 0.3936 0.5642      0.00367       21997908    82
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.1714 0.3933 0.5467      0.00385       23437042    78
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.1846 0.4317 0.6161      0.00494       26068293    61
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1848 0.4369 0.6283      0.00537       26035537    56
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.1864 0.4413 0.6343      0.00625       26011695    48
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.1874 0.4460 0.6404      0.00795       31385104    38
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1923 0.4713 0.7096      0.00768       46097658    40
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.1942 0.4772 0.7173      0.00921       51430596    33
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.1960 0.4842 0.7545      0.00989       85982242    31
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.1982 0.5019 0.8193      0.03104      332830344    10
```

</details>
<details><summary>`IVF262144_HNSW32,PQ16` </summary>
Index size 2574201961

 code_size 16

 log filename: autotune.dbdeep100M.IVF262144_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.1836 0.2816 0.2905      0.00587        4485731    52
nprobe=1,quantizer_efSearch=8,ht=14       0.0076 0.0117 0.0143      0.00367        4498184    82
nprobe=1,quantizer_efSearch=8,ht=34       0.0388 0.0521 0.0574      0.00317        4498184    95
nprobe=1,quantizer_efSearch=8,ht=50       0.1414 0.1918 0.1988      0.00340        4498184    89
nprobe=2,quantizer_efSearch=4,ht=128      0.2045 0.3386 0.3548      0.00345        9027889    87
nprobe=2,quantizer_efSearch=16,ht=54      0.2182 0.3378 0.3508      0.00613        8971412    49
nprobe=2,quantizer_efSearch=16,ht=62      0.2345 0.3870 0.4038      0.00642        8971412    47
nprobe=4,quantizer_efSearch=8,ht=52       0.2494 0.3907 0.4090      0.00770       18044570    39
nprobe=4,quantizer_efSearch=8,ht=64       0.2783 0.4890 0.5203      0.00910       18044570    33
nprobe=8,quantizer_efSearch=8,ht=128      0.3127 0.5870 0.6355      0.00889       36020611    34
nprobe=8,quantizer_efSearch=16,ht=60      0.3215 0.5970 0.6419      0.01642       35894330    19
nprobe=8,quantizer_efSearch=32,ht=64      0.3246 0.6103 0.6594      0.01839       35813940    17
nprobe=16,quantizer_efSearch=16,ht=60     0.3481 0.6769 0.7448      0.02847       71682196    11
nprobe=32,quantizer_efSearch=16,ht=128    0.3662 0.7455 0.8415      0.03110      142839028    10
nprobe=32,quantizer_efSearch=256,ht=58    0.3667 0.7334 0.8169      0.07561      142057612    4
nprobe=32,quantizer_efSearch=256,ht=64    0.3696 0.7554 0.8516      0.08100      142057612    4
nprobe=64,quantizer_efSearch=16,ht=62     0.3733 0.7709 0.8781      0.10512      284342949    3
nprobe=128,quantizer_efSearch=16,ht=128   0.3764 0.7884 0.9093      0.10424      562909507    3
nprobe=64,quantizer_efSearch=512,ht=128   0.3815 0.7964 0.9123      0.10777      282237255    3
nprobe=128,quantizer_efSearch=256,ht=128  0.3876 0.8200 0.9521      0.12686      559123339    3
nprobe=256,quantizer_efSearch=128,ht=64   0.3885 0.8280 0.9677      0.42976     1104525727    1
nprobe=512,quantizer_efSearch=512,ht=128  0.3902 0.8344 0.9806      0.45422     2164692123    1
nprobe=2048,quantizer_efSearch=512,ht=128 0.3906 0.8359 0.9854      1.67676     8201224126    1
```

</details>
<details><summary>`IVF262144_HNSW32,PQ32x4fs` </summary>
Index size 2641272965

 code_size 16

 log filename: autotune.dbdeep100M.IVF262144_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1560 0.3754 0.5056      0.00263       18042519    115
nprobe=2,quantizer_efSearch=4            0.1220 0.2756 0.3492      0.00164        9028730    183
nprobe=4,quantizer_efSearch=4            0.1331 0.3254 0.4407      0.00204       18052163    148
nprobe=2,quantizer_efSearch=8            0.1380 0.3091 0.3911      0.00237        9007035    127
nprobe=4,quantizer_efSearch=8            0.1560 0.3754 0.5056      0.00262       18042519    115
nprobe=8,quantizer_efSearch=4            0.1652 0.4045 0.5882      0.00304       36062097    99
nprobe=8,quantizer_efSearch=8            0.1686 0.4132 0.6011      0.00336       36014875    90
nprobe=8,quantizer_efSearch=16           0.1753 0.4290 0.6231      0.00440       35897350    69
nprobe=16,quantizer_efSearch=8           0.1770 0.4499 0.6868      0.00510       71870520    59
nprobe=16,quantizer_efSearch=16          0.1819 0.4600 0.7025      0.00569       71681186    53
nprobe=32,quantizer_efSearch=16          0.1859 0.4776 0.7548      0.00785      142828271    39
nprobe=32,quantizer_efSearch=32          0.1872 0.4843 0.7655      0.00927      142440722    33
nprobe=64,quantizer_efSearch=32          0.1873 0.4891 0.7939      0.01300      283432884    24
nprobe=64,quantizer_efSearch=64          0.1899 0.4932 0.7990      0.01582      282710171    19
```

</details>
<details><summary>`IVF262144(IVF512,PQ48x4fs,RFlat),PQ16` </summary>
Index size 2511646125

 code_size 16

 log filename: autotune.dbdeep100M.IVF262144_IVF512_PQ48x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.3838 0.7874 0.8952      0.12395      302748568    3
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=128     0.1789 0.2761 0.2863      0.00326        7247317    93
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60     0.1792 0.2739 0.2835      0.00387        7249584    78
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=2,ht=58      0.2023 0.3277 0.3427      0.00807        9377279    38
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=128     0.2499 0.4330 0.4595      0.00632       18423592    48
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64       0.2729 0.4802 0.5060      0.00922       19444738    33
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.2802 0.4834 0.5097      0.00968       19379571    31
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=62     0.2876 0.5044 0.5338      0.01236       20691705    25
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=128   0.2907 0.5118 0.5425      0.01322       38943648    23
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=128    0.3306 0.6160 0.6645      0.01408       41131721    22
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=128    0.3538 0.6891 0.7578      0.01952       82151688    16
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.3746 0.7577 0.8547      0.03543      152550221    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128    0.3841 0.7954 0.9108      0.06637      287432573    5
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128   0.3843 0.7972 0.9128      0.07283      292380761    5
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=64    0.3847 0.7964 0.9106      0.13908      292379806    3
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=62    0.3855 0.7994 0.9231      0.24271      563656366    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60    0.3862 0.8014 0.9279      0.45561     1112686702    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=60  0.3915 0.8223 0.9549      0.48874     1141103913    1
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128   0.3933 0.8327 0.9807      0.49991     2173910926    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=62   0.3936 0.8300 0.9754      1.78130     4252297188    1
nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=256,ht=64 0.3941 0.8341 0.9843      4.06403     8271567013    1
```

</details>
<details><summary>`IVF262144(IVF512,PQ48x4fs,RFlat),PQ32x4fs` </summary>
Index size 2578614729

 code_size 16

 log filename: autotune.dbdeep100M.IVF262144_IVF512_PQ48x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.1091 0.2264 0.2713      0.00169        5208003    178
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.0990 0.2082 0.2486      0.00131        5230572    230
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.1078 0.2238 0.2683      0.00139        5212412    217
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.1091 0.2264 0.2712      0.00150        5209486    200
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.1251 0.2812 0.3565      0.00157        9763487    192
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.1327 0.2978 0.3794      0.00176        9715169    171
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.1374 0.3071 0.3884      0.00184       10412368    164
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.1403 0.3325 0.4398      0.00220       19544392    137
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.1456 0.3477 0.4680      0.00208       18802739    145
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.1490 0.3581 0.4841      0.00219       18764837    137
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.1499 0.3599 0.4869      0.00226       18738248    133
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4   0.1513 0.3681 0.5200      0.00270       37083810    112
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1590 0.3836 0.5145      0.00284       20720160    106
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.1594 0.3893 0.5491      0.00289       37637191    104
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.1605 0.3861 0.5192      0.00303       20696241    99
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.1621 0.3960 0.5772      0.00294       36729479    103
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.1625 0.3965 0.5580      0.00332       38895188    91
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.1685 0.4153 0.5980      0.00299       37393392    101
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1764 0.4282 0.6205      0.00362       38576243    83
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.1774 0.4325 0.6254      0.00458       41166553    66
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.1833 0.4606 0.7026      0.00512       74220985    59
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.1838 0.4619 0.6996      0.00580       76923908    52
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.1849 0.4649 0.7084      0.00602       76747148    50
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.1874 0.4832 0.7598      0.00767      147735908    40
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32 0.1885 0.4873 0.7674      0.00943      147299089    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.1912 0.4918 0.7965      0.01147      288289923    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.1915 0.4937 0.8004      0.01385      292477951    22
```

</details>
<details><summary>`IVF65536_HNSW32,PQ16` </summary>
Index size 2443626089

 code_size 16

 log filename: autotune.dbdeep100M.IVF65536_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3541 0.7696 0.8957      0.04874      558740947    7
nprobe=1,quantizer_efSearch=8,ht=50      0.1696 0.2517 0.2605      0.00280       17772604    108
nprobe=1,quantizer_efSearch=4,ht=60      0.1802 0.2990 0.3142      0.00325       17804642    93
nprobe=1,quantizer_efSearch=16,ht=54     0.1876 0.3007 0.3124      0.00342       17749473    88
nprobe=1,quantizer_efSearch=16,ht=128    0.1975 0.3327 0.3499      0.00351       17749473    86
nprobe=2,quantizer_efSearch=4,ht=128     0.2289 0.4088 0.4393      0.00410       35562638    74
nprobe=2,quantizer_efSearch=16,ht=52     0.2352 0.3901 0.4074      0.00514       35495025    59
nprobe=2,quantizer_efSearch=16,ht=54     0.2429 0.4165 0.4383      0.00499       35495025    61
nprobe=2,quantizer_efSearch=16,ht=62     0.2525 0.4520 0.4849      0.00601       35495025    50
nprobe=4,quantizer_efSearch=8,ht=50      0.2594 0.4376 0.4601      0.00661       71167140    46
nprobe=4,quantizer_efSearch=8,ht=52      0.2711 0.4805 0.5092      0.00663       71167140    46
nprobe=4,quantizer_efSearch=8,ht=64      0.2894 0.5563 0.6106      0.00911       71167140    33
nprobe=8,quantizer_efSearch=8,ht=128     0.3159 0.6479 0.7274      0.00962      141948647    32
nprobe=8,quantizer_efSearch=16,ht=56     0.3186 0.6402 0.7054      0.01279      141594046    24
nprobe=8,quantizer_efSearch=32,ht=56     0.3201 0.6428 0.7081      0.01347      141436198    23
nprobe=8,quantizer_efSearch=16,ht=60     0.3224 0.6612 0.7390      0.01477      141594046    21
nprobe=8,quantizer_efSearch=32,ht=64     0.3252 0.6683 0.7495      0.01716      141436198    18
nprobe=16,quantizer_efSearch=16,ht=60    0.3425 0.7259 0.8290      0.02839      282187040    11
nprobe=32,quantizer_efSearch=16,ht=128   0.3518 0.7694 0.8988      0.03067      559861161    10
nprobe=32,quantizer_efSearch=32,ht=60    0.3541 0.7696 0.8957      0.04971      558740947    7
nprobe=32,quantizer_efSearch=256,ht=58   0.3543 0.7653 0.8845      0.05790      557939088    6
nprobe=32,quantizer_efSearch=256,ht=64   0.3551 0.7764 0.9072      0.06933      557939088    5
nprobe=64,quantizer_efSearch=32,ht=56    0.3559 0.7696 0.8923      0.08044     1104875373    4
nprobe=64,quantizer_efSearch=512,ht=128  0.3610 0.8001 0.9474      0.09178     1101958724    4
nprobe=128,quantizer_efSearch=256,ht=128 0.3635 0.8117 0.9681      0.11840     2167200433    3
nprobe=256,quantizer_efSearch=512,ht=58  0.3636 0.8036 0.9528      0.34902     4244600989    1
nprobe=256,quantizer_efSearch=128,ht=64  0.3638 0.8143 0.9744      0.41836     4248901760    1
nprobe=512,quantizer_efSearch=512,ht=128 0.3644 0.8161 0.9792      0.43445     8272199974    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16` </summary>
Index size 2962276272

 code_size 16

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3679 0.6830 0.7289      0.02980       40194057    11
nprobe=2,quantizer_efSearch=4,ht=46       0.0941 0.1207 0.1246      0.00289        2571988    104
nprobe=2,quantizer_efSearch=4,ht=128      0.1686 0.2491 0.2550      0.00298        2571988    101
nprobe=4,quantizer_efSearch=8,ht=50       0.1868 0.2551 0.2616      0.00506        5128646    60
nprobe=4,quantizer_efSearch=8,ht=52       0.2052 0.2892 0.2964      0.00523        5128646    58
nprobe=8,quantizer_efSearch=8,ht=128      0.2879 0.4874 0.5051      0.00537       10238518    56
nprobe=8,quantizer_efSearch=16,ht=56      0.2891 0.4604 0.4738      0.00977       10176518    31
nprobe=8,quantizer_efSearch=16,ht=60      0.3002 0.5004 0.5171      0.00968       10176518    31
nprobe=8,quantizer_efSearch=32,ht=64      0.3071 0.5185 0.5372      0.01295       10124740    24
nprobe=16,quantizer_efSearch=4,ht=62      0.3086 0.5460 0.5734      0.01389       20436862    22
nprobe=32,quantizer_efSearch=16,ht=128    0.3655 0.6902 0.7427      0.01617       40383620    19
nprobe=32,quantizer_efSearch=32,ht=60     0.3679 0.6830 0.7289      0.03055       40194057    10
nprobe=128,quantizer_efSearch=16,ht=128   0.3835 0.7618 0.8450      0.04697      158617670    7
nprobe=64,quantizer_efSearch=128,ht=60    0.3886 0.7507 0.8166      0.06675       79113642    5
nprobe=64,quantizer_efSearch=128,ht=62    0.3911 0.7617 0.8326      0.06532       79113642    5
nprobe=128,quantizer_efSearch=256,ht=128  0.4048 0.8159 0.9121      0.07737      156241033    4
nprobe=256,quantizer_efSearch=128,ht=64   0.4089 0.8347 0.9413      0.21352      309326195    2
nprobe=512,quantizer_efSearch=512,ht=128  0.4118 0.8531 0.9739      0.23596      604453501    2
nprobe=4096,quantizer_efSearch=512,ht=128 0.4130 0.8611 0.9906      1.31013     4342434247    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16` </summary>
Index size 2702611956

 code_size 16

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.3789 0.7391 0.8062      0.09565      164287466    4
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=58        0.1055 0.1415 0.1442      0.00256        1669635    117
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=52        0.1111 0.1428 0.1454      0.00268        2740884    113
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=62        0.1368 0.1961 0.1996      0.00281        2946249    107
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=58        0.1426 0.2073 0.2110      0.00281        2945720    107
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=60        0.1891 0.2759 0.2802      0.00329        5418437    92
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60      0.2011 0.2925 0.2971      0.00424        8162011    71
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64       0.2069 0.3120 0.3185      0.00563       10715115    54
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=54       0.2184 0.3223 0.3295      0.00540        6558923    56
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=128     0.2592 0.4114 0.4221      0.00677       10685139    45
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=128      0.2861 0.4811 0.4972      0.00699       21114008    43
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128      0.3039 0.5176 0.5373      0.00755       21080031    40
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=56      0.3141 0.5399 0.5626      0.01826       23070874    17
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=64     0.3354 0.6098 0.6441      0.01984       25763091    16
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=58      0.3508 0.6397 0.6770      0.02774       45844782    11
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=58     0.3591 0.6618 0.7038      0.03554       45679618    9
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128    0.3855 0.7651 0.8396      0.04005      162366843    8
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62    0.3873 0.7627 0.8338      0.08364      244404175    4
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=64    0.3878 0.7659 0.8400      0.11301      244401700    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62   0.3991 0.8095 0.8988      0.13802      321547744    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128   0.4106 0.8554 0.9743      0.18297      687722873    2
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=64  0.4109 0.8585 0.9799      0.88061     1224355046    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=512,ht=62   0.4111 0.8560 0.9744      1.46981     2494579053    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=64   0.4113 0.8615 0.9857      1.56038     2465662128    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=128 0.4114 0.8650 0.9939      3.68533     4627221177    1
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16` </summary>
Index size 2540639408

 code_size 16

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3780 0.7490 0.8244      0.02953      141954052    11
nprobe=2,quantizer_efSearch=4,ht=46       0.1262 0.1729 0.1789      0.00246        8969076    123
nprobe=2,quantizer_efSearch=4,ht=128      0.2037 0.3289 0.3417      0.00266        8969076    113
nprobe=2,quantizer_efSearch=16,ht=52      0.2107 0.3078 0.3160      0.00383        8940797    79
nprobe=2,quantizer_efSearch=16,ht=54      0.2219 0.3372 0.3457      0.00387        8940797    78
nprobe=4,quantizer_efSearch=8,ht=50       0.2293 0.3404 0.3515      0.00439       17966523    69
nprobe=4,quantizer_efSearch=8,ht=52       0.2503 0.3872 0.3995      0.00449       17966523    67
nprobe=4,quantizer_efSearch=8,ht=64       0.2797 0.4818 0.5047      0.00543       17966523    56
nprobe=8,quantizer_efSearch=8,ht=128      0.3132 0.5791 0.6182      0.00565       35894562    54
nprobe=8,quantizer_efSearch=16,ht=56      0.3167 0.5593 0.5877      0.00879       35775470    35
nprobe=8,quantizer_efSearch=16,ht=60      0.3245 0.5917 0.6277      0.00914       35775470    33
nprobe=8,quantizer_efSearch=32,ht=64      0.3287 0.6043 0.6440      0.01111       35710640    27
nprobe=16,quantizer_efSearch=4,ht=62      0.3333 0.6381 0.6902      0.01527       71736552    20
nprobe=16,quantizer_efSearch=16,ht=60     0.3526 0.6750 0.7291      0.01588       71414166    19
nprobe=32,quantizer_efSearch=16,ht=128    0.3781 0.7517 0.8332      0.01716      142309319    18
nprobe=32,quantizer_efSearch=256,ht=64    0.3808 0.7600 0.8422      0.04917      141629050    7
nprobe=64,quantizer_efSearch=32,ht=56     0.3818 0.7546 0.8287      0.05315      282542300    6
nprobe=64,quantizer_efSearch=16,ht=62     0.3838 0.7834 0.8790      0.05590      283374589    6
nprobe=128,quantizer_efSearch=16,ht=128   0.3867 0.8000 0.9093      0.06455      561090027    5
nprobe=64,quantizer_efSearch=128,ht=60    0.3903 0.7974 0.8913      0.06249      281594902    5
nprobe=64,quantizer_efSearch=128,ht=62    0.3928 0.8046 0.9039      0.06456      281594902    5
nprobe=64,quantizer_efSearch=512,ht=128   0.3931 0.8081 0.9104      0.07959      281479171    4
nprobe=128,quantizer_efSearch=256,ht=128  0.3990 0.8335 0.9521      0.07923      557649428    4
nprobe=256,quantizer_efSearch=128,ht=64   0.4010 0.8439 0.9701      0.23141     1102126933    2
nprobe=512,quantizer_efSearch=512,ht=128  0.4029 0.8518 0.9852      0.26268     2161658910    2
nprobe=2048,quantizer_efSearch=512,ht=128 0.4037 0.8545 0.9910      0.90539     8200726875    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16` </summary>
Index size 2475855860

 code_size 16

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.3915 0.7936 0.8886      0.05893      303088055    6
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=46       0.1021 0.1327 0.1365      0.00255        4883808    118
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=58       0.1293 0.1883 0.1920      0.00254        5258717    119
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=64       0.1328 0.1966 0.2007      0.00251        5258719    120
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=128     0.1748 0.2665 0.2717      0.00289        7223436    104
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60     0.1769 0.2651 0.2702      0.00301        7212416    100
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=2,ht=58      0.2017 0.3197 0.3301      0.00339        9386875    89
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=62      0.2354 0.3769 0.3887      0.00383       11694592    79
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=128     0.2423 0.4198 0.4416      0.00373       18433056    81
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.2797 0.4714 0.4909      0.00557       19370328    54
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=62     0.2880 0.4935 0.5162      0.00715       20660589    42
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=54      0.3024 0.5163 0.5414      0.00936       37275457    33
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=128    0.3307 0.6010 0.6434      0.01019       41093304    30
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=128    0.3549 0.6686 0.7267      0.01156       82195947    26
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.3817 0.7558 0.8422      0.01887      152612879    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128    0.3928 0.8022 0.9064      0.03421      287607759    9
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128   0.3929 0.8037 0.9083      0.03829      292510761    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60    0.3944 0.8114 0.9267      0.20645     1114681642    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=60  0.4006 0.8307 0.9528      0.23849     1142498020    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128   0.4018 0.8438 0.9827      0.25294     2177048263    2
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=128 0.4025 0.8479 0.9909      1.72376    16036050034    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16` </summary>
Index size 2435229360

 code_size 16

 log filename: autotune.dbdeep100M.OPQ16_64_IVF65536_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3712 0.7859 0.8958      0.03732      558169337    9
nprobe=1,quantizer_efSearch=8,ht=50       0.1684 0.2508 0.2584      0.00252       17749327    119
nprobe=1,quantizer_efSearch=4,ht=60       0.1808 0.2936 0.3047      0.00266       17757612    113
nprobe=1,quantizer_efSearch=8,ht=58       0.1960 0.3146 0.3260      0.00270       17749327    112
nprobe=1,quantizer_efSearch=16,ht=58      0.1985 0.3186 0.3302      0.00295       17708425    102
nprobe=1,quantizer_efSearch=16,ht=128     0.1994 0.3260 0.3386      0.00301       17708425    100
nprobe=2,quantizer_efSearch=16,ht=54      0.2451 0.4155 0.4331      0.00363       35445733    83
nprobe=2,quantizer_efSearch=16,ht=62      0.2564 0.4497 0.4756      0.00463       35445733    65
nprobe=4,quantizer_efSearch=8,ht=50       0.2606 0.4364 0.4556      0.00457       71079489    66
nprobe=4,quantizer_efSearch=8,ht=52       0.2784 0.4818 0.5054      0.00486       71079489    62
nprobe=4,quantizer_efSearch=8,ht=64       0.2984 0.5597 0.6025      0.00724       71079489    42
nprobe=4,quantizer_efSearch=64,ht=64      0.3064 0.5733 0.6162      0.00923       70859179    33
nprobe=8,quantizer_efSearch=8,ht=128      0.3286 0.6490 0.7166      0.00892      141778203    34
nprobe=8,quantizer_efSearch=16,ht=56      0.3327 0.6436 0.6980      0.00965      141434634    32
nprobe=8,quantizer_efSearch=32,ht=56      0.3344 0.6461 0.7007      0.01016      141300507    30
nprobe=8,quantizer_efSearch=16,ht=60      0.3381 0.6656 0.7299      0.01136      141434634    27
nprobe=8,quantizer_efSearch=32,ht=64      0.3411 0.6725 0.7405      0.01344      141300507    23
nprobe=16,quantizer_efSearch=16,ht=60     0.3567 0.7345 0.8240      0.02025      281745780    15
nprobe=16,quantizer_efSearch=64,ht=64     0.3599 0.7444 0.8394      0.02492      281240146    13
nprobe=32,quantizer_efSearch=16,ht=128    0.3705 0.7849 0.8996      0.02924      559168062    11
nprobe=32,quantizer_efSearch=32,ht=60     0.3712 0.7859 0.8958      0.03732      558169337    9
nprobe=32,quantizer_efSearch=256,ht=58    0.3713 0.7801 0.8837      0.04384      557394954    7
nprobe=32,quantizer_efSearch=256,ht=64    0.3734 0.7923 0.9066      0.05394      557394954    6
nprobe=64,quantizer_efSearch=32,ht=56     0.3736 0.7868 0.8927      0.05908     1104182366    6
nprobe=64,quantizer_efSearch=128,ht=60    0.3799 0.8136 0.9381      0.07257     1101787014    5
nprobe=64,quantizer_efSearch=128,ht=62    0.3801 0.8163 0.9454      0.08071     1101787014    4
nprobe=64,quantizer_efSearch=512,ht=128   0.3810 0.8181 0.9497      0.08487     1101585275    4
nprobe=128,quantizer_efSearch=256,ht=128  0.3850 0.8327 0.9734      0.11718     2167061093    3
nprobe=256,quantizer_efSearch=128,ht=64   0.3863 0.8375 0.9826      0.32014     4250413970    1
nprobe=512,quantizer_efSearch=512,ht=128  0.3864 0.8404 0.9881      0.43246     8279790560    1
nprobe=2048,quantizer_efSearch=256,ht=128 0.3865 0.8406 0.9896      1.45617    29146149410    1
nprobe=2048,quantizer_efSearch=512,ht=128 0.3866 0.8411 0.9901      1.54247    30800771291    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576_HNSW32,PQ32x4fs` </summary>
Index size 3233647052

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF1048576_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1580 0.3323 0.3973      0.00301        5153967    100
nprobe=2,quantizer_efSearch=4            0.1123 0.2246 0.2561      0.00180        2593465    167
nprobe=4,quantizer_efSearch=4            0.1315 0.2828 0.3408      0.00197        5171388    153
nprobe=2,quantizer_efSearch=8            0.1328 0.2600 0.2948      0.00284        2581753    106
nprobe=4,quantizer_efSearch=8            0.1580 0.3323 0.3973      0.00295        5153967    102
nprobe=8,quantizer_efSearch=4            0.1705 0.3855 0.4887      0.00289       10321650    104
nprobe=8,quantizer_efSearch=8            0.1761 0.3937 0.5004      0.00320       10298661    94
nprobe=16,quantizer_efSearch=4           0.1827 0.4349 0.5734      0.00360       20578040    84
nprobe=16,quantizer_efSearch=8           0.1934 0.4607 0.6115      0.00433       20522070    70
nprobe=16,quantizer_efSearch=16          0.1976 0.4736 0.6283      0.00533       20448913    57
nprobe=32,quantizer_efSearch=8           0.1999 0.4948 0.6824      0.00558       40843185    54
nprobe=64,quantizer_efSearch=8           0.2018 0.5059 0.7196      0.00728       80955413    42
nprobe=32,quantizer_efSearch=16          0.2095 0.5170 0.7148      0.00673       40670730    45
nprobe=64,quantizer_efSearch=16          0.2137 0.5359 0.7685      0.00927       80803810    33
nprobe=64,quantizer_efSearch=32          0.2173 0.5483 0.7880      0.01172       80324283    26
nprobe=64,quantizer_efSearch=64          0.2179 0.5547 0.7967      0.01629       79877273    19
nprobe=128,quantizer_efSearch=64         0.2203 0.5714 0.8419      0.02024      158375822    15
nprobe=128,quantizer_efSearch=128        0.2216 0.5747 0.8458      0.02900      157631420    11
nprobe=256,quantizer_efSearch=256        0.2229 0.5830 0.8716      0.05253      310087592    6
nprobe=2048,quantizer_efSearch=256       0.2230 0.5860 0.8897      0.12315     2287392796    3
nprobe=2048,quantizer_efSearch=512       0.2234 0.5861 0.8928      0.18722     2321009105    2
```

</details>
<details><summary>`OPQ32_64,IVF1048576_HNSW32,PQ32x4fsr` </summary>
Index size 3233459660

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF1048576_HNSW32_PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2972 0.5179 0.5423      0.02686       10090594    12
nprobe=1,quantizer_efSearch=4            0.1234 0.1753 0.1786      0.00212        1289434    142
nprobe=1,quantizer_efSearch=16           0.1457 0.2072 0.2110      0.00485        1282963    62
nprobe=2,quantizer_efSearch=16           0.1979 0.2980 0.3054      0.00522        2555090    58
nprobe=4,quantizer_efSearch=16           0.2470 0.4005 0.4145      0.00587        5095969    52
nprobe=8,quantizer_efSearch=16           0.2906 0.5077 0.5323      0.00689       10179745    44
nprobe=16,quantizer_efSearch=16          0.3196 0.5983 0.6425      0.00904       20312744    34
nprobe=32,quantizer_efSearch=16          0.3434 0.6757 0.7425      0.01904       40387037    16
nprobe=32,quantizer_efSearch=32          0.3500 0.6869 0.7554      0.02090       40189386    15
nprobe=32,quantizer_efSearch=64          0.3538 0.6927 0.7623      0.02527       40017011    12
nprobe=64,quantizer_efSearch=16          0.3562 0.7209 0.8073      0.03372       80250383    9
nprobe=64,quantizer_efSearch=128         0.3725 0.7535 0.8471      0.04797       79124916    7
nprobe=64,quantizer_efSearch=256         0.3729 0.7539 0.8478      0.06417       79016557    5
nprobe=128,quantizer_efSearch=128        0.3820 0.7951 0.9083      0.08091      156632847    4
nprobe=128,quantizer_efSearch=256        0.3831 0.7954 0.9095      0.09818      156269033    4
nprobe=256,quantizer_efSearch=128        0.3857 0.8166 0.9443      0.12984      309371414    3
nprobe=256,quantizer_efSearch=256        0.3876 0.8186 0.9482      0.14611      308190654    3
nprobe=512,quantizer_efSearch=128        0.3892 0.8237 0.9609      0.22874      608850811    2
nprobe=512,quantizer_efSearch=256        0.3901 0.8296 0.9716      0.24207      606469096    2
nprobe=512,quantizer_efSearch=512        0.3904 0.8292 0.9729      0.27896      604515834    2
nprobe=1024,quantizer_efSearch=512       0.3916 0.8326 0.9808      0.49766     1185048672    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 2973571088

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1525 0.3246 0.3966      0.00246       11695671    122
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.0934 0.1678 0.1828      0.00131        2019962    229
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.0992 0.1792 0.1929      0.00145        2743305    208
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1029 0.1839 0.1993      0.00151        2738655    199
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1128 0.2233 0.2534      0.00150        3299624    200
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1162 0.2334 0.2653      0.00180        3297135    167
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1169 0.2375 0.2754      0.00171        5866505    176
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1294 0.2547 0.2885      0.00178        4012914    169
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1316 0.2712 0.3215      0.00179        5863769    168
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1477 0.2997 0.3561      0.00195        6579302    155
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1513 0.3172 0.3792      0.00208        6563654    145
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1549 0.3125 0.3704      0.00230        7974395    131
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1562 0.3266 0.3910      0.00256        6557160    118
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1665 0.3647 0.4568      0.00257       11684222    117
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1734 0.3859 0.4870      0.00301       11651241    100
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1831 0.4065 0.5095      0.00318       13046508    95
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1848 0.4123 0.5186      0.00407       13023959    74
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1931 0.4631 0.6115      0.00494       23093550    61
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1946 0.4563 0.5988      0.00494       25853084    61
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1983 0.4743 0.6248      0.00518       25826514    58
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2009 0.4939 0.6746      0.00600       43204044    51
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2047 0.5062 0.6974      0.00741       43081493    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2115 0.5201 0.7150      0.00715       45766157    42
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2139 0.5263 0.7221      0.00896       51043188    34
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2142 0.5476 0.7814      0.01133       85147936    27
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2145 0.5461 0.7734      0.01161       90520002    26
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2168 0.5562 0.7918      0.01266       90296680    24
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2185 0.5586 0.7994      0.01953      100442822    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2190 0.5730 0.8409      0.01948      167819290    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2199 0.5750 0.8450      0.02235      177974961    14
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2216 0.5763 0.8470      0.03073      198519250    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2219 0.5832 0.8695      0.03944      350326843    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.2220 0.5832 0.8694      0.04290      391462365    7
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.2225 0.5850 0.8817      0.05804      689735882    6
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2228 0.5875 0.8860      0.06062      646921435    5
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.2231 0.5874 0.8861      0.06597      687863795    5
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.2238 0.5861 0.8899      0.09060     1268142557    4
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.2240 0.5860 0.8909      0.14197     1264949659    3
```

</details>
<details><summary>`OPQ32_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 2973808656

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2489 0.4124 0.4272      0.00453       13073185    67
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2    0.1306 0.1837 0.1875      0.00222        2019267    136
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.1350 0.1989 0.2041      0.00170        2943945    177
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.1420 0.2107 0.2165      0.00197        2938756    153
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1534 0.2221 0.2270      0.00230        4016307    131
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1742 0.2584 0.2650      0.00232        3301482    130
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1844 0.2723 0.2789      0.00245        4013364    123
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1914 0.2825 0.2894      0.00260        4005142    116
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.2001 0.3163 0.3270      0.00270        5873347    112
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.2104 0.3348 0.3475      0.00285        5871760    106
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2209 0.3480 0.3594      0.00297        6579415    102
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2310 0.3695 0.3823      0.00317        6571380    95
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2333 0.3763 0.3904      0.00338        6564708    89
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.2360 0.3985 0.4165      0.00395       10968679    76
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.2420 0.4131 0.4340      0.00418       10953903    72
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2634 0.4468 0.4659      0.00429       11682787    71
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2747 0.4708 0.4910      0.00485       13068907    62
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2755 0.4723 0.4951      0.00489       11642857    62
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2793 0.4791 0.4997      0.00588       15782118    52
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2897 0.5019 0.5257      0.00612       15750294    49
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2905 0.5029 0.5275      0.00623       13022153    49
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.2965 0.5343 0.5677      0.00647       21813817    47
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.2998 0.5463 0.5836      0.00681       21764237    45
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3109 0.5678 0.6038      0.00714       23167174    42
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3174 0.5822 0.6225      0.00745       23108906    41
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.3179 0.5859 0.6288      0.00827       23099423    37
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3182 0.5847 0.6219      0.01277       41344279    24
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.3263 0.6075 0.6531      0.01292       31023040    24
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32  0.3264 0.6079 0.6533      0.01583       31017779    19
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.3268 0.6086 0.6542      0.01604       41331023    19
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3397 0.6464 0.7024      0.01662       43188370    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3456 0.6604 0.7214      0.01673       43121621    18
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3481 0.6667 0.7263      0.01698       45803604    18
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3543 0.6819 0.7461      0.01742       45706419    18
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.3550 0.6908 0.7577      0.02473       61270503    13
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3564 0.6898 0.7554      0.02748       80416122    11
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.3590 0.6978 0.7700      0.03162       90652496    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3701 0.7396 0.8256      0.03171       85126583    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3723 0.7392 0.8254      0.03296      100141128    10
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.3750 0.7540 0.8444      0.04193      100171108    8
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.3753 0.7544 0.8450      0.04556      100300073    7
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.3767 0.7557 0.8519      0.06999      168981636    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3835 0.7861 0.8930      0.07151      177737098    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3854 0.7923 0.9035      0.07392      177209353    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.3869 0.7939 0.9066      0.07812      177257388    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.3895 0.8098 0.9336      0.12627      320566010    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3902 0.8129 0.9387      0.13213      349108624    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3921 0.8167 0.9456      0.13331      329414066    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3923 0.8173 0.9462      0.13869      349254333    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.3925 0.8176 0.9467      0.14289      390416247    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3951 0.8285 0.9674      0.24211      647372863    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3954 0.8315 0.9710      0.26652      645298928    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.3955 0.8324 0.9721      0.27894      646412524    2
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fs` </summary>
Index size 2607554764

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF262144_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2251 0.5829 0.8799      0.06978     1102857975    5
nprobe=2,quantizer_efSearch=4            0.1335 0.2782 0.3374      0.00136        9003701    220
nprobe=4,quantizer_efSearch=4            0.1485 0.3351 0.4341      0.00175       18020602    172
nprobe=2,quantizer_efSearch=8            0.1541 0.3167 0.3803      0.00174        8980678    173
nprobe=4,quantizer_efSearch=8            0.1728 0.3871 0.4939      0.00218       18012817    138
nprobe=8,quantizer_efSearch=4            0.1854 0.4358 0.5829      0.00259       36034414    117
nprobe=8,quantizer_efSearch=8            0.1890 0.4452 0.5954      0.00272       35997666    111
nprobe=8,quantizer_efSearch=16           0.1969 0.4635 0.6163      0.00346       35872847    87
nprobe=16,quantizer_efSearch=8           0.2056 0.4981 0.6940      0.00425       71820117    71
nprobe=16,quantizer_efSearch=16          0.2080 0.5074 0.7074      0.00463       71630825    65
nprobe=32,quantizer_efSearch=8           0.2115 0.5231 0.7520      0.00586      143222575    52
nprobe=32,quantizer_efSearch=16          0.2162 0.5434 0.7793      0.00655      142743938    46
nprobe=32,quantizer_efSearch=32          0.2185 0.5476 0.7876      0.00804      142376429    38
nprobe=32,quantizer_efSearch=64          0.2194 0.5486 0.7892      0.01065      142120565    29
nprobe=64,quantizer_efSearch=32          0.2224 0.5661 0.8342      0.01207      283408625    25
nprobe=64,quantizer_efSearch=64          0.2238 0.5700 0.8377      0.01329      282701486    23
nprobe=64,quantizer_efSearch=128         0.2240 0.5700 0.8389      0.01821      282380130    17
nprobe=128,quantizer_efSearch=128        0.2242 0.5802 0.8671      0.02271      559593029    14
nprobe=256,quantizer_efSearch=128        0.2248 0.5820 0.8790      0.03066     1105228108    10
nprobe=256,quantizer_efSearch=256        0.2250 0.5828 0.8797      0.04130     1103389699    8
nprobe=256,quantizer_efSearch=512        0.2251 0.5829 0.8799      0.06581     1102857975    5
nprobe=1024,quantizer_efSearch=512       0.2254 0.5843 0.8884      0.11891     4236140248    3
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fsr` </summary>
Index size 2607709900

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF262144_HNSW32_PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3179 0.5953 0.6475      0.01306       35691455    23
nprobe=1,quantizer_efSearch=4            0.1556 0.2401 0.2479      0.00120        4490245    250
nprobe=1,quantizer_efSearch=16           0.1764 0.2702 0.2783      0.00240        4465019    125
nprobe=2,quantizer_efSearch=16           0.2285 0.3790 0.3961      0.00265        8936301    114
nprobe=2,quantizer_efSearch=32           0.2303 0.3822 0.3991      0.00405        8922976    74
nprobe=4,quantizer_efSearch=16           0.2772 0.4893 0.5193      0.00318       17915148    95
nprobe=8,quantizer_efSearch=16           0.3151 0.5910 0.6426      0.00425       35792177    71
nprobe=8,quantizer_efSearch=32           0.3174 0.5948 0.6469      0.00556       35727084    54
nprobe=16,quantizer_efSearch=16          0.3433 0.6714 0.7482      0.00618       71424031    49
nprobe=16,quantizer_efSearch=64          0.3462 0.6785 0.7565      0.01001       71168271    30
nprobe=32,quantizer_efSearch=16          0.3623 0.7354 0.8332      0.01537      142306220    20
nprobe=32,quantizer_efSearch=32          0.3653 0.7437 0.8435      0.01587      141951808    19
nprobe=32,quantizer_efSearch=64          0.3670 0.7463 0.8457      0.01867      141714182    17
nprobe=64,quantizer_efSearch=16          0.3716 0.7668 0.8820      0.02768      283397871    11
nprobe=64,quantizer_efSearch=64          0.3796 0.7841 0.9056      0.03100      281900942    10
nprobe=64,quantizer_efSearch=128         0.3804 0.7857 0.9073      0.03504      281597611    9
nprobe=64,quantizer_efSearch=256         0.3805 0.7856 0.9072      0.04393      281507921    7
nprobe=128,quantizer_efSearch=128        0.3850 0.8111 0.9490      0.06833      557959166    5
nprobe=128,quantizer_efSearch=256        0.3854 0.8109 0.9488      0.08171      557583271    4
nprobe=128,quantizer_efSearch=512        0.3857 0.8113 0.9491      0.11171      557489283    3
nprobe=256,quantizer_efSearch=128        0.3881 0.8220 0.9704      0.11796     1102098800    3
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 2542813200

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1203 0.2280 0.2634      0.00139        5212180    217
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.0891 0.1690 0.1966      0.00125        4696633    240
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1055 0.2029 0.2329      0.00129        5222153    233
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1185 0.2257 0.2610      0.00130        5221585    231
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1196 0.2271 0.2626      0.00130        5212315    231
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1222 0.2314 0.2673      0.00142        5893574    212
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1244 0.2348 0.2712      0.00143        5888311    210
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1474 0.3055 0.3679      0.00162        9731823    186
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1508 0.3087 0.3708      0.00161       10406628    187
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1551 0.3201 0.3838      0.00189       10375382    159
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1678 0.3747 0.4783      0.00198       18760159    152
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1686 0.3768 0.4808      0.00215       18747969    140
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.1729 0.3790 0.4775      0.00230       20727030    131
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1796 0.3943 0.5013      0.00245       20693246    123
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1805 0.3994 0.5073      0.00267       20683917    113
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1847 0.4261 0.5706      0.00267       36794807    113
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1978 0.4586 0.6110      0.00341       38578206    89
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2030 0.4863 0.6739      0.00416       73392579    73
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2080 0.5036 0.6963      0.00518       77008705    58
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2088 0.5223 0.7530      0.00660      144374132    46
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2148 0.5363 0.7701      0.00760      147948439    40
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.2150 0.5377 0.7714      0.00897      153050330    34
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2180 0.5459 0.7865      0.00851      147339446    36
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2198 0.5559 0.8145      0.00947      286867588    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2201 0.5618 0.8248      0.01076      288740770    28
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2205 0.5587 0.8217      0.01138      285934460    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2229 0.5665 0.8334      0.01127      287811924    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2237 0.5687 0.8365      0.01306      292744926    23
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2241 0.5774 0.8616      0.01686      571071378    18
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2251 0.5813 0.8767      0.03005     1125537027    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2255 0.5832 0.8797      0.03259     1122469430    10
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.2256 0.5841 0.8858      0.05623     2210338496    6
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2261 0.5854 0.8889      0.07151     4265898265    5
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 2542954000

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.3808 0.8104 0.9484      0.06538      579123459    5
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1757 0.2876 0.2978      0.00153        9786883    197
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2    0.2022 0.3326 0.3456      0.00163        9370814    185
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2231 0.3738 0.3879      0.00179       10391376    168
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2596 0.4603 0.4873      0.00202       18770991    149
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2611 0.4643 0.4924      0.00217       18746551    139
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2697 0.4787 0.5066      0.00224       19404627    134
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2715 0.4819 0.5115      0.00256       19375024    118
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2764 0.4881 0.5165      0.00279       20702332    108
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2769 0.4906 0.5204      0.00296       20676769    102
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2928 0.5487 0.5969      0.00308       36769469    98
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2939 0.5495 0.5984      0.00335       36728184    90
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2985 0.5580 0.6043      0.00316       37431405    95
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3061 0.5783 0.6286      0.00335       37302124    90
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3109 0.5863 0.6363      0.00373       38597502    81
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3133 0.5906 0.6414      0.00398       38548674    76
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3141 0.5925 0.6440      0.00507       41129350    60
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3227 0.6155 0.6751      0.00530       74873373    57
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3319 0.6548 0.7273      0.00533       73110907    57
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3398 0.6720 0.7463      0.00559       74207424    54
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.3401 0.6730 0.7477      0.00597       74146980    51
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3434 0.6754 0.7512      0.00648       76698064    47
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.3436 0.6782 0.7544      0.00779       76611495    39
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3437 0.6759 0.7518      0.00812       81877877    37
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.3439 0.6792 0.7556      0.00953       81749603    32
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.3440 0.6795 0.7559      0.01130       92188557    27
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=64  0.3441 0.6793 0.7558      0.01081       81782490    28
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.3442 0.6796 0.7561      0.01284       92205499    24
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3569 0.7250 0.8223      0.01417      145489351    22
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.3603 0.7345 0.8324      0.01638      144927244    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3623 0.7404 0.8401      0.01562      147385214    20
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.3639 0.7426 0.8422      0.01736      147238464    18
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.3640 0.7442 0.8440      0.01875      152345270    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3756 0.7785 0.8956      0.02778      293277962    11
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.3783 0.7847 0.9027      0.02962      287526280    11
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.3790 0.7857 0.9053      0.03003      292457752    11
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.3792 0.7866 0.9061      0.04096      322672485    8
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.3793 0.7865 0.9063      0.04692      302623303    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3802 0.8104 0.9479      0.06921      579179689    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.3808 0.8104 0.9484      0.07266      578992766    5
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.3809 0.8102 0.9482      0.07598      599278043    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3865 0.8188 0.9678      0.12412     1114922366    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3866 0.8204 0.9696      0.12257     1112703362    3
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fs` </summary>
Index size 2451958476

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF65536_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1797 0.4292 0.5781      0.00300       71153793    101
nprobe=2,quantizer_efSearch=4            0.1457 0.3260 0.4202      0.00180       35580161    167
nprobe=2,quantizer_efSearch=8            0.1604 0.3550 0.4564      0.00195       35561115    155
nprobe=2,quantizer_efSearch=16           0.1625 0.3618 0.4636      0.00230       35478638    131
nprobe=4,quantizer_efSearch=8            0.1797 0.4292 0.5781      0.00305       71153793    99
nprobe=4,quantizer_efSearch=16           0.1849 0.4405 0.5916      0.00337       71004333    89
nprobe=4,quantizer_efSearch=32           0.1856 0.4419 0.5932      0.00387       70947820    78
nprobe=8,quantizer_efSearch=4            0.1903 0.4680 0.6651      0.00448      142133025    67
nprobe=8,quantizer_efSearch=8            0.1942 0.4773 0.6778      0.00445      141955869    68
nprobe=8,quantizer_efSearch=16           0.2024 0.4942 0.7002      0.00482      141665883    63
nprobe=8,quantizer_efSearch=32           0.2033 0.4961 0.7032      0.00560      141475555    54
nprobe=16,quantizer_efSearch=8           0.2064 0.5217 0.7638      0.00586      282762924    52
nprobe=16,quantizer_efSearch=16          0.2106 0.5300 0.7772      0.00627      282209332    48
nprobe=16,quantizer_efSearch=32          0.2120 0.5336 0.7828      0.00669      281782537    45
nprobe=32,quantizer_efSearch=16          0.2165 0.5522 0.8273      0.00808      559995995    38
nprobe=32,quantizer_efSearch=32          0.2180 0.5549 0.8326      0.00894      558874780    34
nprobe=64,quantizer_efSearch=16          0.2182 0.5632 0.8543      0.01028     1108947827    30
nprobe=64,quantizer_efSearch=32          0.2198 0.5676 0.8627      0.01089     1106000617    28
nprobe=64,quantizer_efSearch=64          0.2203 0.5686 0.8651      0.01562     1104200394    20
nprobe=64,quantizer_efSearch=128         0.2204 0.5686 0.8654      0.02197     1103455120    14
nprobe=128,quantizer_efSearch=32         0.2212 0.5726 0.8768      0.02054     2180488041    15
nprobe=128,quantizer_efSearch=128        0.2215 0.5754 0.8827      0.02820     2171209672    11
nprobe=256,quantizer_efSearch=256        0.2227 0.5774 0.8896      0.04259     4253318171    8
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fsr` </summary>
Index size 2451949260

 code_size 16

 log filename: autotune.dbdeep100M.OPQ32_64_IVF65536_HNSW32_PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3652 0.8147 0.9845      0.23171     8280197395    2
nprobe=1,quantizer_efSearch=16           0.1921 0.3221 0.3393      0.00174       17703751    173
nprobe=2,quantizer_efSearch=16           0.2432 0.4436 0.4755      0.00238       35456837    126
nprobe=4,quantizer_efSearch=16           0.2910 0.5606 0.6155      0.00339       70927978    89
nprobe=8,quantizer_efSearch=16           0.3233 0.6559 0.7377      0.00493      141458495    61
nprobe=8,quantizer_efSearch=64           0.3262 0.6580 0.7407      0.00685      141292801    44
nprobe=16,quantizer_efSearch=128         0.3426 0.7274 0.8387      0.01235      281236718    25
nprobe=32,quantizer_efSearch=256         0.3566 0.7704 0.9046      0.02957      557376691    11
nprobe=64,quantizer_efSearch=16          0.3580 0.7829 0.9308      0.03322     1106962279    10
nprobe=64,quantizer_efSearch=256         0.3614 0.7933 0.9464      0.04161     1101731353    8
nprobe=64,quantizer_efSearch=512         0.3615 0.7933 0.9464      0.06248     1101708791    5
nprobe=128,quantizer_efSearch=256        0.3634 0.8087 0.9698      0.07282     2167045320    5
nprobe=128,quantizer_efSearch=512        0.3635 0.8087 0.9698      0.09212     2166936645    4
nprobe=256,quantizer_efSearch=128        0.3639 0.8134 0.9779      0.11727     4250420817    3
nprobe=256,quantizer_efSearch=512        0.3643 0.8154 0.9805      0.13903     4246415933    3
nprobe=512,quantizer_efSearch=256        0.3654 0.8143 0.9842      0.23387     8288188680    2
nprobe=1024,quantizer_efSearch=512       0.3658 0.8149 0.9861      0.43811    16048701575    1
```

</details>
<details><summary>`PCAR32,IVF1048576_HNSW32,SQ4` </summary>
Index size 2828018773

 code_size 16

 log filename: autotune.dbdeep100M.PCAR32_IVF1048576_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1890 0.4856 0.7975      0.10023      293268783    3
nprobe=1,quantizer_efSearch=4            0.0772 0.1385 0.1491      0.00229        1241045    131
nprobe=2,quantizer_efSearch=4            0.0973 0.1928 0.2150      0.00237        2455802    127
nprobe=4,quantizer_efSearch=4            0.1175 0.2467 0.2921      0.00250        4878852    121
nprobe=8,quantizer_efSearch=4            0.1469 0.3298 0.4205      0.00350        9742045    86
nprobe=8,quantizer_efSearch=8            0.1504 0.3390 0.4318      0.00371        9734183    81
nprobe=16,quantizer_efSearch=4           0.1546 0.3684 0.5004      0.00501       19389087    60
nprobe=8,quantizer_efSearch=16           0.1574 0.3525 0.4501      0.00477        9688656    63
nprobe=16,quantizer_efSearch=8           0.1618 0.3881 0.5306      0.00548       19353923    55
nprobe=16,quantizer_efSearch=16          0.1664 0.3951 0.5434      0.00658       19300793    46
nprobe=16,quantizer_efSearch=32          0.1676 0.3985 0.5476      0.00822       19227646    37
nprobe=32,quantizer_efSearch=8           0.1697 0.4191 0.6064      0.00826       38475110    37
nprobe=32,quantizer_efSearch=16          0.1763 0.4326 0.6345      0.00913       38357037    33
nprobe=32,quantizer_efSearch=32          0.1788 0.4370 0.6421      0.01056       38213901    29
nprobe=32,quantizer_efSearch=64          0.1794 0.4394 0.6449      0.01455       38100244    21
nprobe=64,quantizer_efSearch=16          0.1798 0.4497 0.6913      0.01470       76103970    21
nprobe=64,quantizer_efSearch=32          0.1836 0.4597 0.7068      0.01623       75794179    19
nprobe=64,quantizer_efSearch=64          0.1845 0.4616 0.7112      0.01920       75523574    16
nprobe=64,quantizer_efSearch=128         0.1855 0.4623 0.7121      0.02657       75348736    12
nprobe=128,quantizer_efSearch=32         0.1858 0.4698 0.7516      0.02697      150192923    12
nprobe=128,quantizer_efSearch=64         0.1868 0.4751 0.7620      0.02998      149519935    11
nprobe=128,quantizer_efSearch=128        0.1877 0.4773 0.7656      0.03570      149022566    9
nprobe=128,quantizer_efSearch=256        0.1878 0.4774 0.7662      0.05018      148791584    6
nprobe=256,quantizer_efSearch=128        0.1884 0.4852 0.7946      0.05750      294306473    6
nprobe=256,quantizer_efSearch=256        0.1889 0.4855 0.7973      0.06903      293560358    5
nprobe=256,quantizer_efSearch=512        0.1890 0.4856 0.7975      0.10019      293268783    3
nprobe=512,quantizer_efSearch=128        0.1891 0.4900 0.8122      0.10365      579306504    3
nprobe=512,quantizer_efSearch=256        0.1894 0.4909 0.8146      0.13224      577848205    3
nprobe=512,quantizer_efSearch=512        0.1896 0.4910 0.8154      0.15089      576651028    2
nprobe=1024,quantizer_efSearch=256       0.1901 0.4925 0.8216      0.19771     1133573731    2
nprobe=1024,quantizer_efSearch=512       0.1903 0.4927 0.8229      0.23595     1131855499    2
nprobe=4096,quantizer_efSearch=512       0.1904 0.4935 0.8259      0.71437     4164529149    1
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ4` </summary>
Index size 2507045205

 code_size 16

 log filename: autotune.dbdeep100M.PCAR32_IVF262144_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1859 0.4856 0.8151      0.12351     1082109029    3
nprobe=1,quantizer_efSearch=4            0.0925 0.1795 0.2073      0.00212        4476653    142
nprobe=2,quantizer_efSearch=4            0.1132 0.2406 0.2936      0.00215        8914984    140
nprobe=2,quantizer_efSearch=8            0.1225 0.2626 0.3192      0.00237        8915103    127
nprobe=2,quantizer_efSearch=16           0.1248 0.2685 0.3250      0.00281        8894274    107
nprobe=4,quantizer_efSearch=4            0.1284 0.2940 0.3832      0.00288       17793947    105
nprobe=4,quantizer_efSearch=8            0.1428 0.3257 0.4238      0.00313       17820585    96
nprobe=4,quantizer_efSearch=16           0.1455 0.3338 0.4337      0.00367       17771593    82
nprobe=4,quantizer_efSearch=32           0.1471 0.3358 0.4358      0.00443       17747613    68
nprobe=8,quantizer_efSearch=8            0.1553 0.3816 0.5282      0.00469       35498989    64
nprobe=8,quantizer_efSearch=16           0.1608 0.3929 0.5450      0.00513       35407275    59
nprobe=8,quantizer_efSearch=32           0.1632 0.3953 0.5478      0.00628       35365108    48
nprobe=8,quantizer_efSearch=64           0.1636 0.3956 0.5482      0.00819       35347508    37
nprobe=16,quantizer_efSearch=16          0.1699 0.4304 0.6341      0.00799       70578732    38
nprobe=16,quantizer_efSearch=32          0.1723 0.4352 0.6407      0.00878       70460314    35
nprobe=16,quantizer_efSearch=64          0.1729 0.4357 0.6410      0.01002       70409760    30
nprobe=32,quantizer_efSearch=32          0.1788 0.4602 0.7130      0.01388      140049649    22
nprobe=32,quantizer_efSearch=64          0.1798 0.4616 0.7157      0.01687      139894690    18
nprobe=32,quantizer_efSearch=128         0.1801 0.4624 0.7169      0.02286      139837162    14
nprobe=64,quantizer_efSearch=32          0.1813 0.4717 0.7647      0.02507      278068103    12
nprobe=64,quantizer_efSearch=128         0.1821 0.4751 0.7694      0.03016      277376914    10
nprobe=64,quantizer_efSearch=256         0.1823 0.4754 0.7698      0.03849      277317291    8
nprobe=128,quantizer_efSearch=32         0.1825 0.4766 0.7915      0.05189      550810909    6
nprobe=128,quantizer_efSearch=128        0.1841 0.4821 0.8008      0.05266      548840662    6
nprobe=128,quantizer_efSearch=256        0.1844 0.4826 0.8012      0.05882      548577435    6
nprobe=128,quantizer_efSearch=512        0.1845 0.4827 0.8013      0.08340      548514884    4
nprobe=256,quantizer_efSearch=64         0.1846 0.4829 0.8110      0.09753     1085269065    4
nprobe=256,quantizer_efSearch=128        0.1854 0.4850 0.8145      0.10156     1083662511    3
nprobe=256,quantizer_efSearch=256        0.1858 0.4854 0.8147      0.09605     1082414643    4
nprobe=256,quantizer_efSearch=512        0.1859 0.4856 0.8151      0.12797     1082109029    3
nprobe=1024,quantizer_efSearch=256       0.1860 0.4872 0.8247      0.37395     4156489071    1
nprobe=2048,quantizer_efSearch=512       0.1861 0.4872 0.8254      0.72442     8086075604    1
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ4` </summary>
Index size 2440039065

 code_size 16

 log filename: autotune.dbdeep100M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1776 0.4516 0.6934      0.03973      288937530    8
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.0587 0.1128 0.1288      0.00253        5238141    119
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.0868 0.1700 0.1933      0.00262        7235236    115
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.0983 0.1935 0.2212      0.00254        5879472    118
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0987 0.1925 0.2193      0.00273        7232367    111
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1128 0.2419 0.2889      0.00285       10399009    106
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1143 0.2490 0.2984      0.00289        9724752    104
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4    0.1175 0.2557 0.3070      0.00299        9670729    101
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1239 0.2660 0.3194      0.00307       10333268    98
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1352 0.3111 0.4009      0.00412       18700005    73
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1398 0.3169 0.4023      0.00432       20690851    70
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.1402 0.3170 0.4027      0.00468       23301539    65
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=8    0.1433 0.3313 0.4243      0.00508       19217947    60
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.1460 0.3363 0.4317      0.00557       23162457    54
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.1467 0.3352 0.4307      0.00580       28435838    52
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1476 0.3557 0.4867      0.00640       36623752    47
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1510 0.3668 0.5064      0.00678       36406668    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1565 0.3771 0.5170      0.00684       38508520    44
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.1619 0.3905 0.5370      0.00731       40961364    42
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32   0.1623 0.3948 0.5430      0.00953       40791192    32
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1638 0.4055 0.5818      0.01160       74247148    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1646 0.4066 0.5836      0.01157       76743803    26
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1690 0.4203 0.6198      0.01147       72445679    27
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1729 0.4278 0.6242      0.01196       76445485    26
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.1735 0.4334 0.6368      0.01410       91801233    22
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32  0.1738 0.4350 0.6402      0.01917       75937001    16
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1743 0.4412 0.6834      0.02072      143036650    15
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1774 0.4495 0.6793      0.02072      147125044    15
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.1800 0.4592 0.7096      0.02186      151550702    14
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.1802 0.4614 0.7149      0.02208      145932279    14
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.1809 0.4627 0.7185      0.02504      150850445    12
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1822 0.4701 0.7517      0.03886      282781923    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1830 0.4746 0.7601      0.03963      284832705    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.1831 0.4756 0.7619      0.03978      289845417    8
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.1842 0.4779 0.7678      0.04098      289003794    8
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.1843 0.4777 0.7680      0.04501      299082933    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.1859 0.4843 0.7964      0.07445      562401766    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.1861 0.4861 0.8078      0.13881     1134497419    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.1869 0.4875 0.8124      0.14371     1096025654    3
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ4` </summary>
Index size 2426800981

 code_size 16

 log filename: autotune.dbdeep100M.PCAR32_IVF65536_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1081 0.2302 0.2847      0.00484       17705512    62
nprobe=1,quantizer_efSearch=16           0.1072 0.2290 0.2835      0.00327       17719446    92
nprobe=1,quantizer_efSearch=32           0.1079 0.2299 0.2844      0.00368       17703698    82
nprobe=2,quantizer_efSearch=4            0.1221 0.2778 0.3704      0.00416       35474082    73
nprobe=2,quantizer_efSearch=16           0.1346 0.3028 0.4033      0.00462       35431237    65
nprobe=2,quantizer_efSearch=32           0.1355 0.3042 0.4047      0.00515       35408986    59
nprobe=2,quantizer_efSearch=64           0.1356 0.3044 0.4050      0.00611       35412345    50
nprobe=4,quantizer_efSearch=4            0.1370 0.3304 0.4680      0.00694       70712873    44
nprobe=4,quantizer_efSearch=8            0.1520 0.3622 0.5137      0.00721       70815785    42
nprobe=4,quantizer_efSearch=16           0.1550 0.3687 0.5223      0.00725       70662798    42
nprobe=4,quantizer_efSearch=32           0.1562 0.3702 0.5242      0.00771       70612944    39
nprobe=4,quantizer_efSearch=64           0.1564 0.3703 0.5245      0.00911       70609618    33
nprobe=8,quantizer_efSearch=8            0.1650 0.4071 0.6110      0.01195      141010437    26
nprobe=8,quantizer_efSearch=16           0.1691 0.4166 0.6253      0.01227      140746237    25
nprobe=8,quantizer_efSearch=32           0.1705 0.4190 0.6280      0.01262      140626453    24
nprobe=8,quantizer_efSearch=128          0.1706 0.4190 0.6281      0.01594      140606702    19
nprobe=16,quantizer_efSearch=8           0.1742 0.4421 0.6936      0.02288      279841865    14
nprobe=16,quantizer_efSearch=16          0.1767 0.4492 0.7039      0.02188      279452931    14
nprobe=16,quantizer_efSearch=32          0.1780 0.4517 0.7076      0.02273      279140119    14
nprobe=16,quantizer_efSearch=64          0.1783 0.4518 0.7081      0.02479      279044123    13
nprobe=16,quantizer_efSearch=128         0.1784 0.4519 0.7082      0.02614      279022435    12
nprobe=32,quantizer_efSearch=32          0.1825 0.4676 0.7628      0.05215      552702138    6
nprobe=32,quantizer_efSearch=64          0.1830 0.4681 0.7640      0.04259      552333647    8
nprobe=32,quantizer_efSearch=128         0.1832 0.4684 0.7645      0.04569      552249577    7
nprobe=32,quantizer_efSearch=256         0.1833 0.4685 0.7647      0.05182      552225791    6
nprobe=64,quantizer_efSearch=32          0.1847 0.4757 0.7899      0.08275     1091853764    4
nprobe=64,quantizer_efSearch=64          0.1856 0.4770 0.7932      0.08346     1090382963    4
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF1048576_HNSW32,PQ48x4fs` </summary>
Index size 4299172741

 code_size 24

 log filename: autotune.dbdeep100M.IVF1048576_HNSW32_PQ48x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2091 0.3812 0.4161      0.00440        5184892    69
nprobe=2,quantizer_efSearch=4            0.1477 0.2498 0.2657      0.00267        2595000    113
nprobe=4,quantizer_efSearch=4            0.1731 0.3176 0.3496      0.00290        5189894    104
nprobe=8,quantizer_efSearch=4            0.2338 0.4500 0.5055      0.00416       10377253    73
nprobe=8,quantizer_efSearch=8            0.2419 0.4624 0.5196      0.00448       10355958    67
nprobe=16,quantizer_efSearch=4           0.2582 0.5166 0.5969      0.00490       20660635    62
nprobe=16,quantizer_efSearch=8           0.2759 0.5500 0.6371      0.00642       20603397    47
nprobe=32,quantizer_efSearch=8           0.2861 0.5976 0.7117      0.00726       40971810    42
nprobe=32,quantizer_efSearch=16          0.2996 0.6274 0.7480      0.00977       40781242    31
nprobe=32,quantizer_efSearch=32          0.3074 0.6413 0.7632      0.01272       40569734    24
nprobe=64,quantizer_efSearch=16          0.3105 0.6639 0.8094      0.01206       80988102    25
nprobe=64,quantizer_efSearch=32          0.3181 0.6835 0.8341      0.01625       80531387    19
nprobe=128,quantizer_efSearch=32         0.3240 0.7067 0.8773      0.02077      159657573    15
nprobe=128,quantizer_efSearch=64         0.3295 0.7195 0.8955      0.02695      158708643    12
nprobe=256,quantizer_efSearch=64         0.3330 0.7328 0.9239      0.03601      313612578    9
nprobe=256,quantizer_efSearch=128        0.3343 0.7403 0.9348      0.04699      311924003    7
nprobe=512,quantizer_efSearch=128        0.3353 0.7449 0.9489      0.06503      613878933    5
nprobe=256,quantizer_efSearch=256        0.3355 0.7414 0.9366      0.06739      310683649    5
nprobe=2048,quantizer_efSearch=256       0.3356 0.7525 0.9638      0.16773     2288945404    2
nprobe=4096,quantizer_efSearch=256       0.3360 0.7528 0.9646      0.21630     3786020155    2
nprobe=4096,quantizer_efSearch=512       0.3366 0.7548 0.9677      0.33546     4365710578    1
```

</details>
<details><summary>`IVF1048576(IVF1024,PQ48x4fs,RFlat),PQ48x4fs` </summary>
Index size 4047940297

 code_size 24

 log filename: autotune.dbdeep100M.IVF1048576_IVF1024_PQ48x4fs_RFlat__PQ48x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2163 0.3962 0.4385      0.00282       11805499    107
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1101 0.1706 0.1780      0.00151        2051313    200
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1154 0.1816 0.1895      0.00155        2047343    194
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1183 0.1854 0.1932      0.00162        2041150    186
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1267 0.2117 0.2260      0.00168        2961050    179
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1334 0.2151 0.2276      0.00172        3349124    175
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1495 0.2540 0.2696      0.00178        3335576    169
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1700 0.2856 0.3027      0.00209        4042592    144
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1845 0.3329 0.3633      0.00218        5923182    138
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1987 0.3553 0.3863      0.00231        6638026    130
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2064 0.3704 0.4038      0.00243        6623671    124
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2163 0.3962 0.4385      0.00285       11804998    106
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2338 0.4382 0.4904      0.00295       11773540    102
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2406 0.4524 0.5105      0.00352       11747039    86
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2544 0.4761 0.5345      0.00354       13140118    85
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2591 0.4977 0.5662      0.00422       23439434    72
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2637 0.5079 0.5873      0.00402       21998258    75
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2684 0.5213 0.6057      0.00500       21945370    60
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2866 0.5559 0.6407      0.00521       26065462    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2888 0.5650 0.6546      0.00564       26022558    54
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2969 0.6032 0.7150      0.00644       43552644    47
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3077 0.6366 0.7573      0.00811       46106866    37
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3083 0.6376 0.7587      0.00938       46083492    32
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3088 0.6453 0.7757      0.00982       86543481    31
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3110 0.6445 0.7657      0.00934       51408871    33
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3117 0.6449 0.7671      0.01117       51371536    27
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3191 0.6746 0.8220      0.01056       85993927    29
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3194 0.6791 0.8293      0.01202       85803279    25
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3229 0.6851 0.8344      0.01192       91173830    26
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3232 0.6896 0.8413      0.01380       90969377    22
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3244 0.6935 0.8454      0.01946      101285511    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3318 0.7195 0.8949      0.02180      168909240    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3326 0.7246 0.9010      0.02437      179205969    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3332 0.7326 0.9236      0.02815      322925277    11
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3340 0.7334 0.9269      0.03430      322224432    9
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3345 0.7410 0.9416      0.04871      659014410    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.3357 0.7405 0.9361      0.04698      393635148    7
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3364 0.7495 0.9545      0.06454      650796143    5
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3366 0.7549 0.9669      0.15090     2363045157    3
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3369 0.7552 0.9679      0.24505     4543644436    2
```

</details>
<details><summary>`IVF262144_HNSW32,PQ48x4fs` </summary>
Index size 3473696133

 code_size 24

 log filename: autotune.dbdeep100M.IVF262144_HNSW32_PQ48x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2462 0.4618 0.5205      0.00287       17982828    105
nprobe=2,quantizer_efSearch=4            0.1842 0.3241 0.3544      0.00180        8992798    167
nprobe=4,quantizer_efSearch=4            0.2124 0.4036 0.4567      0.00223       17986647    135
nprobe=4,quantizer_efSearch=8            0.2462 0.4618 0.5205      0.00299       17982828    101
nprobe=8,quantizer_efSearch=4            0.2673 0.5320 0.6172      0.00356       35933098    85
nprobe=8,quantizer_efSearch=8            0.2736 0.5440 0.6306      0.00371       35893172    81
nprobe=8,quantizer_efSearch=16           0.2848 0.5670 0.6546      0.00484       35789692    63
nprobe=16,quantizer_efSearch=4           0.2851 0.5880 0.6994      0.00534       71770370    57
nprobe=16,quantizer_efSearch=8           0.2975 0.6198 0.7379      0.00580       71627888    52
nprobe=16,quantizer_efSearch=16          0.3033 0.6339 0.7542      0.00648       71447632    47
nprobe=32,quantizer_efSearch=8           0.3078 0.6565 0.8002      0.00792      142714031    38
nprobe=32,quantizer_efSearch=32          0.3212 0.6908 0.8408      0.01096      141974812    28
nprobe=64,quantizer_efSearch=16          0.3217 0.7024 0.8719      0.01255      283313316    24
nprobe=32,quantizer_efSearch=64          0.3223 0.6925 0.8436      0.01393      141733609    22
nprobe=64,quantizer_efSearch=32          0.3280 0.7176 0.8918      0.01517      282463632    20
nprobe=128,quantizer_efSearch=32         0.3336 0.7349 0.9228      0.01924      560389527    16
nprobe=128,quantizer_efSearch=64         0.3361 0.7435 0.9346      0.02339      558815244    13
nprobe=128,quantizer_efSearch=128        0.3365 0.7453 0.9367      0.02937      557738467    11
nprobe=256,quantizer_efSearch=64         0.3371 0.7514 0.9491      0.03314     1103561975    10
nprobe=256,quantizer_efSearch=128        0.3387 0.7533 0.9537      0.04048     1101013193    8
nprobe=256,quantizer_efSearch=256        0.3396 0.7541 0.9549      0.05386     1099247660    6
nprobe=1024,quantizer_efSearch=256       0.3401 0.7592 0.9651      0.10043     4214185283    3
nprobe=2048,quantizer_efSearch=256       0.3408 0.7594 0.9657      0.13685     7950927973    3
```

</details>
<details><summary>`IVF262144(IVF512,PQ48x4fs,RFlat),PQ48x4fs` </summary>
Index size 3411096521

 code_size 24

 log filename: autotune.dbdeep100M.IVF262144_IVF512_PQ48x4fs_RFlat__PQ48x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3276 0.7028 0.8650      0.01335      291468403    23
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1195 0.1975 0.2120      0.00136        4692685    221
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.1337 0.2175 0.2311      0.00155        5934141    194
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1572 0.2580 0.2749      0.00158        5914245    190
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1902 0.3327 0.3633      0.00160        9763121    188
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2021 0.3535 0.3870      0.00189        9716052    159
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2085 0.3631 0.3962      0.00252       10409713    119
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2100 0.3877 0.4335      0.00227       18909795    132
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2135 0.4007 0.4554      0.00235       18446867    128
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2143 0.4032 0.4582      0.00237       18427803    127
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2286 0.4288 0.4832      0.00224       18803156    135
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2346 0.4432 0.5010      0.00236       18765475    128
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2353 0.4458 0.5043      0.00246       18738170    123
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2440 0.4571 0.5123      0.00291       20760732    104
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2515 0.4721 0.5315      0.00300       20720008    100
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2581 0.5040 0.5735      0.00331       37638436    91
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2609 0.5234 0.6078      0.00347       36730712    87
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.2612 0.5115 0.5823      0.00370       38891881    82
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2833 0.5629 0.6515      0.00409       38569986    74
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2854 0.5794 0.6826      0.00483       73855329    63
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2962 0.6145 0.7300      0.00495       73273748    61
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3021 0.6271 0.7445      0.00566       74400347    53
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3040 0.6328 0.7503      0.00661       76930364    46
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3106 0.6564 0.8010      0.00698      144526212    43
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3118 0.6609 0.8071      0.00761      144161163    40
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3233 0.6855 0.8352      0.00925      147735908    33
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3267 0.7074 0.8790      0.01189      286358307    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3312 0.7190 0.8921      0.01361      288282497    23
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3317 0.7264 0.9123      0.01802      564840742    17
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3326 0.7328 0.9152      0.02048      577292344    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3367 0.7430 0.9329      0.02241      570013162    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.3368 0.7453 0.9362      0.03168      599343339    10
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3384 0.7531 0.9530      0.03589     1122978752    9
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3390 0.7523 0.9536      0.04075     1111446756    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3391 0.7536 0.9549      0.04070     1120961322    8
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3393 0.7583 0.9628      0.05346     2184867327    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.3394 0.7585 0.9630      0.05832     2204655356    6
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.3395 0.7585 0.9627      0.08194     2181364594    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.3396 0.7588 0.9629      0.08803     2201140824    4
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3399 0.7599 0.9665      0.13925     8266810243    3
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.3400 0.7604 0.9669      0.13519     8278659146    3
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.3401 0.7603 0.9669      0.18331     8271701909    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.3402 0.7605 0.9674      0.20935    16338512378    2
```

</details>
<details><summary>`IVF65536_HNSW32,PQ48x4fs` </summary>
Index size 3268471429

 code_size 24

 log filename: autotune.dbdeep100M.IVF65536_HNSW32_PQ48x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3353 0.7538 0.9617      0.07284     4237522083    5
nprobe=1,quantizer_efSearch=4            0.1653 0.2904 0.3194      0.00195       17770977    154
nprobe=1,quantizer_efSearch=8            0.1778 0.3140 0.3453      0.00219       17755271    138
nprobe=1,quantizer_efSearch=16           0.1813 0.3204 0.3519      0.00255       17732062    118
nprobe=2,quantizer_efSearch=4            0.2023 0.3864 0.4380      0.00298       35484749    101
nprobe=2,quantizer_efSearch=8            0.2212 0.4230 0.4785      0.00321       35501598    94
nprobe=2,quantizer_efSearch=16           0.2252 0.4315 0.4875      0.00358       35451153    84
nprobe=2,quantizer_efSearch=32           0.2259 0.4324 0.4884      0.00432       35434915    70
nprobe=4,quantizer_efSearch=4            0.2355 0.4743 0.5545      0.00459       70984100    66
nprobe=4,quantizer_efSearch=8            0.2580 0.5224 0.6098      0.00492       71061544    61
nprobe=4,quantizer_efSearch=16           0.2641 0.5360 0.6244      0.00530       70898374    57
nprobe=4,quantizer_efSearch=32           0.2644 0.5369 0.6251      0.00601       70866177    50
nprobe=8,quantizer_efSearch=4            0.2794 0.5870 0.7059      0.00696      141923889    44
nprobe=8,quantizer_efSearch=8            0.2856 0.6005 0.7222      0.00698      141708322    43
nprobe=8,quantizer_efSearch=16           0.2946 0.6189 0.7430      0.00739      141374570    41
nprobe=8,quantizer_efSearch=32           0.2967 0.6222 0.7463      0.00815      141222909    37
nprobe=16,quantizer_efSearch=8           0.3070 0.6621 0.8151      0.00980      282177874    31
nprobe=16,quantizer_efSearch=16          0.3125 0.6761 0.8300      0.01007      281701555    30
nprobe=16,quantizer_efSearch=32          0.3156 0.6809 0.8361      0.01131      281303193    27
nprobe=32,quantizer_efSearch=16          0.3212 0.7087 0.8867      0.01256      558812200    24
nprobe=32,quantizer_efSearch=32          0.3236 0.7140 0.8937      0.01585      557737229    19
nprobe=64,quantizer_efSearch=32          0.3271 0.7331 0.9287      0.01740     1102789758    18
nprobe=64,quantizer_efSearch=64          0.3285 0.7363 0.9329      0.01876     1100892986    16
nprobe=128,quantizer_efSearch=32         0.3299 0.7421 0.9443      0.02444     2173524676    13
nprobe=128,quantizer_efSearch=64         0.3325 0.7464 0.9506      0.03006     2167514417    10
nprobe=128,quantizer_efSearch=128        0.3334 0.7484 0.9524      0.03364     2164196421    9
nprobe=128,quantizer_efSearch=256        0.3338 0.7489 0.9529      0.04197     2163355200    8
nprobe=256,quantizer_efSearch=128        0.3342 0.7528 0.9605      0.04749     4241494028    7
nprobe=256,quantizer_efSearch=256        0.3353 0.7538 0.9617      0.05661     4238569078    6
nprobe=512,quantizer_efSearch=256        0.3359 0.7559 0.9644      0.08097     8268119583    4
nprobe=1024,quantizer_efSearch=256       0.3360 0.7558 0.9654      0.12703    16000493739    3
nprobe=1024,quantizer_efSearch=512       0.3362 0.7559 0.9659      0.16321    16010317308    2
nprobe=2048,quantizer_efSearch=256       0.3363 0.7559 0.9654      0.18332    29167160051    2
nprobe=2048,quantizer_efSearch=512       0.3367 0.7562 0.9661      0.25553    30756869867    2
```

</details>
<details><summary>`OPQ32_128,IVF1048576_HNSW32,PQ32` </summary>
Index size 4830801840

 code_size 32

 log filename: autotune.dbdeep100M.OPQ32_128_IVF1048576_HNSW32_PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6653 0.9286 0.9331      0.44863      310130393    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0092 0.0129 0.0137      0.00372        1298253    81
nprobe=1,quantizer_efSearch=4,ht=106      0.1197 0.1377 0.1386      0.00375        1298253    80
nprobe=2,quantizer_efSearch=4,ht=102      0.1391 0.1620 0.1633      0.00493        2595615    61
nprobe=1,quantizer_efSearch=16,ht=114     0.1750 0.1995 0.2005      0.00762        1294459    40
nprobe=1,quantizer_efSearch=16,ht=116     0.1803 0.2058 0.2068      0.00756        1294459    40
nprobe=2,quantizer_efSearch=16,ht=110     0.2270 0.2622 0.2636      0.00906        2581382    34
nprobe=4,quantizer_efSearch=8,ht=118      0.3270 0.3954 0.3972      0.00945        5186134    32
nprobe=4,quantizer_efSearch=32,ht=256     0.3589 0.4371 0.4388      0.01338        5127482    23
nprobe=8,quantizer_efSearch=16,ht=128     0.4355 0.5468 0.5488      0.01778       10294285    17
nprobe=16,quantizer_efSearch=4,ht=122     0.4563 0.5881 0.5909      0.02571       20671976    12
nprobe=16,quantizer_efSearch=4,ht=126     0.4604 0.5952 0.5980      0.02573       20671976    12
nprobe=16,quantizer_efSearch=8,ht=118     0.4780 0.6099 0.6126      0.02617       20604985    12
nprobe=16,quantizer_efSearch=16,ht=124    0.5045 0.6500 0.6527      0.02914       20529118    11
nprobe=16,quantizer_efSearch=64,ht=256    0.5234 0.6746 0.6772      0.02774       20334938    11
nprobe=32,quantizer_efSearch=8,ht=124     0.5374 0.7111 0.7145      0.04910       40975771    7
nprobe=32,quantizer_efSearch=32,ht=116    0.5519 0.7167 0.7198      0.05336       40567698    6
nprobe=128,quantizer_efSearch=64,ht=256   0.6523 0.9097 0.9144      0.09283      158709615    4
nprobe=128,quantizer_efSearch=512,ht=256  0.6555 0.9167 0.9213      0.17825      157453289    2
nprobe=256,quantizer_efSearch=512,ht=122  0.6700 0.9400 0.9449      0.44435      310130393    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6712 0.9433 0.9482      0.72127      613854025    1
nprobe=512,quantizer_efSearch=64,ht=128   0.6737 0.9526 0.9582      0.72223      612955105    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6838 0.9721 0.9777      1.48040     1193364334    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.6881 0.9844 0.9906      1.51791     1193364334    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6903 0.9919 0.9981      2.36740     4365717045    1
```

</details>
<details><summary>`OPQ32_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ32` </summary>
Index size 4588441076

 code_size 32

 log filename: autotune.dbdeep100M.OPQ32_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.3325 0.4194 0.4222      0.68110      584029097    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=88       0.0461 0.0551 0.0562      0.00328        2751788    92
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=96      0.0660 0.0768 0.0779      0.00367        1666144    82
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=2,ht=126     0.1707 0.1948 0.1959      0.00392        2038862    77
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=114      0.2359 0.2728 0.2742      0.00475        4037810    64
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=128      0.2585 0.3003 0.3014      0.00544        5465173    56
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=256    0.2736 0.3210 0.3223      0.00611        8223012    50
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=116      0.3258 0.3848 0.3868      0.00766        6618675    40
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=116     0.3266 0.3867 0.3888      0.01021        6613166    30
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=128     0.3422 0.4102 0.4122      0.01069        6609955    29
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128    0.3638 0.4348 0.4368      0.01140       10761748    27
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=256    0.3649 0.4372 0.4391      0.01348       26200431    23
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=120      0.4055 0.4997 0.5020      0.01400       11744070    22
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=128    0.4457 0.5519 0.5542      0.01750       15870908    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=114     0.4704 0.5858 0.5884      0.02609       23299582    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=128    0.5179 0.6614 0.6640      0.02826       26019080    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.5222 0.6683 0.6709      0.03344       41624617    9
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=256   0.5807 0.7650 0.7684      0.04584       45968454    7
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=126   0.5841 0.7707 0.7742      0.07044      123660355    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=120    0.6119 0.8196 0.8231      0.09819       85768005    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=122    0.6215 0.8389 0.8426      0.09951       90766348    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=256   0.6491 0.8926 0.8968      0.09114      163975305    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=122  0.6760 0.9426 0.9476      0.36963      352733320    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=512,ht=256 0.6916 0.9767 0.9825      0.52992      773494316    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=512,ht=256 0.6980 0.9927 0.9995      2.34992     4651611940    1
```

</details>
<details><summary>`OPQ32_128,IVF262144_HNSW32,PQ32` </summary>
Index size 4207838384

 code_size 32

 log filename: autotune.dbdeep100M.OPQ32_128_IVF262144_HNSW32_PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6714 0.9634 0.9692      0.45583     1098751032    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0123 0.0167 0.0179      0.00319        4485531    94
nprobe=1,quantizer_efSearch=4,ht=106      0.1712 0.1987 0.2000      0.00321        4485531    94
nprobe=2,quantizer_efSearch=4,ht=102      0.1948 0.2296 0.2311      0.00444        8993764    68
nprobe=1,quantizer_efSearch=16,ht=114     0.2364 0.2752 0.2766      0.00594        4476249    51
nprobe=1,quantizer_efSearch=16,ht=116     0.2403 0.2809 0.2823      0.00598        4476249    51
nprobe=2,quantizer_efSearch=16,ht=110     0.3005 0.3546 0.3561      0.00734        8951011    41
nprobe=4,quantizer_efSearch=8,ht=118      0.4116 0.5116 0.5135      0.00907       17986991    34
nprobe=4,quantizer_efSearch=32,ht=256     0.4345 0.5439 0.5459      0.01137       17896902    27
nprobe=8,quantizer_efSearch=16,ht=114     0.4864 0.6213 0.6241      0.01641       35790206    19
nprobe=8,quantizer_efSearch=16,ht=128     0.5059 0.6603 0.6632      0.01846       35790206    17
nprobe=16,quantizer_efSearch=4,ht=114     0.5119 0.6669 0.6702      0.02603       71787799    12
nprobe=16,quantizer_efSearch=64,ht=256    0.5705 0.7727 0.7765      0.02624       71193813    12
nprobe=32,quantizer_efSearch=8,ht=124     0.5891 0.8116 0.8156      0.05484      142712997    6
nprobe=32,quantizer_efSearch=32,ht=116    0.6024 0.8200 0.8238      0.05450      141972746    6
nprobe=64,quantizer_efSearch=64,ht=126    0.6457 0.9127 0.9178      0.11592      281777378    3
nprobe=64,quantizer_efSearch=128,ht=126   0.6471 0.9145 0.9196      0.12699      281452389    3
nprobe=128,quantizer_efSearch=64,ht=256   0.6654 0.9547 0.9604      0.11994      558813692    3
nprobe=128,quantizer_efSearch=512,ht=256  0.6669 0.9571 0.9629      0.17696      557257684    2
nprobe=256,quantizer_efSearch=512,ht=120  0.6714 0.9634 0.9692      0.45253     1098751032    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6731 0.9707 0.9765      0.45966     1098751032    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6745 0.9702 0.9763      0.77158     2163985153    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6794 0.9835 0.9899      1.60005     4218690267    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.6812 0.9883 0.9956      1.70475     4218690267    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6826 0.9914 0.9991      3.16160    14975241528    1
```

</details>
<details><summary>`OPQ32_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ32` </summary>
Index size 4147512820

 code_size 32

 log filename: autotune.dbdeep100M.OPQ32_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.6486 0.9114 0.9163      0.11844      287393116    3
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=72        0.0230 0.0282 0.0293      0.00263        4690043    114
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=108       0.1858 0.2163 0.2172      0.00276        4864458    109
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=120       0.2201 0.2658 0.2666      0.00447        9278185    68
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=256      0.2452 0.2998 0.3010      0.00467        9200559    65
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=128       0.3261 0.3939 0.3947      0.00550       10427783    55
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=122      0.3356 0.4058 0.4069      0.00642       11726656    47
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2,ht=120       0.3362 0.4135 0.4144      0.00813       18536244    37
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=116      0.4101 0.5067 0.5080      0.00894       20730814    34
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=256       0.4208 0.5434 0.5450      0.00956       36525634    32
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=256    0.4326 0.5443 0.5460      0.01366       38763386    22
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=118       0.4361 0.5567 0.5582      0.01448       37000783    21
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=124       0.4983 0.6481 0.6504      0.01646       37283963    19
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=126      0.5029 0.6501 0.6521      0.01676       38613722    18
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=128     0.5112 0.6655 0.6679      0.02809       46358006    11
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=126     0.5455 0.7236 0.7259      0.03216       77356329    10
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=128    0.5682 0.7650 0.7679      0.03435       74141412    9
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=116      0.5836 0.7905 0.7937      0.05324      144079419    6
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=120     0.6157 0.8448 0.8483      0.05676      147314117    6
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=256     0.6206 0.8694 0.8741      0.10086      285387588    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=116     0.6269 0.8630 0.8671      0.10287      285564652    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=116    0.6344 0.8755 0.8798      0.11191      287393116    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=126    0.6486 0.9114 0.9163      0.11995      287393116    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128    0.6501 0.9147 0.9198      0.12537      292346443    3
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=256  0.6703 0.9563 0.9621      0.19977      578760562    2
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=128  0.6705 0.9556 0.9612      0.24729      599117780    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=256   0.6767 0.9715 0.9773      0.27872     1108488522    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=118  0.6799 0.9676 0.9726      1.48103     4250970609    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=120   0.6803 0.9745 0.9793      1.51263     4247371256    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=126   0.6851 0.9872 0.9930      1.63023     4247371256    1
nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=256  0.6862 0.9903 0.9964      1.98739     8277392107    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=256 0.6877 0.9935 0.9998      4.92616    15984602239    1
```

</details>
<details><summary>`OPQ32_128,IVF65536_HNSW32,PQ32` </summary>
Index size 4052096688

 code_size 32

 log filename: autotune.dbdeep100M.OPQ32_128_IVF65536_HNSW32_PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6634 0.9795 0.9875      0.49262     4237476725    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0152 0.0191 0.0206      0.00276       17791525    109
nprobe=1,quantizer_efSearch=4,ht=106      0.2231 0.2683 0.2701      0.00318       17791525    95
nprobe=1,quantizer_efSearch=16,ht=114     0.2751 0.3388 0.3409      0.00438       17732649    69
nprobe=1,quantizer_efSearch=16,ht=116     0.2774 0.3437 0.3457      0.00433       17732649    70
nprobe=2,quantizer_efSearch=16,ht=110     0.3563 0.4465 0.4493      0.00574       35451727    53
nprobe=2,quantizer_efSearch=64,ht=120     0.3799 0.4891 0.4917      0.00931       35427457    33
nprobe=2,quantizer_efSearch=64,ht=124     0.3816 0.4913 0.4941      0.00994       35427457    31
nprobe=4,quantizer_efSearch=8,ht=118      0.4562 0.6076 0.6108      0.01027       71055760    30
nprobe=4,quantizer_efSearch=32,ht=116     0.4640 0.6163 0.6194      0.01139       70865680    27
nprobe=4,quantizer_efSearch=32,ht=256     0.4730 0.6312 0.6345      0.01100       70865680    28
nprobe=8,quantizer_efSearch=32,ht=106     0.4756 0.6148 0.6182      0.01596      141223912    19
nprobe=8,quantizer_efSearch=16,ht=114     0.5333 0.7240 0.7279      0.01720      141365984    18
nprobe=8,quantizer_efSearch=32,ht=116     0.5400 0.7381 0.7420      0.01829      141223912    17
nprobe=8,quantizer_efSearch=64,ht=116     0.5414 0.7396 0.7435      0.02003      141180852    15
nprobe=8,quantizer_efSearch=16,ht=128     0.5476 0.7532 0.7574      0.02357      141365984    13
nprobe=8,quantizer_efSearch=64,ht=126     0.5510 0.7573 0.7615      0.02497      141180852    13
nprobe=16,quantizer_efSearch=64,ht=110    0.5652 0.7702 0.7744      0.03195      281108956    10
nprobe=16,quantizer_efSearch=64,ht=256    0.6013 0.8515 0.8571      0.03304      281108956    10
nprobe=32,quantizer_efSearch=32,ht=116    0.6215 0.8902 0.8959      0.06055      557725151    5
nprobe=32,quantizer_efSearch=512,ht=128   0.6333 0.9147 0.9214      0.11639      556983178    3
nprobe=64,quantizer_efSearch=256,ht=114   0.6372 0.9166 0.9229      0.12070     1100021910    3
nprobe=64,quantizer_efSearch=128,ht=126   0.6535 0.9541 0.9620      0.15704     1100167599    2
nprobe=128,quantizer_efSearch=64,ht=256   0.6615 0.9740 0.9823      0.22228     2167520368    2
nprobe=128,quantizer_efSearch=512,ht=256  0.6633 0.9765 0.9847      0.24638     2163219889    2
nprobe=256,quantizer_efSearch=512,ht=120  0.6634 0.9795 0.9875      0.48028     4237476725    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6652 0.9831 0.9915      0.51054     4237476725    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6677 0.9880 0.9966      1.82084    16010068027    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.6687 0.9902 0.9989      2.24076    16010068027    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6693 0.9912 0.9998      5.40768    54068519903    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64x4fs` </summary>
Index size 5364981708

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF1048576_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3843 0.8052 0.9494      0.14750      310128756    3
nprobe=2,quantizer_efSearch=4            0.1631 0.2587 0.2685      0.00296        2595714    102
nprobe=4,quantizer_efSearch=4            0.1968 0.3335 0.3529      0.00324        5191162    93
nprobe=8,quantizer_efSearch=4            0.2633 0.4707 0.5058      0.00476       10382812    64
nprobe=8,quantizer_efSearch=8            0.2702 0.4844 0.5194      0.00550       10361727    55
nprobe=16,quantizer_efSearch=4           0.2915 0.5452 0.5980      0.00564       20667908    54
nprobe=16,quantizer_efSearch=8           0.3103 0.5825 0.6387      0.00711       20607436    43
nprobe=32,quantizer_efSearch=8           0.3288 0.6381 0.7161      0.00865       40980944    35
nprobe=32,quantizer_efSearch=16          0.3434 0.6707 0.7517      0.01171       40785109    26
nprobe=64,quantizer_efSearch=16          0.3559 0.7134 0.8142      0.01381       80990270    22
nprobe=64,quantizer_efSearch=32          0.3652 0.7356 0.8409      0.01886       80534099    16
nprobe=128,quantizer_efSearch=32         0.3705 0.7647 0.8874      0.02330      159658025    13
nprobe=128,quantizer_efSearch=64         0.3766 0.7793 0.9054      0.03249      158709965    10
nprobe=128,quantizer_efSearch=128        0.3794 0.7845 0.9112      0.04635      157982732    7
nprobe=256,quantizer_efSearch=64         0.3809 0.7954 0.9370      0.04524      313593422    7
nprobe=512,quantizer_efSearch=64         0.3830 0.8022 0.9490      0.05737      612925160    6
nprobe=512,quantizer_efSearch=128        0.3850 0.8129 0.9635      0.07146      613860013    5
nprobe=1024,quantizer_efSearch=128       0.3860 0.8163 0.9703      0.09351     1189528626    4
nprobe=2048,quantizer_efSearch=128       0.3868 0.8171 0.9719      0.12803     2088808407    3
nprobe=1024,quantizer_efSearch=256       0.3884 0.8218 0.9784      0.13242     1197063648    3
nprobe=2048,quantizer_efSearch=256       0.3886 0.8228 0.9805      0.18252     2288916181    2
nprobe=1024,quantizer_efSearch=512       0.3887 0.8237 0.9804      0.18357     1193359383    2
nprobe=4096,quantizer_efSearch=256       0.3891 0.8236 0.9813      0.23260     3785998122    2
nprobe=4096,quantizer_efSearch=512       0.3894 0.8260 0.9844      0.34028     4365692583    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64x4fsr` </summary>
Index size 5364911052

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF1048576_HNSW32_PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4932 0.6672 0.6726      0.02525       20408155    12
nprobe=1,quantizer_efSearch=4            0.1565 0.1853 0.1867      0.00325        1300650    93
nprobe=2,quantizer_efSearch=4            0.2192 0.2659 0.2676      0.00384        2601052    79
nprobe=4,quantizer_efSearch=4            0.2795 0.3505 0.3533      0.00502        5198453    60
nprobe=4,quantizer_efSearch=8            0.3317 0.4146 0.4173      0.00658        5186863    46
nprobe=8,quantizer_efSearch=4            0.3891 0.5055 0.5093      0.00826       10380970    37
nprobe=8,quantizer_efSearch=8            0.3991 0.5186 0.5223      0.00891       10360698    34
nprobe=8,quantizer_efSearch=16           0.4214 0.5470 0.5509      0.01147       10293830    27
nprobe=8,quantizer_efSearch=32           0.4266 0.5549 0.5587      0.01607       10239337    19
nprobe=16,quantizer_efSearch=4           0.4469 0.5969 0.6022      0.01648       20668574    19
nprobe=16,quantizer_efSearch=8           0.4733 0.6366 0.6422      0.01739       20602536    18
nprobe=16,quantizer_efSearch=16          0.4827 0.6536 0.6590      0.01835       20528269    17
nprobe=16,quantizer_efSearch=32          0.4932 0.6672 0.6726      0.02436       20408155    13
nprobe=16,quantizer_efSearch=64          0.4966 0.6715 0.6769      0.03187       20334544    10
nprobe=32,quantizer_efSearch=8           0.5160 0.7129 0.7205      0.03272       40978427    10
nprobe=32,quantizer_efSearch=16          0.5383 0.7480 0.7557      0.03599       40782218    9
nprobe=32,quantizer_efSearch=32          0.5480 0.7638 0.7713      0.03871       40570914    8
nprobe=32,quantizer_efSearch=64          0.5535 0.7705 0.7781      0.04330       40394150    7
nprobe=32,quantizer_efSearch=128         0.5542 0.7725 0.7801      0.06153       40297623    5
nprobe=64,quantizer_efSearch=16          0.5648 0.8116 0.8209      0.07304       80987773    5
nprobe=64,quantizer_efSearch=32          0.5786 0.8375 0.8471      0.07585       80535843    4
nprobe=64,quantizer_efSearch=64          0.5850 0.8476 0.8573      0.08126       80111921    4
nprobe=64,quantizer_efSearch=128         0.5877 0.8505 0.8602      0.08992       79848300    4
nprobe=128,quantizer_efSearch=32         0.6016 0.8843 0.8956      0.13641      159660311    3
nprobe=128,quantizer_efSearch=64         0.6116 0.9024 0.9142      0.14212      158707424    3
nprobe=128,quantizer_efSearch=128        0.6142 0.9086 0.9203      0.15120      157982434    2
nprobe=128,quantizer_efSearch=256        0.6149 0.9097 0.9213      0.17140      157596489    2
nprobe=256,quantizer_efSearch=64         0.6250 0.9335 0.9467      0.27895      313608377    2
nprobe=256,quantizer_efSearch=128        0.6301 0.9448 0.9583      0.28842      311924382    2
nprobe=256,quantizer_efSearch=256        0.6304 0.9473 0.9602      0.30138      310678919    1
nprobe=256,quantizer_efSearch=512        0.6315 0.9473 0.9601      0.35306      310128487    1
nprobe=512,quantizer_efSearch=256        0.6335 0.9670 0.9812      0.56451      611183386    1
nprobe=512,quantizer_efSearch=512        0.6348 0.9693 0.9831      0.57493      609130612    1
nprobe=1024,quantizer_efSearch=256       0.6354 0.9758 0.9906      1.03595     1197041979    1
nprobe=1024,quantizer_efSearch=512       0.6383 0.9770 0.9928      1.07491     1193358920    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 5122558480

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.3829 0.7951 0.9300      0.03819      355859368    8
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.1318 0.1898 0.1952      0.00206        2045132    146
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1382 0.2188 0.2260      0.00205        2970516    147
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.1429 0.2243 0.2318      0.00212        2961033    142
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1       0.1440 0.2264 0.2340      0.00226        2955767    133
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.1667 0.2574 0.2647      0.00237        4062796    127
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1668 0.2805 0.2949      0.00245        5533712    123
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1882 0.3099 0.3242      0.00254        5953238    119
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2010 0.3384 0.3556      0.00260        5931353    116
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2087 0.3485 0.3673      0.00275        5921539    110
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2256 0.3781 0.3965      0.00288        6640150    105
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2329 0.3897 0.4098      0.00299        6623249    101
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.2336 0.3928 0.4112      0.00333        8038968    91
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.2495 0.4364 0.4621      0.00342       11817533    88
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2631 0.4678 0.5020      0.00354       11774411    85
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.2680 0.4791 0.5145      0.00434       11746594    70
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.2822 0.5041 0.5395      0.00443       13143581    68
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.2842 0.5030 0.5379      0.00500       15908147    60
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2973 0.5467 0.5913      0.00511       23445744    59
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3224 0.5976 0.6517      0.00645       26060222    47
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3236 0.6061 0.6626      0.00691       26014095    44
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3247 0.6073 0.6652      0.00787       25991885    39
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3339 0.6540 0.7299      0.00781       43521934    39
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3493 0.6815 0.7623      0.00982       46082606    31
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3538 0.6889 0.7701      0.01174       51328407    26
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3545 0.7060 0.7987      0.01185       86618778    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3624 0.7284 0.8311      0.01288       85935669    24
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3627 0.7317 0.8361      0.01479       85759247    21
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.3689 0.7391 0.8432      0.01500       91016838    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3700 0.7435 0.8489      0.01676       90795175    18
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3799 0.7809 0.9064      0.02608      168736598    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3811 0.7859 0.9121      0.02918      178561050    11
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3852 0.7981 0.9398      0.03482      322696842    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.3867 0.8041 0.9474      0.03553      331895080    9
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.3869 0.8139 0.9626      0.05693      659276317    6
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3875 0.8223 0.9765      0.09019     1250807923    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.3881 0.8245 0.9807      0.10677     1273373485    3
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.3888 0.8265 0.9832      0.14770     2435223264    3
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=512  0.3889 0.8264 0.9833      0.16996     2519464224    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3893 0.8243 0.9817      0.21885     4621528296    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.3894 0.8260 0.9845      0.24068     2397499235    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512  0.3895 0.8260 0.9846      0.25285     2478287903    2
nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=512 0.3896 0.8272 0.9859      1.89559     4649152415    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 5122445840

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3917 0.5099 0.5130      0.00542       13159308    56
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1669 0.1953 0.1965      0.00202        2039432    149
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2485 0.3016 0.3030      0.00240        4038984    126
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2526 0.3056 0.3070      0.00250        4036025    121
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2713 0.3409 0.3427      0.00281        5920188    108
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2909 0.3655 0.3680      0.00294        5913511    103
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2929 0.3673 0.3701      0.00298        5907908    101
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3034 0.3811 0.3830      0.00306        6624207    99
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3229 0.4063 0.4088      0.00316        6617092    95
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3252 0.4094 0.4123      0.00331        6611632    91
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3254 0.4098 0.4127      0.00364        6609675    83
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3438 0.4471 0.4509      0.00442       11045893    68
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3462 0.4507 0.4544      0.00472       11036869    64
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3724 0.4843 0.4873      0.00456       11779153    66
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3896 0.5070 0.5110      0.00473       11750159    64
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3917 0.5099 0.5130      0.00515       13165069    59
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4087 0.5344 0.5386      0.00539       13128169    56
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.4123 0.5391 0.5435      0.00689       13095029    44
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4185 0.5460 0.5502      0.00669       15794974    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4212 0.5502 0.5543      0.00716       15766198    42
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4214 0.5510 0.5553      0.00737       15783040    41
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4223 0.5525 0.5566      0.00900       20165388    34
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4227 0.5536 0.5579      0.00989       20434298    31
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4228 0.5536 0.5580      0.01048       20418038    29
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4236 0.5545 0.5588      0.01259       30255092    24
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4459 0.6030 0.6082      0.01386       21947586    22
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.4463 0.6054 0.6109      0.01385       21935204    22
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4549 0.6145 0.6192      0.01431       23359433    21
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4733 0.6413 0.6468      0.01414       23298641    22
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4750 0.6448 0.6505      0.01451       23276072    21
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.4754 0.6452 0.6510      0.01519       23261152    20
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4865 0.6589 0.6646      0.01550       25902150    20
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.4919 0.6682 0.6740      0.02278       30980801    14
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4921 0.6661 0.6717      0.02067       40632049    15
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4927 0.6683 0.6740      0.02068       40984725    15
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.4934 0.6688 0.6746      0.02239       41072559    14
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5090 0.7108 0.7172      0.02679       43594728    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5201 0.7308 0.7388      0.02680       43418283    12
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5216 0.7342 0.7418      0.02653       43406430    12
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5386 0.7599 0.7677      0.02839       45955925    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5449 0.7680 0.7757      0.03016       50726497    10
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5457 0.7685 0.7762      0.03166       51066742    10
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5468 0.7700 0.7777      0.03428       60687133    9
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5661 0.8148 0.8247      0.06497       86143914    5
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5734 0.8267 0.8367      0.06414       90885270    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5740 0.8329 0.8439      0.06446       85604970    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5817 0.8450 0.8557      0.06539      100911179    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.5826 0.8477 0.8588      0.06521      121596175    5
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.5832 0.8478 0.8589      0.07552      101100389    4
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5901 0.8704 0.8821      0.11327      164907703    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6062 0.9077 0.9201      0.11775      178143991    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.6071 0.9134 0.9277      0.21970      317990791    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6182 0.9372 0.9509      0.22315      333039125    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6241 0.9449 0.9593      0.23362      331026367    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6290 0.9597 0.9753      0.43170      635334040    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6296 0.9607 0.9765      0.43526      651292953    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6313 0.9741 0.9909      0.82865     1237730383    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.6336 0.9730 0.9905      0.85042     1214044275    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.6337 0.9757 0.9929      0.88803     1272527243    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fs` </summary>
Index size 4339742924

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF262144_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2681 0.4824 0.5231      0.00482       17987510    63
nprobe=2,quantizer_efSearch=4            0.1990 0.3342 0.3557      0.00234        8992714    129
nprobe=4,quantizer_efSearch=4            0.2344 0.4208 0.4580      0.00295       17978393    102
nprobe=4,quantizer_efSearch=8            0.2681 0.4824 0.5231      0.00380       17987510    79
nprobe=8,quantizer_efSearch=4            0.2974 0.5630 0.6218      0.00473       35934212    64
nprobe=8,quantizer_efSearch=8            0.3031 0.5746 0.6352      0.00503       35888718    60
nprobe=8,quantizer_efSearch=16           0.3157 0.5979 0.6602      0.00647       35790160    47
nprobe=16,quantizer_efSearch=4           0.3214 0.6226 0.7055      0.00674       71779414    45
nprobe=16,quantizer_efSearch=8           0.3378 0.6566 0.7450      0.00785       71621482    39
nprobe=16,quantizer_efSearch=16          0.3470 0.6721 0.7616      0.00869       71448972    35
nprobe=32,quantizer_efSearch=8           0.3493 0.7039 0.8085      0.01119      142723667    27
nprobe=16,quantizer_efSearch=32          0.3512 0.6799 0.7690      0.01082       71275788    28
nprobe=32,quantizer_efSearch=16          0.3628 0.7307 0.8388      0.01257      142347569    24
nprobe=32,quantizer_efSearch=32          0.3665 0.7410 0.8497      0.01470      141976092    21
nprobe=64,quantizer_efSearch=16          0.3701 0.7595 0.8816      0.01767      283329252    17
nprobe=64,quantizer_efSearch=32          0.3786 0.7763 0.9016      0.02026      282460683    15
nprobe=128,quantizer_efSearch=32         0.3834 0.7979 0.9348      0.02573      560383779    12
nprobe=128,quantizer_efSearch=64         0.3880 0.8072 0.9466      0.03267      558811272    10
nprobe=128,quantizer_efSearch=128        0.3890 0.8098 0.9488      0.03910      557735387    8
nprobe=128,quantizer_efSearch=256        0.3894 0.8099 0.9492      0.05664      557350687    6
nprobe=256,quantizer_efSearch=128        0.3909 0.8202 0.9681      0.05318     1101017765    6
nprobe=256,quantizer_efSearch=256        0.3916 0.8212 0.9693      0.06768     1099246687    5
nprobe=1024,quantizer_efSearch=256       0.3922 0.8257 0.9812      0.11983     4214216999    3
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fsr` </summary>
Index size 4339804364

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF262144_HNSW32_PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4781 0.6636 0.6696      0.02234       35675408    14
nprobe=1,quantizer_efSearch=4            0.2037 0.2553 0.2568      0.00204        4486733    147
nprobe=2,quantizer_efSearch=16           0.3169 0.4083 0.4116      0.00492        8949973    61
nprobe=4,quantizer_efSearch=16           0.4047 0.5391 0.5434      0.00560       17925794    54
nprobe=8,quantizer_efSearch=16           0.4753 0.6579 0.6638      0.00706       35787146    43
nprobe=8,quantizer_efSearch=32           0.4778 0.6625 0.6685      0.00980       35715655    31
nprobe=16,quantizer_efSearch=16          0.5275 0.7598 0.7681      0.01437       71448229    21
nprobe=16,quantizer_efSearch=64          0.5316 0.7675 0.7760      0.02062       71196311    15
nprobe=32,quantizer_efSearch=16          0.5643 0.8385 0.8487      0.02561      142335149    12
nprobe=32,quantizer_efSearch=32          0.5697 0.8489 0.8592      0.02689      141976910    12
nprobe=32,quantizer_efSearch=64          0.5709 0.8517 0.8620      0.02985      141737242    11
nprobe=32,quantizer_efSearch=128         0.5712 0.8514 0.8617      0.03706      141641233    9
nprobe=32,quantizer_efSearch=256         0.5715 0.8516 0.8619      0.05037      141621292    6
nprobe=64,quantizer_efSearch=16          0.5847 0.8815 0.8943      0.06024      283315438    5
nprobe=64,quantizer_efSearch=64          0.5977 0.9065 0.9198      0.06673      281783179    5
nprobe=64,quantizer_efSearch=128         0.5988 0.9081 0.9215      0.07438      281453150    5
nprobe=128,quantizer_efSearch=64         0.6094 0.9432 0.9595      0.12213      558813570    3
nprobe=128,quantizer_efSearch=128        0.6116 0.9453 0.9618      0.11349      557736215    3
nprobe=256,quantizer_efSearch=128        0.6160 0.9653 0.9818      0.21109     1101013984    2
nprobe=512,quantizer_efSearch=128        0.6177 0.9659 0.9819      0.39232     2164007335    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 4279622160

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1555 0.2462 0.2569      0.00200        4861697    150
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1296 0.2077 0.2176      0.00154        4691904    195
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1553 0.2459 0.2565      0.00177        4861572    170
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1650 0.2640 0.2749      0.00188        5206214    160
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1690 0.2695 0.2805      0.00196        5903692    153
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2097 0.3534 0.3737      0.00208        9740307    144
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2175 0.3660 0.3882      0.00220        9709425    137
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2306 0.3804 0.4027      0.00233       10394695    129
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2354 0.4133 0.4473      0.00260       18466440    116
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2381 0.4230 0.4551      0.00264       18852830    114
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2554 0.4568 0.4941      0.00272       18786718    111
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2584 0.4656 0.5057      0.00285       18735945    106
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2784 0.5169 0.5667      0.00389       37044823    78
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2956 0.5462 0.5978      0.00413       37603844    73
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3002 0.5536 0.6055      0.00459       38864198    66
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3147 0.5955 0.6567      0.00500       38557510    60
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3166 0.6118 0.6936      0.00623       72746338    49
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3238 0.6235 0.6996      0.00626       73757136    48
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3368 0.6525 0.7387      0.00655       73207111    46
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3481 0.6718 0.7606      0.00745       74170440    41
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3483 0.6726 0.7613      0.00815       74144119    37
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3520 0.7091 0.8128      0.00914      144346468    33
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3538 0.7119 0.8167      0.00984      144089789    31
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3661 0.7394 0.8465      0.01170      147597980    26
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3674 0.7428 0.8514      0.01446      152425752    21
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3693 0.7531 0.8697      0.01513      289549344    20
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3732 0.7667 0.8894      0.01370      286115772    22
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3774 0.7792 0.9028      0.01711      288008751    18
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3795 0.7810 0.9056      0.01844      287449766    17
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3803 0.7822 0.9077      0.02051      292404109    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3816 0.7904 0.9246      0.02093      564415468    15
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3829 0.7957 0.9298      0.02581      577231653    12
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3876 0.8075 0.9461      0.02722      569570578    12
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3879 0.8080 0.9469      0.03009      579633333    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3884 0.8078 0.9475      0.03012      568759981    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3891 0.8084 0.9483      0.03294      578817759    10
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3901 0.8143 0.9611      0.04042     1139824180    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3916 0.8198 0.9684      0.04268     1122259874    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3919 0.8206 0.9692      0.05051     1120804714    6
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.3922 0.8222 0.9737      0.06294     2242408950    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.3924 0.8243 0.9775      0.07244     2203034817    5
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.3926 0.8274 0.9834      0.17521     8275931453    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.3930 0.8274 0.9838      0.23307    16356076906    2
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 4279541264

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3084 0.3962 0.3995      0.00249       10390227    121
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1914 0.2371 0.2391      0.00148        4887716    203
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2189 0.2735 0.2760      0.00171        5206813    176
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2727 0.3496 0.3527      0.00200        9388463    150
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2831 0.3617 0.3644      0.00208        9756360    145
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2997 0.3858 0.3894      0.00215        9707488    140
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3003 0.3872 0.3907      0.00232        9710742    130
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3084 0.3962 0.3995      0.00238       10388926    126
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3775 0.5049 0.5094      0.00306       18732620    99
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3780 0.5013 0.5046      0.00324       19437518    93
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3963 0.5302 0.5347      0.00340       19380627    89
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3968 0.5305 0.5352      0.00364       19377657    83
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4025 0.5380 0.5425      0.00417       20689529    73
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4026 0.5381 0.5428      0.00424       20687808    71
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4513 0.6231 0.6284      0.00512       37459317    59
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4627 0.6452 0.6513      0.00517       37289756    59
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4633 0.6475 0.6540      0.00561       37282059    54
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4690 0.6552 0.6613      0.00577       38549095    53
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4692 0.6568 0.6632      0.00596       38533871    51
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4725 0.6623 0.6687      0.00713       41115368    43
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4726 0.6626 0.6692      0.00823       41086027    37
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4761 0.6805 0.6875      0.01367       73058906    22
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.4864 0.6999 0.7082      0.01403       72576665    22
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5137 0.7390 0.7460      0.01431       74629898    21
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5140 0.7447 0.7529      0.01419       73023090    22
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5240 0.7603 0.7686      0.01452       74130346    21
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5289 0.7668 0.7751      0.01576       76652292    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5472 0.8159 0.8253      0.02592      144083572    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5629 0.8402 0.8499      0.02629      144946851    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5679 0.8504 0.8602      0.02632      147298674    12
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5693 0.8515 0.8612      0.03052      147205194    10
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.5695 0.8519 0.8616      0.03114      162687479    10
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5697 0.8520 0.8617      0.03191      162555146    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5708 0.8616 0.8735      0.06001      285402522    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5953 0.9067 0.9185      0.06221      287307172    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5966 0.9080 0.9199      0.06235      292268032    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5969 0.9094 0.9212      0.06813      302583017    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.5971 0.9096 0.9214      0.07281      322717828    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6113 0.9438 0.9577      0.11326      564368849    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.6135 0.9478 0.9617      0.12025      598292895    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.6161 0.9613 0.9767      0.21100     1108428145    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6183 0.9664 0.9822      0.20032     1111295809    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6189 0.9675 0.9833      0.19959     1120756416    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6190 0.9633 0.9788      0.19905     1121266552    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.6192 0.9676 0.9834      0.21460     1140989686    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.6193 0.9755 0.9914      0.36148     2222182889    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6217 0.9761 0.9916      0.37007     2173607795    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6228 0.9774 0.9930      0.37404     2181066401    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6233 0.9792 0.9965      0.67714     4290280440    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6235 0.9794 0.9969      0.67271     4248656289    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fs` </summary>
Index size 4085025484

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF65536_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3921 0.8253 0.9788      0.10029     4237504852    3
nprobe=1,quantizer_efSearch=4            0.1862 0.3006 0.3202      0.00239       17778676    126
nprobe=1,quantizer_efSearch=8            0.2028 0.3257 0.3471      0.00267       17762129    113
nprobe=1,quantizer_efSearch=16           0.2067 0.3319 0.3536      0.00312       17729141    97
nprobe=2,quantizer_efSearch=4            0.2385 0.4081 0.4431      0.00371       35503254    81
nprobe=2,quantizer_efSearch=8            0.2610 0.4447 0.4824      0.00392       35497500    77
nprobe=2,quantizer_efSearch=16           0.2644 0.4525 0.4908      0.00442       35451399    68
nprobe=2,quantizer_efSearch=32           0.2651 0.4535 0.4918      0.00517       35433231    59
nprobe=4,quantizer_efSearch=4            0.2750 0.5050 0.5616      0.00584       70997536    52
nprobe=4,quantizer_efSearch=8            0.3043 0.5533 0.6150      0.00607       71038386    50
nprobe=4,quantizer_efSearch=16           0.3099 0.5675 0.6301      0.00637       70893981    48
nprobe=4,quantizer_efSearch=32           0.3106 0.5684 0.6310      0.00717       70865961    42
nprobe=8,quantizer_efSearch=4            0.3267 0.6276 0.7125      0.00880      141900988    35
nprobe=8,quantizer_efSearch=8            0.3340 0.6420 0.7288      0.00905      141690126    34
nprobe=8,quantizer_efSearch=16           0.3418 0.6614 0.7504      0.00978      141355530    31
nprobe=8,quantizer_efSearch=32           0.3428 0.6643 0.7535      0.01072      141220895    28
nprobe=16,quantizer_efSearch=4           0.3435 0.6822 0.7868      0.01214      282695573    25
nprobe=16,quantizer_efSearch=8           0.3555 0.7155 0.8246      0.01235      282148692    25
nprobe=16,quantizer_efSearch=16          0.3606 0.7296 0.8405      0.01280      281665938    24
nprobe=32,quantizer_efSearch=8           0.3659 0.7486 0.8743      0.01566      560072678    20
nprobe=64,quantizer_efSearch=8           0.3696 0.7647 0.8965      0.01885     1107141663    16
nprobe=32,quantizer_efSearch=16          0.3737 0.7705 0.8999      0.01608      558765302    19
nprobe=64,quantizer_efSearch=32          0.3868 0.8031 0.9433      0.02178     1102789565    14
nprobe=128,quantizer_efSearch=32         0.3881 0.8135 0.9606      0.03911     2173517360    8
nprobe=64,quantizer_efSearch=128         0.3884 0.8078 0.9487      0.04104     1100172030    8
nprobe=128,quantizer_efSearch=64         0.3902 0.8191 0.9673      0.04326     2167532686    7
nprobe=128,quantizer_efSearch=128        0.3904 0.8206 0.9691      0.05286     2164185535    6
nprobe=256,quantizer_efSearch=64         0.3911 0.8221 0.9752      0.06722     4243064866    5
nprobe=256,quantizer_efSearch=128        0.3917 0.8241 0.9776      0.07054     4241463943    5
nprobe=256,quantizer_efSearch=256        0.3921 0.8252 0.9788      0.07219     4238552089    5
nprobe=512,quantizer_efSearch=128        0.3929 0.8260 0.9804      0.10472     8241417876    3
nprobe=512,quantizer_efSearch=256        0.3932 0.8271 0.9816      0.09910     8268078576    4
nprobe=1024,quantizer_efSearch=256       0.3935 0.8282 0.9828      0.16305    16000637436    2
nprobe=1024,quantizer_efSearch=512       0.3936 0.8286 0.9833      0.18417    16010253350    2
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fsr` </summary>
Index size 4085035724

 code_size 32

 log filename: autotune.dbdeep100M.OPQ64_128_IVF65536_HNSW32_PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5122 0.7503 0.7615      0.01486      141172385    21
nprobe=1,quantizer_efSearch=16           0.2702 0.3510 0.3541      0.00291       17733208    104
nprobe=2,quantizer_efSearch=16           0.3617 0.4873 0.4927      0.00354       35453485    85
nprobe=4,quantizer_efSearch=16           0.4411 0.6247 0.6332      0.00491       70896306    62
nprobe=8,quantizer_efSearch=16           0.5079 0.7454 0.7566      0.00797      141358416    38
nprobe=8,quantizer_efSearch=32           0.5109 0.7489 0.7601      0.00880      141221709    35
nprobe=8,quantizer_efSearch=64           0.5121 0.7501 0.7613      0.01074      141181793    28
nprobe=8,quantizer_efSearch=128          0.5122 0.7503 0.7615      0.01462      141172385    21
nprobe=8,quantizer_efSearch=256          0.5123 0.7504 0.7616      0.02291      141169760    14
nprobe=16,quantizer_efSearch=128         0.5621 0.8411 0.8567      0.02401      281066824    13
nprobe=16,quantizer_efSearch=256         0.5622 0.8412 0.8568      0.03702      281057365    9
nprobe=32,quantizer_efSearch=256         0.5924 0.9039 0.9209      0.04880      556987531    7
nprobe=32,quantizer_efSearch=512         0.5925 0.9039 0.9209      0.06737      556984002    5
nprobe=64,quantizer_efSearch=16          0.5997 0.9263 0.9456      0.06620     1105858811    5
nprobe=64,quantizer_efSearch=256         0.6071 0.9423 0.9621      0.08438     1100024076    4
nprobe=128,quantizer_efSearch=512        0.6118 0.9623 0.9833      0.16858     2163222686    2
nprobe=128,quantizer_efSearch=256        0.6120 0.9623 0.9833      0.12729     2163344199    3
nprobe=256,quantizer_efSearch=256        0.6179 0.9706 0.9930      0.22283     4238528061    2
nprobe=512,quantizer_efSearch=256        0.6189 0.9734 0.9958      0.38413     8268121628    1
```

</details>
<details><summary>`PCAR32,IVF1048576_HNSW32,SQ8` </summary>
Index size 4428018773

 code_size 32

 log filename: autotune.dbdeep100M.PCAR32_IVF1048576_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2130 0.5241 0.8246      0.10484      293272181    3
nprobe=1,quantizer_efSearch=4            0.0874 0.1441 0.1499      0.00248        1240897    121
nprobe=2,quantizer_efSearch=4            0.1125 0.2012 0.2158      0.00259        2458049    117
nprobe=4,quantizer_efSearch=4            0.1351 0.2603 0.2941      0.00272        4881967    111
nprobe=4,quantizer_efSearch=8            0.1522 0.2966 0.3354      0.00330        4887938    91
nprobe=8,quantizer_efSearch=4            0.1642 0.3509 0.4236      0.00348        9745815    87
nprobe=8,quantizer_efSearch=8            0.1688 0.3599 0.4349      0.00391        9736203    77
nprobe=16,quantizer_efSearch=4           0.1738 0.3939 0.5072      0.00507       19392132    60
nprobe=16,quantizer_efSearch=8           0.1822 0.4158 0.5382      0.00552       19357070    55
nprobe=16,quantizer_efSearch=16          0.1878 0.4233 0.5508      0.00625       19299875    48
nprobe=32,quantizer_efSearch=8           0.1914 0.4518 0.6176      0.00874       38475311    35
nprobe=32,quantizer_efSearch=16          0.1985 0.4676 0.6458      0.00887       38355860    34
nprobe=32,quantizer_efSearch=32          0.1998 0.4710 0.6531      0.01055       38213267    29
nprobe=32,quantizer_efSearch=64          0.2011 0.4742 0.6562      0.01444       38100083    21
nprobe=64,quantizer_efSearch=16          0.2033 0.4885 0.7090      0.01482       76106513    21
nprobe=64,quantizer_efSearch=32          0.2066 0.4951 0.7246      0.01637       75795543    19
nprobe=64,quantizer_efSearch=64          0.2075 0.4975 0.7293      0.01983       75523802    16
nprobe=128,quantizer_efSearch=32         0.2089 0.5091 0.7740      0.02762      150193200    11
nprobe=128,quantizer_efSearch=64         0.2104 0.5133 0.7843      0.03029      149521641    10
nprobe=128,quantizer_efSearch=128        0.2109 0.5156 0.7883      0.03589      149023738    9
nprobe=256,quantizer_efSearch=64         0.2117 0.5201 0.8143      0.05250      295449195    6
nprobe=256,quantizer_efSearch=128        0.2125 0.5231 0.8223      0.05604      294307935    6
nprobe=256,quantizer_efSearch=256        0.2128 0.5240 0.8244      0.06580      293564066    5
nprobe=256,quantizer_efSearch=512        0.2130 0.5241 0.8246      0.10511      293272181    3
nprobe=512,quantizer_efSearch=128        0.2131 0.5288 0.8413      0.10313      579313418    3
nprobe=512,quantizer_efSearch=256        0.2135 0.5305 0.8432      0.11860      577855758    3
nprobe=512,quantizer_efSearch=512        0.2136 0.5308 0.8439      0.14227      576656645    3
nprobe=1024,quantizer_efSearch=512       0.2138 0.5329 0.8518      0.25176     1131865654    2
```

</details>
<details><summary>`PCAR32,IVF1048576(IVF1024,PQ16x4fs,RFlat),SQ8` </summary>
Index size 4159712153

 code_size 32

 log filename: autotune.dbdeep100M.PCAR32_IVF1048576_IVF1024_PQ16x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2087 0.4980 0.7262      0.02591       86420520    12
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.0853 0.1415 0.1464      0.00220        2701398    137
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1160 0.2086 0.2222      0.00234        3926366    129
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1182 0.2282 0.2518      0.00232        6387414    130
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1233 0.2398 0.2681      0.00240        5645649    125
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1347 0.2598 0.2905      0.00255        6376914    118
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1466 0.2856 0.3216      0.00280        6352758    107
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1542 0.2953 0.3340      0.00304        7740012    99
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1648 0.3387 0.4057      0.00355       12639114    85
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1744 0.3676 0.4412      0.00397       12581841    76
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1765 0.3784 0.4764      0.00493       25017158    62
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1858 0.4121 0.5312      0.00543       22238551    56
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1861 0.4058 0.5197      0.00550       24982957    55
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1887 0.4202 0.5438      0.00594       24930186    51
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.1892 0.4222 0.5467      0.00720       30163239    42
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.1917 0.4273 0.5552      0.00792       30125924    38
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1972 0.4577 0.6253      0.00896       44043732    34
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.1982 0.4620 0.6302      0.00924       49260339    33
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2009 0.4672 0.6429      0.00929       43934270    33
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2014 0.4704 0.6470      0.01064       49160514    29
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2017 0.4707 0.6478      0.01326       59565716    23
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2020 0.4734 0.6818      0.01413       79239603    22
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2060 0.4893 0.7081      0.01459       81740771    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2071 0.4937 0.7135      0.01738       86861924    18
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2084 0.4984 0.7247      0.02019       96956398    15
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2090 0.4991 0.7277      0.02386       96817545    13
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.2091 0.4986 0.7275      0.02539      117534376    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2107 0.5111 0.7744      0.02700      160842962    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2116 0.5124 0.7776      0.02928      171119868    11
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2119 0.5150 0.7848      0.03941      191284072    8
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.2120 0.5158 0.7865      0.04165      170445746    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2128 0.5184 0.8040      0.04822      317791875    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2129 0.5202 0.8140      0.04726      306468097    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2139 0.5224 0.8180      0.05255      336997500    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.2148 0.5278 0.8328      0.09707      665489882    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2154 0.5311 0.8422      0.09800      621071679    4
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.2155 0.5323 0.8488      0.31355     2344067943    1
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ8` </summary>
Index size 4107045205

 code_size 32

 log filename: autotune.dbdeep100M.PCAR32_IVF262144_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2152 0.5317 0.8476      0.11071     1082113196    3
nprobe=2,quantizer_efSearch=4            0.1297 0.2591 0.2973      0.00219        8915915    138
nprobe=2,quantizer_efSearch=8            0.1414 0.2812 0.3218      0.00242        8915757    124
nprobe=4,quantizer_efSearch=4            0.1482 0.3214 0.3906      0.00283       17797123    106
nprobe=4,quantizer_efSearch=8            0.1655 0.3516 0.4305      0.00296       17824107    102
nprobe=4,quantizer_efSearch=16           0.1679 0.3602 0.4396      0.00355       17772535    85
nprobe=8,quantizer_efSearch=4            0.1770 0.4053 0.5275      0.00431       35522140    70
nprobe=8,quantizer_efSearch=8            0.1817 0.4140 0.5396      0.00415       35508863    73
nprobe=8,quantizer_efSearch=16           0.1865 0.4256 0.5547      0.00467       35410893    65
nprobe=8,quantizer_efSearch=32           0.1886 0.4277 0.5575      0.00579       35368590    52
nprobe=16,quantizer_efSearch=8           0.1949 0.4598 0.6390      0.00685       70693888    44
nprobe=16,quantizer_efSearch=16          0.1974 0.4667 0.6513      0.00687       70584943    44
nprobe=16,quantizer_efSearch=32          0.1998 0.4714 0.6583      0.00836       70466913    36
nprobe=16,quantizer_efSearch=64          0.2007 0.4720 0.6584      0.01085       70410877    28
nprobe=32,quantizer_efSearch=32          0.2073 0.5006 0.7357      0.01532      140058706    20
nprobe=32,quantizer_efSearch=64          0.2079 0.5017 0.7379      0.01488      139896862    21
nprobe=32,quantizer_efSearch=128         0.2085 0.5029 0.7394      0.01925      139837926    16
nprobe=64,quantizer_efSearch=32          0.2098 0.5162 0.7911      0.02149      278084793    14
nprobe=64,quantizer_efSearch=128         0.2111 0.5198 0.7963      0.02720      277378221    12
nprobe=64,quantizer_efSearch=256         0.2115 0.5201 0.7967      0.03689      277318232    9
nprobe=128,quantizer_efSearch=64         0.2126 0.5254 0.8267      0.04564      549634366    7
nprobe=128,quantizer_efSearch=128        0.2132 0.5274 0.8301      0.04337      548845021    7
nprobe=128,quantizer_efSearch=256        0.2136 0.5276 0.8306      0.05228      548581091    6
nprobe=128,quantizer_efSearch=512        0.2137 0.5278 0.8308      0.07510      548517903    4
nprobe=256,quantizer_efSearch=128        0.2147 0.5310 0.8469      0.08272     1083667471    4
nprobe=256,quantizer_efSearch=256        0.2151 0.5315 0.8473      0.08928     1082421413    4
nprobe=256,quantizer_efSearch=512        0.2152 0.5317 0.8476      0.10707     1082113196    3
nprobe=512,quantizer_efSearch=256        0.2153 0.5331 0.8557      0.15739     2129119419    2
nprobe=512,quantizer_efSearch=512        0.2154 0.5332 0.8562      0.17741     2127480827    2
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ8` </summary>
Index size 4040039065

 code_size 32

 log filename: autotune.dbdeep100M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2030 0.4883 0.7120      0.02304      288937530    14
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.0662 0.1195 0.1295      0.00248        5237611    122
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1005 0.1789 0.1942      0.00263        7236766    115
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1135 0.2044 0.2224      0.00257        5879899    117
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1328 0.2631 0.3008      0.00276        9725175    109
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4     0.1358 0.2718 0.3093      0.00269        9671182    112
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1423 0.2826 0.3219      0.00274       10332708    110
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1438 0.3018 0.3658      0.00338       18372570    89
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1586 0.3393 0.4131      0.00345       18639340    88
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1592 0.3374 0.4080      0.00353       20689385    86
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1648 0.3506 0.4266      0.00369       20646990    82
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1704 0.3841 0.4949      0.00487       36623076    62
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1803 0.4059 0.5257      0.00511       38504872    59
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1848 0.4179 0.5454      0.00544       38378445    56
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1856 0.4217 0.5507      0.00564       38278312    54
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.1862 0.4244 0.5532      0.00612       40863403    49
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.1863 0.4247 0.5541      0.00730       46086362    42
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1868 0.4308 0.5836      0.00737       73069655    41
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1885 0.4382 0.5939      0.00776       74245029    39
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1973 0.4602 0.6374      0.00795       73927230    38
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1985 0.4646 0.6489      0.00822       73632499    37
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.1988 0.4684 0.6543      0.01034       81215625    30
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.1995 0.4690 0.6556      0.01090       75947754    28
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2005 0.4778 0.7009      0.01313      143034931    23
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2024 0.4846 0.6971      0.01325      147136306    23
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2070 0.4996 0.7337      0.01447      145932279    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2075 0.4974 0.7268      0.01484      151534673    21
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2081 0.5011 0.7372      0.01705      150850445    18
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64   0.2082 0.5012 0.7375      0.02008      150741919    15
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2091 0.5099 0.7670      0.02389      286405780    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2098 0.5176 0.7882      0.02589      289818570    12
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2111 0.5198 0.7940      0.02633      289003794    12
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.2112 0.5196 0.7938      0.02968      319724792    11
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2117 0.5232 0.8133      0.04296      565624131    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2129 0.5260 0.8268      0.04559      562401766    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2130 0.5258 0.8265      0.04366      572689333    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2131 0.5275 0.8292      0.05351      570991326    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2137 0.5276 0.8362      0.07959     1100766416    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2142 0.5293 0.8396      0.07789     1114249511    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2145 0.5295 0.8414      0.08312     1094254007    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2150 0.5316 0.8463      0.08489     1107737246    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2152 0.5330 0.8529      0.14393     2157689579    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2153 0.5329 0.8535      0.14481     2166508746    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2154 0.5334 0.8554      0.16789     2152895998    2
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ8` </summary>
Index size 4026800981

 code_size 32

 log filename: autotune.dbdeep100M.PCAR32_IVF65536_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1241 0.2489 0.2893      0.00517       17705566    59
nprobe=1,quantizer_efSearch=16           0.1229 0.2475 0.2879      0.00369       17721559    82
nprobe=1,quantizer_efSearch=32           0.1239 0.2486 0.2890      0.00418       17703755    72
nprobe=1,quantizer_efSearch=64           0.1241 0.2489 0.2893      0.00521       17705566    58
nprobe=2,quantizer_efSearch=4            0.1441 0.3063 0.3805      0.00531       35483230    57
nprobe=2,quantizer_efSearch=8            0.1532 0.3257 0.4054      0.00550       35481330    55
nprobe=2,quantizer_efSearch=16           0.1553 0.3300 0.4109      0.00581       35429063    52
nprobe=2,quantizer_efSearch=32           0.1562 0.3314 0.4126      0.00636       35409376    48
nprobe=2,quantizer_efSearch=64           0.1563 0.3316 0.4128      0.00733       35412436    41
nprobe=4,quantizer_efSearch=4            0.1607 0.3637 0.4838      0.00935       70730101    33
nprobe=4,quantizer_efSearch=8            0.1757 0.3964 0.5276      0.00960       70795679    32
nprobe=4,quantizer_efSearch=16           0.1784 0.4020 0.5357      0.00990       70662189    31
nprobe=4,quantizer_efSearch=32           0.1797 0.4036 0.5377      0.01047       70612842    29
nprobe=4,quantizer_efSearch=64           0.1798 0.4039 0.5380      0.01145       70609659    27
nprobe=8,quantizer_efSearch=4            0.1858 0.4400 0.6213      0.01763      141071782    18
nprobe=8,quantizer_efSearch=8            0.1907 0.4476 0.6336      0.01760      140986718    18
nprobe=8,quantizer_efSearch=16           0.1949 0.4560 0.6465      0.01786      140756813    17
nprobe=8,quantizer_efSearch=32           0.1963 0.4585 0.6491      0.01837      140625809    17
nprobe=8,quantizer_efSearch=128          0.1964 0.4588 0.6492      0.02162      140606491    14
nprobe=16,quantizer_efSearch=8           0.2026 0.4862 0.7202      0.03410      279847434    9
nprobe=16,quantizer_efSearch=16          0.2053 0.4929 0.7297      0.03339      279445734    9
nprobe=16,quantizer_efSearch=32          0.2064 0.4958 0.7341      0.03417      279143649    9
nprobe=16,quantizer_efSearch=64          0.2068 0.4961 0.7349      0.03514      279042943    9
nprobe=16,quantizer_efSearch=128         0.2069 0.4963 0.7350      0.03759      279022037    8
nprobe=32,quantizer_efSearch=16          0.2112 0.5116 0.7879      0.06145      553612622    5
nprobe=32,quantizer_efSearch=64          0.2122 0.5151 0.7946      0.06298      552332252    5
nprobe=32,quantizer_efSearch=128         0.2125 0.5157 0.7950      0.06526      552248248    5
nprobe=32,quantizer_efSearch=256         0.2126 0.5158 0.7951      0.07142      552224923    5
nprobe=64,quantizer_efSearch=32          0.2134 0.5233 0.8256      0.11526     1091739375    3
nprobe=64,quantizer_efSearch=64          0.2144 0.5256 0.8286      0.11584     1090376593    3
nprobe=128,quantizer_efSearch=128        0.2148 0.5311 0.8468      0.22019     2143600928    2
```

</details>
<details><summary>`PCAR64,IVF1048576_HNSW32,SQ4` </summary>
Index size 4562249173

 code_size 32

 log filename: autotune.dbdeep100M.PCAR64_IVF1048576_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4786 0.8894 0.9521      0.12198      307690546    3
nprobe=1,quantizer_efSearch=4            0.1383 0.1781 0.1804      0.00281        1289448    107
nprobe=2,quantizer_efSearch=4            0.1840 0.2529 0.2560      0.00287        2576268    105
nprobe=4,quantizer_efSearch=4            0.2314 0.3375 0.3420      0.00295        5132947    102
nprobe=4,quantizer_efSearch=8            0.2676 0.3927 0.3976      0.00363        5123987    83
nprobe=8,quantizer_efSearch=4            0.3110 0.4821 0.4920      0.00472       10250647    64
nprobe=8,quantizer_efSearch=8            0.3182 0.4951 0.5050      0.00471       10234023    64
nprobe=8,quantizer_efSearch=16           0.3363 0.5217 0.5324      0.00630       10176518    48
nprobe=16,quantizer_efSearch=4           0.3481 0.5667 0.5839      0.00633       20420907    48
nprobe=16,quantizer_efSearch=8           0.3720 0.6091 0.6273      0.00664       20374583    46
nprobe=16,quantizer_efSearch=16          0.3825 0.6243 0.6430      0.00763       20314344    40
nprobe=16,quantizer_efSearch=32          0.3893 0.6333 0.6524      0.01060       20202611    29
nprobe=32,quantizer_efSearch=8           0.4003 0.6826 0.7082      0.01058       40545085    29
nprobe=32,quantizer_efSearch=16          0.4175 0.7136 0.7425      0.01144       40394610    27
nprobe=32,quantizer_efSearch=32          0.4260 0.7266 0.7558      0.01353       40197148    23
nprobe=64,quantizer_efSearch=16          0.4376 0.7715 0.8106      0.01808       80230717    17
nprobe=64,quantizer_efSearch=32          0.4487 0.7952 0.8371      0.01990       79783768    16
nprobe=64,quantizer_efSearch=64          0.4535 0.8033 0.8457      0.02346       79367217    13
nprobe=128,quantizer_efSearch=32         0.4596 0.8377 0.8887      0.03464      158244078    9
nprobe=128,quantizer_efSearch=64         0.4686 0.8540 0.9062      0.03780      157327929    8
nprobe=128,quantizer_efSearch=128        0.4717 0.8584 0.9121      0.04611      156634400    7
nprobe=256,quantizer_efSearch=64         0.4732 0.8779 0.9395      0.05892      311016604    6
nprobe=256,quantizer_efSearch=128        0.4770 0.8874 0.9502      0.07740      309349834    4
nprobe=256,quantizer_efSearch=256        0.4782 0.8891 0.9517      0.08538      308197409    4
nprobe=512,quantizer_efSearch=128        0.4813 0.9036 0.9721      0.13915      608910161    3
nprobe=512,quantizer_efSearch=256        0.4834 0.9076 0.9758      0.15047      606481181    2
nprobe=1024,quantizer_efSearch=256       0.4854 0.9141 0.9858      0.24609     1188671780    2
nprobe=1024,quantizer_efSearch=512       0.4857 0.9154 0.9877      0.31597     1185269991    1
nprobe=2048,quantizer_efSearch=256       0.4865 0.9172 0.9898      0.45489     2275150841    1
nprobe=2048,quantizer_efSearch=512       0.4870 0.9197 0.9933      0.51529     2309091126    1
nprobe=4096,quantizer_efSearch=512       0.4872 0.9207 0.9945      0.89183     4342932549    1
```

</details>
<details><summary>`PCAR64,IVF1048576(IVF1024,PQ32x4fs,RFlat),SQ4` </summary>
Index size 4302594585

 code_size 32

 log filename: autotune.dbdeep100M.PCAR64_IVF1048576_IVF1024_PQ32x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.4897 0.9181 0.9949      1.82055     4517011409    1
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.1214 0.1555 0.1570      0.00241        2027058    125
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.1350 0.1730 0.1747      0.00247        2023952    122
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.1414 0.1810 0.1829      0.00259        2022877    116
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1473 0.1954 0.1976      0.00271        3312648    111
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1476 0.1958 0.1981      0.00253        2941485    119
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.1575 0.2102 0.2131      0.00255        2943521    118
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1       0.1618 0.2180 0.2211      0.00261        2939294    116
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.1859 0.2488 0.2519      0.00279        3302276    108
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1961 0.2684 0.2716      0.00287        5880770    105
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.2083 0.2840 0.2874      0.00290        4008890    104
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2250 0.3129 0.3176      0.00287        5868119    105
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2368 0.3358 0.3414      0.00304        5862257    99
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2481 0.3482 0.3525      0.00302        6577983    100
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2617 0.3736 0.3791      0.00310        6559559    97
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.2683 0.3856 0.3920      0.00360        6551183    84
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.2803 0.4035 0.4102      0.00396        7955040    76
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3027 0.4496 0.4572      0.00430       11678941    70
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.3278 0.5011 0.5107      0.00497       13047798    61
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.3359 0.5126 0.5224      0.00672       13017770    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.3362 0.5135 0.5234      0.00630       21118614    48
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3404 0.5420 0.5551      0.00684       21810480    44
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.3456 0.5275 0.5379      0.00681       21094697    45
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3521 0.5650 0.5807      0.00727       21760224    42
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3573 0.5715 0.5885      0.00946       21730041    32
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3710 0.5912 0.6063      0.00763       25865918    40
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3842 0.6172 0.6355      0.00841       25811600    36
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3891 0.6272 0.6467      0.00919       25781806    33
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3909 0.6311 0.6506      0.00981       31123309    31
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3978 0.6664 0.6916      0.01189       43224905    26
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4106 0.6941 0.7243      0.01634       43078427    19
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4240 0.7207 0.7523      0.01483       45705939    21
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4263 0.7264 0.7588      0.01595       50963875    19
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4270 0.7286 0.7608      0.01795       61416706    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4384 0.7683 0.8085      0.02195       85369378    14
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4431 0.7767 0.8173      0.02193       90536565    14
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4509 0.7956 0.8394      0.03293       90189991    10
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4538 0.8007 0.8449      0.03201      100481359    10
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.4540 0.8009 0.8452      0.03270      100452045    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4705 0.8486 0.9027      0.04602      167759407    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4720 0.8525 0.9072      0.04528      177973058    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.4732 0.8551 0.9101      0.05465      198335839    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4789 0.8784 0.9416      0.07404      320105719    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.4808 0.8847 0.9483      0.08014      350320672    4
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.4813 0.8862 0.9506      0.11039      350046472    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.4818 0.8869 0.9512      0.10084      391232454    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4825 0.8932 0.9610      0.13346      620173382    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.4867 0.9048 0.9747      0.14383      646941525    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512   0.4869 0.9061 0.9765      0.18601      769332687    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.4886 0.9124 0.9857      0.26926     1268541699    2
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.4890 0.9141 0.9883      0.49063     2423255083    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4896 0.9168 0.9925      0.47816     2352165605    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512  0.4897 0.9180 0.9941      0.53316     2466354427    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.4899 0.9195 0.9965      0.98203     4549387449    1
nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=256 0.4900 0.9195 0.9966      1.89228     4548933235    1
nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=512 0.4901 0.9197 0.9968      1.98769     4628528636    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ4` </summary>
Index size 4140612309

 code_size 32

 log filename: autotune.dbdeep100M.PCAR64_IVF262144_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4823 0.9050 0.9790      0.21835     1100127483    2
nprobe=2,quantizer_efSearch=4            0.2283 0.3382 0.3452      0.00330        8984988    91
nprobe=2,quantizer_efSearch=8            0.2586 0.3795 0.3868      0.00337        8964322    90
nprobe=4,quantizer_efSearch=4            0.2737 0.4338 0.4471      0.00456       17966556    66
nprobe=4,quantizer_efSearch=8            0.3156 0.4948 0.5090      0.00496       17965747    61
nprobe=4,quantizer_efSearch=32           0.3266 0.5107 0.5251      0.00683       17893040    44
nprobe=8,quantizer_efSearch=4            0.3558 0.5858 0.6098      0.00768       35923544    40
nprobe=8,quantizer_efSearch=8            0.3617 0.5987 0.6226      0.00792       35873157    38
nprobe=8,quantizer_efSearch=32           0.3762 0.6236 0.6483      0.00981       35717850    31
nprobe=8,quantizer_efSearch=64           0.3763 0.6242 0.6489      0.01253       35690881    24
nprobe=16,quantizer_efSearch=32          0.4164 0.7207 0.7570      0.01609       71246559    19
nprobe=16,quantizer_efSearch=128         0.4172 0.7223 0.7586      0.02372       71157732    13
nprobe=32,quantizer_efSearch=8           0.4283 0.7614 0.8057      0.02647      142746319    12
nprobe=32,quantizer_efSearch=32          0.4477 0.7984 0.8459      0.02896      141976163    11
nprobe=32,quantizer_efSearch=128         0.4496 0.8006 0.8483      0.03380      141664155    9
nprobe=64,quantizer_efSearch=16          0.4546 0.8324 0.8884      0.04914      283382898    7
nprobe=64,quantizer_efSearch=32          0.4630 0.8504 0.9076      0.05031      282544454    6
nprobe=64,quantizer_efSearch=64          0.4674 0.8550 0.9127      0.05255      281917371    6
nprobe=64,quantizer_efSearch=128         0.4678 0.8562 0.9141      0.05621      281645783    6
nprobe=64,quantizer_efSearch=256         0.4680 0.8565 0.9144      0.06556      281554275    5
nprobe=128,quantizer_efSearch=128        0.4771 0.8883 0.9559      0.10246      558123536    3
nprobe=128,quantizer_efSearch=256        0.4773 0.8889 0.9562      0.10973      557727445    3
nprobe=128,quantizer_efSearch=512        0.4774 0.8890 0.9563      0.13907      557623770    3
nprobe=256,quantizer_efSearch=64         0.4798 0.9003 0.9724      0.18870     1104645517    2
nprobe=256,quantizer_efSearch=128        0.4818 0.9038 0.9777      0.19236     1102370927    2
nprobe=256,quantizer_efSearch=256        0.4824 0.9050 0.9790      0.19323     1100617753    2
nprobe=512,quantizer_efSearch=256        0.4832 0.9122 0.9890      0.36597     2164094541    1
nprobe=512,quantizer_efSearch=512        0.4835 0.9128 0.9896      0.39229     2161948946    1
nprobe=1024,quantizer_efSearch=256       0.4841 0.9152 0.9934      0.68910     4222439084    1
nprobe=1024,quantizer_efSearch=512       0.4843 0.9163 0.9945      0.73105     4227793847    1
nprobe=2048,quantizer_efSearch=512       0.4844 0.9171 0.9955      1.36112     8202549239    1
nprobe=4096,quantizer_efSearch=512       0.4846 0.9177 0.9964      2.40166    14996226520    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ4` </summary>
Index size 4075829785

 code_size 32

 log filename: autotune.dbdeep100M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1769 0.2436 0.2474      0.00269        4884821    112
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1465 0.2013 0.2049      0.00266        4695687    114
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1495 0.2050 0.2085      0.00263        4695314    114
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2     0.1769 0.2436 0.2474      0.00274        4884615    110
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1962 0.2705 0.2744      0.00290        5902914    104
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2054 0.2936 0.2985      0.00317       10464129    95
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2303 0.3332 0.3391      0.00326        9752707    92
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2312 0.3383 0.3451      0.00312        9397618    97
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2508 0.3670 0.3743      0.00316        9729276    95
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.2616 0.3835 0.3909      0.00355       10401805    85
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.2655 0.3865 0.3939      0.00438       14340754    69
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2680 0.4042 0.4131      0.00468       19505037    65
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2707 0.4238 0.4352      0.00470       18458985    64
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2755 0.4306 0.4427      0.00478       18432308    63
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.2765 0.4314 0.4442      0.00472       18422838    64
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3058 0.4788 0.4926      0.00494       18770134    61
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3196 0.4982 0.5109      0.00507       20706887    60
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3250 0.5068 0.5205      0.00524       20695832    58
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32    0.3284 0.5119 0.5266      0.00700       23296744    43
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3453 0.5711 0.5928      0.00740       36813041    41
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3526 0.5781 0.5976      0.01166       37485140    26
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3703 0.6114 0.6339      0.01172       38602412    26
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3719 0.6141 0.6371      0.00850       41211472    36
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=64    0.3754 0.6236 0.6481      0.01157       46367677    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3944 0.6779 0.7087      0.01267       73370539    24
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4046 0.6938 0.7254      0.01299       74508137    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4114 0.7124 0.7478      0.01351       74246591    23
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4149 0.7185 0.7543      0.01401       76761357    22
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4150 0.7191 0.7547      0.01476       81947505    21
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.4168 0.7206 0.7570      0.01640       81868805    19
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.4169 0.7208 0.7572      0.02940       92310357    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4237 0.7523 0.7961      0.02273      144830839    14
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4259 0.7602 0.8068      0.02324      144367089    13
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4429 0.7844 0.8302      0.02465      148002034    13
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4430 0.7853 0.8308      0.02493      153144537    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4450 0.7953 0.8444      0.02589      147388530    12
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4456 0.7942 0.8432      0.02559      152623899    12
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.4460 0.7972 0.8464      0.02962      152476366    11
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64   0.4461 0.7973 0.8465      0.03213      152472014    10
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=256  0.4464 0.7978 0.8470      0.03315      183396418    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4615 0.8422 0.8998      0.04480      288857127    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4629 0.8478 0.9063      0.04563      287946126    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4645 0.8506 0.9089      0.04699      292897123    7
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.4651 0.8513 0.9103      0.07543      287645422    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.4668 0.8536 0.9126      0.04714      292647788    7
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.4672 0.8547 0.9138      0.05377      302952589    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4728 0.8796 0.9462      0.08504      571273296    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4737 0.8805 0.9473      0.08656      581435204    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4759 0.8856 0.9535      0.08868      569596936    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.4767 0.8862 0.9542      0.10082      600233754    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.4768 0.8881 0.9558      0.10441      599892916    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4778 0.8958 0.9684      0.15824     1113565460    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4805 0.9029 0.9775      0.16177     1113408973    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.4807 0.9010 0.9745      0.15836     1146465478    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.4813 0.9039 0.9787      0.20745     1122608859    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.4821 0.9099 0.9874      0.30071     2211051714    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4826 0.9140 0.9942      0.57334     4266952426    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4828 0.9151 0.9957      1.11324     8293032451    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.4829 0.9151 0.9958      1.10819     8306235614    1
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ4` </summary>
Index size 4035202261

 code_size 32

 log filename: autotune.dbdeep100M.PCAR64_IVF65536_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4760 0.9111 0.9900      0.56459     4247593629    1
nprobe=1,quantizer_efSearch=16           0.2297 0.3352 0.3424      0.00436       17720484    69
nprobe=1,quantizer_efSearch=32           0.2303 0.3360 0.3432      0.00479       17704026    63
nprobe=2,quantizer_efSearch=4            0.2676 0.4238 0.4346      0.00606       35531044    50
nprobe=2,quantizer_efSearch=8            0.2937 0.4623 0.4739      0.00636       35574662    48
nprobe=2,quantizer_efSearch=16           0.2978 0.4686 0.4807      0.00657       35466241    46
nprobe=2,quantizer_efSearch=32           0.2985 0.4692 0.4814      0.00723       35442831    42
nprobe=4,quantizer_efSearch=4            0.3184 0.5303 0.5501      0.01137       71001346    27
nprobe=4,quantizer_efSearch=8            0.3489 0.5828 0.6041      0.01093       71113612    28
nprobe=4,quantizer_efSearch=16           0.3584 0.5960 0.6181      0.01121       70942038    27
nprobe=4,quantizer_efSearch=32           0.3603 0.5981 0.6203      0.01178       70889843    26
nprobe=8,quantizer_efSearch=8            0.3918 0.6849 0.7197      0.02000      141864673    15
nprobe=8,quantizer_efSearch=16           0.4028 0.7024 0.7381      0.02054      141457810    15
nprobe=8,quantizer_efSearch=32           0.4050 0.7052 0.7412      0.02228      141289295    14
nprobe=16,quantizer_efSearch=32          0.4356 0.7943 0.8428      0.03937      281422701    8
nprobe=16,quantizer_efSearch=128         0.4362 0.7954 0.8440      0.04226      281235640    8
nprobe=32,quantizer_efSearch=8           0.4421 0.8188 0.8773      0.07283      560669652    5
nprobe=32,quantizer_efSearch=128         0.4567 0.8489 0.9115      0.07823      557599192    4
nprobe=64,quantizer_efSearch=32          0.4642 0.8795 0.9498      0.14217     1104379532    3
nprobe=64,quantizer_efSearch=128         0.4659 0.8824 0.9531      0.14614     1101996427    3
nprobe=128,quantizer_efSearch=128        0.4729 0.9026 0.9782      0.27898     2168316289    2
nprobe=256,quantizer_efSearch=64         0.4751 0.9081 0.9865      0.53821     4253892443    1
nprobe=256,quantizer_efSearch=128        0.4760 0.9106 0.9895      0.54819     4251790277    1
nprobe=512,quantizer_efSearch=128        0.4765 0.9130 0.9931      1.04131     8263099357    1
nprobe=1024,quantizer_efSearch=128       0.4767 0.9143 0.9946      1.95446    15543093707    1
nprobe=1024,quantizer_efSearch=512       0.4768 0.9152 0.9961      2.05209    16051993351    1
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF1048576_HNSW32,PQ128x4fs` </summary>
Index size 9627512780

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF1048576_HNSW32_PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2224 0.3035 0.3073      0.01106        2596388    28
nprobe=2,quantizer_efSearch=4            0.1919 0.2635 0.2671      0.00719        2598595    42
nprobe=4,quantizer_efSearch=4            0.2383 0.3451 0.3518      0.00750        5194361    41
nprobe=8,quantizer_efSearch=4            0.3262 0.4957 0.5072      0.01048       10383967    29
nprobe=16,quantizer_efSearch=4           0.3664 0.5821 0.5999      0.01196       20668310    26
nprobe=32,quantizer_efSearch=4           0.3872 0.6330 0.6564      0.01405       41008007    22
nprobe=16,quantizer_efSearch=8           0.3930 0.6234 0.6421      0.01542       20603009    20
nprobe=32,quantizer_efSearch=8           0.4219 0.6943 0.7205      0.01801       40978359    17
nprobe=64,quantizer_efSearch=8           0.4362 0.7322 0.7655      0.02186       81199959    14
nprobe=32,quantizer_efSearch=16          0.4404 0.7284 0.7563      0.02388       40785500    13
nprobe=64,quantizer_efSearch=16          0.4627 0.7832 0.8211      0.02768       80990886    11
nprobe=64,quantizer_efSearch=32          0.4782 0.8080 0.8466      0.03866       80534092    8
nprobe=128,quantizer_efSearch=32         0.4949 0.8499 0.8944      0.04584      159655616    7
nprobe=128,quantizer_efSearch=64         0.5010 0.8671 0.9126      0.06616      158707041    5
nprobe=256,quantizer_efSearch=64         0.5114 0.8946 0.9450      0.07737      313606182    4
nprobe=512,quantizer_efSearch=64         0.5138 0.9059 0.9584      0.09589      612952699    4
nprobe=256,quantizer_efSearch=128        0.5164 0.9042 0.9559      0.11193      311923557    3
nprobe=512,quantizer_efSearch=128        0.5194 0.9185 0.9733      0.13180      613859002    3
nprobe=1024,quantizer_efSearch=128       0.5208 0.9240 0.9805      0.17662     1189511039    2
nprobe=512,quantizer_efSearch=256        0.5221 0.9239 0.9792      0.19070      611185175    2
nprobe=1024,quantizer_efSearch=256       0.5234 0.9312 0.9890      0.22607     1197067931    2
nprobe=2048,quantizer_efSearch=256       0.5239 0.9330 0.9915      0.28756     2288851105    2
nprobe=1024,quantizer_efSearch=512       0.5245 0.9332 0.9911      0.32663     1193361619    1
nprobe=2048,quantizer_efSearch=512       0.5255 0.9360 0.9947      0.38693     2324282294    1
nprobe=4096,quantizer_efSearch=512       0.5261 0.9369 0.9959      0.49139     4365628405    1
```

</details>
<details><summary>`OPQ128_256,IVF1048576_HNSW32,PQ128x4fsr` </summary>
Index size 9627682764

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF1048576_HNSW32_PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5942 0.6726 0.6737      0.04658       20408764    7
nprobe=1,quantizer_efSearch=4            0.1720 0.1863 0.1871      0.00577        1296757    53
nprobe=2,quantizer_efSearch=4            0.2439 0.2669 0.2679      0.00692        2596963    44
nprobe=4,quantizer_efSearch=4            0.3178 0.3502 0.3512      0.00906        5191227    34
nprobe=4,quantizer_efSearch=8            0.3779 0.4160 0.4171      0.01187        5187293    26
nprobe=4,quantizer_efSearch=16           0.3931 0.4321 0.4332      0.01687        5151212    18
nprobe=8,quantizer_efSearch=4            0.4554 0.5072 0.5084      0.01896       10382633    16
nprobe=8,quantizer_efSearch=8            0.4683 0.5209 0.5221      0.02138       10360154    15
nprobe=8,quantizer_efSearch=16           0.4934 0.5492 0.5504      0.02534       10295149    12
nprobe=8,quantizer_efSearch=32           0.5017 0.5582 0.5594      0.03277       10239188    10
nprobe=16,quantizer_efSearch=4           0.5328 0.6006 0.6017      0.03323       20668978    10
nprobe=16,quantizer_efSearch=8           0.5667 0.6412 0.6423      0.03467       20606194    9
nprobe=16,quantizer_efSearch=16          0.5804 0.6582 0.6593      0.03727       20529320    9
nprobe=16,quantizer_efSearch=32          0.5942 0.6726 0.6737      0.04708       20408764    7
nprobe=16,quantizer_efSearch=64          0.5971 0.6759 0.6770      0.06088       20334534    5
nprobe=32,quantizer_efSearch=8           0.6295 0.7196 0.7209      0.07003       40980122    5
nprobe=32,quantizer_efSearch=16          0.6579 0.7557 0.7570      0.07353       40785204    5
nprobe=32,quantizer_efSearch=32          0.6710 0.7713 0.7726      0.07932       40571986    4
nprobe=32,quantizer_efSearch=64          0.6769 0.7772 0.7785      0.08941       40392967    4
nprobe=32,quantizer_efSearch=128         0.6785 0.7792 0.7805      0.11455       40297023    3
nprobe=64,quantizer_efSearch=16          0.7065 0.8207 0.8222      0.13209       80992993    3
nprobe=64,quantizer_efSearch=32          0.7282 0.8462 0.8477      0.13430       80534046    3
nprobe=64,quantizer_efSearch=64          0.7351 0.8556 0.8571      0.14320       80110021    3
nprobe=64,quantizer_efSearch=128         0.7380 0.8586 0.8601      0.17277       79848021    2
nprobe=64,quantizer_efSearch=256         0.7388 0.8595 0.8610      0.20916       79737752    2
nprobe=128,quantizer_efSearch=32         0.7599 0.8943 0.8959      0.26099      159663237    2
nprobe=128,quantizer_efSearch=64         0.7747 0.9121 0.9138      0.27179      158707021    2
nprobe=128,quantizer_efSearch=128        0.7787 0.9182 0.9199      0.28811      157982872    2
nprobe=128,quantizer_efSearch=256        0.7791 0.9194 0.9211      0.31951      157598086    1
nprobe=128,quantizer_efSearch=512        0.7792 0.9194 0.9211      0.39987      157452514    1
nprobe=256,quantizer_efSearch=128        0.8012 0.9563 0.9578      0.50583      311925158    1
nprobe=256,quantizer_efSearch=256        0.8021 0.9584 0.9599      0.54778      310684292    1
nprobe=512,quantizer_efSearch=128        0.8088 0.9704 0.9731      0.95453      613871638    1
nprobe=512,quantizer_efSearch=256        0.8151 0.9790 0.9808      0.99982      611183455    1
nprobe=512,quantizer_efSearch=512        0.8171 0.9811 0.9829      1.02322      609129316    1
```

</details>
<details><summary>`OPQ128_256,IVF1048576(IVF1024,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 9419858448

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF1048576_IVF1024_PQ128x4fs_RFlat__PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5150 0.8998 0.9496      0.06048      353264005    5
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1265 0.1649 0.1669      0.00329        1671253    92
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1278 0.1668 0.1687      0.00338        1668876    89
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1590 0.2149 0.2177      0.00348        2972993    87
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1658 0.2271 0.2302      0.00352        2960945    86
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1894 0.2566 0.2597      0.00379        3349865    80
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.2081 0.2980 0.3035      0.00413        5536799    73
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2397 0.3399 0.3447      0.00428        5949281    71
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2521 0.3625 0.3689      0.00461        5910985    66
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2647 0.3777 0.3824      0.00488        6647679    62
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2809 0.4028 0.4096      0.00501        6622825    60
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2815 0.4053 0.4122      0.00526        6614014    57
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2836 0.4221 0.4309      0.00510       11111082    59
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3204 0.4784 0.4882      0.00565       11819761    54
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3290 0.4997 0.5109      0.00594       11756983    51
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3307 0.5040 0.5155      0.00702       11742622    43
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3491 0.5312 0.5427      0.00770       13103940    39
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3743 0.5887 0.6064      0.00762       21965627    40
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3760 0.5925 0.6113      0.00837       21941535    36
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3829 0.5993 0.6155      0.00862       23401778    35
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3924 0.6398 0.6629      0.01015       42555391    30
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4019 0.6610 0.6864      0.01094       42185997    28
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4119 0.6462 0.6647      0.01124       25950269    27
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4330 0.7115 0.7387      0.01212       43457057    25
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4505 0.7386 0.7673      0.01576       45968837    20
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4506 0.7392 0.7677      0.01936       45869522    16
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4540 0.7470 0.7757      0.01894       50667145    16
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4701 0.7880 0.8238      0.01807       86585488    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4769 0.8024 0.8404      0.02034       85666580    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4832 0.8163 0.8548      0.02515       90180735    12
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4899 0.8396 0.8830      0.02804      166060178    11
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5025 0.8674 0.9134      0.03880      168696559    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5057 0.8735 0.9195      0.04442      178172454    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5139 0.8984 0.9493      0.04966      321834245    7
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5150 0.8998 0.9496      0.05752      354175729    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5178 0.9063 0.9574      0.05424      331382145    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5180 0.9075 0.9586      0.06588      349470933    5
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5225 0.9215 0.9761      0.08463      659422639    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5226 0.9251 0.9805      0.11535      650254517    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5228 0.9241 0.9798      0.11864      691612001    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5236 0.9312 0.9882      0.13235     1251705005    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5237 0.9314 0.9884      0.14363     1292001568    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5241 0.9330 0.9911      0.16109     1269179782    2
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5247 0.9357 0.9945      0.22473     2441137557    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5251 0.9365 0.9951      0.26395     2359817037    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5258 0.9370 0.9961      0.33477     4625311675    1
```

</details>
<details><summary>`OPQ128_256,IVF1048576(IVF1024,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 9419745808

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF1048576_IVF1024_PQ128x4fs_RFlat__PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4786 0.5317 0.5329      0.01946       13113754    16
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.2147 0.2319 0.2330      0.00398        2954958    76
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2525 0.2722 0.2731      0.00443        3334828    68
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2568 0.2774 0.2785      0.00504        3330696    60
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.2792 0.3046 0.3056      0.00554        5518462    55
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.3315 0.3618 0.3628      0.00590        5916379    51
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3389 0.3694 0.3704      0.00647        5908973    47
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.3390 0.3696 0.3706      0.00808        5909158    38
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3692 0.4034 0.4044      0.00727        6621079    42
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3762 0.4111 0.4121      0.00737        6608586    41
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3770 0.4120 0.4130      0.00805        6612343    38
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.3923 0.4292 0.4302      0.01104        7988481    28
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3988 0.4356 0.4366      0.01164       10724567    26
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3991 0.4361 0.4371      0.01288       10693431    24
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.4069 0.4537 0.4549      0.01488       11039817    21
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4549 0.5053 0.5065      0.01715       11758935    18
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4613 0.5141 0.5153      0.01737       11731107    18
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4786 0.5317 0.5329      0.01997       13113676    16
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4851 0.5421 0.5433      0.02088       13111661    15
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4960 0.5535 0.5547      0.02172       15719205    14
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4967 0.5538 0.5551      0.02271       15831453    14
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4994 0.5566 0.5578      0.02557       21017710    12
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.5000 0.5569 0.5582      0.02567       21045938    12
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.5318 0.6008 0.6019      0.02839       21957322    11
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.5386 0.6101 0.6113      0.02813       21925564    11
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5749 0.6492 0.6505      0.03295       23243871    10
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5894 0.6658 0.6670      0.03538       25971994    9
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.5954 0.6729 0.6741      0.04709       30825567    7
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5955 0.6734 0.6746      0.05021       41315504    6
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.6458 0.7415 0.7426      0.07291       43378321    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6459 0.7414 0.7425      0.07557       43371825    4
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6617 0.7603 0.7615      0.07397       46029318    5
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.6680 0.7687 0.7698      0.07685       51187908    4
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6682 0.7669 0.7681      0.07295       46029482    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6683 0.7668 0.7680      0.06795       45923417    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6765 0.7773 0.7784      0.07559       60485567    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6766 0.7772 0.7783      0.08715       77965430    4
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.6927 0.8034 0.8046      0.13752       83430655    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7235 0.8395 0.8408      0.13565       85847421    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7324 0.8522 0.8535      0.13925       90968896    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.7343 0.8553 0.8566      0.14085      100930962    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.7347 0.8580 0.8593      0.15831      101088713    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7349 0.8579 0.8592      0.14052      119712513    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.7351 0.8580 0.8593      0.16607      159684895    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.7574 0.8912 0.8926      0.25506      164403244    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.7610 0.8953 0.8968      0.26650      164049421    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.7712 0.9109 0.9124      0.25365      168725340    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7765 0.9170 0.9185      0.24104      178928091    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7767 0.9193 0.9208      0.27745      238759364    2
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512 0.7768 0.9193 0.9208      0.34324      322757001    1
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.7953 0.9472 0.9488      0.47998      322553701    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.7984 0.9500 0.9514      0.47262      322194621    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.8060 0.9581 0.9595      0.48187      332047881    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.8084 0.9680 0.9699      0.90388      623823404    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.8149 0.9776 0.9796      0.89360      632466210    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.8161 0.9791 0.9810      0.89001      649860669    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8168 0.9800 0.9821      0.91158      649882539    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.8205 0.9900 0.9924      1.70833      642030024    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=512 0.8207 0.9904 0.9928      1.70084      758688270    1
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fs` </summary>
Index size 7804087500

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF262144_HNSW32_PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3333 0.5084 0.5244      0.00969       17986224    31
nprobe=2,quantizer_efSearch=4            0.2428 0.3496 0.3585      0.00575        8994195    53
nprobe=4,quantizer_efSearch=4            0.2913 0.4463 0.4622      0.00625       17988032    48
nprobe=4,quantizer_efSearch=8            0.3333 0.5084 0.5244      0.00940       17986224    32
nprobe=8,quantizer_efSearch=4            0.3741 0.6012 0.6251      0.00949       35937151    32
nprobe=8,quantizer_efSearch=8            0.3822 0.6145 0.6383      0.01078       35895466    28
nprobe=16,quantizer_efSearch=4           0.4062 0.6798 0.7111      0.01345       71787364    23
nprobe=16,quantizer_efSearch=8           0.4263 0.7171 0.7500      0.01504       71623799    20
nprobe=16,quantizer_efSearch=16          0.4368 0.7326 0.7665      0.01813       71454248    17
nprobe=32,quantizer_efSearch=8           0.4509 0.7752 0.8169      0.02178      142709568    14
nprobe=32,quantizer_efSearch=16          0.4674 0.8032 0.8462      0.02518      142347695    12
nprobe=32,quantizer_efSearch=32          0.4718 0.8148 0.8579      0.03219      141972659    10
nprobe=64,quantizer_efSearch=16          0.4816 0.8414 0.8913      0.03464      283337171    9
nprobe=64,quantizer_efSearch=32          0.4909 0.8607 0.9119      0.04077      282461183    8
nprobe=128,quantizer_efSearch=32         0.5040 0.8893 0.9459      0.05479      560375540    6
nprobe=128,quantizer_efSearch=64         0.5098 0.9003 0.9577      0.06655      558818676    5
nprobe=128,quantizer_efSearch=128        0.5109 0.9026 0.9601      0.09079      557734493    4
nprobe=256,quantizer_efSearch=64         0.5124 0.9146 0.9748      0.08798     1103519910    4
nprobe=256,quantizer_efSearch=128        0.5145 0.9194 0.9802      0.11027     1101013663    3
nprobe=512,quantizer_efSearch=128        0.5155 0.9261 0.9886      0.13292     2163976517    3
nprobe=512,quantizer_efSearch=256        0.5170 0.9277 0.9905      0.18030     2160889738    2
nprobe=1024,quantizer_efSearch=256       0.5177 0.9307 0.9943      0.23394     4214148657    2
nprobe=2048,quantizer_efSearch=256       0.5181 0.9316 0.9953      0.30715     7951214326    1
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fsr` </summary>
Index size 7804040396

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF262144_HNSW32_PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6747 0.7737 0.7752      0.03192       71277826    10
nprobe=1,quantizer_efSearch=16           0.2712 0.2923 0.2931      0.00731        4477863    42
nprobe=2,quantizer_efSearch=16           0.3767 0.4108 0.4119      0.00805        8953285    38
nprobe=4,quantizer_efSearch=16           0.4875 0.5418 0.5432      0.00993       17927058    31
nprobe=4,quantizer_efSearch=32           0.4903 0.5445 0.5459      0.01346       17896954    23
nprobe=8,quantizer_efSearch=16           0.5868 0.6620 0.6635      0.01879       35791976    16
nprobe=8,quantizer_efSearch=32           0.5916 0.6670 0.6685      0.02237       35715550    14
nprobe=8,quantizer_efSearch=64           0.5928 0.6684 0.6699      0.02766       35685506    11
nprobe=16,quantizer_efSearch=16          0.6678 0.7664 0.7679      0.02819       71453210    11
nprobe=16,quantizer_efSearch=32          0.6747 0.7737 0.7752      0.03143       71277826    10
nprobe=16,quantizer_efSearch=64          0.6752 0.7748 0.7763      0.03750       71194462    8
nprobe=16,quantizer_efSearch=128         0.6753 0.7751 0.7766      0.05010       71168633    6
nprobe=32,quantizer_efSearch=16          0.7224 0.8462 0.8478      0.06212      142349700    5
nprobe=32,quantizer_efSearch=64          0.7347 0.8602 0.8618      0.07219      141735339    5
nprobe=64,quantizer_efSearch=16          0.7538 0.8917 0.8933      0.11652      283312895    3
nprobe=64,quantizer_efSearch=64          0.7748 0.9183 0.9198      0.11515      281776898    3
nprobe=64,quantizer_efSearch=128         0.7765 0.9201 0.9216      0.13177      281454110    3
nprobe=64,quantizer_efSearch=256         0.7768 0.9203 0.9218      0.14518      281362657    3
nprobe=128,quantizer_efSearch=64         0.8011 0.9581 0.9596      0.20744      558816239    2
nprobe=128,quantizer_efSearch=128        0.8018 0.9604 0.9619      0.21950      557732188    2
nprobe=128,quantizer_efSearch=256        0.8021 0.9607 0.9622      0.23943      557350396    2
nprobe=256,quantizer_efSearch=64         0.8103 0.9722 0.9734      0.38869     1103526032    1
nprobe=256,quantizer_efSearch=128        0.8140 0.9805 0.9819      0.36653     1101013387    1
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 7752907280

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1986 0.2713 0.2760      0.00374        5207273    81
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1582 0.2156 0.2199      0.00243        4682125    124
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1906 0.2623 0.2665      0.00268        5213472    113
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1989 0.2712 0.2759      0.00291        5205421    104
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2239 0.3246 0.3313      0.00302        9428040    100
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2424 0.3520 0.3588      0.00335        9754585    90
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2609 0.3775 0.3857      0.00327        9720236    92
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2654 0.3819 0.3908      0.00363        9710893    83
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2768 0.3971 0.4062      0.00394       10388700    77
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2917 0.4433 0.4580      0.00434       18433082    70
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3078 0.4684 0.4811      0.00449       18827594    67
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3196 0.4893 0.5044      0.00455       18747052    66
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3254 0.4913 0.5046      0.00480       19473434    63
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3427 0.5218 0.5373      0.00574       20708754    53
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3435 0.5246 0.5402      0.00592       20687083    51
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3572 0.5731 0.5932      0.00673       36976661    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3703 0.5946 0.6173      0.00734       36705564    41
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3774 0.6038 0.6251      0.00695       37559260    44
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3839 0.6145 0.6358      0.00777       38820120    39
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3967 0.6381 0.6622      0.00883       38539504    34
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4000 0.6592 0.6879      0.01072       73297715    28
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4002 0.6432 0.6672      0.01002       41121440    30
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4239 0.7003 0.7306      0.01161       73700916    26
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4309 0.7142 0.7446      0.01194       74837268    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4374 0.7336 0.7670      0.01223       74219305    25
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4408 0.7396 0.7729      0.01422       76748111    22
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4511 0.7694 0.8095      0.01870      145882595    17
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4562 0.7823 0.8248      0.02002      144194719    16
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4621 0.7924 0.8325      0.01954      146697870    16
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4745 0.8172 0.8606      0.02262      147442940    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4749 0.8176 0.8610      0.02578      152568500    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4880 0.8526 0.9035      0.03385      285718842    9
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4942 0.8659 0.9172      0.03391      287617565    9
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4944 0.8655 0.9167      0.03821      287360304    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4954 0.8672 0.9187      0.04039      292291813    8
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4984 0.8782 0.9330      0.04559      572242935    7
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5003 0.8806 0.9367      0.05100      563857835    7
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5110 0.9012 0.9586      0.05669      568548226    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5113 0.9016 0.9592      0.05900      578722048    6
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5114 0.9014 0.9589      0.06467      568612206    5
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5121 0.9132 0.9720      0.06808     1127488384    5
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5140 0.9183 0.9782      0.07070     1138169008    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5155 0.9200 0.9809      0.08280     1120989234    4
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5163 0.9252 0.9875      0.12415     2214415217    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5170 0.9268 0.9895      0.11465     2221623631    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.5172 0.9271 0.9898      0.13351     2240552848    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5180 0.9315 0.9947      0.18353     4333943317    2
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5183 0.9319 0.9951      0.18054     4350806253    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5184 0.9329 0.9965      0.34985     8271932123    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5187 0.9335 0.9971      0.47269    16333243297    1
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 7752874512

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3642 0.3975 0.3985      0.00380       10391818    80
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2308 0.2491 0.2503      0.00245        4868150    123
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.2378 0.2565 0.2577      0.00280        4860653    108
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2552 0.2753 0.2765      0.00291        5207595    103
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.3230 0.3517 0.3528      0.00322        9383808    94
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3265 0.3558 0.3568      0.00327        9376567    92
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3514 0.3832 0.3842      0.00342        9722118    88
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3575 0.3897 0.3906      0.00377        9711275    80
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3576 0.3898 0.3907      0.00355        9709491    85
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3731 0.4066 0.4075      0.00416       10381591    73
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4587 0.5080 0.5094      0.00537       18724236    56
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4776 0.5297 0.5312      0.00569       19389892    53
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.4803 0.5334 0.5347      0.00593       19374941    51
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4876 0.5414 0.5427      0.00692       20667802    44
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4907 0.5448 0.5461      0.00889       23264496    34
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.5713 0.6446 0.6460      0.01310       37335164    23
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.5785 0.6529 0.6541      0.01296       37279133    24
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5870 0.6621 0.6632      0.01412       38535818    22
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.5872 0.6624 0.6635      0.01446       38527383    21
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5920 0.6677 0.6688      0.01617       41084357    19
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6534 0.7513 0.7527      0.02597       73010897    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6663 0.7673 0.7687      0.02589       74120635    12
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.6675 0.7656 0.7671      0.02636       76782627    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6723 0.7738 0.7752      0.02707       76635974    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6726 0.7736 0.7750      0.03032       81686616    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.6727 0.7736 0.7750      0.03662       92165431    9
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.6728 0.7737 0.7751      0.03996      112485348    8
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.7081 0.8255 0.8269      0.06004      144019970    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.7287 0.8500 0.8514      0.05921      144889613    6
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7298 0.8550 0.8562      0.06039      147526782    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7369 0.8598 0.8612      0.06090      147103814    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7370 0.8599 0.8613      0.06142      147236879    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.7409 0.8721 0.8736      0.10860      285354695    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7644 0.9021 0.9033      0.10809      286227241    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7749 0.9159 0.9171      0.11111      288163106    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7762 0.9168 0.9181      0.11381      287375521    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7777 0.9190 0.9203      0.11366      291984700    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.7782 0.9197 0.9210      0.12924      322525981    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.8072 0.9601 0.9614      0.20829      578375945    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.8074 0.9603 0.9616      0.21063      599069072    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.8135 0.9804 0.9823      0.36892     1111183751    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.8147 0.9816 0.9834      0.36773     1140895489    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.8158 0.9889 0.9911      0.71997     2179434708    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.8164 0.9902 0.9924      0.70774     2187624875    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.8175 0.9945 0.9971      1.30092     2150718045    1
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fs` </summary>
Index size 7351150284

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF65536_HNSW32_PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5200 0.9257 0.9915      0.19370     4237481260    2
nprobe=1,quantizer_efSearch=8            0.2422 0.3404 0.3480      0.00489       17762691    62
nprobe=1,quantizer_efSearch=16           0.2474 0.3462 0.3539      0.00603       17730569    50
nprobe=2,quantizer_efSearch=4            0.2813 0.4309 0.4439      0.00654       35522778    46
nprobe=2,quantizer_efSearch=8            0.3108 0.4716 0.4850      0.00701       35513680    43
nprobe=2,quantizer_efSearch=16           0.3162 0.4788 0.4925      0.00820       35451356    37
nprobe=2,quantizer_efSearch=32           0.3169 0.4798 0.4935      0.01022       35435100    30
nprobe=4,quantizer_efSearch=4            0.3386 0.5418 0.5629      0.01041       71024557    29
nprobe=4,quantizer_efSearch=8            0.3749 0.5967 0.6188      0.01137       71073815    27
nprobe=4,quantizer_efSearch=16           0.3844 0.6096 0.6321      0.01233       70891923    25
nprobe=4,quantizer_efSearch=32           0.3850 0.6106 0.6332      0.01449       70865051    21
nprobe=8,quantizer_efSearch=4            0.4150 0.6861 0.7188      0.01608      141902426    19
nprobe=8,quantizer_efSearch=8            0.4255 0.7022 0.7353      0.01670      141715114    18
nprobe=8,quantizer_efSearch=16           0.4384 0.7222 0.7559      0.01979      141361789    16
nprobe=8,quantizer_efSearch=32           0.4410 0.7247 0.7586      0.02127      141219937    15
nprobe=16,quantizer_efSearch=4           0.4421 0.7526 0.7949      0.02296      282705758    14
nprobe=16,quantizer_efSearch=8           0.4620 0.7897 0.8334      0.02445      282174560    13
nprobe=16,quantizer_efSearch=16          0.4701 0.8030 0.8480      0.02634      281667795    12
nprobe=32,quantizer_efSearch=8           0.4787 0.8338 0.8851      0.03196      560086044    10
nprobe=32,quantizer_efSearch=16          0.4903 0.8554 0.9090      0.03968      558783190    8
nprobe=32,quantizer_efSearch=32          0.4947 0.8624 0.9162      0.04362      557734085    7
nprobe=64,quantizer_efSearch=16          0.5015 0.8869 0.9436      0.05515     1105873238    6
nprobe=64,quantizer_efSearch=32          0.5071 0.8960 0.9543      0.06920     1102757629    5
nprobe=64,quantizer_efSearch=64          0.5095 0.9002 0.9588      0.06890     1100892522    5
nprobe=128,quantizer_efSearch=64         0.5162 0.9155 0.9793      0.08510     2167503998    4
nprobe=256,quantizer_efSearch=64         0.5182 0.9221 0.9877      0.12665     4242044070    3
nprobe=256,quantizer_efSearch=128        0.5195 0.9243 0.9902      0.13346     4241489355    3
nprobe=256,quantizer_efSearch=256        0.5200 0.9257 0.9915      0.17439     4238527990    2
nprobe=512,quantizer_efSearch=128        0.5203 0.9269 0.9934      0.22977     8240536356    2
nprobe=512,quantizer_efSearch=256        0.5209 0.9282 0.9947      0.23216     8268129493    2
nprobe=1024,quantizer_efSearch=256       0.5218 0.9295 0.9961      0.30155    16000336543    2
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fsr` </summary>
Index size 7351172812

 code_size 64

 log filename: autotune.dbdeep100M.OPQ128_256_IVF65536_HNSW32_PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6543 0.7606 0.7618      0.03304      141172421    10
nprobe=1,quantizer_efSearch=16           0.3202 0.3532 0.3543      0.00566       17728867    53
nprobe=2,quantizer_efSearch=16           0.4368 0.4922 0.4931      0.00735       35448517    41
nprobe=2,quantizer_efSearch=32           0.4378 0.4932 0.4941      0.00986       35435189    31
nprobe=4,quantizer_efSearch=16           0.5534 0.6325 0.6335      0.01000       70890785    30
nprobe=4,quantizer_efSearch=32           0.5539 0.6333 0.6343      0.01278       70864538    24
nprobe=4,quantizer_efSearch=64           0.5546 0.6338 0.6348      0.01725       70854157    18
nprobe=8,quantizer_efSearch=16           0.6502 0.7564 0.7576      0.02056      141357305    15
nprobe=8,quantizer_efSearch=32           0.6522 0.7591 0.7603      0.02268      141225382    14
nprobe=8,quantizer_efSearch=64           0.6541 0.7604 0.7616      0.02673      141180901    12
nprobe=8,quantizer_efSearch=128          0.6543 0.7606 0.7618      0.03268      141172421    10
nprobe=16,quantizer_efSearch=64          0.7211 0.8554 0.8566      0.04090      281111865    8
nprobe=16,quantizer_efSearch=128         0.7215 0.8558 0.8570      0.04728      281066479    7
nprobe=32,quantizer_efSearch=64          0.7700 0.9193 0.9204      0.07876      557184324    4
nprobe=32,quantizer_efSearch=128         0.7707 0.9198 0.9209      0.08355      557023186    4
nprobe=64,quantizer_efSearch=64          0.7936 0.9596 0.9608      0.14312     1100886303    3
nprobe=64,quantizer_efSearch=128         0.7947 0.9607 0.9619      0.14237     1100172507    3
nprobe=64,quantizer_efSearch=256         0.7948 0.9609 0.9621      0.15294     1100023035    2
nprobe=128,quantizer_efSearch=64         0.8076 0.9800 0.9814      0.25372     2167520571    2
nprobe=128,quantizer_efSearch=256        0.8101 0.9825 0.9838      0.25293     2163343706    2
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7902127762

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.7134 0.9091 0.9109      0.07592      310959040    4
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.2775 0.3173 0.3181      0.00343        5139545    88
k_factor_rf=1,nprobe=4,quantizer_efSearch=8      0.3241 0.3705 0.3713      0.00420        5128110    72
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.3661 0.4201 0.4209      0.00432       10255175    70
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.3774 0.4327 0.4335      0.00461       10238087    66
k_factor_rf=1,nprobe=16,quantizer_efSearch=4     0.4019 0.4600 0.4608      0.00509       20437714    59
k_factor_rf=2,nprobe=8,quantizer_efSearch=8      0.4052 0.4748 0.4753      0.00581       10238087    52
k_factor_rf=1,nprobe=16,quantizer_efSearch=8     0.4295 0.4915 0.4923      0.00586       20370285    52
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.4689 0.5526 0.5530      0.00702       20370285    43
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.4813 0.5687 0.5692      0.00819       20296658    37
k_factor_rf=2,nprobe=32,quantizer_efSearch=8     0.5001 0.5870 0.5876      0.00860       40538223    36
k_factor_rf=2,nprobe=32,quantizer_efSearch=16    0.5254 0.6169 0.6175      0.01041       40380524    29
k_factor_rf=4,nprobe=32,quantizer_efSearch=16    0.5624 0.6773 0.6780      0.01219       40380524    25
k_factor_rf=4,nprobe=32,quantizer_efSearch=32    0.5722 0.6888 0.6895      0.01428       40195030    22
k_factor_rf=4,nprobe=64,quantizer_efSearch=16    0.5804 0.6967 0.6972      0.01539       80205754    20
k_factor_rf=8,nprobe=32,quantizer_efSearch=16    0.5861 0.7186 0.7195      0.01744       40380524    18
k_factor_rf=8,nprobe=32,quantizer_efSearch=32    0.5968 0.7308 0.7317      0.01947       40195030    16
k_factor_rf=8,nprobe=64,quantizer_efSearch=16    0.6118 0.7546 0.7557      0.02101       80205754    15
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.6307 0.7777 0.7786      0.02301       79760412    14
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.6377 0.7875 0.7885      0.02780       79360794    11
k_factor_rf=8,nprobe=128,quantizer_efSearch=64   0.6561 0.8089 0.8100      0.03379      157306469    9
k_factor_rf=16,nprobe=64,quantizer_efSearch=64   0.6602 0.8272 0.8287      0.03784       79360794    8
k_factor_rf=16,nprobe=128,quantizer_efSearch=64  0.6858 0.8627 0.8643      0.04519      157306469    7
k_factor_rf=16,nprobe=256,quantizer_efSearch=64  0.6915 0.8695 0.8711      0.05481      310959040    6
k_factor_rf=32,nprobe=128,quantizer_efSearch=64  0.7013 0.8922 0.8943      0.06403      157306469    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=128 0.7056 0.8981 0.9003      0.07230      156606339    5
k_factor_rf=32,nprobe=256,quantizer_efSearch=64  0.7134 0.9091 0.9109      0.07552      310959040    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.7190 0.9182 0.9200      0.08787      309329970    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=256 0.7220 0.9214 0.9232      0.10319      308143557    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=64  0.7232 0.9299 0.9319      0.12176      310959040    3
k_factor_rf=32,nprobe=512,quantizer_efSearch=256 0.7258 0.9284 0.9303      0.12527      606410776    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=128 0.7298 0.9392 0.9416      0.12840      309329970    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128 0.7377 0.9517 0.9536      0.15354      608854516    2
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7902141842

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7550 0.9855 0.9886      0.30099     1183950170    1
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.1615 0.1816 0.1824      0.00281        1292663    107
k_factor_rf=8,nprobe=2,quantizer_efSearch=8       0.2603 0.2947 0.2955      0.00629        2569553    48
k_factor_rf=4,nprobe=8,quantizer_efSearch=8       0.4266 0.5049 0.5062      0.00836       10239870    36
k_factor_rf=1,nprobe=16,quantizer_efSearch=64     0.5348 0.6375 0.6385      0.01271       20122210    24
k_factor_rf=4,nprobe=16,quantizer_efSearch=64     0.5402 0.6531 0.6544      0.01634       20122210    19
k_factor_rf=1,nprobe=64,quantizer_efSearch=64     0.6500 0.7975 0.7985      0.01757       79319193    18
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.6611 0.8234 0.8245      0.01907       79319193    16
k_factor_rf=2,nprobe=64,quantizer_efSearch=128    0.6634 0.8264 0.8275      0.02609       79067498    12
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6872 0.8746 0.8764      0.03737      158114322    9
k_factor_rf=2,nprobe=128,quantizer_efSearch=128   0.7000 0.8821 0.8835      0.04256      156522358    8
k_factor_rf=4,nprobe=128,quantizer_efSearch=128   0.7071 0.9006 0.9025      0.04746      156522358    7
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.7102 0.9086 0.9109      0.05301      156522358    6
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7108 0.9110 0.9136      0.06291      156522358    5
k_factor_rf=2,nprobe=256,quantizer_efSearch=128   0.7162 0.9097 0.9109      0.06714      309126895    5
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.7285 0.9426 0.9451      0.07755      309126895    4
k_factor_rf=8,nprobe=256,quantizer_efSearch=256   0.7304 0.9448 0.9474      0.09168      307949476    4
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7321 0.9491 0.9520      0.10303      307949476    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7323 0.9497 0.9527      0.12181      307949476    3
k_factor_rf=16,nprobe=512,quantizer_efSearch=128  0.7425 0.9643 0.9667      0.14672      608357248    3
k_factor_rf=8,nprobe=512,quantizer_efSearch=256   0.7466 0.9685 0.9712      0.14732      605897312    3
k_factor_rf=16,nprobe=512,quantizer_efSearch=256  0.7491 0.9740 0.9768      0.16164      605897312    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7496 0.9757 0.9786      0.18384      605897312    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.7497 0.9764 0.9794      0.21537      604023311    2
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.7502 0.9764 0.9787      0.28651     1187338190    2
k_factor_rf=16,nprobe=1024,quantizer_efSearch=512 0.7550 0.9855 0.9886      0.32972     1183950170    1
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7559 0.9875 0.9906      0.35190     1183950170    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7642441174

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                       0.4163 0.4718 0.4725      0.02393       81421187    13
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1769 0.1991 0.1997      0.00316        3305483    95
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1     0.1941 0.2192 0.2198      0.00339        2944499    89
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2784 0.3158 0.3166      0.00355        5853956    85
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3237 0.3703 0.3711      0.00409        7965231    74
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.3456 0.3905 0.3913      0.00502       21101809    60
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.3540 0.4005 0.4012      0.00512       21040230    59
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3913 0.4581 0.4585      0.00590       13048974    51
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.4032 0.4610 0.4617      0.00684       15724868    44
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4340 0.5071 0.5076      0.00814       42079603    37
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4521 0.5147 0.5153      0.00809       43113163    38
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4862 0.5727 0.5732      0.00920       25774059    33
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5258 0.6172 0.6178      0.01024       45737696    30
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5359 0.6304 0.6309      0.01505       61387005    20
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.5707 0.6854 0.6860      0.01861       82595613    17
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5815 0.7117 0.7127      0.02152       85732302    14
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.5899 0.7240 0.7248      0.02193       45670310    14
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.5975 0.7335 0.7343      0.02354       50949415    13
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6338 0.7838 0.7848      0.02680      100612179    12
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6426 0.8047 0.8062      0.03623      100747089    9
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6501 0.8136 0.8151      0.04464      199701974    7
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.6591 0.8267 0.8281      0.04839      319227775    7
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.6756 0.8485 0.8499      0.05319      332921711    6
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.6885 0.8670 0.8684      0.05310      320847097    6
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=512 0.6900 0.8681 0.8698      0.06851      321567665    5
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=512 0.6929 0.8809 0.8828      0.08242      322150156    4
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=32 0.7013 0.8933 0.8953      0.09535      167583596    4
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=64 0.7048 0.8979 0.8999      0.09287      177060234    4
k_factor_rf=64,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.7081 0.9081 0.9106      0.13174      177739079    3
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=512 0.7262 0.9288 0.9307      0.13097      769516354    3
k_factor_rf=64,nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.7304 0.9405 0.9428      0.14791      329487464    3
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512 0.7425 0.9585 0.9606      0.20600      766640451    2
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7642523094

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.2934 0.3420 0.3432      0.00756       10564795    40
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2293 0.2599 0.2606      0.00305        3309897    99
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2903 0.3321 0.3329      0.00340        5872761    89
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2910 0.3343 0.3351      0.00453        5870981    67
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8      0.3550 0.4084 0.4090      0.00503        7954332    60
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4505 0.5313 0.5320      0.00531       23159735    57
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.5087 0.6081 0.6088      0.00567       23118787    53
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.5671 0.6901 0.6908      0.00855       43079796    36
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5854 0.7203 0.7212      0.00965       45786289    32
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.6126 0.7523 0.7530      0.01380       82544862    22
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.6500 0.8119 0.8128      0.01649       84982365    19
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.6584 0.8218 0.8228      0.02141      100175610    15
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.6658 0.8409 0.8432      0.03236       89935793    10
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128    0.6672 0.8418 0.8441      0.03341      120347671    9
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128   0.6677 0.8428 0.8451      0.03654      120038270    9
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6876 0.8762 0.8786      0.04200      163044652    8
k_factor_rf=2,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.6928 0.8757 0.8768      0.04577      167363752    7
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256   0.6968 0.8859 0.8881      0.04661      238451290    7
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7050 0.9002 0.9027      0.04466      167326575    7
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.7053 0.8989 0.9011      0.04478      177154482    7
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7202 0.9265 0.9287      0.07578      319415528    4
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.7246 0.9320 0.9342      0.08840      389728445    4
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7255 0.9388 0.9416      0.09483      348797636    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=512   0.7289 0.9429 0.9457      0.09197      470161297    4
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7434 0.9648 0.9677      0.12779      645380763    3
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.7479 0.9742 0.9768      0.19659      686878500    2
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7507 0.9766 0.9795      0.22395     1225358315    2
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.7515 0.9814 0.9843      0.23702     1207066207    2
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.7547 0.9856 0.9886      0.26174     1262489102    2
k_factor_rf=32,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.7557 0.9875 0.9907      0.27702     1223442156    2
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7375302034

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.5558 0.6610 0.6618      0.01145       71238387    27
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.3163 0.3581 0.3585      0.00326       17977480    92
k_factor_rf=1,nprobe=4,quantizer_efSearch=8      0.3663 0.4134 0.4138      0.00351       17971047    86
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.4065 0.4591 0.4595      0.00401       35893218    75
k_factor_rf=1,nprobe=16,quantizer_efSearch=4     0.4186 0.4709 0.4714      0.00472       71740446    64
k_factor_rf=1,nprobe=8,quantizer_efSearch=16     0.4234 0.4775 0.4779      0.00488       35774177    62
k_factor_rf=1,nprobe=16,quantizer_efSearch=8     0.4403 0.4961 0.4965      0.00518       71598329    58
k_factor_rf=2,nprobe=8,quantizer_efSearch=8      0.4518 0.5233 0.5239      0.00567       35893218    53
k_factor_rf=2,nprobe=8,quantizer_efSearch=16     0.4698 0.5431 0.5437      0.00647       35774177    47
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.5010 0.5798 0.5803      0.00667       71598329    45
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.5088 0.5894 0.5899      0.00722       71418922    42
k_factor_rf=2,nprobe=32,quantizer_efSearch=8     0.5095 0.5876 0.5880      0.00842      142760757    36
k_factor_rf=2,nprobe=32,quantizer_efSearch=16    0.5282 0.6097 0.6101      0.00891      142315200    34
k_factor_rf=4,nprobe=16,quantizer_efSearch=8     0.5414 0.6434 0.6442      0.00963       71598329    32
k_factor_rf=4,nprobe=16,quantizer_efSearch=32    0.5558 0.6610 0.6618      0.01110       71238387    28
k_factor_rf=4,nprobe=32,quantizer_efSearch=16    0.5848 0.6973 0.6978      0.01271      142315200    24
k_factor_rf=4,nprobe=32,quantizer_efSearch=32    0.5906 0.7052 0.7057      0.01338      141954299    23
k_factor_rf=4,nprobe=32,quantizer_efSearch=64    0.5919 0.7062 0.7067      0.01640      141735436    19
k_factor_rf=8,nprobe=32,quantizer_efSearch=16    0.6190 0.7590 0.7598      0.01872      142315200    17
k_factor_rf=8,nprobe=32,quantizer_efSearch=32    0.6253 0.7679 0.7687      0.02033      141954299    15
k_factor_rf=8,nprobe=32,quantizer_efSearch=64    0.6266 0.7688 0.7696      0.02182      141735436    14
k_factor_rf=8,nprobe=64,quantizer_efSearch=16    0.6347 0.7746 0.7756      0.02337      283371039    13
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.6472 0.7906 0.7915      0.02353      282539225    13
k_factor_rf=16,nprobe=32,quantizer_efSearch=32   0.6487 0.8084 0.8096      0.03141      141954299    10
k_factor_rf=16,nprobe=32,quantizer_efSearch=64   0.6501 0.8097 0.8109      0.03265      141735436    10
k_factor_rf=16,nprobe=64,quantizer_efSearch=16   0.6638 0.8279 0.8290      0.03476      283371039    9
k_factor_rf=16,nprobe=64,quantizer_efSearch=64   0.6801 0.8491 0.8503      0.03984      281892509    8
k_factor_rf=16,nprobe=128,quantizer_efSearch=64  0.6925 0.8640 0.8652      0.04484      559057549    7
k_factor_rf=16,nprobe=128,quantizer_efSearch=128 0.6956 0.8676 0.8687      0.04692      558049091    7
k_factor_rf=32,nprobe=64,quantizer_efSearch=32   0.6976 0.8819 0.8838      0.05655      282539225    6
k_factor_rf=32,nprobe=64,quantizer_efSearch=128  0.7014 0.8872 0.8891      0.06366      281597089    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=64  0.7147 0.9070 0.9086      0.06582      559057549    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=128 0.7179 0.9104 0.9120      0.07231      558049091    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=256 0.7183 0.9106 0.9121      0.08325      557649128    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.7230 0.9139 0.9154      0.08660     1102138726    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=256 0.7240 0.9152 0.9167      0.09346     1100321958    4
k_factor_rf=64,nprobe=128,quantizer_efSearch=64  0.7302 0.9348 0.9369      0.11283      559057549    3
k_factor_rf=64,nprobe=128,quantizer_efSearch=256 0.7341 0.9390 0.9409      0.12622      557649128    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=64  0.7383 0.9438 0.9458      0.12790     1104445088    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=128 0.7421 0.9490 0.9509      0.13696     1102138726    3
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7375378578

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.7035 0.9022 0.9035      0.02601      281845781    12
k_factor_rf=1,nprobe=1,quantizer_efSearch=4      0.2136 0.2386 0.2391      0.00263        4479294    115
k_factor_rf=1,nprobe=1,quantizer_efSearch=64     0.2424 0.2710 0.2716      0.00777        4463419    39
k_factor_rf=4,nprobe=1,quantizer_efSearch=32     0.2445 0.2765 0.2769      0.00801        4465083    38
k_factor_rf=1,nprobe=2,quantizer_efSearch=64     0.3381 0.3875 0.3881      0.00822        8918816    37
k_factor_rf=1,nprobe=4,quantizer_efSearch=64     0.4315 0.5051 0.5058      0.00853       17868385    36
k_factor_rf=4,nprobe=4,quantizer_efSearch=32     0.4375 0.5191 0.5196      0.00928       17876428    33
k_factor_rf=1,nprobe=8,quantizer_efSearch=64     0.5156 0.6145 0.6153      0.00893       35680105    34
k_factor_rf=2,nprobe=8,quantizer_efSearch=64     0.5224 0.6300 0.6305      0.01009       35680105    30
k_factor_rf=4,nprobe=8,quantizer_efSearch=32     0.5249 0.6372 0.6379      0.01031       35706259    30
k_factor_rf=1,nprobe=16,quantizer_efSearch=64    0.5919 0.7116 0.7123      0.01026       71153304    30
k_factor_rf=4,nprobe=16,quantizer_efSearch=32    0.6060 0.7464 0.7473      0.01154       71228613    27
k_factor_rf=4,nprobe=16,quantizer_efSearch=64    0.6073 0.7476 0.7485      0.01398       71153304    22
k_factor_rf=2,nprobe=32,quantizer_efSearch=64    0.6536 0.8147 0.8153      0.01363      141726948    23
k_factor_rf=4,nprobe=32,quantizer_efSearch=64    0.6611 0.8323 0.8331      0.01703      141726948    18
k_factor_rf=2,nprobe=64,quantizer_efSearch=64    0.6905 0.8687 0.8694      0.01816      281845781    17
k_factor_rf=4,nprobe=64,quantizer_efSearch=64    0.7011 0.8929 0.8939      0.02108      281845781    15
k_factor_rf=4,nprobe=64,quantizer_efSearch=128   0.7032 0.8951 0.8961      0.02571      281545281    12
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.7035 0.9022 0.9035      0.02663      281845781    12
k_factor_rf=2,nprobe=128,quantizer_efSearch=64   0.7116 0.8979 0.8986      0.03639      558916976    9
k_factor_rf=4,nprobe=128,quantizer_efSearch=32   0.7185 0.9180 0.9189      0.03863      560345370    8
k_factor_rf=4,nprobe=128,quantizer_efSearch=64   0.7258 0.9284 0.9293      0.04020      558916976    8
k_factor_rf=8,nprobe=128,quantizer_efSearch=64   0.7307 0.9422 0.9436      0.04647      558916976    7
k_factor_rf=8,nprobe=128,quantizer_efSearch=128  0.7337 0.9458 0.9472      0.05116      557857816    6
k_factor_rf=16,nprobe=128,quantizer_efSearch=128 0.7345 0.9519 0.9544      0.06196      557857816    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=256 0.7350 0.9523 0.9548      0.07256      557442527    5
k_factor_rf=8,nprobe=256,quantizer_efSearch=128  0.7453 0.9635 0.9651      0.09291     1101998886    4
k_factor_rf=8,nprobe=256,quantizer_efSearch=256  0.7458 0.9649 0.9664      0.10255     1100300546    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=128 0.7479 0.9720 0.9745      0.09873     1101998886    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.7486 0.9743 0.9769      0.12549     1101998886    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=256 0.7490 0.9755 0.9782      0.11818     1100300546    3
k_factor_rf=8,nprobe=512,quantizer_efSearch=256  0.7497 0.9720 0.9735      0.15558     2163263970    2
k_factor_rf=8,nprobe=512,quantizer_efSearch=512  0.7510 0.9742 0.9757      0.16836     2161177752    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=256 0.7530 0.9813 0.9834      0.16239     2163263970    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=512 0.7543 0.9837 0.9861      0.18322     2161177752    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=512 0.7556 0.9871 0.9898      0.20983     2161177752    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7310566870

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.4630 0.5203 0.5207      0.04668      323321081    7
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.2695 0.3035 0.3040      0.00340        9393721    89
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3337 0.3763 0.3767      0.00339       18812202    89
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.3803 0.4361 0.4367      0.00479       19428303    63
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.4534 0.5245 0.5251      0.00568       37348129    53
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.4629 0.5346 0.5352      0.00609       38588165    50
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4924 0.5719 0.5724      0.00718       73037210    42
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.5065 0.5871 0.5876      0.00844       74120173    36
k_factor_rf=4,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.5505 0.6544 0.6551      0.01098       74128991    28
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5810 0.6901 0.6906      0.01330      147934465    23
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5898 0.6971 0.6976      0.01650      286755606    19
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128    0.5919 0.7061 0.7065      0.01777      162815924    17
k_factor_rf=16,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5947 0.7312 0.7323      0.02735       74135452    11
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256    0.6269 0.7686 0.7695      0.02690      183237473    12
k_factor_rf=16,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6406 0.7978 0.7990      0.03138      145124451    10
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6448 0.8047 0.8058      0.03669      285824576    9
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6657 0.8290 0.8300      0.03679      286760234    9
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.6814 0.8504 0.8515      0.04834      322822827    7
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6893 0.8596 0.8607      0.04562      566541954    7
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.6950 0.8669 0.8679      0.06175      579271773    5
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.7128 0.9024 0.9040      0.06660      566541954    5
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.7149 0.9053 0.9069      0.07270      601541695    5
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7183 0.9104 0.9119      0.07169      579515778    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.7197 0.9075 0.9087      0.09351     2183353677    4
k_factor_rf=64,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7336 0.9385 0.9406      0.12786      599972036    3
k_factor_rf=64,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7419 0.9479 0.9498      0.14083     1125295061    3
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7310514390

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.6040 0.7303 0.7310      0.01509      163696599    20
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2       0.2918 0.3353 0.3357      0.00445        9371105    68
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.3258 0.3724 0.3729      0.00425       10387097    71
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.3619 0.4253 0.4258      0.00460       18432252    66
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8       0.4221 0.4949 0.4954      0.00541       19352254    56
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8        0.4478 0.5271 0.5278      0.00543       37539288    56
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4993 0.5944 0.5951      0.00582       37346694    52
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16       0.5101 0.6086 0.6093      0.00678       38534310    45
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5717 0.6872 0.6878      0.00828       73018676    37
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.5941 0.7252 0.7258      0.00998       74197297    31
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.6015 0.7342 0.7348      0.01175       76629279    26
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.6211 0.7669 0.7675      0.01214      144605667    25
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128     0.6318 0.7684 0.7691      0.01550      162708563    20
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.6454 0.7882 0.7889      0.01775      295484010    17
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.6529 0.8195 0.8205      0.01909      144959252    16
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.6538 0.8141 0.8146      0.01767      286221529    17
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.6674 0.8175 0.8182      0.01864      287636930    17
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.6685 0.8204 0.8211      0.02264      302832898    14
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.6880 0.8632 0.8638      0.02444      302728703    13
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.6978 0.8857 0.8866      0.02498      287595977    13
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64      0.7009 0.8911 0.8920      0.02908      292365920    11
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.7034 0.9000 0.9014      0.03631      302807878    9
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.7207 0.9189 0.9198      0.04580      565833317    7
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.7303 0.9451 0.9474      0.08380      564452283    4
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=256   0.7313 0.9433 0.9448      0.07805      599395933    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128    0.7333 0.9471 0.9486      0.08630     1136523019    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.7402 0.9561 0.9576      0.08638     1109837914    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256    0.7443 0.9633 0.9648      0.10268     1142519003    3
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7448 0.9684 0.9708      0.10149     1124233616    3
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.7475 0.9717 0.9739      0.13898     1122039044    3
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256    0.7507 0.9731 0.9746      0.17411     2204183594    2
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7532 0.9813 0.9835      0.18252     2176533365    2
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7542 0.9829 0.9851      0.18324     2184270333    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256   0.7550 0.9874 0.9901      0.26111     2207378376    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.7554 0.9879 0.9907      0.28548     2204025977    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.7576 0.9913 0.9942      0.41085     4253991169    1
k_factor_rf=32,nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.7585 0.9914 0.9941      0.57543     4255954864    1
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=256 0.7587 0.9933 0.9961      0.59717     4274331378    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7243956114

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.5714 0.6802 0.6808      0.01267      281385217    24
k_factor_rf=1,nprobe=1,quantizer_efSearch=8      0.2529 0.2835 0.2843      0.00283       17745964    106
k_factor_rf=1,nprobe=2,quantizer_efSearch=4      0.3020 0.3405 0.3412      0.00317       35531821    95
k_factor_rf=1,nprobe=2,quantizer_efSearch=8      0.3271 0.3683 0.3690      0.00329       35514638    92
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.3500 0.3965 0.3972      0.00396       70976846    76
k_factor_rf=1,nprobe=4,quantizer_efSearch=8      0.3807 0.4322 0.4329      0.00417       71071093    72
k_factor_rf=1,nprobe=4,quantizer_efSearch=16     0.3911 0.4429 0.4436      0.00434       70900081    70
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.4032 0.4561 0.4568      0.00488      141886530    62
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.4098 0.4631 0.4638      0.00516      141780831    59
k_factor_rf=2,nprobe=4,quantizer_efSearch=8      0.4290 0.4962 0.4965      0.00561       71071093    54
k_factor_rf=2,nprobe=4,quantizer_efSearch=16     0.4400 0.5079 0.5082      0.00575       70900081    53
k_factor_rf=2,nprobe=4,quantizer_efSearch=32     0.4412 0.5093 0.5096      0.00630       70872802    48
k_factor_rf=2,nprobe=8,quantizer_efSearch=8      0.4686 0.5436 0.5440      0.00638      141780831    48
k_factor_rf=2,nprobe=8,quantizer_efSearch=16     0.4862 0.5629 0.5633      0.00685      141431563    44
k_factor_rf=2,nprobe=8,quantizer_efSearch=32     0.4887 0.5664 0.5668      0.00793      141296696    38
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.4971 0.5768 0.5772      0.00820      282314607    37
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.5080 0.5884 0.5888      0.00827      281732269    37
k_factor_rf=2,nprobe=16,quantizer_efSearch=32    0.5109 0.5924 0.5928      0.00946      281385217    32
k_factor_rf=4,nprobe=8,quantizer_efSearch=8      0.5162 0.6127 0.6133      0.00938      141780831    33
k_factor_rf=4,nprobe=8,quantizer_efSearch=32     0.5365 0.6359 0.6366      0.01066      141296696    29
k_factor_rf=4,nprobe=16,quantizer_efSearch=8     0.5567 0.6631 0.6637      0.01100      282314607    28
k_factor_rf=4,nprobe=16,quantizer_efSearch=32    0.5714 0.6802 0.6808      0.01203      281385217    25
k_factor_rf=4,nprobe=32,quantizer_efSearch=16    0.5809 0.6905 0.6910      0.01477      559136489    21
k_factor_rf=4,nprobe=32,quantizer_efSearch=32    0.5840 0.6953 0.6958      0.01372      558168687    22
k_factor_rf=4,nprobe=32,quantizer_efSearch=64    0.5853 0.6966 0.6971      0.01484      557601511    21
k_factor_rf=8,nprobe=16,quantizer_efSearch=32    0.6136 0.7465 0.7473      0.01917      281385217    16
k_factor_rf=8,nprobe=16,quantizer_efSearch=64    0.6145 0.7476 0.7484      0.02120      281239494    15
k_factor_rf=8,nprobe=32,quantizer_efSearch=16    0.6304 0.7695 0.7703      0.02119      559136489    15
k_factor_rf=8,nprobe=32,quantizer_efSearch=32    0.6347 0.7758 0.7766      0.02334      558168687    13
k_factor_rf=8,nprobe=32,quantizer_efSearch=64    0.6362 0.7771 0.7779      0.02558      557601511    12
k_factor_rf=8,nprobe=32,quantizer_efSearch=128   0.6364 0.7774 0.7782      0.02796      557429444    11
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.6402 0.7811 0.7819      0.02570     1104168846    12
k_factor_rf=16,nprobe=16,quantizer_efSearch=64   0.6403 0.7939 0.7952      0.03351      281239494    9
k_factor_rf=16,nprobe=32,quantizer_efSearch=32   0.6707 0.8361 0.8375      0.03649      558168687    9
k_factor_rf=16,nprobe=32,quantizer_efSearch=64   0.6721 0.8375 0.8389      0.03642      557601511    9
k_factor_rf=16,nprobe=64,quantizer_efSearch=16   0.6737 0.8389 0.8402      0.04250     1106935303    8
k_factor_rf=16,nprobe=64,quantizer_efSearch=32   0.6799 0.8479 0.8492      0.04294     1104168846    8
k_factor_rf=32,nprobe=32,quantizer_efSearch=16   0.6864 0.8677 0.8696      0.05943      559136489    6
k_factor_rf=32,nprobe=32,quantizer_efSearch=64   0.6919 0.8754 0.8773      0.06129      557601511    5
k_factor_rf=32,nprobe=64,quantizer_efSearch=16   0.7007 0.8880 0.8895      0.06385     1106935303    5
k_factor_rf=32,nprobe=64,quantizer_efSearch=32   0.7075 0.8975 0.8991      0.06940     1104168846    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=64  0.7147 0.9087 0.9103      0.07075     2170887238    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=128 0.7159 0.9102 0.9119      0.07527     2167887047    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=256 0.7188 0.9105 0.9121      0.09413     4247474273    4
k_factor_rf=64,nprobe=64,quantizer_efSearch=128  0.7228 0.9289 0.9310      0.11988     1101782419    3
k_factor_rf=64,nprobe=128,quantizer_efSearch=64  0.7321 0.9427 0.9447      0.13415     2170887238    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=64  0.7344 0.9449 0.9470      0.14278     4251605737    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=128 0.7366 0.9473 0.9493      0.15459     4250386363    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7243983506

 code_size 64

 log filename: autotune.dbdeep100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7611 0.9866 0.9890      0.27932    16030936112    2
k_factor_rf=1,nprobe=1,quantizer_efSearch=64      0.2919 0.3289 0.3294      0.00517       17671591    59
k_factor_rf=1,nprobe=2,quantizer_efSearch=64      0.3955 0.4551 0.4557      0.00527       35412480    57
k_factor_rf=1,nprobe=2,quantizer_efSearch=128     0.3956 0.4552 0.4558      0.00824       35410760    37
k_factor_rf=2,nprobe=8,quantizer_efSearch=64      0.5876 0.7103 0.7108      0.00826      141259176    37
k_factor_rf=4,nprobe=8,quantizer_efSearch=64      0.5948 0.7263 0.7268      0.01080      141259176    28
k_factor_rf=1,nprobe=32,quantizer_efSearch=128    0.6580 0.7992 0.7996      0.01319      557514520    23
k_factor_rf=2,nprobe=32,quantizer_efSearch=128    0.6858 0.8502 0.8508      0.01489      557514520    21
k_factor_rf=4,nprobe=32,quantizer_efSearch=128    0.6988 0.8804 0.8814      0.01898      557514520    16
k_factor_rf=2,nprobe=64,quantizer_efSearch=128    0.7102 0.8822 0.8829      0.02063     1101629734    15
k_factor_rf=2,nprobe=64,quantizer_efSearch=256    0.7103 0.8823 0.8830      0.02675     1101485404    12
k_factor_rf=4,nprobe=64,quantizer_efSearch=256    0.7243 0.9186 0.9196      0.02988     1101485404    11
k_factor_rf=8,nprobe=64,quantizer_efSearch=256    0.7334 0.9400 0.9416      0.03621     1101485404    9
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.7348 0.9473 0.9495      0.03835     1102278956    8
k_factor_rf=16,nprobe=64,quantizer_efSearch=128   0.7352 0.9476 0.9498      0.04164     1101629734    8
k_factor_rf=4,nprobe=128,quantizer_efSearch=128   0.7369 0.9365 0.9374      0.04317     2167404980    7
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.7479 0.9617 0.9631      0.04899     2167404980    7
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7512 0.9715 0.9737      0.06205     2167404980    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=256  0.7513 0.9716 0.9738      0.06847     2166556103    5
k_factor_rf=8,nprobe=256,quantizer_efSearch=256   0.7540 0.9707 0.9724      0.08475     4245541768    4
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7555 0.9778 0.9801      0.08996     4248467127    4
k_factor_rf=16,nprobe=256,quantizer_efSearch=512  0.7569 0.9811 0.9834      0.12603     4244521973    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7596 0.9872 0.9899      0.11796     4245541768    3
k_factor_rf=16,nprobe=512,quantizer_efSearch=512  0.7600 0.9850 0.9874      0.17037     8274065875    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7624 0.9914 0.9943      0.18972     8282060576    2
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7631 0.9928 0.9956      0.30998    16030936112    1
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7643 0.9952 0.9981      0.37098    16030936112    1
```

</details>
<details><summary>`OPQ56_112,IVF1048576_HNSW32,PQ7+56` </summary>
Index size 7863785180

 code_size 63

 log filename: autotune.dbdeep100M.OPQ56_112_IVF1048576_HNSW32_PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.6830 0.8924 0.8940      0.12401              0    3
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.1900 0.2130 0.2136      0.00641              0    47
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.2980 0.3459 0.3464      0.00638              0    47
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.3535 0.4102 0.4108      0.00819              0    37
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.4178 0.4932 0.4938      0.00846              0    36
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=2       0.4360 0.5179 0.5185      0.01194              0    26
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.4738 0.5693 0.5699      0.01174              0    26
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.5051 0.6085 0.6091      0.01265              0    24
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.5198 0.6254 0.6260      0.01410              0    22
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.5308 0.6479 0.6486      0.01714              0    18
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=1      0.5466 0.6650 0.6657      0.01953              0    16
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=1     0.5842 0.7110 0.7116      0.02375              0    13
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.5923 0.7342 0.7349      0.02347              0    13
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.6044 0.7495 0.7502      0.02651              0    12
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=4     0.6104 0.7638 0.7646      0.03313              0    10
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=2     0.6219 0.7821 0.7830      0.03696              0    9
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.6469 0.8153 0.8162      0.04389              0    7
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=2    0.6500 0.8183 0.8192      0.05403              0    6
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.6618 0.8506 0.8519      0.06318              0    5
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=8    0.6649 0.8534 0.8546      0.07155              0    5
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.6705 0.8529 0.8538      0.08570              0    4
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=4    0.6735 0.8700 0.8711      0.09026              0    4
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=4    0.6852 0.8881 0.8891      0.09212              0    4
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.6923 0.9043 0.9057      0.10469              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.6953 0.9100 0.9114      0.11576              0    3
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=4    0.6983 0.9108 0.9119      0.12680              0    3
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=8    0.7072 0.9325 0.9339      0.13374              0    3
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=8   0.7133 0.9425 0.9440      0.15195              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.7187 0.9554 0.9571      0.18838              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.7196 0.9576 0.9595      0.23810              0    2
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=16  0.7259 0.9691 0.9710      0.27015              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.7292 0.9745 0.9765      0.28161              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.7307 0.9765 0.9785      0.32116              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.7318 0.9803 0.9823      0.38555              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=64  0.7321 0.9808 0.9828      0.49947              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=16 0.7349 0.9845 0.9864      0.52057              0    1
nprobe=2048,quantizer_efSearch=256,ht=56,k_factor=32 0.7355 0.9889 0.9912      0.88127              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=32 0.7371 0.9921 0.9943      0.95211              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=32 0.7374 0.9924 0.9947      1.67161              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=64 0.7378 0.9943 0.9969      1.79929              0    1
```

</details>
<details><summary>`OPQ56_112,IVF1048576(IVF1024,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 7617116064

 code_size 63

 log filename: autotune.dbdeep100M.OPQ56_112_IVF1048576_IVF1024_PQ56x4fs_RFlat__PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.1759 0.1957 0.1960      0.00721        1458609    42
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=56,k_factor=1        0.1747 0.1942 0.1949      0.00543         749078    56
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=56,k_factor=1       0.2432 0.2762 0.2769      0.00613         749003    49
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=56,k_factor=1        0.2549 0.2961 0.2968      0.00657         378036    46
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=56,k_factor=2        0.2679 0.3041 0.3047      0.00802        1457366    38
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=56,k_factor=1       0.2760 0.3129 0.3136      0.00872       11054690    35
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=56,k_factor=1        0.3013 0.3551 0.3558      0.00953         378562    32
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=1       0.3179 0.3670 0.3677      0.00950       11074374    32
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1       0.4327 0.5139 0.5146      0.01024        5662927    30
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=2       0.4721 0.5714 0.5721      0.01660        2884782    19
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=56,k_factor=1     0.5263 0.6404 0.6411      0.01903       10966516    16
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.5304 0.6520 0.6526      0.01832        5652732    17
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=2       0.5381 0.6687 0.6692      0.02527        2884782    12
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=1      0.5741 0.7090 0.7097      0.02603        5648790    12
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=4       0.5800 0.7342 0.7349      0.03302        2875746    10
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=4      0.5997 0.7609 0.7615      0.03453        5662927    9
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=4      0.6061 0.7689 0.7695      0.03697       21174480    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=1      0.6112 0.7639 0.7646      0.04583       10773218    7
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=4      0.6243 0.7996 0.8003      0.05371       21607261    6
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=56,k_factor=2     0.6399 0.8193 0.8198      0.05910       21380334    6
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=4     0.6498 0.8429 0.8435      0.06905       80816031    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=56,k_factor=4     0.6518 0.8464 0.8471      0.08213        5662927    4
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=256,ht=56,k_factor=8    0.6530 0.8538 0.8548      0.09850       82626119    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56,k_factor=4   0.6816 0.8956 0.8964      0.09806       42039508    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=56,k_factor=4   0.6817 0.8957 0.8965      0.11806      165401505    3
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=256,ht=56,k_factor=8   0.6866 0.9120 0.9136      0.14243       83091029    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=4    0.6962 0.9216 0.9225      0.13433       83942274    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=16    0.7063 0.9531 0.9552      0.16868       21715162    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=32   0.7065 0.9558 0.9584      0.24652      165401505    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56,k_factor=32   0.7071 0.9569 0.9596      0.22934       42705636    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=16   0.7162 0.9746 0.9768      0.36790      165401505    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56,k_factor=64  0.7170 0.9792 0.9820      0.56709       42705636    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=64   0.7175 0.9794 0.9822      0.52055       83942274    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=16  0.7201 0.9826 0.9851      0.59945       83942274    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=32  0.7219 0.9883 0.9915      0.72907       83942274    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=64  0.7220 0.9894 0.9927      0.80983       83942274    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=64  0.7232 0.9934 0.9968      1.29843       83942274    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=512,ht=56,k_factor=64  0.7236 0.9943 0.9978      2.28832      165401505    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=56,k_factor=64 0.7242 0.9949 0.9984      5.24931      165401505    1
```

</details>
<details><summary>`OPQ56_112,IVF262144_HNSW32,PQ7+56` </summary>
Index size 7291153372

 code_size 63

 log filename: autotune.dbdeep100M.OPQ56_112_IVF262144_HNSW32_PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.7389 0.9413 0.9420      0.08090              0    4
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.2535 0.2798 0.2804      0.00596              0    51
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.4352 0.4951 0.4957      0.00663              0    46
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.4965 0.5730 0.5736      0.00705              0    43
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.5428 0.6333 0.6339      0.00891              0    34
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.5693 0.6659 0.6665      0.00921              0    33
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.5804 0.6800 0.6806      0.01046              0    29
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=1      0.5965 0.7028 0.7034      0.01314              0    23
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.6079 0.7273 0.7278      0.01380              0    22
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=1     0.6212 0.7350 0.7356      0.01584              0    19
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.6505 0.7881 0.7886      0.01700              0    18
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.6583 0.7980 0.7985      0.01945              0    16
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=4     0.6668 0.8203 0.8206      0.02366              0    13
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=2     0.6678 0.8126 0.8132      0.02347              0    13
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=4     0.6754 0.8310 0.8313      0.02508              0    12
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.6846 0.8349 0.8355      0.02889              0    11
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=2    0.6857 0.8366 0.8372      0.03698              0    9
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=8     0.6858 0.8519 0.8527      0.04150              0    8
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.6980 0.8529 0.8535      0.04141              0    8
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=8     0.6983 0.8777 0.8787      0.04205              0    8
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=4    0.7191 0.8937 0.8941      0.04507              0    7
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=4    0.7258 0.9031 0.9035      0.04777              0    7
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.7273 0.9053 0.9057      0.05623              0    6
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=8    0.7310 0.9250 0.9260      0.05731              0    6
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.7388 0.9359 0.9368      0.05990              0    6
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.7403 0.9382 0.9391      0.06923              0    5
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=16   0.7471 0.9528 0.9537      0.09016              0    4
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=16  0.7484 0.9550 0.9559      0.09627              0    4
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=16  0.7485 0.9553 0.9562      0.10653              0    3
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.7579 0.9725 0.9737      0.13258              0    3
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=32  0.7601 0.9788 0.9800      0.17854              0    2
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=16  0.7612 0.9774 0.9786      0.18585              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.7623 0.9793 0.9805      0.19731              0    2
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=32  0.7646 0.9863 0.9876      0.22899              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.7656 0.9883 0.9895      0.24537              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=64  0.7665 0.9911 0.9924      0.35796              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=64 0.7685 0.9947 0.9961      0.47681              0    1
nprobe=2048,quantizer_efSearch=256,ht=56,k_factor=64 0.7687 0.9952 0.9967      0.64197              0    1
nprobe=4096,quantizer_efSearch=256,ht=56,k_factor=64 0.7689 0.9953 0.9968      0.87677              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=64 0.7690 0.9961 0.9976      1.06100              0    1
```

</details>
<details><summary>`OPQ56_112,IVF262144(IVF512,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 7229713824

 code_size 63

 log filename: autotune.dbdeep100M.OPQ56_112_IVF262144_IVF512_PQ56x4fs_RFlat__PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.7383 0.9383 0.9392      0.13627       21056748    3
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1       0.2510 0.2784 0.2789      0.00572        2774522    53
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=56,k_factor=1       0.3828 0.4338 0.4343      0.00601         366527    50
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=56,k_factor=1        0.4181 0.4782 0.4787      0.00569         722136    53
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=1        0.4305 0.4904 0.4909      0.00580        1421220    52
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1       0.5163 0.5981 0.5986      0.00722        2773857    42
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1      0.5788 0.6829 0.6834      0.00944        2772536    32
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=2      0.6076 0.7260 0.7264      0.01315        5423975    23
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=2      0.6125 0.7327 0.7331      0.01515       10695553    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=2       0.6334 0.7647 0.7651      0.01591        1424661    19
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=2     0.6552 0.7924 0.7928      0.02095       21056748    15
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=2     0.6584 0.7977 0.7981      0.02194       21056748    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=4      0.6645 0.8171 0.8174      0.02248        2777876    14
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=4      0.6967 0.8640 0.8644      0.03034        2777876    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=4      0.7060 0.8754 0.8758      0.03214       10695553    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=8      0.7079 0.8886 0.8891      0.04240        2777876    8
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56,k_factor=4    0.7093 0.8799 0.8803      0.04306       21056748    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=4     0.7239 0.9023 0.9026      0.05157        5394800    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=8     0.7253 0.9157 0.9162      0.05537        2777876    6
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=56,k_factor=8    0.7277 0.9188 0.9193      0.07037        2777876    5
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=56,k_factor=4    0.7283 0.9093 0.9096      0.08274       41465620    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=4    0.7321 0.9157 0.9160      0.08617       21056748    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=16   0.7479 0.9512 0.9520      0.08913       41465620    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=8    0.7513 0.9539 0.9545      0.09671       41465620    4
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=16    0.7540 0.9621 0.9629      0.11132       10695553    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=16    0.7594 0.9720 0.9730      0.11372       10695553    3
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=56,k_factor=16  0.7602 0.9741 0.9751      0.15512       41465620    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=32   0.7624 0.9800 0.9811      0.17067       41465620    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=32   0.7673 0.9889 0.9900      0.23916       41465620    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=64   0.7680 0.9917 0.9930      0.35767       41465620    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32   0.7682 0.9898 0.9910      0.56860       10695553    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=32  0.7691 0.9922 0.9934      0.59419       21056748    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=64  0.7702 0.9958 0.9971      0.79421       21056748    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=64  0.7704 0.9960 0.9973      1.11860       21056748    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=56,k_factor=64 0.7705 0.9964 0.9977      2.68431       41465620    1
```

</details>
<details><summary>`OPQ56_112,IVF65536_HNSW32,PQ7+56` </summary>
Index size 7147994588

 code_size 63

 log filename: autotune.dbdeep100M.OPQ56_112_IVF65536_HNSW32_PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.7957 0.9541 0.9547      0.12329              0    3
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.3051 0.3315 0.3320      0.00554              0    55
nprobe=2,quantizer_efSearch=8,ht=56,k_factor=1       0.4074 0.4488 0.4493      0.00635              0    48
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.4531 0.5058 0.5063      0.00732              0    41
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.4985 0.5553 0.5558      0.00763              0    40
nprobe=4,quantizer_efSearch=16,ht=56,k_factor=1      0.5109 0.5688 0.5693      0.00792              0    38
nprobe=4,quantizer_efSearch=32,ht=56,k_factor=1      0.5115 0.5696 0.5701      0.00865              0    35
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.5508 0.6186 0.6191      0.00970              0    31
nprobe=8,quantizer_efSearch=64,ht=56,k_factor=1      0.5822 0.6545 0.6550      0.01213              0    25
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=2       0.5958 0.6793 0.6797      0.01391              0    22
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.6069 0.6863 0.6867      0.01366              0    22
nprobe=8,quantizer_efSearch=16,ht=56,k_factor=2      0.6130 0.6984 0.6988      0.01344              0    23
nprobe=8,quantizer_efSearch=32,ht=56,k_factor=2      0.6155 0.7016 0.7020      0.01509              0    20
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.6170 0.6976 0.6980      0.01409              0    22
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.6204 0.7020 0.7024      0.01488              0    21
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=2      0.6227 0.7141 0.7146      0.01735              0    18
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.6624 0.7615 0.7620      0.01794              0    17
nprobe=16,quantizer_efSearch=128,ht=56,k_factor=2    0.6672 0.7677 0.7682      0.02336              0    13
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=4      0.6808 0.7919 0.7924      0.02438              0    13
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.6864 0.7937 0.7941      0.02579              0    12
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=4     0.6933 0.8067 0.8072      0.02420              0    13
nprobe=16,quantizer_efSearch=64,ht=56,k_factor=4     0.6983 0.8130 0.8135      0.02788              0    11
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=4      0.7061 0.8267 0.8271      0.03266              0    10
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=4     0.7244 0.8500 0.8504      0.03385              0    9
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=4     0.7294 0.8565 0.8569      0.03406              0    9
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=4     0.7307 0.8579 0.8583      0.03504              0    9
nprobe=32,quantizer_efSearch=256,ht=56,k_factor=4    0.7312 0.8584 0.8588      0.04713              0    7
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=8     0.7510 0.8892 0.8897      0.04744              0    7
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=8     0.7526 0.8910 0.8915      0.04979              0    7
nprobe=32,quantizer_efSearch=256,ht=56,k_factor=8    0.7532 0.8917 0.8922      0.06054              0    5
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=8     0.7655 0.9085 0.9089      0.06297              0    5
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.7752 0.9209 0.9213      0.06665              0    5
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=8    0.7755 0.9218 0.9222      0.06864              0    5
nprobe=64,quantizer_efSearch=256,ht=56,k_factor=8    0.7758 0.9220 0.9224      0.07553              0    4
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=16    0.7791 0.9311 0.9316      0.08965              0    4
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=16    0.7894 0.9448 0.9453      0.09137              0    4
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=16   0.7898 0.9458 0.9463      0.09332              0    4
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=16   0.7957 0.9541 0.9547      0.11861              0    3
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=16   0.8005 0.9604 0.9610      0.12361              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=16  0.8018 0.9622 0.9628      0.12552              0    3
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=16  0.8022 0.9627 0.9633      0.13263              0    3
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.8058 0.9683 0.9689      0.19761              0    2
nprobe=256,quantizer_efSearch=32,ht=56,k_factor=32   0.8060 0.9716 0.9721      0.24223              0    2
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=32  0.8135 0.9827 0.9832      0.24953              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.8140 0.9836 0.9841      0.24958              0    2
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=64  0.8180 0.9902 0.9906      0.35841              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=64  0.8188 0.9914 0.9918      0.35254              0    1
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=64  0.8197 0.9928 0.9932      0.48962              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=64  0.8202 0.9940 0.9944      0.49895              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=64  0.8204 0.9940 0.9944      0.48515              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=64 0.8211 0.9951 0.9955      0.73890              0    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64` </summary>
Index size 8030801840

 code_size 64

 log filename: autotune.dbdeep100M.OPQ64_128_IVF1048576_HNSW32_PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.2894 0.3055 0.3061      0.00734        2595320    41
nprobe=1,quantizer_efSearch=4,ht=106      0.0041 0.0052 0.0058      0.00382        1298168    79
nprobe=1,quantizer_efSearch=4,ht=256      0.1767 0.1859 0.1864      0.00408        1298168    74
nprobe=2,quantizer_efSearch=4,ht=230      0.2075 0.2165 0.2171      0.00566        2595363    53
nprobe=2,quantizer_efSearch=8,ht=252      0.2894 0.3055 0.3061      0.00722        2595320    42
nprobe=2,quantizer_efSearch=8,ht=254      0.2895 0.3055 0.3062      0.00884        2595320    34
nprobe=4,quantizer_efSearch=4,ht=238      0.3060 0.3234 0.3240      0.00924        5190275    33
nprobe=4,quantizer_efSearch=32,ht=228     0.3154 0.3305 0.3313      0.01667        5127307    18
nprobe=4,quantizer_efSearch=32,ht=244     0.4015 0.4252 0.4259      0.01646        5127307    19
nprobe=8,quantizer_efSearch=32,ht=244     0.5068 0.5414 0.5421      0.02337       10238812    13
nprobe=16,quantizer_efSearch=16,ht=256    0.6041 0.6565 0.6573      0.03582       20528773    9
nprobe=16,quantizer_efSearch=32,ht=246    0.6082 0.6592 0.6600      0.03840       20406854    8
nprobe=32,quantizer_efSearch=128,ht=512   0.7106 0.7799 0.7806      0.04853       40297240    7
nprobe=64,quantizer_efSearch=512,ht=256   0.7736 0.8579 0.8587      0.20763       79697171    2
nprobe=128,quantizer_efSearch=64,ht=246   0.8029 0.8955 0.8963      0.25311      158709212    2
nprobe=512,quantizer_efSearch=512,ht=512  0.8688 0.9828 0.9835      0.44283      609129314    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8786 0.9974 0.9982      2.71780     4365739463    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64` </summary>
Index size 7788457460

 code_size 64

 log filename: autotune.dbdeep100M.OPQ64_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.8523 0.9640 0.9647      0.93390      623499110    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=2         0.0002 0.0006 0.0011      0.00327        1679830    92
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=78        0.0028 0.0038 0.0045      0.00342        2052331    88
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=234       0.1695 0.1781 0.1788      0.00383        2758109    79
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=252      0.2044 0.2163 0.2169      0.00535        6944611    57
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=250       0.2806 0.2964 0.2971      0.00636        5464090    48
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=252       0.2967 0.3139 0.3146      0.00670        5453429    45
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=236     0.3690 0.3897 0.3903      0.01294       10753444    24
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=244     0.3997 0.4249 0.4256      0.01435       16122493    21
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=256,ht=248    0.4053 0.4316 0.4323      0.03301       88005000    10
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=224    0.4208 0.4444 0.4451      0.04097       31293920    8
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=248   0.6125 0.6641 0.6647      0.05985       62760615    6
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=512    0.7069 0.7778 0.7784      0.05968       61970545    6
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=512    0.7873 0.8799 0.8805      0.09814      170429963    4
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=256    0.8305 0.9365 0.9372      0.49527      335728740    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=252    0.8523 0.9640 0.9647      0.94240      623499110    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=252   0.8609 0.9759 0.9766      0.94942      693175960    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=250  0.8678 0.9864 0.9872      3.63666     2436018195    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=512 0.8766 0.9989 0.9996      5.87138     4649152398    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64` </summary>
Index size 7407838384

 code_size 64

 log filename: autotune.dbdeep100M.OPQ64_128_IVF262144_HNSW32_PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3749 0.3986 0.3991      0.00762        8982971    40
nprobe=1,quantizer_efSearch=4,ht=204      0.0890 0.0944 0.0949      0.00347        4486421    87
nprobe=1,quantizer_efSearch=4,ht=252      0.2420 0.2561 0.2565      0.00437        4486421    69
nprobe=1,quantizer_efSearch=4,ht=256      0.2424 0.2566 0.2571      0.00441        4486421    69
nprobe=1,quantizer_efSearch=16,ht=234     0.2486 0.2623 0.2626      0.00643        4477095    47
nprobe=2,quantizer_efSearch=8,ht=252      0.3749 0.3986 0.3991      0.00762        8982971    40
nprobe=2,quantizer_efSearch=8,ht=254      0.3755 0.3991 0.3997      0.00772        8982971    39
nprobe=4,quantizer_efSearch=4,ht=238      0.4102 0.4391 0.4397      0.00973       17991037    31
nprobe=4,quantizer_efSearch=32,ht=228     0.4109 0.4384 0.4390      0.01514       17896466    20
nprobe=4,quantizer_efSearch=32,ht=244     0.4961 0.5342 0.5347      0.01612       17896466    19
nprobe=8,quantizer_efSearch=32,ht=244     0.6003 0.6554 0.6559      0.02466       35715390    13
nprobe=8,quantizer_efSearch=128,ht=256    0.6108 0.6685 0.6691      0.04166       35675702    8
nprobe=16,quantizer_efSearch=32,ht=246    0.6891 0.7647 0.7653      0.04161       71274410    8
nprobe=16,quantizer_efSearch=16,ht=256    0.6898 0.7666 0.7672      0.04422       71450145    7
nprobe=32,quantizer_efSearch=128,ht=512   0.7664 0.8620 0.8626      0.06362      141642603    5
nprobe=64,quantizer_efSearch=512,ht=256   0.8120 0.9209 0.9215      0.22103      281343198    2
nprobe=128,quantizer_efSearch=64,ht=246   0.8316 0.9466 0.9472      0.27901      558816274    2
nprobe=256,quantizer_efSearch=32,ht=512   0.8414 0.9619 0.9625      0.32872     1101335862    1
nprobe=512,quantizer_efSearch=512,ht=512  0.8630 0.9926 0.9932      0.69189     2158558373    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8673 0.9985 0.9991      4.53476    14975078950    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64` </summary>
Index size 7347510772

 code_size 64

 log filename: autotune.dbdeep100M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.3732 0.3957 0.3963      0.00800       11723215    38
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=138       0.0088 0.0101 0.0108      0.00292        4691973    103
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=230       0.1836 0.1916 0.1922      0.00326        4683189    93
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=230       0.2244 0.2335 0.2341      0.00496        5915410    61
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=246      0.2597 0.2734 0.2738      0.00618        7264643    49
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=234      0.2620 0.2764 0.2769      0.00564        9201115    54
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=240     0.3732 0.3957 0.3963      0.00856       11720671    36
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256      0.3872 0.4116 0.4120      0.01204       19629609    25
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=250     0.5044 0.5425 0.5430      0.01399       23310491    22
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=232       0.5301 0.5687 0.5693      0.01816       37356281    17
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=246      0.6009 0.6539 0.6544      0.02144       38534795    14
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=512      0.6349 0.6997 0.7003      0.02479       72746338    13
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=256    0.6990 0.7730 0.7736      0.04823       81898668    7
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=242    0.7499 0.8334 0.8339      0.07420      163085309    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=238     0.7669 0.8552 0.8557      0.12675      286115772    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256     0.8141 0.9188 0.9194      0.16315      292356170    2
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=246   0.8211 0.9276 0.9281      0.28439      563590084    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=240    0.8377 0.9460 0.9466      0.51416     1111272093    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=250    0.8574 0.9767 0.9773      0.54826     1111329460    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=512    0.8596 0.9809 0.9815      0.64296     2217106787    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=244 0.8606 0.9798 0.9804      4.96604     8271291499    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=512 0.8727 0.9992 0.9998      6.31696    15984602982    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64` </summary>
Index size 7252096688

 code_size 64

 log filename: autotune.dbdeep100M.OPQ64_128_IVF65536_HNSW32_PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.4415 0.4831 0.4838      0.01025       35502471    30
nprobe=1,quantizer_efSearch=4,ht=106      0.0069 0.0088 0.0095      0.00313       17788524    96
nprobe=1,quantizer_efSearch=4,ht=204      0.1254 0.1337 0.1345      0.00325       17788524    93
nprobe=1,quantizer_efSearch=4,ht=252      0.2973 0.3206 0.3213      0.00577       17788524    53
nprobe=1,quantizer_efSearch=4,ht=256      0.2976 0.3208 0.3214      0.00635       17788524    48
nprobe=1,quantizer_efSearch=16,ht=234     0.3104 0.3333 0.3338      0.00504       17731309    60
nprobe=2,quantizer_efSearch=4,ht=230      0.3718 0.4028 0.4034      0.00681       35510995    45
nprobe=2,quantizer_efSearch=16,ht=226     0.3891 0.4189 0.4194      0.00743       35450665    41
nprobe=2,quantizer_efSearch=16,ht=234     0.4268 0.4639 0.4644      0.00779       35450665    39
nprobe=2,quantizer_efSearch=32,ht=242     0.4447 0.4860 0.4865      0.01008       35434279    31
nprobe=4,quantizer_efSearch=4,ht=238      0.4977 0.5493 0.5498      0.01369       70998574    22
nprobe=4,quantizer_efSearch=32,ht=228     0.5104 0.5559 0.5564      0.01380       70864400    22
nprobe=4,quantizer_efSearch=32,ht=244     0.5675 0.6276 0.6281      0.01700       70864400    18
nprobe=8,quantizer_efSearch=32,ht=244     0.6737 0.7529 0.7533      0.03082      141221320    10
nprobe=8,quantizer_efSearch=128,ht=256    0.6804 0.7609 0.7614      0.04448      141171767    7
nprobe=16,quantizer_efSearch=64,ht=232    0.7050 0.7912 0.7916      0.04869      281110544    7
nprobe=16,quantizer_efSearch=128,ht=242   0.7440 0.8443 0.8448      0.05955      281065254    6
nprobe=16,quantizer_efSearch=32,ht=246    0.7490 0.8504 0.8511      0.05849      281290309    6
nprobe=16,quantizer_efSearch=256,ht=246   0.7499 0.8515 0.8522      0.07307      281058292    5
nprobe=32,quantizer_efSearch=32,ht=246    0.7965 0.9130 0.9137      0.11202      557729570    3
nprobe=32,quantizer_efSearch=128,ht=512   0.8021 0.9210 0.9216      0.12854      557022230    3
nprobe=64,quantizer_efSearch=512,ht=256   0.8328 0.9617 0.9623      0.30298     1100001264    1
nprobe=128,quantizer_efSearch=64,ht=246   0.8429 0.9760 0.9767      0.40496     2167507635    1
nprobe=256,quantizer_efSearch=32,ht=512   0.8452 0.9805 0.9811      0.89328     4232856442    1
nprobe=512,quantizer_efSearch=512,ht=248  0.8543 0.9932 0.9939      1.60115     8259924182    1
nprobe=512,quantizer_efSearch=512,ht=512  0.8564 0.9970 0.9976      1.75747     8259924182    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8579 0.9992 0.9998     11.42717    54066211574    1
```

</details>
<details><summary>`PCAR32,IVF1048576_HNSW32,SQfp16` </summary>
Index size 7628018517

 code_size 64

 log filename: autotune.dbdeep100M.PCAR32_IVF1048576_HNSW32_SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2150 0.5241 0.8245      0.09122      293270555    4
nprobe=1,quantizer_efSearch=4            0.0880 0.1440 0.1498      0.00256        1240601    118
nprobe=2,quantizer_efSearch=4            0.1128 0.2009 0.2157      0.00274        2456019    110
nprobe=4,quantizer_efSearch=4            0.1361 0.2606 0.2943      0.00279        4880987    108
nprobe=8,quantizer_efSearch=4            0.1653 0.3504 0.4236      0.00339        9745229    89
nprobe=8,quantizer_efSearch=8            0.1700 0.3603 0.4355      0.00353        9735559    85
nprobe=16,quantizer_efSearch=4           0.1756 0.3943 0.5072      0.00394       19394511    77
nprobe=16,quantizer_efSearch=8           0.1839 0.4163 0.5382      0.00449       19356228    67
nprobe=16,quantizer_efSearch=16          0.1889 0.4237 0.5508      0.00516       19301421    59
nprobe=32,quantizer_efSearch=8           0.1927 0.4525 0.6177      0.00632       38474772    48
nprobe=32,quantizer_efSearch=16          0.1993 0.4672 0.6461      0.00731       38357488    42
nprobe=32,quantizer_efSearch=32          0.2007 0.4709 0.6530      0.00901       38214192    34
nprobe=64,quantizer_efSearch=16          0.2049 0.4896 0.7093      0.01090       76108438    28
nprobe=64,quantizer_efSearch=32          0.2081 0.4968 0.7247      0.01253       75795843    24
nprobe=64,quantizer_efSearch=64          0.2089 0.4991 0.7291      0.01559       75522710    20
nprobe=128,quantizer_efSearch=32         0.2108 0.5101 0.7746      0.01992      150190204    16
nprobe=128,quantizer_efSearch=64         0.2121 0.5139 0.7849      0.02341      149518265    13
nprobe=128,quantizer_efSearch=128        0.2128 0.5159 0.7886      0.03000      149022177    10
nprobe=256,quantizer_efSearch=64         0.2135 0.5204 0.8151      0.03889      295447271    8
nprobe=256,quantizer_efSearch=128        0.2145 0.5228 0.8220      0.04568      294305875    7
nprobe=256,quantizer_efSearch=256        0.2148 0.5240 0.8243      0.05823      293561752    6
nprobe=512,quantizer_efSearch=128        0.2153 0.5289 0.8409      0.08165      579302223    4
nprobe=512,quantizer_efSearch=256        0.2157 0.5307 0.8431      0.09018      577847045    4
nprobe=512,quantizer_efSearch=512        0.2158 0.5309 0.8435      0.12294      576654923    3
nprobe=1024,quantizer_efSearch=512       0.2159 0.5326 0.8523      0.19611     1131862100    2
```

</details>
<details><summary>`PCAR32,IVF1048576(IVF1024,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 7359711897

 code_size 64

 log filename: autotune.dbdeep100M.PCAR32_IVF1048576_IVF1024_PQ16x4fs_RFlat__SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.0992 0.1781 0.1899      0.00253        3200625    119
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.0851 0.1411 0.1464      0.00237        2703654    127
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.0869 0.1560 0.1666      0.00249        2833051    121
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.0912 0.1639 0.1751      0.00251        2823913    120
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.0992 0.1781 0.1899      0.00252        3200262    119
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1079 0.1964 0.2104      0.00256        3189785    118
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1160 0.2083 0.2222      0.00254        3925438    119
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1182 0.2283 0.2518      0.00258        6387354    117
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1351 0.2604 0.2905      0.00269        6377679    112
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1455 0.2941 0.3452      0.00274       11281597    110
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1467 0.2857 0.3216      0.00300        6352057    100
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1652 0.3388 0.4057      0.00319       12635427    94
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1749 0.3679 0.4410      0.00388       12580879    78
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1767 0.3779 0.4766      0.00443       25012240    68
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1859 0.4126 0.5313      0.00475       22237523    64
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1865 0.4066 0.5197      0.00478       24983497    63
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1889 0.4207 0.5440      0.00516       24921130    59
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.1897 0.4224 0.5469      0.00603       30175194    50
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.1915 0.4250 0.5511      0.00648       24870870    47
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1973 0.4565 0.6256      0.00723       44049840    42
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.1984 0.4604 0.6303      0.00830       49270283    37
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2010 0.4665 0.6430      0.00846       43929148    36
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2013 0.4696 0.6468      0.00890       49150004    34
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2016 0.4698 0.6477      0.01074       59565716    28
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2027 0.4736 0.6818      0.01161       79244271    26
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2077 0.4932 0.7135      0.01354       86819137    23
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2086 0.4966 0.7248      0.01742       96956398    18
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2094 0.4980 0.7277      0.02076       96817545    15
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2115 0.5109 0.7743      0.02285      160897527    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2123 0.5124 0.7774      0.02536      171119868    12
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2128 0.5151 0.7845      0.03147      191450667    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.2129 0.5152 0.7845      0.03530      233035609    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2136 0.5176 0.8034      0.04027      317704533    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2139 0.5196 0.8142      0.04222      306469202    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2149 0.5214 0.8180      0.04687      336997500    7
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.2159 0.5265 0.8340      0.08226      665489882    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2164 0.5308 0.8429      0.08444      621071679    4
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.2165 0.5316 0.8492      0.26480     2344067943    2
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQfp16` </summary>
Index size 7307044949

 code_size 64

 log filename: autotune.dbdeep100M.PCAR32_IVF262144_HNSW32_SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2166 0.5320 0.8473      0.16322     1082108040    2
nprobe=1,quantizer_efSearch=4            0.1067 0.1922 0.2099      0.00222        4478665    136
nprobe=2,quantizer_efSearch=4            0.1312 0.2588 0.2970      0.00252        8917910    120
nprobe=2,quantizer_efSearch=8            0.1423 0.2815 0.3218      0.00274        8916098    110
nprobe=2,quantizer_efSearch=16           0.1452 0.2869 0.3276      0.00319        8894839    95
nprobe=4,quantizer_efSearch=4            0.1487 0.3195 0.3902      0.00334       17802409    90
nprobe=4,quantizer_efSearch=8            0.1666 0.3509 0.4306      0.00356       17827259    85
nprobe=4,quantizer_efSearch=16           0.1690 0.3596 0.4398      0.00398       17770736    76
nprobe=4,quantizer_efSearch=32           0.1701 0.3613 0.4415      0.00474       17749779    64
nprobe=8,quantizer_efSearch=4            0.1775 0.4053 0.5268      0.00560       35528947    54
nprobe=8,quantizer_efSearch=8            0.1821 0.4142 0.5395      0.00570       35510416    53
nprobe=8,quantizer_efSearch=16           0.1873 0.4257 0.5547      0.00603       35410012    50
nprobe=8,quantizer_efSearch=32           0.1893 0.4276 0.5572      0.00687       35369260    44
nprobe=8,quantizer_efSearch=64           0.1897 0.4277 0.5576      0.00832       35347662    37
nprobe=16,quantizer_efSearch=8           0.1957 0.4601 0.6393      0.01016       70702002    30
nprobe=16,quantizer_efSearch=16          0.1986 0.4680 0.6517      0.01034       70590128    30
nprobe=16,quantizer_efSearch=32          0.2007 0.4720 0.6580      0.01118       70468652    27
nprobe=16,quantizer_efSearch=64          0.2016 0.4726 0.6581      0.01253       70410891    24
nprobe=16,quantizer_efSearch=128         0.2018 0.4729 0.6585      0.01564       70394408    20
nprobe=32,quantizer_efSearch=16          0.2073 0.4963 0.7283      0.01954      140325084    16
nprobe=32,quantizer_efSearch=32          0.2086 0.5009 0.7360      0.01954      140056389    16
nprobe=32,quantizer_efSearch=64          0.2090 0.5019 0.7381      0.02060      139895268    15
nprobe=32,quantizer_efSearch=128         0.2096 0.5031 0.7396      0.02397      139837353    13
nprobe=32,quantizer_efSearch=256         0.2098 0.5034 0.7399      0.03157      139822649    10
nprobe=64,quantizer_efSearch=32          0.2113 0.5170 0.7910      0.03655      278080574    9
nprobe=64,quantizer_efSearch=64          0.2120 0.5194 0.7950      0.03541      277594007    9
nprobe=64,quantizer_efSearch=128         0.2124 0.5207 0.7963      0.03747      277376509    9
nprobe=64,quantizer_efSearch=256         0.2128 0.5210 0.7967      0.04518      277317396    7
nprobe=128,quantizer_efSearch=64         0.2138 0.5257 0.8263      0.06656      549616576    5
nprobe=128,quantizer_efSearch=128        0.2147 0.5276 0.8297      0.07299      548839719    5
nprobe=128,quantizer_efSearch=256        0.2151 0.5278 0.8301      0.08013      548577379    4
nprobe=256,quantizer_efSearch=128        0.2162 0.5312 0.8465      0.13055     1083652938    3
nprobe=256,quantizer_efSearch=256        0.2166 0.5318 0.8471      0.13228     1082412343    3
nprobe=512,quantizer_efSearch=256        0.2168 0.5331 0.8550      0.25081     2129108235    2
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 7240038809

 code_size 64

 log filename: autotune.dbdeep100M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1875 0.4323 0.5831      0.00776       73071703    39
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1081 0.1938 0.2105      0.00255        5904736    118
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1298 0.2556 0.2911      0.00274       10398981    110
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4     0.1358 0.2709 0.3094      0.00288        9669464    105
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1408 0.2783 0.3181      0.00277       10343327    109
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1426 0.2819 0.3220      0.00297       10333843    102
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1442 0.3010 0.3657      0.00306       18372593    99
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1586 0.3389 0.4130      0.00306       18638079    98
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1599 0.3367 0.4080      0.00331       20689706    91
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1657 0.3499 0.4264      0.00344       20645184    88
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1663 0.3711 0.4744      0.00457       37364382    66
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32    0.1698 0.3586 0.4371      0.00484       23159868    62
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1713 0.3839 0.4950      0.00461       36623102    66
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1815 0.4061 0.5257      0.00493       38509052    61
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1861 0.4182 0.5457      0.00518       38375510    58
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1871 0.4222 0.5508      0.00553       38278321    55
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.1879 0.4246 0.5533      0.00601       40856861    50
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32    0.1884 0.4252 0.5535      0.00979       40780556    31
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1951 0.4549 0.6340      0.00863       72448795    35
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1994 0.4662 0.6490      0.00875       73627661    35
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2001 0.4700 0.6542      0.01112       81203889    28
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.2004 0.4691 0.6517      0.01116       91801233    27
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2019 0.4787 0.7009      0.01468      143034378    21
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2036 0.4848 0.6968      0.01511      147136306    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.2037 0.4850 0.6977      0.01593      152275778    19
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2083 0.5002 0.7339      0.01602      145932279    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2088 0.4974 0.7272      0.01658      151534004    19
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2093 0.5016 0.7374      0.01794      150812728    17
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64   0.2094 0.5017 0.7378      0.02203      150786478    14
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2096 0.5048 0.7579      0.02811      284357543    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2107 0.5098 0.7667      0.02807      286419308    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2111 0.5170 0.7861      0.02949      284832705    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2116 0.5185 0.7880      0.02969      289845417    11
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2128 0.5207 0.7943      0.03029      289003794    10
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.2129 0.5205 0.7941      0.03395      319724792    9
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2134 0.5241 0.8135      0.05585      565624131    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2135 0.5242 0.8136      0.05596      575910370    6
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2144 0.5262 0.8265      0.05531      562401766    6
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2145 0.5261 0.8262      0.05622      572689333    6
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.2146 0.5275 0.8289      0.08134      570971217    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2153 0.5284 0.8361      0.10359     1100766416    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2157 0.5298 0.8393      0.10568     1114034552    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2160 0.5302 0.8411      0.10067     1094254007    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2164 0.5319 0.8454      0.10335     1097687693    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2167 0.5334 0.8529      0.19208     2166508746    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2168 0.5334 0.8550      0.21674     2152895998    2
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQfp16` </summary>
Index size 7226800725

 code_size 64

 log filename: autotune.dbdeep100M.PCAR32_IVF65536_HNSW32_SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1246 0.2482 0.2894      0.00440       17705571    69
nprobe=1,quantizer_efSearch=8            0.1227 0.2448 0.2862      0.00269       17754540    112
nprobe=1,quantizer_efSearch=16           0.1238 0.2472 0.2884      0.00342       17726694    88
nprobe=1,quantizer_efSearch=32           0.1244 0.2479 0.2891      0.00381       17707367    79
nprobe=2,quantizer_efSearch=4            0.1445 0.3065 0.3808      0.00387       35478291    78
nprobe=2,quantizer_efSearch=8            0.1535 0.3262 0.4056      0.00425       35499302    71
nprobe=2,quantizer_efSearch=16           0.1558 0.3309 0.4115      0.00450       35442277    67
nprobe=2,quantizer_efSearch=32           0.1564 0.3318 0.4127      0.00498       35417443    61
nprobe=2,quantizer_efSearch=64           0.1565 0.3320 0.4129      0.00584       35412414    52
nprobe=4,quantizer_efSearch=4            0.1615 0.3636 0.4837      0.00805       70719683    38
nprobe=4,quantizer_efSearch=16           0.1791 0.4030 0.5368      0.00804       70674112    38
nprobe=4,quantizer_efSearch=32           0.1802 0.4041 0.5382      0.00838       70625698    36
nprobe=4,quantizer_efSearch=64           0.1803 0.4043 0.5384      0.00907       70609770    34
nprobe=8,quantizer_efSearch=16           0.1960 0.4564 0.6470      0.01497      140775733    21
nprobe=8,quantizer_efSearch=32           0.1972 0.4583 0.6491      0.01409      140645796    22
nprobe=16,quantizer_efSearch=8           0.2038 0.4857 0.7208      0.02621      279885749    12
nprobe=16,quantizer_efSearch=16          0.2063 0.4929 0.7310      0.02588      279495890    12
nprobe=16,quantizer_efSearch=32          0.2072 0.4954 0.7349      0.02689      279175652    12
nprobe=16,quantizer_efSearch=64          0.2077 0.4959 0.7353      0.02742      279043577    11
nprobe=32,quantizer_efSearch=32          0.2123 0.5156 0.7936      0.05019      552769811    6
nprobe=32,quantizer_efSearch=64          0.2128 0.5160 0.7944      0.05180      552330098    6
nprobe=32,quantizer_efSearch=256         0.2129 0.5166 0.7948      0.05804      552225939    6
nprobe=64,quantizer_efSearch=32          0.2138 0.5236 0.8249      0.09235     1091855762    4
nprobe=64,quantizer_efSearch=64          0.2149 0.5260 0.8281      0.09706     1090371195    4
nprobe=512,quantizer_efSearch=64         0.2153 0.5336 0.8543      0.64520     8087549052    1
```

</details>
<details><summary>`PCAR64,IVF1048576_HNSW32,SQ8` </summary>
Index size 7762249173

 code_size 64

 log filename: autotune.dbdeep100M.PCAR64_IVF1048576_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5666 0.9313 0.9538      0.12919      307689067    3
nprobe=1,quantizer_efSearch=4            0.1526 0.1810 0.1816      0.00275        1290815    110
nprobe=2,quantizer_efSearch=4            0.2080 0.2566 0.2574      0.00293        2579119    103
nprobe=4,quantizer_efSearch=4            0.2641 0.3428 0.3438      0.00312        5139380    97
nprobe=4,quantizer_efSearch=8            0.3043 0.3956 0.3967      0.00386        5125883    78
nprobe=8,quantizer_efSearch=8            0.3651 0.5034 0.5051      0.00484       10238275    62
nprobe=8,quantizer_efSearch=16           0.3852 0.5306 0.5325      0.00601       10178366    50
nprobe=16,quantizer_efSearch=4           0.4030 0.5801 0.5836      0.00572       20429136    53
nprobe=16,quantizer_efSearch=8           0.4288 0.6232 0.6268      0.00668       20379734    45
nprobe=16,quantizer_efSearch=16          0.4403 0.6394 0.6432      0.00721       20320294    42
nprobe=32,quantizer_efSearch=8           0.4664 0.7021 0.7090      0.00938       40553168    32
nprobe=32,quantizer_efSearch=16          0.4848 0.7360 0.7435      0.01055       40396094    29
nprobe=32,quantizer_efSearch=32          0.4930 0.7486 0.7564      0.01366       40197887    22
nprobe=64,quantizer_efSearch=16          0.5118 0.7994 0.8113      0.01892       80238953    16
nprobe=64,quantizer_efSearch=32          0.5255 0.8262 0.8383      0.02061       79786541    15
nprobe=64,quantizer_efSearch=64          0.5316 0.8350 0.8471      0.02347       79369732    13
nprobe=128,quantizer_efSearch=32         0.5422 0.8734 0.8909      0.03255      158249758    10
nprobe=128,quantizer_efSearch=64         0.5529 0.8900 0.9082      0.03301      157331076    10
nprobe=128,quantizer_efSearch=128        0.5561 0.8960 0.9141      0.04374      156632762    7
nprobe=256,quantizer_efSearch=64         0.5615 0.9193 0.9416      0.06124      310994905    6
nprobe=256,quantizer_efSearch=128        0.5659 0.9295 0.9524      0.06618      309348986    5
nprobe=256,quantizer_efSearch=256        0.5664 0.9309 0.9535      0.08464      308195725    4
nprobe=512,quantizer_efSearch=64         0.5668 0.9337 0.9588      0.11743      608185447    3
nprobe=512,quantizer_efSearch=128        0.5719 0.9475 0.9742      0.13050      608845571    3
nprobe=512,quantizer_efSearch=256        0.5734 0.9511 0.9778      0.14752      606482009    3
nprobe=512,quantizer_efSearch=512        0.5736 0.9518 0.9784      0.17492      604552578    2
nprobe=1024,quantizer_efSearch=128       0.5739 0.9535 0.9820      0.21856     1181055957    2
nprobe=1024,quantizer_efSearch=256       0.5758 0.9586 0.9881      0.23528     1188651489    2
nprobe=1024,quantizer_efSearch=512       0.5760 0.9601 0.9899      0.27873     1185272680    2
nprobe=2048,quantizer_efSearch=256       0.5770 0.9619 0.9918      0.39373     2275081099    1
nprobe=2048,quantizer_efSearch=512       0.5775 0.9648 0.9954      0.45525     2309100522    1
nprobe=4096,quantizer_efSearch=512       0.5779 0.9655 0.9967      0.78034     4342866508    1
```

</details>
<details><summary>`PCAR64,IVF1048576(IVF1024,PQ32x4fs,RFlat),SQ8` </summary>
Index size 7502594585

 code_size 64

 log filename: autotune.dbdeep100M.PCAR64_IVF1048576_IVF1024_PQ32x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.5780 0.9655 0.9972      1.70354     4517011409    1
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.1540 0.1823 0.1829      0.00251        2023103    120
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1633 0.1975 0.1981      0.00249        2941449    121
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.1752 0.2125 0.2131      0.00251        2943707    120
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2054 0.2513 0.2519      0.00259        3302770    116
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.2072 0.2616 0.2624      0.00270        5495007    112
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2499 0.3168 0.3176      0.00263        5867521    115
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2650 0.3405 0.3415      0.00274        5861226    110
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2760 0.3516 0.3525      0.00300        6578046    101
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2924 0.3781 0.3792      0.00297        6558380    101
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.3000 0.3933 0.3944      0.00352       11692014    86
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.3022 0.3910 0.3921      0.00335        6550300    90
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.3115 0.4095 0.4108      0.00367       13087127    82
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.3161 0.4268 0.4279      0.00367       10943470    82
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3385 0.4562 0.4574      0.00348       11677536    87
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.3718 0.5095 0.5110      0.00404       13047973    75
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.3812 0.5213 0.5230      0.00556       13025872    54
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3890 0.5529 0.5555      0.00571       21810939    53
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4036 0.5788 0.5815      0.00608       21761270    50
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4094 0.5864 0.5893      0.00620       21729368    49
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4242 0.6033 0.6067      0.00673       25870004    45
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4398 0.6327 0.6363      0.00658       25812296    46
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4460 0.6437 0.6475      0.00724       25781806    42
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4484 0.6476 0.6514      0.00878       31123309    35
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4599 0.6868 0.6924      0.00896       43222166    34
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4861 0.7361 0.7432      0.01074       45746239    28
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4917 0.7458 0.7534      0.01201       45705939    25
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4953 0.7520 0.7599      0.01303       50982200    24
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4967 0.7540 0.7619      0.01565       61416706    20
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.5223 0.8189 0.8306      0.01970       85121083    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5277 0.8292 0.8408      0.01885       90276630    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.5322 0.8345 0.8464      0.02340      100498164    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5509 0.8867 0.9043      0.03432      167806938    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5539 0.8911 0.9088      0.03642      177973058    9
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5544 0.8942 0.9118      0.04348      198359020    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.5619 0.9177 0.9397      0.06209      351217149    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5623 0.9180 0.9398      0.06337      330713494    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.5652 0.9276 0.9505      0.07707      391475405    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.5653 0.9272 0.9502      0.06695      350048834    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5679 0.9377 0.9630      0.11460      620173382    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.5731 0.9500 0.9768      0.13106      646941525    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512   0.5736 0.9516 0.9786      0.16473      769332687    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5759 0.9583 0.9879      0.23506     1267265654    2
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5766 0.9592 0.9892      0.25197     1225149605    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5780 0.9635 0.9947      0.40578     2352165605    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5781 0.9639 0.9956      0.69464     4598444592    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.5782 0.9669 0.9988      0.84611     4549387449    1
nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=256 0.5783 0.9670 0.9989      1.76768     4548933235    1
nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=512 0.5784 0.9671 0.9991      1.87140     4628528636    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ8` </summary>
Index size 7340612309

 code_size 64

 log filename: autotune.dbdeep100M.PCAR64_IVF262144_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5748 0.9540 0.9816      0.16851     1100123557    2
nprobe=2,quantizer_efSearch=4            0.2652 0.3470 0.3475      0.00283        8976849    106
nprobe=2,quantizer_efSearch=8            0.2998 0.3885 0.3892      0.00337        8973268    90
nprobe=2,quantizer_efSearch=16           0.3050 0.3956 0.3963      0.00403        8945831    75
nprobe=4,quantizer_efSearch=8            0.3703 0.5094 0.5117      0.00454       17980188    67
nprobe=4,quantizer_efSearch=16           0.3802 0.5222 0.5244      0.00492       17921882    62
nprobe=8,quantizer_efSearch=8            0.4265 0.6217 0.6259      0.00657       35899074    46
nprobe=8,quantizer_efSearch=16           0.4401 0.6423 0.6466      0.00717       35784246    42
nprobe=8,quantizer_efSearch=32           0.4431 0.6452 0.6496      0.00823       35717934    37
nprobe=8,quantizer_efSearch=64           0.4434 0.6456 0.6500      0.01165       35691982    26
nprobe=16,quantizer_efSearch=8           0.4777 0.7284 0.7365      0.01095       71578339    28
nprobe=16,quantizer_efSearch=16          0.4870 0.7429 0.7510      0.01113       71418856    27
nprobe=16,quantizer_efSearch=32          0.4917 0.7503 0.7588      0.01223       71247050    25
nprobe=16,quantizer_efSearch=64          0.4928 0.7518 0.7602      0.01458       71178519    21
nprobe=32,quantizer_efSearch=16          0.5241 0.8238 0.8374      0.01962      142324965    16
nprobe=32,quantizer_efSearch=32          0.5296 0.8344 0.8480      0.02072      141968103    15
nprobe=32,quantizer_efSearch=64          0.5315 0.8367 0.8502      0.02273      141743401    14
nprobe=64,quantizer_efSearch=16          0.5418 0.8724 0.8902      0.03542      283369230    9
nprobe=64,quantizer_efSearch=32          0.5507 0.8920 0.9100      0.03619      282540667    9
nprobe=64,quantizer_efSearch=64          0.5550 0.8968 0.9151      0.03927      281915636    8
nprobe=64,quantizer_efSearch=256         0.5556 0.8979 0.9165      0.05199      281553551    6
nprobe=128,quantizer_efSearch=32         0.5610 0.9219 0.9445      0.06757      560530743    5
nprobe=128,quantizer_efSearch=64         0.5673 0.9316 0.9555      0.07029      559103483    5
nprobe=128,quantizer_efSearch=128        0.5684 0.9342 0.9584      0.07358      558112142    5
nprobe=128,quantizer_efSearch=256        0.5688 0.9346 0.9587      0.08322      557724524    4
nprobe=256,quantizer_efSearch=64         0.5715 0.9484 0.9748      0.13256     1104680937    3
nprobe=256,quantizer_efSearch=128        0.5738 0.9528 0.9803      0.13635     1102355765    3
nprobe=256,quantizer_efSearch=512        0.5748 0.9540 0.9816      0.17077     1100123557    2
nprobe=512,quantizer_efSearch=128        0.5749 0.9599 0.9890      0.25633     2166984989    2
nprobe=512,quantizer_efSearch=256        0.5761 0.9625 0.9917      0.27397     2164095535    2
nprobe=512,quantizer_efSearch=512        0.5765 0.9630 0.9922      0.28610     2161944684    2
nprobe=1024,quantizer_efSearch=256       0.5771 0.9657 0.9961      0.50303     4222412817    1
nprobe=1024,quantizer_efSearch=512       0.5776 0.9668 0.9973      0.54285     4227756190    1
nprobe=2048,quantizer_efSearch=512       0.5777 0.9673 0.9983      1.00283     8202536942    1
nprobe=4096,quantizer_efSearch=512       0.5779 0.9678 0.9992      1.77445    14996344797    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ8` </summary>
Index size 7275829785

 code_size 64

 log filename: autotune.dbdeep100M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4674 0.7018 0.7099      0.01219       73370462    25
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1695 0.2045 0.2052      0.00261        4695618    115
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.2037 0.2463 0.2472      0.00268        4886021    112
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=2     0.2039 0.2465 0.2474      0.00272        4881752    111
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2156 0.2621 0.2630      0.00289        7238418    104
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2889 0.3738 0.3747      0.00299        9729036    101
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2910 0.3752 0.3762      0.00301       10405229    100
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.3027 0.3904 0.3913      0.00328       10399351    92
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3044 0.3931 0.3940      0.00354       11706624    85
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3046 0.3934 0.3943      0.00412       14333809    73
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3378 0.4578 0.4599      0.00434       18817374    70
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3578 0.4889 0.4910      0.00433       18779446    70
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3593 0.4912 0.4933      0.00452       18769969    67
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3790 0.5190 0.5212      0.00501       20694466    60
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.3803 0.5211 0.5233      0.00689       28545090    44
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32    0.3834 0.5249 0.5272      0.00635       23300503    48
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4071 0.5891 0.5937      0.00699       36811497    43
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4141 0.5941 0.5983      0.00719       37487652    42
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4349 0.6297 0.6348      0.00798       38599078    38
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4404 0.6394 0.6446      0.00778       38564673    39
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4424 0.6429 0.6482      0.00841       41162576    36
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4434 0.6434 0.6485      0.00904       41135358    34
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.4438 0.6442 0.6492      0.01030       46373009    30
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=128   0.4442 0.6446 0.6496      0.01169       56801561    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4674 0.7018 0.7099      0.01230       73372286    25
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4764 0.7184 0.7266      0.01266       74504511    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4882 0.7402 0.7492      0.01286       74246591    24
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4887 0.7417 0.7509      0.01322       74160320    23
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.4940 0.7491 0.7584      0.01628       81886022    19
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.4942 0.7493 0.7586      0.01760       92273951    18
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5057 0.7856 0.7976      0.02299      144830895    14
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5098 0.7953 0.8084      0.02274      144364121    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5240 0.8188 0.8317      0.02341      148002034    13
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5242 0.8194 0.8323      0.02433      153119916    13
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5291 0.8309 0.8448      0.02479      152651888    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5302 0.8320 0.8460      0.02440      147388530    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5304 0.8333 0.8472      0.02551      152520107    12
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5309 0.8340 0.8479      0.02717      162923577    12
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64   0.5310 0.8341 0.8481      0.03932      152472014    8
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=128  0.5316 0.8346 0.8486      0.03230      162876214    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5506 0.8831 0.9015      0.04324      288857127    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5531 0.8890 0.9081      0.04380      287946126    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5545 0.8916 0.9107      0.04481      292897123    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5563 0.8951 0.9144      0.04662      292647788    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.5567 0.8957 0.9151      0.05309      323493882    6
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=128  0.5568 0.8961 0.9156      0.05756      302950295    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5654 0.9249 0.9482      0.08301      571273296    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5663 0.9259 0.9493      0.08637      581435204    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.5684 0.9289 0.9532      0.08685      564917188    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5700 0.9319 0.9563      0.08700      579759503    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5703 0.9331 0.9576      0.09224      599921549    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.5705 0.9335 0.9579      0.09513      579420736    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5722 0.9435 0.9705      0.15634     1113565460    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5747 0.9490 0.9768      0.16094     1126105456    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5762 0.9527 0.9811      0.16325     1122995401    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5763 0.9524 0.9809      0.17236     1142990621    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5775 0.9599 0.9901      0.30724     2211051714    1
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5780 0.9617 0.9917      0.31935     2185267881    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.5782 0.9639 0.9952      0.57696     4256875052    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5793 0.9653 0.9969      0.56899     4266952426    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.5795 0.9669 0.9985      1.14883     8294094432    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5796 0.9672 0.9991      2.13912    16399601193    1
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ8` </summary>
Index size 7235202261

 code_size 64

 log filename: autotune.dbdeep100M.PCAR64_IVF65536_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4203 0.6014 0.6056      0.01150       71120766    27
nprobe=1,quantizer_efSearch=4            0.2428 0.3111 0.3123      0.00363       17760595    83
nprobe=1,quantizer_efSearch=8            0.2624 0.3366 0.3378      0.00380       17763857    79
nprobe=1,quantizer_efSearch=16           0.2664 0.3415 0.3427      0.00423       17721519    71
nprobe=1,quantizer_efSearch=32           0.2671 0.3424 0.3436      0.00465       17702328    65
nprobe=2,quantizer_efSearch=16           0.3556 0.4792 0.4814      0.00680       35468996    45
nprobe=2,quantizer_efSearch=32           0.3562 0.4797 0.4819      0.00727       35439810    42
nprobe=4,quantizer_efSearch=4            0.3844 0.5479 0.5517      0.01095       71041908    28
nprobe=4,quantizer_efSearch=8            0.4203 0.6014 0.6056      0.01148       71120766    27
nprobe=4,quantizer_efSearch=32           0.4317 0.6166 0.6211      0.01207       70890057    25
nprobe=8,quantizer_efSearch=4            0.4657 0.6988 0.7077      0.02042      141995092    15
nprobe=8,quantizer_efSearch=8            0.4740 0.7122 0.7210      0.02055      141868882    15
nprobe=8,quantizer_efSearch=16           0.4863 0.7304 0.7395      0.02132      141464203    15
nprobe=8,quantizer_efSearch=32           0.4884 0.7330 0.7422      0.02218      141287196    14
nprobe=16,quantizer_efSearch=8           0.5169 0.8116 0.8258      0.04127      282367957    8
nprobe=16,quantizer_efSearch=16          0.5252 0.8248 0.8394      0.03906      281851306    8
nprobe=16,quantizer_efSearch=128         0.5286 0.8315 0.8462      0.04286      281236593    7
nprobe=32,quantizer_efSearch=8           0.5362 0.8597 0.8795      0.07483      560644178    5
nprobe=32,quantizer_efSearch=128         0.5523 0.8920 0.9139      0.07797      557602369    4
nprobe=64,quantizer_efSearch=32          0.5638 0.9274 0.9523      0.14130     1104391064    3
nprobe=64,quantizer_efSearch=128         0.5651 0.9303 0.9556      0.14688     1102001645    3
nprobe=128,quantizer_efSearch=128        0.5728 0.9524 0.9808      0.27945     2168326000    2
nprobe=128,quantizer_efSearch=512        0.5730 0.9526 0.9810      0.32273     2167298535    2
nprobe=256,quantizer_efSearch=256        0.5769 0.9625 0.9927      0.54449     4248590796    1
nprobe=1024,quantizer_efSearch=256       0.5772 0.9667 0.9984      1.99485    16035085334    1
nprobe=1024,quantizer_efSearch=512       0.5774 0.9671 0.9988      2.12085    16051946147    1
```

</details>
