# Detailed logs for dataset deep1M

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF1024,PQ16x4fs` </summary>
Index size 16434455

 code_size 8

 log filename: autotune.dbdeep1M.OPQ16_64_IVF1024_PQ16x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1283 0.3845 0.5963      0.00068       23618520    444
nprobe=1                                 0.1114 0.3099 0.4460      0.00057       12091320    524
nprobe=2                                 0.1283 0.3845 0.5963      0.00067       23618520    446
nprobe=4                                 0.1377 0.4359 0.7229      0.00083       46092750    362
nprobe=8                                 0.1450 0.4642 0.8030      0.00104       89113658    290
```

</details>
<details><summary>`OPQ16_64,IVF1024,PQ16x4fsr` </summary>
Index size 16434455

 code_size 8

 log filename: autotune.dbdeep1M.OPQ16_64_IVF1024_PQ16x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1813 0.4810 0.6190      0.00073       23618520    414
nprobe=2                                 0.1813 0.4810 0.6190      0.00073       23618520    413
nprobe=4                                 0.1987 0.5575 0.7619      0.00097       46092750    311
nprobe=8                                 0.2055 0.5995 0.8564      0.00138       89113658    218
nprobe=16                                0.2100 0.6189 0.9088      0.00215      170819148    140
nprobe=512                               0.2107 0.6235 0.9393      0.11454     4505205896    3
```

</details>
<details><summary>`OPQ16_64,IVF4096_HNSW32,PQ16x4fs` </summary>
Index size 18761292

 code_size 8

 log filename: autotune.dbdeep1M.OPQ16_64_IVF4096_HNSW32_PQ16x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1262 0.3921 0.5900      0.00069       11564391    437
nprobe=2,quantizer_efSearch=8            0.1245 0.3569 0.5019      0.00074        5899362    406
nprobe=4,quantizer_efSearch=4            0.1262 0.3921 0.5900      0.00070       11564391    428
nprobe=4,quantizer_efSearch=8            0.1331 0.4185 0.6305      0.00082       11641703    366
nprobe=8,quantizer_efSearch=4            0.1388 0.4474 0.7313      0.00089       22855852    339
nprobe=8,quantizer_efSearch=8            0.1398 0.4519 0.7378      0.00094       22849438    320
nprobe=16,quantizer_efSearch=4           0.1399 0.4623 0.7900      0.00117       44637569    256
nprobe=8,quantizer_efSearch=16           0.1413 0.4541 0.7437      0.00117       22830484    257
nprobe=16,quantizer_efSearch=8           0.1422 0.4689 0.8066      0.00133       44635041    227
nprobe=16,quantizer_efSearch=256         0.1436 0.4743 0.8150      0.01157       44543827    26
```

</details>
<details><summary>`OPQ16_64,IVF4096_HNSW32,PQ16x4fsr` </summary>
Index size 18761036

 code_size 8

 log filename: autotune.dbdeep1M.OPQ16_64_IVF4096_HNSW32_PQ16x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2273 0.6340 0.8795      0.00261       44551331    116
nprobe=1,quantizer_efSearch=4            0.1430 0.3070 0.3451      0.00052        2975143    579
nprobe=2,quantizer_efSearch=4            0.1723 0.4151 0.4874      0.00062        5881653    482
nprobe=2,quantizer_efSearch=8            0.1819 0.4372 0.5137      0.00074        5900736    405
nprobe=4,quantizer_efSearch=4            0.1912 0.4938 0.6159      0.00081       11589456    370
nprobe=4,quantizer_efSearch=8            0.2030 0.5259 0.6566      0.00093       11641314    321
nprobe=4,quantizer_efSearch=16           0.2039 0.5282 0.6595      0.00116       11635304    260
nprobe=8,quantizer_efSearch=4            0.2173 0.5866 0.7748      0.00121       22861966    248
nprobe=8,quantizer_efSearch=8            0.2185 0.5910 0.7817      0.00126       22852079    238
nprobe=8,quantizer_efSearch=16           0.2207 0.5939 0.7864      0.00150       22826983    201
nprobe=16,quantizer_efSearch=4           0.2227 0.6188 0.8531      0.00185       44668315    162
nprobe=16,quantizer_efSearch=8           0.2257 0.6290 0.8708      0.00194       44635887    155
nprobe=16,quantizer_efSearch=16          0.2262 0.6329 0.8770      0.00208       44584709    145
nprobe=16,quantizer_efSearch=32          0.2273 0.6340 0.8795      0.00259       44551331    116
nprobe=16,quantizer_efSearch=64          0.2274 0.6341 0.8799      0.00337       44546377    90
nprobe=32,quantizer_efSearch=16          0.2294 0.6486 0.9186      0.00330       86538997    91
nprobe=32,quantizer_efSearch=32          0.2299 0.6500 0.9220      0.00360       86442490    84
nprobe=32,quantizer_efSearch=64          0.2301 0.6501 0.9225      0.00446       86414980    68
nprobe=64,quantizer_efSearch=16          0.2304 0.6536 0.9371      0.01416      166845392    22
nprobe=64,quantizer_efSearch=256         0.2311 0.6560 0.9440      0.02149      166244146    14
```

</details>
<details><summary>`OPQ8_64,IVF1024_HNSW32,PQ8` </summary>
Index size 16638896

 code_size 8

 log filename: autotune.dbdeep1M.OPQ8_64_IVF1024_HNSW32_PQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                        0.1598 0.3714 0.4538      0.00369       12941984    82
nprobe=1,quantizer_efSearch=4,ht=4      0.0012 0.0025 0.0029      0.00176       12960393    171
nprobe=1,quantizer_efSearch=4,ht=22     0.1009 0.1686 0.1802      0.00183       12960393    164
nprobe=1,quantizer_efSearch=16,ht=22    0.1014 0.1701 0.1819      0.00233       12941984    130
nprobe=1,quantizer_efSearch=8,ht=26     0.1454 0.3004 0.3368      0.00233       12951650    129
nprobe=1,quantizer_efSearch=4,ht=28     0.1522 0.3337 0.3882      0.00242       12960393    125
nprobe=1,quantizer_efSearch=8,ht=30     0.1570 0.3594 0.4287      0.00302       12951650    100
nprobe=1,quantizer_efSearch=16,ht=64    0.1598 0.3714 0.4538      0.00342       12941984    88
nprobe=2,quantizer_efSearch=4,ht=64     0.1883 0.4707 0.6007      0.00369       25326700    82
nprobe=2,quantizer_efSearch=16,ht=30    0.1887 0.4697 0.5803      0.00445       25286659    68
nprobe=4,quantizer_efSearch=8,ht=26     0.1972 0.4798 0.5707      0.00457       49063625    66
nprobe=4,quantizer_efSearch=16,ht=64    0.2118 0.5639 0.7607      0.00523       49038303    58
nprobe=8,quantizer_efSearch=8,ht=64     0.2237 0.6104 0.8603      0.00705       94503194    43
nprobe=8,quantizer_efSearch=16,ht=64    0.2239 0.6123 0.8631      0.00773       94421940    39
nprobe=8,quantizer_efSearch=32,ht=64    0.2241 0.6128 0.8639      0.00846       94408071    36
nprobe=16,quantizer_efSearch=32,ht=64   0.2280 0.6291 0.9130      0.01175      180080255    26
nprobe=64,quantizer_efSearch=32,ht=64   0.2284 0.6378 0.9474      0.03367      647244509    9
nprobe=256,quantizer_efSearch=128,ht=64 0.2285 0.6384 0.9497      0.11955     2373831295    3
```

</details>
<details><summary>`OPQ8_64,IVF1024,PQ8` </summary>
Index size 16360699

 code_size 8

 log filename: autotune.dbdeep1M.OPQ8_64_IVF1024_PQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1961 0.4791 0.5664      0.00258       49021541    117
nprobe=1,ht=64                           0.1564 0.3753 0.4552      0.00191       12937835    157
nprobe=1,ht=32                           0.1568 0.3720 0.4472      0.00201       12937835    150
nprobe=2,ht=26                           0.1756 0.3990 0.4538      0.00207       25278465    145
nprobe=4,ht=26                           0.1961 0.4791 0.5664      0.00257       49021541    117
nprobe=4,ht=64                           0.2095 0.5697 0.7635      0.00265       49021541    114
nprobe=8,ht=64                           0.2201 0.6170 0.8644      0.00416       94390161    73
nprobe=16,ht=30                          0.2219 0.6226 0.8709      0.01044      180017406    29
nprobe=32,ht=64                          0.2238 0.6433 0.9401      0.01003      341106028    30
nprobe=128,ht=64                         0.2243 0.6456 0.9510      0.03192     1233031698    10
```

</details>
<details><summary>`OPQ8_64,IVF4096_HNSW32,PQ8` </summary>
Index size 18283568

 code_size 8

 log filename: autotune.dbdeep1M.OPQ8_64_IVF4096_HNSW32_PQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1871 0.4443 0.5170      0.00340       16808350    89
nprobe=1,quantizer_efSearch=4,ht=22      0.0812 0.1205 0.1264      0.00210        4425519    143
nprobe=1,quantizer_efSearch=4,ht=26      0.1199 0.2228 0.2344      0.00216        4425519    139
nprobe=4,quantizer_efSearch=4,ht=26      0.1762 0.3919 0.4333      0.00315       16808350    96
nprobe=8,quantizer_efSearch=8,ht=64      0.2179 0.5868 0.7723      0.00374       32349581    81
nprobe=8,quantizer_efSearch=64,ht=64     0.2192 0.5936 0.7826      0.00601       32130543    50
nprobe=16,quantizer_efSearch=128,ht=64   0.2251 0.6309 0.8687      0.01039       61236186    29
nprobe=32,quantizer_efSearch=32,ht=32    0.2259 0.6383 0.8967      0.02191      115847442    14
nprobe=64,quantizer_efSearch=256,ht=64   0.2268 0.6523 0.9405      0.02568      216781118    12
nprobe=512,quantizer_efSearch=16,ht=64   0.2269 0.6498 0.9382      0.04973      839270594    7
nprobe=256,quantizer_efSearch=128,ht=64  0.2272 0.6536 0.9492      0.05045      767416172    6
nprobe=256,quantizer_efSearch=256,ht=64  0.2273 0.6537 0.9493      0.05566      766855995    6
```

</details>
<details><summary>`PCAR16,IVF1024_HNSW32,SQ4` </summary>
Index size 16396181

 code_size 8

 log filename: autotune.dbdeep1M.PCAR16_IVF1024_HNSW32_SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                0.0606 0.2197 0.4289      0.00372       22159017    81
nprobe=1,quantizer_efSearch=8   0.0546 0.1808 0.3204      0.00284       11211949    106
nprobe=2,quantizer_efSearch=4   0.0606 0.2197 0.4289      0.00368       22159017    82
nprobe=2,quantizer_efSearch=8   0.0610 0.2225 0.4315      0.00372       22139487    81
nprobe=2,quantizer_efSearch=16  0.0611 0.2224 0.4312      0.00396       22130311    76
nprobe=4,quantizer_efSearch=4   0.0655 0.2448 0.5235      0.00525       43683950    58
nprobe=4,quantizer_efSearch=8   0.0664 0.2489 0.5283      0.00538       43635977    56
nprobe=4,quantizer_efSearch=16  0.0666 0.2493 0.5288      0.00563       43618313    54
nprobe=8,quantizer_efSearch=4   0.0686 0.2598 0.5823      0.00834       85783214    36
nprobe=8,quantizer_efSearch=32  0.0687 0.2618 0.5852      0.00906       85700603    34
nprobe=16,quantizer_efSearch=4  0.0689 0.2645 0.6063      0.01413      167693948    22
nprobe=16,quantizer_efSearch=64 0.0693 0.2669 0.6123      0.01569      167507680    20
```

</details>
<details><summary>`PCAR16,IVF1024,SQ4` </summary>
Index size 16117984

 code_size 8

 log filename: autotune.dbdeep1M.PCAR16_IVF1024_SQ4.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0611 0.2224 0.4312      0.00211       22127567    143
nprobe=2                                 0.0611 0.2224 0.4312      0.00216       22127567    139
nprobe=4                                 0.0666 0.2492 0.5289      0.00302       43612231    100
nprobe=8                                 0.0687 0.2617 0.5852      0.00483       85698237    63
nprobe=16                                0.0693 0.2668 0.6123      0.00992      167503051    31
```

</details>
<details><summary>`PCAR16,IVF4096_HNSW32,SQ4` </summary>
Index size 17451029

 code_size 8

 log filename: autotune.dbdeep1M.PCAR16_IVF4096_HNSW32_SQ4.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0644 0.2292 0.4428      0.00233       12317025    129
nprobe=4,quantizer_efSearch=8            0.0644 0.2292 0.4424      0.00181       12337811    166
nprobe=8,quantizer_efSearch=4            0.0672 0.2513 0.5230      0.00220       24256217    137
nprobe=8,quantizer_efSearch=8            0.0678 0.2532 0.5273      0.00225       24241703    134
nprobe=8,quantizer_efSearch=16           0.0684 0.2527 0.5272      0.00245       24203452    123
nprobe=8,quantizer_efSearch=32           0.0685 0.2526 0.5271      0.00277       24194285    109
nprobe=32,quantizer_efSearch=4           0.0691 0.2668 0.5956      0.00895       92729868    34
nprobe=32,quantizer_efSearch=8           0.0696 0.2708 0.6086      0.00700       92719824    43
nprobe=32,quantizer_efSearch=32          0.0698 0.2711 0.6101      0.00561       92474334    54
```

</details>

## Code sizes in [9, 16]

<details><summary>`OPQ16_64,IVF1024_HNSW32,PQ16` </summary>
Index size 24638896

 code_size 16

 log filename: autotune.dbdeep1M.OPQ16_64_IVF1024_HNSW32_PQ16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3777 0.7289 0.7720      0.04854       49379185    7
nprobe=1,quantizer_efSearch=4,ht=12      0.0020 0.0029 0.0031      0.00196       13025183    154
nprobe=1,quantizer_efSearch=8,ht=14      0.0025 0.0038 0.0040      0.00212       13002806    142
nprobe=1,quantizer_efSearch=16,ht=18     0.0037 0.0059 0.0063      0.00242       13002204    124
nprobe=1,quantizer_efSearch=4,ht=52      0.2284 0.3603 0.3669      0.00242       13025183    125
nprobe=1,quantizer_efSearch=4,ht=54      0.2433 0.3940 0.4025      0.00266       13025183    113
nprobe=1,quantizer_efSearch=4,ht=58      0.2520 0.4270 0.4385      0.00310       13025183    97
nprobe=1,quantizer_efSearch=16,ht=58     0.2559 0.4336 0.4451      0.00376       13002204    80
nprobe=2,quantizer_efSearch=32,ht=50     0.2729 0.4351 0.4426      0.00441       25337348    69
nprobe=2,quantizer_efSearch=32,ht=52     0.2956 0.4924 0.5021      0.00458       25337348    66
nprobe=2,quantizer_efSearch=4,ht=60      0.3165 0.5794 0.6030      0.00502       25385963    60
nprobe=4,quantizer_efSearch=16,ht=54     0.3640 0.6608 0.6853      0.00652       49390386    47
nprobe=4,quantizer_efSearch=16,ht=128    0.3782 0.7289 0.7731      0.00700       49390386    43
nprobe=8,quantizer_efSearch=8,ht=128     0.4073 0.8170 0.8811      0.01026       95109777    30
nprobe=16,quantizer_efSearch=16,ht=128   0.4233 0.8703 0.9498      0.01743      181627578    18
nprobe=16,quantizer_efSearch=64,ht=64    0.4235 0.8702 0.9493      0.03259      181544448    10
nprobe=32,quantizer_efSearch=128,ht=60   0.4245 0.8862 0.9684      0.04927      343269751    7
nprobe=32,quantizer_efSearch=64,ht=62    0.4252 0.8911 0.9760      0.05234      343283081    6
nprobe=64,quantizer_efSearch=128,ht=60   0.4257 0.8948 0.9816      0.08907      650930846    4
nprobe=64,quantizer_efSearch=512,ht=128  0.4273 0.9012 0.9937      0.09122      650919419    4
nprobe=512,quantizer_efSearch=128,ht=128 0.4275 0.9036 0.9983      0.37998     4672924497    1
```

</details>
<details><summary>`OPQ16_64,IVF1024,PQ16` </summary>
Index size 24360699

 code_size 16

 log filename: autotune.dbdeep1M.OPQ16_64_IVF1024_PQ16.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3163 0.5354 0.5501      0.00224       25335091    134
nprobe=1,ht=56                           0.2551 0.4201 0.4313      0.00198       12998321    152
nprobe=1,ht=128                          0.2600 0.4439 0.4582      0.00200       12998321    151
nprobe=2,ht=50                           0.2827 0.4385 0.4468      0.00201       25335091    150
nprobe=2,ht=52                           0.3035 0.4946 0.5054      0.00210       25335091    143
nprobe=2,ht=54                           0.3163 0.5354 0.5501      0.00227       25335091    133
nprobe=2,ht=56                           0.3233 0.5641 0.5835      0.00250       25335091    120
nprobe=2,ht=128                          0.3282 0.5938 0.6207      0.00256       25335091    118
nprobe=4,ht=50                           0.3291 0.5381 0.5493      0.00310       49373608    97
nprobe=4,ht=128                          0.3787 0.7284 0.7728      0.00363       49373608    83
nprobe=4,ht=64                           0.3788 0.7277 0.7717      0.00535       49373608    57
nprobe=8,ht=128                          0.4100 0.8217 0.8853      0.00606       94995488    50
nprobe=8,ht=64                           0.4105 0.8210 0.8841      0.00926       94995488    33
nprobe=16,ht=128                         0.4237 0.8709 0.9504      0.00934      181511621    33
nprobe=32,ht=128                         0.4273 0.8926 0.9805      0.01635      343204541    19
nprobe=64,ht=128                         0.4291 0.9011 0.9936      0.02986      650789661    11
nprobe=128,ht=64                         0.4296 0.9023 0.9963      0.11868     1242905973    3
```

</details>
<details><summary>`OPQ16_64,IVF4096_HNSW32,PQ16` </summary>
Index size 26283568

 code_size 16

 log filename: autotune.dbdeep1M.OPQ16_64_IVF4096_HNSW32_PQ16.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3951 0.7499 0.7826      0.00790       32268847    38
nprobe=1,quantizer_efSearch=4,ht=60      0.2153 0.3312 0.3346      0.00206        4483943    146
nprobe=2,quantizer_efSearch=4,ht=56      0.2713 0.4360 0.4419      0.00209        8761310    144
nprobe=2,quantizer_efSearch=8,ht=56      0.2839 0.4538 0.4598      0.00221        8703105    136
nprobe=4,quantizer_efSearch=4,ht=128     0.3348 0.6049 0.6245      0.00233       16949535    129
nprobe=4,quantizer_efSearch=8,ht=56      0.3416 0.5868 0.5980      0.00330       16874273    92
nprobe=4,quantizer_efSearch=64,ht=62     0.3545 0.6358 0.6547      0.00531       16782043    57
nprobe=4,quantizer_efSearch=64,ht=64     0.3554 0.6395 0.6593      0.00543       16782043    56
nprobe=8,quantizer_efSearch=4,ht=56      0.3764 0.6812 0.7007      0.00549       32529553    55
nprobe=8,quantizer_efSearch=32,ht=56     0.3845 0.6973 0.7173      0.00621       32287637    49
nprobe=8,quantizer_efSearch=32,ht=60     0.3945 0.7421 0.7730      0.00678       32287637    45
nprobe=8,quantizer_efSearch=64,ht=62     0.3951 0.7499 0.7826      0.00789       32268847    39
nprobe=16,quantizer_efSearch=8,ht=54     0.3963 0.7120 0.7325      0.00985       62053794    31
nprobe=16,quantizer_efSearch=16,ht=58    0.4122 0.8007 0.8398      0.01066       61881506    29
nprobe=16,quantizer_efSearch=16,ht=60    0.4156 0.8166 0.8633      0.01113       61881506    27
nprobe=16,quantizer_efSearch=64,ht=64    0.4187 0.8304 0.8829      0.01326       61642523    23
nprobe=32,quantizer_efSearch=32,ht=58    0.4240 0.8465 0.8976      0.02007      116713930    15
nprobe=128,quantizer_efSearch=256,ht=128 0.4355 0.9052 0.9916      0.03462      408003591    9
nprobe=512,quantizer_efSearch=512,ht=128 0.4357 0.9084 0.9982      0.11781     1462639679    3
nprobe=1024,quantizer_efSearch=128,ht=64 0.4359 0.9068 0.9943      0.58371     2493562692    1
```

</details>
<details><summary>`OPQ32_64,IVF1024_HNSW32,PQ32x4fs` </summary>
Index size 24833484

 code_size 16

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_HNSW32_PQ32x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                0.2383 0.5238 0.6031      0.00134       25337338    224
nprobe=1,quantizer_efSearch=4   0.1959 0.3982 0.4445      0.00114       12959451    263
nprobe=1,quantizer_efSearch=8   0.1988 0.4039 0.4506      0.00130       12958910    232
nprobe=2,quantizer_efSearch=4   0.2383 0.5238 0.6031      0.00137       25337338    220
nprobe=2,quantizer_efSearch=8   0.2433 0.5331 0.6133      0.00153       25344174    197
nprobe=4,quantizer_efSearch=4   0.2694 0.6262 0.7466      0.00169       49390223    178
nprobe=4,quantizer_efSearch=8   0.2756 0.6426 0.7658      0.00185       49373950    163
nprobe=4,quantizer_efSearch=16  0.2775 0.6450 0.7687      0.00214       49351318    141
nprobe=8,quantizer_efSearch=4   0.2945 0.7116 0.8690      0.00221       95148210    137
nprobe=8,quantizer_efSearch=8   0.2968 0.7161 0.8749      0.00229       95055495    131
nprobe=8,quantizer_efSearch=16  0.2984 0.7197 0.8794      0.00271       94979320    111
nprobe=8,quantizer_efSearch=32  0.2985 0.7202 0.8799      0.00327       94946553    92
nprobe=16,quantizer_efSearch=4  0.3044 0.7436 0.9196      0.00314      181870950    96
nprobe=16,quantizer_efSearch=8  0.3084 0.7534 0.9334      0.00321      181602595    94
nprobe=16,quantizer_efSearch=16 0.3093 0.7552 0.9365      0.00354      181437343    85
nprobe=16,quantizer_efSearch=32 0.3096 0.7556 0.9376      0.00397      181331198    76
nprobe=32,quantizer_efSearch=8  0.3123 0.7672 0.9595      0.00459      344180544    66
nprobe=32,quantizer_efSearch=16 0.3137 0.7720 0.9660      0.00483      343728539    63
nprobe=32,quantizer_efSearch=32 0.3140 0.7730 0.9676      0.00521      343387151    58
nprobe=64,quantizer_efSearch=16 0.3142 0.7793 0.9771      0.00727      652393682    42
nprobe=64,quantizer_efSearch=32 0.3144 0.7802 0.9788      0.00779      651158591    39
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fs` </summary>
Index size 24553751

 code_size 16

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2443 0.5348 0.6151      0.00077       25328878    392
nprobe=2                                 0.2443 0.5348 0.6151      0.00074       25328878    408
nprobe=4                                 0.2769 0.6448 0.7685      0.00095       49336048    318
nprobe=8                                 0.2983 0.7199 0.8796      0.00123       94928819    245
nprobe=16                                0.3093 0.7555 0.9376      0.00170      181284631    177
nprobe=32                                0.3139 0.7732 0.9679      0.00245      343278394    123
nprobe=64                                0.3144 0.7802 0.9789      0.00402      650621360    75
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fsr` </summary>
Index size 24554775

 code_size 16

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3196 0.5951 0.6258      0.00090       23536247    333
nprobe=1                                 0.2531 0.4442 0.4625      0.00073       12090236    412
nprobe=2                                 0.3196 0.5951 0.6258      0.00090       23536247    335
nprobe=4                                 0.3729 0.7256 0.7756      0.00127       45949821    238
nprobe=8                                 0.3974 0.8117 0.8813      0.00194       89055363    155
nprobe=16                                0.4101 0.8651 0.9499      0.00310      170778033    97
nprobe=32                                0.4163 0.8858 0.9815      0.01406      324754821    22
nprobe=64                                0.4171 0.8917 0.9926      0.02618      618974813    12
```

</details>
<details><summary>`OPQ32_64,IVF4096_HNSW32,PQ32x4fs` </summary>
Index size 27300428

 code_size 16

 log filename: autotune.dbdeep1M.OPQ32_64_IVF4096_HNSW32_PQ32x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3047 0.7239 0.8792      0.00215       61716809    140
nprobe=2,quantizer_efSearch=8            0.2230 0.4539 0.5051      0.00081        8714007    372
nprobe=4,quantizer_efSearch=4            0.2459 0.5395 0.6171      0.00081       16979319    369
nprobe=4,quantizer_efSearch=8            0.2607 0.5714 0.6558      0.00096       16953611    312
nprobe=8,quantizer_efSearch=4            0.2835 0.6479 0.7672      0.00109       32658030    277
nprobe=8,quantizer_efSearch=8            0.2861 0.6570 0.7777      0.00114       32604935    263
nprobe=8,quantizer_efSearch=16           0.2891 0.6645 0.7872      0.00142       32450927    211
nprobe=16,quantizer_efSearch=4           0.2957 0.6998 0.8466      0.00141       62359676    213
nprobe=16,quantizer_efSearch=8           0.3017 0.7174 0.8708      0.00151       62140825    199
nprobe=16,quantizer_efSearch=16          0.3042 0.7227 0.8784      0.00167       61908292    180
nprobe=16,quantizer_efSearch=32          0.3047 0.7239 0.8792      0.00212       61716809    142
nprobe=32,quantizer_efSearch=16          0.3111 0.7534 0.9330      0.00248      117117325    121
nprobe=64,quantizer_efSearch=16          0.3121 0.7673 0.9581      0.00303      219876427    100
nprobe=64,quantizer_efSearch=32          0.3131 0.7721 0.9649      0.00333      218913715    91
nprobe=128,quantizer_efSearch=32         0.3142 0.7772 0.9753      0.00488      410236294    62
nprobe=128,quantizer_efSearch=64         0.3147 0.7790 0.9776      0.00555      409051613    55
```

</details>
<details><summary>`OPQ32_64,IVF4096_HNSW32,PQ32x4fsr` </summary>
Index size 27265100

 code_size 16

 log filename: autotune.dbdeep1M.OPQ32_64_IVF4096_HNSW32_PQ32x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4108 0.8317 0.8915      0.00363       44579082    83
nprobe=2,quantizer_efSearch=4            0.2771 0.4734 0.4876      0.00078        5855693    385
nprobe=2,quantizer_efSearch=8            0.2957 0.5025 0.5178      0.00089        5890453    336
nprobe=4,quantizer_efSearch=4            0.3284 0.6021 0.6252      0.00113       11577881    266
nprobe=4,quantizer_efSearch=8            0.3497 0.6429 0.6682      0.00124       11655147    242
nprobe=4,quantizer_efSearch=16           0.3505 0.6460 0.6715      0.00148       11649871    203
nprobe=4,quantizer_efSearch=32           0.3512 0.6469 0.6723      0.00183       11646737    164
nprobe=8,quantizer_efSearch=8            0.3852 0.7501 0.7917      0.00183       22893782    164
nprobe=8,quantizer_efSearch=16           0.3865 0.7555 0.7978      0.00201       22879915    150
nprobe=16,quantizer_efSearch=8           0.4069 0.8253 0.8841      0.00289       44669552    104
nprobe=16,quantizer_efSearch=16          0.4093 0.8307 0.8903      0.00294       44616484    102
nprobe=16,quantizer_efSearch=32          0.4108 0.8317 0.8915      0.00337       44579082    90
nprobe=16,quantizer_efSearch=128         0.4109 0.8319 0.8919      0.00586       44567929    52
nprobe=32,quantizer_efSearch=8           0.4141 0.8608 0.9314      0.01352       86881231    23
nprobe=32,quantizer_efSearch=16          0.4187 0.8714 0.9439      0.01366       86724281    22
nprobe=32,quantizer_efSearch=32          0.4192 0.8734 0.9465      0.01409       86624662    22
nprobe=32,quantizer_efSearch=64          0.4194 0.8730 0.9463      0.01454       86582284    21
nprobe=32,quantizer_efSearch=128         0.4195 0.8732 0.9465      0.01635       86575431    19
nprobe=64,quantizer_efSearch=32          0.4210 0.8934 0.9778      0.02716      167010118    12
nprobe=64,quantizer_efSearch=64          0.4211 0.8940 0.9780      0.02750      166837943    11
nprobe=64,quantizer_efSearch=128         0.4212 0.8942 0.9782      0.02916      166797535    11
nprobe=128,quantizer_efSearch=32         0.4240 0.9009 0.9907      0.05861      319455985    6
nprobe=128,quantizer_efSearch=64         0.4251 0.9018 0.9923      0.05907      318859325    6
nprobe=128,quantizer_efSearch=256        0.4257 0.9020 0.9928      0.06602      318536786    5
```

</details>
<details><summary>`PCAR16,IVF1024_HNSW32,SQ8` </summary>
Index size 24396181

 code_size 16

 log filename: autotune.dbdeep1M.PCAR16_IVF1024_HNSW32_SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                0.0662 0.2319 0.4366      0.00329       22161474    92
nprobe=1,quantizer_efSearch=8   0.0592 0.1913 0.3231      0.00263       11211507    115
nprobe=2,quantizer_efSearch=4   0.0662 0.2319 0.4366      0.00329       22161474    92
nprobe=2,quantizer_efSearch=8   0.0671 0.2348 0.4405      0.00343       22136124    88
nprobe=2,quantizer_efSearch=16  0.0672 0.2346 0.4401      0.00371       22129492    81
nprobe=4,quantizer_efSearch=4   0.0719 0.2609 0.5338      0.00468       43688185    65
nprobe=4,quantizer_efSearch=8   0.0729 0.2645 0.5398      0.00473       43634372    64
nprobe=4,quantizer_efSearch=16  0.0732 0.2650 0.5400      0.00498       43616379    61
nprobe=8,quantizer_efSearch=4   0.0754 0.2757 0.5982      0.00711       85766133    43
nprobe=8,quantizer_efSearch=16  0.0757 0.2770 0.6009      0.00745       85708609    41
nprobe=8,quantizer_efSearch=32  0.0758 0.2771 0.6009      0.00774       85700309    39
nprobe=16,quantizer_efSearch=4  0.0759 0.2812 0.6250      0.01155      167712666    26
nprobe=16,quantizer_efSearch=16 0.0762 0.2836 0.6295      0.01177      167529256    26
nprobe=16,quantizer_efSearch=32 0.0763 0.2837 0.6295      0.01222      167508629    25
```

</details>
<details><summary>`PCAR16,IVF1024,SQ8` </summary>
Index size 24117984

 code_size 16

 log filename: autotune.dbdeep1M.PCAR16_IVF1024_SQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0672 0.2346 0.4401      0.00285       22127567    106
nprobe=1                                 0.0592 0.1912 0.3230      0.00210       11209857    143
nprobe=2                                 0.0672 0.2346 0.4401      0.00198       22127567    152
nprobe=4                                 0.0732 0.2650 0.5400      0.00269       43612231    112
nprobe=8                                 0.0758 0.2771 0.6009      0.00393       85698237    77
nprobe=16                                0.0763 0.2837 0.6294      0.00651      167503051    47
```

</details>
<details><summary>`PCAR16,IVF4096_HNSW32,SQ8` </summary>
Index size 25451029

 code_size 16

 log filename: autotune.dbdeep1M.PCAR16_IVF4096_HNSW32_SQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0762 0.2792 0.5952      0.00343       47419855    88
nprobe=4,quantizer_efSearch=4            0.0672 0.2330 0.4336      0.00172       12390334    175
nprobe=4,quantizer_efSearch=8            0.0696 0.2404 0.4460      0.00182       12338275    165
nprobe=8,quantizer_efSearch=4            0.0739 0.2644 0.5313      0.00202       24279319    149
nprobe=8,quantizer_efSearch=8            0.0743 0.2659 0.5352      0.00206       24245464    146
nprobe=8,quantizer_efSearch=32           0.0745 0.2655 0.5354      0.00255       24194918    118
nprobe=16,quantizer_efSearch=4           0.0760 0.2762 0.5868      0.00278       47609393    108
nprobe=16,quantizer_efSearch=32          0.0762 0.2792 0.5952      0.00337       47419855    90
nprobe=32,quantizer_efSearch=4           0.0764 0.2803 0.6122      0.00455       92848903    66
nprobe=128,quantizer_efSearch=8          0.0765 0.2841 0.6335      0.01244      341281983    25
nprobe=128,quantizer_efSearch=32         0.0766 0.2851 0.6406      0.01319      347710942    23
```

</details>
<details><summary>`PCAR32,IVF1024_HNSW32,SQ4` </summary>
Index size 24468053

 code_size 16

 log filename: autotune.dbdeep1M.PCAR32_IVF1024_HNSW32_SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                0.1859 0.4636 0.5628      0.00495       23434422    61
nprobe=1,quantizer_efSearch=8   0.1542 0.3547 0.4130      0.00342       11940275    88
nprobe=2,quantizer_efSearch=4   0.1859 0.4636 0.5628      0.00488       23434422    62
nprobe=2,quantizer_efSearch=8   0.1875 0.4682 0.5693      0.00502       23406917    60
nprobe=4,quantizer_efSearch=4   0.2074 0.5491 0.7065      0.00772       45873748    39
nprobe=4,quantizer_efSearch=16  0.2096 0.5581 0.7190      0.00824       45799774    37
nprobe=4,quantizer_efSearch=32  0.2098 0.5581 0.7190      0.00855       45794344    36
nprobe=8,quantizer_efSearch=4   0.2204 0.6087 0.8233      0.01313       89260959    23
nprobe=8,quantizer_efSearch=16  0.2220 0.6146 0.8324      0.01348       89153144    23
nprobe=8,quantizer_efSearch=32  0.2225 0.6155 0.8333      0.01393       89135871    22
nprobe=16,quantizer_efSearch=8  0.2265 0.6428 0.8974      0.02363      172478059    13
nprobe=16,quantizer_efSearch=16 0.2266 0.6441 0.8994      0.02371      172344917    13
nprobe=16,quantizer_efSearch=32 0.2269 0.6447 0.9000      0.02452      172301962    13
nprobe=32,quantizer_efSearch=8  0.2274 0.6559 0.9308      0.04403      330590712    7
nprobe=32,quantizer_efSearch=16 0.2278 0.6590 0.9362      0.04296      330080114    7
nprobe=64,quantizer_efSearch=16 0.2285 0.6632 0.9484      0.07971      633704818    4
nprobe=64,quantizer_efSearch=32 0.2287 0.6647 0.9503      0.08726      632549446    4
```

</details>
<details><summary>`PCAR32,IVF1024,SQ4` </summary>
Index size 24189856

 code_size 16

 log filename: autotune.dbdeep1M.PCAR32_IVF1024_SQ4.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1873 0.4683 0.5694      0.00280       23386342    107
nprobe=2                                 0.1873 0.4683 0.5694      0.00280       23386342    108
nprobe=4                                 0.2098 0.5581 0.7189      0.00448       45791428    67
nprobe=8                                 0.2225 0.6155 0.8332      0.01334       89128799    23
nprobe=16                                0.2270 0.6448 0.9002      0.01561      172279830    20
nprobe=32                                0.2280 0.6603 0.9379      0.02482      329780601    13
nprobe=64                                0.2289 0.6649 0.9508      0.04532      632250711    7
```

</details>
<details><summary>`PCAR32,IVF4096_HNSW32,SQ4` </summary>
Index size 25719509

 code_size 16

 log filename: autotune.dbdeep1M.PCAR32_IVF4096_HNSW32_SQ4.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2221 0.5745 0.7283      0.00601       27434235    50
nprobe=2,quantizer_efSearch=8            0.1742 0.3895 0.4444      0.00183        7266338    165
nprobe=2,quantizer_efSearch=16           0.1747 0.3902 0.4450      0.00202        7245262    149
nprobe=4,quantizer_efSearch=4            0.1925 0.4728 0.5681      0.00209       14255931    144
nprobe=4,quantizer_efSearch=8            0.2007 0.4927 0.5936      0.00220       14189400    137
nprobe=4,quantizer_efSearch=16           0.2020 0.4945 0.5957      0.00238       14134080    126
nprobe=4,quantizer_efSearch=32           0.2024 0.4946 0.5956      0.00271       14119963    111
nprobe=8,quantizer_efSearch=4            0.2168 0.5628 0.7128      0.00311       27635813    97
nprobe=8,quantizer_efSearch=8            0.2184 0.5690 0.7207      0.00316       27586150    95
nprobe=8,quantizer_efSearch=16           0.2215 0.5732 0.7269      0.00334       27479849    90
nprobe=8,quantizer_efSearch=32           0.2220 0.5743 0.7281      0.00368       27442228    82
nprobe=8,quantizer_efSearch=64           0.2221 0.5745 0.7283      0.00436       27435148    69
nprobe=16,quantizer_efSearch=4           0.2268 0.6095 0.8074      0.00497       53378678    61
nprobe=16,quantizer_efSearch=8           0.2284 0.6214 0.8261      0.00502       53200959    60
nprobe=16,quantizer_efSearch=16          0.2292 0.6248 0.8306      0.00511       53046859    59
nprobe=32,quantizer_efSearch=4           0.2298 0.6284 0.8553      0.00836      102295930    36
nprobe=32,quantizer_efSearch=8           0.2330 0.6467 0.8841      0.00844      102124974    36
nprobe=32,quantizer_efSearch=16          0.2339 0.6534 0.8965      0.00859      101740061    35
nprobe=32,quantizer_efSearch=32          0.2340 0.6550 0.8999      0.00882      101531872    35
nprobe=32,quantizer_efSearch=64          0.2341 0.6555 0.9007      0.00950      101436338    32
nprobe=64,quantizer_efSearch=16          0.2346 0.6660 0.9295      0.01508      194455574    20
nprobe=128,quantizer_efSearch=16         0.2349 0.6679 0.9398      0.02706      367988787    12
nprobe=256,quantizer_efSearch=16         0.2350 0.6678 0.9408      0.04719      645642503    7
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF1024,PQ48x4fs` </summary>
Index size 32783824

 code_size 24

 log filename: autotune.dbdeep1M.IVF1024_PQ48x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3270 0.6053 0.6441      0.00085       24359631    354
nprobe=1                                 0.2568 0.4492 0.4728      0.00071       12514199    426
nprobe=2                                 0.3270 0.6053 0.6441      0.00086       24359631    351
nprobe=4                                 0.3760 0.7333 0.7924      0.00108       47374981    279
nprobe=8                                 0.4072 0.8185 0.8938      0.00140       91666359    215
nprobe=16                                0.4219 0.8651 0.9532      0.00201      175416097    150
nprobe=32                                0.4276 0.8845 0.9808      0.00302      332699035    100
nprobe=64                                0.4301 0.8930 0.9938      0.00772      632391204    39
nprobe=128                               0.4314 0.8955 0.9972      0.00759     1208396470    40
```

</details>
<details><summary>`IVF1024,PQ48x4fsr` </summary>
Index size 32783824

 code_size 24

 log filename: autotune.dbdeep1M.IVF1024_PQ48x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4011 0.6338 0.6450      0.00105       24359631    287
nprobe=1                                 0.3119 0.4671 0.4734      0.00077       12514199    388
nprobe=2                                 0.4011 0.6338 0.6450      0.00108       24359631    277
nprobe=4                                 0.4687 0.7765 0.7934      0.00154       47374981    195
nprobe=8                                 0.5080 0.8719 0.8951      0.00258       91666359    117
nprobe=16                                0.5277 0.9268 0.9546      0.00440      175416097    69
nprobe=32                                0.5331 0.9503 0.9822      0.02144      332699035    14
nprobe=64                                0.5385 0.9616 0.9951      0.04304      632391204    7
nprobe=256                               0.5397 0.9633 0.9991      0.18751     2315638246    2
```

</details>
<details><summary>`IVF4096_HNSW32,PQ48x4fs` </summary>
Index size 36300293

 code_size 24

 log filename: autotune.dbdeep1M.IVF4096_HNSW32_PQ48x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4069 0.8229 0.8972      0.00236       45568234    128
nprobe=1,quantizer_efSearch=4            0.2128 0.3418 0.3540      0.00064        3056128    472
nprobe=2,quantizer_efSearch=4            0.2762 0.4811 0.5028      0.00072        6039381    416
nprobe=2,quantizer_efSearch=8            0.2880 0.5037 0.5258      0.00086        6065111    351
nprobe=4,quantizer_efSearch=4            0.3272 0.6109 0.6483      0.00087       11917876    346
nprobe=4,quantizer_efSearch=8            0.3431 0.6419 0.6807      0.00103       11954140    291
nprobe=8,quantizer_efSearch=4            0.3753 0.7370 0.7935      0.00118       23455711    256
nprobe=8,quantizer_efSearch=8            0.3794 0.7451 0.8023      0.00121       23435082    248
nprobe=16,quantizer_efSearch=4           0.3951 0.7982 0.8696      0.00153       45722741    197
nprobe=16,quantizer_efSearch=8           0.4027 0.8144 0.8876      0.00164       45672284    183
nprobe=16,quantizer_efSearch=16          0.4062 0.8208 0.8946      0.00176       45605554    171
nprobe=32,quantizer_efSearch=8           0.4095 0.8463 0.9321      0.00207       88721841    146
nprobe=32,quantizer_efSearch=16          0.4147 0.8573 0.9448      0.00241       88541157    125
nprobe=64,quantizer_efSearch=32          0.4198 0.8814 0.9766      0.00350      169917354    86
nprobe=64,quantizer_efSearch=64          0.4203 0.8828 0.9779      0.00412      169718272    73
nprobe=128,quantizer_efSearch=32         0.4215 0.8898 0.9881      0.00530      324574430    57
nprobe=128,quantizer_efSearch=64         0.4225 0.8923 0.9908      0.00604      323800051    50
nprobe=128,quantizer_efSearch=128        0.4231 0.8931 0.9917      0.00726      323451571    42
nprobe=128,quantizer_efSearch=256        0.4232 0.8931 0.9917      0.01244      323397349    25
```

</details>
<details><summary>`IVF4096_HNSW32,PQ48x4fsr` </summary>
Index size 36300293

 code_size 24

 log filename: autotune.dbdeep1M.IVF4096_HNSW32_PQ48x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5229 0.8824 0.8987      0.00513       45566748    59
nprobe=1,quantizer_efSearch=4            0.2574 0.3534 0.3548      0.00077        3066858    390
nprobe=1,quantizer_efSearch=8            0.2677 0.3681 0.3695      0.00090        3070513    335
nprobe=2,quantizer_efSearch=4            0.3417 0.4989 0.5023      0.00105        6059672    285
nprobe=2,quantizer_efSearch=8            0.3573 0.5237 0.5270      0.00119        6069283    253
nprobe=2,quantizer_efSearch=16           0.3587 0.5253 0.5286      0.00143        6063359    210
nprobe=4,quantizer_efSearch=8            0.4373 0.6757 0.6821      0.00182       11955821    165
nprobe=4,quantizer_efSearch=16           0.4398 0.6793 0.6857      0.00207       11946902    145
nprobe=8,quantizer_efSearch=4            0.4805 0.7856 0.7955      0.00274       23464530    110
nprobe=8,quantizer_efSearch=8            0.4850 0.7939 0.8040      0.00280       23440690    107
nprobe=8,quantizer_efSearch=32           0.4906 0.8036 0.8135      0.00354       23402342    85
nprobe=16,quantizer_efSearch=4           0.5103 0.8559 0.8713      0.00462       45725410    65
nprobe=16,quantizer_efSearch=8           0.5180 0.8733 0.8895      0.00458       45664756    66
nprobe=16,quantizer_efSearch=16          0.5210 0.8794 0.8956      0.00438       45605956    69
nprobe=16,quantizer_efSearch=32          0.5229 0.8824 0.8987      0.00484       45566748    62
nprobe=16,quantizer_efSearch=64          0.5232 0.8831 0.8994      0.00558       45555037    54
nprobe=32,quantizer_efSearch=32          0.5380 0.9277 0.9498      0.02074       88433054    15
nprobe=32,quantizer_efSearch=64          0.5383 0.9284 0.9506      0.02195       88388565    14
nprobe=64,quantizer_efSearch=8           0.5407 0.9351 0.9576      0.03880      170683847    8
nprobe=64,quantizer_efSearch=64          0.5484 0.9561 0.9805      0.04099      169722361    8
nprobe=128,quantizer_efSearch=256        0.5504 0.9673 0.9943      0.09920      323398925    4
nprobe=1024,quantizer_efSearch=128       0.5514 0.9712 0.9996      0.66277     2115967550    1
```

</details>
<details><summary>`OPQ32_128,IVF1024_HNSW32,PQ32` </summary>
Index size 40991152

 code_size 32

 log filename: autotune.dbdeep1M.OPQ32_128_IVF1024_HNSW32_PQ32.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6878 0.9763 0.9793      0.15664      666814377    2
nprobe=1,quantizer_efSearch=8,ht=90      0.0944 0.1089 0.1095      0.00375       13250966    80
nprobe=1,quantizer_efSearch=32,ht=94     0.1395 0.1612 0.1616      0.00506       13244326    60
nprobe=1,quantizer_efSearch=32,ht=102    0.2515 0.2950 0.2956      0.00538       13244326    56
nprobe=1,quantizer_efSearch=8,ht=120     0.3660 0.4558 0.4563      0.00543       13250966    56
nprobe=1,quantizer_efSearch=8,ht=126     0.3697 0.4619 0.4626      0.00631       13250966    48
nprobe=1,quantizer_efSearch=16,ht=124    0.3707 0.4633 0.4637      0.00647       13247531    47
nprobe=1,quantizer_efSearch=16,ht=128    0.3713 0.4644 0.4648      0.00674       13247531    45
nprobe=2,quantizer_efSearch=8,ht=106     0.3943 0.4828 0.4835      0.00626       25955889    48
nprobe=2,quantizer_efSearch=4,ht=124     0.4716 0.6109 0.6117      0.00911       25953509    33
nprobe=2,quantizer_efSearch=128,ht=116   0.4751 0.6091 0.6099      0.01278       25931913    24
nprobe=4,quantizer_efSearch=32,ht=114    0.5592 0.7389 0.7402      0.01300       50540304    24
nprobe=4,quantizer_efSearch=32,ht=118    0.5707 0.7623 0.7639      0.01430       50540304    21
nprobe=4,quantizer_efSearch=8,ht=120     0.5716 0.7662 0.7677      0.01396       50605709    22
nprobe=4,quantizer_efSearch=16,ht=126    0.5773 0.7774 0.7793      0.01699       50565654    18
nprobe=4,quantizer_efSearch=64,ht=128    0.5775 0.7784 0.7800      0.01956       50535151    16
nprobe=8,quantizer_efSearch=8,ht=112     0.6074 0.8122 0.8135      0.02023       97591525    15
nprobe=8,quantizer_efSearch=16,ht=256    0.6407 0.8880 0.8901      0.02066       97539700    15
nprobe=16,quantizer_efSearch=64,ht=118   0.6660 0.9256 0.9276      0.04409      186146482    7
nprobe=16,quantizer_efSearch=64,ht=120   0.6689 0.9348 0.9368      0.04661      186146482    7
nprobe=16,quantizer_efSearch=32,ht=122   0.6712 0.9405 0.9426      0.04820      186158245    7
nprobe=16,quantizer_efSearch=64,ht=126   0.6723 0.9450 0.9475      0.05606      186146482    6
nprobe=32,quantizer_efSearch=64,ht=256   0.6885 0.9801 0.9832      0.06548      351952660    5
nprobe=64,quantizer_efSearch=32,ht=120   0.6898 0.9796 0.9827      0.15731      665607965    2
nprobe=64,quantizer_efSearch=512,ht=124  0.6936 0.9892 0.9926      0.21560      664944038    2
nprobe=128,quantizer_efSearch=256,ht=256 0.6952 0.9950 0.9992      0.23071     1261297395    2
nprobe=256,quantizer_efSearch=64,ht=124  0.6955 0.9935 0.9972      0.66706     2417663616    1
nprobe=512,quantizer_efSearch=512,ht=256 0.6956 0.9958 1.0000      0.83871     4660182295    1
```

</details>
<details><summary>`OPQ32_128,IVF1024,PQ32` </summary>
Index size 40712955

 code_size 32

 log filename: autotune.dbdeep1M.OPQ32_128_IVF1024_PQ32.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3579 0.4392 0.4394      0.00262       13238549    115
nprobe=1,ht=110                          0.3374 0.4099 0.4102      0.00238       13238549    126
nprobe=1,ht=112                          0.3479 0.4259 0.4261      0.00246       13238549    122
nprobe=1,ht=114                          0.3579 0.4392 0.4394      0.00259       13238549    116
nprobe=1,ht=122                          0.3716 0.4622 0.4625      0.00313       13238549    96
nprobe=1,ht=124                          0.3721 0.4634 0.4637      0.00324       13238549    93
nprobe=1,ht=126                          0.3725 0.4641 0.4644      0.00344       13238549    88
nprobe=2,ht=112                          0.4553 0.5768 0.5772      0.00556       25926923    54
nprobe=2,ht=118                          0.4825 0.6193 0.6197      0.00593       25926923    51
nprobe=2,ht=120                          0.4844 0.6241 0.6246      0.00460       25926923    66
nprobe=2,ht=122                          0.4852 0.6267 0.6272      0.00489       25926923    62
nprobe=2,ht=124                          0.4861 0.6286 0.6291      0.00511       25926923    59
nprobe=2,ht=128                          0.4868 0.6299 0.6305      0.00553       25926923    55
nprobe=4,ht=112                          0.5457 0.7141 0.7147      0.00684       50522680    44
nprobe=4,ht=256                          0.5799 0.7798 0.7806      0.00669       50522680    45
nprobe=8,ht=256                          0.6421 0.8883 0.8904      0.01103       97467270    28
nprobe=16,ht=256                         0.6744 0.9463 0.9488      0.01900      186077445    16
nprobe=32,ht=256                         0.6904 0.9808 0.9834      0.03954      351797967    8
nprobe=64,ht=256                         0.6954 0.9922 0.9954      0.06231      664637570    5
nprobe=128,ht=256                        0.6970 0.9959 0.9992      0.17763     1260718002    2
nprobe=256,ht=256                        0.6975 0.9967 1.0000      0.22104     2403209576    2
```

</details>
<details><summary>`OPQ32_128,IVF4096_HNSW32,PQ32` </summary>
Index size 43422256

 code_size 32

 log filename: autotune.dbdeep1M.OPQ32_128_IVF4096_HNSW32_PQ32.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.5171 0.6696 0.6700      0.03317       18391473    10
nprobe=2,quantizer_efSearch=8,ht=256      0.4111 0.5164 0.5169      0.00291        9527234    104
nprobe=4,quantizer_efSearch=16,ht=256     0.5197 0.6747 0.6753      0.00432       18422497    70
nprobe=8,quantizer_efSearch=8,ht=256      0.5942 0.7938 0.7951      0.00621       35579109    49
nprobe=8,quantizer_efSearch=128,ht=256    0.6004 0.8041 0.8053      0.00998       35322890    31
nprobe=16,quantizer_efSearch=16,ht=256    0.6473 0.8881 0.8905      0.01081       67318268    28
nprobe=16,quantizer_efSearch=256,ht=124   0.6477 0.8878 0.8896      0.03125       67040389    10
nprobe=16,quantizer_efSearch=256,ht=128   0.6484 0.8900 0.8923      0.03237       67040389    10
nprobe=32,quantizer_efSearch=64,ht=120    0.6722 0.9341 0.9362      0.04172      126106788    8
nprobe=32,quantizer_efSearch=64,ht=124    0.6772 0.9450 0.9473      0.04302      126106788    7
nprobe=32,quantizer_efSearch=64,ht=128    0.6778 0.9471 0.9500      0.04643      126106788    7
nprobe=64,quantizer_efSearch=32,ht=122    0.6874 0.9688 0.9718      0.08408      236349944    4
nprobe=64,quantizer_efSearch=64,ht=122    0.6881 0.9702 0.9732      0.07992      235691858    4
nprobe=64,quantizer_efSearch=64,ht=128    0.6907 0.9773 0.9809      0.08464      235691858    4
nprobe=64,quantizer_efSearch=256,ht=128   0.6908 0.9777 0.9813      0.09237      235380866    4
nprobe=256,quantizer_efSearch=512,ht=256  0.6973 0.9939 0.9988      0.15057      823235790    2
nprobe=512,quantizer_efSearch=128,ht=256  0.6975 0.9946 0.9995      0.24279     1555309929    2
nprobe=1024,quantizer_efSearch=512,ht=256 0.6978 0.9950 1.0000      0.50182     2966320245    1
```

</details>
<details><summary>`OPQ64_128,IVF1024_HNSW32,PQ64x4fs` </summary>
Index size 41394636

 code_size 32

 log filename: autotune.dbdeep1M.OPQ64_128_IVF1024_HNSW32_PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                 0.3467 0.5924 0.6138      0.00179       25922253    168
nprobe=1,quantizer_efSearch=4    0.2741 0.4395 0.4523      0.00152       13231788    198
nprobe=1,quantizer_efSearch=8    0.2799 0.4496 0.4625      0.00176       13247108    171
nprobe=2,quantizer_efSearch=4    0.3467 0.5924 0.6138      0.00179       25922253    168
nprobe=2,quantizer_efSearch=8    0.3529 0.6059 0.6273      0.00204       25948558    148
nprobe=4,quantizer_efSearch=4    0.3975 0.7180 0.7531      0.00220       50522147    137
nprobe=4,quantizer_efSearch=8    0.4101 0.7407 0.7769      0.00245       50600906    123
nprobe=4,quantizer_efSearch=16   0.4121 0.7437 0.7800      0.00287       50566938    105
nprobe=8,quantizer_efSearch=4    0.4445 0.8288 0.8762      0.00303       97601837    100
nprobe=8,quantizer_efSearch=8    0.4489 0.8368 0.8846      0.00315       97594276    96
nprobe=8,quantizer_efSearch=16   0.4509 0.8415 0.8894      0.00358       97537290    84
nprobe=8,quantizer_efSearch=32   0.4517 0.8424 0.8903      0.00427       97504453    71
nprobe=16,quantizer_efSearch=4   0.4605 0.8735 0.9276      0.00428      186445708    71
nprobe=16,quantizer_efSearch=8   0.4672 0.8885 0.9436      0.00447      186370913    68
nprobe=16,quantizer_efSearch=16  0.4693 0.8917 0.9476      0.00475      186257999    64
nprobe=16,quantizer_efSearch=32  0.4694 0.8929 0.9487      0.00545      186157429    56
nprobe=32,quantizer_efSearch=8   0.4763 0.9134 0.9733      0.00659      352667242    46
nprobe=32,quantizer_efSearch=16  0.4801 0.9192 0.9806      0.00691      352357139    44
nprobe=32,quantizer_efSearch=32  0.4810 0.9214 0.9829      0.00742      352053503    41
nprobe=64,quantizer_efSearch=16  0.4831 0.9280 0.9916      0.01068      666752125    29
nprobe=64,quantizer_efSearch=32  0.4842 0.9311 0.9946      0.01133      665629764    27
nprobe=64,quantizer_efSearch=128 0.4843 0.9315 0.9950      0.01535      664960470    20
```

</details>
<details><summary>`OPQ64_128,IVF1024,PQ64x4fs` </summary>
Index size 41110295

 code_size 32

 log filename: autotune.dbdeep1M.OPQ64_128_IVF1024_PQ64x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4124 0.7440 0.7804      0.00142       50522646    212
nprobe=4                                 0.4124 0.7440 0.7804      0.00119       50522646    252
nprobe=8                                 0.4517 0.8423 0.8902      0.00162       97467356    185
nprobe=16                                0.4695 0.8928 0.9486      0.00226      186078317    133
nprobe=32                                0.4809 0.9215 0.9830      0.00364      351798577    83
nprobe=64                                0.4843 0.9315 0.9950      0.00564      664638089    54
```

</details>
<details><summary>`OPQ64_128,IVF1024,PQ64x4fsr` </summary>
Index size 41094935

 code_size 32

 log filename: autotune.dbdeep1M.OPQ64_128_IVF1024_PQ64x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4725 0.6441 0.6468      0.00129       24366230    233
nprobe=1                                 0.3586 0.4735 0.4748      0.00093       12511981    324
nprobe=2                                 0.4725 0.6441 0.6468      0.00128       24366230    235
nprobe=4                                 0.5590 0.7912 0.7944      0.00196       47389874    154
nprobe=8                                 0.6107 0.8903 0.8951      0.00315       91686133    96
nprobe=16                                0.6370 0.9487 0.9557      0.01423      175411871    22
nprobe=32                                0.6428 0.9748 0.9826      0.02682      332684060    12
nprobe=64                                0.6475 0.9875 0.9956      0.06304      632292298    5
```

</details>
<details><summary>`OPQ64_128,IVF4096_HNSW32,PQ64x4fs` </summary>
Index size 45425228

 code_size 32

 log filename: autotune.dbdeep1M.OPQ64_128_IVF4096_HNSW32_PQ64x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4564 0.8443 0.8928      0.00279       67142375    108
nprobe=4,quantizer_efSearch=8            0.3771 0.6417 0.6676      0.00127       18488785    237
nprobe=8,quantizer_efSearch=4            0.4165 0.7428 0.7793      0.00142       35652963    212
nprobe=8,quantizer_efSearch=8            0.4229 0.7552 0.7925      0.00153       35620660    196
nprobe=8,quantizer_efSearch=16           0.4303 0.7667 0.8042      0.00187       35430319    161
nprobe=16,quantizer_efSearch=4           0.4363 0.8048 0.8512      0.00190       67669144    158
nprobe=16,quantizer_efSearch=8           0.4501 0.8338 0.8814      0.00204       67561069    147
nprobe=16,quantizer_efSearch=16          0.4548 0.8423 0.8902      0.00222       67348946    136
nprobe=16,quantizer_efSearch=32          0.4564 0.8443 0.8928      0.00274       67142375    110
nprobe=32,quantizer_efSearch=8           0.4639 0.8756 0.9311      0.00268      127219991    112
nprobe=32,quantizer_efSearch=16          0.4699 0.8901 0.9463      0.00290      126752842    104
nprobe=32,quantizer_efSearch=32          0.4718 0.8939 0.9503      0.00330      126351255    91
nprobe=64,quantizer_efSearch=16          0.4777 0.9128 0.9731      0.00397      237315020    76
nprobe=64,quantizer_efSearch=32          0.4792 0.9191 0.9804      0.00446      236418090    68
nprobe=64,quantizer_efSearch=64          0.4800 0.9207 0.9818      0.00525      235732765    58
nprobe=128,quantizer_efSearch=32         0.4822 0.9272 0.9906      0.00690      440928457    44
nprobe=128,quantizer_efSearch=64         0.4826 0.9300 0.9936      0.00749      439771825    41
nprobe=256,quantizer_efSearch=64         0.4832 0.9323 0.9971      0.01139      822436254    27
nprobe=256,quantizer_efSearch=128        0.4835 0.9332 0.9981      0.01416      825037373    22
```

</details>
<details><summary>`OPQ64_128,IVF4096_HNSW32,PQ64x4fsr` </summary>
Index size 45389388

 code_size 32

 log filename: autotune.dbdeep1M.OPQ64_128_IVF4096_HNSW32_PQ64x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6205 0.8951 0.8986      0.01580       45565794    19
nprobe=1,quantizer_efSearch=4            0.2836 0.3504 0.3511      0.00094        3053756    320
nprobe=1,quantizer_efSearch=8            0.2981 0.3684 0.3692      0.00109        3070589    277
nprobe=2,quantizer_efSearch=4            0.3827 0.4947 0.4958      0.00118        6029787    254
nprobe=2,quantizer_efSearch=8            0.4051 0.5252 0.5263      0.00134        6067813    224
nprobe=2,quantizer_efSearch=16           0.4075 0.5280 0.5291      0.00162        6064099    186
nprobe=4,quantizer_efSearch=4            0.4720 0.6364 0.6385      0.00174       11882988    173
nprobe=4,quantizer_efSearch=8            0.5023 0.6790 0.6812      0.00185       11957066    163
nprobe=4,quantizer_efSearch=16           0.5064 0.6840 0.6862      0.00219       11947339    138
nprobe=4,quantizer_efSearch=32           0.5072 0.6847 0.6869      0.00270       11942153    112
nprobe=8,quantizer_efSearch=4            0.5672 0.7914 0.7942      0.00303       23460220    100
nprobe=8,quantizer_efSearch=8            0.5734 0.8005 0.8033      0.00301       23442313    100
nprobe=8,quantizer_efSearch=16           0.5801 0.8097 0.8125      0.00326       23415644    92
nprobe=8,quantizer_efSearch=32           0.5812 0.8110 0.8138      0.00381       23402544    79
nprobe=16,quantizer_efSearch=4           0.6013 0.8665 0.8701      0.01406       45714113    22
nprobe=16,quantizer_efSearch=8           0.6127 0.8853 0.8888      0.01510       45668788    20
nprobe=16,quantizer_efSearch=16          0.6180 0.8923 0.8958      0.01513       45603097    20
nprobe=16,quantizer_efSearch=32          0.6205 0.8951 0.8986      0.01582       45565794    19
nprobe=16,quantizer_efSearch=64          0.6209 0.8957 0.8992      0.01659       45555756    19
nprobe=16,quantizer_efSearch=128         0.6210 0.8958 0.8993      0.01853       45553301    17
nprobe=32,quantizer_efSearch=8           0.6315 0.9303 0.9350      0.02716       88716467    12
nprobe=32,quantizer_efSearch=16          0.6397 0.9422 0.9470      0.02907       88540504    11
nprobe=32,quantizer_efSearch=64          0.6424 0.9456 0.9505      0.03075       88392422    10
nprobe=32,quantizer_efSearch=128         0.6428 0.9460 0.9509      0.03239       88383429    10
nprobe=64,quantizer_efSearch=64          0.6562 0.9747 0.9805      0.06749      169721448    5
nprobe=64,quantizer_efSearch=128         0.6566 0.9752 0.9810      0.07044      169672387    5
nprobe=128,quantizer_efSearch=256        0.6579 0.9884 0.9943      0.13307      323402248    3
nprobe=256,quantizer_efSearch=128        0.6591 0.9913 0.9984      0.24825      615981372    2
nprobe=256,quantizer_efSearch=64         0.6592 0.9903 0.9971      0.25026      616553446    2
```

</details>
<details><summary>`PCAR16,IVF1024_HNSW32,SQfp16` </summary>
Index size 40396053

 code_size 32

 log filename: autotune.dbdeep1M.PCAR16_IVF1024_HNSW32_SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                0.0669 0.2301 0.4370      0.00271       22172835    111
nprobe=1,quantizer_efSearch=16  0.0595 0.1905 0.3231      0.00258       11209967    117
nprobe=2,quantizer_efSearch=4   0.0669 0.2301 0.4370      0.00269       22172835    112
nprobe=2,quantizer_efSearch=8   0.0675 0.2334 0.4410      0.00281       22133852    107
nprobe=2,quantizer_efSearch=16  0.0676 0.2333 0.4407      0.00307       22128685    98
nprobe=4,quantizer_efSearch=4   0.0720 0.2608 0.5345      0.00343       43702860    88
nprobe=4,quantizer_efSearch=8   0.0731 0.2651 0.5398      0.00356       43630117    85
nprobe=4,quantizer_efSearch=16  0.0733 0.2654 0.5398      0.00381       43616458    79
nprobe=8,quantizer_efSearch=4   0.0753 0.2757 0.5972      0.00477       85789565    63
nprobe=8,quantizer_efSearch=8   0.0755 0.2771 0.5997      0.00483       85728903    63
nprobe=8,quantizer_efSearch=16  0.0756 0.2773 0.6002      0.00509       85708661    59
nprobe=8,quantizer_efSearch=32  0.0757 0.2774 0.6002      0.00550       85700197    55
nprobe=16,quantizer_efSearch=4  0.0760 0.2808 0.6235      0.00718      167721821    42
nprobe=16,quantizer_efSearch=32 0.0761 0.2839 0.6282      0.00787      167509485    39
nprobe=64,quantizer_efSearch=16 0.0762 0.2851 0.6412      0.02120      634854973    15
```

</details>
<details><summary>`PCAR16,IVF1024,SQfp16` </summary>
Index size 40117856

 code_size 32

 log filename: autotune.dbdeep1M.PCAR16_IVF1024_SQfp16.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0676 0.2333 0.4407      0.00234       22127567    129
nprobe=1                                 0.0595 0.1904 0.3230      0.00200       11209857    151
nprobe=4                                 0.0733 0.2654 0.5398      0.00243       43612231    124
nprobe=8                                 0.0757 0.2774 0.6002      0.00299       85698237    101
nprobe=16                                0.0761 0.2839 0.6282      0.00432      167503051    70
nprobe=64                                0.0762 0.2848 0.6410      0.01229      634271281    25
```

</details>
<details><summary>`PCAR16,IVF4096_HNSW32,SQfp16` </summary>
Index size 41450901

 code_size 32

 log filename: autotune.dbdeep1M.PCAR16_IVF4096_HNSW32_SQfp16.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0761 0.2783 0.5962      0.00264       47417216    114
nprobe=2,quantizer_efSearch=4            0.0614 0.1985 0.3325      0.00178        6297889    169
nprobe=4,quantizer_efSearch=4            0.0678 0.2338 0.4344      0.00188       12386652    160
nprobe=8,quantizer_efSearch=4            0.0732 0.2643 0.5313      0.00186       24262725    162
nprobe=8,quantizer_efSearch=8            0.0737 0.2666 0.5356      0.00189       24239350    159
nprobe=8,quantizer_efSearch=16           0.0741 0.2655 0.5360      0.00207       24199002    146
nprobe=16,quantizer_efSearch=4           0.0750 0.2741 0.5877      0.00222       47594534    136
nprobe=16,quantizer_efSearch=8           0.0753 0.2779 0.5948      0.00228       47508681    132
nprobe=16,quantizer_efSearch=16          0.0759 0.2779 0.5960      0.00243       47445074    124
nprobe=16,quantizer_efSearch=32          0.0761 0.2783 0.5962      0.00269       47417216    112
nprobe=32,quantizer_efSearch=8           0.0762 0.2831 0.6244      0.00333       92737739    91
nprobe=32,quantizer_efSearch=16          0.0767 0.2834 0.6266      0.00317       92564163    95
```

</details>
<details><summary>`PCAR32,IVF1024_HNSW32,SQ8` </summary>
Index size 40468053

 code_size 32

 log filename: autotune.dbdeep1M.PCAR32_IVF1024_HNSW32_SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                0.2061 0.4818 0.5631      0.00424       23426393    71
nprobe=1,quantizer_efSearch=8   0.1722 0.3682 0.4134      0.00309       11938192    98
nprobe=2,quantizer_efSearch=4   0.2061 0.4818 0.5631      0.00423       23426393    71
nprobe=2,quantizer_efSearch=8   0.2094 0.4876 0.5696      0.00436       23398672    69
nprobe=4,quantizer_efSearch=4   0.2316 0.5728 0.7082      0.00647       45860625    47
nprobe=4,quantizer_efSearch=8   0.2352 0.5827 0.7201      0.00660       45824794    46
nprobe=8,quantizer_efSearch=4   0.2464 0.6405 0.8278      0.01068       89253067    29
nprobe=8,quantizer_efSearch=8   0.2478 0.6456 0.8353      0.01072       89216594    28
nprobe=8,quantizer_efSearch=16  0.2481 0.6464 0.8371      0.01099       89150163    28
nprobe=8,quantizer_efSearch=32  0.2485 0.6469 0.8379      0.01142       89136306    27
nprobe=16,quantizer_efSearch=4  0.2529 0.6696 0.8903      0.01874      172648501    17
nprobe=16,quantizer_efSearch=8  0.2543 0.6768 0.9021      0.01884      172467630    16
nprobe=16,quantizer_efSearch=16 0.2544 0.6784 0.9052      0.01900      172336390    16
nprobe=16,quantizer_efSearch=32 0.2548 0.6789 0.9059      0.01945      172302792    16
nprobe=32,quantizer_efSearch=32 0.2567 0.6958 0.9450      0.03424      329841952    9
nprobe=64,quantizer_efSearch=32 0.2577 0.7003 0.9582      0.06273      632500716    5
```

</details>
<details><summary>`PCAR32,IVF1024,SQ8` </summary>
Index size 40189856

 code_size 32

 log filename: autotune.dbdeep1M.PCAR32_IVF1024_SQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2095 0.4879 0.5702      0.00257       23386342    117
nprobe=1                                 0.1721 0.3681 0.4134      0.00219       11936431    137
nprobe=4                                 0.2352 0.5829 0.7211      0.00452       45791428    67
nprobe=8                                 0.2483 0.6469 0.8379      0.00600       89128799    51
nprobe=16                                0.2546 0.6791 0.9060      0.01065      172279830    29
nprobe=32                                0.2567 0.6962 0.9453      0.01961      329780601    16
nprobe=64                                0.2576 0.7007 0.9587      0.03655      632250711    9
nprobe=128                               0.2578 0.7015 0.9624      0.06642     1224069483    5
```

</details>
<details><summary>`PCAR32,IVF4096_HNSW32,SQ8` </summary>
Index size 41719509

 code_size 32

 log filename: autotune.dbdeep1M.PCAR32_IVF4096_HNSW32_SQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2385 0.5952 0.7286      0.00578       27435571    52
nprobe=2,quantizer_efSearch=8            0.1870 0.4016 0.4460      0.00195        7265999    154
nprobe=4,quantizer_efSearch=4            0.2069 0.4901 0.5699      0.00193       14278632    156
nprobe=4,quantizer_efSearch=8            0.2149 0.5097 0.5937      0.00204       14188447    147
nprobe=4,quantizer_efSearch=16           0.2163 0.5116 0.5961      0.00223       14133496    135
nprobe=8,quantizer_efSearch=4            0.2339 0.5838 0.7135      0.00269       27644134    112
nprobe=8,quantizer_efSearch=8            0.2361 0.5903 0.7224      0.00274       27586129    110
nprobe=8,quantizer_efSearch=16           0.2377 0.5945 0.7280      0.00294       27476945    103
nprobe=8,quantizer_efSearch=32           0.2384 0.5952 0.7285      0.00328       27442907    92
nprobe=8,quantizer_efSearch=64           0.2385 0.5952 0.7286      0.00392       27436273    77
nprobe=16,quantizer_efSearch=8           0.2471 0.6447 0.8287      0.00520       53187266    58
nprobe=16,quantizer_efSearch=16          0.2480 0.6483 0.8334      0.00454       53050426    67
nprobe=32,quantizer_efSearch=32          0.2547 0.6831 0.9040      0.01209      101533980    25
nprobe=32,quantizer_efSearch=64          0.2549 0.6832 0.9045      0.00894      101438430    34
nprobe=64,quantizer_efSearch=16          0.2553 0.6948 0.9364      0.01239      194432561    25
nprobe=64,quantizer_efSearch=32          0.2562 0.6975 0.9406      0.01239      193839663    25
nprobe=128,quantizer_efSearch=32         0.2568 0.7007 0.9548      0.02397      368187040    13
nprobe=128,quantizer_efSearch=128        0.2569 0.7013 0.9562      0.02403      367076195    13
nprobe=256,quantizer_efSearch=32         0.2570 0.7009 0.9585      0.04204      688586685    8
nprobe=256,quantizer_efSearch=128        0.2571 0.7019 0.9613      0.04688      701445995    7
```

</details>
<details><summary>`PCAR64,IVF1024_HNSW32,SQ4` </summary>
Index size 40611797

 code_size 32

 log filename: autotune.dbdeep1M.PCAR64_IVF1024_HNSW32_SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                 0.3965 0.6035 0.6128      0.00799       25339576    38
nprobe=1,quantizer_efSearch=16   0.3177 0.4574 0.4629      0.00524       12985456    58
nprobe=1,quantizer_efSearch=32   0.3178 0.4575 0.4630      0.00573       12984409    53
nprobe=2,quantizer_efSearch=4    0.3965 0.6035 0.6128      0.00771       25339576    39
nprobe=2,quantizer_efSearch=8    0.4040 0.6154 0.6247      0.00789       25309714    39
nprobe=2,quantizer_efSearch=16   0.4044 0.6161 0.6254      0.00818       25303291    37
nprobe=2,quantizer_efSearch=32   0.4045 0.6162 0.6255      0.00866       25298151    35
nprobe=4,quantizer_efSearch=4    0.4628 0.7401 0.7547      0.01329       49265657    23
nprobe=4,quantizer_efSearch=8    0.4748 0.7598 0.7744      0.01336       49187538    23
nprobe=4,quantizer_efSearch=16   0.4760 0.7615 0.7761      0.01367       49175246    22
nprobe=4,quantizer_efSearch=32   0.4762 0.7616 0.7762      0.01417       49171720    22
nprobe=8,quantizer_efSearch=4    0.5156 0.8544 0.8749      0.02375       94911170    13
nprobe=8,quantizer_efSearch=16   0.5223 0.8679 0.8884      0.02411       94792982    13
nprobe=8,quantizer_efSearch=32   0.5228 0.8679 0.8885      0.02454       94762839    13
nprobe=16,quantizer_efSearch=4   0.5368 0.9073 0.9312      0.04317      181438815    7
nprobe=16,quantizer_efSearch=8   0.5435 0.9225 0.9469      0.04315      181152274    7
nprobe=16,quantizer_efSearch=32  0.5446 0.9267 0.9514      0.04382      180875486    7
nprobe=32,quantizer_efSearch=8   0.5515 0.9468 0.9739      0.07971      343528224    4
nprobe=32,quantizer_efSearch=16  0.5526 0.9533 0.9809      0.07990      342958929    4
nprobe=32,quantizer_efSearch=32  0.5540 0.9548 0.9824      0.08013      342628560    4
nprobe=64,quantizer_efSearch=16  0.5542 0.9624 0.9919      0.14822      651011663    3
nprobe=64,quantizer_efSearch=64  0.5562 0.9647 0.9948      0.14913      649271273    3
nprobe=128,quantizer_efSearch=64 0.5565 0.9683 0.9990      0.28129     1240550464    2
nprobe=256,quantizer_efSearch=64 0.5568 0.9689 1.0000      0.54095     2394001884    1
```

</details>
<details><summary>`PCAR64,IVF1024,SQ4` </summary>
Index size 40333600

 code_size 32

 log filename: autotune.dbdeep1M.PCAR64_IVF1024_SQ4.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4056 0.6159 0.6254      0.00453       25294533    67
nprobe=1                                 0.3180 0.4575 0.4628      0.00279       12983209    108
nprobe=2                                 0.4056 0.6159 0.6254      0.00493       25294533    61
nprobe=4                                 0.4766 0.7613 0.7761      0.00764       49161249    40
nprobe=8                                 0.5241 0.8678 0.8885      0.01337       94739869    23
nprobe=16                                0.5463 0.9263 0.9513      0.02462      180809642    13
nprobe=32                                0.5558 0.9548 0.9822      0.04664      342422723    7
nprobe=64                                0.5581 0.9646 0.9948      0.08384      649004728    4
nprobe=128                               0.5587 0.9683 0.9990      0.15840     1239397997    2
nprobe=256                               0.5590 0.9690 1.0000      0.30818     2386045165    1
```

</details>
<details><summary>`PCAR64,IVF4096_HNSW32,SQ4` </summary>
Index size 42256469

 code_size 32

 log filename: autotune.dbdeep1M.PCAR64_IVF4096_HNSW32_SQ4.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5329 0.8745 0.8900      0.01221       61854688    25
nprobe=1,quantizer_efSearch=16           0.2669 0.3558 0.3580      0.00228        4452606    132
nprobe=2,quantizer_efSearch=4            0.3375 0.4812 0.4848      0.00246        8750686    123
nprobe=2,quantizer_efSearch=8            0.3525 0.5047 0.5083      0.00257        8709158    117
nprobe=2,quantizer_efSearch=16           0.3540 0.5067 0.5103      0.00280        8669514    108
nprobe=4,quantizer_efSearch=4            0.4115 0.6178 0.6233      0.00385       17027244    78
nprobe=4,quantizer_efSearch=8            0.4343 0.6537 0.6595      0.00399       16968094    76
nprobe=4,quantizer_efSearch=16           0.4373 0.6585 0.6642      0.00423       16885689    71
nprobe=8,quantizer_efSearch=8            0.4912 0.7760 0.7860      0.00667       32676789    46
nprobe=8,quantizer_efSearch=16           0.4970 0.7856 0.7958      0.00687       32523881    44
nprobe=8,quantizer_efSearch=32           0.4980 0.7868 0.7970      0.00729       32459729    42
nprobe=16,quantizer_efSearch=4           0.5153 0.8430 0.8569      0.01172       62482964    26
nprobe=16,quantizer_efSearch=8           0.5277 0.8651 0.8802      0.01141       62241370    27
nprobe=16,quantizer_efSearch=16          0.5311 0.8725 0.8879      0.01157       62024115    26
nprobe=16,quantizer_efSearch=32          0.5329 0.8745 0.8900      0.01195       61854688    26
nprobe=32,quantizer_efSearch=8           0.5409 0.9091 0.9283      0.02169      117779060    14
nprobe=32,quantizer_efSearch=16          0.5479 0.9231 0.9436      0.02126      117298690    15
nprobe=32,quantizer_efSearch=32          0.5494 0.9268 0.9476      0.02180      116930097    14
nprobe=32,quantizer_efSearch=64          0.5499 0.9276 0.9483      0.02226      116737762    14
nprobe=64,quantizer_efSearch=64          0.5592 0.9556 0.9792      0.03919      218872950    8
nprobe=128,quantizer_efSearch=32         0.5624 0.9640 0.9899      0.07135      411190675    5
nprobe=128,quantizer_efSearch=64         0.5643 0.9673 0.9932      0.07189      409916397    5
nprobe=128,quantizer_efSearch=128        0.5645 0.9673 0.9933      0.07307      409060007    5
nprobe=256,quantizer_efSearch=64         0.5654 0.9713 0.9985      0.14170      768897610    3
nprobe=256,quantizer_efSearch=128        0.5657 0.9717 0.9988      0.13231      771087396    3
nprobe=512,quantizer_efSearch=128        0.5662 0.9725 0.9998      0.24248     1459858336    2
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF1024_HNSW32,PQ128x4fs` </summary>
Index size 74500556

 code_size 64

 log filename: autotune.dbdeep1M.OPQ128_256_IVF1024_HNSW32_PQ128x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                  0.4169 0.6076 0.6133      0.00269       25911126    112
nprobe=1,quantizer_efSearch=8     0.3312 0.4595 0.4634      0.00262       13245334    115
nprobe=2,quantizer_efSearch=4     0.4169 0.6076 0.6133      0.00269       25911126    112
nprobe=2,quantizer_efSearch=8     0.4279 0.6227 0.6284      0.00301       25947920    100
nprobe=4,quantizer_efSearch=4     0.4881 0.7457 0.7551      0.00334       50523241    90
nprobe=4,quantizer_efSearch=8     0.5035 0.7687 0.7782      0.00366       50581653    83
nprobe=4,quantizer_efSearch=16    0.5046 0.7704 0.7799      0.00421       50558443    72
nprobe=8,quantizer_efSearch=4     0.5517 0.8639 0.8762      0.00473       97586978    64
nprobe=8,quantizer_efSearch=8     0.5571 0.8728 0.8853      0.00482       97567562    63
nprobe=8,quantizer_efSearch=16    0.5598 0.8769 0.8894      0.00532       97532769    57
nprobe=8,quantizer_efSearch=32    0.5604 0.8776 0.8901      0.00622       97497087    49
nprobe=16,quantizer_efSearch=4    0.5715 0.9132 0.9275      0.00667      186483792    46
nprobe=16,quantizer_efSearch=8    0.5797 0.9289 0.9434      0.00697      186358159    44
nprobe=16,quantizer_efSearch=16   0.5818 0.9326 0.9472      0.00739      186256077    41
nprobe=16,quantizer_efSearch=32   0.5827 0.9338 0.9484      0.00825      186149699    37
nprobe=16,quantizer_efSearch=64   0.5829 0.9341 0.9487      0.00974      186141151    31
nprobe=32,quantizer_efSearch=16   0.5955 0.9649 0.9807      0.01097      352354928    28
nprobe=32,quantizer_efSearch=32   0.5972 0.9670 0.9828      0.01166      352026792    26
nprobe=32,quantizer_efSearch=64   0.5974 0.9673 0.9831      0.01330      351943431    23
nprobe=64,quantizer_efSearch=16   0.5998 0.9753 0.9918      0.01762      666739648    18
nprobe=64,quantizer_efSearch=32   0.6015 0.9781 0.9946      0.01837      665547841    17
nprobe=64,quantizer_efSearch=64   0.6020 0.9788 0.9953      0.01968      665014215    16
nprobe=128,quantizer_efSearch=32  0.6032 0.9815 0.9981      0.03090     1265678404    10
nprobe=128,quantizer_efSearch=64  0.6037 0.9825 0.9991      0.03271     1262486071    10
nprobe=256,quantizer_efSearch=32  0.6038 0.9821 0.9989      0.05414     2389196945    6
nprobe=256,quantizer_efSearch=128 0.6044 0.9830 0.9999      0.06128     2406436174    5
```

</details>
<details><summary>`OPQ128_256,IVF1024,PQ128x4fs` </summary>
Index size 74203927

 code_size 64

 log filename: autotune.dbdeep1M.OPQ128_256_IVF1024_PQ128x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5052 0.7710 0.7805      0.00202       50522581    149
nprobe=4                                 0.5052 0.7710 0.7805      0.00186       50522581    161
nprobe=8                                 0.5607 0.8778 0.8903      0.00264       97467151    114
nprobe=16                                0.5830 0.9341 0.9487      0.00381      186078083    79
nprobe=32                                0.5977 0.9675 0.9833      0.00579      351797472    52
nprobe=64                                0.6020 0.9788 0.9953      0.00959      664636016    32
nprobe=128                               0.6036 0.9825 0.9991      0.01623     1260718560    19
nprobe=256                               0.6045 0.9830 0.9999      0.02969     2403208777    11
```

</details>
<details><summary>`OPQ128_256,IVF1024,PQ128x4fsr` </summary>
Index size 74173207

 code_size 64

 log filename: autotune.dbdeep1M.OPQ128_256_IVF1024_PQ128x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5695 0.6465 0.6468      0.00233       24366202    129
nprobe=1                                 0.4271 0.4746 0.4748      0.00181       12511970    166
nprobe=2                                 0.5695 0.6465 0.6468      0.00227       24366202    132
nprobe=4                                 0.6911 0.7941 0.7944      0.00356       47389852    85
nprobe=8                                 0.7660 0.8948 0.8951      0.01516       91686082    20
nprobe=16                                0.8061 0.9554 0.9557      0.02897      175411771    11
nprobe=32                                0.8263 0.9822 0.9826      0.06284      332684022    5
nprobe=64                                0.8333 0.9951 0.9956      0.10540      632292425    3
```

</details>
<details><summary>`OPQ128_256,IVF4096_HNSW32,PQ128x4fs` </summary>
Index size 81627724

 code_size 64

 log filename: autotune.dbdeep1M.OPQ128_256_IVF4096_HNSW32_PQ128x4fs.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5615 0.8804 0.8930      0.00404       67133583    75
nprobe=4,quantizer_efSearch=8            0.4493 0.6630 0.6697      0.00199       18481203    151
nprobe=8,quantizer_efSearch=4            0.5045 0.7708 0.7810      0.00223       35617049    135
nprobe=8,quantizer_efSearch=8            0.5143 0.7858 0.7960      0.00225       35587332    134
nprobe=8,quantizer_efSearch=16           0.5199 0.7942 0.8044      0.00257       35415902    117
nprobe=16,quantizer_efSearch=4           0.5379 0.8403 0.8527      0.00274       67600759    110
nprobe=16,quantizer_efSearch=8           0.5550 0.8700 0.8826      0.00281       67526561    107
nprobe=16,quantizer_efSearch=16          0.5599 0.8777 0.8903      0.00310       67323249    97
nprobe=32,quantizer_efSearch=8           0.5760 0.9177 0.9318      0.00390      127122290    77
nprobe=32,quantizer_efSearch=16          0.5846 0.9331 0.9476      0.00418      126728059    72
nprobe=32,quantizer_efSearch=32          0.5862 0.9361 0.9506      0.00472      126335690    64
nprobe=32,quantizer_efSearch=64          0.5864 0.9364 0.9509      0.00586      126110281    52
nprobe=64,quantizer_efSearch=16          0.5943 0.9592 0.9743      0.00606      237215576    50
nprobe=64,quantizer_efSearch=32          0.5973 0.9656 0.9808      0.00642      236374561    47
nprobe=64,quantizer_efSearch=64          0.5975 0.9669 0.9822      0.00738      235702797    41
nprobe=128,quantizer_efSearch=32         0.6010 0.9752 0.9911      0.00989      440832184    31
nprobe=128,quantizer_efSearch=64         0.6019 0.9778 0.9938      0.01077      439725683    28
nprobe=128,quantizer_efSearch=128        0.6023 0.9783 0.9944      0.01270      438811068    24
nprobe=256,quantizer_efSearch=64         0.6036 0.9811 0.9973      0.01758      822202498    18
nprobe=256,quantizer_efSearch=128        0.6042 0.9823 0.9985      0.02091      824965986    15
nprobe=512,quantizer_efSearch=256        0.6043 0.9831 0.9996      0.04092     1560473668    8
```

</details>
<details><summary>`OPQ128_256,IVF4096_HNSW32,PQ128x4fsr` </summary>
Index size 81601100

 code_size 64

 log filename: autotune.dbdeep1M.OPQ128_256_IVF4096_HNSW32_PQ128x4fsr.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7782 0.8984 0.8987      0.02670       45568656    12
nprobe=1,quantizer_efSearch=8            0.3406 0.3701 0.3701      0.00185        3072675    163
nprobe=2,quantizer_efSearch=4            0.4513 0.4979 0.4980      0.00229        6037613    132
nprobe=2,quantizer_efSearch=8            0.4770 0.5261 0.5262      0.00257        6072662    117
nprobe=2,quantizer_efSearch=16           0.4784 0.5274 0.5275      0.00287        6063578    105
nprobe=4,quantizer_efSearch=4            0.5679 0.6379 0.6382      0.00310       11889522    97
nprobe=4,quantizer_efSearch=8            0.6045 0.6802 0.6805      0.00328       11963080    92
nprobe=4,quantizer_efSearch=16           0.6083 0.6847 0.6850      0.00358       11947994    84
nprobe=4,quantizer_efSearch=32           0.6091 0.6858 0.6861      0.00421       11941201    72
nprobe=4,quantizer_efSearch=64           0.6092 0.6859 0.6862      0.00530       11940530    57
nprobe=8,quantizer_efSearch=16           0.7115 0.8105 0.8109      0.01365       23416952    22
nprobe=8,quantizer_efSearch=32           0.7132 0.8126 0.8130      0.01442       23404018    21
nprobe=16,quantizer_efSearch=4           0.7590 0.8723 0.8726      0.02411       45703934    13
nprobe=16,quantizer_efSearch=16          0.7751 0.8949 0.8952      0.02542       45603258    12
nprobe=16,quantizer_efSearch=32          0.7782 0.8984 0.8987      0.02674       45568656    12
nprobe=16,quantizer_efSearch=64          0.7786 0.8991 0.8994      0.02776       45555791    11
nprobe=16,quantizer_efSearch=128         0.7787 0.8992 0.8995      0.02845       45552822    11
nprobe=32,quantizer_efSearch=8           0.8000 0.9347 0.9350      0.05735       88689830    6
nprobe=32,quantizer_efSearch=16          0.8090 0.9466 0.9469      0.06122       88543958    5
nprobe=32,quantizer_efSearch=32          0.8119 0.9497 0.9500      0.06438       88448027    5
nprobe=32,quantizer_efSearch=128         0.8125 0.9507 0.9510      0.06834       88389577    5
nprobe=64,quantizer_efSearch=32          0.8308 0.9787 0.9791      0.11038      169935257    3
nprobe=64,quantizer_efSearch=128         0.8324 0.9804 0.9808      0.11590      169687748    3
nprobe=128,quantizer_efSearch=64         0.8326 0.9929 0.9932      0.22587      323826918    2
nprobe=128,quantizer_efSearch=128        0.8337 0.9936 0.9939      0.22989      323468060    2
nprobe=256,quantizer_efSearch=128        0.8371 0.9982 0.9985      0.43934      615946288    1
nprobe=512,quantizer_efSearch=128        0.8377 0.9992 0.9995      0.82403     1184411288    1
```

</details>
<details><summary>`OPQ16_64,IVF1024,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 72592349

 code_size 64

 log filename: autotune.dbdeep1M.OPQ16_64_IVF1024_PQ16x4fs_Refine_OPQ56_112_PQ56_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7346 0.8633 0.8633      0.00405      618431001    75
k_factor_rf=1,nprobe=2                   0.5186 0.5963 0.5963      0.00173       23618520    174
k_factor_rf=1,nprobe=4                   0.6226 0.7229 0.7229      0.00187       46092750    161
k_factor_rf=1,nprobe=8                   0.6873 0.8030 0.8030      0.00207       89113658    145
k_factor_rf=1,nprobe=16                  0.7183 0.8426 0.8426      0.00240      170819148    125
k_factor_rf=1,nprobe=32                  0.7294 0.8571 0.8571      0.00300      324837799    101
k_factor_rf=2,nprobe=16                  0.7549 0.9013 0.9014      0.00328      170819148    92
k_factor_rf=2,nprobe=32                  0.7687 0.9224 0.9225      0.00396      324837799    76
k_factor_rf=2,nprobe=64                  0.7738 0.9289 0.9290      0.00510      618431001    59
k_factor_rf=4,nprobe=32                  0.7873 0.9593 0.9594      0.00604      324837799    50
k_factor_rf=4,nprobe=64                  0.7942 0.9696 0.9697      0.00736      618431001    41
k_factor_rf=4,nprobe=128                 0.7955 0.9705 0.9706      0.00925     1185647808    33
k_factor_rf=8,nprobe=32                  0.7956 0.9744 0.9748      0.00949      324837799    32
k_factor_rf=8,nprobe=64                  0.8047 0.9873 0.9877      0.01104      618431001    28
k_factor_rf=8,nprobe=128                 0.8069 0.9901 0.9905      0.01312     1185647808    23
k_factor_rf=8,nprobe=256                 0.8072 0.9906 0.9910      0.01703     2291777784    18
k_factor_rf=16,nprobe=64                 0.8077 0.9933 0.9938      0.01917      618431001    16
k_factor_rf=16,nprobe=128                0.8097 0.9960 0.9966      0.02198     1185647808    14
k_factor_rf=16,nprobe=256                0.8101 0.9969 0.9975      0.02638     2291777784    12
k_factor_rf=16,nprobe=512                0.8103 0.9971 0.9977      0.03388     4505205896    9
k_factor_rf=32,nprobe=256                0.8106 0.9982 0.9988      0.04373     2291777784    7
k_factor_rf=32,nprobe=512                0.8107 0.9983 0.9989      0.05293     4505205896    6
k_factor_rf=64,nprobe=128                0.8109 0.9983 0.9988      0.06864     1185647808    5
k_factor_rf=64,nprobe=256                0.8111 0.9989 0.9996      0.07801     2291777784    4
k_factor_rf=64,nprobe=512                0.8114 0.9992 0.9998      0.09373     4505205896    4
```

</details>
<details><summary>`OPQ16_64,IVF1024,PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 70500825

 code_size 62

 log filename: autotune.dbdeep1M.OPQ16_64_IVF1024_PQ16x4fs_Refine_PCAR72_SQ6_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6589 0.8628 0.8633      0.00358      618431001    84
k_factor_rf=1,nprobe=2                   0.4825 0.5961 0.5963      0.00133       23618520    227
k_factor_rf=1,nprobe=4                   0.5698 0.7224 0.7229      0.00146       46092750    206
k_factor_rf=1,nprobe=8                   0.6242 0.8024 0.8030      0.00169       89113658    178
k_factor_rf=1,nprobe=16                  0.6485 0.8420 0.8426      0.00198      170819148    152
k_factor_rf=1,nprobe=32                  0.6561 0.8566 0.8571      0.00254      324837799    118
k_factor_rf=2,nprobe=16                  0.6740 0.9002 0.9014      0.00278      170819148    108
k_factor_rf=2,nprobe=32                  0.6846 0.9211 0.9225      0.00342      324837799    88
k_factor_rf=2,nprobe=64                  0.6874 0.9276 0.9290      0.00454      618431001    67
k_factor_rf=4,nprobe=32                  0.6969 0.9575 0.9594      0.00538      324837799    56
k_factor_rf=4,nprobe=64                  0.7010 0.9677 0.9697      0.00664      618431001    46
k_factor_rf=4,nprobe=128                 0.7016 0.9685 0.9706      0.00849     1185647808    36
k_factor_rf=8,nprobe=32                  0.7026 0.9722 0.9748      0.00852      324837799    36
k_factor_rf=8,nprobe=64                  0.7076 0.9849 0.9877      0.01016      618431001    30
k_factor_rf=8,nprobe=128                 0.7089 0.9877 0.9905      0.01230     1185647808    25
k_factor_rf=8,nprobe=256                 0.7090 0.9882 0.9910      0.01591     2291777784    19
k_factor_rf=16,nprobe=64                 0.7104 0.9904 0.9938      0.01781      618431001    17
k_factor_rf=16,nprobe=128                0.7114 0.9931 0.9966      0.02070     1185647808    15
k_factor_rf=16,nprobe=256                0.7115 0.9939 0.9975      0.02522     2291777784    12
k_factor_rf=16,nprobe=512                0.7116 0.9941 0.9977      0.03293     4505205896    10
k_factor_rf=32,nprobe=256                0.7117 0.9952 0.9988      0.04217     2291777784    8
k_factor_rf=64,nprobe=256                0.7118 0.9956 0.9996      0.08755     2291777784    4
k_factor_rf=64,nprobe=512                0.7119 0.9958 0.9998      0.09847     4505205896    4
```

</details>
<details><summary>`OPQ16_64,IVF4096_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 74918674

 code_size 64

 log filename: autotune.dbdeep1M.OPQ16_64_IVF4096_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.8160 0.9955 0.9961      0.02983     1119425904    11
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.5180 0.5936 0.5938      0.00176       11564326    171
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.6307 0.7314 0.7316      0.00204       22860500    147
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.6357 0.7375 0.7377      0.00218       22845137    138
k_factor_rf=1,nprobe=8,quantizer_efSearch=16     0.6402 0.7432 0.7434      0.00230       22825853    131
k_factor_rf=1,nprobe=16,quantizer_efSearch=4     0.6781 0.7906 0.7908      0.00225       44651974    134
k_factor_rf=1,nprobe=16,quantizer_efSearch=8     0.6911 0.8066 0.8068      0.00228       44639220    132
k_factor_rf=1,nprobe=16,quantizer_efSearch=16    0.6968 0.8126 0.8128      0.00242       44587481    125
k_factor_rf=1,nprobe=32,quantizer_efSearch=8     0.7081 0.8292 0.8294      0.00267       86690453    113
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.7190 0.8523 0.8525      0.00305       44639220    99
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.7252 0.8592 0.8594      0.00328       44587481    92
k_factor_rf=2,nprobe=32,quantizer_efSearch=8     0.7431 0.8845 0.8847      0.00351       86690453    86
k_factor_rf=2,nprobe=32,quantizer_efSearch=16    0.7538 0.8967 0.8969      0.00369       86545479    82
k_factor_rf=2,nprobe=32,quantizer_efSearch=32    0.7555 0.8992 0.8994      0.00399       86444783    76
k_factor_rf=2,nprobe=32,quantizer_efSearch=64    0.7561 0.8997 0.8999      0.00478       86415738    63
k_factor_rf=2,nprobe=64,quantizer_efSearch=32    0.7711 0.9187 0.9189      0.00470      166456973    64
k_factor_rf=2,nprobe=64,quantizer_efSearch=64    0.7719 0.9200 0.9202      0.00523      166296216    58
k_factor_rf=2,nprobe=128,quantizer_efSearch=32   0.7731 0.9222 0.9224      0.00584      318550996    52
k_factor_rf=4,nprobe=64,quantizer_efSearch=16    0.7864 0.9482 0.9486      0.00596      166857043    51
k_factor_rf=4,nprobe=64,quantizer_efSearch=64    0.7915 0.9547 0.9551      0.00684      166296216    44
k_factor_rf=4,nprobe=128,quantizer_efSearch=32   0.7954 0.9609 0.9612      0.00755      318550996    40
k_factor_rf=4,nprobe=128,quantizer_efSearch=64   0.7976 0.9632 0.9635      0.00823      317937131    37
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.8002 0.9707 0.9711      0.01053      166456973    29
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.8008 0.9714 0.9718      0.01060      166296216    29
k_factor_rf=8,nprobe=128,quantizer_efSearch=16   0.8020 0.9741 0.9745      0.01144      318989903    27
k_factor_rf=8,nprobe=128,quantizer_efSearch=32   0.8070 0.9813 0.9817      0.01201      318550996    25
k_factor_rf=8,nprobe=128,quantizer_efSearch=64   0.8094 0.9839 0.9843      0.01281      317937131    24
k_factor_rf=8,nprobe=256,quantizer_efSearch=64   0.8111 0.9866 0.9870      0.01572      606688359    20
k_factor_rf=8,nprobe=256,quantizer_efSearch=256  0.8112 0.9868 0.9873      0.02098      606299191    15
k_factor_rf=16,nprobe=128,quantizer_efSearch=64  0.8132 0.9913 0.9918      0.02112      317937131    15
k_factor_rf=16,nprobe=256,quantizer_efSearch=64  0.8158 0.9949 0.9955      0.02543      606688359    12
k_factor_rf=16,nprobe=256,quantizer_efSearch=128 0.8160 0.9954 0.9959      0.02786      606707940    11
k_factor_rf=16,nprobe=512,quantizer_efSearch=128 0.8165 0.9964 0.9969      0.03402     1170758780    9
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.8167 0.9974 0.9981      0.04367      606707940    7
k_factor_rf=32,nprobe=512,quantizer_efSearch=256 0.8170 0.9986 0.9992      0.05886     1165996117    6
```

</details>
<details><summary>`OPQ16_64,IVF4096_HNSW32,PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 72827150

 code_size 62

 log filename: autotune.dbdeep1M.OPQ16_64_IVF4096_HNSW32_PQ16x4fs_Refine_PCAR72_SQ6_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.7035 0.9688 0.9719      0.01048      166295818    29
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.4835 0.5962 0.5962      0.00133       11573404    225
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.5773 0.7312 0.7315      0.00159       22859637    189
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.5820 0.7373 0.7375      0.00164       22844811    183
k_factor_rf=1,nprobe=16,quantizer_efSearch=4     0.6133 0.7890 0.7894      0.00183       44641846    164
k_factor_rf=1,nprobe=16,quantizer_efSearch=8     0.6232 0.8054 0.8058      0.00191       44641059    157
k_factor_rf=1,nprobe=16,quantizer_efSearch=16    0.6283 0.8123 0.8127      0.00210       44592685    143
k_factor_rf=1,nprobe=32,quantizer_efSearch=8     0.6366 0.8289 0.8294      0.00229       86688493    131
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.6459 0.8508 0.8517      0.00278       44641059    108
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.6514 0.8584 0.8593      0.00289       44592685    104
k_factor_rf=1,nprobe=64,quantizer_efSearch=16    0.6527 0.8516 0.8522      0.00320      166859174    94
k_factor_rf=2,nprobe=32,quantizer_efSearch=8     0.6620 0.8830 0.8842      0.00315       86688493    96
k_factor_rf=2,nprobe=32,quantizer_efSearch=16    0.6705 0.8949 0.8962      0.00324       86539329    93
k_factor_rf=2,nprobe=32,quantizer_efSearch=32    0.6710 0.8977 0.8990      0.00369       86441703    82
k_factor_rf=2,nprobe=64,quantizer_efSearch=32    0.6839 0.9173 0.9186      0.00450      166441506    67
k_factor_rf=2,nprobe=64,quantizer_efSearch=64    0.6847 0.9188 0.9201      0.00522      166295818    58
k_factor_rf=4,nprobe=64,quantizer_efSearch=16    0.6947 0.9462 0.9482      0.00531      166859174    57
k_factor_rf=4,nprobe=128,quantizer_efSearch=16   0.6967 0.9520 0.9540      0.00652      318541989    47
k_factor_rf=4,nprobe=64,quantizer_efSearch=64    0.6975 0.9529 0.9550      0.00665      166295818    46
k_factor_rf=4,nprobe=128,quantizer_efSearch=32   0.6998 0.9587 0.9607      0.00716      318394736    42
k_factor_rf=4,nprobe=128,quantizer_efSearch=64   0.7014 0.9613 0.9634      0.00810      317925292    38
k_factor_rf=8,nprobe=64,quantizer_efSearch=32    0.7031 0.9677 0.9708      0.00946      166441506    32
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.7035 0.9688 0.9719      0.01039      166295818    29
k_factor_rf=8,nprobe=128,quantizer_efSearch=32   0.7066 0.9782 0.9813      0.01119      318394736    27
k_factor_rf=8,nprobe=128,quantizer_efSearch=64   0.7085 0.9811 0.9843      0.01219      317925292    25
k_factor_rf=8,nprobe=256,quantizer_efSearch=64   0.7090 0.9838 0.9869      0.01483      606327573    21
k_factor_rf=16,nprobe=128,quantizer_efSearch=32  0.7097 0.9851 0.9889      0.01980      318394736    16
k_factor_rf=16,nprobe=128,quantizer_efSearch=64  0.7113 0.9877 0.9917      0.02053      317925292    15
k_factor_rf=16,nprobe=256,quantizer_efSearch=64  0.7128 0.9914 0.9954      0.02359      606327573    13
k_factor_rf=16,nprobe=256,quantizer_efSearch=128 0.7129 0.9917 0.9958      0.02577      606718400    12
k_factor_rf=32,nprobe=256,quantizer_efSearch=64  0.7133 0.9935 0.9976      0.03987      606327573    8
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 64497657

 code_size 56

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ16x4fs_Refine_PCAR64_SQ6_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5959 0.8614 0.8633      0.00390      618431001    77
k_factor_rf=1,nprobe=2                   0.4425 0.5954 0.5963      0.00131       23618520    229
k_factor_rf=1,nprobe=4                   0.5192 0.7214 0.7229      0.00145       46092750    207
k_factor_rf=1,nprobe=8                   0.5657 0.8009 0.8030      0.00164       89113658    184
k_factor_rf=1,nprobe=16                  0.5854 0.8406 0.8426      0.00202      170819148    149
k_factor_rf=1,nprobe=32                  0.5922 0.8551 0.8571      0.00260      324837799    116
k_factor_rf=2,nprobe=16                  0.6031 0.8963 0.9014      0.00279      170819148    108
k_factor_rf=2,nprobe=32                  0.6119 0.9167 0.9225      0.00349      324837799    86
k_factor_rf=2,nprobe=64                  0.6153 0.9231 0.9290      0.00662      618431001    46
k_factor_rf=4,nprobe=32                  0.6211 0.9508 0.9594      0.00560      324837799    54
k_factor_rf=4,nprobe=64                  0.6255 0.9609 0.9697      0.00684      618431001    44
k_factor_rf=8,nprobe=64                  0.6300 0.9765 0.9877      0.01029      618431001    30
k_factor_rf=8,nprobe=128                 0.6304 0.9788 0.9905      0.01252     1185647808    24
k_factor_rf=16,nprobe=128                0.6307 0.9830 0.9966      0.02199     1185647808    14
k_factor_rf=16,nprobe=256                0.6311 0.9835 0.9975      0.02653     2291777784    12
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 72690141

 code_size 64

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fs_Refine_OPQ48_96_PQ48_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7981 0.9849 0.9853      0.00482      618974813    63
k_factor_rf=1,nprobe=1                   0.3989 0.4604 0.4607      0.00160       12090236    188
k_factor_rf=1,nprobe=2                   0.5301 0.6232 0.6235      0.00170       23536247    177
k_factor_rf=1,nprobe=4                   0.6449 0.7718 0.7721      0.00196       45949821    154
k_factor_rf=1,nprobe=8                   0.7234 0.8761 0.8764      0.00223       89055363    135
k_factor_rf=1,nprobe=16                  0.7694 0.9436 0.9439      0.00265      170778033    114
k_factor_rf=2,nprobe=16                  0.7707 0.9494 0.9497      0.00356      170778033    85
k_factor_rf=1,nprobe=32                  0.7904 0.9740 0.9744      0.00343      324754821    88
k_factor_rf=2,nprobe=32                  0.7920 0.9808 0.9813      0.00461      324754821    66
k_factor_rf=1,nprobe=64                  0.7981 0.9849 0.9853      0.00488      618974813    62
k_factor_rf=2,nprobe=64                  0.7996 0.9921 0.9926      0.00594      618974813    51
k_factor_rf=1,nprobe=128                 0.8010 0.9892 0.9896      0.00802     1187520814    38
k_factor_rf=2,nprobe=128                 0.8024 0.9966 0.9970      0.00817     1187520814    37
k_factor_rf=2,nprobe=256                 0.8027 0.9973 0.9977      0.01247     2295739775    25
k_factor_rf=8,nprobe=256                 0.8031 0.9991 0.9998      0.01964     2295739775    16
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fs,Refine(PCAR48,SQ8)` </summary>
Index size 72610617

 code_size 64

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fs_Refine_PCAR48_SQ8_.b.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                         0.3431 0.6019 0.6181      0.00339       25328878    89
k_factor_rf=1,nprobe=1   0.2740 0.4443 0.4517      0.00210       12952349    144
k_factor_rf=1,nprobe=2   0.3426 0.6012 0.6151      0.00232       25328878    130
k_factor_rf=1,nprobe=4   0.3958 0.7431 0.7685      0.00264       49336048    114
k_factor_rf=1,nprobe=8   0.4260 0.8434 0.8796      0.00315       94928819    96
k_factor_rf=1,nprobe=16  0.4370 0.8920 0.9376      0.00401      181284631    75
k_factor_rf=1,nprobe=32  0.4412 0.9177 0.9679      0.00543      343278394    56
k_factor_rf=1,nprobe=128 0.4427 0.9294 0.9832      0.01223     1242259908    25
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fs,Refine(PQ24)` </summary>
Index size 48652182

 code_size 40

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fs_Refine_PQ24_.b.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                         0.4382 0.8890 0.9376      0.00433      181284631    70
k_factor_rf=1,nprobe=1   0.2552 0.4368 0.4517      0.00238       12952349    126
k_factor_rf=1,nprobe=2   0.3279 0.5915 0.6151      0.00262       25328878    115
k_factor_rf=1,nprobe=4   0.3846 0.7346 0.7685      0.00294       49336048    103
k_factor_rf=1,nprobe=8   0.4222 0.8372 0.8796      0.00345       94928819    87
k_factor_rf=1,nprobe=16  0.4382 0.8890 0.9376      0.00430      181284631    70
k_factor_rf=1,nprobe=32  0.4479 0.9169 0.9679      0.00573      343278394    53
k_factor_rf=1,nprobe=128 0.4522 0.9300 0.9832      0.01247     1242259908    25
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fs,Refine(PQ48)` </summary>
Index size 72652182

 code_size 64

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fs_Refine_PQ48_.b.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                          0.7929 0.9787 0.9789      0.00891      650621360    34
k_factor_rf=1,nprobe=2    0.5231 0.6149 0.6151      0.00293       25328878    103
k_factor_rf=1,nprobe=4    0.6403 0.7683 0.7685      0.00354       49336048    85
k_factor_rf=1,nprobe=8    0.7222 0.8794 0.8796      0.00391       94928819    77
k_factor_rf=1,nprobe=16   0.7653 0.9374 0.9376      0.00467      181284631    65
k_factor_rf=1,nprobe=32   0.7862 0.9677 0.9679      0.00613      343278394    49
k_factor_rf=2,nprobe=32   0.7908 0.9780 0.9780      0.00788      343278394    39
k_factor_rf=1,nprobe=64   0.7929 0.9787 0.9789      0.00853      650621360    36
k_factor_rf=2,nprobe=64   0.7980 0.9896 0.9896      0.01054      650621360    29
k_factor_rf=4,nprobe=64   0.7981 0.9928 0.9931      0.01433      650621360    21
k_factor_rf=2,nprobe=128  0.7996 0.9942 0.9943      0.01481     1242259908    21
k_factor_rf=4,nprobe=128  0.7999 0.9973 0.9977      0.01956     1242259908    16
k_factor_rf=2,nprobe=256  0.8001 0.9950 0.9951      0.02347     2389492167    13
k_factor_rf=4,nprobe=256  0.8003 0.9981 0.9985      0.02765     2389492167    11
k_factor_rf=4,nprobe=512  0.8004 0.9982 0.9986      0.04642     4660580015    7
k_factor_rf=16,nprobe=512 0.8005 0.9993 0.9999      0.07279     4660580015    5
```

</details>
<details><summary>`OPQ32_64,IVF1024,PQ32x4fs,Refine(SQ4)` </summary>
Index size 72554641

 code_size 64

 log filename: autotune.dbdeep1M.OPQ32_64_IVF1024_PQ32x4fs_Refine_SQ4_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6917 0.9762 0.9789      0.00443      650621360    68
k_factor_rf=1,nprobe=2                   0.4684 0.6137 0.6151      0.00139       25328878    217
k_factor_rf=1,nprobe=4                   0.5653 0.7665 0.7685      0.00157       49336048    191
k_factor_rf=1,nprobe=8                   0.6341 0.8775 0.8796      0.00185       94928819    163
k_factor_rf=1,nprobe=16                  0.6672 0.9353 0.9376      0.00229      181284631    132
k_factor_rf=1,nprobe=32                  0.6858 0.9654 0.9679      0.00307      343278394    98
k_factor_rf=1,nprobe=64                  0.6917 0.9762 0.9789      0.00466      650621360    65
k_factor_rf=2,nprobe=64                  0.6926 0.9859 0.9896      0.00551      650621360    55
k_factor_rf=2,nprobe=128                 0.6949 0.9904 0.9943      0.01331     1242259908    23
k_factor_rf=4,nprobe=128                 0.6952 0.9930 0.9977      0.01055     1242259908    29
k_factor_rf=2,nprobe=512                 0.6955 0.9913 0.9952      0.02239     4660580015    14
k_factor_rf=4,nprobe=256                 0.6956 0.9938 0.9985      0.01533     2389492167    20
```

</details>
<details><summary>`OPQ32_64,IVF4096_HNSW32,PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 66823982

 code_size 56

 log filename: autotune.dbdeep1M.OPQ32_64_IVF4096_HNSW32_PQ16x4fs_Refine_PCAR64_SQ6_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6227 0.9796 0.9921      0.02150      597520447    14
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.3877 0.5014 0.5016      0.00129        5900821    233
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.4453 0.5982 0.5986      0.00126       11596477    238
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.5240 0.7316 0.7327      0.00148       22860173    203
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.5268 0.7360 0.7370      0.00161       22842342    186
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.5532 0.7883 0.7900      0.00175       44657297    172
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5625 0.8056 0.8075      0.00191       44643834    157
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5658 0.8116 0.8136      0.00206       44588583    146
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5728 0.8289 0.8309      0.00250       86702936    120
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.5757 0.8492 0.8533      0.00263       44643834    115
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5792 0.8559 0.8602      0.00268       44588583    112
k_factor_rf=1,nprobe=64,quantizer_efSearch=16     0.5842 0.8508 0.8531      0.00299      166865784    101
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5882 0.8815 0.8861      0.00305       86702936    99
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5937 0.8928 0.8976      0.00315       86544787    96
k_factor_rf=2,nprobe=64,quantizer_efSearch=32     0.6032 0.9142 0.9191      0.00420      166459175    72
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.6036 0.9153 0.9201      0.00464      166298839    65
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6095 0.9410 0.9486      0.00540      166865784    56
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6115 0.9469 0.9549      0.00658      166298839    46
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.6126 0.9471 0.9547      0.00658      318263706    46
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6147 0.9531 0.9610      0.00682      318354728    45
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.6156 0.9552 0.9631      0.00802      317936628    38
k_factor_rf=4,nprobe=128,quantizer_efSearch=128   0.6159 0.9557 0.9636      0.00916      317640229    33
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.6162 0.9606 0.9710      0.00945      166459175    32
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.6165 0.9613 0.9715      0.00981      166298839    31
k_factor_rf=8,nprobe=128,quantizer_efSearch=16    0.6184 0.9642 0.9746      0.01072      318263706    28
k_factor_rf=8,nprobe=128,quantizer_efSearch=32    0.6202 0.9706 0.9813      0.01077      318354728    28
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6213 0.9732 0.9839      0.01177      317936628    26
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6214 0.9736 0.9843      0.01309      317640229    23
k_factor_rf=8,nprobe=256,quantizer_efSearch=64    0.6218 0.9756 0.9867      0.01480      605788798    21
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.6220 0.9761 0.9873      0.01651      606735392    19
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6226 0.9793 0.9914      0.01955      317936628    16
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.6228 0.9797 0.9919      0.02062      317640229    15
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6239 0.9825 0.9951      0.02333      605788798    13
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6242 0.9831 0.9959      0.02529      606735392    12
k_factor_rf=16,nprobe=512,quantizer_efSearch=128  0.6245 0.9839 0.9969      0.03209     1170846949    10
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.6246 0.9839 0.9968      0.05292     2277603625    6
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6247 0.9856 0.9993      0.05711     1166024929    6
```

</details>
<details><summary>`OPQ32_64,IVF4096_HNSW32,PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 75401490

 code_size 64

 log filename: autotune.dbdeep1M.OPQ32_64_IVF4096_HNSW32_PQ32x4fs_Refine_OPQ48_96_PQ48_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.7565 0.9250 0.9251      0.00286       86894506    105
k_factor_rf=1,nprobe=2,quantizer_efSearch=8      0.4511 0.5155 0.5156      0.00168        5891759    179
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.5303 0.6195 0.6196      0.00170       11579987    177
k_factor_rf=1,nprobe=8,quantizer_efSearch=4      0.6558 0.7790 0.7791      0.00191       22913067    158
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.6629 0.7876 0.7877      0.00195       22897867    154
k_factor_rf=1,nprobe=8,quantizer_efSearch=16     0.6673 0.7937 0.7938      0.00221       22881016    136
k_factor_rf=1,nprobe=16,quantizer_efSearch=8     0.7255 0.8787 0.8788      0.00231       44674575    130
k_factor_rf=1,nprobe=16,quantizer_efSearch=16    0.7300 0.8844 0.8845      0.00240       44617528    126
k_factor_rf=1,nprobe=32,quantizer_efSearch=8     0.7565 0.9250 0.9251      0.00277       86894506    109
k_factor_rf=1,nprobe=64,quantizer_efSearch=8     0.7718 0.9476 0.9477      0.00336      167741244    90
k_factor_rf=1,nprobe=64,quantizer_efSearch=16    0.7836 0.9647 0.9648      0.00349      167415178    86
k_factor_rf=1,nprobe=64,quantizer_efSearch=32    0.7880 0.9701 0.9702      0.00394      167004053    77
k_factor_rf=1,nprobe=64,quantizer_efSearch=64    0.7883 0.9703 0.9704      0.00437      166833100    69
k_factor_rf=1,nprobe=128,quantizer_efSearch=16   0.7885 0.9729 0.9731      0.00485      319398461    62
k_factor_rf=2,nprobe=64,quantizer_efSearch=32    0.7900 0.9761 0.9764      0.00472      167004053    64
k_factor_rf=1,nprobe=128,quantizer_efSearch=32   0.7954 0.9820 0.9822      0.00523      319437050    58
k_factor_rf=2,nprobe=128,quantizer_efSearch=32   0.7977 0.9888 0.9893      0.00597      319437050    51
k_factor_rf=2,nprobe=128,quantizer_efSearch=64   0.7993 0.9905 0.9910      0.00694      318847091    44
k_factor_rf=2,nprobe=128,quantizer_efSearch=128  0.7997 0.9910 0.9915      0.00802      318572527    38
k_factor_rf=2,nprobe=256,quantizer_efSearch=32   0.7998 0.9922 0.9927      0.00850      601354207    36
k_factor_rf=1,nprobe=256,quantizer_efSearch=64   0.7999 0.9886 0.9888      0.00873      608324081    35
k_factor_rf=2,nprobe=256,quantizer_efSearch=64   0.8025 0.9956 0.9961      0.00973      608324081    31
k_factor_rf=2,nprobe=256,quantizer_efSearch=128  0.8031 0.9962 0.9967      0.01196      609119890    26
k_factor_rf=2,nprobe=512,quantizer_efSearch=128  0.8033 0.9970 0.9974      0.01768     1176055381    17
k_factor_rf=8,nprobe=256,quantizer_efSearch=128  0.8035 0.9982 0.9988      0.01881      609119890    16
k_factor_rf=4,nprobe=512,quantizer_efSearch=256  0.8036 0.9986 0.9992      0.02730     1171301384    11
k_factor_rf=4,nprobe=1024,quantizer_efSearch=128 0.8038 0.9989 0.9995      0.02889     2096578127    11
k_factor_rf=4,nprobe=1024,quantizer_efSearch=256 0.8039 0.9990 0.9996      0.04113     2286315416    8
```

</details>
<details><summary>`OPQ32_64,IVF4096_HNSW32,PQ32x4fs,Refine(SQ4)` </summary>
Index size 75267526

 code_size 64

 log filename: autotune.dbdeep1M.OPQ32_64_IVF4096_HNSW32_PQ32x4fs_Refine_SQ4_.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                0.6454 0.9218 0.9261      0.00250       86887775    120
k_factor_rf=1,nprobe=1,quantizer_efSearch=8     0.2915 0.3626 0.3639      0.00130        2975754    232
k_factor_rf=1,nprobe=4,quantizer_efSearch=4     0.4679 0.6231 0.6255      0.00143       11579898    211
k_factor_rf=1,nprobe=8,quantizer_efSearch=4     0.5688 0.7792 0.7822      0.00161       22918687    187
k_factor_rf=1,nprobe=8,quantizer_efSearch=8     0.5733 0.7863 0.7893      0.00172       22900438    175
k_factor_rf=1,nprobe=8,quantizer_efSearch=16    0.5773 0.7917 0.7946      0.00199       22878690    151
k_factor_rf=1,nprobe=16,quantizer_efSearch=8    0.6224 0.8760 0.8798      0.00207       44671583    146
k_factor_rf=1,nprobe=16,quantizer_efSearch=16   0.6274 0.8818 0.8856      0.00213       44613081    141
k_factor_rf=1,nprobe=32,quantizer_efSearch=8    0.6454 0.9218 0.9261      0.00252       86887775    119
k_factor_rf=1,nprobe=64,quantizer_efSearch=8    0.6564 0.9440 0.9484      0.00323      167748048    93
k_factor_rf=1,nprobe=64,quantizer_efSearch=16   0.6663 0.9606 0.9651      0.00337      167397371    90
k_factor_rf=1,nprobe=64,quantizer_efSearch=32   0.6694 0.9657 0.9703      0.00394      167001152    77
k_factor_rf=1,nprobe=128,quantizer_efSearch=16  0.6698 0.9690 0.9735      0.00444      319604100    68
k_factor_rf=1,nprobe=128,quantizer_efSearch=32  0.6753 0.9778 0.9824      0.00488      319367503    62
k_factor_rf=2,nprobe=128,quantizer_efSearch=64  0.6764 0.9847 0.9910      0.00719      318855843    42
k_factor_rf=1,nprobe=128,quantizer_efSearch=128 0.6770 0.9796 0.9843      0.00694      318576359    44
k_factor_rf=1,nprobe=256,quantizer_efSearch=64  0.6789 0.9842 0.9888      0.00857      608159769    35
k_factor_rf=1,nprobe=1024,quantizer_efSearch=64 0.6795 0.9847 0.9893      0.01655     1551738388    19
```

</details>
<details><summary>`OPQ64_128,IVF1024,PQ64` </summary>
Index size 72712955

 code_size 64

 log filename: autotune.dbdeep1M.OPQ64_128_IVF1024_PQ64.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4858 0.5237 0.5239      0.00536       25926910    56
nprobe=1,ht=234                          0.3980 0.4268 0.4268      0.00483       13238554    63
nprobe=1,ht=238                          0.4114 0.4421 0.4421      0.00463       13238554    65
nprobe=1,ht=250                          0.4291 0.4629 0.4630      0.00471       13238554    64
nprobe=1,ht=256                          0.4309 0.4649 0.4649      0.00501       13238554    60
nprobe=2,ht=228                          0.4858 0.5237 0.5239      0.00539       25926910    56
nprobe=2,ht=232                          0.5208 0.5643 0.5643      0.00539       25926910    56
nprobe=2,ht=244                          0.5670 0.6211 0.6211      0.00661       25926910    46
nprobe=2,ht=512                          0.5752 0.6309 0.6310      0.00722       25926910    42
nprobe=4,ht=230                          0.6181 0.6800 0.6801      0.01028       50522646    30
nprobe=4,ht=234                          0.6512 0.7209 0.7210      0.00992       50522646    31
nprobe=4,ht=238                          0.6716 0.7467 0.7468      0.01052       50522646    29
nprobe=4,ht=240                          0.6794 0.7571 0.7572      0.01088       50522646    28
nprobe=4,ht=512                          0.6969 0.7805 0.7806      0.01231       50522646    25
nprobe=8,ht=232                          0.7185 0.8004 0.8004      0.01902       97467356    16
nprobe=8,ht=238                          0.7585 0.8520 0.8521      0.01943       97467356    16
nprobe=8,ht=248                          0.7805 0.8844 0.8844      0.02394       97467356    13
nprobe=8,ht=252                          0.7830 0.8877 0.8878      0.02709       97467356    12
nprobe=8,ht=254                          0.7837 0.8888 0.8890      0.02828       97467356    11
nprobe=16,ht=240                         0.8099 0.9204 0.9205      0.03916      186078317    8
nprobe=16,ht=248                         0.8241 0.9422 0.9422      0.04569      186078317    7
nprobe=16,ht=250                         0.8255 0.9443 0.9444      0.04846      186078317    7
nprobe=16,ht=256                         0.8280 0.9479 0.9480      0.05395      186078317    6
nprobe=32,ht=246                         0.8475 0.9741 0.9741      0.11609      351798577    3
nprobe=32,ht=256                         0.8530 0.9825 0.9826      0.11294      351798577    3
nprobe=64,ht=252                         0.8597 0.9923 0.9924      0.18545      664638089    2
nprobe=64,ht=254                         0.8608 0.9938 0.9940      0.19059      664638089    2
nprobe=128,ht=250                        0.8610 0.9947 0.9948      0.34106     1260720428    1
nprobe=128,ht=254                        0.8631 0.9976 0.9978      0.37032     1260720428    1
nprobe=512,ht=512                        0.8642 0.9999 1.0000      0.96509     4659657225    1
```

</details>
<details><summary>`OPQ64_128,IVF1024,PQ64x4fs,Refine(PQ24)` </summary>
Index size 65208726

 code_size 56

 log filename: autotune.dbdeep1M.OPQ64_128_IVF1024_PQ64x4fs_Refine_PQ24_.b.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                         0.4356 0.8867 0.9486      0.00545      186078317    56
k_factor_rf=1,nprobe=2   0.3270 0.6021 0.6308      0.00312       25926910    97
k_factor_rf=1,nprobe=4   0.3840 0.7380 0.7804      0.00358       50522646    84
k_factor_rf=1,nprobe=8   0.4203 0.8361 0.8902      0.00432       97467356    70
k_factor_rf=1,nprobe=16  0.4356 0.8867 0.9486      0.00546      186078317    55
k_factor_rf=1,nprobe=32  0.4452 0.9158 0.9830      0.00787      351798577    39
k_factor_rf=1,nprobe=128 0.4475 0.9272 0.9988      0.01835     1260720428    17
```

</details>
<details><summary>`OPQ64_128,IVF4096_HNSW32,PQ64` </summary>
Index size 75422256

 code_size 64

 log filename: autotune.dbdeep1M.OPQ64_128_IVF4096_HNSW32_PQ64.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.7362 0.8071 0.8072      0.02657       23401665    12
nprobe=1,quantizer_efSearch=4,ht=150      0.0056 0.0060 0.0061      0.00258        3056407    117
nprobe=1,quantizer_efSearch=8,ht=206      0.0914 0.0947 0.0947      0.00277        3070508    109
nprobe=2,quantizer_efSearch=4,ht=512      0.4615 0.4939 0.4941      0.00344        6029755    88
nprobe=2,quantizer_efSearch=64,ht=250     0.4880 0.5222 0.5222      0.00703        6060284    43
nprobe=4,quantizer_efSearch=4,ht=250      0.5783 0.6258 0.6259      0.00918       11874145    33
nprobe=4,quantizer_efSearch=32,ht=254     0.6283 0.6818 0.6819      0.01042       11942153    29
nprobe=8,quantizer_efSearch=128,ht=512    0.7402 0.8131 0.8133      0.01360       23401672    23
nprobe=16,quantizer_efSearch=32,ht=252    0.8047 0.8919 0.8920      0.03388       45567989    9
nprobe=16,quantizer_efSearch=512,ht=256   0.8080 0.8968 0.8969      0.05930       45553584    6
nprobe=32,quantizer_efSearch=8,ht=244     0.8156 0.9069 0.9070      0.06110       88695543    5
nprobe=32,quantizer_efSearch=128,ht=244   0.8275 0.9223 0.9224      0.06355       88387026    5
nprobe=32,quantizer_efSearch=32,ht=252    0.8427 0.9427 0.9428      0.06437       88445956    5
nprobe=64,quantizer_efSearch=8,ht=250     0.8456 0.9465 0.9466      0.12619      170585335    3
nprobe=64,quantizer_efSearch=32,ht=254    0.8667 0.9742 0.9743      0.13809      169921136    3
nprobe=64,quantizer_efSearch=256,ht=254   0.8681 0.9762 0.9763      0.13790      169667623    3
nprobe=128,quantizer_efSearch=16,ht=512   0.8717 0.9824 0.9826      0.13456      324903585    3
nprobe=512,quantizer_efSearch=32,ht=512   0.8796 0.9941 0.9943      0.43840      983637186    1
nprobe=256,quantizer_efSearch=512,ht=254  0.8799 0.9941 0.9942      0.51649      615259440    1
nprobe=512,quantizer_efSearch=128,ht=256  0.8819 0.9970 0.9971      0.99342     1184519302    1
nprobe=1024,quantizer_efSearch=512,ht=512 0.8837 0.9998 1.0000      1.06333     2271610059    1
```

</details>
<details><summary>`PCAR32,IVF1024_HNSW32,SQfp16` </summary>
Index size 72467797

 code_size 64

 log filename: autotune.dbdeep1M.PCAR32_IVF1024_HNSW32_SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                 0.2059 0.4818 0.5628      0.00331       23440371    91
nprobe=1,quantizer_efSearch=8    0.1725 0.3684 0.4133      0.00268       11937324    112
nprobe=1,quantizer_efSearch=16   0.1729 0.3685 0.4135      0.00295       11936456    102
nprobe=2,quantizer_efSearch=4    0.2059 0.4818 0.5628      0.00330       23440371    91
nprobe=2,quantizer_efSearch=8    0.2093 0.4876 0.5695      0.00378       23394702    80
nprobe=2,quantizer_efSearch=16   0.2094 0.4880 0.5701      0.00404       23391519    75
nprobe=4,quantizer_efSearch=8    0.2350 0.5834 0.7207      0.00480       45823686    63
nprobe=4,quantizer_efSearch=16   0.2351 0.5839 0.7215      0.00492       45799663    62
nprobe=8,quantizer_efSearch=4    0.2452 0.6413 0.8278      0.00680       89272451    45
nprobe=8,quantizer_efSearch=16   0.2478 0.6473 0.8373      0.00728       89158987    42
nprobe=8,quantizer_efSearch=32   0.2483 0.6479 0.8380      0.00763       89137839    40
nprobe=16,quantizer_efSearch=4   0.2518 0.6711 0.8895      0.01117      172752285    27
nprobe=16,quantizer_efSearch=8   0.2531 0.6784 0.9017      0.01123      172509218    27
nprobe=16,quantizer_efSearch=16  0.2537 0.6805 0.9053      0.01138      172343640    27
nprobe=16,quantizer_efSearch=32  0.2541 0.6808 0.9058      0.01187      172301085    26
nprobe=32,quantizer_efSearch=32  0.2562 0.6979 0.9452      0.02040      329850529    15
nprobe=64,quantizer_efSearch=32  0.2571 0.7027 0.9584      0.03507      632505925    9
nprobe=128,quantizer_efSearch=32 0.2572 0.7033 0.9619      0.06410     1225109711    5
```

</details>
<details><summary>`PCAR32,IVF1024,SQfp16` </summary>
Index size 72189600

 code_size 64

 log filename: autotune.dbdeep1M.PCAR32_IVF1024_SQfp16.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2094 0.4880 0.5701      0.00215       23386342    140
nprobe=1                                 0.1728 0.3684 0.4134      0.00190       11936431    159
nprobe=2                                 0.2094 0.4880 0.5701      0.00207       23386342    146
nprobe=4                                 0.2351 0.5836 0.7212      0.00396       45791428    76
nprobe=8                                 0.2481 0.6478 0.8379      0.00605       89128799    50
nprobe=16                                0.2540 0.6808 0.9058      0.00817      172279830    37
nprobe=32                                0.2562 0.6978 0.9452      0.01189      329780601    26
nprobe=64                                0.2571 0.7027 0.9586      0.02053      632250711    15
nprobe=128                               0.2572 0.7033 0.9624      0.03961     1224069483    8
```

</details>
<details><summary>`PCAR32,IVF4096_HNSW32,SQfp16` </summary>
Index size 73719253

 code_size 64

 log filename: autotune.dbdeep1M.PCAR32_IVF4096_HNSW32_SQfp16.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2476 0.6495 0.8341      0.00355       52958197    85
nprobe=2,quantizer_efSearch=4            0.1807 0.3883 0.4304      0.00195        7311058    154
nprobe=4,quantizer_efSearch=4            0.2081 0.4908 0.5692      0.00199       14275063    152
nprobe=4,quantizer_efSearch=8            0.2147 0.5113 0.5954      0.00208       14192783    145
nprobe=8,quantizer_efSearch=4            0.2328 0.5843 0.7138      0.00221       27633832    136
nprobe=8,quantizer_efSearch=8            0.2354 0.5906 0.7220      0.00225       27580411    134
nprobe=8,quantizer_efSearch=16           0.2374 0.5950 0.7278      0.00247       27483663    122
nprobe=8,quantizer_efSearch=32           0.2379 0.5956 0.7284      0.00277       27444270    109
nprobe=16,quantizer_efSearch=4           0.2452 0.6344 0.8107      0.00298       53365551    101
nprobe=16,quantizer_efSearch=8           0.2469 0.6451 0.8288      0.00311       53191098    97
nprobe=16,quantizer_efSearch=16          0.2477 0.6487 0.8330      0.00320       53045731    94
nprobe=32,quantizer_efSearch=8           0.2527 0.6748 0.8893      0.00491      102102711    62
nprobe=32,quantizer_efSearch=16          0.2545 0.6819 0.9017      0.00491      101744920    62
nprobe=32,quantizer_efSearch=32          0.2549 0.6835 0.9044      0.00516      101539855    59
nprobe=32,quantizer_efSearch=64          0.2551 0.6837 0.9048      0.00582      101440641    52
nprobe=64,quantizer_efSearch=16          0.2559 0.6952 0.9365      0.00824      194456543    37
nprobe=64,quantizer_efSearch=32          0.2564 0.6982 0.9407      0.00842      193844938    36
nprobe=128,quantizer_efSearch=64         0.2570 0.7024 0.9571      0.01551      367444921    20
nprobe=128,quantizer_efSearch=256        0.2571 0.7025 0.9569      0.02136      366991101    15
```

</details>
<details><summary>`PCAR64,IVF1024_HNSW32,SQ8` </summary>
Index size 72611797

 code_size 64

 log filename: autotune.dbdeep1M.PCAR64_IVF1024_HNSW32_SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                  0.4509 0.6113 0.6135      0.00646       25332130    47
nprobe=1,quantizer_efSearch=16    0.3570 0.4616 0.4628      0.00461       12985479    66
nprobe=1,quantizer_efSearch=32    0.3571 0.4617 0.4629      0.00511       12984352    59
nprobe=2,quantizer_efSearch=4     0.4509 0.6113 0.6135      0.00648       25332130    47
nprobe=2,quantizer_efSearch=8     0.4586 0.6223 0.6244      0.00664       25306698    46
nprobe=2,quantizer_efSearch=16    0.4594 0.6233 0.6254      0.00694       25302311    44
nprobe=2,quantizer_efSearch=32    0.4595 0.6234 0.6255      0.00743       25297409    41
nprobe=4,quantizer_efSearch=4     0.5269 0.7504 0.7546      0.01076       49274981    28
nprobe=4,quantizer_efSearch=8     0.5396 0.7706 0.7747      0.01090       49194497    28
nprobe=4,quantizer_efSearch=16    0.5402 0.7719 0.7760      0.01123       49177435    27
nprobe=4,quantizer_efSearch=32    0.5406 0.7722 0.7763      0.01171       49170312    26
nprobe=8,quantizer_efSearch=4     0.5870 0.8683 0.8759      0.01889       94926698    16
nprobe=8,quantizer_efSearch=8     0.5920 0.8768 0.8845      0.01893       94874792    16
nprobe=8,quantizer_efSearch=16    0.5942 0.8808 0.8884      0.01920       94786407    16
nprobe=8,quantizer_efSearch=32    0.5945 0.8810 0.8886      0.01964       94761809    16
nprobe=8,quantizer_efSearch=64    0.5946 0.8811 0.8887      0.02068       94756891    15
nprobe=16,quantizer_efSearch=4    0.6141 0.9229 0.9321      0.03387      181477674    9
nprobe=16,quantizer_efSearch=32   0.6236 0.9418 0.9515      0.03461      180859696    9
nprobe=32,quantizer_efSearch=8    0.6304 0.9638 0.9745      0.06160      343525250    5
nprobe=32,quantizer_efSearch=16   0.6334 0.9705 0.9813      0.06152      342916774    5
nprobe=32,quantizer_efSearch=32   0.6342 0.9718 0.9826      0.06203      342599319    5
nprobe=64,quantizer_efSearch=16   0.6354 0.9795 0.9924      0.11368      650864018    3
nprobe=64,quantizer_efSearch=64   0.6371 0.9818 0.9948      0.11512      649257474    3
nprobe=128,quantizer_efSearch=32  0.6375 0.9849 0.9985      0.21514     1242570847    2
nprobe=128,quantizer_efSearch=64  0.6377 0.9854 0.9990      0.21876     1240444840    2
nprobe=256,quantizer_efSearch=128 0.6380 0.9861 1.0000      0.41896     2387428196    1
```

</details>
<details><summary>`PCAR64,IVF1024,SQ8` </summary>
Index size 72333600

 code_size 64

 log filename: autotune.dbdeep1M.PCAR64_IVF1024_SQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4591 0.6233 0.6254      0.00477       25294533    63
nprobe=1                                 0.3567 0.4615 0.4628      0.00256       12983209    118
nprobe=2                                 0.4591 0.6233 0.6254      0.00368       25294533    82
nprobe=4                                 0.5402 0.7719 0.7761      0.00632       49161249    48
nprobe=8                                 0.5945 0.8810 0.8885      0.01115       94739869    27
nprobe=16                                0.6234 0.9416 0.9513      0.02203      180809642    14
nprobe=32                                0.6338 0.9714 0.9822      0.03396      342422723    9
nprobe=64                                0.6371 0.9818 0.9948      0.06265      649004728    5
nprobe=128                               0.6378 0.9854 0.9990      0.17165     1239397997    2
nprobe=256                               0.6381 0.9861 1.0000      0.23752     2386045165    2
```

</details>
<details><summary>`PCAR64,IVF4096_HNSW32,SQ8` </summary>
Index size 74256469

 code_size 64

 log filename: autotune.dbdeep1M.PCAR64_IVF4096_HNSW32_SQ8.k.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6028 0.8839 0.8903      0.00849       61865044    36
nprobe=1,quantizer_efSearch=4            0.2830 0.3452 0.3458      0.00194        4492427    155
nprobe=2,quantizer_efSearch=4            0.3776 0.4865 0.4877      0.00210        8751519    143
nprobe=2,quantizer_efSearch=8            0.3929 0.5071 0.5083      0.00217        8704961    139
nprobe=2,quantizer_efSearch=16           0.3949 0.5089 0.5101      0.00247        8669754    122
nprobe=4,quantizer_efSearch=4            0.4625 0.6242 0.6263      0.00285       17034275    106
nprobe=4,quantizer_efSearch=8            0.4870 0.6584 0.6605      0.00295       16957476    102
nprobe=4,quantizer_efSearch=16           0.4909 0.6623 0.6644      0.00318       16891858    95
nprobe=8,quantizer_efSearch=8            0.5552 0.7828 0.7869      0.00758       32668721    40
nprobe=8,quantizer_efSearch=16           0.5611 0.7914 0.7957      0.00760       32529042    40
nprobe=8,quantizer_efSearch=32           0.5620 0.7927 0.7970      0.00545       32463522    56
nprobe=16,quantizer_efSearch=4           0.5850 0.8520 0.8580      0.01037       62453869    29
nprobe=16,quantizer_efSearch=8           0.5966 0.8736 0.8800      0.00745       62219182    41
nprobe=16,quantizer_efSearch=16          0.6008 0.8813 0.8878      0.00744       62027982    41
nprobe=16,quantizer_efSearch=32          0.6028 0.8839 0.8903      0.00804       61865044    38
nprobe=32,quantizer_efSearch=8           0.6147 0.9200 0.9286      0.01545      117767241    20
nprobe=32,quantizer_efSearch=16          0.6219 0.9337 0.9426      0.01307      117330539    23
nprobe=32,quantizer_efSearch=32          0.6248 0.9389 0.9477      0.01328      116943578    23
nprobe=32,quantizer_efSearch=64          0.6253 0.9398 0.9486      0.01395      116738921    22
nprobe=64,quantizer_efSearch=64          0.6353 0.9682 0.9791      0.02453      218871539    13
nprobe=128,quantizer_efSearch=64         0.6398 0.9807 0.9931      0.04729      409905458    7
nprobe=128,quantizer_efSearch=128        0.6400 0.9809 0.9933      0.04396      409046073    7
nprobe=128,quantizer_efSearch=256        0.6401 0.9810 0.9934      0.04913      408796566    7
nprobe=256,quantizer_efSearch=64         0.6413 0.9851 0.9985      0.07899      768665023    4
nprobe=256,quantizer_efSearch=128        0.6414 0.9854 0.9987      0.07902      771054379    4
nprobe=256,quantizer_efSearch=256        0.6415 0.9856 0.9989      0.08242      770349571    4
```

</details>
