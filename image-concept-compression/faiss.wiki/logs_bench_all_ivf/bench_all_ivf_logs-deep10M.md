# Detailed logs for dataset deep10M

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF16384_HNSW32,PQ16x4fs` </summary>
Index size 170984396

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF16384_HNSW32_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0855 0.2685 0.5192      0.00122       30154955    247
nprobe=1,quantizer_efSearch=4            0.0656 0.1839 0.2999      0.00075        7637336    403
nprobe=1,quantizer_efSearch=8            0.0667 0.1923 0.3124      0.00084        7631373    356
nprobe=2,quantizer_efSearch=4            0.0750 0.2225 0.3987      0.00091       15206750    331
nprobe=2,quantizer_efSearch=8            0.0793 0.2364 0.4240      0.00104       15201984    289
nprobe=4,quantizer_efSearch=4            0.0794 0.2492 0.4818      0.00107       30090770    280
nprobe=4,quantizer_efSearch=8            0.0855 0.2685 0.5192      0.00123       30154955    244
nprobe=8,quantizer_efSearch=8            0.0857 0.2804 0.5860      0.00149       59701297    202
nprobe=4,quantizer_efSearch=16           0.0860 0.2722 0.5252      0.00147       30119062    204
nprobe=4,quantizer_efSearch=32           0.0863 0.2722 0.5255      0.00188       30082280    160
nprobe=16,quantizer_efSearch=8           0.0864 0.2879 0.6254      0.00195      117536490    155
nprobe=16,quantizer_efSearch=32          0.0886 0.2941 0.6353      0.00281      117149685    108
nprobe=16,quantizer_efSearch=128         0.0888 0.2942 0.6357      0.00563      117068677    54
```

</details>
<details><summary>`OPQ16_64,IVF16384_HNSW32,PQ16x4fsr` </summary>
Index size 170979276

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF16384_HNSW32_PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1919 0.5501 0.8164      0.00351      116892054    86
nprobe=2,quantizer_efSearch=8            0.1527 0.3741 0.4754      0.00107       15161092    280
nprobe=4,quantizer_efSearch=4            0.1608 0.4196 0.5641      0.00123       30012296    245
nprobe=4,quantizer_efSearch=8            0.1738 0.4553 0.6147      0.00143       30119948    211
nprobe=4,quantizer_efSearch=16           0.1750 0.4584 0.6195      0.00168       30077385    179
nprobe=8,quantizer_efSearch=4            0.1833 0.5058 0.7157      0.00174       59606216    173
nprobe=8,quantizer_efSearch=8            0.1852 0.5097 0.7249      0.00180       59554277    167
nprobe=8,quantizer_efSearch=16           0.1872 0.5167 0.7365      0.00215       59480835    140
nprobe=16,quantizer_efSearch=8           0.1903 0.5429 0.8031      0.00278      117239628    108
nprobe=16,quantizer_efSearch=16          0.1918 0.5486 0.8128      0.00282      117057148    107
nprobe=16,quantizer_efSearch=32          0.1919 0.5501 0.8164      0.00341      116892054    89
nprobe=32,quantizer_efSearch=16          0.1952 0.5634 0.8612      0.00481      229234551    63
nprobe=32,quantizer_efSearch=32          0.1962 0.5658 0.8676      0.00465      228780398    65
nprobe=32,quantizer_efSearch=64          0.1963 0.5653 0.8678      0.00567      228544760    54
nprobe=64,quantizer_efSearch=16          0.1970 0.5678 0.8815      0.01696      446763685    18
nprobe=64,quantizer_efSearch=32          0.1971 0.5711 0.8898      0.01762      445532400    18
nprobe=64,quantizer_efSearch=64          0.1974 0.5719 0.8904      0.01825      444686999    17
nprobe=64,quantizer_efSearch=256         0.1977 0.5724 0.8912      0.02548      444356994    12
nprobe=512,quantizer_efSearch=64         0.1982 0.5760 0.9034      0.12964     3100721845    3
nprobe=512,quantizer_efSearch=128        0.1988 0.5756 0.9050      0.12486     3166479456    3
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fs` </summary>
Index size 334169292

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_HNSW32_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0860 0.2388 0.3056      0.00142         898236    212
nprobe=2,quantizer_efSearch=4            0.0761 0.2148 0.2780      0.00115         900558    262
nprobe=4,quantizer_efSearch=4            0.0817 0.2471 0.3698      0.00136        1800753    222
nprobe=2,quantizer_efSearch=8            0.0860 0.2388 0.3056      0.00144         898236    209
nprobe=8,quantizer_efSearch=4            0.0918 0.3036 0.5136      0.00173        3602489    174
nprobe=4,quantizer_efSearch=16           0.0967 0.2870 0.4250      0.00236        1794271    128
nprobe=4,quantizer_efSearch=32           0.0977 0.2888 0.4274      0.00364        1791544    83
nprobe=16,quantizer_efSearch=128         0.0981 0.3276 0.6214      0.01180        7130441    26
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fsr` </summary>
Index size 333976268

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_HNSW32_PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2181 0.5381 0.6760      0.00542        7120185    56
nprobe=2,quantizer_efSearch=4            0.1296 0.2544 0.2741      0.00105         896714    286
nprobe=4,quantizer_efSearch=4            0.1521 0.3312 0.3716      0.00133        1794927    227
nprobe=8,quantizer_efSearch=4            0.1909 0.4457 0.5294      0.00202        3588761    149
nprobe=8,quantizer_efSearch=8            0.1950 0.4560 0.5405      0.00226        3585574    133
nprobe=16,quantizer_efSearch=4           0.2047 0.4991 0.6263      0.00306        7169485    99
nprobe=16,quantizer_efSearch=8           0.2126 0.5245 0.6576      0.00334        7153090    90
nprobe=16,quantizer_efSearch=16          0.2150 0.5337 0.6702      0.00376        7137834    80
nprobe=16,quantizer_efSearch=32          0.2181 0.5381 0.6760      0.00521        7120185    58
nprobe=32,quantizer_efSearch=8           0.2191 0.5618 0.7414      0.00530       14273701    57
nprobe=32,quantizer_efSearch=16          0.2247 0.5758 0.7627      0.00615       14231328    49
nprobe=32,quantizer_efSearch=32          0.2272 0.5815 0.7707      0.00768       14197518    40
nprobe=32,quantizer_efSearch=64          0.2276 0.5831 0.7729      0.00976       14176086    31
nprobe=32,quantizer_efSearch=128         0.2279 0.5836 0.7736      0.01445       14169044    21
nprobe=64,quantizer_efSearch=16          0.2291 0.6025 0.8302      0.01806       28328587    17
nprobe=64,quantizer_efSearch=32          0.2329 0.6116 0.8441      0.01780       28244527    17
nprobe=64,quantizer_efSearch=64          0.2340 0.6141 0.8485      0.01986       28183234    16
nprobe=128,quantizer_efSearch=64         0.2348 0.6286 0.8964      0.03306       55885285    10
nprobe=128,quantizer_efSearch=128        0.2351 0.6305 0.8983      0.03741       55778146    9
nprobe=256,quantizer_efSearch=128        0.2360 0.6346 0.9249      0.07706      110182552    4
nprobe=256,quantizer_efSearch=512        0.2362 0.6359 0.9256      0.10805      109962086    3
nprobe=512,quantizer_efSearch=128        0.2364 0.6358 0.9337      0.14005      216570933    3
nprobe=512,quantizer_efSearch=256        0.2377 0.6369 0.9350      0.15040      216284930    2
nprobe=1024,quantizer_efSearch=512       0.2381 0.6372 0.9384      0.30929      422453475    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 269392144

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.0740 0.1771 0.2041      0.00112        1168634    269
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4  0.0568 0.1425 0.1615      0.00099        1174811    305
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=8  0.0583 0.1458 0.1654      0.00100        1868785    302
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4 0.0736 0.1763 0.2033      0.00101        1169064    298
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0740 0.1803 0.2079      0.00106        1863657    284
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0773 0.2204 0.2768      0.00111        1624994    272
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.0789 0.2261 0.2848      0.00113        2317957    266
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4  0.0842 0.2321 0.2958      0.00112        1619630    268
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0871 0.2619 0.3809      0.00131        2527012    230
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4 0.0917 0.2724 0.4033      0.00152        2519402    198
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4  0.0920 0.2944 0.4964      0.00163        4323570    185
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.0939 0.3005 0.5027      0.00177        5014927    170
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 269327888

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2293 0.6285 0.8951      0.04306       76807486    8
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1256 0.2452 0.2663      0.00116        1275646    259
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1380 0.2697 0.2918      0.00117        2312039    256
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1390 0.2737 0.2970      0.00130        1621452    231
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1423 0.2816 0.3059      0.00133        2308062    227
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1643 0.3494 0.3926      0.00159        2529874    189
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1659 0.3586 0.4064      0.00153        2522627    197
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1706 0.3705 0.4183      0.00157        3206395    191
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1726 0.3734 0.4226      0.00169        3205275    178
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1834 0.4241 0.5014      0.00195        4332099    154
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1855 0.4341 0.5169      0.00201        4323711    149
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1919 0.4446 0.5242      0.00206        5005170    146
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1951 0.4574 0.5423      0.00244        4994070    123
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1968 0.4619 0.5484      0.00285        6325484    106
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1987 0.4901 0.6136      0.00296        7914172    102
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.1998 0.4851 0.5975      0.00331        9945519    91
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2097 0.5212 0.6521      0.00349        8576440    87
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2107 0.5235 0.6558      0.00351        8571995    86
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2125 0.5310 0.6663      0.00392        9891574    77
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2136 0.5342 0.6708      0.00422        9883817    72
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2155 0.5578 0.7307      0.00504       15720292    60
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2182 0.5689 0.7494      0.00550       17007405    55
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2188 0.5627 0.7412      0.00551       15696264    55
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2191 0.5629 0.7421      0.00564       15690460    54
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2193 0.5727 0.7551      0.00628       19608550    48
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2209 0.5763 0.7631      0.00614       16972350    49
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2214 0.5788 0.7670      0.00713       19581651    43
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.2224 0.5810 0.7699      0.00891       19566171    34
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2225 0.5826 0.7722      0.01079       24780995    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.2265 0.6095 0.8394      0.01853       38869378    17
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2272 0.6119 0.8445      0.01824       33600503    17
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2275 0.6126 0.8446      0.01980       33587607    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.2277 0.6146 0.8481      0.02282       49138806    14
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2286 0.6240 0.8870      0.03623       61374074    9
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2288 0.6267 0.8906      0.03187       66558839    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2290 0.6273 0.8935      0.03527       66354875    9
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.2292 0.6274 0.8916      0.03758       97426705    9
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2293 0.6285 0.8951      0.03920       76807486    8
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.2294 0.6287 0.8951      0.04304       97294292    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2307 0.6323 0.9134      0.07135      115802232    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2313 0.6352 0.9194      0.07642      120792096    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.2316 0.6364 0.9210      0.07942      151590354    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2324 0.6372 0.9351      0.23556      444855718    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2329 0.6359 0.9331      0.25356      435253498    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs` </summary>
Index size 203874508

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0837 0.2500 0.3878      0.00098        3559570    306
nprobe=2,quantizer_efSearch=4            0.0778 0.2351 0.3619      0.00079        3561262    381
nprobe=2,quantizer_efSearch=8            0.0837 0.2500 0.3878      0.00098        3559570    305
nprobe=2,quantizer_efSearch=16           0.0852 0.2556 0.3949      0.00132        3550173    228
nprobe=8,quantizer_efSearch=4            0.0906 0.2973 0.5673      0.00140       14215874    215
nprobe=16,quantizer_efSearch=32          0.0914 0.3142 0.6460      0.00271       28172684    111
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fsr` </summary>
Index size 203830732

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2082 0.5539 0.7647      0.00442       28140955    68
nprobe=2,quantizer_efSearch=4            0.1464 0.3199 0.3747      0.00089        3557312    337
nprobe=2,quantizer_efSearch=8            0.1574 0.3434 0.4019      0.00107        3553522    280
nprobe=4,quantizer_efSearch=4            0.1652 0.3973 0.4899      0.00117        7108350    257
nprobe=4,quantizer_efSearch=8            0.1777 0.4289 0.5278      0.00136        7108477    221
nprobe=8,quantizer_efSearch=4            0.1929 0.4877 0.6371      0.00176       14198274    171
nprobe=8,quantizer_efSearch=8            0.1933 0.4931 0.6454      0.00186       14186249    162
nprobe=8,quantizer_efSearch=16           0.1979 0.5046 0.6603      0.00235       14149362    128
nprobe=16,quantizer_efSearch=4           0.2006 0.5273 0.7250      0.00271       28282447    111
nprobe=16,quantizer_efSearch=8           0.2053 0.5444 0.7486      0.00310       28236480    97
nprobe=16,quantizer_efSearch=16          0.2078 0.5510 0.7600      0.00356       28182471    85
nprobe=16,quantizer_efSearch=32          0.2082 0.5539 0.7647      0.00466       28140955    65
nprobe=32,quantizer_efSearch=16          0.2107 0.5814 0.8323      0.00553       55910430    55
nprobe=32,quantizer_efSearch=32          0.2116 0.5842 0.8383      0.00639       55813311    47
nprobe=32,quantizer_efSearch=64          0.2121 0.5852 0.8394      0.00840       55755976    36
nprobe=32,quantizer_efSearch=128         0.2123 0.5852 0.8392      0.01200       55740872    26
nprobe=64,quantizer_efSearch=16          0.2129 0.5924 0.8715      0.01664      110693119    19
nprobe=64,quantizer_efSearch=32          0.2136 0.5953 0.8814      0.01692      110391086    18
nprobe=64,quantizer_efSearch=64          0.2140 0.5964 0.8842      0.01845      110214923    17
nprobe=64,quantizer_efSearch=128         0.2142 0.5970 0.8851      0.02146      110151482    15
nprobe=128,quantizer_efSearch=64         0.2145 0.6027 0.9069      0.03173      216998054    10
nprobe=128,quantizer_efSearch=128        0.2148 0.6033 0.9080      0.03425      216714299    9
nprobe=128,quantizer_efSearch=256        0.2149 0.6031 0.9080      0.03895      216627141    8
nprobe=256,quantizer_efSearch=64         0.2160 0.6057 0.9167      0.07111      424864324    5
nprobe=1024,quantizer_efSearch=256       0.2161 0.6066 0.9251      0.26297     1600943216    2
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 187738384

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.0824 0.2670 0.4627      0.00145        7508588    207
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0726 0.1957 0.2722      0.00114        2472078    263
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8 0.0730 0.1971 0.2746      0.00115        2468083    261
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0766 0.2325 0.3560      0.00122        3934960    246
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4  0.0800 0.2415 0.3732      0.00123        3926449    245
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8  0.0832 0.2488 0.3829      0.00126        4254699    239
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8 0.0843 0.2527 0.3906      0.00135        4245684    223
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16 0.0852 0.2539 0.3925      0.00149        4902423    202
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4 0.0853 0.2750 0.4828      0.00153        7486649    197
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0881 0.2827 0.4973      0.00151        7804180    199
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0887 0.2859 0.4986      0.00173        8460435    173
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0900 0.3053 0.5871      0.00268       16837631    112
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 187706384

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1611 0.3494 0.4075      0.00219        4903318    138
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4    0.1232 0.2477 0.2757      0.00119        2131156    252
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1588 0.3459 0.4031      0.00130        4244324    231
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1605 0.3480 0.4061      0.00151        4903006    199
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1695 0.4004 0.4934      0.00151        7494696    199
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1732 0.4113 0.5072      0.00157        7486521    192
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1763 0.4137 0.5094      0.00153        7817526    196
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1807 0.4256 0.5242      0.00157        7809644    191
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1840 0.4314 0.5321      0.00173        8458336    174
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.1849 0.4358 0.5371      0.00213        8456227    141
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1971 0.4976 0.6496      0.00239       14876591    126
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2020 0.5048 0.6601      0.00269       15507997    112
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2021 0.5061 0.6613      0.00286       16831554    105
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2032 0.5429 0.7420      0.00309       29019563    98
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2038 0.5442 0.7487      0.00311       28964182    97
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2089 0.5538 0.7635      0.00359       29556946    84
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2096 0.5533 0.7591      0.00385       30901066    78
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2098 0.5549 0.7666      0.00408       30844395    74
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2106 0.5550 0.7670      0.00456       30835008    66
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2109 0.5554 0.7674      0.00521       33435906    58
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2164 0.5823 0.8353      0.00547       57288423    55
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2169 0.5814 0.8350      0.00571       58609530    53
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2176 0.5847 0.8407      0.00617       58515901    49
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2180 0.5852 0.8414      0.00709       61079375    43
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2185 0.5931 0.8748      0.01633      112100121    19
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2200 0.5976 0.8828      0.01706      113190453    18
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.2229 0.6039 0.8993      0.03170      229689326    10
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.2230 0.6093 0.9226      0.12590      844518116    3
```

</details>
<details><summary>`OPQ16_64,IVF65536,PQ16x4fs` </summary>
Index size 186007063

 code_size 8

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_PQ16x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0911 0.3058 0.5887      0.00897       14122776    34
nprobe=8                                 0.0911 0.3058 0.5887      0.00891       14122776    35
```

</details>
<details><summary>`OPQ8_64,IVF16384_HNSW32,PQ8` </summary>
Index size 168876208

 code_size 8

 log filename: autotune.dbdeep10M.OPQ8_64_IVF16384_HNSW32_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1074 0.2430 0.3074      0.00253       17271862    119
nprobe=1,quantizer_efSearch=4,ht=64      0.0986 0.2261 0.2875      0.00231       17719778    130
nprobe=1,quantizer_efSearch=8,ht=28      0.1050 0.2330 0.2821      0.00235       17271862    128
nprobe=1,quantizer_efSearch=8,ht=30      0.1071 0.2418 0.3009      0.00248       17271862    121
nprobe=1,quantizer_efSearch=8,ht=32      0.1074 0.2430 0.3074      0.00256       17271862    118
nprobe=2,quantizer_efSearch=8,ht=26      0.1251 0.2929 0.3624      0.00265       33671141    114
nprobe=2,quantizer_efSearch=16,ht=64     0.1342 0.3288 0.4476      0.00316       33284279    95
nprobe=4,quantizer_efSearch=32,ht=24     0.1374 0.3288 0.4048      0.00416       63844300    73
nprobe=4,quantizer_efSearch=32,ht=64     0.1475 0.3925 0.5807      0.00456       63844300    66
nprobe=8,quantizer_efSearch=8,ht=28      0.1478 0.4067 0.6207      0.00713      123846552    43
nprobe=8,quantizer_efSearch=32,ht=30     0.1528 0.4292 0.6698      0.00901      121710110    34
nprobe=16,quantizer_efSearch=16,ht=30    0.1549 0.4473 0.7321      0.01468      232335505    21
nprobe=16,quantizer_efSearch=32,ht=30    0.1558 0.4510 0.7378      0.01520      230639620    20
nprobe=16,quantizer_efSearch=64,ht=30    0.1559 0.4514 0.7384      0.01617      229944868    19
nprobe=32,quantizer_efSearch=32,ht=32    0.1568 0.4567 0.7743      0.02682      430476212    12
nprobe=64,quantizer_efSearch=32,ht=64    0.1573 0.4595 0.7898      0.02567      797124304    12
```

</details>
<details><summary>`OPQ8_64,IVF262144_HNSW32,PQ8` </summary>
Index size 300639408

 code_size 8

 log filename: autotune.dbdeep10M.OPQ8_64_IVF262144_HNSW32_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2154 0.5050 0.5943      0.00845        7196418    36
nprobe=1,quantizer_efSearch=4,ht=28      0.0900 0.1363 0.1415      0.00245         449147    123
nprobe=4,quantizer_efSearch=4,ht=64      0.1692 0.3404 0.3720      0.00247        1803423    122
nprobe=4,quantizer_efSearch=8,ht=30      0.1798 0.3392 0.3606      0.00318        1801711    95
nprobe=8,quantizer_efSearch=8,ht=28      0.1904 0.3722 0.3996      0.00491        3600724    62
nprobe=8,quantizer_efSearch=4,ht=30      0.1998 0.4164 0.4571      0.00466        3605288    65
nprobe=8,quantizer_efSearch=32,ht=30     0.2106 0.4378 0.4793      0.00717        3581119    42
nprobe=16,quantizer_efSearch=4,ht=32     0.2154 0.5050 0.5943      0.00877        7196418    35
nprobe=16,quantizer_efSearch=16,ht=32    0.2272 0.5372 0.6330      0.00934        7159244    33
nprobe=16,quantizer_efSearch=64,ht=32    0.2287 0.5406 0.6367      0.01442        7134531    22
nprobe=32,quantizer_efSearch=128,ht=64   0.2404 0.6109 0.7771      0.01781       14206215    17
nprobe=64,quantizer_efSearch=64,ht=64    0.2473 0.6432 0.8569      0.01507       28270913    20
nprobe=512,quantizer_efSearch=32,ht=64   0.2502 0.6634 0.9207      0.06705      206940496    5
nprobe=1024,quantizer_efSearch=256,ht=64 0.2520 0.6731 0.9544      0.18569      422776988    2
```

</details>
<details><summary>`OPQ8_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ8` </summary>
Index size 235862516

 code_size 8

 log filename: autotune.dbdeep10M.OPQ8_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                               0.1894 0.4542 0.5404      0.03095       29287602    10
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=8       0.0037 0.0054 0.0073      0.00238         635021    126
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=20      0.0342 0.0446 0.0483      0.00239        1867088    126
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=28      0.1011 0.1499 0.1549      0.00246        1864454    122
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=28      0.1140 0.1877 0.1948      0.00257        1272712    117
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=30      0.1142 0.1997 0.2084      0.00256        1084152    117
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=32      0.1363 0.2481 0.2596      0.00254        1267382    118
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=64     0.1396 0.2627 0.2777      0.00260        1266687    116
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=30     0.1456 0.2525 0.2631      0.00278        1618920    108
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=32      0.1599 0.3065 0.3252      0.00284        3224217    106
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=32      0.1608 0.3255 0.3504      0.00297        2172493    102
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64     0.1932 0.4199 0.4675      0.00314        6354107    96
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=64      0.2081 0.4768 0.5444      0.00315        5004332    96
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=64     0.2095 0.4947 0.5782      0.00344        8648517    88
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64     0.2163 0.5572 0.6984      0.00701       15060305    43
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=30   0.2205 0.5100 0.5847      0.01146       12486714    27
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=64  0.2346 0.6091 0.7758      0.01747       35173582    18
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64   0.2412 0.6540 0.8900      0.02168       58900376    14
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=64  0.2450 0.6711 0.9360      0.08434      120805815    4
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=64 0.2457 0.6747 0.9494      0.12849      257741004    3
```

</details>
<details><summary>`OPQ8_64,IVF65536_HNSW32,PQ8` </summary>
Index size 195229360

 code_size 8

 log filename: autotune.dbdeep10M.OPQ8_64_IVF65536_HNSW32_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2293 0.5776 0.7460      0.01060       28173827    29
nprobe=1,quantizer_efSearch=4,ht=30      0.1236 0.2214 0.2345      0.00223        1779981    135
nprobe=1,quantizer_efSearch=4,ht=64      0.1274 0.2406 0.2600      0.00217        1779981    139
nprobe=4,quantizer_efSearch=4,ht=64      0.1817 0.4137 0.4889      0.00236        7119090    127
nprobe=4,quantizer_efSearch=8,ht=30      0.1926 0.4194 0.4781      0.00285        7120891    106
nprobe=4,quantizer_efSearch=32,ht=64     0.2001 0.4550 0.5389      0.00352        7097984    86
nprobe=8,quantizer_efSearch=8,ht=28      0.2036 0.4491 0.5174      0.00469       14206664    65
nprobe=8,quantizer_efSearch=4,ht=30      0.2079 0.4850 0.5808      0.00477       14220286    63
nprobe=8,quantizer_efSearch=32,ht=30     0.2155 0.5053 0.6037      0.00554       14150998    55
nprobe=8,quantizer_efSearch=64,ht=32     0.2174 0.5237 0.6429      0.00675       14145082    45
nprobe=16,quantizer_efSearch=16,ht=32    0.2281 0.5747 0.7427      0.00899       28234324    34
nprobe=16,quantizer_efSearch=64,ht=32    0.2293 0.5776 0.7460      0.01075       28173827    28
nprobe=32,quantizer_efSearch=128,ht=64   0.2364 0.6219 0.8529      0.01343       55823058    23
nprobe=64,quantizer_efSearch=64,ht=64    0.2389 0.6401 0.8994      0.01247      110402356    25
nprobe=4096,quantizer_efSearch=64,ht=64  0.2397 0.6494 0.9391      0.21001     1527932136    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2401 0.6501 0.9439      0.17805     1602185534    2
```

</details>
<details><summary>`OPQ8_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ8` </summary>
Index size 179100660

 code_size 8

 log filename: autotune.dbdeep10M.OPQ8_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.0038 0.0064 0.0101      0.43992      894221149    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=18      0.0243 0.0337 0.0375      0.00258        3145628    117
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=22      0.0783 0.1053 0.1124      0.00254        4926784    119
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=32       0.1273 0.2392 0.2572      0.00248        2138089    121
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=32       0.1283 0.2415 0.2590      0.00251        2478938    120
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=30       0.1426 0.2760 0.3003      0.00243        4267482    124
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=28      0.1517 0.2854 0.3069      0.00257        3920419    117
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=64      0.1712 0.3555 0.4017      0.00271        4910946    111
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=64       0.1926 0.4537 0.5568      0.00293       14714686    103
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=64     0.2153 0.5204 0.6462      0.00516       24615784    59
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=64     0.2272 0.5813 0.7616      0.00583       33519862    52
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=64    0.2338 0.6220 0.8522      0.01115       61085345    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=64    0.2367 0.6363 0.8995      0.01618      120695465    19
nprobe=256,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=64   0.2379 0.6467 0.9353      0.08216      428518020    4
nprobe=256,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=64   0.2380 0.6474 0.9377      0.10965      430215856    3
nprobe=1024,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=64 0.2383 0.6485 0.9433      0.35396     1612538653    1
```

</details>
<details><summary>`OPQ8_64,IVF65536,PQ8` </summary>
Index size 177391867

 code_size 8

 log filename: autotune.dbdeep10M.OPQ8_64_IVF65536_PQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2415 0.6510 0.9246      0.03138      216607908    10
nprobe=1,ht=64                           0.1375 0.2606 0.2811      0.00921        1770702    33
nprobe=2,ht=26                           0.1477 0.2462 0.2605      0.00939        3545831    32
nprobe=4,ht=32                           0.2011 0.4483 0.5194      0.01028        7087100    30
nprobe=16,ht=64                          0.2327 0.5905 0.7715      0.01230       28125549    25
nprobe=32,ht=64                          0.2385 0.6250 0.8515      0.01438       55730053    22
nprobe=128,ht=64                         0.2415 0.6510 0.9246      0.03101      216607908    10
nprobe=512,ht=64                         0.2417 0.6545 0.9421      0.09659      826872344    4
```

</details>
<details><summary>`PCAR16,IVF16384_HNSW32,SQ4` </summary>
Index size 165684373

 code_size 8

 log filename: autotune.dbdeep10M.PCAR16_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0466 0.1613 0.3340      0.00459       34145014    66
nprobe=2,quantizer_efSearch=4            0.0404 0.1297 0.2450      0.00294       17441113    103
nprobe=2,quantizer_efSearch=8            0.0426 0.1357 0.2568      0.00313       17305744    96
nprobe=2,quantizer_efSearch=16           0.0427 0.1365 0.2592      0.00318       17265522    95
nprobe=4,quantizer_efSearch=4            0.0431 0.1514 0.3127      0.00294       34353245    102
nprobe=4,quantizer_efSearch=8            0.0466 0.1613 0.3340      0.00298       34145014    101
nprobe=8,quantizer_efSearch=8            0.0509 0.1770 0.3899      0.00450       67264268    67
nprobe=8,quantizer_efSearch=16           0.0510 0.1782 0.3940      0.00494       67086076    61
nprobe=8,quantizer_efSearch=32           0.0511 0.1784 0.3940      0.00819       67028468    37
nprobe=16,quantizer_efSearch=8           0.0522 0.1841 0.4345      0.00872      131760689    35
nprobe=16,quantizer_efSearch=16          0.0527 0.1862 0.4387      0.00764      131503275    40
nprobe=32,quantizer_efSearch=16          0.0539 0.1893 0.4611      0.01325      257249646    23
nprobe=32,quantizer_efSearch=128         0.0540 0.1901 0.4628      0.01592      256621465    19
nprobe=128,quantizer_efSearch=128        0.0541 0.1922 0.4754      0.05628      972917925    6
```

</details>
<details><summary>`PCAR16,IVF262144_HNSW32,SQ4` </summary>
Index size 250261653

 code_size 8

 log filename: autotune.dbdeep10M.PCAR16_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0452 0.1411 0.2293      0.00197        1766768    153
nprobe=1,quantizer_efSearch=4            0.0339 0.0836 0.1017      0.00171         450433    176
nprobe=2,quantizer_efSearch=4            0.0393 0.1079 0.1529      0.00174         891462    173
nprobe=4,quantizer_efSearch=4            0.0425 0.1313 0.2129      0.00180        1756853    167
nprobe=4,quantizer_efSearch=8            0.0452 0.1411 0.2293      0.00201        1766768    150
nprobe=8,quantizer_efSearch=4            0.0484 0.1564 0.2943      0.00198        3488083    152
nprobe=8,quantizer_efSearch=8            0.0493 0.1577 0.2971      0.00206        3490915    146
nprobe=16,quantizer_efSearch=4           0.0497 0.1654 0.3467      0.00218        6871723    138
nprobe=16,quantizer_efSearch=8           0.0512 0.1703 0.3596      0.00228        6885200    132
nprobe=16,quantizer_efSearch=32          0.0513 0.1722 0.3644      0.00328        6892738    92
nprobe=32,quantizer_efSearch=16          0.0517 0.1791 0.4070      0.00340       13593397    89
nprobe=32,quantizer_efSearch=32          0.0520 0.1804 0.4099      0.00387       13597993    78
nprobe=64,quantizer_efSearch=16          0.0522 0.1831 0.4336      0.00496       26805371    61
nprobe=64,quantizer_efSearch=32          0.0526 0.1849 0.4387      0.00500       26819637    60
```

</details>
<details><summary>`PCAR16,IVF262144(IVF512,PQ8x4fs,RFlat),SQ4` </summary>
Index size 182137689

 code_size 8

 log filename: autotune.dbdeep10M.PCAR16_IVF262144_IVF512_PQ8x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.0469 0.1597 0.3671      0.02672      207681681    12
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8  0.0326 0.0862 0.1041      0.00213        1872223    141
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8  0.0383 0.1128 0.1533      0.00219        2356118    138
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2  0.0387 0.1267 0.2013      0.00230        2196022    131
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0415 0.1352 0.2135      0.00230        2556211    131
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.0428 0.1428 0.2571      0.00232        4525382    130
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0432 0.1357 0.2122      0.00244        4655682    123
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=2  0.0446 0.1476 0.2985      0.00247        7622185    122
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0461 0.1529 0.2814      0.00248        4281198    122
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4  0.0500 0.1665 0.3366      0.00269        7812788    112
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0514 0.1687 0.3483      0.00286        8505524    105
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16 0.0520 0.1708 0.3528      0.00322        9848976    94
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8  0.0522 0.1755 0.3875      0.00354       15508643    85
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32 0.0530 0.1802 0.4036      0.00499       19215849    61
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0536 0.1834 0.4337      0.00626       30186618    48
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0537 0.1842 0.4361      0.00690       32816267    44
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.0538 0.1843 0.4360      0.00783       38071538    39
```

</details>
<details><summary>`PCAR16,IVF65536_HNSW32,SQ4` </summary>
Index size 182600341

 code_size 8

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0484 0.1478 0.2877      0.00205        6803075    147
nprobe=2,quantizer_efSearch=4            0.0401 0.1219 0.2059      0.00187        3413879    161
nprobe=4,quantizer_efSearch=4            0.0454 0.1429 0.2779      0.00187        6825773    161
nprobe=4,quantizer_efSearch=8            0.0484 0.1478 0.2877      0.00207        6803075    145
nprobe=8,quantizer_efSearch=8            0.0500 0.1654 0.3513      0.00248       13599272    121
nprobe=8,quantizer_efSearch=16           0.0508 0.1674 0.3548      0.00245       13575398    123
nprobe=8,quantizer_efSearch=32           0.0511 0.1676 0.3549      0.00305       13566116    99
nprobe=16,quantizer_efSearch=16          0.0518 0.1784 0.4059      0.00337       27094828    90
nprobe=16,quantizer_efSearch=32          0.0519 0.1788 0.4067      0.00364       27070006    83
nprobe=32,quantizer_efSearch=16          0.0527 0.1839 0.4358      0.00567       53973079    53
nprobe=32,quantizer_efSearch=32          0.0528 0.1850 0.4376      0.00524       53904771    58
nprobe=32,quantizer_efSearch=64          0.0529 0.1851 0.4373      0.00597       53883769    51
nprobe=64,quantizer_efSearch=16          0.0535 0.1859 0.4550      0.00910      107437979    33
nprobe=64,quantizer_efSearch=32          0.0539 0.1872 0.4587      0.00772      107261586    39
nprobe=64,quantizer_efSearch=64          0.0540 0.1869 0.4590      0.00848      107195836    36
nprobe=128,quantizer_efSearch=128        0.0541 0.1873 0.4665      0.01678      212517488    18
nprobe=128,quantizer_efSearch=64         0.0542 0.1873 0.4664      0.01566      212614961    20
```

</details>
<details><summary>`PCAR16,IVF65536(IVF256,PQ8x4fs,RFlat),SQ4` </summary>
Index size 165587289

 code_size 8

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_IVF256_PQ8x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.0537 0.1791 0.4457      0.01384      107971459    22
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0350 0.0993 0.1524      0.00223        3163444    135
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4    0.0353 0.1009 0.1562      0.00226        2145787    133
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=1    0.0360 0.1076 0.1869      0.00228        3629969    132
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.0445 0.1368 0.2676      0.00228        7501733    132
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2    0.0453 0.1401 0.2734      0.00235        7239851    128
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=4    0.0490 0.1488 0.2939      0.00238        7386204    126
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.0497 0.1591 0.3430      0.00287       14814930    105
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.0501 0.1634 0.3583      0.00258       14703516    117
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0514 0.1666 0.3650      0.00319       15257127    95
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0523 0.1683 0.3931      0.00398       27874919    76
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.0528 0.1734 0.4088      0.00420       28879653    72
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.0531 0.1769 0.4298      0.00621       55708763    49
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.0532 0.1776 0.4312      0.00588       57018249    52
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.0534 0.1778 0.4343      0.00629       55360202    48
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.0536 0.1788 0.4367      0.00896       59299365    34
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.0542 0.1786 0.4455      0.01205      107594502    25
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.0544 0.1801 0.4487      0.02345      116526413    13
```

</details>
<details><summary>`PCAR16,IVF65536,SQ4` </summary>
Index size 164762848

 code_size 8

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0478 0.1541 0.3062      0.00579        7020863    52
nprobe=2                                 0.0429 0.1307 0.2322      0.00521        3547724    58
nprobe=4                                 0.0478 0.1541 0.3062      0.00575        7020863    53
nprobe=8                                 0.0508 0.1698 0.3686      0.00633       13860448    48
nprobe=16                                0.0512 0.1782 0.4148      0.00752       27344729    40
nprobe=32                                0.0513 0.1819 0.4382      0.01048       53812503    29
nprobe=64                                0.0516 0.1838 0.4493      0.01618      105757340    19
nprobe=128                               0.0517 0.1855 0.4560      0.02741      207397460    11
```

</details>

## Code sizes in [9, 16]

<details><summary>`OPQ16_64,IVF16384_HNSW32,PQ16` </summary>
Index size 248876208

 code_size 16

 log filename: autotune.dbdeep10M.OPQ16_64_IVF16384_HNSW32_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3255 0.7509 0.9035      0.03777      434440553    8
nprobe=2,quantizer_efSearch=4,ht=46      0.1497 0.2183 0.2219      0.00243       34848392    124
nprobe=1,quantizer_efSearch=8,ht=58      0.1681 0.2849 0.2966      0.00264       17452002    114
nprobe=1,quantizer_efSearch=16,ht=128    0.1724 0.2957 0.3095      0.00290       17222912    104
nprobe=2,quantizer_efSearch=16,ht=52     0.2079 0.3626 0.3743      0.00330       33527219    91
nprobe=2,quantizer_efSearch=16,ht=54     0.2149 0.3871 0.4044      0.00361       33527219    84
nprobe=4,quantizer_efSearch=4,ht=50      0.2184 0.3872 0.4012      0.00422       67143049    72
nprobe=2,quantizer_efSearch=16,ht=62     0.2231 0.4186 0.4454      0.00589       33527219    51
nprobe=4,quantizer_efSearch=16,ht=50     0.2404 0.4329 0.4493      0.00482       64885636    63
nprobe=4,quantizer_efSearch=8,ht=52      0.2505 0.4710 0.4957      0.00480       65755859    63
nprobe=4,quantizer_efSearch=8,ht=64      0.2646 0.5337 0.5862      0.00701       65755859    43
nprobe=8,quantizer_efSearch=8,ht=128     0.2921 0.6274 0.7119      0.00823      125728803    37
nprobe=8,quantizer_efSearch=16,ht=56     0.2968 0.6267 0.6979      0.00976      124199115    31
nprobe=8,quantizer_efSearch=32,ht=56     0.2989 0.6300 0.7008      0.01030      123390064    30
nprobe=8,quantizer_efSearch=16,ht=60     0.2992 0.6412 0.7245      0.01152      124199115    27
nprobe=8,quantizer_efSearch=32,ht=64     0.3012 0.6483 0.7341      0.01254      123390064    24
nprobe=16,quantizer_efSearch=16,ht=60    0.3162 0.7113 0.8290      0.03078      234932244    10
nprobe=16,quantizer_efSearch=64,ht=64    0.3187 0.7212 0.8427      0.02799      232490713    11
nprobe=32,quantizer_efSearch=16,ht=128   0.3232 0.7503 0.9044      0.02470      437360413    13
nprobe=32,quantizer_efSearch=32,ht=60    0.3255 0.7509 0.9035      0.03886      434440553    8
nprobe=32,quantizer_efSearch=256,ht=64   0.3259 0.7567 0.9118      0.04857      431843781    7
nprobe=64,quantizer_efSearch=32,ht=56    0.3263 0.7516 0.9111      0.08640      805755025    4
nprobe=64,quantizer_efSearch=512,ht=128  0.3278 0.7719 0.9525      0.10575      798746997    3
nprobe=128,quantizer_efSearch=256,ht=128 0.3285 0.7781 0.9692      0.10276     1468734559    3
nprobe=512,quantizer_efSearch=512,ht=128 0.3286 0.7799 0.9774      0.29943     4919190997    2
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16` </summary>
Index size 380639408

 code_size 16

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_HNSW32_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3970 0.7114 0.7383      0.02567       14240543    12
nprobe=2,quantizer_efSearch=4,ht=46       0.0824 0.0992 0.1010      0.00226         900478    133
nprobe=2,quantizer_efSearch=4,ht=128      0.1996 0.2771 0.2796      0.00231         900478    130
nprobe=2,quantizer_efSearch=16,ht=62      0.2161 0.2986 0.3013      0.00376         895998    80
nprobe=8,quantizer_efSearch=8,ht=128      0.3289 0.5359 0.5453      0.00390        3597218    77
nprobe=8,quantizer_efSearch=16,ht=60      0.3302 0.5196 0.5275      0.00780        3584360    39
nprobe=8,quantizer_efSearch=32,ht=64      0.3376 0.5437 0.5531      0.00960        3576996    32
nprobe=32,quantizer_efSearch=16,ht=128    0.4002 0.7372 0.7725      0.01171       14276215    26
nprobe=128,quantizer_efSearch=16,ht=128   0.4216 0.8293 0.8900      0.03887       56287451    8
nprobe=128,quantizer_efSearch=256,ht=128  0.4353 0.8613 0.9298      0.05918       55916243    6
nprobe=256,quantizer_efSearch=128,ht=64   0.4380 0.8790 0.9549      0.18821      110505586    2
nprobe=512,quantizer_efSearch=512,ht=128  0.4409 0.8953 0.9832      0.18500      216646031    2
nprobe=4096,quantizer_efSearch=512,ht=128 0.4420 0.9010 0.9960      1.11445     1501009905    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16` </summary>
Index size 315855860

 code_size 16

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.4171 0.7781 0.8200      0.05016       49254852    6
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=40       0.0364 0.0466 0.0481      0.00238        2317039    127
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=50       0.1019 0.1261 0.1278      0.00242        1088215    124
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=58       0.1332 0.1750 0.1769      0.00248        1092474    122
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=128     0.2476 0.3699 0.3753      0.00276        2170135    109
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.2626 0.3784 0.3826      0.00473        3207266    64
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64       0.2642 0.3895 0.3932      0.00397        3213567    76
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=62     0.2801 0.4157 0.4208      0.00647        4537697    47
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=128    0.3370 0.5458 0.5575      0.00814        8953113    37
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=128    0.3675 0.6366 0.6575      0.00852       17784008    36
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.4053 0.7448 0.7824      0.01351       24829267    23
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=128   0.4055 0.7459 0.7846      0.01928       55646040    16
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128    0.4255 0.8122 0.8662      0.02331       33592040    13
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128   0.4257 0.8153 0.8692      0.03100       38816103    10
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=62    0.4258 0.8267 0.8875      0.09885       58842004    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128   0.4430 0.8911 0.9827      0.17160      227223703    2
nprobe=1024,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=128 0.4431 0.8882 0.9816      0.41507      431678126    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=128 0.4448 0.8992 0.9970      1.13192     1640589056    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16` </summary>
Index size 275229360

 code_size 16

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_HNSW32_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.4017 0.7939 0.8477      0.02169       55894184    14
nprobe=2,quantizer_efSearch=4,ht=46       0.1115 0.1451 0.1482      0.00232        3558142    130
nprobe=1,quantizer_efSearch=4,ht=60       0.1733 0.2494 0.2523      0.00241        1779592    125
nprobe=1,quantizer_efSearch=8,ht=58       0.1800 0.2577 0.2610      0.00255        1778644    118
nprobe=2,quantizer_efSearch=4,ht=128      0.2335 0.3658 0.3734      0.00248        3558142    122
nprobe=2,quantizer_efSearch=16,ht=62      0.2535 0.3965 0.4041      0.00297        3550238    101
nprobe=8,quantizer_efSearch=8,ht=128      0.3448 0.6295 0.6590      0.00363       14204859    83
nprobe=8,quantizer_efSearch=16,ht=60      0.3505 0.6291 0.6539      0.00665       14166307    46
nprobe=8,quantizer_efSearch=32,ht=64      0.3529 0.6414 0.6699      0.00732       14148533    41
nprobe=32,quantizer_efSearch=16,ht=128    0.4019 0.8052 0.8685      0.01045       56000998    29
nprobe=32,quantizer_efSearch=256,ht=64    0.4035 0.8091 0.8705      0.03293       55811823    10
nprobe=128,quantizer_efSearch=16,ht=128   0.4161 0.8570 0.9434      0.03618      218302709    9
nprobe=128,quantizer_efSearch=256,ht=128  0.4206 0.8745 0.9665      0.04597      216982451    7
nprobe=256,quantizer_efSearch=128,ht=64   0.4230 0.8834 0.9796      0.16654      425543717    2
nprobe=512,quantizer_efSearch=512,ht=128  0.4241 0.8896 0.9932      0.16550      828667085    2
nprobe=2048,quantizer_efSearch=512,ht=128 0.4246 0.8909 0.9971      0.59411     3081645778    1
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16` </summary>
Index size 259101684

 code_size 16

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.3797 0.7008 0.7317      0.01387       38576585    22
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8,ht=44       0.0942 0.1237 0.1266      0.00258        4246662    117
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=54       0.1606 0.2221 0.2247      0.00260        2133852    116
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=128      0.1833 0.2689 0.2721      0.00260        3134885    116
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8,ht=54       0.2241 0.3254 0.3301      0.00265        4246850    114
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58       0.2413 0.3699 0.3758      0.00279        4245846    108
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=128       0.2972 0.5036 0.5188      0.00302        7488434    100
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=128      0.3108 0.5231 0.5381      0.00344        9779776    88
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60       0.3454 0.6137 0.6351      0.00667       15558369    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=58       0.3458 0.6067 0.6264      0.00655       15527484    46
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=60       0.3466 0.6188 0.6423      0.00725       14876670    42
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=128    0.3564 0.6452 0.6738      0.00736       24555946    41
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128    0.3888 0.7407 0.7850      0.00908       29539801    34
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=64     0.3894 0.7405 0.7839      0.01450       33422280    21
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56      0.3919 0.7331 0.7673      0.02210       58497562    14
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=62      0.3973 0.7788 0.8339      0.02832       56810693    11
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.4002 0.7794 0.8326      0.04627      111802752    7
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=60     0.4184 0.8300 0.8986      0.04853      113043896    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=64    0.4265 0.8702 0.9596      0.09117      219905855    4
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=64    0.4270 0.8714 0.9605      0.11436      222197373    3
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=128  0.4297 0.8884 0.9935      0.20212      838641570    2
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=128 0.4301 0.8901 0.9968      2.16995     5950322548    1
```

</details>
<details><summary>`OPQ16_64,IVF65536,PQ16` </summary>
Index size 257391867

 code_size 16

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_PQ16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3658 0.6678 0.6945      0.01816       28111534    17
nprobe=1,ht=60                           0.1829 0.2684 0.2721      0.00935        1769483    33
nprobe=2,ht=128                          0.2496 0.4011 0.4096      0.00920        3541771    33
nprobe=8,ht=128                          0.3478 0.6445 0.6743      0.01050       14122776    29
nprobe=16,ht=54                          0.3509 0.6122 0.6327      0.01766       28111534    17
nprobe=32,ht=128                         0.3974 0.8125 0.8754      0.01764       55722362    18
nprobe=64,ht=128                         0.4087 0.8521 0.9324      0.02741      110127311    11
nprobe=256,ht=128                        0.4165 0.8870 0.9850      0.08336      424482635    4
nprobe=1024,ht=128                       0.4179 0.8930 0.9956      0.30212     1602703991    2
nprobe=2048,ht=128                       0.4181 0.8933 0.9968      0.57637     3085617571    1
```

</details>
<details><summary>`OPQ32_64,IVF16384_HNSW32,PQ32x4fs` </summary>
Index size 253017292

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF16384_HNSW32_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2525 0.6804 0.9507      0.04610     2687162238    7
nprobe=2,quantizer_efSearch=4            0.1561 0.3432 0.4108      0.00127       34726647    238
nprobe=2,quantizer_efSearch=8            0.1693 0.3733 0.4432      0.00146       33798364    205
nprobe=4,quantizer_efSearch=4            0.1783 0.4282 0.5301      0.00157       67296133    192
nprobe=4,quantizer_efSearch=8            0.1968 0.4695 0.5823      0.00163       65750907    184
nprobe=8,quantizer_efSearch=4            0.2119 0.5297 0.6824      0.00200      126182077    151
nprobe=8,quantizer_efSearch=8            0.2183 0.5453 0.7030      0.00216      125739912    139
nprobe=8,quantizer_efSearch=16           0.2227 0.5601 0.7209      0.00258      124014827    117
nprobe=16,quantizer_efSearch=4           0.2242 0.5733 0.7590      0.00252      237375869    119
nprobe=16,quantizer_efSearch=8           0.2326 0.6033 0.8019      0.00266      236712097    113
nprobe=16,quantizer_efSearch=16          0.2364 0.6150 0.8180      0.00284      235133079    106
nprobe=16,quantizer_efSearch=32          0.2394 0.6214 0.8263      0.00375      233444943    80
nprobe=32,quantizer_efSearch=32          0.2451 0.6526 0.8863      0.00453      435505661    67
nprobe=32,quantizer_efSearch=64          0.2452 0.6545 0.8886      0.00562      433720209    54
nprobe=64,quantizer_efSearch=16          0.2469 0.6588 0.9056      0.00647      810494864    47
nprobe=64,quantizer_efSearch=32          0.2501 0.6676 0.9197      0.00637      806530258    48
nprobe=64,quantizer_efSearch=64          0.2512 0.6702 0.9248      0.00731      802291574    42
nprobe=128,quantizer_efSearch=128        0.2517 0.6774 0.9432      0.01252     1472583626    24
nprobe=256,quantizer_efSearch=64         0.2519 0.6793 0.9484      0.01646     2693855643    19
nprobe=256,quantizer_efSearch=128        0.2525 0.6800 0.9502      0.01895     2694952838    16
nprobe=256,quantizer_efSearch=256        0.2526 0.6804 0.9508      0.02417     2689323264    13
nprobe=512,quantizer_efSearch=256        0.2527 0.6800 0.9532      0.03915     4931990858    8
```

</details>
<details><summary>`OPQ32_64,IVF16384_HNSW32,PQ32x4fsr` </summary>
Index size 252992204

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF16384_HNSW32_PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3742 0.7734 0.8613      0.00564      117078496    54
nprobe=2,quantizer_efSearch=4            0.2512 0.4359 0.4560      0.00124       15198570    243
nprobe=2,quantizer_efSearch=8            0.2655 0.4613 0.4827      0.00139       15210996    216
nprobe=2,quantizer_efSearch=16           0.2694 0.4667 0.4881      0.00164       15192066    184
nprobe=4,quantizer_efSearch=4            0.2916 0.5464 0.5829      0.00167       30155241    180
nprobe=4,quantizer_efSearch=16           0.3164 0.5925 0.6319      0.00216       30119123    139
nprobe=8,quantizer_efSearch=4            0.3438 0.6739 0.7350      0.00289       59843181    104
nprobe=8,quantizer_efSearch=8            0.3489 0.6857 0.7484      0.00292       59749888    103
nprobe=8,quantizer_efSearch=16           0.3565 0.6987 0.7628      0.00343       59622701    88
nprobe=8,quantizer_efSearch=32           0.3575 0.7012 0.7650      0.00394       59558794    77
nprobe=16,quantizer_efSearch=4           0.3584 0.7383 0.8192      0.00482      117734180    63
nprobe=16,quantizer_efSearch=8           0.3680 0.7612 0.8468      0.00466      117488108    65
nprobe=16,quantizer_efSearch=16          0.3736 0.7705 0.8577      0.00459      117265003    66
nprobe=16,quantizer_efSearch=32          0.3742 0.7734 0.8613      0.00539      117078496    56
nprobe=16,quantizer_efSearch=64          0.3750 0.7738 0.8619      0.00583      117002669    52
nprobe=32,quantizer_efSearch=8           0.3834 0.7993 0.9025      0.01596      230163851    19
nprobe=32,quantizer_efSearch=16          0.3907 0.8141 0.9209      0.01619      229541792    19
nprobe=64,quantizer_efSearch=32          0.3983 0.8394 0.9614      0.03048      445939578    10
nprobe=64,quantizer_efSearch=64          0.3994 0.8427 0.9647      0.03164      445128106    10
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fs` </summary>
Index size 445694156

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF262144_HNSW32_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1882 0.3841 0.4226      0.00170        1800819    177
nprobe=2,quantizer_efSearch=4            0.1422 0.2647 0.2802      0.00102         901746    294
nprobe=4,quantizer_efSearch=4            0.1655 0.3410 0.3751      0.00120        1802635    251
nprobe=4,quantizer_efSearch=8            0.1882 0.3841 0.4226      0.00167        1800819    180
nprobe=8,quantizer_efSearch=4            0.2075 0.4590 0.5318      0.00180        3600579    167
nprobe=8,quantizer_efSearch=8            0.2112 0.4700 0.5454      0.00194        3596840    155
nprobe=16,quantizer_efSearch=4           0.2215 0.5201 0.6294      0.00221        7200151    136
nprobe=16,quantizer_efSearch=8           0.2313 0.5463 0.6616      0.00263        7181514    115
nprobe=16,quantizer_efSearch=16          0.2324 0.5537 0.6709      0.00306        7161221    98
nprobe=32,quantizer_efSearch=8           0.2367 0.5871 0.7414      0.00317       14325564    95
nprobe=64,quantizer_efSearch=8           0.2398 0.6091 0.7947      0.00429       28473462    70
nprobe=32,quantizer_efSearch=16          0.2436 0.6024 0.7630      0.00384       14277561    79
nprobe=64,quantizer_efSearch=16          0.2478 0.6325 0.8306      0.00483       28425861    63
nprobe=64,quantizer_efSearch=32          0.2518 0.6430 0.8464      0.00600       28340542    50
nprobe=128,quantizer_efSearch=32         0.2536 0.6594 0.8926      0.00872       56225170    35
nprobe=128,quantizer_efSearch=128        0.2549 0.6671 0.9056      0.01537       55953781    20
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fsr` </summary>
Index size 445407948

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF262144_HNSW32_PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3617 0.6511 0.6790      0.00720        7124371    42
nprobe=2,quantizer_efSearch=8            0.2109 0.3053 0.3084      0.00168         896899    179
nprobe=4,quantizer_efSearch=4            0.2429 0.3736 0.3801      0.00176        1800736    171
nprobe=4,quantizer_efSearch=8            0.2689 0.4166 0.4238      0.00222        1796570    136
nprobe=4,quantizer_efSearch=16           0.2736 0.4240 0.4311      0.00287        1791209    105
nprobe=8,quantizer_efSearch=4            0.3123 0.5231 0.5377      0.00291        3593109    103
nprobe=8,quantizer_efSearch=8            0.3172 0.5314 0.5466      0.00309        3588647    98
nprobe=16,quantizer_efSearch=4           0.3414 0.6110 0.6368      0.00450        7176269    67
nprobe=16,quantizer_efSearch=8           0.3555 0.6410 0.6684      0.00481        7159379    63
nprobe=16,quantizer_efSearch=32          0.3617 0.6511 0.6790      0.00692        7124371    44
nprobe=16,quantizer_efSearch=64          0.3629 0.6527 0.6808      0.00931        7116262    33
nprobe=16,quantizer_efSearch=128         0.3632 0.6530 0.6811      0.01424        7113755    22
nprobe=32,quantizer_efSearch=8           0.3750 0.7111 0.7532      0.01577       14281079    20
nprobe=32,quantizer_efSearch=16          0.3828 0.7284 0.7731      0.01820       14235146    17
nprobe=32,quantizer_efSearch=32          0.3868 0.7364 0.7817      0.01819       14198956    17
nprobe=32,quantizer_efSearch=64          0.3890 0.7397 0.7851      0.01897       14175071    16
nprobe=32,quantizer_efSearch=128         0.3895 0.7403 0.7858      0.02357       14166777    13
nprobe=64,quantizer_efSearch=16          0.3960 0.7850 0.8468      0.02801       28341143    11
nprobe=64,quantizer_efSearch=32          0.4026 0.7979 0.8614      0.03078       28255305    10
nprobe=64,quantizer_efSearch=128         0.4055 0.8039 0.8685      0.03642       28159802    9
nprobe=128,quantizer_efSearch=64         0.4120 0.8452 0.9280      0.06575       55899978    5
nprobe=128,quantizer_efSearch=128        0.4129 0.8482 0.9311      0.07032       55794917    5
nprobe=128,quantizer_efSearch=256        0.4131 0.8486 0.9316      0.08344       55757819    4
nprobe=256,quantizer_efSearch=64         0.4149 0.8640 0.9594      0.12882      110440180    3
nprobe=256,quantizer_efSearch=128        0.4158 0.8698 0.9659      0.13503      110200096    3
nprobe=256,quantizer_efSearch=256        0.4165 0.8702 0.9671      0.13754      110019878    3
nprobe=256,quantizer_efSearch=512        0.4166 0.8702 0.9671      0.16171      109970504    2
nprobe=512,quantizer_efSearch=64         0.4172 0.8711 0.9715      0.23960      215025696    2
nprobe=512,quantizer_efSearch=128        0.4190 0.8766 0.9806      0.24925      216628127    2
nprobe=512,quantizer_efSearch=256        0.4199 0.8775 0.9828      0.26044      216337022    2
nprobe=1024,quantizer_efSearch=256       0.4218 0.8838 0.9906      0.48848      422082370    1
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 380806672

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1091 0.1878 0.1946      0.00114         819569    263
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.0931 0.1591 0.1652      0.00099         633826    303
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2    0.1088 0.1875 0.1943      0.00109         819775    276
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.1186 0.2148 0.2257      0.00113        1278091    266
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.1200 0.2216 0.2345      0.00115        1084799    262
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1498 0.2738 0.2880      0.00118        2314081    255
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1524 0.2851 0.3003      0.00118        1622209    255
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1553 0.2884 0.3037      0.00120        2314278    251
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1641 0.3241 0.3507      0.00137        3226884    220
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1795 0.3673 0.4030      0.00131        2523987    229
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1817 0.3716 0.4078      0.00142        2522827    212
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1822 0.3723 0.4093      0.00159        2521056    189
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1920 0.4135 0.4668      0.00161        5039864    187
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2002 0.4478 0.5170      0.00169        4324465    178
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2049 0.4547 0.5223      0.00175        5013876    172
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.2131 0.4887 0.5796      0.00211        8659220    143
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.2153 0.5010 0.6025      0.00207        7945300    145
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2246 0.5305 0.6363      0.00226        8611126    133
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2287 0.5399 0.6489      0.00280        9921445    107
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2316 0.5531 0.6678      0.00305        9898471    99
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2347 0.5772 0.7288      0.00424       15750005    71
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2370 0.5870 0.7440      0.00343       15713609    88
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.2416 0.6071 0.7859      0.00433       31461143    70
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2423 0.5984 0.7548      0.00463       19634173    65
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2482 0.6324 0.8320      0.00490       31158603    62
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2506 0.6388 0.8405      0.00628       33718160    48
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2507 0.6518 0.8818      0.00683       59048870    44
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2514 0.6452 0.8478      0.00696       33629385    44
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2529 0.6474 0.8507      0.00881       38834857    35
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2546 0.6650 0.8998      0.01101       66663023    28
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2549 0.6651 0.9001      0.01319       77057718    23
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 380891152

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2593 0.4692 0.5004      0.45848      230014715    1
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1752 0.2539 0.2578      0.00135        1271305    223
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2040 0.2957 0.2999      0.00139        1619007    217
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.2058 0.2980 0.3023      0.00140        1617433    214
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2098 0.3049 0.3092      0.00147        2311695    204
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2119 0.3074 0.3117      0.00158        2312052    191
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2435 0.3808 0.3879      0.00179        2527531    168
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2639 0.4139 0.4217      0.00194        3213863    155
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2669 0.4181 0.4260      0.00211        3209938    142
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2683 0.4198 0.4277      0.00238        3212098    127
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2705 0.4227 0.4309      0.00243        4541926    124
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2774 0.4553 0.4659      0.00256        5036793    118
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3024 0.5079 0.5221      0.00255        4322700    118
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3044 0.5132 0.5272      0.00259        5018298    116
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.3045 0.5108 0.5253      0.00268        4318334    112
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3164 0.5336 0.5487      0.00297        5000164    102
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3225 0.5418 0.5577      0.00349        6328917    86
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3248 0.5750 0.5964      0.00446        9962542    68
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.3354 0.5978 0.6240      0.00416        7918377    73
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3496 0.6334 0.6609      0.00446        8583991    68
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3577 0.6467 0.6750      0.00492        9893848    62
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3597 0.6503 0.6790      0.00577       12511641    52
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3602 0.6504 0.6791      0.00700       17756287    43
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3604 0.6508 0.6794      0.00917       28151529    33
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3737 0.7035 0.7456      0.01431       15745973    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3759 0.7111 0.7550      0.01562       15706405    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3826 0.7212 0.7649      0.01597       17028061    19
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.3861 0.7298 0.7759      0.01636       16973709    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3887 0.7361 0.7812      0.01609       19591695    19
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3909 0.7548 0.8093      0.02699       31439066    12
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.3946 0.7625 0.8174      0.02869       33996343    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.4041 0.8003 0.8635      0.02927       33620745    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4051 0.7986 0.8601      0.03184       38867530    10
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4116 0.8221 0.8939      0.06423       67141593    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.4153 0.8383 0.9188      0.06378       61462641    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.4164 0.8404 0.9219      0.06484       66620076    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.4165 0.8481 0.9304      0.06858       76865304    5
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.4196 0.8563 0.9442      0.12656      132655753    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.4214 0.8671 0.9624      0.12557      121062496    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4219 0.8748 0.9724      0.24737      230843005    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fs` </summary>
Index size 292013772

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_HNSW32_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2068 0.4514 0.5280      0.00131        7118444    229
nprobe=2,quantizer_efSearch=4            0.1630 0.3316 0.3719      0.00084        3558354    356
nprobe=2,quantizer_efSearch=8            0.1744 0.3562 0.3989      0.00104        3559216    289
nprobe=4,quantizer_efSearch=4            0.1898 0.4150 0.4872      0.00106        7110573    284
nprobe=4,quantizer_efSearch=8            0.2068 0.4514 0.5280      0.00124        7118444    243
nprobe=8,quantizer_efSearch=4            0.2215 0.5214 0.6411      0.00148       14213363    203
nprobe=8,quantizer_efSearch=8            0.2238 0.5278 0.6494      0.00153       14198382    196
nprobe=8,quantizer_efSearch=16           0.2303 0.5427 0.6663      0.00187       14166682    161
nprobe=16,quantizer_efSearch=4           0.2344 0.5719 0.7304      0.00193       28339124    156
nprobe=16,quantizer_efSearch=8           0.2413 0.5938 0.7609      0.00208       28273340    145
nprobe=16,quantizer_efSearch=16          0.2438 0.6027 0.7715      0.00233       28217081    129
nprobe=32,quantizer_efSearch=8           0.2450 0.6214 0.8272      0.00271       56135017    111
nprobe=32,quantizer_efSearch=16          0.2514 0.6341 0.8466      0.00297       55996378    102
nprobe=64,quantizer_efSearch=16          0.2530 0.6509 0.8897      0.00398      110896139    76
nprobe=64,quantizer_efSearch=32          0.2542 0.6557 0.8994      0.00451      110596417    67
nprobe=128,quantizer_efSearch=32         0.2558 0.6652 0.9250      0.00623      218034204    49
nprobe=128,quantizer_efSearch=64         0.2571 0.6681 0.9311      0.00755      217421432    40
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fsr` </summary>
Index size 291971276

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_HNSW32_PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3722 0.7326 0.7895      0.00628       28137508    48
nprobe=2,quantizer_efSearch=4            0.2265 0.3628 0.3729      0.00111        3553314    271
nprobe=2,quantizer_efSearch=8            0.2452 0.3921 0.4036      0.00132        3555668    228
nprobe=4,quantizer_efSearch=4            0.2709 0.4708 0.4905      0.00152        7093504    198
nprobe=4,quantizer_efSearch=8            0.2951 0.5127 0.5349      0.00173        7109627    174
nprobe=4,quantizer_efSearch=16           0.2988 0.5198 0.5423      0.00238        7094539    127
nprobe=8,quantizer_efSearch=4            0.3300 0.6140 0.6494      0.00245       14193172    123
nprobe=8,quantizer_efSearch=8            0.3340 0.6223 0.6587      0.00260       14180036    116
nprobe=8,quantizer_efSearch=16           0.3400 0.6361 0.6728      0.00310       14144236    97
nprobe=8,quantizer_efSearch=32           0.3414 0.6390 0.6755      0.00422       14132403    72
nprobe=16,quantizer_efSearch=8           0.3656 0.7186 0.7749      0.00427       28231451    71
nprobe=16,quantizer_efSearch=16          0.3703 0.7290 0.7856      0.00466       28172643    65
nprobe=16,quantizer_efSearch=32          0.3722 0.7326 0.7895      0.00555       28137508    55
nprobe=16,quantizer_efSearch=64          0.3727 0.7336 0.7903      0.00767       28123957    40
nprobe=16,quantizer_efSearch=128         0.3729 0.7338 0.7905      0.01141       28119710    27
nprobe=32,quantizer_efSearch=32          0.3922 0.7996 0.8741      0.01735       55811348    18
nprobe=32,quantizer_efSearch=64          0.3925 0.7999 0.8748      0.01747       55757671    18
nprobe=64,quantizer_efSearch=16          0.3973 0.8290 0.9161      0.02977      110709939    11
nprobe=64,quantizer_efSearch=32          0.4018 0.8374 0.9282      0.03070      110427634    10
nprobe=64,quantizer_efSearch=64          0.4024 0.8387 0.9308      0.03130      110244217    10
nprobe=128,quantizer_efSearch=64         0.4080 0.8593 0.9641      0.07275      217066593    5
nprobe=128,quantizer_efSearch=256        0.4089 0.8602 0.9659      0.07861      216688311    4
nprobe=512,quantizer_efSearch=32         0.4093 0.8636 0.9760      0.25520      765459863    2
nprobe=512,quantizer_efSearch=256        0.4099 0.8748 0.9928      0.26972      828645081    2
nprobe=1024,quantizer_efSearch=64        0.4112 0.8724 0.9886      0.49907     1367715465    1
nprobe=1024,quantizer_efSearch=512       0.4116 0.8757 0.9956      0.55406     1604626054    1
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 275890192

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ32x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2546 0.6621 0.9140      0.00612      223806627    50
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1015 0.1946 0.2091      0.00116        1882519    259
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1231 0.2325 0.2510      0.00111        1971152    270
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1378 0.2576 0.2777      0.00121        2469475    248
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1522 0.3077 0.3451      0.00124        3767212    242
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1616 0.3307 0.3699      0.00127        3930636    237
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1672 0.3462 0.3886      0.00129        3920634    233
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1681 0.3467 0.3899      0.00131        3916464    229
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1723 0.3560 0.3990      0.00135        4253434    223
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1741 0.3582 0.4016      0.00146        4246507    206
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1742 0.3597 0.4025      0.00139        4909673    216
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1928 0.4250 0.4963      0.00148        7506066    204
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2071 0.4517 0.5297      0.00149        7804964    201
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2127 0.5022 0.6140      0.00184       14643528    163
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2151 0.5099 0.6281      0.00189       14593382    159
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2251 0.5329 0.6540      0.00183       14891683    164
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2258 0.5351 0.6573      0.00189       14881944    160
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2390 0.5817 0.7441      0.00223       29029590    135
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2395 0.5903 0.7557      0.00230       28968977    131
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2421 0.6017 0.7697      0.00259       29560356    116
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2436 0.6132 0.8161      0.00297       56961813    101
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2444 0.6184 0.8243      0.00365       56841983    83
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2501 0.6335 0.8453      0.00364       57302117    83
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2519 0.6502 0.8896      0.00426      112253817    71
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2540 0.6563 0.8984      0.00591      113316232    51
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2557 0.6671 0.9283      0.00679      220396338    45
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2567 0.6680 0.9301      0.01010      219940162    30
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.2570 0.6687 0.9324      0.05091      227284204    6
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 275873808

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ32x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2427 0.3955 0.4075      0.00177        4901993    170
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=2     0.1687 0.2531 0.2594      0.00128        1961804    234
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4     0.1783 0.2664 0.2730      0.00132        2133068    228
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1799 0.2694 0.2750      0.00130        3131930    232
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2181 0.3535 0.3649      0.00141        3754060    214
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2288 0.3715 0.3823      0.00140        4263936    215
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2301 0.3742 0.3857      0.00146        3925014    206
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2340 0.3795 0.3917      0.00147        3919630    204
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2367 0.3856 0.3971      0.00144        4254959    210
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2400 0.3905 0.4019      0.00151        4909765    199
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.2425 0.3950 0.4070      0.00163        4907075    185
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2427 0.3955 0.4075      0.00189        4904590    159
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2559 0.4412 0.4603      0.00194        7354921    155
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2622 0.4548 0.4752      0.00196        7341253    154
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2651 0.4510 0.4670      0.00193        7856784    156
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2739 0.4781 0.4986      0.00191        7508664    158
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2902 0.5088 0.5312      0.00204        7810882    148
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2937 0.5161 0.5389      0.00217        8461466    139
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2952 0.5196 0.5425      0.00263        8455129    114
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.2954 0.5198 0.5427      0.00291        8456005    104
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2960 0.5456 0.5726      0.00295       14731528    102
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3072 0.5638 0.5920      0.00304       15010797    99
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3195 0.5998 0.6338      0.00307       14600627    98
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3203 0.6002 0.6343      0.00313       14589657    96
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3320 0.6234 0.6598      0.00300       14890405    100
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3334 0.6239 0.6590      0.00304       15556395    99
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3362 0.6338 0.6708      0.00317       15517209    95
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3373 0.6354 0.6724      0.00335       15509927    90
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.3376 0.6359 0.6730      0.00353       15508749    85
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.3385 0.6376 0.6748      0.00407       16820472    74
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.3387 0.6377 0.6749      0.00445       19419972    68
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3509 0.6785 0.7240      0.00465       29831409    65
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3580 0.7079 0.7602      0.00481       29032632    63
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3612 0.7158 0.7684      0.00477       28966736    63
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3647 0.7256 0.7788      0.00525       30906287    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3655 0.7297 0.7834      0.00503       29565603    60
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3669 0.7327 0.7868      0.00533       30844023    57
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3680 0.7349 0.7889      0.00563       30830419    54
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.3681 0.7349 0.7889      0.00636       30829411    48
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3685 0.7358 0.7898      0.00632       33425423    48
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.3686 0.7359 0.7899      0.00736       33420698    42
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3752 0.7604 0.8256      0.01485       57988786    21
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3757 0.7690 0.8393      0.01480       56948009    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3782 0.7716 0.8440      0.01507       56833091    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3858 0.7917 0.8659      0.01558       57288044    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3889 0.7964 0.8724      0.01547       58506682    20
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3946 0.8288 0.9198      0.02907      112010249    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.3964 0.8340 0.9261      0.02797      115766737    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3973 0.8364 0.9302      0.03025      115567206    10
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.3977 0.8350 0.9285      0.03208      113044336    10
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.4013 0.8486 0.9499      0.06405      230530721    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4054 0.8569 0.9640      0.06312      222531556    5
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.4056 0.8628 0.9747      0.11984      441990480    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4065 0.8720 0.9912      0.24664      838672725    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.4066 0.8696 0.9873      0.24083      834649482    2
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.4067 0.8720 0.9937      0.51272     1614368038    1
```

</details>
<details><summary>`OPQ32_64,IVF65536,PQ32x4fs` </summary>
Index size 274148631

 code_size 16

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_PQ32x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2121 0.4619 0.5400      0.00830        7085824    37
nprobe=1                                 0.1403 0.2611 0.2814      0.00787        1769620    39
nprobe=2                                 0.1781 0.3640 0.4072      0.00807        3541207    38
nprobe=4                                 0.2121 0.4619 0.5400      0.00848        7085824    36
nprobe=8                                 0.2306 0.5442 0.6683      0.00904       14123305    34
nprobe=16                                0.2447 0.6059 0.7754      0.00957       28110923    32
nprobe=32                                0.2513 0.6393 0.8531      0.01203       55718027    25
nprobe=64                                0.2541 0.6581 0.9026      0.01301      110127117    24
nprobe=128                               0.2569 0.6686 0.9325      0.01415      216628250    22
```

</details>
<details><summary>`PCAR16,IVF16384_HNSW32,SQ8` </summary>
Index size 245684373

 code_size 16

 log filename: autotune.dbdeep10M.PCAR16_IVF16384_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0504 0.1664 0.3371      0.00290       34142713    104
nprobe=2,quantizer_efSearch=4            0.0432 0.1346 0.2479      0.00204       17444798    148
nprobe=2,quantizer_efSearch=8            0.0456 0.1407 0.2591      0.00210       17304082    144
nprobe=2,quantizer_efSearch=16           0.0461 0.1421 0.2618      0.00236       17264583    127
nprobe=4,quantizer_efSearch=4            0.0471 0.1560 0.3189      0.00280       34346537    108
nprobe=4,quantizer_efSearch=8            0.0504 0.1664 0.3371      0.00276       34142713    109
nprobe=4,quantizer_efSearch=16           0.0510 0.1679 0.3405      0.00315       34039521    96
nprobe=8,quantizer_efSearch=4            0.0543 0.1811 0.3904      0.00425       67405854    71
nprobe=8,quantizer_efSearch=64           0.0558 0.1850 0.4011      0.00554       67015633    55
nprobe=16,quantizer_efSearch=8           0.0569 0.1946 0.4427      0.00765      131768713    40
nprobe=16,quantizer_efSearch=16          0.0575 0.1957 0.4464      0.00664      131500017    46
nprobe=32,quantizer_efSearch=16          0.0583 0.1997 0.4698      0.01178      257257369    26
```

</details>
<details><summary>`PCAR16,IVF262144_HNSW32,SQ8` </summary>
Index size 330261653

 code_size 16

 log filename: autotune.dbdeep10M.PCAR16_IVF262144_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0474 0.1514 0.2308      0.00203        1767056    148
nprobe=2,quantizer_efSearch=4            0.0404 0.1145 0.1525      0.00179         891473    168
nprobe=4,quantizer_efSearch=4            0.0444 0.1402 0.2127      0.00185        1756605    163
nprobe=4,quantizer_efSearch=8            0.0474 0.1514 0.2308      0.00202        1767056    149
nprobe=8,quantizer_efSearch=4            0.0508 0.1698 0.2986      0.00208        3487982    145
nprobe=8,quantizer_efSearch=8            0.0514 0.1717 0.3008      0.00221        3491149    136
nprobe=16,quantizer_efSearch=4           0.0534 0.1796 0.3557      0.00230        6870034    131
nprobe=16,quantizer_efSearch=8           0.0545 0.1855 0.3672      0.00248        6884474    121
nprobe=16,quantizer_efSearch=16          0.0553 0.1865 0.3709      0.00260        6892254    116
nprobe=32,quantizer_efSearch=8           0.0555 0.1927 0.4134      0.00325       13575536    93
nprobe=32,quantizer_efSearch=16          0.0569 0.1943 0.4198      0.00346       13594040    87
nprobe=32,quantizer_efSearch=32          0.0575 0.1952 0.4230      0.00444       13598301    68
nprobe=64,quantizer_efSearch=32          0.0581 0.2000 0.4550      0.00578       26820353    52
nprobe=512,quantizer_efSearch=64         0.0582 0.2022 0.4816      0.03039      202553850    10
```

</details>
<details><summary>`PCAR16,IVF262144(IVF512,PQ8x4fs,RFlat),SQ8` </summary>
Index size 262137689

 code_size 16

 log filename: autotune.dbdeep10M.PCAR16_IVF262144_IVF512_PQ8x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0512 0.1723 0.3792      0.03067      207681581    10
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1   0.0348 0.0967 0.1262      0.00224        1133483    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.0357 0.1020 0.1347      0.00229        1331588    132
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8   0.0358 0.0897 0.1042      0.00217        1873479    139
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1    0.0360 0.1066 0.1578      0.00221        2091314    136
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2   0.0402 0.1107 0.1449      0.00217        1318356    139
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0459 0.1386 0.2094      0.00228        2633929    132
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.0496 0.1522 0.2624      0.00235        4525373    128
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0510 0.1629 0.2861      0.00244        4281461    123
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0547 0.1760 0.3432      0.00270        7812844    112
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8   0.0560 0.1791 0.3545      0.00279        8505607    108
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0568 0.1821 0.3593      0.00312        9851924    97
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.0570 0.1951 0.4257      0.00555       30706943    55
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.0576 0.1946 0.4152      0.00510       19218607    59
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.0577 0.1943 0.4150      0.00586       24471800    52
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.0584 0.1972 0.4355      0.00542       28841944    56
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.0585 0.2011 0.4475      0.00711       32815460    43
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.0587 0.2010 0.4472      0.00792       38061762    38
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32 0.0589 0.2018 0.4698      0.01367       58667306    22
```

</details>
<details><summary>`PCAR16,IVF65536_HNSW32,SQ8` </summary>
Index size 262600341

 code_size 16

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0522 0.1662 0.3114      0.00180        7022576    167
nprobe=2,quantizer_efSearch=4            0.0467 0.1341 0.2240      0.00166        3534302    181
nprobe=4,quantizer_efSearch=4            0.0494 0.1585 0.2952      0.00163        6993034    184
nprobe=4,quantizer_efSearch=8            0.0522 0.1662 0.3114      0.00179        7022576    168
nprobe=8,quantizer_efSearch=4            0.0542 0.1814 0.3698      0.00215       13855001    140
nprobe=8,quantizer_efSearch=8            0.0546 0.1837 0.3751      0.00203       13859066    148
nprobe=8,quantizer_efSearch=16           0.0565 0.1861 0.3810      0.00232       13869621    130
nprobe=16,quantizer_efSearch=8           0.0573 0.1944 0.4247      0.00295       27346345    102
nprobe=16,quantizer_efSearch=16          0.0583 0.1963 0.4286      0.00318       27357435    95
nprobe=16,quantizer_efSearch=32          0.0585 0.1969 0.4311      0.00347       27360642    87
nprobe=32,quantizer_efSearch=16          0.0590 0.2010 0.4547      0.00547       53863396    55
nprobe=64,quantizer_efSearch=16          0.0593 0.2016 0.4659      0.00795      105885871    38
nprobe=64,quantizer_efSearch=32          0.0595 0.2019 0.4695      0.00811      105866350    37
nprobe=64,quantizer_efSearch=64          0.0596 0.2022 0.4705      0.00824      105844949    37
```

</details>
<details><summary>`PCAR16,IVF65536(IVF256,PQ8x4fs,RFlat),SQ8` </summary>
Index size 245587289

 code_size 16

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_IVF256_PQ8x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0568 0.1861 0.4163      0.00973      109375883    31
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.0384 0.1042 0.1527      0.00224        3163130    134
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4   0.0393 0.1059 0.1567      0.00225        2145794    134
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8   0.0398 0.1086 0.1595      0.00224        2501000    134
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.0482 0.1520 0.2773      0.00233        8490937    129
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0533 0.1620 0.2992      0.00236        7427183    128
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.0549 0.1804 0.3694      0.00247       14703552    122
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.0562 0.1852 0.3770      0.00299       15257249    101
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0572 0.1873 0.4079      0.00363       27875209    83
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0573 0.1869 0.4081      0.00341       27795932    88
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.0575 0.1907 0.4140      0.00383       30384118    79
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.0580 0.1932 0.4243      0.00368       28883159    82
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.0581 0.1937 0.4256      0.00549       37952682    55
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32 0.0584 0.1953 0.4269      0.00495       30122659    61
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.0590 0.1990 0.4547      0.00729       56767096    42
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.0593 0.1979 0.4623      0.00771      107341557    39
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.0599 0.2015 0.4702      0.01410      111932218    22
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.0600 0.2016 0.4705      0.01309      111523966    23
```

</details>
<details><summary>`PCAR16,IVF65536,SQ8` </summary>
Index size 244762848

 code_size 16

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0590 0.1964 0.4310      0.00797       27344729    38
nprobe=8                                 0.0572 0.1853 0.3830      0.00653       13860448    46
nprobe=16                                0.0590 0.1964 0.4310      0.00793       27344729    38
nprobe=32                                0.0601 0.1998 0.4581      0.01073       53812503    28
nprobe=64                                0.0605 0.2009 0.4719      0.01643      105757340    19
```

</details>
<details><summary>`PCAR32,IVF16384_HNSW32,SQ4` </summary>
Index size 246739285

 code_size 16

 log filename: autotune.dbdeep10M.PCAR32_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1658 0.4062 0.5201      0.00529       45244933    57
nprobe=1,quantizer_efSearch=16           0.1079 0.2281 0.2631      0.00252       11686954    119
nprobe=1,quantizer_efSearch=32           0.1081 0.2286 0.2637      0.00304       11667768    99
nprobe=2,quantizer_efSearch=16           0.1419 0.3204 0.3890      0.00342       22915254    88
nprobe=4,quantizer_efSearch=4            0.1543 0.3797 0.4852      0.00504       45904841    60
nprobe=4,quantizer_efSearch=8            0.1658 0.4062 0.5201      0.00493       45244933    61
nprobe=4,quantizer_efSearch=16           0.1699 0.4144 0.5294      0.00536       44855181    56
nprobe=8,quantizer_efSearch=8            0.1788 0.4756 0.6397      0.01161       87670316    26
nprobe=8,quantizer_efSearch=16           0.1857 0.4879 0.6552      0.00900       86959471    34
nprobe=8,quantizer_efSearch=32           0.1861 0.4890 0.6577      0.00899       86637682    34
nprobe=8,quantizer_efSearch=64           0.1866 0.4896 0.6586      0.00958       86534042    32
nprobe=16,quantizer_efSearch=32          0.1950 0.5352 0.7587      0.01457      166772875    21
nprobe=32,quantizer_efSearch=8           0.1974 0.5517 0.8084      0.02909      322374118    11
nprobe=32,quantizer_efSearch=16          0.2015 0.5652 0.8310      0.02620      320973590    12
nprobe=32,quantizer_efSearch=32          0.2036 0.5689 0.8361      0.02493      319530168    13
nprobe=32,quantizer_efSearch=64          0.2041 0.5694 0.8369      0.02586      318724348    12
nprobe=32,quantizer_efSearch=128         0.2042 0.5693 0.8369      0.02781      318553068    11
nprobe=64,quantizer_efSearch=32          0.2048 0.5813 0.8732      0.05739      609115723    6
nprobe=128,quantizer_efSearch=32         0.2057 0.5852 0.8894      0.13285     1155275524    3
nprobe=128,quantizer_efSearch=512        0.2058 0.5860 0.8934      0.12217     1151095869    3
nprobe=256,quantizer_efSearch=64         0.2061 0.5881 0.9012      0.15553     2174462657    2
nprobe=1024,quantizer_efSearch=64        0.2062 0.5881 0.9026      0.53703     6194666857    1
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ4` </summary>
Index size 347045205

 code_size 16

 log filename: autotune.dbdeep10M.PCAR32_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1981 0.5639 0.8674      0.05253      108423048    6
nprobe=2,quantizer_efSearch=4            0.1101 0.2159 0.2309      0.00189         893201    159
nprobe=4,quantizer_efSearch=4            0.1306 0.2873 0.3207      0.00199        1783747    152
nprobe=4,quantizer_efSearch=8            0.1436 0.3118 0.3472      0.00227        1785275    133
nprobe=8,quantizer_efSearch=4            0.1593 0.3772 0.4506      0.00224        3558730    134
nprobe=8,quantizer_efSearch=8            0.1630 0.3837 0.4569      0.00236        3557555    127
nprobe=16,quantizer_efSearch=4           0.1697 0.4344 0.5536      0.00256        7096349    118
nprobe=16,quantizer_efSearch=8           0.1753 0.4505 0.5760      0.00268        7087645    112
nprobe=16,quantizer_efSearch=16          0.1780 0.4571 0.5852      0.00315        7073983    96
nprobe=32,quantizer_efSearch=8           0.1821 0.4905 0.6656      0.00408       14092293    74
nprobe=32,quantizer_efSearch=16          0.1847 0.4986 0.6825      0.00419       14061241    72
nprobe=32,quantizer_efSearch=32          0.1857 0.5051 0.6891      0.00451       14033029    67
nprobe=64,quantizer_efSearch=16          0.1899 0.5246 0.7555      0.00657       27936244    46
nprobe=64,quantizer_efSearch=32          0.1916 0.5333 0.7678      0.00758       27869532    40
nprobe=128,quantizer_efSearch=32         0.1939 0.5515 0.8225      0.01316       55212892    23
nprobe=128,quantizer_efSearch=64         0.1951 0.5547 0.8288      0.01188       55088205    26
nprobe=128,quantizer_efSearch=128        0.1952 0.5552 0.8302      0.01365       55004509    22
nprobe=256,quantizer_efSearch=64         0.1973 0.5617 0.8629      0.02332      108768549    13
nprobe=256,quantizer_efSearch=128        0.1975 0.5627 0.8665      0.02354      108591372    13
nprobe=256,quantizer_efSearch=256        0.1981 0.5640 0.8674      0.02722      108457208    12
nprobe=1024,quantizer_efSearch=128       0.1983 0.5690 0.8898      0.09462      411111460    4
nprobe=1024,quantizer_efSearch=256       0.1987 0.5700 0.8932      0.10231      416321460    3
nprobe=1024,quantizer_efSearch=512       0.1989 0.5703 0.8935      0.12315      416901334    3
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ4` </summary>
Index size 280039065

 code_size 16

 log filename: autotune.dbdeep10M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1832 0.4943 0.6710      0.00679       33703816    45
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.0956 0.1804 0.1915      0.00223        1078882    135
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1150 0.2198 0.2345      0.00223        1610904    135
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1273 0.2605 0.2849      0.00228        2513175    132
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1312 0.2757 0.3070      0.00230        2159592    131
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1420 0.2977 0.3303      0.00236        2507909    128
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1434 0.3032 0.3376      0.00235        2501916    128
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1516 0.3468 0.4017      0.00238        4991026    126
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1544 0.3584 0.4222      0.00247        4302160    122
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1556 0.3674 0.4380      0.00272        8588916    111
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1580 0.3710 0.4418      0.00268        4279297    112
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1608 0.3983 0.4935      0.00277        7893987    109
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1662 0.4148 0.5148      0.00298        8562536    101
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1690 0.4231 0.5243      0.00309        9884289    97
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1757 0.4527 0.5712      0.00317        9851583    95
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1782 0.4738 0.6280      0.00362       15629530    83
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.1815 0.4864 0.6460      0.00475       19520710    64
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.1833 0.4923 0.6682      0.00568       31150671    53
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.1866 0.5055 0.6894      0.00740       19407826    41
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1897 0.5303 0.7578      0.00791       30738050    38
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.1913 0.5350 0.7646      0.00772       33292599    39
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1928 0.5455 0.8034      0.01398       58448708    22
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.1941 0.5503 0.8117      0.01267       66077238    24
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.1942 0.5560 0.8293      0.01474       60447144    21
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.1943 0.5557 0.8262      0.01530       65783037    20
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.1944 0.5565 0.8298      0.02423       65617744    13
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.1945 0.5567 0.8300      0.02043       96397807    15
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1960 0.5664 0.8637      0.02028      114248967    15
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.1964 0.5711 0.8817      0.04123      225267402    8
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.1966 0.5713 0.8828      0.04607      235266266    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.1969 0.5730 0.8917      0.08528      440542022    4
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ4` </summary>
Index size 266800981

 code_size 16

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1895 0.5670 0.8870      0.06705      420219086    5
nprobe=1,quantizer_efSearch=4            0.1038 0.2073 0.2249      0.00179        1777680    168
nprobe=2,quantizer_efSearch=4            0.1260 0.2857 0.3247      0.00186        3549226    162
nprobe=4,quantizer_efSearch=4            0.1433 0.3537 0.4314      0.00191        7074472    157
nprobe=4,quantizer_efSearch=8            0.1510 0.3782 0.4619      0.00214        7082161    140
nprobe=4,quantizer_efSearch=16           0.1526 0.3829 0.4677      0.00245        7069564    123
nprobe=4,quantizer_efSearch=32           0.1528 0.3834 0.4686      0.00287        7064953    105
nprobe=8,quantizer_efSearch=4            0.1664 0.4375 0.5707      0.00263       14122225    114
nprobe=8,quantizer_efSearch=16           0.1699 0.4520 0.5902      0.00287       14088447    105
nprobe=16,quantizer_efSearch=8           0.1785 0.4965 0.6906      0.00440       28018229    69
nprobe=16,quantizer_efSearch=16          0.1795 0.5022 0.6983      0.00445       27978803    68
nprobe=16,quantizer_efSearch=32          0.1800 0.5044 0.7029      0.00478       27945665    63
nprobe=16,quantizer_efSearch=64          0.1801 0.5045 0.7037      0.00586       27931229    52
nprobe=32,quantizer_efSearch=16          0.1846 0.5322 0.7780      0.00662       55427447    46
nprobe=32,quantizer_efSearch=32          0.1847 0.5351 0.7830      0.00700       55333731    43
nprobe=64,quantizer_efSearch=16          0.1866 0.5484 0.8257      0.01135      109582340    27
nprobe=64,quantizer_efSearch=64          0.1868 0.5537 0.8392      0.01264      109158852    24
nprobe=64,quantizer_efSearch=32          0.1870 0.5520 0.8351      0.01226      109313210    25
nprobe=128,quantizer_efSearch=32         0.1884 0.5601 0.8652      0.02191      215328987    14
nprobe=256,quantizer_efSearch=32         0.1892 0.5637 0.8768      0.04037      419634538    8
nprobe=256,quantizer_efSearch=64         0.1894 0.5664 0.8843      0.03926      420506595    8
nprobe=256,quantizer_efSearch=128        0.1896 0.5668 0.8866      0.04167      420459263    8
nprobe=512,quantizer_efSearch=128        0.1897 0.5681 0.8946      0.07393      817357658    5
nprobe=1024,quantizer_efSearch=128       0.1898 0.5684 0.8968      0.16152     1537505153    2
```

</details>
<details><summary>`PCAR32,IVF65536(IVF256,PQ16x4fs,RFlat),SQ4` </summary>
Index size 250084249

 code_size 16

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_IVF256_PQ16x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.1783 0.4956 0.6847      0.00532       28773647    57
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16     0.1083 0.2161 0.2350      0.00226        3126880    133
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=8      0.1086 0.2177 0.2369      0.00227        2462843    133
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.1090 0.2369 0.2696      0.00229        3763674    132
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.1187 0.2674 0.3068      0.00234        3747686    129
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.1325 0.2964 0.3378      0.00231        4902800    130
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8      0.1330 0.2990 0.3414      0.00235        4236860    128
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16     0.1342 0.3021 0.3447      0.00242        4896546    125
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.1527 0.3696 0.4481      0.00250        8448783    121
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.1528 0.3772 0.4601      0.00260        7786607    116
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.1548 0.3814 0.4643      0.00289        9745359    104
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16     0.1549 0.3824 0.4667      0.00297        8416985    102
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32     0.1557 0.3837 0.4685      0.00305        9733943    99
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.1645 0.4193 0.5422      0.00344       14904033    88
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.1675 0.4402 0.5760      0.00351       14855071    86
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.1697 0.4459 0.5857      0.00350       14825678    86
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.1700 0.4451 0.5831      0.00382       16790432    79
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16     0.1716 0.4504 0.5919      0.00389       15436723    78
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.1728 0.4520 0.5936      0.00399       16766733    76
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=64     0.1731 0.4524 0.5942      0.00480       19330901    63
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1783 0.4956 0.6847      0.00536       28774028    56
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1813 0.5013 0.6943      0.00563       29366513    54
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1816 0.5043 0.7011      0.00589       29332386    51
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.1825 0.5031 0.6971      0.00599       30661816    51
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.1826 0.5031 0.6971      0.00647       33265691    47
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128    0.1830 0.5078 0.7057      0.00740       38348060    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1848 0.5211 0.7615      0.00909       56343441    34
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.1875 0.5307 0.7783      0.00929       56846527    33
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.1882 0.5344 0.7848      0.00966       58091893    32
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.1884 0.5352 0.7851      0.01019       60673733    30
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128    0.1885 0.5352 0.7853      0.01086       65809067    28
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.1886 0.5343 0.7855      0.01179       60566748    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.1895 0.5471 0.8287      0.01658      112428274    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.1901 0.5506 0.8363      0.01706      111998000    18
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.1903 0.5512 0.8371      0.01776      114545330    17
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1908 0.5519 0.8546      0.03019      217664241    10
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.1911 0.5519 0.8431      0.03079      224144965    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.1915 0.5592 0.8682      0.03174      217623423    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.1921 0.5600 0.8699      0.03294      225120539    10
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.1924 0.5602 0.8713      0.05787      435312102    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.1932 0.5640 0.8853      0.05907      432236356    6
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.1934 0.5662 0.8956      0.20970     1596196433    2
nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.1935 0.5660 0.8960      0.33276     1599181608    1
```

</details>
<details><summary>`PCAR32,IVF65536,SQ4` </summary>
Index size 248963488

 code_size 16

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1938 0.5686 0.8874      0.04300      419673424    7
nprobe=8                                 0.1734 0.4522 0.5930      0.00766       14055753    40
nprobe=16                                0.1839 0.5063 0.7051      0.00886       27894881    34
nprobe=32                                0.1904 0.5368 0.7843      0.01122       55206472    27
nprobe=64                                0.1915 0.5545 0.8396      0.01687      108950465    18
nprobe=128                               0.1930 0.5640 0.8717      0.02794      214228438    11
nprobe=256                               0.1938 0.5686 0.8874      0.04855      419673424    7
nprobe=512                               0.1940 0.5698 0.8936      0.08262      818787283    4
nprobe=1024                              0.1941 0.5704 0.8963      0.13609     1587567136    3
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF16384_HNSW32,PQ48x4fs` </summary>
Index size 337169797

 code_size 24

 log filename: autotune.dbdeep10M.IVF16384_HNSW32_PQ48x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2838 0.5531 0.6137      0.00206       80470598    146
nprobe=2,quantizer_efSearch=4            0.2097 0.3890 0.4239      0.00152       42901554    197
nprobe=2,quantizer_efSearch=8            0.2336 0.4310 0.4674      0.00174       41844839    173
nprobe=4,quantizer_efSearch=4            0.2503 0.4918 0.5473      0.00182       81940723    165
nprobe=4,quantizer_efSearch=8            0.2838 0.5531 0.6137      0.00205       80470598    147
nprobe=8,quantizer_efSearch=4            0.3052 0.6314 0.7139      0.00243      153583116    124
nprobe=8,quantizer_efSearch=8            0.3129 0.6469 0.7320      0.00258      153297922    117
nprobe=8,quantizer_efSearch=16           0.3233 0.6699 0.7566      0.00286      151431975    106
nprobe=16,quantizer_efSearch=8           0.3395 0.7229 0.8334      0.00343      284708430    88
nprobe=16,quantizer_efSearch=16          0.3454 0.7402 0.8520      0.00371      283470484    81
nprobe=16,quantizer_efSearch=32          0.3480 0.7465 0.8594      0.00445      281378210    68
nprobe=32,quantizer_efSearch=16          0.3579 0.7814 0.9115      0.00578      521340183    52
nprobe=32,quantizer_efSearch=32          0.3611 0.7912 0.9218      0.00608      517859135    50
nprobe=32,quantizer_efSearch=64          0.3628 0.7930 0.9242      0.00709      515706668    43
nprobe=64,quantizer_efSearch=32          0.3656 0.8127 0.9549      0.00829      947169600    37
nprobe=64,quantizer_efSearch=64          0.3667 0.8166 0.9602      0.00945      942183497    32
nprobe=128,quantizer_efSearch=32         0.3677 0.8220 0.9691      0.01267     1711385598    24
nprobe=128,quantizer_efSearch=64         0.3693 0.8269 0.9767      0.01355     1710122796    23
nprobe=128,quantizer_efSearch=128        0.3704 0.8287 0.9786      0.01596     1704044091    19
nprobe=128,quantizer_efSearch=256        0.3706 0.8287 0.9786      0.02300     1701299797    14
nprobe=256,quantizer_efSearch=128        0.3708 0.8317 0.9853      0.02468     3078056832    13
nprobe=256,quantizer_efSearch=256        0.3711 0.8323 0.9858      0.02885     3071813969    11
```

</details>
<details><summary>`IVF262144_HNSW32,PQ48x4fs` </summary>
Index size 590747781

 code_size 24

 log filename: autotune.dbdeep10M.IVF262144_HNSW32_PQ48x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2424 0.4176 0.4330      0.00221        1805075    136
nprobe=1,quantizer_efSearch=4            0.1336 0.1938 0.1963      0.00125         450049    241
nprobe=2,quantizer_efSearch=4            0.1791 0.2809 0.2874      0.00137         904205    219
nprobe=4,quantizer_efSearch=4            0.2164 0.3705 0.3854      0.00156        1808346    193
nprobe=4,quantizer_efSearch=8            0.2424 0.4176 0.4330      0.00226        1805075    133
nprobe=8,quantizer_efSearch=4            0.2797 0.5137 0.5464      0.00228        3604298    132
nprobe=8,quantizer_efSearch=8            0.2857 0.5258 0.5590      0.00249        3601437    121
nprobe=16,quantizer_efSearch=4           0.3041 0.5960 0.6481      0.00277        7203768    109
nprobe=16,quantizer_efSearch=8           0.3185 0.6277 0.6818      0.00327        7187411    92
nprobe=32,quantizer_efSearch=8           0.3321 0.6892 0.7653      0.00423       14325240    71
nprobe=32,quantizer_efSearch=16          0.3392 0.7093 0.7884      0.00513       14286315    59
nprobe=64,quantizer_efSearch=8           0.3407 0.7232 0.8182      0.00552       28474438    55
nprobe=32,quantizer_efSearch=32          0.3425 0.7157 0.7951      0.00705       14247107    43
nprobe=64,quantizer_efSearch=16          0.3521 0.7543 0.8569      0.00661       28433387    46
nprobe=64,quantizer_efSearch=32          0.3572 0.7690 0.8730      0.00871       28340822    35
nprobe=128,quantizer_efSearch=32         0.3616 0.7944 0.9188      0.01069       56223588    29
nprobe=128,quantizer_efSearch=64         0.3654 0.8031 0.9292      0.01564       56057618    20
nprobe=128,quantizer_efSearch=128        0.3658 0.8049 0.9304      0.02176       55943294    14
nprobe=256,quantizer_efSearch=128        0.3666 0.8172 0.9607      0.02915      110439085    11
nprobe=256,quantizer_efSearch=256        0.3672 0.8191 0.9627      0.03905      110249605    8
```

</details>
<details><summary>`IVF262144(IVF512,PQ48x4fs,RFlat),PQ48x4fs` </summary>
Index size 528192713

 code_size 24

 log filename: autotune.dbdeep10M.IVF262144_IVF512_PQ48x4fs_RFlat__PQ48x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.3498 0.7417 0.8370      0.00743       34024733    41
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.1550 0.2411 0.2456      0.00119        1277104    253
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1841 0.2872 0.2929      0.00117        1625554    258
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.2141 0.3649 0.3782      0.00146        2179928    206
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2322 0.3966 0.4109      0.00143        2530198    210
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2351 0.4064 0.4209      0.00156        2525879    193
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2356 0.4076 0.4230      0.00241        2522894    125
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2570 0.4621 0.4867      0.00183        4358189    164
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2773 0.5089 0.5406      0.00204        4322609    147
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2838 0.5232 0.5544      0.00204        5016948    148
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.2853 0.5497 0.5914      0.00228        7995556    132
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.2966 0.5836 0.6340      0.00243        7933619    124
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.2987 0.5809 0.6237      0.00247        8665097    122
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3123 0.6176 0.6701      0.00265        8606534    114
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3200 0.6305 0.6832      0.00346        9936299    87
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3262 0.6415 0.6960      0.00379        9916003    80
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3309 0.6879 0.7623      0.00362       15732326    83
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3326 0.6948 0.7707      0.00408       15699067    74
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.3356 0.7111 0.8028      0.00498       30232852    61
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3417 0.7128 0.7896      0.00567       19655875    53
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3467 0.7350 0.8294      0.00505       31459968    60
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3528 0.7577 0.8621      0.00559       31131456    54
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3566 0.7679 0.8725      0.00809       33709652    38
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3567 0.7717 0.8769      0.00918       33623067    33
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3583 0.7738 0.8794      0.01090       38849875    28
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.3621 0.7885 0.9081      0.01346       67331998    23
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3641 0.8022 0.9272      0.01472       66614899    21
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3645 0.8028 0.9279      0.01702       76945050    18
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3661 0.8047 0.9307      0.01696       66487363    18
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3664 0.8053 0.9310      0.01919       76836783    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.3665 0.8056 0.9313      0.02280       97152161    14
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.3668 0.8062 0.9318      0.03266       97229115    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3674 0.8206 0.9637      0.02755      130970685    11
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3680 0.8280 0.9788      0.03524      237389946    9
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.3688 0.8282 0.9799      0.06354      237040493    5
```

</details>
<details><summary>`IVF65536_HNSW32,PQ48x4fs` </summary>
Index size 388469125

 code_size 24

 log filename: autotune.dbdeep10M.IVF65536_HNSW32_PQ48x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2748 0.5054 0.5435      0.00153        7117940    197
nprobe=1,quantizer_efSearch=4            0.1594 0.2601 0.2701      0.00090        1783169    332
nprobe=2,quantizer_efSearch=4            0.2114 0.3631 0.3831      0.00102        3557936    295
nprobe=2,quantizer_efSearch=8            0.2292 0.3930 0.4140      0.00126        3557204    240
nprobe=4,quantizer_efSearch=4            0.2515 0.4627 0.4985      0.00126        7112744    238
nprobe=4,quantizer_efSearch=8            0.2748 0.5054 0.5435      0.00152        7117940    198
nprobe=8,quantizer_efSearch=4            0.3023 0.5931 0.6535      0.00180       14219357    167
nprobe=8,quantizer_efSearch=8            0.3088 0.6065 0.6667      0.00185       14196885    163
nprobe=16,quantizer_efSearch=4           0.3237 0.6628 0.7483      0.00233       28321848    129
nprobe=16,quantizer_efSearch=8           0.3365 0.6910 0.7801      0.00250       28264640    120
nprobe=16,quantizer_efSearch=16          0.3397 0.7008 0.7911      0.00293       28213471    103
nprobe=32,quantizer_efSearch=8           0.3493 0.7393 0.8490      0.00338       56108203    89
nprobe=32,quantizer_efSearch=16          0.3539 0.7548 0.8685      0.00371       55974112    81
nprobe=32,quantizer_efSearch=32          0.3568 0.7599 0.8743      0.00463       55865760    65
nprobe=64,quantizer_efSearch=16          0.3640 0.7887 0.9149      0.00566      110793263    53
nprobe=64,quantizer_efSearch=32          0.3670 0.7984 0.9278      0.00577      110475843    53
nprobe=128,quantizer_efSearch=32         0.3690 0.8129 0.9555      0.00754      217731482    40
nprobe=128,quantizer_efSearch=64         0.3706 0.8167 0.9606      0.00958      217106433    32
nprobe=256,quantizer_efSearch=32         0.3711 0.8194 0.9661      0.01274      423997857    24
nprobe=256,quantizer_efSearch=64         0.3728 0.8250 0.9764      0.01330      424986973    23
nprobe=512,quantizer_efSearch=128        0.3741 0.8285 0.9839      0.02607      825270002    12
```

</details>
<details><summary>`IVF65536(IVF256,PQ48x4fs,RFlat),PQ48x4fs` </summary>
Index size 372960457

 code_size 24

 log filename: autotune.dbdeep10M.IVF65536_IVF256_PQ48x4fs_RFlat__PQ48x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.3678 0.8083 0.9471      0.00991      223176068    31
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1689 0.2762 0.2863      0.00125        2482575    240
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.1713 0.2959 0.3135      0.00140        3668824    215
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1942 0.3368 0.3537      0.00141        3946901    213
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1984 0.3449 0.3640      0.00140        3759482    215
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2120 0.3677 0.3882      0.00145        3928103    208
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2164 0.3772 0.3977      0.00144        3923215    208
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2264 0.3908 0.4116      0.00145        4262102    207
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.2284 0.3948 0.4160      0.00186        4257929    162
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.2439 0.4518 0.4863      0.00180        7334541    167
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2747 0.5073 0.5461      0.00180        7815466    167
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3084 0.6037 0.6640      0.00215       14912223    140
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3104 0.6099 0.6716      0.00232       14886811    130
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3329 0.6821 0.7702      0.00274       29004965    110
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3343 0.6859 0.7751      0.00273       28962801    111
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.3455 0.7304 0.8410      0.00341       56879984    88
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.3475 0.7336 0.8446      0.00359       56795179    84
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3488 0.7321 0.8371      0.00430       57984024    70
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3558 0.7544 0.8684      0.00425       57265458    71
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3591 0.7745 0.8974      0.00459      113552951    66
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3642 0.7895 0.9171      0.00493      112021468    61
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3644 0.7919 0.9203      0.00542      111867753    56
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.3650 0.7996 0.9354      0.00696      222636565    44
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.3666 0.8062 0.9461      0.00724      219222653    42
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3668 0.7990 0.9297      0.00780      112906666    39
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3675 0.8005 0.9315      0.00907      115404293    34
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.3678 0.8083 0.9471      0.00950      223176288    32
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.3694 0.8160 0.9594      0.01000      219817523    31
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.3700 0.8155 0.9594      0.01139      219539118    27
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3706 0.8173 0.9620      0.01402      226814973    22
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.3710 0.8218 0.9717      0.01430      437465618    21
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3718 0.8256 0.9769      0.01609      429945020    19
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3723 0.8266 0.9779      0.01790      434756004    17
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.3726 0.8252 0.9791      0.02787      833140832    11
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3733 0.8293 0.9851      0.02795      837091917    11
```

</details>
<details><summary>`IVF65536,PQ48x4fs` </summary>
Index size 370650832

 code_size 24

 log filename: autotune.dbdeep10M.IVF65536_PQ48x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3723 0.8268 0.9790      0.02117      423475758    15
nprobe=2                                 0.2330 0.4001 0.4215      0.00982        3543332    31
nprobe=4                                 0.2791 0.5157 0.5547      0.01027        7084698    30
nprobe=8                                 0.3163 0.6235 0.6851      0.01055       14115526    29
nprobe=16                                0.3418 0.7041 0.7957      0.01320       28092996    24
nprobe=32                                0.3571 0.7619 0.8767      0.01311       55672567    23
nprobe=128                               0.3707 0.8173 0.9622      0.01584      216193366    19
nprobe=256                               0.3723 0.8268 0.9790      0.02164      423475758    14
nprobe=512                               0.3732 0.8295 0.9854      0.03396      825310440    9
```

</details>
<details><summary>`OPQ32_128,IVF16384_HNSW32,PQ32` </summary>
Index size 413160624

 code_size 32

 log filename: autotune.dbdeep10M.OPQ32_128_IVF16384_HNSW32_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.0002 0.0005 0.0008      1.38310     9408547122    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0078 0.0092 0.0096      0.00288       22314886    105
nprobe=1,quantizer_efSearch=16,ht=68      0.0161 0.0189 0.0195      0.00354       21217730    85
nprobe=1,quantizer_efSearch=32,ht=70      0.0187 0.0215 0.0222      0.00415       21149998    73
nprobe=1,quantizer_efSearch=16,ht=114     0.2531 0.3207 0.3215      0.00433       21217730    70
nprobe=1,quantizer_efSearch=16,ht=116     0.2551 0.3257 0.3266      0.00447       21217730    68
nprobe=2,quantizer_efSearch=16,ht=110     0.3308 0.4357 0.4367      0.00606       41294034    50
nprobe=2,quantizer_efSearch=64,ht=120     0.3501 0.4752 0.4770      0.00987       41081835    31
nprobe=2,quantizer_efSearch=64,ht=124     0.3513 0.4780 0.4796      0.01032       41081835    30
nprobe=4,quantizer_efSearch=32,ht=116     0.4302 0.6194 0.6222      0.01241       78809217    25
nprobe=4,quantizer_efSearch=32,ht=256     0.4347 0.6313 0.6350      0.01169       78809217    26
nprobe=8,quantizer_efSearch=32,ht=106     0.4538 0.6394 0.6414      0.01708      150487088    18
nprobe=8,quantizer_efSearch=32,ht=116     0.4970 0.7470 0.7522      0.02085      150487088    15
nprobe=8,quantizer_efSearch=16,ht=128     0.4989 0.7566 0.7629      0.02559      151506070    12
nprobe=8,quantizer_efSearch=64,ht=126     0.5012 0.7603 0.7666      0.02672      150077935    12
nprobe=16,quantizer_efSearch=64,ht=110    0.5208 0.7959 0.8021      0.03323      280450978    10
nprobe=16,quantizer_efSearch=64,ht=256    0.5417 0.8558 0.8673      0.03425      280450978    9
nprobe=32,quantizer_efSearch=32,ht=116    0.5569 0.8981 0.9136      0.07735      518004372    4
nprobe=32,quantizer_efSearch=512,ht=128   0.5626 0.9143 0.9327      0.12091      514653246    3
nprobe=64,quantizer_efSearch=256,ht=114   0.5673 0.9224 0.9435      0.13597      938973714    3
nprobe=64,quantizer_efSearch=128,ht=126   0.5744 0.9459 0.9701      0.15838      939968242    2
nprobe=128,quantizer_efSearch=64,ht=256   0.5777 0.9585 0.9873      0.18826     1710418478    2
nprobe=128,quantizer_efSearch=512,ht=256  0.5784 0.9597 0.9889      0.21835     1700564007    2
nprobe=256,quantizer_efSearch=512,ht=122  0.5785 0.9623 0.9930      0.55744     3069136796    1
nprobe=512,quantizer_efSearch=64,ht=128   0.5787 0.9627 0.9949      1.01824     5316589783    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.5788 0.9636 0.9966      1.98643    10279747285    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.5799 0.9654 0.9995      2.05408    10279747285    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.5801 0.9654 0.9995      3.73511    28562933655    1
```

</details>
<details><summary>`OPQ32_128,IVF262144_HNSW32,PQ32` </summary>
Index size 607838384

 code_size 32

 log filename: autotune.dbdeep10M.OPQ32_128_IVF262144_HNSW32_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6949 0.9254 0.9271      0.39217      110195224    1
nprobe=1,quantizer_efSearch=4,ht=106      0.1071 0.1164 0.1170      0.00287         449459    105
nprobe=2,quantizer_efSearch=4,ht=102      0.1220 0.1342 0.1348      0.00387         903384    78
nprobe=1,quantizer_efSearch=16,ht=114     0.1710 0.1857 0.1863      0.00497         447936    61
nprobe=1,quantizer_efSearch=16,ht=116     0.1792 0.1956 0.1962      0.00483         447936    63
nprobe=2,quantizer_efSearch=16,ht=110     0.2111 0.2332 0.2338      0.00601         897149    50
nprobe=4,quantizer_efSearch=8,ht=118      0.3487 0.4001 0.4008      0.00766        1805142    40
nprobe=4,quantizer_efSearch=32,ht=256     0.3804 0.4432 0.4439      0.00794        1793434    38
nprobe=8,quantizer_efSearch=16,ht=114     0.4160 0.4838 0.4845      0.01415        3588273    22
nprobe=8,quantizer_efSearch=16,ht=128     0.4742 0.5695 0.5702      0.01413        3588273    22
nprobe=16,quantizer_efSearch=64,ht=256    0.5614 0.6998 0.7007      0.01790        7138001    17
nprobe=32,quantizer_efSearch=8,ht=124     0.5949 0.7552 0.7563      0.04560       14322887    7
nprobe=64,quantizer_efSearch=64,ht=126    0.6670 0.8739 0.8754      0.09927       28267856    4
nprobe=128,quantizer_efSearch=512,ht=256  0.6988 0.9360 0.9378      0.12728       55891432    3
nprobe=256,quantizer_efSearch=512,ht=122  0.7036 0.9448 0.9469      0.39514      110195224    1
nprobe=512,quantizer_efSearch=64,ht=128   0.7138 0.9691 0.9716      0.67460      215419148    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.7151 0.9689 0.9712      1.39858      422848503    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.7222 0.9890 0.9917      1.42575      422848503    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.7244 0.9956 0.9987      2.02849     1500186636    1
```

</details>
<details><summary>`OPQ32_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ32` </summary>
Index size 547512820

 code_size 32

 log filename: autotune.dbdeep10M.OPQ32_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ32.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.6650 0.8717 0.8738      0.09529       33618435    4
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=108       0.1271 0.1371 0.1377      0.00265         816741    114
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=116       0.1400 0.1518 0.1524      0.00253         635623    119
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=256      0.2196 0.2467 0.2475      0.00370        1084469    82
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=122      0.2498 0.2782 0.2788      0.00460        3678648    66
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=122      0.2794 0.3132 0.3140      0.00486        3669784    62
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=256       0.4009 0.4807 0.4815      0.00537        3983226    56
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=124       0.4634 0.5556 0.5564      0.01370        5009780    22
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=126      0.4654 0.5618 0.5626      0.01938        6348404    16
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=128     0.4760 0.5734 0.5742      0.01896       14263280    16
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=256     0.5636 0.7202 0.7219      0.02233       15054235    14
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=126     0.6132 0.7854 0.7868      0.04497       16997962    7
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=256     0.6401 0.8381 0.8402      0.06892       29819571    5
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=126    0.6650 0.8717 0.8738      0.10265       33618435    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128    0.6684 0.8777 0.8799      0.10051       38857482    3
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=256  0.6987 0.9352 0.9383      0.16327       76816122    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=256   0.7080 0.9615 0.9653      0.20324      115713277    2
nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=256  0.7212 0.9922 0.9967      1.30461      837153415    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=256 0.7220 0.9947 0.9994      3.59010     1635394861    1
```

</details>
<details><summary>`OPQ32_128,IVF65536_HNSW32,PQ32` </summary>
Index size 452096688

 code_size 32

 log filename: autotune.dbdeep10M.OPQ32_128_IVF65536_HNSW32_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.1829 0.2078 0.2089      0.00343        3560254    88
nprobe=1,quantizer_efSearch=4,ht=56       0.0064 0.0084 0.0089      0.00241        1784334    125
nprobe=1,quantizer_efSearch=4,ht=68       0.0125 0.0157 0.0161      0.00237        1784334    127
nprobe=1,quantizer_efSearch=4,ht=106      0.1682 0.1898 0.1906      0.00249        1784334    121
nprobe=2,quantizer_efSearch=4,ht=102      0.1829 0.2078 0.2089      0.00360        3560254    84
nprobe=1,quantizer_efSearch=16,ht=114     0.2275 0.2598 0.2607      0.00359        1776167    84
nprobe=1,quantizer_efSearch=16,ht=116     0.2351 0.2698 0.2707      0.00362        1776167    83
nprobe=2,quantizer_efSearch=16,ht=110     0.2928 0.3400 0.3412      0.00474        3551253    64
nprobe=4,quantizer_efSearch=32,ht=256     0.4486 0.5533 0.5548      0.00592        7097352    51
nprobe=8,quantizer_efSearch=16,ht=114     0.4959 0.6125 0.6143      0.01153       14162379    27
nprobe=8,quantizer_efSearch=32,ht=116     0.5103 0.6373 0.6392      0.01283       14147062    24
nprobe=8,quantizer_efSearch=16,ht=128     0.5338 0.6813 0.6832      0.01290       14162379    24
nprobe=16,quantizer_efSearch=64,ht=256    0.6022 0.7959 0.7985      0.01435       28153064    21
nprobe=32,quantizer_efSearch=32,ht=116    0.6217 0.8164 0.8190      0.04042       55864063    8
nprobe=32,quantizer_efSearch=8,ht=124     0.6300 0.8452 0.8480      0.04104       56110001    8
nprobe=32,quantizer_efSearch=512,ht=128   0.6483 0.8768 0.8798      0.07936       55786929    4
nprobe=64,quantizer_efSearch=64,ht=126    0.6755 0.9317 0.9352      0.08228      110280458    4
nprobe=128,quantizer_efSearch=512,ht=256  0.6910 0.9675 0.9716      0.10317      216662288    3
nprobe=256,quantizer_efSearch=32,ht=126   0.6912 0.9690 0.9731      0.31399      423838468    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6934 0.9694 0.9737      0.33184      424360460    1
nprobe=512,quantizer_efSearch=64,ht=122   0.6935 0.9712 0.9757      0.58537      816247801    1
nprobe=512,quantizer_efSearch=64,ht=128   0.6995 0.9855 0.9902      0.60693      816247801    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.7021 0.9922 0.9970      1.28388     1602589824    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.7031 0.9947 0.9997      1.89538     5410373056    1
```

</details>
<details><summary>`OPQ32_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ32` </summary>
Index size 437156340

 code_size 32

 log filename: autotune.dbdeep10M.OPQ32_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ32.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2674 0.3057 0.3066      0.00450        4934450    67
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=82        0.0327 0.0379 0.0387      0.00268        2477935    112
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=112       0.1921 0.2177 0.2185      0.00283        2139787    106
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=128      0.2284 0.2629 0.2638      0.00304        1964362    99
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=120      0.2438 0.2793 0.2801      0.00328        2480678    92
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=108      0.2475 0.2846 0.2855      0.00397        3751477    76
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128      0.2482 0.2857 0.2866      0.00484        7081651    62
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=120       0.3104 0.3668 0.3677      0.00417        3751324    72
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=118       0.3256 0.3933 0.3943      0.00641        7234600    47
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=124       0.3944 0.4826 0.4836      0.00678        7331635    45
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=116      0.3971 0.4785 0.4794      0.00807       12451468    38
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=120     0.4109 0.5006 0.5015      0.00952       17538882    32
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=124       0.4563 0.5732 0.5742      0.01155       14488406    26
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=128     0.5377 0.6840 0.6851      0.01356       16845234    23
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=114     0.5567 0.7075 0.7087      0.02120       30876679    15
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=124     0.5990 0.7857 0.7872      0.02233       29547690    14
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=256     0.6425 0.8583 0.8604      0.02231       61725332    14
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=256    0.6858 0.9359 0.9385      0.05421      112876924    6
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=256   0.7004 0.9679 0.9714      0.08696      221805597    4
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=126   0.7062 0.9806 0.9846      0.33080      429478415    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=256   0.7089 0.9880 0.9926      0.58898     1614913856    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=256   0.7090 0.9883 0.9929      1.10210     3175533898    1
nprobe=2048,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=126 0.7102 0.9909 0.9952      2.53982     3093217655    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=256 0.7123 0.9953 0.9999      2.92964     5947887563    1
```

</details>
<details><summary>`OPQ32_128,IVF65536,PQ32` </summary>
Index size 434259195

 code_size 32

 log filename: autotune.dbdeep10M.OPQ32_128_IVF65536_PQ32.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7052 0.9954 0.9999      1.04459     3079792045    1
nprobe=2,ht=256                          0.3530 0.4212 0.4222      0.01469        3543333    21
nprobe=4,ht=116                          0.4252 0.5169 0.5182      0.01641        7084696    19
nprobe=4,ht=124                          0.4457 0.5481 0.5495      0.01683        7084696    18
nprobe=16,ht=256                         0.6021 0.7972 0.7992      0.02060       28092989    15
nprobe=64,ht=256                         0.6783 0.9376 0.9404      0.04631      109947238    7
nprobe=128,ht=256                        0.6924 0.9689 0.9722      0.07953      216193396    4
nprobe=256,ht=256                        0.7013 0.9863 0.9901      0.14490      423475855    3
nprobe=2048,ht=256                       0.7052 0.9954 0.9999      1.03083     3079792045    1
nprobe=4096,ht=256                       0.7053 0.9954 1.0000      1.98971     5929487520    1
```

</details>
<details><summary>`OPQ64_128,IVF16384_HNSW32,PQ64x4fs` </summary>
Index size 421317836

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF16384_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4206 0.8802 0.9914      0.06154     3069219171    5
nprobe=2,quantizer_efSearch=4            0.2348 0.3980 0.4234      0.00195       42942904    154
nprobe=4,quantizer_efSearch=4            0.2748 0.5068 0.5459      0.00225       82106520    134
nprobe=4,quantizer_efSearch=8            0.3082 0.5749 0.6160      0.00251       80539451    120
nprobe=4,quantizer_efSearch=16           0.3182 0.5905 0.6318      0.00311       79291090    97
nprobe=8,quantizer_efSearch=8            0.3473 0.6748 0.7349      0.00328      153522518    92
nprobe=8,quantizer_efSearch=16           0.3609 0.6990 0.7602      0.00385      151543357    78
nprobe=16,quantizer_efSearch=8           0.3783 0.7582 0.8380      0.00481      284957954    63
nprobe=16,quantizer_efSearch=16          0.3879 0.7752 0.8564      0.00449      283437799    67
nprobe=16,quantizer_efSearch=32          0.3929 0.7811 0.8629      0.00633      281400067    48
nprobe=32,quantizer_efSearch=16          0.4035 0.8227 0.9166      0.00684      521512345    44
nprobe=32,quantizer_efSearch=32          0.4089 0.8313 0.9263      0.00828      517995989    37
nprobe=32,quantizer_efSearch=64          0.4104 0.8339 0.9293      0.00925      515817558    33
nprobe=64,quantizer_efSearch=32          0.4146 0.8585 0.9608      0.01089      947309606    28
nprobe=64,quantizer_efSearch=128         0.4165 0.8626 0.9662      0.01473      939933258    21
nprobe=128,quantizer_efSearch=32         0.4169 0.8688 0.9756      0.01905     1711837454    16
nprobe=128,quantizer_efSearch=64         0.4189 0.8737 0.9827      0.01799     1710524738    17
nprobe=128,quantizer_efSearch=128        0.4200 0.8753 0.9843      0.01965     1704215603    16
nprobe=256,quantizer_efSearch=128        0.4203 0.8801 0.9913      0.03112     3078549343    10
nprobe=256,quantizer_efSearch=256        0.4206 0.8803 0.9915      0.03439     3072146172    9
nprobe=512,quantizer_efSearch=256        0.4208 0.8816 0.9939      0.05989     5576735388    6
nprobe=1024,quantizer_efSearch=256       0.4209 0.8823 0.9947      0.09542    10077915290    4
```

</details>
<details><summary>`OPQ64_128,IVF16384_HNSW32,PQ64x4fsr` </summary>
Index size 421326028

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF16384_HNSW32_PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5852 0.8644 0.8715      0.01811      120417907    17
nprobe=1,quantizer_efSearch=8            0.2765 0.3494 0.3509      0.00179        7899387    168
nprobe=1,quantizer_efSearch=16           0.2789 0.3522 0.3537      0.00221        7886912    136
nprobe=2,quantizer_efSearch=4            0.3495 0.4631 0.4653      0.00225       15695285    134
nprobe=2,quantizer_efSearch=8            0.3709 0.4909 0.4931      0.00244       15661281    123
nprobe=2,quantizer_efSearch=16           0.3760 0.4968 0.4990      0.00280       15638269    108
nprobe=4,quantizer_efSearch=4            0.4298 0.5917 0.5954      0.00323       31111720    94
nprobe=4,quantizer_efSearch=8            0.4623 0.6369 0.6408      0.00339       31116808    89
nprobe=4,quantizer_efSearch=16           0.4695 0.6454 0.6492      0.00384       31048941    79
nprobe=4,quantizer_efSearch=32           0.4700 0.6467 0.6505      0.00455       31025634    66
nprobe=8,quantizer_efSearch=8            0.5287 0.7567 0.7617      0.00530       61459444    57
nprobe=8,quantizer_efSearch=16           0.5416 0.7722 0.7772      0.00562       61339741    54
nprobe=8,quantizer_efSearch=32           0.5420 0.7746 0.7796      0.00636       61281965    48
nprobe=8,quantizer_efSearch=128          0.5422 0.7750 0.7800      0.01012       61258902    31
nprobe=16,quantizer_efSearch=4           0.5577 0.8218 0.8288      0.01715      121089542    18
nprobe=16,quantizer_efSearch=8           0.5748 0.8489 0.8559      0.01691      120868260    18
nprobe=16,quantizer_efSearch=16          0.5837 0.8611 0.8681      0.01710      120603152    18
nprobe=16,quantizer_efSearch=32          0.5852 0.8644 0.8715      0.01728      120417907    18
nprobe=16,quantizer_efSearch=64          0.5855 0.8646 0.8717      0.01919      120353814    16
nprobe=16,quantizer_efSearch=128         0.5857 0.8649 0.8720      0.02184      120329251    14
nprobe=32,quantizer_efSearch=8           0.6007 0.9012 0.9114      0.03120      236543562    10
nprobe=32,quantizer_efSearch=16          0.6139 0.9202 0.9309      0.03239      235883130    10
nprobe=32,quantizer_efSearch=32          0.6152 0.9252 0.9359      0.03280      235379519    10
nprobe=32,quantizer_efSearch=128         0.6157 0.9258 0.9366      0.03686      235081399    9
nprobe=32,quantizer_efSearch=512         0.6158 0.9260 0.9368      0.06072      235069980    5
nprobe=64,quantizer_efSearch=64          0.6239 0.9593 0.9717      0.06837      456859360    5
nprobe=64,quantizer_efSearch=128         0.6247 0.9601 0.9725      0.06851      456549548    5
nprobe=128,quantizer_efSearch=256        0.6314 0.9776 0.9905      0.12595      880899530    3
nprobe=128,quantizer_efSearch=512        0.6315 0.9778 0.9908      0.14746      880846563    3
nprobe=256,quantizer_efSearch=128        0.6319 0.9821 0.9967      0.23111     1691529425    2
nprobe=256,quantizer_efSearch=512        0.6329 0.9826 0.9972      0.26514     1688318478    2
nprobe=512,quantizer_efSearch=128        0.6330 0.9841 0.9986      0.49803     3225924046    1
nprobe=1024,quantizer_efSearch=512       0.6332 0.9848 0.9998      0.93926     6126410390    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fs` </summary>
Index size 735868108

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF262144_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4049 0.8200 0.9082      0.02391      171039629    13
nprobe=2,quantizer_efSearch=4            0.1922 0.2831 0.2880      0.00178         903375    169
nprobe=4,quantizer_efSearch=4            0.2377 0.3753 0.3861      0.00191        1806271    158
nprobe=4,quantizer_efSearch=8            0.2670 0.4213 0.4327      0.00284        1804464    106
nprobe=8,quantizer_efSearch=4            0.3072 0.5259 0.5449      0.00269        3603732    112
nprobe=8,quantizer_efSearch=8            0.3130 0.5377 0.5576      0.00306        3600191    99
nprobe=16,quantizer_efSearch=4           0.3371 0.6157 0.6472      0.00350        7202404    86
nprobe=16,quantizer_efSearch=8           0.3527 0.6495 0.6820      0.00400        7185666    76
nprobe=16,quantizer_efSearch=16          0.3596 0.6600 0.6937      0.00523        7168067    58
nprobe=32,quantizer_efSearch=8           0.3719 0.7162 0.7662      0.00578       14323115    52
nprobe=64,quantizer_efSearch=8           0.3821 0.7541 0.8186      0.00735       28475050    41
nprobe=32,quantizer_efSearch=16          0.3828 0.7368 0.7901      0.00688       14286509    44
nprobe=64,quantizer_efSearch=16          0.3983 0.7884 0.8589      0.00839       28433755    36
nprobe=128,quantizer_efSearch=16         0.4022 0.8116 0.8942      0.00973       56278788    31
nprobe=64,quantizer_efSearch=32          0.4048 0.8029 0.8741      0.01204       28340388    25
nprobe=128,quantizer_efSearch=32         0.4121 0.8341 0.9225      0.01408       56223781    22
nprobe=256,quantizer_efSearch=32         0.4141 0.8464 0.9430      0.01906      110498512    16
nprobe=128,quantizer_efSearch=64         0.4158 0.8434 0.9332      0.01860       56056307    17
nprobe=256,quantizer_efSearch=64         0.4174 0.8593 0.9602      0.02543      110705509    12
nprobe=512,quantizer_efSearch=64         0.4202 0.8673 0.9719      0.03212      215417274    10
nprobe=1024,quantizer_efSearch=64        0.4204 0.8673 0.9739      0.04700      383668532    7
nprobe=512,quantizer_efSearch=128        0.4226 0.8743 0.9813      0.03639      217012944    9
nprobe=1024,quantizer_efSearch=128       0.4233 0.8770 0.9864      0.05025      416746948    6
nprobe=512,quantizer_efSearch=256        0.4235 0.8766 0.9836      0.05484      216672508    6
nprobe=1024,quantizer_efSearch=256       0.4260 0.8808 0.9912      0.07106      422460809    5
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fsr` </summary>
Index size 735298764

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF262144_HNSW32_PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5208 0.6972 0.7000      0.01670        7126056    18
nprobe=2,quantizer_efSearch=4            0.2399 0.2853 0.2860      0.00234         900635    129
nprobe=2,quantizer_efSearch=8            0.2642 0.3153 0.3160      0.00331         897714    91
nprobe=4,quantizer_efSearch=4            0.3101 0.3836 0.3846      0.00310        1800597    97
nprobe=4,quantizer_efSearch=8            0.3502 0.4325 0.4336      0.00393        1799449    77
nprobe=8,quantizer_efSearch=4            0.4229 0.5454 0.5472      0.00498        3592784    61
nprobe=8,quantizer_efSearch=8            0.4331 0.5577 0.5595      0.00535        3589948    57
nprobe=8,quantizer_efSearch=16           0.4459 0.5721 0.5739      0.00686        3577951    44
nprobe=8,quantizer_efSearch=32           0.4485 0.5748 0.5766      0.00931        3570311    33
nprobe=8,quantizer_efSearch=64           0.4501 0.5764 0.5782      0.01400        3567336    22
nprobe=16,quantizer_efSearch=4           0.4827 0.6468 0.6494      0.01546        7177314    20
nprobe=16,quantizer_efSearch=16          0.5179 0.6922 0.6950      0.01519        7144784    20
nprobe=16,quantizer_efSearch=32          0.5208 0.6972 0.7000      0.01785        7126056    17
nprobe=16,quantizer_efSearch=64          0.5225 0.6996 0.7024      0.01985        7117653    16
nprobe=16,quantizer_efSearch=128         0.5228 0.6997 0.7025      0.02667        7115088    12
nprobe=32,quantizer_efSearch=8           0.5571 0.7660 0.7697      0.02698       14274326    12
nprobe=32,quantizer_efSearch=16          0.5690 0.7879 0.7917      0.03034       14237989    10
nprobe=32,quantizer_efSearch=32          0.5718 0.7945 0.7983      0.03387       14202324    9
nprobe=32,quantizer_efSearch=64          0.5741 0.7980 0.8018      0.03410       14178323    9
nprobe=32,quantizer_efSearch=256         0.5743 0.7982 0.8020      0.05065       14166707    6
nprobe=64,quantizer_efSearch=16          0.6020 0.8566 0.8612      0.06459       28331344    5
nprobe=64,quantizer_efSearch=32          0.6109 0.8721 0.8767      0.06421       28243192    5
nprobe=64,quantizer_efSearch=64          0.6152 0.8785 0.8832      0.06574       28174229    5
nprobe=64,quantizer_efSearch=128         0.6163 0.8799 0.8846      0.07558       28142521    4
nprobe=64,quantizer_efSearch=256         0.6170 0.8804 0.8850      0.09470       28133736    4
nprobe=128,quantizer_efSearch=64         0.6350 0.9310 0.9366      0.12927       55868188    3
nprobe=128,quantizer_efSearch=128        0.6366 0.9325 0.9381      0.13306       55760841    3
nprobe=128,quantizer_efSearch=256        0.6371 0.9330 0.9386      0.14504       55723100    3
nprobe=128,quantizer_efSearch=512        0.6373 0.9332 0.9388      0.17178       55714121    2
nprobe=256,quantizer_efSearch=64         0.6468 0.9568 0.9642      0.23332      110330680    2
nprobe=256,quantizer_efSearch=128        0.6485 0.9621 0.9697      0.23530      110082017    2
nprobe=256,quantizer_efSearch=256        0.6490 0.9636 0.9717      0.25905      109905598    2
nprobe=512,quantizer_efSearch=128        0.6570 0.9770 0.9860      0.45291      216338252    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 675605008

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4229 0.8730 0.9807      0.04112      439556205    8
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1742 0.2528 0.2561      0.00132        1275206    228
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1924 0.2851 0.2899      0.00155        1268542    194
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2346 0.3686 0.3794      0.00163        2176891    184
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2381 0.3759 0.3866      0.00178        2173769    169
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2572 0.4034 0.4145      0.00176        2528401    171
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2608 0.4114 0.4226      0.00193        2523522    156
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2613 0.4123 0.4240      0.00239        2523387    126
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2876 0.4827 0.4988      0.00212        4354418    142
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3063 0.5223 0.5420      0.00251        4320787    120
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3154 0.5419 0.5615      0.00250        5015813    120
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3225 0.5770 0.6051      0.00273        7983834    110
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3338 0.6053 0.6373      0.00303        7923970    100
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3378 0.6097 0.6390      0.00294        8655421    103
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3510 0.6427 0.6762      0.00323        8599237    93
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3615 0.6632 0.6971      0.00442        9913979    68
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3664 0.6934 0.7394      0.00419       15873488    72
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3775 0.7163 0.7677      0.00487       15717635    62
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3794 0.7203 0.7722      0.00538       15691107    56
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3849 0.7499 0.8142      0.00527       30217310    57
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3888 0.7421 0.7952      0.00656       19640979    46
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3949 0.7770 0.8427      0.00584       31453330    52
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4010 0.7945 0.8657      0.00734       31108004    41
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4022 0.7956 0.8678      0.00900       31048629    34
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4048 0.8039 0.8754      0.00899       33682412    34
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4070 0.8066 0.8790      0.01217       33625131    25
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4075 0.8244 0.9136      0.01088       58927713    28
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4084 0.8085 0.8812      0.01244       38851561    25
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4130 0.8310 0.9151      0.01403       67324010    22
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4160 0.8417 0.9314      0.01563       66557065    20
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4171 0.8442 0.9347      0.01829       66489884    17
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4182 0.8526 0.9511      0.01760      117621074    18
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.4198 0.8568 0.9549      0.02255      132910419    14
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4204 0.8634 0.9666      0.02575      131157436    12
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4219 0.8668 0.9720      0.02266      226543706    14
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4240 0.8716 0.9774      0.02964      231082189    11
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.4249 0.8797 0.9886      0.04639      443015007    7
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.4254 0.8807 0.9902      0.04834      452500459    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4266 0.8810 0.9921      0.06283      444043416    5
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 675601936

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2725 0.3197 0.3202      0.00227        2318162    132
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1612 0.1857 0.1863      0.00131         820581    230
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1848 0.2127 0.2133      0.00145        1170684    208
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1853 0.2134 0.2140      0.00149        1170469    202
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.2001 0.2358 0.2364      0.00173        1088776    175
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2428 0.2859 0.2864      0.00180        1270333    167
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2523 0.2962 0.2968      0.00194        1626259    155
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2633 0.3106 0.3111      0.00195        1620367    155
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2634 0.3109 0.3114      0.00233        1620673    129
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.3034 0.3700 0.3707      0.00267        2183648    113
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3418 0.4186 0.4197      0.00294        2524993    102
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3454 0.4232 0.4244      0.00302        2521641    100
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3592 0.4393 0.4406      0.00335        3217366    90
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4274 0.5394 0.5409      0.00472        5029124    64
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4446 0.5638 0.5657      0.00470        5008454    64
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4510 0.5719 0.5738      0.00513        6347127    59
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4521 0.5741 0.5762      0.00611        6346934    50
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4539 0.5755 0.5775      0.00620        8981925    49
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.4545 0.5750 0.5769      0.00778       14257380    39
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4756 0.6237 0.6265      0.01540        7957765    20
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5110 0.6731 0.6760      0.01531        9954490    20
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5205 0.6959 0.6992      0.01625        9906840    19
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5228 0.6991 0.7024      0.01899       12545548    16
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5230 0.6997 0.7030      0.01937       17751587    16
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.5320 0.7179 0.7219      0.02521       15053402    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5643 0.7693 0.7733      0.02813       15687322    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5647 0.7698 0.7738      0.03067       15684091    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5761 0.7910 0.7951      0.03038       16998383    10
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5792 0.7965 0.8005      0.03285       19613144    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5804 0.7969 0.8010      0.02940       24847634    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5955 0.8353 0.8403      0.06457       29809700    5
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.6166 0.8659 0.8705      0.06305       49298590    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6187 0.8772 0.8824      0.06021       33549696    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.6207 0.8792 0.8844      0.06847       48945446    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6216 0.8794 0.8844      0.06187       38659166    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.6362 0.9205 0.9260      0.11563       61771486    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.6407 0.9317 0.9380      0.13478       96783534    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6539 0.9633 0.9707      0.24743      120556121    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.6545 0.9650 0.9723      0.24715      151400815    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6553 0.9773 0.9860      0.45155      239109479    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6559 0.9800 0.9889      0.48032      237019191    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6560 0.9801 0.9889      0.45512      237012365    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.6566 0.9802 0.9891      0.53614      257358324    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6575 0.9860 0.9953      0.87470      447219260    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.6577 0.9871 0.9967      0.93402      463880952    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fs` </summary>
Index size 485069516

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_HNSW32_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3037 0.5196 0.5450      0.00194        7118792    155
nprobe=1,quantizer_efSearch=4            0.1732 0.2643 0.2708      0.00114        1784377    263
nprobe=2,quantizer_efSearch=4            0.2293 0.3701 0.3844      0.00135        3557683    223
nprobe=4,quantizer_efSearch=4            0.2803 0.4763 0.5007      0.00162        7110006    185
nprobe=4,quantizer_efSearch=8            0.3037 0.5196 0.5450      0.00193        7118792    156
nprobe=8,quantizer_efSearch=4            0.3380 0.6142 0.6562      0.00226       14218646    133
nprobe=8,quantizer_efSearch=8            0.3460 0.6266 0.6691      0.00274       14199341    110
nprobe=8,quantizer_efSearch=16           0.3520 0.6394 0.6829      0.00298       14161080    101
nprobe=16,quantizer_efSearch=4           0.3655 0.6911 0.7507      0.00278       28318631    109
nprobe=16,quantizer_efSearch=8           0.3801 0.7209 0.7828      0.00304       28264163    99
nprobe=16,quantizer_efSearch=16          0.3858 0.7290 0.7921      0.00380       28211457    79
nprobe=32,quantizer_efSearch=8           0.3959 0.7758 0.8533      0.00371       56108155    81
nprobe=64,quantizer_efSearch=8           0.4028 0.8009 0.8884      0.00474      110939872    64
nprobe=64,quantizer_efSearch=16          0.4117 0.8271 0.9198      0.00590      110791346    51
nprobe=128,quantizer_efSearch=16         0.4163 0.8428 0.9428      0.00801      218048022    38
nprobe=128,quantizer_efSearch=32         0.4218 0.8589 0.9621      0.00887      217723225    35
nprobe=128,quantizer_efSearch=64         0.4238 0.8636 0.9679      0.01293      217105298    24
nprobe=128,quantizer_efSearch=128        0.4240 0.8644 0.9686      0.01581      216761137    19
nprobe=256,quantizer_efSearch=64         0.4262 0.8747 0.9843      0.01827      424963404    17
nprobe=256,quantizer_efSearch=128        0.4264 0.8763 0.9859      0.01966      424786736    16
nprobe=512,quantizer_efSearch=64         0.4269 0.8771 0.9889      0.02199      816362900    14
nprobe=1024,quantizer_efSearch=128       0.4272 0.8813 0.9939      0.04121     1552970376    8
nprobe=1024,quantizer_efSearch=256       0.4273 0.8818 0.9950      0.05775     1601856002    6
nprobe=2048,quantizer_efSearch=512       0.4276 0.8828 0.9964      0.13382     3078345175    3
nprobe=4096,quantizer_efSearch=512       0.4277 0.8825 0.9964      0.18167     5410698916    2
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fsr` </summary>
Index size 485078732

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_HNSW32_PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5590 0.7940 0.7981      0.01692       28125843    18
nprobe=2,quantizer_efSearch=8            0.3319 0.4134 0.4150      0.00197        3552874    153
nprobe=2,quantizer_efSearch=16           0.3361 0.4194 0.4210      0.00249        3546622    121
nprobe=4,quantizer_efSearch=4            0.3851 0.4987 0.5005      0.00253        7104831    119
nprobe=4,quantizer_efSearch=16           0.4246 0.5513 0.5533      0.00341        7091396    88
nprobe=8,quantizer_efSearch=4            0.4794 0.6520 0.6553      0.00420       14191701    72
nprobe=8,quantizer_efSearch=8            0.4900 0.6656 0.6689      0.00439       14171929    69
nprobe=8,quantizer_efSearch=16           0.5006 0.6813 0.6847      0.00493       14139600    61
nprobe=8,quantizer_efSearch=32           0.5016 0.6834 0.6868      0.00594       14125172    51
nprobe=8,quantizer_efSearch=64           0.5021 0.6836 0.6870      0.00754       14121187    40
nprobe=16,quantizer_efSearch=8           0.5511 0.7804 0.7844      0.01434       28213482    21
nprobe=16,quantizer_efSearch=16          0.5581 0.7906 0.7948      0.01526       28162960    20
nprobe=16,quantizer_efSearch=32          0.5590 0.7940 0.7981      0.01584       28125843    19
nprobe=16,quantizer_efSearch=64          0.5596 0.7949 0.7990      0.01705       28107490    18
nprobe=16,quantizer_efSearch=128         0.5597 0.7949 0.7990      0.02018       28102881    15
nprobe=32,quantizer_efSearch=16          0.5979 0.8694 0.8752      0.02683       55871699    12
nprobe=32,quantizer_efSearch=64          0.6031 0.8762 0.8820      0.02896       55711305    11
nprobe=32,quantizer_efSearch=128         0.6034 0.8765 0.8823      0.03265       55695294    10
nprobe=32,quantizer_efSearch=512         0.6036 0.8766 0.8824      0.06120       55691624    5
nprobe=64,quantizer_efSearch=16          0.6194 0.9171 0.9233      0.06554      110580133    5
nprobe=64,quantizer_efSearch=64          0.6266 0.9332 0.9399      0.07140      110077597    5
nprobe=128,quantizer_efSearch=64         0.6355 0.9627 0.9712      0.12056      216703362    3
nprobe=256,quantizer_efSearch=32         0.6372 0.9670 0.9769      0.23272      423026517    2
nprobe=256,quantizer_efSearch=64         0.6409 0.9770 0.9874      0.23435      424092131    2
nprobe=256,quantizer_efSearch=256        0.6432 0.9799 0.9900      0.24982      423752862    2
nprobe=512,quantizer_efSearch=64         0.6476 0.9820 0.9921      0.46144      814810338    1
```

</details>
<details><summary>`OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 470131216

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ64x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4219 0.8573 0.9586      0.01186      223541202    26
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1782 0.2711 0.2778      0.00137        2139740    220
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2243 0.3593 0.3714      0.00144        4271685    209
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2340 0.3767 0.3909      0.00149        3929441    202
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2475 0.3994 0.4141      0.00153        4261429    197
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2542 0.4253 0.4450      0.00170        7392848    177
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2831 0.4780 0.4994      0.00182        7867007    166
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2901 0.4946 0.5191      0.00179        7505368    168
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2996 0.5130 0.5374      0.00183        7831311    164
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3039 0.5215 0.5470      0.00202        7816676    149
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3352 0.6006 0.6368      0.00214       15024536    140
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3466 0.6264 0.6684      0.00237       14916201    127
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3470 0.6293 0.6728      0.00233       14888864    129
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3680 0.6899 0.7454      0.00274       29260802    110
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3793 0.7149 0.7778      0.00300       28956957    100
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3858 0.7287 0.7927      0.00385       29555622    78
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3923 0.7691 0.8487      0.00373       56863497    81
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3928 0.7699 0.8494      0.00405       56787817    75
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4027 0.7910 0.8730      0.00513       57256015    59
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4135 0.8326 0.9252      0.00683      112009099    44
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4162 0.8479 0.9472      0.00883      222971565    34
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4167 0.8404 0.9343      0.00771      113051448    39
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4175 0.8416 0.9356      0.00881      112893790    35
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4176 0.8427 0.9372      0.00963      115396968    32
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4219 0.8573 0.9586      0.01048      223544871    29
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.4237 0.8634 0.9668      0.01088      219744059    28
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4245 0.8747 0.9822      0.01603      438173163    19
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4251 0.8757 0.9854      0.01717      429775821    18
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4257 0.8764 0.9864      0.01943      434619227    16
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4259 0.8769 0.9867      0.02256      434287986    14
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.4267 0.8802 0.9920      0.02484      854601739    13
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4268 0.8812 0.9935      0.02876      836890098    11
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4272 0.8819 0.9955      0.04310     1610884459    7
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.4276 0.8827 0.9965      0.07157     3093913673    5
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.4277 0.8827 0.9966      0.18183     5947851638    2
```

</details>
<details><summary>`OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 470146576

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ64x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5600 0.7941 0.7990      0.02629       38460080    12
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=2     0.2201 0.2630 0.2639      0.00173        1964058    174
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4     0.2320 0.2771 0.2780      0.00158        2138897    190
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3356 0.4154 0.4167      0.00210        4259545    143
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3385 0.4191 0.4204      0.00232        4933281    130
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.3615 0.4658 0.4678      0.00283        7369801    106
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3745 0.4838 0.4859      0.00281        7336117    107
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4036 0.5236 0.5257      0.00295        7488943    102
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4179 0.5421 0.5442      0.00290        7817922    104
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.4205 0.5461 0.5482      0.00300        7812293    100
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4256 0.5527 0.5548      0.00321        8477963    94
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4258 0.5532 0.5553      0.00426        9804486    71
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4630 0.6257 0.6291      0.00437       14676417    69
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4730 0.6397 0.6433      0.00445       14586659    68
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4733 0.6397 0.6433      0.00427       14597467    71
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5017 0.6794 0.6830      0.00458       15537148    66
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.5019 0.6806 0.6843      0.00485       15526501    62
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5029 0.6824 0.6861      0.00543       16838646    56
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.5032 0.6828 0.6865      0.00650       19438198    47
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=128    0.5033 0.6828 0.6865      0.00869       24523259    35
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.5201 0.7281 0.7326      0.01591       28712001    19
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5486 0.7733 0.7781      0.01503       29736413    20
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5492 0.7749 0.7796      0.01455       28951592    21
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5500 0.7765 0.7812      0.01630       31017924    19
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5588 0.7899 0.7947      0.01772       29547957    17
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.5600 0.7940 0.7988      0.01850       38440963    17
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6007 0.8750 0.8812      0.02786       58479913    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6009 0.8750 0.8812      0.02822       58464261    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6014 0.8763 0.8825      0.02900       61051395    11
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.6017 0.8761 0.8823      0.03117       66121154    10
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6270 0.9330 0.9398      0.07041      115361416    5
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.6276 0.9283 0.9354      0.07080      121312909    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6355 0.9639 0.9721      0.13667      226812996    3
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.6394 0.9741 0.9836      0.24434      431728497    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6430 0.9790 0.9887      0.24463      438207943    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6431 0.9807 0.9900      0.24733      434143719    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6447 0.9868 0.9969      0.48317      836506691    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6453 0.9875 0.9988      0.91976     1610358566    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6467 0.9876 0.9987      0.90292     1624818770    1
```

</details>
<details><summary>`OPQ64_128,IVF65536,PQ64x4fs` </summary>
Index size 467235095

 code_size 32

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_PQ64x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4262 0.8772 0.9869      0.02321      423475872    13
nprobe=2                                 0.2524 0.4068 0.4219      0.01166        3543331    26
nprobe=4                                 0.3083 0.5295 0.5553      0.01205        7084697    25
nprobe=8                                 0.3538 0.6425 0.6863      0.01255       14115501    24
nprobe=16                                0.3882 0.7339 0.7978      0.01343       28092982    23
nprobe=32                                0.4065 0.7983 0.8807      0.01454       55672546    21
nprobe=64                                0.4178 0.8432 0.9377      0.01658      109947174    19
nprobe=128                               0.4243 0.8648 0.9692      0.01869      216193401    17
nprobe=256                               0.4262 0.8772 0.9869      0.02513      423475872    12
nprobe=512                               0.4268 0.8812 0.9938      0.03298      825310318    10
nprobe=1024                              0.4271 0.8822 0.9955      0.04550     1598493307    7
nprobe=2048                              0.4276 0.8830 0.9966      0.07317     3079791438    5
```

</details>
<details><summary>`PCAR16,IVF16384_HNSW32,SQfp16` </summary>
Index size 405684245

 code_size 32

 log filename: autotune.dbdeep10M.PCAR16_IVF16384_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0508 0.1666 0.3369      0.00288       34145222    105
nprobe=2,quantizer_efSearch=4            0.0437 0.1363 0.2486      0.00211       17460300    142
nprobe=2,quantizer_efSearch=8            0.0455 0.1418 0.2594      0.00216       17311895    139
nprobe=2,quantizer_efSearch=16           0.0460 0.1431 0.2621      0.00245       17263830    123
nprobe=4,quantizer_efSearch=4            0.0475 0.1566 0.3172      0.00247       34362253    122
nprobe=4,quantizer_efSearch=8            0.0508 0.1666 0.3369      0.00264       34145222    114
nprobe=4,quantizer_efSearch=16           0.0511 0.1680 0.3401      0.00279       34039820    108
nprobe=8,quantizer_efSearch=16           0.0558 0.1857 0.4013      0.00445       67076363    68
nprobe=16,quantizer_efSearch=4           0.0559 0.1906 0.4300      0.00587      132039539    52
nprobe=16,quantizer_efSearch=8           0.0570 0.1946 0.4426      0.00583      131770298    52
nprobe=16,quantizer_efSearch=16          0.0575 0.1964 0.4471      0.00662      131505020    46
nprobe=32,quantizer_efSearch=64          0.0584 0.1999 0.4711      0.01346      256650053    23
nprobe=64,quantizer_efSearch=16          0.0585 0.2008 0.4787      0.02034      501594036    15
nprobe=64,quantizer_efSearch=32          0.0587 0.2013 0.4823      0.02082      500607154    15
```

</details>
<details><summary>`PCAR16,IVF262144_HNSW32,SQfp16` </summary>
Index size 490261525

 code_size 32

 log filename: autotune.dbdeep10M.PCAR16_IVF262144_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0481 0.1511 0.2306      0.00207        1766723    146
nprobe=2,quantizer_efSearch=4            0.0409 0.1148 0.1528      0.00176         891569    171
nprobe=4,quantizer_efSearch=4            0.0457 0.1410 0.2132      0.00188        1756774    160
nprobe=4,quantizer_efSearch=8            0.0481 0.1511 0.2306      0.00207        1766723    145
nprobe=8,quantizer_efSearch=4            0.0516 0.1688 0.2973      0.00208        3487191    145
nprobe=8,quantizer_efSearch=8            0.0522 0.1709 0.3001      0.00218        3490575    138
nprobe=16,quantizer_efSearch=4           0.0541 0.1797 0.3550      0.00229        6870634    131
nprobe=16,quantizer_efSearch=8           0.0556 0.1855 0.3670      0.00237        6884661    127
nprobe=16,quantizer_efSearch=16          0.0564 0.1868 0.3711      0.00268        6892085    113
nprobe=32,quantizer_efSearch=8           0.0568 0.1923 0.4127      0.00320       13575559    94
nprobe=32,quantizer_efSearch=16          0.0579 0.1944 0.4192      0.00335       13593070    90
nprobe=32,quantizer_efSearch=32          0.0586 0.1960 0.4233      0.00431       13597904    70
nprobe=64,quantizer_efSearch=16          0.0587 0.1994 0.4514      0.00485       26804601    62
nprobe=64,quantizer_efSearch=32          0.0594 0.2006 0.4553      0.00575       26819541    53
nprobe=128,quantizer_efSearch=32         0.0598 0.2021 0.4707      0.00845       52844659    36
```

</details>
<details><summary>`PCAR16,IVF262144(IVF512,PQ8x4fs,RFlat),SQfp16` </summary>
Index size 422137561

 code_size 32

 log filename: autotune.dbdeep10M.PCAR16_IVF262144_IVF512_PQ8x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.0543 0.1825 0.3816      0.00507       33788437    60
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8  0.0360 0.0894 0.1042      0.00220        1872858    137
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2  0.0402 0.1117 0.1464      0.00226        1282665    133
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4  0.0419 0.1177 0.1534      0.00225        1634685    134
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2   0.0430 0.1312 0.1996      0.00230        2273271    131
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0460 0.1384 0.2094      0.00239        2633763    126
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0508 0.1594 0.2979      0.00265        8258041    114
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4  0.0516 0.1628 0.2861      0.00254        4281942    119
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8  0.0563 0.1790 0.3546      0.00289        8506376    104
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16 0.0575 0.1816 0.3593      0.00314        9851046    96
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32 0.0585 0.1944 0.4154      0.00497       19215859    61
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8  0.0593 0.1962 0.4356      0.00528       28841162    57
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16 0.0595 0.1998 0.4451      0.00579       30188593    52
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0596 0.2001 0.4473      0.00652       32809757    47
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.0598 0.2000 0.4470      0.00734       38088328    41
```

</details>
<details><summary>`PCAR16,IVF65536_HNSW32,SQfp16` </summary>
Index size 422600213

 code_size 32

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0525 0.1659 0.3106      0.00210        7022204    143
nprobe=1,quantizer_efSearch=4            0.0385 0.1040 0.1547      0.00178        1788515    169
nprobe=2,quantizer_efSearch=4            0.0464 0.1332 0.2243      0.00184        3533888    163
nprobe=4,quantizer_efSearch=4            0.0496 0.1580 0.2950      0.00193        6990496    156
nprobe=8,quantizer_efSearch=4            0.0544 0.1800 0.3703      0.00209       13854173    144
nprobe=8,quantizer_efSearch=8            0.0549 0.1819 0.3751      0.00220       13858154    137
nprobe=8,quantizer_efSearch=16           0.0568 0.1849 0.3814      0.00247       13869640    122
nprobe=16,quantizer_efSearch=8           0.0577 0.1926 0.4240      0.00279       27344727    108
nprobe=16,quantizer_efSearch=16          0.0588 0.1949 0.4282      0.00289       27357468    104
nprobe=16,quantizer_efSearch=32          0.0590 0.1955 0.4307      0.00349       27360349    86
nprobe=32,quantizer_efSearch=16          0.0596 0.1987 0.4550      0.00462       53864018    65
nprobe=64,quantizer_efSearch=32          0.0601 0.2002 0.4698      0.00733      105866546    41
nprobe=64,quantizer_efSearch=64          0.0602 0.2005 0.4709      0.00857      105844417    35
```

</details>
<details><summary>`PCAR16,IVF65536(IVF256,PQ8x4fs,RFlat),SQfp16` </summary>
Index size 405587161

 code_size 32

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_IVF256_PQ8x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.0568 0.1856 0.4158      0.00985      109374838    31
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0390 0.1040 0.1527      0.00228        3162804    132
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4    0.0400 0.1058 0.1566      0.00230        2145802    131
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=16   0.0407 0.1089 0.1606      0.00231        3156263    131
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.0484 0.1341 0.2239      0.00239        3949529    126
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0495 0.1388 0.2308      0.00242        4944658    125
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.0531 0.1625 0.2990      0.00251        7427256    120
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.0536 0.1762 0.3534      0.00258       14814384    117
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.0542 0.1786 0.3553      0.00291       16792662    103
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.0548 0.1767 0.3594      0.00285       14366166    106
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.0562 0.1852 0.3765      0.00324       15255731    93
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0571 0.1862 0.4077      0.00362       27875340    83
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.0580 0.1923 0.4247      0.00391       28878866    77
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.0583 0.1947 0.4269      0.00445       30124297    68
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.0585 0.1955 0.4461      0.00443       55059057    68
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.0588 0.1983 0.4500      0.00544       57020347    56
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.0593 0.1977 0.4529      0.00607       55451593    50
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.0594 0.1985 0.4555      0.00641       59410051    47
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.0596 0.1991 0.4555      0.00982       59309216    31
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.0600 0.2005 0.4711      0.01261      111932218    24
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.0601 0.2004 0.4716      0.02708      116526224    12
```

</details>
<details><summary>`PCAR16,IVF65536,SQfp16` </summary>
Index size 404762720

 code_size 32

 log filename: autotune.dbdeep10M.PCAR16_IVF65536_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0588 0.1959 0.4314      0.00682       27344729    45
nprobe=8                                 0.0566 0.1850 0.3830      0.00618       13860448    49
nprobe=16                                0.0588 0.1959 0.4314      0.00689       27344729    44
nprobe=32                                0.0597 0.1994 0.4581      0.00859       53812503    35
nprobe=64                                0.0601 0.2003 0.4717      0.01219      105757340    25
```

</details>
<details><summary>`PCAR32,IVF16384_HNSW32,SQ8` </summary>
Index size 406739285

 code_size 32

 log filename: autotune.dbdeep10M.PCAR32_IVF16384_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1730 0.4212 0.5227      0.00448       45213567    67
nprobe=2,quantizer_efSearch=8            0.1468 0.3276 0.3864      0.00293       23105401    103
nprobe=2,quantizer_efSearch=16           0.1479 0.3301 0.3896      0.00310       22919076    97
nprobe=2,quantizer_efSearch=32           0.1480 0.3302 0.3898      0.00379       22871091    80
nprobe=4,quantizer_efSearch=4            0.1587 0.3882 0.4815      0.00460       45873047    66
nprobe=4,quantizer_efSearch=8            0.1730 0.4212 0.5227      0.00437       45213567    69
nprobe=4,quantizer_efSearch=16           0.1755 0.4293 0.5313      0.00468       44846920    65
nprobe=4,quantizer_efSearch=32           0.1760 0.4291 0.5318      0.00502       44730195    60
nprobe=4,quantizer_efSearch=64           0.1763 0.4290 0.5318      0.00601       44687229    51
nprobe=8,quantizer_efSearch=8            0.1893 0.4889 0.6427      0.00759       87684840    40
nprobe=8,quantizer_efSearch=16           0.1945 0.5026 0.6583      0.00713       86943597    43
nprobe=8,quantizer_efSearch=32           0.1948 0.5044 0.6607      0.00767       86636129    40
nprobe=8,quantizer_efSearch=64           0.1956 0.5052 0.6617      0.00839       86537336    36
nprobe=16,quantizer_efSearch=4           0.1964 0.5275 0.7195      0.02005      169146436    15
nprobe=16,quantizer_efSearch=8           0.2016 0.5459 0.7520      0.01245      168412076    25
nprobe=16,quantizer_efSearch=16          0.2038 0.5536 0.7633      0.01194      167520488    26
nprobe=16,quantizer_efSearch=32          0.2041 0.5546 0.7663      0.01266      166779873    24
nprobe=16,quantizer_efSearch=64          0.2048 0.5553 0.7674      0.01336      166510392    23
nprobe=32,quantizer_efSearch=32          0.2124 0.5913 0.8451      0.03530      319577070    9
nprobe=32,quantizer_efSearch=128         0.2134 0.5921 0.8462      0.02465      318560439    13
nprobe=64,quantizer_efSearch=32          0.2135 0.6044 0.8849      0.06725      609138112    5
nprobe=64,quantizer_efSearch=128         0.2138 0.6058 0.8867      0.05658      606761125    6
nprobe=64,quantizer_efSearch=64          0.2139 0.6056 0.8863      0.03959      607359616    8
nprobe=128,quantizer_efSearch=128        0.2147 0.6096 0.9050      0.07565     1151974306    4
nprobe=256,quantizer_efSearch=32         0.2149 0.6106 0.9069      0.21269     2145284936    2
nprobe=256,quantizer_efSearch=64         0.2151 0.6122 0.9126      0.19875     2174522917    2
nprobe=1024,quantizer_efSearch=64        0.2152 0.6127 0.9144      0.50405     6194934364    1
nprobe=1024,quantizer_efSearch=256       0.2153 0.6129 0.9167      0.46860     7671525713    1
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ8` </summary>
Index size 507045205

 code_size 32

 log filename: autotune.dbdeep10M.PCAR32_IVF262144_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2133 0.6082 0.8855      0.05743      108424265    6
nprobe=1,quantizer_efSearch=4            0.0963 0.1558 0.1584      0.00185         449291    163
nprobe=2,quantizer_efSearch=4            0.1215 0.2207 0.2305      0.00189         893586    159
nprobe=4,quantizer_efSearch=4            0.1438 0.2970 0.3194      0.00195        1783973    154
nprobe=4,quantizer_efSearch=8            0.1562 0.3226 0.3482      0.00219        1786057    138
nprobe=8,quantizer_efSearch=8            0.1758 0.4018 0.4584      0.00238        3558790    127
nprobe=16,quantizer_efSearch=4           0.1832 0.4596 0.5568      0.00244        7098756    123
nprobe=16,quantizer_efSearch=8           0.1898 0.4765 0.5795      0.00263        7089683    115
nprobe=16,quantizer_efSearch=16          0.1923 0.4831 0.5885      0.00290        7074810    104
nprobe=16,quantizer_efSearch=32          0.1935 0.4866 0.5926      0.00396        7061615    76
nprobe=32,quantizer_efSearch=16          0.2001 0.5317 0.6889      0.00448       14062107    67
nprobe=32,quantizer_efSearch=32          0.2016 0.5370 0.6957      0.00506       14033610    60
nprobe=64,quantizer_efSearch=16          0.2051 0.5642 0.7650      0.00661       27939185    46
nprobe=64,quantizer_efSearch=32          0.2076 0.5721 0.7778      0.00812       27870666    37
nprobe=64,quantizer_efSearch=64          0.2079 0.5744 0.7803      0.00883       27816889    34
nprobe=128,quantizer_efSearch=32         0.2104 0.5925 0.8362      0.01181       55216129    26
nprobe=128,quantizer_efSearch=64         0.2107 0.5954 0.8418      0.01324       55088919    23
nprobe=128,quantizer_efSearch=128        0.2108 0.5964 0.8434      0.01921       55005401    16
nprobe=256,quantizer_efSearch=64         0.2128 0.6059 0.8803      0.02280      108768937    14
nprobe=256,quantizer_efSearch=256        0.2133 0.6081 0.8854      0.03051      108458483    10
nprobe=512,quantizer_efSearch=128        0.2134 0.6111 0.9032      0.04425      213473997    7
nprobe=512,quantizer_efSearch=256        0.2140 0.6120 0.9051      0.05363      213298287    6
nprobe=512,quantizer_efSearch=512        0.2143 0.6123 0.9047      0.06720      213122241    5
nprobe=1024,quantizer_efSearch=512       0.2145 0.6148 0.9122      0.12687      416906194    3
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ8` </summary>
Index size 440039065

 code_size 32

 log filename: autotune.dbdeep10M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1978 0.5215 0.6775      0.00762       33707487    40
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.0868 0.1481 0.1531      0.00221        1262150    136
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.1038 0.1839 0.1915      0.00221        1078875    136
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1244 0.2252 0.2345      0.00223        1611552    135
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1457 0.2936 0.3156      0.00233        2513405    129
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1520 0.3070 0.3304      0.00236        2506863    127
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1532 0.3137 0.3377      0.00240        2502390    126
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1632 0.3600 0.4020      0.00244        4991349    123
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1659 0.3732 0.4226      0.00248        4301609    122
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1669 0.3843 0.4389      0.00264        8589688    114
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1709 0.3863 0.4427      0.00262        4278565    115
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1728 0.3923 0.4440      0.00273        6308600    110
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1782 0.4074 0.4632      0.00299        6295824    101
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1813 0.4465 0.5415      0.00328        7836053    92
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1889 0.4733 0.5727      0.00332        9847831    91
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1899 0.4699 0.5712      0.00325        8502682    93
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1934 0.4982 0.6331      0.00407       15629973    74
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1973 0.5151 0.6641      0.00457       15552567    66
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2019 0.5356 0.6947      0.00626       19409055    49
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2020 0.5361 0.6949      0.00738       24638166    41
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2041 0.5552 0.7450      0.00686       30895393    44
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2045 0.5591 0.7507      0.00841       33461884    36
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2062 0.5659 0.7670      0.00758       30735988    40
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2079 0.5713 0.7737      0.00841       33300773    36
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2084 0.5826 0.8154      0.01146       58442419    27
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2100 0.5885 0.8236      0.01539       66087960    20
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.2106 0.5899 0.8364      0.02077      116582037    15
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2119 0.6065 0.8802      0.02463      114248967    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2121 0.6022 0.8709      0.02469      114881996    13
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.2124 0.6053 0.8798      0.03978      229208825    8
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.2128 0.6113 0.9013      0.05599      255845210    6
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2129 0.6115 0.9040      0.04769      224135144    7
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=64 0.2130 0.6135 0.9134      0.15545      850209756    2
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ8` </summary>
Index size 426800981

 code_size 32

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2152 0.6119 0.9063      0.06713      420217144    5
nprobe=1,quantizer_efSearch=4            0.1141 0.2140 0.2243      0.00185        1779247    163
nprobe=2,quantizer_efSearch=4            0.1408 0.2949 0.3236      0.00195        3550887    155
nprobe=4,quantizer_efSearch=4            0.1617 0.3715 0.4308      0.00204        7077432    147
nprobe=4,quantizer_efSearch=8            0.1734 0.3982 0.4637      0.00213        7084348    142
nprobe=4,quantizer_efSearch=16           0.1744 0.4036 0.4696      0.00237        7069599    127
nprobe=8,quantizer_efSearch=4            0.1857 0.4634 0.5740      0.00239       14122464    126
nprobe=8,quantizer_efSearch=8            0.1886 0.4703 0.5839      0.00245       14116034    123
nprobe=8,quantizer_efSearch=16           0.1921 0.4798 0.5949      0.00285       14088037    106
nprobe=16,quantizer_efSearch=16          0.2038 0.5370 0.7084      0.00432       27977607    70
nprobe=16,quantizer_efSearch=32          0.2045 0.5395 0.7126      0.00508       27945781    60
nprobe=32,quantizer_efSearch=32          0.2108 0.5753 0.7962      0.00667       55333641    45
nprobe=32,quantizer_efSearch=64          0.2114 0.5755 0.7968      0.00775       55288037    39
nprobe=64,quantizer_efSearch=32          0.2132 0.5934 0.8508      0.01077      109310035    28
nprobe=64,quantizer_efSearch=128         0.2135 0.5956 0.8541      0.01437      109106180    21
nprobe=128,quantizer_efSearch=32         0.2142 0.6042 0.8815      0.01871      215329565    17
nprobe=128,quantizer_efSearch=64         0.2146 0.6064 0.8872      0.01939      214825556    16
nprobe=256,quantizer_efSearch=64         0.2152 0.6111 0.9042      0.03984      420543784    8
nprobe=256,quantizer_efSearch=128        0.2153 0.6116 0.9061      0.03769      420447224    8
nprobe=512,quantizer_efSearch=128        0.2156 0.6128 0.9132      0.08729      817354872    4
nprobe=1024,quantizer_efSearch=128       0.2158 0.6134 0.9155      0.15513     1537410238    2
nprobe=1024,quantizer_efSearch=256       0.2159 0.6136 0.9162      0.16759     1587127354    2
```

</details>
<details><summary>`PCAR32,IVF65536(IVF256,PQ16x4fs,RFlat),SQ8` </summary>
Index size 410084249

 code_size 32

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_IVF256_PQ16x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1992 0.5263 0.6910      0.00420       28773804    72
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1206 0.2243 0.2353      0.00233        3124278    129
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1472 0.3031 0.3336      0.00243        3910641    124
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1483 0.3081 0.3386      0.00240        4902394    126
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1542 0.3528 0.4111      0.00252        7298228    120
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1567 0.3511 0.4042      0.00246        7504381    122
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1722 0.3965 0.4616      0.00265        7785669    114
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1736 0.4015 0.4676      0.00264        8421681    114
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1902 0.4731 0.5889      0.00316       14824773    95
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.1905 0.4724 0.5862      0.00345       16793045    87
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1911 0.4775 0.5952      0.00352       15434937    86
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.1927 0.4790 0.5969      0.00370       16762865    82
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1992 0.5263 0.6910      0.00443       28774096    68
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2032 0.5319 0.7013      0.00474       29367867    64
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2036 0.5359 0.7079      0.00506       29329639    60
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2042 0.5337 0.7041      0.00494       30658323    62
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2043 0.5337 0.7040      0.00581       33252878    52
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2064 0.5592 0.7722      0.00727       56344267    42
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2095 0.5688 0.7886      0.00782       56844403    39
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2107 0.5744 0.7951      0.00968       60679351    31
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2109 0.5744 0.7952      0.01188       60554034    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2116 0.5829 0.8345      0.01223      111324156    25
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.2130 0.5891 0.8424      0.01276      120068082    24
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2134 0.5928 0.8496      0.01241      112003602    25
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2137 0.5932 0.8507      0.01300      114545330    24
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2144 0.6045 0.8856      0.02621      217623423    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2146 0.6053 0.8871      0.02551      225120539    12
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.2150 0.6046 0.8897      0.03784      435312102    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2156 0.6097 0.9044      0.04856      427235740    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2157 0.6096 0.9044      0.04058      432195919    8
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2159 0.6115 0.9164      0.16236     1602446232    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2160 0.6118 0.9170      0.30468     3075155099    2
```

</details>
<details><summary>`PCAR32,IVF65536,SQ8` </summary>
Index size 408963488

 code_size 32

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2149 0.6102 0.9063      0.05646      419673424    6
nprobe=8                                 0.1924 0.4796 0.5970      0.00782       14055753    39
nprobe=32                                0.2110 0.5741 0.7961      0.01613       55206472    19
nprobe=64                                0.2132 0.5938 0.8541      0.01987      108950465    16
nprobe=128                               0.2144 0.6057 0.8880      0.03165      214228438    10
nprobe=256                               0.2149 0.6102 0.9063      0.05645      419673424    6
nprobe=512                               0.2154 0.6117 0.9140      0.10436      818787283    3
nprobe=1024                              0.2155 0.6122 0.9165      0.19079     1587567136    2
nprobe=2048                              0.2156 0.6126 0.9171      0.34855     3058665784    1
```

</details>
<details><summary>`PCAR64,IVF16384_HNSW32,SQ4` </summary>
Index size 408849109

 code_size 32

 log filename: autotune.dbdeep10M.PCAR64_IVF16384_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5414 0.9574 0.9950      0.37181     2687988420    1
nprobe=1,quantizer_efSearch=16           0.2302 0.3106 0.3133      0.00413       17272051    73
nprobe=2,quantizer_efSearch=16           0.3128 0.4465 0.4512      0.00645       33517018    47
nprobe=2,quantizer_efSearch=32           0.3135 0.4471 0.4518      0.00712       33358507    43
nprobe=4,quantizer_efSearch=4            0.3476 0.5253 0.5331      0.01093       67262377    28
nprobe=4,quantizer_efSearch=16           0.3930 0.5980 0.6068      0.01054       65027812    29
nprobe=8,quantizer_efSearch=4            0.4341 0.6860 0.6989      0.01864      126337077    17
nprobe=8,quantizer_efSearch=16           0.4570 0.7263 0.7399      0.01839      124473242    17
nprobe=8,quantizer_efSearch=32           0.4597 0.7297 0.7434      0.01901      123604492    16
nprobe=16,quantizer_efSearch=16          0.4964 0.8239 0.8433      0.03415      235236009    9
nprobe=16,quantizer_efSearch=32          0.4995 0.8304 0.8499      0.03289      233446222    10
nprobe=32,quantizer_efSearch=8           0.5074 0.8586 0.8835      0.06101      440700441    5
nprobe=32,quantizer_efSearch=32          0.5221 0.8934 0.9198      0.06112      436363995    5
nprobe=64,quantizer_efSearch=16          0.5263 0.9156 0.9459      0.10838      811450304    3
nprobe=64,quantizer_efSearch=32          0.5328 0.9299 0.9612      0.10522      807079324    3
nprobe=64,quantizer_efSearch=64          0.5332 0.9324 0.9639      0.10635      802903254    3
nprobe=64,quantizer_efSearch=128         0.5334 0.9327 0.9641      0.11161      800968593    3
nprobe=128,quantizer_efSearch=128        0.5393 0.9500 0.9856      0.19326     1473195199    2
nprobe=128,quantizer_efSearch=512        0.5394 0.9500 0.9856      0.21681     1470444636    2
nprobe=256,quantizer_efSearch=64         0.5404 0.9552 0.9920      0.34341     2694315466    1
nprobe=256,quantizer_efSearch=256        0.5414 0.9574 0.9949      0.36197     2690329488    1
nprobe=512,quantizer_efSearch=128        0.5418 0.9596 0.9978      0.62674     4909666287    1
nprobe=512,quantizer_efSearch=256        0.5419 0.9603 0.9986      0.63317     4931958709    1
nprobe=1024,quantizer_efSearch=128       0.5421 0.9602 0.9985      1.06829     8470222559    1
nprobe=1024,quantizer_efSearch=256       0.5423 0.9610 0.9995      1.14144     9007474181    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ4` </summary>
Index size 540612309

 code_size 32

 log filename: autotune.dbdeep10M.PCAR64_IVF262144_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5013 0.9236 0.9682      0.06920      110279532    5
nprobe=1,quantizer_efSearch=4            0.1552 0.1924 0.1935      0.00213         450751    141
nprobe=2,quantizer_efSearch=4            0.2110 0.2760 0.2784      0.00219         900928    138
nprobe=8,quantizer_efSearch=4            0.3514 0.5265 0.5338      0.00264        3601397    114
nprobe=8,quantizer_efSearch=8            0.3586 0.5375 0.5449      0.00293        3595539    103
nprobe=16,quantizer_efSearch=4           0.3910 0.6232 0.6347      0.00321        7194652    94
nprobe=16,quantizer_efSearch=8           0.4094 0.6536 0.6663      0.00404        7179053    75
nprobe=16,quantizer_efSearch=16          0.4135 0.6642 0.6772      0.00462        7161251    65
nprobe=32,quantizer_efSearch=4           0.4157 0.6840 0.7021      0.00516       14330432    59
nprobe=32,quantizer_efSearch=8           0.4368 0.7289 0.7489      0.00516       14320655    59
nprobe=32,quantizer_efSearch=16          0.4473 0.7507 0.7721      0.00593       14278540    51
nprobe=32,quantizer_efSearch=32          0.4508 0.7580 0.7796      0.00694       14240029    44
nprobe=64,quantizer_efSearch=8           0.4569 0.7852 0.8112      0.00944       28466624    32
nprobe=64,quantizer_efSearch=16          0.4713 0.8222 0.8499      0.00934       28426505    33
nprobe=64,quantizer_efSearch=32          0.4782 0.8336 0.8627      0.01006       28339734    30
nprobe=64,quantizer_efSearch=64          0.4795 0.8384 0.8677      0.01777       28272520    17
nprobe=64,quantizer_efSearch=128         0.4805 0.8403 0.8693      0.02115       28242890    15
nprobe=128,quantizer_efSearch=32         0.4888 0.8805 0.9183      0.01624       56219231    19
nprobe=128,quantizer_efSearch=64         0.4930 0.8904 0.9285      0.01917       56066808    16
nprobe=128,quantizer_efSearch=128        0.4942 0.8932 0.9312      0.02407       55959539    13
nprobe=256,quantizer_efSearch=64         0.4981 0.9164 0.9606      0.02970      110776697    11
nprobe=256,quantizer_efSearch=128        0.5009 0.9225 0.9667      0.03447      110518920    9
nprobe=256,quantizer_efSearch=256        0.5012 0.9233 0.9679      0.04260      110332916    8
nprobe=512,quantizer_efSearch=128        0.5036 0.9346 0.9819      0.06226      217221975    5
nprobe=512,quantizer_efSearch=256        0.5038 0.9363 0.9843      0.06995      216895715    5
nprobe=512,quantizer_efSearch=512        0.5044 0.9370 0.9850      0.08878      216665759    4
nprobe=1024,quantizer_efSearch=128       0.5046 0.9394 0.9883      0.11579      417485937    3
nprobe=1024,quantizer_efSearch=256       0.5052 0.9427 0.9928      0.13533      423080708    3
nprobe=1024,quantizer_efSearch=512       0.5056 0.9437 0.9937      0.17826      423551481    2
nprobe=2048,quantizer_efSearch=512       0.5062 0.9464 0.9976      0.26375      821473966    2
nprobe=4096,quantizer_efSearch=512       0.5063 0.9469 0.9984      0.46985     1501391420    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ4` </summary>
Index size 475829785

 code_size 32

 log filename: autotune.dbdeep10M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4653 0.7869 0.8089      0.00943       34039303    32
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1727 0.2241 0.2255      0.00229        1276033    131
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2273 0.2964 0.2984      0.00227        1621074    133
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2318 0.3019 0.3039      0.00232        2315484    130
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2805 0.3850 0.3880      0.00245        2529068    123
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2905 0.4004 0.4035      0.00247        2527322    122
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3197 0.4583 0.4622      0.00261        5044125    115
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3464 0.5127 0.5191      0.00253        4327327    119
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3512 0.5160 0.5216      0.00264        5021717    114
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3565 0.5416 0.5484      0.00311        8000327    97
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3660 0.5458 0.5522      0.00320        6331328    94
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3756 0.5692 0.5766      0.00309        8666066    97
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.3895 0.6134 0.6242      0.00369        7914556    82
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4079 0.6407 0.6513      0.00378        9926321    80
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4177 0.6583 0.6700      0.00414        9896421    73
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4191 0.6725 0.6863      0.00559       15890410    54
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4259 0.6881 0.7020      0.00578       17172950    53
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4377 0.7230 0.7403      0.00507       15757738    60
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4430 0.7342 0.7527      0.00627       15711346    48
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4510 0.7471 0.7648      0.00721       19653049    42
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4579 0.7628 0.7824      0.00764       19586365    40
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.4595 0.7639 0.7836      0.00942       24835099    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4771 0.8200 0.8471      0.00943       31166608    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4818 0.8299 0.8573      0.01010       33741259    30
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4830 0.8376 0.8656      0.01187       33649092    26
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4846 0.8385 0.8669      0.01223       38871021    25
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.4850 0.8388 0.8671      0.01383       49264480    22
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.4857 0.8425 0.8707      0.01458       38831355    21
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4909 0.8603 0.8910      0.01798       67394907    17
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.4913 0.8603 0.8910      0.01844       77757564    17
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4993 0.8948 0.9313      0.02056       66536241    15
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4995 0.8955 0.9320      0.02131       76942555    15
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.4996 0.8956 0.9322      0.02585       97457473    12
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.4999 0.8963 0.9328      0.02733       97424600    11
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5004 0.9029 0.9423      0.03126      133200241    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5033 0.9192 0.9613      0.03955      115901392    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5054 0.9228 0.9646      0.03185      131572320    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5059 0.9257 0.9685      0.03652      131184251    9
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5061 0.9263 0.9693      0.04517      151725103    7
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5069 0.9287 0.9729      0.05788      241820999    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5091 0.9367 0.9834      0.05456      227871648    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5096 0.9379 0.9843      0.06392      238050349    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.5097 0.9397 0.9879      0.09722      444220874    4
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5102 0.9411 0.9892      0.11620      453862086    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5107 0.9441 0.9933      0.13277      445582905    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5108 0.9442 0.9934      0.13056      465895912    3
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5115 0.9470 0.9975      0.22447      867877136    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.5116 0.9471 0.9978      0.23929      866672092    2
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ4` </summary>
Index size 435202261

 code_size 32

 log filename: autotune.dbdeep10M.PCAR64_IVF65536_HNSW32_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5044 0.9373 0.9870      0.10073      425253354    3
nprobe=1,quantizer_efSearch=4            0.1956 0.2589 0.2610      0.00191        1779472    158
nprobe=2,quantizer_efSearch=4            0.2620 0.3720 0.3752      0.00196        3559826    153
nprobe=4,quantizer_efSearch=4            0.3168 0.4849 0.4906      0.00234        7109390    129
nprobe=4,quantizer_efSearch=8            0.3434 0.5284 0.5351      0.00254        7124288    119
nprobe=4,quantizer_efSearch=16           0.3496 0.5373 0.5443      0.00285        7105920    106
nprobe=4,quantizer_efSearch=32           0.3499 0.5377 0.5447      0.00338        7100214    89
nprobe=8,quantizer_efSearch=4            0.3882 0.6365 0.6491      0.00367       14220084    82
nprobe=8,quantizer_efSearch=8            0.3939 0.6477 0.6602      0.00376       14207968    80
nprobe=8,quantizer_efSearch=16           0.4023 0.6606 0.6737      0.00408       14168236    74
nprobe=16,quantizer_efSearch=4           0.4298 0.7266 0.7467      0.00613       28349663    49
nprobe=16,quantizer_efSearch=8           0.4427 0.7533 0.7747      0.00629       28277807    48
nprobe=16,quantizer_efSearch=16          0.4500 0.7661 0.7880      0.00649       28226443    47
nprobe=16,quantizer_efSearch=32          0.4507 0.7686 0.7906      0.00706       28181868    43
nprobe=16,quantizer_efSearch=64          0.4516 0.7693 0.7913      0.00808       28165283    38
nprobe=32,quantizer_efSearch=8           0.4624 0.8158 0.8445      0.01101       56150371    28
nprobe=32,quantizer_efSearch=128         0.4764 0.8443 0.8740      0.01505       55837957    20
nprobe=64,quantizer_efSearch=16          0.4860 0.8794 0.9175      0.02041      110918614    15
nprobe=64,quantizer_efSearch=32          0.4900 0.8896 0.9284      0.02092      110606327    15
nprobe=64,quantizer_efSearch=128         0.4927 0.8932 0.9320      0.02406      110356985    13
nprobe=128,quantizer_efSearch=128        0.5012 0.9207 0.9668      0.04145      217126655    8
nprobe=256,quantizer_efSearch=64         0.5027 0.9335 0.9830      0.07430      425941512    5
nprobe=256,quantizer_efSearch=128        0.5039 0.9363 0.9859      0.07712      425692764    4
nprobe=256,quantizer_efSearch=256        0.5044 0.9374 0.9870      0.08149      425359330    4
nprobe=512,quantizer_efSearch=64         0.5049 0.9399 0.9901      0.13802      818754702    3
nprobe=512,quantizer_efSearch=128        0.5059 0.9438 0.9942      0.14367      827201322    3
nprobe=1024,quantizer_efSearch=128       0.5062 0.9442 0.9953      0.25940     1555958032    2
nprobe=1024,quantizer_efSearch=256       0.5064 0.9453 0.9969      0.28543     1604846237    2
nprobe=1024,quantizer_efSearch=512       0.5065 0.9457 0.9975      0.32512     1606365868    1
nprobe=2048,quantizer_efSearch=512       0.5067 0.9464 0.9985      0.57510     3082259410    1
```

</details>
<details><summary>`PCAR64,IVF65536(IVF256,PQ32x4fs,RFlat),SQ4` </summary>
Index size 419078169

 code_size 32

 log filename: autotune.dbdeep10M.PCAR64_IVF65536_IVF256_PQ32x4fs_RFlat__SQ4.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4969 0.9071 0.9479      0.03679      223784318    9
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2     0.1936 0.2570 0.2593      0.00242        1965518    124
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4     0.2032 0.2712 0.2735      0.00250        2135160    120
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2101 0.2787 0.2810      0.00252        3134054    120
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2771 0.3927 0.3969      0.00248        4256329    121
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2805 0.3962 0.4005      0.00252        4913891    119
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.2847 0.4023 0.4064      0.00265        4913049    114
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2862 0.4041 0.4082      0.00276        4909805    109
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3037 0.4504 0.4557      0.00290        7874859    104
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3463 0.5268 0.5344      0.00302        8467421    100
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.3523 0.5361 0.5437      0.00327        8461044    92
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.3526 0.5364 0.5440      0.00355        8456519    85
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3534 0.5362 0.5439      0.00362        9780136    83
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4003 0.6515 0.6641      0.00431       14885656    70
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.4070 0.6612 0.6741      0.00488       15513351    62
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4076 0.6626 0.6755      0.00502       16827454    60
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4077 0.6627 0.6756      0.00581       19418830    52
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.4079 0.6628 0.6758      0.00613       19415058    49
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4196 0.6916 0.7076      0.00631       29328831    48
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4281 0.7030 0.7195      0.00602       29911182    50
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4500 0.7584 0.7780      0.00647       30933915    47
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4523 0.7689 0.7894      0.00661       29554828    46
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4526 0.7672 0.7874      0.00615       29571479    49
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4536 0.7693 0.7896      0.00673       30850998    45
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4538 0.7697 0.7901      0.00749       33448942    41
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.4539 0.7716 0.7921      0.00891       38564599    34
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.4540 0.7698 0.7902      0.00851       38582393    36
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4774 0.8397 0.8670      0.01044       57323824    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4799 0.8453 0.8730      0.01106       58548381    28
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.4803 0.8467 0.8747      0.02369       66217014    13
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4924 0.8892 0.9256      0.02325      113387823    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4943 0.8939 0.9315      0.02117      113106207    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4954 0.8954 0.9332      0.02058      115605152    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4962 0.9098 0.9522      0.03796      219803012    8
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4969 0.9071 0.9479      0.03765      223793755    8
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5010 0.9195 0.9637      0.03491      220431341    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5012 0.9201 0.9653      0.04037      219990891    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5024 0.9221 0.9673      0.03934      227306318    8
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5028 0.9224 0.9676      0.04359      227264805    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5052 0.9371 0.9860      0.07228      436164860    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5059 0.9384 0.9875      0.11053      435467797    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5067 0.9405 0.9909      0.13904      856140489    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5078 0.9439 0.9955      0.14094      839939823    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.5082 0.9442 0.9955      0.15391      834623385    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5083 0.9445 0.9960      0.14574      838934149    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5085 0.9461 0.9980      0.26225     1616245355    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5086 0.9463 0.9980      0.48754     3106561863    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5087 0.9469 0.9988      0.48724     3101124928    1
```

</details>
<details><summary>`PCAR64,IVF65536,SQ4` </summary>
Index size 417364768

 code_size 32

 log filename: autotune.dbdeep10M.PCAR64_IVF65536_SQ4.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5081 0.9383 0.9874      0.08082      424597865    4
nprobe=8                                 0.4114 0.6638 0.6760      0.01148       14126580    27
nprobe=16                                0.4576 0.7708 0.7921      0.01294       28117000    24
nprobe=32                                0.4814 0.8463 0.8747      0.01690       55742660    18
nprobe=64                                0.4949 0.8934 0.9321      0.02510      110152701    12
nprobe=128                               0.5041 0.9214 0.9670      0.04424      216662768    7
nprobe=256                               0.5081 0.9383 0.9874      0.07740      424597865    4
nprobe=512                               0.5097 0.9450 0.9957      0.14261      827752336    3
nprobe=1024                              0.5104 0.9461 0.9978      0.27414     1603126059    2
nprobe=2048                              0.5107 0.9469 0.9988      0.48808     3086397938    1
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF16384_HNSW32,PQ128x4fs` </summary>
Index size 758013132

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF16384_HNSW32_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5655 0.9619 0.9956      0.08491     3068919645    4
nprobe=2,quantizer_efSearch=4            0.2929 0.4208 0.4277      0.00362       42966817    83
nprobe=4,quantizer_efSearch=4            0.3575 0.5392 0.5502      0.00390       82257387    78
nprobe=4,quantizer_efSearch=8            0.4017 0.6069 0.6186      0.00432       80459727    70
nprobe=8,quantizer_efSearch=4            0.4506 0.7057 0.7231      0.00538      153771047    56
nprobe=8,quantizer_efSearch=16           0.4736 0.7437 0.7617      0.00684      151413374    44
nprobe=16,quantizer_efSearch=4           0.4778 0.7740 0.7955      0.00675      285320884    45
nprobe=16,quantizer_efSearch=16          0.5136 0.8353 0.8591      0.00800      283428698    38
nprobe=32,quantizer_efSearch=16          0.5363 0.8922 0.9197      0.01329      521258995    23
nprobe=32,quantizer_efSearch=32          0.5424 0.9017 0.9294      0.01490      517936801    21
nprobe=32,quantizer_efSearch=64          0.5441 0.9050 0.9326      0.01538      515744249    20
nprobe=64,quantizer_efSearch=32          0.5553 0.9333 0.9639      0.01902      947129569    16
nprobe=64,quantizer_efSearch=128         0.5583 0.9390 0.9699      0.02238      939877862    14
nprobe=128,quantizer_efSearch=32         0.5591 0.9469 0.9789      0.02726     1711240053    12
nprobe=128,quantizer_efSearch=64         0.5628 0.9538 0.9867      0.02872     1710116759    11
nprobe=128,quantizer_efSearch=128        0.5643 0.9558 0.9887      0.03149     1703974166    10
nprobe=256,quantizer_efSearch=64         0.5648 0.9595 0.9930      0.04336     3074427338    7
nprobe=256,quantizer_efSearch=128        0.5661 0.9620 0.9956      0.04718     3077874419    7
nprobe=512,quantizer_efSearch=128        0.5666 0.9641 0.9978      0.07726     5539846215    4
nprobe=512,quantizer_efSearch=256        0.5668 0.9645 0.9983      0.08930     5575294286    4
nprobe=1024,quantizer_efSearch=256       0.5669 0.9654 0.9993      0.14706    10073415534    3
nprobe=4096,quantizer_efSearch=256       0.5670 0.9656 0.9995      0.28042    20047542720    2
```

</details>
<details><summary>`OPQ128_256,IVF16384_HNSW32,PQ128x4fsr` </summary>
Index size 757951692

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF16384_HNSW32_PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7426 0.8706 0.8713      0.03186      120407978    10
nprobe=1,quantizer_efSearch=8            0.3200 0.3496 0.3500      0.00274        7902753    110
nprobe=1,quantizer_efSearch=16           0.3224 0.3522 0.3526      0.00332        7881767    91
nprobe=2,quantizer_efSearch=4            0.4163 0.4633 0.4637      0.00361       15698752    84
nprobe=2,quantizer_efSearch=8            0.4436 0.4938 0.4942      0.00398       15676198    76
nprobe=2,quantizer_efSearch=16           0.4477 0.4985 0.4989      0.00442       15644287    68
nprobe=4,quantizer_efSearch=4            0.5269 0.5933 0.5940      0.00595       31090079    51
nprobe=4,quantizer_efSearch=8            0.5675 0.6399 0.6406      0.00638       31123587    48
nprobe=4,quantizer_efSearch=16           0.5749 0.6482 0.6489      0.00681       31055703    45
nprobe=4,quantizer_efSearch=32           0.5766 0.6498 0.6505      0.00757       31030911    40
nprobe=4,quantizer_efSearch=64           0.5768 0.6500 0.6507      0.00928       31023756    33
nprobe=4,quantizer_efSearch=128          0.5769 0.6501 0.6508      0.01241       31022796    25
nprobe=8,quantizer_efSearch=4            0.6520 0.7487 0.7494      0.01695       61543722    18
nprobe=8,quantizer_efSearch=8            0.6613 0.7605 0.7612      0.01686       61470026    18
nprobe=8,quantizer_efSearch=16           0.6733 0.7751 0.7758      0.01750       61343305    18
nprobe=8,quantizer_efSearch=32           0.6765 0.7785 0.7792      0.01780       61284885    17
nprobe=8,quantizer_efSearch=64           0.6767 0.7786 0.7793      0.01911       61260982    16
nprobe=8,quantizer_efSearch=128          0.6770 0.7789 0.7796      0.02284       61254339    14
nprobe=16,quantizer_efSearch=16          0.7393 0.8666 0.8673      0.03198      120618310    10
nprobe=16,quantizer_efSearch=32          0.7426 0.8706 0.8713      0.03246      120407978    10
nprobe=16,quantizer_efSearch=64          0.7430 0.8710 0.8717      0.03381      120346083    9
nprobe=16,quantizer_efSearch=128         0.7433 0.8713 0.8720      0.03890      120324852    8
nprobe=32,quantizer_efSearch=8           0.7656 0.9086 0.9094      0.07269      236583236    5
nprobe=32,quantizer_efSearch=32          0.7855 0.9341 0.9349      0.07405      235375858    5
nprobe=32,quantizer_efSearch=64          0.7869 0.9347 0.9355      0.07519      235155753    4
nprobe=32,quantizer_efSearch=128         0.7873 0.9350 0.9358      0.07232      235087763    5
nprobe=64,quantizer_efSearch=16          0.7996 0.9587 0.9595      0.13060      459148508    3
nprobe=64,quantizer_efSearch=32          0.8071 0.9682 0.9691      0.13483      457685302    3
nprobe=64,quantizer_efSearch=64          0.8091 0.9703 0.9712      0.13583      456863952    3
nprobe=64,quantizer_efSearch=256         0.8096 0.9711 0.9720      0.14477      456484666    3
nprobe=128,quantizer_efSearch=64         0.8202 0.9880 0.9888      0.25768      882839678    2
nprobe=128,quantizer_efSearch=256        0.8209 0.9896 0.9904      0.26020      880872608    2
nprobe=256,quantizer_efSearch=128        0.8234 0.9957 0.9969      0.47630     1691495941    1
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fs` </summary>
Index size 1316061388

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF262144_HNSW32_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3120 0.4288 0.4324      0.00632        1805041    48
nprobe=2,quantizer_efSearch=4            0.2166 0.2856 0.2875      0.00371         902902    81
nprobe=4,quantizer_efSearch=4            0.2792 0.3825 0.3859      0.00394        1805946    77
nprobe=8,quantizer_efSearch=4            0.3740 0.5391 0.5462      0.00565        3604561    54
nprobe=16,quantizer_efSearch=4           0.4225 0.6400 0.6508      0.00643        7202323    47
nprobe=16,quantizer_efSearch=8           0.4417 0.6723 0.6831      0.00831        7186966    37
nprobe=32,quantizer_efSearch=4           0.4497 0.7019 0.7165      0.00772       14338270    39
nprobe=32,quantizer_efSearch=8           0.4774 0.7514 0.7673      0.00973       14326064    31
nprobe=64,quantizer_efSearch=8           0.4949 0.8008 0.8204      0.01174       28478766    26
nprobe=64,quantizer_efSearch=16          0.5135 0.8402 0.8607      0.01536       28434484    20
nprobe=128,quantizer_efSearch=16         0.5253 0.8719 0.8963      0.01889       56279934    16
nprobe=256,quantizer_efSearch=16         0.5283 0.8830 0.9094      0.02343      108553920    13
nprobe=128,quantizer_efSearch=32         0.5385 0.8980 0.9246      0.02415       56223543    13
nprobe=256,quantizer_efSearch=32         0.5425 0.9164 0.9461      0.02904      110498305    11
nprobe=128,quantizer_efSearch=64         0.5444 0.9078 0.9352      0.03468       56055895    9
nprobe=256,quantizer_efSearch=64         0.5488 0.9314 0.9630      0.04020      110704845    8
nprobe=512,quantizer_efSearch=64         0.5533 0.9421 0.9750      0.04858      215406750    7
nprobe=512,quantizer_efSearch=128        0.5569 0.9501 0.9844      0.06470      217013394    5
nprobe=1024,quantizer_efSearch=128       0.5575 0.9547 0.9897      0.08338      416703475    4
nprobe=2048,quantizer_efSearch=128       0.5577 0.9554 0.9905      0.09866      692804965    4
nprobe=512,quantizer_efSearch=256        0.5592 0.9524 0.9869      0.10100      216669431    3
nprobe=1024,quantizer_efSearch=256       0.5606 0.9592 0.9946      0.11591      422449836    3
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fsr` </summary>
Index size 1315129548

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF262144_HNSW32_PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6260 0.6996 0.7002      0.03381        7125952    9
nprobe=4,quantizer_efSearch=4            0.3571 0.3835 0.3839      0.00633        1798173    48
nprobe=4,quantizer_efSearch=8            0.4021 0.4333 0.4337      0.00832        1798642    37
nprobe=4,quantizer_efSearch=16           0.4123 0.4437 0.4441      0.01149        1792271    27
nprobe=8,quantizer_efSearch=8            0.5096 0.5566 0.5573      0.01522        3588158    20
nprobe=8,quantizer_efSearch=16           0.5242 0.5725 0.5732      0.01781        3577533    17
nprobe=16,quantizer_efSearch=4           0.5824 0.6471 0.6477      0.02692        7174465    12
nprobe=16,quantizer_efSearch=8           0.6116 0.6821 0.6827      0.02640        7162077    12
nprobe=16,quantizer_efSearch=16          0.6219 0.6944 0.6950      0.02837        7144153    11
nprobe=16,quantizer_efSearch=32          0.6260 0.6996 0.7002      0.03111        7125952    10
nprobe=16,quantizer_efSearch=64          0.6277 0.7020 0.7026      0.04004        7117589    8
nprobe=32,quantizer_efSearch=8           0.6808 0.7679 0.7685      0.06735       14274316    5
nprobe=32,quantizer_efSearch=16          0.7018 0.7914 0.7919      0.06699       14238261    5
nprobe=32,quantizer_efSearch=32          0.7063 0.7978 0.7983      0.06553       14202157    5
nprobe=32,quantizer_efSearch=64          0.7092 0.8013 0.8018      0.07784       14178177    4
nprobe=32,quantizer_efSearch=256         0.7093 0.8015 0.8020      0.10908       14166787    3
nprobe=64,quantizer_efSearch=8           0.7170 0.8207 0.8215      0.12337       28369323    3
nprobe=64,quantizer_efSearch=16          0.7499 0.8609 0.8617      0.12367       28333432    3
nprobe=64,quantizer_efSearch=32          0.7629 0.8759 0.8767      0.12878       28242466    3
nprobe=64,quantizer_efSearch=128         0.7694 0.8837 0.8845      0.15089       28142590    2
nprobe=64,quantizer_efSearch=256         0.7696 0.8842 0.8850      0.16220       28133672    2
nprobe=128,quantizer_efSearch=64         0.8066 0.9355 0.9365      0.24122       55868506    2
nprobe=128,quantizer_efSearch=128        0.8080 0.9372 0.9381      0.25402       55761032    2
nprobe=128,quantizer_efSearch=256        0.8086 0.9378 0.9387      0.25736       55723276    2
nprobe=128,quantizer_efSearch=512        0.8087 0.9380 0.9389      0.30907       55714527    2
nprobe=256,quantizer_efSearch=32         0.8090 0.9450 0.9459      0.44016      110097368    1
nprobe=256,quantizer_efSearch=64         0.8218 0.9630 0.9640      0.43720      110329453    1
nprobe=256,quantizer_efSearch=128        0.8247 0.9686 0.9696      0.46830      110083718    1
nprobe=256,quantizer_efSearch=512        0.8269 0.9710 0.9720      0.50270      109857052    1
nprobe=512,quantizer_efSearch=64         0.8271 0.9749 0.9760      0.84333      214706516    1
nprobe=512,quantizer_efSearch=128        0.8344 0.9849 0.9860      0.86419      216341558    1
nprobe=512,quantizer_efSearch=256        0.8370 0.9873 0.9885      0.90873      216043083    1
nprobe=1024,quantizer_efSearch=128       0.8372 0.9896 0.9909      1.63897      208935462    1
nprobe=1024,quantizer_efSearch=256       0.8387 0.9945 0.9957      1.66722      211737504    1
nprobe=1024,quantizer_efSearch=512       0.8398 0.9949 0.9961      1.73593      211921990    1
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 1264840208

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5570 0.9505 0.9849      0.06284      439266025    5
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2085 0.2692 0.2710      0.00216        1274658    139
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2192 0.2877 0.2898      0.00257        1267770    117
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2793 0.3804 0.3842      0.00259        2173924    116
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2824 0.3844 0.3882      0.00288        2173200    105
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3053 0.4154 0.4191      0.00283        2525203    107
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3083 0.4161 0.4194      0.00312        3229286    97
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3101 0.4208 0.4245      0.00296        2523141    102
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3618 0.5136 0.5202      0.00320        4347576    94
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3800 0.5389 0.5455      0.00356        5036176    85
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4163 0.6204 0.6303      0.00392        7979811    77
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4378 0.6560 0.6660      0.00439        8650679    69
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4462 0.6739 0.6850      0.00498        8588640    61
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4519 0.6873 0.6983      0.00618        9919011    49
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4824 0.7578 0.7740      0.00762       15702281    40
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4876 0.7663 0.7825      0.00712       17168329    43
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4976 0.7856 0.8021      0.00923       19631734    33
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5024 0.8112 0.8309      0.00868       30219855    35
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5162 0.8401 0.8611      0.01258       31442695    24
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5189 0.8477 0.8692      0.01166       31066787    26
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5223 0.8498 0.8711      0.01218       34029938    25
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5255 0.8600 0.8817      0.01370       33634928    22
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5311 0.8839 0.9100      0.01384       59707925    22
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5344 0.8911 0.9176      0.01645       58873308    19
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5420 0.9020 0.9289      0.02001       67327660    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5442 0.9094 0.9365      0.02206       66515827    14
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5512 0.9303 0.9613      0.02513      117618670    12
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5524 0.9353 0.9665      0.03209      132908317    10
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5572 0.9464 0.9802      0.03910      226424939    8
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5589 0.9523 0.9865      0.04184      231025422    8
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5595 0.9535 0.9878      0.04566      241113180    7
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.5597 0.9537 0.9880      0.05547      261464712    6
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5601 0.9590 0.9946      0.07051      452225974    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5603 0.9592 0.9948      0.07793      472412134    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5606 0.9606 0.9964      0.10745      463951394    3
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5607 0.9619 0.9979      0.17004      844745379    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5610 0.9627 0.9990      0.21422     1670171744    2
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 1264895504

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2309 0.2470 0.2474      0.00347        1084362    87
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1842 0.1943 0.1948      0.00249         817708    121
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2033 0.2138 0.2143      0.00282        1170931    107
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.2309 0.2470 0.2474      0.00334        1084548    90
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2645 0.2824 0.2828      0.00325        1268893    93
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2868 0.3058 0.3062      0.00341        1621329    88
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2915 0.3108 0.3112      0.00352        1620339    86
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3033 0.3234 0.3238      0.00380        2319171    80
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4048 0.4361 0.4365      0.00530        3221145    57
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.4075 0.4398 0.4402      0.00550        3218214    55
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4081 0.4397 0.4401      0.00609        4565281    50
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4125 0.4450 0.4454      0.00641        4564769    47
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4126 0.4451 0.4455      0.00665        4564816    46
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4129 0.4453 0.4457      0.00853        7212665    36
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.4132 0.4456 0.4460      0.01225       12484334    25
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.4422 0.4875 0.4882      0.01606        3976562    19
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4897 0.5410 0.5417      0.01533        4319347    20
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.5206 0.5754 0.5761      0.01657        6342913    19
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5218 0.5765 0.5772      0.01755        8982985    18
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.5223 0.5770 0.5777      0.02019       14088936    15
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.5748 0.6439 0.6446      0.02837        7906975    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6085 0.6834 0.6841      0.02975        8578671    11
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.6206 0.6980 0.6987      0.03205        9911252    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6207 0.6981 0.6988      0.02979        9902730    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6234 0.7012 0.7019      0.03141       12545300    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6236 0.7018 0.7025      0.03284       17781143    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.6824 0.7732 0.7740      0.06269       15688912    5
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.7019 0.7971 0.7979      0.06892       24907514    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7038 0.7998 0.8005      0.06826       19613344    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7044 0.8005 0.8012      0.07161       24806553    5
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.7666 0.8831 0.8839      0.13294       49064105    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7674 0.8835 0.8843      0.13059       38534730    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.7676 0.8839 0.8847      0.14089       67023000    3
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.8001 0.9314 0.9323      0.23427       61450475    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.8040 0.9342 0.9351      0.24335       61299720    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8068 0.9376 0.9384      0.24450       76813104    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.8232 0.9699 0.9707      0.52901      120726643    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.8241 0.9715 0.9722      0.49290      151392292    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.8282 0.9794 0.9804      0.86107      222849174    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8371 0.9879 0.9889      0.89285      236358387    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.8377 0.9952 0.9963      1.70152      233977228    1
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fs` </summary>
Index size 871215820

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF65536_HNSW32_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5563 0.9571 0.9898      0.07334      424359760    5
nprobe=2,quantizer_efSearch=4            0.2740 0.3794 0.3829      0.00233        3557117    129
nprobe=4,quantizer_efSearch=4            0.3378 0.4930 0.4991      0.00266        7111999    113
nprobe=8,quantizer_efSearch=4            0.4185 0.6460 0.6559      0.00378       14217546    80
nprobe=8,quantizer_efSearch=8            0.4263 0.6587 0.6688      0.00406       14198512    74
nprobe=16,quantizer_efSearch=4           0.4585 0.7372 0.7528      0.00490       28319614    62
nprobe=16,quantizer_efSearch=8           0.4767 0.7677 0.7838      0.00535       28264628    57
nprobe=16,quantizer_efSearch=16          0.4835 0.7774 0.7936      0.00638       28212709    48
nprobe=32,quantizer_efSearch=8           0.5045 0.8341 0.8553      0.00700       56107688    43
nprobe=32,quantizer_efSearch=16          0.5147 0.8522 0.8742      0.00799       55974547    38
nprobe=64,quantizer_efSearch=8           0.5163 0.8664 0.8912      0.00924      110939019    33
nprobe=32,quantizer_efSearch=32          0.5187 0.8580 0.8802      0.01027       55864764    30
nprobe=64,quantizer_efSearch=16          0.5313 0.8961 0.9228      0.01061      110797860    29
nprobe=64,quantizer_efSearch=32          0.5368 0.9086 0.9357      0.01292      110474836    24
nprobe=128,quantizer_efSearch=16         0.5379 0.9165 0.9461      0.01503      218054015    20
nprobe=128,quantizer_efSearch=32         0.5471 0.9341 0.9651      0.01636      217728407    19
nprobe=128,quantizer_efSearch=64         0.5496 0.9396 0.9705      0.02029      217106321    15
nprobe=256,quantizer_efSearch=32         0.5502 0.9444 0.9768      0.02237      423990543    14
nprobe=256,quantizer_efSearch=64         0.5551 0.9545 0.9871      0.02819      424982038    11
nprobe=256,quantizer_efSearch=128        0.5559 0.9563 0.9890      0.03513      424794702    9
nprobe=512,quantizer_efSearch=128        0.5575 0.9620 0.9954      0.04483      825271636    7
nprobe=512,quantizer_efSearch=256        0.5581 0.9631 0.9966      0.06042      827830339    5
nprobe=512,quantizer_efSearch=512        0.5583 0.9635 0.9969      0.08187      826962233    4
nprobe=1024,quantizer_efSearch=512       0.5584 0.9646 0.9988      0.11926     1602597782    3
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fsr` </summary>
Index size 871260876

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF65536_HNSW32_PQ128x4fsr.l.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7000 0.7974 0.7981      0.03229       28125565    10
nprobe=1,quantizer_efSearch=4            0.2540 0.2710 0.2714      0.00361        1780466    84
nprobe=1,quantizer_efSearch=8            0.2687 0.2863 0.2867      0.00464        1778170    65
nprobe=2,quantizer_efSearch=4            0.3542 0.3834 0.3840      0.00486        3549312    62
nprobe=2,quantizer_efSearch=8            0.3839 0.4147 0.4153      0.00591        3552277    51
nprobe=4,quantizer_efSearch=4            0.4517 0.4995 0.5001      0.00734        7095968    41
nprobe=4,quantizer_efSearch=8            0.4900 0.5444 0.5450      0.00840        7105919    36
nprobe=4,quantizer_efSearch=16           0.4969 0.5526 0.5532      0.01010        7090852    30
nprobe=4,quantizer_efSearch=32           0.4987 0.5550 0.5556      0.01271        7087699    24
nprobe=8,quantizer_efSearch=4            0.5820 0.6552 0.6559      0.01522       14190349    20
nprobe=8,quantizer_efSearch=8            0.5955 0.6691 0.6698      0.01573       14171126    20
nprobe=8,quantizer_efSearch=16           0.6073 0.6835 0.6842      0.01745       14139102    18
nprobe=8,quantizer_efSearch=32           0.6097 0.6861 0.6868      0.02016       14125323    15
nprobe=8,quantizer_efSearch=64           0.6100 0.6863 0.6870      0.02419       14121178    13
nprobe=16,quantizer_efSearch=4           0.6597 0.7508 0.7515      0.02501       28260109    12
nprobe=16,quantizer_efSearch=8           0.6889 0.7836 0.7843      0.02579       28210129    12
nprobe=16,quantizer_efSearch=16          0.6976 0.7936 0.7943      0.02940       28163949    11
nprobe=16,quantizer_efSearch=32          0.7000 0.7974 0.7981      0.03241       28125565    10
nprobe=16,quantizer_efSearch=64          0.7010 0.7983 0.7990      0.03684       28107636    9
nprobe=16,quantizer_efSearch=128         0.7011 0.7983 0.7990      0.04483       28103042    7
nprobe=32,quantizer_efSearch=8           0.7413 0.8549 0.8557      0.05442       56001176    6
nprobe=32,quantizer_efSearch=16          0.7579 0.8746 0.8754      0.05637       55869354    6
nprobe=32,quantizer_efSearch=32          0.7619 0.8802 0.8810      0.05917       55765256    6
nprobe=32,quantizer_efSearch=64          0.7623 0.8812 0.8820      0.06476       55711344    5
nprobe=32,quantizer_efSearch=128         0.7625 0.8815 0.8823      0.07373       55695266    5
nprobe=32,quantizer_efSearch=256         0.7626 0.8816 0.8824      0.08886       55692015    4
nprobe=64,quantizer_efSearch=8           0.7667 0.8904 0.8913      0.10037      110717980    3
nprobe=64,quantizer_efSearch=16          0.7908 0.9224 0.9234      0.10363      110576056    3
nprobe=64,quantizer_efSearch=32          0.8002 0.9352 0.9362      0.10712      110265645    3
nprobe=64,quantizer_efSearch=64          0.8028 0.9389 0.9399      0.11192      110078113    3
nprobe=64,quantizer_efSearch=128         0.8031 0.9392 0.9402      0.12096      110004928    3
nprobe=128,quantizer_efSearch=16         0.8054 0.9450 0.9461      0.19248      217624089    2
nprobe=128,quantizer_efSearch=32         0.8190 0.9641 0.9653      0.19603      217304407    2
nprobe=128,quantizer_efSearch=64         0.8240 0.9699 0.9709      0.20317      216703925    2
nprobe=128,quantizer_efSearch=128        0.8242 0.9710 0.9721      0.21217      216372003    2
nprobe=128,quantizer_efSearch=256        0.8246 0.9710 0.9721      0.23204      216288879    2
nprobe=256,quantizer_efSearch=64         0.8296 0.9854 0.9870      0.38570      424208420    1
nprobe=256,quantizer_efSearch=128        0.8306 0.9875 0.9889      0.39706      424048178    1
nprobe=256,quantizer_efSearch=256        0.8312 0.9886 0.9898      0.41695      423751200    1
nprobe=256,quantizer_efSearch=512        0.8315 0.9887 0.9899      0.44015      423646440    1
nprobe=512,quantizer_efSearch=128        0.8337 0.9936 0.9953      0.74629      823923263    1
```

</details>
<details><summary>`OPQ128_256,IVF65536(IVF256,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 858701328

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF65536_IVF256_PQ128x4fs_RFlat__PQ128x4fs.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.5480 0.9354 0.9659      0.01616      223393999    19
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.1732 0.2277 0.2296      0.00177        1877015    170
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.2648 0.3698 0.3735      0.00218        3757483    138
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2840 0.3946 0.3984      0.00230        3925888    131
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2887 0.3997 0.4035      0.00231        3920047    130
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2994 0.4129 0.4167      0.00249        4258565    121
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.3182 0.4621 0.4678      0.00255        7383383    118
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.3293 0.4803 0.4865      0.00258        7335534    117
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3563 0.5204 0.5272      0.00284        7491194    106
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3599 0.5209 0.5271      0.00296        7866831    102
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3695 0.5378 0.5446      0.00299        7822299    101
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3702 0.5409 0.5476      0.00343        7815284    88
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.3703 0.5634 0.5724      0.00335       14604188    90
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4012 0.6189 0.6284      0.00336       14725087    90
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4204 0.6459 0.6554      0.00389       15026032    78
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4305 0.6638 0.6739      0.00419       14885676    72
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4371 0.6724 0.6825      0.00445       15538356    68
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.4407 0.7058 0.7208      0.00449       29045861    67
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.4682 0.7477 0.7629      0.00465       29279322    65
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.4744 0.7621 0.7778      0.00482       28965631    63
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4761 0.7616 0.7770      0.00522       29875983    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.4837 0.7781 0.7944      0.00581       29548357    52
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.5022 0.8293 0.8512      0.00623       56814387    49
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5115 0.8432 0.8648      0.00672       57995952    45
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.5157 0.8522 0.8748      0.00743       57248829    41
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.5174 0.8721 0.8981      0.00970      111734259    31
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5199 0.8599 0.8821      0.00960       61081010    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5331 0.9017 0.9283      0.01015      111889747    30
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5352 0.9035 0.9306      0.01178      114626836    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5387 0.9110 0.9380      0.01156      112935820    26
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.5409 0.9220 0.9520      0.01504      222811843    20
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.5500 0.9388 0.9696      0.01710      219572969    18
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.5513 0.9413 0.9721      0.02546      226783542    12
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.5549 0.9544 0.9867      0.02485      437769454    13
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5563 0.9562 0.9887      0.02807      429528872    11
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5567 0.9572 0.9898      0.03109      434301904    10
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5575 0.9623 0.9956      0.03989      853665018    8
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5579 0.9625 0.9957      0.04334      832376784    7
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5583 0.9634 0.9967      0.04401      836571772    7
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.5584 0.9635 0.9968      0.05501      836501947    6
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.5587 0.9642 0.9983      0.13579     3100211890    3
```

</details>
<details><summary>`OPQ128_256,IVF65536,PQ128x4fs` </summary>
Index size 853425431

 code_size 64

 log filename: autotune.dbdeep10M.OPQ128_256_IVF65536_PQ128x4fs.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5564 0.9573 0.9900      0.03815      423475802    8
nprobe=64                                0.5396 0.9133 0.9403      0.02894      109947166    11
nprobe=128                               0.5510 0.9413 0.9721      0.03122      216193412    10
nprobe=256                               0.5564 0.9573 0.9900      0.03886      423475802    8
nprobe=512                               0.5584 0.9636 0.9970      0.05023      825310138    6
nprobe=4096                              0.5585 0.9656 0.9999      0.18862     5929487877    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 829550038

 code_size 64

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.2777 0.3091 0.3096      0.00737       21944060    41
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1671 0.1823 0.1828      0.00194        1173719    155
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4       0.1866 0.2041 0.2046      0.00233        1166939    129
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.2104 0.2352 0.2357      0.00231        1084957    131
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2485 0.2774 0.2778      0.00241        1267600    125
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2712 0.3014 0.3018      0.00262        2312929    115
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8       0.2763 0.3085 0.3089      0.00299        2310927    101
k_factor_rf=2,nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8       0.2773 0.3095 0.3100      0.00316        2309895    95
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.3383 0.3805 0.3809      0.00272        2527401    111
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3516 0.3967 0.3972      0.00325        3215982    93
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.4304 0.4977 0.4981      0.00380        7578757    79
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4617 0.5364 0.5369      0.00423        5005431    71
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.4626 0.5307 0.5311      0.00497        8952795    61
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.4693 0.5458 0.5463      0.00477        6327824    63
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5433 0.6420 0.6425      0.00735        8575385    41
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5652 0.6550 0.6553      0.00645       19632686    47
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5805 0.6706 0.6708      0.00664       31142553    46
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.6437 0.7668 0.7672      0.00956       33705084    32
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.6662 0.8109 0.8114      0.01088       31141071    28
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6906 0.8417 0.8421      0.01444       59032934    21
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.7343 0.9142 0.9149      0.02663      116123287    12
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7351 0.9219 0.9229      0.03511      117669648    9
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7536 0.9500 0.9510      0.04166      120797382    8
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.7547 0.9522 0.9532      0.06205      131145809    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7611 0.9668 0.9679      0.06761      223140111    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.7676 0.9763 0.9775      0.08929      237374630    4
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7678 0.9770 0.9782      0.11924      847634625    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7683 0.9777 0.9788      0.14122      846643278    3
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.7699 0.9814 0.9825      0.14442      227109175    3
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.7753 0.9908 0.9923      0.18926     1672252958    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.7755 0.9904 0.9918      0.23615      464856136    2
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.7762 0.9919 0.9932      0.58014      846622590    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 809458514

 code_size 62

 log filename: autotune.dbdeep10M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR72_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.2690 0.3090 0.3096      0.00630       21902823    48
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1629 0.1823 0.1826      0.00157        1173800    191
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2        0.1687 0.1877 0.1879      0.00163         818671    185
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4        0.1806 0.2019 0.2021      0.00163        1169668    185
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4       0.1824 0.2041 0.2044      0.00180        1167942    167
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1        0.1979 0.2244 0.2250      0.00190        1088166    158
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1        0.2024 0.2308 0.2314      0.00196        1086580    154
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.2055 0.2351 0.2357      0.00195        1085025    154
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2417 0.2772 0.2778      0.00198        1268148    152
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2623 0.3014 0.3017      0.00213        2312963    141
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.3240 0.3804 0.3809      0.00219        2527269    138
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3363 0.3966 0.3970      0.00305        3214650    99
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.4029 0.4973 0.4981      0.00393        7579209    77
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4367 0.5363 0.5367      0.00414        5005480    73
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.4370 0.5305 0.5311      0.00418        8954475    72
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.4440 0.5457 0.5461      0.00436        6328050    69
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.4816 0.6028 0.6036      0.00483       12538042    63
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.4977 0.6414 0.6424      0.00489        8574988    62
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5171 0.6673 0.6682      0.00586       17152489    52
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5261 0.6700 0.6708      0.00637       31140701    48
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5622 0.7521 0.7533      0.00906       19639794    34
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5759 0.7661 0.7671      0.00903       33700472    34
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5917 0.8095 0.8113      0.01013       31138824    30
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.6042 0.8281 0.8298      0.01409       62125065    22
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6109 0.8405 0.8420      0.01358       59037121    23
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.6177 0.8478 0.8493      0.02565      153450657    12
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.6263 0.8833 0.8864      0.02985       58847039    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6437 0.9117 0.9149      0.02573      116123287    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256    0.6471 0.9175 0.9208      0.03365      151942690    9
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6569 0.9469 0.9509      0.04022      120838413    8
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.6570 0.9473 0.9512      0.06074      120811765    5
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.6580 0.9491 0.9531      0.06449      131145809    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6620 0.9626 0.9678      0.06735      223140111    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.6668 0.9718 0.9774      0.08929      237374630    4
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6678 0.9723 0.9781      0.12105      847634625    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6681 0.9729 0.9788      0.17947      866365953    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.6723 0.9853 0.9922      0.20833     1672252958    2
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.6726 0.9864 0.9931      0.57708      846622590    1
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.6727 0.9859 0.9928      0.63503     1621482158    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 764026258

 code_size 64

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7730 0.9713 0.9723      0.05523      425659870    6
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.2300 0.2543 0.2549      0.00217        1780231    138
k_factor_rf=1,nprobe=2,quantizer_efSearch=4       0.3245 0.3618 0.3620      0.00225        3560404    134
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.3468 0.3873 0.3875      0.00236        3559849    128
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.4089 0.4602 0.4604      0.00242        7108997    125
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4970 0.5676 0.5678      0.00269       14214743    112
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.5019 0.5743 0.5745      0.00280       14201584    108
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.5340 0.6119 0.6122      0.00308       28336034    98
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5519 0.6330 0.6333      0.00337       28275195    89
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5594 0.6424 0.6427      0.00354       28214325    85
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5652 0.6486 0.6489      0.00391       56131483    77
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.5955 0.6991 0.6997      0.00466       28275195    65
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.6042 0.7104 0.7110      0.00480       28214325    63
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.6200 0.7295 0.7300      0.00496       56131483    61
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.6338 0.7467 0.7472      0.00541       55993559    56
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.6373 0.7505 0.7510      0.00598       55887690    51
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.6414 0.7546 0.7551      0.00625      110861999    49
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.6763 0.8120 0.8127      0.00784       55993559    39
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6910 0.8325 0.8332      0.00900      110861999    34
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6996 0.8431 0.8438      0.01145      110387491    27
k_factor_rf=8,nprobe=32,quantizer_efSearch=32     0.7006 0.8523 0.8529      0.01387       55887690    22
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.7018 0.8455 0.8462      0.01241      217368623    25
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.7288 0.8925 0.8932      0.01691      110387491    18
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.7386 0.9070 0.9078      0.01984      217368623    16
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.7425 0.9185 0.9193      0.02522      110387491    12
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.7575 0.9424 0.9434      0.03003      217368623    10
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7583 0.9437 0.9447      0.03166      217053102    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7640 0.9516 0.9525      0.04015      425516163    8
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7646 0.9520 0.9529      0.04176      425208019    8
k_factor_rf=32,nprobe=128,quantizer_efSearch=64   0.7652 0.9592 0.9603      0.04747      217368623    7
k_factor_rf=32,nprobe=128,quantizer_efSearch=128  0.7661 0.9607 0.9617      0.04890      217053102    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.7730 0.9713 0.9723      0.05733      425659870    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.7749 0.9742 0.9752      0.06110      425516163    5
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7755 0.9747 0.9757      0.06630      425208019    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.7764 0.9746 0.9757      0.06193      818247453    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7783 0.9784 0.9794      0.07736      829473327    4
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.7786 0.9787 0.9797      0.09291      828656919    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=64   0.7796 0.9845 0.9859      0.10561      818247453    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.7819 0.9885 0.9899      0.11296      826805002    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.7824 0.9897 0.9909      0.12574     1555713815    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7831 0.9906 0.9920      0.14000     1604349000    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7836 0.9909 0.9923      0.17751     1605955182    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.7838 0.9906 0.9919      0.26674     5383556907    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCA72,SQ6)` </summary>
Index size 743932430

 code_size 62

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCA72_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6646 0.9669 0.9721      0.05818      425703209    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3774 0.4575 0.4579      0.00195        7102840    155
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.4071 0.4952 0.4956      0.00211        7120082    143
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4559 0.5664 0.5669      0.00230       14213655    131
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4604 0.5727 0.5732      0.00241       14200728    125
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4855 0.6107 0.6113      0.00274       28334500    110
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5099 0.6422 0.6429      0.00307       28214734    98
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5120 0.6477 0.6483      0.00339       56131096    89
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.5383 0.6988 0.7000      0.00390       28272129    77
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5461 0.7102 0.7115      0.00414       28214734    73
k_factor_rf=2,nprobe=16,quantizer_efSearch=32     0.5470 0.7125 0.7137      0.00471       28172763    64
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5563 0.7284 0.7295      0.00454       56131096    67
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5672 0.7457 0.7469      0.00478       55991326    63
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.5689 0.7495 0.7507      0.00525       55885345    58
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.5717 0.7540 0.7551      0.00584      110857292    52
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.6003 0.8106 0.8125      0.00694       55991326    44
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.6022 0.8146 0.8166      0.00838       55885345    36
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6091 0.8305 0.8330      0.00881      110857292    35
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6157 0.8410 0.8435      0.00987      110385190    31
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.6174 0.8441 0.8460      0.01150      217360297    27
k_factor_rf=8,nprobe=64,quantizer_efSearch=16     0.6282 0.8784 0.8817      0.01446      110857292    21
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.6349 0.8894 0.8929      0.01620      110385190    19
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6432 0.9038 0.9076      0.01828      217360297    17
k_factor_rf=16,nprobe=128,quantizer_efSearch=16   0.6438 0.9172 0.9217      0.02729      218271505    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6554 0.9384 0.9432      0.03034      217360297    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.6556 0.9395 0.9444      0.03143      217052139    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6586 0.9450 0.9496      0.03710      425703209    9
k_factor_rf=16,nprobe=512,quantizer_efSearch=64   0.6603 0.9459 0.9503      0.04136      818280089    8
k_factor_rf=32,nprobe=256,quantizer_efSearch=32   0.6604 0.9583 0.9635      0.05596      424819860    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.6646 0.9669 0.9721      0.05424      425703209    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6647 0.9694 0.9749      0.05869      425516841    6
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.6673 0.9705 0.9758      0.06341      818280089    5
k_factor_rf=32,nprobe=2048,quantizer_efSearch=128 0.6676 0.9724 0.9778      0.08744     2374191023    4
k_factor_rf=64,nprobe=256,quantizer_efSearch=64   0.6678 0.9743 0.9800      0.09610      425703209    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.6710 0.9839 0.9900      0.11871      826853344    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.6715 0.9853 0.9921      0.14448     1604393456    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.6716 0.9856 0.9924      0.18446     1605964047    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.6719 0.9851 0.9920      0.25840     5383142802    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 743935758

 code_size 62

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCAR72_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.6658 0.9656 0.9717      0.05271      425594861    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3797 0.4589 0.4591      0.00190        7105940    158
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.4090 0.4963 0.4965      0.00209        7118171    144
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4594 0.5680 0.5683      0.00227       14211823    133
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4639 0.5740 0.5743      0.00231       14201010    130
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4879 0.6128 0.6131      0.00260       28338006    116
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5054 0.6343 0.6346      0.00272       28272565    111
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.5126 0.6432 0.6436      0.00294       28213401    102
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.5152 0.6489 0.6493      0.00344       56127082    88
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.5409 0.7001 0.7010      0.00380       28272565    79
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5486 0.7113 0.7122      0.00396       28213401    76
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5573 0.7295 0.7305      0.00441       56127082    69
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5674 0.7457 0.7467      0.00469       55989980    64
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.5702 0.7499 0.7508      0.00512       55885747    59
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.5725 0.7540 0.7550      0.00575      110856303    53
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.6009 0.8108 0.8126      0.00757       55989980    40
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.6038 0.8149 0.8168      0.00786       55885747    39
k_factor_rf=4,nprobe=32,quantizer_efSearch=64     0.6048 0.8167 0.8186      0.00822       55828114    37
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6153 0.8413 0.8435      0.01079      110385564    28
k_factor_rf=8,nprobe=32,quantizer_efSearch=32     0.6170 0.8499 0.8527      0.01232       55885747    25
k_factor_rf=8,nprobe=32,quantizer_efSearch=64     0.6178 0.8517 0.8544      0.01347       55828114    23
k_factor_rf=8,nprobe=64,quantizer_efSearch=16     0.6279 0.8780 0.8817      0.01423      110856303    22
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.6338 0.8864 0.8898      0.01492      110567209    21
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.6360 0.8893 0.8928      0.01625      110385564    19
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6436 0.9032 0.9073      0.01845      217362337    17
k_factor_rf=16,nprobe=64,quantizer_efSearch=128   0.6439 0.9154 0.9199      0.02627      110318015    12
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6550 0.9376 0.9429      0.02925      217362337    11
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6588 0.9441 0.9493      0.03283      425594861    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6593 0.9465 0.9520      0.03489      425514199    9
k_factor_rf=16,nprobe=512,quantizer_efSearch=64   0.6600 0.9451 0.9500      0.03829      818167346    8
k_factor_rf=32,nprobe=128,quantizer_efSearch=64   0.6603 0.9536 0.9597      0.04498      217362337    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=32   0.6623 0.9570 0.9632      0.05181      424663298    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.6658 0.9656 0.9717      0.05215      425594861    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6667 0.9681 0.9746      0.05373      425514199    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.6670 0.9687 0.9752      0.05843      425206399    6
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.6681 0.9691 0.9754      0.06376      818167346    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6695 0.9725 0.9792      0.07503      829479335    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.6697 0.9727 0.9795      0.08963      828657636    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.6724 0.9826 0.9896      0.11092      826766699    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.6727 0.9836 0.9907      0.12270     1555584883    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.6733 0.9847 0.9919      0.14006     1604410926    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.6735 0.9850 0.9922      0.16598     1605973092    2
k_factor_rf=64,nprobe=2048,quantizer_efSearch=256 0.6736 0.9841 0.9913      0.17600     2916914963    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.6745 0.9846 0.9918      0.27002     5382953926    2
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 747896278

 code_size 64

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.4695 0.5425 0.5434      0.03998       17471765    8
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.4213 0.4769 0.4771      0.00297        7494408    102
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.4305 0.4893 0.4895      0.00309       14517829    98
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.4381 0.4985 0.4987      0.00314       14476021    96
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.4837 0.5540 0.5542      0.00313       14588554    96
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4898 0.5577 0.5580      0.00345       29056140    87
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.5116 0.5838 0.5841      0.00397       57465089    76
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.5134 0.5885 0.5887      0.00342       28815465    88
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.5887 0.6885 0.6891      0.00451       29027358    67
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.6239 0.7315 0.7321      0.00665      111819441    46
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.6747 0.8112 0.8119      0.01062      113593840    29
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64     0.6786 0.8172 0.8179      0.01170      117113013    26
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6942 0.8352 0.8359      0.01262      219335751    24
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.6973 0.8477 0.8483      0.01477       57277237    21
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.7008 0.8523 0.8530      0.01465       58519853    21
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.7014 0.8530 0.8536      0.01608       61088972    19
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.7222 0.8840 0.8848      0.01658      111990189    19
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.7331 0.9063 0.9071      0.02425      112221798    13
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7405 0.9075 0.9083      0.02720      428775274    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7424 0.9105 0.9113      0.02755      435953191    11
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.7582 0.9436 0.9445      0.03653      227267502    9
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7642 0.9581 0.9592      0.05334      227708101    6
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7753 0.9746 0.9755      0.05777      430422485    6
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7772 0.9760 0.9771      0.06641      854500821    5
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.7790 0.9782 0.9792      0.13370     3098265170    3
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.7834 0.9903 0.9915      0.20299     6101484420    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=64 0.7837 0.9908 0.9922      0.23301     1612369798    2
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64 0.7841 0.9909 0.9922      0.30292     3104345401    1
```

</details>
<details><summary>`OPQ16_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR72,SQ6)` </summary>
Index size 727804754

 code_size 62

 log filename: autotune.dbdeep10M.OPQ16_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR72_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.4346 0.5428 0.5434      0.04057       17512593    8
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.3274 0.3864 0.3865      0.00248        4913892    122
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4        0.3941 0.4769 0.4771      0.00243        7494361    124
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.4062 0.4983 0.4987      0.00273       14475964    110
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4519 0.5577 0.5580      0.00294       29056140    103
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4677 0.5837 0.5841      0.00382       57465089    79
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.4722 0.5885 0.5887      0.00295       28815361    102
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.5355 0.6882 0.6891      0.00420       29027377    72
k_factor_rf=4,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.5549 0.7297 0.7312      0.00677       29026618    45
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.5596 0.7310 0.7321      0.00658      111817132    46
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5899 0.7905 0.7924      0.00843       56811331    36
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5975 0.8098 0.8119      0.00955      113595496    32
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.6011 0.8159 0.8179      0.01092      117119534    28
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128     0.6061 0.8181 0.8200      0.01250       66202044    24
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=16     0.6101 0.8334 0.8357      0.01430      111990102    22
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.6139 0.8455 0.8483      0.01389       57277237    22
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.6156 0.8502 0.8530      0.01420       58523849    22
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64      0.6165 0.8508 0.8536      0.01542       61088972    20
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.6306 0.8812 0.8848      0.01884      111987606    16
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.6376 0.8922 0.8962      0.02528      219322567    12
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.6437 0.9044 0.9083      0.02574      428768082    12
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.6561 0.9387 0.9445      0.03400      227267502    9
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6601 0.9526 0.9592      0.04968      227708101    7
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6670 0.9687 0.9755      0.06068      430416734    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.6690 0.9702 0.9771      0.06377      854434135    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=64,quantizer_nprobe=64   0.6699 0.9727 0.9798      0.15431      834350823    2
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6713 0.9725 0.9792      0.13024     3098265170    3
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.6745 0.9840 0.9915      0.20738     6101484420    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.6750 0.9848 0.9922      0.99391     5950322548    1
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 749412850

 code_size 56

 log filename: autotune.dbdeep10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR64_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.2573 0.3094 0.3099      0.00616       21911339    49
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.1601 0.1856 0.1861      0.00149        1170656    202
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.1743 0.2028 0.2032      0.00165        1170697    182
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4      0.1759 0.2043 0.2048      0.00179        1168687    168
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1       0.1967 0.2356 0.2361      0.00189        1083530    159
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2      0.2324 0.2792 0.2797      0.00200        1270972    151
k_factor_rf=2,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.2325 0.2797 0.2802      0.00205        1271916    147
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.2517 0.3032 0.3036      0.00209        2312658    144
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3049 0.3803 0.3809      0.00214        2529678    140
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.3161 0.3997 0.4003      0.00285        3216337    106
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.3706 0.4873 0.4883      0.00329        7583203    92
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.4004 0.5387 0.5402      0.00372        5004473    81
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.4014 0.5298 0.5307      0.00410        8958776    74
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.4059 0.5467 0.5481      0.00423        6325724    71
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4385 0.5921 0.5936      0.00460       12548486    66
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4551 0.6364 0.6390      0.00479        8576389    63
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4624 0.6547 0.6574      0.00553       17170447    55
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.4639 0.6508 0.6533      0.00601        9884627    50
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4720 0.6531 0.6549      0.00616       31157810    49
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5046 0.7414 0.7464      0.00878       19641834    35
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5119 0.7485 0.7527      0.00866       33721777    35
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5120 0.7569 0.7621      0.00928       19603100    33
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.5126 0.7578 0.7628      0.01064       24829542    29
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.5265 0.7929 0.7999      0.00982       31158425    31
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5360 0.8187 0.8251      0.01299       59049121    24
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5385 0.8360 0.8467      0.02158       31155467    14
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256   0.5415 0.8253 0.8319      0.02474      153584093    13
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5632 0.8974 0.9119      0.02706       61510577    12
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5674 0.9058 0.9207      0.03267      117799283    10
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5751 0.9298 0.9465      0.04153      120862392    8
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5781 0.9436 0.9637      0.06316      223194148    5
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5818 0.9525 0.9734      0.08829      237390677    4
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5820 0.9526 0.9726      0.11807      847952503    3
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.5822 0.9593 0.9816      0.15148      227246496    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5843 0.9661 0.9888      0.19854     1675509006    2
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 860942038

 code_size 64

 log filename: autotune.dbdeep10M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fs_Refine_OPQ48_96_PQ48_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.2798 0.3097 0.3106      0.00676       21938081    45
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2        0.1635 0.1769 0.1772      0.00196         820970    153
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1721 0.1855 0.1859      0.00191        1171180    157
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2        0.1740 0.1884 0.1887      0.00198         820962    152
k_factor_rf=4,nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4        0.1867 0.2027 0.2031      0.00201        1170952    149
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1        0.2039 0.2260 0.2268      0.00222        1086465    136
k_factor_rf=2,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4        0.2169 0.2388 0.2393      0.00236        1628589    128
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2510 0.2795 0.2804      0.00241        1271008    125
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2735 0.3029 0.3035      0.00264        2314182    114
k_factor_rf=4,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8       0.2814 0.3112 0.3118      0.00273        2310485    110
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.3398 0.3847 0.3856      0.00263        2529222    115
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3536 0.3999 0.4005      0.00342        3217415    88
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.3664 0.4312 0.4322      0.00361        7377642    84
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4        0.3889 0.4474 0.4479      0.00385        4358145    78
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4659 0.5431 0.5438      0.00412        5001445    73
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.4725 0.5511 0.5517      0.00466        6322715    65
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.4737 0.5515 0.5524      0.00457        8948282    66
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32       0.4774 0.5548 0.5557      0.00505        8957207    60
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5510 0.6596 0.6604      0.00550        8574207    55
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.6056 0.7405 0.7416      0.00593       29381129    51
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.6099 0.7444 0.7455      0.00679       29302567    45
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.6216 0.7537 0.7548      0.00648       19638141    47
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.6755 0.8309 0.8320      0.00733       31161330    41
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.6868 0.8523 0.8530      0.00998       33721777    31
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64      0.6912 0.8496 0.8507      0.01072       38822798    28
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.7159 0.9037 0.9047      0.01510       59052821    20
k_factor_rf=2,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.7388 0.9336 0.9346      0.02391      153584093    13
k_factor_rf=1,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.7453 0.9353 0.9364      0.02998      261889480    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.7476 0.9566 0.9577      0.02826      116114873    11
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.7614 0.9783 0.9794      0.03268      227784727    10
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256    0.7631 0.9808 0.9818      0.06308      257872509    5
k_factor_rf=2,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7640 0.9776 0.9786      0.08253      847952503    4
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7653 0.9902 0.9913      0.08316      435104564    4
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7671 0.9926 0.9938      0.08868      444671724    4
k_factor_rf=4,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7684 0.9914 0.9925      0.11149      846828569    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7686 0.9964 0.9980      0.15163      846828569    2
k_factor_rf=8,nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7693 0.9957 0.9967      0.18652     1621931230    2
k_factor_rf=32,nprobe=4096,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.7694 0.9975 0.9991      0.49818     1621914210    1
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ16x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 683979922

 code_size 56

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_HNSW32_PQ16x4fs_Refine_OPQ48_96_PQ48_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7507 0.9644 0.9659      0.05348      425864650    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3928 0.4474 0.4476      0.00227        7104752    132
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.4249 0.4847 0.4849      0.00236        7115617    127
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4794 0.5501 0.5504      0.00264       14214052    114
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4852 0.5578 0.5581      0.00272       14195875    111
k_factor_rf=1,nprobe=8,quantizer_efSearch=16      0.4994 0.5737 0.5740      0.00303       14166677    99
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.5115 0.5883 0.5887      0.00296       28332970    102
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.5315 0.6124 0.6128      0.00308       28271412    98
k_factor_rf=2,nprobe=8,quantizer_efSearch=16      0.5353 0.6293 0.6297      0.00390       14166677    77
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.5822 0.6888 0.6892      0.00418       28271412    72
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5901 0.6983 0.6987      0.00441       28216692    69
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.6018 0.7124 0.7129      0.00477       56137307    63
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.6159 0.7304 0.7309      0.00517       55997550    59
k_factor_rf=4,nprobe=16,quantizer_efSearch=32     0.6213 0.7520 0.7522      0.00709       28173577    43
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.6545 0.7968 0.7973      0.00727       55997550    42
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.6591 0.8022 0.8027      0.00806       55883260    38
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6677 0.8140 0.8145      0.00837      110893598    36
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6744 0.8242 0.8247      0.00980      110415496    31
k_factor_rf=8,nprobe=32,quantizer_efSearch=16     0.6772 0.8413 0.8419      0.01236       55997550    25
k_factor_rf=8,nprobe=32,quantizer_efSearch=32     0.6813 0.8468 0.8474      0.01276       55883260    24
k_factor_rf=8,nprobe=64,quantizer_efSearch=16     0.6984 0.8696 0.8700      0.01472      110893598    21
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.7052 0.8794 0.8798      0.01494      110594681    21
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.7062 0.8814 0.8819      0.01579      110415496    19
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.7126 0.8889 0.8892      0.01904      217420216    16
k_factor_rf=16,nprobe=64,quantizer_efSearch=32    0.7191 0.9098 0.9108      0.02321      110594681    13
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.7198 0.9116 0.9126      0.02384      110415496    13
k_factor_rf=16,nprobe=64,quantizer_efSearch=128   0.7213 0.9130 0.9140      0.02622      110341824    12
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.7368 0.9344 0.9354      0.02776      217420216    11
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.7398 0.9390 0.9397      0.03300      425864650    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7416 0.9420 0.9428      0.03780      425558278    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7422 0.9427 0.9433      0.04381      425271115    7
k_factor_rf=32,nprobe=128,quantizer_efSearch=64   0.7439 0.9532 0.9548      0.04735      217420216    7
k_factor_rf=32,nprobe=128,quantizer_efSearch=128  0.7451 0.9551 0.9567      0.05039      217103121    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.7507 0.9644 0.9659      0.05424      425864650    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.7527 0.9674 0.9688      0.05857      425558278    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7531 0.9682 0.9696      0.06763      425271115    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7557 0.9722 0.9735      0.07727      829633637    4
k_factor_rf=64,nprobe=256,quantizer_efSearch=128  0.7572 0.9787 0.9806      0.09610      425558278    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=64   0.7595 0.9821 0.9842      0.11071      818456657    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.7620 0.9859 0.9880      0.11745      826971745    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.7624 0.9856 0.9876      0.12065     1555695138    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7633 0.9869 0.9888      0.14164     1604556367    3
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCA64,SQ6)` </summary>
Index size 683910574

 code_size 56

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCA64_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.5747 0.9484 0.9662      0.05421      425853908    6
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.3019 0.3836 0.3840      0.00188        3559818    160
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3401 0.4461 0.4466      0.00183        7115340    164
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3686 0.4853 0.4858      0.00207        7120768    145
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4106 0.5589 0.5597      0.00232       14203104    130
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4272 0.5891 0.5900      0.00267       28346193    113
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.4495 0.6199 0.6211      0.00306       28223070    99
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.4789 0.6872 0.6894      0.00394       28277550    77
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.4857 0.6963 0.6989      0.00420       28223070    72
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.4879 0.7111 0.7134      0.00450       56147036    67
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.4975 0.7285 0.7310      0.00476       56007359    64
k_factor_rf=4,nprobe=16,quantizer_efSearch=32     0.5032 0.7483 0.7523      0.00699       28178950    43
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.5238 0.7922 0.7975      0.00747       56007359    41
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.5259 0.7970 0.8026      0.00807       55893227    38
k_factor_rf=4,nprobe=32,quantizer_efSearch=64     0.5260 0.7976 0.8032      0.00921       55838505    33
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.5267 0.8050 0.8099      0.01018      218358898    30
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.5354 0.8183 0.8243      0.01005      110420379    30
k_factor_rf=8,nprobe=32,quantizer_efSearch=16     0.5359 0.8339 0.8419      0.01217       56007359    25
k_factor_rf=8,nprobe=32,quantizer_efSearch=32     0.5378 0.8389 0.8473      0.01290       55893227    24
k_factor_rf=8,nprobe=32,quantizer_efSearch=64     0.5381 0.8396 0.8480      0.01343       55838505    23
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.5525 0.8706 0.8797      0.01517      110600269    20
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.5541 0.8726 0.8820      0.01521      110420379    20
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.5558 0.8794 0.8895      0.01895      217427541    16
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.5600 0.9004 0.9125      0.02512      110420379    12
k_factor_rf=16,nprobe=64,quantizer_efSearch=128   0.5609 0.9016 0.9139      0.02732      110346860    12
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.5688 0.9214 0.9356      0.02828      217427541    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.5696 0.9225 0.9372      0.03137      217111166    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.5707 0.9269 0.9402      0.03439      425853908    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.5713 0.9290 0.9428      0.03633      425574937    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.5716 0.9293 0.9432      0.04065      425284074    8
k_factor_rf=32,nprobe=128,quantizer_efSearch=64   0.5723 0.9383 0.9550      0.04618      217427541    7
k_factor_rf=32,nprobe=128,quantizer_efSearch=128  0.5732 0.9395 0.9566      0.04704      217111166    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.5747 0.9484 0.9662      0.05360      425853908    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.5752 0.9503 0.9687      0.05596      425574937    6
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.5766 0.9518 0.9694      0.06565      818409567    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.5769 0.9545 0.9734      0.07625      829647614    4
k_factor_rf=64,nprobe=256,quantizer_efSearch=128  0.5784 0.9603 0.9804      0.09488      425574937    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.5802 0.9668 0.9878      0.10911      826960401    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.5808 0.9665 0.9875      0.12921     1555504010    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.5810 0.9674 0.9885      0.14393     1604491523    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.5814 0.9677 0.9888      0.17049     1606098458    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 683917230

 code_size 56

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_HNSW32_PQ16x4fs_Refine_PCAR64_SQ6_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.5775 0.9485 0.9661      0.05376      425795013    6
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3400 0.4467 0.4469      0.00189        7101322    159
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3672 0.4858 0.4860      0.00211        7116762    143
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.4045 0.5509 0.5516      0.00221       14209669    137
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4087 0.5591 0.5598      0.00257       14195498    117
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.4272 0.5890 0.5900      0.00260       28332030    116
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.4426 0.6122 0.6133      0.00276       28271652    109
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.4481 0.6196 0.6208      0.00295       28217098    102
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.4784 0.6873 0.6897      0.00385       28271652    78
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.4839 0.6960 0.6985      0.00399       28217098    76
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.4902 0.7115 0.7144      0.00458       56133605    66
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.4987 0.7280 0.7311      0.00471       55991902    64
k_factor_rf=4,nprobe=16,quantizer_efSearch=32     0.5050 0.7480 0.7523      0.00696       28172739    44
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.5248 0.7926 0.7975      0.00698       55991902    43
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.5275 0.7978 0.8028      0.00737       55877217    41
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.5319 0.8086 0.8145      0.00857      110887904    36
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.5368 0.8183 0.8246      0.00941      110404840    32
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.5370 0.8153 0.8215      0.01101      218013821    28
k_factor_rf=8,nprobe=32,quantizer_efSearch=16     0.5378 0.8346 0.8420      0.01205       55991902    25
k_factor_rf=8,nprobe=32,quantizer_efSearch=32     0.5403 0.8398 0.8474      0.01266       55877217    24
k_factor_rf=8,nprobe=64,quantizer_efSearch=16     0.5492 0.8613 0.8701      0.01402      110887904    22
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.5540 0.8704 0.8798      0.01444      110585399    21
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.5544 0.8726 0.8819      0.01552      110404840    20
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.5575 0.8792 0.8894      0.01839      217400372    17
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.5610 0.9001 0.9125      0.02478      110404840    13
k_factor_rf=16,nprobe=128,quantizer_efSearch=16   0.5616 0.9000 0.9134      0.02728      218309209    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.5709 0.9209 0.9355      0.02806      217400372    11
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.5710 0.9218 0.9371      0.03188      217083480    10
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.5721 0.9255 0.9401      0.03346      425795013    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.5722 0.9278 0.9427      0.03686      425532278    9
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.5726 0.9282 0.9431      0.04170      425248538    8
k_factor_rf=32,nprobe=128,quantizer_efSearch=128  0.5746 0.9392 0.9565      0.04813      217083480    7
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.5775 0.9485 0.9661      0.05309      425795013    6
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.5777 0.9506 0.9687      0.05802      425532278    6
k_factor_rf=32,nprobe=512,quantizer_efSearch=64   0.5792 0.9517 0.9694      0.06001      818561616    5
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.5797 0.9543 0.9733      0.07531      829614390    4
k_factor_rf=64,nprobe=512,quantizer_efSearch=64   0.5826 0.9642 0.9844      0.10580      818561616    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.5829 0.9670 0.9880      0.10775      826905319    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.5835 0.9667 0.9876      0.12207     1555874901    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.5840 0.9674 0.9886      0.14157     1604603184    3
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.5841 0.9674 0.9889      0.17282     1606162365    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.5845 0.9671 0.9879      0.26186     5382578671    2
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 772136338

 code_size 64

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_HNSW32_PQ32x4fs_Refine_OPQ48_96_PQ48_.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7616 0.9851 0.9862      0.03363     1535416636    9
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.4174 0.4885 0.4889      0.00230        7111784    131
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.4507 0.5275 0.5279      0.00245        7116652    123
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.5352 0.6411 0.6415      0.00267       14213358    113
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.5407 0.6493 0.6497      0.00276       14197706    109
k_factor_rf=1,nprobe=8,quantizer_efSearch=16      0.5544 0.6659 0.6663      0.00309       14167480    97
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.5989 0.7302 0.7307      0.00321       28335519    94
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.6210 0.7594 0.7599      0.00350       28273931    86
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.6305 0.7710 0.7715      0.00365       28218150    83
k_factor_rf=1,nprobe=32,quantizer_efSearch=4      0.6375 0.7859 0.7864      0.00375       56199667    80
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.6663 0.8265 0.8271      0.00400       56132272    76
k_factor_rf=1,nprobe=32,quantizer_efSearch=32     0.6853 0.8510 0.8516      0.00466       55884313    65
k_factor_rf=1,nprobe=64,quantizer_efSearch=8      0.6890 0.8592 0.8598      0.00498      111036116    61
k_factor_rf=1,nprobe=64,quantizer_efSearch=16     0.7103 0.8895 0.8902      0.00524      110898646    58
k_factor_rf=1,nprobe=64,quantizer_efSearch=32     0.7170 0.8989 0.8995      0.00581      110594946    52
k_factor_rf=1,nprobe=128,quantizer_efSearch=64    0.7381 0.9307 0.9312      0.00873      217418085    35
k_factor_rf=2,nprobe=128,quantizer_efSearch=32    0.7413 0.9461 0.9470      0.00897      218027545    34
k_factor_rf=2,nprobe=128,quantizer_efSearch=64    0.7459 0.9522 0.9531      0.01097      217418085    28
k_factor_rf=2,nprobe=128,quantizer_efSearch=128   0.7468 0.9537 0.9546      0.01305      217102358    23
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.7481 0.9609 0.9620      0.01248      217418085    25
k_factor_rf=2,nprobe=256,quantizer_efSearch=64    0.7537 0.9662 0.9671      0.01364      425805547    22
k_factor_rf=4,nprobe=256,quantizer_efSearch=64    0.7568 0.9767 0.9779      0.01914      425805547    16
k_factor_rf=4,nprobe=512,quantizer_efSearch=64    0.7612 0.9842 0.9854      0.02288      818484577    14
k_factor_rf=8,nprobe=512,quantizer_efSearch=64    0.7627 0.9878 0.9898      0.03467      818484577    9
k_factor_rf=4,nprobe=1024,quantizer_efSearch=128  0.7646 0.9899 0.9911      0.04641     1555706065    7
k_factor_rf=8,nprobe=512,quantizer_efSearch=256   0.7653 0.9923 0.9943      0.04756      829636504    7
k_factor_rf=8,nprobe=1024,quantizer_efSearch=128  0.7660 0.9936 0.9955      0.04814     1555706065    7
k_factor_rf=8,nprobe=1024,quantizer_efSearch=256  0.7665 0.9947 0.9967      0.06471     1604565583    5
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.7668 0.9957 0.9980      0.08102     1604565583    4
k_factor_rf=16,nprobe=2048,quantizer_efSearch=512 0.7673 0.9969 0.9992      0.15599     3081955265    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.7674 0.9974 0.9996      0.30149     5382069093    2
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ16x4fs,Refine(PCAR64,SQ6)` </summary>
Index size 667790834

 code_size 56

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ16x4fs_Refine_PCAR64_SQ6_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.3944 0.5410 0.5428      0.02646       17516388    12
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4       0.2189 0.2677 0.2680      0.00206        2135051    146
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16       0.2612 0.3295 0.3297      0.00216        4929229    139
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.3040 0.3870 0.3872      0.00231        4908970    130
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4        0.3146 0.4114 0.4116      0.00226        7546161    133
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4        0.3543 0.4684 0.4686      0.00228        7490231    132
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.3585 0.4802 0.4808      0.00266       14475643    113
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4       0.3936 0.5379 0.5386      0.00273       14590224    110
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.3959 0.5354 0.5361      0.00284       29061307    106
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4112 0.5607 0.5614      0.00346       57541065    87
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.4143 0.5665 0.5675      0.00296       28809177    102
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.4705 0.6708 0.6731      0.00395       29030123    77
k_factor_rf=4,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.4895 0.7185 0.7228      0.00594       29030414    51
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5222 0.7869 0.7927      0.00867      113695784    35
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.5267 0.7938 0.8002      0.01078      117234143    28
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128     0.5284 0.7990 0.8042      0.01164       66203991    26
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.5316 0.8079 0.8139      0.01467      219352096    21
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.5372 0.8337 0.8411      0.01269       57280341    24
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.5411 0.8404 0.8480      0.01307       58524251    23
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64      0.5413 0.8405 0.8480      0.01411       61096133    22
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.5485 0.8631 0.8720      0.01699      111996391    18
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=16     0.5486 0.8632 0.8721      0.02476      111993440    13
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5578 0.8781 0.8881      0.02735      428809630    11
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128    0.5583 0.8786 0.8891      0.02474      436048932    13
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5711 0.9219 0.9375      0.03365      227285435    9
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.5740 0.9376 0.9548      0.05090      227776058    6
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5777 0.9507 0.9692      0.05711      430449711    6
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.5783 0.9507 0.9689      0.06311      856358944    5
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5797 0.9509 0.9695      0.12844     3098711499    3
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5844 0.9667 0.9875      0.20361     6117745978    2
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64  0.5845 0.9671 0.9881      0.30377     3104862009    1
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128 0.5847 0.9671 0.9881      0.98133     5951372017    1
```

</details>
<details><summary>`OPQ32_64,IVF65536(IVF256,PQ32x4fs,RFlat),PQ32x4fs,Refine(OPQ48_96,PQ48)` </summary>
Index size 756025558

 code_size 64

 log filename: autotune.dbdeep10M.OPQ32_64_IVF65536_IVF256_PQ32x4fs_RFlat__PQ32x4fs_Refine_OPQ48_96_PQ48_.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                        0.7513 0.9512 0.9517      0.02111      835436456    15
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.4379 0.5101 0.5106      0.00282        7490101    107
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.4669 0.5520 0.5525      0.00298       14523137    101
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.4732 0.5635 0.5640      0.00305       14475587    99
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.5233 0.6274 0.6279      0.00313       14590520    96
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.5525 0.6620 0.6625      0.00331       29061191    91
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.5841 0.7074 0.7079      0.00348       28809036    87
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.6049 0.7380 0.7385      0.00401       57541341    75
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6251 0.7598 0.7603      0.00464       30903110    65
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.6655 0.8238 0.8243      0.00734       56815715    41
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.6856 0.8520 0.8526      0.00688       58496310    44
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7189 0.9037 0.9043      0.00741      223235240    41
k_factor_rf=1,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7360 0.9278 0.9283      0.00913      220392999    33
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.7393 0.9478 0.9489      0.01672      219353801    18
k_factor_rf=1,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7444 0.9423 0.9428      0.01808      428809630    17
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7475 0.9605 0.9615      0.01656      222708940    19
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.7488 0.9627 0.9638      0.02056      227263412    15
k_factor_rf=1,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7513 0.9512 0.9517      0.01889      835448025    16
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.7579 0.9804 0.9823      0.02647      428795594    12
k_factor_rf=2,nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7610 0.9781 0.9789      0.02825     1647959491    11
k_factor_rf=4,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7643 0.9887 0.9899      0.03047      835448025    10
k_factor_rf=4,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7652 0.9914 0.9926      0.04533     1616016355    7
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7666 0.9951 0.9970      0.04770     1616063349    7
k_factor_rf=8,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7674 0.9959 0.9978      0.07255     3100704261    5
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.7676 0.9976 0.9997      0.15636     3098711499    2
```

</details>
<details><summary>`OPQ56_112,IVF262144(IVF512,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 839713824

 code_size 63

 log filename: autotune.dbdeep10M.OPQ56_112_IVF262144_IVF512_PQ56x4fs_RFlat__PQ7+56.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                           0.4047 0.4458 0.4461      0.01615       10695553    19
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=56,k_factor=2       0.2013 0.2139 0.2141      0.00381         722144    79
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=1       0.2074 0.2202 0.2204      0.00403        1422289    75
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=1      0.2093 0.2221 0.2223      0.00442        2770237    68
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=56,k_factor=2     0.2096 0.2227 0.2229      0.00459        2772778    66
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1       0.2596 0.2806 0.2810      0.00457        1422031    66
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=56,k_factor=8       0.2836 0.3073 0.3077      0.00506         722256    60
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=56,k_factor=1       0.3835 0.4205 0.4209      0.00508         721435    60
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=1       0.3865 0.4258 0.4262      0.00530        1422268    57
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=56,k_factor=1      0.5177 0.5891 0.5895      0.00615         722115    49
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1     0.5950 0.6864 0.6868      0.00753        2773398    40
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=1     0.6319 0.7374 0.7378      0.00982        5418083    31
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=2      0.6514 0.7679 0.7682      0.01197        1422941    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=1      0.6711 0.7982 0.7986      0.01303        1422836    24
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=1     0.6852 0.8121 0.8125      0.01512       10678375    20
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=1     0.7034 0.8381 0.8385      0.01862        5413963    17
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1    0.7125 0.8563 0.8567      0.02309        2777876    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=4     0.7163 0.8671 0.8674      0.02220        2777876    14
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=4     0.7228 0.8756 0.8758      0.02360       10671118    13
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=2    0.7358 0.8952 0.8954      0.02577        2777876    12
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=2    0.7509 0.9142 0.9145      0.02838       10695553    11
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=4    0.7567 0.9294 0.9297      0.04152        5402804    8
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=2    0.7576 0.9238 0.9242      0.04377        5399200    7
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=56,k_factor=4   0.7691 0.9502 0.9505      0.05559       41465620    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=4   0.7754 0.9608 0.9612      0.05817       21056748    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=8   0.7768 0.9674 0.9677      0.06775       41207773    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=8   0.7779 0.9684 0.9687      0.07297       41465620    5
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=4   0.7794 0.9677 0.9681      0.09168       20890672    4
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=56,k_factor=8   0.7823 0.9777 0.9780      0.10606       41465620    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=32  0.7876 0.9884 0.9889      0.18602       41465620    2
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=16 0.7898 0.9922 0.9930      0.19505       21056748    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32  0.7914 0.9956 0.9965      0.38399       10695553    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=32 0.7922 0.9974 0.9983      0.43286       21056748    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=64 0.7925 0.9979 0.9990      0.76427       21056748    1
```

</details>
<details><summary>`OPQ56_112,IVF65536_HNSW32,PQ7+56` </summary>
Index size 757994588

 code_size 63

 log filename: autotune.dbdeep10M.OPQ56_112_IVF65536_HNSW32_PQ7+56.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.8221 0.9640 0.9643      0.06701              0    5
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.2687 0.2853 0.2856      0.00467              0    65
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.4904 0.5346 0.5350      0.00494              0    61
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.5764 0.6348 0.6352      0.00511              0    59
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.6419 0.7149 0.7153      0.00623              0    49
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.6686 0.7449 0.7453      0.00646              0    47
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.6765 0.7544 0.7548      0.00688              0    44
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.6793 0.7582 0.7586      0.00781              0    39
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=1      0.7077 0.7935 0.7939      0.00894              0    34
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=1     0.7245 0.8156 0.8160      0.01068              0    29
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.7464 0.8491 0.8494      0.01202              0    25
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.7510 0.8551 0.8554      0.01257              0    24
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=2     0.7514 0.8559 0.8563      0.01436              0    21
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=2     0.7739 0.8846 0.8850      0.01754              0    18
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.7840 0.8985 0.8989      0.01882              0    16
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=2    0.7841 0.8986 0.8990      0.02235              0    14
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.7964 0.9158 0.9162      0.03064              0    10
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=2   0.7973 0.9169 0.9173      0.03167              0    10
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.8038 0.9355 0.9359      0.03429              0    9
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.8159 0.9488 0.9491      0.03768              0    8
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.8223 0.9640 0.9644      0.04499              0    7
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.8227 0.9647 0.9651      0.04840              0    7
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=4    0.8240 0.9593 0.9596      0.05771              0    6
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=8    0.8325 0.9776 0.9780      0.06554              0    5
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=8   0.8334 0.9792 0.9796      0.06879              0    5
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.8378 0.9875 0.9877      0.09636              0    4
nprobe=512,quantizer_efSearch=64,ht=56,k_factor=16   0.8386 0.9884 0.9887      0.12822              0    3
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.8412 0.9930 0.9933      0.14488              0    3
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.8414 0.9932 0.9935      0.16234              0    2
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=32  0.8415 0.9941 0.9944      0.17465              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.8424 0.9955 0.9957      0.18474              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.8426 0.9958 0.9960      0.20255              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=64  0.8428 0.9967 0.9971      0.29712              0    2
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=32 0.8432 0.9982 0.9984      0.47570              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=64 0.8434 0.9993 0.9997      0.83640              0    1
```

</details>
<details><summary>`OPQ56_112,IVF65536(IVF256,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 742758176

 code_size 63

 log filename: autotune.dbdeep10M.OPQ56_112_IVF65536_IVF256_PQ56x4fs_RFlat__PQ7+56.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.7388 0.8504 0.8508      0.14876         705389    3
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=56,k_factor=1        0.3711 0.3990 0.3994      0.00521         359489    58
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1        0.5526 0.6083 0.6087      0.00570         705389    53
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=1        0.5908 0.6525 0.6529      0.00617         704919    49
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=56,k_factor=1      0.6451 0.7189 0.7193      0.00706        1382358    43
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1      0.6733 0.7542 0.7546      0.00778        1381199    39
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1       0.6840 0.7677 0.7681      0.00930         705389    33
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=2      0.6893 0.7789 0.7793      0.01140        5311787    27
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=1     0.7203 0.8158 0.8162      0.01355       10397478    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.7421 0.8497 0.8501      0.01400        1377417    22
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=2      0.7443 0.8531 0.8535      0.01336        2709694    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=2      0.7472 0.8566 0.8570      0.01480        5310138    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.7715 0.8888 0.8892      0.01861        1383375    17
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=2      0.7718 0.8890 0.8894      0.01968        1380732    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=2     0.7791 0.8987 0.8991      0.02287       10374315    14
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=2     0.7856 0.9079 0.9083      0.02819        2705509    11
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=4     0.8032 0.9380 0.9383      0.03497        2713129    9
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=56,k_factor=8     0.8037 0.9413 0.9417      0.04510        1383375    7
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=8     0.8125 0.9542 0.9546      0.04395        5317995    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8     0.8189 0.9626 0.9629      0.04872        2713129    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56,k_factor=8   0.8211 0.9653 0.9656      0.06171       10397478    5
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8     0.8292 0.9763 0.9767      0.07801        2691404    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=16    0.8330 0.9827 0.9831      0.09342        2693380    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=16    0.8387 0.9919 0.9923      0.14462        5317995    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32    0.8403 0.9947 0.9951      0.17828        5317995    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=64   0.8416 0.9961 0.9965      0.26937       10397478    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=32  0.8421 0.9977 0.9981      0.27665       10397478    2
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56,k_factor=32  0.8424 0.9984 0.9988      0.48900       10397478    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=32  0.8425 0.9985 0.9989      0.74300       10397478    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=56,k_factor=64 0.8430 0.9992 0.9996      1.60716       10397478    1
```

</details>
<details><summary>`OPQ64_128,IVF16384_HNSW32,PQ64` </summary>
Index size 733160624

 code_size 64

 log filename: autotune.dbdeep10M.OPQ64_128_IVF16384_HNSW32_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.4111 0.4696 0.4700      0.01458       41844748    21
nprobe=1,quantizer_efSearch=4,ht=106      0.0032 0.0041 0.0045      0.00388       22242098    78
nprobe=1,quantizer_efSearch=8,ht=156      0.0138 0.0155 0.0159      0.00395       21567990    77
nprobe=1,quantizer_efSearch=16,ht=234     0.2877 0.3199 0.3203      0.00600       21246071    51
nprobe=2,quantizer_efSearch=4,ht=230      0.3528 0.3961 0.3967      0.00881       42937848    35
nprobe=2,quantizer_efSearch=16,ht=226     0.3787 0.4235 0.4239      0.00881       41318421    35
nprobe=2,quantizer_efSearch=16,ht=234     0.4062 0.4599 0.4604      0.00973       41318421    31
nprobe=2,quantizer_efSearch=32,ht=242     0.4179 0.4758 0.4763      0.01353       41143418    23
nprobe=4,quantizer_efSearch=32,ht=228     0.5051 0.5820 0.5825      0.01607       78833540    19
nprobe=4,quantizer_efSearch=32,ht=244     0.5388 0.6322 0.6328      0.02354       78833540    13
nprobe=8,quantizer_efSearch=64,ht=222     0.5529 0.6396 0.6399      0.02771      150067642    11
nprobe=8,quantizer_efSearch=32,ht=244     0.6336 0.7631 0.7636      0.04064      150427289    8
nprobe=8,quantizer_efSearch=128,ht=256    0.6355 0.7660 0.7665      0.05201      149970327    6
nprobe=16,quantizer_efSearch=64,ht=232    0.6772 0.8284 0.8291      0.06298      280451091    5
nprobe=16,quantizer_efSearch=128,ht=242   0.6961 0.8608 0.8615      0.07765      280143798    4
nprobe=16,quantizer_efSearch=32,ht=246    0.6974 0.8632 0.8640      0.07774      281390105    4
nprobe=16,quantizer_efSearch=256,ht=246   0.6986 0.8640 0.8648      0.08905      280042300    4
nprobe=32,quantizer_efSearch=16,ht=230    0.7019 0.8686 0.8695      0.10400      521675837    3
nprobe=32,quantizer_efSearch=32,ht=232    0.7138 0.8894 0.8903      0.11065      518000714    3
nprobe=32,quantizer_efSearch=128,ht=512   0.7377 0.9319 0.9328      0.11853      514928241    3
nprobe=64,quantizer_efSearch=32,ht=234    0.7394 0.9360 0.9370      0.22204      947374799    2
nprobe=64,quantizer_efSearch=512,ht=256   0.7570 0.9687 0.9700      0.33233      938791594    1
nprobe=128,quantizer_efSearch=64,ht=246   0.7629 0.9832 0.9849      0.51243     1710367733    1
nprobe=512,quantizer_efSearch=512,ht=512  0.7683 0.9967 0.9988      1.27562     5573937181    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.7684 0.9973 0.9999      7.04781    28556889233    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64` </summary>
Index size 927838384

 code_size 64

 log filename: autotune.dbdeep10M.OPQ64_128_IVF262144_HNSW32_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.2969 0.3097 0.3102      0.00583         900160    52
nprobe=1,quantizer_efSearch=4,ht=106      0.0023 0.0026 0.0031      0.00305         449315    99
nprobe=1,quantizer_efSearch=4,ht=256      0.1870 0.1928 0.1933      0.00329         449315    92
nprobe=2,quantizer_efSearch=4,ht=230      0.1947 0.2007 0.2012      0.00487         903219    62
nprobe=2,quantizer_efSearch=8,ht=254      0.2989 0.3118 0.3123      0.00576         900160    53
nprobe=4,quantizer_efSearch=4,ht=238      0.3163 0.3296 0.3301      0.00831        1806446    37
nprobe=4,quantizer_efSearch=32,ht=244     0.3943 0.4125 0.4130      0.01298        1793877    24
nprobe=8,quantizer_efSearch=32,ht=244     0.5066 0.5362 0.5367      0.01975        3580525    16
nprobe=16,quantizer_efSearch=16,ht=256    0.6397 0.6905 0.6910      0.03078        7166903    10
nprobe=32,quantizer_efSearch=128,ht=512   0.7327 0.8007 0.8012      0.03608       14211882    9
nprobe=64,quantizer_efSearch=512,ht=256   0.7956 0.8783 0.8788      0.17174       28221254    2
nprobe=128,quantizer_efSearch=64,ht=246   0.8067 0.8937 0.8942      0.27675       56055484    2
nprobe=512,quantizer_efSearch=512,ht=512  0.8749 0.9878 0.9883      0.34825      216419141    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8821 0.9985 0.9990      2.12907     1500206277    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64` </summary>
Index size 867510772

 code_size 64

 log filename: autotune.dbdeep10M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2762 0.2856 0.2861      0.00680        3670604    45
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=138       0.0034 0.0039 0.0043      0.00261         634321    115
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=230       0.1179 0.1210 0.1215      0.00271         633174    111
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=230       0.1411 0.1444 0.1449      0.00308        1872364    98
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=246      0.1958 0.2008 0.2013      0.00374        3225134    81
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=238      0.2565 0.2648 0.2653      0.00798       11581520    38
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256      0.3080 0.3203 0.3208      0.00853       11588588    36
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=512      0.5915 0.6377 0.6382      0.01089        7924875    28
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=256    0.6456 0.6971 0.6976      0.04983       17765613    7
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=254      0.7042 0.7657 0.7662      0.07907       15692685    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256     0.7975 0.8778 0.8783      0.12394       38858011    3
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=512    0.8079 0.8981 0.8986      0.18305      112858352    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=512    0.8694 0.9754 0.9759      0.28632      226543706    2
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=512 0.8852 0.9989 0.9994      3.67017     1635403317    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64` </summary>
Index size 772096688

 code_size 64

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_HNSW32_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3890 0.4115 0.4119      0.00557        3556492    54
nprobe=1,quantizer_efSearch=4,ht=106      0.0020 0.0029 0.0033      0.00260        1785131    116
nprobe=1,quantizer_efSearch=8,ht=156      0.0083 0.0096 0.0100      0.00289        1779870    104
nprobe=1,quantizer_efSearch=4,ht=256      0.2577 0.2697 0.2701      0.00302        1785131    100
nprobe=2,quantizer_efSearch=4,ht=230      0.2857 0.2988 0.2991      0.00449        3561421    67
nprobe=2,quantizer_efSearch=8,ht=254      0.3896 0.4123 0.4127      0.00533        3556492    57
nprobe=4,quantizer_efSearch=4,ht=238      0.4295 0.4574 0.4577      0.00797        7118356    38
nprobe=4,quantizer_efSearch=32,ht=244     0.5009 0.5352 0.5355      0.01023        7097276    30
nprobe=8,quantizer_efSearch=32,ht=244     0.6106 0.6634 0.6637      0.01813       14146669    17
nprobe=8,quantizer_efSearch=128,ht=256    0.6269 0.6842 0.6846      0.02422       14141287    13
nprobe=16,quantizer_efSearch=16,ht=256    0.7155 0.7909 0.7913      0.03344       28212175    9
nprobe=32,quantizer_efSearch=128,ht=512   0.7880 0.8810 0.8814      0.03440       55790029    9
nprobe=64,quantizer_efSearch=512,ht=256   0.8314 0.9362 0.9366      0.15868      110183434    2
nprobe=256,quantizer_efSearch=32,ht=512   0.8617 0.9766 0.9770      0.19406      423892914    2
nprobe=512,quantizer_efSearch=512,ht=512  0.8753 0.9966 0.9970      0.41404      826953848    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8762 0.9993 0.9997      2.67117     5410813920    1
```

</details>
<details><summary>`OPQ64_128,IVF65536(IVF256,PQ64x4fs,RFlat),PQ64` </summary>
Index size 757156340

 code_size 64

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_IVF256_PQ64x4fs_RFlat__PQ64.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.1435 0.1505 0.1509      0.22069      218088985    2
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=174       0.0189 0.0199 0.0204      0.00283        2483193    107
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=190       0.0318 0.0329 0.0332      0.00286        1969387    105
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=206       0.0610 0.0631 0.0635      0.00281        1876597    107
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=208       0.0672 0.0697 0.0701      0.00288        1878584    105
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=238       0.2064 0.2138 0.2142      0.00305        1876619    99
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=242      0.2431 0.2518 0.2522      0.00308        1964376    98
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=252       0.3452 0.3638 0.3642      0.00527        3761109    57
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=256     0.3554 0.3739 0.3743      0.00827       13937194    37
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=254     0.3961 0.4182 0.4186      0.00745        8863481    41
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=240      0.4492 0.4755 0.4759      0.01000       12452422    30
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=250     0.5141 0.5486 0.5490      0.01234       17478724    25
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=248      0.6196 0.6685 0.6689      0.01688       15559365    18
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=234      0.6244 0.6705 0.6709      0.03047       28946148    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=240     0.6903 0.7481 0.7485      0.03306       33446534    10
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=256    0.7282 0.7962 0.7966      0.03779       38515238    8
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=512      0.8061 0.8977 0.8981      0.05349      111833284    6
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=512    0.8401 0.9395 0.9399      0.05788      115396805    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=250    0.8568 0.9620 0.9624      0.24282      222039389    2
nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=512 0.8825 0.9984 0.9988      0.90470     1610355638    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=512 0.8833 0.9995 0.9999      3.78702     5947850909    1
```

</details>
<details><summary>`OPQ64_128,IVF65536,PQ64` </summary>
Index size 754259195

 code_size 64

 log filename: autotune.dbdeep10M.OPQ64_128_IVF65536_PQ64.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.8430 0.9400 0.9404      0.06280      109947174    5
nprobe=1,ht=162                          0.0110 0.0120 0.0124      0.01342        1772167    23
nprobe=2,ht=162                          0.0126 0.0137 0.0141      0.01428        3543331    22
nprobe=2,ht=256                          0.3997 0.4204 0.4208      0.01505        3543331    20
nprobe=8,ht=512                          0.6364 0.6868 0.6872      0.01877       14115501    16
nprobe=32,ht=512                         0.7958 0.8822 0.8826      0.03787       55672546    8
nprobe=64,ht=512                         0.8430 0.9400 0.9404      0.06334      109947174    5
nprobe=256,ht=512                        0.8789 0.9897 0.9901      0.20871      423475872    2
nprobe=512,ht=252                        0.8798 0.9905 0.9909      0.94138      825310318    1
nprobe=4096,ht=512                       0.8851 0.9996 1.0000      2.87900     5929488072    1
```

</details>
<details><summary>`PCAR32,IVF16384_HNSW32,SQfp16` </summary>
Index size 726739029

 code_size 64

 log filename: autotune.dbdeep10M.PCAR32_IVF16384_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1737 0.4211 0.5234      0.00439       45261851    69
nprobe=1,quantizer_efSearch=16           0.1128 0.2339 0.2634      0.00251       11681579    120
nprobe=2,quantizer_efSearch=4            0.1388 0.3086 0.3634      0.00264       23567094    114
nprobe=2,quantizer_efSearch=8            0.1466 0.3272 0.3856      0.00276       23103039    109
nprobe=2,quantizer_efSearch=16           0.1485 0.3305 0.3896      0.00297       22913493    101
nprobe=4,quantizer_efSearch=8            0.1737 0.4211 0.5234      0.00422       45261851    72
nprobe=4,quantizer_efSearch=16           0.1763 0.4281 0.5311      0.00486       44846641    62
nprobe=4,quantizer_efSearch=32           0.1766 0.4282 0.5317      0.00532       44733546    57
nprobe=4,quantizer_efSearch=64           0.1769 0.4283 0.5318      0.00596       44686375    51
nprobe=8,quantizer_efSearch=8            0.1887 0.4899 0.6441      0.00777       87678035    39
nprobe=8,quantizer_efSearch=32           0.1947 0.5046 0.6616      0.00839       86633973    36
nprobe=8,quantizer_efSearch=64           0.1955 0.5053 0.6627      0.00955       86529607    32
nprobe=16,quantizer_efSearch=4           0.1958 0.5266 0.7199      0.01453      169295075    21
nprobe=16,quantizer_efSearch=16          0.2036 0.5543 0.7628      0.01532      167526847    20
nprobe=16,quantizer_efSearch=64          0.2050 0.5560 0.7669      0.01637      166506772    19
nprobe=32,quantizer_efSearch=64          0.2140 0.5924 0.8464      0.02913      318723699    11
nprobe=64,quantizer_efSearch=64          0.2144 0.6058 0.8870      0.05379      607323229    6
nprobe=128,quantizer_efSearch=64         0.2154 0.6114 0.9052      0.09693     1153926962    4
nprobe=256,quantizer_efSearch=128        0.2159 0.6140 0.9142      0.18401     2176312061    2
nprobe=256,quantizer_efSearch=256        0.2160 0.6141 0.9143      0.19886     2174362352    2
nprobe=4096,quantizer_efSearch=64        0.2161 0.6143 0.9152      0.57869     6757862256    1
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQfp16` </summary>
Index size 827044949

 code_size 64

 log filename: autotune.dbdeep10M.PCAR32_IVF262144_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2131 0.6081 0.8851      0.06110      108423143    5
nprobe=1,quantizer_efSearch=4            0.0962 0.1562 0.1587      0.00193         449225    156
nprobe=2,quantizer_efSearch=4            0.1216 0.2210 0.2305      0.00199         894618    151
nprobe=4,quantizer_efSearch=4            0.1428 0.2972 0.3195      0.00208        1784648    145
nprobe=4,quantizer_efSearch=8            0.1554 0.3217 0.3474      0.00231        1785323    131
nprobe=8,quantizer_efSearch=4            0.1716 0.3978 0.4520      0.00233        3560188    129
nprobe=8,quantizer_efSearch=8            0.1754 0.4029 0.4581      0.00252        3558481    119
nprobe=16,quantizer_efSearch=4           0.1836 0.4608 0.5573      0.00293        7099674    103
nprobe=16,quantizer_efSearch=8           0.1897 0.4769 0.5791      0.00298        7088406    101
nprobe=16,quantizer_efSearch=16          0.1919 0.4838 0.5884      0.00328        7073989    92
nprobe=32,quantizer_efSearch=8           0.1974 0.5219 0.6724      0.00458       14091614    66
nprobe=32,quantizer_efSearch=16          0.1995 0.5317 0.6889      0.00498       14060750    61
nprobe=32,quantizer_efSearch=32          0.2009 0.5368 0.6953      0.00552       14033711    55
nprobe=64,quantizer_efSearch=16          0.2047 0.5640 0.7648      0.00713       27935755    43
nprobe=64,quantizer_efSearch=32          0.2075 0.5731 0.7777      0.00787       27870852    39
nprobe=64,quantizer_efSearch=64          0.2080 0.5757 0.7809      0.01001       27816857    30
nprobe=128,quantizer_efSearch=32         0.2103 0.5928 0.8355      0.01374       55215301    22
nprobe=128,quantizer_efSearch=64         0.2105 0.5957 0.8414      0.01486       55088732    21
nprobe=128,quantizer_efSearch=128        0.2107 0.5969 0.8429      0.01876       55004316    16
nprobe=256,quantizer_efSearch=32         0.2113 0.6007 0.8688      0.02335      108597987    13
nprobe=256,quantizer_efSearch=64         0.2122 0.6057 0.8801      0.02541      108765991    12
nprobe=256,quantizer_efSearch=128        0.2124 0.6071 0.8844      0.02828      108590204    11
nprobe=256,quantizer_efSearch=256        0.2131 0.6080 0.8850      0.03653      108457117    9
nprobe=512,quantizer_efSearch=256        0.2135 0.6124 0.9054      0.05900      213297543    6
nprobe=512,quantizer_efSearch=512        0.2137 0.6126 0.9052      0.07752      213120846    4
nprobe=1024,quantizer_efSearch=512       0.2140 0.6151 0.9135      0.14110      416897696    3
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 760038809

 code_size 64

 log filename: autotune.dbdeep10M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2135 0.6102 0.9010      0.07595      436114068    4
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.1045 0.1839 0.1915      0.00231        1078791    131
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1    0.1063 0.1869 0.1945      0.00232        1076939    130
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1108 0.1922 0.1998      0.00229        1613705    132
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1244 0.2252 0.2345      0.00232        1611016    130
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1323 0.2395 0.2500      0.00233        2289688    129
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1460 0.2942 0.3156      0.00241        2513553    125
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1522 0.3080 0.3304      0.00244        2507676    123
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1532 0.3146 0.3377      0.00249        2501819    121
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1645 0.3604 0.4020      0.00249        4990968    121
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1682 0.3847 0.4389      0.00261        8589341    115
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1718 0.3878 0.4427      0.00279        4276663    108
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1737 0.3933 0.4440      0.00291        6310757    104
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1824 0.4477 0.5416      0.00293        7834818    103
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1897 0.4744 0.5728      0.00320        9849832    94
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1905 0.4715 0.5713      0.00309        8502312    98
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1940 0.5000 0.6331      0.00383       15627806    79
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1978 0.5169 0.6641      0.00413       15553707    73
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2026 0.5371 0.6947      0.00577       19409937    53
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2048 0.5566 0.7450      0.00658       30889987    46
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2056 0.5608 0.7508      0.00771       33458002    39
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2066 0.5672 0.7673      0.00738       30731843    41
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2083 0.5726 0.7740      0.00869       33295907    35
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2093 0.5842 0.8158      0.01186       58448708    26
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2110 0.5901 0.8237      0.01388       66089211    22
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.2112 0.5969 0.8425      0.02373       65617744    13
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.2114 0.5912 0.8366      0.02251      116582037    14
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2131 0.6071 0.8801      0.02565      114248967    12
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2136 0.6079 0.8831      0.02692      119315665    12
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2140 0.6120 0.9015      0.04594      225267402    7
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2141 0.6122 0.9023      0.04904      235549179    7
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=64 0.2142 0.6142 0.9138      0.14413      850209756    3
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQfp16` </summary>
Index size 746800725

 code_size 64

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_HNSW32_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2155 0.6098 0.9064      0.06716      420219197    5
nprobe=1,quantizer_efSearch=4            0.1147 0.2143 0.2248      0.00190        1779839    158
nprobe=1,quantizer_efSearch=8            0.1203 0.2239 0.2350      0.00206        1776476    146
nprobe=2,quantizer_efSearch=4            0.1417 0.2956 0.3247      0.00205        3551941    147
nprobe=4,quantizer_efSearch=4            0.1619 0.3717 0.4315      0.00222        7076282    136
nprobe=8,quantizer_efSearch=4            0.1857 0.4630 0.5734      0.00242       14121862    125
nprobe=8,quantizer_efSearch=8            0.1883 0.4705 0.5840      0.00253       14115154    119
nprobe=8,quantizer_efSearch=16           0.1918 0.4789 0.5943      0.00285       14088201    106
nprobe=8,quantizer_efSearch=32           0.1924 0.4805 0.5969      0.00338       14075145    89
nprobe=16,quantizer_efSearch=4           0.1960 0.5115 0.6731      0.00358       28053224    84
nprobe=16,quantizer_efSearch=8           0.2021 0.5300 0.7000      0.00369       28016112    82
nprobe=16,quantizer_efSearch=16          0.2035 0.5356 0.7077      0.00379       27977179    80
nprobe=16,quantizer_efSearch=32          0.2046 0.5382 0.7122      0.00441       27945494    69
nprobe=32,quantizer_efSearch=8           0.2074 0.5590 0.7737      0.00636       55550203    48
nprobe=32,quantizer_efSearch=16          0.2102 0.5701 0.7903      0.00644       55425068    47
nprobe=32,quantizer_efSearch=32          0.2110 0.5733 0.7957      0.00695       55333764    44
nprobe=32,quantizer_efSearch=64          0.2116 0.5736 0.7966      0.00794       55288005    38
nprobe=32,quantizer_efSearch=128         0.2117 0.5740 0.7968      0.01041       55278299    29
nprobe=64,quantizer_efSearch=16          0.2129 0.5871 0.8404      0.01166      109578352    26
nprobe=64,quantizer_efSearch=32          0.2135 0.5912 0.8506      0.01200      109311272    25
nprobe=64,quantizer_efSearch=64          0.2137 0.5930 0.8548      0.01309      109158140    23
nprobe=128,quantizer_efSearch=32         0.2144 0.6023 0.8813      0.02293      215326331    14
nprobe=128,quantizer_efSearch=64         0.2149 0.6047 0.8877      0.02238      214825679    14
nprobe=256,quantizer_efSearch=64         0.2156 0.6092 0.9043      0.04227      420533263    8
nprobe=512,quantizer_efSearch=64         0.2158 0.6101 0.9107      0.08562      809513606    4
nprobe=512,quantizer_efSearch=128        0.2159 0.6110 0.9137      0.08210      817378454    4
nprobe=1024,quantizer_efSearch=64        0.2160 0.6105 0.9109      0.12743     1344575735    3
nprobe=1024,quantizer_efSearch=128       0.2161 0.6115 0.9155      0.15079     1537349438    2
nprobe=1024,quantizer_efSearch=256       0.2162 0.6117 0.9162      0.16617     1587143895    2
```

</details>
<details><summary>`PCAR32,IVF65536(IVF256,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 730083993

 code_size 64

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_IVF256_PQ16x4fs_RFlat__SQfp16.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2143 0.5944 0.8569      0.02265      221776307    14
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1207 0.2241 0.2353      0.00236        3126187    127
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1211 0.2253 0.2362      0.00241        2464522    125
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1348 0.2800 0.3077      0.00252        3747732    120
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.1482 0.3078 0.3386      0.00250        4903697    120
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1495 0.3113 0.3422      0.00256        4237359    118
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1516 0.3143 0.3457      0.00257        4896085    117
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1527 0.3439 0.4000      0.00260        7319062    116
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1718 0.3975 0.4616      0.00268        7786215    112
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1731 0.4023 0.4676      0.00289        8422312    104
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.1819 0.4444 0.5446      0.00292       14903651    103
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.1831 0.4467 0.5486      0.00286       15537249    105
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.1903 0.4741 0.5887      0.00284       14824898    106
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.1912 0.4784 0.5950      0.00331       15435551    91
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.1928 0.4800 0.5967      0.00350       16766054    86
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1969 0.5094 0.6634      0.00406       28888138    74
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2033 0.5321 0.7016      0.00416       29368800    73
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2039 0.5363 0.7083      0.00494       29328382    61
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2045 0.5340 0.7043      0.00507       33257364    60
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.2056 0.5404 0.7129      0.00613       38335624    49
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2100 0.5690 0.7887      0.00710       56842561    43
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2111 0.5742 0.7948      0.00746       58090464    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.2113 0.5748 0.7955      0.00878       65809067    35
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.2114 0.5747 0.7953      0.00968       60566748    31
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64   0.2115 0.5747 0.7954      0.01247       60552999    25
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2116 0.5820 0.8346      0.01246      111325761    25
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2133 0.5880 0.8419      0.01286      112428274    24
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.2139 0.5923 0.8511      0.01368      114515175    22
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.2145 0.5950 0.8577      0.02393      224144965    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2147 0.6045 0.8856      0.02495      217623423    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2150 0.6054 0.8872      0.02697      225120539    12
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.2153 0.6046 0.8890      0.04558      435312102    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2159 0.6102 0.9044      0.04556      427235740    7
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2161 0.6119 0.9161      0.16184     1602446232    2
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2162 0.6119 0.9161      0.16833     1596196433    2
```

</details>
<details><summary>`PCAR32,IVF65536,SQfp16` </summary>
Index size 728963232

 code_size 64

 log filename: autotune.dbdeep10M.PCAR32_IVF65536_SQfp16.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2156 0.6099 0.9063      0.05504      419673424    6
nprobe=8                                 0.1922 0.4809 0.5972      0.00754       14055753    40
nprobe=16                                0.2045 0.5393 0.7135      0.00878       27894881    35
nprobe=32                                0.2116 0.5738 0.7964      0.01257       55206472    25
nprobe=64                                0.2139 0.5930 0.8541      0.01932      108950465    16
nprobe=128                               0.2150 0.6055 0.8879      0.03362      214228438    9
nprobe=256                               0.2156 0.6099 0.9063      0.06067      419673424    5
nprobe=512                               0.2160 0.6113 0.9143      0.11243      818787283    3
nprobe=1024                              0.2161 0.6118 0.9164      0.19565     1587567136    2
nprobe=2048                              0.2162 0.6121 0.9170      0.36215     3058665784    1
```

</details>
<details><summary>`PCAR64,IVF16384_HNSW32,SQ8` </summary>
Index size 728849109

 code_size 64

 log filename: autotune.dbdeep10M.PCAR64_IVF16384_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5942 0.9746 0.9951      0.31634     2687952669    1
nprobe=2,quantizer_efSearch=8            0.3339 0.4457 0.4467      0.00531       33971618    57
nprobe=2,quantizer_efSearch=16           0.3366 0.4503 0.4513      0.00565       33534982    54
nprobe=2,quantizer_efSearch=32           0.3375 0.4510 0.4520      0.00626       33361015    48
nprobe=4,quantizer_efSearch=16           0.4290 0.6036 0.6066      0.00920       65067301    33
nprobe=4,quantizer_efSearch=32           0.4293 0.6039 0.6069      0.00930       64701758    33
nprobe=8,quantizer_efSearch=8            0.4842 0.7128 0.7185      0.01556      125821980    20
nprobe=8,quantizer_efSearch=32           0.5002 0.7374 0.7432      0.01592      123600682    19
nprobe=8,quantizer_efSearch=64           0.5005 0.7375 0.7434      0.01708      123265314    18
nprobe=8,quantizer_efSearch=256          0.5006 0.7377 0.7436      0.02513      123173887    12
nprobe=16,quantizer_efSearch=8           0.5355 0.8166 0.8262      0.04487      236815418    7
nprobe=16,quantizer_efSearch=16          0.5438 0.8327 0.8424      0.02784      235212428    11
nprobe=16,quantizer_efSearch=32          0.5467 0.8398 0.8495      0.02731      233467388    11
nprobe=32,quantizer_efSearch=8           0.5558 0.8694 0.8815      0.05164      440648300    6
nprobe=32,quantizer_efSearch=256         0.5720 0.9080 0.9213      0.05756      433692482    6
nprobe=32,quantizer_efSearch=64          0.5723 0.9077 0.9211      0.04861      434465642    7
nprobe=64,quantizer_efSearch=16          0.5781 0.9298 0.9457      0.12806      811727330    3
nprobe=64,quantizer_efSearch=32          0.5845 0.9446 0.9608      0.08874      807107171    4
nprobe=64,quantizer_efSearch=128         0.5856 0.9487 0.9648      0.08883      800958905    4
nprobe=128,quantizer_efSearch=128        0.5923 0.9670 0.9860      0.16545     1473150051    2
nprobe=128,quantizer_efSearch=256        0.5925 0.9671 0.9861      0.16225     1470925417    2
nprobe=256,quantizer_efSearch=128        0.5941 0.9740 0.9944      0.32594     2695339149    1
nprobe=256,quantizer_efSearch=256        0.5942 0.9746 0.9950      0.29837     2690272029    2
nprobe=512,quantizer_efSearch=512        0.5948 0.9773 0.9986      0.57324     4930055267    1
nprobe=512,quantizer_efSearch=256        0.5949 0.9774 0.9987      0.53101     4931911014    1
nprobe=1024,quantizer_efSearch=128       0.5950 0.9771 0.9986      0.88755     8470927054    1
nprobe=1024,quantizer_efSearch=256       0.5953 0.9781 0.9996      0.94995     9006947123    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ8` </summary>
Index size 860612309

 code_size 64

 log filename: autotune.dbdeep10M.PCAR64_IVF262144_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5936 0.9523 0.9687      0.07703      110280364    4
nprobe=1,quantizer_efSearch=4            0.1709 0.1933 0.1937      0.00213         450631    141
nprobe=2,quantizer_efSearch=4            0.2361 0.2772 0.2775      0.00224         900484    135
nprobe=4,quantizer_efSearch=4            0.3049 0.3739 0.3743      0.00233        1800976    129
nprobe=8,quantizer_efSearch=4            0.4032 0.5305 0.5314      0.00283        3601856    107
nprobe=8,quantizer_efSearch=8            0.4124 0.5425 0.5435      0.00295        3595645    102
nprobe=16,quantizer_efSearch=4           0.4537 0.6314 0.6337      0.00388        7196252    78
nprobe=16,quantizer_efSearch=8           0.4754 0.6638 0.6664      0.00409        7178435    74
nprobe=16,quantizer_efSearch=16          0.4811 0.6743 0.6771      0.00432        7161174    70
nprobe=16,quantizer_efSearch=32          0.4848 0.6784 0.6810      0.00575        7141544    53
nprobe=32,quantizer_efSearch=8           0.5070 0.7442 0.7489      0.00601       14318297    50
nprobe=32,quantizer_efSearch=16          0.5226 0.7676 0.7728      0.00694       14278348    44
nprobe=32,quantizer_efSearch=32          0.5277 0.7747 0.7801      0.00805       14239984    38
nprobe=64,quantizer_efSearch=8           0.5317 0.8039 0.8111      0.00944       28466715    32
nprobe=64,quantizer_efSearch=16          0.5524 0.8424 0.8510      0.00928       28425755    33
nprobe=64,quantizer_efSearch=32          0.5604 0.8545 0.8635      0.01155       28340380    26
nprobe=64,quantizer_efSearch=64          0.5617 0.8593 0.8685      0.01455       28272913    21
nprobe=128,quantizer_efSearch=32         0.5772 0.9069 0.9193      0.01688       56220164    18
nprobe=128,quantizer_efSearch=64         0.5818 0.9167 0.9295      0.02115       56067937    15
nprobe=128,quantizer_efSearch=128        0.5830 0.9184 0.9317      0.02693       55960093    12
nprobe=256,quantizer_efSearch=64         0.5901 0.9457 0.9615      0.03579      110776921    9
nprobe=256,quantizer_efSearch=128        0.5926 0.9511 0.9674      0.03740      110520586    9
nprobe=256,quantizer_efSearch=256        0.5935 0.9520 0.9684      0.05110      110333866    6
nprobe=512,quantizer_efSearch=128        0.5966 0.9641 0.9824      0.07871      217226175    4
nprobe=512,quantizer_efSearch=256        0.5969 0.9661 0.9847      0.08415      216899489    4
nprobe=512,quantizer_efSearch=512        0.5974 0.9667 0.9853      0.10789      216670292    3
nprobe=1024,quantizer_efSearch=128       0.5979 0.9695 0.9890      0.12036      417494053    3
nprobe=1024,quantizer_efSearch=256       0.5989 0.9731 0.9931      0.13243      423092441    3
nprobe=1024,quantizer_efSearch=512       0.5994 0.9742 0.9941      0.17196      423567205    2
nprobe=2048,quantizer_efSearch=512       0.5997 0.9772 0.9979      0.30490      821500265    1
nprobe=4096,quantizer_efSearch=512       0.5998 0.9779 0.9988      0.46950     1501421027    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ8` </summary>
Index size 795829785

 code_size 64

 log filename: autotune.dbdeep10M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4612 0.6376 0.6396      0.00345        8608868    87
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2123 0.2485 0.2488      0.00223        2321429    135
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2579 0.3036 0.3039      0.00232        2315861    130
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3134 0.3876 0.3880      0.00236        2528852    128
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3255 0.4031 0.4035      0.00238        2526783    127
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3600 0.4617 0.4622      0.00254        5043888    119
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3946 0.5183 0.5191      0.00261        4327679    115
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3972 0.5209 0.5216      0.00255        5019121    118
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4047 0.5471 0.5484      0.00305        7999672    99
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4177 0.5513 0.5522      0.00330        6333882    91
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4612 0.6376 0.6396      0.00340        8609175    89
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4688 0.6491 0.6513      0.00404        9925937    75
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4780 0.6674 0.6700      0.00421        9901351    72
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4811 0.6736 0.6762      0.00613        9884983    49
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5066 0.7356 0.7403      0.00596       15756902    51
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5126 0.7478 0.7527      0.00579       15708002    52
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5223 0.7600 0.7648      0.00674       19657228    45
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5297 0.7773 0.7824      0.00781       19585783    39
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5302 0.7785 0.7836      0.00945       24835099    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5535 0.8392 0.8472      0.00968       31169215    31
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5599 0.8490 0.8574      0.01025       33741259    30
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5621 0.8571 0.8657      0.01187       33642515    26
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5624 0.8582 0.8670      0.01256       38871021    24
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5636 0.8619 0.8708      0.01642       38811388    19
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5720 0.8937 0.9055      0.01921       59059579    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.5739 0.8999 0.9118      0.01944       58911051    16
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5839 0.9185 0.9314      0.02294       66512714    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5846 0.9192 0.9321      0.02344       76942555    13
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5848 0.9198 0.9329      0.04338       97424600    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.5851 0.9197 0.9330      0.04351       76717778    7
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5857 0.9284 0.9426      0.03211      133200241    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5920 0.9457 0.9616      0.03415      115883277    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5936 0.9489 0.9649      0.03412      131572320    9
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5942 0.9525 0.9688      0.03921      131257170    8
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5946 0.9532 0.9696      0.05057      151725103    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5984 0.9665 0.9847      0.06407      258514353    5
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5985 0.9671 0.9857      0.08296      237467516    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5986 0.9672 0.9859      0.08149      257930422    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6003 0.9738 0.9937      0.13389      445582905    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.6004 0.9738 0.9938      0.12861      465895912    3
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.6007 0.9772 0.9980      0.22378      867877136    2
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.6008 0.9772 0.9980      0.27335      846926066    2
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ8` </summary>
Index size 755202261

 code_size 64

 log filename: autotune.dbdeep10M.PCAR64_IVF65536_HNSW32_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5976 0.9684 0.9882      0.09115      425270419    4
nprobe=1,quantizer_efSearch=4            0.2176 0.2614 0.2619      0.00196        1780260    154
nprobe=2,quantizer_efSearch=4            0.2985 0.3771 0.3776      0.00211        3559815    142
nprobe=2,quantizer_efSearch=8            0.3197 0.4024 0.4029      0.00226        3565147    133
nprobe=4,quantizer_efSearch=4            0.3691 0.4929 0.4942      0.00231        7114475    131
nprobe=4,quantizer_efSearch=8            0.3990 0.5336 0.5352      0.00241        7126931    125
nprobe=4,quantizer_efSearch=16           0.4059 0.5426 0.5443      0.00283        7106176    107
nprobe=8,quantizer_efSearch=4            0.4563 0.6463 0.6492      0.00366       14227646    82
nprobe=8,quantizer_efSearch=8            0.4646 0.6575 0.6605      0.00350       14214492    86
nprobe=8,quantizer_efSearch=16           0.4729 0.6707 0.6739      0.00378       14169240    80
nprobe=8,quantizer_efSearch=32           0.4735 0.6713 0.6746      0.00449       14151994    67
nprobe=16,quantizer_efSearch=4           0.5045 0.7422 0.7475      0.00562       28359022    54
nprobe=16,quantizer_efSearch=16          0.5280 0.7825 0.7884      0.00617       28225766    49
nprobe=16,quantizer_efSearch=32          0.5298 0.7853 0.7912      0.00652       28183134    47
nprobe=16,quantizer_efSearch=64          0.5302 0.7860 0.7919      0.00753       28166727    40
nprobe=32,quantizer_efSearch=8           0.5490 0.8389 0.8465      0.01022       56153426    30
nprobe=32,quantizer_efSearch=32          0.5613 0.8647 0.8733      0.01075       55911581    28
nprobe=32,quantizer_efSearch=64          0.5620 0.8661 0.8746      0.01184       55858524    26
nprobe=32,quantizer_efSearch=128         0.5624 0.8665 0.8749      0.01416       55840948    22
nprobe=64,quantizer_efSearch=16          0.5747 0.9068 0.9187      0.02052      110927198    15
nprobe=64,quantizer_efSearch=32          0.5797 0.9172 0.9296      0.01941      110610956    16
nprobe=64,quantizer_efSearch=64          0.5817 0.9190 0.9318      0.01909      110437584    16
nprobe=64,quantizer_efSearch=128         0.5821 0.9200 0.9329      0.02122      110362692    15
nprobe=128,quantizer_efSearch=128        0.5917 0.9505 0.9679      0.03941      217137630    8
nprobe=256,quantizer_efSearch=32         0.5927 0.9562 0.9743      0.06544      424979289    5
nprobe=256,quantizer_efSearch=64         0.5968 0.9648 0.9842      0.06530      425963010    5
nprobe=256,quantizer_efSearch=128        0.5973 0.9674 0.9870      0.06637      425709012    5
nprobe=256,quantizer_efSearch=256        0.5976 0.9684 0.9882      0.07075      425376666    5
nprobe=512,quantizer_efSearch=128        0.6001 0.9749 0.9955      0.12354      827225239    3
nprobe=512,quantizer_efSearch=256        0.6002 0.9755 0.9965      0.13288      829833609    3
nprobe=512,quantizer_efSearch=512        0.6004 0.9756 0.9967      0.15319      828978732    2
nprobe=1024,quantizer_efSearch=256       0.6006 0.9771 0.9982      0.25856     1604815311    2
nprobe=1024,quantizer_efSearch=512       0.6009 0.9774 0.9988      0.27154     1606359565    2
nprobe=2048,quantizer_efSearch=512       0.6011 0.9782 0.9998      0.49286     3082186568    1
```

</details>
<details><summary>`PCAR64,IVF65536(IVF256,PQ32x4fs,RFlat),SQ8` </summary>
Index size 739078169

 code_size 64

 log filename: autotune.dbdeep10M.PCAR64_IVF65536_IVF256_PQ32x4fs_RFlat__SQ8.g.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5166 0.7659 0.7712      0.00611       28976531    50
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2     0.2156 0.2590 0.2593      0.00243        1965509    124
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2325 0.2806 0.2810      0.00250        3133352    121
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3121 0.3944 0.3948      0.00252        3923268    120
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3176 0.4001 0.4005      0.00253        4914646    119
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3223 0.4060 0.4064      0.00265        4912635    114
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3960 0.5328 0.5342      0.00271        7806770    111
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3965 0.5330 0.5344      0.00279        8465877    108
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16    0.4051 0.5425 0.5440      0.00342        8459658    89
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.4059 0.5434 0.5449      0.00365        9776964    83
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.4659 0.6610 0.6641      0.00393       14886307    77
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4723 0.6693 0.6726      0.00458       16838868    66
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.4737 0.6709 0.6741      0.00468       15514112    65
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.4945 0.7292 0.7342      0.00650       28747268    47
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5166 0.7659 0.7712      0.00633       28976039    48
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5283 0.7840 0.7896      0.00709       29552706    43
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5298 0.7840 0.7898      0.00730       30860272    42
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5300 0.7845 0.7903      0.00764       33444789    40
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.5302 0.7859 0.7916      0.00808       30838076    38
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5309 0.7866 0.7923      0.00883       38564599    34
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.5432 0.8302 0.8384      0.01045       56982244    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5461 0.8376 0.8458      0.01080       56859259    28
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5586 0.8591 0.8676      0.01119       57323598    27
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5617 0.8657 0.8745      0.01208       58523910    25
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.5622 0.8666 0.8753      0.01584       66199980    19
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=128  0.5623 0.8666 0.8754      0.01734       66216791    18
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5754 0.9069 0.9183      0.01957      112315117    16
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5793 0.9144 0.9263      0.01975      113388957    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5811 0.9200 0.9322      0.02022      113111999    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5825 0.9215 0.9339      0.02079      115605152    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5853 0.9379 0.9529      0.03603      219798170    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5913 0.9496 0.9660      0.03866      219990891    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5922 0.9510 0.9680      0.04037      227346860    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5976 0.9676 0.9869      0.07154      436164860    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5980 0.9689 0.9884      0.07435      435467797    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5990 0.9726 0.9920      0.13341      835725991    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6010 0.9760 0.9970      0.13901      838934149    3
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.6014 0.9768 0.9983      0.25847     1612779817    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6016 0.9775 0.9991      0.25454     1616245355    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6018 0.9782 0.9998      0.52194     3101124928    1
```

</details>
<details><summary>`PCAR64,IVF65536,SQ8` </summary>
Index size 737364768

 code_size 64

 log filename: autotune.dbdeep10M.PCAR64_IVF65536_SQ8.f.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5995 0.9682 0.9882      0.07508      424597865    4
nprobe=8                                 0.4761 0.6730 0.6761      0.01101       14126580    28
nprobe=16                                0.5315 0.7867 0.7924      0.01278       28117000    24
nprobe=32                                0.5637 0.8670 0.8754      0.01713       55742660    18
nprobe=64                                0.5837 0.9199 0.9329      0.02587      110152701    12
nprobe=128                               0.5935 0.9503 0.9678      0.04249      216662768    8
nprobe=256                               0.5995 0.9682 0.9882      0.07578      424597865    4
nprobe=512                               0.6026 0.9754 0.9967      0.13774      827752336    3
nprobe=1024                              0.6031 0.9772 0.9989      0.25448     1603126059    2
nprobe=2048                              0.6035 0.9782 0.9998      0.49740     3086397938    1
```

</details>
