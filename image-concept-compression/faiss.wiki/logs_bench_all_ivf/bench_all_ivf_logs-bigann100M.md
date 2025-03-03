# Detailed logs for dataset bigann100M

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fs` </summary>
Index size 2300930252

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0617 0.2012 0.3370      0.00283        4622194    106
nprobe=2,quantizer_efSearch=4            0.0509 0.1485 0.2202      0.00186        2318175    162
nprobe=4,quantizer_efSearch=4            0.0552 0.1741 0.2920      0.00189        4634521    160
nprobe=2,quantizer_efSearch=8            0.0561 0.1687 0.2494      0.00291        2312030    104
nprobe=4,quantizer_efSearch=8            0.0617 0.2012 0.3370      0.00268        4622194    113
nprobe=16,quantizer_efSearch=8           0.0673 0.2446 0.4972      0.00475       18375637    64
nprobe=16,quantizer_efSearch=32          0.0683 0.2563 0.5226      0.00774       18264454    39
nprobe=16,quantizer_efSearch=128         0.0697 0.2579 0.5270      0.02702       18209785    12
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fsr` </summary>
Index size 2300807372

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1833 0.4888 0.6171      0.01152       18250652    27
nprobe=1,quantizer_efSearch=4            0.0828 0.1476 0.1540      0.00216        1159516    140
nprobe=1,quantizer_efSearch=16           0.0956 0.1685 0.1751      0.00520        1148774    58
nprobe=2,quantizer_efSearch=16           0.1207 0.2415 0.2605      0.00544        2294883    56
nprobe=4,quantizer_efSearch=16           0.1455 0.3290 0.3707      0.00582        4598276    52
nprobe=8,quantizer_efSearch=16           0.1698 0.4105 0.4931      0.00662        9189188    46
nprobe=16,quantizer_efSearch=16          0.1824 0.4813 0.6069      0.00806       18312401    38
nprobe=32,quantizer_efSearch=16          0.1889 0.5270 0.6999      0.01095       36378448    28
nprobe=32,quantizer_efSearch=32          0.1908 0.5357 0.7140      0.01389       36253446    22
nprobe=32,quantizer_efSearch=64          0.1922 0.5401 0.7222      0.02034       36156802    15
nprobe=64,quantizer_efSearch=16          0.1925 0.5505 0.7649      0.02145       72098118    14
nprobe=32,quantizer_efSearch=128         0.1932 0.5414 0.7245      0.03255       36120127    10
nprobe=64,quantizer_efSearch=128         0.2004 0.5751 0.8096      0.03883       71470149    8
nprobe=128,quantizer_efSearch=128        0.2026 0.5925 0.8691      0.05587      141093325    6
nprobe=128,quantizer_efSearch=256        0.2027 0.5935 0.8709      0.07565      140940822    4
nprobe=256,quantizer_efSearch=256        0.2031 0.6038 0.9053      0.10980      277520765    3
nprobe=512,quantizer_efSearch=128        0.2039 0.6031 0.9145      0.15652      547481244    2
nprobe=512,quantizer_efSearch=256        0.2042 0.6055 0.9203      0.17360      545468933    2
nprobe=1024,quantizer_efSearch=256       0.2048 0.6052 0.9266      0.30203     1071363776    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 2041324048

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.0489 0.1449 0.2153      0.00145        3060390    207
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1  0.0293 0.0758 0.0991      0.00111        1538037    272
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1  0.0355 0.0937 0.1248      0.00115        1533269    261
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2  0.0382 0.1012 0.1311      0.00120        1894925    251
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2  0.0416 0.1116 0.1462      0.00125        1894094    240
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2  0.0437 0.1161 0.1525      0.00135        1891253    222
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4  0.0450 0.1202 0.1560      0.00139        2606853    217
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2  0.0489 0.1449 0.2153      0.00148        3060320    203
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2  0.0531 0.1664 0.2753      0.00180        5406986    167
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4  0.0541 0.1639 0.2417      0.00174        3764462    173
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2  0.0549 0.1760 0.2895      0.00187        5389339    161
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0584 0.1822 0.3016      0.00191        6101948    157
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4 0.0622 0.1973 0.3275      0.00247        6077289    122
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16 0.0637 0.2024 0.3296      0.00284       10183299    106
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.0694 0.2381 0.4445      0.00493       20055897    61
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 2041206288

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1512 0.3512 0.4078      0.00387       12105692    78
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1     0.0564 0.1016 0.1050      0.00156        1535276    193
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.0639 0.1155 0.1202      0.00159        1530147    189
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.0656 0.1180 0.1219      0.00167        1892294    181
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.0729 0.1326 0.1378      0.00169        1891149    178
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.0804 0.1445 0.1504      0.00141        1890447    213
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.1010 0.2192 0.2447      0.00243        5012407    124
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1071 0.2136 0.2292      0.00220        3752603    137
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1140 0.2256 0.2433      0.00228        3747795    132
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1226 0.2666 0.2971      0.00242        5382351    125
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1237 0.2739 0.3076      0.00254        5372520    119
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1336 0.2919 0.3278      0.00265        6078152    114
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1384 0.3039 0.3427      0.00282        6071232    107
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1386 0.3056 0.3461      0.00302        6063439    100
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1396 0.3273 0.3814      0.00328       10014629    92
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1444 0.3316 0.3835      0.00334       10753016    90
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1580 0.3727 0.4374      0.00353       10707098    86
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1662 0.3985 0.4697      0.00406       12062877    74
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1681 0.4271 0.5296      0.00501       19900169    60
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1707 0.4093 0.4815      0.00508       14752323    60
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1718 0.4165 0.4950      0.00533       14737961    57
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1791 0.4623 0.5778      0.00556       21238021    54
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1803 0.4722 0.5902      0.00598       21193528    51
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1850 0.4819 0.6013      0.00655       23897625    46
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1882 0.5123 0.6714      0.00855       39352386    36
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1888 0.5167 0.6818      0.00939       39305976    32
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1965 0.5351 0.7014      0.00955       41949411    32
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1966 0.5401 0.7142      0.01058       41895638    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1984 0.5451 0.7228      0.01242       46916794    25
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.1987 0.5465 0.7244      0.01415       46985154    22
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.1989 0.5472 0.7255      0.01744       57284319    18
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.2005 0.5575 0.7562      0.02190       83310476    14
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2014 0.5719 0.7921      0.02241       77492486    14
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2029 0.5759 0.7993      0.02501       92744262    12
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2036 0.5822 0.8089      0.03086      112964471    10
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2038 0.5822 0.8107      0.02966       92555795    11
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.2042 0.5853 0.8269      0.03365      154221366    9
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.2046 0.5865 0.8306      0.04035      164195038    8
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2065 0.6010 0.8648      0.04264      162655436    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2071 0.6071 0.8933      0.07650      290142864    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=512 0.2072 0.6095 0.9016      0.10349      440132757    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.2073 0.6098 0.9113      0.16540      558807786    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.2086 0.6141 0.9286      0.26235     1092853824    2
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fs` </summary>
Index size 1775186892

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0551 0.1846 0.3137      0.00193        8648918    156
nprobe=2,quantizer_efSearch=4            0.0487 0.1679 0.2876      0.00142        8665224    211
nprobe=4,quantizer_efSearch=4            0.0519 0.1938 0.3604      0.00179       17339793    168
nprobe=2,quantizer_efSearch=8            0.0551 0.1846 0.3137      0.00192        8648918    157
nprobe=8,quantizer_efSearch=4            0.0577 0.2264 0.4600      0.00262       34612089    115
nprobe=4,quantizer_efSearch=16           0.0616 0.2243 0.4167      0.00316       17285005    95
nprobe=4,quantizer_efSearch=32           0.0620 0.2258 0.4196      0.00488       17262168    62
nprobe=16,quantizer_efSearch=128         0.0623 0.2507 0.5571      0.01511       68763868    20
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fsr` </summary>
Index size 1775198156

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1797 0.5201 0.7148      0.00946       68827366    32
nprobe=1,quantizer_efSearch=4            0.0953 0.1954 0.2167      0.00169        4333346    178
nprobe=1,quantizer_efSearch=16           0.1011 0.2107 0.2335      0.00377        4302998    80
nprobe=2,quantizer_efSearch=16           0.1292 0.2957 0.3429      0.00405        8626111    75
nprobe=4,quantizer_efSearch=16           0.1507 0.3855 0.4716      0.00455       17266367    66
nprobe=8,quantizer_efSearch=16           0.1689 0.4622 0.6013      0.00547       34516023    55
nprobe=16,quantizer_efSearch=16          0.1770 0.5125 0.7038      0.00712       68929817    43
nprobe=16,quantizer_efSearch=32          0.1797 0.5201 0.7148      0.00939       68827366    32
nprobe=32,quantizer_efSearch=16          0.1846 0.5501 0.7884      0.00994      137101122    31
nprobe=32,quantizer_efSearch=32          0.1847 0.5564 0.7988      0.01212      136867383    25
nprobe=32,quantizer_efSearch=64          0.1860 0.5592 0.8036      0.01662      136737919    19
nprobe=64,quantizer_efSearch=64          0.1875 0.5782 0.8600      0.02682      271098190    12
nprobe=64,quantizer_efSearch=128         0.1876 0.5789 0.8620      0.03227      270951879    10
nprobe=128,quantizer_efSearch=64         0.1880 0.5874 0.8912      0.04277      535884949    8
nprobe=256,quantizer_efSearch=64         0.1883 0.5908 0.9064      0.08274     1057900561    4
nprobe=512,quantizer_efSearch=128        0.1892 0.5925 0.9154      0.15106     2078300265    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 1710443024

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.0362 0.1132 0.1692      0.00130        4525138    231
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=8  0.0369 0.1172 0.1731      0.00131        5716039    229
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8  0.0465 0.1447 0.2167      0.00132        5712500    229
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0505 0.1707 0.2863      0.00152        9412241    198
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4  0.0515 0.1788 0.3053      0.00153        9388770    197
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.0525 0.1784 0.2982      0.00153       10067458    196
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4  0.0538 0.2022 0.3745      0.00186       18175493    162
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4 0.0572 0.2087 0.3892      0.00217       18111790    139
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4 0.0573 0.2247 0.4541      0.00324       35485342    93
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 1710370832

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1220 0.2778 0.3215      0.00197       10061796    153
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1086 0.2423 0.2770      0.00145        9457877    208
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1211 0.2769 0.3209      0.00137        9383165    220
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1213 0.2778 0.3225      0.00139        9382005    217
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1381 0.3506 0.4266      0.00166       18156613    181
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1411 0.3611 0.4407      0.00168       18118901    179
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1422 0.3628 0.4422      0.00178       18098322    169
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1502 0.3817 0.4661      0.00203       18736139    148
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1503 0.3824 0.4674      0.00215       18717034    140
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1540 0.4173 0.5360      0.00226       35611559    133
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1561 0.4243 0.5460      0.00229       35494536    131
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1637 0.4458 0.5755      0.00276       36160574    109
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1661 0.4553 0.5907      0.00295       36031188    102
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1687 0.4533 0.5854      0.00323       37396775    93
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1706 0.4633 0.6016      0.00356       37287126    85
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1712 0.4645 0.6037      0.00363       37260739    83
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.1742 0.4862 0.6545      0.00417       72529289    72
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1759 0.5043 0.6886      0.00397       70640224    76
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1765 0.5058 0.6899      0.00434       70609477    70
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1810 0.5158 0.7072      0.00467       71768653    65
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1813 0.5166 0.7093      0.00524       71712401    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1817 0.5191 0.7126      0.00584       74252432    52
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.1839 0.5319 0.7521      0.00622      141814367    49
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1851 0.5479 0.7870      0.00652      140326797    46
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1859 0.5530 0.7944      0.00759      142633262    40
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1861 0.5509 0.7934      0.00786      140015129    39
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1865 0.5543 0.8015      0.00808      142345922    38
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.1868 0.5564 0.8032      0.00990      147365237    31
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.1869 0.5545 0.8017      0.01049      142291779    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.1870 0.5566 0.8031      0.01186      157554725    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1884 0.5742 0.8539      0.01635      277427147    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1890 0.5758 0.8573      0.01781      276981187    17
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.1898 0.5838 0.8918      0.03323      556453075    10
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.1899 0.5881 0.9089      0.06331     1067178801    5
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs` </summary>
Index size 1643817932

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.0543 0.1921 0.3679      0.00163       34333334    184
nprobe=1,quantizer_efSearch=8            0.0470 0.1563 0.2703      0.00125       17147935    240
nprobe=2,quantizer_efSearch=4            0.0508 0.1799 0.3421      0.00144       34439815    209
nprobe=2,quantizer_efSearch=8            0.0543 0.1921 0.3679      0.00161       34333334    187
nprobe=2,quantizer_efSearch=16           0.0546 0.1938 0.3723      0.00202       34263917    149
nprobe=4,quantizer_efSearch=16           0.0563 0.2176 0.4564      0.00259       68550814    116
nprobe=8,quantizer_efSearch=4            0.0567 0.2272 0.5023      0.00298      137206437    101
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fsr` </summary>
Index size 1643833548

 code_size 8

 log filename: autotune.dbbigann100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1657 0.4844 0.6875      0.01273      136736475    24
nprobe=1,quantizer_efSearch=4            0.1022 0.2337 0.2803      0.00146       17156104    206
nprobe=1,quantizer_efSearch=16           0.1088 0.2486 0.2974      0.00247       17118224    122
nprobe=2,quantizer_efSearch=16           0.1361 0.3405 0.4281      0.00290       34277016    104
nprobe=4,quantizer_efSearch=16           0.1529 0.4212 0.5594      0.00371       68593215    81
nprobe=4,quantizer_efSearch=32           0.1543 0.4227 0.5616      0.00489       68515881    62
nprobe=8,quantizer_efSearch=16           0.1646 0.4806 0.6827      0.00494      136931379    61
nprobe=8,quantizer_efSearch=32           0.1655 0.4837 0.6866      0.00609      136788675    50
nprobe=8,quantizer_efSearch=64           0.1656 0.4840 0.6871      0.00826      136748823    37
nprobe=16,quantizer_efSearch=64          0.1731 0.5230 0.7812      0.01042      271558430    29
nprobe=32,quantizer_efSearch=64          0.1753 0.5447 0.8378      0.01420      537953568    22
nprobe=64,quantizer_efSearch=16          0.1765 0.5478 0.8559      0.02374     1065565603    13
nprobe=64,quantizer_efSearch=64          0.1781 0.5556 0.8733      0.02728     1062585442    11
nprobe=256,quantizer_efSearch=128        0.1790 0.5618 0.8947      0.08637     4105372451    4
```

</details>
<details><summary>`OPQ8_64,IVF1048576_HNSW32,PQ8` </summary>
Index size 2162284464

 code_size 8

 log filename: autotune.dbbigann100M.OPQ8_64_IVF1048576_HNSW32_PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1846 0.4426 0.5276      0.01075       18386885    28
nprobe=1,quantizer_efSearch=4,ht=64      0.0877 0.1498 0.1543      0.00283        1158925    106
nprobe=4,quantizer_efSearch=4,ht=64      0.1362 0.2841 0.3127      0.00307        4622679    98
nprobe=4,quantizer_efSearch=8,ht=30      0.1516 0.3083 0.3331      0.00458        4615695    66
nprobe=8,quantizer_efSearch=4,ht=30      0.1682 0.3718 0.4208      0.00638        9242512    47
nprobe=16,quantizer_efSearch=4,ht=32     0.1846 0.4426 0.5276      0.01086       18386885    28
nprobe=16,quantizer_efSearch=16,ht=30    0.1961 0.4764 0.5606      0.01215       18312295    25
nprobe=16,quantizer_efSearch=16,ht=32    0.1982 0.4897 0.5879      0.01239       18312295    25
nprobe=16,quantizer_efSearch=32,ht=30    0.1987 0.4826 0.5690      0.01515       18245586    20
nprobe=32,quantizer_efSearch=8,ht=32     0.2030 0.5200 0.6490      0.02066       36479861    15
nprobe=64,quantizer_efSearch=32,ht=64    0.2158 0.5941 0.7947      0.02161       71791371    14
nprobe=64,quantizer_efSearch=64,ht=64    0.2173 0.6005 0.8044      0.02876       71561707    11
nprobe=64,quantizer_efSearch=128,ht=64   0.2186 0.6039 0.8098      0.03555       71447021    9
nprobe=128,quantizer_efSearch=64,ht=30   0.2212 0.6106 0.8154      0.07783      141399816    4
nprobe=256,quantizer_efSearch=64,ht=32   0.2247 0.6325 0.8803      0.13981      279086876    3
nprobe=256,quantizer_efSearch=512,ht=32  0.2267 0.6405 0.8962      0.20373      277239453    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2271 0.6496 0.9395      0.27709     1070862183    2
nprobe=2048,quantizer_efSearch=256,ht=32 0.2272 0.6459 0.9251      1.06338     2083814995    1
nprobe=4096,quantizer_efSearch=512,ht=64 0.2273 0.6504 0.9452      1.05270     4034484456    1
```

</details>
<details><summary>`OPQ8_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ8` </summary>
Index size 1902626804

 code_size 8

 log filename: autotune.dbbigann100M.OPQ8_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                               0.2115 0.5351 0.6508      0.02079       47242389    15
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=30      0.0791 0.1359 0.1412      0.00242        2708713    124
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=24      0.0883 0.1330 0.1384      0.00292        3760524    103
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=30      0.1003 0.1752 0.1818      0.00304        5161999    99
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=28      0.1042 0.1938 0.2051      0.00315        5030076    96
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=24      0.1105 0.1826 0.1934      0.00369        6080404    82
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=28     0.1170 0.2024 0.2111      0.00391        3758490    77
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=64    0.1592 0.3392 0.3740      0.00475       10158940    64
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=30     0.1805 0.4013 0.4490      0.00721       14759521    42
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=64    0.1856 0.4268 0.4933      0.00962       14740423    32
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64     0.2050 0.5274 0.6618      0.01101       39431709    28
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=28    0.2110 0.5094 0.6058      0.02127       41899841    15
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=64    0.2240 0.6024 0.8052      0.02377       82559090    13
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=512,ht=64  0.2274 0.6289 0.8713      0.06496      304810557    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=64  0.2297 0.6464 0.9273      0.14601      628896698    3
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=64 0.2301 0.6481 0.9400      0.29371     1109697833    2
```

</details>
<details><summary>`OPQ8_64,IVF262144_HNSW32,PQ8` </summary>
Index size 1740647600

 code_size 8

 log filename: autotune.dbbigann100M.OPQ8_64_IVF262144_HNSW32_PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1903 0.4968 0.6426      0.01015       69152865    30
nprobe=1,quantizer_efSearch=4,ht=28      0.0962 0.1789 0.1913      0.00244        4331924    123
nprobe=1,quantizer_efSearch=4,ht=64      0.1025 0.1997 0.2174      0.00244        4331924    124
nprobe=2,quantizer_efSearch=4,ht=30      0.1280 0.2683 0.2996      0.00254        8676365    119
nprobe=4,quantizer_efSearch=4,ht=64      0.1508 0.3480 0.4133      0.00278       17372504    109
nprobe=4,quantizer_efSearch=8,ht=30      0.1642 0.3776 0.4410      0.00394       17327819    77
nprobe=4,quantizer_efSearch=32,ht=64     0.1711 0.4051 0.4801      0.00571       17264147    53
nprobe=8,quantizer_efSearch=8,ht=28      0.1785 0.4296 0.5158      0.00577       34608001    52
nprobe=8,quantizer_efSearch=4,ht=30      0.1796 0.4404 0.5415      0.00578       34651346    52
nprobe=8,quantizer_efSearch=32,ht=30     0.1905 0.4750 0.5832      0.00836       34470196    36
nprobe=16,quantizer_efSearch=8,ht=28     0.1923 0.4925 0.6163      0.00980       69048362    31
nprobe=16,quantizer_efSearch=16,ht=30    0.2011 0.5268 0.6794      0.01070       68935896    29
nprobe=16,quantizer_efSearch=32,ht=30    0.2034 0.5337 0.6889      0.01239       68836944    25
nprobe=16,quantizer_efSearch=64,ht=30    0.2042 0.5355 0.6907      0.01484       68791883    21
nprobe=16,quantizer_efSearch=64,ht=32    0.2046 0.5465 0.7142      0.01538       68791883    20
nprobe=64,quantizer_efSearch=32,ht=64    0.2144 0.6064 0.8669      0.01972      271328819    16
nprobe=64,quantizer_efSearch=64,ht=64    0.2156 0.6096 0.8722      0.02176      270973642    14
nprobe=64,quantizer_efSearch=128,ht=64   0.2158 0.6104 0.8736      0.02741      270827235    11
nprobe=128,quantizer_efSearch=64,ht=30   0.2159 0.6077 0.8758      0.07101      535709754    5
nprobe=256,quantizer_efSearch=64,ht=32   0.2160 0.6211 0.9142      0.13979     1057395869    3
nprobe=256,quantizer_efSearch=512,ht=30  0.2171 0.6142 0.8982      0.17774     1054302919    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2173 0.6275 0.9362      0.26475     4072583279    2
nprobe=4096,quantizer_efSearch=512,ht=64 0.2175 0.6277 0.9370      0.97952    15230312499    1
```

</details>
<details><summary>`OPQ8_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ8` </summary>
Index size 1675869172

 code_size 8

 log filename: autotune.dbbigann100M.OPQ8_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                               0.1644 0.4334 0.5584      0.03273      280659229    10
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=8       0.0017 0.0024 0.0024      0.00252        4562928    120
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=20      0.0435 0.0653 0.0701      0.00260        5717613    116
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=28      0.1110 0.2255 0.2449      0.00272        9108125    111
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=28     0.1171 0.2420 0.2651      0.00276        9078143    109
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=64     0.1241 0.2679 0.3028      0.00289        9078164    104
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=30     0.1354 0.2852 0.3142      0.00299        9377820    101
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=30      0.1400 0.3150 0.3611      0.00366       17870164    83
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=32      0.1468 0.3315 0.3812      0.00382       18865276    79
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64     0.1753 0.4296 0.5221      0.00403       37652111    75
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=64      0.1878 0.4795 0.5958      0.00432       36056919    70
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=64     0.1888 0.4954 0.6292      0.00551       71466575    55
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=32     0.1916 0.4864 0.6013      0.00911       45045933    33
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=30    0.1917 0.4933 0.6157      0.01092       75072843    28
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=30   0.2002 0.5382 0.6874      0.01272       74271799    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=30   0.2014 0.5368 0.6845      0.01460       89737148    21
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=64    0.2019 0.5691 0.7715      0.01479      139200334    21
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=64  0.2114 0.5996 0.8179      0.02043      157659922    15
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64   0.2117 0.6169 0.8871      0.03110      541385878    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=30    0.2123 0.5997 0.8273      0.03644      282503405    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=32    0.2131 0.6111 0.8593      0.03945      276886096    8
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=30  0.2139 0.6052 0.8370      0.06861      312374337    5
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=64  0.2166 0.6344 0.9294      0.10449     1066800014    3
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=64 0.2168 0.6359 0.9365      0.15479     2115669746    2
```

</details>
<details><summary>`OPQ8_64,IVF65536_HNSW32,PQ8` </summary>
Index size 1635237552

 code_size 8

 log filename: autotune.dbbigann100M.OPQ8_64_IVF65536_HNSW32_PQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1185 0.2558 0.2954      0.00254       17144449    119
nprobe=1,quantizer_efSearch=4,ht=28      0.1072 0.2243 0.2535      0.00211       17178716    143
nprobe=1,quantizer_efSearch=4,ht=64      0.1122 0.2420 0.2819      0.00217       17178716    139
nprobe=1,quantizer_efSearch=8,ht=28      0.1139 0.2394 0.2695      0.00232       17144449    130
nprobe=1,quantizer_efSearch=8,ht=30      0.1172 0.2500 0.2871      0.00240       17144449    125
nprobe=1,quantizer_efSearch=8,ht=32      0.1185 0.2558 0.2954      0.00257       17144449    117
nprobe=2,quantizer_efSearch=8,ht=26      0.1349 0.2968 0.3421      0.00264       34313217    114
nprobe=2,quantizer_efSearch=4,ht=30      0.1363 0.3192 0.3846      0.00302       34412887    100
nprobe=2,quantizer_efSearch=8,ht=30      0.1453 0.3418 0.4126      0.00322       34313217    94
nprobe=2,quantizer_efSearch=16,ht=64     0.1494 0.3534 0.4333      0.00332       34267905    91
nprobe=4,quantizer_efSearch=4,ht=64      0.1558 0.3969 0.5129      0.00395       68847584    77
nprobe=4,quantizer_efSearch=16,ht=28     0.1653 0.4123 0.5160      0.00467       68555992    65
nprobe=4,quantizer_efSearch=8,ht=30      0.1659 0.4241 0.5403      0.00488       68671915    62
nprobe=4,quantizer_efSearch=32,ht=64     0.1698 0.4386 0.5696      0.00514       68490850    59
nprobe=4,quantizer_efSearch=64,ht=64     0.1699 0.4393 0.5704      0.00650       68475714    47
nprobe=8,quantizer_efSearch=8,ht=28      0.1782 0.4726 0.6227      0.00702      137080351    43
nprobe=8,quantizer_efSearch=4,ht=30      0.1786 0.4826 0.6443      0.00799      137233997    38
nprobe=8,quantizer_efSearch=32,ht=30     0.1833 0.5020 0.6769      0.00910      136733020    33
nprobe=8,quantizer_efSearch=64,ht=32     0.1847 0.5085 0.6933      0.01128      136698883    27
nprobe=16,quantizer_efSearch=8,ht=28     0.1873 0.5193 0.7091      0.01230      272234238    25
nprobe=16,quantizer_efSearch=16,ht=30    0.1905 0.5433 0.7618      0.01469      271900569    21
nprobe=16,quantizer_efSearch=32,ht=30    0.1919 0.5481 0.7693      0.01524      271609368    20
nprobe=16,quantizer_efSearch=64,ht=30    0.1924 0.5492 0.7707      0.01660      271507377    19
nprobe=16,quantizer_efSearch=64,ht=32    0.1932 0.5540 0.7874      0.01846      271507377    17
nprobe=32,quantizer_efSearch=128,ht=64   0.1992 0.5821 0.8601      0.02248      537843876    14
nprobe=64,quantizer_efSearch=64,ht=64    0.2022 0.5957 0.8991      0.03513     1062510083    9
nprobe=256,quantizer_efSearch=512,ht=30  0.2024 0.5951 0.8977      0.21304     4101511121    2
nprobe=512,quantizer_efSearch=32,ht=64   0.2028 0.5977 0.9137      0.23056     7784295675    2
nprobe=256,quantizer_efSearch=512,ht=32  0.2031 0.5990 0.9141      0.25538     4101511121    2
nprobe=1024,quantizer_efSearch=256,ht=64 0.2033 0.6009 0.9243      0.48085    15751281423    1
nprobe=4096,quantizer_efSearch=512,ht=64 0.2034 0.6011 0.9244      1.80809    58242226968    1
```

</details>

## Code sizes in [9, 16]

<details><summary>`IVF1048576_HNSW32,PQ16` </summary>
Index size 3230752617

 code_size 16

 log filename: autotune.dbbigann100M.IVF1048576_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3482 0.6858 0.7165      0.03869       36560459    8
nprobe=2,quantizer_efSearch=4,ht=128      0.1578 0.2260 0.2277      0.00361        2335701    84
nprobe=4,quantizer_efSearch=4,ht=50       0.1686 0.2417 0.2439      0.00568        4662299    53
nprobe=4,quantizer_efSearch=8,ht=52       0.2048 0.2987 0.3010      0.00685        4656806    44
nprobe=8,quantizer_efSearch=8,ht=128      0.2699 0.4583 0.4688      0.00708        9314474    43
nprobe=8,quantizer_efSearch=16,ht=60      0.2801 0.4773 0.4870      0.01339        9267371    23
nprobe=16,quantizer_efSearch=4,ht=62      0.2930 0.5265 0.5425      0.01800       18557814    17
nprobe=32,quantizer_efSearch=16,ht=128    0.3440 0.6789 0.7119      0.01967       36705997    16
nprobe=32,quantizer_efSearch=32,ht=60     0.3482 0.6858 0.7165      0.03883       36560459    8
nprobe=128,quantizer_efSearch=16,ht=128   0.3663 0.7594 0.8182      0.05762      143844885    6
nprobe=64,quantizer_efSearch=128,ht=60    0.3715 0.7662 0.8153      0.08657       72040267    4
nprobe=64,quantizer_efSearch=128,ht=62    0.3722 0.7708 0.8225      0.08799       72040267    4
nprobe=128,quantizer_efSearch=256,ht=128  0.3873 0.8248 0.8991      0.10592      142092096    3
nprobe=256,quantizer_efSearch=128,ht=58   0.3920 0.8369 0.9120      0.25407      280535968    2
nprobe=256,quantizer_efSearch=128,ht=64   0.3938 0.8524 0.9394      0.26907      280535968    2
nprobe=512,quantizer_efSearch=512,ht=128  0.3968 0.8702 0.9709      0.30231      549190299    1
nprobe=2048,quantizer_efSearch=256,ht=128 0.3970 0.8774 0.9856      0.84779     2106645659    1
nprobe=1024,quantizer_efSearch=512,ht=60  0.3974 0.8715 0.9706      1.00763     1077265226    1
nprobe=4096,quantizer_efSearch=512,ht=128 0.3977 0.8800 0.9908      1.69753     4080828893    1
```

</details>
<details><summary>`IVF1048576_HNSW32,PQ32x4fs` </summary>
Index size 3499698053

 code_size 16

 log filename: autotune.dbbigann100M.IVF1048576_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2078 0.5800 0.8456      0.02694      142749615    12
nprobe=8,quantizer_efSearch=4            0.1557 0.3629 0.4497      0.00394        9330225    77
nprobe=16,quantizer_efSearch=4           0.1670 0.4097 0.5329      0.00461       18570354    66
nprobe=16,quantizer_efSearch=8           0.1791 0.4422 0.5766      0.00597       18523111    51
nprobe=32,quantizer_efSearch=8           0.1798 0.4799 0.6501      0.00718       36853116    42
nprobe=16,quantizer_efSearch=16          0.1835 0.4582 0.5959      0.00797       18465431    38
nprobe=32,quantizer_efSearch=16          0.1902 0.5064 0.6880      0.00933       36701887    33
nprobe=64,quantizer_efSearch=16          0.1937 0.5289 0.7459      0.01182       72776513    26
nprobe=32,quantizer_efSearch=32          0.1955 0.5171 0.7034      0.01355       36560840    23
nprobe=64,quantizer_efSearch=32          0.1990 0.5470 0.7746      0.01622       72443724    19
nprobe=128,quantizer_efSearch=32         0.2039 0.5659 0.8250      0.01927      143378563    16
nprobe=128,quantizer_efSearch=64         0.2078 0.5800 0.8456      0.02762      142749615    11
nprobe=128,quantizer_efSearch=128        0.2094 0.5843 0.8528      0.04080      142282616    8
```

</details>
<details><summary>`IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ16` </summary>
Index size 2988403117

 code_size 16

 log filename: autotune.dbbigann100M.IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.3745 0.7756 0.8356      0.11656      151606751    3
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=58       0.0941 0.1241 0.1248      0.00298        1536031    101
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=52       0.1035 0.1316 0.1321      0.00331        2610349    91
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=62      0.1142 0.1533 0.1542      0.00332        1892252    91
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=62       0.1276 0.1808 0.1823      0.00340        2705170    89
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=58       0.1282 0.1813 0.1830      0.00347        2695131    87
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=60       0.1742 0.2525 0.2543      0.00421        5167913    72
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60     0.1777 0.2576 0.2596      0.00572        7883741    53
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=50       0.1805 0.2584 0.2610      0.00566        6094328    54
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=52      0.1856 0.2626 0.2644      0.00665       10224953    46
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64      0.2086 0.3208 0.3244      0.00696       10228749    44
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=128     0.2793 0.4835 0.4945      0.00880       20123464    35
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128     0.2822 0.4922 0.5040      0.01011       20046562    30
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=62      0.2942 0.5258 0.5425      0.01707       21529578    18
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=56     0.3036 0.5429 0.5583      0.02291       21247876    14
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=64    0.3217 0.5959 0.6167      0.02460       23885016    13
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=58    0.3221 0.5888 0.6060      0.02889       59583070    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=58     0.3468 0.6693 0.6979      0.03221       42216456    10
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=58    0.3489 0.6744 0.7041      0.04483       42030068    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128   0.3744 0.7762 0.8328      0.05037      153064278    6
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62   0.3745 0.7733 0.8276      0.10870      227160918    3
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56    0.3746 0.7653 0.8178      0.12511      148191765    3
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=64   0.3750 0.7758 0.8314      0.15478      235663462    2
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62  0.3866 0.8239 0.8987      0.18615      302975508    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128  0.3967 0.8699 0.9750      0.24650      630118021    2
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=64 0.3981 0.8770 0.9871      1.11709     1115505095    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=64  0.3982 0.8795 0.9921      1.91754     2261484312    1
```

</details>
<details><summary>`IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ32x4fs` </summary>
Index size 3257433033

 code_size 16

 log filename: autotune.dbbigann100M.IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.1422 0.3279 0.3993      0.00291       10824831    104
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1    0.0612 0.1061 0.1144      0.00160        1541629    188
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1    0.0699 0.1229 0.1317      0.00173        1533811    174
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2    0.0765 0.1363 0.1472      0.00169        1899739    178
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0795 0.1420 0.1524      0.00173        1894582    174
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1    0.0841 0.1628 0.1828      0.00178        2705162    169
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2    0.0905 0.1761 0.1967      0.00192        3068296    157
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.1002 0.2013 0.2255      0.00196        3055120    154
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2    0.1092 0.2318 0.2692      0.00222        5433686    136
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2    0.1153 0.2558 0.3021      0.00225        5411117    134
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1295 0.2859 0.3343      0.00251        6123030    120
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1386 0.3037 0.3546      0.00291        7497783    104
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.1422 0.3279 0.3993      0.00293       10824831    103
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1497 0.3545 0.4376      0.00305       10782366    99
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1535 0.3531 0.4288      0.00341       12188395    88
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1620 0.3857 0.4777      0.00400       12107367    76
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.1676 0.4169 0.5347      0.00429       21532700    70
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1689 0.3944 0.4856      0.00455       14843961    66
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.1836 0.4596 0.5958      0.00567       24020796    53
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1852 0.4645 0.6034      0.00623       23941156    49
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.1856 0.4650 0.6043      0.00737       23918269    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1955 0.5166 0.7046      0.00872       42068974    35
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.1965 0.5256 0.7146      0.01108       47169212    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.1985 0.5455 0.7719      0.01101       78095890    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2012 0.5581 0.7891      0.01312       83075973    23
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2022 0.5607 0.7923      0.01710       93215378    18
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2025 0.5609 0.7921      0.02201      113621232    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32 0.2085 0.5820 0.8513      0.02370      152915230    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.2108 0.5872 0.8581      0.02747      162654612    11
```

</details>
<details><summary>`IVF262144_HNSW32,PQ16` </summary>
Index size 2607789161

 code_size 16

 log filename: autotune.dbbigann100M.IVF262144_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3736 0.7571 0.8242      0.04077      136877891    8
nprobe=2,quantizer_efSearch=4,ht=128     0.1941 0.3057 0.3134      0.00334        8676003    90
nprobe=4,quantizer_efSearch=4,ht=50      0.2084 0.3314 0.3402      0.00638       17383723    48
nprobe=4,quantizer_efSearch=8,ht=50      0.2325 0.3692 0.3785      0.00658       17318829    46
nprobe=4,quantizer_efSearch=8,ht=52      0.2480 0.4033 0.4139      0.00681       17318829    45
nprobe=4,quantizer_efSearch=8,ht=64      0.2627 0.4527 0.4681      0.00775       17318829    39
nprobe=8,quantizer_efSearch=8,ht=128     0.3013 0.5540 0.5819      0.00778       34631901    39
nprobe=8,quantizer_efSearch=16,ht=56     0.3100 0.5604 0.5848      0.01244       34528619    25
nprobe=8,quantizer_efSearch=16,ht=60     0.3135 0.5764 0.6037      0.01293       34528619    24
nprobe=8,quantizer_efSearch=32,ht=64     0.3165 0.5840 0.6127      0.01635       34462863    19
nprobe=16,quantizer_efSearch=4,ht=62     0.3192 0.6122 0.6522      0.01940       69193061    16
nprobe=16,quantizer_efSearch=16,ht=60    0.3475 0.6709 0.7152      0.02111       68954980    15
nprobe=32,quantizer_efSearch=16,ht=128   0.3708 0.7505 0.8184      0.02200      137131338    14
nprobe=32,quantizer_efSearch=32,ht=60    0.3736 0.7571 0.8242      0.04010      136877891    8
nprobe=32,quantizer_efSearch=256,ht=58   0.3742 0.7569 0.8198      0.06934      136599470    5
nprobe=64,quantizer_efSearch=32,ht=56    0.3853 0.7852 0.8605      0.07149      271524925    5
nprobe=64,quantizer_efSearch=128,ht=62   0.3907 0.8118 0.9002      0.08940      270902618    4
nprobe=64,quantizer_efSearch=128,ht=60   0.3908 0.8099 0.8971      0.08911      270902618    4
nprobe=128,quantizer_efSearch=256,ht=128 0.3960 0.8418 0.9487      0.11234      535372006    3
nprobe=128,quantizer_efSearch=128,ht=62  0.3964 0.8401 0.9454      0.15522      535599897    2
nprobe=256,quantizer_efSearch=128,ht=58  0.3977 0.8472 0.9570      0.27441     1057001540    2
nprobe=256,quantizer_efSearch=128,ht=64  0.3984 0.8550 0.9729      0.28983     1057001540    2
nprobe=512,quantizer_efSearch=512,ht=128 0.3995 0.8619 0.9871      0.33006     2076798340    1
nprobe=512,quantizer_efSearch=512,ht=60  0.3998 0.8587 0.9809      0.57046     2076798340    1
nprobe=1024,quantizer_efSearch=256,ht=60 0.3999 0.8593 0.9841      1.03035     4082368773    1
nprobe=1024,quantizer_efSearch=512,ht=60 0.4002 0.8599 0.9853      1.07778     4077315210    1
```

</details>
<details><summary>`IVF262144_HNSW32,PQ32x4fs` </summary>
Index size 2674732165

 code_size 16

 log filename: autotune.dbbigann100M.IVF262144_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1550 0.3682 0.4588      0.00331       17321846    91
nprobe=2,quantizer_efSearch=4            0.1179 0.2560 0.3062      0.00189        8690273    159
nprobe=4,quantizer_efSearch=4            0.1384 0.3263 0.4068      0.00229       17386447    132
nprobe=4,quantizer_efSearch=8            0.1550 0.3682 0.4588      0.00319       17321846    94
nprobe=8,quantizer_efSearch=4            0.1660 0.4215 0.5484      0.00331       34643497    91
nprobe=8,quantizer_efSearch=8            0.1709 0.4354 0.5667      0.00377       34613977    80
nprobe=16,quantizer_efSearch=4           0.1738 0.4659 0.6267      0.00486       69152134    62
nprobe=8,quantizer_efSearch=16           0.1763 0.4556 0.5931      0.00543       34517355    56
nprobe=16,quantizer_efSearch=8           0.1858 0.4967 0.6741      0.00510       69033684    59
nprobe=16,quantizer_efSearch=16          0.1894 0.5100 0.6930      0.00646       68924460    47
nprobe=32,quantizer_efSearch=8           0.1912 0.5240 0.7370      0.00674      137399486    45
nprobe=32,quantizer_efSearch=16          0.1974 0.5483 0.7749      0.00877      137116111    35
nprobe=32,quantizer_efSearch=32          0.1987 0.5563 0.7863      0.01092      136852173    28
nprobe=64,quantizer_efSearch=32          0.2005 0.5781 0.8389      0.01422      271518705    22
nprobe=64,quantizer_efSearch=64          0.2019 0.5832 0.8462      0.01876      271115869    17
```

</details>
<details><summary>`IVF262144(IVF512,PQ64x4fs,RFlat),PQ16` </summary>
Index size 2547478957

 code_size 16

 log filename: autotune.dbbigann100M.IVF262144_IVF512_PQ64x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.3879 0.8159 0.8999      0.07597      291544888    4
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=2        0.0002 0.0002 0.0002      0.00259        4694672    116
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=46       0.0979 0.1297 0.1314      0.00262        4688203    115
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=64       0.1337 0.1932 0.1954      0.00261        5047027    115
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=50       0.1341 0.1965 0.1998      0.00313        8922718    96
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=56      0.1489 0.2131 0.2157      0.00351        5019412    86
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=128     0.1588 0.2323 0.2352      0.00360        7021425    84
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60     0.1590 0.2311 0.2339      0.00393        7015062    77
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=48      0.1743 0.2482 0.2516      0.00404       11344098    75
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=128     0.2303 0.3932 0.4066      0.00467       17777788    65
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64       0.2679 0.4592 0.4736      0.00642       18765448    47
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.2694 0.4592 0.4725      0.00783       18679453    39
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=62     0.2737 0.4743 0.4888      0.00987       19970988    31
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=54      0.3023 0.5368 0.5574      0.01268       35963254    24
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=128    0.3185 0.5895 0.6185      0.01472       39779050    21
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=128    0.3451 0.6812 0.7289      0.01503       79520356    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.3737 0.7708 0.8388      0.02407      147147519    13
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128    0.3882 0.8174 0.9046      0.04452      276472825    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=62    0.3888 0.8281 0.9234      0.14444      540874019    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60    0.3893 0.8345 0.9376      0.26872     1067993514    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128   0.3976 0.8647 0.9891      0.31102     2088678426    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=62   0.3981 0.8654 0.9904      1.05163     4096824733    1
nprobe=1024,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=60 0.3982 0.8643 0.9867      1.70115     4114002227    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=128 0.3985 0.8675 0.9963      2.08797    15679178330    1
```

</details>
<details><summary>`IVF262144(IVF512,PQ64x4fs,RFlat),PQ32x4fs` </summary>
Index size 2614339529

 code_size 16

 log filename: autotune.dbbigann100M.IVF262144_IVF512_PQ64x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                      0.1868 0.4970 0.6740      0.00452       70757470    67
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1   0.0801 0.1549 0.1766      0.00124        4533782    242
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1   0.0810 0.1561 0.1777      0.00133        4531361    227
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.0978 0.1942 0.2210      0.00140        5029719    214
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.0988 0.1959 0.2233      0.00148        5025192    204
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2   0.1054 0.2246 0.2663      0.00143        9103362    210
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4   0.1168 0.2481 0.2918      0.00159        9406983    189
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.1256 0.2689 0.3193      0.00161        9387092    187
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4   0.1283 0.2759 0.3282      0.00183        9350660    165
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.1346 0.2913 0.3448      0.00202       10030492    149
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4   0.1423 0.3272 0.4016      0.00202       18214251    149
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.1504 0.3554 0.4406      0.00205       18142735    147
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4   0.1525 0.3591 0.4463      0.00222       18097086    135
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4   0.1577 0.3958 0.5076      0.00270       35891861    112
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.1607 0.3821 0.4734      0.00297       20053116    101
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8   0.1668 0.4251 0.5429      0.00303       36424227    99
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.1679 0.4315 0.5531      0.00362       37639556    84
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4  0.1718 0.4456 0.5927      0.00391       71239360    77
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.1743 0.4496 0.5808      0.00316       36134395    95
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1792 0.4623 0.6001      0.00391       37244068    77
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8  0.1823 0.4835 0.6471      0.00430       71558587    70
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.1868 0.4970 0.6740      0.00437       70758018    69
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16 0.1886 0.5126 0.6956      0.00493       71866281    61
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.1910 0.5162 0.6999      0.00626       74355409    48
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8  0.1929 0.5310 0.7469      0.00627      139395624    48
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16 0.1965 0.5359 0.7522      0.00696      142335691    44
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.2003 0.5588 0.7893      0.00846      142543056    36
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16 0.2007 0.5748 0.8353      0.01086      275411435    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.2017 0.5816 0.8461      0.01220      277314844    25
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.2031 0.5845 0.8516      0.01609      281316339    19
```

</details>
<details><summary>`IVF65536_HNSW32,PQ16` </summary>
Index size 2452047465

 code_size 16

 log filename: autotune.dbbigann100M.IVF65536_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3615 0.7940 0.8901      0.04618      538008730    7
nprobe=1,quantizer_efSearch=4,ht=60      0.1657 0.2692 0.2760      0.00308       17152932    98
nprobe=2,quantizer_efSearch=4,ht=46      0.1730 0.2696 0.2781      0.00324       34382748    93
nprobe=1,quantizer_efSearch=8,ht=58      0.1771 0.2869 0.2936      0.00324       17119550    93
nprobe=1,quantizer_efSearch=16,ht=58     0.1803 0.2913 0.2981      0.00386       17095184    78
nprobe=1,quantizer_efSearch=16,ht=128    0.1815 0.2956 0.3028      0.00370       17095184    82
nprobe=2,quantizer_efSearch=4,ht=128     0.2173 0.3796 0.3958      0.00415       34382748    73
nprobe=2,quantizer_efSearch=16,ht=52     0.2282 0.3861 0.3992      0.00470       34242041    64
nprobe=4,quantizer_efSearch=4,ht=50      0.2366 0.4215 0.4401      0.00556       68661451    54
nprobe=2,quantizer_efSearch=16,ht=62     0.2372 0.4180 0.4354      0.00560       34242041    54
nprobe=4,quantizer_efSearch=8,ht=50      0.2570 0.4609 0.4803      0.00593       68486621    51
nprobe=4,quantizer_efSearch=16,ht=50     0.2608 0.4685 0.4881      0.00642       68375019    47
nprobe=4,quantizer_efSearch=8,ht=64      0.2777 0.5333 0.5652      0.00818       68486621    37
nprobe=8,quantizer_efSearch=8,ht=48      0.2817 0.5098 0.5342      0.00957      136831978    32
nprobe=8,quantizer_efSearch=8,ht=128     0.3165 0.6418 0.6938      0.00987      136831978    31
nprobe=8,quantizer_efSearch=16,ht=56     0.3225 0.6467 0.6942      0.01193      136630469    26
nprobe=8,quantizer_efSearch=32,ht=56     0.3230 0.6496 0.6973      0.01290      136477152    24
nprobe=8,quantizer_efSearch=16,ht=60     0.3247 0.6582 0.7099      0.01355      136630469    23
nprobe=16,quantizer_efSearch=16,ht=60    0.3453 0.7354 0.8089      0.02392      271735245    13
nprobe=16,quantizer_efSearch=64,ht=64    0.3487 0.7470 0.8237      0.03001      271276848    10
nprobe=32,quantizer_efSearch=16,ht=128   0.3589 0.7895 0.8853      0.03759      538702647    8
nprobe=32,quantizer_efSearch=32,ht=60    0.3615 0.7940 0.8901      0.04459      538008730    7
nprobe=32,quantizer_efSearch=256,ht=58   0.3622 0.7938 0.8883      0.05493      537581097    6
nprobe=32,quantizer_efSearch=256,ht=64   0.3627 0.8005 0.8994      0.06435      537581097    5
nprobe=64,quantizer_efSearch=32,ht=56    0.3644 0.8101 0.9147      0.07221     1064058895    5
nprobe=64,quantizer_efSearch=128,ht=60   0.3674 0.8272 0.9431      0.08736     1062710661    4
nprobe=64,quantizer_efSearch=512,ht=128  0.3682 0.8312 0.9496      0.09330     1062602709    4
nprobe=128,quantizer_efSearch=256,ht=128 0.3710 0.8443 0.9760      0.12342     2092063097    3
nprobe=256,quantizer_efSearch=128,ht=58  0.3718 0.8428 0.9760      0.28481     4110477857    2
nprobe=256,quantizer_efSearch=512,ht=58  0.3720 0.8433 0.9769      0.30882     4107967780    1
nprobe=512,quantizer_efSearch=512,ht=128 0.3731 0.8518 0.9936      0.53456     8055692320    1
```

</details>
<details><summary>`IVF65536_HNSW32,PQ32x4fs` </summary>
Index size 2468761221

 code_size 16

 log filename: autotune.dbbigann100M.IVF65536_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1651 0.4183 0.5470      0.00385       68479507    78
nprobe=2,quantizer_efSearch=4            0.1340 0.3127 0.3878      0.00242       34374421    125
nprobe=2,quantizer_efSearch=8            0.1446 0.3381 0.4200      0.00273       34286453    110
nprobe=2,quantizer_efSearch=16           0.1475 0.3441 0.4267      0.00331       34238920    91
nprobe=4,quantizer_efSearch=4            0.1515 0.3805 0.4962      0.00347       68625881    87
nprobe=4,quantizer_efSearch=8            0.1651 0.4183 0.5470      0.00383       68479507    79
nprobe=4,quantizer_efSearch=16           0.1682 0.4260 0.5568      0.00437       68370381    69
nprobe=8,quantizer_efSearch=4            0.1767 0.4709 0.6476      0.00523      136935961    58
nprobe=8,quantizer_efSearch=8            0.1814 0.4832 0.6648      0.00526      136812349    58
nprobe=8,quantizer_efSearch=16           0.1868 0.4967 0.6843      0.00585      136622814    52
nprobe=8,quantizer_efSearch=32           0.1875 0.4994 0.6885      0.00707      136470427    43
nprobe=16,quantizer_efSearch=8           0.1914 0.5336 0.7580      0.00736      272111676    41
nprobe=16,quantizer_efSearch=16          0.1923 0.5426 0.7730      0.00800      271733407    38
nprobe=32,quantizer_efSearch=8           0.1967 0.5547 0.8065      0.00949      539867179    32
nprobe=32,quantizer_efSearch=16          0.1988 0.5687 0.8322      0.00974      538654762    31
nprobe=32,quantizer_efSearch=32          0.2012 0.5760 0.8433      0.01081      537985499    29
nprobe=32,quantizer_efSearch=64          0.2024 0.5775 0.8453      0.01524      537674417    20
nprobe=64,quantizer_efSearch=32          0.2028 0.5889 0.8792      0.01525     1064032467    20
nprobe=32,quantizer_efSearch=128         0.2030 0.5784 0.8464      0.02033      537590528    15
nprobe=64,quantizer_efSearch=128         0.2049 0.5929 0.8851      0.02606     1062689884    12
nprobe=128,quantizer_efSearch=128        0.2071 0.5975 0.9026      0.02970     2092307879    11
nprobe=256,quantizer_efSearch=128        0.2074 0.5993 0.9101      0.03704     4110399404    9
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16` </summary>
Index size 2962284464

 code_size 16

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3768 0.6838 0.7070      0.03115       36269600    10
nprobe=2,quantizer_efSearch=4,ht=128      0.1611 0.2238 0.2256      0.00296        2319566    102
nprobe=4,quantizer_efSearch=8,ht=50       0.1689 0.2279 0.2298      0.00521        4625139    58
nprobe=4,quantizer_efSearch=8,ht=52       0.1910 0.2634 0.2654      0.00530        4625139    57
nprobe=4,quantizer_efSearch=8,ht=64       0.2324 0.3510 0.3544      0.00563        4625139    54
nprobe=8,quantizer_efSearch=8,ht=128      0.2783 0.4575 0.4647      0.00550        9240115    55
nprobe=8,quantizer_efSearch=16,ht=56      0.2817 0.4380 0.4431      0.00999        9193256    31
nprobe=8,quantizer_efSearch=16,ht=60      0.2926 0.4735 0.4801      0.00992        9193256    31
nprobe=8,quantizer_efSearch=32,ht=64      0.2986 0.4867 0.4945      0.01317        9159574    23
nprobe=16,quantizer_efSearch=4,ht=62      0.3079 0.5255 0.5378      0.01448       18393420    21
nprobe=32,quantizer_efSearch=16,ht=128    0.3714 0.6799 0.7061      0.01470       36384169    21
nprobe=32,quantizer_efSearch=32,ht=60     0.3768 0.6838 0.7070      0.02960       36269600    11
nprobe=128,quantizer_efSearch=16,ht=128   0.4006 0.7710 0.8168      0.04338      142516548    7
nprobe=64,quantizer_efSearch=128,ht=62    0.4092 0.7780 0.8178      0.06711       71522585    5
nprobe=128,quantizer_efSearch=256,ht=128  0.4274 0.8425 0.8996      0.07848      141058684    4
nprobe=256,quantizer_efSearch=128,ht=58   0.4294 0.8445 0.8977      0.20825      278301157    2
nprobe=256,quantizer_efSearch=128,ht=64   0.4332 0.8718 0.9401      0.20743      278301157    2
nprobe=512,quantizer_efSearch=512,ht=128  0.4374 0.8925 0.9721      0.24586      545083377    2
nprobe=4096,quantizer_efSearch=512,ht=128 0.4395 0.9045 0.9954      1.33114     4041055155    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16` </summary>
Index size 2702633460

 code_size 16

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.3944 0.7726 0.8137      0.09414      150090504    4
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=12        0.0003 0.0003 0.0003      0.00248        1530721    122
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=58        0.0906 0.1201 0.1202      0.00254        1533428    119
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=52        0.0930 0.1174 0.1174      0.00262        2606862    115
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=58        0.1295 0.1793 0.1803      0.00280        2693384    108
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=60        0.1749 0.2455 0.2466      0.00328        5149779    92
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=60      0.1829 0.2560 0.2575      0.00425        7861711    71
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64       0.1957 0.2912 0.2929      0.00553       10188492    55
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=54       0.1984 0.2893 0.2911      0.00545        6074192    56
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=128      0.2782 0.4603 0.4673      0.00747       20088842    41
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128      0.2943 0.4898 0.4979      0.00732       20032467    41
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=56      0.3207 0.5362 0.5460      0.01828       21208323    17
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=64     0.3405 0.6006 0.6164      0.01975       23871410    16
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=58      0.3617 0.6554 0.6741      0.02638       42060379    12
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=58     0.3641 0.6703 0.6922      0.04464       41896344    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=58      0.3873 0.7395 0.7724      0.05385       77551364    6
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128    0.3953 0.7786 0.8230      0.03889      154010529    8
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62    0.3957 0.7760 0.8189      0.08314      235218131    4
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=64    0.3966 0.7793 0.8232      0.12811      235216961    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=62   0.4123 0.8360 0.8927      0.13614      304855073    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=128   0.4248 0.8933 0.9738      0.20092      628061562    2
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=64  0.4258 0.8996 0.9854      0.89343     1111147155    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=64   0.4261 0.9026 0.9923      1.55820     2247233319    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=128 0.4265 0.9050 0.9985      3.73839     4240199164    1
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16` </summary>
Index size 2540647600

 code_size 16

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3923 0.7759 0.8194      0.02961      136893191    11
nprobe=2,quantizer_efSearch=4,ht=46      0.1192 0.1576 0.1595      0.00249        8672707    121
nprobe=2,quantizer_efSearch=4,ht=128     0.2043 0.3112 0.3154      0.00262        8672707    115
nprobe=2,quantizer_efSearch=16,ht=54     0.2120 0.3079 0.3113      0.00398        8626216    76
nprobe=2,quantizer_efSearch=16,ht=62     0.2271 0.3450 0.3494      0.00434        8626216    70
nprobe=4,quantizer_efSearch=8,ht=52      0.2438 0.3744 0.3796      0.00463       17321476    65
nprobe=4,quantizer_efSearch=8,ht=64      0.2734 0.4564 0.4648      0.00549       17321476    55
nprobe=8,quantizer_efSearch=8,ht=128     0.3212 0.5713 0.5886      0.00572       34576866    53
nprobe=8,quantizer_efSearch=16,ht=56     0.3235 0.5620 0.5754      0.00912       34484453    33
nprobe=8,quantizer_efSearch=16,ht=60     0.3312 0.5886 0.6055      0.00924       34484453    33
nprobe=8,quantizer_efSearch=32,ht=64     0.3357 0.5987 0.6176      0.01161       34438629    26
nprobe=16,quantizer_efSearch=4,ht=62     0.3364 0.6306 0.6559      0.01556       69107401    20
nprobe=16,quantizer_efSearch=16,ht=60    0.3611 0.6861 0.7145      0.01582       68915281    19
nprobe=32,quantizer_efSearch=16,ht=128   0.3871 0.7705 0.8171      0.01815      137109397    17
nprobe=32,quantizer_efSearch=32,ht=60    0.3923 0.7759 0.8194      0.03045      136893191    10
nprobe=32,quantizer_efSearch=256,ht=64   0.3948 0.7873 0.8343      0.05125      136718644    6
nprobe=64,quantizer_efSearch=32,ht=56    0.4013 0.7952 0.8419      0.05368      271429654    6
nprobe=128,quantizer_efSearch=16,ht=128  0.4035 0.8315 0.9006      0.06310      537978050    5
nprobe=64,quantizer_efSearch=128,ht=62   0.4090 0.8380 0.9000      0.06476      270909381    5
nprobe=128,quantizer_efSearch=256,ht=128 0.4195 0.8735 0.9519      0.08040      535282917    4
nprobe=256,quantizer_efSearch=128,ht=64  0.4218 0.8897 0.9760      0.23828     1056051298    2
nprobe=512,quantizer_efSearch=128,ht=60  0.4233 0.8910 0.9772      0.41014     2079149370    1
nprobe=512,quantizer_efSearch=512,ht=60  0.4242 0.8925 0.9793      0.44397     2074533052    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16` </summary>
Index size 2475867636

 code_size 16

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.4091 0.8342 0.8922      0.05927      292063200    6
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=46       0.0832 0.1067 0.1077      0.00256        4698767    117
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=58       0.1229 0.1706 0.1713      0.00266        5049801    113
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=64       0.1250 0.1760 0.1769      0.00258        5049797    117
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=50       0.1277 0.1764 0.1785      0.00283        8942615    106
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=56      0.1523 0.2110 0.2123      0.00295        5026577    102
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=128     0.1651 0.2340 0.2355      0.00318        7027489    95
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=2,ht=58      0.1889 0.2902 0.2938      0.00361        9071254    84
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=62      0.2227 0.3447 0.3487      0.00382       11357125    79
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=128     0.2385 0.3935 0.4032      0.00398       17819211    76
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=64       0.2661 0.4421 0.4514      0.00563       18799533    54
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=58      0.2715 0.4486 0.4579      0.00579       18726237    52
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=62     0.2799 0.4680 0.4784      0.00753       20017943    40
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=8,ht=54      0.3096 0.5230 0.5356      0.00952       36023294    32
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32,ht=128    0.3334 0.6000 0.6199      0.01254       39812865    24
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=128    0.3601 0.6859 0.7156      0.01255       79567159    24
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.3944 0.7882 0.8345      0.02111      147505611    15
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=128   0.3947 0.7895 0.8361      0.02644      178017201    12
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=128    0.4099 0.8389 0.9023      0.03696      276979496    9
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128   0.4118 0.8419 0.9058      0.04293      281803281    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=60    0.4127 0.8603 0.9329      0.21110     1068515081    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56  0.4148 0.8521 0.9183      0.22759     1077043750    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=60  0.4206 0.8840 0.9666      0.23938     1097406724    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128   0.4238 0.8976 0.9909      0.25235     2089280441    2
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=128 0.4243 0.9014 0.9986      1.69141    15661535023    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16` </summary>
Index size 2435237552

 code_size 16

 log filename: autotune.dbbigann100M.OPQ16_64_IVF65536_HNSW32_PQ16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3920 0.8224 0.8940      0.03785      538450980    8
nprobe=1,quantizer_efSearch=8,ht=34       0.0230 0.0307 0.0319      0.00249       17146294    121
nprobe=1,quantizer_efSearch=4,ht=60       0.1798 0.2762 0.2817      0.00254       17160721    119
nprobe=1,quantizer_efSearch=8,ht=58       0.1881 0.2896 0.2951      0.00259       17146294    116
nprobe=1,quantizer_efSearch=16,ht=58      0.1896 0.2925 0.2980      0.00296       17123045    102
nprobe=1,quantizer_efSearch=16,ht=128     0.1911 0.2965 0.3026      0.00307       17123045    98
nprobe=2,quantizer_efSearch=16,ht=54      0.2458 0.3987 0.4078      0.00377       34263640    80
nprobe=2,quantizer_efSearch=16,ht=62      0.2543 0.4258 0.4386      0.00463       34263640    65
nprobe=4,quantizer_efSearch=8,ht=50       0.2653 0.4359 0.4467      0.00485       68686619    62
nprobe=4,quantizer_efSearch=8,ht=52       0.2821 0.4794 0.4930      0.00514       68686619    59
nprobe=4,quantizer_efSearch=8,ht=64       0.3004 0.5431 0.5654      0.00723       68686619    42
nprobe=8,quantizer_efSearch=8,ht=128      0.3398 0.6575 0.6965      0.00938      137100821    32
nprobe=8,quantizer_efSearch=16,ht=56      0.3405 0.6537 0.6880      0.01001      136866520    30
nprobe=8,quantizer_efSearch=32,ht=56      0.3415 0.6571 0.6912      0.01061      136756586    29
nprobe=8,quantizer_efSearch=16,ht=60      0.3452 0.6707 0.7102      0.01137      136866520    27
nprobe=8,quantizer_efSearch=32,ht=64      0.3464 0.6774 0.7177      0.01380      136756586    22
nprobe=16,quantizer_efSearch=4,ht=62      0.3584 0.7114 0.7627      0.02161      272598610    14
nprobe=16,quantizer_efSearch=16,ht=60     0.3718 0.7578 0.8127      0.02041      271997358    15
nprobe=16,quantizer_efSearch=64,ht=64     0.3747 0.7684 0.8252      0.02521      271625771    12
nprobe=32,quantizer_efSearch=8,ht=54      0.3751 0.7474 0.7971      0.03185      540138283    10
nprobe=32,quantizer_efSearch=16,ht=128    0.3900 0.8161 0.8896      0.03130      539205858    10
nprobe=32,quantizer_efSearch=32,ht=60     0.3920 0.8224 0.8940      0.03814      538450980    8
nprobe=32,quantizer_efSearch=256,ht=64    0.3939 0.8275 0.9022      0.05673      538074041    6
nprobe=64,quantizer_efSearch=32,ht=56     0.3982 0.8354 0.9100      0.06364     1063892817    5
nprobe=64,quantizer_efSearch=128,ht=60    0.4031 0.8591 0.9459      0.07527     1062647610    4
nprobe=64,quantizer_efSearch=128,ht=62    0.4039 0.8605 0.9497      0.08142     1062647610    4
nprobe=64,quantizer_efSearch=512,ht=128   0.4041 0.8619 0.9522      0.09042     1062543522    4
nprobe=128,quantizer_efSearch=256,ht=128  0.4083 0.8792 0.9801      0.12801     2090857691    3
nprobe=256,quantizer_efSearch=128,ht=64   0.4094 0.8857 0.9913      0.31999     4106265311    1
nprobe=512,quantizer_efSearch=512,ht=128  0.4101 0.8889 0.9968      0.46397     8043809388    1
nprobe=2048,quantizer_efSearch=256,ht=128 0.4103 0.8890 0.9974      1.65387    30262717972    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576_HNSW32,PQ32x4fs` </summary>
Index size 3231190988

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF1048576_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1328 0.3005 0.3526      0.00272        4661143    111
nprobe=2,quantizer_efSearch=4            0.0979 0.1983 0.2244      0.00170        2342522    177
nprobe=4,quantizer_efSearch=4            0.1178 0.2627 0.3089      0.00197        4680736    153
nprobe=4,quantizer_efSearch=8            0.1328 0.3005 0.3526      0.00272        4661143    111
nprobe=8,quantizer_efSearch=4            0.1524 0.3638 0.4445      0.00277        9317464    109
nprobe=8,quantizer_efSearch=8            0.1559 0.3738 0.4578      0.00316        9309683    95
nprobe=16,quantizer_efSearch=4           0.1684 0.4160 0.5315      0.00364       18523435    83
nprobe=16,quantizer_efSearch=8           0.1770 0.4497 0.5743      0.00444       18493490    68
nprobe=16,quantizer_efSearch=16          0.1806 0.4624 0.5912      0.00536       18446027    57
nprobe=32,quantizer_efSearch=8           0.1837 0.4897 0.6503      0.00591       36718201    51
nprobe=32,quantizer_efSearch=16          0.1912 0.5131 0.6831      0.00783       36570204    39
nprobe=64,quantizer_efSearch=16          0.1935 0.5367 0.7502      0.00894       72489895    34
nprobe=32,quantizer_efSearch=32          0.1955 0.5225 0.6971      0.00891       36440841    34
nprobe=64,quantizer_efSearch=32          0.1994 0.5531 0.7753      0.01140       72187725    27
nprobe=128,quantizer_efSearch=64         0.2043 0.5878 0.8436      0.02087      142133472    15
nprobe=256,quantizer_efSearch=64         0.2044 0.5958 0.8736      0.02575      280570557    12
nprobe=128,quantizer_efSearch=128        0.2051 0.5924 0.8507      0.02791      141719767    11
nprobe=256,quantizer_efSearch=128        0.2058 0.6053 0.8880      0.03542      279350448    9
nprobe=128,quantizer_efSearch=256        0.2065 0.5941 0.8529      0.04524      141549474    7
nprobe=256,quantizer_efSearch=256        0.2073 0.6058 0.8903      0.05139      278712273    6
```

</details>
<details><summary>`OPQ32_64,IVF1048576_HNSW32,PQ32x4fsr` </summary>
Index size 3231199692

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF1048576_HNSW32_PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3311 0.5957 0.6165      0.01332       18264578    23
nprobe=1,quantizer_efSearch=4            0.1121 0.1522 0.1528      0.00238        1158557    127
nprobe=1,quantizer_efSearch=16           0.1266 0.1716 0.1722      0.00577        1148848    53
nprobe=2,quantizer_efSearch=16           0.1772 0.2565 0.2592      0.00601        2298792    50
nprobe=4,quantizer_efSearch=16           0.2280 0.3614 0.3669      0.00667        4597569    45
nprobe=8,quantizer_efSearch=16           0.2843 0.4796 0.4910      0.00745        9191336    41
nprobe=16,quantizer_efSearch=16          0.3262 0.5865 0.6073      0.00941       18329987    32
nprobe=16,quantizer_efSearch=32          0.3311 0.5957 0.6165      0.01302       18264578    24
nprobe=32,quantizer_efSearch=16          0.3536 0.6744 0.7078      0.01933       36389803    16
nprobe=32,quantizer_efSearch=32          0.3599 0.6877 0.7214      0.02182       36270558    14
nprobe=32,quantizer_efSearch=64          0.3644 0.6957 0.7299      0.02767       36185384    11
nprobe=64,quantizer_efSearch=16          0.3729 0.7328 0.7805      0.03040       72139698    10
nprobe=64,quantizer_efSearch=128         0.3893 0.7718 0.8237      0.05181       71525458    6
nprobe=128,quantizer_efSearch=128        0.4059 0.8287 0.8982      0.08903      141217222    4
nprobe=128,quantizer_efSearch=256        0.4060 0.8292 0.8998      0.11379      141062555    3
nprobe=256,quantizer_efSearch=128        0.4098 0.8605 0.9417      0.15400      278324331    2
nprobe=256,quantizer_efSearch=256        0.4108 0.8627 0.9456      0.17073      277752330    2
nprobe=256,quantizer_efSearch=512        0.4110 0.8635 0.9465      0.20103      277563108    2
nprobe=512,quantizer_efSearch=128        0.4115 0.8702 0.9636      0.26214      547961359    2
nprobe=512,quantizer_efSearch=256        0.4130 0.8761 0.9702      0.26685      545978833    2
nprobe=1024,quantizer_efSearch=256       0.4149 0.8835 0.9846      0.47892     1072470102    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 2971788304

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1379 0.3095 0.3651      0.00266       10801938    113
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.0773 0.1440 0.1551      0.00137        1895059    219
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.0793 0.1565 0.1740      0.00144        2704583    209
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.0821 0.1648 0.1849      0.00148        2698746    204
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.0836 0.1672 0.1882      0.00154        2696249    196
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.0847 0.1530 0.1646      0.00157        2603679    192
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.0983 0.1975 0.2211      0.00154        3061303    196
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1118 0.2212 0.2454      0.00186        3760451    162
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1162 0.2479 0.2865      0.00186        5396634    162
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1187 0.2569 0.3006      0.00192        5388656    156
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1272 0.2726 0.3136      0.00208        6097067    144
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1318 0.2869 0.3341      0.00213        6088482    141
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1353 0.2857 0.3294      0.00245        7468453    123
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1379 0.3095 0.3651      0.00257       10801327    117
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1490 0.3465 0.4177      0.00268       10743643    112
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1534 0.3605 0.4417      0.00312       10702309    97
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1606 0.3824 0.4647      0.00325       12069395    93
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1608 0.4009 0.5100      0.00387       19971906    78
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1638 0.4131 0.5279      0.00406       19911541    75
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1645 0.3889 0.4749      0.00427       12051548    71
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1784 0.4528 0.5795      0.00527       21215557    57
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1836 0.4646 0.5952      0.00530       23916403    57
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1847 0.4716 0.6025      0.00616       23883265    49
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.1851 0.4720 0.6036      0.00728       23879865    42
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1959 0.5237 0.6975      0.00793       41983740    38
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1968 0.5252 0.7004      0.00887       41943177    34
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1985 0.5302 0.7065      0.00959       47198494    32
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.1987 0.5309 0.7092      0.01068       47159625    29
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1996 0.5478 0.7634      0.01011       77910500    30
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2007 0.5548 0.7751      0.01128       77679120    27
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2024 0.5558 0.7754      0.01166       82995962    26
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2037 0.5637 0.7882      0.01307       82759869    23
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2040 0.5588 0.7787      0.01444       93250563    21
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2056 0.5910 0.8478      0.02032      152702213    15
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2069 0.5868 0.8420      0.02347      183510019    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2075 0.5950 0.8543      0.02325      162826195    13
```

</details>
<details><summary>`OPQ32_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 2971642896

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2473 0.4075 0.4141      0.00447       12128035    68
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.0926 0.1212 0.1217      0.00165        1532911    182
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.0990 0.1289 0.1293      0.00174        1899341    173
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1083 0.1426 0.1431      0.00178        1892901    169
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1137 0.1504 0.1510      0.00184        1888690    164
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1144 0.1571 0.1581      0.00196        2705484    154
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1258 0.1754 0.1771      0.00199        2701213    151
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1357 0.1883 0.1895      0.00206        3069870    146
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1575 0.2231 0.2253      0.00236        3052356    128
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1640 0.2295 0.2319      0.00236        3761183    128
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1684 0.2383 0.2406      0.00245        3756672    123
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1725 0.2445 0.2471      0.00263        3755708    115
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1765 0.2640 0.2672      0.00263        5411981    115
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1917 0.2912 0.2950      0.00271        5382836    111
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1976 0.3026 0.3064      0.00285        5377328    106
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2088 0.3227 0.3267      0.00295        6081726    102
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2155 0.3345 0.3389      0.00315        6075762    96
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2175 0.3396 0.3443      0.00336        6073921    90
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2299 0.3757 0.3835      0.00385       10033770    78
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2358 0.3842 0.3901      0.00398       10767975    76
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2578 0.4276 0.4356      0.00415       10711694    73
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2715 0.4547 0.4638      0.00470       12071266    64
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2774 0.4660 0.4756      0.00570       14750374    53
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2831 0.4812 0.4917      0.00595       14725667    51
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2902 0.5201 0.5358      0.00634       19910491    48
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2942 0.5199 0.5343      0.00664       21370528    46
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3104 0.5636 0.5823      0.00700       21246634    43
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3148 0.5725 0.5924      0.00748       21220531    41
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3216 0.5825 0.6027      0.00789       23900963    39
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.3220 0.5861 0.6069      0.01323       39310736    23
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.3283 0.5974 0.6205      0.01296       29032171    24
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.3285 0.5976 0.6207      0.01620       29057119    19
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3384 0.6382 0.6665      0.01809       42239886    17
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3396 0.6555 0.6876      0.01791       39340295    17
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3521 0.6805 0.7138      0.01814       41965341    17
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3546 0.6879 0.7230      0.01664       41915789    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3586 0.6951 0.7309      0.02032       46918622    15
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.3588 0.6968 0.7326      0.02686       77224163    12
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.3589 0.6972 0.7334      0.02772       56937070    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3750 0.7523 0.8042      0.03101       77693649    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3753 0.7559 0.8076      0.03026       77547338    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.3808 0.7656 0.8179      0.03725       92750819    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3823 0.7718 0.8245      0.04116       92305861    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.3828 0.7705 0.8235      0.04177      113072606    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.3829 0.7705 0.8234      0.04951      151030543    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.3831 0.7721 0.8250      0.05239      151756645    6
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.3883 0.7955 0.8569      0.06677      154282083    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3894 0.7989 0.8601      0.06884      164093141    5
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3997 0.8243 0.8944      0.07094      161809111    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4007 0.8286 0.8995      0.07399      162253757    5
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4008 0.8288 0.8998      0.07890      181733162    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.4058 0.8572 0.9413      0.12446      299741562    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=512  0.4060 0.8614 0.9473      0.17310      441541703    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4088 0.8760 0.9720      0.23408      587428287    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.4092 0.8752 0.9728      0.24516      587525535    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.4094 0.8749 0.9714      0.25878      567902356    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.4099 0.8755 0.9730      0.24970      627868709    2
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=512 0.4102 0.8760 0.9735      0.34474      709133179    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ32x4fs` </summary>
Index size 2988786192

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1522 0.3557 0.4344      0.00255       10717920    118
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1     0.0818 0.1616 0.1815      0.00137        2699524    219
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.0839 0.1682 0.1896      0.00135        2694756    222
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.0984 0.1955 0.2184      0.00144        3058265    209
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.0997 0.2015 0.2269      0.00152        3051221    197
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1096 0.2176 0.2416      0.00168        3763037    179
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1196 0.2596 0.3064      0.00177        5378942    170
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1201 0.2610 0.3081      0.00188        5375780    160
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1303 0.2842 0.3313      0.00197        6089070    153
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1325 0.2892 0.3394      0.00203        6076531    148
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1326 0.2906 0.3411      0.00214        6073083    141
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1334 0.3148 0.3871      0.00258       10005643    117
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1383 0.3010 0.3503      0.00252        7461300    119
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1522 0.3557 0.4344      0.00256       10717346    118
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1532 0.3622 0.4433      0.00277       10693388    109
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.1618 0.3798 0.4630      0.00307       12077590    98
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1641 0.4135 0.5294      0.00371       19871834    81
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1642 0.3890 0.4751      0.00334       12040198    90
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1748 0.4451 0.5696      0.00427       21256986    71
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1787 0.4534 0.5802      0.00531       21199233    57
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1855 0.4715 0.6026      0.00536       23864925    56
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1859 0.5009 0.6667      0.00602       39319962    50
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1866 0.4753 0.6078      0.00743       29097167    41
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1963 0.5256 0.7006      0.00793       41899517    38
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.1985 0.5317 0.7099      0.01014       47087659    30
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.1995 0.5516 0.7719      0.00952       77893265    32
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1998 0.5542 0.7775      0.01000       77539171    30
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1999 0.5544 0.7778      0.01109       77509556    28
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2028 0.5643 0.7915      0.01233       82484854    25
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.2039 0.5760 0.8239      0.01519      148485409    20
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2055 0.5922 0.8496      0.02122      152227628    15
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2069 0.5961 0.8562      0.02424      161850836    13
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.2082 0.5952 0.8526      0.02553      183010605    12
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.2083 0.6069 0.8897      0.04265      359361251    8
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fs` </summary>
Index size 2607822028

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF262144_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1547 0.3712 0.4580      0.00215       17370283    140
nprobe=2,quantizer_efSearch=4            0.1231 0.2643 0.3129      0.00136        8698675    220
nprobe=4,quantizer_efSearch=4            0.1446 0.3349 0.4147      0.00171       17400578    176
nprobe=4,quantizer_efSearch=8            0.1547 0.3712 0.4580      0.00216       17370283    139
nprobe=8,quantizer_efSearch=4            0.1702 0.4328 0.5574      0.00254       34700845    119
nprobe=8,quantizer_efSearch=8            0.1743 0.4449 0.5739      0.00279       34660980    108
nprobe=8,quantizer_efSearch=16           0.1802 0.4634 0.5957      0.00363       34552893    83
nprobe=16,quantizer_efSearch=8           0.1860 0.5042 0.6810      0.00412       69152708    73
nprobe=16,quantizer_efSearch=16          0.1896 0.5175 0.6987      0.00472       69035654    64
nprobe=32,quantizer_efSearch=8           0.1907 0.5375 0.7473      0.00593      137649205    51
nprobe=32,quantizer_efSearch=16          0.1967 0.5638 0.7843      0.00690      137368880    44
nprobe=32,quantizer_efSearch=32          0.1980 0.5690 0.7950      0.00799      137139695    38
nprobe=64,quantizer_efSearch=16          0.2000 0.5811 0.8333      0.00974      272543939    31
nprobe=64,quantizer_efSearch=32          0.2034 0.5912 0.8524      0.01106      271921058    28
nprobe=64,quantizer_efSearch=64          0.2046 0.5957 0.8580      0.01359      271492615    23
nprobe=128,quantizer_efSearch=64         0.2053 0.6077 0.8951      0.01915      536870599    16
nprobe=128,quantizer_efSearch=256        0.2057 0.6099 0.8981      0.03763      536034876    8
nprobe=128,quantizer_efSearch=512        0.2059 0.6100 0.8982      0.06291      536016255    5
nprobe=1024,quantizer_efSearch=256       0.2060 0.6173 0.9293      0.08415     4080412198    4
```

</details>
<details><summary>`OPQ32_64,IVF262144_HNSW32,PQ32x4fsr` </summary>
Index size 2607656652

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF262144_HNSW32_PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3469 0.6899 0.7323      0.01118       68817820    27
nprobe=1,quantizer_efSearch=4            0.1439 0.2142 0.2168      0.00182        4318643    166
nprobe=1,quantizer_efSearch=16           0.1603 0.2354 0.2381      0.00393        4299665    77
nprobe=2,quantizer_efSearch=16           0.2181 0.3458 0.3526      0.00429        8622492    70
nprobe=4,quantizer_efSearch=16           0.2717 0.4680 0.4828      0.00497       17286385    61
nprobe=8,quantizer_efSearch=16           0.3143 0.5850 0.6134      0.00626       34493374    48
nprobe=8,quantizer_efSearch=32           0.3171 0.5901 0.6189      0.00862       34436632    35
nprobe=16,quantizer_efSearch=16          0.3425 0.6828 0.7242      0.00863       68931089    35
nprobe=16,quantizer_efSearch=32          0.3469 0.6899 0.7323      0.01117       68817820    27
nprobe=16,quantizer_efSearch=64          0.3480 0.6924 0.7347      0.01533       68783897    20
nprobe=32,quantizer_efSearch=16          0.3648 0.7601 0.8180      0.01863      137107820    17
nprobe=32,quantizer_efSearch=32          0.3692 0.7694 0.8291      0.02064      136895232    15
nprobe=32,quantizer_efSearch=64          0.3711 0.7735 0.8338      0.02475      136779281    13
nprobe=64,quantizer_efSearch=16          0.3729 0.7996 0.8739      0.03026      272022399    10
nprobe=64,quantizer_efSearch=64          0.3836 0.8226 0.9025      0.03835      271094717    8
nprobe=64,quantizer_efSearch=128         0.3842 0.8236 0.9040      0.04628      270941246    7
nprobe=64,quantizer_efSearch=256         0.3850 0.8242 0.9044      0.06125      270902278    5
nprobe=128,quantizer_efSearch=64         0.3891 0.8514 0.9457      0.07044      536028507    5
nprobe=128,quantizer_efSearch=128        0.3914 0.8545 0.9496      0.07669      535450659    4
nprobe=128,quantizer_efSearch=256        0.3917 0.8547 0.9499      0.09442      535283897    4
nprobe=256,quantizer_efSearch=64         0.3932 0.8633 0.9685      0.12573     1058198192    3
nprobe=256,quantizer_efSearch=128        0.3961 0.8681 0.9756      0.13439     1056068047    3
nprobe=512,quantizer_efSearch=128        0.3965 0.8745 0.9867      0.23779     2079150882    2
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fs` </summary>
Index size 2542921232

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2011 0.5757 0.8188      0.01202      282240391    25
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.0772 0.1520 0.1742      0.00135        4524946    223
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.0995 0.2005 0.2276      0.00140        5025278    215
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1036 0.2070 0.2345      0.00156        5702162    193
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1202 0.2589 0.3030      0.00161        9409287    187
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1251 0.2693 0.3139      0.00162       10061043    186
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1299 0.2807 0.3317      0.00160        9381105    189
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1306 0.2862 0.3359      0.00177       10055613    170
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1344 0.2913 0.3425      0.00196       10036174    153
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1347 0.3080 0.3674      0.00200       18297308    150
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1529 0.3603 0.4423      0.00221       18135776    136
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.1547 0.3637 0.4470      0.00262       20088605    115
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1604 0.3826 0.4700      0.00266       20044665    113
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1686 0.4261 0.5459      0.00284       35549313    106
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1757 0.4434 0.5664      0.00292       36187137    103
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1812 0.4634 0.5969      0.00352       37306527    86
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1871 0.4991 0.6706      0.00430       70948170    70
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1899 0.5138 0.6916      0.00480       72068203    63
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1906 0.5215 0.7044      0.00500       71860879    60
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1912 0.5236 0.7068      0.00548       71778914    55
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1923 0.5410 0.7492      0.00655      139751403    46
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1925 0.5445 0.7549      0.00706      139340279    43
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1993 0.5681 0.7917      0.00785      142953496    39
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2006 0.5695 0.7934      0.00918      147997019    33
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.2011 0.5757 0.8188      0.01104      282240391    28
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2019 0.5901 0.8463      0.01136      275323044    27
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2053 0.5974 0.8604      0.01215      277196487    25
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2063 0.6001 0.8634      0.01378      282004788    22
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2066 0.6096 0.8954      0.02132      558604513    15
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.2069 0.6103 0.9032      0.03002     1122236638    10
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.2072 0.6191 0.9292      0.10095     2117460369    3
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 2542906896

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3894 0.8272 0.9054      0.03177      281813332    10
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1857 0.2830 0.2884      0.00153        9450427    197
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1884 0.2950 0.3034      0.00151        9072893    199
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2100 0.3285 0.3367      0.00153        9380618    197
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2105 0.3257 0.3330      0.00163       10048922    184
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.2175 0.3390 0.3474      0.00179       10036950    168
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2584 0.4345 0.4492      0.00200       18119084    151
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2593 0.4370 0.4518      0.00210       18113185    143
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2674 0.4550 0.4704      0.00223       18739145    135
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.2688 0.4577 0.4733      0.00262       18727192    115
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2734 0.4644 0.4795      0.00292       20033413    103
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2914 0.5239 0.5458      0.00297       36373094    101
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3076 0.5639 0.5895      0.00308       36098891    98
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3117 0.5763 0.6027      0.00338       36016164    89
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3142 0.5776 0.6039      0.00353       37331494    85
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3169 0.5877 0.6152      0.00369       37267461    82
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3179 0.5892 0.6167      0.00402       37254780    75
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3200 0.6112 0.6462      0.00475       70258033    64
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3394 0.6648 0.7044      0.00514       70651546    59
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3471 0.6869 0.7282      0.00563       71764828    54
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3480 0.6873 0.7288      0.00590       71750709    51
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3500 0.6931 0.7348      0.00640       74259495    47
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3503 0.6939 0.7357      0.00801       79340580    38
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.3509 0.6936 0.7355      0.00804       74204349    38
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3596 0.7325 0.7842      0.01513      139432849    20
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3615 0.7393 0.7900      0.01515      141804819    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3708 0.7640 0.8203      0.01591      140263271    19
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3723 0.7704 0.8288      0.01585      142597185    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3738 0.7744 0.8337      0.01637      142386708    19
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.3740 0.7756 0.8353      0.01890      157687159    16
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.3802 0.8066 0.8785      0.03457      295470085    9
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.3885 0.8234 0.9012      0.03049      282074443    10
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3894 0.8272 0.9054      0.03624      281738708    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.3896 0.8269 0.9053      0.03815      291979333    8
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3926 0.8514 0.9419      0.06557      542879636    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3941 0.8448 0.9314      0.06242      554925093    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3951 0.8684 0.9705      0.11969     1064705243    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3960 0.8742 0.9778      0.12358     1077766232    3
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3961 0.8653 0.9653      0.12150     1084209943    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.3965 0.8742 0.9783      0.13556     1076810726    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.3966 0.8743 0.9783      0.12126     1097390843    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3983 0.8776 0.9858      0.19512     2122937599    2
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.3989 0.8810 0.9935      0.39006     4158582373    1
```

</details>
<details><summary>`OPQ32_64,IVF262144(IVF512,PQ64x4fs,RFlat),PQ32x4fs` </summary>
Index size 2547197456

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF262144_IVF512_PQ64x4fs_RFlat__PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.2052 0.5969 0.8580      0.01072      278406518    28
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1    0.0782 0.1557 0.1785      0.00139        4520338    216
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2   0.0903 0.1834 0.2091      0.00138        4691549    219
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1    0.0967 0.2038 0.2398      0.00161        8925700    186
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2    0.1119 0.2460 0.2887      0.00163        9082378    184
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1300 0.2810 0.3305      0.00164        9383060    184
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1307 0.2820 0.3321      0.00184        9376563    164
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1348 0.2919 0.3427      0.00193       10031104    156
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1524 0.3619 0.4451      0.00215       18109986    140
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1580 0.3723 0.4569      0.00234       18778056    129
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.1604 0.3801 0.4663      0.00301       20070469    100
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.1661 0.4249 0.5420      0.00308       35631713    98
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8    0.1740 0.4500 0.5756      0.00323       36172998    93
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1788 0.4557 0.5862      0.00330       36007806    91
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1801 0.4676 0.6003      0.00393       37235035    77
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.1878 0.5053 0.6814      0.00468       70619443    65
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.1901 0.5188 0.6974      0.00504       71987284    60
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.1903 0.5235 0.7064      0.00521       71730245    58
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1907 0.5240 0.7066      0.00625       71730422    48
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.1912 0.5283 0.7124      0.00610       74213745    50
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.1917 0.5286 0.7128      0.00733       74214354    41
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.1980 0.5650 0.7873      0.00791      140645761    38
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.1995 0.5749 0.8024      0.00860      142352028    35
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32  0.2052 0.5969 0.8580      0.01292      278393280    24
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2057 0.6017 0.8637      0.01617      281770211    19
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64 0.2072 0.6115 0.8991      0.01972      549750888    16
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fs` </summary>
Index size 2451976396

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF65536_HNSW32_PQ32x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1684 0.4205 0.5488      0.00289       68747852    104
nprobe=2,quantizer_efSearch=4            0.1346 0.3149 0.3923      0.00179       34456398    168
nprobe=2,quantizer_efSearch=8            0.1451 0.3414 0.4241      0.00196       34341864    154
nprobe=2,quantizer_efSearch=16           0.1457 0.3436 0.4270      0.00238       34271282    127
nprobe=4,quantizer_efSearch=4            0.1562 0.3877 0.5032      0.00262       68916122    115
nprobe=4,quantizer_efSearch=8            0.1684 0.4205 0.5488      0.00291       68747852    104
nprobe=4,quantizer_efSearch=16           0.1694 0.4269 0.5579      0.00334       68605165    90
nprobe=8,quantizer_efSearch=4            0.1829 0.4797 0.6503      0.00413      137363597    73
nprobe=8,quantizer_efSearch=8            0.1862 0.4900 0.6667      0.00401      137284738    75
nprobe=8,quantizer_efSearch=16           0.1905 0.5037 0.6857      0.00447      137009596    68
nprobe=8,quantizer_efSearch=32           0.1912 0.5071 0.6895      0.00547      136870517    55
nprobe=16,quantizer_efSearch=4           0.1922 0.5194 0.7264      0.00556      272972884    54
nprobe=16,quantizer_efSearch=16          0.2021 0.5491 0.7768      0.00641      272331514    47
nprobe=16,quantizer_efSearch=32          0.2039 0.5558 0.7850      0.00670      272004544    45
nprobe=16,quantizer_efSearch=64          0.2046 0.5561 0.7863      0.00877      271887772    35
nprobe=32,quantizer_efSearch=16          0.2090 0.5782 0.8410      0.00782      539791163    39
nprobe=32,quantizer_efSearch=32          0.2106 0.5842 0.8508      0.00927      539040978    33
nprobe=32,quantizer_efSearch=64          0.2115 0.5861 0.8537      0.01138      538733234    27
nprobe=32,quantizer_efSearch=128         0.2119 0.5869 0.8541      0.01590      538642378    19
nprobe=256,quantizer_efSearch=64         0.2123 0.6075 0.9215      0.03125     4116512709    10
nprobe=256,quantizer_efSearch=128        0.2130 0.6085 0.9240      0.03411     4110105122    9
nprobe=256,quantizer_efSearch=256        0.2132 0.6091 0.9250      0.04094     4108066156    8
```

</details>
<details><summary>`OPQ32_64,IVF65536_HNSW32,PQ32x4fsr` </summary>
Index size 2451996876

 code_size 16

 log filename: autotune.dbbigann100M.OPQ32_64_IVF65536_HNSW32_PQ32x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3310 0.6692 0.7186      0.01517      136715723    20
nprobe=1,quantizer_efSearch=4            0.1746 0.2768 0.2838      0.00181       17176269    167
nprobe=1,quantizer_efSearch=16           0.1860 0.2956 0.3032      0.00292       17107636    103
nprobe=2,quantizer_efSearch=16           0.2397 0.4208 0.4380      0.00371       34251218    81
nprobe=4,quantizer_efSearch=16           0.2922 0.5429 0.5744      0.00482       68550122    63
nprobe=8,quantizer_efSearch=16           0.3294 0.6644 0.7132      0.00667      136889408    45
nprobe=8,quantizer_efSearch=32           0.3306 0.6688 0.7182      0.00814      136760323    37
nprobe=8,quantizer_efSearch=64           0.3309 0.6692 0.7186      0.01070      136725793    29
nprobe=8,quantizer_efSearch=128          0.3310 0.6692 0.7186      0.01499      136715723    21
nprobe=16,quantizer_efSearch=128         0.3577 0.7553 0.8242      0.01942      271590010    16
nprobe=32,quantizer_efSearch=128         0.3743 0.8140 0.9013      0.02900      538058698    11
nprobe=32,quantizer_efSearch=256         0.3744 0.8141 0.9014      0.03894      538045878    8
nprobe=64,quantizer_efSearch=16          0.3760 0.8271 0.9292      0.03976     1065805660    8
nprobe=64,quantizer_efSearch=256         0.3808 0.8453 0.9507      0.05563     1062607520    6
nprobe=64,quantizer_efSearch=512         0.3809 0.8454 0.9508      0.08629     1062573181    4
nprobe=128,quantizer_efSearch=256        0.3854 0.8603 0.9788      0.09323     2090706281    4
nprobe=128,quantizer_efSearch=512        0.3856 0.8605 0.9790      0.11970     2090621421    3
nprobe=1024,quantizer_efSearch=256       0.3874 0.8695 0.9956      0.49912    15763742094    1
nprobe=1024,quantizer_efSearch=512       0.3876 0.8687 0.9958      0.53216    15752402452    1
```

</details>
<details><summary>`PCAR32,IVF1048576_HNSW32,SQ4` </summary>
Index size 2828051797

 code_size 16

 log filename: autotune.dbbigann100M.PCAR32_IVF1048576_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2329 0.6591 0.9110      0.09662      272917319    4
nprobe=1,quantizer_efSearch=4            0.0846 0.1388 0.1423      0.00243        1143899    124
nprobe=2,quantizer_efSearch=4            0.1138 0.2084 0.2184      0.00249        2286961    121
nprobe=4,quantizer_efSearch=4            0.1345 0.2764 0.2984      0.00264        4564749    114
nprobe=8,quantizer_efSearch=4            0.1646 0.3817 0.4286      0.00326        9101265    92
nprobe=8,quantizer_efSearch=8            0.1675 0.3915 0.4409      0.00363        9097357    83
nprobe=16,quantizer_efSearch=4           0.1811 0.4451 0.5218      0.00497       18115515    61
nprobe=16,quantizer_efSearch=8           0.1915 0.4748 0.5622      0.00520       18107157    58
nprobe=16,quantizer_efSearch=16          0.1955 0.4841 0.5742      0.00598       18067325    51
nprobe=32,quantizer_efSearch=8           0.2013 0.5217 0.6452      0.00801       35892690    38
nprobe=32,quantizer_efSearch=16          0.2083 0.5470 0.6811      0.00945       35820358    32
nprobe=32,quantizer_efSearch=32          0.2116 0.5535 0.6936      0.01079       35739006    28
nprobe=64,quantizer_efSearch=16          0.2163 0.5881 0.7575      0.01356       70940891    23
nprobe=64,quantizer_efSearch=32          0.2215 0.6029 0.7819      0.01568       70727239    20
nprobe=128,quantizer_efSearch=32         0.2264 0.6302 0.8436      0.02370      139703029    13
nprobe=128,quantizer_efSearch=64         0.2293 0.6387 0.8608      0.02781      139229335    11
nprobe=128,quantizer_efSearch=128        0.2299 0.6424 0.8655      0.03267      138954725    10
nprobe=256,quantizer_efSearch=64         0.2311 0.6516 0.8977      0.04616      274398764    7
nprobe=256,quantizer_efSearch=128        0.2326 0.6573 0.9078      0.05415      273497415    6
nprobe=256,quantizer_efSearch=256        0.2331 0.6593 0.9109      0.06509      273060621    5
nprobe=512,quantizer_efSearch=128        0.2336 0.6651 0.9335      0.09416      537822318    4
nprobe=512,quantizer_efSearch=256        0.2343 0.6670 0.9392      0.10328      536128477    3
nprobe=1024,quantizer_efSearch=256       0.2345 0.6694 0.9504      0.17591     1052244076    2
```

</details>
<details><summary>`PCAR32,IVF1048576(IVF1024,PQ16x4fs,RFlat),SQ4` </summary>
Index size 2559747225

 code_size 16

 log filename: autotune.dbbigann100M.PCAR32_IVF1048576_IVF1024_PQ16x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2300 0.6512 0.9045      0.06292      285138683    5
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.0645 0.1074 0.1103      0.00230        2675280    131
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1035 0.1869 0.1966      0.00237        6038188    127
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1256 0.2391 0.2546      0.00236        6050570    127
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1357 0.2724 0.2933      0.00248        6040980    121
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1469 0.2998 0.3231      0.00282        6020492    107
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1527 0.3228 0.3560      0.00290       10641507    104
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1678 0.3744 0.4194      0.00318       11981593    95
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1685 0.3724 0.4175      0.00319       10605205    95
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1736 0.3986 0.4498      0.00398       11941843    76
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1809 0.4290 0.5026      0.00445       19721341    68
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1832 0.4383 0.5173      0.00504       19672663    60
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.1926 0.4728 0.5596      0.00517       21007234    59
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1934 0.4706 0.5551      0.00520       23720374    58
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1969 0.4838 0.5757      0.00569       23681289    53
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.1980 0.4895 0.5818      0.00679       23644108    45
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.1984 0.4931 0.5861      0.00748       28884139    41
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2045 0.5312 0.6600      0.00836       38839682    36
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2068 0.5394 0.6698      0.00807       41577766    38
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2081 0.5420 0.6740      0.00908       46786975    34
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2118 0.5510 0.6865      0.00917       41476837    33
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2127 0.5542 0.6920      0.01006       46685522    30
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2133 0.5682 0.7276      0.01267       77125636    24
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2144 0.5723 0.7348      0.01354       82219820    23
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2188 0.5917 0.7683      0.01545       76752605    20
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2195 0.5985 0.7776      0.01448       81875369    21
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2210 0.6073 0.7911      0.01840       91957657    17
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2265 0.6336 0.8548      0.02585      150838985    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2269 0.6352 0.8572      0.02701      161022854    12
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2274 0.6386 0.8635      0.03352      181012220    9
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.2275 0.6367 0.8613      0.03459      150262027    9
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.2277 0.6393 0.8654      0.03715      160432940    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2285 0.6445 0.8894      0.04515      297595944    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2288 0.6494 0.9006      0.05049      285988369    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2294 0.6522 0.9052      0.05059      315931276    6
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2297 0.6509 0.9041      0.05445      285296299    6
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.2307 0.6534 0.9098      0.06195      355920858    5
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2312 0.6542 0.9117      0.07244      315212079    5
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2316 0.6613 0.9378      0.09177      579476566    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512  0.2320 0.6618 0.9403      0.11966      700021533    3
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2321 0.6623 0.9402      0.13085      578102160    3
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2324 0.6644 0.9532      0.18031     1093640155    2
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ4` </summary>
Index size 2507078229

 code_size 16

 log filename: autotune.dbbigann100M.PCAR32_IVF262144_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2253 0.6575 0.9381      0.13185     1048166447    3
nprobe=1,quantizer_efSearch=4            0.1019 0.1883 0.2005      0.00221        4334802    136
nprobe=2,quantizer_efSearch=4            0.1302 0.2639 0.2909      0.00238        8672737    127
nprobe=2,quantizer_efSearch=8            0.1415 0.2881 0.3185      0.00263        8653819    115
nprobe=2,quantizer_efSearch=16           0.1439 0.2938 0.3239      0.00315        8638013    96
nprobe=4,quantizer_efSearch=4            0.1538 0.3390 0.3919      0.00330       17340470    91
nprobe=4,quantizer_efSearch=8            0.1681 0.3780 0.4380      0.00347       17323600    87
nprobe=4,quantizer_efSearch=16           0.1717 0.3851 0.4459      0.00387       17294479    78
nprobe=8,quantizer_efSearch=4            0.1854 0.4496 0.5438      0.00450       34611657    67
nprobe=8,quantizer_efSearch=8            0.1882 0.4624 0.5594      0.00467       34572311    65
nprobe=8,quantizer_efSearch=16           0.1923 0.4744 0.5747      0.00522       34525754    58
nprobe=8,quantizer_efSearch=32           0.1938 0.4788 0.5796      0.00611       34495026    50
nprobe=16,quantizer_efSearch=4           0.1947 0.5016 0.6359      0.00718       68979606    42
nprobe=16,quantizer_efSearch=8           0.2015 0.5295 0.6741      0.00774       68937068    39
nprobe=16,quantizer_efSearch=16          0.2039 0.5397 0.6887      0.00787       68852205    39
nprobe=16,quantizer_efSearch=32          0.2054 0.5463 0.6969      0.00868       68768890    35
nprobe=16,quantizer_efSearch=64          0.2063 0.5478 0.6982      0.01121       68739826    27
nprobe=32,quantizer_efSearch=8           0.2074 0.5715 0.7519      0.01466      136939278    21
nprobe=32,quantizer_efSearch=32          0.2159 0.6016 0.7936      0.01431      136529370    21
nprobe=32,quantizer_efSearch=64          0.2164 0.6044 0.7982      0.01635      136426887    19
nprobe=32,quantizer_efSearch=128         0.2170 0.6049 0.7995      0.02171      136392800    14
nprobe=64,quantizer_efSearch=32          0.2200 0.6292 0.8595      0.02822      270345611    11
nprobe=64,quantizer_efSearch=128         0.2217 0.6329 0.8668      0.03008      269928558    10
nprobe=128,quantizer_efSearch=128        0.2241 0.6492 0.9127      0.05080      532654005    6
nprobe=128,quantizer_efSearch=256        0.2243 0.6491 0.9135      0.06311      532516207    5
nprobe=256,quantizer_efSearch=128        0.2250 0.6571 0.9367      0.09265     1048991891    4
nprobe=256,quantizer_efSearch=256        0.2253 0.6575 0.9381      0.09809     1048302968    4
nprobe=512,quantizer_efSearch=256        0.2256 0.6605 0.9485      0.18917     2059862578    2
nprobe=2048,quantizer_efSearch=256       0.2257 0.6610 0.9529      0.69495     7799215838    1
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ4` </summary>
Index size 2440067993

 code_size 16

 log filename: autotune.dbbigann100M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1391 0.2849 0.3112      0.00283        9403515    106
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1105 0.2016 0.2140      0.00258        5732715    117
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.1345 0.2741 0.2973      0.00270       10115366    112
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4    0.1390 0.2849 0.3112      0.00275        9407346    109
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1418 0.2912 0.3180      0.00297       11411443    102
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1447 0.2960 0.3229      0.00290       10068928    104
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1485 0.3236 0.3643      0.00353       18299781    86
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1587 0.3522 0.4022      0.00354       18241341    85
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1639 0.3666 0.4209      0.00365       18140183    83
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1655 0.3718 0.4240      0.00373       20171468    81
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1717 0.3873 0.4436      0.00388       20118852    78
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.1718 0.3901 0.4487      0.00510       22663931    59
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.1798 0.4276 0.5081      0.00518       36419904    58
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1913 0.4684 0.5611      0.00561       37538582    54
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.1925 0.4691 0.5620      0.00604       40120122    50
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.1947 0.4784 0.5767      0.00613       37356634    49
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.1964 0.4802 0.5787      0.00664       39941727    46
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.2031 0.5310 0.6692      0.00911       70687241    33
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2044 0.5348 0.6766      0.00900       72046290    34
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2060 0.5442 0.6897      0.00968       71849393    31
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.2084 0.5498 0.6969      0.01247       79443430    25
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=64  0.2086 0.5491 0.6955      0.01295       79399814    24
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2088 0.5561 0.7248      0.01495      139970933    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2120 0.5724 0.7514      0.01514      139291929    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2139 0.5740 0.7518      0.01562      143342515    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2179 0.5968 0.7879      0.01756      147741611    18
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2190 0.6007 0.7970      0.01715      142264814    18
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2196 0.6020 0.7976      0.01847      147353952    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2218 0.6192 0.8385      0.02813      278366660    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2237 0.6284 0.8610      0.02853      276843281    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2239 0.6302 0.8632      0.02872      281790539    11
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2243 0.6345 0.8849      0.05160      542506399    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2253 0.6444 0.9007      0.04902      548323579    7
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2255 0.6438 0.8999      0.05429      558355065    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2265 0.6524 0.9318      0.09777     1058980250    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32  0.2267 0.6524 0.9318      0.10134     1057706728    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2269 0.6558 0.9377      0.09165     1062451239    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.2271 0.6563 0.9382      0.09888     1061067488    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2276 0.6571 0.9459      0.18441     2094433790    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2282 0.6581 0.9482      0.18686     2075383306    2
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ4` </summary>
Index size 2426834005

 code_size 16

 log filename: autotune.dbbigann100M.PCAR32_IVF65536_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1227 0.2551 0.2855      0.00535       17189121    57
nprobe=1,quantizer_efSearch=4            0.1171 0.2411 0.2704      0.00346       17267406    87
nprobe=1,quantizer_efSearch=8            0.1217 0.2531 0.2832      0.00358       17203478    84
nprobe=1,quantizer_efSearch=32           0.1226 0.2548 0.2852      0.00443       17190018    68
nprobe=1,quantizer_efSearch=64           0.1227 0.2551 0.2855      0.00559       17189121    54
nprobe=2,quantizer_efSearch=8            0.1523 0.3497 0.4109      0.00584       34447642    52
nprobe=2,quantizer_efSearch=16           0.1525 0.3524 0.4141      0.00609       34406113    50
nprobe=2,quantizer_efSearch=32           0.1532 0.3531 0.4149      0.00660       34384929    46
nprobe=2,quantizer_efSearch=128          0.1535 0.3538 0.4155      0.00981       34378870    31
nprobe=4,quantizer_efSearch=4            0.1673 0.4162 0.5061      0.01007       69045119    30
nprobe=4,quantizer_efSearch=8            0.1776 0.4435 0.5428      0.01037       68969014    29
nprobe=4,quantizer_efSearch=32           0.1790 0.4508 0.5519      0.01096       68813100    28
nprobe=4,quantizer_efSearch=64           0.1791 0.4511 0.5524      0.01212       68800664    25
nprobe=4,quantizer_efSearch=128          0.1792 0.4513 0.5526      0.01420       68794480    22
nprobe=8,quantizer_efSearch=8            0.1959 0.5239 0.6684      0.01869      137221097    17
nprobe=8,quantizer_efSearch=16           0.1985 0.5339 0.6808      0.01888      137049169    16
nprobe=8,quantizer_efSearch=32           0.1994 0.5367 0.6842      0.01939      136938125    16
nprobe=8,quantizer_efSearch=128          0.1995 0.5376 0.6852      0.02268      136899294    14
nprobe=16,quantizer_efSearch=32          0.2122 0.5955 0.7917      0.03605      271589957    9
nprobe=16,quantizer_efSearch=64          0.2125 0.5956 0.7918      0.03702      271524341    9
nprobe=16,quantizer_efSearch=128         0.2126 0.5963 0.7925      0.03916      271504659    8
nprobe=32,quantizer_efSearch=32          0.2158 0.6236 0.8632      0.06747      537097874    5
nprobe=32,quantizer_efSearch=64          0.2163 0.6249 0.8652      0.06874      536855237    5
nprobe=32,quantizer_efSearch=256         0.2164 0.6254 0.8658      0.08100      536758239    4
nprobe=64,quantizer_efSearch=16          0.2168 0.6304 0.8932      0.12592     1061474513    3
nprobe=64,quantizer_efSearch=32          0.2180 0.6368 0.9066      0.13765     1059881178    3
nprobe=64,quantizer_efSearch=256         0.2188 0.6398 0.9124      0.13826     1058650728    3
nprobe=128,quantizer_efSearch=128        0.2201 0.6475 0.9374      0.24655     2081957004    2
nprobe=128,quantizer_efSearch=256        0.2202 0.6477 0.9376      0.26249     2081659630    2
nprobe=256,quantizer_efSearch=256        0.2207 0.6516 0.9506      0.50895     4080893137    1
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF1048576_HNSW32,PQ64x4fs` </summary>
Index size 5359972229

 code_size 32

 log filename: autotune.dbbigann100M.IVF1048576_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3866 0.8255 0.9410      0.12177      278476819    3
nprobe=2,quantizer_efSearch=4            0.1462 0.2211 0.2281      0.00262        2321705    115
nprobe=4,quantizer_efSearch=4            0.1833 0.2997 0.3123      0.00287        4637381    105
nprobe=8,quantizer_efSearch=4            0.2412 0.4299 0.4559      0.00418        9271752    72
nprobe=8,quantizer_efSearch=8            0.2481 0.4421 0.4687      0.00478        9262425    63
nprobe=16,quantizer_efSearch=4           0.2705 0.5029 0.5429      0.00541       18449678    56
nprobe=16,quantizer_efSearch=8           0.2920 0.5456 0.5885      0.00652       18421923    46
nprobe=32,quantizer_efSearch=8           0.3137 0.6114 0.6683      0.00834       36648720    36
nprobe=64,quantizer_efSearch=8           0.3241 0.6455 0.7142      0.01082       72524143    28
nprobe=32,quantizer_efSearch=16          0.3267 0.6467 0.7089      0.01036       36518528    29
nprobe=64,quantizer_efSearch=16          0.3432 0.6993 0.7758      0.01291       72400355    24
nprobe=64,quantizer_efSearch=32          0.3538 0.7272 0.8081      0.01749       72098543    18
nprobe=128,quantizer_efSearch=32         0.3682 0.7672 0.8641      0.02212      142679300    14
nprobe=128,quantizer_efSearch=64         0.3765 0.7867 0.8877      0.03012      142090926    10
nprobe=256,quantizer_efSearch=64         0.3806 0.8098 0.9231      0.03899      280547190    8
nprobe=256,quantizer_efSearch=128        0.3862 0.8225 0.9379      0.05146      279331322    6
nprobe=256,quantizer_efSearch=256        0.3868 0.8256 0.9409      0.07591      278689002    4
nprobe=512,quantizer_efSearch=256        0.3878 0.8397 0.9664      0.09326      548071995    4
nprobe=1024,quantizer_efSearch=256       0.3882 0.8466 0.9797      0.13081     1077211040    3
nprobe=1024,quantizer_efSearch=512       0.3889 0.8486 0.9819      0.19258     1073480588    2
```

</details>
<details><summary>`IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 5118197193

 code_size 32

 log filename: autotune.dbbigann100M.IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3813 0.8139 0.9267      0.03836      327717778    8
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1167 0.1779 0.1836      0.00197        2705293    153
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1395 0.2101 0.2152      0.00230        3784448    131
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1469 0.2368 0.2471      0.00231        5050009    130
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1637 0.2614 0.2710      0.00242        5432989    124
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1779 0.2920 0.3037      0.00251        5411265    120
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1816 0.2969 0.3091      0.00260        5395442    116
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1989 0.3305 0.3441      0.00290        6107068    104
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1996 0.3390 0.3564      0.00303       10132155    99
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2064 0.3438 0.3563      0.00326        7500026    93
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2236 0.3834 0.4035      0.00333       10824344    91
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2404 0.4183 0.4430      0.00345       10783189    88
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2537 0.4557 0.4831      0.00429       12105523    70
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2784 0.5079 0.5446      0.00492       21534136    61
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2844 0.5383 0.5857      0.00637       38845249    48
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2981 0.5638 0.6077      0.00635       24023645    48
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3018 0.5707 0.6150      0.00691       23925557    44
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3177 0.6234 0.6813      0.00743       39632645    41
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.3230 0.6300 0.6853      0.00972       47790420    31
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3345 0.6609 0.7242      0.00946       42047394    32
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3451 0.6969 0.7715      0.01136       79265725    27
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3533 0.7220 0.8037      0.01337       78079619    23
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3547 0.7240 0.8061      0.01420       77826548    22
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3594 0.7378 0.8208      0.01442       83068008    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3604 0.7411 0.8250      0.01648       82625855    19
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3617 0.7489 0.8430      0.01694      151671630    18
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3759 0.7898 0.8920      0.02498      152811598    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3786 0.7963 0.8992      0.02875      162229341    11
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3810 0.8151 0.9305      0.03347      291636657    9
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3813 0.8139 0.9267      0.03906      326911511    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3843 0.8240 0.9403      0.03738      300944405    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3851 0.8248 0.9415      0.04294      320509555    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3861 0.8278 0.9434      0.05083      319326210    6
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3863 0.8342 0.9600      0.05713      604632473    6
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.3872 0.8429 0.9695      0.07548      630281351    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.3883 0.8481 0.9795      0.08445     1148905714    4
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.3891 0.8483 0.9805      0.12727     1097574453    3
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.3892 0.8507 0.9838      0.13750     1115168212    3
```

</details>
<details><summary>`IVF262144_HNSW32,PQ64x4fs` </summary>
Index size 4339532933

 code_size 32

 log filename: autotune.dbbigann100M.IVF262144_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2520 0.4409 0.4700      0.00380       17267392    79
nprobe=2,quantizer_efSearch=4            0.1843 0.2989 0.3137      0.00236        8650686    128
nprobe=4,quantizer_efSearch=4            0.2244 0.3926 0.4197      0.00284       17322871    106
nprobe=4,quantizer_efSearch=8            0.2520 0.4409 0.4700      0.00400       17267392    75
nprobe=8,quantizer_efSearch=4            0.2823 0.5248 0.5678      0.00431       34546653    70
nprobe=8,quantizer_efSearch=8            0.2903 0.5401 0.5842      0.00483       34515309    63
nprobe=8,quantizer_efSearch=16           0.3026 0.5653 0.6107      0.00647       34441357    47
nprobe=16,quantizer_efSearch=4           0.3066 0.5953 0.6557      0.00628       68973858    48
nprobe=16,quantizer_efSearch=8           0.3260 0.6381 0.7021      0.00718       68859083    42
nprobe=16,quantizer_efSearch=16          0.3333 0.6551 0.7207      0.00817       68764563    37
nprobe=32,quantizer_efSearch=8           0.3468 0.6981 0.7781      0.01041      137051319    29
nprobe=32,quantizer_efSearch=32          0.3629 0.7429 0.8302      0.01403      136567569    22
nprobe=64,quantizer_efSearch=16          0.3674 0.7719 0.8718      0.01661      271531416    19
nprobe=64,quantizer_efSearch=32          0.3759 0.7901 0.8944      0.01920      270925567    16
nprobe=64,quantizer_efSearch=64          0.3787 0.7967 0.9018      0.02389      270554935    13
nprobe=128,quantizer_efSearch=32         0.3799 0.8135 0.9305      0.02743      536403186    11
nprobe=128,quantizer_efSearch=64         0.3855 0.8243 0.9433      0.03248      535199772    10
nprobe=128,quantizer_efSearch=128        0.3871 0.8275 0.9476      0.04209      534634189    8
nprobe=256,quantizer_efSearch=64         0.3881 0.8392 0.9657      0.04299     1057331870    7
nprobe=256,quantizer_efSearch=128        0.3898 0.8456 0.9737      0.05481     1055205849    6
nprobe=512,quantizer_efSearch=128        0.3903 0.8514 0.9833      0.07420     2078620356    5
nprobe=512,quantizer_efSearch=256        0.3910 0.8534 0.9859      0.09502     2075118713    4
nprobe=1024,quantizer_efSearch=128       0.3916 0.8534 0.9872      0.10288     4061429497    3
nprobe=1024,quantizer_efSearch=256       0.3922 0.8565 0.9906      0.13626     4077383821    3
nprobe=2048,quantizer_efSearch=256       0.3924 0.8572 0.9917      0.17471     7901809527    2
nprobe=2048,quantizer_efSearch=512       0.3925 0.8580 0.9930      0.22731     7987250747    2
```

</details>
<details><summary>`IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 4279225801

 code_size 32

 log filename: autotune.dbbigann100M.IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.3726 0.7774 0.8785      0.01496      282536860    21
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.1132 0.1720 0.1773      0.00148        4533788    204
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1       0.1138 0.1731 0.1785      0.00155        4531448    194
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.1347 0.2070 0.2128      0.00165        5035374    182
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.1398 0.2157 0.2217      0.00166        5029841    181
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.1407 0.2180 0.2241      0.00175        5025453    172
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.1499 0.2385 0.2504      0.00184        8898516    164
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1586 0.2571 0.2684      0.00177        9103441    170
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.1912 0.3091 0.3222      0.00190        9387578    158
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.1944 0.3173 0.3317      0.00215        9350451    140
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.2064 0.3336 0.3483      0.00232       10031329    130
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2149 0.3733 0.3976      0.00232       17855811    130
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2171 0.3781 0.4040      0.00245       17799849    123
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.2231 0.3848 0.4071      0.00241       18215265    125
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2409 0.4225 0.4486      0.00243       18142393    124
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2427 0.4272 0.4550      0.00257       18097999    117
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.2444 0.4290 0.4569      0.00281       18072399    107
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.2619 0.4597 0.4886      0.00341       20008335    89
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.2785 0.5200 0.5577      0.00358       36426372    84
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.2842 0.5296 0.5679      0.00407       37641713    74
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.2962 0.5518 0.5958      0.00371       36135589    81
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.3043 0.5690 0.6150      0.00452       37243149    67
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3147 0.6138 0.6728      0.00532       71559463    57
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3238 0.6364 0.7006      0.00554       70757621    55
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3353 0.6606 0.7273      0.00657       71643033    46
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3454 0.7045 0.7865      0.00868      139396511    35
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3505 0.7097 0.7898      0.00883      142332953    34
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.3634 0.7440 0.8303      0.01037      142542122    29
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3653 0.7486 0.8353      0.01241      142043012    25
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3691 0.7690 0.8667      0.01290      280756889    24
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.3726 0.7774 0.8785      0.01516      282536801    20
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3748 0.7855 0.8876      0.01554      274668658    20
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3783 0.7963 0.9021      0.01647      276534127    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.3797 0.7992 0.9050      0.01910      281286607    16
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.3817 0.8169 0.9341      0.02418      559681589    13
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.3862 0.8285 0.9486      0.02475      547263939    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3871 0.8285 0.9488      0.04376      545840964    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3880 0.8415 0.9684      0.04521     1066020976    7
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.3883 0.8413 0.9684      0.03786     1105738429    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.3904 0.8481 0.9765      0.04141     1078267448    8
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.3912 0.8551 0.9883      0.06716     2099698797    5
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3914 0.8539 0.9870      0.07403     2088749523    5
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3919 0.8565 0.9912      0.10072     4096824733    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3924 0.8582 0.9931      0.10430     4099685092    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.3925 0.8582 0.9931      0.10763     4118728031    3
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.3926 0.8583 0.9932      0.26590     4114002227    2
```

</details>
<details><summary>`IVF65536_HNSW32,PQ64x4fs` </summary>
Index size 4084942469

 code_size 32

 log filename: autotune.dbbigann100M.IVF65536_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2771 0.5211 0.5631      0.00532       68399011    57
nprobe=1,quantizer_efSearch=8            0.1756 0.2834 0.2979      0.00249       17103670    121
nprobe=1,quantizer_efSearch=16           0.1793 0.2879 0.3025      0.00313       17085123    96
nprobe=2,quantizer_efSearch=4            0.2112 0.3704 0.3945      0.00317       34318819    95
nprobe=2,quantizer_efSearch=8            0.2293 0.4033 0.4289      0.00351       34250189    86
nprobe=2,quantizer_efSearch=16           0.2349 0.4110 0.4370      0.00419       34213222    72
nprobe=4,quantizer_efSearch=4            0.2560 0.4715 0.5111      0.00473       68518618    64
nprobe=4,quantizer_efSearch=8            0.2771 0.5211 0.5631      0.00532       68399011    57
nprobe=4,quantizer_efSearch=16           0.2814 0.5308 0.5736      0.00589       68304009    51
nprobe=4,quantizer_efSearch=32           0.2829 0.5333 0.5762      0.00696       68252083    44
nprobe=8,quantizer_efSearch=4            0.3087 0.6113 0.6739      0.00785      136742870    39
nprobe=8,quantizer_efSearch=8            0.3164 0.6278 0.6910      0.00763      136640783    40
nprobe=8,quantizer_efSearch=16           0.3265 0.6480 0.7129      0.00857      136459814    35
nprobe=8,quantizer_efSearch=32           0.3282 0.6522 0.7175      0.00949      136317942    32
nprobe=16,quantizer_efSearch=4           0.3326 0.6759 0.7561      0.01074      271953777    28
nprobe=16,quantizer_efSearch=16          0.3512 0.7271 0.8129      0.01188      271357642    26
nprobe=16,quantizer_efSearch=32          0.3538 0.7354 0.8220      0.01447      271032279    21
nprobe=32,quantizer_efSearch=8           0.3573 0.7531 0.8534      0.01476      539012704    21
nprobe=32,quantizer_efSearch=16          0.3665 0.7771 0.8829      0.01765      537898912    17
nprobe=32,quantizer_efSearch=32          0.3692 0.7872 0.8944      0.02061      537249290    15
nprobe=32,quantizer_efSearch=64          0.3703 0.7894 0.8965      0.02208      536988732    14
nprobe=64,quantizer_efSearch=16          0.3728 0.8075 0.9241      0.02531     1064247249    12
nprobe=64,quantizer_efSearch=32          0.3771 0.8201 0.9408      0.02732     1062496817    12
nprobe=64,quantizer_efSearch=64          0.3791 0.8245 0.9457      0.03385     1061623065    9
nprobe=64,quantizer_efSearch=128         0.3793 0.8255 0.9470      0.03781     1061290123    8
nprobe=128,quantizer_efSearch=32         0.3795 0.8353 0.9639      0.03893     2094488150    8
nprobe=128,quantizer_efSearch=64         0.3827 0.8415 0.9712      0.04287     2090984304    7
nprobe=128,quantizer_efSearch=128        0.3828 0.8432 0.9732      0.04538     2089619224    7
nprobe=256,quantizer_efSearch=64         0.3839 0.8486 0.9828      0.06008     4111546907    5
nprobe=256,quantizer_efSearch=128        0.3841 0.8503 0.9853      0.06131     4105567534    5
nprobe=512,quantizer_efSearch=64         0.3846 0.8496 0.9857      0.08897     8046544476    4
nprobe=512,quantizer_efSearch=128        0.3848 0.8519 0.9898      0.10413     8059547740    3
```

</details>
<details><summary>`OPQ32_128,IVF1048576_HNSW32,PQ32` </summary>
Index size 4830818224

 code_size 32

 log filename: autotune.dbbigann100M.OPQ32_128_IVF1048576_HNSW32_PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6445 0.9280 0.9312      0.45465      278471146    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0022 0.0025 0.0025      0.00368        1165579    82
nprobe=1,quantizer_efSearch=4,ht=106      0.1029 0.1126 0.1126      0.00396        1165579    76
nprobe=2,quantizer_efSearch=4,ht=102      0.1225 0.1371 0.1371      0.00494        2322684    61
nprobe=1,quantizer_efSearch=16,ht=114     0.1440 0.1582 0.1582      0.00776        1154921    39
nprobe=1,quantizer_efSearch=16,ht=116     0.1474 0.1626 0.1626      0.00749        1154921    41
nprobe=2,quantizer_efSearch=16,ht=110     0.1927 0.2187 0.2187      0.00906        2308235    34
nprobe=4,quantizer_efSearch=8,ht=118      0.2953 0.3512 0.3513      0.00915        4634699    33
nprobe=4,quantizer_efSearch=32,ht=256     0.3171 0.3802 0.3803      0.01316        4596765    23
nprobe=8,quantizer_efSearch=16,ht=114     0.3719 0.4553 0.4555      0.01693        9220972    18
nprobe=8,quantizer_efSearch=16,ht=128     0.3962 0.4952 0.4956      0.01754        9220972    18
nprobe=16,quantizer_efSearch=4,ht=114     0.4028 0.5065 0.5068      0.02488       18458614    13
nprobe=16,quantizer_efSearch=4,ht=122     0.4224 0.5403 0.5408      0.02561       18458614    12
nprobe=16,quantizer_efSearch=4,ht=126     0.4244 0.5439 0.5444      0.02576       18458614    12
nprobe=16,quantizer_efSearch=8,ht=118     0.4476 0.5727 0.5733      0.02607       18424044    12
nprobe=16,quantizer_efSearch=64,ht=256    0.4787 0.6241 0.6246      0.02793       18251663    11
nprobe=32,quantizer_efSearch=8,ht=124     0.5040 0.6668 0.6677      0.04873       36648070    7
nprobe=32,quantizer_efSearch=32,ht=116    0.5234 0.6902 0.6910      0.05380       36392502    6
nprobe=64,quantizer_efSearch=64,ht=126    0.5926 0.8226 0.8245      0.10690       71857153    3
nprobe=64,quantizer_efSearch=128,ht=126   0.5952 0.8271 0.8290      0.12091       71732916    3
nprobe=128,quantizer_efSearch=512,ht=256  0.6312 0.9006 0.9035      0.17748      141446279    2
nprobe=256,quantizer_efSearch=512,ht=116  0.6341 0.8984 0.9002      0.45425      278471146    1
nprobe=256,quantizer_efSearch=512,ht=120  0.6445 0.9280 0.9312      0.45492      278471146    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6473 0.9352 0.9389      0.45176      278471146    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6496 0.9469 0.9514      0.72031      550265079    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6607 0.9751 0.9807      1.50537     1073452343    1
nprobe=1024,quantizer_efSearch=512,ht=128 0.6643 0.9820 0.9884      1.52571     1073452343    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6658 0.9895 0.9966      2.38749     4068395676    1
```

</details>
<details><summary>`OPQ32_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ32` </summary>
Index size 4588470772

 code_size 32

 log filename: autotune.dbbigann100M.OPQ32_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                0.2884 0.3625 0.3631      0.68607      543327076    1
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=66       0.0045 0.0050 0.0050      0.00343        1890279    88
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=72       0.0073 0.0080 0.0080      0.00342        1890202    88
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=88       0.0261 0.0290 0.0290      0.00357        2614183    84
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=96      0.0506 0.0549 0.0549      0.00402        1530306    75
nprobe=1,quantizer_k_factor_rf=64,quantizer_nprobe=2,ht=126     0.1388 0.1556 0.1556      0.00419        1892009    72
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=114      0.1983 0.2274 0.2275      0.00532        3772775    57
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=128      0.2180 0.2521 0.2522      0.00568        5162257    53
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=256    0.2288 0.2649 0.2650      0.00604        7860531    50
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=116      0.2756 0.3290 0.3293      0.00794        6098773    38
nprobe=4,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=128     0.2870 0.3463 0.3467      0.01084        6091994    28
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128    0.3140 0.3781 0.3785      0.01113       10155652    27
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=120      0.3571 0.4424 0.4428      0.01357       10734588    23
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=128    0.3984 0.4986 0.4991      0.01686       14737459    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=114     0.4277 0.5475 0.5482      0.02609       21257547    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=128    0.4728 0.6165 0.6174      0.02835       23909257    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=128    0.4761 0.6231 0.6240      0.03347       39306245    9
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=256   0.5377 0.7262 0.7277      0.04738       42004548    7
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=126   0.5412 0.7351 0.7366      0.07267      116236328    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=120    0.5766 0.7932 0.7955      0.09586       77782385    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=122    0.5890 0.8168 0.8196      0.09660       82531472    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=256   0.6138 0.8686 0.8715      0.08995      148186616    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=122   0.6262 0.8918 0.8953      0.35829      287099890    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=122  0.6470 0.9363 0.9405      0.37480      319400050    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=512,ht=256 0.6597 0.9714 0.9771      0.53518      711282443    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=512,ht=256 0.6660 0.9919 0.9993      2.33629     4260487971    1
```

</details>
<details><summary>`OPQ32_128,IVF262144_HNSW32,PQ32` </summary>
Index size 4207854768

 code_size 32

 log filename: autotune.dbbigann100M.OPQ32_128_IVF262144_HNSW32_PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6433 0.9653 0.9722      0.45923     1054136145    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0028 0.0037 0.0037      0.00324        4325543    93
nprobe=1,quantizer_efSearch=4,ht=68       0.0079 0.0099 0.0099      0.00317        4325543    95
nprobe=1,quantizer_efSearch=4,ht=106      0.1482 0.1699 0.1702      0.00327        4325543    92
nprobe=2,quantizer_efSearch=4,ht=102      0.1772 0.2096 0.2102      0.00457        8656815    66
nprobe=1,quantizer_efSearch=16,ht=116     0.1971 0.2289 0.2292      0.00654        4287754    46
nprobe=2,quantizer_efSearch=16,ht=110     0.2611 0.3116 0.3122      0.00757        8595821    40
nprobe=4,quantizer_efSearch=8,ht=118      0.3663 0.4634 0.4643      0.00906       17269371    34
nprobe=4,quantizer_efSearch=32,ht=256     0.3853 0.4907 0.4918      0.01145       17211297    27
nprobe=8,quantizer_efSearch=16,ht=114     0.4463 0.5819 0.5833      0.01651       34440560    19
nprobe=8,quantizer_efSearch=16,ht=128     0.4606 0.6102 0.6118      0.01844       34440560    17
nprobe=16,quantizer_efSearch=4,ht=114     0.4674 0.6243 0.6262      0.02754       68984057    11
nprobe=16,quantizer_efSearch=64,ht=256    0.5291 0.7333 0.7356      0.02844       68624102    11
nprobe=32,quantizer_efSearch=8,ht=124     0.5493 0.7744 0.7775      0.05644      137064959    6
nprobe=32,quantizer_efSearch=32,ht=116    0.5729 0.8072 0.8098      0.05689      136572012    6
nprobe=128,quantizer_efSearch=64,ht=256   0.6325 0.9412 0.9470      0.11801      535213636    3
nprobe=128,quantizer_efSearch=512,ht=256  0.6348 0.9468 0.9528      0.18629      534400545    2
nprobe=256,quantizer_efSearch=512,ht=116  0.6370 0.9481 0.9535      0.43818     1054136145    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6444 0.9692 0.9763      0.46042     1054136145    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6457 0.9726 0.9803      0.78086     2078646325    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6494 0.9847 0.9931      1.59940     4072977444    1
nprobe=2048,quantizer_efSearch=512,ht=122 0.6498 0.9859 0.9944      3.08593     7987294037    1
nprobe=4096,quantizer_efSearch=512,ht=256 0.6501 0.9899 0.9987      3.25038    15298536805    1
```

</details>
<details><summary>`OPQ32_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ32` </summary>
Index size 4147540468

 code_size 32

 log filename: autotune.dbbigann100M.OPQ32_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.0003 0.0003 0.0003      0.00778       19193301    39
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=2         0.0001 0.0001 0.0001      0.00258        4553581    117
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=74        0.0118 0.0143 0.0143      0.00262        4541259    115
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=108       0.1546 0.1778 0.1779      0.00281        4684663    107
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=120       0.1923 0.2285 0.2288      0.00451        8952434    67
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=1,ht=256      0.2091 0.2515 0.2519      0.00458        8886844    66
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=122      0.2604 0.3149 0.3152      0.00577       11390535    53
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=122      0.2892 0.3519 0.3523      0.00593       11323723    51
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2,ht=120       0.3006 0.3731 0.3736      0.00841       17925294    36
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=116      0.3689 0.4631 0.4637      0.00874       20011706    35
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=256       0.3693 0.4822 0.4832      0.00946       35274488    32
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=256    0.3863 0.4909 0.4915      0.01419       37822039    22
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4,ht=118       0.4012 0.5173 0.5183      0.01433       35730732    21
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=124       0.4539 0.6016 0.6026      0.01633       35960890    19
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=126      0.4582 0.6086 0.6096      0.01680       37235294    18
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=128     0.4652 0.6174 0.6185      0.02324       44950314    13
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=126     0.5115 0.6985 0.7001      0.03117       74786398    10
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=128    0.5275 0.7283 0.7302      0.03337       71576907    9
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=116      0.5469 0.7685 0.7713      0.05264      138921343    6
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=124    0.5623 0.7986 0.8015      0.06143      159029095    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=120     0.5782 0.8275 0.8307      0.05468      142041932    6
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=256     0.5826 0.8400 0.8435      0.11915      274457543    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=116     0.5979 0.8624 0.8661      0.10248      274589782    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=116    0.6042 0.8752 0.8792      0.11199      276441711    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=126    0.6144 0.9004 0.9049      0.11821      276441711    3
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=128    0.6172 0.9039 0.9084      0.12029      281261835    3
nprobe=128,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=256  0.6339 0.9478 0.9539      0.26247      555728381    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=256   0.6409 0.9670 0.9742      0.31303     1063902657    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=118  0.6436 0.9743 0.9819      1.47324     4096199055    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=120   0.6464 0.9799 0.9882      1.50565     4092510106    1
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=120 0.6465 0.9815 0.9899      1.64118     4113972220    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=126   0.6491 0.9867 0.9956      1.59815     4092510106    1
nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=256  0.6495 0.9888 0.9979      1.90930     8020740090    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=256 0.6496 0.9909 1.0000      4.90509    15672910993    1
```

</details>
<details><summary>`OPQ32_128,IVF65536_HNSW32,PQ32` </summary>
Index size 4052113072

 code_size 32

 log filename: autotune.dbbigann100M.OPQ32_128_IVF65536_HNSW32_PQ32.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.6295 0.9811 0.9903      0.47690     4103270412    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0049 0.0056 0.0056      0.00294       17144696    102
nprobe=1,quantizer_efSearch=4,ht=106      0.1965 0.2364 0.2368      0.00313       17144696    96
nprobe=1,quantizer_efSearch=16,ht=114     0.2388 0.2938 0.2942      0.00431       17084626    70
nprobe=1,quantizer_efSearch=16,ht=116     0.2421 0.2988 0.2993      0.00448       17084626    68
nprobe=2,quantizer_efSearch=16,ht=110     0.3194 0.4029 0.4036      0.00581       34210550    52
nprobe=2,quantizer_efSearch=64,ht=120     0.3385 0.4372 0.4380      0.00999       34175958    31
nprobe=4,quantizer_efSearch=8,ht=118      0.4139 0.5588 0.5608      0.00993       68419998    31
nprobe=4,quantizer_efSearch=32,ht=116     0.4210 0.5679 0.5698      0.01111       68254816    27
nprobe=4,quantizer_efSearch=32,ht=118     0.4231 0.5720 0.5741      0.01139       68254816    27
nprobe=4,quantizer_efSearch=32,ht=256     0.4249 0.5770 0.5792      0.01088       68254816    28
nprobe=8,quantizer_efSearch=32,ht=106     0.4501 0.6043 0.6062      0.01616      136318195    19
nprobe=8,quantizer_efSearch=16,ht=114     0.4938 0.6946 0.6977      0.01678      136466868    18
nprobe=8,quantizer_efSearch=32,ht=116     0.5011 0.7083 0.7114      0.01806      136318195    17
nprobe=8,quantizer_efSearch=64,ht=116     0.5020 0.7093 0.7124      0.01993      136290846    16
nprobe=8,quantizer_efSearch=64,ht=126     0.5053 0.7193 0.7225      0.02479      136290846    13
nprobe=16,quantizer_efSearch=4,ht=114     0.5121 0.7368 0.7402      0.03015      272019552    10
nprobe=16,quantizer_efSearch=64,ht=110    0.5337 0.7592 0.7623      0.03309      270936206    10
nprobe=16,quantizer_efSearch=64,ht=256    0.5605 0.8238 0.8279      0.03263      270936206    10
nprobe=32,quantizer_efSearch=32,ht=116    0.5888 0.8830 0.8882      0.06026      537253412    5
nprobe=32,quantizer_efSearch=512,ht=128   0.5948 0.8980 0.9040      0.11619      536869025    3
nprobe=64,quantizer_efSearch=256,ht=114   0.6075 0.9231 0.9288      0.12788     1061223348    3
nprobe=64,quantizer_efSearch=64,ht=126    0.6147 0.9451 0.9527      0.14848     1061617627    3
nprobe=64,quantizer_efSearch=128,ht=126   0.6151 0.9464 0.9540      0.14941     1061291070    3
nprobe=128,quantizer_efSearch=64,ht=256   0.6247 0.9703 0.9790      0.21867     2090958305    2
nprobe=128,quantizer_efSearch=512,ht=256  0.6254 0.9723 0.9810      0.24186     2089262765    2
nprobe=256,quantizer_efSearch=512,ht=116  0.6264 0.9718 0.9800      0.44641     4103270412    1
nprobe=256,quantizer_efSearch=512,ht=120  0.6295 0.9811 0.9903      0.49314     4103270412    1
nprobe=256,quantizer_efSearch=512,ht=122  0.6301 0.9830 0.9924      0.50510     4103270412    1
nprobe=1024,quantizer_efSearch=512,ht=122 0.6318 0.9880 0.9983      1.79748    15769123590    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64x4fs` </summary>
Index size 5360042956

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF1048576_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3721 0.8255 0.9412      0.12032      278463429    3
nprobe=2,quantizer_efSearch=4            0.1434 0.2200 0.2283      0.00260        2321300    116
nprobe=4,quantizer_efSearch=4            0.1796 0.2970 0.3135      0.00289        4635580    104
nprobe=4,quantizer_efSearch=8            0.2050 0.3460 0.3625      0.00411        4630851    73
nprobe=8,quantizer_efSearch=4            0.2370 0.4284 0.4573      0.00412        9269396    73
nprobe=8,quantizer_efSearch=8            0.2438 0.4391 0.4690      0.00484        9261760    62
nprobe=16,quantizer_efSearch=4           0.2631 0.5058 0.5456      0.00553       18449946    55
nprobe=16,quantizer_efSearch=8           0.2827 0.5467 0.5895      0.00659       18422131    46
nprobe=16,quantizer_efSearch=16          0.2908 0.5655 0.6089      0.00803       18370492    38
nprobe=32,quantizer_efSearch=8           0.3054 0.6118 0.6697      0.00821       36645457    37
nprobe=32,quantizer_efSearch=16          0.3188 0.6490 0.7099      0.01045       36514786    29
nprobe=32,quantizer_efSearch=32          0.3272 0.6646 0.7263      0.01328       36389661    23
nprobe=64,quantizer_efSearch=16          0.3324 0.6959 0.7759      0.01442       72395803    21
nprobe=64,quantizer_efSearch=32          0.3434 0.7258 0.8082      0.01755       72092968    18
nprobe=128,quantizer_efSearch=32         0.3546 0.7664 0.8644      0.02331      142666771    13
nprobe=128,quantizer_efSearch=64         0.3625 0.7883 0.8884      0.03033      142090294    10
nprobe=256,quantizer_efSearch=64         0.3668 0.8113 0.9236      0.04000      280539575    8
nprobe=256,quantizer_efSearch=128        0.3713 0.8233 0.9383      0.05211      279312529    6
nprobe=256,quantizer_efSearch=256        0.3716 0.8256 0.9411      0.07262      278674799    5
nprobe=512,quantizer_efSearch=128        0.3722 0.8367 0.9611      0.07441      550259136    5
nprobe=512,quantizer_efSearch=256        0.3732 0.8413 0.9670      0.10585      548053339    3
nprobe=1024,quantizer_efSearch=256       0.3754 0.8477 0.9801      0.13573     1077172010    3
nprobe=1024,quantizer_efSearch=512       0.3766 0.8492 0.9821      0.19674     1073442874    2
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64x4fsr` </summary>
Index size 5360027596

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF1048576_HNSW32_PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4567 0.6203 0.6221      0.02711       18294660    12
nprobe=1,quantizer_efSearch=4            0.1337 0.1536 0.1536      0.00338        1165146    89
nprobe=2,quantizer_efSearch=4            0.1927 0.2273 0.2276      0.00400        2322629    75
nprobe=4,quantizer_efSearch=4            0.2531 0.3115 0.3122      0.00519        4636661    58
nprobe=4,quantizer_efSearch=8            0.2917 0.3621 0.3629      0.00691        4631366    44
nprobe=8,quantizer_efSearch=4            0.3561 0.4568 0.4578      0.00864        9272922    35
nprobe=8,quantizer_efSearch=8            0.3647 0.4689 0.4699      0.00924        9263494    33
nprobe=8,quantizer_efSearch=16           0.3841 0.4951 0.4961      0.01227        9220895    25
nprobe=8,quantizer_efSearch=32           0.3889 0.5011 0.5021      0.01761        9186070    18
nprobe=16,quantizer_efSearch=4           0.4072 0.5450 0.5465      0.01780       18452984    17
nprobe=16,quantizer_efSearch=8           0.4355 0.5891 0.5908      0.01916       18422043    16
nprobe=16,quantizer_efSearch=16          0.4502 0.6093 0.6111      0.02026       18370343    15
nprobe=16,quantizer_efSearch=32          0.4567 0.6203 0.6221      0.02674       18294660    12
nprobe=16,quantizer_efSearch=64          0.4595 0.6226 0.6244      0.03387       18251552    9
nprobe=32,quantizer_efSearch=8           0.4788 0.6692 0.6721      0.03564       36648505    9
nprobe=32,quantizer_efSearch=16          0.5040 0.7090 0.7121      0.03696       36518604    9
nprobe=32,quantizer_efSearch=32          0.5141 0.7258 0.7289      0.03932       36392112    8
nprobe=32,quantizer_efSearch=64          0.5198 0.7337 0.7367      0.05099       36290143    6
nprobe=32,quantizer_efSearch=128         0.5212 0.7358 0.7388      0.06615       36244101    5
nprobe=64,quantizer_efSearch=16          0.5369 0.7745 0.7791      0.07409       72401331    5
nprobe=64,quantizer_efSearch=32          0.5538 0.8075 0.8123      0.07622       72099984    4
nprobe=64,quantizer_efSearch=64          0.5634 0.8220 0.8268      0.07992       71860533    4
nprobe=64,quantizer_efSearch=128         0.5666 0.8271 0.8317      0.09681       71736887    4
nprobe=64,quantizer_efSearch=256         0.5673 0.8283 0.8329      0.12807       71690472    3
nprobe=128,quantizer_efSearch=32         0.5811 0.8639 0.8702      0.14047      142679331    3
nprobe=128,quantizer_efSearch=64         0.5910 0.8869 0.8937      0.14339      142093765    3
nprobe=128,quantizer_efSearch=128        0.5950 0.8951 0.9021      0.15264      141672097    2
nprobe=128,quantizer_efSearch=256        0.5965 0.8965 0.9036      0.17611      141499766    2
nprobe=256,quantizer_efSearch=64         0.6050 0.9215 0.9296      0.28036      280546527    2
nprobe=256,quantizer_efSearch=128        0.6115 0.9360 0.9447      0.28836      279326096    2
nprobe=256,quantizer_efSearch=256        0.6149 0.9393 0.9477      0.30832      278683984    1
nprobe=512,quantizer_efSearch=128        0.6206 0.9589 0.9680      0.52809      550267158    1
nprobe=512,quantizer_efSearch=256        0.6237 0.9644 0.9740      0.56624      548067908    1
nprobe=512,quantizer_efSearch=512        0.6254 0.9662 0.9753      0.58875      547091726    1
nprobe=1024,quantizer_efSearch=256       0.6263 0.9766 0.9876      1.00819     1077197105    1
nprobe=1024,quantizer_efSearch=512       0.6265 0.9789 0.9898      1.05065     1073456094    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 5118383632

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3665 0.8115 0.9215      0.03949      324561618    8
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1020 0.1448 0.1480      0.00199        1898723    151
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1046 0.1501 0.1535      0.00203        1894030    148
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1149 0.1738 0.1806      0.00210        2705253    143
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1181 0.1806 0.1877      0.00217        2694810    139
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1244 0.1901 0.1955      0.00222        3077257    136
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1384 0.2178 0.2254      0.00233        3055563    129
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1454 0.2321 0.2452      0.00250        5031298    120
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1598 0.2609 0.2730      0.00258        5424424    117
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1743 0.2880 0.3029      0.00267        5394806    113
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1766 0.2931 0.3093      0.00278        5388357    109
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1907 0.3212 0.3373      0.00296        6103271    102
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1956 0.3375 0.3589      0.00323       10114012    93
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2010 0.3398 0.3559      0.00358        7486410    84
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2167 0.3840 0.4078      0.00352       10813314    86
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2321 0.4128 0.4424      0.00364       10746538    83
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2344 0.4213 0.4511      0.00435       10725558    69
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2491 0.4511 0.4817      0.00445       12098881    68
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2603 0.4955 0.5358      0.00498       19981292    61
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2688 0.5080 0.5462      0.00512       21473629    59
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2781 0.5386 0.5880      0.00664       38689112    46
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2826 0.5488 0.5914      0.00659       21243977    46
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2928 0.5717 0.6148      0.00699       23917851    43
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3103 0.6239 0.6818      0.00771       39542069    39
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3245 0.6613 0.7235      0.00964       42040102    32
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3255 0.6625 0.7246      0.01125       42024513    27
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3276 0.6715 0.7339      0.01184       47188584    26
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3336 0.6965 0.7702      0.01175       78916917    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3403 0.7207 0.8019      0.01272       77970793    24
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3424 0.7241 0.8064      0.01436       77782924    21
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3455 0.7366 0.8190      0.01475       82952423    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3472 0.7394 0.8235      0.01635       82725304    19
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.3475 0.7483 0.8391      0.01777      150854826    17
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.3509 0.7441 0.8283      0.02279       92308290    14
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3624 0.7902 0.8912      0.02473      152821473    13
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3651 0.7957 0.8980      0.02894      162324614    11
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3679 0.8147 0.9281      0.03352      291188640    9
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.3692 0.8164 0.9299      0.04021      290805556    8
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3713 0.8225 0.9376      0.03799      300667975    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3727 0.8272 0.9429      0.04904      320211486    7
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3737 0.8340 0.9580      0.05443      600977386    6
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3745 0.8422 0.9693      0.07493      588303330    5
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.3759 0.8475 0.9785      0.08418     1141544031    4
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.3767 0.8505 0.9834      0.11062     1157019167    3
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.3776 0.8518 0.9871      0.14927     2234301549    3
```

</details>
<details><summary>`OPQ64_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 5118294544

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3699 0.4683 0.4693      0.00526       12123382    57
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1338 0.1551 0.1551      0.00199        1893478    151
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1988 0.2347 0.2349      0.00240        3776244    125
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2103 0.2498 0.2499      0.00270        3769331    111
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2415 0.2953 0.2959      0.00284        5412913    106
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2693 0.3291 0.3296      0.00325        6113231    93
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2789 0.3469 0.3474      0.00332        6091593    91
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.3053 0.3906 0.3914      0.00456       10038252    66
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3066 0.3927 0.3935      0.00461       10037697    66
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3451 0.4382 0.4391      0.00472       10761087    64
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3507 0.4494 0.4503      0.00497       10724464    61
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3699 0.4683 0.4693      0.00526       12123505    58
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3777 0.4823 0.4832      0.00544       12085507    56
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3896 0.4984 0.4993      0.00667       14687987    46
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3898 0.4993 0.5002      0.00737       14692348    41
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.3928 0.5023 0.5032      0.00885       19658775    34
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4044 0.5413 0.5425      0.01441       19917340    21
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4335 0.5787 0.5802      0.01446       21339165    21
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4409 0.5917 0.5931      0.01493       21238173    21
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4561 0.6151 0.6165      0.01603       23848362    19
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4609 0.6213 0.6229      0.02076       37441064    15
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.4618 0.6234 0.6251      0.02064       28968302    15
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4845 0.6775 0.6796      0.02838       39639418    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4950 0.6908 0.6932      0.02821       39442666    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4958 0.6906 0.6931      0.02803       39434627    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5177 0.7238 0.7265      0.02804       41911438    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5184 0.7250 0.7278      0.02878       41952905    11
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5237 0.7354 0.7382      0.03084       46835948    10
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5238 0.7355 0.7382      0.03161       46895000    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5247 0.7363 0.7389      0.03260       56343585    10
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5257 0.7377 0.7404      0.03397       56578050    9
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5259 0.7373 0.7400      0.03795       75530247    8
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=256  0.5261 0.7374 0.7401      0.04696      117312568    7
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.5602 0.8099 0.8139      0.06508       82953789    5
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.5630 0.8138 0.8177      0.06754       92409830    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5719 0.8295 0.8334      0.07137       92078068    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5727 0.8298 0.8336      0.06214       91274543    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5798 0.8587 0.8630      0.11985      149203561    3
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5920 0.8815 0.8865      0.11105      153647069    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6038 0.9001 0.9048      0.12391      162890581    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.6039 0.9003 0.9050      0.12807      183100918    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6124 0.9311 0.9378      0.22973      290925692    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6150 0.9354 0.9421      0.23621      321593516    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6194 0.9429 0.9499      0.25252      320737589    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.6257 0.9640 0.9727      0.45973      593943473    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.6259 0.9640 0.9727      0.43095      631712875    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.6270 0.9682 0.9766      0.44760      589560872    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.6272 0.9686 0.9770      0.43876      588264650    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6273 0.9688 0.9772      0.46654      629957233    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6291 0.9817 0.9915      0.88641     1115459166    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=512 0.6294 0.9796 0.9889      0.83395     1245269705    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=512 0.6305 0.9818 0.9914      0.89659     1231540361    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fs` </summary>
Index size 4339648716

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF262144_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3802 0.8440 0.9723      0.11346     1054172322    3
nprobe=2,quantizer_efSearch=4            0.1787 0.2961 0.3132      0.00229        8654931    132
nprobe=4,quantizer_efSearch=4            0.2203 0.3916 0.4188      0.00283       17333599    107
nprobe=4,quantizer_efSearch=8            0.2461 0.4402 0.4698      0.00375       17271348    80
nprobe=8,quantizer_efSearch=4            0.2734 0.5182 0.5634      0.00436       34549205    69
nprobe=8,quantizer_efSearch=8            0.2838 0.5365 0.5819      0.00508       34523269    60
nprobe=16,quantizer_efSearch=4           0.2946 0.5892 0.6507      0.00614       68970616    49
nprobe=8,quantizer_efSearch=16           0.3009 0.5639 0.6106      0.00641       34438703    47
nprobe=16,quantizer_efSearch=8           0.3164 0.6351 0.7003      0.00702       68868071    43
nprobe=16,quantizer_efSearch=16          0.3276 0.6536 0.7200      0.00812       68767724    37
nprobe=32,quantizer_efSearch=8           0.3365 0.6941 0.7772      0.01003      137065427    30
nprobe=32,quantizer_efSearch=16          0.3517 0.7290 0.8169      0.01132      136804732    27
nprobe=32,quantizer_efSearch=32          0.3532 0.7386 0.8283      0.01403      136579826    22
nprobe=64,quantizer_efSearch=16          0.3607 0.7680 0.8693      0.01660      271545974    19
nprobe=64,quantizer_efSearch=32          0.3660 0.7867 0.8917      0.01921      270956854    16
nprobe=64,quantizer_efSearch=64          0.3676 0.7926 0.8989      0.02412      270582112    13
nprobe=128,quantizer_efSearch=32         0.3709 0.8121 0.9270      0.02693      536461331    12
nprobe=128,quantizer_efSearch=64         0.3753 0.8218 0.9404      0.03164      535230722    10
nprobe=128,quantizer_efSearch=128        0.3770 0.8249 0.9444      0.04192      534659816    8
nprobe=256,quantizer_efSearch=64         0.3773 0.8350 0.9624      0.04293     1057404331    7
nprobe=256,quantizer_efSearch=128        0.3801 0.8419 0.9704      0.05360     1055245882    6
nprobe=512,quantizer_efSearch=128        0.3806 0.8470 0.9798      0.07300     2078690442    5
nprobe=512,quantizer_efSearch=256        0.3808 0.8495 0.9828      0.09205     2075186087    4
nprobe=1024,quantizer_efSearch=128       0.3809 0.8479 0.9837      0.10454     4061585107    3
nprobe=1024,quantizer_efSearch=256       0.3811 0.8514 0.9875      0.12960     4077495625    3
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64x4fsr` </summary>
Index size 4339612876

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF262144_HNSW32_PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5102 0.7291 0.7330      0.02299       68665865    14
nprobe=1,quantizer_efSearch=4            0.1804 0.2162 0.2167      0.00294        4322407    102
nprobe=1,quantizer_efSearch=16           0.1977 0.2365 0.2370      0.00661        4292539    46
nprobe=2,quantizer_efSearch=16           0.2828 0.3525 0.3534      0.00731        8599956    42
nprobe=4,quantizer_efSearch=16           0.3707 0.4862 0.4876      0.00855       17235943    36
nprobe=8,quantizer_efSearch=16           0.4443 0.6093 0.6122      0.01102       34442006    28
nprobe=8,quantizer_efSearch=32           0.4485 0.6146 0.6175      0.01490       34395136    21
nprobe=16,quantizer_efSearch=16          0.5043 0.7191 0.7229      0.01999       68766932    16
nprobe=16,quantizer_efSearch=32          0.5102 0.7291 0.7330      0.02293       68665865    14
nprobe=16,quantizer_efSearch=64          0.5118 0.7316 0.7355      0.02849       68614596    11
nprobe=32,quantizer_efSearch=16          0.5538 0.8156 0.8212      0.03470      136791419    9
nprobe=32,quantizer_efSearch=32          0.5580 0.8271 0.8332      0.03666      136569592    9
nprobe=32,quantizer_efSearch=64          0.5609 0.8321 0.8383      0.04232      136423731    8
nprobe=32,quantizer_efSearch=128         0.5614 0.8326 0.8388      0.05240      136371924    6
nprobe=64,quantizer_efSearch=16          0.5765 0.8667 0.8739      0.06686      271514574    5
nprobe=64,quantizer_efSearch=128         0.5945 0.9002 0.9076      0.08331      270396629    4
nprobe=64,quantizer_efSearch=256         0.5956 0.9011 0.9085      0.10372      270337195    3
nprobe=128,quantizer_efSearch=128        0.6098 0.9421 0.9517      0.13809      534597552    3
nprobe=128,quantizer_efSearch=256        0.6103 0.9434 0.9529      0.15456      534381251    2
nprobe=256,quantizer_efSearch=128        0.6185 0.9674 0.9786      0.23227     1055111687    2
nprobe=256,quantizer_efSearch=256        0.6191 0.9687 0.9799      0.24936     1054219156    2
nprobe=512,quantizer_efSearch=128        0.6226 0.9774 0.9887      0.40980     2078431243    1
nprobe=512,quantizer_efSearch=256        0.6244 0.9794 0.9916      0.42505     2074928950    1
nprobe=512,quantizer_efSearch=512        0.6246 0.9798 0.9922      0.48303     2073774460    1
nprobe=1024,quantizer_efSearch=256       0.6252 0.9844 0.9964      0.84529     4076966885    1
nprobe=1024,quantizer_efSearch=512       0.6255 0.9854 0.9971      0.90260     4072554553    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 4279433744

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3626 0.7734 0.8753      0.01618      281240606    19
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1123 0.1698 0.1759      0.00158        4535034    190
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1318 0.2027 0.2098      0.00178        4685318    169
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1356 0.2164 0.2273      0.00185        8969978    163
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1436 0.2354 0.2495      0.00191        8888437    157
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1586 0.2576 0.2708      0.00192        9126064    157
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1890 0.3057 0.3222      0.00205        9389897    147
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1916 0.3138 0.3318      0.00221        9351654    136
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1987 0.3294 0.3475      0.00237       10018581    127
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2115 0.3730 0.3990      0.00256       17847319    117
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2138 0.3775 0.4047      0.00264       17785681    114
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2272 0.3860 0.4096      0.00259       18226301    116
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2392 0.4203 0.4480      0.00266       18124330    113
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2420 0.4261 0.4553      0.00277       18069972    109
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2429 0.4275 0.4563      0.00295       18062817    102
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.2430 0.4278 0.4565      0.00328       18063633    92
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.2553 0.4518 0.4804      0.00349       20030698    87
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.2580 0.4599 0.4896      0.00365       19979871    83
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.2587 0.4598 0.4898      0.00394       19973283    77
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2626 0.4823 0.5219      0.00368       35821178    82
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2779 0.5185 0.5596      0.00388       36364840    78
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2944 0.5529 0.5984      0.00403       36058099    75
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3008 0.5669 0.6133      0.00475       37210389    64
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3010 0.5676 0.6140      0.00516       37195576    59
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3128 0.6162 0.6744      0.00586       71365113    52
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3187 0.6375 0.7011      0.00602       70622598    50
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3283 0.6594 0.7255      0.00689       71609239    44
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.3288 0.6602 0.7265      0.00749       71579451    41
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3297 0.6614 0.7266      0.00766       74225144    40
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3311 0.6653 0.7314      0.00794       74104709    38
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3396 0.7019 0.7858      0.00914      139218360    33
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3402 0.7064 0.7861      0.00946      141832140    32
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3406 0.7038 0.7877      0.00998      138943153    31
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3537 0.7415 0.8297      0.01105      142330567    28
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3554 0.7443 0.8333      0.01316      142031391    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3564 0.7446 0.8340      0.01334      147054922    23
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.3567 0.7625 0.8619      0.01449      279423252    21
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.3626 0.7734 0.8753      0.01562      281250199    20
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3640 0.7803 0.8835      0.01501      275075525    21
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3684 0.7907 0.8962      0.01603      276945749    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.3693 0.7956 0.9021      0.01975      281265045    16
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.3707 0.8056 0.9206      0.02189      541692067    14
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3751 0.8149 0.9293      0.02478      556452152    13
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3772 0.8259 0.9450      0.02590      546611014    12
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3774 0.8263 0.9453      0.02842      556419105    11
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3778 0.8383 0.9652      0.03669     1065129081    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3801 0.8442 0.9725      0.04061     1077069461    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3803 0.8448 0.9737      0.04928     1075946969    7
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3805 0.8464 0.9786      0.05712     2136188897    6
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.3806 0.8471 0.9796      0.06144     2143892197    5
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3817 0.8514 0.9858      0.06432     2097969832    5
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3819 0.8528 0.9903      0.09714     4097487365    4
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 4279330320

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.5795 0.8839 0.8911      0.05908      274578635    6
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1     0.1439 0.1704 0.1709      0.00204        4526806    148
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.1637 0.1957 0.1961      0.00212        4687931    142
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1735 0.2076 0.2080      0.00214        4690121    141
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1752 0.2099 0.2104      0.00218        4681962    138
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1758 0.2103 0.2108      0.00226        4682050    133
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1782 0.2129 0.2133      0.00235        5017087    128
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1878 0.2246 0.2250      0.00235        5021525    128
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1883 0.2260 0.2265      0.00249        5016297    121
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.2327 0.2902 0.2912      0.00276        9075164    109
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.2419 0.3027 0.3038      0.00280        9041966    108
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2549 0.3174 0.3184      0.00296        9375741    102
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2666 0.3327 0.3338      0.00310        9344013    97
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2778 0.3468 0.3479      0.00340       10007894    89
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2787 0.3491 0.3502      0.00368       10011664    82
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.3086 0.4058 0.4075      0.00408       17769369    74
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3469 0.4555 0.4572      0.00417       18070325    72
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3483 0.4582 0.4599      0.00435       18062337    70
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3520 0.4604 0.4621      0.00454       18753411    67
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3651 0.4811 0.4828      0.00484       18678000    63
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3697 0.4868 0.4885      0.00545       19976468    56
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3720 0.4902 0.4919      0.00564       19962361    54
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4087 0.5588 0.5611      0.00637       35418477    48
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4272 0.5863 0.5884      0.00665       36171715    46
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4363 0.6010 0.6033      0.00684       35956577    44
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4440 0.6140 0.6162      0.00774       37194591    39
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4453 0.6150 0.6172      0.00803       37192984    38
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4461 0.6155 0.6178      0.00938       39734521    32
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4477 0.6167 0.6190      0.00971       39736577    31
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.4518 0.6342 0.6386      0.01471       70550602    21
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.4600 0.6460 0.6503      0.01575       70085289    20
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.4939 0.7053 0.7095      0.01673       70463214    18
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4979 0.7123 0.7169      0.01689       72022898    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.5069 0.7258 0.7301      0.01768       71560832    17
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5105 0.7315 0.7358      0.01913       74050791    16
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5109 0.7318 0.7361      0.02192       79130597    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.5336 0.7873 0.7935      0.02727      138919344    12
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5471 0.8089 0.8147      0.02760      140777855    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5559 0.8230 0.8296      0.02805      139740031    11
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5593 0.8315 0.8381      0.03112      142019093    10
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5597 0.8317 0.8384      0.03223      142009170    10
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5782 0.8749 0.8821      0.05841      277000159    6
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5795 0.8839 0.8911      0.05921      274580361    6
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5839 0.8900 0.8976      0.05992      278717180    6
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5853 0.8982 0.9057      0.06107      276398554    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5854 0.8983 0.9058      0.06278      276437579    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.5859 0.9008 0.9084      0.06893      281189334    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5864 0.9014 0.9090      0.07252      291299460    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16  0.5934 0.9171 0.9252      0.10932      545936989    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.5941 0.9193 0.9279      0.11157      540843633    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.6026 0.9403 0.9487      0.11335      541566933    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6030 0.9446 0.9536      0.11963      555514131    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.6032 0.9446 0.9536      0.12719      575688853    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.6034 0.9447 0.9537      0.13098      576105508    3
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.6105 0.9666 0.9773      0.21434     1086360190    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.6108 0.9631 0.9737      0.21598     1063890800    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.6109 0.9667 0.9771      0.21200     1076760852    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.6137 0.9693 0.9804      0.21591     1066312562    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6142 0.9702 0.9813      0.21981     1075530034    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.6143 0.9704 0.9815      0.23748     1074569645    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.6154 0.9766 0.9896      0.39048     2109101520    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.6192 0.9792 0.9921      0.39654     2088555001    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.6201 0.9809 0.9938      0.41180     2096298321    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fs` </summary>
Index size 4085042892

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF65536_HNSW32_PQ64x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2673 0.5188 0.5628      0.00621       68413060    49
nprobe=1,quantizer_efSearch=4            0.1567 0.2636 0.2782      0.00247       17136420    122
nprobe=1,quantizer_efSearch=8            0.1687 0.2825 0.2976      0.00279       17109598    108
nprobe=1,quantizer_efSearch=16           0.1722 0.2874 0.3025      0.00338       17085018    89
nprobe=2,quantizer_efSearch=4            0.2054 0.3674 0.3942      0.00376       34333404    80
nprobe=2,quantizer_efSearch=8            0.2213 0.4005 0.4285      0.00403       34252866    75
nprobe=2,quantizer_efSearch=16           0.2267 0.4083 0.4362      0.00458       34211992    66
nprobe=2,quantizer_efSearch=32           0.2270 0.4094 0.4374      0.00552       34181785    55
nprobe=4,quantizer_efSearch=4            0.2477 0.4695 0.5108      0.00589       68545900    51
nprobe=4,quantizer_efSearch=8            0.2673 0.5188 0.5628      0.00617       68413060    49
nprobe=4,quantizer_efSearch=16           0.2718 0.5293 0.5737      0.00676       68304176    45
nprobe=4,quantizer_efSearch=32           0.2727 0.5315 0.5763      0.00793       68256206    38
nprobe=8,quantizer_efSearch=4            0.3037 0.6121 0.6738      0.00901      136749099    34
nprobe=8,quantizer_efSearch=8            0.3109 0.6278 0.6913      0.00897      136649086    34
nprobe=8,quantizer_efSearch=16           0.3202 0.6482 0.7134      0.01032      136466324    30
nprobe=8,quantizer_efSearch=32           0.3214 0.6521 0.7178      0.01145      136321398    27
nprobe=16,quantizer_efSearch=4           0.3250 0.6777 0.7558      0.01248      271984096    25
nprobe=16,quantizer_efSearch=8           0.3385 0.7126 0.7972      0.01273      271722127    24
nprobe=16,quantizer_efSearch=16          0.3446 0.7258 0.8130      0.01430      271369498    22
nprobe=16,quantizer_efSearch=32          0.3484 0.7342 0.8226      0.01631      271038870    19
nprobe=32,quantizer_efSearch=8           0.3510 0.7548 0.8543      0.01636      539026602    19
nprobe=32,quantizer_efSearch=32          0.3648 0.7912 0.8955      0.01791      537273927    17
nprobe=64,quantizer_efSearch=16          0.3680 0.8060 0.9246      0.02027     1064263082    15
nprobe=64,quantizer_efSearch=32          0.3728 0.8211 0.9417      0.02453     1062531628    13
nprobe=128,quantizer_efSearch=32         0.3783 0.8348 0.9645      0.03078     2094473566    11
nprobe=128,quantizer_efSearch=64         0.3808 0.8408 0.9723      0.04117     2090983077    8
nprobe=256,quantizer_efSearch=64         0.3824 0.8473 0.9839      0.06388     4111528074    5
nprobe=256,quantizer_efSearch=128        0.3833 0.8494 0.9864      0.07382     4105629770    5
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64x4fsr` </summary>
Index size 4084928204

 code_size 32

 log filename: autotune.dbbigann100M.OPQ64_128_IVF65536_HNSW32_PQ64x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.4909 0.7175 0.7231      0.02415      136286205    13
nprobe=1,quantizer_efSearch=4            0.2165 0.2769 0.2777      0.00285       17130502    106
nprobe=1,quantizer_efSearch=16           0.2378 0.3030 0.3038      0.00467       17085401    65
nprobe=2,quantizer_efSearch=16           0.3253 0.4375 0.4393      0.00593       34211996    51
nprobe=2,quantizer_efSearch=32           0.3256 0.4381 0.4399      0.00781       34180544    39
nprobe=4,quantizer_efSearch=16           0.4116 0.5738 0.5772      0.00786       68308617    39
nprobe=4,quantizer_efSearch=32           0.4132 0.5759 0.5793      0.01008       68255248    30
nprobe=8,quantizer_efSearch=16           0.4874 0.7124 0.7180      0.01142      136473824    27
nprobe=8,quantizer_efSearch=32           0.4900 0.7164 0.7220      0.01342      136317939    23
nprobe=8,quantizer_efSearch=64           0.4907 0.7173 0.7229      0.01773      136294686    17
nprobe=8,quantizer_efSearch=128          0.4909 0.7175 0.7231      0.02445      136286205    13
nprobe=16,quantizer_efSearch=128         0.5443 0.8205 0.8287      0.03384      270907736    9
nprobe=32,quantizer_efSearch=128         0.5746 0.8923 0.9040      0.05180      536918951    6
nprobe=64,quantizer_efSearch=16          0.5805 0.9181 0.9313      0.08340     1064273324    4
nprobe=64,quantizer_efSearch=128         0.5907 0.9406 0.9544      0.09128     1061305942    4
nprobe=64,quantizer_efSearch=256         0.5908 0.9406 0.9544      0.10073     1061233979    3
nprobe=64,quantizer_efSearch=512         0.5910 0.9408 0.9546      0.12962     1061202940    3
nprobe=128,quantizer_efSearch=128        0.6016 0.9664 0.9809      0.15506     2089652875    2
nprobe=128,quantizer_efSearch=512        0.6018 0.9665 0.9810      0.19004     2089285338    2
nprobe=256,quantizer_efSearch=256        0.6033 0.9768 0.9937      0.28334     4103755098    2
nprobe=256,quantizer_efSearch=512        0.6034 0.9770 0.9939      0.30802     4103311577    1
nprobe=512,quantizer_efSearch=256        0.6040 0.9822 0.9983      0.48833     8050929211    1
nprobe=512,quantizer_efSearch=512        0.6046 0.9827 0.9986      0.50989     8048434784    1
nprobe=1024,quantizer_efSearch=256       0.6056 0.9814 0.9993      0.90315    15779234459    1
```

</details>
<details><summary>`PCAR32,IVF1048576_HNSW32,SQ8` </summary>
Index size 4428051797

 code_size 32

 log filename: autotune.dbbigann100M.PCAR32_IVF1048576_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2649 0.7004 0.9200      0.09327      272916195    4
nprobe=1,quantizer_efSearch=4            0.0937 0.1415 0.1425      0.00242        1143035    125
nprobe=2,quantizer_efSearch=4            0.1254 0.2120 0.2169      0.00251        2285354    120
nprobe=4,quantizer_efSearch=4            0.1507 0.2855 0.2972      0.00263        4563887    114
nprobe=4,quantizer_efSearch=8            0.1676 0.3199 0.3332      0.00320        4565677    94
nprobe=8,quantizer_efSearch=4            0.1858 0.3964 0.4286      0.00330        9101963    92
nprobe=8,quantizer_efSearch=8            0.1908 0.4074 0.4417      0.00377        9098599    80
nprobe=8,quantizer_efSearch=16           0.1970 0.4234 0.4592      0.00468        9070570    65
nprobe=16,quantizer_efSearch=4           0.2053 0.4679 0.5227      0.00466       18121058    65
nprobe=16,quantizer_efSearch=8           0.2177 0.4996 0.5636      0.00532       18109402    57
nprobe=16,quantizer_efSearch=16          0.2213 0.5094 0.5767      0.00655       18065970    46
nprobe=32,quantizer_efSearch=8           0.2288 0.5506 0.6481      0.00721       35896095    42
nprobe=32,quantizer_efSearch=16          0.2370 0.5764 0.6837      0.00797       35821817    38
nprobe=32,quantizer_efSearch=32          0.2399 0.5856 0.6958      0.00961       35739627    32
nprobe=64,quantizer_efSearch=16          0.2463 0.6207 0.7616      0.01239       70942985    25
nprobe=64,quantizer_efSearch=32          0.2510 0.6362 0.7861      0.01408       70727351    22
nprobe=64,quantizer_efSearch=64          0.2526 0.6421 0.7950      0.01767       70553936    17
nprobe=64,quantizer_efSearch=128         0.2535 0.6445 0.7980      0.02531       70470526    12
nprobe=128,quantizer_efSearch=64         0.2601 0.6760 0.8682      0.02528      139229561    12
nprobe=128,quantizer_efSearch=128        0.2613 0.6798 0.8728      0.03303      138951593    10
nprobe=256,quantizer_efSearch=64         0.2626 0.6927 0.9069      0.03795      274401806    8
nprobe=256,quantizer_efSearch=128        0.2644 0.6984 0.9168      0.04701      273488157    7
nprobe=256,quantizer_efSearch=256        0.2649 0.7002 0.9196      0.06133      273056523    5
nprobe=512,quantizer_efSearch=128        0.2658 0.7080 0.9436      0.08523      537810734    4
nprobe=512,quantizer_efSearch=256        0.2665 0.7102 0.9493      0.11221      536125106    3
nprobe=1024,quantizer_efSearch=256       0.2668 0.7139 0.9620      0.19393     1052235347    2
```

</details>
<details><summary>`PCAR32,IVF1048576(IVF1024,PQ16x4fs,RFlat),SQ8` </summary>
Index size 4159747225

 code_size 32

 log filename: autotune.dbbigann100M.PCAR32_IVF1048576_IVF1024_PQ16x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.1060 0.1804 0.1837      0.00237        3041426    127
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.0912 0.1372 0.1381      0.00231        2591956    130
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1042 0.1699 0.1732      0.00240        2681602    126
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1060 0.1804 0.1837      0.00239        3040878    126
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1193 0.2002 0.2048      0.00246        3029881    123
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1406 0.2572 0.2679      0.00261        5342260    116
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1466 0.2737 0.2846      0.00261        5325263    116
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1506 0.2817 0.2932      0.00276        5317570    109
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1652 0.3105 0.3233      0.00292        6020884    103
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1657 0.3318 0.3540      0.00320        9928342    94
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1710 0.3352 0.3562      0.00312       10641473    97
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1873 0.3890 0.4198      0.00335       11982517    90
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1889 0.3880 0.4180      0.00353       10604033    86
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1947 0.4158 0.4503      0.00415       11941798    73
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1955 0.4182 0.4611      0.00465       19761988    65
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2053 0.4494 0.5033      0.00486       19719350    62
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2097 0.4596 0.5180      0.00542       19674885    56
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2168 0.4934 0.5557      0.00550       23736122    55
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2178 0.4960 0.5601      0.00563       21005464    54
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2240 0.5089 0.5764      0.00608       23677982    50
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.2251 0.5135 0.5824      0.00704       23640011    43
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.2256 0.5177 0.5868      0.00790       28875393    38
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2373 0.5713 0.6720      0.00870       41577766    35
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2382 0.5743 0.6763      0.00950       46756314    32
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2407 0.5823 0.6893      0.01231       41475393    25
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2415 0.5862 0.6948      0.01123       46685522    27
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2418 0.5872 0.6964      0.01197       57043151    26
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2452 0.6029 0.7309      0.01367       77125636    22
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2458 0.6075 0.7382      0.01441       82234018    21
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2525 0.6341 0.7820      0.01547       81875369    20
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2546 0.6435 0.7959      0.02125       91957657    15
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2612 0.6746 0.8631      0.02765      150838985    11
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2617 0.6805 0.8718      0.03554      181012220    9
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.2619 0.6784 0.8697      0.03691      150225906    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.2624 0.6896 0.8968      0.04739      297595944    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.2629 0.6892 0.8965      0.04818      317861125    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2630 0.6956 0.9086      0.04875      285988369    7
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2635 0.6990 0.9133      0.05224      316143234    6
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2638 0.6979 0.9120      0.05500      285296299    6
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.2641 0.7012 0.9174      0.06283      355920858    5
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2646 0.7021 0.9192      0.08520      315212079    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2655 0.7107 0.9467      0.09588      579476566    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512  0.2659 0.7111 0.9494      0.12301      700021533    3
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2662 0.7150 0.9625      0.17555     1093640155    2
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQ8` </summary>
Index size 4107078229

 code_size 32

 log filename: autotune.dbbigann100M.PCAR32_IVF262144_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2641 0.7110 0.9526      0.15939     1048170673    2
nprobe=1,quantizer_efSearch=4            0.1153 0.1954 0.2006      0.00231        4336032    130
nprobe=2,quantizer_efSearch=4            0.1479 0.2773 0.2910      0.00231        8673410    131
nprobe=2,quantizer_efSearch=8            0.1607 0.3037 0.3197      0.00263        8653570    114
nprobe=2,quantizer_efSearch=16           0.1626 0.3078 0.3241      0.00303        8637757    99
nprobe=4,quantizer_efSearch=4            0.1743 0.3587 0.3907      0.00329       17346054    92
nprobe=4,quantizer_efSearch=8            0.1928 0.4029 0.4389      0.00355       17327439    85
nprobe=4,quantizer_efSearch=16           0.1959 0.4095 0.4463      0.00399       17295895    76
nprobe=8,quantizer_efSearch=4            0.2120 0.4801 0.5452      0.00535       34610545    57
nprobe=8,quantizer_efSearch=8            0.2167 0.4949 0.5613      0.00550       34575923    55
nprobe=8,quantizer_efSearch=32           0.2216 0.5122 0.5813      0.00672       34494087    45
nprobe=8,quantizer_efSearch=64           0.2228 0.5136 0.5832      0.00828       34483585    37
nprobe=16,quantizer_efSearch=4           0.2262 0.5408 0.6384      0.00918       68997574    33
nprobe=16,quantizer_efSearch=8           0.2339 0.5701 0.6762      0.00933       68932443    33
nprobe=16,quantizer_efSearch=16          0.2372 0.5825 0.6923      0.00971       68855501    31
nprobe=16,quantizer_efSearch=32          0.2388 0.5876 0.7004      0.01068       68768259    29
nprobe=16,quantizer_efSearch=128         0.2399 0.5898 0.7023      0.01529       68730954    20
nprobe=32,quantizer_efSearch=8           0.2421 0.6167 0.7586      0.01694      136946468    18
nprobe=32,quantizer_efSearch=16          0.2478 0.6369 0.7885      0.01685      136734945    18
nprobe=32,quantizer_efSearch=32          0.2510 0.6446 0.8006      0.01770      136528588    17
nprobe=32,quantizer_efSearch=128         0.2523 0.6486 0.8065      0.02247      136393110    14
nprobe=64,quantizer_efSearch=32          0.2570 0.6761 0.8699      0.03185      270342491    10
nprobe=64,quantizer_efSearch=64          0.2584 0.6807 0.8765      0.03300      270058609    10
nprobe=64,quantizer_efSearch=128         0.2588 0.6820 0.8780      0.03614      269930685    9
nprobe=128,quantizer_efSearch=32         0.2600 0.6927 0.9123      0.06057      534279231    5
nprobe=128,quantizer_efSearch=64         0.2614 0.6991 0.9235      0.06009      533167059    5
nprobe=128,quantizer_efSearch=128        0.2619 0.6999 0.9267      0.06280      532657443    5
nprobe=256,quantizer_efSearch=64         0.2630 0.7068 0.9451      0.10889     1051104571    3
nprobe=256,quantizer_efSearch=128        0.2640 0.7102 0.9514      0.11235     1049000001    3
nprobe=256,quantizer_efSearch=256        0.2641 0.7112 0.9527      0.11897     1048314856    3
nprobe=512,quantizer_efSearch=128        0.2647 0.7131 0.9628      0.21404     2063233786    2
nprobe=512,quantizer_efSearch=256        0.2650 0.7143 0.9654      0.24063     2059895443    2
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQ8` </summary>
Index size 4040067993

 code_size 32

 log filename: autotune.dbbigann100M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.2233 0.5386 0.6305      0.00925       71213533    33
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.0768 0.1250 0.1288      0.00251        5093594    120
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.1238 0.2101 0.2159      0.00260        5731373    116
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.1639 0.3089 0.3233      0.00275       10080964    110
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1702 0.3452 0.3746      0.00362       17896426    83
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.1881 0.3874 0.4217      0.00371       18139904    82
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.1884 0.3920 0.4248      0.00383       20171434    79
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.1965 0.4095 0.4444      0.00404       20118538    75
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32   0.1970 0.4130 0.4498      0.00520       22657685    58
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.1972 0.4121 0.4489      0.00572       27841473    53
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2061 0.4544 0.5098      0.00554       36420633    55
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4    0.2109 0.4680 0.5300      0.00589       35519937    51
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2211 0.5052 0.5749      0.00609       37419167    50
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.2231 0.5108 0.5814      0.00712       39937568    43
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2233 0.5386 0.6305      0.00969       71213434    31
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.2241 0.5290 0.6202      0.00957       70202835    32
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.2268 0.5529 0.6484      0.00961       72379203    32
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2345 0.5743 0.6811      0.00979       72045783    31
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2376 0.5833 0.6943      0.01032       71849484    30
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2390 0.5873 0.6985      0.01262       89737795    24
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.2393 0.5895 0.7014      0.01212       79443430    25
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.2442 0.6180 0.7591      0.01682      139291929    18
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2446 0.6191 0.7595      0.01748      143342515    18
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2518 0.6435 0.7966      0.01841      147723360    17
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.2524 0.6491 0.8061      0.01943      147353952    16
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.2526 0.6486 0.8058      0.02041      147226233    15
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2557 0.6688 0.8486      0.03160      278350490    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2572 0.6800 0.8721      0.03223      276825921    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2582 0.6815 0.8743      0.03314      281790539    10
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.2599 0.6964 0.9135      0.06012      548260547    5
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=64 0.2601 0.7013 0.9277      0.07562      544163128    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2607 0.7038 0.9422      0.11261     1064820087    3
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.2608 0.6983 0.9202      0.11391     1100330239    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2615 0.7070 0.9470      0.11004     1058980250    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2622 0.7106 0.9529      0.11148     1062451239    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.2626 0.7110 0.9537      0.11716     1061067488    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2628 0.7130 0.9631      0.20779     2094433790    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2638 0.7143 0.9658      0.21122     2075383306    2
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQ8` </summary>
Index size 4026834005

 code_size 32

 log filename: autotune.dbbigann100M.PCAR32_IVF65536_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1413 0.2691 0.2861      0.00517       17188886    59
nprobe=1,quantizer_efSearch=8            0.1400 0.2667 0.2836      0.00332       17210907    91
nprobe=1,quantizer_efSearch=16           0.1403 0.2678 0.2848      0.00357       17198757    84
nprobe=1,quantizer_efSearch=32           0.1412 0.2688 0.2858      0.00408       17189080    74
nprobe=1,quantizer_efSearch=64           0.1413 0.2691 0.2861      0.00509       17188886    59
nprobe=2,quantizer_efSearch=4            0.1699 0.3539 0.3888      0.00522       34562250    58
nprobe=2,quantizer_efSearch=8            0.1780 0.3745 0.4117      0.00536       34456633    56
nprobe=2,quantizer_efSearch=16           0.1787 0.3777 0.4148      0.00565       34404205    54
nprobe=2,quantizer_efSearch=32           0.1793 0.3788 0.4160      0.00612       34382769    50
nprobe=2,quantizer_efSearch=64           0.1794 0.3793 0.4163      0.00713       34381537    43
nprobe=4,quantizer_efSearch=4            0.1989 0.4451 0.5090      0.00917       69044012    33
nprobe=4,quantizer_efSearch=8            0.2091 0.4746 0.5455      0.00933       68971079    33
nprobe=4,quantizer_efSearch=16           0.2104 0.4799 0.5520      0.00972       68854339    31
nprobe=4,quantizer_efSearch=32           0.2113 0.4826 0.5545      0.01010       68809675    30
nprobe=4,quantizer_efSearch=64           0.2114 0.4830 0.5550      0.01125       68799763    27
nprobe=4,quantizer_efSearch=128          0.2115 0.4831 0.5551      0.01333       68794649    23
nprobe=8,quantizer_efSearch=4            0.2270 0.5520 0.6580      0.01697      137359031    18
nprobe=8,quantizer_efSearch=8            0.2320 0.5630 0.6728      0.01703      137214758    18
nprobe=8,quantizer_efSearch=16           0.2348 0.5722 0.6849      0.01735      137040842    18
nprobe=8,quantizer_efSearch=32           0.2356 0.5753 0.6884      0.01761      136934053    18
nprobe=8,quantizer_efSearch=128          0.2357 0.5763 0.6892      0.02095      136899282    15
nprobe=16,quantizer_efSearch=8           0.2461 0.6254 0.7795      0.03220      272072543    10
nprobe=16,quantizer_efSearch=32          0.2515 0.6428 0.8007      0.03271      271580640    10
nprobe=32,quantizer_efSearch=8           0.2519 0.6555 0.8431      0.05951      538452980    6
nprobe=32,quantizer_efSearch=16          0.2570 0.6725 0.8668      0.05932      537741969    6
nprobe=32,quantizer_efSearch=32          0.2583 0.6782 0.8749      0.05979      537085730    6
nprobe=32,quantizer_efSearch=64          0.2587 0.6801 0.8774      0.06086      536853265    5
nprobe=32,quantizer_efSearch=128         0.2588 0.6804 0.8777      0.06297      536775390    5
nprobe=32,quantizer_efSearch=256         0.2589 0.6805 0.8778      0.06960      536758138    5
nprobe=64,quantizer_efSearch=16          0.2597 0.6880 0.9082      0.10871     1061471559    3
nprobe=64,quantizer_efSearch=32          0.2621 0.6964 0.9216      0.11393     1059845812    3
nprobe=64,quantizer_efSearch=64          0.2623 0.6987 0.9259      0.11063     1059035472    3
nprobe=64,quantizer_efSearch=128         0.2625 0.6995 0.9273      0.11751     1058697405    3
nprobe=64,quantizer_efSearch=256         0.2626 0.6996 0.9275      0.11964     1058651879    3
nprobe=128,quantizer_efSearch=32         0.2642 0.7047 0.9446      0.20615     2086758933    2
nprobe=128,quantizer_efSearch=128        0.2646 0.7082 0.9527      0.20850     2081951644    2
nprobe=128,quantizer_efSearch=256        0.2647 0.7083 0.9530      0.21240     2081665366    2
nprobe=256,quantizer_efSearch=64         0.2650 0.7118 0.9645      0.39664     4088892181    1
nprobe=256,quantizer_efSearch=128        0.2653 0.7127 0.9669      0.39592     4082607655    1
nprobe=256,quantizer_efSearch=256        0.2654 0.7130 0.9674      0.40327     4080896170    1
nprobe=512,quantizer_efSearch=256        0.2655 0.7136 0.9703      0.76836     7990939016    1
```

</details>
<details><summary>`PCAR64,IVF1048576_HNSW32,SQ4` </summary>
Index size 4562286293

 code_size 32

 log filename: autotune.dbbigann100M.PCAR64_IVF1048576_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5101 0.9200 0.9483      0.13263      277576090    3
nprobe=1,quantizer_efSearch=4            0.1249 0.1548 0.1549      0.00279        1156280    108
nprobe=4,quantizer_efSearch=4            0.2281 0.3145 0.3165      0.00306        4628873    98
nprobe=8,quantizer_efSearch=4            0.3059 0.4503 0.4545      0.00444        9243990    68
nprobe=8,quantizer_efSearch=8            0.3150 0.4654 0.4698      0.00478        9233588    63
nprobe=8,quantizer_efSearch=16           0.3279 0.4881 0.4929      0.00630        9190876    48
nprobe=16,quantizer_efSearch=4           0.3503 0.5379 0.5448      0.00643       18394125    47
nprobe=16,quantizer_efSearch=8           0.3730 0.5805 0.5883      0.00691       18371137    44
nprobe=16,quantizer_efSearch=16          0.3844 0.5989 0.6069      0.00803       18330853    38
nprobe=32,quantizer_efSearch=8           0.4056 0.6586 0.6707      0.01037       36503933    29
nprobe=32,quantizer_efSearch=32          0.4305 0.7109 0.7239      0.01454       36276861    21
nprobe=32,quantizer_efSearch=64          0.4344 0.7193 0.7323      0.01894       36188814    16
nprobe=64,quantizer_efSearch=16          0.4489 0.7614 0.7791      0.01946       72167021    16
nprobe=64,quantizer_efSearch=32          0.4616 0.7913 0.8093      0.02081       71881533    15
nprobe=64,quantizer_efSearch=64          0.4683 0.8038 0.8220      0.02457       71659428    13
nprobe=64,quantizer_efSearch=128         0.4701 0.8073 0.8257      0.03424       71539207    9
nprobe=128,quantizer_efSearch=32         0.4811 0.8438 0.8659      0.03883      142201262    8
nprobe=128,quantizer_efSearch=64         0.4933 0.8682 0.8912      0.04338      141598351    7
nprobe=128,quantizer_efSearch=128        0.4958 0.8751 0.8983      0.04630      141224577    7
nprobe=256,quantizer_efSearch=64         0.5037 0.9021 0.9297      0.06586      279481148    5
nprobe=256,quantizer_efSearch=128        0.5078 0.9153 0.9437      0.07849      278360795    4
nprobe=256,quantizer_efSearch=256        0.5099 0.9193 0.9478      0.09536      277773451    4
nprobe=256,quantizer_efSearch=512        0.5101 0.9200 0.9483      0.13387      277576090    3
nprobe=512,quantizer_efSearch=128        0.5142 0.9358 0.9665      0.13524      548055846    3
nprobe=512,quantizer_efSearch=256        0.5170 0.9415 0.9727      0.15886      546053110    2
nprobe=512,quantizer_efSearch=512        0.5172 0.9423 0.9735      0.17274      545138920    2
nprobe=1024,quantizer_efSearch=256       0.5195 0.9529 0.9868      0.25122     1072655536    2
nprobe=1024,quantizer_efSearch=512       0.5199 0.9544 0.9885      0.30456     1069050808    1
nprobe=2048,quantizer_efSearch=256       0.5202 0.9558 0.9910      0.44787     2087251067    1
nprobe=2048,quantizer_efSearch=512       0.5205 0.9590 0.9944      0.53365     2094987125    1
nprobe=4096,quantizer_efSearch=512       0.5206 0.9599 0.9963      0.88013     4041544213    1
```

</details>
<details><summary>`PCAR64,IVF1048576(IVF1024,PQ32x4fs,RFlat),SQ4` </summary>
Index size 4302621977

 code_size 32

 log filename: autotune.dbbigann100M.PCAR64_IVF1048576_IVF1024_PQ32x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.2642 0.3658 0.3675      0.00387       10808158    78
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2       0.1249 0.1535 0.1540      0.00239        1891035    126
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1384 0.1751 0.1759      0.00249        2702417    121
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1408 0.1792 0.1797      0.00244        3077039    123
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1       0.1475 0.1887 0.1896      0.00249        2697001    121
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.1704 0.2203 0.2211      0.00257        3060780    117
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1       0.1771 0.2335 0.2348      0.00268        5040683    112
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2       0.1920 0.2536 0.2546      0.00264        5422047    114
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.2135 0.2873 0.2886      0.00273        5404955    111
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2254 0.3061 0.3079      0.00281        5393070    107
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2324 0.3139 0.3152      0.00285        6109896    106
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.2461 0.3356 0.3373      0.00296        6094539    102
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.2474 0.3325 0.3339      0.00315        7484416    96
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.2509 0.3427 0.3447      0.00345        6080202    87
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.2685 0.3660 0.3680      0.00381        7450176    79
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.2688 0.3858 0.3893      0.00398       10031098    76
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.2930 0.4183 0.4212      0.00399       10755393    76
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.3263 0.4742 0.4780      0.00453       12076435    67
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.3320 0.4828 0.4867      0.00543       12045387    56
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3424 0.5168 0.5224      0.00616       19965355    49
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3538 0.5345 0.5404      0.00647       19911941    47
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3547 0.5374 0.5432      0.00717       19890484    42
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3781 0.5789 0.5853      0.00716       23949626    42
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8      0.3860 0.5905 0.5974      0.00755       21217065    40
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3908 0.6013 0.6083      0.00781       23905966    39
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3950 0.6097 0.6167      0.00810       23886235    38
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3968 0.6129 0.6197      0.00952       29155443    32
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32    0.3978 0.6145 0.6216      0.01100       29142978    28
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4173 0.6621 0.6716      0.01092       39499638    28
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4367 0.7096 0.7210      0.01205       41986885    25
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4371 0.7117 0.7233      0.01346       41943419    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4419 0.7182 0.7296      0.01352       47200753    23
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4422 0.7206 0.7324      0.01446       47134909    21
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4429 0.7221 0.7339      0.01659       57491949    19
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4455 0.7263 0.7378      0.01867       78714307    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4648 0.7792 0.7938      0.01953       77864729    16
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4714 0.7900 0.8052      0.02054       82958851    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4765 0.8036 0.8194      0.02181       82738588    14
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.4791 0.8110 0.8268      0.02687       92939700    12
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.4798 0.8116 0.8274      0.03374       92778042    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5001 0.8713 0.8916      0.03972      152721686    8
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5022 0.8777 0.8983      0.04003      162816398    8
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.5042 0.8804 0.9012      0.04352      182886085    7
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.5043 0.8799 0.9008      0.05071      162436259    6
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.5045 0.8806 0.9014      0.05402      223186957    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5100 0.9024 0.9266      0.06810      291101102    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.5128 0.9130 0.9378      0.06681      321038637    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5132 0.9124 0.9372      0.06791      300849157    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5148 0.9106 0.9358      0.06900      290203845    5
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5151 0.9107 0.9361      0.07078      290025971    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.5178 0.9209 0.9463      0.07971      360521251    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.5189 0.9228 0.9483      0.08998      360334007    4
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256  0.5190 0.9229 0.9484      0.11161      360306003    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5202 0.9281 0.9556      0.11378      561493068    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.5248 0.9449 0.9743      0.13995      587932930    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.5250 0.9451 0.9745      0.14223      628286064    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=512  0.5271 0.9570 0.9886      0.24953     1235989847    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5272 0.9571 0.9887      0.23034     1154782054    2
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=512  0.5274 0.9583 0.9903      0.31400     1232389934    2
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.5275 0.9584 0.9904      0.38383     1151168270    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512  0.5280 0.9634 0.9966      0.53414     2252926682    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.5282 0.9651 0.9993      0.91034     4161036976    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ4` </summary>
Index size 4140649429

 code_size 32

 log filename: autotune.dbbigann100M.PCAR64_IVF262144_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5140 0.9438 0.9802      0.21215     1055167055    2
nprobe=2,quantizer_efSearch=4            0.2253 0.3138 0.3166      0.00293        8685970    103
nprobe=2,quantizer_efSearch=8            0.2441 0.3390 0.3419      0.00339        8651362    89
nprobe=2,quantizer_efSearch=16           0.2501 0.3468 0.3498      0.00422        8626619    72
nprobe=4,quantizer_efSearch=4            0.2828 0.4158 0.4213      0.00427       17364849    71
nprobe=4,quantizer_efSearch=8            0.3116 0.4591 0.4650      0.00468       17326363    65
nprobe=4,quantizer_efSearch=16           0.3224 0.4753 0.4815      0.00572       17286460    53
nprobe=4,quantizer_efSearch=32           0.3245 0.4775 0.4838      0.00734       17265389    41
nprobe=8,quantizer_efSearch=8            0.3717 0.5804 0.5900      0.00745       34578135    41
nprobe=8,quantizer_efSearch=16           0.3844 0.6046 0.6145      0.00823       34497206    37
nprobe=8,quantizer_efSearch=32           0.3888 0.6108 0.6207      0.00954       34445089    32
nprobe=8,quantizer_efSearch=64           0.3898 0.6125 0.6224      0.01268       34428460    24
nprobe=16,quantizer_efSearch=4           0.3990 0.6510 0.6640      0.01292       69136506    24
nprobe=16,quantizer_efSearch=8           0.4203 0.6900 0.7047      0.01306       69047592    23
nprobe=16,quantizer_efSearch=16          0.4313 0.7110 0.7263      0.01424       68941601    22
nprobe=16,quantizer_efSearch=64          0.4372 0.7221 0.7378      0.01816       68791489    17
nprobe=32,quantizer_efSearch=8           0.4449 0.7610 0.7794      0.02392      137383408    13
nprobe=32,quantizer_efSearch=128         0.4714 0.8160 0.8363      0.03430      136732349    9
nprobe=32,quantizer_efSearch=256         0.4717 0.8165 0.8368      0.04616      136727854    7
nprobe=64,quantizer_efSearch=16          0.4838 0.8525 0.8761      0.04643      272086053    7
nprobe=64,quantizer_efSearch=32          0.4912 0.8735 0.8984      0.04625      271491902    7
nprobe=64,quantizer_efSearch=128         0.4941 0.8811 0.9067      0.05385      270948622    6
nprobe=64,quantizer_efSearch=256         0.4946 0.8818 0.9074      0.06599      270924696    5
nprobe=128,quantizer_efSearch=32         0.5015 0.9061 0.9363      0.09019      537239378    4
nprobe=128,quantizer_efSearch=64         0.5053 0.9164 0.9484      0.09035      536097677    4
nprobe=128,quantizer_efSearch=128        0.5065 0.9198 0.9520      0.09474      535524185    4
nprobe=128,quantizer_efSearch=256        0.5069 0.9207 0.9529      0.10498      535369794    3
nprobe=256,quantizer_efSearch=64         0.5108 0.9358 0.9719      0.16959     1058216705    2
nprobe=256,quantizer_efSearch=128        0.5130 0.9423 0.9786      0.17424     1056102624    2
nprobe=256,quantizer_efSearch=256        0.5137 0.9435 0.9799      0.18501     1055352372    2
nprobe=256,quantizer_efSearch=512        0.5140 0.9438 0.9802      0.20925     1055167055    2
nprobe=512,quantizer_efSearch=256        0.5155 0.9534 0.9917      0.33883     2075882336    1
nprobe=1024,quantizer_efSearch=128       0.5159 0.9545 0.9934      0.61912     4058301046    1
nprobe=1024,quantizer_efSearch=256       0.5164 0.9569 0.9964      0.63580     4076677838    1
nprobe=2048,quantizer_efSearch=256       0.5167 0.9577 0.9976      1.21520     7887001494    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ4` </summary>
Index size 4075871001

 code_size 32

 log filename: autotune.dbbigann100M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4179 0.6786 0.6916      0.01234       70929568    25
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1367 0.1732 0.1739      0.00265        5065388    114
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.1800 0.2322 0.2335      0.00284        5695774    106
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2144 0.2984 0.3009      0.00310        9078740    97
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2227 0.3048 0.3075      0.00306        9433206    99
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2304 0.3157 0.3185      0.00315       10087578    96
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4     0.2392 0.3310 0.3336      0.00336        9379349    90
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.2511 0.3456 0.3482      0.00368       11359223    82
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.2530 0.3469 0.3494      0.00437       13973769    69
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2943 0.4270 0.4320      0.00459       18174522    66
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3035 0.4453 0.4509      0.00474       18109285    64
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3092 0.4510 0.4558      0.00493       20103507    61
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3226 0.4744 0.4799      0.00524       20031122    58
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3247 0.4742 0.4795      0.00565       22669862    54
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32    0.3253 0.4786 0.4844      0.00667       22635463    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3586 0.5510 0.5593      0.00738       35532774    41
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3694 0.5717 0.5799      0.00743       36215345    41
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3851 0.6021 0.6112      0.00788       37318252    39
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3880 0.6059 0.6149      0.00825       39903728    37
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=64    0.3899 0.6126 0.6222      0.01166       44986017    26
nprobe=8,quantizer_k_factor_rf=64,quantizer_nprobe=32    0.3901 0.6125 0.6221      0.01184       39837094    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4179 0.6786 0.6916      0.01694       70929568    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4351 0.7121 0.7267      0.01309       71858672    23
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4361 0.7153 0.7302      0.01396       71766797    22
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4384 0.7184 0.7332      0.01399       74358046    22
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.4393 0.7232 0.7385      0.01695       79393085    18
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4713 0.8034 0.8233      0.02357      142898668    13
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4716 0.8046 0.8245      0.02421      147899522    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4756 0.8147 0.8358      0.02455      142394989    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.4760 0.8160 0.8372      0.02631      147464627    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4933 0.8675 0.8936      0.04363      278005151    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4958 0.8761 0.9030      0.04231      277178792    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4968 0.8791 0.9065      0.04375      282021379    7
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.4991 0.8917 0.9227      0.08159      543313615    4
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4994 0.8887 0.9183      0.07885      557551000    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5087 0.9155 0.9484      0.07767      548487113    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5095 0.9189 0.9522      0.09736      546990417    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.5098 0.9197 0.9531      0.09337      546756193    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5155 0.9343 0.9707      0.14917     1067733537    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5175 0.9400 0.9769      0.15643     1100391887    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.5188 0.9434 0.9805      0.15848     1067608140    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5190 0.9434 0.9807      0.15115     1077351975    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5219 0.9540 0.9931      0.31967     2117447586    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.5223 0.9544 0.9936      0.55899     4188977330    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.5229 0.9567 0.9965      0.57241     4091535477    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5230 0.9568 0.9967      0.54802     4097716219    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5234 0.9579 0.9979      0.55434     4101263119    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.5235 0.9591 0.9996      1.12617     8017184495    1
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ4` </summary>
Index size 4035239381

 code_size 32

 log filename: autotune.dbbigann100M.PCAR64_IVF65536_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2158 0.3016 0.3044      0.00682       17089216    45
nprobe=1,quantizer_efSearch=16           0.2148 0.3004 0.3032      0.00494       17099843    62
nprobe=1,quantizer_efSearch=32           0.2155 0.3014 0.3042      0.00539       17091118    56
nprobe=2,quantizer_efSearch=4            0.2662 0.3951 0.4012      0.00726       34404446    42
nprobe=2,quantizer_efSearch=8            0.2868 0.4261 0.4324      0.00705       34302856    43
nprobe=2,quantizer_efSearch=16           0.2891 0.4311 0.4375      0.00744       34252475    41
nprobe=2,quantizer_efSearch=64           0.2893 0.4317 0.4380      0.00939       34223244    32
nprobe=4,quantizer_efSearch=4            0.3247 0.5090 0.5186      0.01229       68776953    25
nprobe=4,quantizer_efSearch=8            0.3482 0.5529 0.5634      0.01242       68646595    25
nprobe=4,quantizer_efSearch=16           0.3537 0.5642 0.5748      0.01291       68550443    24
nprobe=4,quantizer_efSearch=64           0.3539 0.5662 0.5768      0.01504       68479035    20
nprobe=4,quantizer_efSearch=128          0.3540 0.5663 0.5769      0.01761       68475258    18
nprobe=8,quantizer_efSearch=4            0.3967 0.6622 0.6780      0.02383      137118694    13
nprobe=8,quantizer_efSearch=8            0.4013 0.6753 0.6921      0.02300      137034929    14
nprobe=8,quantizer_efSearch=64           0.4140 0.7005 0.7180      0.02542      136708694    12
nprobe=8,quantizer_efSearch=128          0.4141 0.7006 0.7181      0.02795      136695371    11
nprobe=16,quantizer_efSearch=16          0.4529 0.7933 0.8163      0.04518      272002657    7
nprobe=16,quantizer_efSearch=32          0.4560 0.8005 0.8237      0.04443      271724752    7
nprobe=16,quantizer_efSearch=64          0.4572 0.8018 0.8250      0.04563      271616199    7
nprobe=32,quantizer_efSearch=16          0.4779 0.8613 0.8906      0.08469      539117825    4
nprobe=32,quantizer_efSearch=128         0.4829 0.8730 0.9032      0.08872      538085551    4
nprobe=64,quantizer_efSearch=16          0.4888 0.9002 0.9339      0.16311     1065619450    2
nprobe=64,quantizer_efSearch=32          0.4934 0.9138 0.9487      0.16290     1063893160    2
nprobe=64,quantizer_efSearch=128         0.4963 0.9189 0.9542      0.16693     1062634815    2
nprobe=128,quantizer_efSearch=64         0.5026 0.9399 0.9779      0.31874     2092596513    1
nprobe=128,quantizer_efSearch=128        0.5037 0.9423 0.9804      0.31644     2091094389    1
nprobe=128,quantizer_efSearch=256        0.5038 0.9427 0.9808      0.32394     2090834606    1
nprobe=128,quantizer_efSearch=512        0.5039 0.9427 0.9808      0.37095     2090761179    1
nprobe=256,quantizer_efSearch=128        0.5063 0.9523 0.9928      0.61291     4106462211    1
nprobe=256,quantizer_efSearch=256        0.5066 0.9528 0.9934      0.61737     4104595322    1
nprobe=256,quantizer_efSearch=512        0.5067 0.9527 0.9933      0.63317     4104165834    1
nprobe=512,quantizer_efSearch=128        0.5071 0.9554 0.9969      1.18623     8056631663    1
nprobe=512,quantizer_efSearch=256        0.5077 0.9564 0.9981      1.20053     8047145756    1
nprobe=1024,quantizer_efSearch=256       0.5080 0.9570 0.9992      2.32033    15765131336    1
nprobe=2048,quantizer_efSearch=512       0.5081 0.9572 0.9995      4.59322    30865589719    1
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF1048576_HNSW32,PQ128x4fs` </summary>
Index size 9618128844

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF1048576_HNSW32_PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1898 0.2545 0.2560      0.00973        2320007    31
nprobe=2,quantizer_efSearch=4            0.1686 0.2251 0.2265      0.00616        2321675    49
nprobe=4,quantizer_efSearch=4            0.2166 0.3078 0.3117      0.00675        4635260    45
nprobe=8,quantizer_efSearch=4            0.2943 0.4488 0.4577      0.00914        9272327    33
nprobe=8,quantizer_efSearch=8            0.3022 0.4603 0.4693      0.01048        9263469    29
nprobe=16,quantizer_efSearch=4           0.3354 0.5329 0.5460      0.01046       18451952    29
nprobe=32,quantizer_efSearch=4           0.3601 0.5871 0.6043      0.01275       36602340    24
nprobe=16,quantizer_efSearch=8           0.3606 0.5753 0.5901      0.01360       18422974    23
nprobe=32,quantizer_efSearch=8           0.3947 0.6518 0.6719      0.01581       36644068    20
nprobe=64,quantizer_efSearch=8           0.4110 0.6952 0.7175      0.01980       72516181    16
nprobe=32,quantizer_efSearch=16          0.4160 0.6904 0.7121      0.02134       36513495    15
nprobe=64,quantizer_efSearch=16          0.4391 0.7535 0.7787      0.02492       72390091    13
nprobe=64,quantizer_efSearch=32          0.4525 0.7847 0.8114      0.03515       72091220    9
nprobe=128,quantizer_efSearch=32         0.4748 0.8377 0.8696      0.04387      142661147    7
nprobe=128,quantizer_efSearch=64         0.4855 0.8604 0.8933      0.06379      142079212    5
nprobe=256,quantizer_efSearch=64         0.4952 0.8926 0.9298      0.07744      280512789    4
nprobe=512,quantizer_efSearch=64         0.5015 0.9083 0.9475      0.09354      551752671    4
nprobe=256,quantizer_efSearch=128        0.5020 0.9063 0.9446      0.10909      279282886    3
nprobe=512,quantizer_efSearch=128        0.5086 0.9262 0.9679      0.12775      550195746    3
nprobe=1024,quantizer_efSearch=128       0.5099 0.9335 0.9769      0.15998     1078222613    2
nprobe=512,quantizer_efSearch=256        0.5105 0.9313 0.9738      0.18469      547986390    2
nprobe=1024,quantizer_efSearch=256       0.5137 0.9431 0.9877      0.21088     1077028890    2
nprobe=2048,quantizer_efSearch=256       0.5139 0.9454 0.9906      0.27308     2098602707    2
nprobe=1024,quantizer_efSearch=512       0.5144 0.9446 0.9896      0.31201     1073313604    1
nprobe=2048,quantizer_efSearch=512       0.5154 0.9481 0.9942      0.36792     2104276606    1
nprobe=4096,quantizer_efSearch=512       0.5159 0.9494 0.9961      0.47373     4067851863    1
```

</details>
<details><summary>`OPQ128_256,IVF1048576_HNSW32,PQ128x4fsr` </summary>
Index size 9617905612

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF1048576_HNSW32_PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5579 0.6224 0.6224      0.05012       18300066    6
nprobe=1,quantizer_efSearch=4            0.1477 0.1550 0.1550      0.00633        1165553    48
nprobe=2,quantizer_efSearch=4            0.2156 0.2282 0.2282      0.00746        2322283    41
nprobe=4,quantizer_efSearch=4            0.2913 0.3131 0.3131      0.00978        4637164    31
nprobe=4,quantizer_efSearch=8            0.3373 0.3633 0.3633      0.01308        4631666    23
nprobe=4,quantizer_efSearch=16           0.3504 0.3785 0.3785      0.01889        4615089    16
nprobe=8,quantizer_efSearch=4            0.4207 0.4591 0.4591      0.02034        9270161    15
nprobe=8,quantizer_efSearch=8            0.4311 0.4712 0.4712      0.02135        9262935    15
nprobe=8,quantizer_efSearch=16           0.4543 0.4972 0.4972      0.02749        9223902    11
nprobe=16,quantizer_efSearch=4           0.4918 0.5470 0.5470      0.03067       18453535    10
nprobe=16,quantizer_efSearch=8           0.5302 0.5913 0.5913      0.03282       18425607    10
nprobe=16,quantizer_efSearch=16          0.5484 0.6119 0.6119      0.03906       18374472    8
nprobe=16,quantizer_efSearch=32          0.5579 0.6224 0.6224      0.05050       18300066    6
nprobe=16,quantizer_efSearch=64          0.5599 0.6246 0.6246      0.06515       18254149    5
nprobe=32,quantizer_efSearch=8           0.5956 0.6721 0.6721      0.07241       36651493    5
nprobe=32,quantizer_efSearch=16          0.6275 0.7130 0.7130      0.07721       36521218    4
nprobe=32,quantizer_efSearch=32          0.6424 0.7296 0.7296      0.08438       36396474    4
nprobe=32,quantizer_efSearch=64          0.6490 0.7368 0.7368      0.09669       36291064    4
nprobe=32,quantizer_efSearch=128         0.6513 0.7390 0.7390      0.12691       36245263    3
nprobe=64,quantizer_efSearch=32          0.7066 0.8121 0.8121      0.13927       72103895    3
nprobe=64,quantizer_efSearch=64          0.7183 0.8271 0.8271      0.15411       71861542    2
nprobe=64,quantizer_efSearch=128         0.7224 0.8319 0.8319      0.18595       71737634    2
nprobe=64,quantizer_efSearch=256         0.7237 0.8331 0.8331      0.22561       71689744    2
nprobe=128,quantizer_efSearch=32         0.7511 0.8701 0.8701      0.26412      142686957    2
nprobe=128,quantizer_efSearch=64         0.7706 0.8937 0.8937      0.27933      142097647    2
nprobe=128,quantizer_efSearch=128        0.7777 0.9023 0.9023      0.30264      141674228    1
nprobe=128,quantizer_efSearch=256        0.7786 0.9037 0.9037      0.34565      141496832    1
nprobe=256,quantizer_efSearch=64         0.7927 0.9299 0.9299      0.51330      280546557    1
nprobe=256,quantizer_efSearch=128        0.8041 0.9451 0.9451      0.52013      279327761    1
nprobe=256,quantizer_efSearch=256        0.8066 0.9479 0.9479      0.57207      278679981    1
nprobe=256,quantizer_efSearch=512        0.8069 0.9481 0.9481      0.67244      278466914    1
nprobe=512,quantizer_efSearch=128        0.8214 0.9683 0.9683      0.96654      550273427    1
nprobe=512,quantizer_efSearch=256        0.8245 0.9741 0.9741      1.01400      548056725    1
nprobe=512,quantizer_efSearch=512        0.8264 0.9755 0.9755      1.09603      547086456    1
nprobe=1024,quantizer_efSearch=256       0.8324 0.9878 0.9879      1.83975      540659326    1
nprobe=1024,quantizer_efSearch=512       0.8339 0.9899 0.9900      1.89165      538839297    1
```

</details>
<details><summary>`OPQ128_256,IVF1048576(IVF1024,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 9411713552

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF1048576_IVF1024_PQ128x4fs_RFlat__PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4998 0.9023 0.9399      0.07213      323668338    5
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1047 0.1319 0.1324      0.00315        1533743    96
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1353 0.1778 0.1786      0.00334        2715961    90
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1414 0.1883 0.1895      0.00350        2700767    86
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1592 0.2122 0.2131      0.00364        3077376    83
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1676 0.2245 0.2260      0.00382        3054338    79
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1773 0.2457 0.2490      0.00403        5031155    75
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2047 0.2857 0.2890      0.00421        5422838    72
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2144 0.3031 0.3070      0.00435        5393199    69
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2160 0.3060 0.3101      0.00469        5387568    64
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2402 0.3410 0.3449      0.00502        6096147    60
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2462 0.3622 0.3703      0.00542       10128368    56
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2819 0.4172 0.4255      0.00602       10810664    50
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2928 0.4397 0.4492      0.00632       10729682    48
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2950 0.4435 0.4529      0.00784       10717489    39
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3010 0.4485 0.4569      0.00773       12146456    39
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3374 0.5271 0.5406      0.00870       19948084    35
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3378 0.5294 0.5428      0.00947       19910465    32
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3527 0.5530 0.5676      0.00980       21433518    31
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3645 0.5889 0.6064      0.01216       38688485    25
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.3659 0.5790 0.5937      0.01288       21205309    24
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.3738 0.6039 0.6222      0.01284       38202092    24
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3748 0.5986 0.6141      0.01300       23829951    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3765 0.6021 0.6174      0.01375       23767381    22
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4104 0.6702 0.6908      0.01487       39462042    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4263 0.7050 0.7262      0.01938       41863506    16
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4266 0.7051 0.7263      0.02287       41917989    14
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4525 0.7713 0.7965      0.02308       78860087    14
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4561 0.7835 0.8098      0.02529       77685364    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4627 0.8008 0.8285      0.02806       82355344    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4628 0.8005 0.8279      0.03237       82511019    10
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4738 0.8279 0.8587      0.03411      150723842    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.4882 0.8633 0.8966      0.04843      152612517    7
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.4916 0.8700 0.9041      0.05303      162339887    6
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.4918 0.8699 0.9037      0.05735      181314430    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5004 0.8987 0.9365      0.06225      290514522    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5024 0.9087 0.9470      0.07014      299980752    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5037 0.9104 0.9488      0.07665      318686037    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5039 0.9105 0.9492      0.09627      319421367    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5066 0.9193 0.9603      0.10417      561567517    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5095 0.9294 0.9715      0.10704      601596378    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.5115 0.9333 0.9764      0.13576      630315856    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5144 0.9434 0.9888      0.16168     1141490733    2
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.5146 0.9436 0.9890      0.17919     1181384655    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.5154 0.9461 0.9913      0.22191     1154485513    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5165 0.9496 0.9963      0.31270     2142576484    2
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5177 0.9513 0.9981      0.37541     4250552188    1
```

</details>
<details><summary>`OPQ128_256,IVF1048576(IVF1024,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 9411420688

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF1048576_IVF1024_PQ128x4fs_RFlat__PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4395 0.4809 0.4809      0.02228       12084088    14
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1474 0.1549 0.1549      0.00496        1890822    61
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1788 0.1887 0.1887      0.00522        2701883    58
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2107 0.2236 0.2236      0.00570        3060568    53
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2138 0.2274 0.2274      0.00615        3049423    49
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2332 0.2473 0.2473      0.00668        3775116    45
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2352 0.2499 0.2499      0.00677        3768144    45
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2362 0.2511 0.2511      0.00694        3767536    44
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2363 0.2510 0.2510      0.00723        3763872    42
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2891 0.3086 0.3086      0.00769        5394031    40
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2898 0.3114 0.3114      0.00789        5384943    39
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3236 0.3470 0.3470      0.00872        6099400    35
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.3426 0.3688 0.3688      0.01297        7464129    24
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3500 0.3748 0.3748      0.01430       10135047    21
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3527 0.3791 0.3791      0.01448       10150831    21
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.4136 0.4537 0.4537      0.01795       10719218    17
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4137 0.4537 0.4537      0.01837       10724093    17
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4395 0.4809 0.4809      0.02128       12074492    15
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4423 0.4842 0.4842      0.02219       12053589    14
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4562 0.4997 0.4997      0.02533       14752619    12
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.4596 0.5033 0.5033      0.02987       19977851    11
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4885 0.5426 0.5426      0.03251       19966666    10
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5299 0.5890 0.5890      0.03289       21286595    10
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.5341 0.5933 0.5933      0.03434       21235813    9
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.5342 0.5933 0.5933      0.03478       21200128    9
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5549 0.6182 0.6182      0.03665       23910914    9
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=64    0.5595 0.6227 0.6227      0.04831       38972777    7
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5622 0.6254 0.6254      0.04865       38837362    7
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.6117 0.6884 0.6884      0.07009       39511407    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.6138 0.6913 0.6913      0.07113       39447448    5
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6419 0.7245 0.7245      0.07325       42016701    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6441 0.7274 0.7274      0.07387       42024638    5
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.6500 0.7343 0.7343      0.07824       47014365    4
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6553 0.7394 0.7394      0.08665       57297823    4
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.6670 0.7591 0.7592      0.13151       75643519    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.7090 0.8078 0.8079      0.13381       78078207    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7228 0.8268 0.8269      0.13760       82802266    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.7262 0.8328 0.8329      0.14448       92931581    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.7263 0.8329 0.8330      0.15359       92846992    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7743 0.8991 0.8992      0.27307      163077919    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7752 0.9005 0.9006      0.27404      184145399    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7799 0.9042 0.9043      0.26767      162418463    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7801 0.9046 0.9047      0.27308      182872114    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.8011 0.9365 0.9366      0.49004      290921734    1
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.8081 0.9461 0.9462      0.49266      320682256    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.8085 0.9478 0.9479      0.50217      299869723    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8099 0.9495 0.9496      0.50299      318030250    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.8112 0.9576 0.9577      0.93762      563789133    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.8246 0.9750 0.9751      0.89838      592243861    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.8253 0.9767 0.9768      0.90591      589561451    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=512  0.8255 0.9769 0.9770      0.98350      711288676    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.8330 0.9875 0.9877      1.75781      561062415    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=512 0.8356 0.9902 0.9903      1.77066      704975408    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.8360 0.9903 0.9904      1.73397      581386275    1
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fs` </summary>
Index size 7803870412

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF262144_HNSW32_PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3014 0.4590 0.4698      0.00831       17267350    37
nprobe=2,quantizer_efSearch=4            0.2156 0.3085 0.3137      0.00489        8660771    62
nprobe=4,quantizer_efSearch=4            0.2722 0.4102 0.4203      0.00567       17335466    53
nprobe=4,quantizer_efSearch=8            0.3014 0.4590 0.4698      0.00824       17267350    37
nprobe=8,quantizer_efSearch=4            0.3451 0.5509 0.5657      0.00869       34549278    35
nprobe=8,quantizer_efSearch=8            0.3551 0.5687 0.5839      0.00973       34520805    31
nprobe=16,quantizer_efSearch=4           0.3803 0.6345 0.6542      0.01147       68975991    27
nprobe=16,quantizer_efSearch=8           0.4061 0.6821 0.7022      0.01354       68861986    23
nprobe=16,quantizer_efSearch=16          0.4175 0.6996 0.7209      0.01667       68760310    18
nprobe=32,quantizer_efSearch=8           0.4321 0.7511 0.7784      0.01820      137056840    17
nprobe=32,quantizer_efSearch=16          0.4525 0.7912 0.8204      0.02256      136794317    14
nprobe=32,quantizer_efSearch=32          0.4579 0.8032 0.8324      0.02932      136570500    11
nprobe=64,quantizer_efSearch=16          0.4688 0.8395 0.8728      0.03128      271542813    10
nprobe=64,quantizer_efSearch=32          0.4768 0.8614 0.8961      0.03853      270934748    8
nprobe=128,quantizer_efSearch=32         0.4870 0.8918 0.9331      0.05024      536423161    6
nprobe=128,quantizer_efSearch=64         0.4909 0.9047 0.9464      0.06458      535208189    5
nprobe=256,quantizer_efSearch=32         0.4933 0.9057 0.9489      0.06860     1059004856    5
nprobe=256,quantizer_efSearch=64         0.4975 0.9246 0.9691      0.08411     1057378300    4
nprobe=512,quantizer_efSearch=64         0.5005 0.9308 0.9770      0.10935     2078351800    3
nprobe=512,quantizer_efSearch=128        0.5023 0.9393 0.9869      0.13744     2078714009    3
nprobe=512,quantizer_efSearch=256        0.5030 0.9423 0.9898      0.17188     2075233442    2
nprobe=1024,quantizer_efSearch=128       0.5039 0.9425 0.9908      0.16427     4061661051    2
nprobe=1024,quantizer_efSearch=256       0.5052 0.9457 0.9946      0.21196     4077612141    2
nprobe=1024,quantizer_efSearch=512       0.5053 0.9468 0.9955      0.27760     4073186812    2
nprobe=2048,quantizer_efSearch=512       0.5055 0.9474 0.9970      0.35833     7987750913    1
nprobe=4096,quantizer_efSearch=512       0.5057 0.9477 0.9973      0.48440    15299520717    1
```

</details>
<details><summary>`OPQ128_256,IVF262144_HNSW32,PQ128x4fsr` </summary>
Index size 7803856076

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF262144_HNSW32_PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6437 0.7324 0.7324      0.04387       68663327    7
nprobe=1,quantizer_efSearch=4            0.2046 0.2161 0.2161      0.00523        4325131    58
nprobe=1,quantizer_efSearch=16           0.2244 0.2364 0.2364      0.01195        4288884    26
nprobe=2,quantizer_efSearch=16           0.3298 0.3530 0.3530      0.01309        8596650    23
nprobe=4,quantizer_efSearch=16           0.4434 0.4869 0.4869      0.01557       17229394    20
nprobe=4,quantizer_efSearch=32           0.4469 0.4909 0.4909      0.02328       17209860    13
nprobe=8,quantizer_efSearch=16           0.5483 0.6112 0.6112      0.02395       34436725    13
nprobe=8,quantizer_efSearch=32           0.5534 0.6170 0.6170      0.03050       34387331    10
nprobe=16,quantizer_efSearch=16          0.6356 0.7222 0.7222      0.03724       68766485    9
nprobe=16,quantizer_efSearch=32          0.6437 0.7324 0.7324      0.04337       68663327    7
nprobe=16,quantizer_efSearch=64          0.6453 0.7348 0.7348      0.05407       68622366    6
nprobe=32,quantizer_efSearch=16          0.7122 0.8207 0.8207      0.07230      136786996    5
nprobe=32,quantizer_efSearch=32          0.7197 0.8326 0.8326      0.07685      136565746    4
nprobe=32,quantizer_efSearch=64          0.7236 0.8376 0.8376      0.08681      136433938    4
nprobe=32,quantizer_efSearch=128         0.7238 0.8384 0.8384      0.10596      136378577    3
nprobe=64,quantizer_efSearch=16          0.7480 0.8735 0.8735      0.12785      271525022    3
nprobe=64,quantizer_efSearch=128         0.7736 0.9070 0.9070      0.15767      270405241    2
nprobe=64,quantizer_efSearch=256         0.7743 0.9078 0.9078      0.18857      270348684    2
nprobe=64,quantizer_efSearch=512         0.7745 0.9080 0.9080      0.24611      270347505    2
nprobe=128,quantizer_efSearch=128        0.8005 0.9511 0.9511      0.26773      534622478    2
nprobe=128,quantizer_efSearch=256        0.8019 0.9524 0.9524      0.28198      534408920    2
nprobe=256,quantizer_efSearch=128        0.8148 0.9778 0.9779      0.46586     1055171071    1
nprobe=256,quantizer_efSearch=256        0.8162 0.9792 0.9793      0.48095     1054278681    1
nprobe=256,quantizer_efSearch=512        0.8164 0.9797 0.9798      0.54051     1054092513    1
nprobe=512,quantizer_efSearch=256        0.8213 0.9909 0.9910      0.88650     2075027578    1
nprobe=512,quantizer_efSearch=512        0.8216 0.9915 0.9916      0.95121     2073894455    1
nprobe=1024,quantizer_efSearch=512       0.8226 0.9963 0.9965      1.64439     2045160157    1
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fs` </summary>
Index size 7752612368

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5033 0.9389 0.9860      0.14407     4196749397    3
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1333 0.1769 0.1792      0.00248        4526200    121
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1750 0.2475 0.2518      0.00293        8890821    103
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2238 0.3130 0.3178      0.00312        9412549    97
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2307 0.3267 0.3318      0.00315        9368552    96
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.2345 0.3281 0.3329      0.00427       10075276    71
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2618 0.3950 0.4039      0.00435       17806431    69
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2637 0.3981 0.4075      0.00404       17784524    75
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2882 0.4282 0.4367      0.00393       18213613    77
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2948 0.4448 0.4543      0.00409       18094799    74
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2975 0.4485 0.4586      0.00425       18070044    71
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3019 0.4505 0.4597      0.00440       18833454    69
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3376 0.5281 0.5415      0.00556       35779999    54
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.3598 0.5679 0.5815      0.00595       36317937    51
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.3668 0.5806 0.5943      0.00679       37550365    45
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3780 0.6017 0.6170      0.00735       37199951    41
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3795 0.6037 0.6191      0.00908       39771157    34
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4134 0.6870 0.7077      0.00924       70541507    33
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4249 0.7092 0.7303      0.01085       71576998    28
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4264 0.7119 0.7335      0.01174       74149782    26
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4267 0.7142 0.7358      0.01252       74071998    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4271 0.7146 0.7362      0.01629       79076117    19
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4383 0.7538 0.7798      0.01374      140922695    22
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4616 0.8077 0.8375      0.01731      142133591    18
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.4619 0.8094 0.8391      0.02145      147187255    15
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.4794 0.8609 0.8953      0.02503      281092646    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.4827 0.8710 0.9056      0.02691      276545693    12
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.4863 0.8877 0.9260      0.03681      551263402    9
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.4946 0.9090 0.9501      0.03975      556062243    8
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.4971 0.9256 0.9689      0.05521     1085697962    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5005 0.9295 0.9739      0.06616     1064205823    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5025 0.9357 0.9810      0.07692     1075942606    4
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5039 0.9433 0.9904      0.09683     2133816102    4
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.5045 0.9443 0.9918      0.09523     2141597968    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5046 0.9453 0.9930      0.12034     2096745344    3
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5067 0.9484 0.9973      0.17164     4187187343    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5069 0.9490 0.9978      0.18784     4095450246    2
```

</details>
<details><summary>`OPQ128_256,IVF262144(IVF512,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 7752415760

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF262144_IVF512_PQ128x4fs_RFlat__PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.3271 0.3495 0.3495      0.00394       10015057    77
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2     0.1982 0.2089 0.2089      0.00250        4693539    121
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=2    0.1998 0.2107 0.2107      0.00282        4683609    107
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.2852 0.3047 0.3047      0.00332        9040870    91
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.3121 0.3332 0.3332      0.00352        9344806    86
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.3271 0.3495 0.3495      0.00393       10012770    77
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4194 0.4583 0.4583      0.00504       18065584    60
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.4198 0.4587 0.4587      0.00502       18063827    60
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4383 0.4786 0.4786      0.00531       18715337    57
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4410 0.4819 0.4819      0.00548       18678009    55
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4489 0.4910 0.4910      0.00662       19962055    46
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4496 0.4918 0.4918      0.00818       22497996    37
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5386 0.5998 0.5998      0.01423       36070920    22
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5540 0.6168 0.6168      0.01495       37186646    21
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.5560 0.6188 0.6188      0.01742       39692459    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.5739 0.6493 0.6493      0.02802       70086528    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.6245 0.7086 0.7086      0.02655       70451934    12
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.6382 0.7257 0.7257      0.02892       71780351    11
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.6416 0.7293 0.7293      0.02700       74233391    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.6423 0.7300 0.7300      0.02716       71528751    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.6464 0.7354 0.7354      0.02844       73925535    11
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.6467 0.7358 0.7358      0.03120       78892612    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.6911 0.7927 0.7927      0.06244      138892775    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.7207 0.8297 0.8297      0.06174      139692567    5
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7221 0.8330 0.8330      0.06451      147280346    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.7256 0.8381 0.8381      0.06350      141991087    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7266 0.8392 0.8392      0.06459      146673873    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7780 0.9083 0.9083      0.11968      280697446    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.7784 0.9089 0.9089      0.13134      291429434    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.8022 0.9484 0.9484      0.20145      541373435    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.8070 0.9537 0.9537      0.22150      575451529    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.8212 0.9806 0.9807      0.39132     1066462733    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.8226 0.9918 0.9918      0.70648     2098848751    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.8244 0.9922 0.9922      0.74068     2087900845    1
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fs` </summary>
Index size 7351070412

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF65536_HNSW32_PQ128x4fs.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3433 0.5500 0.5651      0.01142       68399824    27
nprobe=1,quantizer_efSearch=8            0.2083 0.2937 0.2987      0.00517       17106864    59
nprobe=1,quantizer_efSearch=16           0.2125 0.2987 0.3037      0.00631       17085598    48
nprobe=2,quantizer_efSearch=4            0.2571 0.3856 0.3952      0.00651       34312822    47
nprobe=2,quantizer_efSearch=8            0.2787 0.4204 0.4306      0.00718       34252816    42
nprobe=2,quantizer_efSearch=16           0.2853 0.4288 0.4391      0.00836       34211706    36
nprobe=2,quantizer_efSearch=32           0.2857 0.4296 0.4399      0.01057       34182210    29
nprobe=4,quantizer_efSearch=4            0.3136 0.4967 0.5110      0.01062       68517397    29
nprobe=4,quantizer_efSearch=8            0.3433 0.5500 0.5651      0.01121       68399824    27
nprobe=4,quantizer_efSearch=16           0.3500 0.5612 0.5764      0.01263       68301747    24
nprobe=4,quantizer_efSearch=32           0.3521 0.5639 0.5791      0.01456       68255805    21
nprobe=8,quantizer_efSearch=8            0.3951 0.6737 0.6947      0.01704      136651468    18
nprobe=8,quantizer_efSearch=16           0.4076 0.6957 0.7170      0.01924      136459113    16
nprobe=8,quantizer_efSearch=32           0.4096 0.7002 0.7216      0.02227      136322238    14
nprobe=8,quantizer_efSearch=64           0.4106 0.7013 0.7227      0.02611      136293357    12
nprobe=16,quantizer_efSearch=4           0.4183 0.7327 0.7595      0.02603      271928678    12
nprobe=16,quantizer_efSearch=8           0.4375 0.7726 0.8009      0.02527      271736362    12
nprobe=16,quantizer_efSearch=16          0.4454 0.7887 0.8175      0.02895      271363771    11
nprobe=16,quantizer_efSearch=32          0.4496 0.7978 0.8269      0.02995      271038192    11
nprobe=16,quantizer_efSearch=64          0.4509 0.7987 0.8277      0.03669      270942735    9
nprobe=32,quantizer_efSearch=8           0.4598 0.8261 0.8587      0.03861      538997982    8
nprobe=64,quantizer_efSearch=8           0.4691 0.8529 0.8895      0.05391     1065121023    6
nprobe=32,quantizer_efSearch=16          0.4725 0.8546 0.8886      0.04106      537887328    8
nprobe=64,quantizer_efSearch=16          0.4835 0.8917 0.9307      0.05794     1064204824    6
nprobe=64,quantizer_efSearch=32          0.4893 0.9066 0.9479      0.05487     1062524114    6
nprobe=64,quantizer_efSearch=64          0.4920 0.9115 0.9530      0.07167     1061632242    5
nprobe=128,quantizer_efSearch=32         0.4963 0.9267 0.9713      0.07986     2094502045    4
nprobe=128,quantizer_efSearch=64         0.4996 0.9338 0.9789      0.09579     2090991413    4
nprobe=128,quantizer_efSearch=128        0.5003 0.9353 0.9807      0.11639     2089632296    3
nprobe=256,quantizer_efSearch=64         0.5031 0.9433 0.9908      0.12496     4111543057    3
nprobe=256,quantizer_efSearch=128        0.5040 0.9452 0.9931      0.13836     4105592342    3
nprobe=512,quantizer_efSearch=128        0.5047 0.9488 0.9976      0.18882     8059543142    2
nprobe=1024,quantizer_efSearch=256       0.5049 0.9504 0.9992      0.30624    15779266890    1
```

</details>
<details><summary>`OPQ128_256,IVF65536_HNSW32,PQ128x4fsr` </summary>
Index size 7351031500

 code_size 64

 log filename: autotune.dbbigann100M.OPQ128_256_IVF65536_HNSW32_PQ128x4fsr.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7118 0.8273 0.8274      0.04907      271036385    7
nprobe=1,quantizer_efSearch=4            0.2585 0.2788 0.2788      0.00534       17138319    57
nprobe=1,quantizer_efSearch=16           0.2810 0.3037 0.3037      0.00966       17085581    32
nprobe=2,quantizer_efSearch=16           0.4001 0.4390 0.4390      0.01201       34214139    25
nprobe=4,quantizer_efSearch=16           0.5152 0.5770 0.5770      0.01600       68308805    19
nprobe=4,quantizer_efSearch=32           0.5169 0.5791 0.5791      0.02052       68258153    15
nprobe=8,quantizer_efSearch=16           0.6301 0.7182 0.7182      0.02650      136466226    12
nprobe=8,quantizer_efSearch=32           0.6332 0.7218 0.7218      0.03021      136322419    10
nprobe=8,quantizer_efSearch=64           0.6342 0.7228 0.7228      0.03619      136294792    9
nprobe=16,quantizer_efSearch=16          0.7060 0.8184 0.8185      0.04585      271364057    7
nprobe=16,quantizer_efSearch=32          0.7118 0.8273 0.8274      0.04916      271036385    7
nprobe=16,quantizer_efSearch=64          0.7123 0.8279 0.8280      0.05429      270939632    6
nprobe=16,quantizer_efSearch=128         0.7129 0.8285 0.8286      0.06544      270904061    5
nprobe=32,quantizer_efSearch=16          0.7577 0.8894 0.8894      0.08717      537884297    4
nprobe=32,quantizer_efSearch=32          0.7664 0.9009 0.9009      0.08973      537262160    4
nprobe=32,quantizer_efSearch=64          0.7687 0.9031 0.9031      0.09505      536984428    4
nprobe=32,quantizer_efSearch=128         0.7696 0.9039 0.9039      0.10359      536909696    3
nprobe=64,quantizer_efSearch=16          0.7841 0.9313 0.9313      0.15305     1064221885    2
nprobe=64,quantizer_efSearch=64          0.8012 0.9533 0.9533      0.15852     1061633011    2
nprobe=64,quantizer_efSearch=128         0.8020 0.9543 0.9543      0.16895     1061297510    2
nprobe=128,quantizer_efSearch=64         0.8147 0.9792 0.9792      0.27924     2090984901    2
nprobe=128,quantizer_efSearch=128        0.8155 0.9808 0.9809      0.28468     2089632171    2
nprobe=128,quantizer_efSearch=256        0.8156 0.9808 0.9809      0.31142     2089374405    1
nprobe=128,quantizer_efSearch=512        0.8158 0.9809 0.9810      0.34646     2089277346    1
nprobe=256,quantizer_efSearch=64         0.8216 0.9912 0.9912      0.50237     4111595038    1
nprobe=256,quantizer_efSearch=128        0.8224 0.9933 0.9933      0.47607     4105606489    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7901111698

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7006 0.9150 0.9159      0.07789      279422131    4
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.2556 0.2924 0.2925      0.00353        4634660    85
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.3500 0.4041 0.4041      0.00434        9247916    70
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.3616 0.4175 0.4175      0.00467        9238140    65
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.3960 0.4580 0.4580      0.00536       18392937    56
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.4294 0.4979 0.4979      0.00592       18376112    51
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.4413 0.5126 0.5126      0.00685       18324976    44
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.4541 0.5273 0.5273      0.00734       36511456    41
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.4601 0.5482 0.5483      0.00719       18376112    42
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.4725 0.5636 0.5637      0.00805       18324976    38
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.4946 0.5939 0.5940      0.00920       36511456    33
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5189 0.6252 0.6253      0.01062       36384965    29
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.5445 0.6711 0.6712      0.01279       36384965    24
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.5566 0.6876 0.6877      0.01667       36266557    18
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.5773 0.7164 0.7166      0.01632       72143645    19
k_factor_rf=4,nprobe=128,quantizer_efSearch=16    0.5873 0.7279 0.7281      0.02047      142521400    15
k_factor_rf=8,nprobe=64,quantizer_efSearch=16     0.5988 0.7537 0.7542      0.02185       72143645    14
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6162 0.7682 0.7684      0.02284      142186200    14
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.6188 0.7826 0.7831      0.02369       71866988    13
k_factor_rf=4,nprobe=128,quantizer_efSearch=64    0.6321 0.7899 0.7901      0.02860      141562055    11
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6582 0.8420 0.8424      0.03389      141562055    9
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6647 0.8501 0.8505      0.04305      141214504    7
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6731 0.8698 0.8703      0.04679      141562055    7
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.6798 0.8785 0.8790      0.05508      141214504    6
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.6812 0.8738 0.8741      0.05323      278302984    6
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6911 0.8976 0.8982      0.05811      279422131    6
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7024 0.9139 0.9145      0.06442      278302984    5
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.7120 0.9315 0.9324      0.09020      278302984    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7138 0.9344 0.9353      0.10583      277742606    3
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7257 0.9540 0.9550      0.13281      545983749    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.7273 0.9589 0.9602      0.16592      547976204    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=512  0.7322 0.9665 0.9678      0.21772      545084514    2
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7324 0.9652 0.9661      0.21669     1068909709    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7374 0.9760 0.9774      0.22655     1072499863    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7393 0.9783 0.9797      0.27433     1068909709    2
k_factor_rf=64,nprobe=4096,quantizer_efSearch=512 0.7415 0.9824 0.9836      0.42406     4041057160    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7900957842

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_HNSW32_PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7409 0.9862 0.9880      0.38296     1067840300    1
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.1406 0.1546 0.1546      0.00411        1160177    74
k_factor_rf=2,nprobe=1,quantizer_efSearch=8       0.1573 0.1720 0.1720      0.00541        1153870    56
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.2274 0.2555 0.2555      0.00554        2303836    55
k_factor_rf=2,nprobe=2,quantizer_efSearch=8       0.2275 0.2558 0.2558      0.00657        2303836    46
k_factor_rf=1,nprobe=2,quantizer_efSearch=16      0.2317 0.2604 0.2604      0.00746        2294619    41
k_factor_rf=4,nprobe=4,quantizer_efSearch=4       0.2690 0.3117 0.3118      0.00851        4631106    36
k_factor_rf=4,nprobe=8,quantizer_efSearch=8       0.3933 0.4695 0.4697      0.01124        9230653    27
k_factor_rf=2,nprobe=32,quantizer_efSearch=4      0.4876 0.6037 0.6043      0.01213       36451360    25
k_factor_rf=1,nprobe=64,quantizer_efSearch=8      0.5678 0.7130 0.7134      0.01586       72184450    19
k_factor_rf=2,nprobe=32,quantizer_efSearch=64     0.5825 0.7287 0.7296      0.02352       36155916    13
k_factor_rf=1,nprobe=64,quantizer_efSearch=64     0.6359 0.8057 0.8063      0.02705       71585653    12
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.6405 0.8178 0.8187      0.02846       71585653    11
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6421 0.8224 0.8235      0.03192       71585653    10
k_factor_rf=2,nprobe=64,quantizer_efSearch=128    0.6437 0.8218 0.8227      0.03981       71470864    8
k_factor_rf=4,nprobe=64,quantizer_efSearch=128    0.6450 0.8262 0.8273      0.04363       71470864    7
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6657 0.8645 0.8656      0.04807      142060924    7
k_factor_rf=2,nprobe=128,quantizer_efSearch=64    0.6790 0.8787 0.8796      0.04877      141454411    7
k_factor_rf=2,nprobe=128,quantizer_efSearch=128   0.6851 0.8869 0.8878      0.05837      141095238    6
k_factor_rf=4,nprobe=128,quantizer_efSearch=128   0.6876 0.8949 0.8961      0.06377      141095238    5
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6882 0.8969 0.8982      0.07067      141095238    5
k_factor_rf=4,nprobe=128,quantizer_efSearch=256   0.6897 0.8973 0.8985      0.08131      140942788    4
k_factor_rf=1,nprobe=256,quantizer_efSearch=128   0.6991 0.9002 0.9007      0.08713      278083161    4
k_factor_rf=2,nprobe=256,quantizer_efSearch=128   0.7104 0.9261 0.9270      0.09068      278083161    4
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.7147 0.9399 0.9413      0.10235      278083161    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7149 0.9406 0.9422      0.11528      278083161    3
k_factor_rf=8,nprobe=256,quantizer_efSearch=256   0.7181 0.9440 0.9454      0.12338      277523551    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7183 0.9448 0.9464      0.13626      277523551    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7184 0.9448 0.9464      0.15800      277523551    2
k_factor_rf=2,nprobe=512,quantizer_efSearch=128   0.7218 0.9437 0.9446      0.15551      547491502    2
k_factor_rf=8,nprobe=512,quantizer_efSearch=128   0.7276 0.9616 0.9630      0.16553      547491502    2
k_factor_rf=4,nprobe=512,quantizer_efSearch=256   0.7305 0.9648 0.9661      0.18031      545475052    2
k_factor_rf=8,nprobe=512,quantizer_efSearch=256   0.7326 0.9689 0.9704      0.18713      545475052    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=256  0.7327 0.9701 0.9718      0.20421      545475052    2
k_factor_rf=8,nprobe=512,quantizer_efSearch=512   0.7330 0.9701 0.9716      0.22970      544590302    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=512  0.7332 0.9713 0.9730      0.24188      544590302    2
k_factor_rf=4,nprobe=1024,quantizer_efSearch=256  0.7361 0.9760 0.9773      0.30451     1071378587    1
k_factor_rf=8,nprobe=1024,quantizer_efSearch=256  0.7393 0.9822 0.9837      0.30702     1071378587    1
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.7397 0.9836 0.9854      0.32509     1071378587    1
k_factor_rf=16,nprobe=1024,quantizer_efSearch=512 0.7409 0.9862 0.9880      0.37890     1067840300    1
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7414 0.9866 0.9884      0.40714     1067840300    1
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7415 0.9867 0.9885      0.45945     1067840300    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7641496278

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.4285 0.4952 0.4952      0.02407       73947003    13
k_factor_rf=2,nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1        0.1137 0.1251 0.1251      0.00316        1533416    95
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2        0.1558 0.1727 0.1728      0.00327        3065995    92
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2        0.2555 0.2894 0.2895      0.00384        5388544    79
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.2856 0.3232 0.3233      0.00392        7477709    77
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2982 0.3388 0.3389      0.00415        7465120    73
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.3360 0.3847 0.3848      0.00515       19264139    59
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.3381 0.3873 0.3874      0.00524       19216356    58
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3757 0.4400 0.4402      0.00588       12105825    52
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16      0.3869 0.4465 0.4466      0.00671       14749491    45
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.4293 0.5078 0.5079      0.00782       38517174    39
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.4695 0.5434 0.5435      0.00779       39372918    39
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.4822 0.5747 0.5748      0.00901       23882432    34
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.5314 0.6408 0.6409      0.01154       41927220    26
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.5364 0.6424 0.6425      0.01337       75146470    23
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64      0.5415 0.6527 0.6528      0.01546       57463015    20
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5709 0.7024 0.7025      0.01826       75117690    17
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.5733 0.6942 0.6943      0.02155      113460906    14
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5848 0.7260 0.7264      0.02082       78577064    15
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64      0.6307 0.7970 0.7975      0.02660       92984900    12
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128    0.6406 0.7993 0.7995      0.03158      183143996    10
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6565 0.8358 0.8361      0.04048      286587588    8
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6677 0.8562 0.8568      0.05018      291021803    6
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6961 0.9015 0.9021      0.05613      290750697    6
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.7007 0.9086 0.9092      0.08461      289698724    4
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7102 0.9246 0.9252      0.09511     1115947615    4
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256  0.7113 0.9264 0.9270      0.10445     1172690406    3
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.7153 0.9333 0.9338      0.11560     1152354925    3
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=512   0.7274 0.9567 0.9577      0.12459      709319398    3
k_factor_rf=32,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7318 0.9651 0.9660      0.14202     1113334112    3
k_factor_rf=32,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7329 0.9672 0.9682      0.19095     2136352835    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512   0.7338 0.9688 0.9701      0.20715      709189098    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7385 0.9771 0.9785      0.21475     1092800668    2
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.7395 0.9795 0.9807      0.30034     4215764863    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.7401 0.9800 0.9814      0.31929     1111147155    1
k_factor_rf=64,nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512 0.7427 0.9853 0.9866      2.86117     4240199164    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576(IVF1024,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7641378518

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF1048576_IVF1024_PQ32x4fs_RFlat__PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.7359 0.9721 0.9732      0.25853     1167171700    2
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1        0.0972 0.1050 0.1050      0.00355        1535291    85
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2        0.1657 0.1828 0.1828      0.00408        3062757    74
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=2        0.1911 0.2125 0.2125      0.00412        3049467    73
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2        0.2003 0.2233 0.2233      0.00420        3047476    72
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1        0.2098 0.2385 0.2385      0.00444        5017322    68
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2        0.2595 0.2971 0.2971      0.00456        5383357    66
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=8        0.2663 0.3009 0.3010      0.00657        7465526    46
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.3790 0.4535 0.4535      0.00673       19224061    45
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=2       0.3838 0.4614 0.4614      0.00707       19185019    43
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.4397 0.5251 0.5251      0.00730       21361001    42
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.4864 0.5901 0.5902      0.00805       21196380    38
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.4989 0.6036 0.6041      0.01020       23886591    30
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.5501 0.6813 0.6818      0.01112       39299140    27
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5700 0.7080 0.7087      0.01306       41952340    23
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.5734 0.7210 0.7218      0.02004       75912366    15
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.6034 0.7576 0.7584      0.02133       78327351    15
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.6358 0.8058 0.8066      0.02313       77420381    13
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64      0.6460 0.8216 0.8224      0.02534       92220657    12
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=32     0.6462 0.8262 0.8273      0.04122       82422753    8
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128     0.6485 0.8296 0.8307      0.04345      112947644    7
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6726 0.8657 0.8669      0.05262      147868888    6
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.6885 0.8941 0.8955      0.05566      152270837    6
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.6918 0.8986 0.8997      0.05608      161828951    6
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=256    0.6921 0.8989 0.9000      0.06204      221980373    5
k_factor_rf=1,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6945 0.8928 0.8933      0.07295      290004050    5
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.7117 0.9316 0.9327      0.08274      289604359    4
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256    0.7191 0.9422 0.9433      0.09575      360088024    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=512    0.7199 0.9452 0.9467      0.11982      436268234    3
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128    0.7339 0.9706 0.9721      0.16154      587157888    2
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256  0.7344 0.9729 0.9746      0.25166      627424409    2
k_factor_rf=4,nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=256   0.7359 0.9721 0.9732      0.26397     1165883036    2
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7422 0.9861 0.9877      0.27487     1110399707    2
k_factor_rf=16,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7429 0.9887 0.9906      0.31427     1149929861    1
k_factor_rf=32,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7431 0.9893 0.9912      0.33329     1110027205    1
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=64,quantizer_nprobe=512 0.7433 0.9893 0.9912      0.98010     1231145510    1
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7375336594

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7230 0.9509 0.9519      0.07980     1058236115    4
k_factor_rf=1,nprobe=4,quantizer_efSearch=4       0.3165 0.3622 0.3622      0.00318       17340778    95
k_factor_rf=1,nprobe=4,quantizer_efSearch=8       0.3515 0.4017 0.4017      0.00368       17325450    82
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.4108 0.4759 0.4759      0.00515       34583688    59
k_factor_rf=1,nprobe=16,quantizer_efSearch=8      0.4567 0.5311 0.5311      0.00658       69023722    46
k_factor_rf=1,nprobe=16,quantizer_efSearch=16     0.4696 0.5467 0.5467      0.00608       68920570    50
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.4736 0.5512 0.5512      0.00696      137377759    44
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.5056 0.6043 0.6043      0.00691       69023722    44
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.5184 0.6212 0.6212      0.00761       68920570    40
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.5275 0.6340 0.6340      0.00917      137377759    33
k_factor_rf=2,nprobe=32,quantizer_efSearch=16     0.5512 0.6633 0.6633      0.00964      137118927    32
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.5582 0.6746 0.6746      0.01103      136914124    28
k_factor_rf=2,nprobe=64,quantizer_efSearch=16     0.5636 0.6800 0.6800      0.01255      272043938    24
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.5954 0.7367 0.7367      0.01274      137118927    24
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.6028 0.7483 0.7484      0.01435      136914124    21
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.6121 0.7602 0.7603      0.01568      272043938    20
k_factor_rf=8,nprobe=32,quantizer_efSearch=16     0.6197 0.7797 0.7800      0.01882      137118927    16
k_factor_rf=4,nprobe=64,quantizer_efSearch=64     0.6293 0.7863 0.7865      0.02017      271055861    15
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.6344 0.7883 0.7885      0.02252      537219772    14
k_factor_rf=8,nprobe=64,quantizer_efSearch=16     0.6441 0.8162 0.8165      0.02410      272043938    13
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.6578 0.8374 0.8379      0.02463      271449873    13
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.6625 0.8439 0.8444      0.02671      271055861    12
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6782 0.8677 0.8682      0.03371      536014528    9
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.6808 0.8716 0.8721      0.03732      535449225    9
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.6811 0.8801 0.8808      0.04080      271055861    8
k_factor_rf=16,nprobe=64,quantizer_efSearch=128   0.6822 0.8815 0.8822      0.04421      270906065    7
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.7020 0.9119 0.9126      0.04858      536014528    7
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7046 0.9158 0.9165      0.05531      535449225    6
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.7074 0.9215 0.9222      0.05619     1058236115    6
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7124 0.9286 0.9293      0.06120     1056054792    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=128  0.7152 0.9373 0.9381      0.07331      535449225    5
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.7230 0.9509 0.9519      0.08072     1058236115    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.7279 0.9576 0.9586      0.08625     1056054792    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.7283 0.9583 0.9593      0.09738     1055292281    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=512  0.7285 0.9584 0.9594      0.12451     1055118097    3
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7319 0.9651 0.9660      0.12052     2075709233    3
k_factor_rf=64,nprobe=256,quantizer_efSearch=128  0.7339 0.9699 0.9711      0.13739     1056054792    3
k_factor_rf=64,nprobe=512,quantizer_efSearch=128  0.7383 0.9790 0.9801      0.16186     2079157895    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.7391 0.9806 0.9817      0.19753     4058041610    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=512  0.7398 0.9810 0.9821      0.20309     2074551588    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7410 0.9829 0.9840      0.21607     4076463603    2
```

</details>
<details><summary>`OPQ16_64,IVF262144_HNSW32,PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7375380370

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_HNSW32_PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7453 0.9946 0.9957      0.35983     4069547543    1
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.1918 0.2139 0.2139      0.00391        4331715    77
k_factor_rf=4,nprobe=8,quantizer_efSearch=8       0.4775 0.5848 0.5851      0.01116       34600621    27
k_factor_rf=1,nprobe=8,quantizer_efSearch=64      0.5015 0.6083 0.6084      0.01461       34443875    21
k_factor_rf=1,nprobe=16,quantizer_efSearch=64     0.5773 0.7160 0.7161      0.01631       68771398    19
k_factor_rf=2,nprobe=16,quantizer_efSearch=64     0.5829 0.7288 0.7292      0.01816       68771398    17
k_factor_rf=2,nprobe=32,quantizer_efSearch=64     0.6439 0.8219 0.8224      0.02130      136726054    15
k_factor_rf=1,nprobe=64,quantizer_efSearch=64     0.6756 0.8595 0.8596      0.02493      271082180    13
k_factor_rf=2,nprobe=64,quantizer_efSearch=64     0.6875 0.8858 0.8862      0.02696      271082180    12
k_factor_rf=4,nprobe=64,quantizer_efSearch=128    0.6922 0.8997 0.9003      0.04012      270948304    8
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.7064 0.9252 0.9258      0.05023      537091259    6
k_factor_rf=2,nprobe=128,quantizer_efSearch=128   0.7124 0.9279 0.9283      0.05502      535308688    6
k_factor_rf=4,nprobe=128,quantizer_efSearch=128   0.7172 0.9415 0.9421      0.06035      535308688    5
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.7199 0.9478 0.9486      0.06744      535308688    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7205 0.9491 0.9500      0.08156      535308688    4
k_factor_rf=2,nprobe=256,quantizer_efSearch=128   0.7253 0.9479 0.9483      0.08749     1055771777    4
k_factor_rf=2,nprobe=256,quantizer_efSearch=256   0.7269 0.9495 0.9499      0.09948     1054948542    4
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.7345 0.9741 0.9749      0.09954     1055771777    4
k_factor_rf=8,nprobe=256,quantizer_efSearch=256   0.7358 0.9757 0.9765      0.11299     1054948542    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7363 0.9773 0.9782      0.12772     1054948542    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=512  0.7364 0.9774 0.9783      0.16234     1054794892    2
k_factor_rf=4,nprobe=512,quantizer_efSearch=256   0.7385 0.9772 0.9778      0.17728     2074764964    2
k_factor_rf=8,nprobe=512,quantizer_efSearch=256   0.7425 0.9875 0.9883      0.18825     2074764964    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=256  0.7430 0.9897 0.9907      0.19688     2074764964    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=512  0.7434 0.9901 0.9912      0.22200     2073652196    2
k_factor_rf=8,nprobe=1024,quantizer_efSearch=256  0.7443 0.9918 0.9925      0.30227     4074187441    1
k_factor_rf=16,nprobe=1024,quantizer_efSearch=256 0.7450 0.9945 0.9955      0.30992     4074187441    1
k_factor_rf=16,nprobe=1024,quantizer_efSearch=512 0.7453 0.9946 0.9957      0.35718     4069547543    1
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7454 0.9955 0.9966      0.38318     4069547543    1
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7310615254

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.5110 0.5979 0.5979      0.05156      312417994    6
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.1794 0.1991 0.1991      0.00312        5041061    97
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1        0.2046 0.2288 0.2288      0.00331        8920073    91
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2       0.2468 0.2772 0.2772      0.00341        9075278    88
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.3316 0.3745 0.3745      0.00364       18175098    83
k_factor_rf=2,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8        0.3690 0.4253 0.4253      0.00502       18800365    60
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=32       0.3837 0.4396 0.4396      0.00589       40269275    51
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4        0.3861 0.4466 0.4466      0.00564       35924242    54
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32       0.4288 0.4952 0.4952      0.00588       39881106    51
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4535 0.5363 0.5364      0.00606       36065422    50
k_factor_rf=2,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16       0.4649 0.5500 0.5501      0.00664       37291922    46
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.4678 0.5431 0.5431      0.00697       74456110    44
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5055 0.6039 0.6039      0.00782       70640358    39
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16      0.5240 0.6239 0.6239      0.00954      141951012    32
k_factor_rf=4,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.5535 0.6768 0.6770      0.01142       71757964    27
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5756 0.6974 0.6974      0.01351      277669103    23
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32      0.5969 0.7398 0.7399      0.01377      142734305    22
k_factor_rf=4,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.6049 0.7508 0.7509      0.01435      142451190    21
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.6177 0.7673 0.7675      0.01670      275783043    18
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6290 0.7791 0.7793      0.02273      542821121    14
k_factor_rf=8,nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256     0.6318 0.7981 0.7985      0.02762      178124540    11
k_factor_rf=16,nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.6350 0.8089 0.8094      0.03239      140131627    10
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.6358 0.7915 0.7917      0.03589     1118265041    9
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.6397 0.8184 0.8190      0.03783      274935557    8
k_factor_rf=16,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.6689 0.8584 0.8591      0.03733      275772728    9
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.6821 0.8725 0.8730      0.04108     1066818045    8
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7010 0.9089 0.9096      0.04649      543657965    7
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.7021 0.9114 0.9121      0.05548     1086004930    6
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7136 0.9304 0.9311      0.06216     1067679605    5
k_factor_rf=32,nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7152 0.9380 0.9388      0.07620      556787332    4
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128   0.7187 0.9442 0.9450      0.08498     1098060136    4
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7268 0.9559 0.9568      0.10360     2093474403    3
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.7327 0.9666 0.9675      0.11898     2097186949    3
k_factor_rf=64,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7344 0.9706 0.9716      0.14266     1079141589    3
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.7396 0.9809 0.9819      0.20441     2089279965    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=256 0.7415 0.9844 0.9855      0.28622     4113486717    2
```

</details>
<details><summary>`OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7310543062

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF262144_IVF512_PQ32x4fs_RFlat__PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.6082 0.7596 0.7599      0.01563      158932568    20
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1        0.1309 0.1446 0.1446      0.00361        4564247    84
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4       0.2016 0.2235 0.2235      0.00411        5035637    73
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8       0.2095 0.2322 0.2322      0.00424        5702690    71
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.2962 0.3365 0.3365      0.00428       10036808    71
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2        0.3351 0.3924 0.3925      0.00467       17812963    65
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8       0.3980 0.4677 0.4678      0.00543       18708173    56
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8        0.4388 0.5253 0.5254      0.00551       36402773    55
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8        0.4867 0.5891 0.5893      0.00587       36044031    52
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16       0.4975 0.6035 0.6037      0.00691       37259566    44
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8       0.5586 0.6897 0.6899      0.00844       70611274    36
k_factor_rf=2,nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.5764 0.7198 0.7201      0.01009       71764477    30
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.6119 0.7745 0.7748      0.01239      139465591    25
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128     0.6304 0.7942 0.7944      0.01627      157608424    19
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.6464 0.8245 0.8249      0.01840      275249090    17
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64      0.6557 0.8296 0.8299      0.01840      285371339    17
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.6740 0.8570 0.8573      0.01937      276969601    16
k_factor_rf=1,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.6763 0.8610 0.8613      0.02357      291914105    13
k_factor_rf=2,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.6884 0.8887 0.8891      0.02566      291903833    12
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.6905 0.8952 0.8958      0.02517      276930018    12
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64      0.6941 0.9005 0.9011      0.02963      281525114    11
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128     0.6949 0.9044 0.9052      0.03747      291870105    9
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256     0.6950 0.9044 0.9052      0.04546      312043368    7
k_factor_rf=4,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.7138 0.9342 0.9349      0.04893      542964510    7
k_factor_rf=16,nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7160 0.9425 0.9434      0.07083      543013213    5
k_factor_rf=2,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128    0.7247 0.9469 0.9473      0.07671     1077759057    4
k_factor_rf=4,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=256    0.7281 0.9558 0.9565      0.08796     1115559155    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128    0.7303 0.9630 0.9640      0.08830     1095046815    4
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.7318 0.9679 0.9688      0.09347     1064357112    4
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.7367 0.9775 0.9784      0.10675     1077759057    3
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.7371 0.9783 0.9792      0.13140     1067153484    3
k_factor_rf=8,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256    0.7432 0.9878 0.9888      0.17523     2116365153    2
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.7435 0.9909 0.9920      0.18314     2096255082    2
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7436 0.9911 0.9921      0.21065     2088297677    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.7440 0.9920 0.9931      0.29651     2115686139    2
k_factor_rf=8,nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7448 0.9912 0.9922      0.28804     4088884917    2
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.7460 0.9960 0.9970      0.42883     4088873519    1
k_factor_rf=32,nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.7462 0.9966 0.9977      0.53516     4092045003    1
k_factor_rf=64,nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=256 0.7463 0.9968 0.9979      0.63911     4111212784    1
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 7243992210

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                 0.7277 0.9563 0.9572      0.08816     4112395032    4
k_factor_rf=1,nprobe=1,quantizer_efSearch=8      0.2419 0.2705 0.2705      0.00310       17140935    97
k_factor_rf=1,nprobe=2,quantizer_efSearch=4      0.3034 0.3409 0.3409      0.00322       34435848    94
k_factor_rf=1,nprobe=2,quantizer_efSearch=8      0.3273 0.3680 0.3680      0.00340       34324815    89
k_factor_rf=1,nprobe=4,quantizer_efSearch=4      0.3614 0.4112 0.4112      0.00399       68802327    76
k_factor_rf=1,nprobe=4,quantizer_efSearch=8      0.3915 0.4473 0.4473      0.00430       68705757    70
k_factor_rf=1,nprobe=4,quantizer_efSearch=16     0.3990 0.4560 0.4560      0.00455       68545209    66
k_factor_rf=1,nprobe=8,quantizer_efSearch=8      0.4459 0.5132 0.5132      0.00538      137135674    56
k_factor_rf=1,nprobe=8,quantizer_efSearch=16     0.4576 0.5269 0.5269      0.00550      136858530    55
k_factor_rf=1,nprobe=16,quantizer_efSearch=4     0.4596 0.5305 0.5305      0.00687      272594185    44
k_factor_rf=2,nprobe=8,quantizer_efSearch=8      0.4920 0.5837 0.5838      0.00703      137135674    43
k_factor_rf=2,nprobe=8,quantizer_efSearch=16     0.5043 0.5987 0.5988      0.00712      136858530    43
k_factor_rf=2,nprobe=16,quantizer_efSearch=8     0.5391 0.6411 0.6413      0.00865      272310057    35
k_factor_rf=2,nprobe=16,quantizer_efSearch=16    0.5459 0.6514 0.6515      0.00872      271992392    35
k_factor_rf=2,nprobe=16,quantizer_efSearch=32    0.5504 0.6575 0.6576      0.00922      271722887    33
k_factor_rf=2,nprobe=32,quantizer_efSearch=8     0.5516 0.6564 0.6565      0.01066      540138468    29
k_factor_rf=2,nprobe=32,quantizer_efSearch=16    0.5685 0.6765 0.6766      0.01028      539200661    30
k_factor_rf=4,nprobe=16,quantizer_efSearch=8     0.5824 0.7105 0.7108      0.01144      272310057    27
k_factor_rf=4,nprobe=16,quantizer_efSearch=32    0.5958 0.7290 0.7293      0.01246      271722887    25
k_factor_rf=4,nprobe=32,quantizer_efSearch=16    0.6204 0.7617 0.7621      0.01378      539200661    22
k_factor_rf=4,nprobe=32,quantizer_efSearch=32    0.6274 0.7712 0.7716      0.01380      538452919    22
k_factor_rf=4,nprobe=64,quantizer_efSearch=64    0.6395 0.7904 0.7907      0.01959     1062978337    16
k_factor_rf=4,nprobe=128,quantizer_efSearch=64   0.6404 0.7926 0.7929      0.02239     2092545141    14
k_factor_rf=8,nprobe=32,quantizer_efSearch=16    0.6501 0.8186 0.8190      0.02261      539200661    14
k_factor_rf=8,nprobe=32,quantizer_efSearch=32    0.6570 0.8285 0.8289      0.02322      538452919    13
k_factor_rf=8,nprobe=32,quantizer_efSearch=64    0.6588 0.8309 0.8313      0.02408      538165850    13
k_factor_rf=8,nprobe=64,quantizer_efSearch=64    0.6749 0.8578 0.8582      0.02703     1062978337    12
k_factor_rf=8,nprobe=128,quantizer_efSearch=64   0.6828 0.8679 0.8684      0.03161     2092545141    10
k_factor_rf=16,nprobe=64,quantizer_efSearch=16   0.6872 0.8865 0.8870      0.04013     1065694383    8
k_factor_rf=16,nprobe=64,quantizer_efSearch=64   0.6994 0.9055 0.9060      0.04141     1062978337    8
k_factor_rf=16,nprobe=128,quantizer_efSearch=64  0.7085 0.9186 0.9192      0.05021     2092545141    7
k_factor_rf=16,nprobe=256,quantizer_efSearch=64  0.7098 0.9208 0.9214      0.05397     4112395032    6
k_factor_rf=16,nprobe=256,quantizer_efSearch=128 0.7124 0.9245 0.9251      0.05655     4106261213    6
k_factor_rf=32,nprobe=64,quantizer_efSearch=128  0.7130 0.9327 0.9338      0.06981     1062642815    5
k_factor_rf=32,nprobe=128,quantizer_efSearch=64  0.7249 0.9512 0.9521      0.08078     2092545141    4
k_factor_rf=32,nprobe=128,quantizer_efSearch=128 0.7266 0.9537 0.9546      0.07795     2091110918    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=64  0.7277 0.9563 0.9572      0.09397     4112395032    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=128 0.7300 0.9595 0.9604      0.08953     4106261213    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=256 0.7305 0.9601 0.9610      0.10006     4104350194    3
k_factor_rf=32,nprobe=512,quantizer_efSearch=256 0.7308 0.9602 0.9611      0.12104     8046518018    3
k_factor_rf=64,nprobe=128,quantizer_efSearch=64  0.7312 0.9665 0.9680      0.14208     2092545141    3
k_factor_rf=64,nprobe=128,quantizer_efSearch=256 0.7337 0.9697 0.9712      0.15012     2090848074    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=64  0.7359 0.9747 0.9762      0.15977     4112395032    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=128 0.7386 0.9783 0.9798      0.15988     4106261213    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=128 0.7399 0.9798 0.9813      0.17167     8056058192    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=512 0.7408 0.9812 0.9827      0.20381     8043825921    2
```

</details>
<details><summary>`OPQ16_64,IVF65536_HNSW32,PQ16x4fsr,Refine(OPQ56_112,PQ56)` </summary>
Index size 7243999634

 code_size 64

 log filename: autotune.dbbigann100M.OPQ16_64_IVF65536_HNSW32_PQ16x4fsr_Refine_OPQ56_112_PQ56_.e.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.7475 0.9968 0.9977      0.35904    15746373499    1
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.2445 0.2815 0.2815      0.00379       17147417    80
k_factor_rf=1,nprobe=1,quantizer_efSearch=64      0.2581 0.2979 0.2979      0.00807       17098067    38
k_factor_rf=1,nprobe=2,quantizer_efSearch=64      0.3631 0.4283 0.4283      0.00862       34226704    35
k_factor_rf=1,nprobe=4,quantizer_efSearch=64      0.4656 0.5613 0.5614      0.00957       68495782    32
k_factor_rf=1,nprobe=8,quantizer_efSearch=64      0.5557 0.6866 0.6867      0.01105      136746420    28
k_factor_rf=2,nprobe=8,quantizer_efSearch=64      0.5630 0.7037 0.7038      0.01284      136746420    24
k_factor_rf=1,nprobe=16,quantizer_efSearch=64     0.6259 0.7805 0.7806      0.01361      271553752    23
k_factor_rf=4,nprobe=16,quantizer_efSearch=64     0.6424 0.8202 0.8207      0.01920      271553752    16
k_factor_rf=1,nprobe=32,quantizer_efSearch=128    0.6663 0.8382 0.8383      0.02240      537870590    14
k_factor_rf=2,nprobe=32,quantizer_efSearch=128    0.6802 0.8729 0.8731      0.02468      537870590    13
k_factor_rf=1,nprobe=64,quantizer_efSearch=128    0.6916 0.8742 0.8743      0.02814     1062274227    11
k_factor_rf=2,nprobe=64,quantizer_efSearch=128    0.7092 0.9164 0.9168      0.03047     1062274227    10
k_factor_rf=4,nprobe=64,quantizer_efSearch=128    0.7167 0.9379 0.9386      0.03532     1062274227    9
k_factor_rf=4,nprobe=128,quantizer_efSearch=32    0.7264 0.9538 0.9545      0.05568     2095553316    6
k_factor_rf=4,nprobe=128,quantizer_efSearch=128   0.7327 0.9628 0.9635      0.06037     2090578799    5
k_factor_rf=4,nprobe=128,quantizer_efSearch=256   0.7329 0.9631 0.9638      0.06774     2090283210    5
k_factor_rf=8,nprobe=128,quantizer_efSearch=128   0.7356 0.9752 0.9760      0.06931     2090578799    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.7366 0.9790 0.9799      0.08645     2090578799    4
k_factor_rf=16,nprobe=128,quantizer_efSearch=256  0.7368 0.9793 0.9802      0.09508     2090283210    4
k_factor_rf=8,nprobe=256,quantizer_efSearch=128   0.7429 0.9863 0.9871      0.10039     4105326767    3
k_factor_rf=8,nprobe=256,quantizer_efSearch=256   0.7431 0.9867 0.9875      0.11427     4103470314    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.7444 0.9913 0.9922      0.11839     4105326767    3
k_factor_rf=16,nprobe=256,quantizer_efSearch=256  0.7447 0.9917 0.9926      0.13010     4103470314    3
k_factor_rf=8,nprobe=512,quantizer_efSearch=256   0.7456 0.9901 0.9908      0.18311     8044343409    2
k_factor_rf=8,nprobe=512,quantizer_efSearch=512   0.7457 0.9902 0.9909      0.20005     8041645618    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=128  0.7467 0.9953 0.9962      0.19293     8053693667    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=256  0.7472 0.9959 0.9968      0.20580     8044343409    2
k_factor_rf=16,nprobe=512,quantizer_efSearch=512  0.7473 0.9960 0.9969      0.22616     8041645618    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.7476 0.9969 0.9980      0.23672     8044343409    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.7477 0.9970 0.9981      0.25507     8041645618    2
k_factor_rf=32,nprobe=1024,quantizer_efSearch=512 0.7481 0.9980 0.9991      0.38272    15746373499    1
```

</details>
<details><summary>`OPQ56_112,IVF1048576_HNSW32,PQ7+56` </summary>
Index size 7863799516

 code_size 63

 log filename: autotune.dbbigann100M.OPQ56_112_IVF1048576_HNSW32_PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.6386 0.8309 0.8314      0.09539              0    4
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.2711 0.3086 0.3086      0.00628              0    48
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.3126 0.3585 0.3585      0.00730              0    42
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.3782 0.4448 0.4448      0.00869              0    35
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.4366 0.5228 0.5229      0.01166              0    26
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.4731 0.5708 0.5709      0.01300              0    24
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.4891 0.5908 0.5909      0.01429              0    21
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.4960 0.6041 0.6042      0.01716              0    18
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=1      0.5198 0.6353 0.6355      0.01895              0    16
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=1     0.5573 0.6890 0.6893      0.02297              0    14
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.5704 0.7145 0.7148      0.02619              0    12
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=4     0.5747 0.7259 0.7262      0.03203              0    10
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=1     0.5751 0.7169 0.7171      0.03137              0    10
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=2     0.5759 0.7224 0.7227      0.03280              0    10
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=2     0.5934 0.7543 0.7545      0.03472              0    9
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=1     0.6035 0.7577 0.7580      0.04244              0    8
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.6250 0.7997 0.8000      0.04470              0    7
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=2    0.6279 0.8033 0.8036      0.05637              0    6
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.6356 0.8260 0.8265      0.06329              0    5
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=8    0.6386 0.8297 0.8302      0.07210              0    5
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.6527 0.8468 0.8472      0.08588              0    4
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=2   0.6581 0.8543 0.8547      0.09753              0    4
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.6715 0.8862 0.8867      0.10316              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.6758 0.8980 0.8988      0.11656              0    3
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=4    0.6788 0.9043 0.9051      0.12211              0    3
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=8    0.6851 0.9228 0.9240      0.13369              0    3
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=8   0.6939 0.9366 0.9377      0.14433              0    3
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.6970 0.9444 0.9459      0.19033              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.6973 0.9456 0.9471      0.24253              0    2
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=16  0.7038 0.9625 0.9643      0.26123              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.7071 0.9690 0.9709      0.28402              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.7075 0.9702 0.9721      0.32148              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.7079 0.9731 0.9750      0.37408              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=16 0.7128 0.9828 0.9848      0.52243              0    1
nprobe=2048,quantizer_efSearch=256,ht=56,k_factor=32 0.7138 0.9867 0.9894      0.88707              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=32 0.7149 0.9902 0.9931      0.96037              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=32 0.7154 0.9911 0.9941      1.68982              0    1
```

</details>
<details><summary>`OPQ56_112,IVF1048576(IVF1024,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 7617111584

 code_size 63

 log filename: autotune.dbbigann100M.OPQ56_112_IVF1048576_IVF1024_PQ56x4fs_RFlat__PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.4997 0.6035 0.6035      0.02029       10808578    15
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=56,k_factor=1        0.1419 0.1545 0.1545      0.00538         725363    56
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=2,ht=56,k_factor=2       0.1449 0.1578 0.1578      0.00630         724759    48
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=56,k_factor=2       0.1524 0.1663 0.1663      0.00629        1443245    48
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=56,k_factor=1       0.2087 0.2320 0.2320      0.00625         725648    49
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=56,k_factor=1        0.2229 0.2517 0.2517      0.00659         366873    46
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=56,k_factor=2        0.2237 0.2509 0.2509      0.00812        1442932    37
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=2        0.2304 0.2595 0.2595      0.00845        2845699    36
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=56,k_factor=1        0.2652 0.3060 0.3060      0.00918         366499    33
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=1       0.2780 0.3157 0.3157      0.00930       10858488    33
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=56,k_factor=2        0.2937 0.3360 0.3360      0.00932        1444163    33
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1       0.4077 0.4782 0.4782      0.01068        5566355    29
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=56,k_factor=2      0.4187 0.4973 0.4973      0.01667        5572151    19
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=2       0.4502 0.5396 0.5397      0.01711        2848147    18
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.4999 0.6104 0.6105      0.01896        5572151    16
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=2       0.5145 0.6348 0.6349      0.02760        2839882    11
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=1      0.5574 0.6870 0.6872      0.02631        5572151    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=2      0.5729 0.7155 0.7158      0.03106       21253370    10
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=2     0.5783 0.7243 0.7246      0.03611       41878138    9
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=4      0.5839 0.7349 0.7354      0.03810       20886219    8
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=1      0.6066 0.7589 0.7591      0.04796       10878384    7
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=56,k_factor=4      0.6098 0.7786 0.7793      0.05582       21210158    6
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=4     0.6390 0.8252 0.8259      0.07373       81471807    5
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=56,k_factor=4     0.6412 0.8301 0.8308      0.08455        5572151    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=4     0.6764 0.8876 0.8884      0.09408       20316011    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56,k_factor=4    0.6767 0.8882 0.8890      0.10396       40140507    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=56,k_factor=4   0.6770 0.8883 0.8891      0.14116      162164986    3
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=4    0.6837 0.9033 0.9042      0.15835       40557718    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=4    0.6937 0.9207 0.9216      0.16927       82494674    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=4    0.6945 0.9223 0.9233      0.18874       82494674    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=16    0.7004 0.9445 0.9458      0.20459       20964653    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56,k_factor=32   0.7013 0.9470 0.9485      0.27156       41878138    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=32   0.7015 0.9472 0.9487      0.27982      163862211    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=16   0.7130 0.9733 0.9750      0.37744      163862211    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=56,k_factor=32  0.7132 0.9749 0.9769      0.71052       41878138    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=8   0.7133 0.9725 0.9741      0.56581       41878138    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=16  0.7170 0.9843 0.9864      0.60011       82494674    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=32  0.7178 0.9884 0.9910      0.71824       82494674    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=512,ht=56,k_factor=16  0.7181 0.9860 0.9882      1.13092      163862211    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=16  0.7184 0.9875 0.9897      1.13135       41878138    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=32  0.7195 0.9924 0.9954      1.20608       41878138    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=64  0.7197 0.9933 0.9964      1.30949       82494674    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=56,k_factor=32 0.7198 0.9928 0.9958      2.60780       82494674    1
```

</details>
<details><summary>`OPQ56_112,IVF262144_HNSW32,PQ7+56` </summary>
Index size 7291167708

 code_size 63

 log filename: autotune.dbbigann100M.OPQ56_112_IVF262144_HNSW32_PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.7252 0.9316 0.9322      0.08252              0    4
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.2122 0.2303 0.2303      0.00620              0    49
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.4013 0.4547 0.4547      0.00674              0    45
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.4653 0.5360 0.5360      0.00714              0    43
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.5171 0.6045 0.6045      0.00864              0    35
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.5512 0.6482 0.6482      0.00934              0    33
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.5651 0.6654 0.6654      0.01026              0    30
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.5708 0.6743 0.6743      0.01311              0    23
nprobe=32,quantizer_efSearch=8,ht=56,k_factor=1      0.5863 0.6971 0.6971      0.01215              0    25
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=1     0.6190 0.7411 0.7411      0.01590              0    19
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.6409 0.7795 0.7795      0.01713              0    18
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.6471 0.7909 0.7910      0.01885              0    16
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=4     0.6542 0.8058 0.8059      0.02319              0    13
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=2     0.6641 0.8147 0.8147      0.02218              0    14
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.6809 0.8411 0.8411      0.02913              0    11
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=2    0.6827 0.8437 0.8437      0.03761              0    8
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=2    0.6882 0.8522 0.8522      0.03616              0    9
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=2    0.6948 0.8619 0.8619      0.04103              0    8
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=4    0.7116 0.8990 0.8992      0.04242              0    8
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=4    0.7190 0.9099 0.9101      0.04707              0    7
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.7217 0.9140 0.9142      0.05591              0    6
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=8    0.7223 0.9223 0.9227      0.05845              0    6
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.7303 0.9344 0.9349      0.06183              0    5
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.7331 0.9390 0.9394      0.06648              0    5
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=8    0.7403 0.9512 0.9517      0.08898              0    4
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=8   0.7441 0.9587 0.9592      0.09523              0    4
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.7512 0.9739 0.9747      0.13173              0    3
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=16  0.7515 0.9744 0.9752      0.16755              0    2
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=16  0.7535 0.9799 0.9807      0.17316              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.7553 0.9818 0.9826      0.18747              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.7557 0.9826 0.9834      0.21673              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.7572 0.9878 0.9888      0.24045              0    2
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.7576 0.9886 0.9896      0.27440              0    2
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=64  0.7581 0.9902 0.9912      0.34404              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=64  0.7585 0.9912 0.9922      0.36442              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=64 0.7604 0.9943 0.9954      0.43635              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=64 0.7610 0.9966 0.9977      1.05467              0    1
```

</details>
<details><summary>`OPQ56_112,IVF262144(IVF512,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 7229733536

 code_size 63

 log filename: autotune.dbbigann100M.OPQ56_112_IVF262144_IVF512_PQ56x4fs_RFlat__PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                           0.7009 0.8847 0.8848      0.04226       20804490    8
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=56,k_factor=1       0.1805 0.1977 0.1977      0.00517         353262    59
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=56,k_factor=1       0.1952 0.2164 0.2164      0.00550         178105    55
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=56,k_factor=1       0.2141 0.2336 0.2336      0.00553        1393803    55
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=8,ht=56,k_factor=1       0.2668 0.2962 0.2962      0.00559        1392164    54
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=56,k_factor=1       0.3889 0.4391 0.4392      0.00572         705927    53
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=1       0.4013 0.4562 0.4563      0.00604        1395392    50
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1      0.4954 0.5769 0.5770      0.00728        2731333    42
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=1     0.5668 0.6708 0.6709      0.01017        2722970    30
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=2     0.5883 0.7076 0.7077      0.01346        5360744    23
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=56,k_factor=1     0.5951 0.7127 0.7128      0.01359        5360744    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=2      0.6217 0.7577 0.7578      0.01584        1392979    19
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=4      0.6331 0.7791 0.7792      0.02184        1395392    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=56,k_factor=2    0.6478 0.7951 0.7952      0.02125       20804490    15
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=2    0.6505 0.7989 0.7990      0.02196       20711925    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=4     0.6532 0.8109 0.8110      0.02258        2725241    14
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=4     0.6897 0.8674 0.8675      0.02980        2731333    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=4     0.6978 0.8810 0.8811      0.03252       10553518    10
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=56,k_factor=4   0.7009 0.8847 0.8848      0.04213       20804490    8
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=4    0.7171 0.9104 0.9105      0.05204        5332479    6
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=4   0.7274 0.9294 0.9295      0.08679       20804490    4
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=56,k_factor=8  0.7298 0.9391 0.9393      0.09602       20804490    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=16  0.7313 0.9464 0.9468      0.09062       41216679    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=8   0.7400 0.9587 0.9590      0.09777       41216679    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=8   0.7403 0.9592 0.9595      0.10015       40793196    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=16   0.7449 0.9735 0.9739      0.11499       10553518    3
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=56,k_factor=16 0.7454 0.9741 0.9745      0.15025       41216679    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=56,k_factor=32  0.7472 0.9795 0.9800      0.17336       41216679    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=56,k_factor=32 0.7476 0.9799 0.9804      0.20365       41216679    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=32  0.7517 0.9910 0.9916      0.23645       41216679    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=64  0.7522 0.9926 0.9933      0.35915       41216679    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32  0.7537 0.9944 0.9950      0.56610       10553518    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=32 0.7545 0.9962 0.9968      0.60069       20804490    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=56,k_factor=64 0.7551 0.9984 0.9991      0.69562       20804490    1
```

</details>
<details><summary>`OPQ56_112,IVF65536_HNSW32,PQ7+56` </summary>
Index size 7148008924

 code_size 63

 log filename: autotune.dbbigann100M.OPQ56_112_IVF65536_HNSW32_PQ7+56.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.7831 0.9598 0.9599      0.11765              0    3
nprobe=1,quantizer_efSearch=8,ht=56,k_factor=1       0.2669 0.2919 0.2919      0.00562              0    54
nprobe=2,quantizer_efSearch=8,ht=56,k_factor=1       0.3719 0.4127 0.4127      0.00622              0    49
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.4254 0.4762 0.4762      0.00707              0    43
nprobe=4,quantizer_efSearch=8,ht=56,k_factor=1       0.4700 0.5274 0.5274      0.00754              0    40
nprobe=4,quantizer_efSearch=16,ht=56,k_factor=1      0.4790 0.5378 0.5378      0.00830              0    37
nprobe=4,quantizer_efSearch=32,ht=56,k_factor=1      0.4807 0.5396 0.5396      0.00905              0    34
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.5368 0.6107 0.6107      0.00932              0    33
nprobe=8,quantizer_efSearch=64,ht=56,k_factor=1      0.5692 0.6497 0.6497      0.01291              0    24
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=2       0.5735 0.6614 0.6614      0.01305              0    23
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.5787 0.6617 0.6617      0.01339              0    23
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.6040 0.6939 0.6939      0.01322              0    23
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.6141 0.7067 0.7067      0.01368              0    22
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.6208 0.7142 0.7142      0.01451              0    21
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.6506 0.7611 0.7611      0.01755              0    18
nprobe=16,quantizer_efSearch=128,ht=56,k_factor=2    0.6595 0.7719 0.7719      0.02344              0    13
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=4     0.6716 0.7937 0.7938      0.02461              0    13
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.6894 0.8092 0.8092      0.02517              0    12
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.6966 0.8189 0.8189      0.02598              0    12
nprobe=32,quantizer_efSearch=128,ht=56,k_factor=2    0.6987 0.8225 0.8225      0.03180              0    10
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=4     0.7162 0.8531 0.8532      0.03251              0    10
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=4     0.7228 0.8629 0.8630      0.03355              0    9
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=4     0.7241 0.8657 0.8658      0.03483              0    9
nprobe=32,quantizer_efSearch=256,ht=56,k_factor=4    0.7247 0.8664 0.8665      0.04669              0    7
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=8     0.7350 0.8862 0.8863      0.04806              0    7
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=8     0.7364 0.8890 0.8891      0.04863              0    7
nprobe=32,quantizer_efSearch=256,ht=56,k_factor=8    0.7370 0.8897 0.8898      0.06020              0    5
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=8     0.7516 0.9094 0.9095      0.06210              0    5
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.7648 0.9291 0.9292      0.06518              0    5
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=8    0.7656 0.9301 0.9302      0.06732              0    5
nprobe=64,quantizer_efSearch=256,ht=56,k_factor=8    0.7657 0.9302 0.9303      0.07520              0    4
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=16    0.7745 0.9463 0.9464      0.09311              0    4
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.7760 0.9469 0.9470      0.09702              0    4
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.7772 0.9484 0.9485      0.09649              0    4
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=16   0.7831 0.9598 0.9599      0.11829              0    3
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=16   0.7875 0.9677 0.9678      0.11954              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=16  0.7885 0.9691 0.9692      0.12547              0    3
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=16  0.7887 0.9692 0.9693      0.13211              0    3
nprobe=128,quantizer_efSearch=512,ht=56,k_factor=16  0.7888 0.9694 0.9695      0.15237              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.7942 0.9778 0.9779      0.18908              0    2
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=32  0.7988 0.9879 0.9881      0.23633              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.7992 0.9883 0.9885      0.24494              0    2
nprobe=256,quantizer_efSearch=128,ht=56,k_factor=64  0.8013 0.9927 0.9929      0.34266              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=64  0.8017 0.9931 0.9933      0.34878              0    1
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=64  0.8030 0.9958 0.9960      0.46040              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=64  0.8035 0.9962 0.9964      0.46772              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=64  0.8036 0.9965 0.9967      0.48697              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=64 0.8037 0.9971 0.9973      0.69604              0    1
nprobe=4096,quantizer_efSearch=512,ht=56,k_factor=64 0.8038 0.9974 0.9976      2.02273              0    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64` </summary>
Index size 8030818224

 code_size 64

 log filename: autotune.dbbigann100M.OPQ64_128_IVF1048576_HNSW32_PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.2021 0.2114 0.2114      1.59705      926395480    1
nprobe=1,quantizer_efSearch=4,ht=256      0.1466 0.1526 0.1526      0.00470        1163317    64
nprobe=2,quantizer_efSearch=4,ht=230      0.1822 0.1906 0.1906      0.00591        2319724    51
nprobe=2,quantizer_efSearch=8,ht=254      0.2427 0.2547 0.2547      0.00766        2319045    40
nprobe=4,quantizer_efSearch=4,ht=238      0.2771 0.2933 0.2933      0.00942        4634434    32
nprobe=4,quantizer_efSearch=32,ht=228     0.2846 0.2989 0.2989      0.01769        4596202    17
nprobe=4,quantizer_efSearch=32,ht=244     0.3491 0.3706 0.3706      0.01789        4596202    17
nprobe=8,quantizer_efSearch=32,ht=244     0.4580 0.4903 0.4903      0.02505        9186919    12
nprobe=16,quantizer_efSearch=16,ht=256    0.5637 0.6106 0.6106      0.03578       18369559    9
nprobe=16,quantizer_efSearch=32,ht=246    0.5673 0.6130 0.6130      0.03916       18296373    8
nprobe=32,quantizer_efSearch=128,ht=512   0.6741 0.7390 0.7390      0.05483       36242018    6
nprobe=256,quantizer_efSearch=32,ht=512   0.7989 0.8963 0.8963      0.18303      281492585    2
nprobe=512,quantizer_efSearch=512,ht=512  0.8619 0.9754 0.9754      0.46034      547100767    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8750 0.9964 0.9964      2.71162     4068466458    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576(IVF1024,PQ64x4fs,RFlat),PQ64` </summary>
Index size 7788463604

 code_size 64

 log filename: autotune.dbbigann100M.OPQ64_128_IVF1048576_IVF1024_PQ64x4fs_RFlat__PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.8469 0.9567 0.9567      0.91988      562474948    1
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=234       0.1396 0.1451 0.1451      0.00376        2610154    80
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=252      0.1641 0.1705 0.1705      0.00523        6728481    58
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=252       0.2473 0.2597 0.2597      0.00652        5163805    47
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=236     0.3306 0.3492 0.3492      0.01254       10171984    24
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=244     0.3512 0.3728 0.3728      0.01367       15462419    22
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=228       0.3614 0.3826 0.3826      0.01765       12090813    18
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=224    0.4095 0.4332 0.4332      0.04028       28845468    8
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=512    0.6754 0.7397 0.7397      0.05912       57305342    6
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=512    0.7776 0.8656 0.8656      0.09647      155645684    4
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64,ht=256    0.8237 0.9245 0.9245      0.48206      305695529    1
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=244   0.8309 0.9329 0.9329      0.48205      320733958    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=252    0.8469 0.9567 0.9567      0.91637      561768687    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256,ht=252   0.8608 0.9737 0.9737      0.93331      623233897    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256,ht=250  0.8726 0.9911 0.9911      3.58988     2235161511    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=512 0.8766 0.9992 0.9992      5.73713     4259112137    1
```

</details>
<details><summary>`OPQ64_128,IVF262144_HNSW32,PQ64` </summary>
Index size 7407854768

 code_size 64

 log filename: autotune.dbbigann100M.OPQ64_128_IVF262144_HNSW32_PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3245 0.3424 0.3424      0.00736        8616614    41
nprobe=1,quantizer_efSearch=4,ht=106      0.0003 0.0003 0.0003      0.00334        4321357    90
nprobe=1,quantizer_efSearch=8,ht=156      0.0012 0.0012 0.0012      0.00431        4298223    70
nprobe=1,quantizer_efSearch=4,ht=256      0.2073 0.2165 0.2165      0.00422        4321357    72
nprobe=2,quantizer_efSearch=4,ht=230      0.2649 0.2778 0.2778      0.00576        8650839    53
nprobe=2,quantizer_efSearch=8,ht=252      0.3245 0.3424 0.3424      0.00754        8616614    40
nprobe=2,quantizer_efSearch=8,ht=254      0.3248 0.3428 0.3428      0.00752        8616614    40
nprobe=4,quantizer_efSearch=4,ht=238      0.3791 0.4051 0.4051      0.00993       17319432    31
nprobe=4,quantizer_efSearch=32,ht=228     0.3956 0.4186 0.4186      0.01478       17212064    21
nprobe=4,quantizer_efSearch=32,ht=244     0.4541 0.4862 0.4862      0.01568       17212064    20
nprobe=8,quantizer_efSearch=32,ht=244     0.5622 0.6116 0.6116      0.02384       34389518    13
nprobe=8,quantizer_efSearch=128,ht=256    0.5693 0.6197 0.6197      0.04169       34364702    8
nprobe=16,quantizer_efSearch=16,ht=256    0.6549 0.7218 0.7218      0.04378       68770050    7
nprobe=16,quantizer_efSearch=32,ht=246    0.6613 0.7289 0.7289      0.04212       68662451    8
nprobe=32,quantizer_efSearch=128,ht=512   0.7507 0.8391 0.8391      0.06438      136381966    5
nprobe=64,quantizer_efSearch=32,ht=234    0.7536 0.8378 0.8378      0.12875      270939510    3
nprobe=64,quantizer_efSearch=512,ht=256   0.8060 0.9085 0.9085      0.21736      270355963    2
nprobe=128,quantizer_efSearch=64,ht=246   0.8308 0.9425 0.9425      0.27764      535199848    2
nprobe=256,quantizer_efSearch=32,ht=512   0.8343 0.9499 0.9499      0.32828     1058976370    1
nprobe=512,quantizer_efSearch=512,ht=512  0.8660 0.9921 0.9921      0.67818     2073932902    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8702 0.9988 0.9988      4.67337    15298060962    1
```

</details>
<details><summary>`OPQ64_128,IVF262144(IVF512,PQ64x4fs,RFlat),PQ64` </summary>
Index size 7347536372

 code_size 64

 log filename: autotune.dbbigann100M.OPQ64_128_IVF262144_IVF512_PQ64x4fs_RFlat__PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.3283 0.3452 0.3452      0.00806       11326969    38
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=138       0.0003 0.0003 0.0003      0.00293        4534761    103
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=164       0.0021 0.0021 0.0021      0.00300        4527232    100
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=230       0.1524 0.1584 0.1584      0.00323        4527269    93
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=230       0.1872 0.1939 0.1939      0.00363        5707821    83
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=246      0.2136 0.2226 0.2226      0.00450        7029394    67
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=234      0.2245 0.2359 0.2359      0.00562        8885622    54
nprobe=2,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=240     0.3283 0.3452 0.3452      0.00830       11322280    37
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256      0.3366 0.3547 0.3547      0.01035       19146753    29
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=250     0.4577 0.4906 0.4906      0.01400       22582153    22
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=232       0.5060 0.5457 0.5457      0.01815       36057263    17
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=246      0.5617 0.6119 0.6119      0.02084       37202961    15
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=512      0.5893 0.6466 0.6466      0.02371       70254956    13
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=256    0.6654 0.7353 0.7353      0.04735       79132726    7
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32,ht=236     0.6860 0.7605 0.7605      0.06720      144131252    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=242    0.7353 0.8208 0.8208      0.07561      157609415    4
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=238     0.7669 0.8586 0.8586      0.13214      275075525    3
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=256     0.8027 0.9077 0.9077      0.16519      281268180    2
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=246   0.8143 0.9217 0.9217      0.28774      540860264    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=240    0.8421 0.9575 0.9575      0.52023     1066510788    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=250    0.8561 0.9784 0.9784      0.54686     1066544399    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=256,ht=244  0.8603 0.9832 0.9832      1.17998     2116388004    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=244 0.8641 0.9892 0.9892      5.16247     8022706999    1
nprobe=4096,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=512 0.8708 1.0000 1.0000      6.25710    15672795263    1
```

</details>
<details><summary>`OPQ64_128,IVF65536_HNSW32,PQ64` </summary>
Index size 7252113072

 code_size 64

 log filename: autotune.dbbigann100M.OPQ64_128_IVF65536_HNSW32_PQ64.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3975 0.4304 0.4304      0.01018       34254595    30
nprobe=1,quantizer_efSearch=4,ht=106      0.0003 0.0003 0.0003      0.00321       17143107    94
nprobe=1,quantizer_efSearch=4,ht=204      0.1032 0.1088 0.1088      0.00339       17143107    89
nprobe=1,quantizer_efSearch=16,ht=234     0.2726 0.2894 0.2894      0.00499       17087122    61
nprobe=2,quantizer_efSearch=4,ht=230      0.3459 0.3714 0.3714      0.00674       34347050    45
nprobe=2,quantizer_efSearch=16,ht=226     0.3588 0.3841 0.3841      0.00754       34213277    40
nprobe=2,quantizer_efSearch=16,ht=234     0.3908 0.4211 0.4211      0.00804       34213277    38
nprobe=2,quantizer_efSearch=8,ht=254      0.3976 0.4305 0.4305      0.01065       34254595    29
nprobe=4,quantizer_efSearch=32,ht=228     0.4831 0.5244 0.5244      0.01425       68257747    22
nprobe=4,quantizer_efSearch=32,ht=244     0.5265 0.5761 0.5761      0.01707       68257747    18
nprobe=8,quantizer_efSearch=64,ht=222     0.5365 0.5818 0.5818      0.02729      136293386    11
nprobe=8,quantizer_efSearch=32,ht=244     0.6475 0.7180 0.7180      0.03084      136322714    10
nprobe=8,quantizer_efSearch=128,ht=256    0.6512 0.7229 0.7229      0.04402      136284123    7
nprobe=16,quantizer_efSearch=64,ht=232    0.7047 0.7854 0.7854      0.05112      270941272    6
nprobe=16,quantizer_efSearch=32,ht=246    0.7360 0.8247 0.8247      0.05802      271040701    6
nprobe=16,quantizer_efSearch=256,ht=246   0.7373 0.8259 0.8259      0.07331      270894123    5
nprobe=32,quantizer_efSearch=32,ht=232    0.7593 0.8552 0.8552      0.09234      537271817    4
nprobe=32,quantizer_efSearch=32,ht=246    0.7913 0.8981 0.8981      0.11080      537271817    3
nprobe=32,quantizer_efSearch=128,ht=512   0.7958 0.9039 0.9039      0.12038      536909172    3
nprobe=64,quantizer_efSearch=32,ht=234    0.8028 0.9141 0.9141      0.17484     1062531488    2
nprobe=64,quantizer_efSearch=512,ht=256   0.8321 0.9544 0.9544      0.30092     1061194193    1
nprobe=128,quantizer_efSearch=64,ht=246   0.8485 0.9761 0.9761      0.40794     2090993797    1
nprobe=256,quantizer_efSearch=32,ht=512   0.8517 0.9813 0.9813      0.86554     4117669720    1
nprobe=512,quantizer_efSearch=512,ht=248  0.8627 0.9973 0.9973      1.60019     8048396809    1
nprobe=512,quantizer_efSearch=512,ht=512  0.8630 0.9986 0.9986      1.70799     8048396809    1
nprobe=4096,quantizer_efSearch=512,ht=512 0.8638 0.9999 0.9999     12.20279    58545643036    1
```

</details>
<details><summary>`PCAR128,IVF1048576_HNSW32,SQ4` </summary>
Index size 8030755285

 code_size 64

 log filename: autotune.dbbigann100M.PCAR128_IVF1048576_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7330 0.9459 0.9479      0.17250      278471593    2
nprobe=2,quantizer_efSearch=4            0.2002 0.2273 0.2275      0.00357        2319715    84
nprobe=4,quantizer_efSearch=4            0.2663 0.3119 0.3122      0.00404        4634263    75
nprobe=4,quantizer_efSearch=8            0.3111 0.3628 0.3632      0.00533        4631233    57
nprobe=8,quantizer_efSearch=4            0.3806 0.4577 0.4585      0.00635        9269918    48
nprobe=8,quantizer_efSearch=8            0.3895 0.4691 0.4700      0.00697        9260208    44
nprobe=8,quantizer_efSearch=16           0.4131 0.4960 0.4969      0.00898        9220625    34
nprobe=16,quantizer_efSearch=4           0.4460 0.5456 0.5466      0.00871       18449813    35
nprobe=16,quantizer_efSearch=8           0.4828 0.5906 0.5920      0.01003       18420417    30
nprobe=16,quantizer_efSearch=16          0.4988 0.6103 0.6117      0.01183       18370991    26
nprobe=32,quantizer_efSearch=8           0.5398 0.6707 0.6724      0.01607       36644404    19
nprobe=32,quantizer_efSearch=16          0.5716 0.7119 0.7136      0.01766       36517455    17
nprobe=32,quantizer_efSearch=32          0.5848 0.7276 0.7293      0.02039       36391525    15
nprobe=64,quantizer_efSearch=16          0.6189 0.7786 0.7804      0.02745       72397058    11
nprobe=64,quantizer_efSearch=32          0.6421 0.8104 0.8123      0.03028       72094677    10
nprobe=128,quantizer_efSearch=32         0.6786 0.8680 0.8700      0.05387      142670158    6
nprobe=128,quantizer_efSearch=64         0.6969 0.8921 0.8940      0.06345      142091047    5
nprobe=128,quantizer_efSearch=128        0.7032 0.9002 0.9021      0.06773      141670531    5
nprobe=128,quantizer_efSearch=256        0.7041 0.9018 0.9037      0.09210      141497955    4
nprobe=256,quantizer_efSearch=64         0.7189 0.9279 0.9299      0.11044      280545992    3
nprobe=256,quantizer_efSearch=128        0.7305 0.9428 0.9448      0.11740      279322369    3
nprobe=256,quantizer_efSearch=256        0.7330 0.9457 0.9477      0.13161      278683335    3
nprobe=512,quantizer_efSearch=128        0.7443 0.9660 0.9681      0.19338      550266595    2
nprobe=512,quantizer_efSearch=256        0.7474 0.9720 0.9741      0.20499      548064893    2
nprobe=512,quantizer_efSearch=512        0.7488 0.9733 0.9754      0.25977      547093352    2
nprobe=1024,quantizer_efSearch=256       0.7546 0.9857 0.9878      0.37186     1077194233    1
nprobe=1024,quantizer_efSearch=512       0.7565 0.9878 0.9899      0.43599     1073462052    1
nprobe=2048,quantizer_efSearch=512       0.7583 0.9924 0.9946      0.72967     2104571231    1
nprobe=4096,quantizer_efSearch=512       0.7592 0.9944 0.9966      1.32051     4068447909    1
```

</details>
<details><summary>`PCAR128,IVF1048576(IVF1024,PQ64x4fs,RFlat),SQ4` </summary>
Index size 7788395545

 code_size 64

 log filename: autotune.dbbigann100M.PCAR128_IVF1048576_IVF1024_PQ64x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.7204 0.9256 0.9275      0.09998      326288741    4
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1188 0.1307 0.1307      0.00301        1535393    100
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1316 0.1455 0.1455      0.00307        1900471    98
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1380 0.1526 0.1526      0.00304        1895960    99
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1513 0.1679 0.1679      0.00309        2721173    98
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1675 0.1876 0.1878      0.00320        2698158    94
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1774 0.1983 0.1983      0.00321        3083785    94
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1998 0.2257 0.2260      0.00332        3059985    91
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2211 0.2495 0.2498      0.00375        3769141    81
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2631 0.3013 0.3018      0.00407        5403725    74
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2678 0.3086 0.3092      0.00414        5388731    73
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.2692 0.3100 0.3106      0.00437        5387415    69
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2955 0.3392 0.3397      0.00436        6108222    69
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3000 0.3465 0.3472      0.00513        6092284    59
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3119 0.3581 0.3586      0.00487        7492705    62
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.3257 0.3759 0.3765      0.00584       10182633    52
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3484 0.4062 0.4068      0.00599       10833420    51
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3719 0.4404 0.4414      0.00619       10756810    49
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3813 0.4519 0.4529      0.00689       10727983    44
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.4068 0.4820 0.4829      0.00692       12097986    44
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.4092 0.4841 0.4851      0.00785       12086015    39
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4229 0.4998 0.5007      0.01038       20004902    29
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4423 0.5361 0.5372      0.01002       19985120    30
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.4466 0.5413 0.5425      0.01001       19931903    30
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4528 0.5441 0.5452      0.00970       21499563    31
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5001 0.6075 0.6086      0.01118       23962718    27
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5058 0.6154 0.6166      0.01132       23900088    27
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5067 0.6170 0.6182      0.01270       23895009    24
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.5068 0.6172 0.6184      0.01462       23904664    21
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.5126 0.6234 0.6246      0.01677       28954480    18
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5128 0.6236 0.6248      0.01459       29103718    21
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5129 0.6244 0.6256      0.01905       38718590    16
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.5590 0.6899 0.6916      0.01960       39444035    16
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5845 0.7236 0.7253      0.01813       42023978    17
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.5860 0.7252 0.7269      0.02011       42023404    15
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5935 0.7363 0.7380      0.02270       47124217    14
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5936 0.7367 0.7384      0.02242       56979840    14
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5945 0.7379 0.7396      0.02392       57014786    13
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6172 0.7699 0.7716      0.02906       79038427    11
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6402 0.8017 0.8035      0.03068       77990308    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6461 0.8081 0.8099      0.03513       77703450    9
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6534 0.8204 0.8222      0.03048       82913174    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6584 0.8262 0.8280      0.03250       82710586    10
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.6626 0.8314 0.8332      0.04841       92011685    7
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.6665 0.8408 0.8426      0.05064      151086524    6
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.7050 0.8961 0.8980      0.05698      152940654    6
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7097 0.9030 0.9049      0.06432      162747212    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7331 0.9446 0.9466      0.10361      300786821    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7340 0.9458 0.9478      0.11310      320148098    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7357 0.9475 0.9495      0.12067      357927163    3
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.7361 0.9478 0.9498      0.14104      361306020    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128  0.7443 0.9642 0.9662      0.19824      602232539    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7508 0.9749 0.9769      0.20658      588982658    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.7512 0.9753 0.9773      0.21138      630038406    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.7585 0.9893 0.9913      0.37871     1157180644    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7604 0.9943 0.9963      0.73048     2144685553    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.7612 0.9950 0.9971      0.72597     2180344846    1
nprobe=4096,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7614 0.9967 0.9987      1.29216     4149335228    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.7616 0.9968 0.9988      1.37214     4145826063    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.7619 0.9973 0.9994      1.44494     4179003776    1
```

</details>
<details><summary>`PCAR128,IVF262144_HNSW32,SQ4` </summary>
Index size 7407791829

 code_size 64

 log filename: autotune.dbbigann100M.PCAR128_IVF262144_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7328 0.9770 0.9799      0.34865     1054111668    1
nprobe=1,quantizer_efSearch=4            0.1875 0.2159 0.2161      0.00335        4321648    90
nprobe=1,quantizer_efSearch=8            0.2023 0.2318 0.2320      0.00415        4301411    73
nprobe=2,quantizer_efSearch=4            0.2686 0.3143 0.3150      0.00457        8659459    66
nprobe=2,quantizer_efSearch=8            0.2940 0.3430 0.3438      0.00540        8621477    56
nprobe=2,quantizer_efSearch=16           0.3019 0.3521 0.3529      0.00678        8602628    45
nprobe=4,quantizer_efSearch=4            0.3496 0.4197 0.4204      0.00714       17332734    42
nprobe=4,quantizer_efSearch=8            0.3916 0.4702 0.4710      0.00789       17272897    39
nprobe=4,quantizer_efSearch=16           0.4042 0.4862 0.4870      0.00925       17237024    33
nprobe=4,quantizer_efSearch=32           0.4080 0.4908 0.4916      0.01187       17219456    26
nprobe=8,quantizer_efSearch=4            0.4570 0.5649 0.5665      0.01349       34552242    23
nprobe=8,quantizer_efSearch=8            0.4725 0.5829 0.5845      0.01286       34521261    24
nprobe=8,quantizer_efSearch=16           0.4944 0.6100 0.6115      0.01433       34447015    21
nprobe=8,quantizer_efSearch=32           0.4991 0.6163 0.6178      0.01664       34400104    19
nprobe=8,quantizer_efSearch=64           0.4994 0.6169 0.6184      0.02136       34383353    15
nprobe=16,quantizer_efSearch=4           0.5164 0.6523 0.6543      0.02195       68990888    14
nprobe=16,quantizer_efSearch=8           0.5547 0.7006 0.7026      0.02330       68877253    13
nprobe=16,quantizer_efSearch=16          0.5682 0.7198 0.7217      0.02351       68781880    13
nprobe=16,quantizer_efSearch=32          0.5768 0.7310 0.7330      0.02631       68675023    12
nprobe=16,quantizer_efSearch=64          0.5783 0.7335 0.7355      0.03098       68630060    10
nprobe=16,quantizer_efSearch=128         0.5789 0.7344 0.7364      0.03834       68617392    8
nprobe=32,quantizer_efSearch=8           0.6029 0.7765 0.7790      0.04226      137074045    8
nprobe=32,quantizer_efSearch=32          0.6432 0.8304 0.8330      0.04455      136582896    7
nprobe=32,quantizer_efSearch=64          0.6470 0.8353 0.8379      0.04917      136446725    7
nprobe=64,quantizer_efSearch=16          0.6676 0.8712 0.8739      0.07931      271572107    4
nprobe=64,quantizer_efSearch=32          0.6829 0.8944 0.8971      0.08069      270952060    4
nprobe=64,quantizer_efSearch=128         0.6905 0.9045 0.9073      0.09277      270436943    4
nprobe=64,quantizer_efSearch=256         0.6909 0.9050 0.9078      0.10837      270357304    3
nprobe=128,quantizer_efSearch=32         0.7040 0.9310 0.9339      0.15464      536438421    2
nprobe=128,quantizer_efSearch=64         0.7131 0.9444 0.9473      0.15629      535206869    2
nprobe=128,quantizer_efSearch=128        0.7159 0.9486 0.9515      0.16423      534656938    2
nprobe=128,quantizer_efSearch=256        0.7166 0.9495 0.9524      0.17875      534417469    2
nprobe=128,quantizer_efSearch=512        0.7167 0.9497 0.9526      0.21177      534382851    2
nprobe=256,quantizer_efSearch=128        0.7315 0.9754 0.9783      0.30398     1055207945    1
nprobe=256,quantizer_efSearch=256        0.7322 0.9764 0.9793      0.31458     1054295444    1
nprobe=256,quantizer_efSearch=512        0.7328 0.9770 0.9799      0.35014     1054111668    1
nprobe=512,quantizer_efSearch=128        0.7368 0.9853 0.9883      0.57260     2078616398    1
nprobe=512,quantizer_efSearch=256        0.7383 0.9880 0.9910      0.59102     2075071707    1
nprobe=512,quantizer_efSearch=512        0.7389 0.9888 0.9918      0.65358     2073938973    1
nprobe=1024,quantizer_efSearch=256       0.7407 0.9926 0.9958      1.12969     4077278201    1
nprobe=1024,quantizer_efSearch=512       0.7414 0.9936 0.9968      1.15610     4072854091    1
nprobe=2048,quantizer_efSearch=256       0.7415 0.9935 0.9968      2.13707     7901549738    1
nprobe=2048,quantizer_efSearch=512       0.7424 0.9949 0.9982      2.20448     7987089827    1
```

</details>
<details><summary>`PCAR128,IVF262144(IVF512,PQ64x4fs,RFlat),SQ4` </summary>
Index size 7347474457

 code_size 64

 log filename: autotune.dbbigann100M.PCAR128_IVF262144_IVF512_PQ64x4fs_RFlat__SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.4996 0.6174 0.6184      0.02186       55115724    14
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.1715 0.1970 0.1972      0.00308        5057081    98
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1963 0.2263 0.2265      0.00327        5019073    92
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.2035 0.2337 0.2339      0.00347        5692968    87
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=16    0.2064 0.2364 0.2366      0.00400        7017951    75
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.2136 0.2498 0.2504      0.00438        8895255    69
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2318 0.2702 0.2708      0.00431        9122576    70
nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2521 0.2924 0.2930      0.00432        9417846    70
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2843 0.3329 0.3335      0.00459        9351488    66
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.2975 0.3470 0.3476      0.00465       10022777    65
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.2999 0.3498 0.3504      0.00496       10012644    61
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=8     0.3000 0.3499 0.3505      0.00553       10012131    55
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.3044 0.3547 0.3553      0.00549       11325683    55
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.3050 0.3553 0.3559      0.00643       13954044    47
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3406 0.4078 0.4084      0.00717       18245796    42
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3725 0.4469 0.4477      0.00724       18129871    42
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3806 0.4582 0.4590      0.00737       18066974    41
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.3807 0.4583 0.4591      0.00792       18065060    38
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.3993 0.4775 0.4783      0.00831       20042986    37
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4058 0.4883 0.4891      0.00830       19988903    37
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4078 0.4901 0.4909      0.00824       19974522    37
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32    0.4090 0.4913 0.4921      0.01048       22577796    29
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.4264 0.5207 0.5218      0.01290       35860156    24
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.4534 0.5605 0.5616      0.01284       35439682    24
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8      0.4550 0.5540 0.5551      0.01322       36390357    23
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4614 0.5632 0.5643      0.01327       37607107    23
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4824 0.5934 0.5946      0.01373       36074063    22
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16     0.4971 0.6146 0.6156      0.01337       37224289    23
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16     0.4984 0.6161 0.6171      0.01359       37201185    23
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.4991 0.6166 0.6176      0.01430       39801168    21
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.5003 0.6182 0.6192      0.01446       39783824    21
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5349 0.6690 0.6704      0.02233       71420934    14
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5736 0.7250 0.7265      0.02353       71756209    13
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5785 0.7333 0.7350      0.02456       74094297    13
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=256   0.5789 0.7338 0.7355      0.03300      109929484    10
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.6414 0.8326 0.8346      0.04631      147411168    7
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6439 0.8370 0.8390      0.04686      147117959    7
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6442 0.8370 0.8390      0.04796      147086928    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6485 0.8427 0.8448      0.08747      274478717    4
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6773 0.8851 0.8873      0.08455      275090279    4
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6870 0.8990 0.9012      0.08592      276972107    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.6900 0.9031 0.9053      0.08444      276481653    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6916 0.9061 0.9083      0.08801      281301051    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.6919 0.9065 0.9087      0.09797      311862601    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128   0.6923 0.9069 0.9091      0.09017      291477164    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.7006 0.9249 0.9272      0.15912      541713734    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7174 0.9506 0.9530      0.16325      546716414    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7179 0.9510 0.9534      0.16697      556699521    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7183 0.9512 0.9536      0.16556      555784759    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.7184 0.9513 0.9537      0.17763      576131096    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.7331 0.9789 0.9815      0.32626     1075938709    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.7393 0.9907 0.9933      0.60609     2118331446    1
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.7400 0.9912 0.9938      0.63285     2096350967    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.7402 0.9940 0.9966      1.14965     4094955390    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.7416 0.9960 0.9986      1.17174     4097707941    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.7417 0.9961 0.9987      1.17477     4116727877    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.7420 0.9972 0.9998      2.27214     8026251556    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.7422 0.9974 1.0000      4.47912    16039467142    1
```

</details>
<details><summary>`PCAR128,IVF65536_HNSW32,SQ4` </summary>
Index size 7252050133

 code_size 64

 log filename: autotune.dbbigann100M.PCAR128_IVF65536_HNSW32_SQ4.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7366 0.9902 0.9939      1.08577     4103290343    1
nprobe=1,quantizer_efSearch=8            0.2551 0.2971 0.2978      0.00840       17106746    36
nprobe=1,quantizer_efSearch=16           0.2602 0.3026 0.3033      0.00697       17087479    44
nprobe=1,quantizer_efSearch=32           0.2608 0.3032 0.3039      0.00790       17073804    38
nprobe=2,quantizer_efSearch=4            0.3283 0.3946 0.3958      0.01099       34332580    28
nprobe=2,quantizer_efSearch=16           0.3644 0.4373 0.4386      0.01165       34213821    26
nprobe=2,quantizer_efSearch=32           0.3653 0.4384 0.4397      0.01262       34183061    24
nprobe=2,quantizer_efSearch=64           0.3654 0.4385 0.4398      0.01490       34179026    21
nprobe=2,quantizer_efSearch=128          0.3657 0.4388 0.4401      0.01783       34176896    17
nprobe=4,quantizer_efSearch=4            0.4145 0.5111 0.5129      0.02047       68554182    15
nprobe=4,quantizer_efSearch=8            0.4563 0.5626 0.5644      0.02094       68410007    15
nprobe=4,quantizer_efSearch=32           0.4685 0.5773 0.5791      0.02167       68257698    14
nprobe=4,quantizer_efSearch=64           0.4691 0.5778 0.5796      0.02337       68245186    13
nprobe=4,quantizer_efSearch=128          0.4695 0.5782 0.5800      0.02678       68238887    12
nprobe=8,quantizer_efSearch=8            0.5456 0.6906 0.6932      0.04374      136658647    7
nprobe=8,quantizer_efSearch=16           0.5640 0.7144 0.7170      0.03931      136467500    8
nprobe=8,quantizer_efSearch=32           0.5676 0.7190 0.7216      0.03896      136322104    8
nprobe=8,quantizer_efSearch=64           0.5688 0.7202 0.7228      0.04087      136295767    8
nprobe=8,quantizer_efSearch=128          0.5690 0.7205 0.7231      0.04461      136287008    7
nprobe=16,quantizer_efSearch=8           0.6184 0.7979 0.8007      0.08045      271725630    4
nprobe=16,quantizer_efSearch=16          0.6294 0.8141 0.8169      0.07272      271373428    5
nprobe=16,quantizer_efSearch=32          0.6369 0.8243 0.8271      0.07430      271044947    5
nprobe=16,quantizer_efSearch=64          0.6377 0.8251 0.8279      0.07542      270949174    4
nprobe=16,quantizer_efSearch=128         0.6385 0.8259 0.8287      0.07885      270908940    4
nprobe=32,quantizer_efSearch=16          0.6751 0.8857 0.8888      0.14139      537926067    3
nprobe=32,quantizer_efSearch=64          0.6837 0.9000 0.9031      0.14255      537000110    3
nprobe=32,quantizer_efSearch=128         0.6845 0.9009 0.9040      0.14866      536921279    3
nprobe=64,quantizer_efSearch=16          0.7005 0.9279 0.9311      0.27434     1064285430    2
nprobe=64,quantizer_efSearch=32          0.7102 0.9447 0.9480      0.27997     1062558235    2
nprobe=64,quantizer_efSearch=128         0.7143 0.9509 0.9543      0.28229     1061316035    2
nprobe=64,quantizer_efSearch=512         0.7144 0.9510 0.9544      0.31162     1061200175    1
nprobe=128,quantizer_efSearch=128        0.7296 0.9773 0.9810      0.53872     2089659632    1
nprobe=128,quantizer_efSearch=512        0.7297 0.9773 0.9810      0.56639     2089274716    1
nprobe=256,quantizer_efSearch=64         0.7346 0.9873 0.9910      1.05087     4111678110    1
nprobe=256,quantizer_efSearch=256        0.7366 0.9901 0.9938      1.05204     4103741654    1
nprobe=512,quantizer_efSearch=128        0.7379 0.9941 0.9979      2.05558     8059702736    1
nprobe=512,quantizer_efSearch=256        0.7382 0.9946 0.9984      2.06230     8050896452    1
nprobe=512,quantizer_efSearch=512        0.7383 0.9948 0.9986      2.05642     8048389529    1
nprobe=1024,quantizer_efSearch=256       0.7388 0.9957 0.9995      3.99919    15779250894    1
nprobe=1024,quantizer_efSearch=512       0.7390 0.9960 0.9998      4.03334    15769107391    1
nprobe=2048,quantizer_efSearch=512       0.7391 0.9961 0.9999      7.81858    30904943523    1
```

</details>
<details><summary>`PCAR32,IVF1048576_HNSW32,SQfp16` </summary>
Index size 7628051541

 code_size 64

 log filename: autotune.dbbigann100M.PCAR32_IVF1048576_HNSW32_SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2640 0.7026 0.9197      0.10291      272919057    3
nprobe=1,quantizer_efSearch=4            0.0927 0.1408 0.1416      0.00239        1144243    126
nprobe=2,quantizer_efSearch=4            0.1252 0.2122 0.2175      0.00252        2286472    120
nprobe=4,quantizer_efSearch=4            0.1506 0.2855 0.2970      0.00260        4563764    116
nprobe=8,quantizer_efSearch=4            0.1839 0.3956 0.4281      0.00324        9101084    93
nprobe=8,quantizer_efSearch=8            0.1888 0.4059 0.4405      0.00357        9098771    85
nprobe=16,quantizer_efSearch=4           0.2045 0.4671 0.5223      0.00403       18120041    75
nprobe=16,quantizer_efSearch=8           0.2172 0.4996 0.5634      0.00468       18106800    65
nprobe=16,quantizer_efSearch=16          0.2213 0.5088 0.5763      0.00568       18065669    53
nprobe=32,quantizer_efSearch=8           0.2278 0.5504 0.6481      0.00690       35897664    44
nprobe=32,quantizer_efSearch=16          0.2366 0.5760 0.6835      0.00757       35820651    40
nprobe=32,quantizer_efSearch=32          0.2391 0.5851 0.6956      0.00962       35738021    32
nprobe=64,quantizer_efSearch=16          0.2459 0.6205 0.7615      0.01129       70938617    27
nprobe=64,quantizer_efSearch=32          0.2508 0.6364 0.7861      0.01379       70725574    22
nprobe=64,quantizer_efSearch=64          0.2521 0.6427 0.7950      0.01772       70550693    17
nprobe=128,quantizer_efSearch=32         0.2568 0.6676 0.8505      0.01946      139706559    16
nprobe=128,quantizer_efSearch=64         0.2593 0.6776 0.8681      0.02478      139227277    13
nprobe=128,quantizer_efSearch=128        0.2606 0.6816 0.8731      0.03266      138956419    10
nprobe=256,quantizer_efSearch=64         0.2616 0.6948 0.9072      0.03805      274392850    8
nprobe=256,quantizer_efSearch=128        0.2637 0.7003 0.9166      0.04690      273499500    7
nprobe=256,quantizer_efSearch=256        0.2641 0.7024 0.9196      0.06271      273060110    5
nprobe=512,quantizer_efSearch=128        0.2647 0.7098 0.9438      0.07836      537829042    4
nprobe=512,quantizer_efSearch=256        0.2653 0.7125 0.9497      0.09135      536132500    4
nprobe=1024,quantizer_efSearch=256       0.2658 0.7153 0.9623      0.15975     1052251737    2
```

</details>
<details><summary>`PCAR32,IVF1048576(IVF1024,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 7359746969

 code_size 64

 log filename: autotune.dbbigann100M.PCAR32_IVF1048576_IVF1024_PQ16x4fs_RFlat__SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2633 0.6989 0.9127      0.06400      284996974    5
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.0913 0.1371 0.1381      0.00235        2591962    128
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1247 0.2109 0.2155      0.00262        3739298    115
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1389 0.2466 0.2547      0.00262        6051297    115
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1407 0.2574 0.2679      0.00264        5342269    114
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1512 0.2819 0.2934      0.00277        6041112    109
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1656 0.3321 0.3540      0.00294        9928798    103
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1703 0.3356 0.3562      0.00296       10641461    102
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.1871 0.3889 0.4198      0.00330       11982578    92
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.1888 0.3886 0.4180      0.00340       10601843    89
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.1890 0.3928 0.4225      0.00376       10583970    80
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.1949 0.4165 0.4503      0.00399       11940875    76
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1955 0.4184 0.4610      0.00429       19763354    70
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.2052 0.4498 0.5033      0.00463       19721998    65
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2099 0.4599 0.5180      0.00493       19674261    61
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2184 0.4958 0.5602      0.00518       21004099    58
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2240 0.5089 0.5764      0.00557       23678356    54
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.2246 0.5141 0.5824      0.00664       23633331    46
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=32   0.2252 0.5179 0.5868      0.00752       28875947    40
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2369 0.5707 0.6721      0.00795       41559870    38
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.2400 0.5819 0.6893      0.00899       41482058    34
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2408 0.5858 0.6948      0.01082       46662191    28
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2411 0.5869 0.6964      0.01115       57043151    27
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.2453 0.6028 0.7306      0.01255       77125636    24
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.2457 0.6076 0.7380      0.01326       82219772    23
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2512 0.6268 0.7732      0.01400       76761104    22
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.2517 0.6340 0.7820      0.01423       81875369    22
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.2539 0.6435 0.7959      0.01808       91957657    17
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2603 0.6749 0.8630      0.02508      150838985    12
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.2606 0.6767 0.8652      0.02636      160969218    12
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.2609 0.6805 0.8718      0.03507      180869073    9
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.2611 0.6877 0.8943      0.04241      287755125    8
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.2624 0.6963 0.9091      0.04521      285988369    7
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.2631 0.6985 0.9124      0.05106      285296299    6
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.2635 0.7022 0.9179      0.05964      355492861    6
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.2640 0.7030 0.9196      0.06940      314952336    5
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.2648 0.7123 0.9467      0.09009      579476566    4
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=512  0.2651 0.7124 0.9493      0.11702      700021533    3
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=128 0.2654 0.7158 0.9634      0.16345     1093640155    2
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.2655 0.7169 0.9678      0.28185     2108794670    2
```

</details>
<details><summary>`PCAR32,IVF262144_HNSW32,SQfp16` </summary>
Index size 7307077973

 code_size 64

 log filename: autotune.dbbigann100M.PCAR32_IVF262144_HNSW32_SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2652 0.7106 0.9530      0.11644     1048167160    3
nprobe=1,quantizer_efSearch=4            0.1153 0.1951 0.2008      0.00233        4332571    129
nprobe=2,quantizer_efSearch=4            0.1485 0.2776 0.2913      0.00251        8674589    120
nprobe=2,quantizer_efSearch=8            0.1612 0.3036 0.3195      0.00276        8657343    109
nprobe=4,quantizer_efSearch=4            0.1754 0.3621 0.3927      0.00289       17341977    104
nprobe=4,quantizer_efSearch=8            0.1940 0.4035 0.4391      0.00306       17329136    98
nprobe=4,quantizer_efSearch=16           0.1971 0.4105 0.4467      0.00356       17300203    85
nprobe=8,quantizer_efSearch=8            0.2176 0.4958 0.5620      0.00399       34579875    76
nprobe=8,quantizer_efSearch=16           0.2218 0.5079 0.5766      0.00449       34532601    67
nprobe=8,quantizer_efSearch=32           0.2229 0.5124 0.5815      0.00580       34494557    52
nprobe=16,quantizer_efSearch=8           0.2352 0.5715 0.6774      0.00678       68944319    45
nprobe=16,quantizer_efSearch=16          0.2382 0.5832 0.6929      0.00919       68858024    33
nprobe=16,quantizer_efSearch=32          0.2400 0.5884 0.7007      0.00862       68770103    35
nprobe=16,quantizer_efSearch=64          0.2410 0.5897 0.7020      0.01055       68737806    29
nprobe=32,quantizer_efSearch=8           0.2435 0.6179 0.7591      0.01225      136961330    25
nprobe=32,quantizer_efSearch=16          0.2492 0.6370 0.7885      0.01201      136740598    25
nprobe=32,quantizer_efSearch=32          0.2525 0.6447 0.8014      0.01267      136530750    24
nprobe=32,quantizer_efSearch=64          0.2531 0.6475 0.8057      0.01617      136422163    19
nprobe=32,quantizer_efSearch=128         0.2537 0.6485 0.8070      0.01845      136391362    17
nprobe=64,quantizer_efSearch=16          0.2541 0.6643 0.8494      0.02254      270856691    14
nprobe=64,quantizer_efSearch=32          0.2585 0.6764 0.8705      0.02297      270342659    14
nprobe=64,quantizer_efSearch=64          0.2596 0.6809 0.8771      0.02468      270053801    13
nprobe=64,quantizer_efSearch=128         0.2601 0.6819 0.8783      0.02759      269928680    11
nprobe=128,quantizer_efSearch=32         0.2612 0.6928 0.9128      0.04229      534266928    8
nprobe=128,quantizer_efSearch=64         0.2623 0.6994 0.9242      0.04465      533151938    7
nprobe=128,quantizer_efSearch=128        0.2628 0.7000 0.9273      0.04693      532651700    7
nprobe=256,quantizer_efSearch=64         0.2640 0.7066 0.9455      0.08375     1051084723    4
nprobe=256,quantizer_efSearch=128        0.2650 0.7100 0.9518      0.08760     1049006208    4
nprobe=256,quantizer_efSearch=256        0.2652 0.7109 0.9531      0.09323     1048309899    4
nprobe=512,quantizer_efSearch=256        0.2662 0.7146 0.9659      0.17127     2059895394    2
```

</details>
<details><summary>`PCAR32,IVF262144(IVF512,PQ16x4fs,RFlat),SQfp16` </summary>
Index size 7240067737

 code_size 64

 log filename: autotune.dbbigann100M.PCAR32_IVF262144_IVF512_PQ16x4fs_RFlat__SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.2483 0.6289 0.7819      0.02301      282204189    14
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=4    0.0770 0.1248 0.1288      0.00262        5093777    115
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.1208 0.2025 0.2087      0.00262        5054292    115
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8   0.1246 0.2079 0.2141      0.00262        5731535    115
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=4    0.1350 0.2392 0.2504      0.00268        9493474    112
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.1424 0.2657 0.2788      0.00282        9111358    107
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1563 0.2910 0.3058      0.00278        9437192    108
nprobe=2,quantizer_k_factor_rf=32,quantizer_nprobe=4   0.1592 0.2958 0.3115      0.00283        9407948    107
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1625 0.3034 0.3183      0.00298       11413093    101
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.1718 0.3458 0.3746      0.00317       17896203    95
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.1825 0.3736 0.4029      0.00322       18241173    94
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1885 0.3865 0.4190      0.00320       18192845    94
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.1898 0.3885 0.4217      0.00320       18140429    94
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1904 0.3929 0.4248      0.00333       20170812    91
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1984 0.4103 0.4444      0.00347       20119215    87
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4    0.2077 0.4592 0.5173      0.00419       35686915    72
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.2084 0.4553 0.5098      0.00404       36419978    75
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.2123 0.4688 0.5300      0.00459       35520935    66
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.2182 0.4991 0.5633      0.00462       37537289    65
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.2230 0.5064 0.5749      0.00484       37415429    62
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=16  0.2242 0.5103 0.5793      0.00495       37355293    61
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=32  0.2249 0.5120 0.5814      0.00568       39938551    53
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8   0.2257 0.5395 0.6306      0.00661       71214851    46
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16  0.2291 0.5534 0.6484      0.00698       72373094    44
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.2365 0.5750 0.6809      0.00681       72046936    45
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16  0.2398 0.5841 0.6942      0.00747       71852702    41
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.2414 0.5899 0.7013      0.01041       79431909    29
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8   0.2463 0.6182 0.7590      0.01218      139284556    25
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2467 0.6191 0.7596      0.01302      143328648    24
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2537 0.6439 0.7966      0.01402      147741611    22
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64  0.2541 0.6496 0.8061      0.01459      147353952    21
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.2544 0.6488 0.8058      0.01624      147209023    19
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.2572 0.6715 0.8618      0.02306      274752368    14
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.2579 0.6688 0.8489      0.02303      278366660    14
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32  0.2593 0.6792 0.8722      0.02351      276843281    13
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.2604 0.6811 0.8744      0.02464      281765492    13
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64 0.2622 0.6959 0.9138      0.04465      548323579    7
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32 0.2628 0.7039 0.9423      0.08358     1064820087    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.2645 0.7109 0.9525      0.08754     1062451239    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64 0.2649 0.7113 0.9537      0.08906     1061067488    4
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64 0.2660 0.7145 0.9659      0.16016     2075383306    2
```

</details>
<details><summary>`PCAR32,IVF65536_HNSW32,SQfp16` </summary>
Index size 7226833749

 code_size 64

 log filename: autotune.dbbigann100M.PCAR32_IVF65536_HNSW32_SQfp16.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1413 0.2690 0.2861      0.00474       17188388    64
nprobe=1,quantizer_efSearch=8            0.1403 0.2669 0.2839      0.00256       17206137    118
nprobe=1,quantizer_efSearch=32           0.1412 0.2687 0.2858      0.00371       17187852    81
nprobe=2,quantizer_efSearch=8            0.1782 0.3755 0.4123      0.00465       34447490    65
nprobe=2,quantizer_efSearch=16           0.1783 0.3781 0.4150      0.00433       34402391    70
nprobe=2,quantizer_efSearch=64           0.1789 0.3794 0.4163      0.00599       34380769    51
nprobe=4,quantizer_efSearch=4            0.1979 0.4438 0.5062      0.00615       69032777    49
nprobe=4,quantizer_efSearch=8            0.2093 0.4755 0.5453      0.00609       68953297    50
nprobe=4,quantizer_efSearch=16           0.2100 0.4807 0.5517      0.00656       68846102    46
nprobe=4,quantizer_efSearch=32           0.2109 0.4835 0.5544      0.00709       68807487    43
nprobe=4,quantizer_efSearch=64           0.2110 0.4839 0.5549      0.00802       68797981    38
nprobe=4,quantizer_efSearch=128          0.2111 0.4840 0.5550      0.01040       68794682    29
nprobe=8,quantizer_efSearch=8            0.2313 0.5618 0.6720      0.01092      137213048    28
nprobe=8,quantizer_efSearch=16           0.2341 0.5710 0.6847      0.01120      137035588    27
nprobe=8,quantizer_efSearch=32           0.2345 0.5744 0.6884      0.01212      136930476    25
nprobe=8,quantizer_efSearch=128          0.2346 0.5754 0.6892      0.01562      136899377    20
nprobe=16,quantizer_efSearch=4           0.2384 0.6013 0.7444      0.02022      272208369    15
nprobe=16,quantizer_efSearch=16          0.2486 0.6362 0.7921      0.02059      271818665    15
nprobe=16,quantizer_efSearch=32          0.2506 0.6431 0.8004      0.02132      271576546    15
nprobe=16,quantizer_efSearch=128         0.2507 0.6438 0.8011      0.02462      271504147    13
nprobe=32,quantizer_efSearch=32          0.2576 0.6775 0.8748      0.04390      537078249    7
nprobe=32,quantizer_efSearch=128         0.2580 0.6801 0.8775      0.04433      536774534    7
nprobe=32,quantizer_efSearch=256         0.2581 0.6802 0.8776      0.05101      536756893    6
nprobe=64,quantizer_efSearch=16          0.2590 0.6877 0.9079      0.08028     1061427705    4
nprobe=64,quantizer_efSearch=64          0.2613 0.6986 0.9258      0.07727     1059029705    4
nprobe=64,quantizer_efSearch=128         0.2615 0.6995 0.9271      0.08132     1058695823    4
nprobe=64,quantizer_efSearch=256         0.2616 0.6996 0.9273      0.08782     1058647413    4
nprobe=128,quantizer_efSearch=32         0.2634 0.7051 0.9452      0.14771     2086682210    3
nprobe=128,quantizer_efSearch=64         0.2637 0.7078 0.9523      0.14585     2083290166    3
nprobe=128,quantizer_efSearch=256        0.2638 0.7086 0.9536      0.15702     2081657722    2
nprobe=256,quantizer_efSearch=64         0.2641 0.7116 0.9643      0.28143     4088842402    2
nprobe=256,quantizer_efSearch=128        0.2644 0.7127 0.9668      0.28321     4082602162    2
nprobe=256,quantizer_efSearch=256        0.2645 0.7129 0.9673      0.29768     4080894561    2
nprobe=512,quantizer_efSearch=256        0.2646 0.7132 0.9707      0.55011     7990898225    1
```

</details>
<details><summary>`PCAR64,IVF1048576_HNSW32,SQ8` </summary>
Index size 7762286293

 code_size 64

 log filename: autotune.dbbigann100M.PCAR64_IVF1048576_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5920 0.9405 0.9481      0.12906      277580022    3
nprobe=2,quantizer_efSearch=4            0.1932 0.2286 0.2286      0.00299        2315383    101
nprobe=4,quantizer_efSearch=4            0.2544 0.3143 0.3143      0.00313        4627932    96
nprobe=4,quantizer_efSearch=8            0.2910 0.3595 0.3595      0.00397        4618361    76
nprobe=8,quantizer_efSearch=4            0.3469 0.4540 0.4541      0.00443        9243146    68
nprobe=8,quantizer_efSearch=8            0.3578 0.4701 0.4703      0.00477        9233433    63
nprobe=8,quantizer_efSearch=16           0.3732 0.4928 0.4930      0.00591        9191335    51
nprobe=16,quantizer_efSearch=4           0.3977 0.5430 0.5435      0.00565       18392524    54
nprobe=16,quantizer_efSearch=8           0.4227 0.5879 0.5887      0.00665       18370813    46
nprobe=16,quantizer_efSearch=16          0.4362 0.6064 0.6073      0.00802       18330609    38
nprobe=32,quantizer_efSearch=8           0.4635 0.6690 0.6708      0.01043       36499941    29
nprobe=32,quantizer_efSearch=16          0.4842 0.7050 0.7069      0.01120       36397649    27
nprobe=32,quantizer_efSearch=32          0.4942 0.7215 0.7234      0.01288       36273157    24
nprobe=64,quantizer_efSearch=16          0.5169 0.7759 0.7792      0.01602       72160775    19
nprobe=64,quantizer_efSearch=32          0.5327 0.8056 0.8088      0.01833       71875115    17
nprobe=64,quantizer_efSearch=64          0.5377 0.8176 0.8209      0.02475       71652045    13
nprobe=128,quantizer_efSearch=32         0.5566 0.8603 0.8652      0.02978      142190548    11
nprobe=128,quantizer_efSearch=64         0.5691 0.8848 0.8905      0.03940      141583530    8
nprobe=128,quantizer_efSearch=128        0.5734 0.8924 0.8981      0.04226      141225219    8
nprobe=128,quantizer_efSearch=256        0.5750 0.8951 0.9007      0.05932      141079208    6
nprobe=256,quantizer_efSearch=64         0.5829 0.9214 0.9289      0.05832      279460575    6
nprobe=256,quantizer_efSearch=128        0.5899 0.9359 0.9434      0.06288      278361232    5
nprobe=256,quantizer_efSearch=256        0.5920 0.9400 0.9476      0.08230      277776690    4
nprobe=512,quantizer_efSearch=128        0.5960 0.9577 0.9664      0.11306      548054817    3
nprobe=512,quantizer_efSearch=256        0.5993 0.9639 0.9727      0.13223      546056119    3
nprobe=512,quantizer_efSearch=512        0.5994 0.9648 0.9736      0.16570      545140744    2
nprobe=1024,quantizer_efSearch=256       0.6022 0.9775 0.9868      0.24111     1072655669    2
nprobe=1024,quantizer_efSearch=512       0.6026 0.9790 0.9886      0.29030     1069048387    2
nprobe=2048,quantizer_efSearch=256       0.6033 0.9811 0.9909      0.39298     2087246751    1
nprobe=2048,quantizer_efSearch=512       0.6037 0.9841 0.9944      0.44817     2094986905    1
nprobe=4096,quantizer_efSearch=512       0.6041 0.9856 0.9963      0.79365     4041554584    1
```

</details>
<details><summary>`PCAR64,IVF1048576(IVF1024,PQ32x4fs,RFlat),SQ8` </summary>
Index size 7502621977

 code_size 64

 log filename: autotune.dbbigann100M.PCAR64_IVF1048576_IVF1024_PQ32x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.2922 0.3671 0.3675      0.00371       10808866    81
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1356 0.1540 0.1540      0.00244        1891785    123
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.1392 0.1575 0.1575      0.00247        2603642    122
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1602 0.1854 0.1855      0.00248        2699029    121
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1634 0.1895 0.1896      0.00260        2696893    116
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1884 0.2211 0.2211      0.00266        3060989    113
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1954 0.2347 0.2348      0.00268        5040784    112
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.2104 0.2546 0.2546      0.00263        5422019    115
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2375 0.2885 0.2886      0.00277        5404298    109
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2514 0.3078 0.3079      0.00278        5392679    108
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.2567 0.3151 0.3152      0.00290        6109503    104
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2729 0.3373 0.3373      0.00308        6092389    98
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4     0.2797 0.3447 0.3447      0.00340        6078262    89
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2922 0.3671 0.3675      0.00373       10807676    81
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.2976 0.3680 0.3680      0.00391        7446604    77
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.3024 0.3892 0.3893      0.00386       10030323    78
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3249 0.4209 0.4212      0.00382       10754330    79
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.3646 0.4779 0.4780      0.00423       12068912    71
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=8     0.3704 0.4865 0.4867      0.00540       12042715    56
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.3714 0.4912 0.4913      0.00628       20069550    48
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.3806 0.5035 0.5040      0.00611       21441926    50
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.4251 0.5847 0.5853      0.00653       23941473    46
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.4326 0.5967 0.5974      0.00709       21219358    43
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4389 0.6076 0.6083      0.00751       23910241    40
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.4430 0.6158 0.6167      0.00815       23881795    37
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4446 0.6189 0.6197      0.00892       29145537    34
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4712 0.6703 0.6716      0.01033       39499045    30
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4947 0.7192 0.7210      0.01142       41980374    27
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.4985 0.7277 0.7296      0.01277       47172510    24
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4989 0.7304 0.7324      0.01355       47151873    23
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.4990 0.7287 0.7307      0.01534       57549484    20
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5020 0.7361 0.7378      0.01637       78714307    19
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5284 0.7911 0.7938      0.01827       77855067    17
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5327 0.8023 0.8052      0.01965       82958851    16
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5384 0.8164 0.8195      0.02096       82738588    15
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.5417 0.8236 0.8269      0.02487       92939700    13
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.5421 0.8242 0.8275      0.03084       92720941    10
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5682 0.8868 0.8918      0.03590      152721686    9
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5709 0.8934 0.8985      0.03703      162742114    9
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.5728 0.8961 0.9014      0.04591      182881558    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5731 0.8963 0.9016      0.05214      223615867    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5803 0.9201 0.9268      0.06002      291168463    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5848 0.9305 0.9374      0.06021      300849157    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5855 0.9293 0.9360      0.06166      290072187    5
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.5857 0.9295 0.9363      0.07170      290025971    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.5897 0.9394 0.9465      0.07269      359502966    5
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=256  0.5903 0.9414 0.9485      0.08859      360334007    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5915 0.9481 0.9558      0.10001      561493068    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5981 0.9660 0.9745      0.13078      587932930    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.5985 0.9662 0.9747      0.13560      627428244    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256 0.6016 0.9799 0.9889      0.22126     1154782054    2
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=256 0.6018 0.9822 0.9914      0.36092     2226590448    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.6024 0.9870 0.9968      0.41139     2171925703    1
nprobe=4096,quantizer_k_factor_rf=1,quantizer_nprobe=512 0.6025 0.9867 0.9965      0.67601     4352520793    1
nprobe=4096,quantizer_k_factor_rf=4,quantizer_nprobe=256 0.6031 0.9890 0.9995      0.78071     4161036976    1
```

</details>
<details><summary>`PCAR64,IVF262144_HNSW32,SQ8` </summary>
Index size 7340649429

 code_size 64

 log filename: autotune.dbbigann100M.PCAR64_IVF262144_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.6011 0.9707 0.9800      0.18534     1055139019    2
nprobe=1,quantizer_efSearch=4            0.1837 0.2170 0.2170      0.00244        4334590    123
nprobe=2,quantizer_efSearch=4            0.2552 0.3160 0.3161      0.00276        8686264    109
nprobe=2,quantizer_efSearch=8            0.2773 0.3424 0.3425      0.00320        8654188    94
nprobe=4,quantizer_efSearch=4            0.3222 0.4203 0.4206      0.00394       17361561    77
nprobe=4,quantizer_efSearch=16           0.3668 0.4806 0.4809      0.00526       17286696    58
nprobe=8,quantizer_efSearch=8            0.4266 0.5900 0.5907      0.00678       34577342    45
nprobe=8,quantizer_efSearch=16           0.4409 0.6134 0.6141      0.00752       34498532    40
nprobe=8,quantizer_efSearch=32           0.4459 0.6199 0.6206      0.00889       34447773    34
nprobe=16,quantizer_efSearch=8           0.4855 0.7021 0.7036      0.01174       69049711    26
nprobe=16,quantizer_efSearch=16          0.4972 0.7234 0.7250      0.01213       68938391    25
nprobe=16,quantizer_efSearch=32          0.5027 0.7321 0.7341      0.01352       68835505    23
nprobe=16,quantizer_efSearch=64          0.5054 0.7357 0.7377      0.01628       68796565    19
nprobe=32,quantizer_efSearch=8           0.5164 0.7751 0.7775      0.03350      137374992    9
nprobe=32,quantizer_efSearch=16          0.5405 0.8170 0.8196      0.02184      137127417    14
nprobe=32,quantizer_efSearch=32          0.5468 0.8285 0.8317      0.02235      136909830    14
nprobe=32,quantizer_efSearch=64          0.5490 0.8322 0.8355      0.02546      136791450    12
nprobe=32,quantizer_efSearch=128         0.5494 0.8328 0.8361      0.03033      136733982    10
nprobe=32,quantizer_efSearch=256         0.5498 0.8333 0.8366      0.04024      136723473    8
nprobe=64,quantizer_efSearch=32          0.5727 0.8928 0.8981      0.04337      271486633    7
nprobe=64,quantizer_efSearch=64          0.5765 0.9002 0.9058      0.04272      271111784    8
nprobe=64,quantizer_efSearch=256         0.5770 0.9015 0.9071      0.05800      270914940    6
nprobe=128,quantizer_efSearch=32         0.5854 0.9285 0.9357      0.07581      537243800    4
nprobe=128,quantizer_efSearch=64         0.5906 0.9402 0.9480      0.07775      536111861    4
nprobe=128,quantizer_efSearch=128        0.5913 0.9440 0.9517      0.08441      535523288    4
nprobe=128,quantizer_efSearch=256        0.5920 0.9449 0.9526      0.09657      535354694    4
nprobe=256,quantizer_efSearch=64         0.5981 0.9624 0.9714      0.14503     1058238599    3
nprobe=256,quantizer_efSearch=128        0.5999 0.9692 0.9784      0.15084     1056096238    2
nprobe=256,quantizer_efSearch=256        0.6009 0.9703 0.9796      0.16045     1055319733    2
nprobe=256,quantizer_efSearch=512        0.6011 0.9707 0.9800      0.18787     1055139019    2
nprobe=512,quantizer_efSearch=128        0.6028 0.9794 0.9893      0.28471     2079283942    2
nprobe=512,quantizer_efSearch=256        0.6043 0.9812 0.9914      0.29831     2075820868    2
nprobe=1024,quantizer_efSearch=256       0.6055 0.9858 0.9962      0.55580     4076584214    1
nprobe=2048,quantizer_efSearch=256       0.6058 0.9869 0.9975      1.04139     7886774157    1
```

</details>
<details><summary>`PCAR64,IVF262144(IVF512,PQ32x4fs,RFlat),SQ8` </summary>
Index size 7275871001

 code_size 64

 log filename: autotune.dbbigann100M.PCAR64_IVF262144_IVF512_PQ32x4fs_RFlat__SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                          0.5558 0.8496 0.8538      0.04095      281921011    8
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1       0.1469 0.1723 0.1723      0.00254        4545267    119
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.1941 0.2276 0.2276      0.00263        5026754    115
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=8      0.1996 0.2343 0.2343      0.00274        5700486    110
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.2604 0.3185 0.3185      0.00299       10088521    101
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=8       0.2770 0.3397 0.3397      0.00300       10057352    100
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.2835 0.3482 0.3482      0.00345       11357193    87
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.2849 0.3494 0.3494      0.00418       13981146    72
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4       0.3358 0.4318 0.4320      0.00418       18175218    72
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4      0.3480 0.4507 0.4509      0.00473       18109301    64
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.3681 0.4797 0.4799      0.00496       20035582    61
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=32      0.3682 0.4793 0.4795      0.00532       22661328    57
nprobe=4,quantizer_k_factor_rf=32,quantizer_nprobe=32     0.3704 0.4842 0.4844      0.00625       22632132    48
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=8       0.3862 0.5165 0.5168      0.00664       36489922    46
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4       0.4130 0.5590 0.5594      0.00688       35532289    44
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.4228 0.5795 0.5800      0.00716       36216462    42
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=32      0.4474 0.6210 0.6216      0.00827       39846552    37
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4593 0.6491 0.6501      0.01199       72632860    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=16     0.4958 0.7097 0.7113      0.01215       72042505    25
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.4979 0.7147 0.7163      0.01272       74555980    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5050 0.7314 0.7334      0.01303       74358046    24
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.5058 0.7330 0.7350      0.01406       79495812    22
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.5071 0.7366 0.7387      0.01557       79393085    20
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.5205 0.7735 0.7767      0.02203      139709160    14
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8      0.5247 0.7835 0.7867      0.02292      139339299    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5449 0.8197 0.8235      0.02250      142898668    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64     0.5457 0.8210 0.8247      0.02357      147925354    13
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.5490 0.8323 0.8360      0.02388      147568870    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.5504 0.8323 0.8360      0.02419      142395827    13
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64    0.5511 0.8336 0.8374      0.02653      147447032    12
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64    0.5512 0.8337 0.8375      0.02935      147415101    11
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.5520 0.8389 0.8429      0.04097      280127840    8
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.5558 0.8496 0.8538      0.04127      281921011    8
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32     0.5714 0.8875 0.8938      0.04084      278005151    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.5753 0.8970 0.9032      0.04150      277158308    8
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64     0.5775 0.9005 0.9067      0.04295      282021379    7
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64     0.5779 0.9011 0.9073      0.04493      281818590    7
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128   0.5782 0.9015 0.9077      0.04977      292012845    7
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.5796 0.9156 0.9229      0.07712      543291749    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64    0.5895 0.9402 0.9486      0.07899      548487113    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5915 0.9445 0.9524      0.08063      546990417    4
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64   0.5927 0.9453 0.9533      0.08839      546756193    4
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.5992 0.9638 0.9733      0.14790     1065067085    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=256   0.5998 0.9678 0.9771      0.15793     1099743395    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.6020 0.9708 0.9803      0.15182     1067852705    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6021 0.9711 0.9807      0.16134     1067608140    2
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=256   0.6060 0.9829 0.9933      0.30219     2117447586    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.6069 0.9864 0.9969      0.53798     4097388386    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=256  0.6070 0.9875 0.9981      0.53964     4120512571    1
nprobe=1024,quantizer_k_factor_rf=16,quantizer_nprobe=128 0.6071 0.9875 0.9982      0.61307     4094476494    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.6073 0.9888 0.9998      1.06475     8017184495    1
```

</details>
<details><summary>`PCAR64,IVF65536_HNSW32,SQ8` </summary>
Index size 7235239381

 code_size 64

 log filename: autotune.dbbigann100M.PCAR64_IVF65536_HNSW32_SQ8.c.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.2425 0.3043 0.3044      0.00632       17089156    48
nprobe=1,quantizer_efSearch=8            0.2398 0.3008 0.3009      0.00401       17124957    75
nprobe=1,quantizer_efSearch=16           0.2416 0.3032 0.3033      0.00442       17099670    68
nprobe=1,quantizer_efSearch=32           0.2422 0.3040 0.3041      0.00507       17091164    60
nprobe=1,quantizer_efSearch=64           0.2425 0.3043 0.3044      0.00638       17089156    48
nprobe=2,quantizer_efSearch=8            0.3271 0.4318 0.4320      0.00668       34301029    45
nprobe=2,quantizer_efSearch=16           0.3309 0.4371 0.4373      0.00696       34252543    44
nprobe=4,quantizer_efSearch=4            0.3761 0.5158 0.5166      0.01130       68772144    27
nprobe=4,quantizer_efSearch=8            0.4054 0.5622 0.5634      0.01160       68655487    26
nprobe=4,quantizer_efSearch=32           0.4131 0.5748 0.5759      0.01318       68497885    23
nprobe=4,quantizer_efSearch=64           0.4138 0.5756 0.5767      0.01376       68479117    22
nprobe=4,quantizer_efSearch=128          0.4139 0.5757 0.5768      0.01643       68475355    19
nprobe=8,quantizer_efSearch=4            0.4660 0.6752 0.6770      0.02140      137175732    15
nprobe=8,quantizer_efSearch=64           0.4881 0.7162 0.7181      0.02349      136707561    13
nprobe=8,quantizer_efSearch=128          0.4882 0.7163 0.7182      0.02610      136694216    12
nprobe=16,quantizer_efSearch=4           0.5083 0.7578 0.7610      0.04025      272606011    8
nprobe=16,quantizer_efSearch=8           0.5288 0.7990 0.8027      0.04026      272338813    8
nprobe=16,quantizer_efSearch=16          0.5353 0.8123 0.8163      0.04060      272021886    8
nprobe=16,quantizer_efSearch=32          0.5399 0.8195 0.8235      0.04102      271718174    8
nprobe=16,quantizer_efSearch=128         0.5418 0.8212 0.8253      0.04513      271587944    7
nprobe=16,quantizer_efSearch=512         0.5419 0.8213 0.8254      0.07173      271579543    5
nprobe=32,quantizer_efSearch=8           0.5513 0.8563 0.8618      0.07740      539962227    4
nprobe=32,quantizer_efSearch=16          0.5679 0.8844 0.8904      0.07755      539120060    4
nprobe=32,quantizer_efSearch=32          0.5711 0.8932 0.8993      0.07835      538414918    4
nprobe=32,quantizer_efSearch=64          0.5731 0.8961 0.9022      0.07954      538166236    4
nprobe=32,quantizer_efSearch=128         0.5739 0.8972 0.9033      0.08244      538082119    4
nprobe=64,quantizer_efSearch=32          0.5891 0.9404 0.9486      0.14723     1063880623    3
nprobe=64,quantizer_efSearch=128         0.5922 0.9462 0.9543      0.15091     1062632335    2
nprobe=64,quantizer_efSearch=256         0.5924 0.9464 0.9545      0.15842     1062555755    2
nprobe=128,quantizer_efSearch=128        0.6008 0.9715 0.9806      0.29220     2091079170    2
nprobe=128,quantizer_efSearch=512        0.6012 0.9720 0.9811      0.31616     2090753683    1
nprobe=256,quantizer_efSearch=64         0.6027 0.9792 0.9892      0.54785     4112711894    1
nprobe=256,quantizer_efSearch=128        0.6042 0.9829 0.9930      0.54912     4106451595    1
nprobe=256,quantizer_efSearch=256        0.6045 0.9836 0.9937      0.55068     4104582788    1
nprobe=512,quantizer_efSearch=128        0.6052 0.9867 0.9970      1.05843     8056629365    1
nprobe=512,quantizer_efSearch=256        0.6057 0.9880 0.9984      1.07284     8047119145    1
nprobe=1024,quantizer_efSearch=256       0.6058 0.9890 0.9995      2.05673    15765199261    1
```

</details>
