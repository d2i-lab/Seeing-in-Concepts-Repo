# Detailed logs for dataset bigann1B

## Code sizes in [0, 8]

<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fsr` </summary>
Index size 16700657356

 code_size 8

 log filename: autotune.dbbigann1B.OPQ16_64_IVF1048576_HNSW32_PQ16x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1433 0.3886 0.5524      0.01126       91583985    27
nprobe=1,quantizer_efSearch=4            0.0750 0.1606 0.1885      0.00245       11603106    123
nprobe=2,quantizer_efSearch=4            0.0924 0.2169 0.2731      0.00287       23173417    105
nprobe=4,quantizer_efSearch=4            0.1074 0.2679 0.3601      0.00354       46272432    85
nprobe=4,quantizer_efSearch=8            0.1244 0.3152 0.4194      0.00465       46139905    65
nprobe=8,quantizer_efSearch=4            0.1310 0.3495 0.4951      0.00543       92386569    56
nprobe=8,quantizer_efSearch=8            0.1353 0.3632 0.5156      0.00590       92280392    51
nprobe=16,quantizer_efSearch=4           0.1356 0.3808 0.5699      0.00769      183900262    40
nprobe=8,quantizer_efSearch=16           0.1410 0.3834 0.5460      0.00792       91886022    38
nprobe=16,quantizer_efSearch=8           0.1442 0.4096 0.6205      0.00868      183593506    35
nprobe=16,quantizer_efSearch=16          0.1481 0.4248 0.6463      0.01006      183121724    30
nprobe=16,quantizer_efSearch=32          0.1534 0.4345 0.6618      0.01368      182478997    22
nprobe=32,quantizer_efSearch=32          0.1566 0.4595 0.7348      0.01744      362614188    18
nprobe=32,quantizer_efSearch=64          0.1584 0.4651 0.7436      0.02403      361660982    13
nprobe=64,quantizer_efSearch=32          0.1600 0.4746 0.7871      0.02973      718334845    11
nprobe=64,quantizer_efSearch=64          0.1611 0.4799 0.7992      0.03390      716078174    9
nprobe=128,quantizer_efSearch=128        0.1630 0.4852 0.8330      0.06475     1411187953    5
```

</details>
<details><summary>`OPQ16_64,IVF4194304_HNSW32,PQ16x4fsr` </summary>
Index size 18804815820

 code_size 8

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_HNSW32_PQ16x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1627 0.4918 0.7946      0.08004      756844152    4
nprobe=2,quantizer_efSearch=4            0.0783 0.1684 0.1985      0.00310        6268900    97
nprobe=4,quantizer_efSearch=4            0.0930 0.2177 0.2687      0.00362       12520986    83
nprobe=4,quantizer_efSearch=8            0.1099 0.2638 0.3241      0.00491       12531705    62
nprobe=8,quantizer_efSearch=4            0.1194 0.3109 0.4075      0.00531       25012935    57
nprobe=8,quantizer_efSearch=8            0.1234 0.3217 0.4224      0.00579       25002093    52
nprobe=8,quantizer_efSearch=16           0.1318 0.3486 0.4588      0.00807       24862175    38
nprobe=16,quantizer_efSearch=8           0.1389 0.3853 0.5328      0.00820       49652096    37
nprobe=16,quantizer_efSearch=16          0.1440 0.4030 0.5615      0.00967       49498977    32
nprobe=16,quantizer_efSearch=32          0.1479 0.4176 0.5839      0.01386       49213156    22
nprobe=32,quantizer_efSearch=16          0.1518 0.4408 0.6483      0.01337       98298913    23
nprobe=32,quantizer_efSearch=32          0.1557 0.4548 0.6720      0.01683       97847310    18
nprobe=32,quantizer_efSearch=64          0.1573 0.4623 0.6835      0.02780       97453920    11
nprobe=64,quantizer_efSearch=32          0.1601 0.4780 0.7396      0.02920      194011823    11
nprobe=64,quantizer_efSearch=64          0.1616 0.4870 0.7554      0.04029      193057255    8
nprobe=128,quantizer_efSearch=64         0.1618 0.5008 0.8053      0.05497      381681818    6
nprobe=128,quantizer_efSearch=128        0.1621 0.5041 0.8143      0.07595      379930017    4
nprobe=256,quantizer_efSearch=32         0.1627 0.4918 0.7946      0.09798      756844152    4
nprobe=256,quantizer_efSearch=64         0.1646 0.5062 0.8273      0.09193      753791303    4
nprobe=512,quantizer_efSearch=128        0.1648 0.5115 0.8557      0.18726     1476185825    2
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16x4fs` </summary>
Index size 17764992016

 code_size 8

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16x4fs.a.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                       0.0336 0.1049 0.1906      0.00213        9135271    141
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1    0.0249 0.0691 0.1103      0.00148        4562657    204
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1    0.0258 0.0724 0.1154      0.00151        4557456    200
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1    0.0269 0.0820 0.1464      0.00165        7692332    182
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0285 0.0828 0.1296      0.00180        6000236    167
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2    0.0299 0.0866 0.1366      0.00184        5990750    163
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0336 0.1049 0.1906      0.00205        9137711    147
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2    0.0349 0.1247 0.2478      0.00266       15361261    113
nprobe=2,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.0377 0.1185 0.2146      0.00301       11893826    100
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4   0.0406 0.1411 0.2847      0.00447       18097962    68
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=4   0.0411 0.1509 0.3635      0.00583       55199651    52
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.0432 0.1765 0.4525      0.05025      422753054    6
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16x4fsr` </summary>
Index size 17764565008

 code_size 8

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                        0.1448 0.3845 0.5291      0.02907       54886352    11
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1     0.0454 0.0859 0.0941      0.00370        4555972    82
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.0697 0.1448 0.1648      0.00476        7670485    64
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.0851 0.1825 0.2096      0.00725        9146699    42
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1     0.0884 0.2012 0.2434      0.00823       13850108    37
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2     0.1010 0.2292 0.2745      0.00902       15377772    34
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2     0.1042 0.2423 0.2935      0.00957       15361269    32
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2     0.1050 0.2456 0.2984      0.01104       15357119    28
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.1099 0.2547 0.3041      0.01340       18115343    23
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1126 0.2678 0.3245      0.01401       18104509    22
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.1177 0.2833 0.3510      0.01644       30454401    19
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2    0.1217 0.3003 0.3849      0.01829       27694023    17
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4     0.1333 0.3347 0.4317      0.01909       30448932    16
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.1345 0.3389 0.4384      0.02118       30467520    15
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4    0.1448 0.3845 0.5291      0.02935       54922925    11
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8    0.1451 0.3880 0.5359      0.03128       60355877    10
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8    0.1501 0.4079 0.5659      0.03678       60241507    9
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.1520 0.4155 0.5777      0.03905       70942186    8
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16   0.1526 0.4211 0.5853      0.04105       70901689    8
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8    0.1560 0.4412 0.6474      0.04767      108847617    7
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1580 0.4481 0.6569      0.05183      139229662    6
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.1604 0.4615 0.6854      0.06108      180112600    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.1626 0.4803 0.7456      0.07001      233866866    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.1627 0.4758 0.7359      0.06592      215086308    5
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.1647 0.4862 0.7608      0.07636      274665077    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.1653 0.4870 0.7636      0.08096      234012632    4
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128  0.1659 0.4880 0.7652      0.10132      357324710    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.1664 0.4966 0.8025      0.11215      422263378    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16  0.1672 0.4944 0.7967      0.11786      402272771    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64 0.1684 0.5025 0.8170      0.17972      462584270    2
```

</details>
<details><summary>`OPQ8_64,IVF1048576_HNSW32,PQ8` </summary>
Index size 16562284464

 code_size 8

 log filename: autotune.dbbigann1B.OPQ8_64_IVF1048576_HNSW32_PQ8.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1310 0.3381 0.4250      0.01086       45914493    28
nprobe=1,quantizer_efSearch=4,ht=6       0.0014 0.0023 0.0031      0.00305       11580467    99
nprobe=1,quantizer_efSearch=4,ht=12      0.0070 0.0109 0.0133      0.00301       11580467    100
nprobe=1,quantizer_efSearch=4,ht=18      0.0293 0.0437 0.0494      0.00310       11580467    97
nprobe=1,quantizer_efSearch=4,ht=20      0.0415 0.0647 0.0715      0.00319       11580467    94
nprobe=1,quantizer_efSearch=8,ht=20      0.0469 0.0715 0.0785      0.00404       11541874    75
nprobe=1,quantizer_efSearch=8,ht=24      0.0751 0.1299 0.1413      0.00427       11541874    71
nprobe=1,quantizer_efSearch=4,ht=32      0.0785 0.1612 0.1842      0.00417       11580467    72
nprobe=2,quantizer_efSearch=4,ht=26      0.0878 0.1933 0.2209      0.00439       23122298    69
nprobe=4,quantizer_efSearch=4,ht=26      0.1032 0.2460 0.2955      0.00650       46197231    47
nprobe=4,quantizer_efSearch=8,ht=30      0.1264 0.3242 0.4087      0.00889       46123327    34
nprobe=4,quantizer_efSearch=8,ht=32      0.1270 0.3282 0.4181      0.00959       46123327    32
nprobe=4,quantizer_efSearch=16,ht=28     0.1281 0.3263 0.4015      0.01006       45914493    30
nprobe=8,quantizer_efSearch=4,ht=64      0.1363 0.3706 0.5021      0.01092       92370720    28
nprobe=8,quantizer_efSearch=16,ht=64     0.1461 0.4084 0.5538      0.01327       91900110    23
nprobe=8,quantizer_efSearch=64,ht=32     0.1486 0.4138 0.5574      0.02672       91405095    12
nprobe=16,quantizer_efSearch=16,ht=32    0.1530 0.4513 0.6480      0.02791      183161790    11
nprobe=16,quantizer_efSearch=32,ht=28    0.1537 0.4484 0.6193      0.02637      182543724    12
nprobe=16,quantizer_efSearch=32,ht=32    0.1560 0.4625 0.6631      0.03121      182543724    10
nprobe=32,quantizer_efSearch=64,ht=64    0.1640 0.5015 0.7596      0.04328      361605021    7
nprobe=64,quantizer_efSearch=32,ht=64    0.1647 0.5139 0.8042      0.06600      718311663    5
nprobe=64,quantizer_efSearch=64,ht=30    0.1663 0.5165 0.7962      0.08771      715952776    4
nprobe=64,quantizer_efSearch=256,ht=32   0.1665 0.5204 0.8135      0.12671      714374476    3
nprobe=128,quantizer_efSearch=128,ht=30  0.1686 0.5261 0.8335      0.15048     1410854778    2
```

</details>
<details><summary>`OPQ8_64,IVF4194304_HNSW32,PQ8` </summary>
Index size 18248830384

 code_size 8

 log filename: autotune.dbbigann1B.OPQ8_64_IVF4194304_HNSW32_PQ8.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.1728 0.4987 0.6983      0.07607       97081392    4
nprobe=1,quantizer_efSearch=4,ht=12      0.0025 0.0050 0.0066      0.00364        3130659    83
nprobe=1,quantizer_efSearch=4,ht=20      0.0265 0.0400 0.0440      0.00354        3130659    85
nprobe=1,quantizer_efSearch=4,ht=32      0.0613 0.1179 0.1291      0.00412        3130659    73
nprobe=2,quantizer_efSearch=4,ht=26      0.0764 0.1428 0.1555      0.00454        6258662    67
nprobe=2,quantizer_efSearch=4,ht=28      0.0803 0.1621 0.1781      0.00470        6258662    64
nprobe=2,quantizer_efSearch=4,ht=30      0.0832 0.1724 0.1910      0.00490        6258662    62
nprobe=2,quantizer_efSearch=8,ht=26      0.0880 0.1653 0.1790      0.00558        6257062    54
nprobe=4,quantizer_efSearch=4,ht=26      0.0931 0.1852 0.2082      0.00607       12512880    50
nprobe=2,quantizer_efSearch=8,ht=30      0.0954 0.2001 0.2219      0.00595        6257062    51
nprobe=2,quantizer_efSearch=8,ht=32      0.0963 0.2068 0.2305      0.00613        6257062    49
nprobe=4,quantizer_efSearch=8,ht=24      0.0985 0.1810 0.1995      0.00707       12518544    43
nprobe=4,quantizer_efSearch=8,ht=26      0.1099 0.2236 0.2493      0.00729       12518544    42
nprobe=4,quantizer_efSearch=8,ht=30      0.1168 0.2675 0.3095      0.00782       12518544    39
nprobe=4,quantizer_efSearch=8,ht=32      0.1175 0.2743 0.3222      0.00806       12518544    38
nprobe=8,quantizer_efSearch=4,ht=64      0.1329 0.3329 0.4125      0.00805       25000342    38
nprobe=8,quantizer_efSearch=16,ht=64     0.1458 0.3704 0.4608      0.01087       24817587    28
nprobe=8,quantizer_efSearch=32,ht=32     0.1473 0.3753 0.4647      0.01822       24706846    17
nprobe=16,quantizer_efSearch=16,ht=26    0.1493 0.3643 0.4355      0.01956       49480022    16
nprobe=16,quantizer_efSearch=8,ht=32     0.1535 0.4112 0.5358      0.01983       49629640    16
nprobe=32,quantizer_efSearch=8,ht=64     0.1584 0.4432 0.6109      0.02060       98639256    15
nprobe=16,quantizer_efSearch=32,ht=32    0.1611 0.4440 0.5807      0.02560       49210483    12
nprobe=32,quantizer_efSearch=64,ht=64    0.1723 0.4956 0.6940      0.03392       97400734    9
nprobe=64,quantizer_efSearch=32,ht=64    0.1746 0.5165 0.7574      0.04160      193922556    8
nprobe=64,quantizer_efSearch=64,ht=30    0.1758 0.5172 0.7410      0.07330      192952875    5
nprobe=128,quantizer_efSearch=256,ht=64  0.1803 0.5506 0.8404      0.11881      378781954    3
nprobe=256,quantizer_efSearch=256,ht=64  0.1819 0.5593 0.8746      0.16711      745604425    2
nprobe=1024,quantizer_efSearch=512,ht=64 0.1823 0.5642 0.8995      0.52515     2866248266    1
```

</details>
<details><summary>`OPQ8_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ8` </summary>
Index size 17208594420

 code_size 8

 log filename: autotune.dbbigann1B.OPQ8_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ8.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                               0.1048 0.2332 0.2677      0.00537       15385250    56
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=32      0.0489 0.0838 0.0895      0.00329        4544473    92
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=26      0.0554 0.0879 0.0919      0.00320        4544828    94
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=1,ht=32      0.0626 0.1107 0.1190      0.00349        4543276    87
nprobe=2,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=28      0.0883 0.1665 0.1804      0.00403        9146398    75
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=64      0.1048 0.2332 0.2677      0.00453       15384315    67
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64      0.1202 0.2800 0.3291      0.00647       18088026    47
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=64     0.1275 0.3142 0.3829      0.00789       27696607    39
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=30     0.1328 0.3350 0.4151      0.01565       52388866    20
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=64     0.1484 0.4086 0.5301      0.01259       54959738    24
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=64     0.1585 0.4359 0.5688      0.01505       60392775    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=26     0.1605 0.4109 0.5060      0.03140      109065487    10
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=64   0.1735 0.4951 0.6867      0.02990      119293321    11
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=32   0.1741 0.4972 0.6868      0.05983      140249937    6
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16,ht=64   0.1759 0.5181 0.7544      0.06740      406058673    5
nprobe=64,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=64   0.1793 0.5309 0.7755      0.06733      235334900    5
nprobe=64,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=64  0.1794 0.5325 0.7782      0.12208      358946876    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=64   0.1813 0.5444 0.8413      0.11811      770774409    3
nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=64   0.1816 0.5481 0.8569      0.26548     1490313639    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=64   0.1820 0.5558 0.8772      0.25835     1513459168    2
nprobe=256,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=32 0.1822 0.5553 0.8653      0.36354      911556656    1
nprobe=512,quantizer_k_factor_rf=32,quantizer_nprobe=512,ht=64 0.1824 0.5608 0.8903      0.47655     2114698184    1
nprobe=512,quantizer_k_factor_rf=64,quantizer_nprobe=256,ht=32 0.1826 0.5591 0.8829      0.86445     1789855940    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=32 0.1830 0.5603 0.8917      0.94132     3189802692    1
```

</details>

## Code sizes in [9, 16]

<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16` </summary>
Index size 24562284464

 code_size 16

 log filename: autotune.dbbigann1B.OPQ16_64_IVF1048576_HNSW32_PQ16.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3412 0.7121 0.7752      0.06499      363939544    5
nprobe=1,quantizer_efSearch=4,ht=28       0.0056 0.0084 0.0088      0.00341       11581542    88
nprobe=1,quantizer_efSearch=4,ht=40       0.0384 0.0520 0.0534      0.00350       11581542    86
nprobe=1,quantizer_efSearch=8,ht=44       0.0698 0.0943 0.0960      0.00444       11543058    68
nprobe=1,quantizer_efSearch=4,ht=54       0.1154 0.1681 0.1711      0.00387       11581542    78
nprobe=2,quantizer_efSearch=4,ht=50       0.1409 0.2109 0.2163      0.00496       23177212    61
nprobe=2,quantizer_efSearch=8,ht=48       0.1416 0.2050 0.2097      0.00589       23113260    51
nprobe=4,quantizer_efSearch=4,ht=48       0.1598 0.2442 0.2524      0.00753       46301668    40
nprobe=4,quantizer_efSearch=8,ht=46       0.1648 0.2403 0.2477      0.00845       46222749    36
nprobe=4,quantizer_efSearch=16,ht=48      0.1937 0.2936 0.3022      0.01058       45975234    29
nprobe=4,quantizer_efSearch=16,ht=50      0.2144 0.3390 0.3490      0.01078       45975234    28
nprobe=4,quantizer_efSearch=16,ht=52      0.2292 0.3751 0.3867      0.01108       45975234    28
nprobe=4,quantizer_efSearch=16,ht=54      0.2368 0.3994 0.4126      0.01143       45975234    27
nprobe=4,quantizer_efSearch=32,ht=56      0.2448 0.4205 0.4356      0.01537       45855417    20
nprobe=8,quantizer_efSearch=16,ht=56      0.2827 0.5219 0.5458      0.01804       91936792    17
nprobe=16,quantizer_efSearch=16,ht=54     0.3072 0.5935 0.6270      0.02898      183305069    11
nprobe=16,quantizer_efSearch=16,ht=62     0.3154 0.6360 0.6806      0.03602      183305069    9
nprobe=16,quantizer_efSearch=64,ht=54     0.3159 0.6147 0.6492      0.03882      182216410    8
nprobe=16,quantizer_efSearch=64,ht=60     0.3242 0.6574 0.7019      0.04363      182216410    7
nprobe=16,quantizer_efSearch=64,ht=64     0.3247 0.6596 0.7056      0.04773      182216410    7
nprobe=32,quantizer_efSearch=32,ht=58     0.3434 0.7176 0.7784      0.06104      362762940    5
nprobe=64,quantizer_efSearch=32,ht=54     0.3510 0.7370 0.7965      0.09724      718860562    4
nprobe=64,quantizer_efSearch=256,ht=58    0.3611 0.7889 0.8694      0.14044      715069688    3
nprobe=64,quantizer_efSearch=512,ht=58    0.3612 0.7893 0.8698      0.18208      714960050    2
nprobe=128,quantizer_efSearch=64,ht=128   0.3678 0.8227 0.9265      0.16848     1416000868    2
nprobe=128,quantizer_efSearch=128,ht=60   0.3687 0.8252 0.9281      0.21500     1412302926    2
nprobe=128,quantizer_efSearch=512,ht=128  0.3693 0.8294 0.9358      0.23552     1410334817    2
nprobe=256,quantizer_efSearch=256,ht=128  0.3726 0.8493 0.9684      0.33183     2777597470    1
nprobe=256,quantizer_efSearch=512,ht=128  0.3729 0.8500 0.9690      0.37440     2775645078    1
nprobe=512,quantizer_efSearch=512,ht=128  0.3743 0.8581 0.9834      0.63364     5450224683    1
nprobe=2048,quantizer_efSearch=512,ht=128 0.3746 0.8617 0.9916      2.18710    20935601755    1
```

</details>
<details><summary>`OPQ16_64,IVF4194304_HNSW32,PQ16` </summary>
Index size 26248830384

 code_size 16

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_HNSW32_PQ16.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.3194 0.6381 0.6783      0.03447       98424455    9
nprobe=2,quantizer_efSearch=4,ht=50       0.1055 0.1463 0.1486      0.00364        6281781    83
nprobe=1,quantizer_efSearch=8,ht=128      0.1118 0.1555 0.1567      0.00466        3140339    65
nprobe=4,quantizer_efSearch=4,ht=48       0.1219 0.1745 0.1778      0.00530       12526762    57
nprobe=4,quantizer_efSearch=16,ht=50      0.1673 0.2475 0.2515      0.00801       12480436    38
nprobe=4,quantizer_efSearch=16,ht=52      0.1822 0.2788 0.2832      0.00794       12480436    38
nprobe=4,quantizer_efSearch=16,ht=54      0.1957 0.3057 0.3108      0.00809       12480436    38
nprobe=4,quantizer_efSearch=32,ht=56      0.2031 0.3254 0.3311      0.01149       12406385    27
nprobe=8,quantizer_efSearch=16,ht=56      0.2467 0.4247 0.4353      0.01173       24900973    26
nprobe=16,quantizer_efSearch=4,ht=56      0.2523 0.4547 0.4702      0.01686       49802869    18
nprobe=16,quantizer_efSearch=4,ht=60      0.2573 0.4760 0.4952      0.01746       49802869    18
nprobe=16,quantizer_efSearch=16,ht=54     0.2767 0.4949 0.5111      0.01838       49588002    17
nprobe=16,quantizer_efSearch=16,ht=62     0.2906 0.5491 0.5743      0.01955       49588002    16
nprobe=16,quantizer_efSearch=64,ht=60     0.3024 0.5740 0.5992      0.02884       49111276    11
nprobe=16,quantizer_efSearch=64,ht=64     0.3032 0.5787 0.6060      0.02884       49111276    11
nprobe=64,quantizer_efSearch=8,ht=128     0.3101 0.6224 0.6681      0.03347      195567590    9
nprobe=32,quantizer_efSearch=16,ht=60     0.3189 0.6346 0.6726      0.03443       98424455    9
nprobe=32,quantizer_efSearch=32,ht=58     0.3273 0.6464 0.6843      0.03536       98003904    9
nprobe=32,quantizer_efSearch=32,ht=64     0.3298 0.6603 0.7050      0.03758       98003904    8
nprobe=32,quantizer_efSearch=128,ht=128   0.3322 0.6741 0.7198      0.03770       97325679    8
nprobe=64,quantizer_efSearch=16,ht=58     0.3328 0.6756 0.7224      0.05903      195450008    6
nprobe=64,quantizer_efSearch=32,ht=56     0.3413 0.6941 0.7391      0.06134      194422842    5
nprobe=64,quantizer_efSearch=64,ht=56     0.3465 0.7089 0.7561      0.06588      193410651    5
nprobe=128,quantizer_efSearch=64,ht=128   0.3671 0.7918 0.8724      0.07040      382353662    5
nprobe=128,quantizer_efSearch=128,ht=60   0.3680 0.7962 0.8727      0.13261      380670928    3
nprobe=128,quantizer_efSearch=512,ht=128  0.3704 0.8044 0.8877      0.14715      379340670    3
nprobe=256,quantizer_efSearch=256,ht=128  0.3761 0.8343 0.9344      0.16151      747499203    2
nprobe=256,quantizer_efSearch=512,ht=128  0.3765 0.8351 0.9358      0.19551      746212521    2
nprobe=512,quantizer_efSearch=512,ht=128  0.3820 0.8553 0.9671      0.32081     1465423762    1
nprobe=2048,quantizer_efSearch=512,ht=128 0.3831 0.8669 0.9894      1.01487     5640701761    1
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16` </summary>
Index size 25208589300

 code_size 16

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.1303 0.1816 0.1832      0.00518        9113130    58
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=26        0.0014 0.0025 0.0025      0.00286        4569676    105
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=26        0.0018 0.0030 0.0030      0.00290        4562681    104
nprobe=1,quantizer_k_factor_rf=2,quantizer_nprobe=4,ht=42        0.0285 0.0352 0.0354      0.00420        8768629    72
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=52        0.0930 0.1197 0.1201      0.00438        8766941    69
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=128      0.1138 0.1534 0.1544      0.00478        8763226    63
nprobe=2,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=52       0.1303 0.1816 0.1832      0.00518        9113492    58
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=58        0.1488 0.2210 0.2243      0.00711       13878731    43
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=56        0.1759 0.2735 0.2776      0.00757       15362001    40
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=52        0.1794 0.2627 0.2654      0.01009       23677648    30
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56        0.1929 0.2981 0.3023      0.01040       23579660    29
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=128      0.2016 0.3196 0.3248      0.01223       34325365    25
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=128      0.2399 0.4274 0.4428      0.01436       52335697    21
nprobe=8,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128     0.2661 0.4649 0.4789      0.01863       46550233    17
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56       0.2927 0.5266 0.5434      0.02619       60469712    12
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=54      0.2974 0.5193 0.5345      0.03015       71040421    10
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=128    0.3078 0.5757 0.6008      0.03253       71121332    10
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=60     0.3081 0.5736 0.5975      0.04214       91174622    8
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=64      0.3282 0.6418 0.6783      0.06664      180451983    5
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=32,ht=60     0.3390 0.6695 0.7102      0.06810      139564532    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=128     0.3625 0.7502 0.8133      0.07337      277489443    5
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=62   0.3783 0.8021 0.8832      0.25906      708326780    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=56    0.3813 0.8060 0.8827      0.36435      913280314    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=60    0.3821 0.8207 0.9122      0.35489      915543879    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=58    0.3841 0.8236 0.9112      0.37727      831712080    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=512,ht=128  0.3899 0.8576 0.9679      0.55084     2117675309    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=128,ht=62   0.3902 0.8608 0.9776      1.37129     3033973744    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=128 0.3911 0.8682 0.9916      2.84540     6253969986    1
```

</details>
<details><summary>`OPQ32_64,IVF1048576_HNSW32,PQ32x4fsr` </summary>
Index size 24830667724

 code_size 16

 log filename: autotune.dbbigann1B.OPQ32_64_IVF1048576_HNSW32_PQ32x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3517 0.8123 0.9630      0.16795     2783681256    2
nprobe=2,quantizer_efSearch=4            0.1548 0.2646 0.2774      0.00345       23135622    88
nprobe=4,quantizer_efSearch=4            0.1899 0.3475 0.3675      0.00451       46255513    67
nprobe=4,quantizer_efSearch=8            0.2200 0.4025 0.4260      0.00569       46158443    53
nprobe=8,quantizer_efSearch=8            0.2519 0.4928 0.5302      0.00754       92327588    40
nprobe=8,quantizer_efSearch=16           0.2677 0.5262 0.5664      0.00971       91933197    31
nprobe=16,quantizer_efSearch=8           0.2870 0.5956 0.6550      0.01177      183662808    26
nprobe=16,quantizer_efSearch=32          0.3073 0.6357 0.7001      0.01715      182659851    18
nprobe=16,quantizer_efSearch=64          0.3086 0.6408 0.7058      0.02415      182246141    13
nprobe=32,quantizer_efSearch=16          0.3191 0.6919 0.7753      0.02489      363936996    13
nprobe=32,quantizer_efSearch=64          0.3264 0.7125 0.8022      0.03220      361917104    10
nprobe=64,quantizer_efSearch=16          0.3300 0.7257 0.8287      0.04187      721554595    8
nprobe=64,quantizer_efSearch=32          0.3372 0.7513 0.8646      0.04596      718915634    7
nprobe=64,quantizer_efSearch=64          0.3400 0.7635 0.8783      0.05115      716668337    6
nprobe=64,quantizer_efSearch=128         0.3412 0.7687 0.8843      0.06151      715440110    5
nprobe=64,quantizer_efSearch=256         0.3415 0.7692 0.8847      0.08315      715053921    4
nprobe=128,quantizer_efSearch=32         0.3427 0.7753 0.9015      0.08991     1422237570    4
nprobe=128,quantizer_efSearch=128        0.3473 0.7961 0.9321      0.10129     1412480054    3
nprobe=128,quantizer_efSearch=256        0.3476 0.7983 0.9339      0.12401     1410935617    3
nprobe=256,quantizer_efSearch=128        0.3517 0.8123 0.9630      0.17414     2783681256    2
nprobe=256,quantizer_efSearch=512        0.3519 0.8147 0.9667      0.23400     2775922789    2
nprobe=512,quantizer_efSearch=128        0.3525 0.8172 0.9755      0.31093     5479382852    1
nprobe=512,quantizer_efSearch=256        0.3535 0.8205 0.9805      0.32381     5459492638    1
```

</details>
<details><summary>`OPQ32_64,IVF4194304_HNSW32,PQ32x4fsr` </summary>
Index size 27327961548

 code_size 16

 log filename: autotune.dbbigann1B.OPQ32_64_IVF4194304_HNSW32_PQ32x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3574 0.8074 0.9271      0.17005      750439144    2
nprobe=1,quantizer_efSearch=4            0.0927 0.1347 0.1373      0.00305        3135916    99
nprobe=2,quantizer_efSearch=4            0.1249 0.1959 0.2004      0.00345        6269504    88
nprobe=4,quantizer_efSearch=4            0.1575 0.2658 0.2750      0.00414       12518439    73
nprobe=4,quantizer_efSearch=8            0.1890 0.3211 0.3318      0.00547       12544227    55
nprobe=8,quantizer_efSearch=8            0.2260 0.4121 0.4321      0.00691       25023254    44
nprobe=8,quantizer_efSearch=16           0.2436 0.4459 0.4675      0.00937       24895163    33
nprobe=16,quantizer_efSearch=8           0.2673 0.5172 0.5510      0.01037       49729646    29
nprobe=16,quantizer_efSearch=16          0.2800 0.5419 0.5775      0.01198       49596294    26
nprobe=16,quantizer_efSearch=32          0.2888 0.5614 0.5990      0.01646       49305117    19
nprobe=32,quantizer_efSearch=16          0.3079 0.6257 0.6780      0.02227       98451742    14
nprobe=32,quantizer_efSearch=32          0.3156 0.6476 0.7034      0.02700       98041536    12
nprobe=32,quantizer_efSearch=64          0.3171 0.6571 0.7140      0.03419       97583488    9
nprobe=64,quantizer_efSearch=16          0.3248 0.6738 0.7430      0.03801      195448554    8
nprobe=64,quantizer_efSearch=32          0.3376 0.7115 0.7864      0.04144      194410783    8
nprobe=64,quantizer_efSearch=64          0.3421 0.7248 0.8040      0.05203      193408242    6
nprobe=64,quantizer_efSearch=128         0.3437 0.7308 0.8103      0.06479      192744176    5
nprobe=128,quantizer_efSearch=64         0.3493 0.7713 0.8696      0.09109      382376365    4
nprobe=128,quantizer_efSearch=128        0.3521 0.7828 0.8824      0.09895      380624324    4
nprobe=128,quantizer_efSearch=256        0.3533 0.7868 0.8868      0.12812      379653208    3
nprobe=256,quantizer_efSearch=128        0.3574 0.8074 0.9271      0.18036      750439144    2
nprobe=256,quantizer_efSearch=256        0.3595 0.8125 0.9333      0.18730      747481463    2
nprobe=256,quantizer_efSearch=512        0.3597 0.8144 0.9347      0.24131      746180554    2
nprobe=512,quantizer_efSearch=128        0.3602 0.8214 0.9528      0.33354     1479006509    1
nprobe=512,quantizer_efSearch=256        0.3629 0.8294 0.9642      0.34514     1470163370    1
nprobe=1024,quantizer_efSearch=256       0.3636 0.8372 0.9773      0.61097     2890512385    1
nprobe=1024,quantizer_efSearch=512       0.3641 0.8388 0.9811      0.64386     2874788373    1
```

</details>
<details><summary>`OPQ32_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ32x4fsr` </summary>
Index size 26287503376

 code_size 16

 log filename: autotune.dbbigann1B.OPQ32_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ32x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3042 0.6298 0.6846      0.02541      108815388    12
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1105 0.1660 0.1696      0.00161        7663865    187
nprobe=4,quantizer_k_factor_rf=1,quantizer_nprobe=2      0.1524 0.2437 0.2498      0.00253       15350323    119
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.1683 0.2792 0.2873      0.00255       15373671    118
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1741 0.2879 0.2963      0.00270       15364952    112
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1754 0.2902 0.2996      0.00297       15344849    101
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.1821 0.3051 0.3142      0.00328       18132253    92
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.2027 0.3579 0.3700      0.00436       30527448    69
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.2088 0.3747 0.3917      0.00481       27692022    63
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2334 0.4219 0.4402      0.00476       30486755    63
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2342 0.4232 0.4420      0.00538       30474976    56
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.2343 0.4307 0.4477      0.00616       35914727    49
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2420 0.4518 0.4743      0.00637       55180083    48
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2638 0.5137 0.5449      0.00809       54967537    38
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2731 0.5313 0.5636      0.00860       60386504    35
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2810 0.5517 0.5856      0.00969       60338495    31
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2816 0.5526 0.5868      0.01075       60346518    28
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=8    0.2817 0.5529 0.5872      0.01267       60310201    24
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=16   0.2876 0.5657 0.6018      0.01320       70617746    23
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.2882 0.5691 0.6050      0.01549       90785286    20
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3035 0.6281 0.6824      0.01917      108795151    16
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.3041 0.6298 0.6846      0.02213      108894835    14
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3116 0.6444 0.6991      0.02404      139073812    13
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3143 0.6590 0.7174      0.02752      139860865    11
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.3203 0.6881 0.7585      0.03312      204593863    10
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3302 0.7105 0.7827      0.03406      214915780    9
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3326 0.7179 0.7928      0.03604      214843832    9
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.3353 0.7320 0.8103      0.04050      234553674    8
nprobe=64,quantizer_k_factor_rf=16,quantizer_nprobe=128  0.3359 0.7332 0.8116      0.05658      359261256    6
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=128   0.3363 0.7333 0.8119      0.05372      354913263    6
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.3366 0.7444 0.8286      0.07855      422225971    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.3437 0.7673 0.8603      0.07715      403035861    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.3471 0.7833 0.8811      0.08492      423069322    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3476 0.7858 0.8834      0.08833      462070495    4
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3478 0.7858 0.8832      0.08644      541171727    4
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3499 0.8086 0.9205      0.13633      791770709    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3510 0.8120 0.9258      0.14919      830865734    3
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3515 0.8119 0.9257      0.15860      911363334    2
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3544 0.8167 0.9335      0.16358      913204499    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3556 0.8243 0.9518      0.30594     1513136079    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3566 0.8314 0.9602      0.27903     1550628143    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3569 0.8340 0.9645      0.28974     1631222473    2
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.3577 0.8399 0.9800      0.57894     3194608566    1
```

</details>

## Code sizes in [17, 32]

<details><summary>`IVF1048576_HNSW32,PQ64x4fs` </summary>
Index size 41359547269

 code_size 32

 log filename: autotune.dbbigann1B.IVF1048576_HNSW32_PQ64x4fs.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3349 0.7788 0.9495      0.13411     2792509235    3
nprobe=2,quantizer_efSearch=4            0.1507 0.2553 0.2752      0.00347       23208118    87
nprobe=4,quantizer_efSearch=4            0.1775 0.3305 0.3653      0.00449       46323655    67
nprobe=4,quantizer_efSearch=8            0.2049 0.3893 0.4278      0.00575       46320784    53
nprobe=8,quantizer_efSearch=4            0.2304 0.4576 0.5145      0.00717       92761023    42
nprobe=8,quantizer_efSearch=8            0.2373 0.4738 0.5327      0.00765       92657367    40
nprobe=8,quantizer_efSearch=16           0.2538 0.5085 0.5724      0.01005       92207626    30
nprobe=16,quantizer_efSearch=8           0.2677 0.5663 0.6496      0.01215      184209818    25
nprobe=16,quantizer_efSearch=16          0.2782 0.5906 0.6773      0.01366      183729028    22
nprobe=16,quantizer_efSearch=32          0.2864 0.6061 0.6954      0.01807      182964868    17
nprobe=32,quantizer_efSearch=16          0.3003 0.6545 0.7649      0.02146      365170448    14
nprobe=32,quantizer_efSearch=32          0.3083 0.6729 0.7876      0.02458      363861298    13
nprobe=32,quantizer_efSearch=64          0.3107 0.6825 0.7993      0.03217      362863343    10
nprobe=64,quantizer_efSearch=32          0.3212 0.7192 0.8561      0.04172      720958351    8
nprobe=64,quantizer_efSearch=64          0.3248 0.7319 0.8715      0.04664      718576818    7
nprobe=64,quantizer_efSearch=128         0.3268 0.7355 0.8770      0.06153      717327071    5
nprobe=128,quantizer_efSearch=128        0.3324 0.7620 0.9228      0.08510     1416520116    4
nprobe=128,quantizer_efSearch=256        0.3330 0.7641 0.9245      0.11011     1414737837    3
nprobe=256,quantizer_efSearch=128        0.3349 0.7788 0.9495      0.13637     2792509235    3
nprobe=256,quantizer_efSearch=256        0.3362 0.7815 0.9534      0.16201     2786017645    2
nprobe=512,quantizer_efSearch=256        0.3367 0.7855 0.9659      0.23723     5478425958    2
```

</details>
<details><summary>`IVF4194304_HNSW32,PQ64x4fs` </summary>
Index size 45448087941

 code_size 32

 log filename: autotune.dbbigann1B.IVF4194304_HNSW32_PQ64x4fs.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.3302 0.7597 0.9177      0.09610      761726447    4
nprobe=1,quantizer_efSearch=4            0.0869 0.1304 0.1355      0.00457        3200457    66
nprobe=2,quantizer_efSearch=4            0.1167 0.1912 0.2020      0.00481        6393882    63
nprobe=4,quantizer_efSearch=4            0.1450 0.2569 0.2766      0.00519       12765421    58
nprobe=4,quantizer_efSearch=8            0.1751 0.3142 0.3366      0.00723       12752378    42
nprobe=8,quantizer_efSearch=4            0.1981 0.3764 0.4121      0.00727       25493361    42
nprobe=8,quantizer_efSearch=8            0.2075 0.3944 0.4322      0.00799       25480381    38
nprobe=16,quantizer_efSearch=4           0.2166 0.4381 0.4907      0.00884       50613886    34
nprobe=16,quantizer_efSearch=8           0.2426 0.4924 0.5517      0.01083       50552795    28
nprobe=16,quantizer_efSearch=16          0.2563 0.5160 0.5769      0.01318       50380588    23
nprobe=32,quantizer_efSearch=8           0.2589 0.5415 0.6199      0.01349      100442351    23
nprobe=32,quantizer_efSearch=16          0.2804 0.5891 0.6742      0.01696       99975295    18
nprobe=64,quantizer_efSearch=16          0.2917 0.6303 0.7368      0.02229      198447352    14
nprobe=64,quantizer_efSearch=32          0.3020 0.6641 0.7791      0.02870      197366045    11
nprobe=128,quantizer_efSearch=32         0.3107 0.6989 0.8309      0.03914      390616303    8
nprobe=128,quantizer_efSearch=64         0.3195 0.7220 0.8620      0.05084      388094477    6
nprobe=128,quantizer_efSearch=128        0.3251 0.7318 0.8742      0.07266      386156040    5
nprobe=256,quantizer_efSearch=128        0.3302 0.7597 0.9177      0.09291      761726447    4
nprobe=256,quantizer_efSearch=256        0.3325 0.7636 0.9229      0.13280      758367702    3
nprobe=512,quantizer_efSearch=256        0.3343 0.7774 0.9479      0.17819     1492073790    2
nprobe=1024,quantizer_efSearch=128       0.3345 0.7761 0.9482      0.19784     2947504263    2
nprobe=1024,quantizer_efSearch=256       0.3365 0.7831 0.9612      0.24988     2934899237    2
```

</details>
<details><summary>`IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ64x4fs` </summary>
Index size 44473473481

 code_size 32

 log filename: autotune.dbbigann1B.IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ64x4fs.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.3303 0.7628 0.9228      0.09600      919355435    4
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1285 0.2259 0.2422      0.00368       14046358    82
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=1      0.1403 0.2491 0.2700      0.00380       26755525    79
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1523 0.2752 0.2988      0.00397       26666729    76
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=1      0.1529 0.2788 0.3038      0.00433       26532651    70
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.1595 0.2812 0.3012      0.00438       15537015    69
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=2      0.1597 0.2840 0.3038      0.00508       15501283    60
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=1     0.1677 0.3142 0.3492      0.00581       51454932    52
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=1     0.1681 0.3169 0.3526      0.00608       51255102    50
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.1888 0.3540 0.3875      0.00718       28019512    42
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2149 0.4022 0.4399      0.00805       30880962    38
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.2162 0.4059 0.4439      0.00857       30852060    36
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2251 0.4365 0.4830      0.00901       56261435    34
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.2429 0.4843 0.5415      0.01086       55672190    28
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.2484 0.5042 0.5711      0.01187      106275123    26
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.2581 0.5115 0.5698      0.01256       61433875    24
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.2589 0.5329 0.6077      0.01249      105532445    25
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.2592 0.5217 0.5837      0.01399       61162064    22
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.2596 0.5219 0.5846      0.01596       61163373    19
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.2671 0.5375 0.5994      0.01686       71935433    18
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2686 0.5505 0.6230      0.01484      111454312    21
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2827 0.5906 0.6758      0.01643      110430630    19
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.2897 0.6101 0.7080      0.02020      210267077    15
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.2968 0.6410 0.7510      0.02287      207623739    14
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.2998 0.6363 0.7392      0.02447      220547113    13
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.3067 0.6672 0.7808      0.02529      218856062    12
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.3079 0.6708 0.7869      0.02717      217942613    12
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.3108 0.6774 0.7927      0.03100      238372358    10
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32    0.3121 0.6826 0.7995      0.03297      238328823    10
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.3175 0.7054 0.8334      0.04052      434372588    8
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3234 0.7258 0.8657      0.04299      430159625    7
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32   0.3258 0.7332 0.8733      0.05375      428584811    6
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3282 0.7551 0.9117      0.06379      805045448    5
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3301 0.7600 0.9179      0.07105      844272902    5
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3303 0.7628 0.9228      0.09554      923191602    4
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.3310 0.7633 0.9227      0.09507      839522221    4
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.3312 0.7698 0.9386      0.10401     1538068072    3
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.3326 0.7683 0.9346      0.10439     1606765426    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.3328 0.7761 0.9468      0.11015     1577539381    3
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.3330 0.7767 0.9479      0.12751     1658421541    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.3338 0.7790 0.9501      0.12820     1571136677    3
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.3341 0.7794 0.9512      0.14278     1652906082    3
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.3354 0.7857 0.9643      0.19090     3005438921    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.3362 0.7863 0.9660      0.19400     3085686985    2
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=512 0.3364 0.7862 0.9659      0.29420     3562291916    2
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.3366 0.7879 0.9677      0.35787     3236229749    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=512 0.3367 0.7904 0.9756      0.50787     6336669680    1
```

</details>
<details><summary>`OPQ32_128,IVF1048576_HNSW32,PQ32` </summary>
Index size 40830818224

 code_size 32

 log filename: autotune.dbbigann1B.OPQ32_128_IVF1048576_HNSW32_PQ32.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.4310 0.5871 0.5897      0.10328       91601086    3
nprobe=1,quantizer_efSearch=4,ht=14       0.0002 0.0002 0.0002      0.00576       11611213    53
nprobe=1,quantizer_efSearch=4,ht=116      0.1589 0.1897 0.1897      0.00776       11611213    39
nprobe=1,quantizer_efSearch=4,ht=118      0.1593 0.1909 0.1909      0.00801       11611213    38
nprobe=2,quantizer_efSearch=4,ht=114      0.2198 0.2721 0.2724      0.01135       23192469    27
nprobe=2,quantizer_efSearch=4,ht=122      0.2226 0.2773 0.2776      0.01291       23192469    24
nprobe=2,quantizer_efSearch=4,ht=126      0.2228 0.2776 0.2779      0.01376       23192469    22
nprobe=2,quantizer_efSearch=8,ht=120      0.2469 0.3099 0.3102      0.01451       23174400    21
nprobe=4,quantizer_efSearch=4,ht=108      0.2595 0.3323 0.3333      0.01732       46316666    18
nprobe=4,quantizer_efSearch=8,ht=122      0.3269 0.4298 0.4310      0.02336       46288586    13
nprobe=4,quantizer_efSearch=32,ht=110     0.3275 0.4214 0.4225      0.02961       45915724    11
nprobe=8,quantizer_efSearch=4,ht=108      0.3531 0.4666 0.4685      0.03185       92695211    10
nprobe=8,quantizer_efSearch=4,ht=120      0.3796 0.5169 0.5191      0.03233       92695211    10
nprobe=8,quantizer_efSearch=32,ht=256     0.4300 0.5852 0.5879      0.03619       91807530    9
nprobe=16,quantizer_efSearch=8,ht=114     0.4561 0.6388 0.6417      0.05387      184207738    6
nprobe=16,quantizer_efSearch=64,ht=116    0.4909 0.6962 0.6996      0.07614      182501293    5
nprobe=16,quantizer_efSearch=32,ht=128    0.4917 0.6999 0.7038      0.07503      182907608    4
nprobe=32,quantizer_efSearch=8,ht=256     0.4971 0.7222 0.7270      0.09199      366444657    4
nprobe=32,quantizer_efSearch=16,ht=256    0.5237 0.7698 0.7754      0.09389      365135787    4
nprobe=32,quantizer_efSearch=32,ht=256    0.5395 0.7935 0.7991      0.10314      363821687    3
nprobe=64,quantizer_efSearch=64,ht=118    0.5753 0.8709 0.8780      0.21832      718575081    2
nprobe=128,quantizer_efSearch=256,ht=110  0.5783 0.8619 0.8680      0.40197     1414734671    1
nprobe=128,quantizer_efSearch=32,ht=120   0.5839 0.8930 0.9016      0.37180     1426573297    1
nprobe=128,quantizer_efSearch=512,ht=116  0.5983 0.9172 0.9251      0.46178     1414223165    1
nprobe=256,quantizer_efSearch=128,ht=114  0.6021 0.9310 0.9403      0.66857     2792459211    1
nprobe=256,quantizer_efSearch=64,ht=122   0.6027 0.9405 0.9521      0.74355     2804831630    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6122 0.9611 0.9736      1.39327     5500445142    1
nprobe=1024,quantizer_efSearch=128,ht=124 0.6147 0.9685 0.9821      2.89178    10777426885    1
nprobe=1024,quantizer_efSearch=128,ht=128 0.6148 0.9690 0.9829      3.27489    10777426885    1
nprobe=2048,quantizer_efSearch=512,ht=256 0.6186 0.9818 0.9964      4.71473    21028104238    1
```

</details>
<details><summary>`OPQ32_128,IVF4194304_HNSW32,PQ32` </summary>
Index size 43322670512

 code_size 32

 log filename: autotune.dbbigann1B.OPQ32_128_IVF4194304_HNSW32_PQ32.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.2821 0.3569 0.3573      0.02471       25464287    13
nprobe=1,quantizer_efSearch=4,ht=80       0.0157 0.0198 0.0198      0.00613        3190895    49
nprobe=1,quantizer_efSearch=4,ht=118      0.1175 0.1347 0.1347      0.00656        3190895    46
nprobe=1,quantizer_efSearch=8,ht=116      0.1321 0.1522 0.1522      0.00847        3179695    36
nprobe=2,quantizer_efSearch=4,ht=122      0.1695 0.2020 0.2020      0.00943        6378583    32
nprobe=2,quantizer_efSearch=4,ht=126      0.1701 0.2026 0.2026      0.00969        6378583    31
nprobe=2,quantizer_efSearch=8,ht=120      0.1980 0.2372 0.2372      0.01127        6359013    27
nprobe=4,quantizer_efSearch=8,ht=122      0.2692 0.3345 0.3347      0.01671       12739631    18
nprobe=8,quantizer_efSearch=4,ht=108      0.2821 0.3569 0.3573      0.02476       25464287    13
nprobe=8,quantizer_efSearch=4,ht=120      0.3167 0.4118 0.4122      0.02615       25464287    12
nprobe=8,quantizer_efSearch=32,ht=256     0.3687 0.4835 0.4840      0.02799       25159438    11
nprobe=16,quantizer_efSearch=8,ht=114     0.4013 0.5319 0.5330      0.04723       50522065    7
nprobe=32,quantizer_efSearch=8,ht=256     0.4520 0.6264 0.6276      0.05187      100381957    6
nprobe=32,quantizer_efSearch=32,ht=256    0.4992 0.7056 0.7073      0.05975       99527838    6
nprobe=64,quantizer_efSearch=16,ht=118    0.5111 0.7350 0.7368      0.17055      198413211    2
nprobe=64,quantizer_efSearch=32,ht=126    0.5378 0.7866 0.7890      0.17783      197348071    2
nprobe=64,quantizer_efSearch=64,ht=118    0.5423 0.7916 0.7937      0.18783      196231274    2
nprobe=128,quantizer_efSearch=32,ht=120   0.5587 0.8329 0.8366      0.34555      390638013    1
nprobe=128,quantizer_efSearch=512,ht=116  0.5774 0.8627 0.8662      0.47548      384582062    1
nprobe=256,quantizer_efSearch=128,ht=114  0.5911 0.8879 0.8921      0.66340      761760317    1
nprobe=256,quantizer_efSearch=64,ht=122   0.5916 0.9014 0.9065      0.65907      766697273    1
nprobe=512,quantizer_efSearch=128,ht=120  0.6063 0.9412 0.9480      1.24509     1501603874    1
nprobe=1024,quantizer_efSearch=128,ht=124 0.6114 0.9559 0.9637      2.32541     2947618969    1
nprobe=2048,quantizer_efSearch=512,ht=256 0.6200 0.9816 0.9913      2.49068     5728067406    1
```

</details>
<details><summary>`OPQ32_128,IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ32` </summary>
Index size 42350058996

 code_size 32

 log filename: autotune.dbbigann1B.OPQ32_128_IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ32.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.2422 0.3049 0.3057      0.01740       26478319    18
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=1,ht=14       0.0003 0.0003 0.0003      0.00413        4599000    73
nprobe=1,quantizer_k_factor_rf=16,quantizer_nprobe=1,ht=112      0.1023 0.1157 0.1157      0.00397        4599342    76
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=122       0.1312 0.1497 0.1497      0.00514        6034849    59
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=128       0.1325 0.1513 0.1513      0.00537        6025531    56
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1,ht=110       0.1374 0.1599 0.1599      0.00517        7772030    59
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=106       0.2020 0.2403 0.2406      0.00973       15537066    31
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=116       0.2404 0.2937 0.2942      0.01019       15509383    30
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=110       0.2535 0.3064 0.3070      0.01304       18258517    23
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=4,ht=120      0.2770 0.3397 0.3403      0.01397       18261146    22
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=256     0.3510 0.4640 0.4655      0.02071       52845035    15
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=124      0.3650 0.4726 0.4735      0.02671       47022790    12
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=4,ht=114      0.3951 0.5203 0.5215      0.03223       55668610    10
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=126      0.4355 0.5879 0.5894      0.03877       61224408    8
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=8,ht=118      0.4858 0.6732 0.6751      0.06343      110400309    5
nprobe=32,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=126     0.5031 0.7083 0.7105      0.06583      120969158    5
nprobe=32,quantizer_k_factor_rf=32,quantizer_nprobe=64,ht=118    0.5058 0.7098 0.7117      0.09085      181096052    4
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=120   0.5078 0.7156 0.7176      0.09667      259234849    4
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=112     0.5172 0.7322 0.7344      0.10872      218241348    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=126     0.5403 0.7875 0.7907      0.11922      218010235    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=114    0.5550 0.8158 0.8186      0.20869      408883072    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=116    0.5630 0.8361 0.8394      0.22031      408390594    2
nprobe=128,quantizer_k_factor_rf=32,quantizer_nprobe=16,ht=128   0.5725 0.8598 0.8637      0.27957      408375616    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=256  0.6043 0.9321 0.9392      0.31658      923822553    1
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=124   0.6131 0.9574 0.9653      0.87622     1653681397    1
nprobe=1024,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=120 0.6187 0.9692 0.9777      2.44777     3074594151    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=256 0.6222 0.9844 0.9952      3.55808     6335339778    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64x4fsr` </summary>
Index size 41359604684

 code_size 32

 log filename: autotune.dbbigann1B.OPQ64_128_IVF1048576_HNSW32_PQ64x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5743 0.9505 0.9669      0.33388     2792321017    1
nprobe=1,quantizer_efSearch=8            0.1715 0.2105 0.2109      0.00396       11590934    76
nprobe=2,quantizer_efSearch=4            0.2155 0.2781 0.2792      0.00336       23200419    90
nprobe=4,quantizer_efSearch=4            0.2691 0.3674 0.3699      0.00478       46310921    63
nprobe=4,quantizer_efSearch=8            0.3124 0.4307 0.4335      0.00604       46315245    50
nprobe=4,quantizer_efSearch=16           0.3269 0.4509 0.4537      0.00800       46138877    38
nprobe=8,quantizer_efSearch=16           0.3988 0.5734 0.5778      0.01088       92224964    28
nprobe=8,quantizer_efSearch=32           0.4052 0.5830 0.5875      0.01454       91848961    21
nprobe=8,quantizer_efSearch=64           0.4056 0.5837 0.5884      0.02146       91680887    14
nprobe=16,quantizer_efSearch=8           0.4347 0.6526 0.6584      0.02182      184221105    14
nprobe=16,quantizer_efSearch=16          0.4517 0.6794 0.6854      0.02303      183739573    14
nprobe=16,quantizer_efSearch=32          0.4622 0.6977 0.7038      0.02599      182936854    12
nprobe=16,quantizer_efSearch=64          0.4658 0.7033 0.7097      0.03459      182505577    9
nprobe=32,quantizer_efSearch=16          0.4967 0.7675 0.7759      0.04145      365152922    8
nprobe=32,quantizer_efSearch=64          0.5143 0.8007 0.8098      0.05009      362845113    6
nprobe=32,quantizer_efSearch=128         0.5155 0.8036 0.8128      0.06020      362390451    5
nprobe=32,quantizer_efSearch=256         0.5163 0.8045 0.8137      0.08230      362254622    4
nprobe=64,quantizer_efSearch=16          0.5196 0.8185 0.8292      0.08428      723940179    4
nprobe=64,quantizer_efSearch=32          0.5386 0.8589 0.8696      0.08942      720857142    4
nprobe=64,quantizer_efSearch=64          0.5455 0.8740 0.8848      0.09175      718504884    4
nprobe=64,quantizer_efSearch=128         0.5488 0.8796 0.8904      0.10160      717272808    3
nprobe=64,quantizer_efSearch=256         0.5492 0.8808 0.8916      0.12208      716785343    3
nprobe=64,quantizer_efSearch=512         0.5494 0.8810 0.8918      0.16693      716657452    2
nprobe=128,quantizer_efSearch=128        0.5639 0.9234 0.9382      0.18150     1416404104    2
nprobe=128,quantizer_efSearch=256        0.5655 0.9251 0.9400      0.19789     1414632140    2
nprobe=256,quantizer_efSearch=128        0.5743 0.9505 0.9669      0.34351     2792321017    1
nprobe=256,quantizer_efSearch=256        0.5774 0.9542 0.9711      0.36843     2785808411    1
nprobe=512,quantizer_efSearch=128        0.5798 0.9613 0.9786      0.63785     5500115494    1
nprobe=512,quantizer_efSearch=256        0.5810 0.9669 0.9847      0.69525     5478046747    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304_HNSW32,PQ64x4fsr` </summary>
Index size 45448159692

 code_size 32

 log filename: autotune.dbbigann1B.OPQ64_128_IVF4194304_HNSW32_PQ64x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.5714 0.9193 0.9330      0.30648      761721126    1
nprobe=1,quantizer_efSearch=4            0.1160 0.1354 0.1356      0.00538        3195148    56
nprobe=2,quantizer_efSearch=4            0.1634 0.2002 0.2010      0.00585        6384898    52
nprobe=4,quantizer_efSearch=4            0.2153 0.2755 0.2770      0.00699       12747757    43
nprobe=4,quantizer_efSearch=8            0.2583 0.3332 0.3349      0.00955       12742336    32
nprobe=8,quantizer_efSearch=4            0.3076 0.4113 0.4136      0.01107       25476374    28
nprobe=8,quantizer_efSearch=8            0.3194 0.4298 0.4322      0.01198       25458608    26
nprobe=8,quantizer_efSearch=16           0.3489 0.4703 0.4728      0.01651       25305623    19
nprobe=16,quantizer_efSearch=4           0.3576 0.4950 0.4980      0.02027       50582945    15
nprobe=16,quantizer_efSearch=8           0.3964 0.5554 0.5589      0.02234       50510812    14
nprobe=16,quantizer_efSearch=16          0.4124 0.5792 0.5829      0.02737       50360462    11
nprobe=16,quantizer_efSearch=32          0.4271 0.6021 0.6059      0.03456       50087520    9
nprobe=32,quantizer_efSearch=8           0.4343 0.6251 0.6301      0.03761      100420659    8
nprobe=32,quantizer_efSearch=16          0.4648 0.6775 0.6829      0.04162       99959277    8
nprobe=32,quantizer_efSearch=32          0.4784 0.7013 0.7073      0.05112       99515161    6
nprobe=32,quantizer_efSearch=64          0.4841 0.7127 0.7186      0.06382       98969779    5
nprobe=64,quantizer_efSearch=16          0.4923 0.7405 0.7482      0.07454      198404831    5
nprobe=64,quantizer_efSearch=32          0.5132 0.7825 0.7902      0.07831      197344171    4
nprobe=64,quantizer_efSearch=64          0.5212 0.7973 0.8058      0.08969      196227506    4
nprobe=64,quantizer_efSearch=128         0.5254 0.8050 0.8133      0.11762      195462611    3
nprobe=128,quantizer_efSearch=32         0.5345 0.8344 0.8442      0.14617      390639195    3
nprobe=128,quantizer_efSearch=64         0.5511 0.8643 0.8752      0.15016      388094318    3
nprobe=128,quantizer_efSearch=128        0.5562 0.8763 0.8874      0.16361      386153884    2
nprobe=128,quantizer_efSearch=256        0.5579 0.8794 0.8904      0.21313      385030369    2
nprobe=256,quantizer_efSearch=64         0.5619 0.8986 0.9115      0.30015      766710925    1
nprobe=256,quantizer_efSearch=128        0.5714 0.9193 0.9330      0.31626      761721126    1
nprobe=256,quantizer_efSearch=256        0.5742 0.9251 0.9386      0.35826      758354764    1
nprobe=256,quantizer_efSearch=512        0.5753 0.9262 0.9396      0.45572      756804758    1
nprobe=512,quantizer_efSearch=128        0.5757 0.9407 0.9559      0.57888     1501505741    1
nprobe=512,quantizer_efSearch=256        0.5804 0.9502 0.9661      0.59399     1492057128    1
nprobe=512,quantizer_efSearch=512        0.5820 0.9523 0.9687      0.63095     1486650708    1
nprobe=1024,quantizer_efSearch=256       0.5854 0.9627 0.9803      1.12932     2934916918    1
nprobe=1024,quantizer_efSearch=512       0.5868 0.9666 0.9850      1.17065     2917528550    1
nprobe=2048,quantizer_efSearch=512       0.5877 0.9711 0.9914      2.05923     2873468131    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ64x4fsr` </summary>
Index size 44473243152

 code_size 32

 log filename: autotune.dbbigann1B.OPQ64_128_IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ64x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.5740 0.9273 0.9391      0.27982      923751429    2
nprobe=2,quantizer_k_factor_rf=2,quantizer_nprobe=1      0.1502 0.1810 0.1812      0.00296        7751464    102
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.1924 0.2419 0.2426      0.00440       14025432    69
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2386 0.3003 0.3013      0.00545       15484147    56
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2402 0.3034 0.3044      0.00564       15495565    54
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.2697 0.3425 0.3436      0.00867       18274137    35
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.2941 0.3880 0.3898      0.00938       28018317    32
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=4      0.3208 0.4208 0.4222      0.01048       30781528    29
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=4      0.3356 0.4460 0.4478      0.01090       30797838    28
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.3363 0.4462 0.4480      0.01153       30740026    27
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.3500 0.4695 0.4713      0.01399       36290149    22
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.3757 0.5170 0.5191      0.01706       55639498    18
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.3916 0.5458 0.5489      0.01916       55662725    16
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8     0.4171 0.5869 0.5897      0.02053       61077770    15
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=8     0.4172 0.5883 0.5914      0.02380       61187565    13
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.4277 0.6042 0.6071      0.02561       71930189    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=4     0.4319 0.6191 0.6236      0.03070      104940083    10
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.4539 0.6556 0.6599      0.03184      110271606    10
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.4688 0.6809 0.6851      0.03561      120902100    9
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=32    0.4726 0.6886 0.6929      0.04397      141025275    7
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.4871 0.7188 0.7233      0.04508      141337280    7
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.4976 0.7570 0.7628      0.07041      207231349    5
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.5222 0.8036 0.8096      0.08363      238054634    4
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.5242 0.8083 0.8146      0.09185      278881692    4
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.5382 0.8419 0.8492      0.12687      408371247    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=16   0.5444 0.8568 0.8646      0.13378      407879935    3
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32   0.5471 0.8608 0.8687      0.13516      427306572    3
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5496 0.8656 0.8736      0.14443      466425425    3
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5536 0.8772 0.8858      0.13465      428712185    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.5564 0.8812 0.8903      0.14302      469032555    3
nprobe=128,quantizer_k_factor_rf=16,quantizer_nprobe=64  0.5566 0.8812 0.8903      0.17641      469926898    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.5578 0.8904 0.9009      0.27152      782110214    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.5683 0.9166 0.9276      0.26018      838470496    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.5728 0.9256 0.9372      0.27762      842031828    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.5734 0.9263 0.9379      0.26902      923668946    2
nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=64   0.5738 0.9267 0.9385      0.30340      842122785    1
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5740 0.9273 0.9391      0.28800      923751429    2
nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.5772 0.9436 0.9579      0.50819     1532125027    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=32   0.5785 0.9438 0.9582      0.50466     1532245918    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=128  0.5814 0.9532 0.9684      0.52683     1646525689    1
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=256  0.5815 0.9533 0.9685      0.54648     1813964297    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.5856 0.9670 0.9844      0.93018     2992671088    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.5865 0.9686 0.9860      0.95053     3074102136    1
nprobe=2048,quantizer_k_factor_rf=1,quantizer_nprobe=128 0.5877 0.9749 0.9943      1.75912     3019891906    1
```

</details>

## Code sizes in [33, 64]

<details><summary>`OPQ128_256,IVF1048576_HNSW32,PQ128x4fsr` </summary>
Index size 74416953292

 code_size 64

 log filename: autotune.dbbigann1B.OPQ128_256_IVF1048576_HNSW32_PQ128x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                         0.7980 0.9662 0.9665      0.65976     2792533681    1
nprobe=1,quantizer_efSearch=8            0.1981 0.2113 0.2113      0.01012       11606304    30
nprobe=2,quantizer_efSearch=4            0.2556 0.2770 0.2770      0.00872       23203628    35
nprobe=2,quantizer_efSearch=8            0.2871 0.3122 0.3122      0.01191       23201286    26
nprobe=4,quantizer_efSearch=4            0.3342 0.3693 0.3693      0.01225       46346435    25
nprobe=4,quantizer_efSearch=8            0.3906 0.4321 0.4321      0.01557       46331281    20
nprobe=4,quantizer_efSearch=16           0.4117 0.4542 0.4542      0.02097       46113982    15
nprobe=8,quantizer_efSearch=8            0.4796 0.5410 0.5410      0.02458       92653598    13
nprobe=8,quantizer_efSearch=16           0.5128 0.5785 0.5785      0.03335       92217558    9
nprobe=8,quantizer_efSearch=32           0.5216 0.5881 0.5881      0.04209       91860948    8
nprobe=16,quantizer_efSearch=8           0.5724 0.6580 0.6580      0.04309      184216600    7
nprobe=16,quantizer_efSearch=16          0.5965 0.6855 0.6855      0.04492      183735143    7
nprobe=16,quantizer_efSearch=32          0.6147 0.7042 0.7042      0.05529      182948322    6
nprobe=16,quantizer_efSearch=64          0.6192 0.7098 0.7098      0.07415      182520579    5
nprobe=32,quantizer_efSearch=8           0.6263 0.7276 0.7276      0.09114      366473654    4
nprobe=32,quantizer_efSearch=16          0.6657 0.7759 0.7759      0.08924      365178343    4
nprobe=32,quantizer_efSearch=32          0.6852 0.7993 0.7993      0.10166      363868670    3
nprobe=32,quantizer_efSearch=64          0.6941 0.8098 0.8098      0.11686      362878144    3
nprobe=32,quantizer_efSearch=128         0.6956 0.8126 0.8126      0.14029      362418893    3
nprobe=64,quantizer_efSearch=32          0.7378 0.8697 0.8697      0.17255      720976505    2
nprobe=64,quantizer_efSearch=64          0.7504 0.8850 0.8850      0.18231      718584123    2
nprobe=64,quantizer_efSearch=128         0.7547 0.8904 0.8904      0.20866      717325137    2
nprobe=64,quantizer_efSearch=256         0.7553 0.8915 0.8915      0.26553      716845172    2
nprobe=64,quantizer_efSearch=512         0.7555 0.8919 0.8919      0.33984      716718217    1
nprobe=128,quantizer_efSearch=32         0.7589 0.9054 0.9054      0.37037     1426656082    1
nprobe=128,quantizer_efSearch=128        0.7814 0.9378 0.9379      0.38102     1416512098    1
nprobe=128,quantizer_efSearch=256        0.7830 0.9397 0.9398      0.45515     1414749661    1
nprobe=128,quantizer_efSearch=512        0.7831 0.9403 0.9404      0.51588     1414236671    1
nprobe=256,quantizer_efSearch=128        0.7980 0.9662 0.9665      0.69660     2792533681    1
nprobe=256,quantizer_efSearch=256        0.8009 0.9705 0.9708      0.73793     2786043227    1
nprobe=256,quantizer_efSearch=512        0.8017 0.9713 0.9716      0.82064     2783909400    1
nprobe=512,quantizer_efSearch=128        0.8018 0.9781 0.9784      1.35386     5500475868    1
nprobe=512,quantizer_efSearch=256        0.8063 0.9843 0.9846      1.33490     5478476172    1
nprobe=512,quantizer_efSearch=512        0.8070 0.9862 0.9865      1.38094     5468771939    1
```

</details>
<details><summary>`OPQ128_256,IVF4194304(IVF1024,PQ128x4fs,RFlat),PQ128x4fsr` </summary>
Index size 80845980176

 code_size 64

 log filename: autotune.dbbigann1B.OPQ128_256_IVF4194304_IVF1024_PQ128x4fs_RFlat__PQ128x4fsr.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                         0.7046 0.8142 0.8142      0.26781      359755933    2
nprobe=4,quantizer_k_factor_rf=8,quantizer_nprobe=1      0.2276 0.2443 0.2443      0.00898       14032990    34
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=2      0.2825 0.3046 0.3046      0.00980       15492888    31
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2      0.2831 0.3053 0.3053      0.01001       15490061    30
nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4      0.3187 0.3436 0.3436      0.01415       18303007    22
nprobe=8,quantizer_k_factor_rf=16,quantizer_nprobe=2     0.3585 0.3918 0.3918      0.01939       28016890    16
nprobe=8,quantizer_k_factor_rf=8,quantizer_nprobe=4      0.4082 0.4471 0.4471      0.02202       30825560    14
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=8      0.4334 0.4766 0.4766      0.02681       36342004    12
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4     0.4880 0.5409 0.5409      0.03181       55816781    10
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4     0.4945 0.5491 0.5491      0.03475       55687403    9
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.5254 0.5850 0.5850      0.03779       61261520    8
nprobe=16,quantizer_k_factor_rf=16,quantizer_nprobe=8    0.5303 0.5915 0.5915      0.04443       61171681    7
nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.5384 0.6003 0.6003      0.04422       71944850    7
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.5439 0.6076 0.6076      0.04541       72042100    7
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.5471 0.6126 0.6126      0.05656       93176005    6
nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6249 0.7088 0.7088      0.06996      121193802    5
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6351 0.7232 0.7232      0.07893      141418615    4
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.6357 0.7237 0.7237      0.08462      141771509    4
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=64    0.6363 0.7244 0.7244      0.10559      181931507    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=8     0.6578 0.7560 0.7560      0.12103      207930474    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=8     0.6655 0.7629 0.7629      0.12011      207400796    3
nprobe=64,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6871 0.7916 0.7916      0.12709      218397418    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6942 0.7994 0.7994      0.13518      217874887    3
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.7033 0.8116 0.8116      0.15527      237398057    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=64    0.7051 0.8145 0.8145      0.17929      278913090    2
nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.7350 0.8606 0.8606      0.24409      409437977    2
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.7545 0.8877 0.8878      0.25823      428748192    2
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=64   0.7568 0.8906 0.8907      0.28237      469885621    2
nprobe=256,quantizer_k_factor_rf=1,quantizer_nprobe=16   0.7603 0.8999 0.8999      0.45946      784633283    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16   0.7611 0.9017 0.9017      0.46461      782155106    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64   0.7869 0.9388 0.9389      0.48594      839393058    1
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128  0.7873 0.9392 0.9393      0.49559      923720523    1
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=64   0.8047 0.9660 0.9661      0.87128     1575964739    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=32  0.8085 0.9738 0.9739      1.65024     1506760522    1
nprobe=1024,quantizer_k_factor_rf=1,quantizer_nprobe=64  0.8138 0.9834 0.9835      1.65496     1550031872    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64  0.8156 0.9845 0.9846      1.68471     1545031574    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128 0.8165 0.9860 0.9861      1.65725     1625596147    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256 0.8168 0.9862 0.9863      1.83553     1787914061    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=64  0.8193 0.9932 0.9933      3.36613     1526165811    1
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=128 0.8220 0.9950 0.9951      3.25303     1622425252    1
```

</details>
<details><summary>`OPQ16_64,IVF1048576_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 72700667282

 code_size 64

 log filename: autotune.dbbigann1B.OPQ16_64_IVF1048576_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.5996 0.7693 0.7697      0.04177      718347519    8
k_factor_rf=1,nprobe=1,quantizer_efSearch=4       0.1588 0.1742 0.1743      0.00486       11585730    62
k_factor_rf=1,nprobe=2,quantizer_efSearch=4       0.2117 0.2335 0.2335      0.00520       23169613    58
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.2420 0.2676 0.2676      0.00625       23031131    48
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.3266 0.3722 0.3722      0.00724       92383076    42
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.3415 0.3899 0.3899      0.00771       92292170    39
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.3521 0.4020 0.4020      0.00886      183884252    34
k_factor_rf=2,nprobe=8,quantizer_efSearch=4       0.3630 0.4255 0.4256      0.00940       92383076    32
k_factor_rf=2,nprobe=16,quantizer_efSearch=4      0.3944 0.4670 0.4671      0.01133      183884252    27
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.4318 0.5134 0.5135      0.01237      183560538    25
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.4480 0.5345 0.5346      0.01370      183110448    22
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.4508 0.5370 0.5370      0.01590      364726969    19
k_factor_rf=4,nprobe=16,quantizer_efSearch=8      0.4717 0.5750 0.5751      0.01613      183560538    19
k_factor_rf=4,nprobe=16,quantizer_efSearch=16     0.4892 0.5985 0.5986      0.01704      183110448    18
k_factor_rf=4,nprobe=32,quantizer_efSearch=8      0.4947 0.6068 0.6070      0.01944      364726969    16
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.5249 0.6487 0.6489      0.02179      363785397    14
k_factor_rf=4,nprobe=32,quantizer_efSearch=32     0.5381 0.6650 0.6652      0.02522      362613802    12
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.5410 0.6707 0.6709      0.02900      721117545    11
k_factor_rf=4,nprobe=64,quantizer_efSearch=32     0.5593 0.6964 0.6966      0.03197      718347519    10
k_factor_rf=8,nprobe=32,quantizer_efSearch=32     0.5681 0.7224 0.7228      0.03541      362613802    9
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.5996 0.7693 0.7697      0.04308      718347519    7
k_factor_rf=8,nprobe=64,quantizer_efSearch=64     0.6091 0.7820 0.7824      0.04830      716119630    7
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6225 0.8022 0.8026      0.06087     1414932248    5
k_factor_rf=16,nprobe=64,quantizer_efSearch=64    0.6328 0.8297 0.8306      0.06817      716119630    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=32   0.6349 0.8359 0.8367      0.07423     1421072118    5
k_factor_rf=16,nprobe=64,quantizer_efSearch=128   0.6356 0.8340 0.8349      0.07868      714928414    4
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6500 0.8573 0.8582      0.07998     1414932248    4
k_factor_rf=16,nprobe=128,quantizer_efSearch=128  0.6542 0.8641 0.8650      0.09088     1411214300    4
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6605 0.8759 0.8766      0.11430     2781241814    3
k_factor_rf=16,nprobe=512,quantizer_efSearch=128  0.6616 0.8766 0.8772      0.14753     5474540803    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6832 0.9222 0.9234      0.15538     2781241814    2
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.6852 0.9258 0.9270      0.16929     2775481018    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=64   0.6856 0.9338 0.9355      0.20509     2792402069    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=128  0.6934 0.9461 0.9479      0.21628     2781241814    2
k_factor_rf=64,nprobe=256,quantizer_efSearch=256  0.6954 0.9498 0.9516      0.23949     2775481018    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=128 0.6962 0.9527 0.9544      0.31045    10714747266    1
k_factor_rf=64,nprobe=512,quantizer_efSearch=512  0.6990 0.9589 0.9606      0.30903     5445623200    1
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.7002 0.9603 0.9620      0.34309    10710216681    1
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7008 0.9616 0.9633      0.40115    10675229342    1
```

</details>
<details><summary>`OPQ16_64,IVF4194304_HNSW32,PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 74805056402

 code_size 64

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_HNSW32_PQ16x4fs_Refine_OPQ56_112_PQ56_.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                  0.5696 0.7278 0.7282      0.03348      194028923    9
k_factor_rf=1,nprobe=2,quantizer_efSearch=4       0.1620 0.1830 0.1830      0.00540        6266692    56
k_factor_rf=1,nprobe=2,quantizer_efSearch=8       0.1925 0.2159 0.2159      0.00663        6260865    46
k_factor_rf=1,nprobe=8,quantizer_efSearch=4       0.2957 0.3393 0.3393      0.00698       25024568    43
k_factor_rf=1,nprobe=8,quantizer_efSearch=8       0.3058 0.3512 0.3512      0.00740       25007984    41
k_factor_rf=2,nprobe=8,quantizer_efSearch=4       0.3157 0.3742 0.3743      0.00863       25024568    35
k_factor_rf=1,nprobe=16,quantizer_efSearch=4      0.3248 0.3735 0.3735      0.00797       49735553    38
k_factor_rf=1,nprobe=32,quantizer_efSearch=4      0.3380 0.3878 0.3878      0.00952       98584321    32
k_factor_rf=2,nprobe=16,quantizer_efSearch=4      0.3580 0.4249 0.4250      0.00977       49735553    31
k_factor_rf=1,nprobe=32,quantizer_efSearch=8      0.3811 0.4392 0.4392      0.01089       98689194    28
k_factor_rf=2,nprobe=16,quantizer_efSearch=8      0.3968 0.4722 0.4723      0.01145       49649160    27
k_factor_rf=2,nprobe=16,quantizer_efSearch=16     0.4160 0.4985 0.4986      0.01285       49490438    24
k_factor_rf=2,nprobe=32,quantizer_efSearch=8      0.4239 0.5035 0.5036      0.01341       98689194    23
k_factor_rf=4,nprobe=16,quantizer_efSearch=16     0.4376 0.5378 0.5379      0.01711       49490438    18
k_factor_rf=4,nprobe=32,quantizer_efSearch=8      0.4531 0.5570 0.5571      0.01794       98689194    17
k_factor_rf=2,nprobe=32,quantizer_efSearch=32     0.4743 0.5677 0.5678      0.01876       97860824    16
k_factor_rf=4,nprobe=32,quantizer_efSearch=16     0.4897 0.6050 0.6051      0.01963       98298725    16
k_factor_rf=4,nprobe=64,quantizer_efSearch=16     0.5154 0.6375 0.6377      0.02370      195014415    13
k_factor_rf=4,nprobe=64,quantizer_efSearch=32     0.5416 0.6728 0.6730      0.02717      194028923    12
k_factor_rf=8,nprobe=64,quantizer_efSearch=32     0.5696 0.7278 0.7282      0.03690      194028923    9
k_factor_rf=8,nprobe=128,quantizer_efSearch=32    0.5879 0.7563 0.7566      0.04463      383866776    7
k_factor_rf=8,nprobe=128,quantizer_efSearch=64    0.6074 0.7833 0.7836      0.05146      381682669    6
k_factor_rf=16,nprobe=128,quantizer_efSearch=32   0.6079 0.7984 0.7992      0.06413      383866776    5
k_factor_rf=8,nprobe=256,quantizer_efSearch=64    0.6153 0.7940 0.7943      0.06486      753800943    5
k_factor_rf=16,nprobe=128,quantizer_efSearch=64   0.6271 0.8272 0.8280      0.06962      381682669    5
k_factor_rf=16,nprobe=256,quantizer_efSearch=64   0.6417 0.8484 0.8490      0.08518      753800943    4
k_factor_rf=16,nprobe=256,quantizer_efSearch=128  0.6525 0.8638 0.8644      0.09841      749194681    4
k_factor_rf=32,nprobe=256,quantizer_efSearch=64   0.6578 0.8823 0.8836      0.12169      753800943    3
k_factor_rf=16,nprobe=512,quantizer_efSearch=128  0.6589 0.8749 0.8755      0.12552     1476144005    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=128  0.6684 0.8991 0.9004      0.13442      749194681    3
k_factor_rf=32,nprobe=256,quantizer_efSearch=256  0.6726 0.9054 0.9067      0.14917      746380334    3
k_factor_rf=32,nprobe=512,quantizer_efSearch=128  0.6778 0.9160 0.9171      0.15446     1476144005    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=256  0.6851 0.9285 0.9296      0.18031     1467587702    2
k_factor_rf=32,nprobe=512,quantizer_efSearch=512  0.6856 0.9289 0.9300      0.23228     1462838812    2
k_factor_rf=32,nprobe=1024,quantizer_efSearch=256 0.6889 0.9336 0.9347      0.23467     2885346689    2
k_factor_rf=64,nprobe=512,quantizer_efSearch=256  0.6942 0.9515 0.9532      0.25263     1467587702    2
k_factor_rf=64,nprobe=1024,quantizer_efSearch=256 0.6998 0.9611 0.9627      0.30168     2885346689    1
k_factor_rf=64,nprobe=1024,quantizer_efSearch=512 0.7009 0.9637 0.9654      0.35919     2869548940    1
```

</details>
<details><summary>`OPQ16_64,IVF4194304(IVF1024,PQ32x4fs,RFlat),PQ16x4fs,Refine(OPQ56_112,PQ56)` </summary>
Index size 73764737238

 code_size 64

 log filename: autotune.dbbigann1B.OPQ16_64_IVF4194304_IVF1024_PQ32x4fs_RFlat__PQ16x4fs_Refine_OPQ56_112_PQ56_.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                         0.4263 0.5305 0.5309      0.02842       54916514    11
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1        0.0830 0.0904 0.0904      0.00429        4553888    70
k_factor_rf=1,nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=2        0.0946 0.1039 0.1039      0.00481        6013442    63
k_factor_rf=1,nprobe=2,quantizer_k_factor_rf=1,quantizer_nprobe=2        0.1399 0.1543 0.1543      0.00503        9143877    60
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=1        0.2169 0.2437 0.2437      0.00556       26159784    54
k_factor_rf=1,nprobe=4,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.2397 0.2710 0.2710      0.00643       18131343    47
k_factor_rf=1,nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=4        0.3013 0.3439 0.3439      0.00720       30485555    42
k_factor_rf=1,nprobe=16,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.3169 0.3605 0.3605      0.00802       55100802    38
k_factor_rf=1,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=2       0.3195 0.3621 0.3621      0.00932      100744884    33
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=1,quantizer_nprobe=4       0.3908 0.4593 0.4594      0.01230      103875413    25
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=8       0.4441 0.5283 0.5284      0.01509      108959560    20
k_factor_rf=2,nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.4625 0.5507 0.5508      0.01853      119516167    17
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=4       0.4779 0.5852 0.5854      0.02484      199047653    13
k_factor_rf=2,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=16     0.4960 0.5898 0.5899      0.02788      404891094    11
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16      0.5411 0.6675 0.6676      0.02664      215052387    12
k_factor_rf=4,nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=16      0.5505 0.6816 0.6818      0.03249      214420705    10
k_factor_rf=8,nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16      0.5758 0.7328 0.7332      0.03703      214905748    9
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=1,quantizer_nprobe=32     0.5838 0.7421 0.7423      0.04893      425278421    7
k_factor_rf=8,nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32     0.6138 0.7919 0.7922      0.06350      422862714    5
k_factor_rf=8,nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32     0.6232 0.8063 0.8066      0.07434      790087865    5
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=16    0.6348 0.8355 0.8361      0.08061      771857949    4
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=16    0.6350 0.8351 0.8357      0.09866     1504419994    4
k_factor_rf=16,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6519 0.8635 0.8641      0.10994      911784676    3
k_factor_rf=32,nprobe=256,quantizer_k_factor_rf=8,quantizer_nprobe=16    0.6542 0.8749 0.8762      0.13471      770555258    3
k_factor_rf=16,nprobe=512,quantizer_k_factor_rf=8,quantizer_nprobe=32    0.6600 0.8781 0.8787      0.15379     1508445817    2
k_factor_rf=32,nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=16    0.6616 0.8897 0.8908      0.15482     1491689373    2
k_factor_rf=64,nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=32    0.6712 0.9092 0.9109      0.18541      791690201    2
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=2,quantizer_nprobe=128   0.6925 0.9458 0.9475      0.22410     1631389823    2
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=32   0.6926 0.9493 0.9509      0.34922     5663967195    1
k_factor_rf=64,nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=256  0.6963 0.9529 0.9546      0.35990     1790508002    1
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=64  0.7008 0.9635 0.9650      1.00432     5685481663    1
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=128 0.7019 0.9660 0.9675      1.05989     5759276299    1
k_factor_rf=64,nprobe=2048,quantizer_k_factor_rf=32,quantizer_nprobe=512 0.7021 0.9661 0.9676      1.22078     6243656201    1
```

</details>
<details><summary>`OPQ56_112,IVF1048576_HNSW32,PQ7+56` </summary>
Index size 71763799516

 code_size 63

 log filename: autotune.dbbigann1B.OPQ56_112_IVF1048576_HNSW32_PQ7+56.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.4698 0.5820 0.5820      0.03267              0    10
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.2949 0.3457 0.3457      0.01293              0    24
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=1       0.3966 0.4786 0.4786      0.01970              0    16
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=2       0.4125 0.5088 0.5089      0.02470              0    13
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=2      0.4411 0.5539 0.5540      0.03413              0    9
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.4538 0.5584 0.5584      0.03057              0    10
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.4698 0.5820 0.5820      0.03256              0    10
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.4823 0.5990 0.5990      0.03801              0    8
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.4967 0.6331 0.6332      0.03785              0    8
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=4     0.5124 0.6629 0.6630      0.04836              0    7
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=4     0.5249 0.6818 0.6819      0.05316              0    6
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.5379 0.6963 0.6964      0.05776              0    6
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.5514 0.7157 0.7158      0.06175              0    5
nprobe=64,quantizer_efSearch=16,ht=56,k_factor=4     0.5780 0.7721 0.7726      0.10105              0    3
nprobe=32,quantizer_efSearch=128,ht=56,k_factor=4    0.5793 0.7716 0.7720      0.09773              0    4
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.5796 0.7632 0.7635      0.10341              0    3
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=8     0.5811 0.7816 0.7822      0.09324              0    4
nprobe=32,quantizer_efSearch=128,ht=56,k_factor=8    0.5889 0.7959 0.7966      0.11367              0    3
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.6229 0.8577 0.8589      0.13166              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.6416 0.8948 0.8964      0.23825              0    2
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=16   0.6460 0.9134 0.9155      0.19397              0    2
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=16  0.6495 0.9208 0.9229      0.25177              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.6610 0.9442 0.9470      0.41484              0    1
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=16  0.6612 0.9449 0.9477      0.45211              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.6651 0.9603 0.9641      0.48768              0    1
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=32  0.6655 0.9611 0.9649      0.54979              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=64  0.6705 0.9772 0.9824      0.87191              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=64 0.6729 0.9838 0.9891      1.38978              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=64 0.6734 0.9846 0.9902      1.67557              0    1
```

</details>
<details><summary>`OPQ56_112,IVF4194304_HNSW32,PQ7+56` </summary>
Index size 74054325212

 code_size 63

 log filename: autotune.dbbigann1B.OPQ56_112_IVF4194304_HNSW32_PQ7+56.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                     0.1726 0.1984 0.1984      0.00953              0    32
nprobe=1,quantizer_efSearch=4,ht=56,k_factor=1       0.1206 0.1342 0.1342      0.00840              0    36
nprobe=2,quantizer_efSearch=4,ht=56,k_factor=1       0.1726 0.1984 0.1984      0.00956              0    32
nprobe=2,quantizer_efSearch=8,ht=56,k_factor=1       0.2084 0.2390 0.2390      0.01131              0    27
nprobe=4,quantizer_efSearch=4,ht=56,k_factor=1       0.2257 0.2674 0.2674      0.01150              0    27
nprobe=8,quantizer_efSearch=4,ht=56,k_factor=1       0.3242 0.3965 0.3965      0.01622              0    19
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=1       0.3388 0.4155 0.4155      0.01680              0    18
nprobe=8,quantizer_efSearch=8,ht=56,k_factor=2       0.3454 0.4284 0.4287      0.02133              0    15
nprobe=16,quantizer_efSearch=4,ht=56,k_factor=1      0.3642 0.4565 0.4565      0.02315              0    13
nprobe=16,quantizer_efSearch=8,ht=56,k_factor=1      0.4053 0.5125 0.5125      0.02480              0    13
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=1     0.4205 0.5346 0.5347      0.02684              0    12
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=2     0.4341 0.5624 0.5628      0.03160              0    10
nprobe=16,quantizer_efSearch=32,ht=56,k_factor=1     0.4364 0.5561 0.5561      0.03276              0    10
nprobe=16,quantizer_efSearch=16,ht=56,k_factor=4     0.4388 0.5739 0.5746      0.04097              0    8
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=1     0.4637 0.6000 0.6001      0.04102              0    8
nprobe=32,quantizer_efSearch=16,ht=56,k_factor=2     0.4851 0.6446 0.6452      0.04594              0    7
nprobe=32,quantizer_efSearch=32,ht=56,k_factor=2     0.5011 0.6686 0.6695      0.05067              0    6
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=2     0.5083 0.6793 0.6802      0.06139              0    5
nprobe=32,quantizer_efSearch=64,ht=56,k_factor=4     0.5176 0.7035 0.7049      0.07020              0    5
nprobe=64,quantizer_efSearch=32,ht=56,k_factor=2     0.5323 0.7266 0.7278      0.07541              0    4
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=2     0.5410 0.7410 0.7421      0.08459              0    4
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=4     0.5555 0.7803 0.7820      0.09356              0    4
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=4    0.5591 0.7858 0.7876      0.13295              0    3
nprobe=64,quantizer_efSearch=64,ht=56,k_factor=8     0.5611 0.7962 0.7983      0.11131              0    3
nprobe=128,quantizer_efSearch=32,ht=56,k_factor=4    0.5621 0.8008 0.8027      0.11289              0    3
nprobe=64,quantizer_efSearch=128,ht=56,k_factor=8    0.5648 0.8017 0.8040      0.12378              0    3
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=4    0.5804 0.8320 0.8340      0.13824              0    3
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=4   0.5867 0.8426 0.8449      0.16442              0    2
nprobe=128,quantizer_efSearch=64,ht=56,k_factor=8    0.5887 0.8561 0.8588      0.16789              0    2
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=8   0.5946 0.8671 0.8704      0.17944              0    2
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=8   0.5960 0.8691 0.8725      0.20916              0    2
nprobe=128,quantizer_efSearch=128,ht=56,k_factor=16  0.5985 0.8784 0.8828      0.22729              0    2
nprobe=128,quantizer_efSearch=256,ht=56,k_factor=16  0.6000 0.8805 0.8851      0.25029              0    2
nprobe=256,quantizer_efSearch=64,ht=56,k_factor=16   0.6044 0.8971 0.9018      0.28660              0    2
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=16  0.6153 0.9229 0.9283      0.32622              0    1
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=16  0.6158 0.9235 0.9288      0.40317              0    1
nprobe=256,quantizer_efSearch=256,ht=56,k_factor=32  0.6165 0.9289 0.9360      0.44156              0    1
nprobe=256,quantizer_efSearch=512,ht=56,k_factor=32  0.6170 0.9294 0.9366      0.46502              0    1
nprobe=512,quantizer_efSearch=128,ht=56,k_factor=16  0.6184 0.9349 0.9409      0.45465              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=16  0.6234 0.9438 0.9507      0.49387              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=16  0.6237 0.9451 0.9518      0.54618              0    1
nprobe=512,quantizer_efSearch=256,ht=56,k_factor=32  0.6248 0.9538 0.9630      0.57169              0    1
nprobe=512,quantizer_efSearch=512,ht=56,k_factor=32  0.6253 0.9548 0.9640      0.62274              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=16 0.6260 0.9517 0.9585      0.82566              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=16 0.6273 0.9551 0.9622      0.87702              0    1
nprobe=1024,quantizer_efSearch=256,ht=56,k_factor=32 0.6283 0.9645 0.9742      0.87985              0    1
nprobe=1024,quantizer_efSearch=512,ht=56,k_factor=64 0.6297 0.9716 0.9829      1.15109              0    1
nprobe=2048,quantizer_efSearch=512,ht=56,k_factor=64 0.6315 0.9765 0.9885      1.75454              0    1
```

</details>
<details><summary>`OPQ56_112,IVF4194304(IVF1024,PQ56x4fs,RFlat),PQ7+56` </summary>
Index size 73064805280

 code_size 63

 log filename: autotune.dbbigann1B.OPQ56_112_IVF4194304_IVF1024_PQ56x4fs_RFlat__PQ7+56.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                            0.5921 0.8613 0.8633      0.31704       43417738    1
nprobe=1,quantizer_k_factor_rf=1,quantizer_nprobe=1,ht=56,k_factor=1        0.0880 0.0983 0.0983      0.00537        1443706    56
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=56,k_factor=1        0.1997 0.2345 0.2345      0.00733        1447299    41
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=56,k_factor=1        0.2465 0.2937 0.2937      0.00818        2884657    37
nprobe=8,quantizer_k_factor_rf=2,quantizer_nprobe=2,ht=56,k_factor=2        0.3005 0.3692 0.3692      0.01315        2880705    23
nprobe=16,quantizer_k_factor_rf=8,quantizer_nprobe=4,ht=56,k_factor=1       0.3999 0.5036 0.5036      0.01782        5664753    17
nprobe=16,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=56,k_factor=1      0.4006 0.5047 0.5047      0.02085        5695242    15
nprobe=16,quantizer_k_factor_rf=2,quantizer_nprobe=8,ht=56,k_factor=2       0.4272 0.5517 0.5518      0.02284       11191876    14
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=56,k_factor=2       0.4375 0.5664 0.5666      0.02334       11237398    13
nprobe=16,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=56,k_factor=2      0.4458 0.5814 0.5816      0.02644       22043093    12
nprobe=32,quantizer_k_factor_rf=2,quantizer_nprobe=32,ht=56,k_factor=1      0.4766 0.6234 0.6234      0.03400       42654434    9
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=16,ht=56,k_factor=1     0.4793 0.6275 0.6275      0.03436       22001840    9
nprobe=64,quantizer_k_factor_rf=2,quantizer_nprobe=16,ht=56,k_factor=1      0.5007 0.6611 0.6611      0.04466       21665406    7
nprobe=32,quantizer_k_factor_rf=8,quantizer_nprobe=16,ht=56,k_factor=8      0.5135 0.7068 0.7082      0.05605       22039944    6
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=56,k_factor=4     0.5148 0.7083 0.7091      0.05749       83328732    6
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=4      0.5545 0.7851 0.7864      0.06507       42869933    5
nprobe=64,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=56,k_factor=8      0.5598 0.8014 0.8034      0.07748       42011167    4
nprobe=128,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=2     0.5609 0.7824 0.7835      0.09135       81756484    4
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=64,ht=56,k_factor=4     0.5836 0.8418 0.8436      0.11168       84176612    3
nprobe=128,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=56,k_factor=16    0.5932 0.8761 0.8800      0.13894       42983696    3
nprobe=256,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=56,k_factor=8     0.6033 0.8980 0.9020      0.17839       42785897    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=32,ht=56,k_factor=8    0.6035 0.8983 0.9024      0.20788       43417738    2
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=56,k_factor=8    0.6064 0.9064 0.9106      0.20807       85070606    2
nprobe=256,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32    0.6124 0.9246 0.9309      0.26000       82409686    2
nprobe=512,quantizer_k_factor_rf=4,quantizer_nprobe=64,ht=56,k_factor=8     0.6137 0.9217 0.9262      0.29936       82345673    2
nprobe=512,quantizer_k_factor_rf=1,quantizer_nprobe=128,ht=56,k_factor=16   0.6147 0.9313 0.9371      0.32314      161821134    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=64,ht=56,k_factor=32   0.6227 0.9556 0.9634      0.45546       85070606    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=16   0.6240 0.9563 0.9630      0.53994       85070606    1
nprobe=1024,quantizer_k_factor_rf=4,quantizer_nprobe=512,ht=56,k_factor=16  0.6255 0.9590 0.9656      0.66017      654590151    1
nprobe=1024,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=32  0.6278 0.9702 0.9787      0.71303      329449940    1
nprobe=1024,quantizer_k_factor_rf=32,quantizer_nprobe=128,ht=56,k_factor=32 0.6279 0.9698 0.9783      0.94378      167194871    1
nprobe=2048,quantizer_k_factor_rf=2,quantizer_nprobe=64,ht=56,k_factor=32   0.6291 0.9732 0.9819      1.06004       85070606    1
nprobe=2048,quantizer_k_factor_rf=8,quantizer_nprobe=256,ht=56,k_factor=32  0.6297 0.9752 0.9841      1.24319      329449940    1
nprobe=2048,quantizer_k_factor_rf=4,quantizer_nprobe=256,ht=56,k_factor=64  0.6302 0.9813 0.9919      1.26195      329449940    1
```

</details>
<details><summary>`OPQ64_128,IVF1048576_HNSW32,PQ64` </summary>
Index size 72830818224

 code_size 64

 log filename: autotune.dbbigann1B.OPQ64_128_IVF1048576_HNSW32_PQ64.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.7913 0.9022 0.9022      0.47081     1426680833    1
nprobe=1,quantizer_efSearch=4,ht=56       0.0001 0.0001 0.0001      0.00608       11626660    50
nprobe=1,quantizer_efSearch=4,ht=202      0.0626 0.0644 0.0644      0.00616       11626660    49
nprobe=1,quantizer_efSearch=4,ht=232      0.1735 0.1818 0.1818      0.00687       11626660    44
nprobe=1,quantizer_efSearch=16,ht=228     0.1899 0.1980 0.1980      0.01137       11521135    27
nprobe=4,quantizer_efSearch=4,ht=218      0.2573 0.2720 0.2720      0.01619       46332337    19
nprobe=2,quantizer_efSearch=8,ht=230      0.2749 0.2891 0.2891      0.01210       23181185    25
nprobe=4,quantizer_efSearch=4,ht=228      0.3183 0.3390 0.3390      0.01720       46332337    18
nprobe=4,quantizer_efSearch=16,ht=236     0.4136 0.4408 0.4408      0.02329       46113575    13
nprobe=8,quantizer_efSearch=8,ht=242      0.4957 0.5381 0.5381      0.03844       92583897    8
nprobe=16,quantizer_efSearch=32,ht=224    0.5459 0.5928 0.5928      0.06653      182918523    5
nprobe=16,quantizer_efSearch=32,ht=232    0.6052 0.6651 0.6651      0.06983      182918523    5
nprobe=16,quantizer_efSearch=16,ht=246    0.6178 0.6833 0.6833      0.07821      183696323    4
nprobe=16,quantizer_efSearch=64,ht=234    0.6186 0.6812 0.6812      0.07931      182506337    4
nprobe=16,quantizer_efSearch=128,ht=236   0.6279 0.6920 0.6920      0.09622      182362688    4
nprobe=32,quantizer_efSearch=32,ht=230    0.6698 0.7438 0.7438      0.11794      363838574    3
nprobe=32,quantizer_efSearch=32,ht=234    0.6889 0.7690 0.7690      0.12072      363838574    3
nprobe=32,quantizer_efSearch=16,ht=250    0.6915 0.7750 0.7750      0.15125      365163552    2
nprobe=32,quantizer_efSearch=128,ht=240   0.7163 0.8033 0.8033      0.15416      362405783    2
nprobe=32,quantizer_efSearch=256,ht=250   0.7244 0.8135 0.8135      0.20613      362271048    2
nprobe=64,quantizer_efSearch=256,ht=236   0.7696 0.8689 0.8689      0.26871      716805802    2
nprobe=64,quantizer_efSearch=64,ht=250    0.7786 0.8846 0.8846      0.30006      718519024    2
nprobe=64,quantizer_efSearch=256,ht=246   0.7833 0.8897 0.8897      0.31729      716805802    1
nprobe=128,quantizer_efSearch=64,ht=236   0.7960 0.9065 0.9065      0.44416     1420651270    1
nprobe=128,quantizer_efSearch=64,ht=238   0.8002 0.9137 0.9137      0.44495     1420651270    1
nprobe=128,quantizer_efSearch=64,ht=250   0.8103 0.9297 0.9297      0.53789     1420651270    1
nprobe=128,quantizer_efSearch=256,ht=256  0.8193 0.9402 0.9402      0.64841     1414674969    1
nprobe=256,quantizer_efSearch=256,ht=240  0.8350 0.9602 0.9602      0.88846     2785921805    1
nprobe=512,quantizer_efSearch=128,ht=240  0.8386 0.9677 0.9677      1.61460     5500318668    1
nprobe=512,quantizer_efSearch=128,ht=242  0.8416 0.9721 0.9721      1.42032     5500318668    1
nprobe=1024,quantizer_efSearch=256,ht=240 0.8476 0.9805 0.9805      2.94925    10764790786    1
nprobe=2048,quantizer_efSearch=512,ht=250 0.8569 0.9956 0.9956      7.18776    21027871703    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304_HNSW32,PQ64` </summary>
Index size 75322670512

 code_size 64

 log filename: autotune.dbbigann1B.OPQ64_128_IVF4194304_HNSW32_PQ64.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                          0.7416 0.8363 0.8363      0.34508      390557595    1
nprobe=1,quantizer_efSearch=4,ht=232      0.1198 0.1233 0.1233      0.00597        3196901    51
nprobe=2,quantizer_efSearch=8,ht=230      0.2031 0.2106 0.2106      0.01001        6364627    30
nprobe=4,quantizer_efSearch=4,ht=226      0.2229 0.2343 0.2343      0.01250       12753467    25
nprobe=4,quantizer_efSearch=4,ht=228      0.2299 0.2420 0.2420      0.01262       12753467    24
nprobe=4,quantizer_efSearch=16,ht=236     0.3191 0.3378 0.3378      0.01985       12663805    16
nprobe=8,quantizer_efSearch=8,ht=242      0.3974 0.4262 0.4262      0.02979       25464161    11
nprobe=16,quantizer_efSearch=16,ht=246    0.5335 0.5804 0.5804      0.05813       50371916    6
nprobe=16,quantizer_efSearch=128,ht=236   0.5366 0.5823 0.5823      0.09390       49727568    4
nprobe=32,quantizer_efSearch=8,ht=238     0.5574 0.6090 0.6090      0.09490      100393538    4
nprobe=32,quantizer_efSearch=32,ht=230    0.5722 0.6251 0.6251      0.09920       99487063    4
nprobe=32,quantizer_efSearch=32,ht=234    0.6012 0.6600 0.6600      0.10074       99487063    3
nprobe=32,quantizer_efSearch=16,ht=250    0.6176 0.6821 0.6821      0.10555       99958890    3
nprobe=32,quantizer_efSearch=128,ht=240   0.6438 0.7075 0.7075      0.13404       98693523    3
nprobe=32,quantizer_efSearch=256,ht=250   0.6545 0.7232 0.7232      0.17975       98579331    2
nprobe=64,quantizer_efSearch=256,ht=246   0.7229 0.8100 0.8100      0.25443      195118450    2
nprobe=128,quantizer_efSearch=64,ht=238   0.7526 0.8457 0.8457      0.35454      388095972    1
nprobe=128,quantizer_efSearch=64,ht=250   0.7706 0.8728 0.8728      0.37586      388095972    1
nprobe=128,quantizer_efSearch=256,ht=256  0.7841 0.8903 0.8903      0.44655      385020900    1
nprobe=256,quantizer_efSearch=256,ht=240  0.8090 0.9195 0.9195      0.71828      758338565    1
nprobe=512,quantizer_efSearch=128,ht=240  0.8193 0.9367 0.9367      1.30100     1501517298    1
nprobe=512,quantizer_efSearch=128,ht=242  0.8227 0.9433 0.9433      1.29112     1501517298    1
nprobe=1024,quantizer_efSearch=256,ht=240 0.8380 0.9609 0.9609      2.10379     2934858148    1
nprobe=2048,quantizer_efSearch=512,ht=512 0.8557 0.9916 0.9916      3.13607     5727887628    1
```

</details>
<details><summary>`OPQ64_128,IVF4194304(IVF1024,PQ64x4fs,RFlat),PQ64` </summary>
Index size 74350072308

 code_size 64

 log filename: autotune.dbbigann1B.OPQ64_128_IVF4194304_IVF1024_PQ64x4fs_RFlat__PQ64.d.stdout

```
parameters                                   R@1   R@10  R@100  time(ms/q)   nb distances #runs
                                                                 0.1620 0.1681 0.1681      0.07147      318445064    5
nprobe=1,quantizer_k_factor_rf=4,quantizer_nprobe=1,ht=512       0.1170 0.1219 0.1219      0.00400        4613801    76
nprobe=1,quantizer_k_factor_rf=8,quantizer_nprobe=2,ht=244       0.1422 0.1473 0.1473      0.00584        6033578    52
nprobe=1,quantizer_k_factor_rf=32,quantizer_nprobe=4,ht=240      0.1510 0.1563 0.1563      0.00944        8820970    32
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=2,ht=226       0.2361 0.2483 0.2483      0.01102       15520109    28
nprobe=4,quantizer_k_factor_rf=16,quantizer_nprobe=2,ht=244      0.2831 0.3009 0.3009      0.01305       15502426    23
nprobe=4,quantizer_k_factor_rf=4,quantizer_nprobe=8,ht=232       0.3046 0.3230 0.3230      0.01809       23794440    17
nprobe=8,quantizer_k_factor_rf=1,quantizer_nprobe=2,ht=242       0.3124 0.3332 0.3332      0.01974       28241197    16
nprobe=8,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=244      0.4413 0.4742 0.4742      0.03131       47267687    10
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=4,ht=254     0.5029 0.5479 0.5479      0.05531       55661716    6
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=8,ht=246     0.5354 0.5854 0.5854      0.05553       60977092    6
nprobe=16,quantizer_k_factor_rf=64,quantizer_nprobe=128,ht=238   0.5417 0.5908 0.5908      0.08877      210496937    4
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=16,ht=236    0.6179 0.6787 0.6787      0.10110      121076197    3
nprobe=32,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=242   0.6430 0.7115 0.7115      0.11029      262459817    3
nprobe=32,quantizer_k_factor_rf=64,quantizer_nprobe=64,ht=248    0.6493 0.7201 0.7201      0.11908      182781928    3
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=16,ht=252     0.7112 0.7975 0.7975      0.15544      218033594    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=32,ht=252     0.7213 0.8100 0.8100      0.16083      238888278    2
nprobe=64,quantizer_k_factor_rf=4,quantizer_nprobe=128,ht=254    0.7220 0.8117 0.8117      0.18537      362507048    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=246    0.7756 0.8801 0.8801      0.29288      428823674    2
nprobe=128,quantizer_k_factor_rf=8,quantizer_nprobe=32,ht=252    0.7789 0.8857 0.8857      0.30613      426101084    1
nprobe=256,quantizer_k_factor_rf=16,quantizer_nprobe=256,ht=512  0.8179 0.9391 0.9391      0.42362     1086117263    1
nprobe=512,quantizer_k_factor_rf=16,quantizer_nprobe=128,ht=244  0.8305 0.9574 0.9574      1.16008     1652059131    1
nprobe=1024,quantizer_k_factor_rf=2,quantizer_nprobe=128,ht=512  0.8487 0.9854 0.9854      1.23161     3081073224    1
nprobe=2048,quantizer_k_factor_rf=64,quantizer_nprobe=512,ht=512 0.8548 0.9955 0.9955      3.94201     6335515537    1
```

</details>
