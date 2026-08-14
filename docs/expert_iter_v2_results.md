# Expert iteration v2 -- mix-experiment results (comparative stage)

Written by analyze_v2_mixes.py BEFORE narrative interpretation (prereg recording plan).
This stage SELECTS, never concludes; claims require the fresh-seed confirmation battery.

| mix | delta vs baseline | hier. SE | 95% CI | p (max-T adj) | candidate-eligible |
|---|---|---|---|---|---|
| a_nat60 | +0.1046 | 0.0026 | [+0.0965, +0.1128] | 1.0000 | yes |
| b_even50 | +0.1527 | 0.0129 | [+0.1116, +0.1939] | 1.0000 | yes |
| c_seed65 | +0.1425 | 0.0045 | [+0.1282, +0.1568] | 1.0000 | yes |
| d_natonly | +0.1554 | 0.0168 | [+0.1021, +0.2088] | 1.0000 | yes |
| e_seedspread | +0.1011 | 0.0044 | [+0.0869, +0.1152] | 1.0000 | yes |
| f_ct_nat50 | -0.0010 | 0.0046 | [-0.0155, +0.0135] | 0.8111 | NO (exploratory) |
| g_ct_2x50 | +0.0088 | 0.0041 | [-0.0041, +0.0217] | 0.9853 | NO (exploratory) |
| h_ct_nat100 | +0.0045 | 0.0023 | [-0.0028, +0.0118] | 0.9464 | NO (exploratory) |

Units: 6400 (block, match) pairs; CRN check: OK.

| contrast | delta | SE | verdict |
|---|---|---|---|
| a_nat60 - b_even50 | -0.0481 | 0.0076 | distinct |
| a_nat60 - c_seed65 | -0.0379 | 0.0077 | distinct |
| a_nat60 - d_natonly | -0.0508 | 0.0076 | distinct |
| a_nat60 - e_seedspread | +0.0036 | 0.0079 | indistinguishable |
| a_nat60 - f_ct_nat50 | +0.1057 | 0.0086 | distinct |
| a_nat60 - g_ct_2x50 | +0.0959 | 0.0086 | distinct |
| a_nat60 - h_ct_nat100 | +0.1002 | 0.0088 | distinct |
| b_even50 - c_seed65 | +0.0102 | 0.0074 | indistinguishable |
| b_even50 - d_natonly | -0.0027 | 0.0079 | indistinguishable |
| b_even50 - e_seedspread | +0.0517 | 0.0080 | distinct |
| b_even50 - f_ct_nat50 | +0.1537 | 0.0089 | distinct |
| b_even50 - g_ct_2x50 | +0.1439 | 0.0090 | distinct |
| b_even50 - h_ct_nat100 | +0.1482 | 0.0091 | distinct |
| c_seed65 - d_natonly | -0.0129 | 0.0079 | indistinguishable |
| c_seed65 - e_seedspread | +0.0414 | 0.0076 | distinct |
| c_seed65 - f_ct_nat50 | +0.1435 | 0.0088 | distinct |
| c_seed65 - g_ct_2x50 | +0.1337 | 0.0089 | distinct |
| c_seed65 - h_ct_nat100 | +0.1380 | 0.0090 | distinct |
| d_natonly - e_seedspread | +0.0544 | 0.0084 | distinct |
| d_natonly - f_ct_nat50 | +0.1564 | 0.0088 | distinct |
| d_natonly - g_ct_2x50 | +0.1466 | 0.0088 | distinct |
| d_natonly - h_ct_nat100 | +0.1509 | 0.0092 | distinct |
| e_seedspread - f_ct_nat50 | +0.1021 | 0.0087 | distinct |
| e_seedspread - g_ct_2x50 | +0.0923 | 0.0089 | distinct |
| e_seedspread - h_ct_nat100 | +0.0966 | 0.0089 | distinct |
| f_ct_nat50 - g_ct_2x50 | -0.0098 | 0.0073 | indistinguishable |
| f_ct_nat50 - h_ct_nat100 | -0.0055 | 0.0075 | indistinguishable |
| g_ct_2x50 - h_ct_nat100 | +0.0043 | 0.0073 | indistinguishable |
