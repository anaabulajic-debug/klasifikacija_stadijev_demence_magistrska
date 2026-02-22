import numpy as np
import pandas
import pandas as pd
from config import nacc_file, lethe_file
from data_loading import nacc_early_derivations

lethe_map = dict(Age="age",
                 Sex="sex",
                 Educ_years="educ_years",
                 Marital_status="marital_status",
                 Liv_alone="liv_alone",
                 Cog_status="cog_status",
                 Dem_binary="dem_binary",
                 CDR_total="cdr_total",
                 TMT_A="tmt_a",
                 TMT_B="tmt_b",
                 RAVLT_dela="ravlt_dela",
                 MMSE="mmse",
                 Height="height",
                 Weight="weight",
                 BMI="bmi",
                 DBP="dbp",
                 SBP="sbp",
                 ApoE_var1="apoe_var1",
                 ApoE_var2="apoe_var2",
                 Depr_GDS15="depr_gds15",
                 curr_med_total="curr_med_total",
                 antihypertensives="antihypertensives",
                 antidepressants="antidepressants",
                 antidiabetics="antidiabetics",
                 lipid_drugs="lipid_drugs",
                 antipsychotics="antipsychotics",
                 stroke="stroke",
                 PD="pd",
                 #diabetes_any="diabetes_any",
                 cancer="cancer",
                 heart_bypass="heart_bypass",
                 depression="depression",
                 hypertension="hypertension",
                 PTSD="ptsd",
                 thyroid_dis="thyroid_dis",
                 sleep_apnea="sleep_apnea",
                 epilepsy="epilepsy",
                 dem_fam_hist="dem_fam_hist",
                 Smoke_comb="smoke_comb",
                 diabetes_t1="diabetes_t1",
                 diabetes_t2="diabetes_t2",
                 RAVLT_imm="ravlt_imm",
                 sleep_dis="sleep_dis",
                 rheuma_arth="rheuma_arth",
                 participant_id="participant_id",
                 row_id="row_id")

nacc_map = dict(NACCAGEB="age",
                SEX="sex",
                EDUC="educ_years",
                MARISTAT="marital_status",
                NACCLIVS="liv_alone",
                Cog_status="cog_status",
                DEMENTED="dem_binary",
                CDRGLOB="cdr_total",
                TRAILA="tmt_a",
                TRAILB="tmt_b",
                REYDREC="ravlt_dela",
                NACCMMSE="mmse",
                HEIGHT="height",
                WEIGHT="weight",
                NACCBMI="bmi",
                BPDIAS="dbp",
                BPSYS="sbp",
                NACCAPOE="apoe_var1",
                NACCNE4S="apoe_var2",
                NACCGDS="depr_gds15",
                NACCAMD="curr_med_total",
                NACCAHTN="antihypertensives",
                NACCADEP="antidepressants",
                NACCDBMD="antidiabetics",
                NACCLIPL="lipid_drugs",
                NACCAPSY="antipsychotics",
                CBSTROKE="stroke",
                PD="pd",
                CANCER="cancer",
                CVBYPASS="heart_bypass",
                DEP="depression",
                HYPERT="hypertension",
                PTSD="ptsd",
                THYDIS="thyroid_dis",
                SLEEPAP="sleep_apnea",
                EPILEP="epilepsy",
                NACCFAM="dem_fam_hist",
                Smoke_comb="smoke_comb",
                diabetes_t1="diabetes_t1",
                diabetes_t2="diabetes_t2",
                #diabetes_any = "diabetes_any",
                RAVLT_imm="ravlt_imm",
                sleep_dis="sleep_dis",
                rheuma_arth="rheuma_arth",
                participant_id="participant_id",
                row_id="row_id")

def recode (nacc, lethe):

    """
    nacc["cog_status"]=np.select(
        [
            nacc["cog_status"] == 1,
            nacc["cog_status"] == 3,
            nacc["cog_status"] == 4
        ],
        [
            0, #normal
            2, #MCI
            3 #AD
        ]
    )
    """
    lethe["cog_status"] = np.select(
        [
            lethe["cog_status"] == 0,
            lethe["cog_status"] == 2,
            lethe["cog_status"] == 3,
        ],
        [
            0,
            1,
            2
        ]
    )

    #df je nacc
    nacc["height"] *= 2.54
    nacc["weight"] *= 0.45359237
    # mogoče dodaj DBP pa SBP checks


    nacc["sex"] = np.select(
        [
            nacc["sex"] == 1,
            nacc["sex"] == 2
        ],
        [
            0,
            1
        ])

#marital status
    nacc["marital_status"] = np.select(
        [
            (nacc["marital_status"] <= 4),
            (nacc["marital_status"] == 5) | (nacc["marital_status"] == 6),
            (nacc["marital_status"] == 9)
        ],
        [
            0, #not married (single, widowed, divorced
            1, #married (or cohabiting)
            np.nan
        ])

#liv_alone
    nacc["liv_alone"] = np.select(
        [
            (nacc["liv_alone"] == 1),
            (nacc["liv_alone"] == 2) | (nacc["liv_alone"] == 3) | (nacc["liv_alone"] == 4),
            (nacc["liv_alone"].isin([5, 9]))
        ],
        [
            1, #living alone
            2, #living w someone
           # 3 #living in an institution - ni podatkov v nacc, smo dal stran tud v lethe
            3
        ])

#CDR mapping
    lethe["cdr_total"] = np.select(
        [
            (lethe["cdr_total"] == 1),
            (lethe["cdr_total"] == 2),
            (lethe["cdr_total"] == 3),
            (lethe["cdr_total"] == 4),
            (lethe["cdr_total"] == 5)
        ],
        [
            0, #no dementia
            0.5, #0.5 questionable impairment
            1, #MCI
            2, #moderate cog impairment
            3 #severe cog impairment
        ]
    )

#nared tmt_a pa tmt_b
    nacc["tmt_a"] = np.select(
        [
            (nacc["tmt_a"]<150),
            (nacc["tmt_a"].isin([995, 996, 997, 998, -4]))
        ],
        [
            (nacc["tmt_a"]),
            np.nan
            ]
    )

    nacc["tmt_b"] = np.select(
        [
            (nacc["tmt_a"]<300),
            (nacc["tmt_a"].isin([995, 996, 997, 998, -4]))
        ],
        [
            (nacc["tmt_b"]),
            np.nan
            ]
    )

    nacc["ravlt_dela"] = np.select(
        [
            (nacc["ravlt_dela"] < 15),
            (nacc["ravlt_dela"].isin([88, 95, 96, 97, 98, -4]))
        ],
        [
            (nacc["ravlt_dela"]),
            np.nan
        ]
    )

    # APOE dosage
    nacc["apoe_var1"] = np.select(
        [
            (nacc["apoe_var1"] == 6),
            (nacc["apoe_var1"] == 3),
            (nacc["apoe_var1"] == 5),
            (nacc["apoe_var1"] == 1),
            (nacc["apoe_var1"] == 2),
            (nacc["apoe_var1"] == 4),
            (nacc["apoe_var1"] == 9)
        ],
        [
            1, #2.2
            2, #2.3
            3, #2.4
            4, #3.3
            5, # 3.4
            6,#4.4
            np.nan
        ]
    )

    #nacc apoe var2
    nacc["apoe_var2"] = np.select(
        [
            (nacc["apoe_var2"] == 0),
            (nacc["apoe_var2"] == 1) | (nacc["apoe_var2"] == 2),
            (nacc["apoe_var2"] == 9)
        ],
        [
            0, #no
            1, #yes
            np.nan
        ]
    )
#dem_binary (mislm da kasnej izklučm ker leakage)

#meds_binaries
    nacc["curr_med_total"] = np.select(
        [
            (nacc["curr_med_total"] < 40),
            (nacc["curr_med_total"] == -4)
        ],
        [
            nacc["curr_med_total"],
            np.nan
        ]
    )

    nacc["antihypertensives"] = np.select(
        [
            (nacc["antihypertensives"] == 0),
            (nacc["antihypertensives"] == 1),
            (nacc["antihypertensives"] == -4)
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["antidepressants"] = np.select(
        [
            (nacc["antidepressants"] == 0),
            (nacc["antidepressants"] == 1),
            (nacc["antidepressants"] == -4)
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["antidiabetics"] = np.select(
        [
            (nacc["antidiabetics"] == 0),
            (nacc["antidiabetics"] == 1),
            (nacc["antidiabetics"] == -4)
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["lipid_drugs"] = np.select(
        [
            (nacc["lipid_drugs"] == 0),
            (nacc["lipid_drugs"] == 1),
            (nacc["lipid_drugs"] == -4)
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["antipsychotics"] = np.select(
        [
            (nacc["antipsychotics"] == 0),
            (nacc["antipsychotics"] == 1),
            (nacc["antipsychotics"] == -4)
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["stroke"] = np.select(
        [
            (nacc["stroke"] == 0),
            (nacc["stroke"] == 1) | (nacc["stroke"] == 2),
            (nacc["stroke"].isin([9, -4]))
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["pd"] = np.select(
        [
            (nacc["pd"] == 0),
            (nacc["pd"] == 1),
            (nacc["pd"].isin([9, -4]))
        ],
        [
            0,
            1,
            np.nan
        ]
    )
#condition diagnoses_binary
#cancer
    nacc["cancer"] = np.select(
        [
            (nacc["cancer"] == 0),
            (nacc["cancer"] == 1) | (nacc["cancer"] == 2)
        ],
        [
            0,
            1
        ]
    )

    nacc["heart_bypass"] = np.select(
        [
            (nacc["heart_bypass"] == 0),
            (nacc["heart_bypass"] == 1) | (nacc["heart_bypass"] == 2),
            (nacc["heart_bypass"].isin([9, -4]))
            ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["hypertension"] = np.select(
        [
            (nacc["hypertension"] == 0),
            (nacc["hypertension"] == 1) | (nacc["hypertension"] == 2),
            (nacc["hypertension"].isin([9, -4]))
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["ptsd"] = np.select(
        [
            (nacc["ptsd"] == 0),
            (nacc["ptsd"] == 1) | (nacc["ptsd"] == 2),
            (nacc["ptsd"].isin([9, -4]))
            ],
        [
            0,
            1,
            np.nan
            ]
    )

    nacc["thyroid_dis"] = np.select(
        [
            (nacc["thyroid_dis"] == 0),
            (nacc["thyroid_dis"] == 1),
            (nacc["thyroid_dis"].isin([8, -4]))
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["sleep_apnea"] = np.select(
        [
            (nacc["sleep_apnea"] == 0),
            (nacc["sleep_apnea"] == 1),
            (nacc["sleep_apnea"].isin([8, -4]))
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    nacc["epilepsy"] = np.select(
        [
            (nacc["epilepsy"] == 0),
            (nacc["epilepsy"] == 1),
            (nacc["epilepsy"] == -4)
        ],
        [
            0,
            1,
            np.nan
        ]
    )

    #lethe.replace("NA", np.nan, inplace=True)



def unify_dataset(lethe, nacc):
    lethe.rename(columns=lethe_map, inplace=True)
    nacc.rename(columns=nacc_map, inplace=True)

    print("AFTER rename")
    print("lethe columns:", repr(lethe.columns.tolist()))
    print("nacc  columns:", repr(nacc.columns.tolist()))

    recode(nacc, lethe)
    print(nacc_map.values())
    print (lethe_map.values())

    print("AFTER recode")
    print("lethe columns:", repr(lethe.columns.tolist()))
    print("nacc  columns:", repr(nacc.columns.tolist()))

    both = pd.concat([lethe, nacc])
    print("Concatenated dfs:", both.head(10))
    return both

#sex
    print(lethe['sex'].unique())
    for e in lethe["sex"].unique():
        print(e, len(lethe[lethe["sex"] == e]))
        print(lethe[lethe["sex"] == e]["height"].mean())
        print(lethe[lethe["sex"] == e]["weight"].mean())

#   both = pd.concat([lethe, nacc])
  #  print(both.head(100))
   # return both
