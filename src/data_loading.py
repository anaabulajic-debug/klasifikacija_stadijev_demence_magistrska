import pandas as pd
import numpy as np
from config import nacc_file, lethe_file

def lethe_drop_bad(df):
    df.sort_values(["subject_ID", "visit_month"], inplace=True)
    df.drop_duplicates(subset="subject_ID", inplace=True, keep="first")

    df.drop(df[~df["Cog_status"].isin([0, 2, 3])].index, inplace=True)

def nacc_drop_bad(df):
    df.sort_values(["NACCID", "VISITMO"], inplace=True)
    df.drop_duplicates(subset="NACCID", inplace=True, keep="first")

    df.drop(
        df[
            (~df["NACCUDSD"].isin([1, 3])) &
            ~((df["NACCUDSD"] == 4) & (df["NACCETPR"] == 1))
            ].index,
        inplace=True
    )



def nacc_early_derivations(df):
    df["cog_status"] = np.select(
        [
            (df["NACCUDSD"] == 1),
            (df["NACCUDSD"] == 3),
            (df["NACCUDSD"] == 4) & (df["NACCETPR"] == 1)
        ],
        [
            0,
            2,
            3
        ]
    )

    #smoke, diabetes subtype, RAVLT
    df["Smoke_comb"] = np.select(
        [
            (df["TOBAC30"] == 1) & (df["TOBAC100"] == 1),
            (df["TOBAC30"] == 0) & (df["TOBAC100"] == 1),
            (df["TOBAC100"] == 0),
            (df["TOBAC100"] == 9) | (df["TOBAC100"] == -4) | (df["TOBAC30"] == 9) | (df["TOBAC30"] == -4 ),
        ],
        [
            1,#"smoker",
            2,#"former",
            0,#"never",
            np.nan,
        ])

    df["diabetes_any"] = np.select(
        [
            (df["DIABETES"] == 0),
            (df["DIABETES"] == 1) | (df["DIABETES"] == 2),
            (df["DIABETES"].isin([9, -4]))
        ],
        [
            0, #absent
            1, #diagnosed for any type of diabetes
            np.nan
        ]
    )

    df["diabetes_t1"] = np.select(
        [
            (df["DIABETES"]) == 1 & (df["DIABTYPE"] == 1),
            (df["DIABTYPE"] == 8) | (df["DIABTYPE"] == 9)| (df["DIABTYPE"] == -4),
        ],
        [
            1, #tip 1
            np.nan,
        ])


    df["diabetes_t2"] = np.select(
        [
            (df["DIABETES"]) == 1 & (df["DIABTYPE"] == 2),
            (df["DIABTYPE"] == 8) | (df["DIABTYPE"] == 9)| (df["DIABTYPE"] == -4),
        ],
        [
            1, #tip 2
            np.nan,
        ])

    """
    df["diabetes_other"] = np.select(
        [
            (df["DIABETES"]) == 1 & (df["DIABTYPE"] == 3),
            (df["DIABTYPE"] == 8) | (df["DIABTYPE"] == 9) | (df["DIABTYPE"] == -4),
        ],
        [
            1, #other
            np.nan
        ]
    )
    """

    df["RAVLT_imm"] = np.select(
        [
            (df["REY1REC"].between(0,15) & df["REY2REC"].between(0,15) & df["REY3REC"].between(0,15)
            & df["REY4REC"].between(0,15) & df["REY5REC"].between(0,15)),
            (df["REY1REC"].isin([88, 95, 96, 97, 98, -4]) | df["REY2REC"].isin([88, 95, 96, 97, 98, -4]) | df["REY3REC"].isin([88, 95, 96, 97, 98, -4]) |
            df["REY4REC"].isin([88, 95, 96, 97, 98, -4]) | df["REY5REC"].isin([88, 95, 96, 97, 98, -4]))
        ],
        [
            df["REY1REC"] + df["REY2REC"] + df["REY3REC"] + df[
                "REY4REC"] + df["REY5REC"],
            np.nan
        ]
    )

    #SLEEP_DIS, RHEUMA
    df["sleep_dis"] = np.select(
        [
            (df["SLEEPAP"] == 1) | (df["REMDIS"] == 1) | (df["HYPOSOM"] == 1) | (df["SLEEPOTH"] == 1),
            (df["SLEEPAP"] == 0) & (df["REMDIS"] == 0) & (df["HYPOSOM"] == 0) & (df["SLEEPOTH"] == 0),
            (df["SLEEPAP"].isin([8, -4])) | (df["REMDIS"].isin([8, -4])) |
            (df["HYPOSOM"].isin([8, -4])) | (df["SLEEPOTH"].isin([8, -4])),
        ],
        [
            1,  # sleep disorder present
            0,  # no sleep disorder
            np.nan,  # missing / unknown
        ],
    )

    df["rheuma_arth"] = np.select(
        [
            (df["ARTYPE"]) == 1,
            (df["ARTYPE"].isin([2,3,8,9,-4])),
        ],
    [
        1, #rheuma
        np.nan, #other types or missing
    ],
    )


def load_dataset():

    lethe = pd.read_csv(lethe_file, low_memory=False)
    nacc = pd.read_csv(nacc_file, low_memory=False)
    print("lethe len,shape: ",len(lethe),"\t",lethe.shape, "\nnacc len,shape: ",len(nacc),"\t",nacc.shape)

    lethe_drop_bad(lethe)
    nacc_drop_bad(nacc)
    nacc_early_derivations(nacc)
 #   print(nacc[["REY1REC", "REY2REC", "REY3REC"]].head(100))
    print("lethe len, shape: ",len(lethe),"\t",lethe.shape, "\nnacc len,shape: ",len(nacc),"\t",nacc.shape)
    return lethe, nacc