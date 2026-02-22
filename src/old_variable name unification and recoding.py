
# =============================================================================
# 3) Reload derived files (strings) and subset+rename to common names
# =============================================================================
# Instead of reloading from CSV, just cast to str in-memory
lethe = lethe_raw.astype(str)
nacc  = nacc_raw.astype(str)
print("Reloaded derived (in-memory cast):", lethe.shape, nacc.shape)

lethe_map = {
    "Age":"age","Sex":"sex","Educ_years":"educ_years","Marital_status":"marital_status",
    "Liv_alone":"liv_alone","Cog_status":"cog_status","Dem_binary":"dem_binary","CDR_total":"cdr_total",
    "TMT_A":"tmt_a","TMT_B":"tmt_b","RAVLT_imm":"ravlt_imm","RAVLT_dela":"ravlt_dela","MMSE":"mmse",
    "Height":"height","Weight":"weight","BMI":"bmi","DBP":"dbp","SBP":"sbp",
    "ApoE_var1":"apoe_var1","ApoE_var2":"apoe_var2","Depr_GDS15":"depr_gds15",
    "Smoke_comb":"smoke_comb","curr_med_total":"curr_med_total","antihypertensives":"antihypertensives",
    "antidepressants":"antidepressants","antidiabetics":"antidiabetics","lipid_drugs":"lipid_drugs","antipsychotics":"antipsychotics",
    "stroke":"stroke","PD":"pd","diabetes_any":"diabetes_any","diabetes_t1":"diabetes_t1","diabetes_t2":"diabetes_t2",
    "cancer":"cancer","heart_bypass":"heart_bypass","depression":"depression","hypertension":"hypertension",
    "sleep_dis":"sleep_dis","PTSD":"ptsd","thyroid_dis":"thyroid_dis","sleep_apnea":"sleep_apnea","rheuma_arth":"rheuma_arth",
    "epilepsy":"epilepsy","dem_fam_hist":"dem_fam_hist",
    # IDs
    "participant_id":"participant_id","row_id":"row_id",
}
nacc_map = {
    "NACCAGEB":"age","SEX":"sex","EDUC":"educ_years","MARISTAT":"marital_status","NACCLIVS":"liv_alone",
    "Cog_status":"cog_status","DEMENTED":"dem_binary","CDRGLOB":"cdr_total",
    "TRAILA":"tmt_a","TRAILB":"tmt_b","REYDREC":"ravlt_dela","NACCMMSE":"mmse",
    "HEIGHT":"height","WEIGHT":"weight","NACCBMI":"bmi","BPDIAS":"dbp","BPSYS":"sbp",
    "NACCAPOE":"apoe_var1","NACCNE4S":"apoe_var2","NACCGDS":"depr_gds15",
    "NACCAMD":"curr_med_total","NACCAHTN":"antihypertensives","NACCADEP":"antidepressants",
    "NACCDBMD":"antidiabetics","NACCLIPL":"lipid_drugs","NACCAPSY":"antipsychotics",
    "CBSTROKE":"stroke","PD":"pd","DIABETES":"diabetes_any","CANCER":"cancer","CVBYPASS":"heart_bypass",
    "DEP":"depression","HYPERT":"hypertension","PTSD":"ptsd","THYDIS":"thyroid_dis",
    "SLEEPAP":"sleep_apnea","EPILEP":"epilepsy","NACCFAM":"dem_fam_hist",
    "Smoke_comb":"smoke_comb","diabetes_t1":"diabetes_t1","diabetes_t2":"diabetes_t2","RAVLT_imm":"ravlt_imm",
    "sleep_dis":"sleep_dis","rheuma_arth":"rheuma_arth",
    # IDs
    "participant_id":"participant_id","row_id":"row_id",
}

lethe = lethe[[c for c in lethe_map if c in lethe.columns]].rename(columns=lethe_map)
nacc  = nacc [[c for c in nacc_map  if c in nacc.columns ]].rename(columns=nacc_map)

# COG STAT NUM
def build_cog_status_num(s):
    out = pd.Series(pd.NA, index=s.index, dtype="Int8")
    sn = pd.to_numeric(s, errors="coerce")              # LETHE numeric 0/2/3
    out.loc[sn.isin([0,2,3])] = sn.map({0:0,2:1,3:2}).astype("Int8")
    ss = s.astype(str).str.lower()                      # NACC strings
    out.loc[ss.isin(["no cognitive impairment","mci","ad"])] = \
        ss.map({"no cognitive impairment":0,"mci":1,"ad":2}).astype("Int8")
    return out

for df in (lethe, nacc):
    if "cog_status" in df.columns:
        df["cog_status_num"] = build_cog_status_num(df["cog_status"])

# Restrict to rows with cog_status present (hard row filter)
def restrict_to_cog_present(df, name):
    if "cog_status_num" not in df.columns:
        raise RuntimeError(f"[{name}] Missing 'cog_status_num' — cannot restrict.")
    y = pd.to_numeric(df["cog_status_num"], errors="coerce")
    mask = y.notna()
    n0, n1 = len(df), int(mask.sum())
    print(f"[{name}] Restricting to cog_status present: {n1}/{n0} rows kept ({n1/n0:.1%}).")
    return df.loc[mask].copy()

lethe = restrict_to_cog_present(lethe, "LETHE")
nacc  = restrict_to_cog_present(nacc,  "NACC")

# =============================================================================
# 4) Vitals & BMI (metric), clamping, SBP<=DBP fix
# =============================================================================
for df in (lethe, nacc):
    for c in ["height","weight","bmi","sbp","dbp","age"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

# LETHE metric; NACC convert
lethe["height_cm"] = lethe.get("height")
lethe["weight_kg"] = lethe.get("weight")
if "height" in nacc.columns: nacc["height_cm"] = nacc["height"] * 2.54
if "weight" in nacc.columns: nacc["weight_kg"] = nacc["weight"] * 0.45359237

for df in (lethe, nacc):
    if "height_cm" in df.columns:
        df.loc[(df["height_cm"] < 120) | (df["height_cm"] > 220), "height_cm"] = np.nan
    if "weight_kg" in df.columns:
        df.loc[(df["weight_kg"] < 30) | (df["weight_kg"] > 200), "weight_kg"] = np.nan

def compute_bmi(df):
    if "height_cm" not in df.columns or "weight_kg" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    h_m = df["height_cm"] / 100
    out = df["weight_kg"] / (h_m ** 2)
    out[(h_m <= 0) | h_m.isna()] = np.nan
    return out

lethe["bmi"] = compute_bmi(lethe)
nacc["bmi"]  = compute_bmi(nacc)

def clamp(df, col, lo, hi):
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s.where((s >= lo) & (s <= hi))

for name, df in [("LETHE", lethe), ("NACC", nacc)]:
    clamp(df,"age",40,110)
    clamp(df,"sbp",70,260)
    clamp(df,"dbp",40,160)
    if {"sbp","dbp"}.issubset(df.columns):
        bad = df["sbp"].notna() & df["dbp"].notna() & (df["sbp"] <= df["dbp"])
        nbad = int(bad.sum())
        if nbad:
            print(f"{name}: SBP<=DBP in {nbad} rows → setting both to NaN.")
            df.loc[bad, ["sbp","dbp"]] = np.nan

# =============================================================================
# 5) Core recodes + diabetes consistency
# =============================================================================
num = lambda x: pd.to_numeric(x, errors="coerce")

# sex: 1=male,2=female (LETHE sometimes 0/1)
for df, name in [(lethe,"LETHE"), (nacc,"NACC")]:
    if "sex" in df.columns:
        s = num(df["sex"])
        df["sex"] = (s.map({0:2,1:1,2:2}) if name=="LETHE" else s).where(lambda x: x.isin([1,2]), np.nan)

# marital (cohort-specific)
def harmonize_marital(df, cohort):
    if "marital_status" not in df.columns: return df
    s = num(df["marital_status"])
    if cohort=="LETHE":
        s = s.where(s.isin([0,1]), np.nan)
    else:
        s = s.map({1:1,6:1,2:0,3:0,4:0,5:0,9:np.nan})
    df["marital_status"] = s.astype("Int8")
    return df
lethe = harmonize_marital(lethe,"LETHE")
nacc  = harmonize_marital(nacc,"NACC")

# liv_alone harmonization
def harmonize_liv_alone(df, cohort):
    if "liv_alone" not in df.columns:
        return df

    s_raw = pd.to_numeric(df["liv_alone"], errors="coerce")
    s = s_raw.copy()

    if cohort == "LETHE":
        # Assume LETHE already uses {1=alone, 2=with others, 3=institution}
        unexpected = s.dropna()[~s.isin([1,2,3])].unique()
        if len(unexpected):
            print(f"[WARN][LETHE] unexpected liv_alone codes: {sorted(map(int, unexpected))} → set to NaN")
        s = s.where(s.isin([1,2,3]), np.nan)

    else:  # NACC
        nacc_map = {1:1, 2:2, 3:2, 4:3, 5:2, 9:np.nan}  # 2/3/5 → 'with others'
        mapped = s.map(nacc_map)
        unexpected = s.dropna()[~s.dropna().astype(int).isin(nacc_map.keys())].unique()
        if len(unexpected):
            print(f"[WARN][NACC] unexpected liv_alone codes: {sorted(map(int, unexpected))} → set to NaN")
            mapped = mapped.where(s.dropna().astype(int).isin(nacc_map.keys()), np.nan)
        s = mapped

    df["liv_alone"] = pd.to_numeric(s, errors="coerce").astype("Int8")
    return df

lethe = harmonize_liv_alone(lethe, "LETHE")
nacc  = harmonize_liv_alone(nacc,  "NACC")

# CDR mapping to 1..5
cdr_map = {0.0:1,0.5:2,1.0:3,2.0:4,3.0:5}
for df in (lethe,nacc):
    if "cdr_total" in df.columns:
        c = num(df["cdr_total"])
        needs = c.isin(list(cdr_map.keys()))
        c.loc[needs] = c.loc[needs].map(cdr_map)
        df["cdr_total"] = c.where(c.isin([1,2,3,4,5]), np.nan)

# dem_binary (QC-only; cast to Int8; sentinel-aware)
for name, df in [("LETHE", lethe), ("NACC", nacc)]:
    if "dem_binary" in df.columns:
        s = pd.to_numeric(df["dem_binary"], errors="coerce") \
               .replace({8: pd.NA, 9: pd.NA, -4: pd.NA})
        s = s.where(s.isin([0, 1]), pd.NA).astype("Int8")
        df["dem_binary"] = s
        if "cog_status_num" in df.columns:
            y = pd.to_numeric(df["cog_status_num"], errors="coerce")
            bad = int(((y == 2) & (s == 0)).sum())
            print(f"[{name}] QC: y=AD but dem_binary=0 -> {bad} rows")

# chair_stand recode (will drop later)
for df in (lethe,nacc):
    if "chair_stand" in df.columns:
        s = num(df["chair_stand"])
        df["chair_stand"] = s.map({0:1,1:2,2:2,3:2,4:2,8:np.nan}).where(lambda x: x.isin([1,2]), np.nan)

# meds binaries
med_cols = ["antihypertensives","antidepressants","antidiabetics","lipid_drugs","antipsychotics"]
for df in (lethe,nacc):
    for c in med_cols:
        if c in df.columns:
            s = num(df[c]).replace({-4:np.nan})
            df[c] = s.where(s.isin([0,1]), np.nan)

# diagnoses -> 0/1/NaN
def recode_bin(df, col, yes, no, na=()):
    if col not in df.columns: return
    s = num(df[col])
    mapped = np.where(s.isin(yes), 1, np.where(s.isin(no), 0, np.nan))
    df[col] = pd.Series(mapped, index=df.index)

for df in (lethe,nacc):
    recode_bin(df,"stroke",[1,2],[0],[9,-4])
    recode_bin(df,"pd",[1],[0],[9,-4])
    recode_bin(df,"diabetes_any",[1,2],[0],[])
    recode_bin(df,"cancer",[1,2],[0],[])
    recode_bin(df,"heart_bypass",[1,2],[0],[9,-4])
    recode_bin(df,"depression",[1],[0],[])
    recode_bin(df,"hypertension",[1,2],[0],[9,-4])
    recode_bin(df,"sleep_apnea",[1],[0],[8,-4])
    recode_bin(df,"epilepsy",[1],[0],[-4])
    recode_bin(df,"dem_fam_hist",[1],[0],[])
    recode_bin(df,"ptsd",[1,2],[0],[9,-4])
    recode_bin(df,"thyroid_dis",[1],[0],[8,-4])

# diabetes consistency
def fix_diabetes(df):
    for c in ["diabetes_any","diabetes_t1","diabetes_t2"]:
        if c in df.columns:
            df[c] = num(df[c]).where(lambda x: x.isin([0,1]), np.nan)
    if {"diabetes_any","diabetes_t1","diabetes_t2"}.issubset(df.columns):
        df.loc[df["diabetes_any"] == 0, ["diabetes_t1","diabetes_t2"]] = 0
        sub = (df["diabetes_t1"] == 1) | (df["diabetes_t2"] == 1)
        df.loc[sub, "diabetes_any"] = 1

fix_diabetes(lethe)
fix_diabetes(nacc)

# =============================================================================
# 6) APOE dosage (Int8) + neuropsych + education + curr_med_total
# =============================================================================
def finalize_apoe(df, cohort):
    _nan = lambda: pd.Series(np.nan, index=df.index)

    s1 = pd.to_numeric(df.get("apoe_var1"), errors="coerce")

    if cohort == "LETHE":
        # ApoE_var1: 1=2/2, 2=2/3, 3=2/4, 4=3/3, 5=3/4, 6=4/4  → ε4 dosage: 0,0,1,0,1,2
        dose1   = s1.map({1:0, 2:0, 3:1, 4:0, 5:1, 6:2})
        carrier = pd.to_numeric(df.get("apoe_var2"), errors="coerce").where(lambda x: x.isin([0,1]), np.nan)

        df["_apoe_from_var1"]    = dose1
        df["_apoe_var2_carrier"] = carrier
        df["_apoe_var2_dosage"]  = _nan()

        out = dose1.copy()
        fb  = out.isna() & (carrier == 1)
        out.loc[fb] = 1

        both = dose1.notna() & carrier.notna()
        mismatch = ((dose1.eq(0) & (carrier == 1)) | (dose1.gt(0) & (carrier == 0))) & both
        df["qc_apoe_mismatch_var1_var2"] = mismatch.astype("Int8")

    elif cohort == "NACC":
        # NACCAPOE: 1=e3/e3, 2=e3/e4, 3=e3/e2, 4=e4/e4, 5=e4/e2, 6=e2/e2, 9=missing
        dose1 = s1.map({1:0, 2:1, 3:0, 4:2, 5:1, 6:0, 9:np.nan})
        dose2 = pd.to_numeric(df.get("apoe_var2"), errors="coerce").where(lambda x: x.isin([0,1,2]), np.nan)

        df["_apoe_from_var1"]    = dose1
        df["_apoe_var2_dosage"]  = dose2
        df["_apoe_var2_carrier"] = _nan()

        out = dose2.copy()
        fb  = out.isna() & dose1.notna()
        out.loc[fb] = dose1.loc[fb]

        both = dose1.notna() & dose2.notna()
        df["qc_apoe_mismatch_var1_var2"] = ((dose1 != dose2) & both).astype("Int8")

    else:
        raise ValueError("cohort must be 'LETHE' or 'NACC'")

    s = pd.to_numeric(out, errors="coerce")
    bad = s.notna() & ~s.isin([0,1,2])
    if bad.any():
        raise ValueError(f"[{cohort}] invalid APOE dosage values found: {s[bad].unique()[:5]}")
    df["apoe_e4_dosage"] = s.astype("Int8")
    return df

lethe = finalize_apoe(lethe, "LETHE")
nacc  = finalize_apoe(nacc,  "NACC")

# APOE summary / QC prints
for name, df, cohort in [("LETHE", lethe, "LETHE"), ("NACC", nacc, "NACC")]:
    vc = df["apoe_e4_dosage"].value_counts(dropna=False).sort_index()
    if cohort == "LETHE":
        both_count = int((df["_apoe_from_var1"].notna() & df["_apoe_var2_carrier"].notna()).sum())
    else:
        both_count = int((df["_apoe_from_var1"].notna() & df["_apoe_var2_dosage"].notna()).sum())
    mism = int(df.get("qc_apoe_mismatch_var1_var2", 0).sum())
    print(f"[{name}] ε4 dosage: {vc.to_dict()} | mismatches: {mism}/{both_count}")

# Drop raw APOE + helpers
for df in (lethe, nacc):
    df.drop(
        columns=[
            "apoe_var1","apoe_var2",
            "_apoe_from_var1","_apoe_var2_dosage","_apoe_var2_carrier"
        ],
        inplace=True, errors="ignore"
    )

# Neuropsych & education & curr_med_total cleaning
def clean_tmt(s, upper):
    s = pd.to_numeric(s, errors="coerce").replace({995:np.nan,996:np.nan,997:np.nan,998:np.nan,-4:np.nan})
    return s.where((s>=0) & (s<=upper), np.nan)

def clean_mmse(s):
    s = pd.to_numeric(s, errors="coerce").replace({88:np.nan,95:np.nan,96:np.nan,97:np.nan,98:np.nan,-4:np.nan})
    return s.where((s>=0) & (s<=30), np.nan)

def clean_ravlt_delayed(s):
    s = pd.to_numeric(s, errors="coerce").replace({88:np.nan,95:np.nan,96:np.nan,97:np.nan,98:np.nan,-4:np.nan})
    return s.where((s>=0) & (s<=15), np.nan)

def clean_ravlt_imm(s):
    s = pd.to_numeric(s, errors="coerce").replace({88:np.nan,95:np.nan,96:np.nan,97:np.nan,98:np.nan,-4:np.nan})
    return s.where((s>=0) & (s<=75), np.nan)

def clean_gds15(s):
    s = pd.to_numeric(s, errors="coerce").replace({88:np.nan,95:np.nan,96:np.nan,97:np.nan,98:np.nan,-4:np.nan})
    return s.where((s>=0) & (s<=15), np.nan)

for df in (lethe, nacc):
    if "depr_gds15" in df.columns:
        df["depr_gds15"] = clean_gds15(df["depr_gds15"])

for df in (lethe, nacc):
    if "tmt_a" in df.columns: df["tmt_a"] = clean_tmt(df["tmt_a"], 150)
    if "tmt_b" in df.columns: df["tmt_b"] = clean_tmt(df["tmt_b"], 300)
    if "mmse"  in df.columns: df["mmse"]  = clean_mmse(df["mmse"])
    if "ravlt_dela" in df.columns: df["ravlt_dela"] = clean_ravlt_delayed(df["ravlt_dela"])
    if "ravlt_imm"  in df.columns: df["ravlt_imm"]  = clean_ravlt_imm(df["ravlt_imm"])
    if "educ_years" in df.columns:
        df["educ_years"] = pd.to_numeric(df["educ_years"], errors="coerce").where(lambda e: (e>=0) & (e<=36), np.nan)
    if "curr_med_total" in df.columns:
        c = pd.to_numeric(df["curr_med_total"], errors="coerce").replace({-4:np.nan})
        df["curr_med_total"] = c.where((c>=0) & (c<=40), np.nan)

        # =============================================================================
        # 7.1) Canonicalize units & add cohort indicator (pre-freeze) + drop cog_status
        # =============================================================================
        for df in (lethe_aligned, nacc_aligned):
            if "height_cm" in df.columns and "height" in df.columns:
                df.drop(columns=["height"], inplace=True)
            if "weight_kg" in df.columns and "weight" in df.columns:
                df.drop(columns=["weight"], inplace=True)

        lethe_aligned["cohort"] = "LETHE"
        nacc_aligned["cohort"] = "NACC"

        for df in (lethe_aligned, nacc_aligned):
            df.drop(columns=["cog_status"], inplace=True, errors="ignore")

        # Med/diagnosis plausibility flags (flag-only; do not mutate)
        NEW_QC_BIN_FLAGS = []


        def _pct(n, d):
            return f"{(100.0 * n / d):.1f}%" if d else "n/a"


        def med_diag_flags_brief(df, name):
            created = []
            if {"antidiabetics", "diabetes_any"}.issubset(df.columns):
                f = (df["antidiabetics"] == 1) & (df["diabetes_any"] != 1)
                df["qc_med_antidiab_no_diag"] = f.astype("Int8");
                created.append("qc_med_antidiab_no_diag")
                base = int((df["antidiabetics"] == 1).sum());
                n = int(f.sum());
                nall = len(df)
                print(f"[{name}] antidiabetics→diabetes_any mismatch: {n} "
                      f"({_pct(n, base)} of med-users; {_pct(n, nall)} overall)")
            if {"antihypertensives", "hypertension"}.issubset(df.columns):
                f = (df["antihypertensives"] == 1) & (df["hypertension"] != 1)
                df["qc_med_ahtn_no_htn"] = f.astype("Int8");
                created.append("qc_med_ahtn_no_htn")
                base = int((df["antihypertensives"] == 1).sum());
                n = int(f.sum());
                nall = len(df)
                print(f"[{name}] antihypertensives→hypertension mismatch: {n} "
                      f"({_pct(n, base)} of med-users; {_pct(n, nall)} overall)")
            if {"antidepressants", "depression"}.issubset(df.columns):
                f = (df["antidepressants"] == 1) & (df["depression"] != 1)
                df["qc_med_ad_no_dep"] = f.astype("Int8");
                created.append("qc_med_ad_no_dep")
                base = int((df["antidepressants"] == 1).sum());
                n = int(f.sum());
                nall = len(df)
                print(f"[{name}] antidepressants→depression mismatch: {n} "
                      f"({_pct(n, base)} of med-users; {_pct(n, nall)} overall)")
            if "lipid_drugs" in df.columns:
                f = (df["lipid_drugs"] == 1)
                df["qc_med_lipid_drugs"] = f.astype("Int8");
                created.append("qc_med_lipid_drugs")
                n = int(f.sum());
                nall = len(df)
                print(f"[{name}] lipid_drugs present: {n} ({_pct(n, nall)} overall)")
            return created


        NEW_QC_BIN_FLAGS = sorted(set(
            med_diag_flags_brief(lethe_aligned, "LETHE") +
            med_diag_flags_brief(nacc_aligned, "NACC")
        ))

#######################################################################################################
    ###################################################################################################
    ###################################################################################################
    __________________________________________________________________________________________________

    # =============================================================================
    # EDA FIGURES — class balance, numeric, categorical
    # Colors: NACC = blue (#1f77b4), LETHE = orange (#ff7f0e)
    # =============================================================================

    # =============================================================================
    # EDA FIGURES FOR THESIS:
    #  - Class balance (target)
    #  - Numeric distributions (age, mmse, educ_years, sbp, dbp, bmi, curr_med_total)
    #  - Categorical distributions (sex, apoe_e4_dosage, liv_alone,
    #                              hypertension, antihypertensives, antidepressants)
    # Colours:
    #  - LETHE = orange
    #  - NACC  = blue
    #  - Overall (class balance) = sage green
    # =============================================================================
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path

    PLOT_DIR = Path(r"C:\Users\anaab\Desktop\1 urejeno magistrska\nove spyder scripts\no dropping")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    dfL = lethe_final_h.copy()
    dfN = nacc_final_h.copy()
    TARGET_COL = "cog_status_num"

    # Shared palette
    MAT_ORANGE = "#ff7f0e"  # LETHE
    MAT_BLUE = "#1f77b4"  # NACC
    MAT_SAGE = "#a3c9a8"  # Overall (light sage)

    # -----------------------------------------------------------------------------
    # 1) CLASS BALANCE (target)
    # -----------------------------------------------------------------------------
    label_map = {0: "NCI", 1: "MCI", 2: "AD"}

    counts_L = dfL[TARGET_COL].value_counts().reindex([0, 1, 2], fill_value=0)
    counts_N = dfN[TARGET_COL].value_counts().reindex([0, 1, 2], fill_value=0)
    counts_overall = counts_L + counts_N

    fig_cb, axes_cb = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    x = np.arange(3)
    labels = [label_map[i] for i in [0, 1, 2]]

    # Overall = sage
    ax = axes_cb[0]
    ax.bar(x, counts_overall.values, color=MAT_SAGE)
    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_title("Overall");
    ax.set_ylabel("Count")

    # LETHE = orange
    ax = axes_cb[1]
    ax.bar(x, counts_L.values, color=MAT_ORANGE)
    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_title("LETHE");
    ax.set_ylabel("Count")

    # NACC = blue
    ax = axes_cb[2]
    ax.bar(x, counts_N.values, color=MAT_BLUE)
    ax.set_xticks(x);
    ax.set_xticklabels(labels)
    ax.set_title("NACC");
    ax.set_ylabel("Count")

    fig_cb.suptitle("Class balance (target)", fontsize=16)
    fig_cb.savefig(PLOT_DIR / "01_class_balance.png", dpi=300)
    plt.close(fig_cb)
    print(f"[PLOTS] Saved class balance → {PLOT_DIR / '01_class_balance.png'}")


    # -----------------------------------------------------------------------------
    # Helper: numeric hist weights in percent
    # -----------------------------------------------------------------------------
    def _weights_percent(x):
        n = len(x)
        if n == 0:
            return None, "Percent"
        return np.ones(n) * 100.0 / n, "Percent"


    # -----------------------------------------------------------------------------
    # 2) NUMERIC DISTRIBUTIONS – 9 clinically most important variables
    #    (age, educ_years, mmse, ravlt_imm, tmt_b, sbp, dbp, bmi, curr_med_total)
    # -----------------------------------------------------------------------------
    numeric_clinical = [
        "age",
        "educ_years",
        "mmse",
        "ravlt_imm",
        "tmt_b",
        "sbp",
        "dbp",
        "bmi",
        "curr_med_total",
    ]

    # keep only those present in BOTH cohorts, preserve order, cap at 9
    numeric_vars = [
        v for v in numeric_clinical
        if v in dfL.columns and v in dfN.columns
    ][:9]

    print("Numeric vars in distribution panel:", numeric_vars)

    fig_num, axes_num = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes_num = axes_num.ravel()

    for i, var in enumerate(numeric_vars):
        ax = axes_num[i]
        xL = dfL[var].dropna().astype(float)
        xN = dfN[var].dropna().astype(float)

        if len(xL) == 0 and len(xN) == 0:
            ax.axis("off")
            continue

        all_vals = pd.concat([xL, xN], ignore_index=True)
        bins = np.histogram_bin_edges(all_vals, bins=30)

        wL, ylab = _weights_percent(xL)
        wN, _ = _weights_percent(xN)

        # NACC = blue outline
        ax.hist(xN, bins=bins, weights=wN,
                histtype="step", linewidth=2, color=MAT_BLUE)

        # LETHE = orange fill
        ax.hist(xL, bins=bins, weights=wL,
                alpha=0.5, color=MAT_ORANGE, edgecolor="none")

        ax.set_title(var)
        ax.set_ylabel("Percent")

    # hide any unused axes (if fewer than 9 survived)
    for j in range(len(numeric_vars), 9):
        axes_num[j].axis("off")

    fig_num.suptitle("Distribution checks\nNumeric overview (LETHE vs NACC)", fontsize=18)
    fig_num.savefig(PLOT_DIR / "02_distribution_numeric.png", dpi=300)
    plt.close(fig_num)

    print(f"[PLOTS] Saved numeric distribution panel → {PLOT_DIR / '02_distribution_numeric.png'}")

    # -----------------------------------------------------------------------------
    # 3) CATEGORICAL DISTRIBUTIONS
    #    Variables as per methods: sex, apoe_e4_dosage, liv_alone,
    #                               hypertension, antihypertensives, antidepressants
    # -----------------------------------------------------------------------------
    categorical_vars = [
        "sex", "apoe_e4_dosage", "liv_alone",
        "hypertension", "antihypertensives", "antidepressants"
    ]
    categorical_vars = [v for v in categorical_vars if v in dfL.columns and v in dfN.columns]

    fig_cat, axes_cat = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes_cat = axes_cat.ravel()

    for i, var in enumerate(categorical_vars):
        ax = axes_cat[i]

        vcL = dfL[var].value_counts(dropna=False)
        vcN = dfN[var].value_counts(dropna=False)

        cats = sorted(set(vcL.index).union(vcN.index), key=lambda x: str(x))
        pL = (vcL.reindex(cats, fill_value=0) / max(1, vcL.sum())) * 100
        pN = (vcN.reindex(cats, fill_value=0) / max(1, vcN.sum())) * 100

        x_pos = np.arange(len(cats))
        w = 0.4

        # LETHE = orange
        ax.bar(x_pos - w / 2, pL.values, width=w, color=MAT_ORANGE, alpha=0.9, label="LETHE")
        # NACC  = blue
        ax.bar(x_pos + w / 2, pN.values, width=w, color=MAT_BLUE, alpha=0.9, label="NACC")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(c) for c in cats], rotation=45, ha="right")
        ax.set_ylabel("Percent")
        ax.set_title(var)

    # hide unused axes if fewer than 6 vars
    for j in range(len(categorical_vars), 6):
        axes_cat[j].axis("off")

    handles, labels = axes_cat[0].get_legend_handles_labels()
    if handles:
        fig_cat.legend(handles, labels, loc="upper right")
    fig_cat.suptitle("Categorical", fontsize=18)

    fig_cat.savefig(PLOT_DIR / "03_categorical.png", dpi=300)
    plt.close(fig_cat)
    print(f"[PLOTS] Saved categorical panel → {PLOT_DIR / '03_categorical.png'}")

    # After dedup: confirm one row per participant
    print("LETHE after dedup:", len(lethe_raw), "rows,",
          lethe_raw["participant_id"].nunique(), "unique participants")

    # Optional: what does visit_month look like now?
    print(lethe_raw["visit_month"].value_counts().sort_index().head(20))

    # After your existing NACC dedup
    print("\nNACC after dedup:", len(nacc_raw), "rows,",
          nacc_raw[id_col].nunique(), "unique participants")

    if visitnum and visitnum in nacc_raw.columns:
        print("\nVisitnum distribution after dedup (first 20 values):")
        print(nacc_raw[visitnum].value_counts().sort_index().head(20))

    if year_col and year_col in nacc_raw.columns:
        print("\nVISITYR distribution after dedup (first 20 values):")
        print(pd.to_numeric(nacc_raw[year_col], errors="coerce").value_counts().sort_index().head(20))

    if month_col and month_col in nacc_raw.columns:
        print("\nVISITMO distribution after dedup (first 20 values):")
        print(pd.to_numeric(nacc_raw[month_col], errors="coerce").value_counts().sort_index().head(20))