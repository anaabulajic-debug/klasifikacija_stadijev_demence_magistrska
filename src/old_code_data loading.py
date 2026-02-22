
# =============================================================================
# 0) Load raw datasets
# =============================================================================
lethe_raw = pd.read_csv("lethe_raw.csv")
nacc_raw  = pd.read_csv("nacc_raw.csv", low_memory=False)
print("Loaded LETHE:", lethe_raw.shape, "  NACC:", nacc_raw.shape)

to_num = lambda s: pd.to_numeric(s, errors="coerce")

# Before dedup: how many visits per participant and what is the earliest visit_month?
lethe_raw = pd.read_csv("lethe_raw.csv")

lethe_raw["participant_id"] = lethe_raw["subject_ID"].astype(str).str.strip()
lethe_raw["visit_month"] = pd.to_numeric(lethe_raw["visit_month"], errors="coerce")

visits_per_participant = lethe_raw.groupby("participant_id")["visit_month"].agg(
    n_visits="count",
    earliest="min",
    latest="max"
)

print(visits_per_participant["n_visits"].value_counts().sort_index())
print(visits_per_participant["earliest"].value_counts().sort_index().head(20))

# Identify ID and visit variables as in your pipeline
id_col     = next((c for c in ["NACCID","PTID","SUBJID","RID"] if c in nacc_raw.columns), None)
visitnum   = next((c for c in ["VISITNUM","VISIT","PACKET"] if c in nacc_raw.columns), None)
year_col   = next((c for c in ["VISITYR","VISYEAR","YEAR"] if c in nacc_raw.columns), None)
month_col  = next((c for c in ["VISITMO","VISMONTH","MONTH"] if c in nacc_raw.columns), None)

print("NACC ID column:", id_col)
print("NACC visitnum column:", visitnum)
print("NACC year/month cols:", year_col, month_col)

# Visits per participant using visitnum (if available)
if visitnum:
    nacc_raw[visitnum] = pd.to_numeric(nacc_raw[visitnum], errors="coerce")
    visits_per_nacc = nacc_raw.groupby(id_col)[visitnum].agg(
        n_visits="count",
        earliest="min",
        latest="max"
    )
    print("\nVisits per NACC participant (counts):")
    print(visits_per_nacc["n_visits"].value_counts().sort_index())

    print("\nEarliest visitnum per participant (first 20 values):")
    print(visits_per_nacc["earliest"].value_counts().sort_index().head(20))

# If you prefer to look at VISITYR/VISITMO instead:
if year_col and month_col:
    y = pd.to_numeric(nacc_raw[year_col], errors="coerce")
    m = pd.to_numeric(nacc_raw[month_col], errors="coerce")
    visits_per_year = nacc_raw.groupby(id_col)[[year_col, month_col]].agg(
        earliest_year=(year_col, "min"),
        earliest_month=(month_col, "min")
    )
    print("\nEarliest VISITYR per participant (first 20 values):")
    print(visits_per_year["earliest_year"].value_counts().sort_index().head(20))


# -------------------------------------------------------------------------
# LETHE: stable IDs and reduction to earliest visit per participant
# -------------------------------------------------------------------------
# Use the actual ID column in LETHE
id_col = "subject_ID"  # this exists in lethe_raw

lethe_raw["participant_id"] = lethe_raw[id_col].astype(str).str.strip()

# OPTIONAL: keep subject_ID as original ID as well
# lethe_raw["subject_ID"] = lethe_raw["participant_id"]

# Visit column
visit_col = "visit_month"  # this exists in lethe_raw

if lethe_raw["participant_id"].duplicated().any():
    # make sure visit_month is numeric
    lethe_raw[visit_col] = pd.to_numeric(lethe_raw[visit_col], errors="coerce")

    n0 = len(lethe_raw)
    lethe_raw = (
        lethe_raw
        .sort_values(["participant_id", visit_col])
        .drop_duplicates(subset=["participant_id"], keep="first")
    )
    print(
        f"[LETHE] Reduced to one row/participant using earliest {visit_col}: "
        f"{len(lethe_raw):,} (from {n0:,})."
    )
else:
    print("[LETHE] No duplicate participant_id values found.")

# Row-level ID for traceability
lethe_raw["row_id"] = "LETHE_" + lethe_raw["participant_id"]



# =============================================================================
# 1) NACC EARLY DERIVATIONS (on RAW, before renaming)
# =============================================================================
# AD-focused keep and label
cs = to_num(nacc_raw.get("NACCUDSD"))
if "NACCETPR" in nacc_raw.columns:
    et = to_num(nacc_raw["NACCETPR"])
    keep = (cs == 1) | (cs == 3) | ((cs == 4) & (et == 1))
    nacc_raw = nacc_raw.loc[keep].copy()
    cs_f, et_f = to_num(nacc_raw["NACCUDSD"]), to_num(nacc_raw["NACCETPR"])
    is_nci, is_mci, is_ad = (cs_f == 1), (cs_f == 3), ((cs_f == 4) & (et_f == 1))
    nacc_raw["Cog_status"] = np.where(is_nci, "no cognitive impairment",
                              np.where(is_mci, "MCI", "AD"))
else:
    print("[WARNING] NACCETPR not found — keeping only 1 (no CI) and 3 (MCI)")
    keep = cs.isin([1, 3])
    nacc_raw = nacc_raw.loc[keep].copy()
    nacc_raw["Cog_status"] = cs.map({1: "no cognitive impairment", 3: "MCI"})

# --- NACC: deduplicate to earliest visit (DATE > VISITNUM > FIRST) ---
id_col     = next((c for c in ["NACCID","PTID","SUBJID","RID"] if c in nacc_raw.columns), None)
visitnum   = next((c for c in ["VISITNUM","VISIT","PACKET"] if c in nacc_raw.columns), None)
year_col   = next((c for c in ["VISITYR","VISYEAR","YEAR"] if c in nacc_raw.columns), None)
month_col  = next((c for c in ["VISITMO","VISMONTH","MONTH"] if c in nacc_raw.columns), None)
day_col    = next((c for c in ["VISITDAY","VISDAY","DAY"] if c in nacc_raw.columns), None)

if id_col:
    df = nacc_raw.copy()

    # try date-first
    have_date = (year_col in df.columns) and (month_col in df.columns)
    if have_date:
        y = pd.to_numeric(df[year_col],  errors="coerce")
        m = pd.to_numeric(df[month_col], errors="coerce").clip(1,12)
        if day_col in df.columns:
            d = pd.to_numeric(df[day_col], errors="coerce").clip(1,31)
        else:
            d = pd.Series(15, index=df.index, dtype="float64")
        visitdate = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        df["_visitdate"] = visitdate
        n0 = len(df)
        df = (df.sort_values([id_col, "_visitdate"])
                .drop_duplicates(subset=[id_col], keep="first")
                .drop(columns=["_visitdate"]))
        print(f"[NACC] Reduced to one row/participant using VISIT DATE: {len(df):,} (from {n0:,}).")
    # else fall back to visit number if available
    elif visitnum:
        df[visitnum] = pd.to_numeric(df[visitnum], errors="coerce")
        n0 = len(df)
        df = (df.sort_values([id_col, visitnum])
                .drop_duplicates(subset=[id_col], keep="first"))
        print(f"[NACC] Reduced to one row/participant using {visitnum}: {len(df):,} (from {n0:,}).")
    else:
        n0 = len(df)
        df = df.drop_duplicates(subset=[id_col], keep="first")
        print(f"[NACC] Reduced to one row/participant (first per {id_col}): {len(df):,} (from {n0:,}).")

    nacc_raw = df
else:
    print("[NACC] Participant ID column not found — skipping dedup.")

# After the dedup block finishes and nacc_raw is the deduplicated frame
if id_col  in nacc_raw.columns:
    pid = nacc_raw[id_col].astype(str).str.strip()
    # fill any empty/NaN IDs with the current index as a last resort
    null_mask = pid.isna() | (pid == "")
    if null_mask.any():
        pid.loc[null_mask] = nacc_raw.index[null_mask].astype(str)
        print(f"[NACC] Filled {int(null_mask.sum())} missing/blank {id_col} with index.")
else:
    pid = nacc_raw.index.astype(str)
    print("[NACC] id_col not found after dedup — using index as participant_id.")

nacc_raw["participant_id"] = pid
nacc_raw["row_id"] = "NACC_" + nacc_raw["participant_id"]