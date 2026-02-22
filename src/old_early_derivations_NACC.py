
# =============================================================================
# 2) EARLY DERIVATIONS (NACC): smoke, diabetes subtypes, strict RAVLT imm, sleep_dis, rheuma
# =============================================================================
def clean_rey_raw(s):
    s = to_num(s).replace({88:np.nan,95:np.nan,96:np.nan,97:np.nan,98:np.nan,-4:np.nan})
    return s.where(s.between(0,15), np.nan)

# Strict smoking (no optimistic "former")
t100 = to_num(nacc_raw.get("TOBAC100")).where(lambda x: x.isin([0,1]), np.nan)
t30  = to_num(nacc_raw.get("TOBAC30")).where(lambda x: x.isin([0,1]), np.nan)
smoke = pd.Series(np.nan, index=nacc_raw.index, dtype="object")
smoke.loc[t30 == 1] = "current"
smoke.loc[(t100 == 0) & (t30 == 0)] = "never"
smoke.loc[(t100 == 1) & (t30 == 0)] = "former"
nacc_raw["Smoke_comb"] = smoke.map({"never":0,"former":1,"current":2})

# Diabetes subtypes
diab_any  = to_num(nacc_raw.get("DIABETES")).where(lambda x: x.isin([0,1,2]), np.nan)
diab_type = to_num(nacc_raw.get("DIABTYPE")).replace({8:np.nan,9:np.nan,-4:np.nan})
diab_t1 = pd.Series(np.nan, index=nacc_raw.index)
diab_t2 = pd.Series(np.nan, index=nacc_raw.index)
diab_t1[diab_any == 0] = 0; diab_t2[diab_any == 0] = 0
mask_yes = diab_any.isin([1,2]) & diab_type.notna()
diab_t1[mask_yes] = np.where(diab_type[mask_yes] == 1, 1, 0)
diab_t2[mask_yes] = np.where(diab_type[mask_yes] == 2, 1, 0)
nacc_raw["diabetes_t1"], nacc_raw["diabetes_t2"] = diab_t1, diab_t2

# Safe composite for sleep_dis (works with any subset of subcomponents)
def safe_bin01(df, col):
    if col not in df.columns: return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s.where(s.isin([0,1]), np.nan)

sleep_sources = []
for c in ["SLEEPAP","REMDIS","HYPOSOM","SLEEPOTH"]:
    s = safe_bin01(nacc_raw, c)
    if s is not None:
        s = s.replace({8: np.nan, 9: np.nan, -4: np.nan})
        sleep_sources.append(s.rename(c))

if sleep_sources:
    stack = pd.concat(sleep_sources, axis=1)
    has_one = (stack == 1).any(axis=1)
    all_zero_known = (stack.fillna(0) == 0).all(axis=1) & stack.notna().any(axis=1)
    sleep_dis = pd.Series(np.nan, index=nacc_raw.index)
    sleep_dis[has_one] = 1
    sleep_dis[all_zero_known & ~has_one] = 0
    nacc_raw["sleep_dis"] = sleep_dis
else:
    print("[NACC] No sleep subcomponents found; 'sleep_dis' not created.")

# rheuma_arth only if ARTYPE exists
if "ARTYPE" in nacc_raw.columns:
    at = pd.to_numeric(nacc_raw["ARTYPE"], errors="coerce")
    rheu = pd.Series(np.nan, index=nacc_raw.index)
    rheu[at == 1] = 1
    rheu[at.isin([2,3,8])] = 0
    nacc_raw["rheuma_arth"] = rheu

# RAVLT imm strict (all 5 trials)
rey_cols = ["REY1REC","REY2REC","REY3REC","REY4REC","REY5REC"]
if all(c in nacc_raw.columns for c in rey_cols):
    rey_clean = [clean_rey_raw(nacc_raw[c]) for c in rey_cols]
    nacc_raw["RAVLT_imm"] = pd.concat(rey_clean, axis=1).sum(axis=1, min_count=5)
else:
    nacc_raw["RAVLT_imm"] = np.nan

# --- ultra-compact sanity checks ---
chk = {"Smoke_comb":[0,1,2], "diabetes_t1":[0,1], "diabetes_t2":[0,1], "sleep_dis":[0,1]}
for c, dom in chk.items():
    if c in nacc_raw.columns:
        s = pd.to_numeric(nacc_raw[c], errors="coerce")
        print(f"[{c}] dist:", s.value_counts(dropna=False).to_dict())
        bad = s.dropna()[~s.dropna().isin(dom)]
        assert bad.empty, f"{c} has illegal values: {bad.unique().tolist()[:5]}"

if "RAVLT_imm" in nacc_raw.columns:
    r = pd.to_numeric(nacc_raw["RAVLT_imm"], errors="coerce")
    print("[RAVLT_imm] range:", float(np.nanmin(r)), "→", float(np.nanmax(r)) if r.notna().any() else "n/a")
    assert r.dropna().between(0,75).all(), "RAVLT_imm out of [0,75]"

if {"TOBAC100","TOBAC30","Smoke_comb"}.issubset(nacc_raw.columns):
    t100 = pd.to_numeric(nacc_raw["TOBAC100"], errors="coerce")
    t30  = pd.to_numeric(nacc_raw["TOBAC30"],  errors="coerce")
    sc   = pd.to_numeric(nacc_raw["Smoke_comb"], errors="coerce")
    mism = ((t30==1)&(sc!=2)) | ((t100==0)&(t30==0)&(sc!=0)) | ((t100==1)&(t30==0)&(sc!=1))
    print("[Smoke_comb] mapping mismatches:", int(mism.fillna(False).sum()))
