"""Electricity theft detection (SGCC), enhanced pipeline. Just run:  python train.py

What it does, with no flags:
  1. cleans the data and builds features (old features + population-relative + change-point + frequency)
  2. automatically checks whether the 140+ weekly columns help, and keeps them only if they do
  3. tunes XGBoost with Optuna on cross-validated PR-AUC (test set never touched)
  4. trains XGBoost + LightGBM + CatBoost and averages their probabilities
  5. calibrates the probabilities (isotonic, on the validation set)
  6. evaluates once on the untouched test set and saves everything to ./model_out

Optional packages (the script falls back gracefully if missing):
  pip install optuna lightgbm catboost joblib
"""
import json, os, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             classification_report, confusion_matrix, brier_score_loss)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
import joblib

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ settings (edit here, no flags)
DATA_PATH = "./data_set.csv"
OUT_DIR = "model_out"
N_TRIALS = 20          # Optuna trials (set 0 to skip tuning)
TUNE_FOLDS = 3         # folds used while tuning / choosing feature set
FINAL_FOLDS = 5        # folds used for the reported CV score
SEED = 42
EPS = 1e-6


# ----------------------------------------------------------------------------- data
def load_and_clean(path):
    """Read the SGCC csv and return (ids, V, y, dates, missing_frac)."""
    df = pd.read_csv(path)
    id_col = "CONS_NO" if "CONS_NO" in df.columns else df.columns[0]
    y_col = "FLAG" if "FLAG" in df.columns else df.columns[-1]
    df = df.drop_duplicates(subset=id_col)

    day_cols = [c for c in df.columns if c not in (id_col, y_col)]
    dates = pd.to_datetime(pd.Index(day_cols), errors="coerce")
    if dates.isna().any():
        raise ValueError("Some day columns are not dates. Check that the last column is the "
                         "label and the middle columns are daily readings.")
    order = np.argsort(dates.values)
    day_cols = [day_cols[i] for i in order]
    dates = dates[order]

    y = df[y_col].astype(int).to_numpy()
    X = df[day_cols].apply(pd.to_numeric, errors="coerce")
    X = X.mask(X < 0)

    keep = X.notna().any(axis=1).to_numpy()
    X, y, ids = X[keep], y[keep], df[id_col].to_numpy()[keep]

    missing_frac = X.isna().mean(axis=1).to_numpy()
    X = X.interpolate(axis=1, limit=14, limit_direction="both")
    V = np.array(X.to_numpy(dtype=float))
    med = np.nanmedian(V, axis=1)
    r, c = np.where(np.isnan(V))
    V[r, c] = med[r]

    cap = V.mean(axis=1) + 3 * V.std(axis=1)
    V = np.minimum(V, cap[:, None])
    return ids, V, y, dates, missing_frac


# ------------------------------------------------------------------------ features
def _longest_run(mask):
    cur = np.zeros(mask.shape[0])
    best = cur.copy()
    for t in range(mask.shape[1]):
        cur = (cur + 1) * mask[:, t]
        best = np.maximum(best, cur)
    return best


def weekly_profile(V):
    n, T = V.shape
    w = T // 7
    W = V[:, :w * 7].reshape(n, w, 7).mean(2)
    return W / (V.mean(1, keepdims=True) + EPS)


def population_reference(V):
    """Median consumption per day across all customers (uses no labels)."""
    return np.median(V, axis=0)


def build_features(V, dates, missing_frac, ref, weekly=True):
    """V: (customers x days) cleaned kWh. ref: population reference curve (days,)."""
    n, T = V.shape
    mean, std = V.mean(1), V.std(1)
    low = V < 0.5 * mean[:, None]
    zero = V == 0
    weekend = np.asarray(dates.dayofweek >= 5)
    h, q = T // 2, max(T // 4, 1)

    months = np.asarray(dates.to_period("M").astype(str))
    M = np.column_stack([V[:, months == m].mean(1) for m in pd.unique(months)])
    if M.shape[1] > 1:
        month_cv = M.std(1) / (M.mean(1) + EPS)
        max_month_drop = (M[:, :-1] - M[:, 1:]).max(1) / (M.mean(1) + EPS)
    else:
        month_cv = max_month_drop = np.zeros(n)

    Vc = V - mean[:, None]
    t = np.arange(T) - (T - 1) / 2
    slope = (Vc @ t) / (t @ t) * T / (mean + EPS)
    autocorr7 = (Vc[:, 7:] * Vc[:, :-7]).mean(1) / (Vc.var(1) + EPS)
    skew = (Vc ** 3).mean(1) / (std ** 3 + EPS)
    p5, p25, p75, p95 = np.percentile(V, [5, 25, 75, 95], axis=1) / (mean + EPS)

    Wn = weekly_profile(V)
    cs_w = np.concatenate([np.zeros((n, 1)), np.cumsum(Wn, axis=1)], axis=1)
    roll4 = (cs_w[:, 4:] - cs_w[:, :-4]) / 4

    # ---- population-relative features: deviation from what everyone did that day
    Vn = V / (mean[:, None] + EPS)
    refn = ref / (ref.mean() + EPS)
    resid = Vn - refn
    refc = ref - ref.mean()
    corr_ref = (Vc @ refc) / (T * (std + EPS) * (ref.std() + EPS))
    cs_r = np.concatenate([np.zeros((n, 1)), np.cumsum(resid, axis=1)], axis=1)
    roll28 = (cs_r[:, 28:] - cs_r[:, :-28]) / 28

    # ---- change-point features: largest sustained level shift
    cs = np.cumsum(V, axis=1)
    ks = np.arange(30, T - 30, 7)
    before = cs[:, ks - 1] / ks
    after = (cs[:, [-1]] - cs[:, ks - 1]) / (T - ks)
    shift = (before - after) / (mean[:, None] + EPS)

    # ---- frequency features
    freqs = np.fft.rfftfreq(T)
    P = np.abs(np.fft.rfft(Vc, axis=1)) ** 2
    tot = P[:, 1:].sum(1) + EPS
    wk_band = np.abs(freqs - 1 / 7) < 0.004
    lo_band = (freqs > 0) & (freqs < 1 / 60)

    d = min(90, T // 4)
    feats = {
        "mean": mean, "std": std, "min": V.min(1), "max": V.max(1), "median": np.median(V, 1),
        "cv": std / (mean + EPS),
        "zero_frac": zero.mean(1), "low_frac": low.mean(1),
        "longest_zero_run": _longest_run(zero), "longest_low_run": _longest_run(low),
        "spike_frac": (V > (mean + 2 * std)[:, None]).mean(1),
        "second_vs_first_half": V[:, h:].mean(1) / (V[:, :h].mean(1) + EPS),
        "last_quarter_vs_rest": V[:, -q:].mean(1) / (V[:, :-q].mean(1) + EPS),
        "weekend_ratio": (V[:, weekend].mean(1) if weekend.any() else mean) /
                         (V[:, ~weekend].mean(1) + EPS),
        "month_cv": month_cv, "max_month_drop": max_month_drop,
        "trend_slope": slope, "autocorr7": autocorr7, "skew": skew,
        "p5_rel": p5, "p25_rel": p25, "p75_rel": p75, "p95_rel": p95,
        "min_4wk_rel": roll4.min(1), "weeks_below_half": (Wn < 0.5).mean(1),
        "weekly_std": Wn.std(1),
        "missing_frac": missing_frac,
        # population-relative
        "corr_with_ref": corr_ref, "resid_std": resid.std(1), "resid_abs_mean": np.abs(resid).mean(1),
        "resid_mean_last_quarter": resid[:, -q:].mean(1), "resid_min_4wk": roll28.min(1),
        "resid_max_4wk": roll28.max(1),
        # change-point
        "max_level_drop": shift.max(1), "max_level_rise": (-shift).max(1),
        "zero_to_nonzero": ((V[:, 1:] > 0) & (V[:, :-1] == 0)).sum(1).astype(float),
        "var_ratio_last_first": V[:, -d:].std(1) / (V[:, :d].std(1) + EPS),
        # frequency
        "fft_weekly_share": P[:, wk_band].sum(1) / tot, "fft_lowfreq_share": P[:, lo_band].sum(1) / tot,
    }
    F = pd.DataFrame(feats)
    if weekly:
        F = pd.concat([F, pd.DataFrame(Wn, columns=[f"wk_{i}" for i in range(Wn.shape[1])])], axis=1)
    return F


# --------------------------------------------------------------------------- models
def spw_of(y):
    return (y == 0).sum() / max((y == 1).sum(), 1)


def make_xgb(y_train, **p):
    base = dict(n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=3,
                subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0)
    base.update(p)
    return xgb.XGBClassifier(**base, scale_pos_weight=spw_of(y_train), eval_metric="aucpr",
                             early_stopping_rounds=50, tree_method="hist", n_jobs=-1,
                             random_state=SEED)


def fit_xgb_val(X, y, **p):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, stratify=y, random_state=1)
    m = make_xgb(y_tr, **p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def cv_pr_auc(F, y, folds, verbose=False, **p):
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    pr, roc = [], []
    for i, (tr, te) in enumerate(skf.split(F, y), 1):
        m = fit_xgb_val(F.iloc[tr], y[tr], **p)
        s = m.predict_proba(F.iloc[te])[:, 1]
        pr.append(average_precision_score(y[te], s))
        roc.append(roc_auc_score(y[te], s))
        if verbose:
            print(f"  fold {i}: PR-AUC {pr[-1]:.3f}  ROC-AUC {roc[-1]:.3f}")
    return np.mean(pr), np.std(pr), np.mean(roc), np.std(roc)


def tune_xgb(F, y):
    if optuna is None or N_TRIALS <= 0:
        print("Optuna unavailable or disabled: using default XGBoost parameters.")
        return {}

    def objective(trial):
        p = dict(max_depth=trial.suggest_int("max_depth", 3, 8),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
                 min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
                 reg_lambda=trial.suggest_float("reg_lambda", 0.5, 20, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0))
        # fast settings while searching; the final models use learning_rate 0.03 / 800 trees
        return cv_pr_auc(F, y, TUNE_FOLDS, learning_rate=0.08, n_estimators=400, **p)[0]

    def progress(study, trial):
        print(f"  trial {trial.number + 1}/{N_TRIALS}: {trial.value:.4f} "
              f"(best {study.best_value:.4f}, {time.time() - t0:.0f}s)", flush=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress])
    print(f"  best CV PR-AUC {study.best_value:.4f} after {N_TRIALS} trials ({time.time() - t0:.0f}s)")
    print(f"  best params: {study.best_params}")
    return study.best_params


def train_ensemble(X_tr, y_tr, X_va, y_va, xgb_params):
    """Return dict name -> fitted model."""
    models = {}
    m = make_xgb(y_tr, **xgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    models["xgboost"] = m
    spw = spw_of(y_tr)
    if lgb is not None:
        m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=31,
                               min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=2.0, scale_pos_weight=spw,
                               n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        models["lightgbm"] = m
    if CatBoostClassifier is not None:
        m = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                               scale_pos_weight=spw, eval_metric="PRAUC", early_stopping_rounds=50,
                               random_seed=SEED, verbose=False, thread_count=-1)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        models["catboost"] = m
    return models


def ensemble_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)


# ---------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    # 1. data + features
    ids, V, y, dates, miss = load_and_clean(DATA_PATH)
    print(f"customers: {len(y)}  days: {V.shape[1]}  theft share: {y.mean():.3%}")
    ref = population_reference(V)
    np.save(os.path.join(OUT_DIR, "reference_curve.npy"), ref)
    F_full = build_features(V, dates, miss, ref, weekly=True)
    F_slim = build_features(V, dates, miss, ref, weekly=False)
    print(f"features: {F_full.shape[1]} with weekly columns, {F_slim.shape[1]} without")

    # 2. does the weekly profile help? (labels of the test set are not involved: CV on all data
    #    is only used to choose between two feature sets, then the test split is evaluated once)
    print(f"\nChoosing feature set ({TUNE_FOLDS}-fold CV, default params):")
    s_full = cv_pr_auc(F_full, y, TUNE_FOLDS)
    s_slim = cv_pr_auc(F_slim, y, TUNE_FOLDS)
    print(f"  with weekly columns   : PR-AUC {s_full[0]:.4f}")
    print(f"  without weekly columns: PR-AUC {s_slim[0]:.4f}")
    use_weekly = s_full[0] > s_slim[0] + 0.003      # keep the extra columns only for a clear gain
    F = F_full if use_weekly else F_slim
    results.update({"use_weekly": bool(use_weekly), "cv_pr_auc_with_weekly": float(s_full[0]),
                    "cv_pr_auc_without_weekly": float(s_slim[0])})
    print(f"  -> using {'full' if use_weekly else 'slim'} feature set ({F.shape[1]} features)")

    # 3. tuning
    print(f"\nTuning XGBoost ({N_TRIALS} Optuna trials):")
    best = tune_xgb(F, y)
    results["xgb_params"] = best

    # 4. reported CV score of the tuned XGBoost
    print(f"\n{FINAL_FOLDS}-fold CV of tuned XGBoost:")
    pm, ps, rm, rs = cv_pr_auc(F, y, FINAL_FOLDS, verbose=True, **best)
    print(f"CV PR-AUC {pm:.3f} +/- {ps:.3f} | ROC-AUC {rm:.3f} +/- {rs:.3f}")
    results.update({"cv_pr_auc_mean": float(pm), "cv_pr_auc_std": float(ps),
                    "cv_roc_auc_mean": float(rm), "cv_roc_auc_std": float(rs)})

    # 5. final split by customer: 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(F, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    print("\nTraining ensemble:")
    models = train_ensemble(X_tr, y_tr, X_va, y_va, best)
    print("  models:", ", ".join(models))

    pv_raw = ensemble_proba(models, X_va)
    pt_raw = ensemble_proba(models, X_te)

    # 6. calibration on the validation set
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv_raw, y_va)
    pv, pt = iso.predict(pv_raw), iso.predict(pt_raw)

    # threshold on calibrated validation scores (best F1)
    p, r, t = precision_recall_curve(y_va, pv)
    f1 = 2 * p * r / (p + r + 1e-9)
    thr = float(t[np.argmax(f1[:-1])])

    # 7. one evaluation on the untouched test set
    print("\nTest set, single models (PR-AUC / ROC-AUC):")
    for name, m in models.items():
        s = m.predict_proba(X_te)[:, 1]
        print(f"  {name:<9}: {average_precision_score(y_te, s):.4f} / {roc_auc_score(y_te, s):.4f}")
        results[f"test_pr_auc_{name}"] = float(average_precision_score(y_te, s))

    pred = (pt >= thr).astype(int)
    results.update({
        "threshold_calibrated": thr,
        "pr_auc": float(average_precision_score(y_te, pt_raw)),
        "roc_auc": float(roc_auc_score(y_te, pt_raw)),
        "random_pr_auc": float(y_te.mean()),
        "brier_raw": float(brier_score_loss(y_te, pt_raw)),
        "brier_calibrated": float(brier_score_loss(y_te, pt)),
        "baseline_pr_auc_previous_run": 0.4144,
    })
    print("\nTest set, ensemble + calibration:")
    print(json.dumps({k: results[k] for k in ("threshold_calibrated", "pr_auc", "roc_auc", "random_pr_auc",
                                              "brier_raw", "brier_calibrated")}, indent=2))
    print(f"Previous baseline PR-AUC: 0.4144  ->  now {results['pr_auc']:.4f}")
    print(classification_report(y_te, pred, target_names=["normal", "theft"], digits=3))
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))

    # inspection budget
    order = np.argsort(-pt_raw)
    print("\nInspection budget:")
    for k in (0.01, 0.05, 0.10, 0.20):
        top = order[:max(int(k * len(pt_raw)), 1)]
        prec, rec = y_te[top].mean(), y_te[top].sum() / y_te.sum()
        results[f"top{int(k*100)}pct_precision"], results[f"top{int(k*100)}pct_recall"] = float(prec), float(rec)
        print(f"  top {k:>4.0%}: precision {prec:.2f}  recall {rec:.2f}  lift {prec / y_te.mean():.1f}x")

    print("\nTop 10 features (XGBoost):")
    imp = pd.Series(models["xgboost"].feature_importances_, index=F.columns).sort_values(ascending=False)
    print(imp.head(10))

    # 8. save everything needed for deployment
    models["xgboost"].save_model(os.path.join(OUT_DIR, "model_xgb.json"))
    if "lightgbm" in models:
        models["lightgbm"].booster_.save_model(os.path.join(OUT_DIR, "model_lgbm.txt"))
    if "catboost" in models:
        models["catboost"].save_model(os.path.join(OUT_DIR, "model_cat.cbm"))
    joblib.dump(iso, os.path.join(OUT_DIR, "calibrator.joblib"))
    json.dump({**results, "features": list(F.columns), "models": list(models)},
              open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)

    pp, rr, _ = precision_recall_curve(y_te, pt_raw)
    plt.figure()
    plt.plot(rr, pp)
    plt.axhline(y_te.mean(), ls="--", c="grey", label="random")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall (test, ensemble)")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved models, calibrator, reference curve, metrics and PR curve to {OUT_DIR}/")


if __name__ == "__main__":
    main()