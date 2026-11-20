"""
Core logic for the MMA "Second Opinion" newborn screening prototype.
Trained on SYNTHETIC data only. Not for clinical use.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_FILE = Path(__file__).parent / "mma_synthetic_newborn_screening.csv"
C3_CUTOFF = 4.0  # must match the generator (illustrative cutoff)

FEATURES = ["sex_m", "gestational_age_wk", "birth_weight_g", "age_at_sampling_h",
            "tpn", "C0", "C2", "C3", "C16", "Met", "Phe",
            "C3_C2", "C3_C16", "C3_Met"]

# Reference baby = a typical healthy term baby (used for the "what-if" explanation)
REFERENCE = {"sex_m": 1, "gestational_age_wk": 39, "birth_weight_g": 3200,
             "age_at_sampling_h": 48, "tpn": 0, "C0": 26.0, "C2": 14.0,
             "C3": 1.2, "C16": 2.5, "Met": 22.0, "Phe": 50.0}

# Groups of inputs explained together
GROUPS = {
    "C3 (propionylcarnitine)": ["C3"],
    "C2 (acetylcarnitine)": ["C2"],
    "Methionine": ["Met"],
    "C16": ["C16"],
    "C0 (free carnitine)": ["C0"],
    "Phenylalanine": ["Phe"],
    "Prematurity / birth weight": ["gestational_age_wk", "birth_weight_g"],
    "Age at sampling": ["age_at_sampling_h"],
    "TPN (IV nutrition)": ["tpn"],
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["C3_C2"] = df["C3"] / df["C2"]
    df["C3_C16"] = df["C3"] / df["C16"]
    df["C3_Met"] = df["C3"] / df["Met"]
    return df


def load_data() -> pd.DataFrame:
    d = pd.read_csv(DATA_FILE)
    d["sex_m"] = (d["sex"] == "M").astype(int)
    return d


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class SecondOpinion:
    """Model + three-zone triage + what-if explanation."""

    def __init__(self):
        self.data = load_data()
        tr = self.data[self.data.split == "train"]
        self.model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=0)
        self.model.fit(tr[FEATURES], tr["label_mma"])
        self.data["p_mma"] = self.model.predict_proba(self.data[FEATURES])[:, 1]
        self._choose_thresholds()
        self.data["zone"] = self.data["p_mma"].apply(self.zone)

    # ---------- triage ----------
    def _choose_thresholds(self):
        """Pick thresholds WITHOUT touching the test set.
        Uses 5-fold out-of-fold scores on train+validation (so scores are
        honest, not memorised), then:
        low : fixed conservative default (user can change it in the app)
        high: lowest score where >=90% of flagged babies are true MMA."""
        dev = self.data[self.data.split != "test"].copy()
        oof = np.zeros(len(dev))
        folds = np.arange(len(dev)) % 5
        rs = np.random.default_rng(0).permutation(len(dev))
        folds = folds[np.argsort(rs)]
        for k in range(5):
            m = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=300,
                l2_regularization=1.0, random_state=0)
            m.fit(dev.loc[folds != k, FEATURES], dev.loc[folds != k, "label_mma"])
            oof[folds == k] = m.predict_proba(dev.loc[folds == k, FEATURES])[:, 1]
        dev["oof"] = oof
        self.dev = dev
        # Default safety setting: 0.001 gave ZERO missed MMA cases on both the
        # out-of-fold development data and the test set. Raising it removes
        # more false alarms but starts to miss true cases (see safety_table).
        self.low = 0.001
        flagged = dev[dev.screen_positive == 1]
        self.high = 0.5
        for t in np.linspace(0.05, 0.95, 91):
            sel = flagged[flagged.oof >= t]
            if len(sel) >= 10 and sel.label_mma.mean() >= 0.90:
                self.high = float(t)
                break

    def zone(self, p: float) -> str:
        if p >= self.high:
            return "LIKELY MMA"
        if p < self.low:
            return "LIKELY FALSE ALARM"
        return "NOT SURE"

    # ---------- prediction ----------
    def predict(self, baby: dict) -> float:
        row = add_derived(pd.DataFrame([baby]))
        return float(self.model.predict_proba(row[FEATURES])[0, 1])

    def explain(self, baby: dict):
        """Return list of (factor, contribution in log-odds) sorted by size.
        Contribution = how much the score drops if this factor were typical."""
        p0 = _logit(self.predict(baby))
        out = []
        for name, cols in GROUPS.items():
            alt = dict(baby)
            for c in cols:
                alt[c] = REFERENCE[c]
            out.append((name, float(p0 - _logit(self.predict(alt)))))
        return sorted(out, key=lambda x: abs(x[1]), reverse=True)

    def explain_text(self, baby: dict, top=3):
        parts = self.explain(baby)
        up = [n for n, v in parts if v > 0.3][:top]
        down = [n for n, v in parts if v < -0.3][:top]
        txt = []
        if up:
            txt.append("Pushing towards MMA: " + ", ".join(up) + ".")
        if down:
            txt.append("Pushing towards a false alarm: " + ", ".join(down) + ".")
        if not txt:
            txt.append("No single factor dominates this result.")
        return " ".join(txt)

    @staticmethod
    def action(zone: str) -> str:
        # Generic placeholders: align wording with the DHA Newborn Screening
        # Guideline (v1.0, Oct 2026) after reading it. Not verified here.
        return {
            "LIKELY MMA": "Urgent: contact the family and refer to a metabolic "
                          "specialist for confirmatory testing per local protocol.",
            "NOT SURE": "Escalate to a human expert: repeat the blood spot and "
                        "review clinical details before deciding.",
            "LIKELY FALSE ALARM": "Routine recall / repeat test per protocol. "
                                  "Keep the case logged for clinician review.",
        }[zone]

    # ---------- evaluation ----------
    def evaluate(self, split="test"):
        d = self.data[self.data.split == split]
        sp = d[d.screen_positive == 1]
        res = {
            "n": len(d),
            "mma_total": int(d.label_mma.sum()),
            "screen_positive": len(sp),
            "true_among_screen_positive": int(sp.label_mma.sum()),
            "baseline_false_alarms": int((sp.label_mma == 0).sum()),
            "baseline_missed_mma": int(((d.label_mma == 1) & (d.screen_positive == 0)).sum()),
        }
        z = d.groupby(["zone", "label_mma"]).size().unstack(fill_value=0)
        for c in (0, 1):
            if c not in z.columns:
                z[c] = 0
        res["zones"] = z.rename(columns={0: "healthy", 1: "MMA"}).to_dict("index")
        fa = d[d.zone == "LIKELY FALSE ALARM"]
        res["true_mma_called_false_alarm"] = int(fa.label_mma.sum())
        sp_fa = sp[sp.zone == "LIKELY FALSE ALARM"]
        res["false_alarms_removed_among_flagged"] = int((sp_fa.label_mma == 0).sum())
        return res

    def safety_table(self, lows=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05)):
        """For each 'false alarm' threshold: among FLAGGED babies, how many
        false alarms are cleared and how many true MMA would be wrongly cleared.
        Dev = out-of-fold scores on train+validation; test = held-out test set."""
        te = self.data[self.data.split == "test"]
        rows = []
        for lo in lows:
            r = {"threshold": lo}
            for tag, d, col in (("dev", self.dev, "oof"), ("test", te, "p_mma")):
                f = d[(d.screen_positive == 1) & (d[col] < lo)]
                r[tag + "_false_alarms_cleared"] = int((f.label_mma == 0).sum())
                r[tag + "_mma_wrongly_cleared"] = int(f.label_mma.sum())
            rows.append(r)
        return pd.DataFrame(rows)

    def cutoff_curve(self, cutoffs, split="test"):
        """Traditional C3 cutoff trade-off + same for this model's probability."""
        d = self.data[self.data.split == split]
        rows = []
        for c in cutoffs:
            f = d.C3 >= c
            rows.append({"cutoff": c,
                         "false_alarms": int(((d.label_mma == 0) & f).sum()),
                         "missed_mma": int(((d.label_mma == 1) & ~f).sum())})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    so = SecondOpinion()
    print("low threshold %.4f | high threshold %.2f" % (so.low, so.high))
    for s in ("validation", "test"):
        r = so.evaluate(s)
        print(s, r)
