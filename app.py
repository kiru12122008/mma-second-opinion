"""
MMA Newborn Screening "Second Opinion" dashboard (SYNTHETIC DATA PROTOTYPE)
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import SecondOpinion, REFERENCE, C3_CUTOFF

st.set_page_config(page_title="MMA Second Opinion", page_icon="🧬", layout="wide")

ZONE_COLOR = {"LIKELY MMA": "#c0392b", "NOT SURE": "#e1a100",
              "LIKELY FALSE ALARM": "#2e8b57"}


@st.cache_resource(show_spinner="Training model on synthetic data...")
def get_engine():
    return SecondOpinion()


so = get_engine()

# ------------------------------------------------------------------ header
st.title("🧬 Newborn Screening Second Opinion: Methylmalonic Acidemia (MMA)")
st.caption("Explainable decision support that separates real MMA from false alarms, "
           "and says 'not sure' when it should.")
st.warning("**Prototype on SYNTHETIC data.** Marker ranges are illustrative. "
           "Results show how the system behaves, not clinical performance. "
           "Needs validation on real UAE screening data before any clinical use.")

# ------------------------------------------------------------------ presets
PRESETS = {
    "Typical healthy baby": dict(REFERENCE),
    "Preterm baby, C3 slightly high": dict(
        sex_m=1, gestational_age_wk=31, birth_weight_g=1700, age_at_sampling_h=40,
        tpn=0, C0=22.0, C2=13.0, C3=4.9, C16=2.4, Met=22.0, Phe=50.0),
    "Classic MMA pattern": dict(
        sex_m=0, gestational_age_wk=39, birth_weight_g=3100, age_at_sampling_h=48,
        tpn=0, C0=25.0, C2=10.0, C3=12.0, C16=2.2, Met=12.0, Phe=48.0),
    "Ambiguous: mild C3 rise, full-term": dict(
        sex_m=1, gestational_age_wk=39, birth_weight_g=3200, age_at_sampling_h=50,
        tpn=0, C0=25.0, C2=12.0, C3=4.6, C16=2.5, Met=19.0, Phe=50.0),
}
KEYS = ["sex_m", "gestational_age_wk", "birth_weight_g", "age_at_sampling_h",
        "tpn", "C0", "C2", "C3", "C16", "Met", "Phe"]
for k in KEYS:
    st.session_state.setdefault("in_" + k, REFERENCE[k])


def load_preset(name):
    for k in KEYS:
        st.session_state["in_" + k] = PRESETS[name][k]


tab_sim, tab_perf, tab_safety, tab_cases, tab_impact = st.tabs(
    ["1. Live simulator", "2. Tool vs plain cutoff", "3. Cutoff & safety trade-off",
     "4. Case browser", "5. Impact estimate"])

# ------------------------------------------------------------------ tab 1
with tab_sim:
    st.subheader("Enter a baby's screening result")
    st.write("Pick a demo case or move the sliders. The answer updates instantly.")
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, on_click=load_preset, args=(name,), use_container_width=True)

    left, mid, right = st.columns([1, 1, 1.3])
    with left:
        st.markdown("**Baby details**")
        sex = st.radio("Sex", ["Male", "Female"],
                       index=0 if st.session_state["in_sex_m"] == 1 else 1,
                       horizontal=True)
        ga = st.slider("Gestational age (weeks)", 26, 42, key="in_gestational_age_wk")
        bw = st.slider("Birth weight (g)", 800, 4800, step=50, key="in_birth_weight_g")
        age = st.slider("Age at sampling (hours)", 6, 120, key="in_age_at_sampling_h")
        tpn = st.checkbox("On intravenous nutrition (TPN)",
                          value=bool(st.session_state["in_tpn"]))
    with mid:
        st.markdown("**Markers (µmol/L)**")
        c3 = st.slider("C3 (propionylcarnitine)", 0.2, 30.0, step=0.1, key="in_C3")
        c2 = st.slider("C2 (acetylcarnitine)", 2.0, 40.0, step=0.5, key="in_C2")
        met = st.slider("Methionine", 4.0, 60.0, step=0.5, key="in_Met")
        c16 = st.slider("C16", 0.5, 8.0, step=0.1, key="in_C16")
        c0 = st.slider("C0 (free carnitine)", 5.0, 80.0, step=0.5, key="in_C0")
        phe = st.slider("Phenylalanine", 20.0, 150.0, step=1.0, key="in_Phe")

    baby = dict(sex_m=1 if sex == "Male" else 0, gestational_age_wk=ga,
                birth_weight_g=bw, age_at_sampling_h=age, tpn=int(tpn),
                C0=c0, C2=c2, C3=c3, C16=c16, Met=met, Phe=phe)
    p = so.predict(baby)
    zone = so.zone(p)

    with right:
        st.markdown("**Result**")
        st.markdown(
            f"<div style='padding:14px;border-radius:8px;background:{ZONE_COLOR[zone]};"
            f"color:white;font-size:22px;font-weight:700;text-align:center'>{zone}</div>",
            unsafe_allow_html=True)
        st.metric("Model score for MMA", f"{p*100:.1f}%")
        flagged = c3 >= C3_CUTOFF
        st.write(f"Plain cutoff (C3 ≥ {C3_CUTOFF}): "
                 f"**{'FLAGGED' if flagged else 'not flagged'}**")
        st.info(so.explain_text(baby))
        st.success("**Suggested next step:** " + so.action(zone))

        contrib = so.explain(baby)
        names = [n for n, v in contrib if abs(v) > 0.05][:7][::-1]
        vals = [v for n, v in contrib if abs(v) > 0.05][:7][::-1]
        if names:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            ax.barh(names, vals,
                    color=["#c0392b" if v > 0 else "#2e8b57" for v in vals])
            ax.axvline(0, color="grey", lw=0.8)
            ax.set_xlabel("← towards false alarm        towards MMA →")
            ax.set_title("Why this result? (what-if: factor set to typical)", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
    st.caption("Explanation method: each factor is set to a typical healthy value in "
               "turn, and the change in the model score is shown. Red pushes towards "
               "MMA, green towards a false alarm.")

# ------------------------------------------------------------------ tab 2
with tab_perf:
    r = so.evaluate("test")
    sp_n, sp_true = r["screen_positive"], r["true_among_screen_positive"]
    z = r["zones"]
    urgent = z.get("LIKELY MMA", {"healthy": 0, "MMA": 0})
    urgent_n = urgent["healthy"] + urgent["MMA"]
    st.subheader("Held-out test set (754 synthetic babies, never used for training)")
    a, b, c = st.columns(3)
    a.metric("Babies flagged by plain cutoff", sp_n,
             f"only {100*sp_true/sp_n:.0f}% truly have MMA", delta_color="off")
    b.metric("Urgent queue with this tool", urgent_n,
             f"{100*urgent['MMA']/max(urgent_n,1):.0f}% truly have MMA", delta_color="off")
    c.metric("True MMA called 'false alarm'", r["true_mma_called_false_alarm"])
    st.write(
        f"Of **{sp_true}** truly affected flagged babies, the urgent queue catches "
        f"**{urgent['MMA']}**. The rest are sent to **NOT SURE** for a human expert, "
        "not dismissed.")
    zt = pd.DataFrame(z).T.fillna(0).astype(int)
    zt["total"] = zt.sum(axis=1)
    st.markdown("**All test babies by zone**")
    st.dataframe(zt, use_container_width=True)
    st.markdown("**By scenario (how each type of baby was triaged)**")
    dd = so.data[so.data.split == "test"]
    st.dataframe(dd.groupby("scenario").zone.value_counts().unstack(fill_value=0),
                 use_container_width=True)
    st.caption("Honest note: on this data the tool's main strength is **prioritising** "
               "the urgent queue and keeping uncertain cases with a human. At the safest "
               "setting it clears almost no false alarms among flagged babies (see tab 3).")

# ------------------------------------------------------------------ tab 3
with tab_safety:
    st.subheader("The trade-off no tool can avoid")
    cut = st.slider("Plain C3 cutoff (µmol/L)", 2.0, 8.0, C3_CUTOFF, 0.1)
    cc = so.cutoff_curve(np.round(np.arange(2.0, 8.01, 0.25), 2))
    one = so.cutoff_curve([cut]).iloc[0]
    x, y = st.columns(2)
    x.metric("False alarms at this cutoff (test)", int(one["false_alarms"]))
    y.metric("True MMA missed at this cutoff (test)", int(one["missed_mma"]))
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(cc.cutoff, cc.false_alarms, label="False alarms", color="#e1a100")
    ax.plot(cc.cutoff, cc.missed_mma, label="Missed MMA", color="#c0392b")
    ax.axvline(cut, color="grey", ls="--")
    ax.set_xlabel("C3 cutoff"); ax.set_ylabel("Babies"); ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.write("Raising the cutoff cuts false alarms but starts to miss sick babies. "
             "That is why safe hospitals keep cutoffs low, and why a second opinion helps.")

    st.divider()
    st.subheader("Safety setting for 'LIKELY FALSE ALARM'")
    st.write("Among **flagged** babies, how many false alarms can be cleared, and at "
             "what risk of wrongly clearing a true case?")
    st.dataframe(so.safety_table().rename(columns={
        "threshold": "score below",
        "dev_false_alarms_cleared": "dev: false alarms cleared",
        "dev_mma_wrongly_cleared": "dev: MMA wrongly cleared",
        "test_false_alarms_cleared": "test: false alarms cleared",
        "test_mma_wrongly_cleared": "test: MMA wrongly cleared"}),
        use_container_width=True, hide_index=True)
    st.caption("'dev' = out-of-fold scores on train+validation (larger sample). "
               "'test' = held-out set. The app default is 0.001: zero wrongly cleared "
               "in both. Higher settings clear more false alarms but risk missed cases.")

# ------------------------------------------------------------------ tab 4
with tab_cases:
    st.subheader("Browse test babies")
    dd = so.data[so.data.split == "test"].copy()
    scen = st.selectbox("Scenario", ["(all)"] + sorted(dd.scenario.unique()))
    if scen != "(all)":
        dd = dd[dd.scenario == scen]
    show = dd[["baby_id", "scenario", "gestational_age_wk", "birth_weight_g",
               "C3", "C3_C2", "Met", "p_mma", "zone", "label_mma"]].rename(
        columns={"label_mma": "truly_MMA", "p_mma": "model_score"})
    st.dataframe(show.round(3), use_container_width=True, hide_index=True)
    if len(dd):
        pick = st.selectbox("Explain this baby", dd.baby_id.tolist())
        row = dd[dd.baby_id == pick].iloc[0]
        b = {k: float(row[k]) for k in KEYS}
        st.write(f"**{row.zone}** (score {row.p_mma*100:.1f}%, truly MMA: "
                 f"{'yes' if row.label_mma == 1 else 'no'})")
        st.info(so.explain_text(b))

# ------------------------------------------------------------------ tab 5
with tab_impact:
    st.subheader("Illustrative workload estimate")
    st.write("Enter how many babies your lab flags per year. The estimate reuses the "
             "ratios from the synthetic test set, so treat it as an example of the "
             "calculation, **not a forecast**.")
    n_flag = st.number_input("Babies flagged per year by the plain cutoff", 10, 100000, 500)
    r = so.evaluate("test")
    share_urgent = (z["LIKELY MMA"]["healthy"] + z["LIKELY MMA"]["MMA"]) / r["screen_positive"] \
        if "LIKELY MMA" in z else 0
    st.metric("Flagged babies needing URGENT review (not all flagged)",
              f"{int(round(n_flag*share_urgent))}",
              f"{100*share_urgent:.0f}% of flagged", delta_color="off")
    st.metric("Flagged babies sent to NOT SURE (human expert review)",
              f"{int(round(n_flag*(1-share_urgent)))}")
    st.caption("Real prevalence is far lower than in this synthetic set, so real "
               "numbers would differ. Replace with local data when available.")

st.divider()
st.caption("Built for the Dubai Future Health innovation challenge. Synthetic data, "
           "prototype only. Not a medical device.")
