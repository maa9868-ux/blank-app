## Step 00 - Import of the packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.set_page_config(
    page_title="Song Longevity Predictor 🎵",
    layout="centered",
    page_icon="🎵",
)

## Step 01 - Setup
st.sidebar.title("Streaming Analytics 🎵")
page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction 📘", "Visualization 📊", "Model Comparison 🔬",
     "Prediction 🎯", "Recommendations 💡"],
)

if os.path.exists("header.png"):
    st.image("header.png")

st.write("   ")


# --------------------------------------------------------------------
# Data loading and preparation (cached so it runs once, not per click)
# --------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("song_longevity.csv")


@st.cache_data
def prepare(df):
    """Encode categories and log the heavy-tailed columns."""
    d = df.copy().dropna()
    d["genre_enc"] = LabelEncoder().fit_transform(d["genre"])
    d["label_tier_enc"] = LabelEncoder().fit_transform(d["label_tier"])
    d["log_first_month"] = np.log10(d.first_month_streams.clip(lower=1))
    d["log_prior_listeners"] = np.log10(d.artist_prior_monthly_listeners.clip(lower=1))
    d["log_streams"] = np.log10(d.streams_3yr)
    return d


df = load_data()

# Columns that must never be used as predictors:
#   streams_3yr   -> the target itself
#   slow_burner   -> derived from the target
#   is_hit        -> derived from the target
#   halflife_days -> measured over the full 3 years, unknown at prediction time
#   track_id      -> an identifier, not information

AUDIO_F = ["genre_enc", "energy", "valence", "danceability", "acousticness",
           "instrumentalness", "tempo_bpm", "song_length_sec"]
BUSINESS_F = ["label_tier_enc", "marketing_budget", "playlist_adds_first_month",
              "editorial_playlist", "tiktok_virality", "log_prior_listeners",
              "featured_artist", "log_first_month"]

PRETTY = {
    "genre_enc": "genre", "label_tier_enc": "label_tier",
    "log_first_month": "first_month_streams (log)",
    "log_prior_listeners": "prior_listeners (log)",
}


@st.cache_data
def fit_model(d, feature_list, log_target=True, test_size=0.2):
    X = d[feature_list]
    y = d["log_streams"] if log_target else d["streams_3yr"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    m = LinearRegression().fit(X_tr, y_tr)
    p = m.predict(X_te)
    return {
        "model": m,
        "r2": metrics.r2_score(y_te, p),
        "mae": metrics.mean_absolute_error(y_te, p),
        "mse": metrics.mean_squared_error(y_te, p),
        "y_test": y_te, "pred": p,
        "coefs": pd.Series(m.coef_, index=feature_list),
    }


# ====================================================================
# PAGE 1 - INTRODUCTION
# ====================================================================
if page == "Introduction 📘":

    st.subheader("01 Introduction 📘")

    st.markdown("""
    ### The business problem

    A record label has a limited marketing budget and has to decide **which songs
    to keep investing in**. Pushing a track that fades costs money; dropping one
    that would have lasted costs far more.

    Our question: **can we predict how many streams a song will accumulate over
    three years, and what actually drives that number?**

    If the answer is the music itself, labels should invest in A&R. If the answer
    is distribution, they should invest in playlists and promotion. Those are very
    different companies to run.
    """)

    st.markdown("##### The dataset")

    c1, c2, c3 = st.columns(3)
    c1.metric("Songs", f"{df.shape[0]:,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Median streams (3yr)", f"{df.streams_3yr.median():,.0f}")

    st.markdown("""
    Each row is one released track. The columns fall into three groups:

    - **Audio features** — how the song sounds: energy, valence, danceability,
      acousticness, instrumentalness, tempo, length, genre
    - **Distribution features** — how it was pushed: label tier, marketing budget,
      playlist adds, editorial placement, TikTok virality, the artist's existing
      audience
    - **Outcomes** — first-month streams, half-life in days, and our target,
      `streams_3yr`
    """)

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Data quality check")

    q1, q2, q3 = st.columns(3)
    q1.metric("Missing values", int(df.isnull().sum().sum()))
    q2.metric("Duplicate rows", int(df.duplicated().sum()))
    q3.metric("Numeric columns", df.select_dtypes(include=np.number).shape[1])

    if df.isnull().sum().sum() == 0:
        st.success("✅ No missing values found — no imputation needed")
    else:
        st.warning("⚠️ you have missing values")
        st.write(df.isnull().sum())

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

    st.markdown("##### ⚠️ One thing to notice before modelling")
    m1, m2 = st.columns(2)
    m1.metric("Median streams", f"{df.streams_3yr.median():,.0f}")
    m2.metric("Maximum streams", f"{df.streams_3yr.max():,.0f}")

    ratio = df.streams_3yr.max() / df.streams_3yr.median()

    # computed live so the claim can never drift from what the model does
    _d = prepare(df)
    _raw = fit_model(_d, AUDIO_F + BUSINESS_F, log_target=False)["r2"]
    _log = fit_model(_d, AUDIO_F + BUSINESS_F, log_target=True)["r2"]

    st.markdown(f"""
    The biggest song in the dataset has **{ratio:,.0f} times** the streams of a
    typical one. A handful of mega-hits sit far above everything else.

    That matters for the model: fitted on raw streams, a linear regression spends
    all its effort on those few outliers and fits everything else badly. We model
    `log10(streams)` instead, which turns a multiplicative problem into an
    additive one.
    """)

    r1, r2c = st.columns(2)
    r1.metric("R² on raw streams", f"{_raw:.3f}")
    r2c.metric("R² on log10(streams)", f"{_log:.3f}")
    st.caption(
        "Same features, same split — the only change is the scale of the target. "
        "You can reproduce this on the Prediction page with the "
        "'Log-transform the target' checkbox."
    )


# ====================================================================
# PAGE 2 - VISUALIZATION
# ====================================================================
elif page == "Visualization 📊":

    st.subheader("02 Data Viz 📊")

    d = prepare(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution 📈", "By Category 📊", "Correlation Heatmap 🔥", "Audio vs Business 🎧"
    ])

    with tab1:
        st.markdown("#### The target is extremely skewed")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.histplot(d.streams_3yr, bins=60, ax=ax1, color="#4878A6")
        ax1.set_title("Raw streams")
        ax1.set_xlabel("3-year streams")

        sns.histplot(d.log_streams, bins=60, ax=ax2, color="#4878A6")
        ax2.set_title("log10(streams)")
        ax2.set_xlabel("log10(3-year streams)")
        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "Left: almost every song is crushed against zero because a few hits "
            "stretch the axis. Right: the same data on a log scale becomes roughly "
            "bell-shaped — which is what linear regression assumes."
        )

    with tab2:
        st.markdown("#### Median streams by category")

        cat = st.selectbox("Group by", ["genre", "label_tier", "editorial_playlist"])
        grouped = d.groupby(cat)["streams_3yr"].median().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(x=grouped.index.astype(str), y=grouped.values, ax=ax, color="#4878A6")
        ax.set_ylabel("Median 3-year streams")
        ax.set_xlabel(cat)
        ax.set_title(f"Median 3-year streams by {cat}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(grouped.round(0).astype(int).rename("median_streams"))

    with tab3:
        st.markdown("#### Correlation Matrix")

        num = d.select_dtypes(include=np.number).drop(
            columns=["slow_burner", "is_hit", "genre_enc", "label_tier_enc"]
        )

        fig_corr, ax_corr = plt.subplots(figsize=(11, 9))
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax_corr, annot_kws={"size": 7})
        plt.tight_layout()
        st.pyplot(fig_corr)

        st.info(
            "Read the `log_streams` row. The strong correlations are all business "
            "variables — first-month streams, TikTok virality, playlist adds. "
            "The audio columns sit near zero."
        )

    with tab4:
        st.markdown("#### Does how a song *sounds* predict how it performs?")

        audio_pick = st.selectbox(
            "Audio feature",
            ["energy", "valence", "danceability", "acousticness",
             "instrumentalness", "tempo_bpm", "song_length_sec"],
        )
        biz_pick = st.selectbox(
            "Business feature",
            ["tiktok_virality", "playlist_adds_first_month", "marketing_budget"],
        )

        sample = d.sample(3000, random_state=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        ax1.scatter(sample[audio_pick], sample.log_streams, alpha=0.15, s=10, color="#4878A6")
        ax1.set_title(f"{audio_pick}  (r = {d[audio_pick].corr(d.log_streams):.3f})")
        ax1.set_xlabel(audio_pick)
        ax1.set_ylabel("log10(3-year streams)")

        ax2.scatter(sample[biz_pick], sample.log_streams, alpha=0.15, s=10, color="#C06C4F")
        ax2.set_title(f"{biz_pick}  (r = {d[biz_pick].corr(d.log_streams):.3f})")
        ax2.set_xlabel(biz_pick)
        plt.tight_layout()
        st.pyplot(fig)

        st.success(
            "Left is a shapeless cloud whatever you pick. Right slopes. "
            "This is the finding the whole project rests on."
        )


# ====================================================================
# PAGE 3 - MODEL COMPARISON
# ====================================================================
elif page == "Model Comparison 🔬":

    st.subheader("03 Model Comparison 🔬")

    st.markdown("""
    We fit the **same linear regression three times**, changing only which
    features it is allowed to see. Everything else is identical: same rows, same
    80/20 split, same random seed.
    """)

    d = prepare(df)

    audio = fit_model(d, AUDIO_F)
    biz = fit_model(d, BUSINESS_F)
    both = fit_model(d, AUDIO_F + BUSINESS_F)

    c1, c2, c3 = st.columns(3)
    c1.metric("🎧 Audio only", f"{audio['r2']:.3f}")
    c2.metric("💼 Business only", f"{biz['r2']:.3f}")
    c3.metric("🎧+💼 Combined", f"{both['r2']:.3f}")
    st.caption("R² — the share of variation in 3-year streams the model explains.")

    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["Audio only", "Business only", "Combined"]
    vals = [audio["r2"], biz["r2"], both["r2"]]
    bars = ax.bar(names, vals, color=["#C06C4F", "#4878A6", "#4878A6"])
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.set_title("How much does each group of features explain?")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.03,
                f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.error(
        f"**Audio features explain {audio['r2']:.1%} of the variation.** Energy, "
        "danceability, tempo, valence, acousticness, length and genre — all eight "
        "of them together — tell you essentially nothing about whether a song will "
        "last."
    )
    st.success(
        f"**Business features explain {biz['r2']:.1%}.** Adding the audio features "
        f"on top moves it to {both['r2']:.1%} — a gain of "
        f"{(both['r2'] - biz['r2']):.3f}, which is noise."
    )

    st.markdown("##### Full comparison")
    st.dataframe(pd.DataFrame({
        "Feature set": names,
        "Features used": [len(AUDIO_F), len(BUSINESS_F), len(AUDIO_F) + len(BUSINESS_F)],
        "R²": [round(v, 3) for v in vals],
        "MAE (log10)": [round(audio["mae"], 3), round(biz["mae"], 3), round(both["mae"], 3)],
    }))

    st.markdown("##### Why this matters")
    st.markdown("""
    This is not a statement that music quality is irrelevant to human beings. It is
    a statement about **what a label can act on**. Within this dataset, two songs
    with near-identical audio profiles can end up three orders of magnitude apart
    in streams, and what separates them is how they were distributed.
    """)


# ====================================================================
# PAGE 4 - PREDICTION
# ====================================================================
elif page == "Prediction 🎯":

    st.subheader("04 Prediction with Linear Regression 🎯")

    d = prepare(df)

    st.sidebar.markdown("### Model settings")
    preset = st.sidebar.radio(
        "Feature set", ["All features", "Audio only", "Business only", "Custom"]
    )

    if preset == "All features":
        features_selection = AUDIO_F + BUSINESS_F
    elif preset == "Audio only":
        features_selection = AUDIO_F
    elif preset == "Business only":
        features_selection = BUSINESS_F
    else:
        features_selection = st.sidebar.multiselect(
            "Select Features (X)", AUDIO_F + BUSINESS_F, default=AUDIO_F + BUSINESS_F
        )

    log_target = st.sidebar.checkbox("Log-transform the target", value=True)
    test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2)
    selected_metrics = st.sidebar.multiselect(
        "Metrics to display",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["R² Score", "Mean Absolute Error (MAE)"],
    )

    if len(features_selection) == 0:
        st.warning("Select at least one feature in the sidebar.")
        st.stop()

    res = fit_model(d, features_selection, log_target, test_size)

    st.markdown("##### Features (X)")
    st.dataframe(d[features_selection].head())

    st.markdown("##### Model performance")
    shown = selected_metrics if selected_metrics else ["R² Score"]
    cols = st.columns(len(shown))
    i = 0
    if "R² Score" in shown:
        cols[i].metric("R² Score", f"{res['r2']:.3f}"); i += 1
    if "Mean Absolute Error (MAE)" in shown:
        cols[i].metric("MAE", f"{res['mae']:,.3f}"); i += 1
    if "Mean Squared Error (MSE)" in shown:
        cols[i].metric("MSE", f"{res['mse']:,.3f}"); i += 1

    if log_target:
        st.caption(
            f"On the log scale, a MAE of {res['mae']:.2f} means the typical "
            f"prediction is off by a factor of about {10 ** res['mae']:.1f}×."
        )

    if res["r2"] > 0.7:
        st.success(f"✅ Strong fit — the model explains {res['r2']:.1%} of the variation.")
    elif res["r2"] > 0.3:
        st.warning(f"⚠️ Moderate fit — {res['r2']:.1%} explained.")
    else:
        st.error(f"❌ Weak fit — only {res['r2']:.1%} explained. Try another feature set.")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(res["y_test"], res["pred"], alpha=0.15, s=10, color="#4878A6")
    lo, hi = res["y_test"].min(), res["y_test"].max()
    ax.plot([lo, hi], [lo, hi], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("##### What drives the prediction")
    coefs = res["coefs"].sort_values(key=abs, ascending=False)
    labels = [PRETTY.get(c, c) for c in coefs.index]

    fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.32 * len(coefs))))
    colors = ["#4878A6" if v > 0 else "#C06C4F" for v in coefs.values]
    ax2.barh(labels[::-1], coefs.values[::-1], color=colors[::-1])
    ax2.axvline(0, color="#666", linewidth=1)
    ax2.set_xlabel("Coefficient")
    ax2.set_title("Driving variables")
    plt.tight_layout()
    st.pyplot(fig2)

    if log_target and "log_first_month" in coefs.index:
        st.info(
            f"`first_month_streams (log)` has a coefficient of "
            f"{coefs['log_first_month']:.2f}. On a log-log scale that is close to "
            "proportional: ten times the first-month streams means roughly ten "
            "times the three-year streams. Early momentum compounds."
        )

    st.markdown("---")
    st.markdown("##### 🎧 Predict a song")
    st.caption("Set the inputs and the model predicts its 3-year streams.")

    if preset == "All features" and log_target:
        c1, c2 = st.columns(2)
        in_first = c1.number_input("First-month streams", 100, 5_000_000, 20_000, step=1000)
        in_tiktok = c1.slider("TikTok virality (0-15)", 0.0, 15.0, 2.0)
        in_playlist = c1.slider("Playlist adds, first month", 0, 60, 5)
        in_editorial = c1.selectbox("Editorial playlist?", [0, 1])
        in_genre = c2.selectbox("Genre", sorted(df.genre.unique()))
        in_label = c2.selectbox("Label tier", sorted(df.label_tier.unique()))
        in_budget = c2.slider("Marketing budget (0-12)", 0.0, 12.0, 2.0)
        in_prior = c2.number_input("Artist's prior monthly listeners", 100, 20_000_000, 50_000, step=1000)

        genre_map = dict(zip(d.genre, d.genre_enc))
        label_map = dict(zip(d.label_tier, d.label_tier_enc))

        row = {c: float(d[c].median()) for c in features_selection}
        row.update({
            "log_first_month": np.log10(max(in_first, 1)),
            "tiktok_virality": in_tiktok,
            "playlist_adds_first_month": in_playlist,
            "editorial_playlist": in_editorial,
            "genre_enc": genre_map[in_genre],
            "label_tier_enc": label_map[in_label],
            "marketing_budget": in_budget,
            "log_prior_listeners": np.log10(max(in_prior, 1)),
        })

        x_new = pd.DataFrame([[row[c] for c in features_selection]], columns=features_selection)
        pred = res["model"].predict(x_new)[0]

        st.metric("Predicted 3-year streams", f"{10 ** pred:,.0f}")
        st.caption(
            f"Typical range given the model's error: "
            f"{10 ** (pred - res['mae']):,.0f} to {10 ** (pred + res['mae']):,.0f}"
        )
    else:
        st.caption(
            "Switch the feature set to **All features** with the log target on "
            "to use the predictor."
        )


# ====================================================================
# PAGE 5 - RECOMMENDATIONS
# ====================================================================
elif page == "Recommendations 💡":

    st.subheader("05 Recommendations 💡")

    d = prepare(df)
    both = fit_model(d, AUDIO_F + BUSINESS_F)
    co = both["coefs"]

    ed_mult = 10 ** co["editorial_playlist"]
    tk_mult = 10 ** (co["tiktok_virality"] * 5)
    ed_yes = d[d.editorial_playlist == 1].streams_3yr.median()
    ed_no = d[d.editorial_playlist == 0].streams_3yr.median()

    st.markdown("""
    ### What should the label actually do?

    Three recommendations, each tied to a number the model produced.
    """)

    st.markdown("#### 1. Stop screening on the music, start screening on the rollout")
    a1, a2 = st.columns(2)
    a1.metric("Audio features R²", f"{fit_model(d, AUDIO_F)['r2']:.3f}")
    a2.metric("Business features R²", f"{fit_model(d, BUSINESS_F)['r2']:.3f}")
    st.markdown("""
    A&R meetings spend their time on how a track sounds. In this data that is
    the one thing with no predictive power at all. The decision that matters is
    not *is this a good song* but *can we get it placed*.
    """)

    st.markdown("#### 2. Editorial playlist placement is the single biggest lever you control")
    b1, b2 = st.columns(2)
    b1.metric("Median streams, placed", f"{ed_yes:,.0f}")
    b2.metric("Median streams, not placed", f"{ed_no:,.0f}")
    st.markdown(f"""
    Holding everything else constant, the model puts editorial placement at a
    **{ed_mult:.1f}× multiplier** on 3-year streams. Unlike TikTok virality, which
    is largely exogenous, placement is something a label can negotiate for.
    Budget should move toward playlist pitching.
    """)

    st.markdown("#### 3. Judge a song at one month, not at release")
    st.markdown(f"""
    The coefficient on log first-month streams is **{co['log_first_month']:.2f}** —
    on a log-log scale, almost exactly proportional. Ten times the first-month
    streams implies roughly ten times the three-year streams.

    Practically: a song's first month is a very strong signal of its eventual
    ceiling. Re-allocate marketing at the 30-day mark rather than committing the
    full budget at release.
    """)

    st.markdown("---")
    st.markdown("#### ⚠️ What this model cannot tell you")
    st.markdown(f"""
    - **{(1 - both['r2']):.0%} of the variation is still unexplained.** Even the best
      model here misses most of what makes an individual song succeed.
    - **This is correlation, not causation.** Songs that get editorial placement
      may be better songs in ways the audio columns do not capture — the playlist
      may be a symptom of quality rather than a cause of success.
    - **There is a feedback loop.** Early streams drive placement, which drives
      more streams. The model cannot separate the two directions.
    - **Typical error is {10 ** both['mae']:.1f}×.** Useful for ranking a catalogue,
      not for forecasting one track's revenue.
    """)
