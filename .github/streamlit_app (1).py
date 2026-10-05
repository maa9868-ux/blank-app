## Step 00 - Import of the packages

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
    ["Introduction 📘", "Visualization 📊", "Prediction 🎯"],
)

st.image("header.png")

st.write("   ")
st.write("   ")

df = pd.read_csv("song_longevity.csv")

# Columns that must never be used as predictors:
#   streams_3yr  -> the target itself
#   slow_burner  -> derived from the target
#   is_hit       -> derived from the target
#   halflife_days-> measured over the full 3 years, so unknown at prediction time
#   track_id     -> an identifier, not information
LEAKAGE = ["track_id", "streams_3yr", "slow_burner", "is_hit", "halflife_days"]

AUDIO = ["genre", "energy", "valence", "danceability", "acousticness",
         "instrumentalness", "tempo_bpm", "song_length_sec"]
BUSINESS = ["label_tier", "marketing_budget", "playlist_adds_first_month",
            "editorial_playlist", "tiktok_virality",
            "artist_prior_monthly_listeners", "featured_artist",
            "first_month_streams"]


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

    col1, col2, col3 = st.columns(3)
    col1.metric("Songs", f"{df.shape[0]:,}")
    col2.metric("Variables", df.shape[1])
    col3.metric("Median streams (3yr)", f"{df.streams_3yr.median():,.0f}")

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

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ you have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

    st.markdown("##### ⚠️ One thing to notice before modelling")
    c1, c2 = st.columns(2)
    c1.metric("Median streams", f"{df.streams_3yr.median():,.0f}")
    c2.metric("Maximum streams", f"{df.streams_3yr.max():,.0f}")

    st.markdown("""
    The biggest song in the dataset has over **ten thousand times** the streams of
    a typical one. A handful of mega-hits sit far above everything else.

    That matters for the model: fitted on raw streams, a linear regression spends
    all its effort on those few outliers and fits the other 44,990 songs badly.
    On the Prediction page we model `log10(streams)` instead, which turns a
    multiplicative problem into an additive one. It lifts R² from 0.33 to 0.81.
    """)


# ====================================================================
# PAGE 2 - VISUALIZATION
# ====================================================================
elif page == "Visualization 📊":

    st.subheader("02 Data Viz 📊")

    df_viz = df.copy()
    df_viz["log_streams"] = np.log10(df_viz.streams_3yr)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution 📈", "By Category 📊", "Correlation Heatmap 🔥", "Audio vs Business 🎧"
    ])

    # ---- Tab 1: the skew ----
    with tab1:
        st.markdown("#### The target is extremely skewed")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.histplot(df_viz.streams_3yr, bins=60, ax=ax1, color="#4878A6")
        ax1.set_title("Raw streams")
        ax1.set_xlabel("3-year streams")

        sns.histplot(df_viz.log_streams, bins=60, ax=ax2, color="#4878A6")
        ax2.set_title("log10(streams)")
        ax2.set_xlabel("log10(3-year streams)")
        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "Left: almost every song is crushed against zero because a few hits "
            "stretch the axis. Right: the same data on a log scale becomes roughly "
            "bell-shaped — which is what linear regression assumes."
        )

    # ---- Tab 2: categories ----
    with tab2:
        st.markdown("#### Median streams by category")

        cat = st.selectbox("Group by", ["genre", "label_tier", "editorial_playlist"])

        grouped = (df_viz.groupby(cat)["streams_3yr"]
                   .median().sort_values(ascending=False))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(x=grouped.index.astype(str), y=grouped.values, ax=ax, color="#4878A6")
        ax.set_ylabel("Median 3-year streams")
        ax.set_xlabel(cat)
        ax.set_title(f"Median 3-year streams by {cat}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(grouped.round(0).astype(int).rename("median_streams"))

    # ---- Tab 3: heatmap ----
    with tab3:
        st.markdown("#### Correlation Matrix")

        df_numeric = df_viz.select_dtypes(include=np.number).drop(
            columns=["slow_burner", "is_hit"]
        )

        fig_corr, ax_corr = plt.subplots(figsize=(11, 9))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax_corr,
                    annot_kws={"size": 7})
        plt.tight_layout()
        st.pyplot(fig_corr)

        st.info(
            "Read the `log_streams` row. The strong correlations are all business "
            "variables — first-month streams, TikTok virality, playlist adds. "
            "The audio columns sit near zero."
        )

    # ---- Tab 4: the headline finding ----
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

        sample = df_viz.sample(3000, random_state=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

        ax1.scatter(sample[audio_pick], sample.log_streams, alpha=0.15, s=10, color="#4878A6")
        ax1.set_title(f"{audio_pick}  (r = {df_viz[audio_pick].corr(df_viz.log_streams):.3f})")
        ax1.set_xlabel(audio_pick)
        ax1.set_ylabel("log10(3-year streams)")

        ax2.scatter(sample[biz_pick], sample.log_streams, alpha=0.15, s=10, color="#C06C4F")
        ax2.set_title(f"{biz_pick}  (r = {df_viz[biz_pick].corr(df_viz.log_streams):.3f})")
        ax2.set_xlabel(biz_pick)

        plt.tight_layout()
        st.pyplot(fig)

        st.success(
            "Left is a shapeless cloud whatever you pick. Right slopes. "
            "This is the finding the whole project rests on."
        )


# ====================================================================
# PAGE 3 - PREDICTION
# ====================================================================
elif page == "Prediction 🎯":

    st.subheader("03 Prediction with Linear Regression 🎯")

    df2 = df.copy().dropna()

    ### Label Encoder to change text categories into number categories
    le_genre, le_label = LabelEncoder(), LabelEncoder()
    df2["genre"] = le_genre.fit_transform(df2["genre"])
    df2["label_tier"] = le_label.fit_transform(df2["label_tier"])

    ### Log-transform the heavy-tailed columns
    df2["log_first_month"] = np.log10(df2.first_month_streams.clip(lower=1))
    df2["log_prior_listeners"] = np.log10(df2.artist_prior_monthly_listeners.clip(lower=1))

    AUDIO_F = ["genre", "energy", "valence", "danceability", "acousticness",
               "instrumentalness", "tempo_bpm", "song_length_sec"]
    BUSINESS_F = ["label_tier", "marketing_budget", "playlist_adds_first_month",
                  "editorial_playlist", "tiktok_virality", "log_prior_listeners",
                  "featured_artist", "log_first_month"]

    st.sidebar.markdown("### Model settings")

    preset = st.sidebar.radio(
        "Feature set",
        ["All features", "Audio only", "Business only", "Custom"],
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

    ### i) X and y
    X = df2[features_selection]
    y = np.log10(df2["streams_3yr"]) if log_target else df2["streams_3yr"]

    st.markdown("##### Features (X)")
    st.dataframe(X.head())

    ### ii) train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    ### Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    ### iv) Evaluation
    st.markdown("##### Model performance")

    r2 = metrics.r2_score(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    mse = metrics.mean_squared_error(y_test, predictions)

    cols = st.columns(len(selected_metrics) if selected_metrics else 1)
    i = 0
    if "R² Score" in selected_metrics:
        cols[i].metric("R² Score", f"{r2:.3f}"); i += 1
    if "Mean Absolute Error (MAE)" in selected_metrics:
        cols[i].metric("MAE", f"{mae:,.3f}"); i += 1
    if "Mean Squared Error (MSE)" in selected_metrics:
        cols[i].metric("MSE", f"{mse:,.3f}"); i += 1

    if log_target:
        st.caption(
            f"On the log scale, a MAE of {mae:.2f} means the typical prediction is "
            f"off by a factor of about {10**mae:.1f}×."
        )

    if r2 > 0.7:
        st.success(f"✅ Strong fit — the model explains {r2:.1%} of the variation.")
    elif r2 > 0.3:
        st.warning(f"⚠️ Moderate fit — {r2:.1%} explained.")
    else:
        st.error(f"❌ Weak fit — only {r2:.1%} explained. Try a different feature set.")

    ### Actual vs predicted
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_test, predictions, alpha=0.15, s=10, color="#4878A6")
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "--r", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    plt.tight_layout()
    st.pyplot(fig)

    ### Which variables drive the prediction
    st.markdown("##### What drives the prediction")

    coefs = (pd.Series(model.coef_, index=features_selection)
             .sort_values(key=abs, ascending=False))

    fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.32 * len(coefs))))
    colors = ["#4878A6" if v > 0 else "#C06C4F" for v in coefs.values]
    ax2.barh(coefs.index[::-1], coefs.values[::-1], color=colors[::-1])
    ax2.axvline(0, color="#666", linewidth=1)
    ax2.set_xlabel("Coefficient")
    ax2.set_title("Driving variables")
    plt.tight_layout()
    st.pyplot(fig2)

    if log_target and "log_first_month" in coefs.index:
        st.info(
            f"`log_first_month` has a coefficient near {coefs['log_first_month']:.2f}. "
            "On a log-log scale that is close to proportional: ten times the "
            "first-month streams means roughly ten times the three-year streams. "
            "Early momentum compounds — success breeds success."
        )

    ### Try it yourself
    st.markdown("---")
    st.markdown("##### 🎧 Predict a song")
    st.caption("Set the inputs and the model predicts its 3-year streams.")

    if preset == "All features":
        c1, c2 = st.columns(2)
        with c1:
            in_first = c1.number_input("First-month streams", 100, 5_000_000, 20_000, step=1000)
            in_tiktok = c1.slider("TikTok virality (0-15)", 0.0, 15.0, 2.0)
            in_playlist = c1.slider("Playlist adds, first month", 0, 60, 5)
            in_editorial = c1.selectbox("Editorial playlist?", [0, 1])
        with c2:
            in_genre = c2.selectbox("Genre", sorted(df.genre.unique()))
            in_label = c2.selectbox("Label tier", sorted(df.label_tier.unique()))
            in_budget = c2.slider("Marketing budget (0-12)", 0.0, 12.0, 2.0)
            in_prior = c2.number_input("Artist's prior monthly listeners", 100, 20_000_000, 50_000, step=1000)

        row = {c: float(df2[c].median()) for c in features_selection}
        row.update({
            "log_first_month": np.log10(max(in_first, 1)),
            "tiktok_virality": in_tiktok,
            "playlist_adds_first_month": in_playlist,
            "editorial_playlist": in_editorial,
            "genre": int(le_genre.transform([in_genre])[0]),
            "label_tier": int(le_label.transform([in_label])[0]),
            "marketing_budget": in_budget,
            "log_prior_listeners": np.log10(max(in_prior, 1)),
        })

        x_new = pd.DataFrame([[row[c] for c in features_selection]], columns=features_selection)
        pred = model.predict(x_new)[0]
        pred_streams = 10 ** pred if log_target else pred

        st.metric("Predicted 3-year streams", f"{pred_streams:,.0f}")
        if log_target:
            lo, hi = 10 ** (pred - mae), 10 ** (pred + mae)
            st.caption(f"Typical range given the model's error: {lo:,.0f} to {hi:,.0f}")
    else:
        st.caption("Switch the feature set to **All features** to use the predictor.")
