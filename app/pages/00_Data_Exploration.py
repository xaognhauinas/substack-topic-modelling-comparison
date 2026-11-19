"""Notebook 00: Cleaning and filtering a raw 3.5M row scrape into a modelling ready corpus."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import COLOR_PALETTE, PLOT_TEMPLATE, callout, callouts, format_int, load_csv, load_json, get_artefact

folder_name = "00_data_exploration"

st.set_page_config(page_title="Data Exploration", page_icon="🔎", layout="wide")

# Overview
ov = get_artefact(load_json, folder_name, "corpus_overview.json")
dtypes = get_artefact(load_csv, folder_name, "dataset_dtypes.csv")
stats = get_artefact(load_csv, folder_name, "dataset_stats.csv")
sample = get_artefact(load_csv, folder_name, "dataset_sample.csv")   
# Dates
yearly = get_artefact(load_csv, folder_name, "yearly_posts.csv")
# Categories
cats = get_artefact(load_csv, folder_name, "categories.csv")
cat_year = get_artefact(load_csv, folder_name, "category_by_year.csv")
# Text quality
text_qual = get_artefact(load_csv, folder_name, "text_quality_flags.csv")
wc_dist = get_artefact(load_csv, folder_name, "combined_text_word_count.csv")             
wc_summary = get_artefact(load_csv, folder_name, "combined_text_word_count_summary.csv").set_index('statistic')
# Duplicates
dups = get_artefact(load_csv, folder_name, "duplicates_summary.csv")
# Language
langs = get_artefact(load_csv, folder_name, "language_shares.csv")
scripts = get_artefact(load_csv, folder_name, "script_shares.csv")
# Low information
low_info = get_artefact(load_csv, folder_name, "low_information_summary.csv")
# Corpus funnel
funnel = get_artefact(load_csv, folder_name, "corpus_funnel.csv")


removed = funnel.set_index("step")["dropped"]                 # posts removed at each pipeline step
STEP_DATES, STEP_LANG, STEP_DEDUP, STEP_LOWINFO, STEP_QUALITY = removed.index[1:6]
dup_rows = dups.set_index(dups["metric"].str.extract(r"^(\d+)\.")[0].astype(int))   # duplicate checks by number

st.title("Data Exploration")
st.caption("""Notebook 00: Explores the raw dataset of **3.5 million posts** to define inclusion criteria 
              for topic modelling and examine the characteristics of the filtered corpus.
           """)
with st.expander("How to read the notes in each tab"):
    callout("observation", "The figures and patterns observed in the data.")
    callout("decision", "The practical choices made for the cleaning pipeline, and the reasoning behind them.")
    callout("takeaway", "Key insights and open questions that inform the next steps.")

tabs = st.tabs(["1. Overview", "2. Dates", "3. Categories", "4. Publications", "5. Text quality",
                "6. Duplicates", "7. Language", "8. Low information posts", "9. Funnel and decisions"])

# 1 Overview ---------------------------------------------------------------------------------------------
COLUMN_NOTES = {
    "publication_domain": "Newsletter domain, which includes the author's handle.",
    "post_title": "Post title, with some empty values.",
    "article_url": "Post URL, where the segment following /p/ is the slug.",
    "author_name": "Unused and excluded from all displays.",
    "publish_date": "Publication timestamp, with a minimum recorded value of 0001-03-08.",
    "subtitle_summary": "Post subtitle, stored as an empty string when missing.",
    "title_char_length": "Character count of the title, as provided in the raw dataset.",
    "title_word_count": "Word count of the title, as provided in the raw dataset.",
    "has_subtitle": "Binary flag (1 if a subtitle is present), equal to 1 in nearly all rows.",
    "category": "Rule-based topic label across 11 categories.",
}

NUM_COLS = ["title_char_length", "title_word_count", "has_subtitle"]

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts", format_int(ov["raw_posts"]))
    c2.metric("Publications", format_int(ov["publications"]))
    c3.metric("Authors", format_int(ov["authors"]))
    c4.metric("Categories", ov["categories"])

    st.subheader("Sample rows")
    seed = st.session_state.get("sample_seed", 0)
    if st.button("Show another sample", key="another_sample"):
        seed += 1
        st.session_state["sample_seed"] = seed
    st.dataframe(
        sample.sample(min(10, len(sample)), random_state=seed), 
        hide_index=True)
    st.caption("""Random sample of rows from the raw dataset. Identifying fields (such as publication domains, author names, and article URLs) 
               have been removed, and long text fields are truncated."""
)
        

    st.subheader("Data types and completeness")
    dt = dtypes.copy()
    dt["unique %"] = (100 * dt["unique"] / dt["non_null"]).round(1)
    dt["note"] = dt["column"].map(COLUMN_NOTES)
    st.dataframe(dt, 
                 hide_index=True, 
                 column_config={"non_null": st.column_config.NumberColumn("non-null", format="%d"),
                                "null": st.column_config.NumberColumn(format="%d"),
                                "unique": st.column_config.NumberColumn(format="%d")})

    st.subheader("Basic statistics")
    s = stats.set_index("statistic")
    fmt = lambda v: "-" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Numeric columns**")
        st.dataframe(s[NUM_COLS].astype(float).apply(lambda col: col.map(fmt)), width="stretch")
    with right:
        st.markdown("**Dates (`publish_date`)**")
        dates = s["publish_date"].dropna().astype(str)
        dates["count"] = format_int(float(dates["count"]))
        st.dataframe(dates.to_frame("publish_date"), width="stretch")

    unique_title = 100 * dtypes.set_index("column").loc["post_title", "unique"] / ov["raw_posts"]
    unique_sub = 100 * dtypes.set_index("column").loc["subtitle_summary", "unique"] / ov["raw_posts"]

    
    callouts(
        "observation",
        "There are no null values in any column, though several titles have a length of zero. "
        "Because missing text is stored as empty strings, standard null checks detect nothing.",
        "Each row contains only a title and a subtitle, with **no full article text**. "
        "These short documents favour embedding-based models over word-count approaches.",
        f"There are no duplicate URLs. About {unique_title:.1f}% of titles and {unique_sub:.1f}% of subtitles are unique, "
        f"showing that some text repeats (see Tab 6. Duplicates).",
        "`publish_date` includes unrealistic values (such as a minimum of `0001-03-08`), which calls for a closer look. "
        "Meanwhile, `has_subtitle` is almost always 1 and carries almost no useful information.",
    )
    callout(
        "decision",
        "Check fields for empty strings, whitespace-only entries, and text without letters "
        "rather than relying only on null checks (see Tab 5. Text quality).",
    )



# 2 Dates -----------------------------------------------------------------------------------------------
with tabs[1]:
    st.caption(
        "Examining publication volume over time and distinguishing corrupted dates (such as year 1 or year 24) from genuine "
        "back-dated imports, which occur when blogs migrated to Substack retain their original publication dates."
    )

    st.subheader("Dates before the Substack launch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Substack Launch Date", ov["launch_date"])
    c2.metric("Posts before launch", format_int(ov["posts_before_launch"]), 
              f"{100 * ov['posts_before_launch'] / ov['raw_posts']:.3f}% of all posts", delta_color="off", delta_arrow="off")
    c3.metric("Publications before launch", format_int(ov["publications_before_launch"]))

    callouts(
        "observation",
        f"Publication dates span from `{ov['date_min']}` to `{ov['date_max']}`, whereas Substack launched in 2017.",
        f"A total of {format_int(ov['posts_before_launch'])} posts across {format_int(ov['publications_before_launch'])} publications "
        f"are dated before launch ({ov['launch_date']}), accounting for {100 * ov['posts_before_launch'] / ov['raw_posts']:.3f}% of all posts.",
        "Manual inspection of sample posts indicates that dates such as 2014 represent genuine back-dated imports from older blogs, "
        "whereas years 1, 2, and 24 are parsing errors.",
    )


    st.subheader("Volume over time")

    total = yearly["posts_count"].sum()
    last_two = yearly[yearly["publish_year"] >= 2025]["posts_count"].sum()
    per_year = yearly.set_index("publish_year")["posts_count"]
    recent_years = yearly[yearly["publish_year"] >= 2025]
    new_share = 100 * recent_years["new_domains_count"].sum() / recent_years["domains_count"].sum()

    start = st.slider("Select window start year", 2017, 2024, 2018, key="start_year")
    kept = yearly[yearly["publish_year"] >= start]["posts_count"].sum()
    st.write(f"Starting in **{start}** retains **{format_int(kept)}** posts ({100 * kept / total:.2f}% of the dataset).")

    st.caption(f"*Annual post volume from {start} onwards. Hover over a bar for additional details.*")
    log_y = st.toggle("Log scale", value=False, key="log_years")
    fig = px.bar(
        yearly[yearly["publish_year"] >= start],
        x="publish_year",
        y="posts_count",
        template=PLOT_TEMPLATE,
        labels={
            "publish_year": "Year",
            "posts_count": "Posts",
            "domains_count": "Active Domains",
            "new_domains_count": "New Domains",
        },
        hover_data={
            "publish_year": True,
            "posts_count": ":,",           
            "domains_count": ":,",
            "new_domains_count": ":,",
        },
        text="posts_count",
        log_y=log_y,
    )
    fig.update_traces(marker_color=COLOR_PALETTE[0],texttemplate="%{text:,}")
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    callouts(
        "observation",
        "Post and publication volumes increase sharply over time, rising from a low base between 2017 and 2019, nearly doubling "
        "year over year from 2020 to 2024, and accelerating further from 2025 onwards.",
        f"**{100 * last_two / total:.0f}% of all posts originate from 2025 and 2026.** Annual post volume grew from "
        f"{format_int(per_year[2020])} in 2020 to {format_int(per_year[2026])} in 2026, roughly a "
        f"{per_year[2026] / per_year[2020]:.0f}-fold increase.",
        f"This growth reflects both an influx of new newsletters and higher publishing frequency, with {new_share:.0f}% of "
        f"publications active in 2025 and 2026 being new to the platform.",
    )

    callouts(
        "decision",
        f"Restrict the analysis window to **2018-01-01** through **{ov['date_max']}**, capturing complete calendar years "
        f"following Substack's launch ({ov['launch_date']}).",
        f"Only {format_int(removed[STEP_DATES])} rows ({100 * removed[STEP_DATES] / ov['raw_posts']:.2f}%) fall outside "
        f"this window, meaning the cutoff primarily serves to remove impossible dates without substantially reducing sample size.",
        "Use **annual proportions** rather than raw counts for any topic analysis over time, as raw counts primarily capture "
        "overall platform growth.",
    )


# 3 Categories -----------------------------------------------------------------------------------------
with tabs[2]:
    st.caption(
        "The dataset card describes **rule-based topic classification** using regex matching. "
        "Examining how posts and publications are distributed across these categories."
    )
    
    st.subheader("Posts and publications by category")
    by = st.radio("Show share of", ["posts", "publications"], horizontal=True, key="cat_by")
    st.caption(f"*Share of {by} by category. Hover over a bar for additional details.*")
    col = "post_pct_share" if by == "posts" else "domains_pct_share"
    if col not in cats:                                                  
        cats["domains_pct_share"] = 100 * cats["domains_count"] / cats["domains_count"].sum()
    fig = px.bar(
        cats.sort_values(col), 
        x=col, 
        y="category", 
        orientation="h", 
        template=PLOT_TEMPLATE,
        labels={col: f"% of {by}", "category": ""},
        hover_data={"category": False, col: ":.1f"},
        text=col,
    )
    fig.update_traces(marker_color=COLOR_PALETTE[3], texttemplate="%{text:.1f}")
    fig.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    share = cats.set_index("category")["post_pct_share"]
    callouts(
        "observation",
        f"{share['General & Uncategorized']:.0f}% of posts fall under General & Uncategorized, with a Gini coefficient of "
        f"{ov['gini_posts_per_category']:.2f} across categories.",
        f"Posts and publications are concentrated in a few dominant categories: {share['Technology & AI']:.0f}% of posts are "
        f"in Technology & AI and {share['Culture, Arts & Literature']:.1f}% are in Culture, Arts & Literature. Niche "
        f"categories such as Science & Environment and Sports each account for under 5% of posts.",
        "Manual review of random samples per category shows that labels align reasonably well with post titles overall. "
        "However, clear misclassifications appear, such as wellness and cocktail posts labelled as Technology & AI, or "
        "social media and faith posts placed under Sports, suggesting the regex rules are overly broad.",
    )

    st.subheader("Categories over time")
    mode = st.radio("Display metric", ["Share of the year's posts", "Post counts"], horizontal=True, key="cat_year_mode")
    cy = cat_year[cat_year["publish_year"] >= 2018].melt("publish_year", var_name="category", value_name="posts").fillna(0)
    if mode.startswith("Share"):
        cy["posts"] = 100 * cy["posts"] / cy.groupby("publish_year")["posts"].transform("sum")
    st.caption(
        f"*{'Share of annual posts' if mode.startswith('Share') else 'Post count'} by category over time. "
        f"Hover over a point for additional details. Double-click a legend item to isolate a category, or double-click again to reset.*"
    )

    fig = px.line(
        cy, 
        x="publish_year", 
        y="posts", 
        color="category", 
        markers=True, 
        template=PLOT_TEMPLATE,
        labels={"publish_year": "Year", "posts": "% of annual posts" if mode.startswith("Share") else "Posts", "category": "Category"},
        hover_data={"category": True, "posts": ":.1f" if mode.startswith("Share") else ":,"},
        )
    fig.update_layout(height=580, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    callouts(
        "observation",
        "Post volume increases across all categories rather than being driven by a single theme.",
    )
    callout(
        "decision",
        "Do **not** use `category` as ground truth for evaluating topic models. Retain it solely as a rough contextual label.",
    )

# 4 Publications --------------------------------------------------------------------------------------
with tabs[3]:
    st.caption("Examining how posts are concentrated across publications to determine whether a per-publication cap is required.")

    st.subheader("Publications concentration")
    q = ov["posts_per_publication"]
    qdf = pd.DataFrame({"statistic": ["min", "25%", "median", "75%", "max"],
                        "posts per publication": [q["min"], q["25%"], q["50%"], q["75%"], q["max"]]})
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            qdf, 
            x="statistic", 
            y="posts per publication", 
            template=PLOT_TEMPLATE, 
            text="posts per publication")
        fig.update_traces(marker_color=COLOR_PALETTE[2])
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("Publications", format_int(ov["publications"]))
        st.metric("Mean posts per publication", f"{q['mean']:.1f}")
        st.metric("Gini coefficient (posts per publication)", f"{ov['gini_posts_per_publication']:.3f}")

    callouts(
        "observation",
        f"The Gini coefficient of posts per publication is {ov['gini_posts_per_publication']:.3f}, which at first suggests "
        f"posts are evenly spread across the platform.",
        f"However, the maximum is exactly {q['max']:.0f}, and the median and 75th percentile are "
        f"{'both ' + format(q['50%'], '.0f') if q['50%'] == q['75%'] else format(q['50%'], '.0f') + ' and ' + format(q['75%'], '.0f')}. "
        f"This apparent balance is likely a **result of how the data was collected**, reflecting a scraping cap per publication.",
        "Manual checks of prominent Substack newsletters confirm that their actual archives are much larger than what appears in the dataset.",
    )
    callouts(
        "decision",
        f"Apply no additional cap per publication, as the dataset is already capped at {q['max']:.0f} posts per publication.",
        "While no single publication dominates the sample, true publishing volume and posting frequency "
        "cannot be reliably measured.",
    )


# 5 Text quality ---------------------------------------------------------------------------------------
with tabs[4]:
    st.caption(
        "Standard null checks (`df.info()`) find no missing values, yet some titles have a length of zero. "
        "Because missing text is stored as **empty strings**, this section examines empty, whitespace-only, "
        "very short, and non-alphabetic entries."
    )
    f = text_qual.copy()
    parts = f["metric"].str.extract(r"^(post_title|subtitle_summary|combined_text)_(.+)$")
    f["field"], f["check"] = parts[0], parts[1].str.replace("_", " ")

    st.subheader("Text quality checks")
    st.caption(f"*Share of posts flagging each quality check across text fields. Hover over a bar for additional details.*")
    fig = px.bar(
        f, 
        x="check", 
        y="pct_true", 
        color="field", 
        barmode="group", 
        template=PLOT_TEMPLATE,
        labels={"pct_true": "% of posts", "check": "", "count_true": "Posts"}, 
        hover_data={"count_true": ":,", "pct_true": ":.1f", "check": False},
        color_discrete_sequence=[COLOR_PALETTE[1],COLOR_PALETTE[3],COLOR_PALETTE[0]],
        text="pct_true")
    fig.update_traces(texttemplate="%{text:.1f}")
    fig.update_layout(height=380, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    row = lambda name: f.loc[f["metric"] == name, "pct_true"].iloc[0]

    callouts(
        "observation",
        f"{row('combined_text_has_html'):.1f}% of posts contain raw HTML entities in the combined title and subtitle.",
        "Some entities represent emoji or characters encoded as numeric codes (such as `😺` stored as `&#128570;`), "
        "so HTML entities are decoded prior to running emoji checks and computing structural text metrics.",
        f"Letter-poor text is defined as containing fewer than half alphabetic characters. "
        f"Approximately {row('combined_text_is_letter_poor'):.1f}% of combined text entries are letter-poor, "
        f"and {row('combined_text_has_emoji'):.1f}% contain emoji. No encoding corruption (mojibake) was detected.",
    )
    callouts(
        "decision",
        "Decode HTML entities and strip HTML tags in both fields, then concatenate them as `title. subtitle`.",
    )

    st.subheader("Word count distribution of combined title and subtitle")
    max_cap = int(wc_dist["word_count"].max()) - 1               # the last bin pools everything above the cap
    
    k1, k2, k3 = st.columns([2, 1, 1])
    cap = k1.slider("Maximum word count shown", 10, max_cap, min(40, max_cap), key="wc_cap")
    measure = k2.radio("Display metric", ["Posts", "Share of posts"], key="wc_measure")
    log_y = k3.toggle("Log scale", value=False, key="wc_log")

    shown = wc_dist.loc[wc_dist["word_count"] <= cap, ["word_count", "posts"]].copy()
    tail = int(wc_dist.loc[wc_dist["word_count"] > cap, "posts"].sum())

    shown = pd.concat(
        [shown, pd.DataFrame({"word_count": [cap + 1], "posts": [tail]})], 
        ignore_index=True)
    shown["label"] = [str(n) if n <= cap else f"{cap + 1}+" for n in shown["word_count"]]
    shown["share"] = 100 * shown["posts"] / shown["posts"].sum()
    shown["rule"] = ["under 4 words (dropped)" if n < 4 else "kept" for n in shown["word_count"]]
    y = "posts" if measure == "Posts" else "share"

    st.caption(
        f"*Raw dataset after HTML decoding. Counts above {cap} words are pooled into the final bar. "
        f"Hover over a bar for additional details.*"
    )
    fig = px.bar(
        shown, 
        x="label", 
        y=y, 
        color="rule", 
        template=PLOT_TEMPLATE, 
        log_y=log_y,
        color_discrete_map={"kept": COLOR_PALETTE[3], "under 4 words (dropped)": COLOR_PALETTE[0]},
        labels={"label": "Words in title + subtitle", "posts": "Posts", "share": "% of posts", "rule": ""},
        hover_data={"rule": False, "posts": ":,", "share": ":.2f"})
    fig.update_layout(height=360, margin=dict(t=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    

    cum = wc_dist.set_index("word_count")["cumulative_proportion"]
    over_cap = 100 * wc_dist["posts"].iloc[-1] / wc_dist["posts"].sum()

    callouts(
        "observation",
        f"Combined text entries are typically concise, with a median word count of {int(wc_summary.loc['50%'].iloc[0])}.",
        f"The distribution has a long right tail, with {over_cap:.1f}% of posts exceeding {int(wc_dist['word_count'].max()) - 1} words"
        f" and the longest reaching {int(wc_summary.loc['max'].iloc[0]):,} words.",
        f"Overall, around {100 * cum.loc[3]:.1f}% of posts contain three words or fewer once the title and subtitle are combined. "
        "Manual review confirms that these very short posts are typically low-information entries.",
    )

    callout(
        "decision",
        "Remove posts that are letter-poor or contain fewer than four words.",
    )


# 6 Duplicates -----------------------------------------------------------------------------------------
with tabs[5]:
    st.caption(
        "Examining different types of duplicates, including exact copies of publication, title, and subtitle, "
        "recurring series with the same publication and title, and titles or subtitles reused across the platform."
    )
    metric_name = {
        1: "Duplicate URLs",
        2: "Exact clones within a publication",
        3: "Exact clones within a publication (normalised series numbers)",
        4: "Recurring titles within a publication",
        5: "Recurring titles within a publication (normalised series numbers)",
        6: "Same title and subtitle across publications",
        7: "Reused titles (global)",
        8: "Reused subtitles (global)",
        9: "Reused URL slugs (global)",
    }
    d = dups.copy()
    d["check"] = d["metric"].str.extract(r"^(\d+)\.")[0].astype(int).map(metric_name)

    st.subheader("Types of duplicates")
    st.caption(
        "*Share of posts by duplicate type. The categories overlap, hence bars should not be summed. "
        "Hover over a bar for additional details.*"
    )
    fig = px.bar(
        d.iloc[::-1], 
        x="pct_share", 
        y="check", 
        orientation="h", 
        template=PLOT_TEMPLATE,
        labels={"pct_share": "% of posts", "check": "", "rows": "Posts"}, 
        hover_data={"rows": ":,", "metric": True, "check": False, "pct_share": ":.2f"},
        text="pct_share")
    fig.update_traces(marker_color=COLOR_PALETTE[0],texttemplate="%{text:.1f}")
    fig.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    
    share_of = lambda n: dup_rows.loc[n, "pct_share"]
    callouts(
        "observation",
        f"There are no duplicate URLs, and exact clones (matching publication, title, and subtitle) account for under 0.4% "
        f"({share_of(2):.2f}%), consisting mostly of recurring editions.",
        f"Recurring editions frequently contain issue or chapter numbers. Replacing digits with a placeholder (turning both "
        f"`Chapter 33` and `#133` into `0`) increases exact clones from {share_of(2):.1f}% to {share_of(3):.1f}%, and "
        f"recurring titles within a publication from {share_of(4):.1f}% to {share_of(5):.1f}%.",
        f"Generic titles, subtitles, and URL slugs are widely reused across different publications, appearing in "
        f"{share_of(7):.0f}% to {share_of(9):.0f}% of posts.",
    )
    with st.expander("Examples of generic cross-publication reuse"):
        st.markdown("- **Announcement:** `coming soon`\n"
                    "- **Greetings:** `dear friends,`, `hi everyone,`, `hello!`\n"
                    "- **Recruitment:** `join my new subscriber chat`, `join me on notes`, `join-my-new-subscriber-chat`")
    callouts(
        "decision",
        f"Remove duplicates sharing the same publication and **normalised title and subtitle**. This affects "
        f"{format_int(dup_rows.loc[3, 'rows'])} posts ({share_of(3):.1f}%) in the raw dataset.",
        f"Retain recurring titles within the same publication that feature **distinct subtitles** to help capture nuanced micro-themes within an ongoing series. "
        f"This preserves roughly {share_of(5) - share_of(3):.0f}% of posts ({share_of(5):.1f}% minus {share_of(3):.1f}%, approximated due to overlapping checks).",
        "Filter out generic, low-information entries (such as announcements, greetings, and subscriber prompts) in a separate step (see Tab 8. Low-information posts).",
    )

# 7 Language -------------------------------------------------------------------------------------------
with tabs[6]:
    st.caption(
        "The dataset includes non-English posts, which can skew English topic models. "
        "Language is detected using `fastText`, as it handles large text datasets efficiently "
        "and outperformed `langdetect` in initial sample tests."
    )

    st.subheader("Language and script distribution")
    st.caption(
            "*Share of posts by detected language and Unicode script. Hover over a bar for additional details.*"
        )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            langs.head(8).iloc[::-1], 
            x="pct_share", 
            y="language", 
            orientation="h", 
            template=PLOT_TEMPLATE,
            labels={"pct_share": "% of posts", "language": "Language (fastText)", "size": "Posts"},
            hover_data={"language": True, "pct_share": ":.1f", "size": ":,"},
            text="pct_share")
        fig.update_traces(marker_color=COLOR_PALETTE[3],texttemplate="%{text:.1f}")
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(
            scripts.head(6).iloc[::-1], 
            x="pct_share", 
            y="script", 
            orientation="h", 
            template=PLOT_TEMPLATE,
            labels={"pct_share": "% of posts", "size": "Posts", "script": "Script (Unicode)"},
            hover_data={"script": True, "pct_share": ":.1f", "size": ":,"},
            text="pct_share")
        fig.update_traces(marker_color=COLOR_PALETTE[2],texttemplate="%{text:.1f}")
        fig.update_layout(height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    pct_of = lambda df, key, val: df.loc[df[key] == val, "pct_share"].iloc[0]

    callouts(
        "observation",
        f"Latin script accounts for {pct_of(scripts, 'script', 'LATIN'):.1f}% of posts, meaning script filtering alone "
        f"cannot distinguish English from Spanish ({pct_of(langs, 'language', 'es'):.1f}%), "
        f"Portuguese ({pct_of(langs, 'language', 'pt'):.1f}%), French ({pct_of(langs, 'language', 'fr'):.1f}%), "
        f"or Italian ({pct_of(langs, 'language', 'it'):.1f}%).",
        f"According to `fastText`, {pct_of(langs, 'language', 'en'):.0f}% of posts are English, with a mean confidence "
        f"score of 0.85.",
        "`fastText` exhibits lower confidence on short text, prompting the use of Unicode script as a secondary check. "
        "Manual review of English predictions scoring between 0.1 and 0.3 confidence confirmed that most were valid predictions.",
        "Several English-labelled posts use decorative Unicode characters (such as mathematical bold alphabets) or "
        "consist solely of symbols and digits.",
    )

    callout(
        "decision",
        f"Retain posts classified as English in Latin or mathematical-alphabet scripts with a confidence score above 0.1. "
        f"This removes {format_int(ov['removed_by_language'])} posts ({ov['pct_removed_by_language']:.0f}% of raw posts), "
        f"representing the largest single reduction in the pipeline.",
    )

    callout(
        "takeaway",
        "Titles using decorative Unicode alphabets pass the filter but are **not normalised to standard ASCII characters**, "
        "causing word-count models to treat them as out-of-vocabulary tokens. Planned improvement: apply Unicode normalisation (NFKC).",
    )

# 8 Low information -------------------------------------------------------------------------------------
with tabs[7]:
    st.caption(
        "Keywords (such as `coming soon`, `join my ...`, and `join us ...`) and placeholder slugs (such as "
        "`/p/coming-soon`) serve as a baseline filter for announcements, greetings, and subscriber posts. "
        "These rules are intentionally **lenient** to avoid dropping deep-dive articles that contain common low-information phrases "
        "(for example, an article examining the cultural impact of Zoom), so some promotional or placeholder posts may still pass through."
    )

    st.subheader("Low-information flags")
    st.caption(
        "*Share of posts matching each low-information flag. Hover over a bar for additional details.*"
    )
    a = low_info.copy()
    a["metric"] = a["metric"].map(lambda m: "keyword flag" if m.startswith("keyword") else "slug flag" if m.startswith("slug")
                                  else "empty slug" if m.startswith("empty") else m)
    fig = px.bar(
        a.iloc[::-1], 
        x="pct_share", 
        y="metric", 
        orientation="h", 
        template=PLOT_TEMPLATE,
        labels={"pct_share": "% of posts", "metric": "", "rows": "Posts"}, 
        hover_data={"metric": False, "pct_share": ":.1f", "rows": ":,"},
        text="pct_share")
    fig.update_traces(marker_color=COLOR_PALETTE[0],texttemplate="%{text:.1f}")
    fig.update_layout(height=300, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    by_name = low_info.set_index(low_info["metric"].str.split().str[0].str.split("-").str[0])   # keyword, slug, empty, either

    callouts(
        "observation",
        f"Filters for \"coming soon\", \"join my...\", and welcome-style URL slugs flag approximately "
        f"{by_name.loc['either', 'pct_share']:.0f}% of posts (keyword: {by_name.loc['keyword', 'pct_share']:.2f}%, "
        f"slug: {by_name.loc['slug', 'pct_share']:.2f}%, missing `/p/` path: {by_name.loc['empty', 'pct_share']:.2f}%).",
        "Several posts lacking a `/p/` URL path carry the title \"No title\", which indicates broken or invalid links.",
    )
    callouts(
        "decision",
        f"Remove flagged posts. Overall, {format_int(by_name.loc['either', 'rows'])} posts "
        f"({by_name.loc['either', 'pct_share']:.0f}%) are flagged in the raw dataset. "
        f"Given the dataset's scale, tolerate a small number of false positives rather than retain low-information placeholder text in the corpus.",
        "Remove posts whose URL slugs do not follow the standard `/p/` structure.",
    )

# 9 Funnel and decisions ------------------------------------------------------------------------------
with tabs[8]:

    st.caption(
        "Consolidating every filtering rule into a single cleaning pipeline and tracking remaining post counts after each step."
    )

    st.subheader("Cleaning funnel")
    fig = px.funnel(funnel, 
                    x="remaining", 
                    y="step", 
                    template=PLOT_TEMPLATE,
                    labels={
                        "remaining": "Remaining posts",
                        "dropped": "Posts removed",
                        "dropped_pct_of_previous": "% removed from previous step",
                        "remaining_pct_of_raw": "% of raw posts",
                    },
                    hover_data={
                        "step": False,
                        "remaining": ":,",
                        "dropped": ":,",
                        "dropped_pct_of_previous": ":.1f",
                        "remaining_pct_of_raw": ":.1f",
                    },
                    )
    fig.update_traces(marker_color=COLOR_PALETTE[0], texttemplate="%{x:.3s}")
    fig.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    callout(
        "observation",
        f"**{format_int(funnel['remaining'].iloc[-1])}** of {format_int(ov['raw_posts'])} posts remain "
        f"({funnel['remaining_pct_of_raw'].iloc[-1]:.1f}% of the raw dataset).",
    )

    st.subheader("Summary of findings and decisions")
    st.dataframe(
        pd.DataFrame({
            "Topic": [
                "Dates",
                "`category`",
                "Publications",
                "Language",
                "Duplicates and series",
                "Low-information posts",
                "Short or letter-poor text",
            ],
            "Observations": [
                f"{100 * last_two / total:.0f}% of posts are from 2025 to 2026, with {format_int(ov['posts_before_launch'])} posts predating Substack's launch",
                "Rule-based labels with visible misclassifications",
                f"Capped at {q['max']:.0f} posts per publication during collection, resulting in a low Gini coefficient",
                f"{100 - ov['pct_removed_by_language']:.0f}% English, though `fastText` shows lower confidence on short text",
                f"Normalising edition numbers raises exact clones from {share_of(2):.1f}% to {share_of(3):.1f}%, while roughly "
                f"{share_of(5) - share_of(3):.0f}% are recurring titles with distinct subtitles",
                f"Approximately {by_name.loc['either', 'pct_share']:.0f}% flagged by lenient keyword and URL slug rules",
                f"Around {100 * cum.loc[3]:.0f}% of posts contain three words or fewer",
            ],
            "Decisions": [
                "Restrict window to 2018-01-01 onwards",
                "Exclude as ground truth and retain as rough context",
                "Apply no additional per-publication cap",
                "Retain English posts (confidence > 0.1) in Latin or mathematical-alphabet scripts",
                "Remove exact clones (normalised title and subtitle) within the same publication",
                "Remove flagged posts and non-standard `/p/` URL slugs",
                "Remove letter-poor posts and posts with fewer than 4 words",
            ],
            "Posts removed": [
                format_int(removed[STEP_DATES]),
                "0",
                "0",
                format_int(removed[STEP_LANG]),
                format_int(removed[STEP_DEDUP]),
                format_int(removed[STEP_LOWINFO]),
                format_int(removed[STEP_QUALITY]),
            ],
        }),
        hide_index=True,
        row_height=80
    )

    st.subheader("Next steps")
    callouts(
        "takeaway",
        "The next notebook fits and evaluates four topic-modelling approaches on a sample of this cleaned corpus.",
    )


