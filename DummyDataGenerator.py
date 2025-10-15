import streamlit as st
import pandas as pd
from faker import Faker
import random
import io
import uuid
from datetime import datetime, timedelta
import math

fake = Faker()

# --- Field types (generators without per-field params) ---
FIELD_TYPES = {
    "Unique ID (Sequential)": None,  # handled separately
    "Unique ID (UUID)": lambda: str(uuid.uuid4()),
    "Full Name": lambda: fake.name(),         # replaced old "Name"
    "First Name": lambda: fake.first_name(),  # new
    "Last Name": lambda: fake.last_name(),    # new
    "Email": lambda: fake.email(),
    "Phone": lambda: fake.phone_number(),
    "Address": lambda: fake.address().replace("\n", ", "),
    "Company": lambda: fake.company(),
    "Age": lambda: random.randint(18, 70),
    "Job Title": lambda: fake.job(),
    "Country": lambda: fake.country(),
    "Date": lambda: fake.date_between(start_date="-5y", end_date="today"),
    "Custom Text": lambda: fake.word(),
    "Custom Number": lambda: random.randint(1000, 9999),
    # param-handled types
    "Range (0-10)": None,
    "Comment (Sentiment)": None,
    "Conditional Range (Based on Comment Sentiment)": None,
    "Constant": None,
    "Custom Enum": None,
}

# --- Comment pools ---
COMMENTS_POSITIVE = [
    "Great experience, highly recommended.",
    "Really pleased with the result!",
    "Exceeded expectations — very happy.",
    "Fantastic service and friendly staff.",
    "Absolutely loved it, 10/10.",
]

COMMENTS_NEUTRAL = [
    "It was okay, nothing special.",
    "Average experience overall.",
    "Met basic expectations.",
    "Neither good nor bad — acceptable.",
    "Satisfactory but could be better.",
]

COMMENTS_NEGATIVE = [
    "Very disappointed with the service.",
    "Not what I expected, poor experience.",
    "Would not recommend, needs improvement.",
    "Bad experience — will not return.",
    "Unhappy with the outcome.",
]


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _normalize_probs(p):
    s = sum(p)
    if s == 0:
        return [1/3, 1/3, 1/3]
    return [x / s for x in p]


def _apply_trend(base_probs, time_factor, trend_type, strength):
    bp = list(base_probs)
    pos, neu, neg = bp

    if trend_type == "Increasing Positive":
        pos = pos + strength * time_factor * (1 - pos)
        neg = neg - strength * time_factor * neg
        neu = 1 - pos - neg
    elif trend_type == "Decreasing Positive":
        pos = pos - strength * time_factor * pos
        neg = neg + strength * time_factor * (1 - neg)
        neu = 1 - pos - neg
    elif trend_type == "Cyclical":
        cyc = math.sin(2 * math.pi * time_factor)
        pos = pos + strength * (cyc * 0.5)
        pos = _clamp(pos, 0.01, 0.99)
        remain = 1 - pos
        base_pair_sum = neu + neg
        if base_pair_sum == 0:
            neu = neg = remain / 2
        else:
            neu = remain * (neu / base_pair_sum)
            neg = remain * (neg / base_pair_sum)
    elif trend_type == "Random Fluctuation":
        jitter_pos = random.gauss(0, 0.1 * strength)
        jitter_neu = random.gauss(0, 0.1 * strength)
        jitter_neg = random.gauss(0, 0.1 * strength)
        pos = pos + jitter_pos
        neu = neu + jitter_neu
        neg = neg + jitter_neg

    pos = _clamp(pos, 0.0, 1.0)
    neu = _clamp(neu, 0.0, 1.0)
    neg = _clamp(neg, 0.0, 1.0)
    pos, neu, neg = _normalize_probs((pos, neu, neg))
    return pos, neu, neg


def _parse_enum_values(raw: str):
    """Parse comma-separated enum values; return list of trimmed non-empty strings."""
    items = [s.strip() for s in raw.split(",")]
    items = [s for s in items if s != ""]
    return items


def _parse_weights(raw: str, n):
    """Parse comma-separated weights into list of floats length n; if invalid return None."""
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    try:
        weights = [float(p) for p in parts]
    except Exception:
        return None
    if len(weights) != n:
        return None
    # normalize non-negative
    weights = [max(0.0, w) for w in weights]
    if sum(weights) == 0:
        return None
    return weights


def generate_dummy_data(rows, schema, global_timeline=None):
    """
    Generate dataset:
      - First pass: generate non-comment, non-conditional fields (so date fields exist)
      - Second pass: generate comment fields (trend-aware)
      - Third pass: evaluate conditional ranges & sequential IDs
    """
    # PASS 1: base rows (includes Constants & Custom Enums)
    base_rows = []
    for i in range(rows):
        row = {}
        for field in schema:
            fname = field["name"]
            ftype = field["type"]

            if ftype == "Unique ID (Sequential)":
                # sequential handled later to account for pad/start/step per-field
                continue

            if ftype == "Constant":
                row[fname] = field.get("value", "")
                continue

            if ftype == "Custom Enum":
                raw_vals = field.get("values_raw", "")
                vals = _parse_enum_values(raw_vals)
                if not vals:
                    row[fname] = None
                    continue
                mode = field.get("enum_mode", "Random")
                weights_raw = field.get("weights_raw", "")
                weights = _parse_weights(weights_raw, len(vals))
                if mode == "Cycle":
                    row[fname] = vals[i % len(vals)]
                else:  # Random (respect weights if present)
                    if weights:
                        row[fname] = random.choices(vals, weights=weights, k=1)[0]
                    else:
                        row[fname] = random.choice(vals)
                continue

            if ftype == "Range (0-10)":
                min_v = int(field.get("min", 0))
                max_v = int(field.get("max", 10))
                if min_v > max_v:
                    min_v, max_v = max_v, min_v
                if field.get("float", False):
                    row[fname] = round(random.uniform(min_v, max_v), field.get("precision", 2))
                else:
                    row[fname] = random.randint(min_v, max_v)
                continue

            if ftype == "Date Joined":
                gen = FIELD_TYPES.get(ftype)
                if callable(gen):
                    row[fname] = gen()
                else:
                    row[fname] = None
                continue

            # skip comment and conditional-range types here (we'll handle later)
            if ftype in ("Comment (Sentiment)", "Conditional Range (Based on Comment Sentiment)"):
                continue

            gen = FIELD_TYPES.get(ftype)
            if callable(gen):
                row[fname] = gen()
            else:
                row[fname] = None

        base_rows.append(row)

    # PASS 2: comments (trend aware)
    sentiments_per_row = [dict() for _ in range(rows)]

    def _compute_date_range(field_name):
        dates = []
        for r in base_rows:
            v = r.get(field_name)
            if v is None:
                continue
            try:
                dt = pd.to_datetime(v)
                dates.append(dt)
            except Exception:
                continue
        if not dates:
            return None, None
        min_d = min(dates)
        max_d = max(dates)
        return min_d, max_d

    for i, row in enumerate(base_rows):
        for field in schema:
            fname = field["name"]
            ftype = field["type"]

            if ftype != "Comment (Sentiment)":
                continue

            # Trend settings
            trend_enabled = field.get("trend_enabled", False)
            trend_type = field.get("trend_type", "Increasing Positive")
            trend_strength = float(field.get("trend_strength", 0.5))
            timeline_source = field.get("timeline_source", "Global timeline")
            date_field_ref = field.get("timeline_date_field", "")
            base_preset = field.get("base_preset", "Balanced")

            if base_preset == "Balanced":
                base_prob = (0.34, 0.33, 0.33)
            elif base_preset == "Positive-heavy":
                base_prob = (0.6, 0.2, 0.2)
            elif base_preset == "Negative-heavy":
                base_prob = (0.2, 0.2, 0.6)
            elif base_preset == "Neutral-heavy":
                base_prob = (0.2, 0.6, 0.2)
            else:
                base_prob = (0.34, 0.33, 0.33)

            time_factor = 0.0
            if trend_enabled:
                if timeline_source == "Global timeline" and global_timeline is not None:
                    if rows > 1:
                        time_factor = i / (rows - 1)
                    else:
                        time_factor = 0.0
                elif timeline_source == "Date field" and date_field_ref:
                    min_d, max_d = _compute_date_range(date_field_ref)
                    try:
                        this_dt = row.get(date_field_ref)
                        if this_dt is not None:
                            this_dt = pd.to_datetime(this_dt)
                        if min_d is not None and max_d is not None and this_dt is not None and min_d != max_d:
                            total = (max_d - min_d).total_seconds()
                            elapsed = (this_dt - min_d).total_seconds()
                            time_factor = _clamp(elapsed / total)
                        else:
                            time_factor = i / (rows - 1) if rows > 1 else 0.0
                    except Exception:
                        time_factor = i / (rows - 1) if rows > 1 else 0.0
                else:
                    time_factor = i / (rows - 1) if rows > 1 else 0.0

            if trend_enabled:
                p_pos, p_neu, p_neg = _apply_trend(base_prob, time_factor, trend_type, trend_strength)
            else:
                sentiment_choice = field.get("sentiment", "Random")
                if sentiment_choice == "Random":
                    p_pos, p_neu, p_neg = base_prob
                elif sentiment_choice == "Positive":
                    p_pos, p_neu, p_neg = (1.0, 0.0, 0.0)
                elif sentiment_choice == "Neutral":
                    p_pos, p_neu, p_neg = (0.0, 1.0, 0.0)
                elif sentiment_choice == "Negative":
                    p_pos, p_neu, p_neg = (0.0, 0.0, 1.0)
                else:
                    p_pos, p_neu, p_neg = base_prob

            r = random.random()
            if r < p_pos:
                sentiment = "Positive"
                comment_text = random.choice(COMMENTS_POSITIVE)
            elif r < p_pos + p_neu:
                sentiment = "Neutral"
                comment_text = random.choice(COMMENTS_NEUTRAL)
            else:
                sentiment = "Negative"
                comment_text = random.choice(COMMENTS_NEGATIVE)

            row[fname] = comment_text
            sentiments_per_row[i][fname] = sentiment

    # PASS 3: conditional ranges and sequential ids
    final_rows = []
    for i, row in enumerate(base_rows):
        # Sequential IDs: any field of that type should be added now
        for field in schema:
            fname = field["name"]
            ftype = field["type"]

            if ftype == "Unique ID (Sequential)":
                start = int(field.get("start", 1))
                step = int(field.get("step", 1))
                pad_zeros = int(field.get("pad_zeros", 0))
                # width = digits of start + pad_zeros
                width = len(str(start)) + max(0, pad_zeros)
                val = start + i * step
                row[fname] = str(val).zfill(width)

            elif ftype == "Conditional Range (Based on Comment Sentiment)":
                depends_on = field.get("depends_on", "")
                pmin, pmax = int(field.get("positive_min", 0)), int(field.get("positive_max", 10))
                nmin, nmax = int(field.get("neutral_min", 0)), int(field.get("neutral_max", 10))
                negmin, negmax = int(field.get("negative_min", 0)), int(field.get("negative_max", 10))
                amin, amax = int(field.get("any_min", 0)), int(field.get("any_max", 10))
                use_float = field.get("float", False)
                precision = int(field.get("precision", 2))

                actual_sent = sentiments_per_row[i].get(depends_on)
                if actual_sent == "Positive":
                    min_v, max_v = pmin, pmax
                elif actual_sent == "Neutral":
                    min_v, max_v = nmin, nmax
                elif actual_sent == "Negative":
                    min_v, max_v = negmin, negmax
                else:
                    min_v, max_v = amin, amax

                if min_v > max_v:
                    min_v, max_v = max_v, min_v

                if use_float:
                    val = round(random.uniform(min_v, max_v), precision)
                else:
                    val = random.randint(min_v, max_v)
                row[fname] = val

        final_rows.append(row)

    return pd.DataFrame(final_rows)


# --- UI ---
st.set_page_config(page_title="Custom Dummy Data Generator", layout="wide")
st.title("📊 Custom Dummy Data Generator")
st.markdown("Generate dummy data. Full/First/Last name fields added; everything else preserved.")

st.sidebar.header("⚙️ Settings")
rows = st.sidebar.slider("Number of rows", 10, 5000, 100, step=10)

# Global timeline (optional)
st.sidebar.subheader("📈 Global timeline (optional)")
use_global_timeline = st.sidebar.checkbox("Use global timeline mapping (index → dates)", value=False, key="use_global_tl")
global_timeline = None
if use_global_timeline:
    gstart = st.sidebar.date_input("Global start date", value=datetime.now().date() - timedelta(days=365))
    gend = st.sidebar.date_input("Global end date", value=datetime.now().date())
    if gstart >= gend:
        st.sidebar.error("Global start date must be before end date.")
    else:
        global_timeline = {"start_date": pd.to_datetime(gstart), "end_date": pd.to_datetime(gend)}

st.sidebar.subheader("🛠️ Add Custom Fields")
num_fields = st.sidebar.number_input("Number of fields", 1, 40, 6)

schema = []
type_options = list(FIELD_TYPES.keys())

EMOJI = {
    "Unique ID (Sequential)": "🔢",
    "Unique ID (UUID)": "🆔",
    "Full Name": "👤",
    "First Name": "🙂",
    "Last Name": "🔖",
    "Email": "✉️",
    "Phone": "📞",
    "Address": "🏠",
    "Company": "🏢",
    "Age": "🎂",
    "Job Title": "💼",
    "Country": "🌍",
    "Date Joined": "📅",
    "Custom Text": "📝",
    "Custom Number": "🔣",
    "Range (0-10)": "📏",
    "Comment (Sentiment)": "💬",
    "Conditional Range (Based on Comment Sentiment)": "🎯",
    "Constant": "🔒",
    "Custom Enum": "🧩",
}

for i in range(num_fields):
    with st.sidebar.expander(f"Field {i+1}", expanded=(i < 6)):
        col1, col2 = st.columns([2, 2])
        default_name = f"Field{i+1}"
        with col1:
            field_name = st.text_input("Name", value=default_name, key=f"name_{i}")
        with col2:
            field_type = st.selectbox("Type", options=type_options, index=min(i, len(type_options) - 1), key=f"type_{i}")

        st.markdown(f"**{EMOJI.get(field_type, '')} {field_name or default_name} — _{field_type}_**")

        field_def = {"name": field_name or default_name, "type": field_type, "default_name": default_name}

        # Unique ID (Sequential) options
        if field_type == "Unique ID (Sequential)":
            start = st.number_input("Start", value=1, step=1, key=f"seq_start_{i}")
            step = st.number_input("Step", value=1, step=1, key=f"seq_step_{i}")
            pad_zeros = st.number_input("Number of zeros (additional)", min_value=0, value=3, step=1, key=f"seq_pad_{i}")
            st.caption("Width = digits(start) + Number of zeros. Example: start=1, zeros=3 → 0001")
            field_def.update({"start": int(start), "step": int(step), "pad_zeros": int(pad_zeros)})

        # Constant field
        if field_type == "Constant":
            const_val = st.text_input("Constant value (will repeat for every row)", value="", key=f"const_val_{i}")
            st.caption("The exact value you type here will be used for every generated row.")
            field_def.update({"value": const_val})

        # Custom Enum field
        if field_type == "Custom Enum":
            vals = st.text_area("Enum values (comma-separated)", value="A,B,C", help="Enter values separated by commas, e.g. red, green, blue", key=f"enum_vals_{i}")
            mode = st.selectbox("Mode", options=["Random", "Cycle"], index=0, key=f"enum_mode_{i}")
            weights = st.text_input("Optional weights (comma-separated, same length as values) — leave empty for uniform", value="", key=f"enum_weights_{i}")
            st.caption("Random: picks one value per row at random (weights respected). Cycle: round-robin across rows.")
            field_def.update({"values_raw": vals, "enum_mode": mode, "weights_raw": weights})

        # Range options
        if field_type == "Range (0-10)":
            rcol1, rcol2 = st.columns([1, 1])
            with rcol1:
                min_val = st.number_input("Min", value=0, key=f"min_{i}")
            with rcol2:
                max_val = st.number_input("Max", value=10, key=f"max_{i}")
            float_toggle = st.checkbox("Float output?", value=False, key=f"float_{i}")
            precision = st.number_input("Precision (if float)", min_value=0, max_value=6, value=2, key=f"prec_{i}")
            field_def.update({"min": min_val, "max": max_val, "float": float_toggle, "precision": precision})

        # Date Joined note
        if field_type == "Date Joined":
            st.caption("This date field can be used as a timeline source for comment trends.")

        # Comment options with trend controls
        if field_type == "Comment (Sentiment)":
            sentiment = st.selectbox("Sentiment (override)", options=["Random", "Positive", "Neutral", "Negative"], index=0, key=f"sentiment_{i}")
            field_def["sentiment"] = sentiment

            st.markdown("**Trend over time**")
            trend_enabled = st.checkbox("Enable trend for this comment field?", value=False, key=f"trend_enabled_{i}")
            field_def["trend_enabled"] = trend_enabled

            if trend_enabled:
                tl_source = st.selectbox("Timeline source", options=["Global timeline", "Date field"], key=f"tl_src_{i}")
                field_def["timeline_source"] = tl_source
                if tl_source == "Date field":
                    date_field_options = [f["name"] for f in schema if f.get("type") == "Date Joined"]
                    if date_field_options:
                        chosen_df = st.selectbox("Date field to use as timeline", options=date_field_options + ["(enter manually)"], key=f"df_choice_{i}")
                        if chosen_df == "(enter manually)":
                            date_field_ref = st.text_input("Enter date field name", value="", key=f"df_manual_{i}")
                        else:
                            date_field_ref = chosen_df
                    else:
                        date_field_ref = st.text_input("Enter date field name to use for timeline", value="", key=f"df_manual_{i}")
                    field_def["timeline_date_field"] = date_field_ref

                trend_type = st.selectbox("Trend type", options=["Increasing Positive", "Decreasing Positive", "Cyclical", "Random Fluctuation"], key=f"trend_type_{i}")
                trend_strength = st.slider("Trend strength", 0.0, 1.0, 0.5, step=0.01, key=f"trend_strength_{i}")
                field_def["trend_type"] = trend_type
                field_def["trend_strength"] = float(trend_strength)

                base_preset = st.selectbox("Base distribution", options=["Balanced", "Positive-heavy", "Neutral-heavy", "Negative-heavy"], key=f"base_preset_{i}")
                field_def["base_preset"] = base_preset

        # Conditional Range options (kept)
        if field_type == "Conditional Range (Based on Comment Sentiment)":
            comment_field_options = [f["name"] for f in schema if f.get("type") == "Comment (Sentiment)"]
            if comment_field_options:
                chosen = st.selectbox("Depends on", options=comment_field_options + ["(enter manually)"], key=f"cr_dep_choice_{i}")
                if chosen == "(enter manually)":
                    depends_on = st.text_input("Enter comment field name", value="", key=f"cr_dep_manual_{i}")
                else:
                    depends_on = chosen
            else:
                depends_on = st.text_input("Comment field name to depend on", value="", key=f"cr_dep_manual_{i}")

            st.markdown("**Range when comment is Positive**")
            pcol1, pcol2 = st.columns([1, 1])
            with pcol1:
                pmin = st.number_input("Pos min", value=7, key=f"pos_min_{i}")
            with pcol2:
                pmax = st.number_input("Pos max", value=10, key=f"pos_max_{i}")

            st.markdown("**Range when comment is Neutral**")
            ncol1, ncol2 = st.columns([1, 1])
            with ncol1:
                nmin = st.number_input("Neu min", value=4, key=f"neu_min_{i}")
            with ncol2:
                nmax = st.number_input("Neu max", value=6, key=f"neu_max_{i}")

            st.markdown("**Range when comment is Negative**")
            negcol1, negcol2 = st.columns([1, 1])
            with negcol1:
                negmin = st.number_input("Neg min", value=0, key=f"neg_min_{i}")
            with negcol2:
                negmax = st.number_input("Neg max", value=3, key=f"neg_max_{i}")

            st.markdown("**Range when comment sentiment unknown / Any**")
            acol1, acol2 = st.columns([1, 1])
            with acol1:
                amin = st.number_input("Any min", value=0, key=f"any_min_{i}")
            with acol2:
                amax = st.number_input("Any max", value=10, key=f"any_max_{i}")

            float_toggle = st.checkbox("Float output?", value=False, key=f"cr_float_{i}")
            precision = st.number_input("Precision (if float)", min_value=0, max_value=6, value=2, key=f"cr_prec_{i}")

            field_def.update({
                "depends_on": depends_on,
                "positive_min": pmin, "positive_max": pmax,
                "neutral_min": nmin, "neutral_max": nmax,
                "negative_min": negmin, "negative_max": negmax,
                "any_min": amin, "any_max": amax,
                "float": float_toggle, "precision": precision
            })

        # append field to schema
        schema.append(field_def)

# Generate dataset
df = generate_dummy_data(rows, schema, global_timeline=global_timeline)

# Show preview
st.subheader(f"Preview of {rows} rows")
st.dataframe(df, use_container_width=True)

# Download as Excel
towrite = io.BytesIO()
df.to_excel(towrite, index=False, engine="openpyxl")
towrite.seek(0)

st.download_button(
    label="📥 Download Excel File",
    data=towrite,
    file_name="custom_dummy_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet"
)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Faker")
