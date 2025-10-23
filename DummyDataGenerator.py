import streamlit as st
import pandas as pd
from faker import Faker
import random
import io
import uuid
from datetime import datetime, timedelta, time
import math
from typing import List, Dict, Any

fake = Faker()

# --- Field types (generators without per-field params) ---
FIELD_TYPES = {
    "Unique ID (Sequential)": None,  # handled separately
    "Unique ID (UUID)": lambda: str(uuid.uuid4()),
    "Full Name": lambda: fake.name(),
    "First Name": lambda: fake.first_name(),
    "Last Name": lambda: fake.last_name(),
    "Email": lambda: fake.email(),
    "Phone": lambda: fake.phone_number(),
    "Address": lambda: fake.address().replace("\n", ", "),
    "Company": lambda: fake.company(),
    "Age": lambda: random.randint(18, 70),
    "Job Title": lambda: fake.job(),
    "Country": lambda: fake.country(),
    "Date": None,  # we'll generate datetime with time in code to control format
    "Date (Sequential)": None,  # handled separately
    "Custom Text": lambda: fake.word(),
    "Custom Number": lambda: random.randint(1000, 9999),
    # param-handled types
    "Range (0-10)": None,
    "Comment (Sentiment)": None,
    "Conditional Range (Based on Comment Sentiment)": None,
    "Constant": None,
    "Custom Enum": None,
    "Grouped Enum": None,   # grouped enum (multi-output columns)
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

# -----------------
# Helper functions
# -----------------
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
    if raw is None:
        return []
    items = [s.strip() for s in str(raw).split(",")]
    return [s for s in items if s != ""]

def _parse_grouped_values(raw: str) -> List[List[str]]:
    """
    groups separated by ';', fields in each group separated by '|' .
    "Acme|A001; Beta|B002" -> [["Acme","A001"], ["Beta","B002"]]
    """
    if raw is None or str(raw).strip() == "":
        return []
    groups = []
    for chunk in str(raw).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split("|")]
        if any(p != "" for p in parts):
            groups.append(parts)
    return groups

def _parse_weights(raw: str, n):
    if not raw:
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    try:
        weights = [float(p) for p in parts]
    except Exception:
        return None
    if len(weights) != n:
        return None
    weights = [max(0.0, w) for w in weights]
    if sum(weights) == 0:
        return None
    return weights

def _generate_sequential_dates(rows, start_date, end_date, entries_per_date):
    date_list = []
    num_unique_dates = math.ceil(rows / entries_per_date)
    if num_unique_dates <= 1:
        date_list = [start_date] * rows
    else:
        total_days = (end_date - start_date).days
        day_increment = max(1, total_days // (num_unique_dates - 1))
        for i in range(num_unique_dates):
            date = start_date + timedelta(days=min(i * day_increment, total_days))
            for _ in range(min(entries_per_date, rows - len(date_list))):
                date_list.append(date)
                if len(date_list) >= rows:
                    break
            if len(date_list) >= rows:
                break
    return date_list[:rows]

def _safe_lower(x):
    try:
        return str(x).strip().lower()
    except Exception:
        return ""

def _safe_str(x: Any) -> str:
    try:
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x)
    except Exception:
        return ""

def _random_time() -> time:
    return time(random.randrange(0,24), random.randrange(0,60))

# -----------------------------
# CSV/XLSX schema → app schema
# -----------------------------
def map_row_to_field(name: str, field_code: str, values: str) -> Dict[str, Any]:
    """
    VALUES FORCES ENUM:
      - If 'values' non-empty:
          * If '|' present (and Name has '|'), -> Grouped Enum
          * Else -> Custom Enum
      Overrides any suffix for that row.

    Legacy suffix rules used only when values empty:
      *_txt   -> UUID
      *_auto  -> Sequential ID
      yn/_yn  -> Enum(1,2)
      *_date  -> Date
      *_email -> Email
      *_enum/_alt/UNIT -> Custom Enum
      *_cmt   -> Sentiment Comment
      *_scale11 -> Range (0-10)
      default -> Custom Text
    """
    n_raw = _safe_str(name).strip() or "Field"
    f = _safe_lower(field_code)
    v = _safe_str(values).strip()

    if v:
        if ("|" in n_raw) or ("|" in v):
            group_fields = [s.strip() for s in n_raw.split("|") if s.strip() != ""]
            grouped = _parse_grouped_values(v)
            if group_fields and all(len(g) == len(group_fields) for g in grouped):
                return {
                    "type": "Grouped Enum",
                    "group_fields": group_fields,
                    "group_values": grouped,
                    "enum_mode": "Random",
                    "weights_raw": "",
                }
            # Fallback: treat as single-field enum
            enum_vals = _parse_enum_values(v)
            return {"name": n_raw, "type": "Custom Enum",
                    "values_raw": ",".join(enum_vals), "enum_mode": "Random", "weights_raw": ""}

        enum_vals = _parse_enum_values(v)
        return {"name": n_raw, "type": "Custom Enum",
                "values_raw": ",".join(enum_vals), "enum_mode": "Random", "weights_raw": ""}

    if f.endswith("_auto"):
        return {"name": n_raw, "type": "Unique ID (Sequential)", "start": 1, "step": 1, "pad_zeros": 3}
    if f.endswith("_txt"):
        return {"name": n_raw, "type": "Unique ID (UUID)"}
    if f.endswith("_email"):
        return {"name": n_raw, "type": "Email"}
    if f == "yn" or f.endswith("_yn"):
        return {"name": n_raw, "type": "Custom Enum", "values_raw": "1,2", "enum_mode": "Random", "weights_raw": ""}
    if f.endswith("_date"):
        return {"name": n_raw, "type": "Date"}
    if f.endswith("_enum") or f.endswith("_alt") or f == "unit":
        enum_vals = ["cm","m","km","in","ft"] if f == "unit" else ["A","B","C"]
        return {"name": n_raw, "type": "Custom Enum", "values_raw": ",".join(enum_vals), "enum_mode": "Random", "weights_raw": ""}
    if f.endswith("_cmt"):
        return {"name": n_raw, "type": "Comment (Sentiment)", "sentiment": "Random"}
    if f.endswith("_scale11"):
        return {"name": n_raw, "type": "Range (0-10)", "min": 0, "max": 10, "float": False, "precision": 0}
    return {"name": n_raw, "type": "Custom Text"}

def build_schema_from_dataframe(upload_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = {c.strip().lower(): c for c in upload_df.columns}
    required = ["name", "field", "values"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required column: '{r}'")
    name_c = cols["name"]
    field_c = cols["field"]
    values_c = cols["values"]
    schema: List[Dict[str, Any]] = []
    for _, row in upload_df.iterrows():
        name = row.get(name_c, "")
        field_code = row.get(field_c, "")
        values = row.get(values_c, "")
        schema.append(map_row_to_field(name, field_code, values))
    return schema

def read_uploaded_schema(file) -> pd.DataFrame:
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# -----------------------
# Data generation engine
# -----------------------
def generate_dummy_data(rows, schema, global_timeline=None):
    # PASS 1: base rows (includes Constants, Custom Enums, Grouped Enums)
    base_rows = []
    for i in range(rows):
        row = {}
        for field in schema:
            ftype = field["type"]

            # Grouped Enum writes multiple columns
            if ftype == "Grouped Enum":
                group_fields = field.get("group_fields", [])
                grouped = field.get("group_values", [])
                if not group_fields or not grouped:
                    continue
                mode = field.get("enum_mode", "Random")
                weights = _parse_weights(field.get("weights_raw", ""), len(grouped))
                if mode == "Cycle":
                    idx = i % len(grouped)
                else:
                    idx = random.choices(range(len(grouped)), weights=weights, k=1)[0] if weights else random.randrange(len(grouped))
                chosen = grouped[idx]
                for gf, val in zip(group_fields, chosen):
                    row[gf] = val
                continue

            fname = field.get("name", "Field")

            if ftype == "Unique ID (Sequential)":
                continue
            if ftype == "Date (Sequential)":
                continue

            if ftype == "Constant":
                row[fname] = field.get("value", "")
                continue

            if ftype == "Custom Enum":
                vals = _parse_enum_values(field.get("values_raw", ""))
                if not vals:
                    row[fname] = None
                    continue
                mode = field.get("enum_mode", "Random")
                weights = _parse_weights(field.get("weights_raw", ""), len(vals))
                row[fname] = (vals[i % len(vals)] if mode == "Cycle"
                              else (random.choices(vals, weights=weights, k=1)[0] if weights else random.choice(vals)))
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

            if ftype == "Date":
                # Random date within ~5 years + random time so we can format consistently later
                start = datetime.now() - timedelta(days=5*365)
                end = datetime.now()
                rand_date = start + (end - start) * random.random()
                row[fname] = rand_date
                continue

            if ftype in ("Comment (Sentiment)", "Conditional Range (Based on Comment Sentiment)"):
                continue

            gen = FIELD_TYPES.get(ftype)
            row[fname] = gen() if callable(gen) else None
        base_rows.append(row)

    # sequential dates
    for field in schema:
        if field["type"] == "Date (Sequential)":
            fname = field["name"]
            start_date = pd.to_datetime(field.get("seq_start_date"))
            end_date = pd.to_datetime(field.get("seq_end_date"))
            entries_per_date = int(field.get("entries_per_date", 1))
            date_list = _generate_sequential_dates(rows, start_date, end_date, entries_per_date)
            for i, row in enumerate(base_rows):
                d = pd.to_datetime(date_list[i]).date()
                # add a random time component
                row[fname] = datetime.combine(d, _random_time())

    # PASS 2: comments
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
        return min(dates), max(dates)

    for i, row in enumerate(base_rows):
        for field in schema:
            if field["type"] != "Comment (Sentiment)":
                continue
            fname = field["name"]

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
                if timeline_source == "Global timeline":
                    time_factor = (i / (rows - 1)) if rows > 1 else 0.0
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
                            time_factor = (i / (rows - 1)) if rows > 1 else 0.0
                    except Exception:
                        time_factor = (i / (rows - 1)) if rows > 1 else 0.0
                else:
                    time_factor = (i / (rows - 1)) if rows > 1 else 0.0

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
                comment_text = COMMENTS_POSITIVE[random.randrange(len(COMMENTS_POSITIVE))]
            elif r < p_pos + p_neu:
                sentiment = "Neutral"
                comment_text = COMMENTS_NEUTRAL[random.randrange(len(COMMENTS_NEUTRAL))]
            else:
                sentiment = "Negative"
                comment_text = COMMENTS_NEGATIVE[random.randrange(len(COMMENTS_NEGATIVE))]

            row[fname] = comment_text
            sentiments_per_row[i][fname] = sentiment

    # PASS 3: conditional ranges & sequential IDs
    final_rows = []
    for i, row in enumerate(base_rows):
        for field in schema:
            ftype = field["type"]
            if ftype == "Unique ID (Sequential)":
                fname = field["name"]
                start = int(field.get("start", 1))
                step = int(field.get("step", 1))
                pad_zeros = int(field.get("pad_zeros", 3))
                width = len(str(start)) + max(0, pad_zeros)
                val = start + i * step
                row[fname] = str(val).zfill(width)

            elif ftype == "Conditional Range (Based on Comment Sentiment)":
                fname = field["name"]
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

# -----------------
# Streamlit UI
# -----------------
st.set_page_config(page_title="Custom Dummy Data Generator", layout="wide")
st.title("📊 Custom Dummy Data Generator")
st.markdown("Generate dummy data with manual fields **or** from an uploaded CSV/XLSX schema (columns: **Name, field, values**).")

st.sidebar.header("⚙️ Settings")

# Rows input: slider or number field
st.sidebar.subheader("Rows")
rows_input_mode = st.sidebar.radio("Input mode", ["Slider", "Number"], horizontal=True, key="rows_input_mode")
if rows_input_mode == "Slider":
    rows = st.sidebar.slider("Number of rows", min_value=10, max_value=100000, value=100, step=10, key="rows_slider")
else:
    rows = st.sidebar.number_input("Number of rows", min_value=1, max_value=1_000_000, value=100, step=1, key="rows_number")
rows = int(rows)

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

# Upload schema section
st.sidebar.subheader("📄 Upload schema (CSV/XLSX)")
uploaded_file = st.sidebar.file_uploader("Schema file with columns: Name, field, values", type=["csv", "xlsx", "xls"])

type_options = list(FIELD_TYPES.keys())

EMOJI = {
    "Unique ID (Sequential)": "🔢",
    "Unique ID (UUID)": "🆔",
    "Full Name": "👤",
    "First Name": "👤",
    "Last Name": "👤",
    "Email": "✉️",
    "Phone": "📞",
    "Address": "🏠",
    "Company": "🏢",
    "Age": "🎂",
    "Job Title": "💼",
    "Country": "🌍",
    "Date": "📅",
    "Date (Sequential)": "📆",
    "Custom Text": "📝",
    "Custom Number": "🔣",
    "Range (0-10)": "📏",
    "Comment (Sentiment)": "💬",
    "Conditional Range (Based on Comment Sentiment)": "🎯",
    "Constant": "🔒",
    "Custom Enum": "🧩",
    "Grouped Enum": "🧩👥",
}

DEFAULT_FIELD_ORDER = [
    ("First name", "First Name"),
    ("Last name", "Last Name"),
    ("Comment", "Comment (Sentiment)"),
    ("LTR", "Conditional Range (Based on Comment Sentiment)"),
]

# ----------------------------
# Editable uploaded schema — CENTERED
# ----------------------------
def _ensure_session_schema(items: List[Dict[str, Any]]):
    out = []
    for it in items:
        it = dict(it)
        if "_uid" not in it:
            it["_uid"] = str(uuid.uuid4())
        out.append(it)
    return out

def _render_field_editor(item: Dict[str, Any], idx: int, all_comment_names: List[str]):
    """Render one field editor row with delete button; mutate item in-place via session_state widgets."""
    uid = item["_uid"]
    label = item.get('name', '') if item.get('type') != "Grouped Enum" else " | ".join(item.get('group_fields', []))
    with st.expander(f"{EMOJI.get(item.get('type'), '')} Field {idx+1}: {label}", expanded=(idx < 6)):
        top = st.columns([3, 3, 1])
        with top[0]:
            if item.get("type") == "Grouped Enum":
                st.text_input("Name (single field)", disabled=True, value="—", key=f"name_disabled_{uid}")
            else:
                item["name"] = st.text_input("Name", value=item.get("name", f"Field{idx+1}"), key=f"name_{uid}")
        with top[1]:
            current_type = item.get("type", "Custom Text")
            if current_type not in type_options:
                current_type = "Custom Text"
            try:
                type_index = type_options.index(current_type)
            except ValueError:
                type_index = 0
            item["type"] = st.selectbox("Type", options=type_options, index=type_index, key=f"type_{uid}")
        with top[2]:
            if st.button("🗑️", key=f"del_{uid}", help="Delete this field", use_container_width=True):
                return "DELETE"

        ftype = item["type"]

        if ftype == "Unique ID (Sequential)":
            cols = st.columns(3)
            item["start"] = int(cols[0].number_input("Start", value=int(item.get("start", 1)), step=1, key=f"seq_start_{uid}"))
            item["step"] = int(cols[1].number_input("Step", value=int(item.get("step", 1)), step=1, key=f"seq_step_{uid}"))
            item["pad_zeros"] = int(cols[2].number_input("Zeros (additional)", min_value=0, value=int(item.get("pad_zeros", 3)), step=1, key=f"seq_pad_{uid}"))
            st.caption("Width = digits(start) + zeros. Example: start=1, zeros=3 → 0001")

        if ftype == "Date (Sequential)":
            default_start = item.get("seq_start_date") or (datetime.now().date() - timedelta(days=365))
            default_end   = item.get("seq_end_date")   or datetime.now().date()
            item["seq_start_date"] = st.date_input("Start date", value=default_start, key=f"seq_date_start_{uid}")
            item["seq_end_date"] = st.date_input("End date", value=default_end, key=f"seq_date_end_{uid}")
            item["entries_per_date"] = int(st.number_input("Max entries per date", min_value=1, value=int(item.get("entries_per_date", 1)), step=1, key=f"entries_per_date_{uid}"))
            st.caption("Dates will be sequential. Multiple rows can share the same date up to the max specified.")

        if ftype == "Constant":
            item["value"] = st.text_input("Constant value", value=item.get("value", ""), key=f"const_val_{uid}")

        if ftype == "Custom Enum":
            item["values_raw"] = st.text_area("Enum values (comma-separated)", value=item.get("values_raw", "A,B,C"), key=f"enum_vals_{uid}")
            item["enum_mode"] = st.selectbox("Mode", options=["Random", "Cycle"], index=0 if item.get("enum_mode","Random")=="Random" else 1, key=f"enum_mode_{uid}")
            item["weights_raw"] = st.text_input("Optional weights (same length)", value=item.get("weights_raw", ""), key=f"enum_weights_{uid}")
            st.caption("Random = weighted/uniform random pick. Cycle = round-robin per row.")

        if ftype == "Grouped Enum":
            st.markdown("**Grouped Enum configuration**")
            gf = st.text_input(
                "Output field names (use `|` to separate)",
                value="|".join(item.get("group_fields", ["name","id"])),
                key=f"group_fields_{uid}",
                help="Example: client_name|client_id"
            )
            item["group_fields"] = [s.strip() for s in gf.split("|") if s.strip() != ""]
            gv = st.text_area(
                "Group options (one group; fields separated by `|`, groups separated by `;`)",
                value="; ".join(["|".join(g) for g in item.get("group_values", [["Acme","A001"],["Beta","B002"]])]),
                key=f"group_values_{uid}",
                help="Example: Acme|A001; Beta|B002; Gamma|G003"
            )
            item["group_values"] = _parse_grouped_values(gv)
            item["enum_mode"] = st.selectbox("Mode", options=["Random","Cycle"], index=0 if item.get("enum_mode","Random")=="Random" else 1, key=f"group_mode_{uid}")
            item["weights_raw"] = st.text_input("Optional weights for groups (comma-separated)", value=item.get("weights_raw",""), key=f"group_weights_{uid}")
            if item["group_values"] and item["group_fields"] and not all(len(g)==len(item["group_fields"]) for g in item["group_values"]):
                st.warning("Each group's number of values must match the number of output fields.")

        if ftype == "Range (0-10)":
            cols = st.columns(4)
            item["min"] = int(cols[0].number_input("Min", value=int(item.get("min", 0)), key=f"min_{uid}"))
            item["max"] = int(cols[1].number_input("Max", value=int(item.get("max", 10)), key=f"max_{uid}"))
            item["float"] = bool(cols[2].checkbox("Float?", value=bool(item.get("float", False)), key=f"float_{uid}"))
            item["precision"] = int(cols[3].number_input("Precision", min_value=0, max_value=6, value=int(item.get("precision", 2)), key=f"prec_{uid}"))

        if ftype == "Comment (Sentiment)":
            item["sentiment"] = st.selectbox("Sentiment (override)", options=["Random", "Positive", "Neutral", "Negative"], index=["Random","Positive","Neutral","Negative"].index(item.get("sentiment","Random")), key=f"sent_{uid}")
            st.markdown("**Trend over time**")
            item["trend_enabled"] = bool(st.checkbox("Enable trend", value=bool(item.get("trend_enabled", False)), key=f"trend_enabled_{uid}"))
            if item["trend_enabled"]:
                item["timeline_source"] = st.selectbox("Timeline source", options=["Global timeline", "Date field"], index=0 if item.get("timeline_source","Global timeline")=="Global timeline" else 1, key=f"tl_src_{uid}")
                if item["timeline_source"] == "Date field":
                    item["timeline_date_field"] = st.text_input("Date field name", value=item.get("timeline_date_field",""), key=f"df_ref_{uid}")
                item["trend_type"] = st.selectbox("Trend type", options=["Increasing Positive","Decreasing Positive","Cyclical","Random Fluctuation"], index=["Increasing Positive","Decreasing Positive","Cyclical","Random Fluctuation"].index(item.get("trend_type","Increasing Positive")), key=f"trend_type_{uid}")
                item["trend_strength"] = float(st.slider("Trend strength", 0.0, 1.0, float(item.get("trend_strength", 0.5)), step=0.01, key=f"trend_strength_{uid}"))
                item["base_preset"] = st.selectbox("Base distribution", options=["Balanced","Positive-heavy","Neutral-heavy","Negative-heavy"], index=["Balanced","Positive-heavy","Neutral-heavy","Negative-heavy"].index(item.get("base_preset","Balanced")), key=f"base_preset_{uid}")

        if ftype == "Conditional Range (Based on Comment Sentiment)":
            st.markdown("**Depends on (comment field)**")
            if all_comment_names:
                default_name = item.get("depends_on", all_comment_names[0] if all_comment_names else "")
                try:
                    default_idx = all_comment_names.index(default_name)
                except ValueError:
                    default_idx = 0
                sel = st.selectbox("Comment field", options=all_comment_names, index=default_idx, key=f"cr_dep_sel_{uid}")
                item["depends_on"] = sel
            else:
                st.info("No comment fields available yet. You can type a name (will use 'Any' ranges if not found).")
                item["depends_on"] = st.text_input("Comment field name", value=item.get("depends_on",""), key=f"cr_dep_txt_{uid}")

            st.markdown("**Ranges by sentiment**")
            cols1 = st.columns(2)
            item["positive_min"] = int(cols1[0].number_input("Pos min", value=int(item.get("positive_min", 9)), key=f"pmin_{uid}"))
            item["positive_max"] = int(cols1[1].number_input("Pos max", value=int(item.get("positive_max", 10)), key=f"pmax_{uid}"))
            cols2 = st.columns(2)
            item["neutral_min"] = int(cols2[0].number_input("Neu min", value=int(item.get("neutral_min", 7)), key=f"nmin_{uid}"))
            item["neutral_max"] = int(cols2[1].number_input("Neu max", value=int(item.get("neutral_max", 8)), key=f"nmax_{uid}"))
            cols3 = st.columns(2)
            item["negative_min"] = int(cols3[0].number_input("Neg min", value=int(item.get("negative_min", 0)), key=f"negmin_{uid}"))
            item["negative_max"] = int(cols3[1].number_input("Neg max", value=int(item.get("negative_max", 6)), key=f"negmax_{uid}"))
            cols4 = st.columns(3)
            item["any_min"] = int(cols4[0].number_input("Any min", value=int(item.get("any_min", 0)), key=f"amin_{uid}"))
            item["any_max"] = int(cols4[1].number_input("Any max", value=int(item.get("any_max", 10)), key=f"amax_{uid}"))
            item["float"] = bool(cols4[2].checkbox("Float?", value=bool(item.get("float", False)), key=f"cr_float_{uid}"))
            if item["float"]:
                item["precision"] = int(st.number_input("Precision", min_value=0, max_value=6, value=int(item.get("precision", 2)), key=f"cr_prec_{uid}"))

        return "OK"

# Keep an app-level storage of uploaded/parsed fields for editing
if "schema_items" not in st.session_state:
    st.session_state.schema_items: List[Dict[str, Any]] = []
if "schema_loaded_name" not in st.session_state:
    st.session_state.schema_loaded_name = None

schema_from_upload_mode = False
if uploaded_file:
    try:
        upload_df = read_uploaded_schema(uploaded_file)
        parsed_schema = build_schema_from_dataframe(upload_df)
        if st.session_state.schema_loaded_name != uploaded_file.name:
            st.session_state.schema_items = _ensure_session_schema(parsed_schema)
            st.session_state.schema_loaded_name = uploaded_file.name
        schema_from_upload_mode = True
    except Exception as e:
        st.error(f"Failed to read/parse schema: {e}")
        st.stop()

# ===== CENTERED layout for field editors when using uploaded schema =====
if schema_from_upload_mode:
    spacer_left, center_col, spacer_right = st.columns([0.5, 1.6, 0.5])
    with center_col:
        st.subheader("🧩 Fields (from CSV/XLSX) — Centered")

        # 🔎 Search bar
        search_query = st.text_input(
            "Search fields (matches name / type / enum values)",
            value="",
            placeholder="e.g. email, client, uuid, date…"
        ).strip().lower()

        # Build list of all comment-field names (across full list)
        all_comment_names = []
        for it in st.session_state.schema_items:
            if it.get("type") == "Comment (Sentiment)":
                nm = it.get("name", "").strip()
                if nm:
                    all_comment_names.append(nm)

        # Filter for display only
        def _item_text_for_search(it: Dict[str, Any]) -> str:
            ftype = _safe_lower(it.get("type", ""))
            if it.get("type") == "Grouped Enum":
                names_part = " ".join([s.lower() for s in it.get("group_fields", [])])
                values_part = "; ".join(["|".join(g) for g in it.get("group_values", [])]).lower()
                return f"{ftype} {names_part} {values_part}"
        else:
                name_part = _safe_lower(it.get("name", ""))
                values_part = _safe_lower(it.get("values_raw", ""))
                return f"{ftype} {name_part} {values_part}"

        items = st.session_state.schema_items
        display_items = [it for it in items if (search_query in _item_text_for_search(it))] if search_query else items

        c1, c2 = st.columns([3,1])
        with c1:
            st.caption(f"Showing {len(display_items)} of {len(items)} fields")
        with c2:
            if search_query and st.button("Clear search"):
                st.experimental_rerun()

        if len(display_items) == 0:
            st.info("No fields match your search.")
        else:
            to_delete = []
            for i, item in enumerate(display_items):
                result = _render_field_editor(item, i, all_comment_names)
                if result == "DELETE":
                    to_delete.append(item["_uid"])
            if to_delete:
                st.session_state.schema_items = [it for it in st.session_state.schema_items if it["_uid"] not in to_delete]
                st.experimental_rerun()

        cols = st.columns(2)
        if cols[0].button("➕ Add empty field"):
            st.session_state.schema_items.append({
                "_uid": str(uuid.uuid4()),
                "name": f"Field{len(st.session_state.schema_items)+1}",
                "type": "Custom Text"
            })
            st.experimental_rerun()
        if cols[1].button("🧹 Clear uploaded schema"):
            st.session_state.schema_items = []
            st.experimental_rerun()

    # Build schema to use (strip _uid)
    schema: List[Dict[str, Any]] = [{k: v for k, v in it.items() if k != "_uid"} for it in st.session_state.schema_items]

else:
    # Manual builder (still in the sidebar)
    st.sidebar.subheader("🛠️ Add Custom Fields (manual)")
    num_fields = st.sidebar.number_input("Number of fields", 1, 40, 4)
    schema: List[Dict[str, Any]] = []
    for i in range(num_fields):
        with st.sidebar.expander(f"Field {i+1}", expanded=(i < 6)):
            col1, col2 = st.columns([2, 2])
            if i < len(DEFAULT_FIELD_ORDER):
                default_name, default_type = DEFAULT_FIELD_ORDER[i]
            else:
                default_name, default_type = f"Field{i+1}", None

            with col1:
                field_name = st.text_input("Name", value=default_name, key=f"name_{i}")

            type_options_manual = type_options
            if default_type and default_type in type_options_manual:
                default_type_index = type_options_manual.index(default_type)
            else:
                default_type_index = min(i, len(type_options_manual) - 1)

            with col2:
                field_type = st.selectbox("Type", options=type_options_manual, index=default_type_index, key=f"type_{i}")

            st.markdown(f"**{EMOJI.get(field_type, '')} {field_name or default_name} — _{field_type}_**")
            field_def = {"type": field_type}

            if field_type == "Grouped Enum":
                gf = st.text_input("Output field names (use `|`)", value=f"{field_name or default_name}_name|{field_name or default_name}_id", key=f"m_group_fields_{i}")
                gv = st.text_area("Group options (use `;` between groups, `|` within group)", value="Acme|A001; Beta|B002", key=f"m_group_values_{i}")
                mode = st.selectbox("Mode", options=["Random","Cycle"], index=0, key=f"m_group_mode_{i}")
                weights = st.text_input("Optional weights (for groups)", value="", key=f"m_group_weights_{i}")
                field_def.update({
                    "group_fields": [s.strip() for s in gf.split("|") if s.strip()!=""],
                    "group_values": _parse_grouped_values(gv),
                    "enum_mode": mode,
                    "weights_raw": weights,
                })
            else:
                field_def["name"] = field_name or default_name

            if field_type == "Unique ID (Sequential)":
                start = st.number_input("Start", value=1, step=1, key=f"seq_start_{i}")
                step = st.number_input("Step", value=1, step=1, key=f"seq_step_{i}")
                pad_zeros = st.number_input("Number of zeros (additional)", min_value=0, value=3, step=1, key=f"seq_pad_{i}")
                st.caption("Width = digits(start) + Number of zeros. Example: start=1, zeros=3 → 0001")
                field_def.update({"start": int(start), "step": int(step), "pad_zeros": int(pad_zeros)})

            if field_type == "Date (Sequential)":
                seq_start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365), key=f"seq_date_start_{i}")
                seq_end_date = st.date_input("End date", value=datetime.now().date(), key=f"seq_date_end_{i}")
                entries_per_date = st.number_input("Max entries per date", min_value=1, value=1, step=1, key=f"entries_per_date_{i}")
                st.caption("Dates will be sequential. Multiple rows can share the same date up to the max specified.")
                field_def.update({
                    "seq_start_date": seq_start_date,
                    "seq_end_date": seq_end_date,
                    "entries_per_date": int(entries_per_date)
                })

            if field_type == "Constant":
                const_val = st.text_input("Constant value (will repeat for every row)", value="", key=f"const_val_{i}")
                st.caption("The exact value you type here will be used for every generated row.")
                field_def.update({"value": const_val})

            if field_type == "Custom Enum":
                vals = st.text_area("Enum values (comma-separated)", value="A,B,C", help="Enter values separated by commas", key=f"enum_vals_{i}")
                mode = st.selectbox("Mode", options=["Random", "Cycle"], index=0, key=f"enum_mode_{i}")
                weights = st.text_input("Optional weights (same length)", value="", key=f"enum_weights_{i}")
                st.caption("Random: weighted/uniform; Cycle: round-robin.")
                field_def.update({"values_raw": vals, "enum_mode": mode, "weights_raw": weights})

            if field_type == "Range (0-10)":
                rcol1, rcol2 = st.columns([1, 1])
                with rcol1:
                    min_val = st.number_input("Min", value=0, key=f"min_{i}")
                with rcol2:
                    max_val = st.number_input("Max", value=10, key=f"max_{i}")
                float_toggle = st.checkbox("Float output?", value=False, key=f"float_{i}")
                precision = st.number_input("Precision (if float)", min_value=0, max_value=6, value=2, key=f"prec_{i}")
                field_def.update({"min": min_val, "max": max_val, "float": float_toggle, "precision": precision})

            if field_type == "Date":
                st.caption("This generates random datetimes. Use 'Date (Sequential)' for ordered dates.")

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
                        date_field_options = [f["name"] for f in schema if f.get("type") in ["Date", "Date (Sequential)"]]
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

            if field_type == "Conditional Range (Based on Comment Sentiment)":
                comment_field_options = [f["name"] for f in schema if f.get("type") == "Comment (Sentiment)"]
                if comment_field_options:
                    chosen = st.selectbox("Depends on", options=comment_field_options, index=0, key=f"cr_dep_choice_{i}")
                    depends_on = chosen
                else:
                    st.info("No comment fields available yet. You can type a name (will use 'Any' ranges if not found).")
                    depends_on = st.text_input("Comment field name", value="", key=f"cr_dep_manual_{i}")

                st.markdown("**Range when comment is Positive**")
                pcol1, pcol2 = st.columns([1, 1])
                with pcol1:
                    pmin = st.number_input("Pos min", value=9, key=f"pos_min_{i}")
                with pcol2:
                    pmax = st.number_input("Pos max", value=10, key=f"pos_max_{i}")

                st.markdown("**Range when comment is Neutral**")
                ncol1, ncol2 = st.columns([1, 1])
                with ncol1:
                    nmin = st.number_input("Neu min", value=7, key=f"neu_min_{i}")
                with ncol2:
                    nmax = st.number_input("Neu max", value=8, key=f"neu_max_{i}")

                st.markdown("**Range when comment is Negative**")
                negcol1, negcol2 = st.columns([1, 1])
                with negcol1:
                    negmin = st.number_input("Neg min", value=0, key=f"neg_min_{i}")
                with negcol2:
                    negmax = st.number_input("Neg max", value=6, key=f"neg_max_{i}")

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

            schema.append(field_def)

# ==========================
# 📥 Downloadable schema templates
# ==========================
st.subheader("📥 Download schema template")

blank_template = pd.DataFrame(columns=["Name", "field", "values"])
example_template = pd.DataFrame([
    {"Name": "id",                         "field": "user_auto",        "values": ""},                        
    {"Name": "uuid",                       "field": "user_txt",         "values": ""},                        
    {"Name": "contact_email",              "field": "contact_email",    "values": ""},                        
    {"Name": "answer",                     "field": "question_yn",      "values": "1,2"},                     
    {"Name": "visit_date",                 "field": "visit_date",       "values": ""},                        
    {"Name": "rating",                     "field": "csat_scale11",     "values": ""},                        
    {"Name": "mood",                       "field": "mood_enum",        "values": "happy,neutral,sad"},       
    {"Name": "note",                       "field": "feedback_cmt",     "values": ""},                        
    {"Name": "unit",                       "field": "UNIT",             "values": "kg,g,lb"},                 
    {"Name": "client_name|client_id",      "field": "client_group",     "values": "Acme|A001; Beta|B002; Gamma|G003"}, 
])

def _to_csv_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    buf.write(df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))
    buf.seek(0)
    return buf

def _to_xlsx_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf

colA, colB = st.columns(2)
with colA:
    st.markdown("**Blank template**")
    st.download_button(
        "Download CSV (blank)",
        data=_to_csv_bytes(blank_template),
        file_name="schema_template_blank.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        "Download XLSX (blank)",
        data=_to_xlsx_bytes(blank_template),
        file_name="schema_template_blank.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with colB:
    st.markdown("**Example template**")
    st.download_button(
        "Download CSV (example)",
        data=_to_csv_bytes(example_template),
        file_name="schema_template_example.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        "Download XLSX (example)",
        data=_to_xlsx_bytes(example_template),
        file_name="schema_template_example.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.caption("Templates include required headers: **Name, field, values**. For Grouped Enum, put multiple output column names in **Name** separated by `|`, and group options in **values** separated by `;` with fields in each group separated by `|`.")

# ---------------
# Generate & show
# ---------------
schema_to_use = schema
df = generate_dummy_data(rows, schema_to_use, global_timeline=global_timeline)

# ✅ Format date columns as "mm/dd/yyyy, %I:%M %p"
date_cols = [f.get("name") for f in schema_to_use
             if f.get("type") in ("Date", "Date (Sequential)") and f.get("name")]

df_fmt = df.copy()
for col in date_cols:
    if col in df_fmt.columns:
        dtcol = pd.to_datetime(df_fmt[col], errors="coerce")
        df_fmt[col] = dtcol.dt.strftime("%m/%d/%Y, %I:%M %p")

st.subheader(f"Preview of {rows} rows")
st.dataframe(df_fmt, use_container_width=True)

towrite = io.BytesIO()
df_fmt.to_excel(towrite, index=False, engine="openpyxl")
towrite.seek(0)
st.download_button(
    label="📥 Download Excel File",
    data=towrite,
    file_name="custom_dummy_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.caption("Made using Streamlit & Faker")
