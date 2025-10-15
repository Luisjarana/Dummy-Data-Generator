import streamlit as st
import pandas as pd
from faker import Faker
import random
import io
import uuid

fake = Faker()

# --- Field types (generators without per-field params) ---
FIELD_TYPES = {
    "Unique ID (Sequential)": None,  # handled separately
    "Unique ID (UUID)": lambda: str(uuid.uuid4()),
    "Name": lambda: fake.name(),
    "Email": lambda: fake.email(),
    "Phone": lambda: fake.phone_number(),
    "Address": lambda: fake.address().replace("\n", ", "),
    "Company": lambda: fake.company(),
    "Age": lambda: random.randint(18, 70),
    "Salary": lambda: round(random.uniform(30000, 120000), 2),
    "Job Title": lambda: fake.job(),
    "Country": lambda: fake.country(),
    "Date Joined": lambda: fake.date_between(start_date="-5y", end_date="today"),
    "Custom Text": lambda: fake.word(),
    "Custom Number": lambda: random.randint(1000, 9999),
    # param-handled types
    "Range (0-10)": None,
    "Comment (Sentiment)": None,
    "Conditional (Based on Comment Sentiment)": None,
    "Conditional Range (Based on Comment Sentiment)": None,  # NEW
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


def generate_dummy_data(rows, schema):
    data = []
    for i in range(rows):
        row = {}
        sentiments = {}

        # FIRST PASS
        for field in schema:
            fname = field["name"]
            ftype = field["type"]

            if ftype == "Unique ID (Sequential)":
                row[fname] = i + 1
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

            if ftype == "Comment (Sentiment)":
                sentiment_choice = field.get("sentiment", "Random")
                if sentiment_choice == "Random":
                    sentiment_choice = random.choice(["Negative", "Neutral", "Positive"])
                if sentiment_choice == "Positive":
                    comment_text = random.choice(COMMENTS_POSITIVE)
                    sentiment = "Positive"
                elif sentiment_choice == "Neutral":
                    comment_text = random.choice(COMMENTS_NEUTRAL)
                    sentiment = "Neutral"
                else:
                    comment_text = random.choice(COMMENTS_NEGATIVE)
                    sentiment = "Negative"

                row[fname] = comment_text
                sentiments[fname] = sentiment
                continue

            # skip conditional types in first pass
            if ftype in ("Conditional (Based on Comment Sentiment)", "Conditional Range (Based on Comment Sentiment)"):
                continue

            gen = FIELD_TYPES.get(ftype)
            if callable(gen):
                row[fname] = gen()
            else:
                row[fname] = None

        # SECOND PASS: conditionals
        for field in schema:
            fname = field["name"]
            ftype = field["type"]

            if ftype == "Conditional (Based on Comment Sentiment)":
                depends_on = field.get("depends_on", "")
                trigger = field.get("trigger_sentiment", "Negative")
                true_val = field.get("true_value", "")
                false_val = field.get("false_value", "")

                actual_sent = sentiments.get(depends_on)
                if trigger == "Any":
                    condition_true = actual_sent is not None
                else:
                    condition_true = (actual_sent == trigger)

                row[fname] = true_val if condition_true else false_val

            elif ftype == "Conditional Range (Based on Comment Sentiment)":
                depends_on = field.get("depends_on", "")
                pmin, pmax = int(field.get("positive_min", 0)), int(field.get("positive_max", 10))
                nmin, nmax = int(field.get("neutral_min", 0)), int(field.get("neutral_max", 10))
                negmin, negmax = int(field.get("negative_min", 0)), int(field.get("negative_max", 10))
                amin, amax = int(field.get("any_min", 0)), int(field.get("any_max", 10))

                use_float = field.get("float", False)
                precision = int(field.get("precision", 2))

                actual_sent = sentiments.get(depends_on)
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

        data.append(row)

    return pd.DataFrame(data)


# --- UI ---
st.set_page_config(page_title="Custom Dummy Data Generator", layout="wide")
st.title("📊 Custom Dummy Data Generator")
st.markdown("Generate dummy data. Fields are now grouped into collapsible cards for clarity.")

st.sidebar.header("⚙️ Settings")
rows = st.sidebar.slider("Number of rows", 10, 5000, 100, step=10)

st.sidebar.subheader("🛠️ Add Custom Fields")
num_fields = st.sidebar.number_input("Number of fields", 1, 30, 4)

schema = []
type_options = list(FIELD_TYPES.keys())

# small emoji map to help visually identify type inside the expander
EMOJI = {
    "Unique ID (Sequential)": "🔢",
    "Unique ID (UUID)": "🆔",
    "Name": "👤",
    "Email": "✉️",
    "Phone": "📞",
    "Address": "🏠",
    "Company": "🏢",
    "Age": "🎂",
    "Salary": "💰",
    "Job Title": "💼",
    "Country": "🌍",
    "Date Joined": "📅",
    "Custom Text": "📝",
    "Custom Number": "🔣",
    "Range (0-10)": "📏",
    "Comment (Sentiment)": "💬",
    "Conditional (Based on Comment Sentiment)": "🔀",
    "Conditional Range (Based on Comment Sentiment)": "🎯",
}

for i in range(num_fields):
    # each field gets its own expander (collapsible card)
    with st.sidebar.expander(f"Field {i+1}", expanded=(i < 3)):
        col1, col2 = st.columns([2, 2])
        default_name = f"Field{i+1}"
        with col1:
            field_name = st.text_input(f"Name", value=default_name, key=f"name_{i}")
        with col2:
            field_type = st.selectbox(
                "Type",
                options=type_options,
                index=min(i, len(type_options) - 1),
                key=f"type_{i}"
            )

        # show a small header for quick recognition
        st.markdown(f"**{EMOJI.get(field_type, '')} {field_name or default_name} — _{field_type}_**")

        field_def = {"name": field_name or default_name, "type": field_type, "default_name": default_name}

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

        # Comment options
        if field_type == "Comment (Sentiment)":
            sentiment = st.selectbox("Sentiment", options=["Random", "Positive", "Neutral", "Negative"], index=0, key=f"sentiment_{i}")
            field_def["sentiment"] = sentiment

        # Conditional string field options
        if field_type == "Conditional (Based on Comment Sentiment)":
            comment_field_options = [f["name"] for f in schema if f.get("type") == "Comment (Sentiment)"]
            if comment_field_options:
                chosen = st.selectbox("Depends on", options=comment_field_options + ["(enter manually)"], key=f"depends_choice_{i}")
                if chosen == "(enter manually)":
                    depends_on = st.text_input("Comment field name", value="", key=f"depends_manual_{i}")
                else:
                    depends_on = chosen
            else:
                depends_on = st.text_input("Comment field name to depend on", value="", key=f"depends_manual_{i}")

            trigger = st.selectbox("Trigger sentiment", options=["Positive", "Neutral", "Negative", "Any"], index=3, key=f"trigger_{i}")
            true_val = st.text_input(f"Value if {trigger}", value="TRUE_VALUE", key=f"true_{i}")
            false_val = st.text_input(f"Value if NOT {trigger}", value="FALSE_VALUE", key=f"false_{i}")

            field_def.update({
                "depends_on": depends_on,
                "trigger_sentiment": trigger,
                "true_value": true_val,
                "false_value": false_val
            })

        # Conditional Range options
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

        # append the configured field to schema after the expander content
        schema.append(field_def)

# Generate and display
df = generate_dummy_data(rows, schema)

st.subheader(f"Preview of {rows} rows")
st.dataframe(df, use_container_width=True)

# Download
towrite = io.BytesIO()
df.to_excel(towrite, index=False, engine="openpyxl")
towrite.seek(0)

st.download_button(
    label="📥 Download Excel File",
    data=towrite,
    file_name="custom_dummy_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit & Faker")
