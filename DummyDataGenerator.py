import streamlit as st
import pandas as pd
import random
import io
import uuid
import math
from typing import List, Dict, Any
from datetime import datetime, timedelta, time
from faker import Faker

# =========================
# 🔤 Simple i18n dictionary
# =========================
I18N = {
    "en": {
        "app_title": "📊 Custom Dummy Data Generator",
        "app_desc": "Generate dummy data with manual fields **or** from an uploaded CSV/XLSX schema (columns: **Name, field, values**).",
        "settings": "⚙️ Settings",
        "language": "Language",
        "rows": "Rows",
        "rows_input_mode": "Input mode",
        "slider": "Slider",
        "number": "Number",
        "global_tl": "📈 Global timeline (optional)",
        "use_global_tl": "Use global timeline mapping (index → dates)",
        "global_start": "Global start date",
        "global_end": "Global end date",
        "upload_schema": "📄 Upload schema (CSV/XLSX)",
        "upload_label": "Schema file with columns: Name, field, values",
        "csv_fields": "🧩 Fields (from CSV/XLSX)",
        "search": "Search (name / type / enum values)",
        "search_ph": "e.g. email, client, uuid, date…",
        "clear": "Clear",
        "showing": "Showing",
        "of": "of",
        "fields": "fields",
        "no_matches": "No fields match your search.",
        "add_field": "➕ Add field",
        "clear_schema": "🧹 Clear schema",
        "name": "Name",
        "type": "Type",
        "delete": "Delete this field",
        "seq_id_caption": "Width = digits(start) + zeros. Example: start=1, zeros=3 → 0001",
        "seq_dates_caption": "Dates will be sequential. Multiple rows can share the same date up to the max specified.",
        "const_value": "Constant value",
        "enum_values": "Enum values (comma-separated)",
        "enum_mode": "Mode",
        "enum_weights": "Optional weights (same length)",
        "enum_note": "Random = weighted/uniform random pick. Cycle = round-robin per row.",
        "grouped_title": "**Grouped Enum configuration**",
        "grouped_fields": "Output field names (use `|` to separate)",
        "grouped_fields_help": "Example: client_name|client_id",
        "grouped_values": "Group options (one group; fields separated by `|`, groups separated by `;`)",
        "grouped_values_help": "Example: Acme|A001; Beta|B002; Gamma|G003",
        "grouped_weights": "Optional weights for groups (comma-separated)",
        "grouped_warn": "Each group's number of values must match the number of output fields.",
        "min": "Min",
        "max": "Max",
        "float": "Float?",
        "precision": "Precision",
        "date_note": "This generates random datetimes. Use 'Date (Sequential)' for ordered dates.",
        "sentiment": "Sentiment (override)",
        "trend_title": "**Trend over time**",
        "trend_enable": "Enable trend",
        "timeline_source": "Timeline source",
        "date_field_name": "Date field name",
        "trend_strength": "Trend strength",
        "base_dist": "Base distribution",
        "depends_on": "**Depends on (comment field)**",
        "no_comments": "No comment fields available yet. Type a name (will use 'Any' ranges if not found).",
        "ranges_title": "**Ranges by sentiment**",
        "pos_min": "Pos min",
        "pos_max": "Pos max",
        "neu_min": "Neu min",
        "neu_max": "Neu max",
        "neg_min": "Neg min",
        "neg_max": "Neg max",
        "any_min": "Any min",
        "any_max": "Any max",
        "download_templates": "📥 Download schema template",
        "blank_template": "**Blank template**",
        "dl_csv_blank": "Download CSV (blank)",
        "dl_xlsx_blank": "Download XLSX (blank)",
        "example_template": "**Example template**",
        "dl_csv_example": "Download CSV (example)",
        "dl_xlsx_example": "Download XLSX (example)",
        "templates_caption": "Templates include required headers: **Name, field, values**. For Grouped Enum, put multiple output column names in **Name** separated by `|`, and group options in **values** separated by `;` with fields in each group separated by `|`.",
        "preview": "Preview of {rows} rows",
        "download_excel": "📥 Download Excel File",
        "footer": "Made using Streamlit & Faker",
        "errors_missing_col": "Missing required column",
        "failed_parse": "Failed to read/parse schema",
        "field_label": "Field",
        "name_single_disabled": "Name (single field)",
        "mode_random": "Random",
        "mode_cycle": "Cycle",
        "tl_global": "Global timeline",
        "tl_datefield": "Date field",
        "trend_inc": "Increasing Positive",
        "trend_dec": "Decreasing Positive",
        "trend_cyc": "Cyclical",
        "trend_jit": "Random Fluctuation",
        "base_bal": "Balanced",
        "base_pos": "Positive-heavy",
        "base_neu": "Neutral-heavy",
        "base_neg": "Negative-heavy",
    },
    "es": {
        "app_title": "📊 Generador de Datos Ficticios",
        "app_desc": "Genera datos con campos manuales **o** desde un esquema CSV/XLSX (columnas: **Name, field, values**).",
        "settings": "⚙️ Ajustes",
        "language": "Idioma",
        "rows": "Filas",
        "rows_input_mode": "Modo de entrada",
        "slider": "Deslizador",
        "number": "Número",
        "global_tl": "📈 Línea de tiempo global (opcional)",
        "use_global_tl": "Usar línea de tiempo global (índice → fechas)",
        "global_start": "Fecha inicial global",
        "global_end": "Fecha final global",
        "upload_schema": "📄 Subir esquema (CSV/XLSX)",
        "upload_label": "Archivo con columnas: Name, field, values",
        "csv_fields": "🧩 Campos (de CSV/XLSX)",
        "search": "Buscar (nombre / tipo / valores enum)",
        "search_ph": "ej. email, cliente, uuid, fecha…",
        "clear": "Limpiar",
        "showing": "Mostrando",
        "of": "de",
        "fields": "campos",
        "no_matches": "No hay coincidencias.",
        "add_field": "➕ Agregar campo",
        "clear_schema": "🧹 Limpiar esquema",
        "name": "Nombre",
        "type": "Tipo",
        "delete": "Eliminar este campo",
        "seq_id_caption": "Ancho = dígitos(inicio) + ceros. Ej: inicio=1, ceros=3 → 0001",
        "seq_dates_caption": "Las fechas serán secuenciales. Varias filas pueden compartir la misma fecha.",
        "const_value": "Valor constante",
        "enum_values": "Valores enum (separados por comas)",
        "enum_mode": "Modo",
        "enum_weights": "Pesos opcionales (misma longitud)",
        "enum_note": "Aleatorio = con/sin pesos. Ciclo = rotación fila a fila.",
        "grouped_title": "**Enum agrupado**",
        "grouped_fields": "Nombres de salida (separa con `|`)",
        "grouped_fields_help": "Ejemplo: cliente_nombre|cliente_id",
        "grouped_values": "Opciones (campos con `|`, grupos con `;`)",
        "grouped_values_help": "Ejemplo: Acme|A001; Beta|B002; Gamma|G003",
        "grouped_weights": "Pesos opcionales para grupos (comas)",
        "grouped_warn": "Cada grupo debe tener tantos valores como campos de salida.",
        "min": "Mín",
        "max": "Máx",
        "float": "Decimal?",
        "precision": "Precisión",
        "date_note": "Genera fechas con hora. Usa 'Date (Sequential)' para fechas ordenadas.",
        "sentiment": "Sentimiento (forzar)",
        "trend_title": "**Tendencia en el tiempo**",
        "trend_enable": "Habilitar tendencia",
        "timeline_source": "Fuente de tiempo",
        "date_field_name": "Nombre del campo de fecha",
        "trend_strength": "Fuerza de tendencia",
        "base_dist": "Distribución base",
        "depends_on": "**Depende de (campo de comentario)**",
        "no_comments": "Aún no hay comentarios. Escribe un nombre (usará rangos 'Any' si no existe).",
        "ranges_title": "**Rangos por sentimiento**",
        "pos_min": "Pos mín",
        "pos_max": "Pos máx",
        "neu_min": "Neu mín",
        "neu_max": "Neu máx",
        "neg_min": "Neg mín",
        "neg_max": "Neg máx",
        "any_min": "Cualq mín",
        "any_max": "Cualq máx",
        "download_templates": "📥 Descargar plantilla de esquema",
        "blank_template": "**Plantilla en blanco**",
        "dl_csv_blank": "Descargar CSV (blanco)",
        "dl_xlsx_blank": "Descargar XLSX (blanco)",
        "example_template": "**Plantilla de ejemplo**",
        "dl_csv_example": "Descargar CSV (ejemplo)",
        "dl_xlsx_example": "Descargar XLSX (ejemplo)",
        "templates_caption": "Incluye cabeceras: **Name, field, values**. Para Enum agrupado, usa varios nombres en **Name** separados por `|`, y en **values** separa grupos con `;` y campos con `|`.",
        "preview": "Vista previa de {rows} filas",
        "download_excel": "📥 Descargar Excel",
        "footer": "Hecho con Streamlit & Faker",
        "errors_missing_col": "Falta la columna requerida",
        "failed_parse": "No se pudo leer/parsear el esquema",
        "field_label": "Campo",
        "name_single_disabled": "Nombre (campo simple)",
        "mode_random": "Aleatorio",
        "mode_cycle": "Ciclo",
        "tl_global": "Línea global",
        "tl_datefield": "Campo de fecha",
        "trend_inc": "Aumenta Positivo",
        "trend_dec": "Disminuye Positivo",
        "trend_cyc": "Cíclica",
        "trend_jit": "Fluctuación aleatoria",
        "base_bal": "Balanceada",
        "base_pos": "Positiva",
        "base_neu": "Neutral",
        "base_neg": "Negativa",
    }
}

def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))

# =========================
# Stable field type codes
# =========================
FT_CODES = [
    "seq", "uuid", "full", "first", "last", "email", "phone", "address", "company",
    "age", "job", "country", "date", "date_seq", "ctext", "cnum",
    "range", "comment", "cond_range", "const", "cenum", "genum"
]

def ft_label(lang: str, code: str) -> str:
    mapping = {
        "seq": "Unique ID (Sequential)",
        "uuid": "Unique ID (UUID)",
        "full": "Full Name",
        "first": "First Name",
        "last": "Last Name",
        "email": "Email",
        "phone": "Phone",
        "address": "Address",
        "company": "Company",
        "age": "Age",
        "job": "Job Title",
        "country": "Country",
        "date": "Date",
        "date_seq": "Date (Sequential)",
        "ctext": "Custom Text",
        "cnum": "Custom Number",
        "range": "Range (0-10)",
        "comment": "Comment (Sentiment)",
        "cond_range": "Conditional Range (Based on Comment Sentiment)",
        "const": "Constant",
        "cenum": "Custom Enum",
        "genum": "Grouped Enum",
    }
    # (Optional) translate labels if you later add localized labels.
    return mapping.get(code, code)

EMOJI = {
    "seq": "🔢", "uuid": "🆔", "full": "👤", "first": "👤", "last": "👤",
    "email": "✉️", "phone": "📞", "address": "🏠", "company": "🏢",
    "age": "🎂", "job": "💼", "country": "🌍", "date": "📅", "date_seq": "📆",
    "ctext": "📝", "cnum": "🔣", "range": "📏", "comment": "💬", "cond_range": "🎯",
    "const": "🔒", "cenum": "🧩", "genum": "🧩👥",
}

# Reverse label → code (to migrate legacy items that stored 'type' as a label)
LABEL_TO_CODE = {ft_label("en", c): c for c in FT_CODES}
# Also accept raw codes as labels just in case
LABEL_TO_CODE.update({c: c for c in FT_CODES})

# ==============
# Streamlit page
# ==============
st.set_page_config(page_title="Custom Dummy Data Generator", layout="wide")

# Language
if "lang" not in st.session_state:
    st.session_state.lang = "en"
st.sidebar.selectbox("Language / Idioma", options=["en", "es"],
                     index=0 if st.session_state.lang=="en" else 1, key="lang")
lang = st.session_state.lang

st.title(t(lang, "app_title"))
st.markdown(t(lang, "app_desc"))

# Faker per language
FAKER_LOCALE = "en_US" if lang == "en" else "es_MX"
fake = Faker(FAKER_LOCALE)

# Generators by type code
FIELD_GEN = {
    "seq": None,
    "uuid": lambda: str(uuid.uuid4()),
    "full": lambda: fake.name(),
    "first": lambda: fake.first_name(),
    "last": lambda: fake.last_name(),
    "email": lambda: fake.email(),
    "phone": lambda: fake.phone_number(),
    "address": lambda: fake.address().replace("\n", ", "),
    "company": lambda: fake.company(),
    "age": lambda: random.randint(18, 70),
    "job": lambda: fake.job(),
    "country": lambda: fake.country(),
    "date": lambda: fake.date_time_between(start_date="-5y", end_date="now"),  # datetime
    "date_seq": None,
    "ctext": lambda: fake.word(),
    "cnum": lambda: random.randint(1000, 9999),
    "range": None,
    "comment": None,
    "cond_range": None,
    "const": None,
    "cenum": None,
    "genum": None,
}

# =========================
# Comment pools (localized)
# =========================
if lang == "en":
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
else:
    COMMENTS_POSITIVE = [
        "Excelente experiencia, muy recomendable.",
        "¡Muy satisfecho con el resultado!",
        "Superó las expectativas — muy contento.",
        "Servicio fantástico y personal amable.",
        "Me encantó, 10/10.",
    ]
    COMMENTS_NEUTRAL = [
        "Estuvo bien, nada especial.",
        "Experiencia promedio en general.",
        "Cumplió con lo básico.",
        "Ni bueno ni malo — aceptable.",
        "Satisfactorio pero mejorable.",
    ]
    COMMENTS_NEGATIVE = [
        "Muy decepcionado con el servicio.",
        "No era lo esperado, mala experiencia.",
        "No lo recomendaría; necesita mejorar.",
        "Mala experiencia — no volveré.",
        "Insatisfecho con el resultado.",
    ]

# =========================
# Helpers
# =========================
def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def _normalize_probs(p):
    s = sum(p)
    if s == 0:
        return [1/3, 1/3, 1/3]
    return [x / s for x in p]

def _apply_trend(base_probs, time_factor, trend_type, strength):
    pos, neu, neg = base_probs
    if trend_type == t(lang, "trend_inc"):
        pos = pos + strength * time_factor * (1 - pos)
        neg = neg - strength * time_factor * neg
        neu = 1 - pos - neg
    elif trend_type == t(lang, "trend_dec"):
        pos = pos - strength * time_factor * pos
        neg = neg + strength * time_factor * (1 - neg)
        neu = 1 - pos - neg
    elif trend_type == t(lang, "trend_cyc"):
        import math as _m
        cyc = _m.sin(2 * _m.pi * time_factor)
        pos = pos + strength * (cyc * 0.5)
        pos = _clamp(pos, 0.01, 0.99)
        remain = 1 - pos
        pair_sum = neu + neg
        if pair_sum == 0:
            neu = neg = remain / 2
        else:
            neu = remain * (neu / pair_sum)
            neg = remain * (neg / pair_sum)
    elif trend_type == t(lang, "trend_jit"):
        pos = pos + random.gauss(0, 0.1 * strength)
        neu = neu + random.gauss(0, 0.1 * strength)
        neg = neg + random.gauss(0, 0.1 * strength)
    pos = _clamp(pos, 0.0, 1.0)
    neu = _clamp(neu, 0.0, 1.0)
    neg = _clamp(neg, 0.0, 1.0)
    return _normalize_probs((pos, neu, neg))

def _parse_enum_values(raw: str):
    if raw is None:
        return []
    items = [s.strip() for s in str(raw).split(",")]
    return [s for s in items if s != ""]

def _parse_grouped_values(raw: str) -> List[List[str]]:
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

def _random_time():
    return time(random.randrange(0,24), random.randrange(0,60))

def _clear_schema_search():
    st.session_state["schema_search"] = ""

# ---------- NEW: migration helpers (fixes your dropdown mismatch) ----------
def _ensure_uid(item: Dict[str, Any]) -> None:
    if "_uid" not in item or not item["_uid"]:
        item["_uid"] = str(uuid.uuid4())

def _migrate_type_to_ft(item: Dict[str, Any]) -> None:
    """Normalize an item to have 'ft' code; accept legacy 'type' labels/codes."""
    if "ft" in item and item["ft"] in FT_CODES:
        return
    # Legacy path: 'type' exists (label or code)
    legacy_type = item.get("type")
    if isinstance(legacy_type, str):
        code = LABEL_TO_CODE.get(legacy_type.strip(), None)
        if code is None:
            # try a few known alternates
            alt = legacy_type.strip().lower()
            # very rough fallbacks
            if alt in ["unique id (sequential)", "sequential id", "id seq", "unique id"]:
                code = "seq"
            elif alt in ["unique id (uuid)", "uuid"]:
                code = "uuid"
            elif "comment" in alt:
                code = "comment"
            elif "conditional" in alt and "range" in alt:
                code = "cond_range"
            elif "date" in alt and "sequential" in alt:
                code = "date_seq"
            elif "date" in alt:
                code = "date"
            elif "enum" in alt and "group" in alt:
                code = "genum"
            elif "enum" in alt:
                code = "cenum"
            elif "range" in alt:
                code = "range"
            elif "constant" in alt:
                code = "const"
            elif "phone" in alt:
                code = "phone"
            elif "email" in alt:
                code = "email"
            elif "first" in alt:
                code = "first"
            elif "last" in alt:
                code = "last"
            elif "full" in alt and "name" in alt:
                code = "full"
            elif "company" in alt:
                code = "company"
            elif "job" in alt:
                code = "job"
            elif "country" in alt:
                code = "country"
            elif "number" in alt:
                code = "cnum"
            elif "text" in alt:
                code = "ctext"
        if code in FT_CODES:
            item["ft"] = code
            item.pop("type", None)
            return
    # If still nothing, default to ctext
    item["ft"] = "ctext"
    item.pop("type", None)

def _normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        it = dict(it)
        _ensure_uid(it)
        _migrate_type_to_ft(it)
        out.append(it)
    return out
# ---------------------------------------------------------------------------

# =========================
# CSV → app schema mapping
# =========================
def map_row_to_field(name: str, field_code: str, values: str) -> Dict[str, Any]:
    """
    If 'values' is non-empty -> enum (Grouped if '|' present in Name or values).
    Otherwise use suffix rules:
      *_txt -> UUID
      *_auto -> Sequential ID
      yn/_yn -> Enum(1,2)
      *_date -> Date
      *_email -> Email
      *_enum / *_alt / UNIT -> Custom Enum
      *_cmt -> Comment
      *_scale11 -> Range (0-10)
      default -> Custom Text
    """
    n_raw = _safe_str(name).strip() or "Field"
    f = _safe_lower(field_code)
    v = _safe_str(values).strip()

    # Force enum if values present
    if v:
        if ("|" in n_raw) or ("|" in v):
            group_fields = [s.strip() for s in n_raw.split("|") if s.strip() != ""]
            grouped = _parse_grouped_values(v)
            if group_fields and all(len(g) == len(group_fields) for g in grouped):
                return {
                    "ft": "genum",
                    "group_fields": group_fields,
                    "group_values": grouped,
                    "enum_mode": "Random",
                    "weights_raw": "",
                    "_uid": str(uuid.uuid4()),
                }
        enum_vals = _parse_enum_values(v)
        return {"name": n_raw, "ft": "cenum", "values_raw": ",".join(enum_vals),
                "enum_mode": "Random", "weights_raw": "", "_uid": str(uuid.uuid4())}

    # Suffix rules when no values provided
    if f.endswith("_auto"):
        return {"name": n_raw, "ft": "seq", "start": 1, "step": 1, "pad_zeros": 3, "_uid": str(uuid.uuid4())}
    if f.endswith("_txt"):
        return {"name": n_raw, "ft": "uuid", "_uid": str(uuid.uuid4())}
    if f.endswith("_email"):
        return {"name": n_raw, "ft": "email", "_uid": str(uuid.uuid4())}
    if f == "yn" or f.endswith("_yn"):
        return {"name": n_raw, "ft": "cenum", "values_raw": "1,2", "enum_mode": "Random", "weights_raw": "", "_uid": str(uuid.uuid4())}
    if f.endswith("_date"):
        return {"name": n_raw, "ft": "date", "_uid": str(uuid.uuid4())}
    if f.endswith("_enum") or f.endswith("_alt") or f == "unit":
        enum_vals = ["cm","m","km","in","ft"] if f == "unit" else ["A","B","C"]
        return {"name": n_raw, "ft": "cenum", "values_raw": ",".join(enum_vals), "enum_mode": "Random", "weights_raw": "", "_uid": str(uuid.uuid4())}
    if f.endswith("_cmt"):
        return {"name": n_raw, "ft": "comment", "sentiment": "Random", "_uid": str(uuid.uuid4())}
    if f.endswith("_scale11"):
        return {"name": n_raw, "ft": "range", "min": 0, "max": 10, "float": False, "precision": 0, "_uid": str(uuid.uuid4())}
    return {"name": n_raw, "ft": "ctext", "_uid": str(uuid.uuid4())}

def build_schema_from_dataframe(upload_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = {c.strip().lower(): c for c in upload_df.columns}
    for r in ["name", "field", "values"]:
        if r not in cols:
            raise ValueError(f"{t(lang,'errors_missing_col')}: '{r}'")
    name_c, field_c, values_c = cols["name"], cols["field"], cols["values"]
    schema = []
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

# =========================
# Data generation engine
# =========================
def generate_dummy_data(rows, schema, global_timeline=None):
    base_rows = []
    for i in range(rows):
        row = {}
        for field in schema:
            ft = field["ft"]

            # Grouped Enum writes multiple columns
            if ft == "genum":
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
            if ft == "seq":
                continue
            if ft == "date_seq":
                continue
            if ft == "const":
                row[fname] = field.get("value", "")
                continue
            if ft == "cenum":
                vals = _parse_enum_values(field.get("values_raw", ""))
                if not vals:
                    row[fname] = None
                    continue
                mode = field.get("enum_mode", "Random")
                weights = _parse_weights(field.get("weights_raw", ""), len(vals))
                row[fname] = (vals[i % len(vals)] if mode == "Cycle"
                              else (random.choices(vals, weights=weights, k=1)[0] if weights else random.choice(vals)))
                continue
            if ft == "range":
                min_v = int(field.get("min", 0))
                max_v = int(field.get("max", 10))
                if min_v > max_v:
                    min_v, max_v = max_v, min_v
                if field.get("float", False):
                    row[fname] = round(random.uniform(min_v, max_v), field.get("precision", 2))
                else:
                    row[fname] = random.randint(min_v, max_v)
                continue
            if ft == "date":
                gen = FIELD_GEN.get(ft)
                row[fname] = gen() if callable(gen) else None
                continue
            if ft in ("comment", "cond_range"):
                continue

            gen = FIELD_GEN.get(ft)
            row[fname] = gen() if callable(gen) else None
        base_rows.append(row)

    # Sequential dates with random time
    for field in schema:
        if field["ft"] == "date_seq":
            fname = field["name"]
            start_date = pd.to_datetime(field.get("seq_start_date"))
            end_date = pd.to_datetime(field.get("seq_end_date"))
            entries_per_date = int(field.get("entries_per_date", 1))
            date_list = _generate_sequential_dates(rows, start_date, end_date, entries_per_date)
            for i, row in enumerate(base_rows):
                d = pd.to_datetime(date_list[i]).date()
                row[fname] = datetime.combine(d, _random_time())

    # Comments (trend-aware)
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
            if field["ft"] != "comment":
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
                comment_text = random.choice(COMMENTS_POSITIVE)
            elif r < p_pos + p_neu:
                sentiment = "Neutral"
                comment_text = random.choice(COMMENTS_NEUTRAL)
            else:
                sentiment = "Negative"
                comment_text = random.choice(COMMENTS_NEGATIVE)

            row[fname] = comment_text
            sentiments_per_row[i][fname] = sentiment

    # Conditional ranges and sequential IDs
    final_rows = []
    for i, row in enumerate(base_rows):
        for field in schema:
            ft = field["ft"]
            if ft == "seq":
                fname = field["name"]
                start = int(field.get("start", 1))
                step = int(field.get("step", 1))
                pad_zeros = int(field.get("pad_zeros", 3))
                width = len(str(start)) + max(0, pad_zeros)
                val = start + i * step
                row[fname] = str(val).zfill(width)

            elif ft == "cond_range":
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

# =========================
# Sidebar controls
# =========================
st.sidebar.header(t(lang, "settings"))

st.sidebar.subheader(t(lang, "rows"))
rows_input_mode = st.sidebar.radio(t(lang, "rows_input_mode"), [t(lang, "slider"), t(lang, "number")], horizontal=True, key="rows_input_mode")
if rows_input_mode == t(lang, "slider"):
    rows = st.sidebar.slider(t(lang, "rows"), min_value=10, max_value=100000, value=100, step=10, key="rows_slider")
else:
    rows = st.sidebar.number_input(t(lang, "rows"), min_value=1, max_value=1_000_000, value=100, step=1, key="rows_number")
rows = int(rows)

st.sidebar.subheader(t(lang, "global_tl"))
use_global_timeline = st.sidebar.checkbox(t(lang, "use_global_tl"), value=False, key="use_global_tl")
global_timeline = None
if use_global_timeline:
    gstart = st.sidebar.date_input(t(lang, "global_start"), value=datetime.now().date() - timedelta(days=365))
    gend = st.sidebar.date_input(t(lang, "global_end"), value=datetime.now().date())
    if gstart >= gend:
        st.sidebar.error("Start must be before end." if lang=="en" else "La fecha inicial debe ser anterior a la final.")
    else:
        global_timeline = {"start_date": pd.to_datetime(gstart), "end_date": pd.to_datetime(gend)}

st.sidebar.subheader(t(lang, "upload_schema"))
uploaded_file = st.sidebar.file_uploader(t(lang, "upload_label"), type=["csv", "xlsx", "xls"])

# =========================
# Sidebar: CSV editor + search
# =========================
if "schema_items" not in st.session_state:
    st.session_state.schema_items = []
if "schema_loaded_name" not in st.session_state:
    st.session_state.schema_loaded_name = None

schema_from_upload_mode = False
if uploaded_file:
    try:
        upload_df = read_uploaded_schema(uploaded_file)
        parsed_schema = build_schema_from_dataframe(upload_df)
        # normalize new items (add _uid, convert 'type'→'ft' if any)
        parsed_schema = _normalize_items(parsed_schema)
        if st.session_state.schema_loaded_name != uploaded_file.name:
            st.session_state.schema_items = parsed_schema
            st.session_state.schema_loaded_name = uploaded_file.name
        schema_from_upload_mode = True
    except Exception as e:
        st.error(f"{t(lang,'failed_parse')}: {e}")
        st.stop()

# ALWAYS normalize whatever is in session (handles legacy items)
st.session_state.schema_items = _normalize_items(st.session_state.schema_items)

def _item_search_text(it: Dict[str, Any]) -> str:
    ft = _safe_lower(it.get("ft", ""))
    if it.get("ft") == "genum":
        names_part = " ".join([s.lower() for s in it.get("group_fields", [])])
        values_part = "; ".join(["|".join(g) for g in it.get("group_values", [])]).lower()
        return f"{ft} {names_part} {values_part}"
    else:
        name_part = _safe_lower(it.get("name", ""))
        values_part = _safe_lower(it.get("values_raw", ""))
        return f"{ft} {name_part} {values_part}"

def _render_field_editor(item: Dict[str, Any], idx: int, all_comment_names: List[str], ui):
    uid = item["_uid"]
    label_name = item.get('name', '') if item.get('ft') != "genum" else " | ".join(item.get('group_fields', []))
    with ui.expander(f"{EMOJI.get(item.get('ft'),'')} {t(lang,'field_label')} {idx+1}: {label_name}", expanded=(idx < 6)):
        top = ui.columns([3, 3, 1])
        with top[0]:
            if item.get("ft") == "genum":
                ui.text_input(t(lang,"name_single_disabled"), disabled=True, value="—", key=f"name_disabled_{uid}")
            else:
                item["name"] = ui.text_input(t(lang,"name"), value=item.get("name", f"{t(lang,'field_label')} {idx+1}"), key=f"name_{uid}")
        with top[1]:
            # Select by CODE (stable), show label
            current_code = item.get("ft", "ctext")
            item["ft"] = ui.selectbox(
                t(lang, "type"),
                options=FT_CODES,
                index=FT_CODES.index(current_code) if current_code in FT_CODES else FT_CODES.index("ctext"),
                format_func=lambda c: ft_label(lang, c),
                key=f"type_{uid}"
            )
        with top[2]:
            if ui.button("🗑️", key=f"del_{uid}", help=t(lang,"delete"), use_container_width=True):
                return "DELETE"

        ft = item["ft"]

        if ft == "seq":
            cols = ui.columns(3)
            item["start"] = int(cols[0].number_input("Start", value=int(item.get("start", 1)), step=1, key=f"seq_start_{uid}"))
            item["step"] = int(cols[1].number_input("Step", value=int(item.get("step", 1)), step=1, key=f"seq_step_{uid}"))
            item["pad_zeros"] = int(cols[2].number_input("Zeros (additional)", min_value=0, value=int(item.get("pad_zeros", 3)), step=1, key=f"seq_pad_{uid}"))
            ui.caption(t(lang,"seq_id_caption"))

        if ft == "date_seq":
            default_start = item.get("seq_start_date") or (datetime.now().date() - timedelta(days=365))
            default_end   = item.get("seq_end_date")   or datetime.now().date()
            item["seq_start_date"] = ui.date_input("Start date", value=default_start, key=f"seq_date_start_{uid}")
            item["seq_end_date"] = ui.date_input("End date", value=default_end, key=f"seq_date_end_{uid}")
            item["entries_per_date"] = int(ui.number_input("Max entries per date", min_value=1, value=int(item.get("entries_per_date", 1)), step=1, key=f"entries_per_date_{uid}"))
            ui.caption(t(lang,"seq_dates_caption"))

        if ft == "const":
            item["value"] = ui.text_input(t(lang,"const_value"), value=item.get("value", ""), key=f"const_val_{uid}")

        if ft == "cenum":
            item["values_raw"] = ui.text_area(t(lang,"enum_values"), value=item.get("values_raw", "A,B,C"), key=f"enum_vals_{uid}")
            item["enum_mode"] = ui.selectbox(t(lang,"enum_mode"), options=["Random", "Cycle"], index=0 if item.get("enum_mode","Random")=="Random" else 1, key=f"enum_mode_{uid}")
            item["weights_raw"] = ui.text_input(t(lang,"enum_weights"), value=item.get("weights_raw", ""), key=f"enum_weights_{uid}")
            ui.caption(t(lang,"enum_note"))

        if ft == "genum":
            ui.markdown(t(lang,"grouped_title"))
            gf = ui.text_input(t(lang,"grouped_fields"), value="|".join(item.get("group_fields", ["name","id"])), key=f"group_fields_{uid}", help=t(lang,"grouped_fields_help"))
            item["group_fields"] = [s.strip() for s in gf.split("|") if s.strip() != ""]
            gv = ui.text_area(t(lang,"grouped_values"),
                               value="; ".join(["|".join(g) for g in item.get("group_values", [["Acme","A001"],["Beta","B002"]])]),
                               key=f"group_values_{uid}", help=t(lang,"grouped_values_help"))
            item["group_values"] = _parse_grouped_values(gv)
            item["enum_mode"] = ui.selectbox(t(lang,"enum_mode"), options=["Random","Cycle"], index=0 if item.get("enum_mode","Random")=="Random" else 1, key=f"group_mode_{uid}")
            item["weights_raw"] = ui.text_input(t(lang,"grouped_weights"), value=item.get("weights_raw",""), key=f"group_weights_{uid}")
            if item["group_values"] and item["group_fields"] and not all(len(g)==len(item["group_fields"]) for g in item["group_values"]):
                ui.warning(t(lang,"grouped_warn"))

        if ft == "range":
            cols = ui.columns(4)
            item["min"] = int(cols[0].number_input(t(lang,"min"), value=int(item.get("min", 0)), key=f"min_{uid}"))
            item["max"] = int(cols[1].number_input(t(lang,"max"), value=int(item.get("max", 10)), key=f"max_{uid}"))
            item["float"] = bool(cols[2].checkbox(t(lang,"float"), value=bool(item.get("float", False)), key=f"float_{uid}"))
            item["precision"] = int(cols[3].number_input(t(lang,"precision"), min_value=0, max_value=6, value=int(item.get("precision", 2)), key=f"prec_{uid}"))

        if ft == "date":
            ui.caption(t(lang,"date_note"))

        if ft == "comment":
            item["sentiment"] = ui.selectbox(t(lang,"sentiment"),
                                             options=["Random", "Positive", "Neutral", "Negative"],
                                             index=["Random","Positive","Neutral","Negative"].index(item.get("sentiment","Random")),
                                             key=f"sent_{uid}")
            ui.markdown(t(lang,"trend_title"))
            item["trend_enabled"] = bool(ui.checkbox(t(lang,"trend_enable"), value=bool(item.get("trend_enabled", False)), key=f"trend_enabled_{uid}"))
            if item["trend_enabled"]:
                item["timeline_source"] = ui.selectbox(t(lang,"timeline_source"), options=["Global timeline", "Date field"],
                                                       index=0 if item.get("timeline_source","Global timeline")=="Global timeline" else 1,
                                                       key=f"tl_src_{uid}")
                if item["timeline_source"] == "Date field":
                    item["timeline_date_field"] = ui.text_input(t(lang,"date_field_name"), value=item.get("timeline_date_field",""), key=f"df_ref_{uid}")
                item["trend_type"] = ui.selectbox("Trend type", options=[t(lang,"trend_inc"), t(lang,"trend_dec"), t(lang,"trend_cyc"), t(lang,"trend_jit")],
                                                  index=[t(lang,"trend_inc"), t(lang,"trend_dec"), t(lang,"trend_cyc"), t(lang,"trend_jit")].index(item.get("trend_type", t(lang,"trend_inc"))),
                                                  key=f"trend_type_{uid}")
                item["trend_strength"] = float(ui.slider(t(lang,"trend_strength"), 0.0, 1.0, float(item.get("trend_strength", 0.5)), step=0.01, key=f"trend_strength_{uid}"))
                item["base_preset"] = ui.selectbox(t(lang,"base_dist"),
                                                   options=[ "Balanced","Positive-heavy","Neutral-heavy","Negative-heavy"],
                                                   index=["Balanced","Positive-heavy","Neutral-heavy","Negative-heavy"].index(item.get("base_preset","Balanced")),
                                                   key=f"base_preset_{uid}")

        if ft == "cond_range":
            ui.markdown(t(lang,"depends_on"))
            if all_comment_names:
                default_name = item.get("depends_on", all_comment_names[0])
                try:
                    default_idx = all_comment_names.index(default_name)
                except ValueError:
                    default_idx = 0
                sel = ui.selectbox("Comment field", options=all_comment_names, index=default_idx, key=f"cr_dep_sel_{uid}")
                item["depends_on"] = sel
            else:
                ui.info(t(lang,"no_comments"))
                item["depends_on"] = ui.text_input("Comment field", value=item.get("depends_on",""), key=f"cr_dep_txt_{uid}")

            ui.markdown(t(lang,"ranges_title"))
            cols1 = ui.columns(2)
            item["positive_min"] = int(cols1[0].number_input(t(lang,"pos_min"), value=int(item.get("positive_min", 9)), key=f"pmin_{uid}"))
            item["positive_max"] = int(cols1[1].number_input(t(lang,"pos_max"), value=int(item.get("positive_max", 10)), key=f"pmax_{uid}"))
            cols2 = ui.columns(2)
            item["neutral_min"] = int(cols2[0].number_input(t(lang,"neu_min"), value=int(item.get("neutral_min", 7)), key=f"nmin_{uid}"))
            item["neutral_max"] = int(cols2[1].number_input(t(lang,"neu_max"), value=int(item.get("neutral_max", 8)), key=f"nmax_{uid}"))
            cols3 = ui.columns(2)
            item["negative_min"] = int(cols3[0].number_input(t(lang,"neg_min"), value=int(item.get("negative_min", 0)), key=f"negmin_{uid}"))
            item["negative_max"] = int(cols3[1].number_input(t(lang,"neg_max"), value=int(item.get("negative_max", 6)), key=f"negmax_{uid}"))
            cols4 = ui.columns(3)
            item["any_min"] = int(cols4[0].number_input(t(lang,"any_min"), value=int(item.get("any_min", 0)), key=f"amin_{uid}"))
            item["any_max"] = int(cols4[1].number_input(t(lang,"any_max"), value=int(item.get("any_max", 10)), key=f"amax_{uid}"))
            item["float"] = bool(cols4[2].checkbox(t(lang,"float"), value=bool(item.get("float", False)), key=f"cr_float_{uid}"))
            if item["float"]:
                item["precision"] = int(ui.number_input(t(lang,"precision"), min_value=0, max_value=6, value=int(item.get("precision", 2)), key=f"cr_prec_{uid}"))
    return "OK"

# Build editor if CSV uploaded; else manual builder
if uploaded_file is not None:
    schema_from_upload_mode = True

if schema_from_upload_mode:
    st.sidebar.subheader(t(lang, "csv_fields"))

    # Search box (no 'value='). Clear via callback.
    search_query_raw = st.sidebar.text_input(t(lang, "search"), key="schema_search", placeholder=t(lang, "search_ph"))
    search_query = (search_query_raw or "").strip().lower()
    c1, c2 = st.sidebar.columns([3, 1])
    with c2:
        st.sidebar.button(t(lang, "clear"), key="clear_search_btn", on_click=_clear_schema_search)

    items = st.session_state.schema_items  # already normalized
    display_items = [it for it in items if (search_query in _item_search_text(it))] if search_query else items

    # collect all comment names from FULL list (not filtered)
    all_comment_names = [it.get("name","").strip() for it in items if it.get("ft") == "comment" and it.get("name","").strip()]

    with c1:
        st.sidebar.caption(f"{t(lang,'showing')} {len(display_items)} {t(lang,'of')} {len(items)} {t(lang,'fields')}")

    if len(display_items) == 0:
        st.sidebar.info(t(lang, "no_matches"))
    else:
        to_delete = []
        for i, item in enumerate(display_items):
            result = _render_field_editor(item, i, all_comment_names, st.sidebar)
            if result == "DELETE":
                to_delete.append(item["_uid"])
        if to_delete:
            st.session_state.schema_items = [it for it in items if it["_uid"] not in to_delete]
            st.rerun()

    sbc1, sbc2 = st.sidebar.columns(2)
    if sbc1.button(t(lang, "add_field"), key="add_empty_field_sidebar"):
        st.session_state.schema_items.append({
            "_uid": str(uuid.uuid4()),
            "name": f"{t(lang,'field_label')} {len(st.session_state.schema_items)+1}",
            "ft": "ctext"
        })
        st.rerun()
    if sbc2.button(t(lang, "clear_schema"), key="clear_uploaded_schema"):
        st.session_state.schema_items = []
        st.rerun()

    schema: List[Dict[str, Any]] = [{k: v for k, v in it.items() if k != "_uid"} for it in st.session_state.schema_items]

else:
    # Manual builder (sidebar)
    st.sidebar.subheader("🛠️ Add Custom Fields" if lang=="en" else "🛠️ Agregar Campos")
    num_fields = st.sidebar.number_input("Number of fields" if lang=="en" else "Número de campos", 1, 40, 4)
    schema: List[Dict[str, Any]] = []
    DEFAULT_FIELD_ORDER = [
        ("First name" if lang=="en" else "Nombre", "first"),
        ("Last name" if lang=="en" else "Apellido", "last"),
        ("Comment" if lang=="en" else "Comentario", "comment"),
        ("LTR", "cond_range"),
    ]
    for i in range(num_fields):
        with st.sidebar.expander(f"{t(lang,'field_label')} {i+1}", expanded=(i < 6)):
            col1, col2 = st.columns([2, 2])
            if i < len(DEFAULT_FIELD_ORDER):
                default_name, default_code = DEFAULT_FIELD_ORDER[i]
            else:
                default_name, default_code = f"{t(lang,'field_label')} {i+1}", "ctext"

            with col1:
                field_name = st.text_input(t(lang,"name"), value=default_name, key=f"name_{i}")

            with col2:
                ft_code = st.selectbox(t(lang,"type"), options=FT_CODES,
                                       index=FT_CODES.index(default_code) if default_code in FT_CODES else FT_CODES.index("ctext"),
                                       format_func=lambda c: ft_label(lang, c), key=f"type_{i}")

            st.markdown(f"**{EMOJI.get(ft_code,'')} {field_name or default_name} — _{ft_label(lang, ft_code)}_**")
            field_def = {"ft": ft_code}

            if ft_code == "genum":
                gf = st.text_input(t(lang,"grouped_fields"), value=f"{(field_name or default_name)}_name|{(field_name or default_name)}_id", key=f"m_group_fields_{i}")
                gv = st.text_area(t(lang,"grouped_values"), value="Acme|A001; Beta|B002", key=f"m_group_values_{i}")
                mode = st.selectbox(t(lang,"enum_mode"), options=["Random", "Cycle"], index=0, key=f"m_group_mode_{i}")
                weights = st.text_input(t(lang,"grouped_weights"), value="", key=f"m_group_weights_{i}")
                field_def.update({
                    "group_fields": [s.strip() for s in gf.split("|") if s.strip()!=""],
                    "group_values": _parse_grouped_values(gv),
                    "enum_mode": mode,
                    "weights_raw": weights,
                })
            else:
                field_def["name"] = field_name or default_name

            if ft_code == "seq":
                start = st.number_input("Start", value=1, step=1, key=f"seq_start_{i}")
                step = st.number_input("Step", value=1, step=1, key=f"seq_step_{i}")
                pad_zeros = st.number_input("Zeros (additional)", min_value=0, value=3, step=1, key=f"seq_pad_{i}")
                st.caption(t(lang,"seq_id_caption"))
                field_def.update({"start": int(start), "step": int(step), "pad_zeros": int(pad_zeros)})

            if ft_code == "date_seq":
                seq_start_date = st.date_input("Start date", value=datetime.now().date() - timedelta(days=365), key=f"seq_date_start_{i}")
                seq_end_date = st.date_input("End date", value=datetime.now().date(), key=f"seq_date_end_{i}")
                entries_per_date = st.number_input("Max entries per date", min_value=1, value=1, step=1, key=f"entries_per_date_{i}")
                st.caption(t(lang,"seq_dates_caption"))
                field_def.update({
                    "seq_start_date": seq_start_date,
                    "seq_end_date": seq_end_date,
                    "entries_per_date": int(entries_per_date)
                })

            if ft_code == "const":
                const_val = st.text_input(t(lang,"const_value"), value="", key=f"const_val_{i}")
                field_def.update({"value": const_val})

            if ft_code == "cenum":
                vals = st.text_area(t(lang,"enum_values"), value="A,B,C", key=f"enum_vals_{i}")
                mode = st.selectbox(t(lang,"enum_mode"), options=["Random", "Cycle"], index=0, key=f"enum_mode_{i}")
                weights = st.text_input(t(lang,"enum_weights"), value="", key=f"enum_weights_{i}")
                st.caption(t(lang,"enum_note"))
                field_def.update({"values_raw": vals, "enum_mode": mode, "weights_raw": weights})

            if ft_code == "range":
                rcol1, rcol2 = st.columns([1, 1])
                with rcol1:
                    min_val = st.number_input(t(lang,"min"), value=0, key=f"min_{i}")
                with rcol2:
                    max_val = st.number_input(t(lang,"max"), value=10, key=f"max_{i}")
                float_toggle = st.checkbox(t(lang,"float"), value=False, key=f"float_{i}")
                precision = st.number_input(t(lang,"precision"), min_value=0, max_value=6, value=2, key=f"prec_{i}")
                field_def.update({"min": min_val, "max": max_val, "float": float_toggle, "precision": precision})

            if ft_code == "date":
                st.caption(t(lang,"date_note"))

            if ft_code == "comment":
                sentiment = st.selectbox(t(lang,"sentiment"), options=["Random", "Positive", "Neutral", "Negative"], index=0, key=f"sentiment_{i}")
                field_def["sentiment"] = sentiment
                st.markdown(t(lang,"trend_title"))
                trend_enabled = st.checkbox(t(lang,"trend_enable"), value=False, key=f"trend_enabled_{i}")
                field_def["trend_enabled"] = trend_enabled

                if trend_enabled:
                    tl_source = st.selectbox(t(lang,"timeline_source"), options=["Global timeline", "Date field"], key=f"tl_src_{i}")
                    field_def["timeline_source"] = tl_source
                    if tl_source == "Date field":
                        date_field_options = [f["name"] for f in schema if f.get("ft") in ["date", "date_seq"]]
                        if date_field_options:
                            chosen_df = st.selectbox(t(lang,"date_field_name"), options=date_field_options + ["(manual)"], key=f"df_choice_{i}")
                            if chosen_df == "(manual)":
                                date_field_ref = st.text_input(t(lang,"date_field_name"), value="", key=f"df_manual_{i}")
                            else:
                                date_field_ref = chosen_df
                        else:
                            date_field_ref = st.text_input(t(lang,"date_field_name"), value="", key=f"df_manual_{i}")
                        field_def["timeline_date_field"] = date_field_ref

                    trend_type = st.selectbox("Trend type", options=[t(lang,"trend_inc"), t(lang,"trend_dec"), t(lang,"trend_cyc"), t(lang,"trend_jit")], key=f"trend_type_{i}")
                    trend_strength = st.slider(t(lang,"trend_strength"), 0.0, 1.0, 0.5, step=0.01, key=f"trend_strength_{i}")
                    field_def["trend_type"] = trend_type
                    field_def["trend_strength"] = float(trend_strength)

                    base_preset = st.selectbox(t(lang,"base_dist"), options=["Balanced","Positive-heavy","Neutral-heavy","Negative-heavy"], key=f"base_preset_{i}")
                    field_def["base_preset"] = base_preset

            if ft_code == "cond_range":
                comment_field_options = [f["name"] for f in schema if f.get("ft") == "comment"]
                if comment_field_options:
                    depends_on = st.selectbox("Depends on", options=comment_field_options, index=0, key=f"cr_dep_choice_{i}")
                else:
                    st.info(t(lang,"no_comments"))
                    depends_on = st.text_input("Comment field", value="", key=f"cr_dep_manual_{i}")

                st.markdown(t(lang,"ranges_title"))
                pcol1, pcol2 = st.columns([1, 1])
                with pcol1:
                    pmin = st.number_input(t(lang,"pos_min"), value=9, key=f"pos_min_{i}")
                with pcol2:
                    pmax = st.number_input(t(lang,"pos_max"), value=10, key=f"pos_max_{i}")

                ncol1, ncol2 = st.columns([1, 1])
                with ncol1:
                    nmin = st.number_input(t(lang,"neu_min"), value=7, key=f"neu_min_{i}")
                with ncol2:
                    nmax = st.number_input(t(lang,"neu_max"), value=8, key=f"neu_max_{i}")

                negcol1, negcol2 = st.columns([1, 1])
                with negcol1:
                    negmin = st.number_input(t(lang,"neg_min"), value=0, key=f"neg_min_{i}")
                with negcol2:
                    negmax = st.number_input(t(lang,"neg_max"), value=6, key=f"neg_max_{i}")

                acol1, acol2 = st.columns([1, 1])
                with acol1:
                    amin = st.number_input(t(lang,"any_min"), value=0, key=f"any_min_{i}")
                with acol2:
                    amax = st.number_input(t(lang,"any_max"), value=10, key=f"any_max_{i}")

                float_toggle = st.checkbox(t(lang,"float"), value=False, key=f"cr_float_{i}")
                precision = st.number_input(t(lang,"precision"), min_value=0, max_value=6, value=2, key=f"cr_prec_{i}")

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
# Downloadable schema templates
# ==========================
st.subheader(t(lang, "download_templates"))

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
    st.markdown(t(lang, "blank_template"))
    st.download_button(
        t(lang, "dl_csv_blank"),
        data=_to_csv_bytes(blank_template),
        file_name="schema_template_blank.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        t(lang, "dl_xlsx_blank"),
        data=_to_xlsx_bytes(blank_template),
        file_name="schema_template_blank.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with colB:
    st.markdown(t(lang, "example_template"))
    st.download_button(
        t(lang, "dl_csv_example"),
        data=_to_csv_bytes(example_template),
        file_name="schema_template_example.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        t(lang, "dl_xlsx_example"),
        data=_to_xlsx_bytes(example_template),
        file_name="schema_template_example.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

st.caption(t(lang, "templates_caption"))

# ================
# Generate & show
# ================
schema_to_use = schema
df = generate_dummy_data(rows, schema_to_use, global_timeline=global_timeline)

# ✅ Format date columns as "mm/dd/yyyy, %I:%M %p"
date_fields = [f.get("name") for f in schema_to_use if f.get("ft") in ["date", "date_seq"]]
df_fmt = df.copy()
for col in date_fields:
    if col in df_fmt.columns:
        dtcol = pd.to_datetime(df_fmt[col], errors="coerce")
        df_fmt[col] = dtcol.dt.strftime("%m/%d/%Y, %I:%M %p")

st.subheader(t(lang, "preview").format(rows=rows))
st.dataframe(df_fmt, use_container_width=True)

towrite = io.BytesIO()
df_fmt.to_excel(towrite, index=False, engine="openpyxl")
towrite.seek(0)
st.download_button(
    label=t(lang, "download_excel"),
    data=towrite,
    file_name="custom_dummy_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.markdown("---")
st.caption(t(lang, "footer"))
