# Create a Markdown file summarizing phantom groups and metadata

markdown_content = """# Phantom Groups and Metadata

## Phantom Groups

There are **20 phantoms** organized by their **adipose shell type**:

- **A2 Adipose Shell:**  
  `A2F1`, `A2F11`, `A2F12`, `A2F2`, `A2F3`

- **A3 Adipose Shell:**  
  `A3F1`, `A3F2`, `A3F3`, `A3F11`, `A3F12`

- **A14 Adipose Shell:**  
  `A14F1`, `A14F2`, `A14F3`, `A14F11`, `A14F12`

- **A16 Adipose Shell:**  
  `A16F1`, `A16F2`, `A16F3`, `A16F11`, `A16F14`

---

## Example Metadata (for Phantom `A2F1`)

Below is a simplified view of the metadata entries for **phantom A2F1**:

| Experiment | ID | Tumor Diameter (mm) | Tumor Shape | Tumor Position (x, y, z) | BI-RADS | Adipose Ref | Fibrous Ref | Empirical Ref | Date | Tumor in Fibrous | Notes |
|-------------|----|---------------------|--------------|---------------------------|----------|--------------|--------------|----------------|------|------------------|-------|
| 1 | 1 | 3.0 | sphere | (2.25, 2.25, -6.5) | 1 | 3 | 2 | 16 | 2021-07-30 | 0 | Standard sample |
| 2 | 2 | — | — | — | 1 | 3 | — | 16 | 2021-07-30 | — | Missing tumor data |
| 4 | 4 | 2.5 | sphere | (2.25, 2.25, -6.5) | 1 | 6 | 5 | 16 | 2021-07-30 | 0 | Updated adipose/fibrous refs |
| 5 | 5 | — | — | — | 1 | 6 | — | 16 | 2021-07-30 | — | Missing tumor data |
| 7 | 7 | 2.0 | sphere | (2.25, 2.25, -6.5) | 1 | 9 | 8 | 16 | 2021-07-30 | 0 | Final reference update |

**Key:**  
- `—` = data not available (NaN)  
- **Adipose/Fibrous/Empirical Ref IDs** correspond to material references used in experiments.  
- **BI-RADS** = Breast Imaging Reporting and Data System score.  
- **Tumor in Fibrous (tum_in_fib)** indicates if the tumor is inside the fibrous region (0 = No).
"""






# Create a Markdown file summarizing the provided analysis output

from textwrap import dedent
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

md_path = "/mnt/data/employee_data_analysis.md"

columns = [
    "adi_ref_id","adi_x","adi_y","ant_rad","ant_z","birads","date","emp_ref_id",
    "fib_ang","fib_ref_id","fib_x","fib_y","id","n_expt","n_session","phant_id",
    "tum_diam","tum_in_fib","tum_shape","tum_x","tum_y","tum_z"
]

dtypes = [
    "int","float","float","float","float","int","str","int",
    "float","float, int","float","float","int","int","int","str",
    "float","float, int","str","float","float","float"
]

df_schema = pd.DataFrame({"Column": columns, "Data type(s)": dtypes})

sample_record_items = [
    ("n_expt", "1", "int"),
    ("id", "1", "int"),
    ("phant_id", "A2F1", "str"),
    ("tum_diam", "3.0", "float"),
    ("tum_shape", "sphere", "str"),
    ("tum_x", "2.25", "float"),
    ("tum_y", "2.25", "float"),
    ("tum_z", "-6.5", "float"),
    ("birads", "1", "int"),
    ("adi_ref_id", "3", "int"),
    ("emp_ref_id", "16", "int"),
    ("date", "20210730", "str"),
    ("n_session", "1", "int"),
    ("ant_rad", "18.0", "float"),
    ("ant_z", "-6.5", "float"),
    ("fib_ang", "0.0", "float"),
    ("adi_x", "0.0", "float"),
    ("adi_y", "0.0", "float"),
    ("fib_ref_id", "2", "int"),
    ("fib_x", "0.0", "float"),
    ("fib_y", "0.0", "float"),
    ("tum_in_fib", "0", "int"),
]

md_content = dedent(f"""
# Employee Data: Useful Information

This markdown captures the key, actionable details parsed from the command output you provided.

## Overview
- Source command: `cd /workspace && python detailed_analysis.py`
- High-level tasks performed: Load and inspect two pickles: `md_list_s11_emp.pickle` and `fd_data_s11_emp.pickle`.

---

## 1) `md_list_s11_emp.pickle`
- Records: **200** (list of **dictionary** records)
- **Total unique columns:** 22
- **Column names (normalized):**
  `{columns}`

### Column data types
Below is the consolidated schema inferred from the analysis output.
""")

# Write the initial content
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

# Append the schema as a Markdown table
schema_md = df_schema.to_markdown(index=False)
with open(md_path, "a", encoding="utf-8") as f:
    f.write("\n" + schema_md + "\n")

# Append sample record
sample_intro = dedent("""
### Sample record (first item)

| Field | Value | Type |
|------:|:-----:|:----:|
""")

with open(md_path, "a", encoding="utf-8") as f:
    f.write("\n" + sample_intro)
    for field, value, typ in sample_record_items:
        f.write(f"| `{field}` | `{value}` | `{typ}` |\n")

# Add deprecation note and second file details
dep_note = dedent("""

> **Note on warning**
>
> The script emitted a DeprecationWarning regarding `numpy.core.numeric` → `numpy._core.numeric._frombuffer`.
> This does not affect the parsed values above, but the code should be updated to use public NumPy APIs.

---

## 2) `fd_data_s11_emp.pickle`
- **Type:** `numpy.ndarray`
- **Shape:** `(200, 1001, 72)`
- **Dtype:** `complex128`

### Quick interpretation
- Likely 200 samples × 1001 timepoints (or features) × 72 channels (or sensors).
- Complex dtype suggests frequency-domain or analytic-signal data.

---

## Practical tips
- Prefer explicit dtype conversions when loading mixed-type columns (`fib_ref_id`, `tum_in_fib` show mixed int/float).
- Treat `date` as a string in the input; parse to datetime format on load if needed (e.g., `%Y%m%d` for `20210730`).
- Validate coordinate fields (`tum_x`, `tum_y`, `tum_z`, `ant_z`) are in consistent units and frames.
- Ensure downstream code avoids deprecated NumPy internals mentioned above.

""")

with open(md_path, "a", encoding="utf-8") as f:
    f.write(dep_note)

print(f"Wrote: {md_path}")

