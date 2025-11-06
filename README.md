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

