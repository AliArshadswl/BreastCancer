# UMBMID-Gen3-Dataset: Detailed Description
## Overview
The UMBMID-Gen3-Dataset is a comprehensive medical imaging phantom dataset designed for microwave breast imaging research. This dataset contains 200 experimental measurements with corresponding metadata and frequency domain data, specifically designed to study the effects of breast tissue density on tumor detection in microwave imaging systems.
## Dataset Structure
### Files Included
1. **md_list_s11_emp.pickle**: Metadata containing experimental parameters and phantom characteristics
2. **fd_data_s11_emp.pickle**: Frequency domain measurement data (complex values)
### Data Dimensions
- **Total Experiments**: 200
- **Frequency Points**: 1,001
- **Measurement Channels**: 72
- **Data Type**: Complex64 values (frequency domain S-parameters)
---
## Metadata Columns (22 columns)
### 1. Identification Columns
- **`id`** (int): Unique experiment identifier (1-200)
- **`n_expt`** (int): Experiment number
- **`n_session`** (int): Session number for the experiment
- **`phant_id`** (str): Phantom type identifier (20 unique types: A2F1, A2F11, A2F12, A2F2, A2F3, A3F1, A3F11, A3F12, A3F2, A3F3, A14F1, A14F11, A14F12, A14F2, A14F3, A16F1, A16F11, A16F14, A16F2, A16F3)
- **`date`** (str): Date of experiment in YYYYMMDD format
### 2. Tumor Characteristics
- **`tum_diam`** (float): Tumor diameter in cm
  - Values: 1.0, 1.5, 2.0, 2.5, 3.0 cm (20 samples each)
  - NaN values indicate no tumor present
- **`tum_shape`** (str): Tumor shape (all samples: "sphere")
- **`tum_x`** (float): Tumor X-coordinate position
- **`tum_y`** (float): Tumor Y-coordinate position  
- **`tum_z`** (float): Tumor Z-coordinate position
- **`tum_in_fib`** (float/int): Binary indicator if tumor is in fibroglandular tissue (0 = no, 1 = yes)
  - All tumor samples have value 0 (tumors not in fibroglandular tissue)
### 3. Adipose (Fatty) Tissue Shell
- **`adi_ref_id`** (int): Adipose tissue reference identifier (ranging 3-319)
- **`adi_x`** (float): Adipose tissue X-coordinate (all samples: 0.0)
- **`adi_y`** (float): Adipose tissue Y-coordinate (all samples: 0.0)
### 4. Fibroglandular Tissue Shell
- **`fib_ref_id`** (float/int): Fibroglandular tissue reference identifier
- **`fib_x`** (float): Fibroglandular tissue X-coordinate (all samples: 0.0)
- **`fib_y`** (float): Fibroglandular tissue Y-coordinate (all samples: 0.0)
- **`fib_ang`** (float): Fibroglandular tissue orientation angle in degrees
  - Values: -30.0°, -15.0°, 0.0°, 15.0°, 180.0°
  - Distribution: 0.0° (65 samples), -30.0° (10 samples), 180.0° (10 samples), -15.0° (10 samples), 15.0° (5 samples)
### 5. Phantom Geometry
- **`ant_rad`** (float): Antenna radius (all samples: 18.0)
- **`ant_z`** (float): Antenna Z-coordinate (all samples: -6.5)
### 6. Classification
- **`birads`** (int): BI-RADS classification score
  - Values: 1 (benign, 70 samples), 2 (benign finding, 100 samples), 3 (probably benign, 30 samples)
- **`emp_ref_id`** (int): Employee/reference person identifier
---
## Tissue Distribution Analysis
### Adipose Tissue (Fatty Breast Component)
- **Coverage**: Present in ALL 200 samples (100%)
- **Coordinate System**: Centered at origin (0,0)
- **Reference IDs**: Unique identifiers ranging from 3 to 319
- **Phantom Types**: 4 adipose configurations (A2, A3, A14, A16)
### Fibroglandular Tissue (Dense Breast Component)  
- **Coverage**: Present in only 100 samples (50%)
- **Perfect Correlation with Tumors**: ALL tumor samples have fibroglandular tissue, NO non-tumor samples have fibroglandular tissue
- **Orientation**: Variable angles from -30° to 180°
- **Phantom Types**: 6 fibroglandular configurations (F1, F2, F3, F11, F12, F14)
### Tumor Distribution
- **Total Tumor Samples**: 100 (50%)
- **Total Non-tumor Samples**: 100 (50%)
- **Tumor Sizes**: Evenly distributed across 5 diameter categories (20 samples each)
- **Tumor Location**: All positioned outside fibroglandular tissue (tum_in_fib = 0)
---
## Phantom Configuration Matrix
The dataset uses a systematic phantom naming convention: **A[Adipose]F[Fibroglandular]**
### Adipose Tissue Types (A-component)
- **A2**: Low-density adipose tissue
- **A3**: Medium-low density adipose tissue  
- **A14**: Medium-high density adipose tissue
- **A16**: High-density adipose tissue
### Fibroglandular Tissue Types (F-component)
- **F1**: Type 1 fibroglandular configuration
- **F2**: Type 2 fibroglandular configuration
- **F3**: Type 3 fibroglandular configuration
- **F11**: Type 11 fibroglandular configuration
- **F12**: Type 12 fibroglandular configuration
- **F14**: Type 14 fibroglandular configuration
### Phantom Distribution
- **Total Combinations**: 20 unique phantom types
- **Samples per Phantom**: 10 samples each
- **Study Design**: Balanced to ensure equal representation
---
## Frequency Domain Data Description
### Data Structure
- **Shape**: (200, 1001, 72) complex128 values
- **Dimensions**:
  - **First dimension (200)**: Experiments (matches metadata records)
  - **Second dimension (1001)**: Frequency points (likely microwave frequencies)
  - **Third dimension (72)**: Measurement channels/antennas
### Data Type
- **Complex Values**: Each measurement contains real and imaginary components
- **Precision**: complex128 (double precision complex numbers)
- **Interpretation**: S-parameters or scattering parameters for microwave imaging
### Experimental Setup
- **Antenna Array**: 72-element circular or multi-static antenna array
- **Frequency Sweep**: 1,001 frequency points covering microwave spectrum
- **Measurement Type**: Likely S11 parameters (reflection coefficients)
---
## Clinical Research Applications
### Primary Research Questions
1. **Breast Density Effects**: How does fibroglandular tissue affect microwave imaging?
2. **Tumor Detection**: Performance comparison in fatty vs. dense breast tissue
3. **Phantom Validation**: Validation of imaging system performance
4. **Frequency Analysis**: Impact of frequency on tumor detection capability
### Study Design Implications
- **Balanced Dataset**: Equal representation of tumor/non-tumor cases
- **Controlled Variables**: Systematic variation of tissue components
- **Multi-frequency Analysis**: Rich frequency domain information for spectrum analysis
- **Standardized Phantoms**: Reproducible experimental conditions
### Clinical Relevance
- **Breast Cancer Screening**: Relevant for microwave breast imaging systems
- **Dense Breast Challenges**: Addresses limitations in dense breast tissue screening
- **System Optimization**: Data for optimizing microwave imaging parameters
- **Algorithm Development**: Training data for machine learning approaches
---
## Data Quality and Validation
### Completeness
- **Metadata**: 100% complete for all key parameters
- **Frequency Data**: No missing values in complex measurements
- **Phantom Coverage**: Full factorial design of tissue combinations
### Consistency
- **Coordinate System**: Consistent reference frame across all experiments
- **Measurement Protocol**: Standardized antenna and phantom positioning
- **Data Format**: Uniform data types and structures
### Reproducibility
- **Phantom Types**: Clearly defined and reproducible phantom configurations
- **Experimental Parameters**: Well-documented setup conditions
- **Measurement Accuracy**: Precise frequency domain data with complex values
---
## Usage Recommendations
### Data Loading
```python
import pickle
import numpy as np
# Load metadata
with open('md_list_s11_emp.pickle', 'rb') as f:
    metadata = pickle.load(f)
# Load frequency domain data
with open('fd_data_s11_emp.pickle', 'rb') as f:
    fd_data = np.load(f)
```
### Common Analysis Tasks
1. **Tumor Classification**: Binary classification based on tum_diam presence
2. **Tissue Analysis**: Study adipose vs. fibroglandular effects
3. **Frequency Analysis**: Analyze frequency response patterns
4. **Phantom Comparison**: Compare different tissue combinations
5. **Imaging Algorithm Development**: Use as training/validation data
### Statistical Considerations
- **Balanced Classes**: Equal tumor/non-tumor samples (100 each)
- **Independent Samples**: Each experiment is independent
- **Controlled Variables**: Systematic variation of key parameters
- **Power Analysis**: Sufficient sample size for statistical analysis
---
## Citation and Attribution
**Dataset Name**: UMBMID-Gen3-Dataset  
**Version**: Gen3  
**Total Samples**: 200  
**Date of Creation**: 2021-07-30 (based on sample date)  
**Research Domain**: Microwave Breast Imaging  
**Tissue Types**: Adipose and Fibroglandular Breast Tissue  
**Tumor Models**: Spherical tumors (1.0-3.0 cm diameter)
---
*This dataset represents a significant contribution to microwave breast imaging research, providing a comprehensive foundation for algorithm development, system validation, and clinical translation studies.*
