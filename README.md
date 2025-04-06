# 🧬 Multi-Omics Prediction: Proteomics from Transcriptomics and Methylation 

## Project Overview

This project tackles the challenge of predicting one omics modality (proteomics) from others (transcriptomics and methylation), within a multi-omics framework. The goal was to explore how effectively different data types can be leveraged to infer protein abundance, with the broader context of building models that could one day contribute to predictive tasks such as drug response.

## 🧠 Conceptual and Technical Approach 

### Initial Exploration

I began by exploring the dataset and quickly identified the high dimensionality across the omics layers. Initially, I was curious about predicting drug response using multi-omics data, but upon reviewing the task, I refocused on predicting an omics layer instead- specifically, proteomics.

### Choice of Modalities

I chose transcriptomics and methylation as input modalities to predict proteomics, based on the biological hierarchy and mechanistic flow of gene expression regulation. Although copy number variation was considered as a future addition, its differing distribution and count-based nature made it a less straightforward integration at this stage.

### Dimensionality Challenges

The dataset's high dimensionality posed immediate challenges. To mitigate this, I applied upsampling techniques to increase the sample size, which helped with model stability. I also implemented two baseline models:

- **Linear Regression** – to establish a simple, interpretable benchmark.
- **Ridge Regression** – to introduce L2 regularisation and address overfitting risk.

### Dimensionality Reduction

To further address the data complexity, I used Principal Component Analysis (PCA). I was pleasantly surprised to find that only ~60 principal components were needed to explain 99% of the variance in the data. This validated the use of PCA as a preprocessing step.

### Target Simplification

Given the large number of protein targets, I reduced the output dimensionality to focus on a subset of more predictable proteins (larger variance). This was done to reduce noise and create a more tractable predictive task.

### Advanced Modelling

I then implemented a transformer-based model to explore non-linear and deep learning approaches. Initially, the model suffered from overfitting, but I integrated regularisation strategies and architectural changes to improve generalisation. While the transformer didn’t outperform Ridge Regression across all targets, it showed superior performance on some proteins. This suggests that, with further tuning or with additional data and modalities (e.g., using mid-/late-fusion strategies), the transformer could become more competitive.

### Reflections

The biggest challenge was the small dataset size, which limits the potential of complex models. However, this also made the task more interesting and realistic in the context of biomedical applications, where sample sizes are often limited.

## 📁 Project Structure

```text
.
├── main.ipynb                  # Main notebook
├── multi_omics_env.yml         # Conda environment
└── utils/
    ├── utils.py                # General functions and imports
    └── transformer_utils.py    # Transformer-specific functions and setup
```

## 🛠️ How to Build and Run the Code

### Environment Setup

To replicate the development environment, use the provided Conda environment file.

1. **Create the environment**  
   Run the following command from the project root:

   ```bash
   conda env create -f multi_omics_env.yml
   ```
2. **Activate the environment**

   ```bash
    conda activate multi_omics_env
   ```

3. **Register the environment as a Jupyter kernel**

   This ensures the notebook's run using the correct environment:

   ```bash
   python -m ipykernel install --user --name=multi_omics_env
   ```

4. **Launch Jupyter Lab or Notebook**

   ```bash
   jupyter lab
   ```
   Then, when opening a notebook, select the kernel named multi_omics_env from the kernel selector.

5. **Run the Notebook**

   Once the dependencies have been installed, the notebook should be ready to run.

## ⌛ Time to Completion

This project was completed over the course of a weekend (~2–3 days), including exploration, modelling, and documentation.
