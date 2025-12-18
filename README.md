# Supervised-SADP-
This work is an extension of Spike Agreement Dependent Plasticity: A Scalable Bio-Inspired Learning Paradigm for Spiking Neural Networks (https://doi.org/10.48550/arXiv.2508.16216
). In this study, we extend the original framework by introducing a supervised SADP formulation and evaluating its performance across multiple datasets.

In the primary analysis, we present the main benchmark datasets used to conduct the experiments. In the device-based supervised SADP analysis, we further evaluate data collected from a memtransistor, in order to verify whether the proposed algorithm remains energy-efficient and stable when deployed on realistic neuromorphic synaptic hardware.

Additionally, we extend our study to medical imaging datasets. For colored images, we use the lung and colon histopathology dataset, where lung and colon samples are evaluated separately. Instead of using the multiple cancer sub-categories provided in the dataset, all cancer-related classes are merged into a single “cancerous” class, resulting in a binary classification task (cancerous vs. non-cancerous) (https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

We also evaluate our approach on a grayscale MRI brain tumor dataset, which is used for binary classification of tumor vs. non-tumor images (https://www.kaggle.com/datasets/mohammadhossein77/brain-tumors-dataset). We also conducted an Optuna-based hyperparameter optimization analysis.
