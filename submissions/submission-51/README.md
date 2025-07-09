# CaliPSo: Calibrated Predictive Models with Sharpness as Loss Function
The notebook provided implements **"CaliPSo: Calibrated Predictive Models with Sharpness as Loss Function"**.


## Abstract

Conformal prediction methods have become increasingly common for accurately capturing uncertainty with machine learning models. However, conformal prediction typically recalibrates an existing model, making it heavily reliant on the quality of the uncalibrated model. Moreover, they either enforce marginal calibration strictly, yielding potentially coarse predictive intervals, or attempt to strike a balance between interval coarseness and calibration.

Motivated by these shortcomings, we present CaliPSo a neural network model that is marginally calibrated out-of-the-box and stays so throughout training. This property is achieved by adding a model-dependent constant to the model prediction that shifts it in a way that ensures calibration. During training, we then leverage this to focus exclusively on sharpness - the property of returning tight predictive intervals - rendering the model more useful at test time.

We show thorough experimental results, where our method exhibits superior performance compared to several state-of-the-art approaches.


## Notebook

The notebook provides an implementation of the CaliPSo method, applying the technique to UCI regression datasets and evaluating the performance metrics. The notebook can be run in Google Colab Free tier, and the notebook is configured with the same hyperparameters as the ones reported in the paper.
