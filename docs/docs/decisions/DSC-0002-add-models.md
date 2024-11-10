# DSC-0002: Add models
# Date: 2024-10-23
# Decision: Use sklearn for the model extensions
# Status: Accepted
# Motivation: To simplify the process of adding and experimenting with the models in the project. 
# Reason: Sklearn provides library of ML models with a consistent interface, making it very straightforward to integrate models for classification and regression tasks. 
# Limitations: Sklearn is primarily designed for tabular data, therefore handling unstructured data (eg: images, text) may need additional preprocessing.
# Alternatives: Could use TensorFlow or PyTorch which provides more flexibility and tools.