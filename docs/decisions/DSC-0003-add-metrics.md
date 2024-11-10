# DSC-0003: Add metrics
# Date: 2024-10-24
# Decision: Implemented the metrics: accuracy, precision, and recall for classification tasks and mean_squared_error, mean_absolute_error, and r2_score for regression tasks.
# Status: Accepted
# Motivation: To evaluate and compare the performance of different ML models more effectively. 
# Reason: Each metric offers insights into the model performance. For classification, accuracy gives overall success rate, while precision and recall address class-specific performance, and handles  datasets which are imbalanced. For regression, mean squared error and mean absolute error express prediction accuracy, and R2 score measures how well the model explains the variability of the data.
# Limitations: The metrics we have may not capture all aspects of the model performance.
# Alternatives: Consider adding other metrics for both tasks.