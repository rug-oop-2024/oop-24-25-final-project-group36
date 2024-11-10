# DSC-0006: Serialize pipelines with pickle
# Date: 2024-10-30
# Decision: Used Python’s pickle module to serialize and save pipelines.
# Status: Accepted
# Motivation: To enable easy saving and loading of trained pipelines. 
# Reason: Serialization simplifies deployment and allows consistent use of the same model version across different environments.
# Limitations: Pickle can produce large files which is not good for storage and loading time
# Alternatives: Consider using other tools such as joblib
