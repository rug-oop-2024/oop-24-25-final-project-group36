# DSC-0005: Use file hashes
# Date: 2024-10-25
# Decision: Used file hashes to ensure data integrity and track changes in file versions.
# Status: Accepted
# Motivation: To maintain data integrity and ensure that files used in the project are unaltered or consistent across different versions. 
# Reason: To ensure correct data versions are used in model training or testing.
# Limitations: May impact performance if applied to very large files. 
# Alternatives: Consider using file metadata (eg timestamps or file size) to detect changes. 