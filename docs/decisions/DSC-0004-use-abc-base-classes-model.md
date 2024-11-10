# DSC-0004: Use abc base classes model
# Date: 2024-10-25
# Decision: Used abstract base classes (ABC) from Python’s abc module to define model classes.
# Status: Accepted
# Motivation: To help maintain redability and reusability in the code. Using ABCs ensure a structured and uniform interface for all model classes. 
# Reason: Allows definition of abstract methods that must be implemented by any subclass, so that it stays consistent across model classes.
# Limitations: Could complicate class design and make it difficult to debug when the project has many files
# Alternatives: Using simple inheritance without abstract base classes or relying on duck typing