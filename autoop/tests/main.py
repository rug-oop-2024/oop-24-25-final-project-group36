
import unittest
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline

__all__ = [
    "TestDatabase",
    "TestStorage",
    "TestFeatures",
    "TestPipeline",
]

if __name__ == '__main__':
    unittest.main()
