from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a specified path is not found."""
    def __init__(self, path: str) -> None:
        """
        Initialize NotFoundError.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """Local storage implementation of the Storage interface."""

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize LocalStorage with a base path.

        Args:
            base_path (str): The base directory for local storage.
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a specified key.

        Args:
            data (bytes): Data to save.
            key (str): Key (path) to save data.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a specified key.

        Args:
            key (str): Key (path) to load data.

        Returns:
            bytes: Loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete data at a specified key.

        Args:
            key (str): Key (path) to delete data.
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all files under a given prefix.

        Args:
            prefix (str): Prefix to list files.

        Returns:
            List[str]: List of file paths.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Assert that a specified path exists.

        Args:
            path (str): Path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with the specified path.

        Args:
            path (str): Path to join with the base path.

        Returns:
            str: Joined path.
        """
        return os.path.join(self._base_path, path)
