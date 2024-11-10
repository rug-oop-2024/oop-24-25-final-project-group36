import json
from typing import Tuple, List, Union

from autoop.core.storage import Storage


class Database:
    """
    Database that stores and manages collections of data,
    Each collection contains multiple entries
    which could identified through their unique IDs.

    Attributes:
        _storage (Storage):
            The storage interface for saving and loading data.
        _data (dict):
            In-memory storage for data organized by collections and IDs.
    """
    def __init__(self, storage: Storage) -> None:
        """
        Initialize the Database with a specified storage system
        and load any existing data.

        Args:
            storage (Storage): The storage system to save and load data.
        """
        self._storage = storage
        self._data = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """
        Store new entry in the specified collection,
        or update it if it already exists.

        Args:
            collection (str): The collection to store the data in.
            id (str): The unique identifier of the entry.
            entry (dict): The data to store.

        Returns:
            dict: The data that was stored.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """
        Gets an entry from the collection.

        Args:
            collection (str): The collection to retrieve data from.
            id (str): The unique identifier of the entry.

        Returns:
            Union[dict, None]:
                The data associated with the ID, or None if it doesn't exist.
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """
        Delete an entry from the specified collection.

        Args:
            collection (str): The collection to delete the data from.
            id (str): The unique identifier of the entry.

        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """
        List all entries in a specified collection.

        Args:
            collection (str): The collection to list the data from.

        Returns:
            List[Tuple[str, dict]]:
            A list of tuples containing the ID and data for each entry.
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """
        Reload all data from the storage to ensure the database is up to date.

        Returns:
            None
        """
        self._load()

    def _persist(self) -> None:
        """
        Persist the current in-memory data to storage,
        update any modified entries,
        and delete removed entries from the storage.

        Returns:
            None
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save
                (json.dumps(item).encode(), f"{collection}/{id}")

        # for things that were deleted, we need to remove them from the storage
        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split("/")[-2:]
            if not self._data.get(collection, id):
                self._storage.delete(f"{collection}/{id}")

    def _load(self) -> None:
        """
        Load data from storage into memory.

        Returns:
            None
        """
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split("/")[-2:]
            data = self._storage.load(f"{collection}/{id}")
            # Ensure the collection exists in the dictionary
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
