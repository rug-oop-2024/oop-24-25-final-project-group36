from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    Registry to handle the storage and retrieval of artifacts.
    Artifacts consist of data and metadata.
    Data is stored in a storage system,
    and metadata is stored in a database.

    Attributes:
        _database (Database): The database for storing artifact metadata.
        _storage (Storage): The storage system for storing artifact data.
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initialize the ArtifactRegistry with database and storage.

        Args:
            database (Database): The database to store artifact metadata.
            storage (Storage): The storage system to save artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: "Artifact") -> None:
        """
        Register a new artifact by saving its data to storage
        and its metadata to the database.

        Args:
            artifact (Artifact): The artifact to be registered,
            contains both data and metadata.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Retrieve a list of artifacts, which is optionally filtered by type.

        Args:
            type (Optional[str]):
            When specified, only artifacts of this type will be listed.

        Returns:
            List[Artifact]: A list of artifacts matching the specified type,
            or all artifacts if no type is specified.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieve a specific artifact through their ID.

        Args:
            artifact_id (str):
            The unique identifier of the artifact to retrieve.

        Returns:
            Artifact: The artifact corresponding to the given ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Delete an artifact by their ID,
        removes both data from storage and metadata from database.

        Args:
            artifact_id (str): The unique identifier of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    A class that represents an automated machine learning system;
    managing storage and database resources,
    also as an artifact registry for models and data.

    Attributes:
        _storage (LocalStorage):
            The local storage for saving and loading artifacts.
        _database (Database):
            The database for storing artifact metadata.
        _registry (ArtifactRegistry):
            The registry managing artifacts within the system.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize the AutoMLSystem with
        specific storage and database instances.

        Args:
            storage (LocalStorage): The storage system to manage artifact data.
            database (Database): The database to manage artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieve the singleton instance of the AutoMLSystem.
        If it does not exist,
        creates one with default storage and database paths.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> "ArtifactRegistry":
        """
        Gives access to the artifact registry used by the AutoMLSystem.

        Returns:
            ArtifactRegistry: The artifact registry instance.
        """
        return self._registry
