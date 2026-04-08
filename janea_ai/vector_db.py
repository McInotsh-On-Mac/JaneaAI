from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import (
        HuggingFaceBgeEmbeddings as HuggingFaceEmbeddings,
    )

from .config import Settings


MANIFEST_NAME = ".source_manifest.json"
VECTOR_STORE_NAME = "vector_store.json"


def _build_embeddings(settings: Settings) -> Embeddings:
    return HuggingFaceEmbeddings(model_name=settings.embeddings_model_name)


def _collect_pdf_paths(settings: Settings) -> list[Path]:
    source_path = settings.pdf_source_path.expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF source path not found: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() != ".pdf":
            raise ValueError(f"Source file must be a PDF: {source_path}")
        return [source_path.resolve()]

    pdf_paths = sorted(path.resolve() for path in source_path.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in: {source_path}")
    return pdf_paths


def _build_source_manifest(settings: Settings, pdf_paths: list[Path]) -> dict[str, Any]:
    file_entries: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        stat = pdf_path.stat()
        file_entries.append(
            {
                "path": str(pdf_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    return {
        "source_path": str(settings.pdf_source_path.expanduser().resolve()),
        "embeddings_model_name": settings.embeddings_model_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "pdf_files": file_entries,
    }


def _manifest_file(persist_directory: Path) -> Path:
    return persist_directory / MANIFEST_NAME


def _vector_store_file(persist_directory: Path) -> Path:
    return persist_directory / VECTOR_STORE_NAME


def _read_manifest(persist_directory: Path) -> dict[str, Any] | None:
    path = _manifest_file(persist_directory)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(persist_directory: Path, manifest: dict[str, Any]) -> None:
    persist_directory.mkdir(parents=True, exist_ok=True)
    path = _manifest_file(persist_directory)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_documents(pdf_paths: list[Path]):
    documents = []
    for pdf_path in pdf_paths:
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    return documents


def _create_vector_db(
    settings: Settings,
    pdf_paths: list[Path],
    source_manifest: dict[str, Any],
) -> VectorStore:
    documents = _load_documents(pdf_paths)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = _build_embeddings(settings)

    vector_db = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    settings.persist_directory.mkdir(parents=True, exist_ok=True)
    vector_db.dump(str(_vector_store_file(settings.persist_directory)))
    _write_manifest(settings.persist_directory, source_manifest)
    return vector_db


def create_vector_db(settings: Settings) -> VectorStore:
    pdf_paths = _collect_pdf_paths(settings)
    source_manifest = _build_source_manifest(settings, pdf_paths)
    return _create_vector_db(settings, pdf_paths, source_manifest)


def load_vector_db(settings: Settings) -> VectorStore:
    vector_store_path = _vector_store_file(settings.persist_directory)
    if not vector_store_path.exists():
        raise FileNotFoundError(f"Vector store file not found: {vector_store_path}")
    embeddings = _build_embeddings(settings)
    return InMemoryVectorStore.load(
        path=str(vector_store_path),
        embedding=embeddings,
    )


def _rebuild_vector_db(
    settings: Settings,
    pdf_paths: list[Path],
    source_manifest: dict[str, Any],
) -> VectorStore:
    if settings.persist_directory.exists():
        shutil.rmtree(settings.persist_directory)
    return _create_vector_db(settings, pdf_paths, source_manifest)


def load_or_create_vector_db(settings: Settings) -> VectorStore:
    pdf_paths = _collect_pdf_paths(settings)
    source_manifest = _build_source_manifest(settings, pdf_paths)

    if settings.persist_directory.exists():
        existing_manifest = _read_manifest(settings.persist_directory)
        if existing_manifest == source_manifest:
            try:
                return load_vector_db(settings)
            except Exception:
                pass

    return _rebuild_vector_db(settings, pdf_paths, source_manifest)
