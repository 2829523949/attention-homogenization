from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(*, image_path: Path, processor: str, version: str, model_revision: str, config: dict[str, Any], query: str | None = None) -> tuple[str, dict[str, Any]]:
    fields = {"image_sha256": sha256_file(image_path), "processor": processor, "version": version, "model_revision": model_revision, "config": config, "query": query}
    return hashlib.sha256(stable_json(fields).encode()).hexdigest(), fields


class JsonCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path(key)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path(key)
        encoded = stable_json(payload) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise RuntimeError(f"cache collision/refusing overwrite: {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
        return path


class CachedRegionProcessor:
    def __init__(self, processor: Any, cache: JsonCache, model_revision: str):
        self.processor=processor; self.cache=cache; self.model_revision=model_revision
        self.name=processor.name; self.version=processor.version

    @staticmethod
    def _restore(rows: list[dict[str, Any]]) -> list[Any]:
        from harness.processors.base import RegionPrediction
        return [RegionPrediction(**{**row,"bbox_xyxy_pixels":tuple(row["bbox_xyxy_pixels"]),"bbox_xyxy_normalized":tuple(row["bbox_xyxy_normalized"])}) for row in rows]

    def _cached(self,image_path:Path,query:str|None,config:dict[str,Any],call:Any)->list[Any]:
        key,fields=cache_key(image_path=image_path,processor=self.name,version=self.version,model_revision=self.model_revision,config=config,query=query);cached=self.cache.get(key)
        if cached is None:
            rows=call();cached={"key_fields":fields,"predictions":[row.__dict__ for row in rows]};self.cache.put(key,cached)
        return self._restore(cached["predictions"])

    def predict(self,image_path:Path,**kwargs:Any)->list[Any]:
        return self._cached(image_path,None,dict(kwargs),lambda:self.processor.predict(image_path,**kwargs))

    def predict_query(self,image_path:Path,query:str,**kwargs:Any)->list[Any]:
        return self._cached(image_path,query,{"mode":"query",**kwargs},lambda:self.processor.predict_query(image_path,query,**kwargs))


class CachedColorReader:
    def __init__(self,reader:Any,cache:JsonCache,model_revision:str):
        self.reader=reader;self.cache=cache;self.model_revision=model_revision;self.name=reader.name;self.version=reader.version

    def read(self,image_path:Path,region:Any,mask_path:Path|None=None)->Any:
        from harness.processors.base import ColorPrediction
        config={"bbox":list(region.bbox_xyxy_pixels),"mask_path":str(mask_path) if mask_path else None}
        key,fields=cache_key(image_path=image_path,processor=self.name,version=self.version,model_revision=self.model_revision,config=config)
        cached=self.cache.get(key)
        if cached is None:
            result=self.reader.read(image_path,region,mask_path);cached={"key_fields":fields,"prediction":result.__dict__};self.cache.put(key,cached)
        return ColorPrediction(**cached["prediction"])
