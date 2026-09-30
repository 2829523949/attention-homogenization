from __future__ import annotations

import hashlib, json, os, tempfile, unittest
from pathlib import Path

from harness.cache import CachedRegionProcessor, JsonCache, cache_key
from harness.compiler.fact_compiler import compile_facts, serialize_facts
from harness.configs.paths import Paths, ensure_d_drive_environment
from harness.dataset import composition_cases
from harness.io_utils import write_new
from harness.processors.base import EntityVisualState, RegionPrediction, RelationState, VisualState
from harness.processors.defocus import auto_defocus
from harness.processors.geometry import horizontal_relation


def digest(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


class HarnessTests(unittest.TestCase):
    def test_frozen_case_list_and_scorer(self):
        p=Paths(); self.assertEqual(len(composition_cases(p)),200)
        self.assertEqual(digest(p.families),"b8d366c296403646a8c9a71c09cf8dd8ba007eea2d1d7a3c930cd95edd1816ee")
        self.assertEqual(digest(p.qa),"433ccc24f7a7070ae56f756e8936f10c181105e6fa31b3fa9bf206fcf06c3c87")
        self.assertEqual(digest(p.scorer),"841f6b0dfbbd1710e9c1e74ed583fc609a5113740ad83cf9090ad71bc237f18e")

    def test_gt_unavailable_in_auto_and_visual_state(self):
        r=RegionPrediction.from_pixels([1,1,5,5],10,10,score=1,label="x",source="oracle_gt")
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image
            image=Path(d)/"i.png"; Image.new("RGB",(10,10)).save(image)
            with self.assertRaises(AssertionError): auto_defocus(image,[r],Path(d)/"o.png")
        payload={"case_id":"x","requested_task":"REL2C_LEFT","entities":[],"relations":[],"acquisition_trace":[],"uncertainty":{"gt_bbox":[1,2,3,4]},"processor_versions":{}}
        with self.assertRaises(AssertionError): VisualState.from_dict(payload)

    def test_geometry_deterministic(self):
        a=RegionPrediction.from_pixels([1,1,3,3],10,10,score=1,label="a",source="pred")
        b=RegionPrediction.from_pixels([7,1,9,3],10,10,score=1,label="b",source="pred")
        self.assertEqual(horizontal_relation("A",a,"B",b),horizontal_relation("A",a,"B",b))
        self.assertEqual(horizontal_relation("A",a,"B",b).relation,"left")

    def test_serialization_and_cache_reproducibility(self):
        a=RegionPrediction.from_pixels([1,1,3,3],10,10,score=.9,label="a",source="pred")
        b=RegionPrediction.from_pixels([7,1,9,3],10,10,score=.8,label="b",source="pred")
        state=VisualState("c","REL2C_LEFT",(EntityVisualState("A",a,.9,"apple",.9,"red",.8,.2,.2,("pred",)),EntityVisualState("B",b,.8,"ball",.8,"blue",.7,.8,.2,("pred",))),(horizontal_relation("A",a,"B",b),),(),{}, {})
        facts=compile_facts(state,{"A":"apple","B":"ball"}); self.assertEqual(serialize_facts(facts),serialize_facts(facts)); self.assertNotIn("answer is",serialize_facts(facts).casefold())
        with tempfile.TemporaryDirectory() as d:
            image=Path(d)/"x"; image.write_bytes(b"same")
            k1,f1=cache_key(image_path=image,processor="p",version="1",model_revision="m",config={"x":1})
            k2,f2=cache_key(image_path=image,processor="p",version="1",model_revision="m",config={"x":1})
            self.assertEqual((k1,f1),(k2,f2)); cache=JsonCache(Path(d)/"c"); cache.put(k1,{"fields":f1}); cache.put(k1,{"fields":f1})
            class Fake:
                name="fake";version="1"
                def __init__(self):self.calls=0
                def predict(self,path,**kwargs):
                    self.calls+=1;return [RegionPrediction.from_pixels([0,0,1,1],2,2,score=1,label="x",source="fake")]
            fake=Fake();wrapped=CachedRegionProcessor(fake,JsonCache(Path(d)/"regions"),"m")
            self.assertEqual(wrapped.predict(image,task="x"),wrapped.predict(image,task="x"));self.assertEqual(fake.calls,1)

    def test_paths_downloader_and_no_overwrite(self):
        env=ensure_d_drive_environment(Paths()); self.assertTrue(all(value.startswith("/mnt/d/vlm_repair_v2") for value in env.values()))
        source=(Paths().harness/"scripts/download_assets.py").read_text(); self.assertNotIn("/root/.cache",source)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x"; write_new(p,"a")
            with self.assertRaises(FileExistsError): write_new(p,"b")


if __name__ == "__main__": unittest.main()
