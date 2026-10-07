"""API tests (skipped automatically if FastAPI/httpx are not installed)."""

import importlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_FASTAPI = all(importlib.util.find_spec(m) is not None for m in ("fastapi", "httpx"))


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["RAGQA_INDEX_DIR"] = cls._tmp.name
        os.environ["RAGQA_DATA_DIR"] = str(ROOT / "data")
        import ragqa.api as api

        importlib.reload(api)  # pick up the environment above
        from fastapi.testclient import TestClient

        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_flow(self):
        health = self.client.get("/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertFalse(health["index_ready"])
        self.assertEqual(self.client.post("/ask", json={"question": "What is RRT?"}).status_code, 409)

        r = self.client.post("/ingest", json={"paths": ["sample"]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["documents"], 8)

        r = self.client.post("/ask", json={"question": "What is time-to-collision?", "k": 3})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["abstained"])
        self.assertEqual(len(body["sources"]), 3)
        self.assertIn("lane_change.md", body["sources"][0]["citation"])

        r = self.client.post("/ask", json={"question": "How many moons does Jupiter have?"})
        self.assertTrue(r.json()["abstained"])

    def test_rejects_paths_outside_data_dir(self):
        r = self.client.post("/ingest", json={"paths": ["../README.md"]})
        self.assertEqual(r.status_code, 400)

    def test_validation(self):
        self.assertEqual(self.client.post("/ask", json={"question": "x"}).status_code, 422)
        r = self.client.post("/ingest", json={"paths": ["sample"], "chunk_size": 50, "overlap": 50})
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
