"""API tests (skipped automatically if FastAPI is not installed)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ["RAGQA_INDEX_DIR"] = cls._tmp.name
        import importlib

        import ragqa.api as api

        importlib.reload(api)  # pick up the temporary index directory
        from fastapi.testclient import TestClient

        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_flow(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.post("/ask", json={"question": "What is RRT?"}).status_code, 409)

        r = self.client.post("/ingest", json={"paths": [str(ROOT / "data" / "sample")]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["documents"], 4)

        r = self.client.post("/ask", json={"question": "What is time-to-collision?", "k": 3})
        body = r.json()
        self.assertEqual(r.status_code, 200)
        self.assertFalse(body["abstained"])
        self.assertEqual(len(body["sources"]), 3)

    def test_validation(self):
        self.assertEqual(self.client.post("/ask", json={"question": "x"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
