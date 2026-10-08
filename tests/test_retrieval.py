import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_PATH = PROJECT_ROOT / "02.retrieval.py"


spec = importlib.util.spec_from_file_location("retrieval", RETRIEVAL_PATH)
retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval)


class DeploymentSecretTests(unittest.TestCase):
    def test_streamlit_secrets_populate_environment(self):
        fake_streamlit = SimpleNamespace(
            secrets={
                "GROQ_API_KEY": "groq-test",
                "pinecone_api_key": "pinecone-test",
            }
        )
        with patch.dict("sys.modules", {"streamlit": fake_streamlit}), patch.dict(
            "os.environ", {}, clear=True
        ), patch.object(retrieval, "load_dotenv"):
            retrieval._load_deployment_secrets()
            self.assertEqual(retrieval.os.environ["GROQ_API_KEY"], "groq-test")
            self.assertEqual(
                retrieval.os.environ["PINECONE_API_KEY"], "pinecone-test"
            )


class RetrievalValidationTests(unittest.TestCase):
    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrieval._validate_query("  ")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            retrieval._validate_top_k(0)

    def test_oversized_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "2000"):
            retrieval._validate_query("x" * 2001)

    def test_oversized_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "maximum"):
            retrieval._validate_conversation_history("x" * 12001)

    def test_invalid_chunk_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            retrieval.create_chunks("one two", chunk_size=2, overlap=2)

    def test_valid_chunking_preserves_overlap(self):
        chunks = retrieval.create_chunks(
            "one two three four five",
            chunk_size=3,
            overlap=1,
        )
        self.assertEqual(
            chunks,
            ["one two three", "three four five", "five"],
        )


class GroundingTests(unittest.TestCase):
    def test_supported_answer_is_grounded(self):
        self.assertTrue(
            retrieval._grounded_answer(
                "Attention uses queries and keys.",
                "Attention uses queries, keys, and values.",
            )
        )

    def test_unrelated_answer_is_not_grounded(self):
        self.assertFalse(
            retrieval._grounded_answer(
                "Paris is the capital of France.",
                "Attention uses queries, keys, and values.",
            )
        )


class RetrievalCombinationTests(unittest.TestCase):
    def test_rrf_combines_duplicate_results(self):
        semantic = [{
            "chunk_id": "chunk_1",
            "chunk": "semantic",
            "rank": 1,
            "similarity_score": 0.9,
        }]
        keyword = [{
            "chunk_id": "chunk_1",
            "chunk": "keyword",
            "rank": 1,
        }]

        results = retrieval.combine_results(semantic, keyword, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chunk_1")
        self.assertEqual(results[0]["semantic_rank"], 1)
        self.assertEqual(results[0]["keyword_rank"], 1)
        self.assertGreater(results[0]["rrf_score"], 0)


class PineconeMetadataTests(unittest.TestCase):
    def test_malformed_matches_are_skipped(self):
        matches = [
            SimpleNamespace(id="missing-text", metadata={}, score=0.9),
            SimpleNamespace(id="valid", metadata={"text": "valid chunk"}, score=0.8),
            SimpleNamespace(id="", metadata={"text": "ignored"}, score=0.7),
        ]
        index = SimpleNamespace(
            query=lambda **kwargs: SimpleNamespace(matches=matches)
        )

        with patch.object(retrieval, "initialize_pipeline"), patch.object(
            retrieval, "index", index
        ):
            results = retrieval.get_chunks_by_similarity([0.1, 0.2], top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "valid")
        self.assertEqual(results[0]["chunk"], "valid chunk")


class DirectResponseTests(unittest.TestCase):
    def test_direct_response_skips_retrieval(self):
        rewrite = SimpleNamespace(
            intent="direct_response",
            re_written_query="",
            direct_response="Hello! How can I help?",
        )

        with patch.object(retrieval, "initialize_pipeline"), patch.object(
            retrieval, "query_re_writer", return_value=rewrite
        ), patch.object(
            retrieval, "generate_embedding", side_effect=AssertionError("retrieval called")
        ):
            results = retrieval.retrieve_relevant_chunks("hi")

        self.assertEqual(results, [])
        self.assertEqual(results.direct_response, "Hello! How can I help?")


class HealthApiTests(unittest.TestCase):
    def test_liveness_is_available_without_dependencies(self):
        api_spec = importlib.util.spec_from_file_location(
            "health_api", PROJECT_ROOT / "api.py"
        )
        health_api = importlib.util.module_from_spec(api_spec)
        api_spec.loader.exec_module(health_api)

        response = health_api.liveness()

        self.assertEqual(response, {"status": "ok"})

    def test_readiness_returns_service_unavailable_when_unhealthy(self):
        api_spec = importlib.util.spec_from_file_location(
            "health_api", PROJECT_ROOT / "api.py"
        )
        health_api = importlib.util.module_from_spec(api_spec)
        api_spec.loader.exec_module(health_api)

        with patch.object(
            health_api,
            "load_pipeline_module",
            return_value=SimpleNamespace(
                health_check=lambda: {"ok": False, "pinecone": {"ok": False}}
            ),
        ):
            response = health_api.readiness()

        self.assertEqual(response.status_code, 503)


class LazyImportTests(unittest.TestCase):
    def test_import_does_not_initialize_external_clients(self):
        self.assertIsNone(retrieval.index)
        self.assertIsNone(retrieval.embedding_model)
        self.assertIsNone(retrieval.bm25_retriever)
        self.assertIsNone(retrieval.lm)


if __name__ == "__main__":
    unittest.main()
