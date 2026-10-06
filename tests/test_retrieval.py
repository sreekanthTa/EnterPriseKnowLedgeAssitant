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


class RetrievalValidationTests(unittest.TestCase):
    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            retrieval._validate_query("  ")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            retrieval._validate_top_k(0)

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


class LazyImportTests(unittest.TestCase):
    def test_import_does_not_initialize_external_clients(self):
        self.assertIsNone(retrieval.index)
        self.assertIsNone(retrieval.client)
        self.assertIsNone(retrieval.bm25_retriever)
        self.assertIsNone(retrieval.lm)


if __name__ == "__main__":
    unittest.main()
