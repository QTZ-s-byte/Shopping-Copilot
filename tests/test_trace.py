from __future__ import annotations

import unittest

from shopping_copilot.trace import redact


class TraceTests(unittest.TestCase):
    def test_secrets_are_redacted_but_usage_counts_are_preserved(self) -> None:
        value = redact(
            {
                "api_key": "secret-value",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        )
        self.assertEqual(value["api_key"], "[REDACTED]")
        self.assertEqual(value["prompt_tokens"], 10)
        self.assertEqual(value["completion_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
