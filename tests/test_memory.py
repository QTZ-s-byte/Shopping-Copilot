from __future__ import annotations

import unittest

from shopping_copilot.contracts import IntentResult
from shopping_copilot.memory import InMemoryContextMemory


class ContextMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = InMemoryContextMemory(history_window=2)
        self.memory.reset("s1", {"preference_tags": ["comfort"]})

    def test_apply_turn_accumulates_and_summarizes(self) -> None:
        diff = self.memory.apply_turn(
            "s1",
            "black shoes",
            IntentResult(
                intent="buying",
                hard_constraints={"category": "shoes"},
                soft_preferences={"color": "black"},
            ),
            expected_turn=1,
        )
        state = self.memory.get("s1")
        self.assertEqual(state.turn_count, 1)
        self.assertEqual(state.intent, "buying")
        self.assertEqual(state.hard_constraints["category"], "shoes")
        self.assertEqual(state.soft_preferences["color"], "black")
        self.assertEqual(diff.intent_before, None)
        self.assertEqual(diff.intent_after, "buying")
        self.assertIn("intent=buying", state.summary)

    def test_remove_and_replace_are_explicit(self) -> None:
        self.memory.apply_turn(
            "s1",
            "initial",
            IntentResult(intent="buying", hard_constraints={"category": "shoes", "color": "black"}),
            expected_turn=1,
        )
        diff = self.memory.apply_turn(
            "s1",
            "actually a red bag",
            IntentResult(
                intent="buying",
                replace_fields={"category": "bag"},
                remove_fields=("color",),
                soft_preferences={"color": "red"},
            ),
            expected_turn=2,
        )
        state = self.memory.get("s1")
        self.assertEqual(state.hard_constraints["category"], "bag")
        self.assertNotIn("color", state.hard_constraints)
        self.assertEqual(state.soft_preferences["color"], "red")
        self.assertTrue(diff.removed)
        self.assertTrue(diff.replaced)

    def test_turn_boundary_and_snapshot_restore(self) -> None:
        for turn in range(1, 11):
            self.memory.apply_turn(
                "s1", str(turn), IntentResult(intent="browsing"), expected_turn=turn
            )
        with self.assertRaises(RuntimeError):
            self.memory.apply_turn("s1", "11", IntentResult(), expected_turn=11)
        snapshot = self.memory.snapshot("s1")
        snapshot.turn_count = 3
        self.memory.restore("s1", snapshot)
        self.assertEqual(self.memory.get("s1").turn_count, 3)

    def test_history_window_is_bounded(self) -> None:
        for turn in range(1, 4):
            self.memory.apply_turn(
                "s1", f"message-{turn}", IntentResult(intent="browsing"), expected_turn=turn
            )
        self.assertEqual(self.memory.get("s1").recent_messages, ["message-2", "message-3"])


if __name__ == "__main__":
    unittest.main()

