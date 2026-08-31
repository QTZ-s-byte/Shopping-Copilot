import unittest

from agent import ClarificationPolicy, IntentRouter, SessionState, StateMachine
from agent.types import IntentResult


PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 5.0,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort", "durability"],
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}


class StateMachineTest(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.sm = StateMachine()

    def session(self):
        return self.sm.new_session("dev_001", PROFILE)


class SessionSeedTests(StateMachineTest):
    def test_seeds_preference_tags(self):
        state = self.session()
        self.assertEqual(state.soft_preferences["preference_tags"], ["fit", "comfort", "durability"])
        self.assertEqual(state.user_profile["average_prior_rating"], 5.0)


class AddConditionTests(StateMachineTest):
    def test_add_single_hard_constraint(self):
        state = self.session()
        result = self.router.classify("I need a black running shoe", state)
        self.sm.apply(state, result, "I need a black running shoe", turn=1)
        self.assertEqual(state.hard_constraints["color"], ["black"])
        self.assertEqual(state.hard_constraints["category"], ["running shoes"])

    def test_add_accumulates_conditions(self):
        state = self.session()
        result = self.router.classify("I need a black shoe", state)
        self.sm.apply(state, result, "I need a black shoe", turn=1)
        result = self.router.classify("I want Nike", state)
        self.sm.apply(state, result, "I want Nike", turn=2)
        self.assertEqual(state.hard_constraints["color"], ["black"])
        self.assertIn("nike", state.hard_constraints["brand"])

    def test_add_budget_constraint(self):
        state = self.session()
        result = self.router.classify("My budget is under $100", state)
        self.sm.apply(state, result, "My budget is under $100", turn=1)
        self.assertEqual(state.hard_constraints["budget"], {"min": None, "max": 100.0})


class DeleteConditionTests(StateMachineTest):
    def test_delete_color(self):
        state = self.session()
        result = self.router.classify("I need a black shoe", state)
        self.sm.apply(state, result, "I need a black shoe", turn=1)
        result = self.router.classify("forget the color", state)
        diff = self.sm.apply(state, result, "forget the color", turn=2)
        self.assertNotIn("color", state.hard_constraints)
        self.assertIn("color", diff["removed"])

    def test_delete_budget(self):
        state = self.session()
        result = self.router.classify("My budget is under $100", state)
        self.sm.apply(state, result, "My budget is under $100", turn=1)
        result = self.router.classify("remove the budget", state)
        self.sm.apply(state, result, "remove the budget", turn=2)
        self.assertNotIn("budget", state.hard_constraints)

    def test_delete_brand(self):
        state = self.session()
        result = self.router.classify("I want Nike", state)
        self.sm.apply(state, result, "I want Nike", turn=1)
        result = self.router.classify("drop the brand", state)
        self.sm.apply(state, result, "drop the brand", turn=2)
        self.assertNotIn("brand", state.hard_constraints)


class OverrideConditionTests(StateMachineTest):
    def test_override_replaces_goal(self):
        state = self.session()
        result = self.router.classify("I need black running shoes", state)
        self.sm.apply(state, result, "I need black running shoes", turn=1)
        result = self.router.classify("Actually, I want a leather bag", state)
        diff = self.sm.apply(state, result, "Actually, I want a leather bag", turn=2)
        self.assertTrue(diff["override"])
        self.assertNotIn("running shoes", state.hard_constraints.get("category", []))
        self.assertEqual(state.hard_constraints["category"], ["bags"])
        self.assertIn("leather", state.hard_constraints.get("material", []))

    def test_override_resets_soft_preferences(self):
        state = self.session()
        result = self.router.classify("Show me some summer outfit ideas", state)
        self.sm.apply(state, result, "Show me some summer outfit ideas", turn=1)
        result = self.router.classify("Actually, I want a leather bag", state)
        self.sm.apply(state, result, "Actually, I want a leather bag", turn=2)
        self.assertNotIn("use_case", state.soft_preferences)

    def test_override_records_intent_change(self):
        state = self.session()
        result = self.router.classify("Show me some summer outfit ideas", state)
        self.sm.apply(state, result, "Show me some summer outfit ideas", turn=1)
        result = self.router.classify("Actually, I want a leather bag", state)
        diff = self.sm.apply(state, result, "Actually, I want a leather bag", turn=2)
        self.assertTrue(diff["override"])
        self.assertEqual(diff["intent"], {"from": "browsing", "to": "buying"})


class TurnLimitTests(StateMachineTest):
    def test_terminates_after_ten_turns(self):
        state = self.session()
        for turn in range(1, 11):
            result = self.router.classify("I need a black shoe", state)
            self.sm.apply(state, result, "I need a black shoe", turn=turn)
        self.assertEqual(state.turn_count, 10)
        self.assertTrue(state.terminated)

    def test_eleventh_turn_is_blocked(self):
        state = self.session()
        for turn in range(1, 11):
            result = self.router.classify("I need a black shoe", state)
            self.sm.apply(state, result, "I need a black shoe", turn=turn)
        result = self.router.classify("I need a blue shoe", state)
        diff = self.sm.apply(state, result, "I need a blue shoe", turn=11)
        self.assertEqual(state.turn_count, 10)
        self.assertTrue(diff["terminated"])


class NoPreferenceTests(StateMachineTest):
    def test_no_preference_marks_attribute(self):
        state = self.session()
        result = self.router.classify("no preference", state)
        self.sm.apply(state, result, "no preference", turn=1, asked_attribute="color")
        self.assertIn("color", state.no_preference_attributes)
        self.assertNotIn("color", state.hard_constraints)


class ClarificationTests(StateMachineTest):
    def setUp(self):
        super().setUp()
        self.policy = ClarificationPolicy()

    def test_clarify_missing_attribute(self):
        result = IntentResult(intent="buying", confidence=0.9)
        clarification = self.policy.decide(SessionState(), result)
        self.assertTrue(clarification.needed)
        self.assertEqual(clarification.ask_attribute, "category")

    def test_clarify_candidate_set_too_broad(self):
        result = IntentResult(
            intent="buying",
            confidence=0.9,
            hard_constraints={"category": ["shoes"]},
        )
        clarification = self.policy.decide(SessionState(), result, candidate_count=500)
        self.assertTrue(clarification.needed)
        self.assertEqual(clarification.reason, "candidate_set_too_broad")

    def test_clarify_vague_browsing(self):
        result = IntentResult(intent="browsing", confidence=0.9)
        clarification = self.policy.decide(SessionState(), result)
        self.assertTrue(clarification.needed)
        self.assertEqual(clarification.ask_attribute, "style")


if __name__ == "__main__":
    unittest.main()
