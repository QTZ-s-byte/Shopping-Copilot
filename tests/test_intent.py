import unittest

from agent import IntentRouter, SessionState


class IntentRouterTest(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()

    def classify(self, message):
        return self.router.classify(message, SessionState())


class BuyingIntentTests(IntentRouterTest):
    def test_buying_budget_constraint(self):
        result = self.classify("I need a black running shoe under $80")
        self.assertEqual(result.intent, "buying")
        self.assertEqual(result.hard_constraints["budget"], {"min": None, "max": 80.0})

    def test_buying_brand(self):
        result = self.classify("I want to buy a Nike jacket")
        self.assertEqual(result.intent, "buying")
        self.assertIn("nike", result.hard_constraints.get("brand", []))

    def test_buying_looking_for(self):
        result = self.classify("Looking for a waterproof rain jacket for hiking")
        self.assertEqual(result.intent, "buying")
        self.assertIn("waterproof", result.soft_preferences.get("feature", []))

    def test_buying_color_and_material(self):
        result = self.classify("I want a red leather handbag")
        self.assertEqual(result.intent, "buying")
        self.assertIn("red", result.hard_constraints.get("color", []))
        self.assertIn("leather", result.hard_constraints.get("material", []))

    def test_buying_size(self):
        result = self.classify("Find me size 9 blue sneakers")
        self.assertEqual(result.intent, "buying")
        self.assertIn("9", result.hard_constraints.get("size", []))
        self.assertIn("blue", result.hard_constraints.get("color", []))


class BrowsingIntentTests(IntentRouterTest):
    def test_browsing_outfit_ideas(self):
        result = self.classify("Show me some summer outfit ideas")
        self.assertEqual(result.intent, "browsing")

    def test_browsing_trending(self):
        result = self.classify("What's trending in jewelry?")
        self.assertEqual(result.intent, "browsing")

    def test_browsing_recommend_gift(self):
        result = self.classify("Recommend something for a birthday gift")
        self.assertEqual(result.intent, "browsing")

    def test_browsing_just_looking(self):
        result = self.classify("I'm just looking for inspiration")
        self.assertEqual(result.intent, "browsing")

    def test_browsing_explore(self):
        result = self.classify("Explore gift ideas for my girlfriend")
        self.assertEqual(result.intent, "browsing")


class IntentOverrideTests(IntentRouterTest):
    def test_override_forget_old_goal(self):
        result = self.classify("Actually, forget shoes. I want a leather bag")
        self.assertTrue(result.override)
        self.assertEqual(result.intent, "buying")

    def test_override_never_mind(self):
        result = self.classify("Never mind the jacket, let's look at shoes instead")
        self.assertTrue(result.override)

    def test_override_second_thought(self):
        result = self.classify("On second thought, I'd rather have a backpack")
        self.assertTrue(result.override)


if __name__ == "__main__":
    unittest.main()
