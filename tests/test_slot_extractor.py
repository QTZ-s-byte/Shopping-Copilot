import unittest

from agent import SlotExtractor


class SlotExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = SlotExtractor()

    def extract(self, message):
        return self.extractor.extract(message)

    def test_budget_under(self):
        slots = self.extract("I want something under $80")
        self.assertEqual(slots.budget_max, 80.0)
        self.assertIsNone(slots.budget_min)

    def test_budget_between(self):
        slots = self.extract("between $20 and $50")
        self.assertEqual(slots.budget_min, 20.0)
        self.assertEqual(slots.budget_max, 50.0)

    def test_budget_bare(self):
        slots = self.extract("something for $30")
        self.assertEqual(slots.budget_max, 30.0)

    def test_budget_dollars_word(self):
        slots = self.extract("under 100 dollars")
        self.assertEqual(slots.budget_max, 100.0)

    def test_color_and_material(self):
        slots = self.extract("black leather")
        self.assertIn("black", slots.color)
        self.assertIn("leather", slots.material)

    def test_brand(self):
        slots = self.extract("Nike sneakers")
        self.assertIn("nike", slots.brand)

    def test_category(self):
        slots = self.extract("running shoes")
        self.assertIn("running shoes", slots.category)

    def test_size(self):
        slots = self.extract("size 9")
        self.assertIn("9", slots.size)

    def test_style(self):
        slots = self.extract("a casual style")
        self.assertIn("casual", slots.style)

    def test_feature(self):
        slots = self.extract("something waterproof")
        self.assertIn("waterproof", slots.feature)

    def test_use_case(self):
        slots = self.extract("something for hiking")
        self.assertIn("hiking", slots.use_case)

    def test_negation_not_leather(self):
        slots = self.extract("not leather")
        self.assertIn("leather", slots.negative.get("material", []))
        self.assertNotIn("leather", slots.material)

    def test_discard_forget(self):
        slots = self.extract("forget shoes, I want a bag")
        self.assertNotIn("shoes", slots.category)
        self.assertIn("bags", slots.category)

    def test_negative_keywords_phrase(self):
        slots = self.extract("avoid bright colors")
        self.assertTrue(any("bright" in keyword for keyword in slots.negative_keywords))


if __name__ == "__main__":
    unittest.main()
