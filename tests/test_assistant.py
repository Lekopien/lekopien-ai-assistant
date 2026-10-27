import unittest

from assistant import Assistant
from assistant.naive_bayes import NaiveBayes
from assistant.text import tokenize
from assistant.tfidf import TfidfIndex


class TextTests(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_stems(self):
        self.assertEqual(tokenize("Fixing the broken sites"), ["fix", "broken", "site"])


class TfidfTests(unittest.TestCase):
    def test_finds_most_similar_document(self):
        idx = TfidfIndex(["cats purr softly", "dogs bark loudly", "fish swim"])
        (best, score), *_ = idx.search("why do dogs bark")
        self.assertEqual(best, 1)
        self.assertGreater(score, 0)


class NaiveBayesTests(unittest.TestCase):
    def test_learns_simple_classes(self):
        nb = NaiveBayes().fit([("great awesome love", "pos"), ("terrible awful hate", "neg")])
        self.assertEqual(nb.predict("i love it, awesome")[0], "pos")
        probs = nb.predict_proba("hate")
        self.assertAlmostEqual(sum(probs.values()), 1.0)


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.bot = Assistant()

    def test_answers_known_intents(self):
        cases = {
            "my website keeps crashing and is broken": "fix_site",
            "how much would a website cost me?": "pricing",
            "can you build a new site for my shop": "new_site",
        }
        for msg, intent in cases.items():
            with self.subTest(msg=msg):
                self.assertEqual(self.bot.ask(msg).intent, intent)

    def test_unknown_question_falls_back(self):
        self.assertEqual(self.bot.ask("quantum chromodynamics lattice").source, "fallback")

    def test_history_is_recorded(self):
        self.bot.ask("hello")
        self.assertEqual(len(self.bot.history), 1)


if __name__ == "__main__":
    unittest.main()
