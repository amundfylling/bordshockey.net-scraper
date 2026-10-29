import difflib
import random
import unittest

from map_players import FuzzyMatcher


class FuzzyMatcherTests(unittest.TestCase):
    def test_matches_difflib_get_close_matches(self):
        rng = random.Random(1234)
        alphabet = "abcdefghijklmnopqrstuvwxyzåäöāēīšž -"
        names = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 20))).title()
            for _ in range(800)
        ]
        names += names[:50]  # duplicates in the candidate list
        names += ["Agris Pavārs", "Axel Lönnqvist", "Krisjanis Zemnickis"]
        matcher = FuzzyMatcher(names, cutoff=0.90)

        queries = ["Agris Pavars", "Axel Lonnqvist", "Krisjanis Zemnicki", "", "x", "Ω unknown chars"]
        for name in rng.sample(names, 300):
            chars = list(name)
            for _ in range(rng.randint(0, 3)):
                op = rng.choice("ids")
                pos = rng.randrange(len(chars) + 1)
                if op == "i":
                    chars.insert(pos, rng.choice(alphabet + "Ω"))
                elif chars and pos < len(chars):
                    if op == "d":
                        del chars[pos]
                    else:
                        chars[pos] = rng.choice(alphabet)
            queries.append("".join(chars))

        for cutoff in (0.6, 0.9):
            matcher.cutoff = cutoff
            for query in queries:
                self.assertEqual(
                    matcher.match(query),
                    difflib.get_close_matches(query, names, n=1, cutoff=cutoff),
                    query,
                )


if __name__ == "__main__":
    unittest.main()
