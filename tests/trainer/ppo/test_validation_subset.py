import importlib.util
from pathlib import Path
import unittest
from collections import Counter


# Exercise the sampler without importing the GPU training stack.
spec = importlib.util.spec_from_file_location(
    "validation_subset",
    Path(__file__).resolve().parents[3] / "verl/trainer/ppo/validation_subset.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
validation_indices = module.validation_indices
should_validate = module.should_validate


class ValidationSubsetTest(unittest.TestCase):
    def test_science_schedule_without_initial_or_near_final_duplicate(self):
        steps = [step for step in range(94) if should_validate(step, 93, 23, True)]
        self.assertEqual(steps, [23, 46, 69, 93])

    def test_default_schedule_and_disabled_validation(self):
        self.assertEqual(
            [step for step in range(94) if should_validate(step, 93, 23)],
            [23, 46, 69, 92, 93],
        )
        self.assertFalse(should_validate(93, 93, 0, True))
        self.assertTrue(should_validate(10, 10, 100, True))
        self.assertEqual(
            [step for step in range(101) if should_validate(step, 100, 25, True)],
            [25, 50, 75, 100],
        )

    def test_stratification_and_rounding(self):
        sources = ["math"] * 60 + ["code"] * 400 + ["tool"] * 68 + ["tiny"]
        indices = validation_indices(sources)
        self.assertEqual(Counter(sources[i] for i in indices), {"math": 6, "code": 40, "tool": 7, "tiny": 1})
        self.assertEqual(indices, sorted(set(indices)))

    def test_fixed_subset_independent_of_global_randomness(self):
        import random

        sources = ["science"] * 100
        first = validation_indices(sources)
        random.seed(999)
        self.assertEqual(first, validation_indices(sources))
        self.assertNotEqual(first, validation_indices(sources, seed=43))

    def test_full_standalone_evaluation(self):
        self.assertEqual(validation_indices(["a"] * 20, val_only=True), list(range(20)))
        self.assertEqual(validation_indices(["a"] * 20, fraction=1), list(range(20)))

    def test_invalid_fraction(self):
        for fraction in (0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                validation_indices(["a"], fraction=fraction)


if __name__ == "__main__":
    unittest.main()
