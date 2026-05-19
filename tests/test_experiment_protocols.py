import unittest

from HCC_SRC.experiment_protocols import resolve_protocol


class ExperimentProtocolTests(unittest.TestCase):
    def test_smoke_protocol_matches_lightweight_budget(self):
        protocol = resolve_protocol("smoke")

        self.assertEqual(protocol["name"], "smoke")
        self.assertEqual(protocol["max_fes"], 2000)
        self.assertEqual(protocol["cycle_num"], 5)
        self.assertEqual(protocol["record_fes"], [200, 500, 1000, 1500, 2000])

    def test_paper_protocol_matches_paper_budget(self):
        protocol = resolve_protocol("paper")

        self.assertEqual(protocol["name"], "paper")
        self.assertEqual(protocol["max_fes"], 3_000_000)
        self.assertEqual(protocol["cycle_num"], 25)
        self.assertEqual(protocol["record_fes"], [120_000, 200_000, 1_000_000, 2_000_000, 3_000_000])

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_protocol("mystery")
