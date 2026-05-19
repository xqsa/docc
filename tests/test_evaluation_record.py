import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from HCC_SRC.AOB.utils import evaluation_record


class EvaluationRecordTests(unittest.TestCase):
    def test_record_point_equal_to_available_evaluations_is_recorded(self):
        data = {
            "demo_1": [[9.0, 7.0, 5.0]],
            "demo_1_time": [0.25],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                evaluation_record(data, output_path, record_FEs_list=[3])
            contents = Path(output_path, "evaluation_record.txt").read_text()
            console = stdout.getvalue()

        self.assertNotIn("Warning: Record point 3 exceeds", console)
        self.assertIn("demo_1", contents)
        self.assertIn("3.000e+00", contents)
        self.assertIn("5.000000", contents)
        self.assertIn("Run Time:", contents)


if __name__ == "__main__":
    unittest.main()
