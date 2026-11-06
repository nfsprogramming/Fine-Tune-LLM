import unittest
from scripts.generate_dataset import (
    generate_bug_fix_scenario,
    generate_file_write_scenario,
    generate_git_scenario
)

class TestDatasetGenerator(unittest.TestCase):
    def _validate_scenario(self, scenario):
        self.assertIn("messages", scenario)
        messages = scenario["messages"]
        self.assertGreaterEqual(len(messages), 2)
        roles = [m["role"] for m in messages]
        self.assertIn("system", roles)
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        for m in messages:
            self.assertTrue(len(m["content"]) > 0)

    def test_generate_bug_fix_scenario(self):
        scenario = generate_bug_fix_scenario()
        self._validate_scenario(scenario)
        assistant_content = scenario["messages"][-1]["content"]
        self.assertIn("<thought>", assistant_content)
        self.assertIn("read_file", assistant_content)

    def test_generate_file_write_scenario(self):
        scenario = generate_file_write_scenario()
        self._validate_scenario(scenario)
        assistant_content = scenario["messages"][-1]["content"]
        self.assertIn("<thought>", assistant_content)
        self.assertIn("```", assistant_content)

    def test_generate_git_scenario(self):
        scenario = generate_git_scenario()
        self._validate_scenario(scenario)
        assistant_content = scenario["messages"][-1]["content"]
        self.assertIn("<thought>", assistant_content)
        self.assertIn("git_tool", assistant_content)

if __name__ == "__main__":
    unittest.main()
