import os
import tempfile
import unittest
from src.tools import read_file, write_file, git_tool, get_tool_prompt, AVAILABLE_TOOLS, TOOL_DESCRIPTIONS

class TestTools(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_write_and_read_file(self):
        filepath = os.path.join(self.test_dir.name, "subdir", "test.txt")
        content = "Hello, Autonomous Agent!"
        
        # Write
        result = write_file(filepath, content)
        self.assertIn("Successfully wrote to", result)
        self.assertTrue(os.path.exists(filepath))
        
        # Read
        read_content = read_file(filepath)
        self.assertEqual(read_content, content)

    def test_read_nonexistent_file(self):
        filepath = os.path.join(self.test_dir.name, "nonexistent.txt")
        result = read_file(filepath)
        self.assertTrue(result.startswith("Error reading file"))

    def test_tool_prompt_contains_persona_and_tools(self):
        prompt = get_tool_prompt("Reviewer")
        self.assertIn("Reviewer", prompt)
        for tool in TOOL_DESCRIPTIONS:
            self.assertIn(tool["name"], prompt)

    def test_available_tools_mapping(self):
        expected_keys = {"read_file", "write_file", "execute_terminal", "git_tool", "test_runner"}
        self.assertEqual(set(AVAILABLE_TOOLS.keys()), expected_keys)

if __name__ == "__main__":
    unittest.main()
