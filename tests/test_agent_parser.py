import unittest
from src.agent import parse_tool_call

class TestAgentParser(unittest.TestCase):
    def test_parse_markdown_file_write(self):
        text = "I will write the file now.\nmain.py:\n```python\nprint('hello')\n```\n"
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.get("name"), "write_file")
        self.assertEqual(call["arguments"]["filepath"], "main.py")
        self.assertEqual(call["arguments"]["content"], "print('hello')")

    def test_parse_tool_call_tag(self):
        text = 'Here is the call: <tool_call>{"name": "git_tool", "arguments": {"command": "status"}}</tool_call>'
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.get("name"), "git_tool")
        self.assertEqual(call["arguments"]["command"], "status")

    def test_parse_markdown_json_block(self):
        text = '```json\n{"name": "read_file", "arguments": {"filepath": "config.json"}}\n```'
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.get("name"), "read_file")
        self.assertEqual(call["arguments"]["filepath"], "config.json")

    def test_parse_qwen_native_format(self):
        text = 'execute_terminal\n{"command": "pytest"}'
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.get("name"), "execute_terminal")
        self.assertEqual(call["arguments"]["command"], "pytest")

    def test_parse_raw_code_fallback(self):
        text = "Here is the HTML file to save as index.html:\n```html\n<h1>App</h1>\n```"
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.get("name"), "write_file")
        self.assertIn("App", call["arguments"]["content"])

    def test_parse_no_tool_call(self):
        text = "The task is finished and tests are passing cleanly."
        call = parse_tool_call(text)
        self.assertIsNone(call)

    def test_parse_invalid_json(self):
        text = '<tool_call>{"name": "test", broken json</tool_call>'
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertIn("error", call)

if __name__ == "__main__":
    unittest.main()
