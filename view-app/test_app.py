"""Run with: python -m unittest discover -s view-app -p 'test_*.py'."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from app import Handler, ThreadingHTTPServer, child, discover, preview
from trace import parse_trace


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / 'agent' / 'run'
        self.run.mkdir(parents=True)
        self.trace = self.run / 'solve_out.txt'
        self.trace.touch()

    def tearDown(self):
        self.temp.cleanup()

    def parse(self, rows):
        self.trace.write_text('\n'.join(json.dumps(row) for row in rows))
        return parse_trace(self.trace)

    def test_codex_lifecycle_and_reset(self):
        rows = [dict(type='thread.started'),
                dict(type='item.started', item=dict(type='command_execution', id='0', command='pwd', status='in_progress')),
                dict(type='item.completed', item=dict(type='command_execution', id='0', command='pwd', aggregated_output='/work', exit_code=0)),
                dict(type='thread.started'),
                dict(type='item.completed', item=dict(type='agent_message', id='0', text='done'))]
        events = self.parse(rows)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[1]['title'], 'pwd')
        self.assertEqual(events[1]['body'], 'pwd')
        self.assertEqual(events[1]['output'], '/work')
        self.assertEqual(events[1]['status'], 'exit 0')
        self.assertEqual(events[-1]['body'], 'done')

    def test_claude_call_results(self):
        events = self.parse([
            {'type': 'assistant', 'message': {'role': 'assistant', 'content': [
                {'type': 'thinking', 'thinking': 'plan'},
                {'type': 'tool_use', 'id': 'a', 'name': 'Bash', 'input': {'command': 'false'}}]}},
            {'type': 'user', 'message': {'content': [
                {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'failed', 'is_error': True}]}}])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['kind'], 'reasoning')
        self.assertEqual(events[1]['title'], 'Bash')
        self.assertEqual(events[1]['output'], 'failed')
        self.assertEqual(events[1]['status'], 'error')

    def test_gemini_and_opencode(self):
        events = self.parse([
            {'type': 'tool_use', 'tool_id': 'g', 'tool_name': 'shell', 'parameters': {'command': 'pwd'}},
            {'type': 'tool_result', 'tool_id': 'g', 'output': '/work', 'status': 'success'},
            {'type': 'tool_use', 'part': {'callID': 'o', 'tool': 'bash', 'state': {'input': {'command': 'ls'}, 'output': 'file', 'status': 'completed'}}}])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['title'], 'shell')
        self.assertEqual(events[1]['output'], 'file')

    def test_cursor_repr_and_legacy_codex(self):
        events = self.parse([
            {'type': 'tool_call', 'subtype': 'started', 'call_id': 'c\nfc_a',
             'tool_call': "{'shellToolCall': {'args': {'command': 'pwd'}}}"},
            {'type': 'tool_call', 'subtype': 'completed', 'call_id': 'c\nfc_b',
             'tool_call': {'shellToolCall': {'result': '/app'}}},
            {'msg': {'type': 'exec_command_begin', 'call_id': 'old', 'command': 'ls'}},
            {'msg': {'type': 'exec_command_end', 'call_id': 'old', 'stdout': 'file', 'exit_code': 1}}])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['output'], '/app')
        self.assertEqual(events[1]['title'], 'ls')
        self.assertEqual(events[1]['body'], 'ls')
        self.assertEqual(events[1]['status'], 'exit 1')

    def test_timestamp_and_malformed(self):
        self.trace.write_text('preamble\n[2026-10-18T10:00:00Z] {"type":"item.completed","item":{"type":"reasoning","text":"plan"}}\n{"partial":')
        events = parse_trace(self.trace)
        self.assertEqual([e['kind'] for e in events], ['log', 'reasoning', 'log'])
        self.assertEqual(events[1]['timestamp'], '2026-10-18T10:00:00Z')

    def test_nested_trajectory(self):
        path = self.run / 'trajectory.json'
        path.write_text(json.dumps({'messages': [
            {'role': 'assistant', 'tool_calls': [{'id': 'x', 'function': {'name': 'shell', 'arguments': '{"cmd":"pwd"}'}}]},
            {'role': 'tool', 'tool_call_id': 'x', 'content': '/app'}]}))
        events = parse_trace(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['output'], '/app')

    def test_discovery_stops_at_run(self):
        nested = self.run / 'task'
        nested.mkdir()
        (nested / 'metrics.json').write_text('{}')
        self.assertEqual([x['id'] for x in discover(self.root)], ['agent/run'])
        self.assertEqual(discover(self.run)[0]['id'], '.')

    def test_symlink_boundary(self):
        (self.run / 'task').mkdir()
        (self.run / 'internal').symlink_to(self.run / 'task', target_is_directory=True)
        (self.run / 'outside').symlink_to(self.root)
        self.assertEqual(child(self.run, 'internal'), self.run / 'task')
        for path in ('../', 'outside', '/etc/passwd'):
            with self.assertRaises(PermissionError):
                child(self.run, path)

    def test_discovery_includes_nested_reevaluation(self):
        rerun = self.run / 'reeval_123'
        rerun.mkdir()
        (rerun / 'run_config.json').write_text('{"evaluation_only": true}')
        (self.run / 'reeval_external').symlink_to(self.root, target_is_directory=True)
        self.assertEqual({item['id'] for item in discover(self.root)},
                         {'agent/run', 'agent/run/reeval_123'})
        self.assertEqual({item['id'] for item in discover(self.run)}, {'.', 'reeval_123'})

    def test_preview_is_bounded_and_binary(self):
        path = self.run / 'large.log'
        path.write_bytes(b'a' * 300000)
        data = preview(path)
        self.assertTrue(data['more'])
        self.assertEqual(data['next_offset'], 262144)
        self.assertFalse(preview(path, data['next_offset'])['more'])
        path.write_bytes(b'\0abc')
        self.assertTrue(preview(path)['binary'])

    def test_http_paging_errors_and_traversal(self):
        self.parse([{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': str(i)}} for i in range(105)])
        class TestHandler(Handler):
            root = self.root
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/api/trace?run=agent/run&path=solve_out.txt&offset=100') as response:
                data = json.load(response)
                self.assertEqual(len(data['events']), 5)
                self.assertEqual(data['total'], 105)
            for endpoint, status in [('/api/file?run=agent/run&path=../../secret', 403),
                                     ('/api/file?run=agent/run&path=missing', 404),
                                     ('/api/file?run=agent/run&path=solve_out.txt&offset=oops', 400)]:
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + endpoint)
                self.assertEqual(caught.exception.code, status)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
