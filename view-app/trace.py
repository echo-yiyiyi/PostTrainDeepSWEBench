"""Normalize saved CLI events without executing trace content."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

STAMP = re.compile(r'^\[([^\]]+)\]\s*')


def pretty(value: Any) -> str:
    if isinstance(value, str):
        try:
            return json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        except (ValueError, TypeError):
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def parse_trace(path: Path) -> list[dict]:
    events: list[dict] = []
    calls: dict[str, dict] = {}
    session = 0

    def add(kind: str, title: str, body: Any = '', *, key: str = '', output: Any = None,
            status: Any = '', raw: Any = None, timestamp: Any = '', line: int = 0) -> dict:
        scoped = f'{session}:{key}' if key else ''
        entry = calls.get(scoped)
        if entry is None:
            entry = dict(kind=kind, title=title, body='', output='', status='', timestamp=timestamp,
                         line=line, raw=[])
            events.append(entry)
            if scoped:
                calls[scoped] = entry
        if title and title not in ('tool', 'tool_result', 'exec_command', 'mcp_tool_call', 'patch_apply'):
            entry['title'] = title
        if body != '' and body is not None:
            entry['body'] = pretty(body)
        if output is not None:
            entry['output'] = pretty(output)
        if status != '' and status is not None:
            entry['status'] = str(status)
        if raw is not None:
            entry['raw'].append(raw)
        return entry

    def process(data: Any, line: int, timestamp: str = '') -> None:
        nonlocal session
        if not isinstance(data, dict):
            add('system', 'JSON', data, line=line)
            return
        t = data.get('type', data.get('role', 'event'))
        timestamp = data.get('timestamp', timestamp)
        if t in ('thread.started', 'session_configured'):
            session += 1
        meta = dict(raw=data, timestamp=timestamp, line=line)
        item = data.get('item')
        if isinstance(item, dict):
            it = item.get('type', 'item')
            tool = it in ('command_execution', 'mcp_tool_call', 'file_change', 'web_search')
            add('tool' if tool else ('reasoning' if it == 'reasoning' else 'assistant'),
                item.get('command') or item.get('tool') or it,
                item.get('arguments', item.get('text', item.get('changes', item.get('query', item.get('command', ''))))),
                key='item:' + str(item['id']) if item.get('id') else '',
                output=item.get('aggregated_output', item.get('result')),
                status=f"exit {item['exit_code']}" if item.get('exit_code') is not None else item.get('status', t), **meta)
            return
        msg = data.get('msg', data.get('payload', data))
        if not isinstance(msg, dict):
            msg = data
        mt = msg.get('type', t)
        if mt.startswith(('exec_command_', 'mcp_tool_call_', 'patch_apply_')):
            entry = add('tool', msg.get('command') or msg.get('tool_name') or mt.split('_begin')[0].split('_end')[0],
                msg.get('arguments', msg.get('patch', msg.get('command', ''))), key=str(msg.get('call_id', '')),
                output=msg.get('aggregated_output', msg.get('stdout', msg.get('result'))),
                status=f"exit {msg['exit_code']}" if msg.get('exit_code') is not None else mt, **meta)
            if msg.get('chunk'):
                entry['output'] += str(msg['chunk'])
            if msg.get('stderr'):
                entry['output'] += '\n' + msg['stderr']
            return
        if mt in ('function_call', 'function_call_output'):
            add('tool', msg.get('name', 'tool_result'), msg.get('arguments', ''),
                output=msg.get('output'), key=str(msg.get('call_id', '')), **meta)
            return
        if 'part' in data and isinstance(data['part'], dict):
            part = data['part']
            state = part.get('state', {})
            add('tool' if t == 'tool_use' else 'assistant', part.get('tool', t),
                state.get('input', part.get('text', part)), output=state.get('output', state.get('error')),
                status=state.get('status', ''), key=str(part.get('callID', part.get('id', ''))), **meta)
            return
        if t == 'tool_call':
            call = data.get('tool_call', {})
            if isinstance(call, str):
                try:
                    call = ast.literal_eval(call)
                except (ValueError, SyntaxError):
                    call = {'tool': {'args': call}}
            if isinstance(call, dict):
                for name, inner in call.items():
                    if isinstance(inner, dict):
                        add('tool', name, inner.get('args', ''),
                            output=inner.get('result', inner.get('output', inner.get('response'))),
                            key=str(data.get('call_id', '')).split('\n')[0],
                            status=data.get('subtype', ''), **meta)
                return
        message = data.get('message', msg)
        if isinstance(message, str):
            try:
                message = ast.literal_eval(message)
            except (ValueError, SyntaxError):
                pass
        if isinstance(message, dict) and isinstance(message.get('content'), list):
            for block in message['content']:
                if not isinstance(block, dict):
                    add('system', 'content', block, **meta)
                    continue
                bt = block.get('type', '')
                if bt in ('tool_use', 'tool_result'):
                    add('tool', block.get('name', 'tool_result'), block.get('input', ''),
                        output=block.get('content'), key=str(block.get('tool_use_id', block.get('id', ''))),
                        status='error' if block.get('is_error') else '', **meta)
                else:
                    kind = 'reasoning' if bt in ('thinking', 'reasoning') else message.get('role', t)
                    add(kind, kind, block.get('text', block.get('thinking', block)), **meta)
            return
        if t in ('tool_use', 'tool_result'):
            add('tool', data.get('tool_name', 'tool_result'), data.get('parameters', ''),
                output=data.get('output'), key=str(data.get('tool_id', '')), status=data.get('status', ''), **meta)
            return
        if data.get('tool_calls'):
            if data.get('content'):
                add('assistant', 'assistant', data['content'], **meta)
            for call in data['tool_calls']:
                fn = call.get('function', call)
                add('tool', fn.get('name', 'tool'), fn.get('arguments', ''), key=str(call.get('id', '')), **meta)
            return
        if data.get('role') == 'tool':
            add('tool', 'tool_result', output=data.get('content'), key=str(data.get('tool_call_id', '')), **meta)
            return
        kind = ('reasoning' if 'reasoning' in mt or mt == 'thinking' else 'assistant' if mt in ('agent_message', 'agent_message_delta', 'text')
                else 'user' if mt == 'user_message' else data.get('role', 'system'))
        if t == 'message':
            kind = data.get('role', 'assistant')
        body = msg.get('text', msg.get('delta', msg.get('content', msg)))
        add(kind, mt, body, **meta)

    # Structured trajectory documents are common in nested DeepSWE evaluations.
    if path.suffix == '.json':
        try:
            document = json.loads(path.read_text(errors='replace'))
        except ValueError:
            document = None
        if isinstance(document, (list, dict)):
            rows = document if isinstance(document, list) else document.get('messages', [document])
            for n, row in enumerate(rows, 1):
                process(row, n)
            return events
    with path.open(encoding='utf-8', errors='replace') as source:
        for number, line in enumerate(source, 1):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            match = STAMP.match(line)
            timestamp = match[1] if match else ''
            text = line[match.end():] if match else line
            try:
                process(json.loads(text), number, timestamp)
            except json.JSONDecodeError:
                # Preserve non-JSON preambles and malformed/truncated records.
                if events and events[-1]['kind'] == 'log' and len(events[-1]['body']) < 16000:
                    events[-1]['body'] += '\n' + line
                else:
                    add('log', '日志 / 未解析文本', line, line=number, timestamp=timestamp)
    return events
