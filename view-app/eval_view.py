"""Read saved evaluation conversations; never rerun a model."""
import json
from functools import lru_cache
from pathlib import Path


def sources(run: Path) -> list[dict]:
    paths = set()
    for pattern in ('final_eval/eval_*/*.trajectory.json', 'eval_*/*.trajectory.json',
                    'final_eval/eval_*/inspect_logs/*.json',
                    'final_eval/*/eval_*/*.trajectory.json', 'eval_*/inspect_logs/*.json',
                    'evaluation/inspect_logs/seed*/*.json', 'evaluation/inspect_logs/*.json',
                    'reruns/inspect_logs/seed*/*.json'):
        paths.update(run.glob(pattern))
    result = []
    for path in sorted(paths):
        if not path.resolve().is_relative_to(run.resolve()):
            continue
        # Native mini trajectories contain generation; their Inspect companion
        # records scoring only, so offer the native file instead.
        if path.parent.name == 'inspect_logs' and (path.parent.parent / 'mini-swe-agent.trajectory.json').is_file():
            continue
        relative = path.relative_to(run).as_posix()
        result.append(dict(path=relative, phase='训练期间试评测' if relative.startswith('eval_') else '最终评测'))
    return sorted(result, key=lambda row: (row['phase'] != '最终评测', row['path']))


@lru_cache(maxsize=2)
def load(path: str, mtime: int, size: int) -> tuple[list[dict], str]:
    document = json.loads(Path(path).read_text(errors='replace'))
    calls = []
    samples = document.get('samples', []) if isinstance(document, dict) else []
    if samples:
        for sample in samples:
            model_events = [event for event in sample.get('events', []) if event.get('event') == 'model']
            for event in model_events:
                calls.append(dict(sample=sample.get('id'), input=event.get('input', []),
                                  output=event.get('output', {}), timestamp=event.get('timestamp', ''),
                                  basis='Inspect 保存的模型调用输入与输出'))
        if calls:
            return calls, 'Inspect 模型调用记录'
        groups = [(sample.get('id'), sample.get('messages', [])) for sample in samples]
    elif isinstance(document, dict):
        groups = [('', document.get('messages', []))]
    elif isinstance(document, list):
        groups = [('', document)]
    else:
        raise ValueError('不支持的评测轨迹结构')
    for sample_id, messages in groups:
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            following = []
            for next_message in messages[index+1:]:
                if isinstance(next_message, dict) and next_message.get('role') == 'assistant':
                    break
                following.append(next_message)
            calls.append(dict(sample=sample_id, history=messages, index=index,
                              output=message, following=following,
                              basis='根据保存的历史消息重建输入上下文；不是原始 HTTP 请求包'))
    return calls, '保存的评测对话（输入按历史消息重建）'


def conversation(path: Path, selected: int | None = None) -> dict:
    stat = path.stat()
    calls, description = load(str(path), stat.st_mtime_ns, stat.st_size)
    if selected is not None:
        if selected < 0 or selected >= len(calls):
            raise ValueError('评测轮次不存在')
        call = calls[selected]
        return {key: value for key, value in call.items() if key not in ('history', 'index')} | {
            'input': call['history'][:call['index']] if 'history' in call else call['input']}
    rows = []
    for index, call in enumerate(calls):
        output = call['output']
        text = output.get('content', output.get('completion', '')) if isinstance(output, dict) else str(output)
        if not text and isinstance(output, dict):
            text = output.get('choices', output.get('tool_calls', output))
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        rows.append(dict(index=index, sample=call.get('sample'), preview=text[:220]))
    return dict(calls=rows, description=description)
