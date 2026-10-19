#!/usr/bin/env python3
"""Read-only experiment viewer. Run: python view-app/app.py [--results-dir PATH]."""
from __future__ import annotations

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shlex
import stat
from urllib.parse import parse_qs, urlparse

from trace import parse_trace
from eval_view import sources as evaluation_sources, conversation

APP = Path(__file__).resolve().parent
REPO = APP.parent
MARKERS = ('solve_out.txt', 'solve_out_sanitized.txt', 'solve_parsed.txt', 'run_config.json', 'metrics.json')
PREVIEW_BYTES = 256 * 1024


def default_root() -> Path:
    env = REPO / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.removeprefix('export ').partition('=')
            if sep and key.strip() == 'POST_TRAIN_BENCH_RESULTS_DIR':
                parts = shlex.split(value, comments=True)
                if parts:
                    path = Path(os.path.expandvars(parts[0])).expanduser()
                    return (REPO / path).resolve()
    value = os.environ.get('POST_TRAIN_BENCH_RESULTS_DIR', str(REPO / 'results'))
    return Path(value).expanduser().resolve()


def child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PermissionError('路径越过了当前实验目录（或符号链接指向外部）。')
    return path


def is_run(path: Path) -> bool:
    return any((path / name).is_file() for name in MARKERS)


def discover(root: Path, depth: int = 4) -> list[dict]:
    found = []

    def visit(folder: Path, remaining: int) -> None:
        if is_run(folder):
            found.append(dict(id=folder.relative_to(root).as_posix(), name=folder.name,
                              group=folder.parent.name, modified=folder.stat().st_mtime))
            # Evaluation-only reruns are saved inside the original run. Discover
            # those explicitly, without walking task/checkpoint directories.
            if remaining > 0:
                for rerun in sorted(folder.glob('reeval_*')):
                    if rerun.is_dir() and not rerun.is_symlink() and is_run(rerun):
                        visit(rerun, remaining - 1)
            return
        if remaining <= 0:
            return
        for entry in sorted(folder.iterdir()):
            if entry.name.startswith('.') or entry.is_symlink() or not entry.is_dir():
                continue
            try:
                visit(entry, remaining - 1)
            except PermissionError:
                continue
    visit(root, depth)
    return sorted(found, key=lambda item: item['modified'], reverse=True)


@lru_cache(maxsize=2)
def cached_trace(path: str, modified: int, size: int) -> list[dict]:
    return parse_trace(Path(path))


def preview(path: Path, offset: int = 0) -> dict:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('只能读取普通文件。')
    with path.open('rb') as source:
        source.seek(offset)
        data = source.read(PREVIEW_BYTES)
    binary = b'\0' in data
    size = path.stat().st_size
    return dict(text='' if binary else data.decode('utf-8', errors='replace'), binary=binary,
                size=size, offset=offset, next_offset=offset + len(data), more=offset + len(data) < size)


def workspace(run: Path) -> dict | None:
    """Map the saved task directory back to its container location."""
    for source in ('training_home/task', 'task'):
        try:
            if child(run, source).is_dir():
                return dict(source=source, container_path='/home/ben/task')
        except PermissionError:
            continue
    return None


def summary(run: Path) -> dict:
    names = ['run_config.json', 'status.json', 'metrics.json', 'metrics_averaged.json',
             'metrics_averaged_exact_match.json', 'time_taken.txt', 'cli_version.txt', 'prompt.txt']
    names += [p.relative_to(run).as_posix() for p in sorted(run.glob('judgement*.json'))]
    names += [p.relative_to(run).as_posix() for p in sorted(run.glob('judgement_multi_runs/*.json'))]
    documents = {}
    for name in names:
        try:
            path = child(run, name)
            if not path.is_file():
                continue
            data = preview(path)
            if name.endswith('.json') and not data['more']:
                try:
                    data['json'] = json.loads(data['text'])
                except ValueError:
                    data['error'] = 'JSON 不完整或格式错误；显示原文。'
            documents[name] = data
        except (OSError, ValueError) as exc:
            documents[name] = dict(error=str(exc))
    traces = [p.name for p in sorted(run.iterdir()) if p.is_file() and
              (p.name.startswith(('solve_', 'judge_output')) and p.suffix in ('.txt', '.json', '.jsonl'))]
    traces.sort(key=lambda name: (name != 'solve_out_sanitized.txt', name != 'solve_out.txt', name))
    return dict(documents=documents, traces=traces, workspace=workspace(run), evaluations=evaluation_sources(run),
                workspaces=[p for p in ('task', 'training_home/task', 'final_model') if (run / p).is_dir()])


class Handler(BaseHTTPRequestHandler):
    root: Path
    scan_depth: int = 4

    def send(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            code = 403 if isinstance(exc, PermissionError) else 404 if isinstance(exc, FileNotFoundError) else 400 if isinstance(exc, ValueError) else 500
            self.send(json.dumps({'error': str(exc)}, ensure_ascii=False).encode(), 'application/json; charset=utf-8', code)

    def route(self) -> None:
        url = urlparse(self.path)
        q = {key: values[0] for key, values in parse_qs(url.query).items()}
        if url.path == '/highlight.js' or url.path in {f'/vendor/{name}.js' for name in ('prism', 'prism-python', 'prism-bash', 'prism-json', 'prism-yaml', 'prism-diff')}:
            self.send((APP / 'static' / url.path.lstrip('/')).read_bytes(), 'text/javascript; charset=utf-8')
            return
        if url.path in ('/', '/app.js', '/style.css'):
            name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[url.path]
            mime = {'index.html': 'text/html', 'app.js': 'text/javascript', 'style.css': 'text/css'}[name]
            self.send((APP / 'static' / name).read_bytes(), mime + '; charset=utf-8')
            return
        if url.path == '/api/runs':
            data = dict(root=str(self.root), runs=discover(self.root, self.scan_depth))
        else:
            run = child(self.root, q.get('run', '.'))
            if not run.is_dir() or not is_run(run):
                raise FileNotFoundError('未找到实验目录。')
            file_root = run
            if q.get('scope') == 'workspace':
                saved = workspace(run)
                if saved is None:
                    raise FileNotFoundError('这次实验没有保存工作目录。')
                file_root = child(run, saved['source'])
            if url.path == '/api/summary':
                data = summary(run)
            elif url.path == '/api/evaluation':
                data = conversation(child(run, q['path']), int(q['call']) if 'call' in q else None)
            elif url.path == '/api/files':
                relative = q.get('path', '.')
                folder = child(file_root, relative)
                entries = []
                for p in folder.iterdir():
                    try:
                        entry_path = (Path(relative) / p.name).as_posix()
                        resolved = child(file_root, entry_path)
                        s = resolved.stat()
                        entries.append(dict(name=p.name, path=entry_path,
                                            directory=resolved.is_dir(), size=s.st_size, symlink=p.is_symlink()))
                    except OSError as exc:
                        entries.append(dict(name=p.name, path=(Path(relative) / p.name).as_posix(), directory=False,
                                            blocked=True, error=str(exc), symlink=p.is_symlink(), size=0))
                data = dict(entries=sorted(entries, key=lambda x: (not x['directory'], x['name'].lower())))
            elif url.path == '/api/file':
                data = preview(child(file_root, q['path']), max(0, int(q.get('offset', '0'))))
            elif url.path == '/api/trace':
                path = child(run, q['path'])
                s = path.stat()
                if not stat.S_ISREG(s.st_mode):
                    raise ValueError('只能读取普通轨迹文件。')
                events = cached_trace(str(path), s.st_mtime_ns, s.st_size)
                query = q.get('search', '').lower()
                kind = q.get('kind', '')
                filtered = [e for e in events if (not kind or e['kind'] == kind) and
                            (not query or query in (str(e['title']) + e['body'] + e['output']).lower())]
                offset = max(0, int(q.get('offset', '0')))
                data = dict(total=len(filtered), all_total=len(events), events=filtered[offset:offset + 100])
            else:
                raise FileNotFoundError('Unknown endpoint')
        self.send(json.dumps(data, ensure_ascii=False).encode(), 'application/json; charset=utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', '--run-dir', type=Path, default=default_root(), help='结果根目录或单次实验目录')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--scan-depth', type=int, default=4)
    args = parser.parse_args()
    Handler.root = args.results_dir.expanduser().resolve()
    Handler.scan_depth = args.scan_depth
    if not Handler.root.is_dir():
        parser.error(f'结果目录不存在：{Handler.root}；请使用 --results-dir 指定。')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Experiment viewer: http://{args.host}:{server.server_port}\nResults: {Handler.root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
