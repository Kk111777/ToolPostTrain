import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp/ProjectB')
REPO = ROOT / 'repo'
name = sys.argv[1]
out = Path(sys.argv[2])
meta = Path(sys.argv[3])
expected_mapping_sha = 'ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd'
expected_manifest_sha = '160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16'
mapping_path = REPO / 'manifests/final_holdout_v1_runtime_mapping.json'
manifest_path = REPO / 'manifests/final_holdout_v1_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapping_module = load(REPO / 'scripts/audit_final_eval_runtime_mapping.py', 'projectb_runtime_audit')
mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
jsonls = sorted(out.rglob('*.jsonl')) if out.is_dir() else []
jsonl = jsonls[0] if len(jsonls) == 1 else None
items = []
if jsonl is not None:
    items = [json.loads(line) for line in jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
mapping_check = mapping_module.audit_jsonl(jsonl, mapping.get('rows', [])) if jsonl is not None else {'pass': False}
log_text = ''
for path in sorted(meta.rglob('*')) + sorted(out.rglob('*')):
    if path.is_file() and path.suffix.lower() in {'.log', '.yaml', '.yml', '.txt'}:
        log_text += path.read_text(errors='replace') + '\n'
fatal_patterns = [
    r'Traceback', r'\bFATAL\b', r'CUDA error', r'out of memory', r'\bOOM\b',
    r'optimizer\.step', r'optimizer step', r'Saving checkpoint', r'save_checkpoint',
]
fatal_hits = [pattern for pattern in fatal_patterns if re.search(pattern, log_text, re.I)]
old_test_hits = [token for token in ['test.parquet', '/test.parquet'] if token in log_text]
model_files = [
    str(path.relative_to(out)) for path in out.rglob('*')
    if path.is_file() and (path.suffix.lower() in {'.safetensors', '.pt', '.pth', '.bin'} or 'optimizer' in path.name.lower())
]
config_files = [str(path) for base in (out, meta) for path in base.rglob('*') if path.is_file() and 'resolved_config' in path.name]
exit_code = (meta / 'exit_code.txt').read_text().strip() if (meta / 'exit_code.txt').is_file() else None
checks = {
    'exit_code_zero': exit_code == '0',
    'one_jsonl': len(jsonls) == 1,
    'exactly_108_nonempty_records': len(items) == 108,
    'mapping_108_of_108': bool(mapping_check.get('pass')) and mapping_check.get('matched_rows') == 108 and mapping_check.get('missing_rows') == 0 and mapping_check.get('duplicate_rows') == 0 and mapping_check.get('unique_source_ids') == 108,
    'no_fatal_or_update_markers': not fatal_hits,
    'no_old_test_reference': not old_test_hits,
    'no_checkpoint_or_model_files': not model_files,
    'resolved_config_saved': bool(config_files),
    'total_steps_zero_evidence': 'Total training steps: 0' in log_text and 'Total steps: 0' in log_text,
    'endpoint_mapping_identity': sha(mapping_path) == expected_mapping_sha and sha(manifest_path) == expected_manifest_sha,
}
passed = all(checks.values())
result = {
    'schema': 'projectb_final_holdout_v1_technical_gate_v1',
    'model': name,
    'output_dir': str(out),
    'status': 'PASS' if passed else 'TECHNICAL_FAILURE',
    'jsonl_path': str(jsonl) if jsonl else None,
    'rows': len(items),
    'jsonl_sha256': sha(jsonl) if jsonl else None,
    'mapping': mapping_check,
    'endpoint_mapping_sha256': sha(mapping_path),
    'endpoint_manifest_sha256': sha(manifest_path),
    'resolved_config_files': config_files,
    'fatal_hits': fatal_hits,
    'old_test_hits': old_test_hits,
    'model_files': model_files,
    'checks': checks,
}
meta.mkdir(parents=True, exist_ok=True)
(meta / 'technical_gate.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 1)
