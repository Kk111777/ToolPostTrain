import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp/ProjectB')
REPO = ROOT / 'repo'
NAME = sys.argv[1] if len(sys.argv) > 1 else 'RL_INIT_V1'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'final-test/final_holdout_v1/01_RL_INIT_V1_retry_02'
META = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / 'final-test/execution-metadata/final_holdout_v1/01_RL_INIT_V1_retry_02'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def get(cfg, key):
    for part in key.split('.'):
        cfg = cfg[part]
    return cfg

def extract_config(log):
    clean = re.sub(r'\x1b\[[0-9;]*m', '', log)
    clean = re.sub(r'^\(TaskRunner pid=\d+\) ', '', clean, flags=re.M)
    start = clean.index("{'actor_rollout_ref':")
    lines = clean[start:].splitlines()
    for n, line in enumerate(lines):
        if line.rstrip().endswith('}'):
            try:
                obj = ast.literal_eval('\n'.join(lines[:n+1]))
                if isinstance(obj, dict) and 'trainer' in obj:
                    return obj
            except (SyntaxError, ValueError):
                pass
    raise ValueError('actual resolved config not recoverable from runtime log')

def main():
    checks = {}
    log = (META / 'launcher.log').read_text()
    cfg = extract_config(log)
    files = list(OUT.rglob('*.jsonl'))
    checks['one_jsonl'] = len(files) == 1
    if not checks['one_jsonl']:
        raise RuntimeError('not exactly one JSONL')
    path = files[0]
    before = sha(path)
    items = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    checks['exactly_108_records'] = len(items) == 108
    mapping_path = REPO / 'manifests/final_holdout_v1_runtime_mapping.json'
    endpoint_path = REPO / 'manifests/final_holdout_v1_manifest.json'
    checks['mapping_sha'] = sha(mapping_path) == 'ee7ab8143f57c6c9f60c35fd0a5c7c48df73ee28fb3c72761d6bf7a7eb7d9cdd'
    checks['endpoint_manifest_sha'] = sha(endpoint_path) == '160befdb3c87aab85b8d65254c70ca83823e393ace2478c6b1e226898be97a16'
    endpoint = load(endpoint_path)
    mapping = load(mapping_path)
    pairs = {(x['runtime_input_sha256'], x['ground_truth_sha256_exact']): x for x in mapping['rows']}
    seen = set()
    missing = duplicates = 0
    for item in items:
        pair = (hashlib.sha256(str(item['input']).encode()).hexdigest(), hashlib.sha256(str(item['gts']).encode()).hexdigest())
        missing += pair not in pairs
        duplicates += pair in seen
        seen.add(pair)
    checks['unique_mapping_108'] = len(pairs) == len(seen) == 108 and not missing and not duplicates and seen == set(pairs)
    checks['exit_code_zero'] = (META / 'exit_code.txt').read_text().strip() == '0'
    expected = {
        'actor_rollout_ref.rollout.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.n': 1,
        'actor_rollout_ref.rollout.val_kwargs.temperature': 0,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': False,
        'actor_rollout_ref.rollout.val_kwargs.top_k': -1,
        'actor_rollout_ref.rollout.val_kwargs.top_p': 1.0,
        'data.max_prompt_length': 2048,
        'data.max_response_length': 1024,
        'data.validation_shuffle': False,
        'data.val_max_samples': 108,
        'trainer.val_only': True,
        'trainer.val_before_train': True,
        'trainer.total_epochs': 0,
        'trainer.total_training_steps': 0,
        'trainer.resume_mode': 'disable',
        'trainer.save_freq': -1,
        'algorithm.use_kl_in_reward': False,
        'actor_rollout_ref.actor.use_kl_loss': False,
    }
    config_fields = {k: {'actual': get(cfg, k), 'expected': v} for k, v in expected.items()}
    checks['frozen_generation_and_val_only'] = all(x['actual'] == x['expected'] for x in config_fields.values())
    val_files = get(cfg, 'data.val_files')
    if isinstance(val_files, str):
        val_files = [val_files]
    checks['frozen_endpoint_path'] = val_files == [str(ROOT / 'env-modern/final_holdout_v1.parquet')]
    checks['endpoint_parquet_sha'] = sha(val_files[0]) == endpoint['derived_parquet_sha256'] == 'e8f2e025906fa5e9e3088d9a29494e05da5fb47ddf4705b2550956c4380d8f22'
    models = load(REPO / 'manifests/final_test_models_manifest.json')['models']
    model = models[NAME]
    checks['frozen_model_path'] = get(cfg, 'actor_rollout_ref.model.path') == model['path']
    model_checks = {f: sha(Path(model['path']) / f) == data['sha256'] for f, data in model['files'].items()}
    checks['model_and_tokenizer_file_hashes'] = all(model_checks.values())
    scorer_path = Path(get(cfg, 'reward.custom_reward_function.path'))
    checks['scorer_sha'] = sha(scorer_path) == '4b437fdbc4a834953b640f59c0b966b1838dd771a0d86da89e2398c7fe37046c'
    scorer = module(scorer_path, 'supervision_frozen_scorer')
    deltas = {k: 0.0 for k in ['score', 'accuracy_reward', 'format_reward']}
    finite = True
    for item in items:
        with contextlib.redirect_stdout(io.StringIO()):
            scored = scorer.compute_score('rlla', str(item['output']), str(item['gts']), {'experiment_name': get(cfg, 'trainer.experiment_name')}, step=0)
        for key in deltas:
            a, b = float(item[key]), float(scored[key])
            finite &= math.isfinite(a) and math.isfinite(b)
            deltas[key] = max(deltas[key], abs(a-b))
    checks['scorer_finite_and_consistent'] = finite and max(deltas.values()) <= 1e-6
    forbidden = [str(p) for p in OUT.rglob('*') if p.is_file() and (p.suffix in {'.pt', '.pth', '.bin', '.safetensors'} or 'optimizer' in p.name.lower())]
    checks['no_checkpoint_files'] = not forbidden
    patterns = [r'Traceback', r'CUDA error', r'out of memory', r'optimizer\.step', r'optimizer step', r'Saving checkpoint', r'save_checkpoint']
    fatal = [x for x in patterns if re.search(x, log, re.I)]
    checks['no_fatal_or_update_markers'] = not fatal
    checks['zero_training_steps_runtime'] = 'Total training steps: 0' in log and 'Total steps: 0' in log
    checks['no_optimizer_loop_runtime'] = 'step:1 -' not in log and 'Training Progress:' not in log
    checks['jsonl_unchanged_during_audit'] = sha(path) == before
    if NAME == 'RL_INIT_V1':
        checks['expected_existing_jsonl_sha'] = before == '0063a0c924fbaf3226de7e3ce5eb9e93c716b1237ba33b609301c2d7a753bb4a'
    result = {'status': 'PASS' if all(checks.values()) else 'GPU_FINAL_MATRIX_BLOCKED_EXISTING_OUTPUT_INTEGRITY', 'model': NAME, 'checks': checks, 'jsonl_path': str(path), 'jsonl_sha256': before, 'rows': len(items), 'mapping': {'matched': len(seen)-missing, 'missing': missing, 'duplicates': duplicates, 'unique_source_ids': len({pairs[x]['source_id'] for x in seen if x in pairs})}, 'scorer_max_abs_delta': deltas, 'config_fields': config_fields, 'model_file_checks': model_checks, 'forbidden_files': forbidden, 'fatal_markers': fatal, 'actual_resolved_config_recovered_from_runtime': True}
    print(json.dumps(result, sort_keys=True))
    if len(sys.argv) > 4 and sys.argv[4] == '--save':
        (META / 'actual_resolved_config.json').write_text(json.dumps(cfg, indent=2, sort_keys=True)+'\n')
        (META / 'supervision_integrity_gate.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    raise SystemExit(0 if result['status'] == 'PASS' else 1)

main()
