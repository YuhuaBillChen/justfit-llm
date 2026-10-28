from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess

config = json.loads(Path('.github/release-assets-2.3.1.json').read_text())
output = Path('reviewed-assets')
output.mkdir(exist_ok=False)

def fetch(item):
    name, expected = item
    assert '/' not in name and '\\' not in name
    target = output / name
    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                    '--proto', '=https', '--proto-redir', '=https', '--retry', '5',
                    '--retry-all-errors', '--connect-timeout', '30', '--max-time', '1800',
                    '--continue-at', '-', '--output', str(target), config['source_url'] + '/' + name], check=True)
    digest = hashlib.sha256()
    with target.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, name + ': SHA256 mismatch'
    print(name + ': verified', flush=True)

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, config['files'].items()))
