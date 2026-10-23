"""Only the fixed9x3 chain or explicit27x1 12K chain is accepted. (M6 b27 copy: the 27x1 chain is also allowed at 14,400.)"""
import json
from pathlib import Path

def group_count(assets):
    marker=Path(assets)/'partition.json'
    if not marker.exists():return 9
    row=json.loads(marker.read_text())
    if row.get('capacity') not in (12288,14400) or row.get('groups')!=[[i,i+1] for i in range(27)]:
        raise ValueError('unsupported or incomplete vision block partition')
    return 27
