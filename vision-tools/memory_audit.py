"""Conservative upper bound includes kernel-reported lifetime peaks."""
import json
from pathlib import Path
def audit(folder,budget=1127*2**20):
 rows=[]
 for p in Path(folder).glob('worker-memory-*.jsonl'):
  rows.extend(json.loads(s) for s in p.read_text().splitlines() if s.strip())
 rows=[r for r in rows if r.get('ok') and r.get('pid_ok')]
 if not rows:raise ValueError('no valid physical/neural attribution')
 observed=max(r['ledger']['phys_footprint']+r['ledger']['neural_nofootprint'] for r in rows)
 phys=max(max(r['ledger']['phys_footprint'],r['rusage'][3]) for r in rows)
 neural=max(max(r['ledger']['neural_nofootprint'],r['ledger']['neural_nofootprint_peak'],r['rusage'][2]) for r in rows)
 bound=phys+neural
 return dict(observed_simultaneous_peak_bytes=observed,physical_lifetime_peak_bytes=phys,neural_lifetime_peak_bytes=neural,conservative_lifetime_peak_sum_bytes=bound,budget_bytes=budget,budget_passed=bound<=budget,final_neural_bytes=rows[-1]['ledger']['neural_nofootprint'],scope='sum of separate lifetime peaks is conservative, not simultaneous or unique system memory; covers transient neural peaks absent between50ms samples')
