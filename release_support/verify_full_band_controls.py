"""Run the original small full-band controls without overwriting their log."""
from pathlib import Path
import hashlib
import importlib.util
import tempfile

ROOT=Path(__file__).resolve().parents[1]
script=ROOT/'pole_zero_work/code/full_bz_completion/validate_completion.py'
data=ROOT/'pole_zero_work/data_prl/full_bz_completion'
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir() if p.is_file()}
spec=importlib.util.spec_from_file_location('released_full_band_validator',script)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='full_band_small_checks_') as temporary:
    module.LOG=Path(temporary)/'checks.log'
    module.main()
assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir() if p.is_file()}
print('PASS: original reference tables and archived validation log unchanged')
