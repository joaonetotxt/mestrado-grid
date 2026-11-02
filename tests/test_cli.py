import json
from pathlib import Path
import subprocess
import sys


def invoke(*args):
    return subprocess.run([sys.executable,'-B','-m','mestrado_grid',*map(str,args)],capture_output=True,text=True)


def test_output_directory_is_never_overwritten(tmp_path):
    protected=tmp_path/'existing'
    protected.mkdir()
    marker=protected/'checkpoint.json'
    marker.write_text('{"saved":true}')
    before=marker.read_bytes()
    proc=invoke('run','--method','PSO','--study',tmp_path/'missing.json','--population',tmp_path/'missing.npy',
        '--output',protected,'--budget','88','--regime','normal')
    assert proc.returncode!=0 and 'never overwritten' in proc.stderr
    assert marker.read_bytes()==before
    assert list(protected.iterdir())==[marker]


def test_minimum_budget_rejected_before_creating_output(tmp_path):
    target=tmp_path/'result'
    proc=invoke('run','--method','PSO','--study',tmp_path/'missing.json','--population',tmp_path/'missing.npy',
        '--output',target,'--budget','8','--regime','normal')
    assert proc.returncode!=0 and 'at least 32' in proc.stderr
    assert not target.exists()


def test_installed_methods_and_source_integrity():
    proc=invoke('methods')
    assert proc.returncode==0
    assert set(proc.stdout.splitlines())=={'PSO','HyDE','GA','ACO','Vortex','QPSO','SHADE','L-SHADE','HYBRID','HYBRID_AI'}
    proc=invoke('verify')
    assert proc.returncode==0 and json.loads(proc.stdout)['verified_files']>100
