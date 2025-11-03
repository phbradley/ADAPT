import json
import sys
from pathlib import Path
from os.path import exists

path_to_tcrdock        = Path(__file__).parents[1].resolve() # also path to alphafold
path_to_design_scripts = Path(__file__).parents[0].resolve()
path_to_filelock       = Path(__file__).parents[1].resolve() / 'FileLock'

def setup_import_paths():
    if str(path_to_tcrdock) != sys.path[0]:
        sys.path.insert(0, str(path_to_tcrdock))
    if str(path_to_filelock) not in sys.path:
        sys.path.append(str(path_to_filelock))


config_file = path_to_tcrdock / 'config_paths.json'

with open(config_file) as f:
    config = json.load(f)

for tag in 'DATA_DIR AF2_PYTHON MPNN_PYTHON MPNN_SCRIPT'.split():
    assert tag in config, f'config_paths.json needs to specify {tag}'
    assert exists(config[tag]), f'{tag} does not exist'

DATA_DIR = config['DATA_DIR']
AF2_PYTHON = config['AF2_PYTHON']
MPNN_PYTHON = config['MPNN_PYTHON']
MPNN_SCRIPT = config['MPNN_SCRIPT']

# these are optional and may not exist
if 'RFAB_SCRIPT' in config:
    RFAB_SCRIPT = config['RFAB_SCRIPT'] 
if 'RFAB_PYTHON' in config:
    RFAB_PYTHON = config['RFAB_PYTHON']
if 'PYROSETTA_PYTHON' in config:
    PYROSETTA_PYTHON = config['PYROSETTA_PYTHON']

if not DATA_DIR.endswith('/'):
    DATA_DIR += '/'


# paths to stuff that was in the zenodo download ie in DATA_DIR
PAIRED_TCR_DB = DATA_DIR + 'paired_human_cdr3s.tsv'
AF2_BINDER_FT_PARAMS = DATA_DIR + 'model_2_ptm_ft_binder_20230729.pkl'
SABDAB_DIR = DATA_DIR + 'antibody_frameworks_2024-01-26/'
RFAB_CHK = DATA_DIR + 'RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt'

## silly thing for Fred Hutch system
FRED_HUTCH_HACKS = '/home/pbradley' in AF2_PYTHON # special hack for FRED HUTCH

