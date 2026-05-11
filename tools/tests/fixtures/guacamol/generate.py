# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate synthetic fixtures for GuacaMol tests.

Run once: python tests/fixtures/guacamol/generate.py
Fixtures are committed to the repo — do not regenerate in CI.

Uses real SMILES strings for the valid/large fixtures so RDKit can parse them.
"""

import json
import random
from pathlib import Path

SEED = 42
OUT = Path(__file__).parent

# 30 known-valid drug-like SMILES from public sources
_VALID_POOL = [
    "c1ccccc1",
    "CC(=O)O",
    "CCO",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1ccc2ccccc2c1",
    "c1ccncc1",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc(cc1)C(=O)O",
    "CC(C)(C)c1ccc(cc1)OCC(O)CNC(C)(C)C",
    "CN1CCC[C@H]1c2cccnc2",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "O=C(O)c1ccccc1O",
    "c1ccc(cc1)O",
    "c1ccc2[nH]ccc2c1",
    "CC(=O)c1ccc(cc1)O",
    "O=Cc1ccccc1",
    "NCCc1ccc(O)c(O)c1",
    "C[N+](C)(C)CCO",
    "OC(=O)c1cccnc1",
    "CC(O)=O",
    "c1ccc(Cl)cc1",
    "c1ccc(F)cc1",
    "CC(=O)OCC",
    "NCCO",
    "OCC(O)CO",
    "c1ccc(N)cc1",
    "CC(C)O",
    "O=C(O)CC(O)(CC(=O)O)C(=O)O",
    "C",
    "N",
]

if __name__ == "__main__":
    rng = random.Random(SEED)

    # valid: all 30 + 2 exact duplicates + 1 minimum-length + 1 long-ish
    valid = list(_VALID_POOL)
    valid += [valid[0], valid[1]]
    valid += ["C", "N"]
    valid += ["CC(=O)Nc1ccc(OCC(=O)O)c(c1)OC"]
    (OUT / "valid.smiles").write_text("\n".join(valid) + "\n")

    # invalid: syntactically malformed and semantically broken
    invalid = [
        "not-a-smiles!!!",
        "(((unclosed",
        "",
        "NOTASMILES",
        "123456",
        "[[[[[unclosed_bracket",
    ]
    (OUT / "invalid.smiles").write_text("\n".join(invalid) + "\n")

    # empty: zero bytes
    (OUT / "empty.smiles").write_text("")

    # large: 1 000 SMILES sampled from valid pool (fast; no network needed)
    rng2 = random.Random(SEED + 1)
    large = [rng2.choice(_VALID_POOL) for _ in range(1_000)]
    (OUT / "large.smiles").write_text("\n".join(large) + "\n")

    # metadata
    metadata = {
        "name": "guacamol_synthetic",
        "version": "0.0.0-synthetic",
        "num_valid_samples": len(valid),
        "description": "Synthetic fixture for testing — not real chemistry data.",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Fixtures written to", OUT)