This is to install and test the data packing for mamba hybrid models.

You need to install varlen-mamba from [here](https://github.com/jxiw/varlen_mamba).

Installation commands:

```
conda create -n m1_pack python=3.10
conda activate m1_pack
pip install numpy
pip install torch==2.4.0

git clone --branch varlen_mamba https://github.com/jxiw/varlen_mamba.git
cd varlen_mamba/
python setup.py install
```

Then follow [this](rl/README.md) to install rl environments. And test with this,

```
cd rl/verl/
python tests/pack_mamba/test_mamba_layer.py
python tests/pack_mamba/test_pack_hybrid.py
```

You need to see `0.0` difference for those two checks.

Those two libraries are modified from the awesome packed Mamba packages [here](https://github.com/ptxu78/pack_mamba).

The original [`mha.py`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mha.py) in official Mamba repository does not support `cu_seqlens`. We are able to fix this and support it in [this](rl/verl/verl/models/mamba/mha.py) and [this](rl/verl/verl/models/mamba/rotary.py)