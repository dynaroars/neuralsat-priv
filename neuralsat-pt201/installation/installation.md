1. Install system dependencies for `Python` virtual environment:

```bash
sudo apt-get install libbz2-dev
sudo apt-get install lzma
sudo apt-get install liblzma-dev
```

2. Create `Python` virtual environment:

```bash
python -m venv neuralsat 
source neuralsat/bin/activate
```

3. Install the latest `Pytorch`:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

4. Install other requirements:

```bash
pip install -r reqs.txt 
```

5. Install the latest `Triton` (**Note**: older versions might not work since `proton` was just added to `Triton` recently):

```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```
