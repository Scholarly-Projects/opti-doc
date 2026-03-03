# Set up

## Install system dependencies

brew install pyenv xz poppler ninja

## Set up pyenv in your shell

SHELL_RC="$HOME/.$(basename "$SHELL")rc"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"
exec "$SHELL"

## Install Python 3.11.9 with explicit _lzma (xz) support

env PYTHON_CONFIGURE_OPTS="--with-liblzma" \
    LDFLAGS="-L$(brew --prefix xz)/lib" \
    CPPFLAGS="-I$(brew --prefix xz)/include" \
    pyenv install 3.11.9

pyenv local 3.11.9

## Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

## Install project dependencies

pip install -r requirements.txt

## Force torch back to 2.6.0 

pip install --force-reinstall --no-deps \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

## Verify

## Confirm torch version (must be 2.6.0)

python -c "import torch; print(torch.__version__)"

### Confirm surya is importable

python -c "import surya; print('surya OK')"

### Confirm kraken is importable

python -c "from kraken import blla; print('kraken OK')"

### Run your scripts

python script.py
python review.py