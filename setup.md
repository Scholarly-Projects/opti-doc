# For macOS
brew install pyenv

# Set up pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

exec "$SHELL"

# Install specific Python version
pyenv install 3.11.9

# Create and activate virtual environment
pyenv virtualenv 3.11.9 .venv
pyenv activate .venv

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

brew install poppler

pip install --upgrade torch>=2.6 torchvision torchaudio

python script.py

python review.py