# DTU_ML-Operations

Activate the virtual environment with:

```bash
source .venv/bin/activate
```

Install dependencies with:

```bash
uv sync
```

Install an optional dependency group with:

```bash
uv sync --group <group-name>
```

Install all dependency groups with:

```bash
uv sync --all-groups
```

Add a new dependency with (or add it straight to `pyproject.toml` and run `uv sync`):

```bash
uv add <package-name>
```

Running a script inside the virtual environment can be done with:

```bash
uv run <script-name>.py
```

This can get quite tedious, so we can make an alias `uvr` for this:

```bash
echo "alias uvr='uv run'" >> ~/.bashrc
source ~/.bashrc
```

Now you can run scripts like this:

```bash
uvr script.py
```

Change Python version with:

```bash
uv python pin <version>
```

`uvx` can be used to install _tools_ (which are external command line tools, not libraries used in your code) globally on your machine. Tools include `black`, `ruff`, `pytest` or the simple `cowsay`. You can install such tools with `uvx`. For example:

```bash
uvx add cowsay
```

Then you can run the tool like this:

```bash
uvx cowsay -t "muuh"
```

If you run above command without having installed `cowsay` with `uvx`, it will install it for you automatically.
