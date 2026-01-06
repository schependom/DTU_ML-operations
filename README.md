# DTU Machine Learning Operations

Repository for the DTU Machine Learning Operations course.

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Installation and notes on using `uv`

### Virtual environment

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

### Dependency management

Add a new dependency with (or add it straight to `pyproject.toml` and run `uv sync`):

```bash
uv add <package-name>
# e.g. uv add numpy
```

To add a development dependency, use:

```bash
uv add <package-name> --group dev
# e.g. uv add pytest --group dev
```

### Requirements files

The `requirements.txt` and `requirements-dev.txt` files are automatically generated from `pyproject.toml` and `uv.lock`. They are provided for compatibility with tools that do not support `pyproject.toml` (e.g. some CIs or older docker builds), and for users who prefer `pip` + `conda`.

Do **not** edit these files manually. Instead, update `pyproject.toml` and then run the following command to sync them:

```bash
uv export --format requirements-txt --output-file requirements.txt
uv export --format requirements-txt --only-group dev --output-file requirements-dev.txt
```

### Running scripts

Running a script inside the virtual environment can be done with:

```bash
uv run <script-name>.py
# e.g. uv run src/ml_ops/train.py
```

This can get quite tedious, so we can make an alias `uvr` for this:

```bash
echo "alias uvr='uv run'" >> ~/.bashrc
source ~/.bashrc
```

Now you can run scripts like this:

```bash
uvr script.py
# e.g. uvr src/ml_ops/train.py
```

### Other `uv` commands

Change Python version with:

```bash
uv python pin <version>
```

### Using `uvx` for global tools

`uvx` can be used to install _tools_ (which are external command line tools, not libraries used in your code) globally on your machine. Tools include `black`, `ruff`, `pytest` or the simple `cowsay`. You can install such tools with `uvx`. For example:

```bash
uvx add cowsay
```

Then you can run the tool like this:

```bash
uvx cowsay -t "muuh"
```

If you run above command without having installed `cowsay` with `uvx`, it will install it for you automatically.

## Usage

### Version control

Clone the repo:

```bash
git clone git@github.com:schependom/DTU_ML-Operations.git
cd DTU_ML-Operations
```

Authenticate DVC using SSH (make sure you have access to the remote):

```bash
dvc remote modify --local myremote auth ssh
```

Pull data from DVC remote:

```bash
dvc pull
```

You can use `invoke` to run common tasks. To list available tasks, run:

```bash
invoke --list
# Available tasks:
#
#   build-docs        Build documentation.
#   docker-build      Build docker images.
#   preprocess-data   Preprocess data.
#   serve-docs        Serve documentation.
#   test              Run tests.
#   train             Train model.
```

Now, to run a task, use:

```bash
invoke <task-name>
# e.g. invoke preprocess-data
```

After preprocessing data (v2), you can push (Data Version Control [DVC]) changes to the remote with:

```bash
dvc add data
git add data.dvc # or `git add .`
git commit -m "Add new data"
git tag -a v2.0 -m "Version 2.0"
# Why tag? To mark a specific point in git history as important (e.g., a release)
#   -a to create an annotated tag
#   -m to add a message to the tag
dvc push
git push origin main --tags
```

Or simply use:

```bash
invoke dvc --folder 'data' --message 'Add new data'
```

To go back to a specific version later, you can checkout the git tag:

```bash
git switch v1.0 # or `git checkout v1.0`
dvc checkout
```

To go back to the latest version, use:

```bash
git switch main # or `git checkout main`
dvc checkout
```

### Training

To train the model, run either of the following commands:

```bash
uvr invoke train
uvr src/ml_ops/train.py
uvr train # because we configured a script entry point in pyproject.toml
```
