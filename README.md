# R4R Training

Exemplary training workflow for
    - propensity score based control matching 
    - ATE estimation with pyTMLE

## Installation

These instructions assume that you are using **Visual Studio Code (VS Code)**.
You do not need previous Python or programming experience.

### Before the course: install uv

`uv` creates and manages the Python environment for this course. Install it
once before opening the course folder in VS Code. Follow the official
installation instructions for your operating system:

<https://docs.astral.sh/uv/getting-started/installation/>

After installing `uv`, close and reopen VS Code. You will use `uv` later from
the VS Code terminal, but you only need to install it once on your computer.

### Set up the course in VS Code

1. Download the course repository and extract it, or clone it if you already
    use Git.
2. Open **VS Code**.
3. Select **File > Open Folder...** and select the `r4r-training` folder. Make
    sure the Explorer shows `pyproject.toml`, `uv.lock`, `data`, and `notebooks`.
4. Select **Terminal > New Terminal**. A terminal opens at the bottom of VS
    Code. All commands below should be entered there.
5. Create the course environment by running:

    ```text
    uv sync --locked
    ```

    This creates a local `.venv` folder and installs the tested Python version
    and packages. The first installation may take several minutes.

### Open a notebook

Install the **Python** and **Jupyter** extensions when VS Code offers to do so.
Then open one of the notebooks in the `training` folder. Before selecting a
Jupyter kernel, open the Command Palette and run **Python: Select Interpreter**.
Choose the environment whose name includes `r4r-training`. The environment
will then appear in the notebook's kernel picker; select it when VS Code asks
you to choose a kernel.

The notebooks expect to be run from their original location inside the
repository because they load files from `../data`.

### Updating the environment

Do not run `uv lock` during the course. The committed `uv.lock` file records
the tested dependency versions. To recreate or repair the environment on your
computer, run this in the VS Code terminal:

```text
uv sync --locked
```

### Troubleshooting

If `uv` is not recognized, close and reopen VS Code. If the `r4r-training`
environment does not appear in the notebook's kernel picker, open the Command
Palette (Str+Shift+P), run **Python: Select Interpreter**, and choose the environment whose
name includes `r4r-training`. Then return to the notebook and select that
environment as its kernel. If a notebook cannot find a package, restart the
notebook kernel.
