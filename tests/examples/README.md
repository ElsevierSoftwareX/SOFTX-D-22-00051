# Example reduction scripts

To write a reduction script, you can consult our documentation: https://scse.ornl.gov

To use the example Jupyter notebook:
  - Download an example notebook from this repo.
  - Go to https://jupyter.sns.gov
  - Once on there, click `Open...` from the `File menu and navigate to where you downloaded the notebook to open it.
  - You will now be able to hit `Ctrl-Enter` in each cell to execute the cide.
  - The comparison cell at the end plots the results. If you want to compare to the old EQSANS reduction, execute the `EQSANS_porasil.py` script on one of the analysis computers. This will create the reference data in your home directory.


To write your own example scripts and run them from the command line, you can use the save conda environment used by the Jupyter notebook.
Just do the following before calling python.

```
export PATH="/SNS/software/miniconda2/bin:$PATH"
source activate sans
```