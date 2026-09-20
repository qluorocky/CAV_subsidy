# Python experiments for AV adoption subsidies

This is a new Python implementation based on the MATLAB files supplied with
Qi Luo, Romesh Saigal, Zhibin Chen, and Yafeng Yin, “Accelerating the adoption
of automated vehicles by subsidies: A dynamic games approach,”
*Transportation Research Part B* 129 (2019), 226–243,
[doi:10.1016/j.trb.2019.09.011](https://doi.org/10.1016/j.trb.2019.09.011).

It runs without MATLAB. The supplied PDF and MATLAB files are **inputs for
interpretation**, not files included in this package. No result in this
repository should be described as an exact numerical reproduction of all
figures in the published paper. The table below states what each output is.

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m av_subsidy.figures --output results --figures all
```
 
## Code layout

- `av_subsidy/model.py`: Bass process, benefit function, and HJB solver.
- `av_subsidy/figures.py`: figure and CSV generator.
- `tests/test_model.py`: numerical checks.
- `results/`: generated example outputs and run metadata.

The source MATLAB, figures, and article PDF are not copied into the package.
