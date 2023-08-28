# Download the data

Data images available at: <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

Data split available at: <https://github.com/EUCAIM/demo_dl_data/>

**Clone** the <https://github.com/EUCAIM/demo_dl_data/> repo in the `demo_dl` folder.

# Launch the experiment

Install the requirements: `pip install requirements.txt`

Launch the demo: `python run_demo_dl.py`

Args:

- `--n-split` to trigger the 2 or 3 data provider context. Default to 2.
- `--n-round` to choose the number of round of the compute plan. Default to 5.

The script, if launched for the first time, will create a `substra_data_{NUM-DATA-PROVIDER}_split` to copy the image regarding the given split in the repo.

The chest X-Ray data must be downloaded and place in the `demo_dl_data` folder.
