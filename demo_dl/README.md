# Download the data

Data images available at: <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

Data split available at: <https://github.com/EUCAIM/demo_dl_data/>

**Clone** this [repo](<https://github.com/EUCAIM/demo_dl_data/>) in the `demo_dl` folder.

The [chest X-Ray](<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>) data must be downloaded and placed in the `demo_dl_data` folder.

# Launch the experiment

Install the requirements: `pip install -r requirements.txt`

Launch the demo: `python run_demo_dl.py`

Args:

- `-r`, `--remote` to launch the compute remotely. Note that you need to set the `SUBSTRA_ORG_{num_org}_PASSWORD` env variables to be able to connect to each backend before being able to launch an experiment remotely. The config files are stored in the `config_files` folder. You can also access your compute plan on the frontend when launched remotely (see bellow for frontend urls). Use the same credentials to connect.

For instance:

```sh
export SUBSTRA_ORG_1_PASSWORD="my_pwd"
export SUBSTRA_ORG_2_PASSWORD="my_pwd"
export SUBSTRA_ORG_3_PASSWORD="my_pwd"
```

- `--n-split` to trigger the 2 or 3 data provider context. Default to 2.
- `--n-round` to choose the number of round of the compute plan. Default to 5.

The script, if launched for the first time, will create a `substra_data_{NUM-DATA-PROVIDER}_split` to copy the image regarding the given split in the repo.

Frontend urls:

- <https://substra.org-1.eucaim.cg.owkin.tech/>
- <https://substra.org-2.eucaim.cg.owkin.tech/>
- <https://substra.org-3.eucaim.cg.owkin.tech/>

---
**Note for remote mode:**

To avoid registering the data at every compute plan, we take advantage of a cache system that stores the different keys of datasets and datasamples after a remote registration. These keys are stored in the `susbtra_dl_assets/dataset/cache_keys` folder.

To relaunch the registration, simply delete these files. A new one will be generated with new keys and the data will be re-registered.

---
