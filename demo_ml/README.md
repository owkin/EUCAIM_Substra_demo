# Download the data

Data already split available at: <https://github.com/EUCAIM/demo_ml_data/>

Data first extracted from: <https://github.com/isglobal-brge/rexposome/tree/master/inst/extdata>

**Clone** the <https://github.com/EUCAIM/demo_ml_data/> repo in the `demo_ml` folder.

# Launch the experiment

Install the requirements: `pip install -r requirements.txt`

Launch the demo: `python run_demo_ml.py`

Args:

- `-r`, `--remote` to launch the compute remotely. Note that you need to set the `SUBSTRA_ORG_{num_org}_PASSWORD` env variable to be able to connect to each backend before being able to launch an experiment remotely. The config files are stored in the `config_files` folder. You can also access your compute plan on the frontend when launched remotely (see bellow for frontend urls). Use the same credentials to connect.
- `--n-split` to trigger the 2 or 3 data provider context. Default to 2.

Frontend urls:

- <https://substra.org-1.eucaim.cg.owkin.tech/>
- <https://substra.org-2.eucaim.cg.owkin.tech/>
- <https://substra.org-3.eucaim.cg.owkin.tech/>

---
**Note for remote mode:**

To avoid registering the data at every compute plan, we take advantage of a cache system that stores the different keys of datasets and datasamples after a remote registration. These keys are stored in the `susbtra_ml_assets/dataset/cache_keys` folder.

To relaunch the registration, simply delete these files. A new one will be generated with new keys and the data will be re-registered.

---
