# 🚗 TrajEvo

### About the data

Let's use the public trajectory benchmark dataset for easier comparision with baselines. This ETH/UCY trajectory data is one of the most widely used dataset.

You can check the data loading process in `1-eth-dataset-info.ipynb`.

### About the baselines

I took the baselines value from the Trajectron++ paper for the ETH dataset. Later we could also do it for other four datasets.

Check the `baselines.py` for the comparison calculation function and `2-eth-compare-with-baselines.ipynb` for the comparison with baselines.

I temporarily uploaded the `gpt.py` for your reference as this is a private repo.

### About using TrajEvo

Use the following command to train:
```bash
python main.py problem=trajectory_prediction init_pop_size=4 pop_size=4 max_fe=20 timeout=20 llm_client=azure
```



## Baselines

### CVM

For CVM, see [baselines/cvm/README.md](baselines/cvm/README.md).


### Others
TODO
