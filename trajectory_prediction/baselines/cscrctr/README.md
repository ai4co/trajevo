# CSCRCTR


## Usage


(from the `trajectory_prediction`)
```bash
cd trajectory_prediction
```

To run it, you can use the following command

must use sample:
```bash
python eval.py --code_path baselines/CSCRCTR/model.py --code_function predict_trajectory --code_args "{'sample':True}" --test --dataset eth
```

(you should change the dataset to other four datasets)
