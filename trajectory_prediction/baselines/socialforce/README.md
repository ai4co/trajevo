# PySocialForce

Original code: https://github.com/yuxiang-gao/PySocialForce



## Usage


(from the `trajectory_prediction`)


To run it, you can use the following command (- no sample):
```bash
python eval.py --code_path baselines/social_force/model.py --code_function predict_trajectory --code_args "{'sample':False}" --test --dataset eth
```

With sample:
```bash
python eval.py --code_path baselines/social_force/model.py --code_function predict_trajectory --code_args "{'sample':True}" --test --dataset eth
```

(you should change the dataset to other four datasets)
