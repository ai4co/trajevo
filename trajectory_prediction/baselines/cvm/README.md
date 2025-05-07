# Constant Velocity Model

From https://arxiv.org/abs/1903.07933

Original code: https://github.com/cschoeller/constant_velocity_pedestrian_motion/blob/master/evaluate.py

(Note: the results from that paper are not correct compared to other methods!)


## Usage


(from the `trajectory_prediction`)


To run it, you can use the following command (for CVM - no sample):
```bash
python eval.py --code_path baselines/cvm/model.py --code_function predict_trajectory --test --code_args "{'sample':False}" --test --dataset eth
```

With sample:
```bash
python eval.py --code_path baselines/cvm/model.py --code_function predict_trajectory --code_args "{'sample':True}" --test --dataset eth
```

(you should change the dataset to other four datasets)
