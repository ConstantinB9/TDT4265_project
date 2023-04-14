from train import train
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

def objective(config):
    return train(config)

if __name__ == "__main__":
    search_space= {
        "model":tune.choice([ "yolov8n.pt"]),
        "epochs":tune.choice([60]),
        "image_size": tune.choice([640]),
        "optimizer": tune.choice(['SGD', 'Adam', 'AdamW', 'RMSProp']),
        "cos_lr": tune.choice([True, False]),
        "lr0": tune.uniform(1e-4, 1e-2),
        "lrf": tune.uniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.5, 1 - 1e-5),
        "weight_decay": tune.uniform(0, 1e-3),
        "box": tune.uniform(1, 10),
        "cls": tune.uniform(0.1, 10),
        "dfl": tune.uniform(0.1, 10)
    }
    ray.init(object_store_memory=10**9, num_cpus=8, num_gpus=1, )
    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="metrics/mAP50(B)",
            mode="max",
            search_alg=OptunaSearch(),
            max_concurrent_trials=1
        )
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)