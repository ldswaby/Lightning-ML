import hydra
from hydra.core.config_store import ConfigStore

from lightning_ml.core.utils.config import Config

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Config) -> None:
    components = cfg.instantiate()
    dataset = components["data"]
    print(f"Loaded dataset: {type(dataset).__name__}, length={len(dataset)}")
    print(f"Seed: {cfg.seed}")


if __name__ == "__main__":
    main()
