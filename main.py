# # main.py
# import sys

# sys.path.append("src")
# import hydra
# from hydra.core.config_store import ConfigStore
# from omegaconf import DictConfig

# from lightning_ml.core.utils.config import Config

# cs = ConfigStore.instance()
# cs.store(name="app_config", node=Config)


# @hydra.main(version_base="1.1", config_name="config", config_path="cfg")
# def main(cfg: Config) -> None:
#     """Entry point for application."""
#     print(f"Running with seed={cfg.seed}")
#     print(f"Model: {cfg.model.name}")


# if __name__ == "__main__":
#     main()


# main.py
import sys

import hydra
from hydra.core.config_store import ConfigStore

from lightning_ml.core.utils.config import Config
from lightning_ml.core.utils.enums import Registries

sys.path.append("src")


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    # instantiate data components
    data_components = cfg.data.instantiate()
    loader = data_components["loader"]
    dataset = data_components["dataset"]
    # instantiate model and optimizer
    print(loader)
    print(cfg.optimizer)


if __name__ == "__main__":
    main()
