# main.py
import sys

sys.path.append("src")
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from lightning_ml.core.utils.config import Config

cs = ConfigStore.instance()
cs.store(name="app_config", node=Config)


@hydra.main(version_base="1.1", config_name="config", config_path="cfg")
def main(cfg: Config) -> None:
    """Entry point for application."""
    print(f"Running with seed={cfg.seed}")
    print(f"Model: {cfg.model.name}")


if __name__ == "__main__":
    main()
