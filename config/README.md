Example Hydra configuration files demonstrating how to build components using the registries.

The files in this folder can be loaded with
`lightning_ml.core.utils.registry.instantiate_from_yaml`.

```
from lightning_ml.core.utils.registry import instantiate_from_yaml

# Build dataset defined in dataset/labelled.yaml
dataset = instantiate_from_yaml("config/dataset/labelled.yaml")
```
