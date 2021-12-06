from pytorch_lightning.utilities.cli import LightningCLI
from ssrllib.data import datamodules
from ssrllib.models import modules
# TODO: Write calback to visualize images

# run=false is not needed, but used for consistency
# save_config_overwrite=True is needed to avoid errors
# when test overwrites fit config file
cli = LightningCLI(run=False, save_config_overwrite=True)

# Train model
cli.trainer.fit(cli.model, cli.datamodule)

# Test model
cli.trainer.test(cli.model, cli.datamodule,
                 ckpt_path=cli.trainer.checkpoint_callback.best_model_path)
