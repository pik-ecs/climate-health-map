import typer
from .r_tuning import tune
from .slurm import app as slurm_app
from .r_train import train
from .q_tuning import main as tuning_quality
from .q_train import main as training_quality
# from .r_classify import predict

app = typer.Typer()
app.command('tuning', help='Run hyper-parameter tuning and store best parameters and statistics')(tune)
app.command('train', help='Using a model-info file, train and store a classifier')(train)
app.command('tuning-quality', help='Produce summary statistics of tuning qualities')(tuning_quality)
app.command('training-quality', help='Produce summary statistics of tuning qualities')(training_quality)
# app.command('classify', help='Apply a fitted classifier to a dataset')(predict)

# This adds all the slurmify-commands for tuning, training, classification
app.add_typer(slurm_app)
