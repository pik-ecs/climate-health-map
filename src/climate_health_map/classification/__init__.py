import typer
from .r_tuning import tune
from .slurm import app as slurm_app
from .r_train import train
# from .r_classify import predict

app = typer.Typer()
app.command('tune', help='Run hyper-parameter tuning and store best parameters and statistics')(tune)
app.command('train', help='Using a model-info file, train and store a classifier')(train)
# app.command('classify', help='Apply a fitted classifier to a dataset')(predict)

app.add_typer(slurm_app)
