import torch
import numpy as np
import os
import pytest

from dataloader.ae_data_handler import AEDataHandler
from models.multi_VAE import MultiVAE
from loss.vae_loss import VAELoss
from metric.recall_at_k import RecallAtK
from metric.revenue_at_k import RevenueAtK
from paretomanager.pareto_manager_class import ParetoManager
from validator import Validator
from trainer import Trainer
import torch.nn as nn
from torch.utils.data import DataLoader

# set cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create temporary directories
if not os.path.isdir('test_data_mo'):
    os.mkdir('test_data_mo')
if not os.path.isdir('test_data_mo/models'):
    os.mkdir('test_data_mo/models')

# generate random data
np.random.seed(42)
dir_path = 'test_data_mo/'
train_data_path = os.path.join(
    dir_path, 'movielens_small_training.npy')
validation_input_data_path = os.path.join(
    dir_path, 'movielens_small_validation_input.npy')
validation_output_data_path = os.path.join(
    dir_path, 'movielens_small_validation_test.npy')
test_input_data_path = os.path.join(
    dir_path, 'movielens_small_test_input.npy')
test_output_data_path = os.path.join(
    dir_path, 'movielens_small_test_test.npy')
products_data_path = os.path.join(
    dir_path, 'movielens_products_data.npy')


np.save(train_data_path, np.random.rand(10000, 8936).astype('float32'))
np.save(validation_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(validation_output_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_input_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(test_output_data_path, np.random.rand(2000, 8936).astype('float32'))
np.save(products_data_path, np.random.rand(8936))

dataHandler = AEDataHandler(
    'Testing trainer random dataset', train_data_path, validation_input_data_path,
    validation_output_data_path, test_input_data_path,
    test_output_data_path)

input_dim = dataHandler.get_input_dim()
output_dim = dataHandler.get_output_dim()

products_data_np = np.load(products_data_path)
products_data_torch = torch.tensor(
    products_data_np, dtype=torch.float32).to(device)

# create model
model = MultiVAE(params='yaml_files/params_multi_VAE.yaml')

correctness_loss = VAELoss()
revenue_loss = VAELoss(weighted_vector=products_data_torch)
losses = [correctness_loss, revenue_loss]

recallAtK = RecallAtK(k=10)
revenueAtK = RevenueAtK(k=10, revenue=products_data_np)
validation_metrics = [recallAtK, revenueAtK]

# Set up this
save_to_path = 'test_data_mo/models'
yaml_path = 'yaml_files/trainer_params.yaml'


# test the init arguments
def test_check_input1():
    with pytest.raises(TypeError, match='Please check you are using the right data handler object,'
                       + ' or the right order of the attributes!'):
        trainer = Trainer(None, model, losses, validation_metrics, save_to_path, yaml_path)
        trainer.train()
    with pytest.raises(TypeError, match='Please check you are using the right data handler object,'
                       + ' or the right order of the attributes!'):
        trainer = Trainer(model, dataHandler, losses, validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input2():
    with pytest.raises(TypeError, match='Please check you are using the right model object,'
                       + ' or the right order of the attributes!'):
        trainer = Trainer(dataHandler, None, losses, validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input3():
    class TestModel(nn.Module):
        def forward(self):
            return 1
    with pytest.raises(TypeError, match='Please check if your models has initialize_model\\(\\) method defined!'):
        trainer = Trainer(dataHandler, TestModel(), losses, validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input4():
    with pytest.raises(TypeError, match='Please check you are using the right loss objects,'
                       + ' or the right order of the attributes!'):
        losses_tmp = losses.copy()
        losses_tmp[0] = validation_metrics[0]
        trainer = Trainer(dataHandler, model, losses_tmp, validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input5():
    with pytest.raises(TypeError, match='Please check you are using the right metric objects,'
                       + ' or the right order of the attributes!'):
        validation_metrics_tmp = validation_metrics.copy()
        validation_metrics_tmp[0] = model
        trainer = Trainer(dataHandler, model, losses, validation_metrics_tmp, save_to_path, yaml_path)
        trainer.train()


def test_check_input6():
    with pytest.raises(ValueError, match='Please make sure that the directory where you want'
                       + ' to save the models is empty!'):
        trainer = Trainer(dataHandler, model, losses, validation_metrics, '.', yaml_path)
        trainer.train()


def test_check_input7():
    # check for None losses
    with pytest.raises(ValueError, match='The losses are None, please make sure to give valid losses!'):
        trainer = Trainer(dataHandler, model, None, validation_metrics, save_to_path, yaml_path)
        trainer.train()
    # check the legnth of Losses at least 2
    losses_tmp = []
    losses_tmp.append(losses[0])
    with pytest.raises(ValueError, match='Please check you have defined at least two losses,'
                       + ' for training with one loss use the Single Objective Loss class!'):
        trainer = Trainer(dataHandler, model, losses_tmp, validation_metrics, save_to_path, yaml_path)
        trainer.train()


def test_check_input8():
    # check for None metrics
    with pytest.raises(ValueError, match='The validation_metrics are None,'
                       + ' please make sure to give valid validation_metrics!'):
        trainer = Trainer(dataHandler, model, losses, None, save_to_path, yaml_path)
        trainer.train()
    # check if length is at least 1
    validation_metrics_tmp = []
    with pytest.raises(ValueError, match='Please check you have defined at least one validation metric!'):
        trainer = Trainer(dataHandler, model, losses, validation_metrics_tmp, save_to_path, yaml_path)
        trainer.train()


def test_check_input9():
    with pytest.raises(TypeError, match='Please make sure that the optimizer is a pytorch Optimizer object!'):
        trainer = Trainer(dataHandler, model, losses, validation_metrics, save_to_path, yaml_path, model)
        trainer.train()


# test the reading from the yaml files
def test_read_yaml_params():
    trainer = Trainer(dataHandler, model, losses, validation_metrics, save_to_path, yaml_path)
    assert trainer.seed == 42
    assert trainer.normalize_gradients is True
    assert trainer.learning_rate == 1e-3
    assert trainer.batch_size_training == 500
    assert trainer.shuffle_training is True
    assert trainer.drop_last_batch_training is True
    assert trainer.batch_size_validation == 500
    assert trainer.shuffle_validation is True
    assert trainer.drop_last_batch_validation is False
    assert trainer.number_of_epochs == 50
    assert trainer.frank_wolfe_max_iter == 100
    assert trainer.anneal is True
    assert trainer.beta_start == 0
    assert trainer.beta_cap == 0.3
    assert trainer.beta_step == 0.3/10000


# test the init of the objects
def test_init_objects():
    trainer = Trainer(dataHandler, model, losses, validation_metrics, save_to_path, yaml_path)
    assert type(trainer._train_dataloader) == DataLoader
    assert type(trainer.pareto_manager) == ParetoManager
    assert trainer.pareto_manager.path == save_to_path
    assert type(trainer.validator) == Validator
    assert len(trainer.max_empirical_losses) == 2


# removing generated data
def test_cleanup():
    os.remove(train_data_path)
    os.remove(validation_input_data_path)
    os.remove(validation_output_data_path)
    os.remove(test_input_data_path)
    os.remove(test_output_data_path)
    os.remove(products_data_path)
    os.rmdir('test_data_mo/models')
    os.rmdir('test_data_mo')
