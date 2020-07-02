from models.multi_VAE import MultiVAE
import torch.nn as nn
import numpy as np
import torch

"""The following tests are already covered by the pytorch module nn.ModuleList
      * Either of input_size, output_size, dropout, encoder or decoder being None
      * Dropout not being in [0, 1]
      * Input or output sizes not being positive integers
      * Forward pass not defined

   To perform the tests, cd into the models directory and run pytest in the terminal

   Requirements:
       pytest
       pyyaml
"""

# Model with dropout 0, to have a more predictable behaviour
# Default input and output sizes are 1000
model = MultiVAE(params='yaml_files/params_multi_VAE.yaml')
model.dropout = nn.Dropout(0)
model.initialize_model()

# Dropout doesn't need to be 0 for the other tests
# Default input and output sizes are 1000, dropout = 0.5
model1 = MultiVAE(params='yaml_files/params_multi_VAE.yaml')
model1.initialize_model()

# A 10 x 1000 tensor with all 1s
# Equivalent to an input with batch size 10 and input_size 1000
x1 = torch.tensor(np.ones((10, model.input_size), dtype=float))

# A 10 x 1000 tensor with all 0s
# Equivalent to an input with batch size 10 and input_size 1000
x0 = torch.tensor(np.zeros((10, model.no_latent_features), dtype=float))


# Forward pass and encoder return the same mean tensors for 0 dropout
def test_mean_same():
    a = (model.forward(x1.float()))[1]
    b = (model.encode(x1.float()))[0]
    assert(torch.all(a.eq(b)))


# Forward pass and encoder return the same log var tensors for 0 dropout
def test_logvar_same():
    a = (model.forward(x1.float()))[2]
    b = (model.encode(x1.float()))[1]
    assert(torch.all(a.eq(b)))


# The mean and log variance vectors are of the same length
def test_mean_logvar_length():
    r = model1.forward(x1.float())
    assert(len(r[1][0]) == len(r[2][0]))


# Output size is as expected
def test_output_size_check():
    r = model1.forward(x1.float())
    assert(len(r[0][0]) == model1.output_size)


# Reparamatrize function has an output of the correct size
def test_reparameterize_size():
    mean, logvar = model1.encode(x1.float())
    mean_new = model1.reparameterize(mean, logvar)
    assert(len(mean_new[0]) == model1.no_latent_features)


# decode function has output size as expected
def test_decode_size():
    out = model1.decode(x0.float())
    assert(len(out[0]) == model1.output_size)


# Testing the model with dictionary parameters
# Model with dropout 0, to have a more predictable behaviour
# Default input and output sizes are 1000
params = {
    'dropout': 0.5,
    'no_latent_features': 200,
    'norm_mean': 0.0,
    'norm_std': 0.001,
    'input_size': 1000,
    'output_size': 1000,
    'enc1_out': 600,
    'enc2_in': 600,
    'enc2_out': 400,
    'dec1_in': 200,
    'dec1_out': 600,
    'dec2_in': 600
}
model_dict = MultiVAE(params=params)
model_dict.dropout = nn.Dropout(0)
model_dict.initialize_model()

# Dropout doesn't need to be 0 for the other tests
# Default input and output sizes are 1000, dropout = 0.5
model1_dict = MultiVAE(params=params)
model1_dict.initialize_model()

# A 10 x 1000 tensor with all 1s
# Equivalent to an input with batch size 10 and input_size 1000
x1_dict = torch.tensor(np.ones((10, model_dict.input_size), dtype=float))

# A 10 x 1000 tensor with all 0s
# Equivalent to an input with batch size 10 and input_size 1000
x0_dict = torch.tensor(np.zeros((10, model_dict.no_latent_features), dtype=float))


# Forward pass and encoder return the same mean tensors for 0 dropout
def test_mean_same_dict():
    a = (model_dict.forward(x1_dict.float()))[1]
    b = (model_dict.encode(x1_dict.float()))[0]
    assert(torch.all(a.eq(b)))


# Forward pass and encoder return the same log var tensors for 0 dropout
def test_logvar_same_dict():
    a = (model_dict.forward(x1_dict.float()))[2]
    b = (model_dict.encode(x1_dict.float()))[1]
    assert(torch.all(a.eq(b)))


# The mean and log variance vectors are of the same length
def test_mean_logvar_length_dict():
    r = model1_dict.forward(x1_dict.float())
    assert(len(r[1][0]) == len(r[2][0]))


# Output size is as expected
def test_output_size_check_dict():
    r = model1_dict.forward(x1_dict.float())
    assert(len(r[0][0]) == model1_dict.output_size)


# Reparamatrize function has an output of the correct size
def test_reparameterize_size_dict():
    mean, logvar = model1_dict.encode(x1_dict.float())
    mean_new = model1_dict.reparameterize(mean, logvar)
    assert(len(mean_new[0]) == model1_dict.no_latent_features)


# decode function has output size as expected
def test_decode_size_dict():
    out = model1_dict.decode(x0_dict.float())
    assert(len(out[0]) == model1_dict.output_size)
