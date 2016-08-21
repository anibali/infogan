--[[

"InfoGAN: Interpretable Representation Learning by Information Maximizing
 Generative Adversarial Nets"
  - http://arxiv.org/abs/1606.03657

--]]

package.path = package.path .. ';./src/?.lua'

require('torch')    -- Essential Torch utilities
require('image')    -- Torch image handling
require('nn')       -- Neural network building blocks
require('optim')    -- Optimisation algorithms
require('cutorch')  -- 'torch' on the GPU
require('cunn')     -- 'nn' on the GPU
require('cudnn')    -- Torch bindings to CuDNN

torch.setdefaulttensortype('torch.FloatTensor')

-- Set manual seeds for reproducible RNG
torch.manualSeed(1234)
cutorch.manualSeedAll(1234)
math.randomseed(1234)

local tnt = require('torchnet')
local nninit = require('nninit')
local pl = require('pl.import_into')()

local MnistDataset = require('MnistDataset')

--- CONSTANTS ---

-- Organisation of training examples during training
local n_epochs = 50
local n_updates_per_epoch = 100
local batch_size = 128

-- Total number of generator inputs
local n_gen_inputs = 74
-- Number of noise vars which will be treated as salient attributes
local n_salient_vars = 12

-- "lambda" from the InfoGAN paper
local info_regularisation_coefficient = 1.0

-- Learning rates for the Adam optimisers
local disc_learning_rate = 2e-4
local gen_learning_rate = 1e-3

assert(n_salient_vars >= 10 and n_salient_vars < n_gen_inputs)
local n_noise_vars = n_gen_inputs - n_salient_vars

--- DATA ---

local train_data = MnistDataset.new('data/mnist/train_32x32.t7')
local train_iter = train_data:make_iterator(batch_size)

--- MODEL ---

local Seq = nn.Sequential
local ReLU = cudnn.ReLU

local SpatBatchNorm = function(n_outputs)
  return nn.SpatialBatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 0.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local BatchNorm = function(n_outputs)
  return nn.BatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 0.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local function Conv(...)
  local conv = cudnn.SpatialConvolution(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)

  -- Use deterministic algorithms for convolution
  conv:setMode(
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')

  return conv
end

local function FullConv(...)
  local conv = cudnn.SpatialFullConvolution(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)

  -- Use deterministic algorithms for convolution
  conv:setMode(
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')

  return conv
end

local function LeakyReLU(leakiness, in_place)
  leakiness = leakiness or 0.01
  in_place = in_place == nil and true or in_place
  return nn.LeakyReLU(leakiness, in_place)
end

local function Linear(...)
  return nn.Linear(...)
    :init('weight', nninit.normal, 0.0, 0.02)
    :init('bias', nninit.constant, 0)
end

local generator = Seq()
  -- n_gen_inputs
  :add(Linear(n_gen_inputs, 1024))
  :add(BatchNorm(1024))
  :add(ReLU(true))
  -- 1024
  :add(Linear(1024, 128 * 7 * 7))
  :add(BatchNorm(128 * 7 * 7))
  :add(ReLU(true))
  :add(nn.Reshape(128, 7, 7))
  -- 128 x 7 x 7
  :add(FullConv(128, 64, 4,4, 2,2, 1,1))
  :add(SpatBatchNorm(64))
  :add(ReLU(true))
  -- 64 x 14 x 14
  :add(FullConv(64, 1, 4,4, 2,2, 1,1))
  :add(nn.Sigmoid())
  -- 1 x 28 x 28

local discriminator_body = Seq()
  -- 1 x 28 x 28
  :add(Conv(1, 64, 4,4, 2,2, 1,1))
  :add(LeakyReLU())
  -- 64 x 14 x 14
  :add(Conv(64, 128, 4,4, 2,2, 1,1))
  :add(SpatBatchNorm(128))
  :add(LeakyReLU())
  -- 128 x 7 x 7
  :add(nn.Reshape(128 * 7 * 7))
  :add(Linear(128 * 7 * 7, 1024))
  :add(BatchNorm(1024))
  :add(LeakyReLU())
  -- 1024

local discriminator_head = Seq()
  -- 1024
  :add(Linear(1024, 1))
  :add(nn.Sigmoid())
  -- 1

local info_head = Seq()
  -- 1024
  :add(Linear(1024, 128))
  :add(BatchNorm(128))
  :add(LeakyReLU())
  -- 128
  :add(Linear(128, n_salient_vars))
  -- n_salient_vars

local concat = nn.ConcatTable():add(nn.Narrow(2, 1, 10))
if n_salient_vars > 10 then
  concat:add(nn.Narrow(2, 11, n_salient_vars - 10))
end
info_head:add(concat)

local discriminator = Seq()
  :add(discriminator_body)
  :add(nn.ConcatTable()
    :add(discriminator_head)
    :add(info_head)
  )

generator:cuda()
discriminator:cuda()

--- CRITERIA ---

local disc_head_criterion = nn.BCECriterion()
local info_head_criterion = nn.ParallelCriterion()
  :add(nn.CrossEntropyCriterion())
if n_salient_vars > 10 then
  info_head_criterion:add(nn.MSECriterion())
end

disc_head_criterion:cuda()
info_head_criterion:cuda()

-- LOGGING --

local log_text = require('torchnet.log.view.text')

local log_keys = {'epoch', 'fake_loss', 'info_loss',
  'real_loss', 'gen_loss', 'time'}

local log = tnt.Log{
  keys = log_keys,
  onFlush = {
    log_text{
      keys = log_keys,
      format = {'epoch=%3d', 'fake_loss=%8.6f', 'info_loss=%8.6f',
        'real_loss=%8.6f', 'gen_loss=%8.6f', 'time=%5.2fs'}
    }
  }
}

--- TRAIN ---

local real_input = torch.CudaTensor()
local gen_input = torch.CudaTensor()
local fake_input = torch.CudaTensor()
local disc_target = torch.CudaTensor()

-- Flatten network parameters
local disc_params, disc_grad_params = discriminator:getParameters()
local gen_params, gen_grad_params = generator:getParameters()

-- Meters for gathering statistics to log
local fake_loss_meter = tnt.AverageValueMeter()
local info_loss_meter = tnt.AverageValueMeter()
local real_loss_meter = tnt.AverageValueMeter()
local gen_loss_meter = tnt.AverageValueMeter()
local time_meter = tnt.TimeMeter()

-- Creates targets for the salient part of the generator input
local function salient_input_to_target(tensor)
  local categorical = tensor:narrow(2, 1, 10)
  local max_vals, max_indices = categorical:max(2)
  if n_salient_vars > 10 then
    local continuous = tensor:narrow(2, 11, n_salient_vars - 10):clone()
    return {max_indices, continuous}
  else
    return {max_indices}
  end
end

-- Populates `res` such that each row contains a random one-hot vector.
-- That is, each row will be almost full of 0s, except for a 1 in a random
-- position.
local function random_one_hot(res)
  local batch_size = res:size(1)
  local n_categories = res:size(2)

  local probabilities = res.new(n_categories):fill(1 / n_categories)
  local indices = torch.multinomial(probabilities, batch_size, true):view(-1, 1)

  res:zero():scatter(2, indices, 1)
end

-- Calculate outputs and gradients for the discriminator
local do_discriminator_step = function(new_params)
  if new_params ~= disc_params then
    disc_params:copy(new_params)
  end

  disc_grad_params:zero()

  local batch_size = real_input:size(1)
  disc_target:resize(batch_size, 1)
  gen_input:resize(batch_size, n_gen_inputs)

  local loss_real = 0
  local loss_fake = 0
  local loss_info = 0

  -- Train with real images (from dataset)
  local dbodyout = discriminator_body:forward(real_input)

  local dheadout = discriminator_head:forward(dbodyout)
  disc_target:fill(1)
  loss_real = disc_head_criterion:forward(dheadout, disc_target)
  local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
  local dloss_ddheadin = discriminator_head:backward(dbodyout, dloss_ddheadout)
  discriminator_body:backward(real_input, dloss_ddheadin)

  -- Train with fake images (from generator)
  random_one_hot(gen_input:narrow(2, 1, 10))
  if n_salient_vars > 10 then
    gen_input:narrow(2, 11, n_salient_vars - 10):uniform(-1, 1)
  end
  gen_input:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)
  generator:forward(gen_input)
  fake_input:resizeAs(generator.output):copy(generator.output)
  local dbodyout = discriminator_body:forward(fake_input)

  local dheadout = discriminator_head:forward(dbodyout)
  disc_target:fill(0)
  loss_fake = disc_head_criterion:forward(dheadout, disc_target)
  local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
  local dloss_ddheadin = discriminator_head:backward(dbodyout, dloss_ddheadout)
  discriminator_body:backward(fake_input, dloss_ddheadin)

  local iheadout = info_head:forward(dbodyout)
  local info_target = salient_input_to_target(gen_input:narrow(2, 1, n_salient_vars))
  loss_info = info_head_criterion:forward(iheadout, info_target) * info_regularisation_coefficient
  local dloss_diheadout = info_head_criterion:backward(iheadout, info_target)
  for i, t in ipairs(dloss_diheadout) do
    t:mul(info_regularisation_coefficient)
  end
  local dloss_diheadin = info_head:backward(dbodyout, dloss_diheadout)
  discriminator_body:backward(fake_input, dloss_diheadin)

  -- Update average value meters
  real_loss_meter:add(loss_real)
  fake_loss_meter:add(loss_fake)
  info_loss_meter:add(loss_info)

  -- Calculate combined loss
  local loss = loss_real + loss_fake + loss_info

  return loss, disc_grad_params
end

-- Calculate outputs and gradients for the generator
local do_generator_step = function(new_params)
  if new_params ~= gen_params then
    gen_params:copy(new_params)
  end

  gen_grad_params:zero()

  disc_target:fill(1)
  local dheadout = discriminator_head.output
  local dbodyout = discriminator_body.output
  local gen_loss = disc_head_criterion:forward(dheadout, disc_target)
  local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
  local dloss_ddheadin = discriminator_head:updateGradInput(dbodyout, dloss_ddheadout)
  local dloss_dgout = discriminator_body:updateGradInput(fake_input, dloss_ddheadin + info_head.gradInput)
  gen_loss_meter:add(gen_loss)

  generator:backward(gen_input, dloss_dgout)

  return gen_loss, gen_grad_params
end

-- Discriminator optimiser
local disc_optimiser = {
  method = optim.adam,
  config = {
    learningRate = disc_learning_rate,
    beta1 = 0.5
  },
  state = {}
}

-- Generator optimiser
local gen_optimiser = {
  method = optim.adam,
  config = {
    learningRate = gen_learning_rate,
    beta1 = 0.5
  },
  state = {}
}

-- Takes a tensor containing many images and uses them as tiles to create one
-- big image. Assumes row-major order.
local function tile_images(images, rows, cols)
  local tiled = torch.Tensor(images:size(2), images:size(3) * rows, images:size(4) * cols)
  tiled:zero()
  for i = 1, math.min(images:size(1), rows * cols) do
    local col = (i - 1) % cols
    local row = math.floor((i - 1) / cols)
    tiled
      :narrow(2, row * images:size(3) + 1, images:size(3))
      :narrow(3, col * images:size(4) + 1, images:size(4))
      :copy(images[i])
  end
  return tiled
end

-- Constant noise for each row of the fake input visualisation
local constant_noise = torch.CudaTensor(math.floor(batch_size / 10), n_gen_inputs)
constant_noise:narrow(2, 1, 10):zero()
if n_salient_vars > 10 then
  constant_noise:narrow(2, 11, n_salient_vars - 10):uniform(-1, 1)
end
constant_noise:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)

local iter_inst = train_iter()

-- Training loop
for epoch = 1, n_epochs do
  fake_loss_meter:reset()
  info_loss_meter:reset()
  real_loss_meter:reset()
  gen_loss_meter:reset()
  time_meter:reset()

  -- Do training iterations for the epoch
  for iteration = 1, n_updates_per_epoch do
    local sample = iter_inst()

    if not sample or sample.input:size(1) < batch_size then
      -- Restart iterator
      iter_inst = train_iter()
      sample = iter_inst()
    end

    -- Copy real inputs from the dataset onto the GPU
    input = sample.input:narrow(3, 3, 28):narrow(4, 3, 28)
    real_input:resize(input:size()):copy(input)

    -- Update the discriminator network
    disc_optimiser.method(
      do_discriminator_step,
      disc_params,
      disc_optimiser.config,
      disc_optimiser.state
    )
    -- Update the generator network
    gen_optimiser.method(
      do_generator_step,
      gen_params,
      gen_optimiser.config,
      gen_optimiser.state
    )
  end

  -- Generate fake images. Noise varies across columns (horizontally),
  -- category varies across rows (vertically).
  local cols = constant_noise:size(1)
  for row = 1, 10 do
    local row_tensor = gen_input:narrow(1, 1 + (row - 1) * cols, cols)
    row_tensor:copy(constant_noise)

    local category = row
    for col = 1, cols do
      row_tensor[{col, category}] = 1
    end
  end
  generator:evaluate()
  local fake_images = tile_images(generator:forward(gen_input):float(), 10, 12)
  generator:training()

  -- Update log
  log:set{
    epoch = epoch,
    fake_loss = fake_loss_meter:value(),
    info_loss = info_loss_meter:value(),
    real_loss = real_loss_meter:value(),
    gen_loss = gen_loss_meter:value(),
    time = time_meter:value()
  }
  log:flush()

  -- Save outputs

  local model_dir = pl.path.join('out', 'models')
  pl.dir.makepath(model_dir)

  discriminator:clearState()
  local model_disc_file = pl.path.join(model_dir, 'infogan_mnist_disc.t7')
  torch.save(model_disc_file, discriminator)

  generator:clearState()
  local model_gen_file = pl.path.join(model_dir, 'infogan_mnist_gen.t7')
  torch.save(model_gen_file, generator)

  local image_dir = pl.path.join('out', 'images')
  pl.dir.makepath(image_dir)

  local image_basename = string.format('fake_images_%04d.png', epoch)
  image.save(pl.path.join(image_dir, image_basename), fake_images)

  -- Checkpoint the networks every 10 epochs
  if epoch % 10 == 0 then
    pl.file.copy(model_disc_file,
      pl.path.join(model_dir, string.format('infogan_mnist_disc_%04d.t7', epoch)))
    pl.file.copy(model_gen_file,
      pl.path.join(model_dir, string.format('infogan_mnist_gen_%04d.t7', epoch)))
  end
end
