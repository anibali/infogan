require('cudnn')
local nninit = require('nninit')

local model_builder = {}

local Seq = nn.Sequential
local ReLU = cudnn.ReLU

local function SpatBatchNorm(n_outputs)
  return nn.SpatialBatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 1.0, 0.02) -- Gamma
    :init('bias', nninit.constant, 0)         -- Beta
end

local function BatchNorm(n_outputs)
  return nn.BatchNormalization(n_outputs, 1e-5, 0.1)
    :init('weight', nninit.normal, 1.0, 0.02) -- Gamma
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

function model_builder.build_infogan(n_gen_inputs, n_salient_params)
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
    :add(Linear(128, n_salient_params))
    -- n_salient_params

  return generator, discriminator_body, discriminator_head, info_head
end

return model_builder
