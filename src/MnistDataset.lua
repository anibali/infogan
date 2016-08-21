-- Data loader for the Torch packaging of the MNIST dataset, which can
-- be downloaded from https://s3.amazonaws.com/torch7/data/mnist.t7.tgz

local tnt = require('torchnet')
local argcheck = require('argcheck')

local MnistDataset = torch.class('MnistDataset', {})

MnistDataset.__init = argcheck{
  {name = 'self', type = 'MnistDataset'},
  {name = 'data_file', type = 'string'},
  call = function(self, data_file)
    local raw_data = torch.load(data_file, 'ascii')
    self.inputs = raw_data.data:float():div(255)
    self.targets = raw_data.labels
  end
}

MnistDataset.make_iterator = argcheck{
  {name = 'self', type = 'MnistDataset'},
  {name = 'batch_size', type = 'number', default = 32},
  {name = 'subset', type = 'string', default = 'all'},
  {name = 'n_threads', type = 'number', default = 8},
  call = function(self, batch_size, subset, n_threads)
    local inputs = self.inputs
    local targets = self.targets

    local function load_example_from_index(index)
      return {
        input = inputs[index],
        target = targets:narrow(1, index, 1)
      }
    end

    local gen = torch.Generator()
    torch.manualSeed(gen, 1234)
    local indices = torch.randperm(gen, inputs:size(1)):long()
    if subset == 'training' then
      indices = indices:narrow(1, 1, math.floor(0.8 * indices:size(1)))
    elseif subset == 'validation' then
      indices = indices:narrow(1, math.floor(0.8 * indices:size(1)) + 1,
        indices:size(1) - math.floor(0.8 * indices:size(1)))
    elseif subset == 'all' then
      -- No need to narrow indices
    else
      error('unrecognised subset: ' .. subset)
    end

    return tnt.ParallelDatasetIterator{
      ordered = true,
      nthread = n_threads,
      closure = function()
        local tnt = require('torchnet')
        torch.manualSeed(1234)

        return tnt.BatchDataset{
          batchsize = batch_size,
          dataset = tnt.ListDataset{
            list = indices,
            load = load_example_from_index
          }
        }
      end
    }
  end
}

return MnistDataset
