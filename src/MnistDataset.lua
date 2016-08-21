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
  {name = 'n_threads', type = 'number', default = 8},
  call = function(self, batch_size, n_threads)
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

    return tnt.ParallelDatasetIterator{
      ordered = true,
      nthread = n_threads,
      closure = function()
        local tnt = require('torchnet')

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
