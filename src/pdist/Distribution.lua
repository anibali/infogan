local argcheck = require('argcheck')

local Distribution = torch.class('pdist.Distribution', require('pdist.env'))

Distribution.n_vars = argcheck{
  {name = 'self', type = 'pdist.Distribution'},
  call = function(self)
    error('not implemented')
  end
}

Distribution.n_params = argcheck{
  {name = 'self', type = 'pdist.Distribution'},
  call = function(self)
    error('not implemented')
  end
}

Distribution.shape_params = function(self, flat_params)
  error('not implemented')
end

Distribution.sample = argcheck{
  {name = 'self', type = 'pdist.Distribution'},
  {name = 'dest', type = 'torch.*Tensor'},
  {name = 'params', type = 'table'},
  call = function(self, dest, params)
    error('not implemented')
  end
}

Distribution.nll = function(self, vals, params)
  error('not implemented')
end

return Distribution
