local argcheck = require('argcheck')

local function view_and_expand(t, size)
  local s = t:size():totable()
  while #s < size:size() do
    table.insert(s, 1, 1)
  end
  return torch.reshape(t, unpack(s)):expand(unpack(size:totable()))
end

local Gaussian, Parent = torch.class('pdist.Gaussian', 'pdist.Distribution', require('pdist.env'))

Gaussian.__init = argcheck{
  {name = 'self', type = 'pdist.Gaussian'},
  {name = 'n', type = 'number'},
  {name = 'mean', type = 'torch.*Tensor'},
  {name = 'stddev', type = 'torch.*Tensor'},
  {name = 'fixed_stddev', type = 'boolean', default = 'false'},
  call = function(self, n, mean, stddev, fixed_stddev)
    local prior_params = {
      mean = mean,
      stddev = stddev
    }

    self.n = n
    self.prior_params = prior_params
    self.fixed_stddev = fixed_stddev
  end
}

Gaussian.n_vars = argcheck{
  {name = 'self', type = 'pdist.Gaussian'},
  call = function(self)
    return self.n
  end
}

Gaussian.n_params = argcheck{
  {name = 'self', type = 'pdist.Gaussian'},
  call = function(self)
    local n_params = self.n
    if not self.fixed_stddev then
      n_params = n_params + self.n
    end
    return n_params
  end
}

Gaussian.shape_params = function(self, flat_params)
  local shaped_params = {}

  if flat_params:dim() == 1 then
    flat_params = torch.view(flat_params, 1, flat_params:size(1))
  end

  if flat_params:dim() == 2 then
    shaped_params.mean = torch.narrow(flat_params, 2, 1, self.n)
    if self.fixed_stddev then
      shaped_params.stddev = torch.add(torch.mul(shaped_params.mean, 0), 1)
    else
      shaped_params.stddev = torch.sqrt(torch.exp(torch.narrow(flat_params, 2, self.n + 1, self.n)))
    end
  else
    error('wrong dimensions')
  end

  return shaped_params
end

Gaussian.sample = argcheck{
  {name = 'self', type = 'pdist.Gaussian'},
  {name = 'dest', type = 'torch.*Tensor'},
  {name = 'params', type = 'table'},
  call = function(self, dest, params)
    local mean = view_and_expand(params.mean, dest:size())
    local stddev = view_and_expand(params.stddev, dest:size())

    for i = 1, dest:size(1) do
      for j = 1, dest:size(2) do
        dest:narrow(1, i, 1):narrow(2, j, 1):normal(mean[{i, j}], stddev[{i, j}])
      end
    end

    return dest
  end
}

local half_log_2pi = 0.5 * torch.log(2 * math.pi)

Gaussian.nll = function(self, vals, params)
  if vals:dim() == 1 then
    vals = torch.view(vals, 1, vals:size(1))
  end

  if vals:dim() == 2 then
    local mean = view_and_expand(params.mean, vals:size())
    local stddev = view_and_expand(params.stddev, vals:size())

    local safe_stddev = torch.add(stddev, 1e-8)

    local tmp = torch.pow(torch.cdiv(vals - mean, safe_stddev), 2) / 2
      + torch.log(safe_stddev) + half_log_2pi

    return torch.sum(tmp, 2)
  else
    error('wrong dimensions')
  end
end

return Gaussian
