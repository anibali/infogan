local argcheck = require('argcheck')

local Hybrid, Parent = torch.class('pdist.Hybrid', 'pdist.Distribution', require('pdist.env'))

Hybrid.__init = argcheck{
  {name = 'self', type = 'pdist.Hybrid'},
  call = function(self)
    self.children = {}
    self.prior_params = {}
  end
}

Hybrid.shape_params = function(self, flat_params)
  local shaped_params = {}

  if flat_params:dim() == 1 then
    flat_params = torch.view(flat_params, 1, flat_params:size(1))
  end

  if flat_params:dim() == 2 then
    local pos = 1

    for i = 1, #self.children do
      shaped_params[i] = self.children[i]:shape_params(
        torch.narrow(flat_params, 2, pos, self.children[i]:n_params()))

      pos = pos + self.children[i]:n_params()
    end
  else
    error('wrong dimensions')
  end

  return shaped_params
end

Hybrid.add = argcheck{
  {name = 'self', type = 'pdist.Hybrid'},
  {name = 'dist', type = 'pdist.Distribution'},
  call = function(self, dist)
    local i = #self.children + 1
    self.children[i] = dist
    self.prior_params[i] = dist.prior_params

    return self
  end
}

Hybrid.n_vars = argcheck{
  {name = 'self', type = 'pdist.Hybrid'},
  call = function(self)
    local n = 0

    for i = 1, #self.children do
      n = n + self.children[i]:n_vars()
    end

    return n
  end
}

Hybrid.n_params = argcheck{
  {name = 'self', type = 'pdist.Hybrid'},
  call = function(self)
    local n = 0

    for i = 1, #self.children do
      n = n + self.children[i]:n_params()
    end

    return n
  end
}

Hybrid.sample = argcheck{
  {name = 'self', type = 'pdist.Hybrid'},
  {name = 'dest', type = 'torch.*Tensor'},
  {name = 'params', type = 'table'},
  call = function(self, dest, params)
    local pos = 1
    for i = 1, #self.children do
      self.children[i]:sample(dest:narrow(2, pos, self.children[i]:n_vars()), params[i])
      pos = pos + self.children[i]:n_vars()
    end
    return dest
  end
}

Hybrid.nll = function(self, vals, params)
  local pos = 1

  if vals:dim() == 1 then
    vals = torch.view(vals, 1, vals:size(1))
  end

  if vals:dim() == 2 then
    local total = nil

    for i = 1, #self.children do
      local tmp = self.children[i]:nll(
        torch.narrow(vals, 2, pos, self.children[i]:n_vars()),
        params[i])
      if i == 1 then
        total = tmp
      else
        total = torch.add(total, tmp)
      end
      pos = pos + self.children[i]:n_vars()
    end

    return total
  else
    error('wrong dimensions')
  end
end

return Hybrid
