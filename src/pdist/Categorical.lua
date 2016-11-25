local argcheck = require('argcheck')

local function view_and_expand(t, size)
  local s = t:size():totable()
  while #s < size:size() do
    table.insert(s, 1, 1)
  end
  return torch.reshape(t, unpack(s)):expand(unpack(size:totable()))
end

local Categorical, Parent = torch.class('pdist.Categorical', 'pdist.Distribution', require('pdist.env'))

Categorical.__init = argcheck{
  {name = 'self', type = 'pdist.Categorical'},
  {name = 'n', type = 'number'},
  {name = 'probs', type = 'torch.*Tensor'},
  call = function(self, n, probs)
    local prior_params = {
      probs = probs
    }

    self.n = n
    self.prior_params = prior_params
  end
}

Categorical.n_vars = argcheck{
  {name = 'self', type = 'pdist.Categorical'},
  call = function(self)
    return self.n
  end
}

Categorical.n_params = argcheck{
  {name = 'self', type = 'pdist.Categorical'},
  call = function(self)
    return self.n
  end
}

Categorical.shape_params = function(self, flat_params)
  local shaped_params = {}

  if flat_params:dim() == 1 then
    flat_params = torch.view(flat_params, 1, flat_params:size(1))
  end

  if flat_params:dim() == 2 then
    local exp_z = torch.exp(flat_params)
    local denom = torch.expandAs(torch.sum(exp_z, 2), exp_z)

    shaped_params.probs = torch.cdiv(exp_z, denom)
  else
    error('wrong dimensions')
  end

  return shaped_params
end

Categorical.sample = argcheck{
  {name = 'self', type = 'pdist.Categorical'},
  {name = 'dest', type = 'torch.*Tensor'},
  {name = 'params', type = 'table'},
  call = function(self, dest, params)
    local probs = view_and_expand(params.probs, dest:size())

    dest:zero()

    for i = 1, dest:size(1) do
      local indices = torch.multinomial(probs[i], 1, true):view(-1, 1)
      dest:narrow(1, i, 1):scatter(2, indices, 1)
    end

    return dest
  end
}

Categorical.nll = function(self, vals, params)
  if vals:dim() == 1 then
    vals = torch.view(vals, 1, vals:size(1))
  end

  if vals:dim() == 2 then
    local probs = view_and_expand(params.probs, vals:size())

    local tmp = -torch.cmul(torch.log(probs + 1e-8), vals)

    return torch.sum(tmp, 2)
  else
    error('wrong dimensions')
  end
end

return Categorical
