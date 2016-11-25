local autograd = require('autograd')
require('nn')

local MutualInformationCriterion, Parent = torch.class('pdist.MutualInformationCriterion', 'nn.Criterion', require('pdist.env'))

function MutualInformationCriterion.make_loss_function(dist)
  return function(p, target)
    local out_params = dist:shape_params(p.input)

    -- Lower bound estimate of H(c|G(z,c))
    local out_entropy = torch.mean(dist:nll(target, out_params))
    -- H(c)
    local prior_entropy = torch.mean(dist:nll(target, dist.prior_params))

    -- L_I(G,Q)
    local mutual_information = prior_entropy - out_entropy

    if type(prior_entropy) ~= 'table' then
      assert(prior_entropy == prior_entropy, 'nan found here')
    end

    if type(mutual_information) ~= 'table' then
      assert(mutual_information == mutual_information, 'nan found here')
    end

    -- Return the negative MI so that performing gradient descent to find the
    -- minimum will maximize MI
    return -mutual_information
  end
end

function MutualInformationCriterion:__init(dist)
  Parent.__init(self)

  self.func = autograd(self.make_loss_function(dist))
end

function MutualInformationCriterion:updateOutput(input, target)
  local grads, output = self.func({input = input}, target)

  self.output = output
  self.gradInput = grads.input

  return self.output
end

function MutualInformationCriterion:updateGradInput(input, target)
  return self.gradInput
end

return MutualInformationCriterion
