import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import time
import random
import copy
import os
from models import Models_GNN


def sample_from_model(model, args, batch_size,constraints,lambdas ):
    timing = dict()
    all_molecules_tensor = []
    # Sample molecules for this batch
    for i in range(args.n_sample):
        pass

        all_molecules_tensor.append( mol_generated)
        #generate a full molecule

    # calculating features / constraints over the molecule generated
    all_constraints_value = {}
    for constraint_name in constraints:
        fn = constraints[constraint_name]
        cur_value = []
        for i in range(batch_size):
            mol_generated =  all_molecules_tensor[i]
            #get the smiles conversion with rdkit
            smiles_generated = mol_generated.to_smiles()
            res = fn(smiles_generated)
            cur_values.append(res)
        cur_values = torch.cat(cur_values, dim=0)
        all_constraints_value[constraint_name] = cur_values.to(args.device)

    #coputing exponents for EBM
    lambda_dict = lambdas
    assert set(lambda_dict.keys()) == set(constraints.keys())
    constraint_names= all_constraints_value.keys()

    phi_tensor = torch.stack([all_constraints_value[constraint_name] for constraint_name in constraint_names], dim=1)
    lambda_tensor = torch.stack([lambda_dict[constraint_name] for constraint_name in constraint_names]).repeat(batch_size, 1).to(args.device)
    exponents = lambda_tensor.mul(phi_tensor).sum(dim=1)

    return game_data, timing, query_tensors, all_molecules_tensor, all_constraints_value, exponents



class GDCTrainer():
    #implement a finetuning of our odel (which is concatenation of 3 GNN) with a Gradient distributional policy program
    def get_sampling_model(self):
        return self.ref_model
    
    def get_policy_model(self):
        return self.model 
    
    def get_eval_model(self):
        return self.ref_model
    
    def __init__(self, args, constraints, q_update_criterion, q_update_interval, sampling_function, lr = 1e-4, batch_size = 128, dpg_epochs =4):
        # dpg_epochs is the number of optimization epochs per batch of samples
        self.sampling_function = sampling_function
        self.constraints = constraints

        self.lambdas = {k:0.0 for k in constraints}
        self.q_update_criterion = q_update_criterion
        assert self.q_update_criterion in ['interval', 'tvd', "kld"]
        # q_update_criterion can take one of the following values:
        # - 'interval': Update the GDP policy at regular intervals defined by q_update_interval.
        # - 'tvd': Update the GDP policy when the total variation distance between action probability distributions exceeds a threshold.
        # - 'kld': Update the GDP policy when the Kullback-Leibler divergence between action probability distributions exceeds a threshold.
        

        #to open
        self.model = Models_GNN(args)
        self.orig_model = copy.deepcopy(self.model)
        self.ref_model = copy.deepcopy(self.model)
        for p1, p2 in zip(self.ref_model.parameters(), self.orig_model.parameters()):
            p1.requires_grad= False
            p2.requires_grad= False

        # will hold all values of P(x) / q(x) for estimating TVD
        self.Z_moving_average = 0
        self.iter = 0
        self.min_kld = float("inf")
        self.min_tvd = float("inf")
        self.is_policy_eval = False

        self.q_update_interval = q_update_interval

        #### Compute lambdas
        self.compute_optimal_lambdas(sample_size=self.params["moment_matching_sample_size"])


    def compute_optimal_lambdas(self, sample_size=4096, n_iters=1000, lr=.5):
        """
        This performs the first step: Constraints --> EBM through self-normalized importance sampling. 
        Args:
            sample_size: total number of samples to use for lambda computation
        Returns:
            dicitonary of optimal lambdas per constraint: {'black': lambda_1, 'positive': lambda_2}
        """


        print("Computing Optimal Lambdas for desired moments...")

        min_nabla_lambda = 0.01
        max_n_iters = n_iters

        constraint_names = list(self.constraints.keys())
        mu_star = self.desired_moments

        mu_star = torch.tensor([mu_star[f] for f in constraint_names])
        lambdas = torch.tensor([self.lambdas[f] for f in constraint_names])

        # Collect sample_size samples for this:
        list_constraint_tensor = []
        list_model_input = []
        for i in  range(sample_size):
            _, _, atom, mols_generated, all_constraints_values, exponents = self.sampling_function(self.get_sampling_model(),
                    self.constraints,
                    self.lambdas)

            model_input = torch.cat((atom, mols_generated), axis=1)
            constraint_tensor = torch.stack([all_constraints_values[k] for k in all_constraints_values], dim=1) # B x F

            list_model_input.append(model_input)
            list_constraint_tensor.append(constraint_tensor)

        all_constraint_tensor = torch.cat(list_constraint_tensor, dim=0)  # [sample_sz x F]

        #### check for zero-occuring features. 
        # If a constraint has not occurred in your sample, no lambdas will be learned for that constraint, so we must check.

        for i, constraint  in enumerate(constraint_names):
            assert all_constraint_tensor[:, i].sum().item() > 0, "constraint {constraint} hasn't occurred in the samples, use a larger sample size"

        for step in range(max_n_iters):

            # 1. calculate P_over_q batch wise with current lambdas
            ## compute new exponents
            list_P_over_q = []
            for model_input, constraint_tensor in zip(list_model_input, list_constraint_tensor):
                exponents = lambdas.to(constraint_tensor.get_device()).mul(constraint_tensor).sum(dim=1) # N  ## compute new exponents
                P_over_q , _, _ ,_ = self.compute_rewards(exponents, model_input, mols_generated.shape[1]) # B TODO: use fbs for larger batches
                list_P_over_q.append(P_over_q)

            P_over_q = torch.cat(list_P_over_q, dim=0)

            # 2. compute mu (mean) of features given the current lambda using SNIS
            mu_lambda_numerator = P_over_q.view(1, -1).matmul(all_constraint_tensor ).squeeze(0) # F
            mu_lambda_denominator = P_over_q.sum()
            mu_lambda = mu_lambda_numerator / mu_lambda_denominator # F

            # 3. Update current Lambdas
            nabla_lambda = mu_star - mu_lambda.cpu()
            err = np.linalg.norm(nabla_lambda.cpu().numpy())
            print("step: %s \t ||nabla_lambda|| = %.6f" %(step, err))
            lambdas = lambdas + lr * nabla_lambda
            print("\tlambdas : {} ".format(self.lambdas))
            print("\tμ: {}".format(mu_lambda))
            print("\tμ*: {}".format(mu_star))

            for i, k in enumerate(constraint_names):
                self.lambdas[k] = lambdas[i].item()
            
            ## Check if error is less than tolerance, then break.
            if err < min_nabla_lambda: 
                break
        
    def compute_rewards(self, scores, model_input, gen_len):
        """
        Calculate P(x)/q(x) coefficient
        P(x) = a(x).b(target|x) the energy function
        a(x) = prob. of the sampled sequence by the original model (i.e. gpt-2 orig)
        b(x) = the output of the classifier (scores)
        q(x) = prob. of the sampled sequence by the reference/proposal policy.
        """

        # step1: calculate P(x) = a(x).b(target|x)
        # calculate a(x)


        orig_logits, _ , _ = self.orig_model(model_input)
        orig_logprob = logprobs_from_logits(orig_logits[:,:-1,:], model_input[:, 1:])

        # @todo: rethink if we should calculate log(a(x)) with or without the query
        orig_logprob = orig_logprob[:, -gen_len:] # (might be not necesary) we only keep prob regardless of the query

        # calculate b(x)
        orig_logprob = orig_logprob.detach() # we don't backprob to original model
        orig_logprob = torch.sum(orig_logprob, dim=-1) # log(a(x)) of shape [batchsize]

        assert scores.shape == orig_logprob.shape

        # we move all variables to the gpu of the policy to be trained "gpt2_device"
        scores = scores.to(self.params["gpt2_ref_device"])
        orig_logprob = orig_logprob.to(self.params["gpt2_ref_device"])


        log_P = orig_logprob.detach() + scores.detach() # Log P(x) = Log a(x)* e^{scores} = Log a(x) + scores

        # step2: calculate q(x)

        model_input = model_input.to(self.params["gpt2_ref_device"])
        ref_logits, _ , _ = self.ref_model(model_input)

        #q_prob = probs_from_logits(ref_logits[:,:-1,:], model_input[:, 1:])
        q_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], model_input[:, 1:])
        q_logprobs = q_logprobs[:, -gen_len:]
        q_logprobs = q_logprobs.detach() # do not backpropagate to q(x)

        q_logprobs = torch.sum(q_logprobs, dim=-1) # Log(q(x)) [Batch size]

        # final reward = exp(Log(P(x)) - Log(q(x)))
        P_over_q = torch.exp(log_P - q_logprobs)

        return P_over_q, log_P, q_logprobs, orig_logprob  # P/q , P(x), q(x), a(x)
    

    def loss(self, scores, query, response, model_input):
        """
        Calculates DPG loss on a given batch.
        L = (a(x) b(target|x) / q(x)) log pi(x)
        args:
            q_logprobs (torch.tensor): tensor containing logprobs shape [batch_size, response_length]
            response (torch.tensor): tensor containing response (continuation) token ids, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing b(x_1), b(x_2), .. shape [batch_size]
        returns:
            loss (torch.tensor) []: mean loss value across the batch
            stats (dict) : training statistics
        """
        



        
