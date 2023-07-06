import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import time
import random
import copy
import os
from models import Models_GNN
from DataPipeline.preprocessing import node_encoder
import torch_geometric
from torch_geometric.data import Data, Batch



 
class GDCTrainer_path():
    #implement a finetuning of our odel (which is concatenation of 3 GNN) with a Gradient distributional policy program
    def get_sampling_model(self):
        return self.ref_model
    
    def get_policy_model(self):
        return self.model 
    
    def get_eval_model(self):
        return self.ref_model
    
    def __init__(self, args, features, desired_moments, q_update_criterion, q_update_interval, sampling_function, lr = 1e-4, batch_size = 128, dpg_epochs =4,  min_nabla_lambda = 0.01):
        # dpg_epochs is the number of optimization epochs per batch of samples
        self.sampling_function = sampling_function #generate a batch of paths
        self.features = features

        self.lambdas = {k:0.0 for k in features}
        self.q_update_criterion = q_update_criterion
        self.batch_size = batch_size
        self.desired_moments = desired_moments
        self.min_nabla_lambda =  min_nabla_lambda      # difference betweeb the wanted mu and approximated mu 

        assert self.q_update_criterion in ['interval', 'tvd', "kld"]
        # q_update_criterion can take one of the following values:
        # - 'interval': Update the GDP policy at regular intervals defined by q_update_interval.
        # - 'tvd': Update the GDP policy when the total variation distance between action probability distributions exceeds a threshold.
        # - 'kld': Update the GDP policy when the Kullback-Leibler divergence between action probability distributions exceeds a threshold.
        

        #to open
        self.model = Models_GNN(args) # it is pi
        if args.keku:
            self.edge_size = 3
        else:
            self.edge_size = 4
        self.orig_model = copy.deepcopy(self.model) # it is a 
        self.ref_model = copy.deepcopy(self.model) # it is q
        for p1, p2 in zip(self.ref_model.parameters(), self.orig_model.parameters()):
            p1.requires_grad= False #crucial for not computing the gradient of the reference model which not be impacted i te loss
            p2.requires_grad= False

        # will hold all values of P(x) / q(x) for estimating TVD
        self.Z_moving_average = 0
        self.iter = 0
        self.min_kld = float("inf")
        self.min_tvd = float("inf")
        self.is_policy_eval = False

        self.q_update_interval = q_update_interval

        #### Compute lambdas
        self.compute_optimal_lambdas()


    def compute_optimal_lambdas(self, sample_size=4096, n_iters=1000, lr=.5): #how do they define the learning rate and sample size maybe do more
        """
        This performs the first step: Constraints --> EBM through self-normalized importance sampling. 
        Args:
            sample_size: total number of samples to use for lambda computation
        Returns:
            dicitonary of optimal lambdas per constraint: {'black': lambda_1, 'positive': lambda_2}
        """


        print("Computing Optimal Lambdas for desired moments...")

        max_n_iters = n_iters

        feature_names = list(self.features.keys())
        mu_star = self.desired_moments #name mu_bar in the pseudo code

        mu_star = torch.tensor([mu_star[f] for f in feature_names])
        lambdas = torch.tensor([self.lambdas[f] for f in feature_names])

        # Collect sample_size samples for this:
        list_feature_tensor = []
        for i in  range(sample_size):
            #if we do multi processing, i think here the best 
            #put pi without grad for lambdas
            batch_features_values, exponents, a_value, q_value, pi_value   = self.sampling_function(self.get_sampling_model(),
                                                                                                self.get_eval_model(),
                                                                                                self.get_policy_model(), 
                                                                                                self.features, 
                                                                                                self.lambdas,
                                                                                                self.batch_size)

            #model_input = torch.cat((atom, mols_generated), axis=1)
            feature_tensor = torch.stack([batch_features_values[k] for k in batch_features_values], dim=1) # B x F, du coup c'est censé etre batch_size x nb_features
            
            list_feature_tensor.append(feature_tensor)

        all_feature_tensor = torch.cat(list_feature_tensor, dim=0)  # [sample_sz*size_batch x F]

        #### check for zero-occuring features. 
        # If a constraint has not occurred in your sample, no lambdas will be learned for that constraint, so we must check.

        for i, feature  in enumerate(feature_names):
            assert all_feature_tensor[:, i].sum().item() > 0, "constraint {feature} hasn't occurred in the samples, use a larger sample size"

        for step in range(max_n_iters): #SGD for finding lambdas

            # 1. calculate P_over_q batch wise with current lambdas which will be name w
            ## compute new exponents
            list_w = []
            for feature_tensor in (list_feature_tensor):
                exponents = lambdas.to(feature_tensor.get_device()).mul(feature_tensor).sum(dim=1) # N  ## compute new exponents, do scalar product
                #w is equall to putting the exponents in one exponential 
                w = torch.exp(exponents) # N
                list_w.append(w)

            w = torch.cat(w, dim=0)

            # 2. compute mu (mean) of features given the current lambda using SNIS
            mu_lambda_numerator = w.view(1, -1).matmul(all_feature_tensor ).squeeze(0) # F
            mu_lambda_denominator = w.sum()
            mu_lambda = mu_lambda_numerator / mu_lambda_denominator # F

            # 3. Update current Lambdas
            nabla_lambda = mu_star - mu_lambda.cpu()
            err = np.linalg.norm(nabla_lambda.cpu().numpy())
            print("step: %s \t ||nabla_lambda|| = %.6f" %(step, err))
            lambdas = lambdas + lr * nabla_lambda
            print("\tlambdas : {} ".format(self.lambdas))
            print("\tμ: {}".format(mu_lambda))
            print("\tμ*: {}".format(mu_star))

            for i, k in enumerate(feature):
                self.lambdas[k] = lambdas[i].item()
            
            ## Check if error is less than tolerance, then break.
            if err < self.min_nabla_lambda: 
                break
    

    def step(self,number_samples):
        train_stats = {}
        P_over_q = []
        P_over_pi = []
        pi_over_q = []
        mean_loss = 0
        for k in range (number_samples):
            loss = 0
            #if we do multi processing, i think here the best
            batch_features_values, exponents, a_value, q_value, pi_value = sample_path_from_model(self.get_sampling_model(),
                                                                                                self.get_eval_model(),
                                                                                                self.get_policy_model(), 
                                                                                                self.features, 
                                                                                                self.lambdas,
                                                                                                self.batch_size)
            P_over_q.append(torch.flatten(a_value * exponents / q_value))
            P_over_pi.append(torch.flatten(a_value * exponents / pi_value))
            pi_over_q.append(torch.flatten(pi_value / q_value))

            # compute the loss to trained the model
            loss += torch.sum(a_value*exponents / q_value * torch.log(pi_value))
            loss = loss /(self.batch_size)
            loss.backward()
            mean_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        mean_loss = mean_loss/number_samples


        P_over_q = torch.flatten(torch.stack(P_over_q))
        P_over_pi = torch.flatten(torch.stack(P_over_pi))
        pi_over_q = torch.flatten(torch.stack(pi_over_q))

        ### now we compare the KL divergence betweend p and pi and p and q to perhaps replacce q 
        was_q_updated = False
        z_hat_i = torch.mean(P_over_q)
        z_hat_i_std = torch.std(P_over_q)
        self.Z_moving_average = (self.iter*self.Z_moving_average + z_hat_i)/(self.iter+1)

        tvd_p_pi = 0.5 * torch.sum(torch.abs(pi_over_q -P_over_q/ self.Z_moving_average))/(number_samples*self.batch_size)
        tvd_p_q = 0.5 * torch.sum(torch.abs(1-P_over_q/ self.Z_moving_average))/(number_samples*self.batch_size)

        dkl_p_pi = -torch.log(self.Z_moving_average) + torch.sum((P_over_q * torch.log(P_over_pi)))/(number_samples*self.batch_size* self.Z_moving_average)
        dkl_p_q = -torch.log(self.Z_moving_average) + torch.sum((P_over_q * torch.log(P_over_q)))/(number_samples*self.batch_size* self.Z_moving_average)

        if self.q_update_criterion == 'interval' :
            if (self.iter+1) % self.q_update_interval == 0:
                print("was_q_updated")
                self.ref_model.load_state_dict(self.model.state_dict())
                was_q_updated = True
        elif self.q_update_criterion in ['kld', 'tvd'] and (self.iter+1) % self.q_update_interval == 0:
            if self.q_update_criterion == 'kld':
                if dkl_p_pi < dkl_p_q:
                    self.ref_model.load_state_dict(self.model.state_dict())
                    if dkl_p_pi < self.min_kld:
                        self.min_kld = dkl_p_pi
                    was_q_updated = True
                    print("updating q based on KL divergence")
                else :
                    print ("Worse KL divergence, not updating q")
            
            if self.q_update_criterion == 'tvd':
                
                if tvd_p_pi < tvd_p_q:
                    self.ref_model.load_state_dict(self.model.state_dict())
                    if tvd_p_pi < self.min_tvd:
                        self.min_tvd = tvd_p_pi
                    was_q_updated = True
                    print("updating q based on TVD")
                else :
                    print ("Worse TVD, not updating q")

        train_stats['dkl_p_pi'] = dkl_p_pi
        train_stats['dkl_p_q'] = dkl_p_q
        train_stats['tvd_p_pi'] = tvd_p_pi
        train_stats['tvd_p_q'] = tvd_p_q

        train_stats['Z_moving_average'] = self.Z_moving_average
        train_stats['min_kld'] = self.min_kld
        train_stats['min_tvd'] = self.min_tvd
        train_stats['loss'] = loss
        train_stats['Z_mean'] = z_hat_i
        train_stats['Z_std'] = z_hat_i_std

        train_stats['q_updated?'] = was_q_updated
        self.iter += 1
        return train_stats

def run_steps(self, num_steps):
        train_history = []
        for _ in range(num_steps):
            gc.collect()
            torch.cuda.empty_cache()
            train_stats = self.step()
            train_history.append(train_stats)


        return train_history
