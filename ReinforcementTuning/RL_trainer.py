import torch
import os
import sys

import numpy as np

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from DataPipeline.preprocessing import node_encoder, tensor_to_smiles


from dataclasses import dataclass

import gc

from tqdm import tqdm



class GDCTrainer_path():
    
    def __init__(self, Module_Gen, features, desired_moments, q_update_criterion, lr = 1e-4, minibatch_size = 16, batch_size = 1000, min_nabla_lambda = 0.01, lambdas = None):
   
        # dpg_epochs is the number of optimization epochs per batch of samples
        self.Module_Gen = Module_Gen
        self.Module_Gen.batch_size = batch_size
        self.features = features

        if lambdas is None:
            self.lambdas = torch.zeros(len(self.features)).to(self.Module_Gen.device)
        else:
            self.lambdas = lambdas.to(self.Module_Gen.device)
        Module_Gen.lambdas = self.lambdas # Be sure that the lambdas are initialized to 0 in Module_Gen
        self.q_update_criterion = q_update_criterion
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.desired_moments = desired_moments
        self.min_nabla_lambda =  min_nabla_lambda      # difference between the wanted mu and approximated mu 

        assert self.q_update_criterion in ['interval', 'tvd', "kld"]
        # q_update_criterion can take one of the following values:
        # - 'interval': Update the GDP policy at regular intervals defined by q_update_interval.
        # - 'tvd': Update the GDP policy when the total variation distance between action probability distributions exceeds a threshold.
        # - 'kld': Update the GDP policy when the Kullback-Leibler divergence between action probability distributions exceeds a threshold.
        

        # GNN1_model
        for p1, p2 in zip(self.Module_Gen.GNNs_Models_q.GNN1_model.parameters(), self.Module_Gen.GNNs_Models_a.GNN1_model.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False
        
        # GNN2_model
        for p1, p2 in zip(self.Module_Gen.GNNs_Models_q.GNN2_model.parameters(), self.Module_Gen.GNNs_Models_a.GNN2_model.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False
        
        #GNN3_1_model
        for p1, p2 in zip(self.Module_Gen.GNNs_Models_q.GNN3_1_model.parameters(), self.Module_Gen.GNNs_Models_a.GNN3_1_model.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False
        
        #GNN3_2_model
        for p1, p2 in zip(self.Module_Gen.GNNs_Models_q.GNN3_2_model.parameters(), self.Module_Gen.GNNs_Models_a.GNN3_2_model.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

        # Define optimizers 
        self.optimizer_1 = torch.optim.Adam(self.Module_Gen.GNNs_Models_pi.GNN1_model.parameters(), lr=lr)
        self.optimizer_2 = torch.optim.Adam(self.Module_Gen.GNNs_Models_pi.GNN2_model.parameters(), lr=lr)
        self.optimizer_3_1 = torch.optim.Adam(self.Module_Gen.GNNs_Models_pi.GNN3_1_model.parameters(), lr=lr)
        self.optimizer_3_2 = torch.optim.Adam(self.Module_Gen.GNNs_Models_pi.GNN3_2_model.parameters(), lr=lr)



        # will hold all values of P(x) / q(x) for estimating TVD
        self.Z_moving_average = 0
        self.iter = 0
        self.min_kld = float("inf")
        self.min_tvd = float("inf")
        self.is_policy_eval = False

        #### Compute lambdas
        if lambdas is None:
            self.compute_optimal_lambdas()


    def compute_optimal_lambdas(self, sample_size=5, n_iters=1000, lr=.5, min_nabla_lambda = 0.001): #how do they define the learning rate and sample size maybe do more
        """
        This performs the first step: Constraints --> EBM through self-normalized importance sampling. 
        Args:
            sample_size: total number of samples to use for lambda computation
        Returns:
            dicitonary of optimal lambdas per constraint: {'black': lambda_1, 'positive': lambda_2}
        """


        print("Computing Optimal Lambdas for desired moments...")

        max_n_iters = n_iters

        feature_names = list(self.Module_Gen.features.keys())
        mu_star = self.desired_moments #name mu_bar in the pseudo code

        mu_star = torch.tensor([mu_star[f] for f in feature_names])
        lambdas = self.Module_Gen.lambdas.cpu()

        # Collect sample_size samples for this:
        list_feature_tensor = []
        for i in range(sample_size):
            #if we do multi processing, i think here the best 
            #put pi without grad for lambdas

            self.Module_Gen.full_generation()
            self.Module_Gen.convert_to_smiles()
            self.Module_Gen.compute_features()

            batch_features_values = self.Module_Gen.all_features_values

            
            list_feature_tensor.append(batch_features_values)

        all_feature_tensor = torch.cat(list_feature_tensor, dim=0)  # [sample_sz*size_batch x F]

        #### check for zero-occuring features. 
        # If a constraint has not occurred in your sample, no lambdas will be learned for that constraint, so we must check.

        for i, feature  in enumerate(feature_names):
            assert all_feature_tensor[:, i].sum().item() > 0, "constraint {feature} hasn't occurred in the samples, use a larger sample size"

        for step in range(max_n_iters): #SGD for finding lambdas

            # 1. calculate P_over_q batch wise with current lambdas which will be name w
            ## compute new exponents

            w = torch.exp(torch.matmul(all_feature_tensor, lambdas.to(all_feature_tensor.device)))


            # 2. compute mu (mean) of features given the current lambda using SNIS
            mu_lambda_numerator = w.view(1, -1).matmul(all_feature_tensor ).squeeze(0) # F
            mu_lambda_denominator = w.sum()
            mu_lambda = mu_lambda_numerator / mu_lambda_denominator # F

            # 3. Update current Lambdas
            nabla_lambda = mu_star - mu_lambda.cpu()
            err = np.linalg.norm(nabla_lambda.cpu().numpy())
            print("step: %s \t ||nabla_lambda|| = %.6f" %(step, err))
            lambdas = lambdas + lr * nabla_lambda
            print("\tlambdas : {} ".format(self.Module_Gen.lambdas))
            print("\tμ: {}".format(mu_lambda))
            print("\tμ*: {}".format(mu_star))

            self.Module_Gen.lambdas = lambdas
            
            ## Check if error is less than tolerance, then break.
            if err < min_nabla_lambda: 
                break
    

    def step(self, num_batches, num_mini_batches):
        train_stats = {}
        P_over_q = []
        P_over_pi = []
        pi_over_q = []
        total_loss = 0
        for i in tqdm(range(num_mini_batches)):
            loss = 0
            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            self.optimizer_3_1.zero_grad()
            self.optimizer_3_2.zero_grad()

            self.Module_Gen.full_generation(batch_size = self.minibatch_size)
            self.Module_Gen.convert_to_smiles()
            self.Module_Gen.compute_features()
            
            exponents, all_features_values, q_value, a_value, pi_value = self.Module_Gen.get_all()

            # Check if there is a zero in the tensor q, a or pi

            if (q_value == 0).any():
                # Add a tensor of 1e-38 on all the values of q
                q_value = q_value + torch.ones_like(q_value) * float('1e-38')

            if (a_value == 0).any():
                # Add a tensor of 1e-38 on all the values of a
                a_value = a_value + torch.ones_like(a_value) * float('1e-38')
            
            if (pi_value == 0).any():
                # Add a tensor of 1e-38 on all the values of pi
                pi_value = pi_value + torch.ones_like(pi_value) * float('1e-38')

            assert self.Module_Gen.GNNs_Models_q.GNN1_model.training == False
            assert self.Module_Gen.GNNs_Models_q.GNN2_model.training == False
            assert self.Module_Gen.GNNs_Models_q.GNN3_1_model.training == False
            assert self.Module_Gen.GNNs_Models_q.GNN3_2_model.training == False
            assert self.Module_Gen.GNNs_Models_pi.GNN1_model.training == False
            assert self.Module_Gen.GNNs_Models_pi.GNN2_model.training == False
            assert self.Module_Gen.GNNs_Models_pi.GNN3_1_model.training == False
            assert self.Module_Gen.GNNs_Models_pi.GNN3_2_model.training == False
            assert self.Module_Gen.GNNs_Models_a.GNN1_model.training == False
            assert self.Module_Gen.GNNs_Models_a.GNN2_model.training == False
            assert self.Module_Gen.GNNs_Models_a.GNN3_1_model.training == False
            assert self.Module_Gen.GNNs_Models_a.GNN3_2_model.training == False
            

            P_over_q.append(torch.flatten(a_value * exponents / q_value))
            P_over_pi.append(torch.flatten(a_value * exponents / pi_value))
            pi_over_q.append(torch.flatten(pi_value / q_value))

            # Compute the loss to train the model
            loss = - torch.sum(a_value*exponents / q_value * torch.log(pi_value))
            total_loss += loss.item()   

            # Backward pass and optimizer steps are performed after each mini-batch
            loss.backward()
            self.optimizer_1.step()
            self.optimizer_2.step()
            self.optimizer_3_1.step()
            self.optimizer_3_2.step()

            self.Module_Gen.clean_memory()

            """
            # Delete everything that is not needed anymore
            loss.cpu().detach()
            q_value.cpu().detach()
            a_value.cpu().detach()
            pi_value.cpu().detach()
            exponents.cpu().detach()
            del loss, q_value, a_value, pi_value, exponents

            torch.cuda.empty_cache()

            # Garbage collection
            gc.collect()
            """
        
            
        mean_loss = total_loss / (num_mini_batches * self.minibatch_size)


        P_over_q = torch.flatten(torch.stack(P_over_q))
        P_over_pi = torch.flatten(torch.stack(P_over_pi))
        pi_over_q = torch.flatten(torch.stack(pi_over_q))

        ### now we compare the KL divergence betweend p and pi and p and q to perhaps replacce q 
        was_q_updated = False
        z_hat_i = torch.mean(P_over_q)
        z_hat_i_std = torch.std(P_over_q)
        print('z_hat_i : ', z_hat_i)
        print('i' , self.iter)
        self.Z_moving_average = (self.iter*self.Z_moving_average + z_hat_i)/(self.iter+1)

        tvd_p_pi = 0.5 * torch.sum(torch.abs(pi_over_q -P_over_q/ self.Z_moving_average))/(num_mini_batches * self.minibatch_size)
        tvd_p_q = 0.5 * torch.sum(torch.abs(1-P_over_q/ self.Z_moving_average))/(num_mini_batches * self.minibatch_size)

        print('self.Z_moving_average : ', self.Z_moving_average)
        print('P_over_q : ', P_over_q)
        print('P_over_pi : ', P_over_pi)
        dkl_p_pi = -torch.log(self.Z_moving_average) + torch.sum((P_over_q * torch.log(P_over_pi)))/(num_mini_batches * self.minibatch_size * self.Z_moving_average)
        dkl_p_q = -torch.log(self.Z_moving_average) + torch.sum((P_over_q * torch.log(P_over_q)))/(num_mini_batches * self.minibatch_size * self.Z_moving_average)
        
        print('dkl_p_pi: ', dkl_p_pi)
        print('dkl_p_q: ', dkl_p_q)
        

        if self.q_update_criterion in ['kld', 'tvd']:
            if self.q_update_criterion == 'kld':
                if dkl_p_pi < dkl_p_q:
                    self.Module_Gen.GNNs_Models_q.load_from_state_dict(self.Module_Gen.GNNs_Models_pi.get_state_dict())
                    if dkl_p_pi < self.min_kld:
                        self.min_kld = dkl_p_pi
                    was_q_updated = True
                    print("updating q based on KL divergence")
                else :
                    print ("Worse KL divergence, not updating q")
            
            if self.q_update_criterion == 'tvd':
                
                if tvd_p_pi < tvd_p_q:
                    self.Module_Gen.GNNs_Models_q.load_from_state_dict(self.Module_Gen.GNNs_Models_pi.get_state_dict())
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
    def run_steps(self, num_steps, num_batches, num_mini_batches):
            self.Module_Gen.batch_size = self.minibatch_size
            train_history = []
            for _ in range(num_steps):
                gc.collect()
                torch.cuda.empty_cache()
                train_stats = self.step(num_batches, num_mini_batches)
                train_history.append(train_stats)


            return train_history