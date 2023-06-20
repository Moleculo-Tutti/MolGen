import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import time
import random
import copy
import os
from models import Models_GNN


def append_node_link(true_graphs, next_step_gnn1, next_step_gnn2):
    pass


def sample_path_from_model(model_q, model_a, model_pi,features,lambdas, batch_size = 128, device = 'cuda' ):
    #this function compute a batch of x ( so paths) and the corresponding q(x), a(x), pi(x)
    #it also compute the features of the final molecule so that we can compute the exponents

    #we first sample a batch of x
    #generate a batch of initial_atoms 
    true_graphs = torch.zeros(batch_size).to(device)
    true_graphs = torch([initial_atom() for i in range(batch_size)])
    
    a_value = torch.zeros(batch_size).to(device)
    q_value = torch.zeros(batch_size).to(device)
    pi_value = torch.zeros(batch_size).to(device)
    
    while not_all_atoms_finished :
        #compute the gnn1 and apply softmax
        output_gnn1_q = torch.softmax(model_q.GNN1(true_graphs), dim = 1)
        output_gnn1_a = torch.softmax(model_a.GNN1(true_graphs),  dim = 1)
        output_gnn1_pi = torch.softmax(model_pi.GNN1(true_graphs), dim = 1)

        #for each graph in the ouptut_gnn1_q we sample with a multinomial
        next_step_gnn1 = torch.multinomial(output_gnn1_q, 1) #shape (batch_size, 1)

        a_value = a_value * torch([output_gnn1_a[i][next_step_gnn1[i]] for i in range(batch_size)])
        pi_value = pi_value * torch([output_gnn1_pi[i][next_step_gnn1[i]] for i in range(batch_size)])
        q_value = q_value * torch([output_gnn1_q[i][next_step_gnn1[i]] for i in range(batch_size)])

        #check that next_step_gnn1 is not a stop node 
        if next_step_gnn1 != dim_embedding -1:
            # change which node is the one we are looking at
            #idk if we change which node is on feature embedding finished here
            pass
        else :
            #compute the gnn2 and apply softmax
            output_gnn2_q = torch.softmax(model_q.GNN2(true_graphs,next_step_gnn1), dim = 1)
            output_gnn2_a = torch.softmax(model_a.GNN2(true_graphs,next_step_gnn1),  dim = 1)
            output_gnn2_pi = torch.softmax(model_pi.GNN2(true_graphs,next_step_gnn1), dim = 1)

            #for each graph in the ouptut_gnn2_q we sample with a multinomial
            next_step_gnn2 = torch.multinomial(output_gnn2_q, 1) #shape (batch_size, 1)

            a_value = a_value * torch([output_gnn2_a[i][next_step_gnn2[i]] for i in range(batch_size)])
            pi_value = pi_value * torch([output_gnn2_pi[i][next_step_gnn2[i]] for i in range(batch_size)])
            q_value = q_value * torch([output_gnn2_q[i][next_step_gnn2[i]] for i in range(batch_size)])

            true_graphs = append_node_link(true_graphs, next_step_gnn1, next_step_gnn2)
            graph_for_gnn3 = encode_gnn3(true_graphs.deepcopy())
            #this function also change who is the point of interest to switch to the latest added
            
            #compute the gnn3 and apply softmax to choose the neighbour
            output_gnn3_q = torch.softmax(model_q.GNN3(graph_for_gnn3), dim = 1)
            output_gnn3_a = torch.softmax(model_a.GNN3(graph_for_gnn3),  dim = 1)
            output_gnn3_pi = torch.softmax(model_pi.GNN3(graph_for_gnn3), dim = 1)

            #compute the softmax over the neighbor chosen to choose the kind of link

            if next_step_gnn3 == 0 :#it is a stop,
                #for the probability we compute it over all the node of the graph probability to stop
                a_value
            
            else :
                # multiply by two values, the one if the good neighbor is chosen and then if the good type of link 
                pass

                true_graphs = add_link_closness(true_graphs, node_linked,)
    
    #now we can compute the features of the final molecule
    all_features_values = torch.zeros(batch_size, len(features)).to(device)
    for i,fn in enumerate(features):
        all_features_values[:,i] = compute_features(true_graphs, fn)

    #exponents are the exponential of lambdas * all_features_values for each molecule
    exponents = torch.exp(torch.matmul(all_features_values, lambdas), dim = 1) #check the dimension carefuly

    return all_features_values, exponents, a_value, q_value, pi_value 



class GDCTrainer_path():
    #implement a finetuning of our odel (which is concatenation of 3 GNN) with a Gradient distributional policy program
    def get_sampling_model(self):
        return self.ref_model
    
    def get_policy_model(self):
        return self.model 
    
    def get_eval_model(self):
        return self.ref_model
    
    def __init__(self, name_exp, features, desired_moments, q_update_criterion, q_update_interval, sampling_function, lr = 1e-4, batch_size = 128, dpg_epochs =4):
        # dpg_epochs is the number of optimization epochs per batch of samples
        self.sampling_function = sampling_function
        self.features = features

        self.lambdas = {k:0.0 for k in features}
        self.q_update_criterion = q_update_criterion
        self.batch_size = batch_size
        self.desired_moments = desired_moments
        assert self.q_update_criterion in ['interval', 'tvd', "kld"]
        # q_update_criterion can take one of the following values:
        # - 'interval': Update the GDP policy at regular intervals defined by q_update_interval.
        # - 'tvd': Update the GDP policy when the total variation distance between action probability distributions exceeds a threshold.
        # - 'kld': Update the GDP policy when the Kullback-Leibler divergence between action probability distributions exceeds a threshold.
        

        #to open
        self.model = Models_GNN(name_exp) # it is pi
        self.orig_model = copy.deepcopy(self.model) # it is a 
        self.ref_model = copy.deepcopy(self.model) # it is q
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


    def compute_optimal_lambdas(self, sample_size=4096, n_iters=1000, lr=.5): #how do they define the learning rate and sample size maybe do more
        """
        This performs the first step: Constraints --> EBM through self-normalized importance sampling. 
        Args:
            sample_size: total number of samples to use for lambda computation
        Returns:
            dicitonary of optimal lambdas per constraint: {'black': lambda_1, 'positive': lambda_2}
        """


        print("Computing Optimal Lambdas for desired moments...")

        min_nabla_lambda = 0.01 # difference betweeb the wanted mu and approximated mu 
        max_n_iters = n_iters

        feature_names = list(self.features.keys())
        mu_star = self.desired_moments #name mu_bar in the pseudo code

        mu_star = torch.tensor([mu_star[f] for f in feature_names])
        lambdas = torch.tensor([self.lambdas[f] for f in feature_names])

        # Collect sample_size samples for this:
        list_feature_tensor = []
        for i in  range(sample_size):
            all_features_values, exponents, a_value, q_value, pi_value   = self.sampling_function(self.get_sampling_model(),
                                                                                                self.get_eval_model(),
                                                                                                self.get_policy_model(), 
                                                                                                self.features, 
                                                                                                self.lambdas,
                                                                                                self.batch_size)

            #model_input = torch.cat((atom, mols_generated), axis=1)
            constraint_tensor = torch.stack([all_features_values[k] for k in all_features_values], dim=1) # B x F, du coup c'est censé etre batch_size x nb_features
            
            list_feature_tensor.append(constraint_tensor)

        all_constraint_tensor = torch.cat(list_feature_tensor, dim=0)  # [sample_sz x F]

        #### check for zero-occuring features. 
        # If a constraint has not occurred in your sample, no lambdas will be learned for that constraint, so we must check.

        for i, feature  in enumerate(feature_names):
            assert all_constraint_tensor[:, i].sum().item() > 0, "constraint {feature} hasn't occurred in the samples, use a larger sample size"

        for step in range(max_n_iters): #SGD for finding lambdas

            # 1. calculate P_over_q batch wise with current lambdas which will be name w
            ## compute new exponents
            list_w = []
            for feature_tensor in (list_feature_tensor):
                exponents = lambdas.to(constraint_tensor.get_device()).mul(constraint_tensor).sum(dim=1) # N  ## compute new exponents, do scalar product
                #w is equall to putting the exponents in one exponential 
                w = torch.exp(exponents) # N
                list_w.append(w)

            w = torch.cat(w, dim=0)

            # 2. compute mu (mean) of features given the current lambda using SNIS
            mu_lambda_numerator = w.view(1, -1).matmul(all_constraint_tensor ).squeeze(0) # F
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
            if err < min_nabla_lambda: 
                break
    

    def step(self,query,reponse,number_samples):
        """
        This is the main training function. It runs a off-policy DPG (with proposal q) optimization step which includes :
        1. Sample continuations from proposal distribution q(x) which is self.ref_model 
            (already done and output is passed as response)
        2. Compute P(x) / pi(x)
        3. Compute Loss = (P(x) / q(x)) log pi(x) and update policy pi
        Args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the continuation token ids. These were obtained from gdc.gpt2.respond_to_batch().
            scores (torch.tensor): tensor containing the  P(x)/ q(x), shape [batch_size]
        Returns:
            train_stats (dict): a summary of the training statistics for logging purposes.
        """

        for k in range (number_samples):
            pass
   

    def loss(self):
        pass


        
