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


def append_node_link(true_graphs, next_step_gnn1, next_step_gnn2):
    pass


def create_torch_graph_from_one_atom(atom, edge_size, encoding_option='charged'):
    num_atom = int(atom)

    atom_attribute = node_encoder(num_atom, encoding_option=encoding_option)
    # Create graph
    graph = torch_geometric.data.Data(x=atom_attribute.view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, edge_size)))

    return graph

def sample_first_atom(encoding = 'charged'):
    if encoding == 'reduced' or encoding == 'charged':
        prob_dict = {'60': 0.7385023585929047, 
                    '80': 0.1000143018658728, 
                    '70': 0.12239949901813525, 
                    '90': 0.013786373862576426, 
                    '160': 0.017856330814654413,
                    '170': 0.007441135845856433}
    if encoding == 'polymer':
        prob_dict = {'60': 0.7489344573582472,
                    '70': 0.0561389266682314,
                    '80': 0.0678638375933265,
                    '160': 0.08724385192820308,
                    '90': 0.032130486119902095,
                    '140': 0.007666591133009364,
                    '150': 2.184919908044154e-05}

    
    return random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()))[0]


def sample_path_from_model(model_q, model_a, model_pi,features,lambdas, batch_size = 128, device = 'cuda',  edge_size= 3 ):
    #this function compute a batch of x ( so paths) and the corresponding q(x), a(x), pi(x)
    #it also compute the features of the final molecule so that we can compute the exponents

    #we first sample a batch of x
    #generate a batch of initial_atoms 
    
    true_graphs = [create_torch_graph_from_one_atom(sample_first_atom(), edge_size=edge_size, encoding_option='charged')
                            for i in range(batch_size)]
    
    a_value = torch.zeros(batch_size)
    q_value = torch.zeros(batch_size)
    pi_value = torch.zeros(batch_size)
    queues = [[0] for i in range(batch_size)]

    #not_graphs_finished is a mask of size batch size, initialized at true
    not_mols_finished = [True for i in range(batch_size)]
    
    #while not all atoms are finished
    while torch.sum(not_mols_finished) > 0 :
        current_atoms = torch.zeros(batch_size)
        for i in range(batch_size):
            if len(queues[i]) == 0:
                not_mols_finished[i] = False
                current_atoms[i]= -1 # precise no current_atoms
            else :
                current_atoms[i] = queues[i][0]

        graph_for_gnn1 =[graph.clone() for graph in true_graphs]
        #add feature position
        for graph1 in graph_for_gnn1 :
                graph1.x = torch.cat([graph1.x, torch.zeros(graph1.x.size(0), 1)], dim=1)
                graph1.x[0:current_atoms[i], -1] = 1

        #compute the gnn1 and apply softmax
        batch_for_gnn1 = Batch.from_data_list(graph_for_gnn1)
        batch_for_gnn1.to(device)
        output_gnn1_q = torch.softmax(model_q.GNN1(batch_for_gnn1), dim = 1)
        output_gnn1_a = torch.softmax(model_a.GNN1(batch_for_gnn1),  dim = 1)
        output_gnn1_pi = torch.softmax(model_pi.GNN1(batch_for_gnn1), dim = 1)

        #for each graph in the ouptut_gnn1_q we sample with a multinomial
        next_step_gnn1 = torch.multinomial(output_gnn1_q, 1).item() #shape (batch_size, 1)

        a_value = a_value * torch([output_gnn1_a[i][next_step_gnn1[i]] for i in range(batch_size) if not_mols_finished[i]])
        pi_value = pi_value * torch([output_gnn1_pi[i][next_step_gnn1[i]] for i in range(batch_size) if not_mols_finished[i]])
        q_value = q_value * torch([output_gnn1_q[i][next_step_gnn1[i]] for i in range(batch_size) if not_mols_finished[i]])

        for i in range(batch_size):
            encoded_predicted_node = torch.zeros(output_gnn1_q.size(), dtype=torch.float)
            encoded_predicted_node[0, predicted_node] = 1

        
        self.queue.append(graph1.x.size(0))

        mask_no_stop_step = [next_step_gnn1[i] != 12 for i in range(batch_size)]

        #check that next_step_gnn1 is not a stop node
        for i in range(batch_size):
            if next_step_gnn1[i] == 12:
                queues[i].pop(0) 
       
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
    for i,fn in enumerate(features.values()):
        all_features_values[:,i] = fn(true_graphs)

    #exponents are the exponential of lambdas * all_features_values for each molecule
    exponents = torch.exp(torch.matmul(all_features_values, lambdas), dim = 1) #check the dimension carefuly

    return all_features_values ,exponents, a_value, q_value, pi_value 
    # exponents, a_value, q_value, pi_value are of size batch_size 
    # all_features_values is of size (batch_size, len(features))



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

        max_n_iters = n_iters

        feature_names = list(self.features.keys())
        mu_star = self.desired_moments #name mu_bar in the pseudo code

        mu_star = torch.tensor([mu_star[f] for f in feature_names])
        lambdas = torch.tensor([self.lambdas[f] for f in feature_names])

        # Collect sample_size samples for this:
        list_feature_tensor = []
        for i in  range(sample_size):
            #if we do multi processing, i think here the best 
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
        for k in range (number_samples):
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
        loss = loss /(number_samples*self.batch_size)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()



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


        