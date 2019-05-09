import time 
import random 
import numpy as np
import pandas as pd 
import pyUniDOE as pydoe 
from joblib import Parallel
from joblib import delayed
from matplotlib import pylab as plt
from tqdm import tqdm_notebook as tqdm

EPS = 10**(-10)

class BaseSeqUD(object):
    
    """Base class for sequential uniform design."""
    
    def __init__(self, para_space, level_number, max_runs, max_search_iter = 100, n_jobs=None, rand_seed = 0, verbose = False):

        self.para_space = para_space
        self.level_number = level_number
        self.max_runs = max_runs
        self.max_search_iter = max_search_iter
        self.n_jobs = n_jobs
        self.rand_seed = rand_seed
        self.verbose = verbose
        
        self.stage = 0
        self.stop_flag = False
        self.para_ud_names = []
        self.variable_number = [0]
        self.factor_number = len(self.para_space)
        self.para_names = list(self.para_space.keys())
        for items, values in self.para_space.items():
            if (values['Type']=="categorical"):
                self.variable_number.append(len(values['Mapping']))
                self.para_ud_names.extend([items + "_UD_" + str(i+1) for i in range(len(values['Mapping']))])
            else:
                self.variable_number.append(1)
                self.para_ud_names.append(items+ "_UD")
        self.extend_factor_number = sum(self.variable_number)  
    
    def plot_scores(self):
        if self.logs.shape[0]>0:
            cum_best_score = self.logs["score"].cummax()
            fig = plt.figure(figsize = (6,4))
            plt.plot(cum_best_score)
            plt.xlabel('# of Runs')
            plt.ylabel('Best Scores')
            plt.title('The best found scores during optimization')
            plt.grid(True)
            plt.show()
        else:
            print("No available logs!")

    def _summary(self, para_set_ud, para_set, score):
        """
        This function summarizes the evaluation results and makes records. 
        
        Parameters
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point, 
                and columns are used to represent variables. 
        para_set: A pandas dataframe which contains the trial points in original form. 
        score: A numpy vector, which contains the evaluated scores of trial points in para_set.
        
        """
        self.best_index_ = self.logs.loc[:,"score"].idxmax()
        self.best_params_ = {self.logs.loc[:,self.para_names].columns[j]:\
                             self.logs.loc[:,self.para_names].iloc[self.best_index_,j] 
                              for j in range(self.logs.loc[:,self.para_names].shape[1])}
        self.best_score_ = self.logs.loc[:,"score"].iloc[self.best_index_]
        if self.verbose:
            print("Search completed in %.2f seconds."%self.search_time_consumed_)
            print("The best score is: %.5f."%self.best_score_)
            print("The best configurations are:")
            print("\n".join("%-20s: %s"%(k, v if self.para_space[k]['Type']=="categorical" else round(v, 5))
                            for k, v in self.best_params_.items()))
             
    def _para_mapping(self, para_set_ud):
        
        """
        This function maps trials points in UD space ([0, 1]) to original scales. 
        
        There are three types of variables: 
          - continuous：Perform inverse Maxmin scaling for each value. 
          - integer: Evenly split the UD space, and map each partition to the corresponding integer values. 
          - categorical: The UD space uses one-hot encoding, and this function selects the one with the maximal value as class label.
          
        Parameters
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point, 
                and columns are used to represent variables. 
        
        Returns
        ----------
        para_set: The transformed variables.
        """
        
        para_set = pd.DataFrame(np.zeros((para_set_ud.shape[0],self.factor_number)), columns = self.para_names) 
        for items, values in self.para_space.items():
            if (values['Type']=="continuous"):
                para_set[items] = values['Wrapper'](para_set_ud[items+"_UD"]*(values['Range'][1]-values['Range'][0])+values['Range'][0])
            elif (values['Type'] == "integer"):
                temp = np.linspace(0, 1, len(values['Mapping'])+1)
                for j in range(1,len(temp)):
                    para_set.loc[(para_set_ud[items+"_UD"]>=temp[j-1])&(para_set_ud[items+"_UD"]<temp[j]),items] = values['Mapping'][j-1]
                para_set.loc[para_set_ud[items+"_UD"]==1,items] = values['Mapping'][-1]
                para_set[items] = para_set[items].astype(int)
            elif (values['Type'] == "categorical"):
                column_bool = [items in para_name for para_name in self.para_ud_names]
                col_index = np.argmax(para_set_ud.loc[:,column_bool].values, axis = 1).tolist()
                para_set[items] = np.array(values['Mapping'])[col_index]
        return para_set  
    
    def _generate_init_design(self):
        """
        This function generates the initial uniform design. 
        
        Returns
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point, 
                and columns are used to represent variables.         
        """
        
        self.logs = pd.DataFrame()
        ud_space = np.repeat(np.linspace(1/(2*self.level_number), 1-1/(2*self.level_number), self.level_number).reshape([-1,1]),
                             self.extend_factor_number, axis=1)

        BaseUD = pydoe.DesignQuery(n = self.level_number, s = self.extend_factor_number,
                          q = self.level_number, crit = "CD2", ShowCrit=False)
        if BaseUD is None:
            BaseUD = pydoe.GenUD_MS(n = self.level_number, s = self.extend_factor_number, q = self.level_number, crit="CD2", 
                            maxiter = self.max_search_iter, rand_seed = self.rand_seed, nshoot = 10)
           
        if (not isinstance(BaseUD, np.ndarray)):
            raise ValueError('Uniform design is not correctly constructed!')

        para_set_ud = np.zeros((self.level_number, self.extend_factor_number))
        for i in range(self.factor_number):
            loc_min = np.sum(self.variable_number[:(i+1)])
            loc_max = np.sum(self.variable_number[:(i+2)])
            for k in range(int(loc_min), int(loc_max)):
                para_set_ud[:,k] = ud_space[BaseUD[:,k]-1,k]
        para_set_ud = pd.DataFrame(para_set_ud, columns = self.para_ud_names)
        return para_set_ud
    
    def _generate_augment_design(self, ud_center):
        """
        This function refines the search space to a subspace of interest, and 
        generates augmented uniform designs given existing designs. 
        
                
        Parameters
        ----------
        ud_center: A numpy vector representing the center of the subspace, 
               and corresponding elements denote the position of the center for each variable.         

        Returns
        ----------
        para_set_ud: A pandas dataframe where each row represents a UD trial point, 
                and columns are used to represent variables.         
        """

        
        # 1. Transform the existing Parameters to Standardized Horizon (0-1)
        ud_space = np.zeros((self.level_number,self.extend_factor_number))
        ud_grid_size = 1.0/(self.level_number*2**(self.stage-1))
        left_radius = np.floor((self.level_number-1)/2) * ud_grid_size
        right_radius = (self.level_number - np.floor((self.level_number-1)/2) -1) * ud_grid_size
        for i in range(self.extend_factor_number):
            if ((ud_center[i]-left_radius)<0):
                lb = 0
                ub = ud_center[i] + right_radius - (ud_center[i]-left_radius)
            elif ((ud_center[i] + right_radius)> 1):
                ub = 1
                lb = ud_center[i] - left_radius - (ud_center[i]+ right_radius - 1)
            else:
                lb = max(ud_center[i]-left_radius,0)
                ub = min(ud_center[i]+right_radius,1)
            ud_space[:,i] = np.linspace(lb, ub, self.level_number)

        # 2. Map existing Runs' Parameters to UD Levels "x0" (1 - level_number)
        flag = True
        for i in range(self.extend_factor_number):
            flag = flag & (self.logs.loc[:,self.para_ud_names].iloc[:,i]>=(ud_space[0,i]-EPS)) 
            flag = flag & (self.logs.loc[:,self.para_ud_names].iloc[:,i]<=(ud_space[-1,i]+EPS))
        x0 = self.logs.loc[flag,self.para_ud_names].values
        
        for i in range(x0.shape[0]):
            for j in range(x0.shape[1]):
                x0[i,j] = (np.where(abs(x0[i,j]-ud_space[:,j])<=EPS)[0][0] + 1)
        
        x0 = x0.astype(int)
        # 3. Delete existing UD points on the same levels grids
        for i in range(self.extend_factor_number):
            keep_list = []
            unique = np.unique(x0[:,i])
            for j in range(len(unique)):
                xx_loc=np.where(x0[:,i]==unique[j])[0].tolist()
                keep_list.extend(random.sample(xx_loc, 1))
            x0 = x0[keep_list,:].reshape([-1,self.extend_factor_number])

        # Return if the maximum run has reached.
        if ((self.logs.shape[0] + self.level_number - x0.shape[0])>self.max_runs):
            self.stop_flag = True
            if self.verbose:
                print("Maximum number of runs reached, stop!")
            return
        
        if (x0.shape[0]>=self.level_number):
            self.stop_flag = True
            if self.verbose:
                print("Search space already full, stop!")
            return
        
        # 4. Generate Sequential UD
        BaseUD = pydoe.GenAUD_MS(x0, n = self.level_number, s = self.extend_factor_number, q = self.level_number, crit="CD2",
                         maxiter = self.max_search_iter, rand_seed = self.rand_seed, nshoot = 10)
        if (not isinstance(BaseUD, np.ndarray)):
            raise ValueError('Uniform design is not correctly constructed!')

        BaseUD_Aug = BaseUD[(1+x0.shape[0]):BaseUD.shape[0],:].reshape([-1, self.extend_factor_number])

        para_set_ud = np.zeros((BaseUD_Aug.shape[0], self.extend_factor_number))
        for i in range(self.factor_number):
            loc_min = np.sum(self.variable_number[:(i+1)])
            loc_max = np.sum(self.variable_number[:(i+2)])
            for k in range(int(loc_min), int(loc_max)):
                para_set_ud[:,k] = ud_space[BaseUD_Aug[:,k]-1,k]
        para_set_ud = pd.DataFrame(para_set_ud, columns = self.para_ud_names)
        return para_set_ud
    
    def _evaluate_runs(self, obj_func, para_set_ud):
        """
        This function evaluates the performance scores of given trials. 
        
                
        Parameters
        ----------
        obj_func: A callable function. It takes the values stored in each trial as input parameters, and  
               output the corresponding scores.  
        para_set_ud: A pandas dataframe where each row represents a UD trial point, 
                and columns are used to represent variables.         
        """
        para_set = self._para_mapping(para_set_ud)
        para_set_ud.columns = self.para_ud_names
        candidate_params = [{para_set.columns[j]: para_set.iloc[i,j] 
                             for j in range(para_set.shape[1])} 
                            for i in range(para_set.shape[0])] 
        out = Parallel(n_jobs=self.n_jobs)(delayed(obj_func)(parameters)
                                for parameters in tqdm(candidate_params, 
                                   desc = "Stage %d:" %self.stage, 
                                   postfix = "Current Best Score = %.5f"% 0 if self.logs.shape[0]==0 else \
                                                       (self.logs.loc[:,"score"].max())))
        self.stage += 1
        
    def _run_search(self, obj_func):
        """
        This function controls the procedures for implementing the sequential uniform design method. 
        
        Parameters
        ----------
        obj_func: A callable function. It takes the values stored in each trial as input parameters, and  
               output the corresponding scores.  
        """
        self.stage = 1
        self.logs = pd.DataFrame()
        self.pbar = tqdm(total=self.max_runs) 

        search_start_time = time.time()
        para_set_ud = self._generate_init_design()
        self._evaluate_runs(obj_func, para_set_ud)
        while (True):
            ud_center = self.logs.sort_values("score", ascending = False).loc[:,self.para_ud_names].values[0,:] 
            para_set_ud = self._generate_augment_design(ud_center)
            if not self.stop_flag:
                self._evaluate_runs(obj_func, para_set_ud)
            else:
                break
        search_end_time = time.time()
        self.search_time_consumed_ = search_end_time - search_start_time        
        