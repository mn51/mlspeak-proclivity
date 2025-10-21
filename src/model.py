from utils import *
from data import *
from plttools import *
from time import perf_counter

def compute_loss(dataset:Dataset,betas=None,w=None,loss_scale:str='total',weight_func=None,device=None):
    assert loss_scale in ["convo","group","total"], "Invalid loss scaling type `loss_scale`. Must be one of ['convo','group','total']."
    assert weight_func in [None,get_turn_type_agg_weights,get_turn_type_weights], "Invalid weight function. Must be one of [None,get_turn_type_agg_weights,get_turn_type_weights]."
    assert (betas is None) or (len(betas)==dataset.num_groups and all([len(betas[g])==dataset.num_convos[g] for g in range(dataset.num_groups)])), "Inconsistent number of groups or convos between dataset and betas."
    assert (betas is None) or all([betas[g][c].shape == dataset.speakers_onehot[g][c].shape for g in range(dataset.num_groups) for c in range(dataset.num_convos[g])]), "Inconsistent number of turns or speakers between dataset and betas."
    # assert device in [None,'cpu','cuda'], "Invalid device. Must be cpu or cuda."

    if w is None:
        w = lambda x:torch.exp(-.5*x)
    if betas is None:
        betas = [[compute_betas(w,compute_time_since(dataset.speakers_onehot[g][c])) for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]

    nlls = [[[] for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]
    for g in range(dataset.num_groups):
        for c in range(dataset.num_convos[g]):
            nlls[g][c] = negloglik(dataset.speakers_onehot[g][c],dataset.pi[g],dataset.d[g],dataset.A[g],betas[g][c],device=device)

    if weight_func is None:
        weight_func = lambda convo:torch.ones_like(convo,dtype=torch.float)
    weights = [[weight_func(dataset.convos[g][c]) for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]
    if device is not None:
        weights = [[weights[g][c].to(device) for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]

    if loss_scale=='total':
        nlls = [[weights[g][c] * nlls[g][c] / np.concatenate(dataset.num_times).sum() for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]
    elif loss_scale=='group':
        nlls = [[weights[g][c] * nlls[g][c] / (dataset.num_groups*np.sum(dataset.num_times[g])) for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]
    elif loss_scale=='convo':
        nlls = [[weights[g][c] * nlls[g][c] / (np.sum(dataset.num_convos)*dataset.num_times[g][c]) for c in range(dataset.num_convos[g])] for g in range(dataset.num_groups)]
    
    loss = torch.sum(torch.cat(sum(nlls,[])))
    return loss

def get_turn_type_weights(convo):
    turn_classes = classify_turns(convo)
    flr = turn_classes['floor']
    brkf = turn_classes['broken_floor']
    newf = turn_classes['new_floor']
    reg = turn_classes['regain']
    nonf = turn_classes['nonfloor']

    flr = flr*(1-newf)
    counts = np.array([flr.sum(), brkf.sum(), newf.sum(), reg.sum(), nonf.sum()])
    inv_cnt = 1/(counts + (counts==0).astype(int))
    typ_w = inv_cnt / np.sum(inv_cnt)
    weights = typ_w[0]*flr + typ_w[1]*brkf + typ_w[2]*newf + typ_w[3]*reg + typ_w[4]*nonf
    return weights * 5

def get_turn_type_agg_weights(convo):
    turn_classes = classify_turns(convo)
    flr = turn_classes['floor']
    brkf = turn_classes['broken_floor']
    newf = turn_classes['new_floor']
    reg = turn_classes['regain']
    nonf = turn_classes['nonfloor']

    counts = np.array([flr.sum(), brkf.sum(), reg.sum(), nonf.sum()])
    inv_cnt = 1/(counts + (counts==0).astype(int))
    # typ_w = inv_cnt / np.sum(inv_cnt)
    typ_w = inv_cnt / 4
    weights = typ_w[0]*flr + typ_w[1]*brkf + typ_w[2]*reg + typ_w[3]*nonf
    # return weights * 4
    return weights * len(convo)

def load_model_params_from_config(config_path:str):
    if (config_path is not None) and os.path.exists(config_path):
        with open(config_path,'r') as f:
            CONFIG = json.load(f)

    key = "model_path"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==str, "Invalid model path `model_path`. Expecting a string."
    else:
        CONFIG[key] = ""

    key = "verbose"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag for verbosity `verbose`. Expecting a boolean value."
    else:
        CONFIG[key] = False

    key = "model_trials"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int and CONFIG[key]>0, "Invalid number of model trials `model_trials`. Expecting a positive integer."
    else:
        CONFIG[key] = 1
    
    assert "MODEL" in CONFIG.keys(), "Invalid parameters. Missing `MODEL` containing arguments for model training."

    key = "w"
    if key in CONFIG["MODEL"].keys() and CONFIG["MODEL"][key] is not None:
        assert type(CONFIG["MODEL"][key])==str, "Invalid configuration for memory function `w`. Expecting a string in the form of a function."
        CONFIG["MODEL"][key] = eval(CONFIG["MODEL"][key])

    return CONFIG

# ----------------------------------------------------------------

# Simple MLP
#   Layer 1: Linear  (input: in_dim, output: hid_dim)
#   Layer 2: ReLU    (input: hid_dim, output: hid_dim)
#   Layer 3: Linear  (input: hid_dim, output: out_dim)
#   Layer 4: Sigmoid (input: out_dim, output: out_dim)
#            Output between 0 and 1.
class MLP(torch.nn.Module):
    def __init__(self,in_dim,out_dim,hid_dim=5,nonlin=torch.nn.Sigmoid):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_dim,hid_dim,bias=True)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(hid_dim,out_dim,bias=True)
        self.layer4 = nonlin()
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# LSTM
#   Layer 1: LSTM   (input: in_dim, output: hid_dim)
#   Layer 2: Linear (input: hid_dim, output: out_dim)
#            Output is sequence of same length as input sequence.
class LSTM(torch.nn.Module):
    def __init__(self,in_dim=1,hid_dim=5,out_dim=1, nonlin=None):
        super(LSTM,self).__init__()
        self.hid_dim = hid_dim
        self.lstm = torch.nn.LSTM(in_dim,hid_dim)
        self.layer1 = torch.nn.Linear(hid_dim,out_dim)
        self.layer2 = nonlin() if nonlin is not None else None
    def forward(self,seq):
        x,_ = self.lstm(seq)
        x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        
        return x

# GAT
#   MLP for attention weights
#   LSTM for convo history
#   GAT for pairwise speaker relationships
#   MLP for final speaker likelihood score
class GAT(torch.nn.Module):
    def __init__(self,num_feat,out_dim=1,hid_dim=5,nonlin=torch.nn.Sigmoid):
        super().__init__()
        self.hid_dim = hid_dim
        
        self.a = torch.nn.Sequential(
            torch.nn.Linear(2*num_feat, hid_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, 1, bias=True),
            nonlin()
        )
        self.w = torch.nn.Sequential(
            torch.nn.LSTM(1,hid_dim),
            torch.nn.Linear(hid_dim,hid_dim),
            nonlin()
        )
        self.g = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim, bias=True),
            nonlin()
        )
        self.f = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, out_dim, bias=True),
            nonlin()
        )
        # self.a = MLP(in_dim=2*num_feat,out_dim=1,hid_dim=hid_dim,nonlin=nonlin)
        # self.w = LSTM(in_dim=1,out_dim=hid_dim,hid_dim=hid_dim,nonlin=nonlin)
        # self.g = MLP(in_dim=hid_dim,out_dim=hid_dim,hid_dim=hid_dim,nonlin=nonlin)
        # self.f = MLP(in_dim=num_speakers,out_dim=out_dim,hid_dim=hid_dim)
    def forward(self,psi,time_since):
        assert len(psi)==time_since.shape[1], "Inconsistent number of speakers."
        num_speakers = len(psi)

        alpha = torch.Tensor([[ self.a(torch.cat([psi[i],psi[j]])) for j in range(num_speakers)] for i in range(num_speakers)])
        phi = torch.zeros(time_since.shape[0],self.hid_dim,num_speakers)
        for s in range(num_speakers):
            if len(~time_since[:,s].isinf())>0:
                phi[~time_since[:,s].isinf(),:,s] = self.w[2](self.w[1](self.w[0](time_since[:,s][~time_since[:,s].isinf()].unsqueeze(1))[0]))
        x = phi@alpha.T
        h = torch.stack([self.g(x[:,:,s]) for s in range(num_speakers)])
        f = torch.stack([self.f(h[s]) for s in range(num_speakers)]).squeeze(2).T
        return f


# ----------------------------------------------------------------

class GRACIEModel:
    def __init__(self,seed:int=1000,
                 cuda:bool=False,
                 model_has_memory:bool=True,
                 learn_memory_function:bool=True,
                 learn_affinity_function:bool=True,
                 w=None,
                 lstm:bool=False,
                 weight_func=None,
                 hid_dim:int=5,
                 lr:float=1e-3,
                 lr_decay:float=.5,
                 lr_affinity:float=.5,
                 lr_patience:int=10,
                 num_inner_iter:int=1,
                 num_iter:int=100,
                 decay_window:int=1,
                 affinity_window:int=1,
                 loss_scale:str="total",
                 test_loss_scale:str="total",
                 alpha:float=0.,
                 delta:float=0.,
                 terminate:bool=False,
                 burn_in_iter:int=10,
                 patience:int=10,
                 burn_in_valloss:int=0):
        assert type(seed)==int, "Invalid seed. Must be an integer."
        assert type(cuda)==bool, "Invalid value for CUDA flag `cuda`. Must be a boolean."
        assert type(model_has_memory)==bool, "Invalid value for `model_has_memory`. Must be a boolean."
        assert type(learn_memory_function)==bool, "Invalid value for `learn_memory_function`. Must be a boolean."
        assert type(learn_affinity_function)==bool, "Invalid value for `learn_affinity_function`. Must be a boolean."
        assert type(lstm)==bool, "Invalid value for `lstm`. Must be a boolean."
        assert (model_has_memory or (not learn_memory_function)), "Invalid values for `model_has_memory` and `learn_memory_function`. Cannot have `model_has_memory=False` and `learn_memory_function=True`."
        assert weight_func in [None,get_turn_type_agg_weights,get_turn_type_weights], "Invalid weight function. Must be one of [None,get_turn_type_agg_weights,get_turn_type_weights]."
        assert type(hid_dim)==int and hid_dim>0, "Invalid value for hidden dimensions `hid_dim`. Must be a positive integer."
        assert type(lr)==float and lr>=0, "Invalid learning rate `lr`. Must be a nonnegative float."
        assert type(lr_decay)==float and lr_decay>=0, "Invalid learning rate decay factor `lr_decay`. Must be a nonnegative float."
        assert type(lr_affinity)==float and lr_affinity>=0, "Invalid learning rate affinity factor `lr_affinity`. Must be a nonnegative float."
        assert type(lr_patience)==int and lr_patience>0, "Invalid value for number of learning rate patience iterations `lr_patience`. Must be a positive integer."
        assert type(num_inner_iter)==int and num_inner_iter>0, "Invalid value for number of iterations `num_inner_iter`. Must be a positive integer."
        assert type(num_iter)==int and num_iter>0, "Invalid value for number of iterations `num_iter`. Must be a positive integer."
        assert type(decay_window)==int and decay_window>0, "Invalid value for number of iterations `decay_window`. must be a positive integer."
        assert type(affinity_window)==int and affinity_window>0, "Invalid value for number of iterations `affinity_window`. must be a positive integer."
        assert loss_scale in ["convo","group","total"], "Invalid loss scaling type `loss_scale`. Must be one of ['convo','group','total']."
        assert test_loss_scale in ["convo","group","total"], "Invalid loss scaling type `test_loss_scale`. Must be one of ['convo','group','total']."
        assert type(alpha)==float and alpha>=0., "Invalid affinity penalty weight `alpha`. Must be a nonnegative float."
        assert type(delta)==float and delta>=0., "Invalid memory penalty weight `delta`. Must be a nonnegative float."
        assert type(terminate)==bool, "Invalid value for termination flag `terminate`. Must be a boolean."
        assert (not terminate) or (type(burn_in_iter)==int and burn_in_iter>0), "Invalid value for number of burn-in iterations `burn_in_iter`. Must be a positive integer."
        assert (not terminate) or (type(patience)==int and patience>0), "Invalid value for number of termination patience iterations `patience`. Must be a positive integer."
        assert type(burn_in_valloss)==int and burn_in_valloss>=0, "Invalid value for minimum validation loss iteration `burn_in_valloss`. Must be a nonnegative integer."

        self.seed = seed
        self.cuda = cuda
        self.model_has_memory = model_has_memory
        self.learn_memory_function = learn_memory_function
        self.learn_affinity_function = learn_affinity_function
        self.lstm = lstm
        self.weight_func = weight_func
        self.hid_dim = hid_dim
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_affinity = lr_affinity
        self.lr_patience = lr_patience
        self.num_inner_iter = num_inner_iter if self.learn_memory_function else 1
        self.num_iter = num_iter
        self.decay_window = decay_window
        self.affinity_window = affinity_window
        self.loss_scale = loss_scale
        self.test_loss_scale = test_loss_scale
        self.alpha = alpha
        self.delta = delta
        self.terminate = terminate
        self.burn_in_iter = burn_in_iter
        self.patience = patience
        self.burn_in_valloss = burn_in_valloss

        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.cuda) else 'cpu')
        torch.manual_seed(self.seed)
        
        self.out_dim = 1+int(self.model_has_memory)
        self.model = None
        self.optimizer = None
        self.scheduler = None

        self.decay = None
        self.optim_dec = None
        self.sched_dec = None

        self.affinity = None
        self.optim_aff = None
        self.sched_aff = None
        
        if w is None:
            w = lambda x: torch.exp(-.5*x) if type(x) is torch.Tensor else (np.exp(-.5*x).item() if np.isscalar(x) else np.exp(-.5*x))
        self.w = None if self.learn_memory_function else w
        self.a = None if self.learn_affinity_function else lambda x,y:0.

        self.num_feat = None
        self.train_data_pred = None
        self.val_data_pred = None
        self.test_data_pred = None
        self.loss_train_pred = None
        self.loss_val_pred = None
        self.loss_test_pred = None
        self.loss_train_list = None
        self.dloss_train_list = None
        self.loss_val_list = None
        self.best_loss_val = None
        self.best_iter = None
        self.log = None
    
    def train(self,train_data:Dataset,val_data:Dataset=None,test_data:Dataset=None,verbose:bool=True):
        assert (val_data is None) or (train_data.num_feat==val_data.num_feat), "Inconsistent number of features between train and validation data."
        assert (test_data is None) or (train_data.num_feat==test_data.num_feat), "Inconsistent number of features between train and test data."

        # ------------------------------------
        # Prepare iteration lists for loss and any log messages
        loss_train_list = []
        loss_val_list = None if val_data is None else []
        dloss_train_list = None if not self.learn_memory_function else []
        log = []
        # ------------------------------------


        # ------------------------------------
        # Initialize models to be trained
        self.num_feat = train_data.num_feat
        self.model = MLP(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_iter, eta_min=1e-5)
        if self.learn_memory_function:
            self.decay = (MLP(in_dim=1, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device) if not self.lstm else 
                          LSTM(in_dim=1, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device))
            self.optim_dec = torch.optim.SGD(self.decay.parameters(), lr=self.lr_decay)
            self.sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dec, T_max=self.num_iter, eta_min=1e-5)
        if self.learn_affinity_function:
            self.affinity = MLP(in_dim=self.num_feat*2, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device)
            self.optim_aff = torch.optim.SGD(self.affinity.parameters(), lr=self.lr_affinity)
            self.sched_aff = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_aff, T_max=self.num_iter, eta_min=1e-5)
        # ------------------------------------


        # ------------------------------------
        # Prepare to search for best model during training
        best_model = MLP(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim).to(self.device)
        best_model.load_state_dict(self.model.state_dict())
        if self.learn_memory_function:
            best_decay = (MLP(in_dim=1, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device) if not self.lstm else 
                          LSTM(in_dim=1, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device))
            best_decay.load_state_dict(self.decay.state_dict())
        if self.learn_affinity_function:
            best_affinity = MLP(in_dim=self.num_feat*2, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device)
            best_affinity.load_state_dict(self.affinity.state_dict())
        best_loss = float('inf')
        best_iter = 0
        # ------------------------------------

        # ------------------------------------
        # Prepare data
        train_data = train_data.to(self.device)
        val_data = None if val_data is None else val_data.to(self.device)
        test_data = None if test_data is None else test_data.to(self.device)

        # Output predictions follow same composition as input data
        train_data_pred = train_data.copy()
        val_data_pred = None if val_data is None else val_data.copy()
        test_data_pred = None if test_data is None else test_data.copy()

        # If learning memory function, initialize with something totally uninformative
        if self.learn_memory_function:
            betas_fixed_train = [[(.2*torch.ones_like(train_data.time_since[g][c])).to(self.device) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            betas_pred_val = None if val_data is None else [[(.2*torch.ones_like(val_data.time_since[g][c])).to(self.device) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
            betas_pred_test = None if test_data is None else [[(.2*torch.ones_like(test_data.time_since[g][c])).to(self.device) for c in range(test_data.num_convos[g])] for g in range(test_data.num_groups)]
        # If not learning memory function, compute time-dependent effect using the given memory function self.w
        else:
            betas_fixed_train = [[compute_betas(self.w,train_data.time_since[g][c]).to(self.device) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            betas_pred_val = None if val_data is None else [[compute_betas(self.w,val_data.time_since[g][c]).to(self.device) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
            betas_pred_test = None if test_data is None else [[compute_betas(self.w,test_data.time_since[g][c]).to(self.device) for c in range(test_data.num_convos[g])] for g in range(test_data.num_groups)]

        # If learning pairwise affinities, initialize with all ones, that is, assume some level of affinity between all pairs
        if self.learn_affinity_function:
            A_fixed_train = [.1*(1.-torch.eye(train_data.num_speakers[g])).to(self.device) for g in range(train_data.num_groups)]
            A_pred_val = None if val_data is None else [.1*(1.-torch.eye(val_data.num_speakers[g])).to(self.device) for g in range(val_data.num_groups)]
            A_pred_test = None if test_data is None else [.1*(1.-torch.eye(test_data.num_speakers[g])).to(self.device) for g in range(test_data.num_groups)]
        # If not learning pairwise affinities, set all pairwise affinities to zero since the model is ignoring them
        else:
            A_fixed_train = [torch.zeros((train_data.num_speakers[g],train_data.num_speakers[g])).to(self.device) for g in range(train_data.num_groups)]
            A_pred_val = [torch.zeros((val_data.num_speakers[g],val_data.num_speakers[g])).to(self.device) for g in range(val_data.num_groups)]
            A_pred_test = [torch.zeros((test_data.num_speakers[g],test_data.num_speakers[g])).to(self.device) for g in range(test_data.num_groups)]

        # Make sure the predictions are not using the pairwise affinities in the input data
        train_data_pred.set_A(A_fixed_train)
        val_data_pred.set_A(A_pred_val)
        test_data_pred.set_A(A_pred_test)
        # ------------------------------------

        tic = perf_counter()
        for train_iter in range(self.num_iter):
            # ------------------------------------
            # Prepare model and affinity for training
            self.model.train()
            if self.learn_memory_function:
                self.decay.eval()
            if self.learn_affinity_function:
                self.affinity.train()
            # ------------------------------------


            # ------------------------------------
            # Train models for predicting pi and d (and A if learning pairwise affinities)
            for inner_iter in range(self.num_inner_iter):
                # Predict pi and d for training data
                preds_train = self.model(train_data.tensor)
                pis_pred_train = [preds_train[train_data.inds_grps[g]][:,0] for g in range(train_data.num_groups)]
                ds_pred_train = [preds_train[train_data.inds_grps[g]][:,1] if self.model_has_memory else torch.zeros(train_data.num_speakers[g]).to(self.device) for g in range(train_data.num_groups)]
                train_data_pred.pi = pis_pred_train
                train_data_pred.d = ds_pred_train

                # Predict A for training data (if learning pairwise affinities)
                if self.learn_affinity_function:
                    preds_train_aff = self.affinity(train_data.pairwise_tensor)
                    A_pred_train = [preds_train_aff[train_data.pair_inds_grps[g]].reshape(train_data.num_speakers[g],train_data.num_speakers[g]) for g in range(train_data.num_groups)]
                else:
                    A_pred_train = [torch.zeros((train_data.num_speakers[g],train_data.num_speakers[g])).to(self.device) for g in range(train_data.num_groups)]
                train_data_pred.A = A_pred_train

                # Update models
                loss_train = compute_loss(train_data_pred,betas_fixed_train,self.w,self.loss_scale,self.weight_func,self.device)

                if self.alpha>0:
                    for g in range(train_data.num_groups):
                        loss_train += self.alpha * torch.linalg.norm(A_pred_train[g],'fro')**2
                if self.delta>0:
                    for g in range(train_data.num_groups):
                        loss_train += self.delta * torch.linalg.norm(ds_pred_train[g])**2
                
                loss_train_list.append(loss_train.detach())
                self.optimizer.zero_grad()
                if self.learn_affinity_function:
                    self.optim_aff.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.learn_affinity_function:
                    self.optim_aff.step()
                    self.sched_aff.step()

            train_data_fixed = train_data_pred.copy()
            train_data_fixed.set_pi([a.clone().detach() for a in train_data_fixed.pi])
            train_data_fixed.set_d([a.clone().detach() for a in train_data_fixed.d])
            train_data_fixed.set_A([a.clone().detach() for a in train_data_fixed.A])
            # ------------------------------------


            # ------------------------------------
            # Prepare model and affinity for evaluation
            self.model.eval()
            if self.learn_affinity_function:
                self.affinity.eval()
            # ------------------------------------


            # ------------------------------------
            # Train model for prediction memory function w
            if self.learn_memory_function and train_iter%self.decay_window==0:
                self.decay.train()
                for inner_iter in range(self.num_inner_iter):
                    # Predict beta, output of w given conversations in dataset
                    betas_pred_train = [[compute_betas(self.decay,train_data.time_since[g][c]) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]

                    # Update model
                    dloss_train = compute_loss(train_data_fixed,betas_pred_train,self.w,self.loss_scale,self.weight_func,self.device)
                    dloss_train_list.append(dloss_train.detach())
                    self.optim_dec.zero_grad()
                    dloss_train.backward()
                    self.optim_dec.step()
                    self.sched_dec.step()

                betas_fixed_train = [[betas_pred_train[g][c].clone().detach() for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            # ------------------------------------


            # ------------------------------------
           # Evaluate model on validation data (if given)
            if self.learn_memory_function:
                self.decay.eval()
            if val_data is not None:
                preds_val = self.model(val_data.tensor)
                pis_pred_val = [preds_val[val_data.inds_grps[g]][:,0].detach().to(self.device) for g in range(val_data.num_groups)]
                ds_pred_val = [preds_val[val_data.inds_grps[g]][:,1].detach().to(self.device) if self.model_has_memory else torch.zeros(train_data.num_speakers[g]).to(self.device) for g in range(val_data.num_groups)]
                if self.learn_affinity_function:
                    preds_val_aff = self.affinity(val_data.pairwise_tensor)
                    A_pred_val = [preds_val_aff[val_data.pair_inds_grps[g]].reshape(val_data.num_speakers[g],val_data.num_speakers[g]) for g in range(val_data.num_groups)]
                if self.learn_memory_function:
                    betas_pred_val = [[compute_betas(self.decay,val_data.time_since[g][c]) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
                    betas_pred_val = [[betas_pred_val[g][c].clone().detach() for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
                val_data_pred.set_pi(pis_pred_val)
                val_data_pred.set_d(ds_pred_val)
                val_data_pred.set_A(A_pred_val)
                loss_val = compute_loss(val_data_pred,betas_pred_val,self.w,self.test_loss_scale,self.weight_func,self.device)
                loss_val_list.append(loss_val.detach())
            # ------------------------------------
            

            # ------------------------------------
            # Update best model if validation loss improves (and if validation data given)
            if (val_data is not None and train_iter>=self.burn_in_valloss and loss_val_list[-1]<=best_loss):
                best_loss = loss_val_list[-1]
                best_model.load_state_dict(self.model.state_dict())
                if self.learn_memory_function:
                    best_decay.load_state_dict(self.decay.state_dict())
                if self.learn_affinity_function:
                    best_affinity.load_state_dict(self.affinity.state_dict())
                best_iter = train_iter
            elif (val_data is None):
                best_loss = loss_train_list[-1]
                best_model.load_state_dict(self.model.state_dict())
                if self.learn_memory_function:
                    best_decay.load_state_dict(self.decay.state_dict())
                if self.learn_affinity_function:
                    best_affinity.load_state_dict(self.affinity.state_dict())
                best_iter = train_iter
            # ------------------------------------


            # ------------------------------------
            # Terminate early if conditions hold
            #   If validation loss nondecreasing for long enough or validation loss has very low variance for long enough
            if self.terminate and (val_data is not None):
                # if (train_iter>self.burn_in_iter + self.burn_in_valloss and
                if (train_iter>self.burn_in_iter and 
                    (all(np.diff(np.array(loss_val_list[-self.patience:]))>=0) or
                    # (all(torch.tensor(loss_val_list[-self.patience:]) >= float(loss_val_list[-self.patience])) or 
                     torch.std(torch.tensor(loss_val_list[-2*self.patience:]))<=1e-6 ) ):
                    if verbose:
                        msg = f"TERMINATE @ {train_iter+1} iterations"
                        log.append(msg)
                        print(msg)
                    break
            # ------------------------------------


            # ------------------------------------
            # Print out messages (optional)
            toc = perf_counter()
            if verbose and ((train_iter+1)%100==0 or train_iter==0 or toc-tic>=10):
                msg = (f"ITER. {train_iter+1}/{self.num_iter} | " + 
                    f"Train. loss: {loss_train_list[-1]:.4f}")
                if self.learn_memory_function and len(dloss_train_list)>0:
                    msg += f" | Dec. loss: {dloss_train_list[-1]:.4f}"
                if val_data is not None:
                    msg += f" | Val. loss: {loss_val_list[-1]:.4f}"
                    msg += f" | Best loss: {best_loss:.4f} at iter. {best_iter}"
                log.append(msg)
                print(msg)
                tic = perf_counter()
            # ------------------------------------


        # ------------------------------------
        # Load the final best model obtained during training
        self.model.load_state_dict(best_model.state_dict())
        self.model.eval()
        if self.learn_memory_function:
            self.decay.load_state_dict(best_decay.state_dict())
            self.decay.eval()
        if self.learn_affinity_function:
            self.affinity.load_state_dict(best_affinity.state_dict())
            self.affinity.eval()
        # ------------------------------------


        # ------------------------------------
        # Learned affinity or memory functions
        if self.learn_memory_function:
            def w(x):
                if np.isscalar(x):
                    inp = torch.tensor([x]).to(torch.float)
                else:
                    inp = torch.Tensor(x).unsqueeze(1).to(torch.float)
                y = self.decay(inp.to(self.device)).squeeze().to(inp.device).item() if np.isscalar(x) or inp.numel()==1 else \
                    self.decay(inp.to(self.device)).squeeze().detach().to(inp.device)
                return y
            self.w = w
        if self.learn_affinity_function:
            def a(x,y):
                assert (np.isscalar(x) and np.isscalar(y)) or (not np.isscalar(x) and not np.isscalar(y)), "Invalid inputs. Expecting pair of inputs of same size."
                inp = torch.tensor([x,y]).to(torch.float) if np.isscalar(x) else torch.cat((torch.Tensor(x).to(torch.float), torch.Tensor(y).to(torch.float)))
                s = self.affinity(inp).squeeze().detach().to(inp.device)
                return s
            self.a = a
        # ------------------------------------


        # ------------------------------------
        # Final predictions
        loss_train_pred, train_data_pred = self.test(train_data)
        if val_data is not None:
            loss_val_pred, val_data_pred = self.test(val_data)
        if test_data is not None:
            loss_test_pred, test_data_pred = self.test(test_data)

        if loss_val_list is not None:
            loss_val_rep = torch.tensor(loss_val_list).repeat_interleave(self.num_inner_iter)
            loss_val_rep = loss_val_rep[:len(loss_train_list)]
            loss_val_list = loss_val_rep
        if dloss_train_list is not None:
            dloss_train_rep = torch.tensor(dloss_train_list).repeat_interleave(self.decay_window)
            dloss_train_rep = dloss_train_rep[:len(loss_train_list)]
            dloss_train_rep = torch.cat([dloss_train_rep, dloss_train_rep[-1]*torch.ones(len(loss_train_list)-len(dloss_train_rep))])
            dloss_train_rep = torch.cat([dloss_train_rep[0]*torch.ones(len(loss_train_list)-len(dloss_train_rep)), dloss_train_rep ])
            dloss_train_list = dloss_train_rep
        # ------------------------------------

        if verbose and val_data is not None:
            msg = f"Best validation loss: {best_loss:.5f} at iteration {best_iter}"
            log.append(msg)
            print(msg)

        self.train_data_pred = train_data_pred
        self.val_data_pred = val_data_pred
        self.test_data_pred = test_data_pred
        self.loss_train_pred = loss_train_pred
        if val_data is not None:
            self.loss_val_pred = loss_val_pred
        if test_data is not None:
            self.loss_test_pred = loss_test_pred
        self.loss_train_list = torch.tensor(loss_train_list)
        if val_data_pred is not None:
            self.loss_val_list = torch.tensor(loss_val_list)
        if self.learn_memory_function:
            self.dloss_train_list = torch.tensor(dloss_train_list)
        self.best_loss_val = best_loss
        self.best_iter = best_iter
        self.log = log

        return
    
    def predict(self,test_data:Dataset):
        test_data = test_data.to(self.device)

        preds = self.model(test_data.tensor)
        pis_pred = [preds[test_data.inds_grps[g]][:,0].detach() for g in range(test_data.num_groups)]
        ds_pred = [preds[test_data.inds_grps[g]][:,1].detach() if self.model_has_memory else torch.zeros(test_data.num_speakers[g]).to(self.device) for g in range(test_data.num_groups)]

        if self.learn_affinity_function:
            preds_aff = self.affinity(test_data.pairwise_tensor)
            A_pred = [preds_aff[test_data.pair_inds_grps[g]].reshape(test_data.num_speakers[g],test_data.num_speakers[g]).detach() for g in range(test_data.num_groups)]
        else:
            A_pred = [torch.zeros((test_data.num_speakers[g],test_data.num_speakers[g])).to(self.device) for g in range(test_data.num_groups)]

        grps_pred_test = [Group(test_data.psi[g], pis_pred[g], ds_pred[g], A_pred[g], test_data.convos[g]) for g in range(test_data.num_groups)]
        test_data_pred = Dataset(grps_pred_test)

        return test_data_pred

    def test(self,test_data:Dataset):
        test_data_pred = self.predict(test_data)
        betas_test_pred = [[compute_betas(self.w,test_data_pred.time_since[g][c]) for c in range(test_data_pred.num_convos[g])] for g in range(test_data_pred.num_groups)]
        loss_test = float(compute_loss(test_data_pred,betas_test_pred,self.w,self.test_loss_scale,weight_func=self.weight_func,device=self.device))

        return loss_test, test_data_pred
    
    def get_loss_list_df(self):
        assert hasattr(self,"loss_train_list") and (self.loss_train_list is not None), "No attribute available."

        if self.loss_val_list is None and self.dloss_train_list is None:
            loss_df = pd.DataFrame(torch.stack([self.loss_train_list]).T, columns=["loss_train"])
        elif self.loss_val_list is None:
            loss_df = pd.DataFrame(torch.stack([self.loss_train_list,self.dloss_train_list]).T, columns=["loss_train","dloss_train"])
        elif not self.learn_memory_function:
            loss_df = pd.DataFrame(torch.stack([self.loss_train_list,self.loss_val_list]).T, columns=["loss_train","loss_val"])
        else:
            loss_df = pd.DataFrame(torch.stack([self.loss_train_list,self.dloss_train_list,self.loss_val_list]).T, columns=["loss_train","dloss_train","loss_val"])

        return loss_df

    def get_loss_pred_df(self):
        assert hasattr(self,"loss_train_pred") and hasattr(self,"loss_val_pred") and hasattr(self,"loss_test_pred") and (self.loss_train_list is not None), "No attribute available."
        assert self.loss_train_pred is not None or self.loss_val_pred is not None or self.loss_test_pred is not None, "No attribute available." 

        loss_pred_df = pd.DataFrame(data=np.array([[self.loss_train_pred,self.loss_val_pred,self.loss_test_pred]]), 
                                    columns=["loss_train","loss_val","loss_test"])
        return loss_pred_df

    def save_model(self,path:str="",name:str=""):
        assert self.model is not None, "Model not trained."
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        filename = f"model{name}.pt"
        filepath = os.path.join(path,filename)
        torch.save(self.model.state_dict(), filepath)

        if self.learn_memory_function:
            filename_dec = f"decay{name}.pt"
            filepath_dec = os.path.join(path,filename_dec)
            torch.save(self.decay.state_dict(), filepath_dec)

        if self.learn_affinity_function:
            filename_aff = f"affinity{name}.pt"
            filepath_aff = os.path.join(path,filename_aff)
            torch.save(self.affinity.state_dict(), filepath_aff)

        msg = "\n".join(self.log)
        with open( os.path.join(path,f"log{name}.txt"), "w" ) as f:
            f.write(msg)
        
        return

    def save_results(self,path:str="",name:str="_pred"):
        assert self.model is not None, "Model not trained. Train model by running `.train` method."

        train_names = [f"{v}_train{name}" for v in ["psi","pi","d","A","convo"]]
        self.train_data_pred.save(*train_names,path=path)
        
        if self.val_data_pred is not None:
            val_names = [f"{v}_val{name}" for v in ["psi","pi","d","A","convo"]]
            self.val_data_pred.save(*val_names,path=path)

        if self.test_data_pred is not None:
            test_names = [f"{v}_test{name}" for v in ["psi","pi","d","A","convo"]]
            self.test_data_pred.save(*test_names,path=path)

        loss_list_df = self.get_loss_list_df()
        loss_list_df.to_csv(os.path.join(path,f"loss_list{name}.csv"),index=False)

        loss_pred_df = self.get_loss_pred_df()
        loss_pred_df.to_csv(os.path.join(path,f"loss{name}.csv"),index=False)

        return
    
    def load_model(self,path:str="",name:str=""):
        filename = f"model{name}.pt"
        filepath = os.path.join(path,filename)
        assert os.path.exists(filepath), "Model does not exist."
        if self.learn_memory_function:
            filename_dec = f"decay{name}.pt"
            filepath_dec = os.path.join(path,filename_dec)
            assert os.path.exists(filepath_dec), "Decay model does not exist."
        if self.learn_affinity_function:
            filename_aff = f"affinity{name}.pt"
            filepath_aff = os.path.join(path,filename_aff)
            assert os.path.exists(filepath_aff), "Affinity model does not exist."
        
        num_feat = torch.load(filepath)['layer1.weight'].shape[1]
        self.num_feat = num_feat
        self.model = MLP(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim).to(self.device)
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        if self.learn_memory_function:
            self.decay = (MLP(in_dim=1, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device) if not self.lstm else 
                          LSTM(in_dim=1, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device))
            self.decay.load_state_dict(torch.load(filepath_dec))
            self.decay.eval()

            def w(x):
                if np.isscalar(x):
                    inp = torch.tensor([x]).to(torch.float)
                else:
                    inp = torch.Tensor(x).unsqueeze(1).to(torch.float)
                y = self.decay(inp.to(self.device)).squeeze().to(inp.device).item() if np.isscalar(x) or inp.numel()==1 else \
                    self.decay(inp.to(self.device)).squeeze().detach().to(inp.device)
                return y
            self.w = w
        if self.learn_affinity_function:
            self.affinity = MLP(in_dim=self.num_feat*2, out_dim=1, hid_dim=self.hid_dim, nonlin=torch.nn.Sigmoid).to(self.device)
            self.affinity.load_state_dict(torch.load(filepath_aff))
            self.affinity.eval()

    def load_results(self,path:str="",name:str="_pred"):

        psi_train = load_psi(f"psi_train{name}",path=path)
        pi_train = load_pi(f"pi_train{name}",path=path)
        d_train = load_d(f"d_train{name}",path=path)
        A_train = load_A(f"A_train{name}",path=path)
        convo_train = load_convo(f"convo_train{name}",path=path)
        self.train_data_pred = Dataset(groups=Groups_from_lists(psi_train,pi_train,d_train,A_train,convo_train))

        psi_val = load_psi(f"psi_val{name}",path=path) if os.path.exists(os.path.join(path,f"psi_val{name}.csv")) else None
        pi_val = load_pi(f"pi_val{name}",path=path) if os.path.exists(os.path.join(path,f"pi_val{name}.csv")) else None
        d_val = load_d(f"d_val{name}",path=path) if os.path.exists(os.path.join(path,f"d_val{name}.csv")) else None
        A_val = load_A(f"A_val{name}",path=path) if os.path.exists(os.path.join(path,f"A_val{name}.csv")) else None
        convo_val = load_convo(f"convo_val{name}",path=path) if os.path.exists(os.path.join(path,f"convo_val{name}.csv")) else None
        if psi_val is not None:
            self.val_data_pred = Dataset(groups=Groups_from_lists(psi_val,pi_val,d_val,A_val,convo_val))

        psi_test = load_psi(f"psi_test{name}",path=path) if os.path.exists(os.path.join(path,f"psi_test{name}.csv")) else None
        pi_test = load_pi(f"pi_test{name}",path=path) if os.path.exists(os.path.join(path,f"pi_test{name}.csv")) else None
        d_test = load_d(f"d_test{name}",path=path) if os.path.exists(os.path.join(path,f"d_test{name}.csv")) else None
        A_test = load_A(f"A_test{name}",path=path) if os.path.exists(os.path.join(path,f"A_test{name}.csv")) else None
        convo_test = load_convo(f"convo_test{name}",path=path) if os.path.exists(os.path.join(path,f"convo_test{name}.csv")) else None
        if psi_test is not None:
            self.test_data_pred = Dataset(groups=Groups_from_lists(psi_test,pi_test,d_test,A_test,convo_test))

        loss_list_df = pd.read_csv(os.path.join(path,f"loss_list{name}.csv"))
        self.loss_train_list = torch.tensor(loss_list_df['loss_train'].to_list())
        if "loss_val" in loss_list_df.columns:
            self.loss_val_list = torch.tensor(loss_list_df['loss_val'].to_list())
        if "dloss_train" in loss_list_df.columns:
            self.dloss_val_list = torch.tensor(loss_list_df['dloss_train'].to_list())

        loss_pred_df = pd.read_csv(os.path.join(path,f"loss{name}.csv"))
        self.loss_train_pred = loss_pred_df["loss_train"].item() if not np.isnan(loss_pred_df["loss_train"].item()) else None
        self.loss_val_pred = loss_pred_df["loss_val"].item() if not np.isnan(loss_pred_df["loss_val"].item()) else None
        self.loss_test_pred = loss_pred_df["loss_test"].item() if not np.isnan(loss_pred_df["loss_test"].item()) else None

def save_trained_models_list(models:List[List[GRACIEModel]], path:str="", filetag:str=""):
    data_trials = len(models)
    model_trials = len(models[0])
    assert all([len(models[i])==model_trials for i in range(len(models))]), "Inconsistent number of model trials across data trials."

    for dt in range(data_trials):
        for mt in range(model_trials):
            tag = f"{filetag}_d{dt}_m{mt}"
            models[dt][mt].save_model(path, tag)

    return

def save_preds_list(models:List[List[GRACIEModel]], split:str="train", path:str="", name:str="_pred"):
    data_trials = len(models)
    model_trials = len(models[0])
    assert all([len(models[i])==model_trials for i in range(len(models))]), "Inconsistent number of model trials across data trials."
    assert split in ["train","val","test"], "Invalid split choice. Expecting 'train', 'val', or 'test'."
    if split=="train":
        pred_data = [[models[dt][mt].train_data_pred for mt in range(model_trials)] for dt in range(data_trials)]
    elif split=="val":
        pred_data = [[models[dt][mt].val_data_pred for mt in range(model_trials)] for dt in range(data_trials)]
    elif split=="test":
        pred_data = [[models[dt][mt].test_data_pred for mt in range(model_trials)] for dt in range(data_trials)]

    psi_df_list = [[pred_data[dt][mt].get_psi_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(psi_df_list[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    psi_df = pd.concat(sum(psi_df_list,[]))
    psi_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    psi_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    psi_df = psi_df[["Data_trial","Model_trial","Group","Speaker","Feat","Psi"]]
    psi_df.to_csv(os.path.join(path,f"psi_{split}{name}.csv"),index=False)

    pi_df_list = [[pred_data[dt][mt].get_pi_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(pi_df_list[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    pi_df = pd.concat(sum(pi_df_list,[]))
    pi_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    pi_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    pi_df = pi_df[["Data_trial","Model_trial","Group","Speaker","Pi"]]
    pi_df.to_csv(os.path.join(path,f"pi_{split}{name}.csv"),index=False)

    d_df_list = [[pred_data[dt][mt].get_d_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(d_df_list[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    d_df = pd.concat(sum(d_df_list,[]))
    d_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    d_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    d_df = d_df[["Data_trial","Model_trial","Group","Speaker","D"]]
    d_df.to_csv(os.path.join(path,f"d_{split}{name}.csv"),index=False)

    A_df_list = [[pred_data[dt][mt].get_A_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(A_df_list[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    A_df = pd.concat(sum(A_df_list,[]))
    A_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    A_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    A_df = A_df[["Data_trial","Model_trial","Group","Speaker1","Speaker2","A"]]
    A_df.to_csv(os.path.join(path,f"A_{split}{name}.csv"),index=False)

    convo_df_list = [[pred_data[dt][mt].get_convo_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(convo_df_list[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    convo_df = pd.concat(sum(convo_df_list,[]))
    convo_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    convo_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    convo_df = convo_df[["Data_trial","Model_trial","Group","Convo","Turn","Speaker"]]
    convo_df.to_csv(os.path.join(path,f"convo_{split}{name}.csv"),index=False)

    return

def save_model_results_list(models:List[List[GRACIEModel]], path:str="", name:str="_pred"):
    data_trials = len(models)
    model_trials = len(models[0])
    assert all([len(models[i])==model_trials for i in range(len(models))]), "Inconsistent number of model trials across data trials."

    save_preds_list(models,"train",path,name)
    save_preds_list(models,"val",path,name)
    save_preds_list(models,"test",path,name)

    loss_list_df_list = [[models[dt][mt].get_loss_list_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(loss_list_df_list[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    loss_list_df = pd.concat(sum(loss_list_df_list,[]))
    loss_list_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt,mt],dtype=int) for dt in range(data_trials) for mt in range(model_trials)],axis=0)
    loss_list_df["Model_trial"] = np.concatenate([mt*np.ones(num_ids[dt,mt],dtype=int) for dt in range(data_trials) for mt in range(model_trials)],axis=0)
    # loss_list_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    # loss_list_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    loss_list_df = loss_list_df[["Data_trial","Model_trial","loss_train","loss_val"]]
    loss_list_df.to_csv(os.path.join(path,f"loss_list{name}.csv"),index=False)

    loss_df = [[models[dt][mt].get_loss_pred_df() for mt in range(model_trials)] for dt in range(data_trials)]
    num_ids = np.array([[len(loss_df[dt][mt]) for mt in range(model_trials)] for dt in range(data_trials)])
    loss_df = pd.concat(sum(loss_df,[]))
    loss_df["Data_trial"] = np.concatenate([dt*np.ones(num_ids[dt].sum(),dtype=int) for dt in range(data_trials)],axis=0)
    loss_df["Model_trial"] = np.concatenate([np.arange(model_trials).repeat(num_ids[dt][0]) for dt in range(data_trials)],axis=0)
    loss_df = loss_df[["Data_trial","Model_trial","loss_train","loss_val","loss_test"]]
    loss_df.to_csv(os.path.join(path,f"loss{name}.csv"),index=False)

    return

def load_preds_list(models:List[List[GRACIEModel]], split:str="train", path:str="", name:str="_pred"):
    data_trials = len(models)
    model_trials = len(models[0])
    assert all([len(models[i])==model_trials for i in range(len(models))]), "Inconsistent number of model trials across data trials."
    assert split in ["train","val","test"], "Invalid split choice. Expecting 'train', 'val', or 'test'."

    if split=="train":
        assert os.path.exists(os.path.join(path,f"psi_{split}{name}.csv")), f"File `psi_{split}{name}.csv` not available."
        assert os.path.exists(os.path.join(path,f"pi_{split}{name}.csv")), f"File `pi_{split}{name}.csv` not available."
        assert os.path.exists(os.path.join(path,f"d_{split}{name}.csv")), f"File `d_{split}{name}.csv` not available."
        assert os.path.exists(os.path.join(path,f"A_{split}{name}.csv")), f"File `A_{split}{name}.csv` not available."
        assert os.path.exists(os.path.join(path,f"convo_{split}{name}.csv")), f"File `convo_{split}{name}.csv` not available."
    else:
        if not os.path.exists(os.path.join(path,f"psi_{split}{name}.csv")):
            return models

    psi_df = pd.read_csv(os.path.join(path,f"psi_{split}{name}.csv"))
    assert data_trials==psi_df["Data_trial"].nunique(), "Inconsistent number of data trials."
    assert model_trials==psi_df["Model_trial"].nunique(), "Inconsistent number of model trials."
    num_feat = psi_df["Feat"].nunique()
    psi_gb = psi_df.groupby(["Data_trial","Model_trial"])
    psi_list = [[psi_from_DataFrame(psi_gb.get_group((dt,mt))[["Group","Speaker","Feat","Psi"]]) for mt in range(model_trials)] for dt in range(data_trials)]

    pi_df = pd.read_csv(os.path.join(path,f"pi_{split}{name}.csv"))
    assert data_trials==pi_df["Data_trial"].nunique(), "Inconsistent number of data trials."
    assert model_trials==pi_df["Model_trial"].nunique(), "Inconsistent number of model trials."
    pi_gb = pi_df.groupby(["Data_trial","Model_trial"])
    pi_list = [[pi_from_DataFrame(pi_gb.get_group((dt,mt))[["Group","Speaker","Pi"]]) for mt in range(model_trials)] for dt in range(data_trials)]

    d_df = pd.read_csv(os.path.join(path,f"d_{split}{name}.csv"))
    assert data_trials==d_df["Data_trial"].nunique(), "Inconsistent number of data trials."
    assert model_trials==d_df["Model_trial"].nunique(), "Inconsistent number of model trials."
    d_gb = d_df.groupby(["Data_trial","Model_trial"])
    d_list = [[d_from_DataFrame(d_gb.get_group((dt,mt))[["Group","Speaker","D"]]) for mt in range(model_trials)] for dt in range(data_trials)]

    A_df = pd.read_csv(os.path.join(path,f"A_{split}{name}.csv"))
    assert data_trials==A_df["Data_trial"].nunique(), "Inconsistent number of data trials."
    assert model_trials==A_df["Model_trial"].nunique(), "Inconsistent number of model trials."
    A_gb = A_df.groupby(["Data_trial","Model_trial"])
    A_list = [[A_from_DataFrame(A_gb.get_group((dt,mt))[["Group","Speaker1","Speaker2","A"]]) for mt in range(model_trials)] for dt in range(data_trials)]

    convo_df = pd.read_csv(os.path.join(path,f"convo_{split}{name}.csv"))
    assert data_trials==convo_df["Data_trial"].nunique(), "Inconsistent number of data trials."
    assert model_trials==convo_df["Model_trial"].nunique(), "Inconsistent number of model trials."
    convo_gb = convo_df.groupby(["Data_trial","Model_trial"])
    convo_list = [[convo_from_DataFrame(convo_gb.get_group((dt,mt))[["Group","Convo","Turn","Speaker"]]) for mt in range(model_trials)] for dt in range(data_trials)]

    pred_data = [Datasets_from_lists(psi_list[dt],pi_list[dt],d_list[dt],A_list[dt],convo_list[dt]) for dt in range(data_trials)]
    for dt in range(data_trials):
        for mt in range(model_trials):
            models[dt][mt].num_feat = num_feat
            if split=="train":
                models[dt][mt].train_data_pred = pred_data[dt][mt]
            elif split=="val":
                models[dt][mt].val_data_pred = pred_data[dt][mt]
            elif split=="test":
                models[dt][mt].test_data_pred = pred_data[dt][mt]

    return models

def load_model_results_list(models:List[List[GRACIEModel]],path:str="", name:str="_pred"):
    data_trials = len(models)
    model_trials = len(models[0])
    assert all([len(models[i])==model_trials for i in range(len(models))]), "Inconsistent number of model trials across data trials."

    models = load_preds_list(models,"train",path,name)
    models = load_preds_list(models,"val",path,name)
    models = load_preds_list(models,"test",path,name)

    loss_list_df = pd.read_csv(os.path.join(path,f"loss_list{name}.csv"))
    loss_list_gb = loss_list_df.groupby(["Data_trial","Model_trial"])

    columns = [col for col in loss_list_df.columns if col not in ["Data_trial","Model_trial"]]
    loss_list_dfs = [[loss_list_gb.get_group((dt,mt))[columns] for mt in range(model_trials)] for dt in range(data_trials)]

    for dt in range(data_trials):
        for mt in range(model_trials):
            models[dt][mt].loss_train_list = torch.tensor(loss_list_dfs[dt][mt]['loss_train'].to_list())
            if "loss_val" in loss_list_df.columns:
                models[dt][mt].loss_val_list = torch.tensor(loss_list_dfs[dt][mt]['loss_val'].to_list())
            if "dloss_train" in loss_list_df.columns:
                models[dt][mt].dloss_val_list = torch.tensor(loss_list_dfs[dt][mt]['dloss_train'].to_list())

    loss_df = pd.read_csv(os.path.join(path,f"loss{name}.csv"))
    loss_gb = loss_df.groupby(["Data_trial","Model_trial"])

    columns = [col for col in loss_df.columns if col not in ["Data_trial","Model_trial"]]
    loss_dfs = [[loss_gb.get_group((dt,mt))[columns] for mt in range(model_trials)] for dt in range(data_trials)]

    for dt in range(data_trials):
        for mt in range(model_trials):
            models[dt][mt].loss_train_pred = loss_dfs[dt][mt]["loss_train"].item() if not np.isnan(loss_dfs[dt][mt]["loss_train"].item()) else None
            models[dt][mt].loss_val_pred = loss_dfs[dt][mt]["loss_val"].item() if not np.isnan(loss_dfs[dt][mt]["loss_val"].item()) else None
            models[dt][mt].loss_test_pred = loss_dfs[dt][mt]["loss_test"].item() if not np.isnan(loss_dfs[dt][mt]["loss_test"].item()) else None
    
    return models

def load_trained_models_list(models:List[List[GRACIEModel]],path:str="", filetag:str=""):
    data_trials = len(models)
    model_trials = len(models[0])
    assert all([len(models[i])==model_trials for i in range(len(models))]), "Inconsistent number of model trials across data trials."

    list_files = os.listdir(path)
    model_list = [f for f in list_files if "model" in f and ".pt" in f and ("_d" in f and "_m" in f)]
    decay_list = [f for f in list_files if "decay" in f and ".pt" in f and ("_d" in f and "_m" in f)]
    affinity_list = [f for f in list_files if "affinity" in f and ".pt" in f and ("_d" in f and "_m" in f)]

    dt_list = [int(f[f.find("_d")+2:f.find("_m")]) for f in model_list]
    mt_list = [int(f[f.find("_m")+2:f.find(".pt")]) for f in model_list]

    assert data_trials==len(np.unique(dt_list)), "Inconsistent number of data trials."
    assert model_trials==len(np.unique(mt_list)), "Inconsistent number of model trials."

    names = [f[len("model"):f.find(".pt")] for f in model_list]
    if len(decay_list)>0:
        dec_names = [f[len("decay"):f.find(".pt")] for f in decay_list]
        assert set(dec_names) == set(names), "Inconsistent files between model and decay models."
    if len(affinity_list)>0:
        aff_names = [f[len("affinity"):f.find(".pt")] for f in affinity_list]
        assert set(aff_names) == set(names), "Inconsistent files between model and affinity models."
    trial_names = [[f for f in names if f"_d{dt}" in f] for dt in range(data_trials)]

    for dt in range(data_trials):
        for mt in range(model_trials):
            models[dt][mt].load_model(path,trial_names[dt][mt])

    return models

# ----------------------------------------------------------------

class FullGRACIEModel:
    def __init__(self, seed:int=1000,
                 cuda:bool=False,
                 weight_func=None,
                 hid_dim:int=5,
                 lr:float=1e-3,
                 lr_decay:float=.5,
                 lr_patience:int=10,
                 num_inner_iter:int=1,
                 num_iter:int=100,
                 decay_window:int=1,
                 loss_scale:str="total",
                 test_loss_scale:str="total",
                 terminate:bool=False,
                 burn_in_iter:int=10,
                 patience:int=10,
                 burn_in_valloss:int=0):
        assert type(seed)==int, "invalid seed. must be an integer."
        assert type(cuda)==bool, "invalid value for cuda flag `cuda`. must be a boolean."
        assert weight_func in [None,get_turn_type_agg_weights,get_turn_type_weights], "invalid weight function. must be one of [none,get_turn_type_agg_weights,get_turn_type_weights]."
        assert type(hid_dim)==int and hid_dim>0, "invalid value for hidden dimensions `hid_dim`. must be a positive integer."
        assert type(lr)==float and lr>=0, "invalid learning rate `lr`. must be a nonnegative float."
        assert type(lr_decay)==float and lr_decay>=0 and lr_decay<=1, "invalid learning rate decay factor `lr_decay`. must be a nonnegative float."
        assert type(lr_patience)==int and lr_patience>0, "invalid value for number of learning rate patience iterations `lr_patience`. must be a positive integer."
        assert type(num_inner_iter)==int and num_inner_iter>0, "invalid value for number of iterations `num_inner_iter`. must be a positive integer."
        assert type(num_iter)==int and num_iter>0, "invalid value for number of iterations `num_iter`. must be a positive integer."
        assert type(decay_window)==int and decay_window>0, "invalid value for number of iterations `decay_window`. must be a positive integer."
        assert loss_scale in ["convo","group","total"], "invalid loss scaling type `loss_scale`. must be one of ['convo','group','total']."
        assert test_loss_scale in ["convo","group","total"], "invalid loss scaling type `test_loss_scale`. must be one of ['convo','group','total']."
        assert type(terminate)==bool, "invalid value for termination flag `terminate`. must be a boolean."
        assert (not terminate) or (type(burn_in_iter)==int and burn_in_iter>0), "invalid value for number of burn-in iterations `burn_in_iter`. must be a positive integer."
        assert (not terminate) or (type(patience)==int and patience>0), "invalid value for number of termination patience iterations `patience`. must be a positive integer."
        assert type(burn_in_valloss)==int and burn_in_valloss>=0, "invalid value for minimum validation loss iteration `burn_in_valloss`. must be a nonnegative integer."

        self.seed = seed
        self.cuda = cuda
        self.weight_func = weight_func
        self.hid_dim = hid_dim
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.num_inner_iter = num_inner_iter
        self.num_iter = num_iter
        self.decay_window = decay_window
        self.loss_scale = loss_scale
        self.test_loss_scale = test_loss_scale
        self.terminate = terminate
        self.burn_in_iter = burn_in_iter
        self.patience = patience
        self.burn_in_valloss = burn_in_valloss

        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.cuda) else 'cpu')
        torch.manual_seed(self.seed)
        
        self.out_dim = 1
        self.mlp = None
        self.gat = None
        self.optim_m = None
        self.optim_g = None
        self.sched_m = None
        self.sched_g = None

        self.num_feat = None
        self.pis_pred_train = None
        self.fs_pred_train = None
        self.pis_pred_val = None
        self.fs_pred_val = None
        self.pis_pred_test = None
        self.fs_pred_test = None
        self.loss_train_list = None
        self.loss_val_list = None
        self.best_loss_val = None
        self.best_iter = None
        self.log = None

    def train(self,train_data:Dataset,val_data:Dataset=None,test_data:Dataset=None,verbose:bool=True):
        assert (val_data is None) or (train_data.num_feat==val_data.num_feat), "Inconsistent number of features between train and validation data."
        assert (test_data is None) or (train_data.num_feat==test_data.num_feat), "Inconsistent number of features between train and test data."

        loss_train_list = []
        loss_val_list = None if val_data is None else []
        log = []

        self.num_feat = train_data.num_feat
        self.mlp = MLP(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim).to(self.device)
        self.optim_m = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.sched_m = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_m, T_max=self.num_iter, eta_min=1e-5)

        self.gat = GAT(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim ).to(self.device)
        self.optim_g = torch.optim.SGD(self.gat.parameters(), lr=self.lr)
        self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_g, T_max=self.num_iter, eta_min=1e-5)

        best_mlp = MLP(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim).to(self.device)
        best_mlp.load_state_dict(self.mlp.state_dict())
        best_gat = GAT(in_dim=self.num_feat, out_dim=self.out_dim, hid_dim=self.hid_dim ).to(self.device)
        best_gat.load_state_dict(self.gat.state_dict())
        best_loss_val = float('inf')
        best_iter = 0

        train_data = train_data.to(self.device)
        val_data = None if val_data is None else val_data.to(self.device)
        test_data = None if test_data is None else test_data.to(self.device)

        train_data_pred = train_data.copy()
        val_data_pred = None if val_data is None else val_data.copy()
        test_data_pred = None if test_data is None else test_data.copy()

        # betas_fixed_train = [[(torch.exp(-self.b*train_data.time_since[g][c])).to(self.device) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
        # betas_pred_val = [[(torch.exp(-self.b*val_data.time_since[g][c])).to(self.device) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
        # betas_pred_test = [[(torch.exp(-self.b*test_data.time_since[g][c])).to(self.device) for c in range(test_data.num_convos[g])] for g in range(test_data.num_groups)]
        betas_fixed_train = [[(.2*torch.ones_like(train_data.time_since[g][c])).to(self.device) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
        betas_pred_val = [[(.2*torch.ones_like(val_data.time_since[g][c])).to(self.device) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
        betas_pred_test = [[(.2*torch.ones_like(test_data.time_since[g][c])).to(self.device) for c in range(test_data.num_convos[g])] for g in range(test_data.num_groups)]

        tic = perf_counter()

        tic = perf_counter()
        for train_iter in range(self.num_iter):
            self.mlp.train()
            self.gat.train()

            preds_train = self.mlp(train_data.tensor)
            pis_pred_train = [preds_train[train_data.inds_grps[g]][:,0] for g in range(train_data.num_groups)]
            fs_pred_train = [[self.gat(train_data.psi[g],train_data.time_since[g][c]) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]

            L_pred = [[(pis_pred_train[g][None] + fs_pred_train[g][c]) * (train_data.time_since[g][c]-2>=0) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            l_ratio_pred = [[L_pred[g][c][np.arange(train_data.num_times[g][c]),train_data.convos[g][c]] / L_pred[g][c].sum(1) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            nlls_pred = [[-torch.log(l_ratio_pred[g][c]) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]

            if self.weight_func is None:
                self.weight_func = lambda convo:torch.ones_like(convo,dtype=torch.float)
            weights = [[self.weight_func(train_data.convos[g][c]) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]

            if self.loss_scale=='total':
                nlls_pred = [[weights[g][c] * nlls_pred[g][c] / np.concatenate(train_data.num_times).sum() for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            elif self.loss_scale=='group':
                nlls_pred = [[weights[g][c] * nlls_pred[g][c] / (train_data.num_groups*np.sum(train_data.num_times[g])) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]
            elif self.loss_scale=='convo':
                nlls_pred = [[weights[g][c] * nlls_pred[g][c] / (np.sum(train_data.num_convos)*train_data.num_times[g][c]) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]

            loss_train = torch.sum(torch.cat(sum(nlls_pred,[])))
            loss_train_list.append(loss_train.detach())
            self.optim_m.zero_grad()
            self.optim_g.zero_grad()
            loss_train.backward()
            self.optim_m.step(); self.sched_m.step()
            self.optim_g.step(); self.sched_g.step()

            self.mlp.eval()
            self.gat.eval()

            if val_data is not None:
                preds_val = self.mlp(val_data.tensor)
                pis_pred_val = [preds_val[val_data.inds_grps[g]][:,0] for g in range(val_data.num_groups)]
                fs_pred_val = [[self.gat(val_data.psi[g],val_data.time_since[g][c]) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]

                L_pred = [[(pis_pred_val[g][None] + fs_pred_val[g][c]) * (val_data.time_since[g][c]-2>=0) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
                l_ratio_pred = [[L_pred[g][c][np.arange(val_data.num_times[g][c]),val_data.convos[g][c]] / L_pred[g][c].sum(1) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
                nlls_pred = [[-torch.log(l_ratio_pred[g][c]) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]

                if self.weight_func is None:
                    self.weight_func = lambda convo:torch.ones_like(convo,dtype=torch.float)
                weights = [[self.weight_func(val_data.convos[g][c]) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]

                if self.loss_scale=='total':
                    nlls_pred = [[weights[g][c] * nlls_pred[g][c] / np.concatenate(val_data.num_times).sum() for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
                elif self.loss_scale=='group':
                    nlls_pred = [[weights[g][c] * nlls_pred[g][c] / (val_data.num_groups*np.sum(val_data.num_times[g])) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]
                elif self.loss_scale=='convo':
                    nlls_pred = [[weights[g][c] * nlls_pred[g][c] / (np.sum(val_data.num_convos)*val_data.num_times[g][c]) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]

                loss_val = torch.sum(torch.cat(sum(nlls_pred,[])))
                loss_val_list.append(loss_val.detach())
            
            if train_iter>=self.burn_in_valloss and loss_val_list[-1]<=best_loss_val:
                best_loss_val = loss_val_list[-1]
                best_mlp.load_state_dict(self.mlp.state_dict())
                best_gat.load_state_dict(self.gat.state_dict())
                best_iter = train_iter

            if self.terminate and (val_data is not None):
                if (train_iter>self.burn_in_iter and 
                    (all(torch.tensor(loss_val_list[-self.patience:]) >= float(loss_val_list[-self.patience])) or 
                     torch.std(torch.tensor(loss_val_list[-2*self.patience:]))<=1e-6 ) ):
                    if verbose:
                        msg = f"TERMINATE @ {train_iter+1} iterations"
                        log.append(msg)
                        print(msg)
                    break

            toc = perf_counter()
            if verbose and ((train_iter+1)%100==0 or train_iter==0 or toc-tic>=10):
                msg = (f"ITER. {train_iter+1}/{self.num_iter} | " + 
                    f"Train. loss: {loss_train_list[-1]:.4f}")
                if val_data is not None:
                    msg += f" | Val. loss: {loss_val_list[-1]:.4f}"
                log.append(msg)
                print(msg)
                tic = perf_counter()

        self.mlp.load_state_dict(best_mlp.state_dict())
        self.mlp.eval()
        self.gat.load_state_dict(best_gat.state_dict())
        self.gat.eval()

        preds_train = self.mlp(train_data.tensor)
        pis_pred_train = [preds_train[train_data.inds_grps[g]][:,0].detach().to(self.device) for g in range(train_data.num_groups)]
        fs_pred_train = [[self.gat(train_data.psi[g],train_data.time_since[g][c]).detach().to(self.device) for c in range(train_data.num_convos[g])] for g in range(train_data.num_groups)]

        preds_val = None if val_data is None else self.mlp(val_data.tensor)
        pis_pred_val = None if val_data is None else [preds_val[val_data.inds_grps[g]][:,0].detach().to(self.device) for g in range(val_data.num_groups)]
        fs_pred_val = None if val_data is None else [[self.gat(val_data.psi[g],val_data.time_since[g][c]).detach().to(self.device) for c in range(val_data.num_convos[g])] for g in range(val_data.num_groups)]

        preds_test = None if test_data is None else self.mlp(test_data.tensor)
        pis_pred_test = None if test_data is None else [preds_test[test_data.inds_grps[g]][:,0].detach().to(self.device) for g in range(test_data.num_groups)]
        fs_pred_test = None if test_data is None else [[self.gat(test_data.psi[g],test_data.time_since[g][c]).detach().to(self.device) for c in range(test_data.num_convos[g])] for g in range(test_data.num_groups)]

        if verbose and val_data is not None:
            msg = f"Best validation loss: {best_loss_val:.5f} at iteration {best_iter}"
            log.append(msg)
            print(msg)

        self.pis_pred_train = pis_pred_train
        self.fs_pred_train = fs_pred_train
        self.pis_pred_val = pis_pred_val
        self.fs_pred_val = fs_pred_val
        self.pis_pred_test = pis_pred_test
        self.fs_pred_test = fs_pred_test

        self.loss_train_list = torch.tensor(loss_train_list)
        if val_data_pred is not None:
            self.loss_val_list = torch.tensor(loss_val_list)
        self.best_loss_val = best_loss_val
        self.best_iter = best_iter
        self.log = log

        return
    

# ----------------------------------------------------------------

def plot_predicted_data_1d(models, num_grid:int=30, psi_min:float=.1, psi_max:float=1., agg="mean", scale_d:bool=False):
    clrs = {"pi":bq['red'], "d":bq['blue'], "A":bq['yellow'], "w":bq["green"], 'l':bq['purple']}
    cmaps ={"pi":"viridis", "d":"cividis", "A":"plasma", "w":"inferno", "l":"magma"} 
    surfs = {"pi":'summer', "d":'winter', "A":'autumn', "w":'plasma', 'l':'spring'}
    line_args = { "linestyle":'-', "linewidth":3, }

    data_trials = len(models)
    model_trials = len(models[0])

    if agg=="mean":
        agg_fn = np.mean
    elif agg=="median":
        agg_fn = np.median
    else:
        agg_fn = np.mean

    psi_range = np.linspace(psi_min,psi_max,num_grid)
    psi_input = list(torch.FloatTensor(psi_range.reshape(-1,1)))

    plot_data = Dataset([Group(psi_input)])
    plot_data_preds = [[models[dt][mt].predict(plot_data) for mt in range(model_trials)] for dt in range(data_trials)]

    pi_preds = [[plot_data_preds[dt][mt].pi[0].numpy().astype(float) for mt in range(model_trials)] for dt in range(data_trials)]
    d_preds = [[plot_data_preds[dt][mt].d[0].numpy().astype(float) for mt in range(model_trials)] for dt in range(data_trials)]
    A_preds = [[plot_data_preds[dt][mt].A[0].numpy().astype(float) for mt in range(model_trials)] for dt in range(data_trials)]
    w_preds = [[models[dt][mt].w for mt in range(model_trials)] for dt in range(data_trials)]

    # d_preds = [[d_preds[dt][mt] * models[dt][mt].w(2) for mt in range(model_trials)] for dt in range(data_trials)]
    # w_preds = [[models[dt][mt].w(np.arange(50) + 2) / models[dt][mt].w(2) for mt in range(model_trials)] for dt in range(data_trials)]
    
    # Compute scaling factor
    # psi_mid = .5 * (psi_min + psi_max)
    # ix = torch.tensor([torch.norm(psi-psi_mid) for psi in psi_input]).argmin().item()
    # A_mid = [[A_preds[dt][mt][:,ix] for mt in range(model_trials)] for dt in range(data_trials)]
    # # l_mid = [[pi_preds[dt][mt] + d_preds[dt][mt]*w_preds[dt][mt](2) + A_mid[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # l_mid = [[pi_preds[dt][mt] + d_preds[dt][mt] + A_mid[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # scale = [[l_mid[dt][mt].max().item() for mt in range(model_trials)] for dt in range(data_trials)]
    w_max = [[w_preds[dt][mt](np.arange(50)+2).max().item() for mt in range(model_trials)] for dt in range(data_trials)]
    # scale = [[( pi_preds[dt][mt] + d_preds[dt][mt] * w_preds[dt][mt](2) ).max() for mt in range(model_trials)] for dt in range(data_trials)]
    # scale = [[( pi_preds[dt][mt] + d_preds[dt][mt] * w_max[dt][mt] ).max() for mt in range(model_trials)] for dt in range(data_trials)]
    scale = [[np.mean( pi_preds[dt][mt] + d_preds[dt][mt] * w_max[dt][mt] ) for mt in range(model_trials)] for dt in range(data_trials)]

    pi_preds = [[pi_preds[dt][mt] / scale[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    d_preds = [[d_preds[dt][mt] / scale[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # A_preds = [[A_preds[dt][mt].reshape(num_grid,num_grid) / scale[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # l_preds = [[pi_preds[dt][mt] + d_preds[dt][mt]*w_preds[dt][mt](2) + A_preds[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # l_preds = [[pi_preds[dt][mt] + d_preds[dt][mt] + A_preds[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    l_preds = [[pi_preds[dt][mt] + d_preds[dt][mt] * w_max[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    if scale_d:
        d_preds = [[d_preds[dt][mt] * w_max[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    # Choose mean surface plots across all trials
    pi_range = agg_fn(sum(pi_preds,[]),axis=0)
    d_range = agg_fn(sum(d_preds,[]),axis=0)
    # A_range = agg_fn(sum(A_preds,[]),axis=0).reshape(num_grid,num_grid)
    l_range = agg_fn(sum(l_preds,[]),axis=0)

    fig,axes = plt.subplots(1,2,figsize=(2*fw,fl)); _ = [a.grid(1) for a in axes]
    ax = axes[0]
    for dt in range(data_trials):
        for mt in range(model_trials):
            ax.plot(psi_range,pi_preds[dt][mt], c=clrs['pi'], linestyle='-', linewidth=1, alpha=.4)
    ax.plot(psi_range, pi_range, c=clrs['pi'], **line_args)
    ax.set_xlabel(r"$\psi$"); ax.set_ylabel(r"$\pi$"); ax.set_title(r"Predicted $\pi$")
    ax = axes[1]
    for dt in range(data_trials):
        for mt in range(model_trials):
            ax.plot(psi_range,d_preds[dt][mt], c=clrs['d'], linestyle='-', linewidth=1, alpha=.4)
    ax.plot(psi_range, d_range, c=clrs['d'], **line_args)
    ax.set_xlabel(r"$\psi$"); ax.set_ylabel(r"$d$"); ax.set_title(r"Predicted $d$")
    fig.tight_layout()
    
    fig,ax = plt.subplots(figsize=(fw,fl)); ax.grid(1)
    ax.plot(psi_range,l_range, c=clrs['l'], **line_args)
    for dt in range(data_trials):
        for mt in range(model_trials):
            ax.plot(psi_range,l_preds[dt][mt], c=clrs['l'], linestyle='-', linewidth=1, alpha=.4)
    ax.set_xlabel(r"$\psi$"); ax.set_ylabel(r"$\ell$"); ax.set_title(r"Predicted $\ell$")
    fig.tight_layout()

    # fig,axes = plt.subplots(1,2,figsize=(2.2*fw,fl)); ims = [None,None]
    # ax = axes[0]
    # ims[0] = ax.imshow(A_range.T, cmap=cmaps['A'], origin='lower')
    # ax.set_xticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    # ax.set_yticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1))
    # ax.set_xlabel(r"$\psi$ (current)"); ax.set_ylabel(r"$\psi$ (previous)"); ax.set_title(r"Predicted $A$")
    # ax = axes[1]
    # ims[1] = ax.imshow(l_range.T, cmap=cmaps['l'], origin='lower')
    # ax.set_xticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    # ax.set_yticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1))
    # ax.set_xlabel(r"$\psi$ (current)"); ax.set_ylabel(r"$\psi$ (previous)"); ax.set_title(r"Predicted $\ell$")
    # bbox = [ax.get_position() for ax in axes]
    # cax = [fig.add_axes([bbox[j].x1+.01, bbox[j].y0, .02, bbox[j].height]) for j in range(len(axes))]
    # _ = [fig.colorbar(ims[j], cax=cax[j]) for j in range(len(axes))]

    w_pred_list = [[w_preds[dt][mt](np.arange(50)+2) / w_max[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    w_pred_range = agg_fn(sum(w_pred_list,[]), axis=0)
    # w_pred_range = agg_fn(sum(w_preds,[]), axis=0)
    fig,ax = plt.subplots(figsize=(fw,fl)); ax.grid(1)
    for dt in range(data_trials):
        for mt in range(model_trials):
            ax.plot(np.arange(50), w_pred_list[dt][mt], '-', c=clrs['w'], linewidth=1, alpha=.6)
            # ax.plot(np.arange(50), w_preds[dt][mt], '-', c=clrs['w'], linewidth=1, alpha=.6)
    ax.plot(np.arange(50), w_pred_range, '-', c=clrs['w'], linewidth=3, alpha=1)
    ax.set_xlabel(r"Number of turns $t$"); ax.set_ylabel(r"Memory function $w(t)$")
    fig.tight_layout()


def plot_predicted_data_2d(models, 
        num_grid:int=30, psi_min:float=.1, psi_max:float=1.,
        surf_plot_type:str="fix_prev", scale_d:bool=False
    ):
    clrs = {"pi":bq['red'], "d":bq['blue'], "A":bq['yellow'], "w":bq["green"], 'l':bq['purple']}
    cmaps ={"pi":"viridis", "d":"cividis", "A":"plasma", "w":"inferno", "l":"magma"} 
    surfs = {"pi":'summer', "d":'winter', "A":'autumn', "w":'plasma', 'l':'spring'}
    line_args = { "linestyle":'-', "linewidth":3, }

    assert surf_plot_type in ['fix_prev','fix_psi1','fix_psi2'], "Invalid option for surface plots. Expecting surf_plot_type to be 'fix_prev', 'fix_psi1', or 'fix_psi2'."
    data_trials = len(models)
    model_trials = len(models[0])

    if agg=="mean":
        agg_fn = np.mean
    elif agg=="median":
        agg_fn = np.median
    else:
        agg_fn = np.mean

    psi_range = np.linspace(psi_min,psi_max,num_grid)
    psi1_range, psi2_range = np.meshgrid(psi_range,psi_range)
    psi_input = list(torch.FloatTensor(np.array([psi1_range.reshape(-1), psi2_range.reshape(-1)])).T)

    plot_data = Dataset([Group(psi_input)])
    plot_data_preds = [[models[dt][mt].predict(plot_data) for mt in range(model_trials)] for dt in range(data_trials)]

    pi_preds = [[plot_data_preds[dt][mt].pi[0].numpy().astype(float) for mt in range(model_trials)] for dt in range(data_trials)]
    d_preds = [[plot_data_preds[dt][mt].d[0].numpy().astype(float) for mt in range(model_trials)] for dt in range(data_trials)]
    A_preds = [[plot_data_preds[dt][mt].A[0].numpy().astype(float) for mt in range(model_trials)] for dt in range(data_trials)]

    d_preds = [[d_preds[dt][mt] * models[dt][mt].w(2) for mt in range(model_trials)] for dt in range(data_trials)]
    w_preds = [[models[dt][mt].w for mt in range(model_trials)] for dt in range(data_trials)]
    # w_preds = [[models[dt][mt].w(np.arange(50) + 2) / models[dt][mt].w(2) for mt in range(model_trials)] for dt in range(data_trials)]

    # Compute scaling factor
    # psi_mid = np.array([.5 * (psi_min + psi_max)]*2)
    # ix = torch.tensor([torch.norm(psi-psi_mid) for psi in psi_input]).argmin().item()
    # A_mid = [[A_preds[dt][mt][:,ix] for mt in range(model_trials)] for dt in range(data_trials)]
    # # l_mid = [[pi_preds[dt][mt] + d_preds[dt][mt]*w_preds[dt][mt](2) + A_mid[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # l_mid = [[pi_preds[dt][mt] + d_preds[dt][mt] + A_mid[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    w_max = [[w_preds[dt][mt](np.arange(50)+2).max().item() for mt in range(model_trials)] for dt in range(data_trials)]
    # scale = [[l_mid[dt][mt].max().item() for mt in range(model_trials)] for dt in range(data_trials)]
    # scale = [[l_mid[dt][mt].max().item() for mt in range(model_trials)] for dt in range(data_trials)]
    scale = [[np.mean( pi_preds[dt][mt] + d_preds[dt][mt] * w_max[dt][mt] ) for mt in range(model_trials)] for dt in range(data_trials)]

    pi_preds = [[pi_preds[dt][mt]/scale[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    d_preds = [[d_preds[dt][mt]/scale[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    l_preds = [[pi_preds[dt][mt] + d_preds[dt][mt] * w_max[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # A_preds = [[A_preds[dt][mt]/scale[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    # l_preds = [[pi_preds[dt][mt] + d_preds[dt][mt]*w_preds[dt][mt](2) + A_preds[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    if scale_d:
        d_preds = [[d_preds[dt][mt] * w_max[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    # Choose mean surface plots across all trials
    pi_range = agg_fn(sum(pi_preds,[]),axis=0).reshape(num_grid,num_grid)
    d_range = agg_fn(sum(d_preds,[]),axis=0).reshape(num_grid,num_grid)
    l_range = agg_fn(sum(l_preds,[]),axis=0).reshape(num_grid,num_grid)

    fig = plt.figure(figsize=(2*fl,fl)); axes = [fig.add_subplot(121,projection='3d'), fig.add_subplot(122,projection='3d')]
    ax = axes[0]
    ax.plot_surface(psi1_range, psi2_range, pi_range, cmap=cmaps['pi'])
    ax.set_xlabel(r"$\psi_1$"); ax.set_ylabel(r"$\psi_2$"); ax.set_zlabel(r"$\pi$"); ax.set_title(r"Predicted $\pi$")
    ax = axes[1]
    ax.plot_surface(psi1_range, psi2_range, d_range, cmap=cmaps['d'])
    ax.set_xlabel(r"$\psi_1$"); ax.set_ylabel(r"$\psi_2$"); ax.set_zlabel(r"$d$"); ax.set_title(r"Predicted $d$")
    fig.tight_layout()
    
    fig = plt.figure(figsize=(fw,fl)); ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(psi1_range, psi2_range, l_range, cmap=cmaps['l'])
    ax.set_xlabel(r"$\psi_1$"); ax.set_ylabel(r"$\psi_2$"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"Predicted $\ell$")
    fig.tight_layout()

    w_pred_list = [[w_preds[dt][mt](np.arange(50)+2) / w_max[dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]
    w_pred_range = agg_fn(sum(w_pred_list,[]), axis=0)
    # w_pred_range = agg_fn(sum(w_preds,[]), axis=0)
    fig,ax = plt.subplots(figsize=(fw,fl)); ax.grid(1)
    for dt in range(data_trials):
        for mt in range(model_trials):
            ax.plot(np.arange(50), w_pred_list[dt][mt], '-', c=clrs['w'], linewidth=1, alpha=.6)
            # ax.plot(np.arange(50), w_preds[dt][mt], '-', c=clrs['w'], linewidth=1, alpha=.6)
    ax.plot(np.arange(50), w_pred_range, '-', c=clrs['w'], linewidth=3, alpha=1)
    ax.set_xlabel(r"Number of turns $t$"); ax.set_ylabel(r"Memory function $w(t)$")
    fig.tight_layout()

    # w_preds = [[models[dt][mt].w for mt in range(model_trials)] for dt in range(data_trials)]
    # if surf_plot_type=='fix_psi2':
    #     As_pred = [[None, None], [None, None]]
    #     ls_pred = [[None, None], [None, None]]
    #     psi_range_min = torch.FloatTensor(np.stack([psi_range, psi_min * np.ones(num_grid)]).T)
    #     psi_range_max = torch.FloatTensor(np.stack([psi_range, psi_max * np.ones(num_grid)]).T)
    #     As_pred[0][0] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_min] for psi1 in psi_range_min]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[0][1] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_max] for psi1 in psi_range_min]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[1][0] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_min] for psi1 in psi_range_max]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[1][1] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_max] for psi1 in psi_range_max]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     for i in range(2):
    #         for j in range(2):
    #             ls_pred[i][j] = [[pi_preds[dt][mt] + d_preds[dt][mt] * w_preds[dt][mt](2) + As_pred[i][j][dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    #     As_range = [[None, None], [None, None]]
    #     ls_range = [[None, None], [None, None]]
    #     for i in range(2):
    #         for j in range(2):
    #             As_range[i][j] = np.mean(sum(As_pred[i][j],[]),axis=0).reshape(num_grid,num_grid)
    #             ls_range[i][j] = np.mean(sum(ls_pred[i][j],[]),axis=0).reshape(num_grid,num_grid)

    #     Amin = float(np.min([As_range[i][j].min() for i in range(2) for j in range(2)]))
    #     Amax = float(np.max([As_range[i][j].max() for i in range(2) for j in range(2)]))
    #     A_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
    #     fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ims[i][j] = ax.imshow(As_range[i][j], cmap=cmaps['A'], origin='lower', vmin=Amin, vmax=Amax)
    #             ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    #             ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
    #             ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_title(r"$A$ for $\psi_1$ given "+rf"$\psi_2 = ({A_fixed[i][j][0]},{A_fixed[i][j][1]})$")
    #     fig.tight_layout(); fig.subplots_adjust(wspace=.5)
    #     bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
    #     cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
    #     _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

    #     lmin = float(np.min([ls_range[i][j].min() for i in range(2) for j in range(2)]))
    #     lmax = float(np.max([ls_range[i][j].max() for i in range(2) for j in range(2)]))
    #     l_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
    #     fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ims[i][j] = ax.imshow(ls_range[i][j], cmap=cmaps['l'], origin='lower', vmin=lmin, vmax=lmax)
    #             ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    #             ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
    #             ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_title(r"$\ell$ for $\psi_1$ given "+rf"$\psi_2 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout(); fig.subplots_adjust(wspace=.5)
    #     bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
    #     cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
    #     _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

    #     fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ax.plot_surface(psi1_range,psi2_range,As_range[i][j], cmap=cmaps['A'])
    #             ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_zlabel(r"$A$"); ax.set_title(r"$A$ for $\psi_1$ given "+rf"$\psi_2 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout()

    #     fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ax.plot_surface(psi1_range,psi2_range,ls_range[i][j], cmap=cmaps['l'])
    #             ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"$\ell$ for $\psi_1$ given "+rf"$\psi_2 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout()

    # elif surf_plot_type=='fix_psi1':
    #     As_pred = [[None, None], [None, None]]
    #     ls_pred = [[None, None], [None, None]]
    #     psi_range_min = torch.FloatTensor(np.stack([psi_min * np.ones(num_grid), psi_range]).T)
    #     psi_range_max = torch.FloatTensor(np.stack([psi_max * np.ones(num_grid), psi_range]).T)
    #     As_pred[0][0] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_min] for psi1 in psi_range_min]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[0][1] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_max] for psi1 in psi_range_min]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[1][0] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_min] for psi1 in psi_range_max]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[1][1] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psi_range_max] for psi1 in psi_range_max]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     for i in range(2):
    #         for j in range(2):
    #             ls_pred[i][j] = [[pi_preds[dt][mt] + d_preds[dt][mt] * w_preds[dt][mt](2) + As_pred[i][j][dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    #     As_range = [[None, None], [None, None]]
    #     ls_range = [[None, None], [None, None]]
    #     for i in range(2):
    #         for j in range(2):
    #             As_range[i][j] = np.mean(sum(As_pred[i][j],[]),axis=0).reshape(num_grid,num_grid)
    #             ls_range[i][j] = np.mean(sum(ls_pred[i][j],[]),axis=0).reshape(num_grid,num_grid)

    #     Amin = float(np.min([As_range[i][j].min() for i in range(2) for j in range(2)]))
    #     Amax = float(np.max([As_range[i][j].max() for i in range(2) for j in range(2)]))
    #     A_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
    #     fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ims[i][j] = ax.imshow(As_range[i][j], cmap=cmaps['A'], origin='lower', vmin=Amin, vmax=Amax)
    #             ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    #             ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({A_fixed[i][j][0]},{A_fixed[i][j][1]})$")
    #     fig.tight_layout(); fig.subplots_adjust(wspace=.5)
    #     bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
    #     cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
    #     _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

    #     lmin = float(np.min([ls_range[i][j].min() for i in range(2) for j in range(2)]))
    #     lmax = float(np.max([ls_range[i][j].max() for i in range(2) for j in range(2)]))
    #     l_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
    #     fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ims[i][j] = ax.imshow(ls_range[i][j], cmap=cmaps['l'], origin='lower', vmin=lmin, vmax=lmax)
    #             ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    #             ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout(); fig.subplots_adjust(wspace=.5)
    #     bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
    #     cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
    #     _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

    #     fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ax.plot_surface(psi1_range,psi2_range,As_range[i][j], cmap=cmaps['A'])
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$A$"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout()

    #     fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ax.plot_surface(psi1_range,psi2_range,ls_range[i][j], cmap=cmaps['l'])
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout()

    # elif surf_plot_type=='fix_prev':
    #     As_pred = [[None, None], [None, None]]
    #     ls_pred = [[None, None], [None, None]]
    #     psis_range = [[[torch.FloatTensor([psi_min,psi_min])], [torch.FloatTensor([psi_min,psi_max])]], [[torch.FloatTensor([psi_max,psi_min])], [torch.FloatTensor([psi_max,psi_max])]]]
    #     As_pred[0][0] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psis_range[0][0]] for psi1 in psi_input]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[0][1] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psis_range[0][1]] for psi1 in psi_input]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[1][0] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psis_range[1][0]] for psi1 in psi_input]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     As_pred[1][1] = [[(1/scale[dt][mt]) * np.array([[models[dt][mt].a(psi1,psi2) for psi2 in psis_range[1][1]] for psi1 in psi_input]).reshape(-1) for mt in range(model_trials)] for dt in range(data_trials)]
    #     for i in range(2):
    #         for j in range(2):
    #             ls_pred[i][j] = [[pi_preds[dt][mt] + d_preds[dt][mt] * w_preds[dt][mt](2) + As_pred[i][j][dt][mt] for mt in range(model_trials)] for dt in range(data_trials)]

    #     As_range = [[None, None], [None, None]]
    #     ls_range = [[None, None], [None, None]]
    #     for i in range(2):
    #         for j in range(2):
    #             As_range[i][j] = np.mean(sum(As_pred[i][j],[]),axis=0).reshape(num_grid,num_grid)
    #             ls_range[i][j] = np.mean(sum(ls_pred[i][j],[]),axis=0).reshape(num_grid,num_grid)

    #     Amin = float(np.min([As_range[i][j].min() for i in range(2) for j in range(2)]))
    #     Amax = float(np.max([As_range[i][j].max() for i in range(2) for j in range(2)]))
    #     A_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
    #     fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ims[i][j] = ax.imshow(As_range[i][j], cmap=cmaps['A'], origin='lower', vmin=Amin, vmax=Amax)
    #             ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    #             ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({A_fixed[i][j][0]},{A_fixed[i][j][1]})$")
    #     fig.tight_layout(); fig.subplots_adjust(wspace=.5)
    #     bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
    #     cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
    #     _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

    #     lmin = float(np.min([ls_range[i][j].min() for i in range(2) for j in range(2)]))
    #     lmax = float(np.max([ls_range[i][j].max() for i in range(2) for j in range(2)]))
    #     l_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
    #     fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ims[i][j] = ax.imshow(ls_range[i][j], cmap=cmaps['l'], origin='lower', vmin=lmin, vmax=lmax)
    #             ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    #             ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout(); fig.subplots_adjust(wspace=.5)
    #     bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
    #     cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
    #     _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

    #     fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ax.plot_surface(psi1_range,psi2_range,As_range[i][j], cmap=cmaps['A'])
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$A$"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout()

    #     fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
    #     for i in range(2):
    #         for j in range(2):
    #             ax = axes[i][j]
    #             ax.plot_surface(psi1_range,psi2_range,ls_range[i][j], cmap=cmaps['l'])
    #             ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
    #     fig.tight_layout()

