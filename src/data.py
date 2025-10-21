from utils import *
import os
import pandas as pd
from copy import deepcopy
import commentjson as json
from plttools import *
from typing import List

COMPLEXITY_SETTINGS = [
    'simple_incr',
    'simple_decr',
    'complex_incr',
    'complex_decr',
    'simple_uncorrel',
    'simple_poscorrel',
    'simple_negcorrel',
    'complex_uncorrel',
    'complex_poscorrel',
    'complex_negcorrel'
]
AFFINITY_SETTINGS = ["simple_posdot", "simple_negdot", "simple_posaff", "simple_negaff", "simple_linposaff", "simple_linnegaff", "simple_posthresh", "simple_negthresh"]

# Generate synthetic personality traits psi for one speaker
def simulate_psi(num_feat:int=1):
    psi = torch.rand(num_feat)*.9 + .1
    # if num_feat==1:
    #     psi = psi.item()
    return psi

# Generate inherent speaking probability pi for one speaker from traits
def simulate_pi(psi, COMPLEXITY:str=None):
    assert COMPLEXITY in COMPLEXITY_SETTINGS, "Invalid choice for trait complexity."

    f_simple = lambda x:x
    f_complex = lambda x:torch.sqrt(x)

    num_feat = 1 if np.isscalar(psi) else len(psi)
    psi = torch.tensor(psi) if np.isscalar(psi) else psi
    if COMPLEXITY=='simple_incr':
        assert num_feat==1, "Complexity `simple_incr` for 1 feature."
        pi = float(f_simple(psi))
    elif COMPLEXITY=='simple_decr':
        assert num_feat==1, "Complexity `simple_decr` for 1 feature."
        pi = float(f_simple(1.1-psi))
    elif COMPLEXITY=='complex_incr':
        assert num_feat==1, "Complexity `complex_incr` for 1 feature."
        pi = float(f_complex(psi))
    elif COMPLEXITY=='complex_decr':
        assert num_feat==1, "Complexity `complex_decr` for 1 feature."
        pi = float(f_complex(1.1-psi))

    elif COMPLEXITY=='simple_uncorrel':
        assert num_feat==2, "Complexity `simple_uncorrel` for 2 features."
        pi = f_simple(psi[0]).item()
    elif COMPLEXITY=='simple_poscorrel':
        assert num_feat==2, "Complexity `simple_poscorrel` for 2 features."
        pi = f_simple(torch.mean(psi)).item()
    elif COMPLEXITY=='simple_negcorrel':
        assert num_feat==2, "Complexity `simple_negcorrel` for 2 features."
        pi = f_simple(torch.mean(psi)).item()
    elif COMPLEXITY=='complex_uncorrel':
        assert num_feat==2, "Complexity `complex_uncorrel` for 2 features."
        pi = f_complex(psi[0]).item()
    elif COMPLEXITY=='complex_poscorrel':
        assert num_feat==2, "Complexity `complex_poscorrel` for 2 features."
        pi = f_complex(torch.mean(psi)).item()
    elif COMPLEXITY=='complex_negcorrel':
        assert num_feat==2, "Complexity `complex_negcorrel` for 2 features."
        pi = f_complex(torch.mean(psi)).item()
    
    return pi

# Generate turn-based speaking probability d for one speaker from traits
def simulate_d(psi, COMPLEXITY:str=None):
    assert COMPLEXITY in COMPLEXITY_SETTINGS, "Invalid choice for trait complexity."

    g_simple = lambda x: (5*x+1) / 3
    # g_complex = lambda x: (15/2) * (torch.exp(-2*x) / ( np.exp(-.2)-np.exp(-2) ) - np.exp(-2) + (1/3))
    g_complex = lambda x: (3/2) * ( (torch.exp(-2*(1.1-x)) - np.exp(-2)) / ( np.exp(-.2)-np.exp(-2) ) + (1/3))

    num_feat = 1 if np.isscalar(psi) else len(psi)
    psi = torch.tensor(psi) if np.isscalar(psi) else psi
    if COMPLEXITY=='simple_incr':
        assert num_feat==1, "Complexity `simple_incr` for 1 feature."
        d = float(g_simple(psi))
    elif COMPLEXITY=='simple_decr':
        assert num_feat==1, "Complexity `simple_decr` for 1 feature."
        d = float(g_simple(1.1-psi))
    elif COMPLEXITY=='complex_incr':
        assert num_feat==1, "Complexity `complex_incr` for 1 feature."
        d = float(g_complex(psi))
    elif COMPLEXITY=='complex_decr':
        assert num_feat==1, "Complexity `complex_decr` for 1 feature."
        d = float(g_complex(1.1-psi))

    elif COMPLEXITY=='simple_uncorrel':
        assert num_feat==2, "Complexity `simple_uncorrel` for 2 features."
        d = g_simple(psi[1]).item()
    elif COMPLEXITY=='simple_poscorrel':
        assert num_feat==2, "Complexity `simple_poscorrel` for 2 features."
        d = g_simple(psi.mean()).item()
    elif COMPLEXITY=='simple_negcorrel':
        assert num_feat==2, "Complexity `simple_negcorrel` for 2 features."
        d = g_simple(1.1 - psi.mean()).item()
    elif COMPLEXITY=='complex_uncorrel':
        assert num_feat==2, "Complexity `complex_uncorrel` for 2 features."
        d = g_complex(psi[1]).item()
    elif COMPLEXITY=='complex_poscorrel':
        assert num_feat==2, "Complexity `complex_poscorrel` for 2 features."
        d = g_complex(psi.mean()).item()
    elif COMPLEXITY=='complex_negcorrel':
        assert num_feat==2, "Complexity `complex_negcorrel` for 2 features."
        d = g_complex(1.1 - psi.mean()).item()
    
    return d * np.exp(1.) * 5.

def simulate_a(psi1, psi2, AFFINITY_TYPE:str=None):
    assert AFFINITY_TYPE in AFFINITY_SETTINGS, "Invalid choice for trait complexity."
    assert (np.isscalar(psi1) and np.isscalar(psi2)) or ((not np.isscalar(psi1) and not np.isscalar(psi2)) and len(psi1)==len(psi2)), "Invalid input. Expecting psi1 and psi2 to be the same size."

    num_feat = 1 if np.isscalar(psi1) else len(psi1)

    if AFFINITY_TYPE == "simple_posdot":
        a = float(psi1*psi2) if num_feat==1 else float(psi1@psi2)/len(psi1)
    elif AFFINITY_TYPE == "simple_negdot":
        a = float(psi1*(1.1-psi2)) if num_feat==1 else float(psi1@(1.1-psi2))/len(psi1)

    elif AFFINITY_TYPE == "simple_linnegaff":
        a = float(np.abs(psi1-.5*psi2)/.9) if num_feat==1 else float(torch.abs(psi1-.5*psi2).sum())/.9/num_feat
    elif AFFINITY_TYPE == "simple_linposaff":
        a = float(np.abs(psi1-.5*(1.1-psi2)))/.9 if num_feat==1 else float(torch.abs(psi1-.5*(1.1-psi2)).sum())/.9/num_feat

    elif AFFINITY_TYPE == "simple_negaff":
        a = float(np.abs(psi1-psi2)/.9) if num_feat==1 else float(torch.abs(psi1-psi2).sum())/.9/num_feat
    elif AFFINITY_TYPE == "simple_posaff":
        a = float(np.abs(psi1+psi2-1.1))/.9 if num_feat==1 else float(torch.abs(psi1+psi2-1.1).sum())/.9/num_feat

    elif AFFINITY_TYPE == "simple_posthresh":
        diff = float(np.abs(psi1-psi2)) if num_feat==1 else float(torch.abs(psi1-psi2).sum())/num_feat
        a = 1. if diff<=.2 else .05
    elif AFFINITY_TYPE == "simple_negthresh":
        diff = float(np.abs(psi1+psi2-1.1)) if num_feat==1 else float(torch.abs(psi1+psi2-1.1).sum())/num_feat
        a = 1. if diff<=.2 else .05
    
    return a * 10

# Simulate conversation from a group of speaker probabilities
def simulate_conversation(pis,ds=None,A=None,num_time:int=100,w=None):
    assert all([pis[s]>=0 for s in range(len(pis))]), "Invalid pis. All values must be nonnegative."
    assert (ds is None) or (len(pis)==len(ds)), "Inconsistent number of speakers for pis and ds."
    assert (ds is None) or all([ds[s]>=0 for s in range(len(ds))]), "Invalid ds. All values must be nonnegative."
    assert (A is None) or (A.ndim==2 and A.shape[0]==A.shape[1] and A.shape[0]==len(pis)), "Inconsistent number of speakers for A."
    assert (A is None) or all([A[s1,s2]>=0 for s1 in range(A.shape[0]) for s2 in range(A.shape[1])]), "Invalid A. All values must be nonnegative."
    # assert (ds is None) or np.isscalar(b), "Invalid b. Must be a scalar value."
    if w is None:
        w = lambda x: torch.exp(-.5*x)

    num_speakers = len(pis)
    last_id = 0
    speakers_id = -torch.ones(num_time,dtype=int)
    t_last = -torch.inf * torch.ones(num_speakers,dtype=int)
    delta_t = -t_last
    if ds is None:
        ds = torch.zeros(num_speakers)
    if A is None:
        A = torch.zeros((num_speakers,num_speakers))

    for t in range(num_time):
        delta_t = t - t_last - 2
        # l = (pis + ds * w(delta_t) + A[:,last_id])*(t>0)) * (delta_t >= 0)
        l = (pis + ds * w(delta_t) + A[:,last_id]*(1-torch.eye(num_speakers)[last_id])*(t>0)) * (delta_t >= 0)
        l /= torch.sum(l)
        speakers_id[t] = torch.multinomial(l,1).item()
        last_id = int(speakers_id[t])
        assert t==0 or speakers_id[t]!=speakers_id[t-1], "Invalid convo. Same speaker selected twice in a row."

        for s in range(num_speakers):
            spkr_turns = np.where(speakers_id==s)[0]
            if len(spkr_turns>0):
                t_last[s] = float(np.max(spkr_turns))
        # last_spkrturn = torch.where(torch.arange(num_speakers).reshape(1,-1)==speakers_id.reshape(-1,1))
        # if len(last_spkrturn[0])>0:
        #     t_last[last_spkrturn[1]] = last_spkrturn[0].to(torch.float)

    return speakers_id

# Simulate conversation from a group of speaker probabilities
def simulate_approx_conversation(pis,ds=None,A=None,num_time:int=100,w=None,tau:float=.1):
    assert all([pis[s]>=0 for s in range(len(pis))]), "Invalid pis. All values must be nonnegative."
    assert (ds is None) or (len(pis)==len(ds)), "Inconsistent number of speakers for pis and ds."
    assert (ds is None) or all([ds[s]>=0 for s in range(len(ds))]), "Invalid ds. All values must be nonnegative."
    assert (A is None) or (A.ndim==2 and A.shape[0]==A.shape[1] and A.shape[0]==len(pis)), "Inconsistent number of speakers for A."
    assert (A is None) or all([A[s1,s2]>=0 for s1 in range(A.shape[0]) for s2 in range(A.shape[1])]), "Invalid A. All values must be nonnegative."
    assert np.isscalar(tau), "Invalid tau. Must be a scalar value."
    if w is None:
        w = lambda x: torch.exp(-.5*x)

    num_speakers = len(pis)
    last_id = 0
    speakers_id = -torch.ones(num_time,dtype=int)
    speakers_onehot_approx = -torch.ones((num_time,num_speakers))
    t_last = -torch.inf * torch.ones(num_speakers,dtype=int)
    delta_t = -t_last
    if ds is None:
        ds = torch.zeros(num_speakers)
    if A is None:
        A = torch.zeros((num_speakers,num_speakers))

    for t in range(num_time):
        delta_t = t - t_last - 2
        l = (pis + ds * w(delta_t) + A[:,last_id]*(t>0)) * (delta_t >= 0)
        # l = (pis + ds * w(delta_t)) * (delta_t >= 0)
        # l = (pis + ds * torch.exp(-b*delta_t)) * (delta_t >= 0)
        l /= torch.sum(l)
        # speakers_id[t] = torch.multinomial(l,1).item()
        g = torch.log(l) + torch.FloatTensor(np.random.gumbel(0,1,num_speakers))
        speakers_onehot_approx[t] = torch.softmax(g/tau,dim=0)
        speakers_id[t] = speakers_onehot_approx[t].argmax()

        for s in range(num_speakers):
            spkr_turns = np.where(speakers_id==s)[0]
            if len(spkr_turns>0):
                t_last[s] = float(np.max(spkr_turns))
        # last_spkrturn = torch.where(torch.arange(num_speakers).reshape(1,-1)==speakers_id.reshape(-1,1))
        # if len(last_spkrturn[0])>0:
        #     t_last[last_spkrturn[1]] = last_spkrturn[0].to(torch.float)

    return speakers_onehot_approx

# Generate psis for a group of speakers
def generate_psis(num_feat:int, num_speakers:int):
    assert type(num_speakers)==int and num_speakers>0, "Invalid number of speakers. Must be a positive integer."
    assert type(num_feat)==int and num_feat>0, "Invalid number of features. Must be a positive integer."

    psi = [simulate_psi(num_feat) for s in range(num_speakers)]
    return psi

# Generates pis for a group of speakers given psis
def generate_pis(psis, COMPLEXITY:str=None):
    assert COMPLEXITY in COMPLEXITY_SETTINGS, "Invalid choice for trait complexity."

    pi = torch.FloatTensor([simulate_pi(psi, COMPLEXITY) for psi in psis])
    return pi

# Generates ds for a group of speakers given psis
def generate_ds(psis, COMPLEXITY:str=None):
    assert COMPLEXITY in COMPLEXITY_SETTINGS, "Invalid choice for trait complexity."

    d = torch.FloatTensor([simulate_d(psi, COMPLEXITY) for psi in psis])
    return d

def generate_A(psis, AFFINITY_TYPE:str=None):
    assert AFFINITY_TYPE in AFFINITY_SETTINGS, "Invalid choice for trait complexity."

    # A = torch.FloatTensor([[simulate_s(psis[i],psis[j],AFFINITY_TYPE) if i!=j else 0. for j in range(len(psis))] for i in range(len(psis))])
    A = torch.FloatTensor([[simulate_a(psis[i],psis[j],AFFINITY_TYPE) for j in range(len(psis))] for i in range(len(psis))])
    return A

# Simulate multiple conversations for a group of speakers given pis and ds
def generate_convos(pis,ds=None,A=None,num_times=[100],w=None):
    assert (ds is None) or len(pis)==len(ds), "Inconsistent number of speakers for pis and ds."
    assert (A is None) or (A.ndim==2 and A.shape[0]==A.shape[1] and A.shape[0]==len(pis)), "Inconsistent number of speakers for A."
    # assert (ds is None) or np.isscalar(b), "Invalid b. Must be a scalar value."
    assert (not np.isscalar(num_times)) and all([num_time>=0 for num_time in num_times]), "Invalid number of turns. Expecting a list or array of number of turns per conversation."
    num_convos = len(num_times)

    convos = [simulate_conversation(pis,ds,A,num_times[c],w=w) for c in range(num_convos)]
    return convos

def load_real_data(trait_paths:List[str], convo_path:str, NORMALIZE:bool=True):
    for trait_path in trait_paths:
        assert os.path.exists(trait_path), f"Trait file `{trait_path}` does not exist."
    assert os.path.exists(convo_path), f"Conversation file `{convo_path}` does not exist."

    # ------------------------------------
    psi_pds = [pd.read_csv(trait_path,sep="\t") for trait_path in trait_paths]
    psi_max = [psi_pd["Psi"].max() for psi_pd in psi_pds]
    psi_min = [psi_pd["Psi"].min() for psi_pd in psi_pds]
    if NORMALIZE:
        for i in range(len(psi_pds)):
            psi_pds[i]["Psi"] = ((psi_pds[i]["Psi"] - psi_min[i]) / ((psi_max[i] - psi_min[i]) + int(psi_max[i]==psi_min[i]) ))
    num_traits = len(psi_pds)

    convos_pd = pd.read_csv(convo_path,sep="\t")
    # ------------------------------------


    # ------------------------------------
    # Groups present among all traits
    keep_groups = psi_pds[0]['Group'].unique()
    for i in range(1,num_traits):
        keep_groups = np.intersect1d(keep_groups, psi_pds[i]['Group'].unique())
    # ------------------------------------


    # ------------------------------------
    # Remove groups with NAN psi values
    keep_groups = list(keep_groups)
    for i in range(num_traits):
        psi_pd = psi_pds[i]
        nan_psi_loc = np.where(np.isnan(psi_pd['Psi'].to_numpy()))[0]
        for i_nan in nan_psi_loc:
            del_grp = psi_pd['Group'].iloc[i_nan]
            if del_grp in keep_groups:
                keep_groups.remove(del_grp)
    keep_groups = np.array(keep_groups)
    # ------------------------------------


    # ------------------------------------
    # Remove groups with only 2 speakers
    keep_groups = list(keep_groups)
    for i_group, grp in enumerate(convos_pd['Group'].unique()):
        if convos_pd[convos_pd['Group']==grp]['Speaker_ID'].nunique()<=2:
            keep_groups.remove(grp)
    keep_groups = np.array(keep_groups)
    # ------------------------------------


    # ------------------------------------
    # Remove convos with less than 2 turns
    for i_convo,convo in enumerate(convos_pd["Meeting"].unique()):
        if convos_pd[convos_pd["Meeting"]==convo].shape[0]<=2:
            convos_pd = convos_pd.loc[convos_pd["Meeting"]!=convo,:]
    # ------------------------------------


    # ------------------------------------
    # Remove convos with only 2 speakers
    for i_convo,convo in enumerate(convos_pd["Meeting"].unique()):
        if convos_pd[convos_pd['Meeting']==convo]['Speaker_ID'].nunique()<=2:
            convos_pd = convos_pd.loc[convos_pd["Meeting"]!=convo,:]
    # ------------------------------------


    # ------------------------------------
    psi_gbs = [psi_pd.groupby("Group") for psi_pd in psi_pds]
    convos_gb = convos_pd.groupby("Group")
    psi_pds = [pd.concat([psi_gb.get_group(g) for g in keep_groups]) for psi_gb in psi_gbs]
    convos_pd = pd.concat([convos_gb.get_group(g) for g in keep_groups])
    # ------------------------------------


    # ------------------------------------
    count = 0
    for i in range(num_traits):
        for j,feat in enumerate(psi_pds[i]['Feature'].unique()):
                psi_pds[i].iloc[psi_pds[i]['Feature']==feat, psi_pds[i].columns=='Feature'] = count
                count += 1

    psi_pd = pd.concat(psi_pds).sort_values(by=['Data_trial','Group','Speaker'])[['Data_trial','Group','Speaker','Feature','Psi']]
    # ------------------------------------

    # ------------------------------------
    num_feat = psi_pd["Feature"].nunique()
    psi_gb = psi_pd.groupby("Group")
    grpids = list(psi_pd['Group'].unique())
    spkrids = [list(psi_gb.get_group(g)["Speaker"].unique()) for g in grpids]
    cnvoids = [list(convos_gb.get_group(g)["Meeting"].unique()) for g in grpids]
    # ------------------------------------


    # ------------------------------------
    convo_gb = convos_pd.groupby(["Group","Meeting"])
    psi_gb = psi_pd.groupby(["Group","Speaker"])
    psis_list = [[torch.tensor(psi_gb.get_group((grp,s))["Psi"].to_numpy()).to(torch.float) for s in spkrids[g]] for g,grp in enumerate(grpids)]
    convos_list = [[torch.tensor(convo_gb.get_group((grp,c))["Speaker_ID"].to_numpy()) for c in cnvoids[g]] for g,grp in enumerate(grpids)]
    grpid_list = torch.tensor(np.array(grpids))
    # ------------------------------------

    return psis_list, convos_list, grpid_list

# TODO: Consider adding speaker, group, and convo IDs to Group and Dataset objects
def check_valid_convo(convo):
    assert len(np.unique(convo))>2, "Invalid convo. Only two speakers."
    assert not any(convo[1:]==convo[:-1]), "Invalid convo. Same team member speaks twice in a row."
    return

# ----------------------------------------------------------------

def get_dataset_psi_df(datasets:list):
    assert all(list(map(lambda x:type(x)==Dataset,datasets))), "Invalid input. Must be list of Dataset objects."

    data_trials = len(datasets)
    psi_df_list = list(map(lambda i:datasets[i].get_psi_df(),np.arange(len(datasets))))
    num_ids = list(map(len,psi_df_list))
    all_trials = np.concatenate(list(map(lambda dt:dt*np.ones(num_ids[dt],dtype=int),np.arange(data_trials))))
    psi_df = pd.concat(psi_df_list)
    psi_df["Data_trial"] = all_trials
    psi_df = psi_df[["Data_trial","Group","Speaker","Feat","Psi"]]
    return psi_df
def get_dataset_pi_df(datasets:list):
    assert all(list(map(lambda x:type(x)==Dataset,datasets))), "Invalid input. Must be list of Dataset objects."
    if any([dataset.pi[g] is None for dataset in datasets for g in range(dataset.num_groups)]):
        return None

    data_trials = len(datasets)
    pi_df_list = list(map(lambda i:datasets[i].get_pi_df(),np.arange(len(datasets))))
    num_ids = list(map(len,pi_df_list))
    all_trials = np.concatenate(list(map(lambda dt:dt*np.ones(num_ids[dt],dtype=int),np.arange(data_trials))))
    pi_df = pd.concat(pi_df_list)
    pi_df["Data_trial"] = all_trials
    pi_df = pi_df[["Data_trial","Group","Speaker","Pi"]]
    return pi_df
def get_dataset_d_df(datasets:list):
    assert all(list(map(lambda x:type(x)==Dataset,datasets))), "Invalid input. Must be list of Dataset objects."
    if any([dataset.d[g] is None for dataset in datasets for g in range(dataset.num_groups)]):
        return None

    data_trials = len(datasets)
    d_df_list = list(map(lambda i:datasets[i].get_d_df(),np.arange(len(datasets))))
    num_ids = list(map(len,d_df_list))
    all_trials = np.concatenate(list(map(lambda dt:dt*np.ones(num_ids[dt],dtype=int),np.arange(data_trials))))
    d_df = pd.concat(d_df_list)
    d_df["Data_trial"] = all_trials
    d_df = d_df[["Data_trial","Group","Speaker","D"]]
    return d_df
def get_dataset_A_df(datasets:list):
    assert all(list(map(lambda x:type(x)==Dataset,datasets))), "Invalid input. Must be list of Dataset objects."
    if any([dataset.A[g] is None for dataset in datasets for g in range(dataset.num_groups)]):
        return None

    data_trials = len(datasets)
    A_df_list = list(map(lambda i:datasets[i].get_A_df(),np.arange(len(datasets))))
    num_ids = list(map(len,A_df_list))
    all_trials = np.concatenate(list(map(lambda dt:dt*np.ones(num_ids[dt],dtype=int),np.arange(data_trials))))
    A_df = pd.concat(A_df_list)
    A_df["Data_trial"] = all_trials
    A_df = A_df[["Data_trial","Group","Speaker1","Speaker2","A"]]
    return A_df
def get_dataset_convo_df(datasets:list):
    assert all(list(map(lambda x:type(x)==Dataset,datasets))), "Invalid input. Must be list of Dataset objects."
    if any([dataset.convos[g] is None for dataset in datasets for g in range(dataset.num_groups)]):
        return None

    data_trials = len(datasets)
    convo_df_list = list(map(lambda i:datasets[i].get_convo_df(),np.arange(len(datasets))))
    num_ids = list(map(len,convo_df_list))
    all_trials = np.concatenate(list(map(lambda dt:dt*np.ones(num_ids[dt],dtype=int),np.arange(data_trials))))
    convo_df = pd.concat(convo_df_list)
    convo_df["Data_trial"] = all_trials
    convo_df = convo_df[["Data_trial","Group","Convo","Turn","Speaker"]]
    return convo_df

def save_dataset_list(datasets:list, filetag:str="", path:str=""):
    save_dataset_psi(datasets,f"psi{filetag}",path)
    save_dataset_pi(datasets,f"pi{filetag}",path)
    save_dataset_d(datasets,f"d{filetag}",path)
    save_dataset_A(datasets,f"A{filetag}",path)
    save_dataset_convo(datasets,f"convo{filetag}",path)
def save_dataset_psi(datasets:list, filename:str="psi", path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    psi_df = get_dataset_psi_df(datasets)
    if not os.path.exists(path):
        os.makedirs(path)
    psi_df.to_csv(filepath,index=False)
    return
def save_dataset_pi(datasets:list, filename:str="pi", path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    pi_df = get_dataset_pi_df(datasets)
    if pi_df is None:
        return
    if not os.path.exists(path):
        os.makedirs(path)
    pi_df.to_csv(filepath,index=False)
    return
def save_dataset_d(datasets:list, filename:str="d", path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    d_df = get_dataset_d_df(datasets)
    if d_df is None:
        return
    if not os.path.exists(path):
        os.makedirs(path)
    d_df.to_csv(filepath,index=False)
    return
def save_dataset_A(datasets:list, filename:str="A", path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    A_df = get_dataset_A_df(datasets)
    if A_df is None:
        return
    if not os.path.exists(path):
        os.makedirs(path)
    A_df.to_csv(filepath,index=False)
    return
def save_dataset_convo(datasets:list, filename:str="convo", path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    convo_df = get_dataset_convo_df(datasets)
    if convo_df is None:
        return
    if not os.path.exists(path):
        os.makedirs(path)
    convo_df.to_csv(filepath,index=False)
    return


def psi_from_DataFrame(psi_df:pd.DataFrame):
    if set(psi_df.columns) == set(["Speaker","Psi","Feat"]):
        spkrs = psi_df["Speaker"].unique()
        psi_groupby = psi_df.groupby("Speaker")
        psis = [torch.FloatTensor(psi_groupby.get_group(sp)["Psi"].to_numpy()) for sp in spkrs]
        num_feats = list(map(len,psis))
        assert all(num_feats[0]==np.array(num_feats)), "Inconsistent number of features across speakers."
    elif set(psi_df.columns) == set(["Group","Speaker","Psi","Feat"]):
        grps = psi_df["Group"].unique()
        psi_groupby = psi_df.groupby("Group")
        spkrs = [psi_groupby.get_group((grp))["Speaker"].unique() for grp in grps]

        psi_groupby = psi_df.groupby(["Group","Speaker"])
        psis = [[torch.FloatTensor(psi_groupby.get_group((grp,spkr))["Psi"].to_numpy()) for spkr in spkrs[g]] for g,grp in enumerate(grps)]
        num_feats = list(map(len,sum(psis,[])))
        assert all(num_feats[0]==np.array(num_feats)), "Inconsistent number of features across speakers."
    elif set(psi_df.columns) == set(["Data_trial","Group","Speaker","Psi","Feat"]):
        trials = psi_df["Data_trial"].unique()
        data_trials = len(trials)
        num_feat = psi_df["Feat"].nunique()

        psi_groupby = psi_df.groupby("Data_trial")
        grps = [psi_groupby.get_group(dt)["Group"].unique() for dt in trials]
        num_groups = list(map(len,grps))

        psi_groupby = psi_df.groupby(["Data_trial","Group"])
        spkrs = [[psi_groupby.get_group((trial,grp))["Speaker"].unique() for grp in grps[dt]] for dt,trial in enumerate(trials)]
        num_speakers = list(map(lambda dt:np.array(list(map(len,spkrs[dt]))), np.arange(data_trials)))

        psi_groupby = psi_df.groupby(["Data_trial","Group","Speaker"])
        psis = [[[torch.FloatTensor(psi_groupby.get_group((trial,grp,spkr))["Psi"].to_numpy()) for spkr in spkrs[dt][g]] for g,grp in enumerate(grps[dt])] for dt,trial in enumerate(trials)]
    else:
        print("Invalid type of file. Columns must contain at least ['Speaker','Feat','Psi'].")
        return

    return psis
def pi_from_DataFrame(pi_df:pd.DataFrame):
    if set(pi_df.columns) == set(["Speaker","Pi"]):
        pis = torch.FloatTensor(pi_df["Pi"].to_numpy())
    elif set(pi_df.columns) == set(["Group","Speaker","Pi"]):
        grps = pi_df["Group"].unique()
        pi_groupby = pi_df.groupby("Group")
        pis = [torch.FloatTensor(pi_groupby.get_group(grp)["Pi"].to_numpy()) for grp in grps]

    elif set(pi_df.columns) == set(["Data_trial","Group","Speaker","Pi"]):
        trials = pi_df["Data_trial"].unique()
        pi_groupby = pi_df.groupby("Data_trial")
        grps = [pi_groupby.get_group((trial))["Group"].unique() for trial in trials]

        pi_groupby = pi_df.groupby(["Data_trial","Group"])
        pis = [[torch.FloatTensor(pi_groupby.get_group((trial,grp))["Pi"].to_numpy()) for grp in grps[dt]] for dt,trial in enumerate(trials)]
    else:
        print("Invalid type of file. Columns must contain at least ['Speaker','Pi'].")
        return

    return pis
def d_from_DataFrame(d_df:pd.DataFrame):
    if set(d_df.columns) == set(["Speaker","D"]):
        ds = torch.FloatTensor(d_df["D"].to_numpy())
    elif set(d_df.columns) == set(["Group","Speaker","D"]):
        grps = d_df["Group"].unique()
        d_groupby = d_df.groupby("Group")
        ds = [torch.FloatTensor(d_groupby.get_group(grp)["D"].to_numpy()) for grp in grps]

    elif set(d_df.columns) == set(["Data_trial","Group","Speaker","D"]):
        trials = d_df["Data_trial"].unique()
        d_groupby = d_df.groupby("Data_trial")
        grps = [d_groupby.get_group((trial))["Group"].unique() for trial in trials]

        d_groupby = d_df.groupby(["Data_trial","Group"])
        ds = [[torch.FloatTensor(d_groupby.get_group((trial,grp))["D"].to_numpy()) for grp in grps[dt]] for dt,trial in enumerate(trials)]
    else:
        print("Invalid type of file. Columns must contain at least ['Speaker','D'].")
        return

    return ds
def A_from_DataFrame(A_df:pd.DataFrame):
    cols = ["Speaker1","Speaker2","A"]

    if set(A_df.columns) == set(cols):
        spkrs = A_df["Speaker1"].unique()
        num_speakers = len(spkrs)
        assert all(spkrs==A_df["Speaker2"].unique()), "Speakers in A invalid."
        A = torch.zeros((num_speakers,num_speakers))
        A[A_df["Speaker1"], A_df["Speaker2"]] = torch.FloatTensor(A_df["A"].to_numpy())
    elif set(A_df.columns) == set(["Group"] + cols):
        grps = A_df["Group"].unique()
        num_speakers = list(map(lambda g:A_df[A_df["Group"]==g]["Speaker1"].nunique(), grps))
        A_groupby = A_df.groupby("Group")
        A = [torch.zeros((num_speakers[g],num_speakers[g])) for g in range(len(grps))]
        for g,grp in enumerate(grps):
            A[g][np.array(A_groupby.get_group(grp)["Speaker1"]),np.array(A_groupby.get_group(grp)["Speaker2"])] = torch.FloatTensor(A_groupby.get_group(grp)["A"].to_numpy())
    elif set(A_df.columns) == set(["Data_trial","Group"] + cols):
        trials = A_df["Data_trial"].unique()
        data_trials = len(trials)

        A_groupby = A_df.groupby("Data_trial")
        grps = [A_groupby.get_group(dt)["Group"].unique() for dt in trials]
        num_groups = list(map(len,grps))

        A_groupby = A_df.groupby(["Data_trial","Group"])
        spkrs = [[A_groupby.get_group((trial,grp))["Speaker1"].unique() for grp in grps[dt]] for dt,trial in enumerate(trials)]
        num_speakers = list(map(lambda dt:np.array(list(map(len,spkrs[dt]))), np.arange(data_trials)))
        A = [[torch.zeros((num_speakers[t][g],num_speakers[t][g])) for g in range(len(grps[t]))] for t in range(data_trials)]
        for t,trial in enumerate(trials):
            for g,grp in enumerate(grps[t]):
                A[t][g][np.array(A_groupby.get_group((trial,grp))["Speaker1"]),np.array(A_groupby.get_group((trial,grp))["Speaker2"])] = \
                    torch.FloatTensor(A_groupby.get_group((trial,grp))["A"].to_numpy())
    else:
        print("Invalid type of file. Columns must contain at least ['Speaker','Feat','A'].")
        return

    return A
def convo_from_DataFrame(convo_df:pd.DataFrame):
    if set(convo_df.columns) == set(["Convo","Turn","Speaker"]):
        cnvos = convo_df["Convo"].unique()
        convo_groupby = convo_df.groupby("Convo")
        convos = [torch.tensor(convo_groupby.get_group(cnvo)["Speaker"].to_numpy()) for cnvo in cnvos]

    elif set(convo_df.columns) == set(["Group","Convo","Turn","Speaker"]):
        grps = convo_df["Group"].unique()
        convo_groupby = convo_df.groupby("Group")
        cnvos = [convo_groupby.get_group((grp))["Convo"].unique() for grp in grps]

        convo_groupby = convo_df.groupby(["Group","Convo"])
        convos = [[torch.tensor(convo_groupby.get_group((grp,cnvo))["Speaker"].to_numpy()) for cnvo in cnvos[g]] for g,grp in enumerate(grps)]


    elif set(convo_df.columns) == set(["Data_trial","Group","Convo","Turn","Speaker"]):
        trials = convo_df["Data_trial"].unique()
        data_trials = len(trials)

        convo_groupby = convo_df.groupby("Data_trial")
        grps = [convo_groupby.get_group(dt)["Group"].unique() for dt in trials]

        convo_groupby = convo_df.groupby(["Data_trial","Group"])
        cnvos = [[convo_groupby.get_group((trial,grp))["Convo"].unique() for grp in grps[dt]] for dt,trial in enumerate(trials)]

        convo_groupby = convo_df.groupby(["Data_trial","Group","Convo"])
        convos = [[[torch.tensor(convo_groupby.get_group((trial,grp,cnvo))["Speaker"].to_numpy()) for cnvo in cnvos[dt][g]] for g,grp in enumerate(grps[dt])] for dt,trial in enumerate(trials)]
    else:
        print("Invalid type of file. Columns must contain at least ['Convo','Turn','Speaker'].")
        return

    return convos


def load_psi(filename:str="psi",path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    assert os.path.exists(filepath), f"File at path `{filepath}` does not exist. Current folder: {os.getcwd()}"

    psi_df = pd.read_csv(filepath,sep=',')
    psis = psi_from_DataFrame(psi_df)

    return psis
def load_pi(filename:str="pi",path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    assert os.path.exists(filepath), f"File at path `{filepath}` does not exist. Current folder: {os.getcwd()}"

    pi_df = pd.read_csv(filepath,sep=',')
    pis = pi_from_DataFrame(pi_df)

    return pis
def load_d(filename:str="d",path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    assert os.path.exists(filepath), f"File at path `{filepath}` does not exist. Current folder: {os.getcwd()}"

    d_df = pd.read_csv(filepath,sep=',')
    ds = d_from_DataFrame(d_df)

    return ds
def load_A(filename:str="A",path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    assert os.path.exists(filepath), f"File at path `{filepath}` does not exist. Current folder: {os.getcwd()}"

    A_df = pd.read_csv(filepath,sep=',')
    As = A_from_DataFrame(A_df)

    return As
def load_convo(filename:str="convo",path:str=""):
    filename = f"{filename}.csv"
    filepath = os.path.join(path,filename)
    assert os.path.exists(filepath), f"File at path `{filepath}` does not exist. Current folder: {os.getcwd()}"

    convo_df = pd.read_csv(filepath,sep=',')
    convos = convo_from_DataFrame(convo_df)

    return convos

# ----------------------------------------------------------------

def Groups_from_lists(psi_grp,pi_grp=None,d_grp=None,A_grp=None,convo_grp=None):
    num_groups = len(psi_grp)
    # # if pi_grp is None and d_grp is None and convo_grp is None
    assert (pi_grp is None) or (num_groups==len(pi_grp)), "Inconsistent number of groups between psi and pi."
    assert (d_grp is None) or (num_groups==len(d_grp)), "Inconsistent number of groups between psi and d."
    assert (A_grp is None) or (num_groups==len(A_grp)), "Inconsistent number of groups between psi and A."
    assert (convo_grp is None) or (num_groups==len(convo_grp)), "Inconsistent number of groups between psi and convo."

    if pi_grp is None:
        pi_grp = [None]*num_groups
    if d_grp is None:
        d_grp = [None]*num_groups
    if A_grp is None:
        A_grp = [None]*num_groups
    if convo_grp is None:
        convo_grp = [None]*num_groups

    return [Group(psi_grp[g],pi_grp[g],d_grp[g],A_grp[g],convo_grp[g]) for g in range(num_groups)]

def Datasets_from_lists(psi_tris,pi_tris=None,d_tris=None,A_tris=None,convo_tris=None):
    data_trials = len(psi_tris)
    assert (pi_tris is None) or (data_trials==len(pi_tris)), "Inconsistent number of data trials between psi and pi."
    assert (d_tris is None) or (data_trials==len(d_tris)), "Inconsistent number of data trials between psi and d."
    assert (A_tris is None) or (data_trials==len(A_tris)), "Inconsistent number of data trials between psi and A."
    assert (convo_tris is None) or (data_trials==len(convo_tris)), "Inconsistent number of data trials between psi and convo."

    if pi_tris is None:
        pi_tris = [None]*data_trials
    if d_tris is None:
        d_tris = [None]*data_trials
    if A_tris is None:
        A_tris = [None]*data_trials
    if convo_tris is None:
        convo_tris = [None]*data_trials

    return [Dataset(Groups_from_lists(psi_tris[dt],pi_tris[dt],d_tris[dt],A_tris[dt],convo_tris[dt])) for dt in range(data_trials)]

# ----------------------------------------------------------------

class Group:
    def __init__(self, psi, pi=None, d=None, A=None, convos=None):
        assert (pi is None) or (len(psi)==len(pi)), \
            "Inconsistent number of speakers between psi and pi."
        assert (pi is None) or all([p>=0 for p in pi]), \
               "Invalid pi. All values must be nonnegative."
        assert (d is None) or (len(psi)==len(d)), \
            "Inconsistent number of speakers between psi and d."
        assert (d is None) or all([a>=0 for a in d]), \
               "Invalid d. All values must be nonnegative."
        assert (A is None) or (A.ndim==2 and A.shape[0]==len(psi) and A.shape[1]==len(psi)), \
            "Inconsistent number of speakers between psi and A."
        assert (A is None) or all([A[i1,i2]>=0 for i1 in range(len(psi)) for i2 in range(len(psi))]), \
               "Invalid A. All values must be nonnegative."
        assert all([np.isscalar(p) for p in psi]) or all(torch.Tensor(list(map(len,psi)))==len(psi[0])), \
            "Invalid number of features for psi. Must be consistent across speakers."
        assert (convos is None) or (type(convos)==list), \
            "Invalid conversations. Must be a list of conversations."
        assert (convos is None) or all([all((torch.unique(convo)[:,None].to('cpu')==torch.arange(len(psi))[None]).sum(1)>0) for convo in convos]), \
            "Invalid conversations. Speaker IDs in convo does not match expected number of speakers."
        
        self.num_speakers = len(psi)
        self.num_feat = len(psi[0]) if not np.isscalar(psi[0]) else 1
        self.num_convos = 0 if convos is None else len(convos)
        self.num_time = None if convos is None else [len(convos[c]) for c in range(self.num_convos)]

        self.psi = psi
        self.pi = pi
        self.d = d
        self.A = A
        self.convos = convos
        self.speakers_onehot = None if convos is None else [speakers_id_to_onehot(convos[c],self.num_speakers) for c in range(self.num_convos)]
        self.time_since = None if convos is None else [compute_time_since(self.speakers_onehot[c]) for c in range(self.num_convos)]

        return
    
    def get_psi(self,COPY:bool=False):
        if COPY:
            return [p.clone() for p in self.psi]
        else:
            return self.psi
    def get_pi(self,COPY:bool=False):
        if COPY and self.pi is not None:
            return self.pi.clone()
        else:
            return self.pi
    def get_d(self,COPY:bool=False):
        if COPY and self.d is not None:
            return self.d.clone()
        else:
            return self.d
    def get_A(self,COPY:bool=False):
        if COPY and self.A is not None:
            return self.A.clone()
        else:
            return self.A
    def get_convos(self,i:int=None,COPY:bool=False):
        assert (i is None) or (i in np.arange(self.num_convos)), "Invalid conversation index. Given index is out of range."
        if i is None and COPY:
            return [cnvo.clone() for cnvo in self.convos]
        elif i is None:
            return self.convos
        elif COPY:
            return self.convos[i].clone()
        else:
            return self.convos[i]

    def get_speakers_onehot(self,i:int=None,COPY:bool=False):
        assert (i is None) or (i in np.arange(self.num_convos)), "Invalid conversation index. Given index is out of range."
        if i is None and COPY:
            return [cnvo.clone() for cnvo in self.speakers_onehot]
        elif i is None:
            return self.speakers_onehot
        elif COPY:
            return self.speakers_onehot[i].clone()
        else:
            return self.speakers_onehot[i]
    def get_time_since(self,i:int=None,COPY:bool=False):
        assert (i is None) or (i in np.arange(self.num_convos)), "Invalid conversation index. Given index is out of range."
        if i is None and COPY:
            return [cnvo.clone() for cnvo in self.time_since]
        elif i is None:
            return self.time_since
        elif COPY:
            return self.time_since[i].clone()
        else:
            return self.time_since[i]
    def set_pi(self,pi):
        assert (pi is None) or (len(self.psi)==len(pi)), \
            "Inconsistent number of speakers between psi and pi."
        self.pi = pi
        return
    def set_d(self,d):
        assert (d is None) or (len(self.psi)==len(d)), \
            "Inconsistent number of speakers between psi and d."
        self.d = d
        return
    def set_A(self,A):
        assert (A is None) or (A.ndim==2 and A.shape[0]==len(self.psi) and A.shape[1]==len(self.psi)), \
            "Inconsistent number of speakers between psi and A."
        self.A = A
        return
    def set_convos(self,convos=None):
        assert (convos is None) or (type(convos)==list), \
            "Invalid conversations. Must be a list of conversations."
        assert (convos is None) or all([all((torch.unique(convo)[:,None].to('cpu')==torch.arange(self.num_speakers)[None]).sum(1)>0) for convo in convos]), \
            "Invalid conversations. Speaker IDs in convo does not match expected number of speakers."
        self.num_convos = 0 if convos is None else len(convos)
        self.num_time = None if convos is None else [len(convos[c]) for c in range(self.num_convos)]
        self.convos = convos
        self.speakers_onehot = None if convos is None else [speakers_id_to_onehot(convos[c],self.num_speakers) for c in range(self.num_convos)]
        self.time_since = None if convos is None else [compute_time_since(self.speakers_onehot[c]) for c in range(self.num_convos)]
        return
    def reset_convos(self):
        self.set_convos(None)
        return
    
    def normalize(self):
        scale = (1/3)*(torch.mean(self.pi) + torch.mean(self.d) + torch.mean(self.A))
        scale = scale if scale!=0 else 1.
        self.pi = self.pi/scale
        self.d = self.d/scale
        self.A = self.A/scale
        return

    def get_negloglik(self,i:int=None,w=None):
        assert (i is None) or (i in np.arange(self.num_convos)), "Invalid conversation index. Given index is out of range."
        if w is None:
            w = lambda x:torch.exp(-.5*x)
        if i is None:
            betas = [compute_betas(w,self.time_since[c]) for c in range(self.num_convos)]
            return [negloglik(coh,self.pi,self.d,self.A,betas[c]) for c,coh in enumerate(self.speakers_onehot)]
        else:
            beta = compute_betas(w,self.time_since[i])
            return negloglik(self.speakers_onehot[i],self.pi,self.d,self.A,beta)

    def get_psi_df(self):
        cols = ["Speaker", "Feat", "Psi"]

        # Psi column:
        all_psi = np.concatenate(self.psi)

        # Trait column:
        all_feat = np.tile(np.arange(self.num_feat),self.num_speakers)

        # Speaker column:
        all_spkr = np.repeat(np.arange(self.num_speakers),self.num_feat)

        assert len(all_psi)==len(all_feat) and len(all_psi)==len(all_spkr), "Invalid column lengths for DataFrame."

        psi_pd = pd.DataFrame(columns=cols)
        psi_pd['Speaker'] = all_spkr
        psi_pd['Feat'] = all_feat
        psi_pd['Psi'] = all_psi
        return psi_pd
    def get_pi_df(self):
        assert hasattr(self,"pi") and (self.pi is not None), "No attribute available."
        cols = ["Speaker", "Pi"]
        all_spkr = np.arange(self.num_speakers)
        all_pi = self.pi.numpy().astype(float)

        pi_pd = pd.DataFrame(columns=cols)
        pi_pd["Speaker"] = all_spkr
        pi_pd["Pi"] = all_pi
        return pi_pd
    def get_d_df(self):
        assert hasattr(self,"d") and (self.d is not None), "No attribute available."
        cols = ["Speaker", "D"]
        all_spkr = np.arange(self.num_speakers)
        all_d = self.d.numpy().astype(float)

        d_pd = pd.DataFrame(columns=cols)
        d_pd["Speaker"] = all_spkr
        d_pd["D"] = all_d
        return d_pd
    def get_A_df(self):
        assert hasattr(self,"A") and (self.A is not None), "No attribute available."
        cols = ["Speaker1","Speaker2","A"]
        all_spkr1 = np.repeat(np.arange(self.num_speakers),self.num_speakers)
        all_spkr2 = np.tile(np.arange(self.num_speakers),self.num_speakers)
        all_A = torch.cat(list(self.A)).numpy().astype(float)

        A_pd = pd.DataFrame(columns=cols)
        A_pd["Speaker1"] = all_spkr1
        A_pd["Speaker2"] = all_spkr2
        A_pd["A"] = all_A
        return A_pd
    def get_convo_df(self):
        assert hasattr(self,"convos") and (self.convos is not None), "No attribute available."
        cols = ["Convo","Turn","Speaker"]

        all_cnvos = np.repeat(np.arange(self.num_convos), self.num_time)
        all_turns = np.concatenate(list(map(np.arange,self.num_time)))
        all_spkrs = torch.cat(self.convos).numpy().astype(int)
        cnvo_pd = pd.DataFrame(columns=cols)
        cnvo_pd["Convo"] = all_cnvos
        cnvo_pd["Turn"] = all_turns
        cnvo_pd["Speaker"] = all_spkrs
        return cnvo_pd
    
    def save_psi(self,filename:str="psi",path:str=""):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_psi_df().to_csv(filepath,index=False)
    def save_pi(self,filename:str="pi",path:str=""):
        assert hasattr(self,"pi") and (self.pi is not None), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_pi_df().to_csv(filepath,index=False)
    def save_d(self,filename:str="d",path:str=""):
        assert hasattr(self,"d") and (self.d is not None), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_d_df().to_csv(filepath,index=False)
    def save_A(self,filename:str="A",path:str=""):
        assert hasattr(self,"A") and (self.A is not None), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_A_df().to_csv(filepath,index=False)
    def save_convo(self,filename:str="convo",path:str=""):
        assert hasattr(self,"convos") and (self.convos is not None), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_convo_df().to_csv(filepath,index=False)
    def save(self,psi_file:str, pi_file:str=None, d_file:str=None, A_file:str=None, convo_file:str=None, path:str=""):
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_psi(psi_file,path)
        if self.pi is not None:
            self.save_pi(pi_file,path)
        if self.d is not None:
            self.save_d(d_file,path)
        if self.A is not None:
            self.save_A(A_file,path)
        if self.convos is not None:
            self.save_convo(convo_file,path)

    def copy(self):
        psi_clone = [p.clone() for p in self.psi] if (hasattr(self,"psi") and self.psi is not None) else None
        pi_clone = self.pi.clone() if (hasattr(self,"pi") and self.pi is not None) else None
        d_clone = self.d.clone() if (hasattr(self,"d") and self.d is not None) else None
        A_clone = self.A.clone() if (hasattr(self,"A") and self.A is not None) else None
        convos_clone = [cnvo.clone() if cnvo is not None else None for cnvo in self.convos] if (hasattr(self,"convos") and self.convos is not None) else None
        coh_clone = [cnvo.clone() if cnvo is not None else None for cnvo in self.speakers_onehot] if (hasattr(self,"speakers_onehot") and self.speakers_onehot is not None) else None
        ts_clone = [ts.clone() if ts is not None else None for ts in self.time_since] if (hasattr(self,"time_since") and self.time_since is not None) else None

        group_copy = Group(psi_clone)
        group_copy.pi = pi_clone
        group_copy.d = d_clone
        group_copy.A = A_clone
        group_copy.convos = convos_clone
        group_copy.speakers_onehot = coh_clone
        group_copy.time_since = ts_clone

        group_copy.num_speakers = len(group_copy.psi)
        group_copy.num_feat = len(group_copy.psi[0])
        group_copy.num_convos = 0 if group_copy.convos is None else len(group_copy.convos)
        group_copy.num_time = None if group_copy.convos is None else [len(group_copy.convos[c]) for c in range(group_copy.num_convos)]
        # if coh_clone is not None:
        #     group_copy.speakers_onehot = coh_clone
        return group_copy
    
    def to(self,device):
        group_copy = self.copy()
        if hasattr(group_copy,"psi") and group_copy.psi is not None:
            group_copy.psi = [p.to(device) if p is not None else None for p in group_copy.psi]
        if hasattr(group_copy,"pi") and group_copy.pi is not None:
            group_copy.pi = group_copy.pi.to(device)
        if hasattr(group_copy,"d") and group_copy.d is not None:
            group_copy.d = group_copy.d.to(device)
        if hasattr(group_copy,"A") and group_copy.A is not None:
            group_copy.A = group_copy.A.to(device)
        if hasattr(group_copy,"convos") and group_copy.convos is not None:
            group_copy.convos = [convo.to(device) if convo is not None else None for convo in group_copy.convos]
        if hasattr(group_copy,"speakers_onehot") and group_copy.speakers_onehot is not None:
            group_copy.speakers_onehot = [convo.to(device) if convo is not None else None for convo in group_copy.speakers_onehot]
        if hasattr(group_copy,"time_since") and group_copy.time_since is not None:
            group_copy.time_since = [convo.to(device) if convo is not None else None for convo in group_copy.time_since]
        return group_copy

class Dataset:
    def __init__(self,groups):
        assert type(groups)==list and all([type(grp==Group) for grp in groups]), "Invalid input data. Must be list of Group objects."
        assert all([groups[0].num_feat==groups[g].num_feat for g in range(1,len(groups))]), \
            "Inconsistent number of features in groups."

        self.groups = groups

        self.num_feat = self.groups[0].num_feat
        self.num_groups = len(self.groups)
        self.num_speakers = [self.groups[g].num_speakers for g in range(len(self.groups))]

        self.tensor = torch.stack(sum([self.groups[g].psi for g in range(len(self.groups))],[]))
        # TODO: Check traits
        self.pairwise_tensor = torch.cat( (torch.vstack([torch.stack(self.groups[g].psi).repeat_interleave(self.num_speakers[g],0) for g in range(self.num_groups)]),
                                           torch.vstack([torch.stack(self.groups[g].psi).repeat(self.num_speakers[g],1) for g in range(self.num_groups)])), dim=1 )
        locs_grps = torch.cat(list(map( lambda g:g*torch.ones(self.num_speakers[g],dtype=int),np.arange(self.num_groups))))
        self.inds_grps = [torch.where(locs_grps==g)[0] for g in range(self.num_groups)]
        locs_pair_grps = torch.cat(list(map( lambda g:g*torch.ones(self.num_speakers[g]**2,dtype=int),np.arange(self.num_groups))))
        self.pair_inds_grps = [torch.where(locs_pair_grps==g)[0] for g in range(self.num_groups)]

        self.convos = [grp.convos for grp in self.groups]
        self.speakers_onehot = [grp.speakers_onehot for grp in self.groups]
        self.time_since = [grp.time_since for grp in self.groups]
        self.num_times = [groups[g].num_time for g in range(self.num_groups)]
        self.num_convos = [groups[g].num_convos for g in range(self.num_groups)]

        self.psi = [grp.psi for grp in self.groups]
        self.pi = [grp.pi for grp in self.groups]
        self.d = [grp.d for grp in self.groups]
        self.A = [grp.A for grp in self.groups]
    
    def get_group(self,i:int):
        assert i in torch.arange(self.num_groups), "Invalid group index. Given index is out of range."
        return self.groups[i]

    def set_pi(self,pi):
        assert (pi is None) or (self.num_groups==len(pi)), "Inconsistent number of groups."
        assert (pi is None) or all([len(pi[g])==self.num_speakers[g] for g in range(self.num_groups)]), "Inconsistent number of speakers."
        _ = [self.groups[g].set_pi(pi[g]) for g in range(self.num_groups)]
        self.pi = pi
        return
    def set_d(self,d):
        assert (d is None) or (self.num_groups==len(d)), "Inconsistent number of groups."
        assert (d is None) or all([len(d[g])==self.num_speakers[g] for g in range(self.num_groups)]), "Inconsistent number of speakers."
        _ = [self.groups[g].set_d(d[g]) for g in range(self.num_groups)]
        self.d = d
        return
    def set_A(self,A):
        assert (A is None) or (self.num_groups==len(A)), "Inconsistent number of groups."
        assert (A is None) or all([(A[g].shape[0]==self.num_speakers[g]) and (A[g].shape[1]==self.num_speakers[g]) for g in range(self.num_groups)]), "Inconsistent number of speakers."
        _ = [self.groups[g].set_A(A[g]) for g in range(self.num_groups)]
        self.A = A
        return
    def set_convos(self,convos):
        assert (convos is None) or (self.num_groups==len(convos)), "Inconsistent number of groups."
        assert (convos is None) or all([all((torch.unique(convos[g][c])[:,None].to('cpu')==torch.arange(self.num_speakers[g])[None]).sum(1)>0) for g in range(self.num_groups) for c in range(len(convos[g]))]), \
            "Invalid conversations. Speaker IDs in convo does not match expected number of speakers."
        _ = [self.groups[g].set_convos(convos[g]) for g in range(self.num_groups)]
        self.convos = convos
        self.speakers_onehot = [self.groups[g].speakers_onehot for g in range(self.num_groups)]
        self.time_since = [self.groups[g].time_since for g in range(self.num_groups)]
        self.num_convos = [self.groups[g].num_convos for g in range(self.num_groups)]
        self.num_times = [[self.groups[g].num_time[c] for c in range(self.groups[g].num_convos)] for g in range(self.num_groups)]
        return

    def generate_convos(self,w,num_times=None):
        if num_times is None:
            num_times = self.num_times
        assert (not np.isscalar(num_times) and all([not np.isscalar(num_times[g]) for g in range(len(num_times))]) and all([num_times[g][c]>=0 for g in range(len(num_times)) for c in range(len(num_times[g]))])), \
            "Invalid number of turns. Expecting a list of lists or arrays of numbers of turns per conversation for each group."

        return [generate_convos(self.pi[g], self.d[g], self.A[g], num_times[g], w) for g in range(self.num_groups)]

    def normalize(self):
        pi_mean = torch.cat(self.pi).mean()
        d_mean = torch.cat(self.d).mean()
        A_mean = torch.cat(self.A).mean()
        scale = (1/3)*(pi_mean+d_mean+A_mean)

        for g in range(self.num_groups):
            self.groups[g].pi = self.groups[g].pi/scale
            self.groups[g].d = self.groups[g].d/scale
            self.groups[g].A = self.groups[g].A/scale
        self.pi = [grp.pi for grp in self.groups]
        self.d = [grp.d for grp in self.groups]
        self.A = [grp.A for grp in self.groups]
        return

    def get_psi_df(self):
        psi_df_list = list(map(lambda g:self.groups[g].get_psi_df(),np.arange(self.num_groups)))
        num_grp_inds = self.num_feat * np.array(self.num_speakers)
        all_grps = np.concatenate(list(map(lambda g:g*np.ones(num_grp_inds[g]), np.arange(self.num_groups)))).astype(int)
        psi_df = pd.concat(psi_df_list)
        psi_df["Group"] = all_grps
        psi_df = psi_df[["Group","Speaker","Feat","Psi"]]
        return psi_df
    def get_pi_df(self):
        assert hasattr(self,"pi") and (self.pi is not None), "No attribute available."
        pi_df_list = list(map(lambda g:self.groups[g].get_pi_df(),np.arange(self.num_groups)))
        all_grps = np.concatenate(list(map(lambda g:g*np.ones(self.num_speakers[g]), np.arange(self.num_groups)))).astype(int)
        pi_df = pd.concat(pi_df_list)
        pi_df["Group"] = all_grps
        pi_df = pi_df[["Group","Speaker","Pi"]]
        return pi_df
    def get_d_df(self):
        assert hasattr(self,"d") and (self.d is not None), "No attribute available."
        d_df_list = list(map(lambda g:self.groups[g].get_d_df(),np.arange(self.num_groups)))
        all_grps = np.concatenate(list(map(lambda g:g*np.ones(self.num_speakers[g]), np.arange(self.num_groups)))).astype(int)
        d_df = pd.concat(d_df_list)
        d_df["Group"] = all_grps
        d_df = d_df[["Group","Speaker","D"]]
        return d_df
    def get_A_df(self):
        assert hasattr(self,"A") and (self.A is not None), "No attribute available."
        A_df_list = list(map(lambda g:self.groups[g].get_A_df(),np.arange(self.num_groups)))
        all_grps = np.concatenate(list(map(lambda g:g*np.ones(self.num_speakers[g]**2), np.arange(self.num_groups)))).astype(int)
        A_df = pd.concat(A_df_list)
        A_df["Group"] = all_grps
        A_df = A_df[["Group","Speaker1","Speaker2","A"]]
        return A_df
    def get_convo_df(self):
        assert hasattr(self,"convos") and (self.convos is not None), "No attribute available."
        convo_df_list = list(map(lambda g:self.groups[g].get_convo_df(),np.arange(self.num_groups)))
        num_grp_inds = list(map(np.sum,self.num_times))
        all_grps = np.concatenate(list(map(lambda g:g*np.ones(num_grp_inds[g]), np.arange(self.num_groups)))).astype(int)
        convo_df = pd.concat(convo_df_list)
        convo_df["Group"] = all_grps
        convo_df = convo_df[["Group","Convo","Turn","Speaker"]]
        return convo_df
    
    def save_psi(self,filename:str="psi",path:str=""):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_psi_df().to_csv(filepath,index=False)
    def save_pi(self,filename:str="pi",path:str=""):
        assert hasattr(self,"pi") and all([t is not None for t in self.pi]), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_pi_df().to_csv(filepath,index=False)
    def save_d(self,filename:str="d",path:str=""):
        assert hasattr(self,"d") and all([t is not None for t in self.d]), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_d_df().to_csv(filepath,index=False)
    def save_A(self,filename:str="A",path:str=""):
        assert hasattr(self,"A") and all([t is not None for t in self.A]), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_A_df().to_csv(filepath,index=False)
    def save_convo(self,filename:str="convo",path:str=""):
        assert hasattr(self,"convos") and all([t is not None for t in self.convos]), "No attribute available."
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f"{filename}.csv"
        filepath = os.path.join(path,filename)
        self.get_convo_df().to_csv(filepath,index=False)

    def save(self,psi_file:str, pi_file:str=None, d_file:str=None, A_file:str=None, convo_file:str=None, path:str=""):
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_psi(psi_file,path)
        if all([t is not None for t in self.pi]):
            self.save_pi(pi_file,path)
        if all([t is not None for t in self.d]):
            self.save_d(d_file,path)
        if all([t is not None for t in self.A]):
            self.save_A(A_file,path)
        if all([t is not None for t in self.convos]):
            self.save_convo(convo_file,path)

    def copy(self):
        ds_clone = Dataset([grp.copy() for grp in self.groups])
        ds_clone.tensor = self.tensor.clone() if (hasattr(self,"tensor") and self.tensor is not None) else None
        ds_clone.convos = [[self.convos[g][c].clone() if self.convos[g][c] is not None else None for c in range(self.num_convos[g])] for g in range(self.num_groups)] if (hasattr(self,"convos") and self.convos is not None) else None
        ds_clone.speakers_onehot = [[self.speakers_onehot[g][c].clone() if self.speakers_onehot[g][c] is not None else None for c in range(self.num_convos[g])] for g in range(self.num_groups)] if (hasattr(self,"speakers_onehot") and self.speakers_onehot is not None) else None
        ds_clone.time_since = [[self.time_since[g][c].clone() if self.time_since[g][c] is not None else None for c in range(self.num_convos[g])] for g in range(self.num_groups)] if (hasattr(self,"time_since") and self.time_since is not None) else None
        ds_clone.psi = [[self.psi[g][s].clone() for s in range(self.num_speakers[g])] for g in range(self.num_groups)] if (hasattr(self,"psi") and self.psi is not None) else None
        ds_clone.pi = [self.pi[g].clone() if self.pi[g] is not None else None for g in range(self.num_groups)] if (hasattr(self,"pi") and self.pi is not None) else None
        ds_clone.d = [self.d[g].clone() if self.d[g] is not None else None for g in range(self.num_groups)] if (hasattr(self,"d") and self.d is not None) else None
        ds_clone.A = [self.A[g].clone() if self.A[g] is not None else None for g in range(self.num_groups)] if (hasattr(self,"A") and self.A is not None) else None

        return ds_clone

    def to(self,device):
        ds_clone = self.copy()
        if hasattr(ds_clone,"tensor") and ds_clone.tensor is not None:
            ds_clone.tensor = ds_clone.tensor.to(device)
        if hasattr(ds_clone,"convos") and ds_clone.convos is not None:
            ds_clone.convos = [[ds_clone.convos[g][c].to(device) if ds_clone.convos[g][c] is not None else None for c in range(ds_clone.num_convos[g])] for g in range(ds_clone.num_groups)]
        if hasattr(ds_clone,"speakers_onehot") and ds_clone.speakers_onehot is not None:
            ds_clone.speakers_onehot = [[ds_clone.speakers_onehot[g][c].to(device) if ds_clone.speakers_onehot[g][c] is not None else None for c in range(ds_clone.num_convos[g])] for g in range(ds_clone.num_groups)]
        if hasattr(ds_clone,"time_since") and ds_clone.time_since is not None:
            ds_clone.time_since = [[ds_clone.time_since[g][c].to(device) if ds_clone.time_since[g][c] is not None else None for c in range(ds_clone.num_convos[g])] for g in range(ds_clone.num_groups)]
        if hasattr(ds_clone,"psi") and ds_clone.psi is not None:
            ds_clone.psi = [[ds_clone.psi[g][s].to(device) if ds_clone.psi[g][s] is not None else None for s in range(ds_clone.num_speakers[g])] for g in range(ds_clone.num_groups)]
        if hasattr(ds_clone,"pi") and ds_clone.pi is not None:
            ds_clone.pi = [p.to(device) if p is not None else None for p in ds_clone.pi]
        if hasattr(ds_clone,"d") and ds_clone.d is not None:
            ds_clone.d = [a.to(device) if a is not None else None for a in ds_clone.d]
        if hasattr(ds_clone,"A") and ds_clone.A is not None:
            ds_clone.A = [s.to(device) if s is not None else None for s in ds_clone.A]
        if hasattr(ds_clone,"groups") and ds_clone.groups is not None:
            # ds_clone.groups = [grp.to(device) for grp in ds_clone.groups]
            ds_clone.groups = Groups_from_lists(ds_clone.psi,ds_clone.pi,ds_clone.d,ds_clone.A,ds_clone.convos)
        
        return ds_clone

def save_dataset_with_group_ids(dataset,group_ids, 
                                psi_file:str, pi_file:str=None, d_file:str=None, A_file:str=None, convo_file:str=None, path:str=""):
    assert dataset.num_groups==len(group_ids), "Inconsistent number of groups between `dataset` and `group_ids`."

    group_index2ID = dict(zip(np.arange(len(group_ids)),np.array(group_ids)))
    if not os.path.exists(path):
        os.makedirs(path)

    psi_path = os.path.join(path,f"{psi_file}.csv")
    psi_df = dataset.get_psi_df()
    grps_df = psi_df["Group"].copy()
    for grp in np.arange(dataset.num_groups):
        psi_df.loc[grps_df==grp,"Group"] = group_index2ID[grp]
    psi_df.to_csv(psi_path,index=False)

    if pi_file is not None:
        pi_path = os.path.join(path,f"{pi_file}.csv")
        if all([t is not None for t in dataset.pi]):
            pi_df = dataset.get_pi_df()
            grps_df = pi_df["Group"].copy()
            for grp in np.arange(dataset.num_groups):
                pi_df.loc[grps_df==grp,"Group"] = group_index2ID[grp]
            pi_df.to_csv(pi_path,index=False)

    if d_file is not None:
        d_path = os.path.join(path,f"{d_file}.csv")
        if all([t is not None for t in dataset.d]):
            d_df = dataset.get_d_df()
            grps_df = d_df["Group"].copy()
            for grp in np.arange(dataset.num_groups):
                d_df.loc[grps_df==grp,"Group"] = group_index2ID[grp]
            d_df.to_csv(d_path,index=False)

    if A_file is not None:
        A_path = os.path.join(path,f"{A_file}.csv")
        if all([t is not None for t in dataset.A]):
            A_df = dataset.get_A_df()
            grps_df = A_df["Group"].copy()
            for grp in np.arange(dataset.num_groups):
                A_df.loc[grps_df==grp,"Group"] = group_index2ID[grp]
            A_df.to_csv(A_path,index=False)

    if convo_file is not None:
        convo_path = os.path.join(path,f"{convo_file}.csv")
        if all([t is not None for t in dataset.convos]):
            convo_df = dataset.get_convo_df()
            grps_df = convo_df["Group"].copy()
            for grp in np.arange(dataset.num_groups):
                convo_df.loc[grps_df==grp,"Group"] = group_index2ID[grp]
            convo_df.to_csv(convo_path,index=False)

    return

# ----------------------------------------------------------------

def load_synthdata_params_from_config(config_path:str):
    if (config_path is not None) and os.path.exists(config_path):
        with open(config_path,'r') as f:
            CONFIG = json.load(f)

    key = "dataset_path"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==str, "Invalid dataset path `dataset_path`. Expecting a string."
    else:
        CONFIG[key] = ""

    key = "verbose"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag for verbosity `verbose`. Expecting a boolean value."
    else:
        CONFIG[key] = False

    key = "seed"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int, "Invalid random seed `seed`. Expecting an integer."
    else:
        CONFIG[key] = 1

    key = "data_trials"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int and CONFIG[key]>0, "Invalid number of data trials `data_trials`. Expecting a positive integer."
    else:
        CONFIG[key] = 1

    key = "num_feat"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int and CONFIG[key]>0, "Invalid number of features `num_feat`. Expecting a positive integer."
    else:
        CONFIG[key] = 2

    key = "b"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==float and CONFIG[key]>=0, "Invalid scalar `b`. Expecting a nonnegative float."
    else:
        CONFIG[key] = .5

    key = "complexity_pi"
    if key in CONFIG.keys():
        assert CONFIG[key] is None or type(CONFIG[key])==str, "Invalid setting for pi trait complexity `complexity_pi`. Expecting a string."
    else:
        CONFIG[key] = "simple_uncorrel"

    key = "complexity_d"
    if key in CONFIG.keys():
        assert CONFIG[key] is None or type(CONFIG[key])==str, "Invalid setting for d trait complexity `complexity_d`. Expecting a string."
    else:
        CONFIG[key] = "simple_uncorrel"

    key = "affinity_type"
    if key in CONFIG.keys():
        assert CONFIG[key] is None or type(CONFIG[key])==str, "Invalid setting for pairwise affinity type `affinity_type`. Expecting a string."
    else:
        CONFIG[key] = "simple_posdot"

    key = "data_has_memory"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag `data_has_memory`. Expecting a boolean value."
    else:
        CONFIG[key] = True

    key = "data_has_affinity"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag `data_has_affinity`. Expecting a boolean value."
    else:
        CONFIG[key] = True

    for SPLIT in ["TRAIN","VAL","TEST"]:
        if SPLIT not in CONFIG.keys():
            CONFIG[SPLIT] = {}
        key = "num_speakers"
        if key in CONFIG[SPLIT].keys():
            assert (len(CONFIG[SPLIT][key])>0 and 
                    all([type(s)==int for s in CONFIG[SPLIT][key]]) and
                    all([s>0 for s in CONFIG[SPLIT][key]])), "Invalid number of speakers `num_speakers`. Expecting a list of positive integers."
            CONFIG[SPLIT]["num_groups"] = len(CONFIG[SPLIT][key])
        else:
            CONFIG[SPLIT][key] = [4]
            CONFIG[SPLIT]["num_groups"] = len(CONFIG[SPLIT][key])

        key = "num_time"
        if key in CONFIG[SPLIT].keys():
            assert (len(CONFIG[SPLIT][key])==CONFIG[SPLIT]["num_groups"] and 
                    all(list(map(lambda x:type(x)==int,sum(CONFIG[SPLIT][key],[])))) and
                    all(np.concatenate(CONFIG[SPLIT][key])>0)), "Invalid number of turns per convo `num_time`. Expecting a list of lists of positive integers with same number of groups as `num_speakers`."
            CONFIG[SPLIT]["num_convos"] = list(map(len,CONFIG[SPLIT][key]))
        else:
            CONFIG[SPLIT][key] = [[100] for _ in range(CONFIG[SPLIT]["num_groups"])]
            CONFIG[SPLIT]["num_convos"] = list(map(len,CONFIG[SPLIT][key]))
    
    return CONFIG

def load_synthdata_args_from_config(config_path:str):
    if (config_path is not None) and os.path.exists(config_path):
        with open(config_path,'r') as f:
            CONFIG = json.load(f)

    key = "dataset_path"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==str, "Invalid dataset path `dataset_path`. Expecting a string."
    else:
        CONFIG[key] = ""

    key = "verbose"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag for verbosity `verbose`. Expecting a boolean value."
    else:
        CONFIG[key] = False

    key = "seed"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int, "Invalid random seed `seed`. Expecting an integer."
    else:
        CONFIG[key] = 1

    key = "data_trials"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int and CONFIG[key]>0, "Invalid number of data trials `data_trials`. Expecting a positive integer."
    else:
        CONFIG[key] = 1

    key = "num_feat"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int and CONFIG[key]>0, "Invalid number of features `num_feat`. Expecting a positive integer."
    else:
        CONFIG[key] = 2

    key = "b"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==float and CONFIG[key]>=0, "Invalid scalar `b`. Expecting a nonnegative float."
    else:
        CONFIG[key] = .5

    key = "complexity_pi"
    if key in CONFIG.keys():
        assert CONFIG[key] is None or type(CONFIG[key])==str, "Invalid setting for pi trait complexity `complexity_pi`. Expecting a string."
    else:
        CONFIG[key] = "simple_uncorrel"

    key = "complexity_d"
    if key in CONFIG.keys():
        assert CONFIG[key] is None or type(CONFIG[key])==str, "Invalid setting for d trait complexity `complexity_d`. Expecting a string."
    else:
        CONFIG[key] = "simple_uncorrel"

    key = "affinity_type"
    if key in CONFIG.keys():
        assert CONFIG[key] is None or type(CONFIG[key])==str, "Invalid setting for pairwise affinity type `affinity_type`. Expecting a string."
    else:
        CONFIG[key] = "simple_posdot"

    key = "data_has_memory"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag `data_has_memory`. Expecting a boolean value."
    else:
        CONFIG[key] = True

    key = "data_has_affinity"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag `data_has_affinity`. Expecting a boolean value."
    else:
        CONFIG[key] = True

    for SPLIT in ["TRAIN","VAL","TEST"]:
        if SPLIT not in CONFIG.keys():
            CONFIG[SPLIT] = {}
        key = "num_speakers"
        if key in CONFIG[SPLIT].keys():
            assert (len(CONFIG[SPLIT][key])>0 and 
                    all([type(s)==int for s in CONFIG[SPLIT][key]]) and
                    all([s>0 for s in CONFIG[SPLIT][key]])), "Invalid number of speakers `num_speakers`. Expecting a list of positive integers."
            CONFIG[SPLIT]["num_groups"] = len(CONFIG[SPLIT][key])
        else:
            CONFIG[SPLIT][key] = [4]
            CONFIG[SPLIT]["num_groups"] = len(CONFIG[SPLIT][key])

        key = "num_time"
        if key in CONFIG[SPLIT].keys():
            assert (len(CONFIG[SPLIT][key])==CONFIG[SPLIT]["num_groups"] and 
                    all(list(map(lambda x:type(x)==int,sum(CONFIG[SPLIT][key],[])))) and
                    all(np.concatenate(CONFIG[SPLIT][key])>0)), "Invalid number of turns per convo `num_time`. Expecting a list of lists of positive integers with same number of groups as `num_speakers`."
            CONFIG[SPLIT]["num_convos"] = list(map(len,CONFIG[SPLIT][key]))
        else:
            CONFIG[SPLIT][key] = [[100] for _ in range(CONFIG[SPLIT]["num_groups"])]
            CONFIG[SPLIT]["num_convos"] = list(map(len,CONFIG[SPLIT][key]))
    
    train_args = {
        "num_speakers":CONFIG['TRAIN']['num_speakers'],
        "num_time":CONFIG['TRAIN']['num_time'],
    }
    val_args = {
        "num_speakers":CONFIG['VAL']['num_speakers'],
        "num_time":CONFIG['VAL']['num_time'],
    }
    test_args = {
        "num_speakers":CONFIG['TEST']['num_speakers'],
        "num_time":CONFIG['TEST']['num_time'],
    }
    for key in ['num_feat','complexity_pi','complexity_d','affinity_type','data_has_memory','data_has_affinity']:
        train_args[key] = CONFIG[key]
        val_args[key] = CONFIG[key]
        test_args[key] = CONFIG[key]
    train_args['w'] = lambda x: torch.exp(-CONFIG['b'] * x)
    val_args['w'] = lambda x: torch.exp(-CONFIG['b'] * x)
    test_args['w'] = lambda x: torch.exp(-CONFIG['b'] * x)

    return train_args, val_args, test_args, CONFIG['data_trials']

def load_realdata_params_from_config(config_path:str):
    if (config_path is not None) and os.path.exists(config_path):
        with open(config_path,'r') as f:
            CONFIG = json.load(f)

        key = "dataset_path"
        if key in CONFIG.keys():
            assert type(CONFIG[key])==str, "Invalid dataset path `dataset_path`. Expecting a string."
        else:
            CONFIG[key] = ""

    key = "verbose"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==bool, "Invalid flag for verbosity `verbose`. Expecting a boolean value."
    else:
        CONFIG[key] = False

    key = "seed"
    if key in CONFIG.keys():
        assert type(CONFIG[key])==int, "Invalid random seed `seed`. Expecting an integer."
    else:
        CONFIG[key] = 1

    assert ("VAL" not in CONFIG.keys() and "TEST" not in CONFIG.keys()) or all([len(CONFIG["TRAIN"])==len(CONFIG[SPLIT]) for SPLIT in ["VAL","TEST"] if SPLIT in CONFIG.keys()]), "Invalid number of data trials across data splits."
    CONFIG["data_trials"] = len(CONFIG["TRAIN"])
    return CONFIG

def generate_dataset(num_feat,num_speakers,num_time,complexity_pi,complexity_d,affinity_type,w,data_has_memory:bool=True,data_has_affinity:bool=True):
    assert len(num_speakers)==len(num_time), "Invalid input. Inconsistent number of groups between `num_speakers` and `num_time`."
    num_groups = len(num_speakers)
    psis = [generate_psis(num_feat, num_speakers[g]) for g in range(num_groups)]
    pis = [generate_pis(psis[g],complexity_pi) for g in range(num_groups)]
    ds = [generate_ds(psis[g],complexity_d) if data_has_memory else torch.zeros_like(pis[g]) for g in range(num_groups)]
    As = [generate_A(psis[g],affinity_type) if data_has_affinity else torch.zeros((num_speakers[g],num_speakers[g])) for g in range(num_groups)]
    convos = [generate_convos(pis[g],ds[g],As[g],num_time[g],w) for g in range(num_groups)]
    grps = [Group(psis[g],pis[g],ds[g],As[g],convos[g]) for g in range(num_groups)]
    data = Dataset(grps)
    return data

# ----------------------------------------------------------------

def plot_synthetic_data_1d(complexity_pi, complexity_d, affinity_type, w=None,
        data_has_memory:bool=True, data_has_affinity:bool=True,
        num_grid:int=30, psi_min:float=.1, psi_max:float=1.
    ):
    clrs = {"pi":bq['red'], "d":bq['blue'], "A":bq['yellow'], "w":bq["green"], 'l':bq['purple']}
    cmaps ={"pi":"viridis", "d":"cividis", "A":"plasma", "w":"inferno", "l":"magma"} 
    surfs = {"pi":'summer', "d":'winter', "A":'autumn', "w":'plasma', 'l':'spring'}
    line_args = { "linestyle":'-', "linewidth":3, }

    if w is None:
        w = lambda x: torch.exp(-.5*x) if type(x) is torch.Tensor else (np.exp(-.5*x).item() if np.isscalar(x) else np.exp(-.5*x))

    psi_range = np.linspace(psi_min,psi_max,num_grid)
    pi_range = generate_pis(list(torch.FloatTensor(psi_range[:,None])), complexity_pi).numpy().astype(float)
    d_range = generate_ds(list(torch.FloatTensor(psi_range[:,None])), complexity_d).numpy().astype(float) if data_has_memory else np.zeros(num_grid)
    A_range = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range] for psi1 in psi_range]) if data_has_affinity else np.zeros((num_grid,num_grid))

    # d_range = d_range * w(2)

    # Compute scaling factor
    # psi_mid = .5 * (psi_min + psi_max)
    # A_mid = np.array([simulate_a(psi,psi_mid,affinity_type) for psi in psi_range]) if data_has_affinity else np.zeros(num_grid)
    # # l_mid = pi_range + d_range * w(2) + A_mid
    # l_mid = pi_range + d_range + A_mid
    # scale = l_mid.max().item()
    scale = (pi_range + d_range * w(2)).max()

    # Rescale pi, d, and A
    pi_range = pi_range / scale
    d_range = d_range / scale
    # A_range /= scale

    # Compute likelihoods l
    # # l_range = pi_range + d_range * w(2) + A_range
    # l_range = pi_range + d_range + A_range
    l_range = pi_range + d_range * w(2)

    fig,axes = plt.subplots(1,2,figsize=(2*fw,fl)); _ = [a.grid(1) for a in axes]
    ax = axes[0]
    ax.plot(psi_range, pi_range, c=clrs['pi'], **line_args)
    ax.set_xlabel(r"$\psi$"); ax.set_ylabel(r"$\pi$"); ax.set_title(r"Synthetic $\pi$")
    ax = axes[1]
    ax.plot(psi_range, d_range, c=clrs['d'], **line_args)
    ax.set_xlabel(r"$\psi$"); ax.set_ylabel(r"$d$"); ax.set_title(r"Synthetic $d$")
    fig.tight_layout()

    # fig,axes = plt.subplots(1,2,figsize=(2.2*fw,fl)); ims = [None,None]
    # ax = axes[0]
    # ims[0] = ax.imshow(A_range.T, cmap=cmaps['A'], origin='lower')
    # ax.set_xticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    # ax.set_yticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1))
    # ax.set_xlabel(r"$\psi$ (current)"); ax.set_ylabel(r"$\psi$ (previous)"); ax.set_title(r"Synthetic $A$")
    # ax = axes[1]
    # ims[1] = ax.imshow(l_range.T, cmap=cmaps['l'], origin='lower')
    # ax.set_xticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
    # ax.set_yticks(np.linspace(0,num_grid-1,10),np.round(np.linspace(psi_min,psi_max,10),1))
    # ax.set_xlabel(r"$\psi$ (current)"); ax.set_ylabel(r"$\psi$ (previous)"); ax.set_title(r"Synthetic $\ell$")
    # bbox = [ax.get_position() for ax in axes]
    # cax = [fig.add_axes([bbox[j].x1+.01, bbox[j].y0, .02, bbox[j].height]) for j in range(len(axes))]
    # _ = [fig.colorbar(ims[j], cax=cax[j]) for j in range(len(axes))]

    fig,ax = plt.subplots(figsize=(fw,fl)); ax.grid(1)
    # ax.plot(np.arange(50), w(np.arange(50)), '-', c=clrs['w'], linewidth=3)
    # ax.plot(np.arange(50), w(np.arange(50)) / w(0), '-', c=clrs['w'], linewidth=3)
    ax.plot(np.arange(50), w(np.arange(50)+2) / w(2), '-', c=clrs['w'], linewidth=3)
    ax.set_xlabel(r"Number of turns $t$"); ax.set_ylabel(r"Memory function $w(t)$")
    fig.tight_layout()

def plot_synthetic_data_2d(complexity_pi, complexity_d, affinity_type, w=None,
        data_has_memory:bool=True, data_has_affinity:bool=True,
        num_grid:int=30, psi_min:float=.1, psi_max:float=1.,
        surf_plot_type:str="fix_prev"
    ):
    clrs = {"pi":bq['red'], "d":bq['blue'], "A":bq['yellow'], "w":bq["green"], 'l':bq['purple']}
    cmaps ={"pi":"viridis", "d":"cividis", "A":"plasma", "w":"inferno", "l":"magma"} 
    surfs = {"pi":'summer', "d":'winter', "A":'autumn', "w":'plasma', 'l':'spring'}
    line_args = { "linestyle":'-', "linewidth":3, }

    assert surf_plot_type in ['fix_prev','fix_psi1','fix_psi2'], "Invalid option for surface plots. Expecting surf_plot_type to be 'fix_prev', 'fix_psi1', or 'fix_psi2'."

    if w is None:
        w = lambda x: torch.exp(-.5*x) if type(x) is torch.Tensor else (np.exp(-.5*x).item() if np.isscalar(x) else np.exp(-.5*x))

    psi_range = np.linspace(psi_min,psi_max,num_grid)
    psi1_range, psi2_range = np.meshgrid(psi_range, psi_range)
    psi_input = list(torch.FloatTensor(np.array([psi1_range.reshape(-1), psi2_range.reshape(-1)])).T)
    pi_range = generate_pis(psi_input, complexity_pi).reshape(num_grid,num_grid).numpy().astype(float)
    d_range = generate_ds(psi_input, complexity_d).reshape(num_grid,num_grid).numpy().astype(float) if data_has_memory else np.zeros((num_grid,num_grid))

    d_range = d_range / w(0)

    # Compute scaling factor
    psi_mid = np.array([.5 * (psi_min + psi_max)]*2)
    A_mid = np.array([simulate_a(psi,psi_mid,affinity_type) for psi in psi_input]).reshape(num_grid,num_grid) if data_has_affinity else np.zeros((num_grid,num_grid))
    # l_mid = pi_range + d_range * w(2) + A_mid
    l_mid = pi_range + d_range + A_mid
    scale = l_mid.max().item()

    # Rescale pi, d, and A
    pi_range = pi_range / scale
    d_range = d_range / scale

    fig = plt.figure(figsize=(2*fl,fl)); axes = [fig.add_subplot(121,projection='3d'), fig.add_subplot(122,projection='3d')]
    ax = axes[0]
    ax.plot_surface(psi1_range, psi2_range, pi_range, cmap=cmaps['pi'])
    ax.set_xlabel(r"$\psi_1$"); ax.set_ylabel(r"$\psi_2$"); ax.set_zlabel(r"$\pi$"); ax.set_title(r"Synthetic $\pi$")
    ax = axes[1]
    ax.plot_surface(psi1_range, psi2_range, d_range, cmap=cmaps['d'])
    ax.set_xlabel(r"$\psi_1$"); ax.set_ylabel(r"$\psi_2$"); ax.set_zlabel(r"$d$"); ax.set_title(r"Synthetic $d$")
    fig.tight_layout()

    fig,ax = plt.subplots(figsize=(fw,fl)); ax.grid(1)
    # ax.plot(np.arange(50), w(np.arange(50)), '-', c=clrs['w'], linewidth=3)
    ax.plot(np.arange(50), w(np.arange(50)) / w(0), '-', c=clrs['w'], linewidth=3)
    ax.set_xlabel(r"Number of turns $t$"); ax.set_ylabel(r"Memory function $w(t)$")
    fig.tight_layout()

    if surf_plot_type=='fix_psi2':
        As_range = [[None, None], [None, None]]
        ls_range = [[None, None], [None, None]]
        psi_range_min = torch.FloatTensor(np.stack([psi_range, psi_min * np.ones(num_grid)]).T)
        psi_range_max = torch.FloatTensor(np.stack([psi_range, psi_max * np.ones(num_grid)]).T)
        As_range[0][0] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_min] for psi1 in psi_range_min])
        As_range[0][1] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_max] for psi1 in psi_range_min])
        As_range[1][0] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_min] for psi1 in psi_range_max])
        As_range[1][1] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_max] for psi1 in psi_range_max])
        for i in range(2):
            for j in range(2):
                As_range[i][j] /= scale
                ls_range[i][j] = pi_range + d_range * w(2) + As_range[i][j]

        Amin = float(np.min([As_range[i][j].min() for i in range(2) for j in range(2)]))
        Amax = float(np.max([As_range[i][j].max() for i in range(2) for j in range(2)]))
        A_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
        fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ims[i][j] = ax.imshow(As_range[i][j], cmap=cmaps['A'], origin='lower', vmin=Amin, vmax=Amax)
                ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
                ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
                ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_title(r"$A$ for $\psi_1$ given "+rf"$\psi_2 = ({A_fixed[i][j][0]},{A_fixed[i][j][1]})$")
        fig.tight_layout(); fig.subplots_adjust(wspace=.5)
        bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
        cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
        _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

        lmin = float(np.min([ls_range[i][j].min() for i in range(2) for j in range(2)]))
        lmax = float(np.max([ls_range[i][j].max() for i in range(2) for j in range(2)]))
        l_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
        fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ims[i][j] = ax.imshow(ls_range[i][j], cmap=cmaps['l'], origin='lower', vmin=lmin, vmax=lmax)
                ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
                ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
                ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_title(r"$\ell$ for $\psi_1$ given "+rf"$\psi_2 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout(); fig.subplots_adjust(wspace=.5)
        bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
        cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
        _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

        fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ax.plot_surface(psi1_range,psi2_range,As_range[i][j], cmap=cmaps['A'])
                ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_zlabel(r"$A$"); ax.set_title(r"$A$ for $\psi_1$ given "+rf"$\psi_2 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout()

        fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ax.plot_surface(psi1_range,psi2_range,ls_range[i][j], cmap=cmaps['l'])
                ax.set_xlabel(r"$\psi_1$ (current)"); ax.set_ylabel(r"$\psi_1$ (previous)"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"$\ell$ for $\psi_1$ given "+rf"$\psi_2 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout()

    elif surf_plot_type=='fix_psi1':
        As_range = [[None, None], [None, None]]
        ls_range = [[None, None], [None, None]]
        psi_range_min = torch.FloatTensor(np.stack([psi_min * np.ones(num_grid), psi_range]).T)
        psi_range_max = torch.FloatTensor(np.stack([psi_max * np.ones(num_grid), psi_range]).T)
        As_range[0][0] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_min] for psi1 in psi_range_min])
        As_range[0][1] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_max] for psi1 in psi_range_min])
        As_range[1][0] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_min] for psi1 in psi_range_max])
        As_range[1][1] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psi_range_max] for psi1 in psi_range_max])
        for i in range(2):
            for j in range(2):
                As_range[i][j] /= scale
                ls_range[i][j] = pi_range + d_range * w(2) + As_range[i][j]

        Amin = float(np.min([As_range[i][j].min() for i in range(2) for j in range(2)]))
        Amax = float(np.max([As_range[i][j].max() for i in range(2) for j in range(2)]))
        A_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
        fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ims[i][j] = ax.imshow(As_range[i][j], cmap=cmaps['A'], origin='lower', vmin=Amin, vmax=Amax)
                ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
                ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({A_fixed[i][j][0]},{A_fixed[i][j][1]})$")
        fig.tight_layout(); fig.subplots_adjust(wspace=.5)
        bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
        cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
        _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

        lmin = float(np.min([ls_range[i][j].min() for i in range(2) for j in range(2)]))
        lmax = float(np.max([ls_range[i][j].max() for i in range(2) for j in range(2)]))
        l_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
        fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ims[i][j] = ax.imshow(ls_range[i][j], cmap=cmaps['l'], origin='lower', vmin=lmin, vmax=lmax)
                ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
                ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout(); fig.subplots_adjust(wspace=.5)
        bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
        cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
        _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

        fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ax.plot_surface(psi1_range,psi2_range,As_range[i][j], cmap=cmaps['A'])
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$A$"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout()

        fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ax.plot_surface(psi1_range,psi2_range,ls_range[i][j], cmap=cmaps['l'])
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout()

    elif surf_plot_type=='fix_prev':
        As_range = [[None, None], [None, None]]
        ls_range = [[None, None], [None, None]]
        psis_range = [[[torch.FloatTensor([psi_min,psi_min])], [torch.FloatTensor([psi_min,psi_max])]], [[torch.FloatTensor([psi_max,psi_min])], [torch.FloatTensor([psi_max,psi_max])]]]
        As_range[0][0] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psis_range[0][0]] for psi1 in psi_input]).reshape(num_grid,num_grid)
        As_range[0][1] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psis_range[0][1]] for psi1 in psi_input]).reshape(num_grid,num_grid)
        As_range[1][0] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psis_range[1][0]] for psi1 in psi_input]).reshape(num_grid,num_grid)
        As_range[1][1] = np.array([[simulate_a(psi1,psi2,affinity_type) for psi2 in psis_range[1][1]] for psi1 in psi_input]).reshape(num_grid,num_grid)
        for i in range(2):
            for j in range(2):
                As_range[i][j] /= scale
                ls_range[i][j] = pi_range + d_range * w(2) + As_range[i][j]

        Amin = float(np.min([As_range[i][j].min() for i in range(2) for j in range(2)]))
        Amax = float(np.max([As_range[i][j].max() for i in range(2) for j in range(2)]))
        A_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
        fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ims[i][j] = ax.imshow(As_range[i][j], cmap=cmaps['A'], origin='lower', vmin=Amin, vmax=Amax)
                ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
                ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({A_fixed[i][j][0]},{A_fixed[i][j][1]})$")
        fig.tight_layout(); fig.subplots_adjust(wspace=.5)
        bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
        cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
        _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

        lmin = float(np.min([ls_range[i][j].min() for i in range(2) for j in range(2)]))
        lmax = float(np.max([ls_range[i][j].max() for i in range(2) for j in range(2)]))
        l_fixed = [[(psi_min,psi_min), (psi_min,psi_max)],[(psi_max,psi_min), (psi_max,psi_max)]]
        fig = plt.figure(figsize=(2*fl,2*fl)); axes = fig.subplots(2,2); ims = [[None,None],[None,None]]
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ims[i][j] = ax.imshow(ls_range[i][j], cmap=cmaps['l'], origin='lower', vmin=lmin, vmax=lmax)
                ax.set_xticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1)); ax.tick_params('x',rotation=90)
                ax.set_yticks(np.linspace(0,num_grid-1,10), np.round(np.linspace(psi_min,psi_max,10),1))
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout(); fig.subplots_adjust(wspace=.5)
        bbox = [[axes[i][j].get_position() for j in range(2)] for i in range(2)]
        cax = [[fig.add_axes([bbox[i][j].x1+.01, bbox[i][j].y0, .02, bbox[i][j].height]) for j in range(2)] for i in range(2)]
        _ = [[fig.colorbar(ims[i][j], cax=cax[i][j]) for j in range(2)] for i in range(2)]

        fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ax.plot_surface(psi1_range,psi2_range,As_range[i][j], cmap=cmaps['A'])
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$A$"); ax.set_title(r"$A$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout()

        fig,axes = plt.subplots(2,2, subplot_kw={'projection':'3d'}, figsize=(2*fl,2*fl))
        for i in range(2):
            for j in range(2):
                ax = axes[i][j]
                ax.plot_surface(psi1_range,psi2_range,ls_range[i][j], cmap=cmaps['l'])
                ax.set_xlabel(r"$\psi_2$ (current)"); ax.set_ylabel(r"$\psi_2$ (previous)"); ax.set_zlabel(r"$\ell$"); ax.set_title(r"$\ell$ for $\psi_2$ given "+rf"$\psi_1 = ({l_fixed[i][j][0]},{l_fixed[i][j][1]})$")
        fig.tight_layout()

