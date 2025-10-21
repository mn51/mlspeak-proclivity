import numpy as np
import torch
import networkx as nx
from plttools import *

def compute_betas(w,time_since):
    beta = torch.zeros_like(time_since)
    for s in range(time_since.shape[1]):
        if torch.sum(~time_since[:,s].isinf())==1:
            beta[~time_since[:,s].isinf(),s] = w(time_since[:,s][~time_since[:,s].isinf()].unsqueeze(1))
        elif torch.sum(~time_since[:,s].isinf())>1:
            beta[~time_since[:,s].isinf(),s] = w(time_since[:,s][~time_since[:,s].isinf()].unsqueeze(1)).squeeze()
    return beta

# Compute loss of given conversation for given pi and d
def negloglik(speakers_onehot,pi,d=None,A=None,beta=None,device=None):
    assert all([pi[s]>=0 for s in range(len(pi))]), "Invalid pi. All values must be nonnegative."
    assert (d is None) or (len(pi)==len(d)), "Inconsistent number of speakers for pi and d."
    assert (d is None) or all([d[s]>=0 for s in range(len(d))]), "Invalid d. All values must be nonnegative."
    assert len(pi)==speakers_onehot.shape[1], "Inconsistent number of speakers between pi and conversation."
    assert (d is None) or (len(d)==speakers_onehot.shape[1]), "Inconsistent number of speakers between d and conversation."
    assert (A is None) or (A.ndim==2 and A.shape[0]==A.shape[1] and A.shape[0]==len(pi)), "Inconsistent number of speakers for A."
    assert (A is None) or all([A[s1,s2]>=0 for s1 in range(A.shape[0]) for s2 in range(A.shape[1])]), "Invalid A. All values must be nonnegative."
    assert (d is None) or (beta is None) or speakers_onehot.shape==beta.shape, "Inconsistent number of turns or speakers between speakers_onehot and beta."
    
    if device is not None:
        speakers_onehot = speakers_onehot.to(device)

    # Convert one-hot conversation to array of speakers IDs
    num_time = speakers_onehot.shape[0]
    convo = speakers_onehot_to_id(speakers_onehot)
    time_since = compute_time_since(speakers_onehot) - 2
    if d is None:
        d = torch.zeros_like(pi)
    if A is None:
        A = torch.zeros((len(pi),len(pi)))
    if beta is None:
        beta = torch.exp(-.5 * time_since)

    if device is not None:
        convo = convo.to(device)
        time_since = time_since.to(device)
        pi = pi.to(device)
        d = d.to(device)
        A = A.to(device)
        beta = beta.to(device)

    Affinity = torch.zeros_like(time_since)
    Affinity[1:] = A[:,convo[:-1]].T
    L = (pi[None] + d[None]*beta + Affinity) * (time_since>=0)
    assert all(L.sum(1)>0), "Invalid loss computed. For at least one turn, all speakers have likelihood of zero."

    l_ratio = L[torch.arange(len(convo)),convo] / L.sum(1)
    assert torch.prod(l_ratio>0)>0, "Invalid loss computed. For at least one turn, loss value is not positive. This can happen if one speaker speaks twice in a row."

    nll_ratio = -torch.log(l_ratio)
    return nll_ratio

# Convert symmetric matrix with zero diagonal to lower triangular vector
def mat2lowtri(A):
    assert (A.ndim==2) and (A.shape[0]==A.shape[1]), "Invalid input matrix shape. Must be square matrix."
    if A.diagonal().any():
        print(f"Warning: Diagonal entries are not empty and will be discarded.")
    
    (N,N) = A.shape
    cols,rows = np.triu_indices(N,1)
    return A[rows,cols]

# Convert lower triangular vector to symmetric matrix with zero diagonal
def lowtri2mat(a):
    assert a.ndim==1, "Invalid input vector shape. Must be a 1-D array."

    N = (2*len(a)+.25)**.5 + .5
    assert not N%1, "Invalid input vector length. Must be N(N-1)/2 for a valid square matrix of size N-by-N."
    N = int(N)

    cols,rows = np.triu_indices(N,1)
    A = torch.zeros((N,N),dtype=a.dtype)
    A[rows,cols] = a.clone()
    A = A + A.T

    return A

# Convert speakers_id to speakers_onehot
def speakers_id_to_onehot(speakers_id, num_speakers:int=None):
    assert speakers_id.ndim==1 and (speakers_id.dtype==int or speakers_id.dtype==torch.int or speakers_id.dtype==torch.int64), "Invalid conversation. Must be a 1-D array of integers."
    num_speakers = num_speakers if num_speakers is not None else speakers_id.max()+1

    Eye = torch.eye(num_speakers).to(speakers_id.device)

    return Eye[speakers_id]
    # return torch.eye(num_speakers)[speakers_id]

# Convert speakers_onehot to speakers_id
def speakers_onehot_to_id(speakers_onehot):
    assert ((speakers_onehot==0) + (speakers_onehot==1)).all() and (speakers_onehot.sum(1)==1).all(), "Invalid input conversation. Each row must be one-hot vector indicating speaker."
    return speakers_onehot.argmax(1)

# Compute number of turns since spoken for each speaker for given conversation
def compute_time_since(speakers_onehot):
    num_time,num_speakers = speakers_onehot.shape
    speakers_id = speakers_onehot_to_id(speakers_onehot)
    speakers = torch.unique(speakers_id)
    time_since = torch.inf * torch.ones((num_time,num_speakers))
    end_time = torch.tensor([num_time-1]).to(speakers_id.device)
    for s in speakers:
        # inds = torch.unique(torch.cat([torch.where(speakers_id==s)[0],torch.tensor([num_time-1])]))
        inds = torch.unique(torch.cat([torch.where(speakers_id==s)[0],end_time]))
        inds_diff = torch.diff(inds)
        for i in range(len(inds_diff)):
            time_since[inds[i]+1:inds[i+1]+1,s] = torch.arange(inds_diff[i])+1
    time_since = time_since.to(speakers_onehot.device)
    return time_since

# Compute speaking proportions per speaker given conversation
def compute_speaking_proportions(speakers_onehot):
    num_time,num_speakers = speakers_onehot.shape
    speakers_id = speakers_onehot_to_id(speakers_onehot)
    return (speakers_id[:,None]==np.arange(num_speakers)[None]).sum(0) / num_time

# Compute ABA turn proportions per speaker given conversation
def compute_aba_turn_proportions(speakers_onehot):
    num_time,num_speakers = speakers_onehot.shape
    speakers_id = speakers_onehot_to_id(speakers_onehot)

    floor_idx = np.where(speakers_id[2:]==speakers_id[:-2])[0] + 2
    floor_counts = (speakers_id[floor_idx][:,None]==np.arange(num_speakers)[None]).sum(0)
    speaker_counts = speakers_onehot.sum(0)
    return floor_counts/speaker_counts

# Compute ABAB turn proportions per pair of speakers given conversation
def compute_abab_turn_proportions(speakers_onehot):
    speakers_id = speakers_onehot.argmax(1)
    (num_time,num_speakers) = speakers_onehot.shape
    dyad_turns = torch.zeros((num_speakers,num_speakers))

    i1=0
    i2=1
    # For each pair of speakers
    for i1 in range(num_speakers):
        for i2 in range(i1+1,num_speakers):
            # Count number of ABAB or more turns
            dyad_turns[i1,i2] = 0

            # Create new speaker array, where A or B turns are 1, and other speaker's turns are 0
            speakers_id_dyad = speakers_id.clone()
            speakers_id_dyad[(speakers_id_dyad!=i1)&(speakers_id_dyad!=i2)] = -1
            speakers_id_dyad[speakers_id_dyad>=0] = 1
            speakers_id_dyad[speakers_id_dyad<0]  = 0

            # Loop over conversation until all ABAB+ strings are gone
            while torch.sum(speakers_id_dyad)>=4 and len(speakers_id_dyad)>0:
                # Find first turn of A or B
                start_ind = np.where(speakers_id_dyad)[0][0]

                # Trim all turns before
                speakers_id_dyad = speakers_id_dyad[start_ind:]

                # Find end of sequence of ABAB turns
                end_ind = len(speakers_id_dyad) if (speakers_id_dyad==1).all() else torch.where(speakers_id_dyad==0)[0][0].item()

                # Check if ABAB sequence is at least four turns
                if end_ind>=4:
                    # Add all turns in sequence to total if at least four
                    dyad_turns[i1,i2] += end_ind
                
                # Trim current sequence
                speakers_id_dyad = speakers_id_dyad[end_ind:]
            
            # Divide ABAB turn sum by total number of turns
            dyad_turns[i1,i2] /= num_time
            dyad_turns[i2,i1] = dyad_turns[i1,i2]
    return dyad_turns

# Classify each turn in given conversation
def classify_turns(speakers_id):
    assert speakers_id.ndim==1 and (speakers_id.dtype in [int,torch.int,torch.int64]), "Invalid conversation. Must be a 1-D array of integers."

    floor_idx = torch.where(speakers_id[2:]==speakers_id[:-2])[0] + 2
    broken_floor_idx = torch.where((speakers_id[4-3:-3]==speakers_id[4-1:-1]) * (speakers_id[4-2:-2]!=speakers_id[4:]))[0] + 4
    new_floor_idx = torch.where((speakers_id[5:]==speakers_id[5-2:-2]) * (speakers_id[5:]==speakers_id[5-4:-4]) * (speakers_id[5-1:-1]!=speakers_id[5-3:-3]))[0] + 5
    regain_idx = torch.where((speakers_id[5-4:-4]==speakers_id[5-2:-2]) * (speakers_id[5-3:-3]!=speakers_id[5-1:-1]) * (speakers_id[5-3:-3]==speakers_id[5:]))[0] + 5

    floor_indicator = torch.zeros(len(speakers_id),dtype=int)
    floor_indicator[floor_idx] = 1
    broken_floor_indicator = torch.zeros(len(speakers_id),dtype=int)
    broken_floor_indicator[broken_floor_idx] = 1
    new_floor_indicator = torch.zeros(len(speakers_id),dtype=int)
    new_floor_indicator[new_floor_idx] = 1
    regain_indicator = torch.zeros(len(speakers_id),dtype=int)
    regain_indicator[regain_idx] = 1

    assert (floor_indicator + broken_floor_indicator + regain_indicator).max() <= 1, 'Invalid classification. Floor, broken floor, and regain turns should not overlap.'
    nonfloor_indicator = 1 - (floor_indicator + broken_floor_indicator + regain_indicator)

    convo_class = {
        'floor': floor_indicator,
        'broken_floor': broken_floor_indicator,
        'new_floor': new_floor_indicator,
        'regain': regain_indicator,
        'nonfloor': nonfloor_indicator
    }

    return convo_class

# Compute conversation network from given conversation
def convonetwork_from_speakers_id(speakers_id,num_speakers:int=None):
    assert speakers_id.ndim==1 and speakers_id.dtype in [int,torch.int,torch.int64], "Invalid conversation. Must be a 1-D array of integers."
    num_speakers = num_speakers if num_speakers is not None else speakers_id.max()+1

    if num_speakers is None:
        spkrs = torch.unique(speakers_id)
        num_speakers = len(spkrs)
    else:
        spkrs = torch.arange(num_speakers)
    # num_speakers = num_speakers if num_speakers is not None else len(np.unique(speakers_id))
    # spkrs = np.unique(speakers_id)
    A = torch.zeros((num_speakers,num_speakers))
    for i1 in range(A.shape[0]):
        for i2 in np.delete(np.arange(A.shape[1]),i1):
            A[i1,i2] = torch.sum((speakers_id[:-1]==spkrs[i1])*(speakers_id[1:]==spkrs[i2]))
    A = A/( torch.sum(A) + int(torch.sum(A)==0) )
    return A

# Plot conversation network computed from convonetwork_from_speakers_id
def plot_convonetwork(A):
    G = nx.DiGraph(A.numpy())
    pos = nx.circular_layout(G)
    ew = []
    for e in G.edges:
        ew.append( G.edges[e]['weight'] )
    ew = np.array(ew)/np.sum(ew)*40

    eg_args = {
        'arrows':True,
        'arrowstyle':'<|-',
        'arrowsize':25,
        'connectionstyle':'arc3,rad=0.2',
        'width':ew
    }
    nd_args = {
        'node_color':'c',
        'edgecolors':'k'
    }

    fig = plt.figure(figsize=(4,4))
    ax = fig.subplots()
    nx.draw_networkx_nodes(G,pos=pos,**nd_args)
    nx.draw_networkx_edges(G,pos=pos,**eg_args)
    nx.draw_networkx_labels(G,pos=pos)
    fig.tight_layout()
    return
