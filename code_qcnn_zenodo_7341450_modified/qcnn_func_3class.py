'''
simulate a QCNN consisting of two qubit gates
'''



import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
from itertools import product
from jax.scipy.linalg import expm
import scipy.linalg as spl

def apply_gate1_numpy(U, i1, psi):
    L = len(psi.shape)
    
    order_restore = list(range(1,L))
    order_restore.insert(i1,0)
    psi_out = np.tensordot( U, psi, [[1], [i1]] )
    psi_out = np.transpose(psi_out, axes = tuple(order_restore))
    return psi_out

def fixed_point_wavefunc(L, phase):
    # ferromagnetic
    if phase == 0:
        wave_func = np.zeros(2**L)
        wave_func[0] = 1
        wave_func[-1] = 1
        wave_func /= np.sum(wave_func)**0.5
        return wave_func.reshape(L*[2])
    # paramaggnetic
    elif phase == 1:
        wave_func = np.ones(2**L)
        wave_func /= np.sum(wave_func)**0.5
        return wave_func.reshape(L*[2])
    elif phase == 2:
        wave_func = cluster_state(L)
        return wave_func
def cluster_state(L):
    psi = np.ones(2**L)
    psi /= np.sum(psi)**0.5
    psi = psi.reshape(L*[2])
    
    cz = np.diag([1,1,1,-1]).reshape((2,2,2,2))
    z = np.diag([1,-1])
    
    for n,i in enumerate(range(0,L-1,2)):
     
        order_restore = list(range(2,L))
        order_restore.insert(i,0)
        order_restore.insert(i+1,1)
          
       
        psi = np.tensordot( cz, psi, [[2,3], [i, i+1]] )
        psi = np.transpose(psi, axes = tuple(order_restore))
    for n,i in enumerate(range(1,L-1,2)):
     
        order_restore = list(range(2,L))
        order_restore.insert(i,0)
        order_restore.insert(i+1,1)
        
       
        psi = np.tensordot( cz, psi, [[2,3], [i, i+1]] )
        psi = np.transpose(psi, axes = tuple(order_restore))
    for i in range(L):
        psi = apply_gate1_numpy(z, i, psi)
    return psi        
def apply_gate3_numpy(U, i1, i2,i3, psi):
    ''' apply a two-site matrix (i1',i2', i1,i2) to the state'''
    L = len(psi.shape)
    
    order_restore = list(range(3,L))
    order_restore.insert(i1,0)
    order_restore.insert(i2,1)
    order_restore.insert(i3,2)
    
    psi_out = np.tensordot( U, psi, [[3,4,5], [i1, i2,i3]] )
    psi_out = np.transpose(psi_out, axes = tuple(order_restore))
    return psi_out       


def symmetric_gate(theta_max):
    ''' 
    generate a symmetric gate exp(i*theta_1*IX + i*theta_2*XI + ....), 
    with theta uniformly in [-theta_max, theta_max)
    '''
    pauli = [np.eye(2), 
          np.array([[0,1],[1,0]]),
          np.array([[0,-1.j],[1.j,0]]),
          np.array([[1,0],[0,-1]])]
    basis = list(product(pauli, pauli))
    basis = np.array([np.kron(pair[0], pair[1]) for pair in basis])
    
    # select Pauli strings symmetric under XX and complex conjugation
    symmetric_pauli = [3,7,11,12,13,14]
    symmetric_basis = np.array([basis[i] for i in symmetric_pauli])
    
    theta = theta_max*(2*np.random.rand(len(symmetric_pauli))-1)
    return spl.expm(-1j*np.tensordot(theta, symmetric_basis, [0,0])).reshape((2,2,2,2))
    
def symmetric_layer(theta_max, N_gates):
    ''' generate a list of N_gates symmetric gates'''
    return [symmetric_gate(theta_max) for i in range(N_gates)]
    
def apply_symmetric_layers(theta_max,psi, n_mask_layer, pairing):
    L = len(psi.shape)
    psi_out = psi
    for n in range(n_mask_layer):
        off_set = (n+pairing) % 2
        gate_list = symmetric_layer(theta_max, N_gates = 1 )
        for m,i in enumerate(range(off_set,L-1,2)):
            psi_out = apply_gate_numpy(gate_list[0], i, i+1, psi_out)       
    return psi_out
    
def apply_gate_numpy(U, i1, i2, psi):
    ''' apply a two-site matrix (i1',i2', i1,i2) to the state'''
    if i1 >= i2: # for simplicity, we stick to this convention.
        raise ValueError
    L = len(psi.shape)
    
    order_restore = list(range(2,L))
    order_restore.insert(i1,0)
    order_restore.insert(i2,1)
    psi_out = np.tensordot( U, psi, [[2,3], [i1, i2]] )
    psi_out = np.transpose(psi_out, axes = tuple(order_restore))
    return psi_out
     
    
def generate_data(theta_max, N_data, n_mask_layer):
    L = 8 + 2*(n_mask_layer+1)
    data = []
    labels = []
    phases = [0,1,2]
    
    fixed_points = [ fixed_point_wavefunc(L, phases[0]), 
                    fixed_point_wavefunc(L, phases[1]),
                    fixed_point_wavefunc(L, phases[2])]
    
    for n in range(N_data):
        label = np.random.choice(phases)
        pairing = np.random.randint(2)
        #pairing = 0
        psi = apply_symmetric_layers(theta_max, fixed_points[label], n_mask_layer, pairing)
        data.append(psi)
        labels.append(label)
    
    return {'state':np.array(data), 'label':np.array(labels)}
  



def process_state(batch):
   
    states = batch['state'].astype(jnp.complex128)

    return {
        "state": states,
        "label": batch["label"],
    }



@jit
def apply_qcnn(circuit_list, psi):
    ''' apply the qcnn classifier to the state L = 8'''
    
    # the additional qubits is to make sure the random unitary lightcone is not affected by BC
    L_phys = 8
    L = len(psi.shape)
    n_mask_layer = int((L-L_phys)/2-1)
  
    psi_out = psi
    
    # first level
    gate_list = circuit_list[0]
   
    for n,i in enumerate(range(0,L_phys-1,2)):
     
        ind = n_mask_layer +1 + i
        i1 = ind ; i2 = ind+1
        order_restore = list(range(2,L))
        order_restore.insert(i1,0)
        order_restore.insert(i2,1)
       
        psi_out = jnp.tensordot( gate_list[0][n], psi_out, [[2,3], [i1, i2]] )
        psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
    for n,i in enumerate(range(1,L_phys-1,2)):
        ind = n_mask_layer +1 + i
        i1 = ind ; i2 = ind+1
        order_restore = list(range(2,L))
        order_restore.insert(i1,0)
        order_restore.insert(i2,1)
        psi_out = jnp.tensordot( gate_list[1][n], psi_out, [[2,3], [i1, i2]] )
        psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
    for n,i in enumerate(range(0,L_phys-1,2)):
        ind = n_mask_layer +1 + i
        i1 = ind ; i2 = ind+1
        order_restore = list(range(2,L))
        order_restore.insert(i1,0)
        order_restore.insert(i2,1)
        psi_out = jnp.tensordot( gate_list[2][n], psi_out, [[2,3], [i1, i2]] )
        psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
   
    
    # second level
    gate_list = circuit_list[1]
    for n,i in enumerate(range(1,L_phys-1,4)):
        ind = n_mask_layer +1 + i
        i1 = ind ; i2 = ind+2
        order_restore = list(range(2,L))
        order_restore.insert(i1,0)
        order_restore.insert(i2,1)
        psi_out = jnp.tensordot( gate_list[0][n], psi_out, [[2,3], [i1, i2]] )
        psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
    for n,i in enumerate(range(3,L_phys-1,4)):
        ind = n_mask_layer +1 + i
        i1 = ind ; i2 = ind+2
        order_restore = list(range(2,L))
        order_restore.insert(i1,0)
        order_restore.insert(i2,1)
        psi_out = jnp.tensordot( gate_list[1][n], psi_out, [[2,3], [i1, i2]] )
        psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
    for n,i in enumerate(range(1,L_phys-1,4)):
        ind = n_mask_layer +1 + i
        i1 = ind ; i2 = ind+2
        order_restore = list(range(2,L))
        order_restore.insert(i1,0)
        order_restore.insert(i2,1)
        psi_out = jnp.tensordot( gate_list[2][n], psi_out, [[2,3], [i1, i2]] )
        psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
    
    
    # third level
    gate_list = circuit_list[2]
    i = 3
    ind = n_mask_layer +1 + i
    i1 = ind ; i2 = ind+4
    order_restore = list(range(2,L))
    order_restore.insert(i1,0)
    order_restore.insert(i2,1)
    psi_out = jnp.tensordot( gate_list[0][0], psi_out, [[2,3], [i1, i2]] )
    psi_out = jnp.transpose(psi_out, axes = tuple(order_restore))
    
    return psi_out
    
    
    

@jit
def trace_out(psi): 
    L_phys = 8
    L = len(psi.shape)
    n_mask_layer = int((L-L_phys)/2-1)
    
    Nq_out = 2  # num of qubits for readout, need to customize
    
    # trace out unmeasured qubits
    traced_ind = list(range(0, L-(n_mask_layer+1)-Nq_out -3)) +list(range(L-(n_mask_layer+1)-Nq_out -2, L-(n_mask_layer+1)-1))+ list(range(L-(n_mask_layer+1),L))
    output = jnp.tensordot( psi, psi.conjugate(), [traced_ind, traced_ind] )
  
    return jnp.diag(output.reshape((2**Nq_out, 2**Nq_out)))
   
    
@jit
def get_basis():
    pauli = [jnp.eye(2), 
          jnp.array([[0,1],[1,0]]),
          jnp.array([[0,-1.j],[1.j,0]]),
          jnp.array([[1,0],[0,-1]])]
    basis = list(product(pauli, pauli))
    # get rid of the complex phase
    return jnp.array([jnp.kron(pair[0], pair[1]) for pair in basis])[1:]

@jit
def process_circuit(params_list, basis):
    return [params_to_unitary_gate(params, basis) for params in params_list]

@jit
def params_to_unitary_gate(params, basis):
    return expm(-0.5j*jnp.tensordot(params, basis, [0,0])).reshape((2,2,2,2))

    
@jit
def evaluate(circuit, psi):
    psi_out = apply_qcnn(circuit, psi)
    pred = trace_out(psi_out)
    return jnp.abs(pred)*50

evaluate_batched = jax.jit(jax.vmap(evaluate, in_axes=(None, 0), out_axes=0))

@jit 
def params_to_circuit(params, basis):
    layer11 = process_circuit(params[0:4,:], basis)
    layer12 = process_circuit(params[4:7,:], basis) 
    layer13 = process_circuit(params[7:11,:], basis) 
    
    layer21 = process_circuit(params[11:13,:], basis) 
    layer22 = process_circuit(params[13:14,:], basis) 
    layer23 = process_circuit(params[14:16,:], basis) 
    
    layer31 = process_circuit(params[16:17,:], basis) 
    
    return [[layer11,layer12,layer13], [layer21,layer22,layer23], [layer31]]
    
    

@jit
def get_sum(v):
    return jnp.sum(v)

def loss(params, batch, basis, n_mask_layer):
    num_class = 4
     
    circuit = params_to_circuit(params, basis)
    
    logits = evaluate_batched(circuit, batch["state"])
   
    labels = jax.nn.one_hot(batch["label"], num_class)
    
    lsq =  -get_sum(labels * jax.nn.log_softmax(logits))
    lsq /= labels.shape[0]
    

    return lsq

  

def accuracy(params, batch, basis, n_mask_layer):
    circuit = params_to_circuit(params, basis)
    predictions = evaluate_batched(circuit, batch["state"])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])










