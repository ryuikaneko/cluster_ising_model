""" 
Example script to read out unitary matrices from parameter files. 

Examples are shown for how the QCNN acts on an 8-qubit system.

"""

import numpy as np
from itertools import product
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import List
import numpy.typing as npt

def get_basis() -> npt.ArrayLike:
    """ Returns the basis for the unitary gates, based on Kronecker product of Pauli matrices. """
    pauli = [np.eye(2), 
         np.array([[0,1],[1,0]]),
         np.array([[0,-1.j],[1.j,0]]),
         np.array([[1,0],[0,-1]])]
    basis = list(product(pauli, pauli))

    # Ignore the first identity -- just adds a complex phase.
    return np.array([np.kron(pair[0], pair[1]) for pair in basis])[1:]

def process_circuit(params_list: npt.ArrayLike, basis: npt.ArrayLike) -> List:
    """ Processes a list of parameter lists, and returns a list of unitary matrices.
    Args:
        params_list (np.array): Each element is a list of parameters.
        basis (np.array): basis for the unitary gates, output of get_basis()
    Returns:
        List: list of unitary matrices
    """
    return [params_to_unitary_gate(params, basis) for params in params_list]

def params_to_unitary_gate(params: npt.ArrayLike, basis: npt.ArrayLike) -> npt.ArrayLike:
    """ Converts a list of parameters to a unitary matrix (circuit gate).
    Args:
        params (np.array): List of parameters
        basis (np.array): basis for the unitary gates, output of get_basis()
    Returns:
        np.array: unitary matrix
    """
    return expm(-0.5j*np.tensordot(params, basis, [0,0])).reshape((2,2,2,2))
def apply_qcnn(circuit_list: List ,psi: npt.ArrayLike) -> npt.ArrayLike:
    """ Applies a circuit to a state psi. 
    Args:
        circuit (List): Gates with shape (Nlayers, Ngates, 2, 2, 2, 2).
        psi (np.array): Input states with shape (2,2,....,2)
    Returns:
        np.array: Output state
    """   
    # Assume the state has only 8 qubits
    L_phys = 8   
    psi_out = psi
    
    # first level
    gate_list = circuit_list[0]
   
    for n,i in enumerate(range(0,L_phys-1,2)):   
        i1 = i ; i2 = i+1
        psi_out = apply_gate(gate_list[0][n], psi_out, i1, i2, L_phys)
    for n,i in enumerate(range(1,L_phys-1,2)):   
        i1 = i ; i2 = i+1
        psi_out = apply_gate(gate_list[1][n], psi_out, i1, i2, L_phys)
    for n,i in enumerate(range(0,L_phys-1,2)):      
        i1 = i ; i2 = i+1
        psi_out = apply_gate(gate_list[2][n], psi_out, i1, i2, L_phys)
     
    # second level
    gate_list = circuit_list[1]
    for n,i in enumerate(range(1,L_phys-1,4)):    
        i1 = i ; i2 = i+2
        psi_out = apply_gate(gate_list[0][n], psi_out, i1, i2, L_phys)
    for n,i in enumerate(range(3,L_phys-1,4)):      
        i1 = i ; i2 = i+2
        psi_out = apply_gate(gate_list[1][n], psi_out, i1, i2, L_phys)
    for n,i in enumerate(range(1,L_phys-1,4)):        
        i1 = i ; i2 = i+2
        psi_out = apply_gate(gate_list[2][n], psi_out, i1, i2, L_phys)
        
    # third level
    gate_list = circuit_list[2]
    i = 3
    i1 = i ; i2 = i+4
    psi_out = apply_gate(gate_list[0][0], psi_out, i1, i2, L_phys)
    
    return psi_out
def params_to_circuit(params: npt.ArrayLike, basis: npt.ArrayLike) -> List:
    layer11 = process_circuit(params[0:4,:], basis)
    layer12 = process_circuit(params[4:7,:], basis) 
    layer13 = process_circuit(params[7:11,:], basis) 
    
    layer21 = process_circuit(params[11:13,:], basis) 
    layer22 = process_circuit(params[13:14,:], basis) 
    layer23 = process_circuit(params[14:16,:], basis) 
    
    layer31 = process_circuit(params[16:17,:], basis) 
    
    return [[layer11,layer12,layer13], [layer21,layer22,layer23], [layer31]]

def apply_gate(gate: npt.ArrayLike, psi: npt.ArrayLike, i1: int, i2: int, L: int) -> npt.ArrayLike:
    order_restore = list(range(2,L))
    order_restore.insert(i1,0)
    order_restore.insert(i2,1)
   
    psi_out = np.tensordot( gate, psi, [[2,3], [i1, i2]] )
    psi_out = np.transpose(psi_out, axes = tuple(order_restore))
    return psi_out  
def trace_out(psi: npt.ArrayLike) -> npt.ArrayLike: 
    Nq_out = 2  # num of qubits for readout
    
    # trace out unmeasured qubits
    traced_ind = [0,1,2,4,5,6]
    output = np.tensordot( psi, psi.conjugate(), [traced_ind, traced_ind] )
  
    return np.diag(output.reshape((2**Nq_out, 2**Nq_out)))  
def apply_gate1_numpy(U: npt.ArrayLike, i1: int ,psi: npt.ArrayLike) -> npt.ArrayLike:
    L = len(psi.shape)
    
    order_restore = list(range(1,L))
    order_restore.insert(i1,0)
    psi_out = np.tensordot( U, psi, [[1], [i1]] )
    psi_out = np.transpose(psi_out, axes = tuple(order_restore))
    return psi_out
def cluster_state(L: int) -> npt.ArrayLike:
    psi = np.ones(2**L)
    psi /= np.sum(psi)**0.5
    psi = psi.reshape(L*[2])  
    cz = np.diag([1,1,1,-1]).reshape((2,2,2,2))
    z = np.diag([1,-1])    
    for n,i in enumerate(range(0,L-1,2)):     
        psi = apply_gate(cz, psi, i, i+1, L)
    for n,i in enumerate(range(1,L-1,2)):    
        psi = apply_gate(cz, psi, i, i+1, L)    
    for i in range(L):
        psi = apply_gate1_numpy(z, i, psi)
    
    return psi
if __name__ == "__main__":
#    path = 'c:/users/user/desktop/QCNN_data/'
    path = './'
    params = np.load(path + "params_TR_8q.npy")
    basis = get_basis()
    
    # Get the unitary matrices for each layer, 15 parameters is a 4x4 unitary matrix.
    circuits = params_to_circuit(params, basis)    
    Nqubits = 8
    
    
    ''' Note the prediction yields probability array (SB, Trivial, SPT, Unsuccessful)'''
    psi_input = np.zeros(2**Nqubits)
    psi_input[0] = 1
    psi_input = psi_input/np.sum(psi_input**2)**0.5
    psi_input = psi_input.reshape(Nqubits*[2]) 
    psi_readout = apply_qcnn(circuits, psi_input)
    pred = trace_out(psi_readout)
    print('For a product state |00...0>, the prediction is')
    print(f'SB: {pred[0].real*100:.2f}% | Trivial: {pred[1].real*100:.2f}% | SPT: {pred[2].real*100:.2f}% | Unsuccessful: {pred[3].real*100:.2f}%')
    print('-------------------------')
    
    psi_input = np.zeros(2**Nqubits)
    psi_input[-1] = 1
    psi_input = psi_input/np.sum(psi_input**2)**0.5
    psi_input = psi_input.reshape(Nqubits*[2]) 
    psi_readout = apply_qcnn(circuits, psi_input)
    pred = trace_out(psi_readout)
    print('For a product state |11...1>, the prediction is')
    print(f'SB: {pred[0].real*100:.2f}% | Trivial: {pred[1].real*100:.2f}% | SPT: {pred[2].real*100:.2f}% | Unsuccessful: {pred[3].real*100:.2f}%')
    print('-------------------------')
        
    psi_input = np.ones(2**Nqubits)
    psi_input = psi_input/np.sum(psi_input**2)**0.5
    psi_input = psi_input.reshape(Nqubits*[2])
    psi_readout = apply_qcnn(circuits, psi_input)
    pred = trace_out(psi_readout)
    print('For a product state |++...+>, the prediction is')
    print(f'SB: {pred[0].real*100:.2f}% | Trivial: {pred[1].real*100:.2f}% | SPT: {pred[2].real*100:.2f}% | Unsuccessful: {pred[3].real*100:.2f}%')
    print('-------------------------')
    
    y = np.array([[1,1j], [-1j,1]])*1/2**0.5
    psi_input = np.zeros(2**Nqubits)
    psi_input[0] = 1
    psi_input = psi_input/np.sum(psi_input**2)**0.5
    psi_input = psi_input.reshape(Nqubits*[2]) 
    for i in range(Nqubits):
        psi_input = apply_gate1_numpy(y, i, psi_input)
    psi_readout = apply_qcnn(circuits, psi_input)
    pred = trace_out(psi_readout)
    print('For a product state |+i, +i...,+i>, the prediction is')
    print(f'SB: {pred[0].real*100:.2f}% | Trivial: {pred[1].real*100:.2f}% | SPT: {pred[2].real*100:.2f}% | Unsuccessful: {pred[3].real*100:.2f}%')
    print('-------------------------')
    
    psi_readout = apply_qcnn(circuits, cluster_state(Nqubits))
    pred = trace_out(psi_readout)
    print('For an 8-qubit cluster state |CS>, the prediction is')
    print(f'SB: {pred[0].real*100:.2f}% | Trivial: {pred[1].real*100:.2f}% | SPT: {pred[2].real*100:.2f}% | Unsuccessful: {pred[3].real*100:.2f}%')
    print('-------------------------')
   










