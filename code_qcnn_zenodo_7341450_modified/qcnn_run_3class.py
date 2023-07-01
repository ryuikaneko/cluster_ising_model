import jax
from progress.bar import Bar
import optax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from qcnn_func_3class import get_basis, loss, generate_data, process_state,accuracy
import os



def run_qcnn(**kwargs):
    N_data_train = 60000
    N_data_test = 1000
    batch_size = 50
    n_mask_layer = kwargs['n_mask_layer']
    theta_max = np.pi*kwargs['theta_max']

    train_set = generate_data(theta_max, N_data_train, n_mask_layer)
    test_eval_set = generate_data(theta_max, N_data_test, n_mask_layer)


    train = iter(tfds.as_numpy(tf.data.Dataset.from_tensor_slices( train_set ).take(-1).cache().repeat().shuffle(50*batch_size, seed=0).batch(batch_size)))
    test_eval = iter(tfds.as_numpy(tf.data.Dataset.from_tensor_slices(  test_eval_set ).take(-1).cache().repeat().batch(N_data_test)))
    train_eval = iter(tfds.as_numpy(tf.data.Dataset.from_tensor_slices(  train_set ).take(-1).cache().repeat().batch(1000)))


    grad_rate = 5e-4
    opt = optax.adam(grad_rate)
    basis = get_basis()

    def update(params, opt_state, batch, n_mask_layer):
        grads = jax.grad(loss)(params, batch, basis, n_mask_layer)
        updates, opt_state = opt.update(grads, opt_state)
        tn = optax.apply_updates(params, updates)
        return tn, opt_state



#    Nepochs = 12000
    Nepochs = 2

    seed = 42
    key = jax.random.PRNGKey(seed) 
    key, subkey = jax.random.split(key)
    params = jax.random.uniform(subkey, shape=(17,15))
    losses = []
    acc = []




    process = lambda x: process_state(x)
    opt_state = opt.init(params)
    max_acc = 0

    for epoch in range(Nepochs):
        if epoch < len(acc):
            continue
        bar = Bar(f"[Epoch {epoch+1}/{Nepochs}]", max=1200) # rough batches for (0,1,2,3)
        batch = next(train)
        for m in range(1200):
            batch = process(batch)
            params, opt_state = update(params, opt_state, batch, n_mask_layer)
            batch = next(train)
            bar.next()
          
        bar.finish()
        
        test_accuracy = accuracy(params, process(next(test_eval)) , basis, n_mask_layer)
        train_accuracy = accuracy(params, process(next(train_eval)) , basis, n_mask_layer)
        train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
        
        if test_accuracy >= max_acc:
            max_acc = test_accuracy
            max_trainacc = train_accuracy
        
        print('==== Test for QCNN (8q) =====')
        print('8q symmetry-breaking')
        print(f"Train/Test accuracy for {n_mask_layer}: "
                f"{train_accuracy:.4f}/{test_accuracy:.4f}.")
        print(f'The max test accuracy is {max_trainacc:.4f}/{max_acc:.4f}')
        
        losses.append(loss(params, process(next(test_eval)), basis, n_mask_layer))
        acc.append([train_accuracy, test_accuracy])
        print(f'The losses are {losses[-1]}')
        
        
        savepath = "results_3class/"
        if not os.path.isdir(savepath):
            os.makedirs(savepath)


        np.save(f"results_3class/losses_{n_mask_layer}_layers_c50_{kwargs['theta_max']}.npy", [range(Nepochs), losses])
        np.save(f"results_3class/params_{n_mask_layer}_layers_c50_{kwargs['theta_max']}.npy", params)
        np.save(f"results_3class/train_test_acc_{n_mask_layer}_layers_c50_{kwargs['theta_max']}.npy",np.array(acc))

      
    print(f'layer {n_mask_layer} is finished (bottom)')
    
if __name__ == "__main__":
#    import cluster_jobs
#    cluster_jobs.run_simulation_commandline(globals())
    run_qcnn(n_mask_layer=1,theta_max=1.0)
