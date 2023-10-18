

def DataLoader(x, y, batch_size, verbose=False):
    for i in range(0, x.shape[0], batch_size):
        if i + batch_size < x.shape[0]:
            yield x[i: i + batch_size, :, :, :], y[i: i + batch_size, :]
        else:
            if verbose:
                print('Batches are over')
