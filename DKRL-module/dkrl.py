import numpy as np
from tqdm import tqdm

def DKRL_primal(KZ, KX, y, r, penalty, tol, T):
    N = KZ.shape[0]

    Ut = np.random.normal(loc=0, scale=1, size=(N, r))
    Vt = np.random.normal(loc=0, scale=1, size=(N, r))
    Utt = Ut.copy()
    Vtt = Vt.copy()

    for iter in tqdm(range(T)):
        # update U matrix
        VKX = np.repeat(np.dot(Vt.T, KX), repeats=N, axis=0)
        KZZ = np.tile(KZ, (1, r)) # repeat KZ r times, stack as column
        DesignVt = VKX.T * KZZ  # n * (nr)

        hhat = np.dot(KX, Vtt)
        for i in range(r):
            # Compute residuals first
            ghat = np.dot(KZ, Utt)
            DKZ = DesignVt[:, i * N:(i + 1) * N]
            yhat = (ghat * hhat).sum(axis=1)
            y_residual = y - yhat

            # Update Utt[:, i]
            dUtti = np.linalg.solve(np.dot(DKZ.T, DKZ) + penalty * KZ + 1e-2*np.eye(N), np.dot(DKZ.T, y_residual))
            Utt[:, i] = Utt[:, i] + dUtti
        
        # update V matrix
        UKZ = np.repeat(np.dot(Utt.T, KZ), repeats=N, axis=0)
        KXX = np.tile(KX, (1, r)) 
        DesignUt = UKZ.T * KXX  # n * (nr)
        
        ghat = np.dot(KZ, Utt)
        for i in range(r):
            # Compute residuals first
            hhat = np.dot(KX, Vtt)
            yhat = (ghat * hhat).sum(axis=1)
            y_residual = y - yhat
            
            # Update Vtt[:, i]
            DKX = DesignUt[:, i * N:(i + 1) * N]
            dVtti = np.linalg.solve(np.dot(DKX.T, DKX) + penalty * KX + 1e-2*np.eye(N), np.dot(DKX.T, y_residual))
            Vtt[:, i] = Vtt[:, i] + dVtti

        # check stopping criterion
        # print(iter)
        # print(np.linalg.norm(Utt - Ut)/(np.linalg.norm(Ut) + 1e-4))
        if (np.linalg.norm(Utt - Ut)/(np.linalg.norm(Ut) + 1e-4) < tol and np.linalg.norm(Vtt - Vt)/(np.linalg.norm(Vt) + 1e-3) < tol):
            break

        # if not stop, update Ut and Vt
        Ut = Utt.copy()
        Vt = Vtt.copy()
    
    y_pred = (np.dot(KZ, Utt) * np.dot(KX, Vtt)).sum(axis=1)
    return Utt, Vtt, y_pred
    


def DKRL_dual(Z, X, y, r, penalty, tol, T):
    p = Z.shape[1]
    q = X.shape[1]
    N = Z.shape[0]
    
    Ut = np.random.normal(loc=0, scale=1, size=(p, r))
    Vt = np.random.normal(loc=0, scale=1, size=(q, r))
    Utt = Ut.copy()
    Vtt = Vt.copy()

    for iter in tqdm(range(T)):
        # update U matrix    
        hhat = np.dot(X, Vtt)
        for i in range(r):
            # Compute residuals first
            ghat = np.dot(Z, Utt)
            y_pred = (ghat * hhat).sum(axis=1)
            y_residual = y - y_pred
            DZ = Z * hhat[:, i][:, np.newaxis]

            # Update Utt[:, i]
            dUtti = np.linalg.solve(np.dot(DZ.T, DZ) + penalty * np.eye(p), np.dot(DZ.T, y_residual))
            Utt[:, i] = Utt[:, i] + dUtti
        
        # update V matrix
        ghat = np.dot(Z, Utt)
        for i in range(r):
            # Compute residuals first
            hhat = np.dot(X, Vtt)
            y_pred = (ghat * hhat).sum(axis=1)
            y_residual = y - y_pred
            DX = X * ghat[:, i][:, np.newaxis]
            
            # Update Vtt[:, i]
            dVtti = np.linalg.solve(np.dot(DX.T, DX) + penalty * np.eye(q), np.dot(DX.T, y_residual))
            Vtt[:, i] = Vtt[:, i] + dVtti

        # check stopping criterion
        # print(iter)
        # print(np.linalg.norm(Utt - Ut)/(np.linalg.norm(Ut) + 1e-4))
        if (np.linalg.norm(Utt - Ut)/(np.linalg.norm(Ut) + 1e-4) < tol and np.linalg.norm(Vtt - Vt)/(np.linalg.norm(Vt) + 1e-3) < tol):
            break

        # if not stop, update Ut and Vt
        Ut = Utt.copy()
        Vt = Vtt.copy()

    y_pred = (np.dot(Z, Utt) * np.dot(X, Vtt)).sum(axis=1)
    return Utt, Vtt, y_pred


def DKRL_pred_primal(U, V, KZ_pred, KX_pred):
    y_pred = np.dot(KZ_pred.T, U) * np.dot(KX_pred.T, V)
    y_pred = y_pred.sum(axis=1)
    print(y_pred.shape)
    return y_pred


def DKRL_pred_dual(U, V, Z, X):
    y_pred = np.dot(Z, U) * np.dot(X, V)
    y_pred = y_pred.sum(axis=1)
    return y_pred