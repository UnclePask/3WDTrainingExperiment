'''
Created on Nov 8, 2025

@author: pasquale
'''
import numpy as np
#import pandas as pd
from sklearn.preprocessing import PowerTransformer
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import wasserstein_distance
#import matplotlib.pyplot as plt
#import staticMethodsCollection as static


class halloweenTrain(object):
    '''
    classdocs
    '''

    @staticmethod
    def entropie(probs):
    #return -sum([p * math.log2(p) for p in probs if p > 0])
        probs = np.clip(probs, 1e-9, 1) #<- vettoriale e non scalare
        return -np.sum(probs * np.log2(probs), axis=1) #<- vettoriale e non scalare
    
    @staticmethod
    def margini(probs):
        return np.abs(probs[:,0] - probs[:,1])
    
    @staticmethod
    def integraleDefinitoOf(f):
    
        if len(f) == 0:
            return 0.0
    
        x_vals = np.linspace(0, 1, len(f)) 
        fp = interp1d(x_vals, f, kind='linear', fill_value="extrapolate") 
        Sf, _ = quad(lambda x: float(fp(x)), 0, 1) 
    
        return Sf
    
    @staticmethod
    def compute_tau(margins, entropies, area_m, area_h, eps):
        #m = (margins - margins.min()) / (margins.max() - margins.min() + 1e-12)
        #h = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-12)
        m = margins
        h = entropies
        # EMD
        delta_emd = wasserstein_distance(m, h)
        
        #correggere la formula per trovare l'area
        # area locale
        A = np.mean(np.abs(m - h))
        # tau
        tau = delta_emd / (A + eps)

        return tau, delta_emd, A
    
    @staticmethod
    def trasformata(dist, distribuzione: str):
    
        epsilon = 1e-6
        dist = np.array(dist)
        mu = np.mean(dist)
        sigma = np.std(dist)
        pt = PowerTransformer(method="yeo-johnson")
        gauss = pt.fit_transform(dist.reshape(-1, 1))
        gauss_min = gauss.min()
        gauss_max = gauss.max()
        gauss_in01 = (gauss - gauss_min) / (gauss_max - gauss_min)
    
        in01 = np.array(gauss_in01)
    
        # vado a generare l'asse x della funzione con flatten di numpy
        asse_x = in01.flatten()
    
        # rimuovo eventuali punti nan(asintoti) come da documentazione scypy
        asse_x = asse_x[np.isfinite(asse_x)]
    
        # creo un istogramma numpy non per disegnare il grafico ma per campionare la funzione che ho fatto .flatten() su x
        counts, _ = np.histogram(asse_x, bins=len(asse_x), density=True)
    
        #tratto la funzione campionata come un segnale oscillatorio e ottengo un vettore di valori crescenti
        f = argrelextrema(counts, np.less)[0]
    
    #diciamo che normalizzo tutto anche se questa formula non è proprio "canonica"
        f = np.array(f)
        if len(f) == 0:
            min_f = np.argmin(counts)
            max_f = np.argmax(counts)
        else:
            min_f = np.argmin(f)
            max_f = np.argmax(f)
        
        f_hallo = (abs(f - min_f) / abs((min_f - max_f) + epsilon)) / abs(max_f+ epsilon)
        f_hallo =np.array(f_hallo)
        mu_f = np.mean(f_hallo)
        sigma_f = np.std(f_hallo)
        #asse_y = np.linspace(0, 1, len(f_hallo))
    
        area_trasformata = halloweenTrain.integraleDefinitoOf(f_hallo)
    
        return mu, sigma, gauss_in01, mu_f, sigma_f, f_hallo, area_trasformata    
    
    @staticmethod
    def get_threshold_margine(step, margine_array):
    
        epsilon = 1e-6
        #mu, sigma, margine_array, _, _, _, _  = halloweenTrain.trasformata(margine_array, "")
        if step == 0:
            beta = np.percentile(margine_array, 50-epsilon)
            alpha = np.percentile(margine_array, 50+epsilon) 
        else:
            alpha_range = np.clip(50 + step + epsilon, 50, 100)
            alpha = np.percentile(margine_array, alpha_range)
            beta_range = np.clip(50 - step - epsilon, 50, 100)
            beta = np.percentile(margine_array, beta_range)
    
        return alpha, beta
    
    @staticmethod
    def get_threshold_entropy(step, entropies_array):
    
        epsilon = 1e-9
        #mu, sigma, entropies_array, _, _, _, _ = halloweenTrain.trasformata(entropies_array, "")
        if step == 0:
            beta = np.percentile(entropies_array, 1)
            alpha = np.percentile(entropies_array, 99)
        else:
            alpha_range = np.clip(99 - step + epsilon, 0, 100)
            alpha = np.percentile(entropies_array, alpha_range)
            beta_range = np.clip(1 + step - epsilon, 0, 100)
            beta = np.percentile(entropies_array, beta_range)
    
        return alpha, beta 
    
    @staticmethod
    def get_bnd(iter, x_train_ds):
        
        alpha_h = 0
        eps = 1e-12
        alpha_m = beta_m = 0
        #mu_h, sigma_h, H, mu_f, sigma_f, f_hallo, area_H = halloweenTrain.trasformata(x_train_ds["entropy"], "Entropia")
        #mu_m, sigma_m, M, mu_g, sigma_g, g_hallo, area_M = halloweenTrain.trasformata(x_train_ds["margine"], "Margine")
        mu_h, _, _, _, _, _, area_H = halloweenTrain.trasformata(x_train_ds["entropy"], "Entropia")
        _, _, _, _, _, _, area_M = halloweenTrain.trasformata(x_train_ds["margine"], "Margine")
        D = (area_M - area_H) / (area_M + area_H + eps) # per quando tau è molto vicino a 1
        #ratio_H = area_H/(area_M+eps) # per quando tau è molto vicino a 1
        #ratio_M = area_M/(area_H+eps) # per quando tau è molto vicino a 1
        
        x_train_ds["bnd3wd"] = 0
        #test_adattivo = abs(area_H - area_M)
        
        #inizio test adattivo
        #estraggo la parte decimale
        # Estragg la parte decimale
        
        print(f"\n Test adattivo \nArea_h: {area_H:.9f}\nArea_m: {area_M:.9f}")
        tau, _, _ = halloweenTrain.compute_tau(x_train_ds["margine"], x_train_ds["entropy"], area_M, area_H, 1e-8)
        
        
        test_adattivo = (tau > 1) and np.abs(D) > 0.08 #<- scatta più facilmente
        #test_adattivo = (tau > 1) or (np.abs(D) > 0.5) #<- Tau sembrava così figo ma non va bene per questa applicazione
        #test_adattivo =  #<-condizione che verifica la divergenza delle aree in  [-1, 1]
        print(f"\nD = {D:.4f}\n")
        #test_adattivo = tau > 1 #<- meno preciso ma il mio preferito verifica solo lo stato del sistema
        #
        total_steps = len(x_train_ds["bnd3wd"])
        if test_adattivo:
            
            #ma come è possibile che python non abbia un do-while!
            i = 0
            from rich.progress import (
                Progress,
                SpinnerColumn,
                BarColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
                TaskProgressColumn
            )
            
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn(" mu_h={task.fields[mu_h]:.4f} alpha_h={task.fields[alpha_h]:.4f}"), #alpha_m={task.fields[alpha_m]:.4f} beta_m={task.fields[beta_m]:.4f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            with progress:
                task = progress.add_task("Creazione insieme BND...", mu_h=mu_h, alpha_h=alpha_h, total=None)
                while True:
                    #alpha_h, beta_h = halloweenTrain.get_threshold_entropy(i, x_train_ds["entropy"])
                    #alpha_m, beta_m = halloweenTrain.get_threshold_margine(i,  x_train_ds["margine"])
                    alpha_h, _ = halloweenTrain.get_threshold_entropy(i, x_train_ds["entropy"])
                                    
                    #NEG_INCERTI = ((x_train_ds["prob_ben"] < x_train_ds["prob_mal"])
                    #                   & (x_train_ds["entropy"] > (abs(alpha_h - mu_h) / max(sigma_h, 1e-9)))
                    #                   & (x_train_ds["margine"] < alpha_m)
                    #                   & (x_train_ds["margine"] > beta_m))
                    #
                    #POS_INCERTI = ((x_train_ds["prob_ben"] > x_train_ds["prob_mal"])
                    #                   & (x_train_ds["entropy"] > (abs(alpha_h - mu_h) / max(sigma_h, 1e-9)))
                    #                   & (x_train_ds["margine"] < alpha_m)
                    #                   & (x_train_ds["margine"] > beta_m))
                    #
                    #x_train_ds.loc[POS_INCERTI, "bnd3wd"] = -1
                    #x_train_ds.loc[NEG_INCERTI, "bnd3wd"] = -1
                    INCERTI = x_train_ds["entropy"] > alpha_h
                    x_train_ds.loc[INCERTI, "bnd3wd"] = -1
                    
                    # aggiorna la barra (stima ETA)
                    progress.update(task, advance=1, mu_h=mu_h, alpha_h=alpha_h)
                    
                    #if (x_train_ds["bnd3wd"] == -1).any():
                    #    break
                    
                    #static.modelAnalysis.plotUnspace(x_train_ds, alpha_m, beta_m, beta_h)
                    
                    if alpha_h < mu_h: #or total_steps >= i:
                        break
                                        
                    i = i + 1
        
        mask_bnd = x_train_ds["bnd3wd"]
                  
        return mask_bnd
        
        
