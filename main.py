'''
Created on 4 ago 2025

@author: pasquale
'''
import os
import psutil
import pandas as pd
import numpy as np
import glob
#import math #---òp.lòp-+klo.ò-print
#ùòàò
from staticMethodsCollection import filesMethods as sm
from joblib import load as load
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import load_model
#from sklearn.preprocessing import PowerTransformer#, QuantileTransformer, MinMaxScaler
#import matplotlib.pyplot as plt
#from scipy.stats import truncnorm, norm
#from scipy.signal import argrelextrema
#from scipy.interpolate import interp1d
#from scipy.stats import norm, shapiro, normaltest
#from scipy.integrate import quad
'''

def entropy(prob):
    
    return -sum([p * math.log2(p) for p in prob if p > 0])

def gauss(dist, distribuzione: str):
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
    
    # tratto uguali margine e entropia???
    # vado a generare l'asse x della funzione con flatten di numpy
    asse_x = in01.flatten()
    #asse_y = np.linspace(0, 1, len(gauss_in01))
    #interpolazione = interp1d(asse_x, asse_y, fill_value="extrapolate", assume_sorted=True)
    #y_noas = np.clip(asse_y, gauss_in01.min(), gauss_in01.max()) 
    
    # rimuovo eventuali punti nan(asintoti) come da documentazione scypy
    asse_x = asse_x[np.isfinite(asse_x)]
    #asse_y = asse_y[np.isfinite(asse_y)]
    
    # creo un istogramma numpy non per disegnare il grafico ma per campionare la funzione che ho fatto .flatten() su x
    counts, _ = np.histogram(asse_x, bins=len(asse_x), density=True)
    #centro = (bins[:-1] + bins[1:]) / 2 #<- come reshape
    #interpolazione = interp1d(asse_x, asse_y, fill_value="extrapolate", assume_sorted=True) <- da pensare come usare al posto di histogram
    
    # prendo x tale che f(x) = max(y e x= min(y)
    #max_x = np.argmax(counts)
    #min_x = np.argmin(counts)
    #picco = counts[max_x]
    #sella = counts[min_x]
    
    #tratto la funzione campionata come un segnale oscillatorio e ottengo un vettore di valori crescenti
    f = argrelextrema(counts, np.less)[0]
    #diciamo che normalizzo tutto anche se questa formula non è proprio "canonica"
    f = np.array(f)
    min_f = np.argmin(f)
    max_f = np.argmax(f)
    f_hallo = (abs(f - min_f) / abs((min_f - max_f) + epsilon)) / abs(max_f+ epsilon)
    f_hallo =np.array(f_hallo)
    mu_f = np.mean(f_hallo)
    sigma_f = np.std(f_hallo)
    asse_y = np.linspace(0, 1, len(f_hallo))
    
    
    #if len(f) == 0:
        # se non ci sono minimi locali, usa il minimo globale
    #    sella = centro[np.argmin(counts)]
    #else:
        # prendi il minimo locale più vicino al picco considero la deviazione standard
    #    sella = centro[min_loc[np.argmin(np.abs(centro[min_loc] - picco - sigma))]]
        
    #fx_min, fx_max = sorted([sella, picco])
    #ristretta = gauss_in01[(gauss_in01 >= fx_min) & (gauss_in01 <= fx_max)]
    #ristretta = gauss_in01[(gauss_in01 >= sella) & (gauss_in01 <= picco)]
    
    
    #posizioni = len(dist)
    #counts, bins = np.histogram(gauss_in01, bins=posizioni)
    #ripartizione = gauss_in01[gauss_in01 <= np.max(gauss_in01) + sigma]
    #ripartizione = MinMaxScaler(feature_range=(0, 1)).fit_transform(ripartizione.reshape(-1, 1))
    
    #calcolo la metrica
    area = integrale(f_hallo)
    print(f"{distribuzione}: {area}")
    # Plot dei risultati
    
    plt.figure(figsize=(10, 5))
# Istogramma originale
    plt.subplot(1, 2, 1)
    plt.hist(gauss_in01, bins=30, color='lightgreen', edgecolor='black')
    plt.title(f"Distribuzione campionata (Yeo-Johnson)\nμ={mu:.4f}, σ={sigma:.4f}")
    plt.xlabel("Valore")
    plt.suptitle(f"Funzione {distribuzione}")
    plt.ylabel("Frequenza")
    plt.subplot(1, 2, 2)
    plt.plot(f_hallo, asse_y, color="orange")
    plt.title(f"Distribuzione Halloween xD (in arancio) rispetto alla {distribuzione}\nμ={mu_f:.3e}, σ={sigma_f:.3e}")
    plt.xlabel("Variazione")
    plt.suptitle(f"Trasformazione {distribuzione}")
    plt.ylabel("Variabilità")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    #plt.subplot(1, 2, 2)
    #plt.hist(normal, bins=30, color='lightgreen', edgecolor='black')
    #plt.title("Distribuzione trasformata (normale)")
    #plt.xlabel("Valori normalizzati [0,1]")
    #plt.ylabel("Frequenza")


    qt = QuantileTransformer(output_distribution= "normal", n_quantiles=100, random_state=0)
    normal = qt.fit_transform(gauss_in01)
# Istogramma trasformato
    plt.subplot(1, 3, 2)
    plt.hist(normal, bins=30, color='lightgreen', edgecolor='black')
    plt.title("Distribuzione trasformata (normale)")
    plt.xlabel("Valori normalizzati [0,1]")
    plt.ylabel("Frequenza")
        
    normal_shifted = normal - normal.min()
    traslazione = MinMaxScaler(feature_range=(0, 1))
    normal_01 = traslazione.fit_transform(normal_shifted)
    normal_01_min = gauss.min()
    normal_01_max = gauss.max()
    normal_in01 = (normal_01 - normal_01_min) / (normal_01_max - normal_01_min)
    mu_normal = np.mean(normal_in01)
    traslazione = 1 - mu_normal
    normal_in01 = normal_shifted + traslazione
    normal_in01 = np.clip(normal_in01, 0, 100)
    counts, bins = np.histogram(normal_in01, bins=100)
    moda = (bins[np.argmax(counts)] + bins[np.argmax(counts) + 1]) / 2
    normale_sinistra = normal_in01[normal_in01 <= moda]
    normale_sinistra = MinMaxScaler(feature_range=(0, 1)).fit_transform(normale_sinistra.reshape(-1, 1))
    
    plt.subplot(1, 3, 3)
    plt.hist(normale_sinistra, bins=30, color='red', edgecolor='black')
    plt.title("Distribuzione trasformata (normale trasformata)")
    plt.xlabel("Valori normalizzati [0,1]")
    plt.ylabel("Frequenza")
    plt.suptitle(f"Trasformazione Normale troncata -> [0,1]\nμ={mu:.3f}, σ={sigma:.3f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return mu, sigma, gauss_in01, f_hallo#ripartizione#normale_sinistra#normal#uniform#gauss_in01

def integrale(f):
    
    x_vals = np.linspace(0, 1, len(f)) 
    fp = interp1d(x_vals, f, kind='linear', fill_value="extrapolate") 
    Sf, _ = quad(lambda x: float(fp(x)), 0, 1) 
    
    return Sf

def get_threshold_entropy(step, sigma, entropies_array):
    
    epsilon = 1e-6
    #mi, sigma, entropies_array = gauss(entropies_array)
    if step == 0:
        beta = np.percentile(entropies_array, 1)
        alpha = np.percentile(entropies_array, 99)
    else:
        alpha_range = np.clip(99 - step*sigma + epsilon, 0, 100)
        alpha = np.percentile(entropies_array, alpha_range)
        beta_range = np.clip(1 + step*sigma - epsilon, 0, 100)
        beta = np.percentile(entropies_array, beta_range)
    
    return alpha, beta 

def get_threshold_margine(step, sigma, margine_array):
    
    epsilon = 1e-6
    #mu, sigma, margine_array= gauss(margine_array)
    if step == 0:
        beta = np.percentile(margine_array, 50-epsilon)
        alpha = np.percentile(margine_array, 50+epsilon) 
    else:
        alpha_range = np.clip(50 + step*sigma + epsilon, 50, 100)
        alpha = np.percentile(margine_array, alpha_range)
        beta_range = np.clip(50 - step*sigma - epsilon, 50, 100)
        beta = np.percentile(margine_array, beta_range)
    
    return alpha, beta

def get_bnd_roughset(c_df: pd.DataFrame()):
    
    mu, sigma, _, _ = gauss(c_df["entropy"], "Entropia")
    
    conditions3wd = [
        (c_df["prob_ben"] > c_df["prob_mal"]) & (c_df["entropy"] < mu - sigma),
        (c_df["prob_ben"] < c_df["prob_mal"]) & (c_df["entropy"] < mu - sigma)
    ]
    values = [0, 1] #primo adjustment dei valori
    
    c_df["predict_3wd"] = np.select(conditions3wd, values, default=-1) #-1 BND Roughset
    
    return c_df

def merge_with_predicted_values(c_df: pd.DataFrame(), probs, entropies):
    
    c_df["prob_ben"] = probs[:,0]
    c_df["prob_mal"] = probs[:,1]
    c_df["entropy"] = entropies
    c_df["margine"] = (c_df["prob_ben"] - c_df["prob_mal"]).abs()
    
    return c_df

def twd_refine(data_frame: pd.DataFrame(), path_model: str, path_scaler: str):
    
    model, normalized_data = read_model_and_normalization(path_model, path_scaler, data_frame)
    inferenze = predict_as_batch(model, normalized_data)
    entropies = np.array([entropy(p) for p in inferenze])
    predicted = merge_with_predicted_values(data_frame, inferenze, entropies) #che nomi fighi che ho dato alle funzioni :D
    margine = predicted["margine"]
    
    mu_h, sigma_h, entropies, _ = gauss(entropies, "Entropia") #normalizzo le distribuzioni con nuova funzione
    _, sigma_m, margine, _ = gauss(margine, "Margine")     #
    
    step = 0
    alpha_h, beta_h = get_threshold_entropy(step, sigma_h, entropies)
    alpha_m, beta_m = get_threshold_margine(step, sigma_m, margine)
    
    predicted = get_bnd_roughset(predicted)# aggiungo la colonna predict_3wd E definisco roughset BND in base all'entropia
        
    while alpha_h > mu_h - sigma_h: #la condizione di chiusura del ciclo è la frontiera del roughset BND
        #prevengo eventuali divergenze, non serve ma meglio avrle!                                                   
        if alpha_h <= 0 or beta_h >= 1:
            break
        if alpha_m >= 1 or beta_m <= 0:
            break
        print(f"\nEpoch = {step}:\nSoglia margine {abs(alpha_m-beta_m):.4f}\nSoglia Entropia: {abs(alpha_h):.4f}")
        # Cambio Predict con sintassi pandas
        #all'inizio i margini sono più stretti ed entropia più alta (si lavora sui più INCERTI per il modello)
        #successivamente si arriva a margini più ampi e entropie più basse (si lavora verso i record più CERTI per il modello)
        #MALEVOLI -> Benigni
        POS = (
            (predicted["predict_3wd"] == -1) 
            & (predicted["prob_ben"] < predicted["prob_mal"]) 
            & (predicted["entropy"] > abs(alpha_h - mu_h) / sigma_h)#l'entropia viene approssimata ad una truncated normal
            & (predicted["margine"] < alpha_m)
            & (predicted["margine"] > beta_m)
            )
        #BENIGNI -> Malevoli
        NEG = (
            (predicted["predict_3wd"] == -1) 
            & (predicted["prob_ben"] > predicted["prob_mal"]) 
            & (predicted["entropy"] > abs(alpha_h - mu_h) / sigma_h)#l'entropia viene approssimata ad una truncated normal 
            & (predicted["margine"] < alpha_m) 
            & (predicted["margine"] > beta_m)
            )
        # Aggiorna predizioni in base all'ultima iterazione 3wd
        predicted.loc[POS, "predict_3wd"] = 0
        predicted.loc[NEG, "predict_3wd"] = 1
        
        #Signore fa che funzioni <3
        step = step + 1
        alpha_h, _ = get_threshold_entropy(step, sigma_h, entropies)
        alpha_m, beta_m = get_threshold_margine(step, sigma_m, margine)
    
    # ripristina eventuali residui dell'insieme BND
    # con le predizioni del modello 
    # perché non sono stati rivalutati in quanto sono "punti di frontiera"    
    if (predicted["predict_3wd"] == -1).any():
        conditionsStd = [
            (predicted["predict_3wd"] == -1) & (predicted["prob_ben"] > predicted["prob_mal"]),
            (predicted["predict_3wd"] == -1) & (predicted["prob_ben"] < predicted["prob_mal"])
            ]
        values = [0, 1]
        predicted["predict_3wd"] = np.select(conditionsStd, values, default = predicted["predict_3wd"])    
    
    #print(type(margine), margine.shape if hasattr(margine, 'shape') else len(margine))
    #print(type(entropy), entropy.shape if hasattr(entropy, 'shape') else len(entropy))
    
    plt.scatter(margine, entropies, alpha=0.5)

    plt.axvline(1-alpha_m, linestyle='--')
    plt.axvline(1-beta_m, linestyle='--')
    plt.axhline(1-beta_h, linestyle='--')
    plt.xlabel("Margin")
    plt.ylabel("Entropy")
    plt.title("Uncertainty Space (Margin vs Entropy)")
    plt.show()
    
    return predicted
'''

def load_dataset(dataset_type: str, limit: int):
    if dataset_type == "all" or dataset_type == "debug":
        #file_paths = glob.glob("Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet")        # 599948#Accuracy_No = 0.2300 Accuracy3wd = 0.7776#599948 -> 0.8026 -> Accuracy3wd = 0.8026
        #file_paths = glob.glob("Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet") # 525193#Accuracy_No = 0.8479 Accuracy3wd  = 0.8480#525193 -> 0.4081 -> Accuracy3wd = 0.8480
        #file_paths = glob.glob("DDoS*.parquet")                                                     #1014878# Accuracy_No = 0.4874 Accuracy3wd = 0.6693#1014878 -> 0.2636
        #file_paths = glob.glob("DoS1*.parquet")                                                     # 735975# Accuracy_No = 0.9346 Accuracy3wd = 0.9260#735975 -> 0.9260
        #file_paths = glob.glob("Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet")     # 394584# Accuracy_No = 0.8758 Accuracy3wd = 0.8637#394584 -> 0.8637
        #file_paths = glob.glob("Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet")      # 185975# Accuracy_No = 0.7515 Accuracy3wd = 0.7464#185975 -> 0.7468
        #file_paths = glob.glob("Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet")        # 829707# Accuracy_No = 0.9994 Accuracy3wd = 0.9994#829707 -> 0.2181
        #file_paths = glob.glob("Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet")           # 827959# Accuracy_No = 0.9946 Accuracy3wd = 0.9983#827959 -> 0.9750
        file_paths = glob.glob("*.parquet")                                                      # Accuracy_No = 0.7993 Accuracy3wd = 0.8381 () --> 0.4770 crollata :( ---> 0.7962
    else:
        file_paths = dataset_type
    dfs = [pd.read_parquet(f) for f in file_paths]
    try:
        df = pd.concat(dfs)
    except:
        print(f"Error (0): Non ci sono i file .parquet")
    if dataset_type == "debug":
        df = df.head(limit)
    #df.to_csv("original_data.csv")
    if limit != -1: #test supervisionato
        df = df.drop(columns=['Label'])
    return df

def read_model_and_normalization(path_model: str, path_scaler: str, data_frame: pd.DataFrame()):
    model = load_model(path_model, compile = False)
    scaler = load(path_scaler)
    df = data_frame.select_dtypes(include=['float64', 'int64'])
    normalized_data = scaler.fit_transform(df.to_numpy())
    return model, normalized_data

def predict_as_batch(model, normalized_data):
    normalized_data = normalized_data.astype(np.float32)
    #reshape per CNN se 1D: (none, 16, 1) se 2D: (none, 16, 1, 1) ............ (77, 1) .... ma tu c'è fatt e pall tant fratèèèèèèèèè ma come cazz pigli 32 77???? e bast!!!!!!"!
    normalized_data_reshaped = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
    probs_predicted = model.predict(normalized_data_reshaped)
    return probs_predicted

#def mainEval3wd():
#    print(f"Loading dataset ...")
#    df = load_dataset("all", -1)
#
#    print(f"numero di record del dataset = {df.index.size}")
#    label_col = "Label"
#    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
#    classes = df[label_col].value_counts()
#    print(f"\n\nPositivi Totali = {classes.BENIGN}\nNegativi Totali = {df.index.size - classes.BENIGN}\n\nClassi: {classes}\n")
#    
#    df[label_col] = df[label_col].apply(lambda label_i: 0 if "BENIGN" in label_i else 1)
#    df_y = df["Label"].values
#    
#    encoder = LabelEncoder()
#    y_encoded = encoder.fit_transform(df_y)
#    y_cat = to_categorical(y_encoded)
#    y_true = np.argmax(y_cat, axis=1)
#    
#    #model, normalized_data = read_model_and_normalization("unclepaskm_cnn_multi.keras", "scaler_multi.pkl", df_x)
#
#    data_predicted = twd_refine(data_frame = df,
#                                  path_model = "unclepaskm_cnn_multi.keras",
#                                  path_scaler = "scaler_multi.pkl")
#    #data_predicted.to_csv("file_della_verita.csv")
#    #da rivedere e testare
#    count_correct_predicted = data_predicted[data_predicted["Label"] == data_predicted["predict_3wd"]].shape[0]
#    accuracy_3wd = count_correct_predicted / df.index.size
#    #file_veritas = data_predicted.head(1000)
#    #file_veritas.to_csv("file_veritas.csv")
#    print(f"Accuracy3wd = {accuracy_3wd:.4f}")
#    
#    twd_pred_vect = data_predicted["predict_3wd"].values
#    sm.plotConfusionMatrix(y_true, twd_pred_vect, "unclepaskm_cnn_mono_dim")

def main3wd():
    print(f"Loading dataset ...")
    df = load_dataset("all", -1)
    print(f"numero di record del dataset = {df.index.size}")
    label_col = "Label"
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()
    classes = df[label_col].value_counts()
    print(f"\n\nPositivi Totali = {classes.BENIGN}\nNegativi Totali = {df.index.size - classes.BENIGN}\n\nClassi: {classes}\n")
    
    df[label_col] = df[label_col].apply(lambda label_i: 0 if "BENIGN" in label_i else 1)
    df_x = df.drop(columns=['Label'])
    df_y = df["Label"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df_y)
    y_cat = to_categorical(y_encoded)
    
    # Misura memoria prima del caricamento
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024*1024)
    print(f"Memoria prima del caricamento: {mem_before:.2f} MB")
    
    #model, normalized_data = read_model_and_normalization("unclepaskm_cnn_multi.keras", "scaler_multi.pkl", df_x)
    model, normalized_data = read_model_and_normalization("unclepaskm_cnn_multi_10test2_10.keras", "scaler_multi.pkl", df_x)
    normalized_data = normalized_data.astype(np.float32)
    normalized_data_reshaped = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
    
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    
    try: 
        print(f"model.evaluate(...) runs ...")
        loss, acc = model.evaluate(normalized_data_reshaped, y_cat)
        print(f"Accuracy = {acc:.4f} \nLoss = {loss:.4f}")
        print(f"model.predict(...) runs ...")
        y_pred = model.predict(normalized_data_reshaped)
        mem_after = proc.memory_info().rss / (1024*1024)
        print(f"Memoria dopo il caricamento: {mem_after:.2f} MB")
        print(f"Delta memoria: {mem_after - mem_before:.2f} MB")
        y_pred_classes = (y_pred < 0.5).astype(int)
        y_pred_classes = y_pred_classes[:,0]
        y_true = np.argmax(y_cat, axis=1)
        count_neg = np.sum(y_pred_classes == 1)
        count_ben = np.sum(y_pred_classes == 0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_cat, axis=1)
        report = classification_report(y_true_labels, y_pred_labels)
        sm.appendToTxt("model_multi_kernel_newno3wd.txt", report)   
        correct_pred = np.sum(y_pred_classes == y_true) # 5569191 / 6659532 = 0.83627363 mi trvo con accuracy :)
        accuracy_No = correct_pred / df.index.size
        #print(f"Accuracy_modello_No_3wd = {accuracy_No:.4f}")
        print(f"Accuracy_modello_Training_3wd = {accuracy_No:.4f}")
        print(f"\n(Positivi individuati) = {count_ben}\n(Negativi individuati) = {count_neg}\nPredizioni corrette = {correct_pred}")
        report = classification_report(y_true, y_pred_classes)
        #sm.appendToTxt("model_mono_kernel.txt", report)
        # Misura memoria dopo il caricamento
        sm.appendToTxt("model_mono_kernel.txt", report)
    except:
        print(f"\nErrore (3): Metodo evaluate/predict del modello")
    
    #sm.plotConfusionMatrix(y_true, y_pred_classes, "unclepaskm_cnn_mono_dim")
    #sm.plotConfusionMatrix(y_true, y_pred_classes, "lupin_unclepaskm_cnn_multi_dim")

#def checkRAM():
    

# Path al modello salvato (es. modello .h5)
#    model_path = "unclepaskm_cnn_multi_10wdtr_10.keras"

# Misura memoria prima del caricamento
#    proc = psutil.Process(os.getpid())
#    mem_before = proc.memory_info().rss / (1024*1024)
#    print(f"Memoria prima del caricamento: {mem_before:.2f} MB")

# Caricamento modello
#    mainNo3wd()

# Misura memoria dopo il caricamento
#    mem_after = proc.memory_info().rss / (1024*1024)
#    print(f"Memoria dopo il caricamento: {mem_after:.2f} MB")
#    print(f"Delta memoria: {mem_after - mem_before:.2f} MB")

if __name__ == "__main__":
    main3wd()
    #mainEval3wd()
    #checkRAM()