'''
Created on Nov 8, 2025

@author: pasquale
'''
import numpy as np
import pandas as pd
import halloweenTrain as hTrain
#import staticMethodsCollection as static
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical#, plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from sklearn.metrics import classification_report
from staticMethodsCollection import filesMethods as sm
from joblib import dump as save_scaler
from warnings import filterwarnings
from cnnMultikernel import cnnMultikernel
from sklearn.utils.class_weight import compute_class_weight
from keras.src.losses import loss
filterwarnings("ignore")

def training_experiment(model, dataset: str):
    # 1. Carichiamo dataset ed individuiamo classi e colonna delle realizzazioni
    df = sm.readDatasetParquet(dataset)
#df = df.head(100000)

    label_col = "Label"  # <-- Nome della colonna con le etichette
    label_map = {
        "BENIGN": "0",
        "DDOS ATTACKS-LOIC-HTTP": "1",
        "BOT": "2",
        "SSH-BRUTEFORCE": "3",
        "INFILTERATION": "4",
        "DOS ATTACKS-GOLDENEYE": "5",
        "DOS ATTACKS-SLOWLORIS": "6",
        "BRUTE FORCE -WEB": "7",
        "BRUTE FORCE -XSS": "8",
        "FTP-BRUTEFORCE": "9",
        "SQL INJECTION": "10"
    }
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    # 2. Applica la mappatura
    df["y"] = df[label_col].map(label_map)

    # 3. Rimuovi righe con etichette non mappate (valori NaN)
    df = df[df["y"].notna()]

    # 4. Converti la colonna y in interi
    df["y"] = df["y"].astype(int)
    #print(df[label_col].value_counts(dropna=False))
    # REDUCE: concatena tutti i DataFrame validi
    #df = pd.concat(mapped_df, ignore_index=True)

    # 5. Rimuovi la colonna etichetta testuale
    X = df.drop(columns=[label_col, "y"])
    #y_uni_class = df["y"].unique()
    #y = df["y"].values

    # 6. Codifica binario il dataset: 0 = BENIGN, 1 = Malicious
    df[label_col] = df[label_col].apply(lambda x: 0 if "BENIGN" in x.upper() else 1)
    y = df[label_col].values
    #da decommentare per scriptare più classi
    #le = LabelEncoder()
    #y = le.fit_transform(df[label_col])
    #print(list(le.classes_))
    #print(y)

    # 7. Rimuovi colonne categoriche se presenti
    X = X.select_dtypes(include=["float64", "int64"])

    # 8. Standardizzazione 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. Ridimensionamento per la CNN
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    # 10. One-hot encoding binario
    y_cat = to_categorical(y, num_classes=2)

# 11. Split dal dataset originale del Train dataset di Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, 
        y_cat, 
        test_size=0.4,
        train_size=0.4,
        stratify=np.argmax(y_cat, axis=1), 
        random_state=42 ) #<- attenzione questo restituisce un oggetto list -> se si lavora a backend può mischiare le righe
    #print("Distribuzione classi nel test set:", np.unique(np.argmax(y_test, axis=1), return_counts=True))
    
    # 12. Bilanciamento del Train dataset mediante metodo ziopask undersample
    y_train_int = np.argmax(y_train, axis=1)
    
    class_weight_var = compute_class_weight(class_weight = 'balanced', 
                                            classes = np.unique(y_train_int), 
                                            y = y_train_int)
    
    dict_cw = dict(enumerate(class_weight_var))
    #X_train, y_train = sm.balance_classes_undersample(X_train, y_train)
    
    # 13. salvataggio dello scaler dal metodo dump 
    save_scaler(scaler, "scaler_multi.pkl")
    
    # 14. Definizione del modello di Machine Learning
    #input_shape = (None, X_train.shape[1], 1)
    #input_shape = (X_train.shape[1], 1)
    #modelInit = cnnMultikernel(2, input_shape)
    #model = modelInit.build_cnn_ultra()
    
    #sm.overwriteModelMap(model, "model_multi_kernel6.txt")
    #plot_model(model, to_file="cnn_multi_map.png", show_shapes=True, show_layer_names=True)
    
    # 15. callback per ogni ciclo di addestramento e quando smette di apprendere salviamo il migliore
    callbacks_var = [
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True), 
        ReduceLROnPlateau(monitor="val_loss", patience=20, factor=0.5, min_lr=1e-6)             
    ]
    
    # 16. Machine Learning
    total_epochs = 20
    i = 0
    #loss3wd = acc3wd = rec3wd = pre3wd = 0
    rec3wd = 0
    while i < total_epochs:   
        print(f"\n-------\n3WD epoch {i+1} epoch fase di fit training set (T) ...")
        #history = 
        model.fit(
            X_train, 
            y_train,
            epochs = 1,           
            batch_size = 64,
            validation_data = (X_test, y_test),
            class_weight = dict_cw,
            callbacks = callbacks_var)
        
        X_val, _, y_val, _ = train_test_split(
            X_reshaped, 
            y_cat, 
            train_size=0.4,
            stratify=np.argmax(y_cat, axis=1), 
            random_state=42 )
        
        print(f"\n3WD epoch {i+1} fase di validation set (V) ...")
        probs = model.predict(
            X_val, 
            batch_size=64 )
        
        entropie = hTrain.halloweenTrain.entropie(probs)
        margini = hTrain.halloweenTrain.margini(probs)
        
        x_train_ds = pd.DataFrame(X_val.reshape(X_train.shape[0], -1))
        
        x_train_ds["prob_ben"] = probs[:,0]
        x_train_ds["prob_mal"] = probs[:,1]
        x_train_ds["entropy"] = entropie
        x_train_ds["margine"] = margini
        #per non creare problemi con l'intestazione numerica del tataframe
        x_train_ds.columns = x_train_ds.columns.astype(str)
        
        bnd_roughset = hTrain.halloweenTrain.get_bnd(iter, x_train_ds)
        
        x_train_ds["bnd3wd"] = bnd_roughset
        X_train_2sd = x_train_ds[x_train_ds["bnd3wd"] == -1]
        
        #riferimento debug per capire se stavo lavorando su modelli diversi o se addestravo sempre lo stesso modello
        #model_no_3wd = model
        print(f"\nCardinalità BND: {len(X_train_2sd)}")
        if not X_train_2sd.empty:
            #y_2sd = df[label_col].values
            #indice sparsi
            #y_2sd = pd.Series(y_train_2sd, index=X_train.index)
            #y_2sd_filtered = y_2sd.loc[X_train_2sd.index]
            y_train_2sd = y_val[np.isin(np.arange(y_val.shape[0]), X_train_2sd.index)]
            #y_cat_2sd = to_categorical(y_train_2sd, num_classes=2)
               
            
            X_train_2sd = X_train_2sd.drop(columns=["prob_ben", "prob_mal", "entropy", "margine"])
            X_train_2sd = X_train_2sd.select_dtypes(include=["float64", "int64"])
            X_train_2sd_scaled = scaler.fit_transform(X_train_2sd) #<- attenzione POTREBBE CAMBIARE scala dei dati
            #X_train_2sd_scaled = scaler.transform(X_train_2sd, feature_names_out=None)
            X_reshaped_2nd = X_train_2sd_scaled.reshape((X_train_2sd_scaled.shape[0], X_train_2sd_scaled.shape[1], 1))
            #X_train3wd, y_train3wd = sm.balance_classes_undersample(X_reshaped_2nd, y_train_2sd)
            #X_train3wd, X_test3wd, y_train3wd, y_test3wd = train_test_split(
            #    X_reshaped_2nd,
            #    y_cat,
            #    test_size=0.2,
            #    stratify=np.argmax(y_cat, axis=1),
            #    random_state=42 )
            #X_train3wd, y_train3wd = sm.balance_classes_undersample(X_train3wd, y_train3wd)
            #indice sparsi
            #y_2sd = pd.Series(y_train_2sd, index=X_train.index)
            #y_2sd_filtered = y_2sd.loc[X_train_2sd.index]
            
            #inutile se voglio usare il 100% dei dati incerti
            #X_train_2nd, X_test_2nd, y_train_2nd, y_test_2nd = train_test_split(
            #    X_reshaped_2nd, 
            #    y_cat_2sd, 
            #    test_size=1, 
            #    stratify=np.argmax(y_cat_2sd, axis=1), 
            #    random_state=42 )
            num_inc = len(X_reshaped_2nd)
            print(f"\nEpoch {i+1} fit per dati incerti sul validation set (V)\nNumero record BND: {num_inc}\n")
            #history = 
            model.fit(
                X_reshaped_2nd, 
                y_train_2sd,
                epochs = 1,           
                batch_size = 64,
                validation_data = (X_test, y_test),
                class_weight = dict_cw#,
                #callbacks = callbacks_var
            )
        
        print(f"\n3WD epoch {i+1} fase di evaluate con il test set (S) ...")  
        loss, acc, _, rec = model.evaluate(
            X_test, 
            y_test, 
            batch_size=64)
        
        if rec > rec3wd:
            print(f"\nSalvo modello con accuracy {acc} loss {loss} e recall {rec}")
            model.save("unclepaskm_cnn_multi_10test2_acc.keras")
            rec3wd = rec
                    
        i = i + 1
    
    # 17. Report di Valutazione ========
    #confronto i modelli migliori e libero memoria
    #loss, acc = model.evaluate(X_test, y_test)
    #if model3WD != None:
    #    loss2, acc2 = model3WD.evaluate(X_test, y_test)
    #    if loss2 < loss:
    #        model = model3WD
    #        model3WD = None
    #    else:
    #        model3WD = None
    #y_pred = model_no_3wd.predict(X_test)
    #y_pred_labels = np.argmax(y_pred, axis=1)
    #y_true_labels = np.argmax(y_test, axis=1)
    #report = classification_report(y_true_labels, y_pred_labels)
    #sm.appendToTxt("model_multi_kernel_3wdtr.txt", report)
    # 18. Salva il modello con il nuovo formato keras
    return model
    
def save_model(model):
    model.save("unclepaskm_cnn_multi_10test2_10.keras")
    #y_pred = model.predict(X_test)
    #y_pred_labels = np.argmax(y_pred, axis=1)
    #y_true_labels = np.argmax(y_test, axis=1)
    #report = classification_report(y_true_labels, y_pred_labels)
    #sm.appendToTxt("model_multi_kernel_10test2.txt", report)
    #plotAccu = sm.plotTrainingAccuracy(history, "Accuracy - CNN MultiDimentional Kernel with residual approach 3wd training")
    #plotAccu.show()
    #plotLoss = sm.plotTrainingLoss(history, "Loss - CNN MultiDimentional Kernel with residual approach 3wd training")
    #plotLoss.show()

def init_model():
    input_shape = (None, 16, 1)
    modelInit = cnnMultikernel(2, input_shape)
    model = modelInit.build_cnn_ultra()
    return model

model = init_model()
#print("\n\nBotnet traffic training\n")
#model = training_experiment(model, 'TBot*.parquet')
#print("\n\nBruteforce, Dos and Ddos traffic training\n")
#model = training_experiment(model, 'B*.parquet')
##print("\n\nDdoS & DoS traffic training\n")
##model = training_experiment(model, 'D*.parquet')
#print("\n\nInjection traffic training\n")
#model = training_experiment(model, 'Inf*.parquet')
#print("\n\nWeb traffic training\n")
#model = training_experiment(model, 'Web*.parquet')
print("\n\nAll dataset\n")
model = training_experiment(model, '*.parquet')
save_model(model)
