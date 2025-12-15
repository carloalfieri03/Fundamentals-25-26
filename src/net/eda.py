import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configurazione Percorsi
base_path = os.path.join('data', 'UCI HAR Dataset', 'test') 
signals_path = os.path.join(base_path, 'Inertial Signals')
y_test_path = os.path.join(base_path, 'y_test.txt')

# 2. Caricamento Etichette
try:
    y_test = np.loadtxt(y_test_path)
except OSError:
    print("ERRORE: File y_test.txt non trovato. Controlla i percorsi.")
    y_test = np.zeros(100)

# 3. Dizionario delle attività (ID -> Nome)
activities = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

# 4. Funzione per caricare i dati
def load_signal_sample(signal_type, idx):
    data = []
    for axis in ['x', 'y', 'z']:
        filename = f'{signal_type}_{axis}_test.txt'
        filepath = os.path.join(signals_path, filename)
        full_data = np.loadtxt(filepath)
        data.append(full_data[idx, :])
    return np.array(data)

# 5. Creazione della griglia di grafici (2 righe, 3 colonne)
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten() # Appiattiamo l'array per poterci ciclare sopra facilmente

# Ciclo su tutte e 6 le attività
for i, (class_id, class_name) in enumerate(activities.items()):
    
    # Trova il primo indice disponibile per questa classe
    try:
        idx = np.where(y_test == class_id)[0][0]
    except IndexError:
        print(f"Nessun campione trovato per {class_name}")
        continue

    # Carica i dati (Accelerazione Totale)
    signal_data = load_signal_sample('total_acc', idx)
    
    # Plot nel riquadro corrispondente
    ax = axes[i]
    ax.plot(signal_data[0], label='X', alpha=0.8)
    ax.plot(signal_data[1], label='Y', alpha=0.8)
    ax.plot(signal_data[2], label='Z', alpha=0.8)
    
    # Styling
    ax.set_title(f'{class_name} ')
    if i >= 3: # Metti l'etichetta X solo nella riga in basso
        ax.set_xlabel('Time (samples)')
    if i % 3 == 0: # Metti l'etichetta Y solo a sinistra
        ax.set_ylabel('Accelleration (g)')
        
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()