import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
N_CLIENTI = 500
DATA_INIZIO = pd.Timestamp('2026-01-01')
DATA_FINE = pd.Timestamp('2026-06-30')
FEATURES = ['numero_richieste', 'importo_totale_speso', 'giorni_ultima_richiesta']

CITTA = {'Roma': 140, 'Milano': 120, 'Napoli': 90, 'Torino': 80, 'Bologna': 70}
CANALI = {'Google': 175, 'Instagram': 100, 'Referral': 100, 'LinkedIn': 75, 'Diretto': 50}
BUSINESS_PER_CANALE = {'Google': 105, 'Instagram': 40, 'Referral': 65, 'LinkedIn': 60, 'Diretto': 30}
SERVIZI = {
    'Assistenza informatica': 150,
    'Sviluppo Web': 100,
    'Cybersecurity': 75,
    'Cloud e Backup': 75,
    'Automazione e Consulenza Digitale': 100,
}
BUSINESS_PER_SERVIZIO = {
    'Assistenza informatica': 60,
    'Sviluppo Web': 65,
    'Cybersecurity': 56,
    'Cloud e Backup': 49,
    'Automazione e Consulenza Digitale': 70,
}
PREZZI = {
    'Assistenza informatica': (40, 200, 90),
    'Sviluppo Web': (500, 2000, 750),
    'Cybersecurity': (300, 2000, 600),
    'Cloud e Backup': (80, 500, 180),
    'Automazione e Consulenza Digitale': (150, 900, 300),
}
FREQUENZA_BASE = {
    'Assistenza informatica': 4.2,
    'Sviluppo Web': 1.35,
    'Cybersecurity': 1.4,
    'Cloud e Backup': 2.6,
    'Automazione e Consulenza Digitale': 1.8,
}
PROFILI_GENERAZIONE = {
    'molto_attivo': {'peso': 0.15, 'freq_mult': 1.80, 'recency': (0, 20)},
    'regolare': {'peso': 0.35, 'freq_mult': 1.20, 'recency': (10, 50)},
    'bassa_attivita': {'peso': 0.35, 'freq_mult': 0.75, 'recency': (25, 90)},
    'inattivo': {'peso': 0.15, 'freq_mult': 0.85, 'recency': (90, 180)},
}

def _lista_da_conteggi(conteggi):
    valori = []
    for valore, quantita in conteggi.items():
        valori.extend([valore] * quantita)
    return valori

def _genera_canale_e_tipo(rng):
    coppie = []
    for canale, totale in CANALI.items():
        n_business = BUSINESS_PER_CANALE[canale]
        coppie.extend([(canale, 'Professionista/Piccola impresa')] * n_business)
        coppie.extend([(canale, 'Privato')] * (totale - n_business))
    rng.shuffle(coppie)
    return [x[0] for x in coppie], [x[1] for x in coppie]

def _assegna_servizi(rng, tipi_cliente):
    out = [None] * len(tipi_cliente)
    business_idx = [i for i, t in enumerate(tipi_cliente) if t == 'Professionista/Piccola impresa']
    privati_idx = [i for i, t in enumerate(tipi_cliente) if t == 'Privato']
    servizi_business, servizi_privati = [], []
    for servizio, totale in SERVIZI.items():
        n_business = BUSINESS_PER_SERVIZIO[servizio]
        servizi_business.extend([servizio] * n_business)
        servizi_privati.extend([servizio] * (totale - n_business))
    rng.shuffle(servizi_business)
    rng.shuffle(servizi_privati)
    for idx, servizio in zip(business_idx, servizi_business):
        out[idx] = servizio
    for idx, servizio in zip(privati_idx, servizi_privati):
        out[idx] = servizio
    return out

def _genera_ticket(rng, servizio, numero_richieste, tipo_cliente):
    minimo, massimo, media_target = PREZZI[servizio]
    posizione_media = (media_target - minimo) / (massimo - minimo)
    alpha = 2.0
    beta = alpha * (1 - posizione_media) / posizione_media
    ticket = minimo + (massimo - minimo) * rng.beta(alpha, beta, size=numero_richieste)
    if tipo_cliente == 'Professionista/Piccola impresa':
        ticket *= 1.08
    return np.clip(ticket, minimo, massimo)

def genera_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    citta = _lista_da_conteggi(CITTA)
    rng.shuffle(citta)
    canale, tipo = _genera_canale_e_tipo(rng)
    servizio = _assegna_servizi(rng, tipo)
    giorni_periodo = (DATA_FINE - DATA_INIZIO).days + 1
    offset = rng.integers(0, giorni_periodo, size=N_CLIENTI)
    nomi_profili = list(PROFILI_GENERAZIONE)
    pesi_profili = [PROFILI_GENERAZIONE[p]['peso'] for p in nomi_profili]
    profili = rng.choice(nomi_profili, size=N_CLIENTI, p=pesi_profili)
    limiti = {
        'Assistenza informatica': 12,
        'Sviluppo Web': 4,
        'Cybersecurity': 4,
        'Cloud e Backup': 10,
        'Automazione e Consulenza Digitale': 6,
    }
    righe = []
    for i in range(N_CLIENTI):
        data_iscrizione = DATA_INIZIO + pd.Timedelta(days=int(offset[i]))
        anzianita = (DATA_FINE - data_iscrizione).days
        profilo = PROFILI_GENERAZIONE[profili[i]]
        categoria = servizio[i]
        fattore_anzianita = max(0.35, anzianita / 180)
        lam = FREQUENZA_BASE[categoria] * profilo['freq_mult'] * fattore_anzianita
        n_richieste = max(1, int(rng.poisson(lam)))
        n_richieste = min(n_richieste, limiti[categoria])
        rmin, rmax = profilo['recency']
        if anzianita <= 0:
            giorni_ultima = 0
        else:
            lo = min(rmin, anzianita)
            hi = min(rmax, anzianita)
            giorni_ultima = int(rng.integers(lo, hi + 1))
        ticket = _genera_ticket(rng, categoria, n_richieste, tipo[i])
        totale = round(float(ticket.sum()), 2)
        righe.append({
            'cliente_id': f'ERG{i+1:04d}',
            'tipo_cliente': tipo[i],
            'citta': citta[i],
            'data_iscrizione': data_iscrizione,
            'canale_acquisizione': canale[i],
            'categoria_preferita': categoria,
            'numero_richieste': n_richieste,
            'importo_totale_speso': totale,
            'giorni_ultima_richiesta': giorni_ultima,
            'spesa_media_richiesta': round(totale / n_richieste, 2),
        })
    return pd.DataFrame(righe)

def analizza_clustering(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    confronto = []
    modelli = {}
    for k in (3, 4, 5):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = model.fit_predict(X)
        confronto.append({'k': k, 'silhouette': float(silhouette_score(X, labels)), 'inertia': float(model.inertia_)})
        modelli[k] = model
    model = modelli[4]
    labels = model.labels_
    out = df.copy()
    out['cluster_id'] = labels
    profili = out.groupby('cluster_id').agg(
        clienti=('cliente_id', 'count'),
        richieste_medie=('numero_richieste', 'mean'),
        spesa_media_cliente=('importo_totale_speso', 'mean'),
        giorni_ultima_richiesta=('giorni_ultima_richiesta', 'mean'),
        ticket_medio=('spesa_media_richiesta', 'mean'),
    )
    premium = profili['spesa_media_cliente'].idxmax()
    fedeli = profili.drop(index=premium)['richieste_medie'].idxmax()
    restanti = [i for i in profili.index if i not in (premium, fedeli)]
    da_riattivare = profili.loc[restanti, 'giorni_ultima_richiesta'].idxmax()
    occasionale = [i for i in restanti if i != da_riattivare][0]
    mapping = {
        int(premium): 'Premium',
        int(fedeli): 'Fedeli',
        int(occasionale): 'Occasionali',
        int(da_riattivare): 'Da riattivare',
    }
    out['segmento'] = out['cluster_id'].map(mapping)
    return out, scaler, model, mapping, pd.DataFrame(confronto)

def prepara_progetto(seed=SEED):
    df = genera_dataset(seed)
    segmentato, scaler, model, mapping, confronto = analizza_clustering(df)
    return {
        'raw': df,
        'data': segmentato,
        'scaler': scaler,
        'model': model,
        'mapping': mapping,
        'confronto': confronto,
    }
