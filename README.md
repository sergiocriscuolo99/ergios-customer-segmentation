# Ergios — Customer Segmentation

Project Work universitario dedicato alla segmentazione automatica della clientela di una startup digitale di servizi IT on-demand.

Il progetto genera in modo riproducibile **500 clienti sintetici**, normalizza le variabili numeriche e confronta configurazioni **K-Means con K=3, K=4 e K=5**. La dashboard Streamlit permette di esplorare i segmenti, analizzare canali e servizi e classificare un nuovo cliente.

## Stack
- Python
- Pandas / NumPy
- Scikit-learn
- K-Means
- Streamlit
- Plotly

## Variabili usate dal clustering
- numero_richieste
- importo_totale_speso
- giorni_ultima_richiesta

I nomi dei segmenti (**Premium, Fedeli, Occasionali, Da riattivare**) vengono assegnati solo dopo il clustering, interpretando i profili medi dei gruppi.

## Avvio locale
1. pip install -r requirements.txt
2. streamlit run streamlit_app.py

## Streamlit Community Cloud
- Branch: main
- Main file: streamlit_app.py
- Secrets: nessuno

Autore: **Sergio Criscuolo**
