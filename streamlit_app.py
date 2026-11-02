import pandas as pd
import plotly.express as px
import streamlit as st

from ergios_core import FEATURES, prepara_progetto

STRATEGIE = {
    'Premium': 'Fidelizzazione, servizi premium e cross-selling su Cybersecurity, Cloud e consulenza.',
    'Fedeli': 'Offerte personalizzate, upgrade e proposta di servizi complementari.',
    'Occasionali': 'Incentivo alla seconda/terza richiesta e comunicazioni mirate sul servizio già utilizzato.',
    'Da riattivare': 'Campagna di riattivazione con reminder, promozione dedicata e contatto personalizzato.',
}

st.set_page_config(page_title='Ergios · Customer Intelligence', page_icon='📊', layout='wide')
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1450px;}
.ergios-title {font-size: 2.45rem; font-weight: 850; letter-spacing: -0.045em; margin-bottom: 0;}
.ergios-sub {color: #6b7280; margin-top: .1rem; margin-bottom: 1.25rem;}
div[data-testid="stMetric"] {border: 1px solid #e5e7eb; padding: 14px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def carica_progetto():
    return prepara_progetto()

def euro(v):
    return f'€ {v:,.0f}'.replace(',', '.')

def filtra(df):
    with st.sidebar:
        st.markdown('### ERGIOS')
        pagina = st.radio('Navigazione', ['Dashboard', 'Nuovo cliente', 'Metodo'])
        st.divider()
        if pagina != 'Dashboard':
            return pagina, df
        st.subheader('Filtri')
        citta = st.multiselect('Città', sorted(df.citta.unique()), default=sorted(df.citta.unique()))
        servizi = st.multiselect('Servizio', sorted(df.categoria_preferita.unique()), default=sorted(df.categoria_preferita.unique()))
        canali = st.multiselect('Canale acquisizione', sorted(df.canale_acquisizione.unique()), default=sorted(df.canale_acquisizione.unique()))
        segmenti = st.multiselect('Segmento', ['Premium', 'Fedeli', 'Occasionali', 'Da riattivare'], default=['Premium', 'Fedeli', 'Occasionali', 'Da riattivare'])
        min_date, max_date = df.data_iscrizione.min().date(), df.data_iscrizione.max().date()
        periodo = st.date_input('Periodo iscrizione', value=(min_date, max_date), min_value=min_date, max_value=max_date)
    out = df[df.citta.isin(citta) & df.categoria_preferita.isin(servizi) & df.canale_acquisizione.isin(canali) & df.segmento.isin(segmenti)]
    if isinstance(periodo, (tuple, list)) and len(periodo) == 2:
        out = out[(out.data_iscrizione >= pd.Timestamp(periodo[0])) & (out.data_iscrizione <= pd.Timestamp(periodo[1]))]
    return pagina, out

def pagina_dashboard(df, confronto):
    st.markdown('<p class="ergios-title">ERGIOS</p>', unsafe_allow_html=True)
    st.markdown('<p class="ergios-sub">Customer Intelligence · Segmentazione automatica della clientela con Machine Learning non supervisionato</p>', unsafe_allow_html=True)
    sil4 = float(confronto.loc[confronto.k == 4, 'silhouette'].iloc[0])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Clienti', len(df))
    c2.metric('Fatturato', euro(df.importo_totale_speso.sum()))
    c3.metric('Richieste', int(df.numero_richieste.sum()))
    c4.metric('Silhouette K=4', f'{sil4:.3f}')
    tab1, tab2, tab3 = st.tabs(['Segmenti', 'Marketing', 'Dati'])
    with tab1:
        a, b = st.columns([1.45, 1])
        with a:
            fig = px.scatter(df, x='giorni_ultima_richiesta', y='importo_totale_speso', color='segmento', size='numero_richieste', hover_data=['cliente_id', 'citta', 'categoria_preferita'], title='Distribuzione dei segmenti', labels={'giorni_ultima_richiesta': "Giorni dall'ultima richiesta", 'importo_totale_speso': 'Spesa totale (€)', 'segmento': 'Segmento'})
            st.plotly_chart(fig, use_container_width=True)
        with b:
            counts = df.segmento.value_counts().rename_axis('segmento').reset_index(name='clienti')
            st.plotly_chart(px.bar(counts, x='segmento', y='clienti', text='clienti', title='Clienti per segmento'), use_container_width=True)
        profili = df.groupby('segmento').agg(clienti=('cliente_id','count'), richieste_medie=('numero_richieste','mean'), spesa_media_cliente=('importo_totale_speso','mean'), giorni_ultima_richiesta=('giorni_ultima_richiesta','mean')).round(2).reset_index()
        profili['azione_marketing'] = profili.segmento.map(STRATEGIE)
        st.subheader('Profilo dei cluster e azioni suggerite')
        st.dataframe(profili, use_container_width=True, hide_index=True)
    with tab2:
        a, b = st.columns(2)
        canali = df.groupby('canale_acquisizione', as_index=False).agg(clienti=('cliente_id','count'), spesa_media_cliente=('importo_totale_speso','mean'), fatturato=('importo_totale_speso','sum'))
        servizi = df.groupby('categoria_preferita', as_index=False).agg(fatturato=('importo_totale_speso','sum'), clienti=('cliente_id','count'))
        with a:
            st.plotly_chart(px.bar(canali.sort_values('spesa_media_cliente', ascending=False), x='canale_acquisizione', y='spesa_media_cliente', title='Valore medio per canale'), use_container_width=True)
        with b:
            st.plotly_chart(px.bar(servizi.sort_values('fatturato', ascending=False), x='categoria_preferita', y='fatturato', title='Fatturato per servizio'), use_container_width=True)
        premium = df[df.segmento == 'Premium']
        if not premium.empty:
            st.info(f"Tra i Premium, il canale più frequente è **{premium.canale_acquisizione.value_counts().idxmax()}** e il servizio più frequente è **{premium.categoria_preferita.value_counts().idxmax()}**.")
    with tab3:
        st.dataframe(df.sort_values('importo_totale_speso', ascending=False), use_container_width=True, hide_index=True)
        st.download_button('Scarica CSV segmentato', df.to_csv(index=False).encode('utf-8-sig'), 'ergios_clienti_segmentati.csv', 'text/csv')

def pagina_nuovo_cliente(progetto):
    st.markdown('<p class="ergios-title">Analizza un nuovo cliente</p>', unsafe_allow_html=True)
    st.markdown('<p class="ergios-sub">Il cliente viene assegnato al cluster K-Means più vicino dopo la stessa normalizzazione usata sul dataset.</p>', unsafe_allow_html=True)
    with st.form('nuovo_cliente'):
        a, b = st.columns(2)
        with a:
            richieste = st.number_input('Numero richieste nei 6 mesi', 1, 20, 3)
            spesa = st.number_input('Importo totale speso (€)', 1.0, 20000.0, 900.0, 50.0)
            giorni = st.number_input("Giorni dall'ultima richiesta", 0, 180, 25)
        with b:
            tipo = st.selectbox('Tipo cliente', ['Privato', 'Professionista/Piccola impresa'])
            citta = st.selectbox('Città', ['Roma','Milano','Napoli','Torino','Bologna'])
            canale = st.selectbox('Canale', ['Google','Instagram','Referral','LinkedIn','Diretto'])
            servizio = st.selectbox('Servizio preferito', ['Assistenza informatica','Sviluppo Web','Cybersecurity','Cloud e Backup','Automazione e Consulenza Digitale'])
        ok = st.form_submit_button('Analizza cliente', use_container_width=True)
    if ok:
        x = pd.DataFrame([{'numero_richieste': richieste, 'importo_totale_speso': spesa, 'giorni_ultima_richiesta': giorni}])
        cid = int(progetto['model'].predict(progetto['scaler'].transform(x[FEATURES]))[0])
        segmento = progetto['mapping'][cid]
        st.success(f'Segmento individuato: **{segmento}**')
        st.write(f'**Azione consigliata:** {STRATEGIE[segmento]}')
        st.caption(f'{tipo} · {citta} · {canale} · {servizio}. Le variabili descrittive non entrano direttamente nel K-Means minimale.')

def pagina_metodo(progetto):
    st.markdown('<p class="ergios-title">Metodo</p>', unsafe_allow_html=True)
    st.markdown('<p class="ergios-sub">Confronto delle configurazioni e logica del prototipo.</p>', unsafe_allow_html=True)
    st.write('Il dataset contiene **500 clienti sintetici** relativi a gennaio–giugno 2026. K-Means utilizza tre variabili numeriche: numero di richieste, importo totale speso e giorni dall’ultima richiesta. Le variabili sono standardizzate prima del clustering.')
    confronto = progetto['confronto'].copy()
    confronto['silhouette'] = confronto['silhouette'].round(3)
    st.dataframe(confronto, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(confronto, x='k', y='silhouette', text='silhouette', title='Silhouette Score per configurazione'), use_container_width=True)
    st.info('I nomi Premium, Fedeli, Occasionali e Da riattivare vengono assegnati **dopo** il clustering, interpretando i profili medi dei gruppi. Non sono etichette fornite al modello.')

progetto = carica_progetto()
pagina, filtrato = filtra(progetto['data'])
if pagina == 'Dashboard':
    pagina_dashboard(filtrato, progetto['confronto'])
elif pagina == 'Nuovo cliente':
    pagina_nuovo_cliente(progetto)
else:
    pagina_metodo(progetto)
