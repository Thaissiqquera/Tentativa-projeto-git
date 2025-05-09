
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

def handler(request, context):
    try:
        body = request.json
        transacoes = pd.DataFrame(body.get("transacoes"))
        campanhas = pd.DataFrame(body.get("campanhas"))

        clientes = transacoes.groupby('cliente_id').agg({
            'frequencia_compras': 'max',
            'total_gasto': 'max',
            'ultima_compra': 'max'
        }).reset_index()

        scaler = StandardScaler()
        clientes_scaled = scaler.fit_transform(clientes[['frequencia_compras', 'total_gasto', 'ultima_compra']])
        kmeans = KMeans(n_clusters=3, random_state=42)
        clientes['cluster'] = kmeans.fit_predict(clientes_scaled)

        campanhas_grouped = campanhas.groupby('campanha_id').agg({
            'custo_campanha': 'mean',
            'alcance': 'mean',
            'conversao': 'mean'
        }).reset_index()

        X = campanhas_grouped[['custo_campanha', 'alcance']]
        y = campanhas_grouped['conversao']
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_.tolist()

        return {
            "statusCode": 200,
            "body": json.dumps({
                "clusters": clientes[['cliente_id', 'cluster']].to_dict(orient='records'),
                "regressao_coeficientes": coef
            }),
            "headers": {
                "Content-Type": "application/json"
            }
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"erro": str(e)}),
            "headers": {
                "Content-Type": "application/json"
            }
        }
