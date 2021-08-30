

#######################################################
                        ROADMAP

#######################################################


GOALS :
GOLDEN : Prévision neige par coordonnées + indicateur d'avalance
SILVER : Prédiction neige pour chaque station
BRONZE : Prédiction neige pour 5 meilleure station


STEPS :

1) Get Data
    Source : Météo France
    Get data with scrapper when deployed / Direct download with CSV --> Big Pandas DF (Gdrive)

2) Clean Data
    Drop NaN
    Detect and delete abhérentes values (ex: Neg values for snow fall)

3) Data Selection
    Select 3 last years (test) and the rest = train
    Select 5 stations (+nomination)
    Heatmap colinearity
        >> Select features ()

4) Base model
    If nothing, get metrics

5) Build DL model
    Select DL, grid search

6) Test model

7) Build web app (streamlit) + Data viz

8) Choose to upgrade

8') Build container

9/9') Deployment



