import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import re


def takeData():
    dtp = pd.read_csv('data/DTP.csv')
    fires = pd.read_csv('data/Fires.csv')
    incidents = pd.read_csv('data/Incidents.csv')
    dtp.columns = ['reg', 'type','time', 'reason']
    incidents.columns = ['reg','type', 'time']
    fires.columns = ['reg', 'type','time', 'reason']
    lmap = {
            'Кизел ГО':'ГО город Кизел',
            'Кизеловский ГО': 'ГО город Кизел',
            'Нытвенский  ГО': 'Нытвенский ГО',
            'Березники ГО': 'ГО город Березники',
            'Губахинский ГО': 'Губахинский МО',
            'Верещагино ГО': 'Верещагинский ГО',
            'Гремячинск ГО':'Губахинский МО',
            'Пермский край': '',
    }

    dtp = dtp.dropna()[dtp.dropna().reason.apply(lambda x: 'пдд' not in x.lower())] # убрать нарушение пдд и дтп без указания причины
    fires = fires[fires['reason'].apply(lambda x: 'НППБ' not in x)] # убрать пожары где прична была НППБ
    emergency = pd.concat([dtp, fires])
    del dtp, fires
    emergency.drop('reason', axis=1, inplace=True)
    emergency = pd.concat([emergency, incidents])

    emergency.reg = emergency.reg.apply(lambda x: x.strip())
    emergency.reg = emergency.reg.map(lmap).fillna(emergency.reg)
    emergency.drop(index=emergency[emergency.reg.isin([''])].index, inplace=True)

    def selection(tipe:str)-> str:
        nature = ['заморозки', 'туман','дождь','сильный ветер','просадка', 'абразия', "паводок", "снег", "оползни", "засуха", 'половодье']
        toxic = ['нефте', "АХОВ", "загрязнения"]
        blows = ['пожар', 'взрыв']
        roadTP = ['дтп', 'перерыв в движении', 'транспорт', "аэропорт", "пассажир"]
        GKH = ['канализац', "электроэнергетич", "коммунальн", "питьевой"]

        if re.search(str.join('|', roadTP), tipe.lower().strip()):
            return 'Аварии на транспорте'
        if re.search(str.join('|', toxic), tipe.lower().strip()):
            return 'Аварии с выбросом опасных/токсичных веществ'
        if re.search(str.join('|', nature), tipe.lower().strip()):
            return 'Опасные природные явления'
        if re.search(str.join('|', GKH), tipe.lower().strip()):
            return 'ЖКХ'
        if re.search(str.join('|', blows), tipe.lower().strip()):
            return 'Взрывы/пожары/разрушения'


        return 'Прочие опасности'
    emergency['type'] = emergency['type'].apply(lambda x: selection(x))
    emergency = emergency.reset_index().drop('index', axis=1)
    emergency.time = pd.to_datetime(emergency.time, errors='coerce')
    emergency.dropna(inplace=True)
    emergency.time = pd.to_datetime(emergency.time.dt.strftime('%Y-%m-%d'))  # merge вот эту  табличка
    meteToReg = pd.read_csv('data/Meteo data. MO and meteo accordance.csv')
    meteToReg.columns = ['reg', 'station']
    meteToReg.reg = meteToReg.reg.apply(lambda x: " ".join(x.split()))
    meteToReg.station = meteToReg.station.apply(lambda x: " ".join(x.replace('ё', 'е').split()[1:]))
    meteToReg = dict(zip(meteToReg.reg, meteToReg.station))
    emergency.reg = emergency.reg.map(meteToReg)


    tempStat = pd.read_csv('data/Temperature statistics.csv')
    tempStat = tempStat[~tempStat.year.isin(['среднемноголетняя температура'])]
    tempStat = tempStat.dropna()
    monthRus = {
        'январь':1,
        'ферваль':2,
        'март':3,
        'апрель':4,
        'май':5,
        'июнь':6,
        'июль':7,
        'август':8,
        'сентябрь':9,
        'октябрь':10,
        'ноябрь':11,
        'декабрь':12
    }
    tempStat.month = tempStat.month.map(monthRus)
    tempStat['time'] = pd.to_datetime(tempStat.year+"-"+tempStat.month.astype('string')+"-"+tempStat.day.astype('string'))
    tempStat.drop(['year','month', 'day'], axis=1, inplace=True)


    meteo = pd.read_csv('data/Meteo data.csv')
    meteo.rename(columns={'Местное время':'time'}, inplace=True)
    meteo.time = pd.to_datetime(meteo.time)
    snow_map={
        'Снежный покров не постоянный.': 0,
        'Менее 0.5': 0,
        'Измерение невозможно или неточно.':0,
    }
    meteo.sss = meteo.sss.map(snow_map).fillna(meteo.sss).fillna(0).astype('int')
    col_to_drop = ['DD', 'N', 'WW', 'W1', 'W2', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV', 'RRR', 'E', "E\'"]
    meteo.drop(col_to_drop, axis=1, inplace=True)
    rep_map={
        'Гайны':'Гайны',
        'Октябрьский':'Октябрьский',
        'Чайковский': 'Чайковский'
    }
    meteo.meteostation = meteo.meteostation.map(rep_map).fillna(meteo.meteostation)
    meteo.rename(columns={'meteostation':'reg'}, inplace=True)
    meteo = meteo.groupby([meteo.reg, pd.Grouper(key='time', freq='D')]).agg(['mean', 'std']).transform(lambda val: val.fillna(val.median())).reset_index()
    meteo.columns = list(map(lambda x: f"{x[0]}_{x[1]}",meteo.columns))
    meteo.rename(columns={'reg_':'reg', 'time_':'time'}, inplace=True)


    data = pd.merge(left = emergency, right= meteo, how='right', on=['reg', 'time']).fillna('Nothing')
    data = pd.merge(left = data, right= tempStat, how='inner', on=['time'])
    data.t = data.T_mean-data.t


    

    y = data[(data.time < pd.to_datetime('2022-12-30'))]['type']
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    X = data[(data.time < pd.to_datetime('2022-12-30'))].drop(['type', 'time', 'reg'], axis=1)
    pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier(n_jobs=-1))
    pipeline.fit(X,y)

    answer = pd.DataFrame(columns=['reg']+list(encoder.classes_))
    for reg  in data.reg.unique():
        
        new = [reg]+list((pipeline.predict_proba(data[(data.reg==reg)& (data.time == pd.to_datetime('2022-12-30'))].drop(['type', 'time', 'reg'], axis=1))[0]*10).astype('int'))
        answer.loc[len(answer)] = new

    return answer




with open('regions.json', encoding='utf-8') as reg_file:
    gjson = json.load(reg_file)
MO_acc = pd.DataFrame([i['properties'] | \
                        {'longitude' : i['geometry']['coordinates'][0]} | \
                        {'latitude' : i['geometry']['coordinates'][1]}  for i in gjson['features']])

data = MO_acc.copy()
data['reg'] = data.meteostation.apply(lambda x: " ".join(x.replace('ё', 'е').split()[1:]))
modData = takeData()
modData.drop('Nothing', axis=1, inplace=True)
modData.columns = ['reg', 'ДТП', 'Выброс','Пожар',"ЖКХ","Природное", "Прочее"]
data = pd.merge(left=data, right=modData, how='inner', on='reg')
# data['ДТП'] = pd.Series(np.random.random_sample(len(MO_acc['name']))*10).round().convert_dtypes(convert_integer=True)
# data['Выброс'] = pd.Series(np.random.random_sample(len(MO_acc['name']))*10).round().convert_dtypes(convert_integer=True)
# data['Природное'] = pd.Series(np.random.random_sample(len(MO_acc['name']))*10).round().convert_dtypes(convert_integer=True)
# data['ЖКХ'] = pd.Series(np.random.random_sample(len(MO_acc['name']))*10).round().convert_dtypes(convert_integer=True)
# data['Пожар'] = pd.Series(np.random.random_sample(len(MO_acc['name']))*10).round().convert_dtypes(convert_integer=True)
# data['Прочее'] = pd.Series(np.random.random_sample(len(MO_acc['name']))*10).round().convert_dtypes(convert_integer=True)
data['Максимальная опасность'] = data.loc[:, 'ДТП':].max(axis=1)

def get_hover_text(data, s2=None):
    r = []
    def get_ending(t):
        if t in [2, 3, 4]:
            return ' баллa'
        elif t in [0, 5, 6, 7, 8, 9, 10]:
            return ' баллов'
        else:
            return ' балл'

    for i in range(len(data)):
        r.append('<b>' + data['name'][i] + '</b><br>' + \
                 'Метеостанция ' + data['meteostation'][i] + '<br>')
        if type(s2) == pd.Series:
            r[-1] += 'Опасность - <b>' + str(s2[i]) + get_ending(s2[i]) + '</b>'
        else:
            s_dtp = 'Опасность аварии на транспорте - ' + str(data['ДТП'][i]) + get_ending(data['ДТП'][i])
            if data['Максимальная опасность'][i] - data['ДТП'][i] <= 1:
                s_dtp = '<b><i>' + s_dtp + '</i></b>'
            s_ej = 'Опасность аварии с выбросом веществ - ' + str(data['Выброс'][i]) + get_ending(data['Выброс'][i])
            if data['Максимальная опасность'][i] - data['Выброс'][i] <= 1:
                s_ej = '<b><i>' + s_ej + '</i></b>'
            s_nat = 'Опасность природных явлений - ' + str(data['Природное'][i]) + get_ending(data['Природное'][i])
            if data['Максимальная опасность'][i] - data['Природное'][i] <= 1:
                s_nat = '<b><i>' + s_nat + '</i></b>'
            s_zhkh = 'Опасность аварий ЖКХ - ' + str(data['ЖКХ'][i]) + get_ending(data['ЖКХ'][i])
            if data['Максимальная опасность'][i] - data['ЖКХ'][i] <= 1:
                s_zhkh = '<b><i>' + s_zhkh + '</i></b>'
            s_fire = 'Опасность пожара - ' + str(data['Пожар'][i]) + get_ending(data['Пожар'][i])
            if data['Максимальная опасность'][i] - data['Пожар'][i] <= 1:
                s_fire = '<b><i>' + s_fire + '</i></b>'
            s_other = 'Прочие опасности - ' + str(data['Прочее'][i]) + get_ending(data['Прочее'][i])
            if data['Максимальная опасность'][i] - data['Прочее'][i] <= 1:
                s_other = '<b><i>' + s_other + '</i></b>'
            r[-1] += s_dtp + '<br>' + s_ej + '<br>' + s_nat + '<br>' + s_zhkh + '<br>' + s_fire + '<br>' + s_other
    return r
def map_size(t):
    return 5*(t+1)
def map_color(t, c=1):
    return 10*(1+np.exp(-c*(t - 5)))**-1

fig = go.Figure([
        go.Scattermapbox(name='dtp', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data, data['ДТП']),
                         marker=dict(colorbar=dict(title="Вероятность<br>аварии на транспорте", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['ДТП'].map(map_color), size=data['ДТП'].map(map_size)),
                         showlegend=False, visible=False
                         ),
        go.Scattermapbox(name='ejection', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data, data['Выброс']),
                         marker=dict(colorbar=dict(title="Вероятность аварии с выбросом<br>опасных/токсичных веществ", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['Выброс'].map(map_color), size=data['Выброс'].map(map_size)),
                         showlegend=False, visible=False
                         ),
        go.Scattermapbox(name='nature', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data, data['Природное']),
                         marker=dict(colorbar=dict(title="Вероятность опасных<br>природных явлений", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['Природное'].map(map_color), size=data['Природное'].map(map_size)),
                         showlegend=False, visible=False
                         ),
        go.Scattermapbox(name='zhkh', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data, data['ЖКХ']),
                         marker=dict(colorbar=dict(title="Вероятность<br>аварии ЖКХ", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['ЖКХ'].map(map_color), size=data['ЖКХ'].map(map_size)),
                         showlegend=False, visible=False
                         ),
        go.Scattermapbox(name='fire', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data, data['Пожар']),
                         marker=dict(colorbar=dict(title="Вероятность<br>взрыва/пожара/разрушения", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['Пожар'].map(map_color), size=data['Пожар'].map(map_size)),
                         showlegend=False, visible=True
                         ),
        go.Scattermapbox(name='other', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data, data['Прочее']),
                         marker=dict(colorbar=dict(title="Вероятность<br>прочих опасностей", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['Прочее'].map(map_color), size=data['Прочее'].map(map_size)),
                         showlegend=False, visible=False
                         ),

        go.Scattermapbox(name='overall', lat=data['latitude'], lon=data['longitude'], 
                         hoverinfo='text', hovertext=get_hover_text(data),
                         marker=dict(colorbar=dict(title="Максимальная<br>опасность", tickvals=np.arange(11),
                                                   ticktext=[f'{i} баллов' if i in [0, 10] else i for i in range(11)]), 
                                     colorscale='Viridis', cmin=0, cmax=10, sizemin=0,
                                     color=data['Максимальная опасность'].map(map_color), size=data['Максимальная опасность'].map(map_size)/1.5),
                         showlegend=False, visible=False
                         )
])

fig.update_layout( 
    updatemenus=[ 
        dict( 
            type="buttons", 
            direction="down", 
            buttons=list([
                dict( 
                    args=[dict(visible=[True, False, False, False, False, False, False])], 
                    label="Аварии на транспорте", 
                    method="restyle"
                ),
                dict( 
                    args=[dict(visible=[False, True, False, False, False, False, False])], 
                    label="Аварии с выбросом<br>опасных/токсичных веществ", 
                    method="restyle"
                ),
                dict( 
                    args=[dict(visible=[False, False, True, False, False, False, False])], 
                    label="Природные явления", 
                    method="restyle"
                ),
                dict( 
                    args=[dict(visible=[False, False, False, True, False, False, False])], 
                    label="ЖКХ", 
                    method="restyle"
                ),
                dict( 
                    args=[dict(visible=[False, False, False, False, True, False, False])],
                    label="Пожары", 
                    method="restyle"
                ),
                dict( 
                    args=[dict(visible=[False, False, False, False, False, True, False])], 
                    label="Прочее", 
                    method="restyle"
                ),

                dict( 
                    args=[dict(visible=[False, False, False, False, False, False, True])], 
                    label="<b>Макс. опасность</b>", 
                    method="restyle"
                )
            ]), 
        ), 
    ] 
)

fig.update_layout(title='Риски возникновения опасностей',
                  mapbox_style='open-street-map', margin={"r":250,"t":35,"l":170,"b":20},
                  mapbox=dict(center=dict(lat=59, lon=56), zoom=5.5))
fig.show()