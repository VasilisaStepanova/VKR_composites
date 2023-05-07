# -*- coding: utf-8 -*-
import flask
from flask import render_template, request
import pandas as pd
import pickle
import sklearn
from sklearn.neural_network import MLPRegressor





# Функция для извлечения признаков из данных формы
def get_data_from_form(features, params):
    param_names = features.keys()
    data = dict.fromkeys(param_names, None)
    error = ''
    # Преобразование из строк в числа
    for param_name, param_value in params.items():
        if param_value.strip(' \t') != '':
            try:
                data[param_name] = float(param_value)
            except:
                error += f'{features[param_name]} - некорректное значение "{param_value}"\n'
    # Проверка допустимых диапазонов
    # Соотношение матрица-наполнитель (0..6)
    if 'var1' in data and data['var1'] is not None:
        if data['var1'] < 0 or data['var1'] > 6:
            error += f'{features["var1"]} - значение вне корректного диапазона\n'
    # Плотность, кг/м3 (1700...2300)
    if 'var2' in data and data['var2'] is not None:
        if data['var2'] < 1700 or data['var2'] > 2300:
            error += f'{features["var2"]} - значение вне корректного диапазона\n'
    # Модуль упругости, ГПа (2...2000)
    if 'var3' in data and data['var3'] is not None:
        if data['var3'] < 2 or data['var3'] > 2000:
            error += f'{features["var3"]} - значение вне корректного диапазона\n'
    # Количество отвердителя, м.% (17...200)
    if 'var4' in data and data['var4'] is not None:
        if data['var4'] < 17 or data['var4'] > 200:
            error += f'{features["var4"]} - значение вне корректного диапазона\n'
    # Содержание эпоксидных групп,%_2 (14...34)
    if 'var5' in data and data['var5'] is not None:
        if data['var5'] < 14 or data['var5'] > 34:
            error += f'{features["var5"]} - значение вне корректного диапазона\n'
    # Температура вспышки, С_2 (100...414)
    if 'var6' in data and data['var6'] is not None:
        if data['var6'] < 100 or data['var6'] > 414:
            error += f'{features["var6"]} - значение вне корректного диапазона\n'
    # Поверхностная плотность, г/м2 (0.6...1400)
    if 'var7' in data and data['var7'] is not None:
        if data['var7'] < 0.6 or data['var7'] > 1400:
            error += f'{features["var7"]} - значение вне корректного диапазона\n'
    # Модуль упругости при растяжении, ГПа (64...83)
    if 'var8' in data and data['var8'] is not None:
        if data['var8'] < 64 or data['var8'] > 83:
            error += f'{features["var8"]} - значение вне корректного диапазона\n'
    # Прочность при растяжении, МПа (1036...3849)
    if 'var9' in data and data['var9'] is not None:
        if data['var9'] < 1036 or data['var9'] > 3849:
            error += f'{features["var9"]} - значение вне корректного диапазона\n'
    # Потребление смолы, г/м2 (33...414)
    if 'var10' in data and data['var10'] is not None:
        if data['var10'] < 33 or data['var10'] > 414:
            error += f'{features["var10"]} - значение вне корректного диапазона\n'
    # Угол нашивки, град(0 или 90)
    if 'var11' in data and data['var11'] is not None:
        if data['var11'] != 0.0 and data['var11'] != 90.0:
            error += f'{features["var11"]} - значение вне корректного диапазона\n'
    # Шаг нашивки(0...15)
    if 'var12' in data and data['var12'] is not None:
        if data['var12'] < 0 or data['var12'] > 15:
            error += f'{features["var12"]} - значение вне корректного диапазона\n'
    # Плотность нашивки (0...104)
    if 'var13' in data and data['var13'] is not None:
        if data['var13'] < 0 or data['var13'] > 104:
            error += f'{features["var13"]} - значение вне корректного диапазона\n'
    # Проверка отсутствующих значений
    if None in data.values():
        none = 'Некоторые значения отсутствуют!\n'
        error += f'{none}'
    # Заменить сокращенные имена признаков на полные
    data_clean = dict(zip(features.values(), data.values()))
    return data_clean, error


app = flask.Flask(__name__, template_folder='templates')



@app.route('/', methods = ['POST', 'GET'])
def main():
    # Необходимые признаки
    features = {
        'var2': 'Плотность, кг/м3',
        'var3': 'модуль упругости, ГПа',
        'var4': 'Количество отвердителя, м.%',
        'var5': 'Содержание эпоксидных групп,%_2',
        'var6': 'Температура вспышки, С_2',
        'var7': 'Поверхностная плотность, г/м2',
        'var8': 'Модуль упругости при растяжении, ГПа',
        'var9': 'Прочность при растяжении, МПа',
        'var10': 'Потребление смолы, г/м2',
        'var11': 'Угол нашивки, град',
        'var12': 'Шаг нашивки',
        'var13': 'Плотность нашивки'
    }
    # Переменные для формы
    # params = {'var2': '', 'var3': '', 'var4': '', 'var5': '', 'var6': '', 'var7': '',
    #           'var8': '', 'var9': '', 'var10': '', 'var11': '', 'var12': '', 'var13': ''}
 
    # Получены данные из формы
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        with open('preprocessor3.pkl', 'rb') as f:
            load_model = pickle.load(f)
            preprocessor3 = load_model
        
        with open('best_model3.pkl', 'rb') as f:
            load_model = pickle.load(f)
            best_model3 = load_model
            
        params = request.form.to_dict()
        data, error = get_data_from_form(features, params)
        if error == '':
            # Входные данные корректны, выполняется логика
            x = pd.DataFrame(data, index=[0])
            # для соотношения матрица-наполнитель
           
            x3 = preprocessor3.transform(x)
            y3 = best_model3.predict(x3)
            var1 = y3[0]
    # Отображение результата
    return render_template('main.html', params=params, error=error, inputs=x.to_html(), result = var1)


if __name__ == '__main__':
    app.run()
