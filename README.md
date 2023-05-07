# Прогнозирование конечных свойств новых материалов


## Выпускная квалификационная работа по курсу «Data Science» в Образовательном Центре МГТУ им. Н.Э. Баумана

  Описание: Композиционные материалы - это искусственно созданные материалы, состоящие из нескольких других с четкой границей между ними. Композиты обладают теми свойствами, которые не наблюдаются у компонентов по отдельности. При этом композиты являются монолитным материалом, т.е. компоненты материала неотделимы друг от друга без разрушения конструкции в целом. Яркий пример композита - железобетон. Бетон прекрасно сопротивляется сжатию, но плохо растяжению. Стальная арматура внутри бетона компенсирует его неспособность сопротивляться сжатию, формируя тем самым новые, уникальные свойства. Современные композиты изготавливаются из других материалов: полимеры, керамика, стеклянные и углеродные волокна, но данный принцип сохраняется. У такого подхода есть и недостаток: даже если мы знаем характеристики исходных компонентов, определить характеристики композита, состоящего из этих компонентов, достаточно проблематично. Для решения этой проблемы есть два пути: физические испытания образцов материалов, или прогнозирование характеристик. Суть прогнозирования заключается в симуляции представительного элемента объема композита, на основе данных о характеристиках входящих компонентов (связующего и армирующего компонента).

  На входе имеются данные о начальных свойствах компонентов композиционных материалов (количество связующего, наполнителя, температурный режим отверждения и т.д.). На выходе необходимо спрогнозировать ряд конечных свойств получаемых композиционных материалов.

  Актуальность: Созданные прогнозные модели помогут сократить количество проводимых испытаний, а также пополнить базу данных материалов возможными новыми характеристиками материалов, и цифровыми двойниками новых композитов.

  Целью данной работы является разработка пользовательского приложения для прогнозирования характеристики конечных свойств новых композиционных материалов.

  Этапы работы:

1)  Изучение теоретических основ и методов решения поставленной задачи. Прогнозирование по входным параметрам ряда конечных свойств получаемых композиционных материалов при следующих используемых признаках: • Соотношение матрица-наполнитель • Плотность, кг/м3 • Модуль упругости, ГПа • Количество отвердителя, м.% • Содержание эпоксидных групп,%_2 • Температура вспышки, С_2 • Поверхностная плотность, г/м2 • Потребление смолы, г/м2 • Прочность при растяжении, МПа • Потребление смолы, г/м2 • Угол нашивки, град • Шаг нашивки • Плотность нашивки

2)  Проведение разведочного анализа и представление визуализации предложенных данных. Представлены гистограммы распределения переменнох, диаграммы boxplot, попарные графики рассеяния точек. В таблице представлены для каждой колонки среднее, медианное значение, проведен анализ и исключены выбросы, проверка на наличие пропусков.

3)  Проведение предобработки данных (удалены выбросы, нормализация).

4)  Обучение нескольких моделей для прогноза модуля упругости при растяжении и прочности при растяжении. При построении модели было 30% данных оставлено на тестирование модели, на остальных происходило обучение моделей:

методом ридж регрессии
методом лассо регрессии
методом опорных векторов
методом градиентного бустинга
методом К-ближайших соседей
методом деревья решений
методом случайного леса
методом градиентного спуска

5)  Написание нейронных сетей для рекомендации Соотношения "матрица-наполнитель".

6)  Разработатка пользовательского приложения на Flask для прогноза Соотношение "матрица - наполнитель".

7)  Оценка точности модели.

8)  Создание репозитория в GitHub и размещение кода исследования.

9)  Оформление файла README

  Структура репозитория:

Datasets - папка с файлами: X_bp.xlsx - Первый датасет, X_nup.xlsx - Второй датасет (с нашивками), data_merged.xlsx - объединенный датасет, data_cleaned.xlsx - очищенный от выбросов

App - папка с файлами для корректной работы пользовательского приложения, включая приложение

VKR_Composites_StepanovaV.ipynb - Юпитер Ноутбук с исследованием

reqirements.txt - список внешних зависимостей проекта

Презентация_Степанова В.В.pptx и Презентация_Степанова В.В.pdf - презентация ВКР для защиты в формате pptx и pdf, соответственно

Пояснительная записка_Степанова В.В.docx и Пояснительная записка_Степанова В.В.pdf - описание работы в формате docx и pdf, соответственно

Инструкция использования приложения:

Приложение позволяет решать задачу прогнозирования "Соотношение матрица наполнитель".

Для получения прогноза необходимо

  • Скачать на компьютер все файлы из папки App;

  • Запустить app.py;

  • Совершить запуск всех ячеек;

  • В появившейся строке (Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)) - нажать на ссылку: http://127.0.0.1:5000/.

  • В новом открывшемся окне (сайте) ввести 12 входных параметров, в указанных диапазонах и нажать "Отправить".

  • Появится результат в виде числа с плавающей точкой.

Автор: Степанова Василиса Валерьевна

Выпускная квалификационная работа по программе повышения квалификации «Data Science» в обущчающем центре МГТУ им. Н. Э. Баумана 2023 г.
