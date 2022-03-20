from pdb import Restart
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob

# general variables
# employees_interaction_matrix
# specialists (add info about past coefficients)

# if os.path.isdir('general'):
#     print('general already exists')
# else:
#     os.mkdir('general')
#     print('create general folder')

if os.path.isfile('specialists.csv'):
    specialists = pd.read_csv('specialists.csv')
else:
    specialists = pd.DataFrame(columns=['id', 'is_busy', 'psycho', 'skill_ML', 'skill_DS', 'skill_DA', 'skill_DE', 
                                        'skill_BACKEND', 'skill_FRONTEND', 'skill_DESIGN', 'skill_PM', 'skill_PO'])
    specialists.to_csv('specialists.csv')
    print('Please add specialists to specialists.csv')

# значение это уровень (0-отсутствие, 1-джун, 2-мидл, 3-сеньор)

if os.path.isfile('employees_interaction_matrix.csv'):
    print('employees_interaction_matrix already exists')
    employees_interaction_matrix = pd.read_csv('employees_interaction_matrix.csv').set_index('id')
else:
    employees_interaction_matrix = pd.DataFrame()
    employees_interaction_matrix.index = specialists.id
    employees_interaction_matrix[specialists.id] = np.random.randint(80, 120, size=(employees_interaction_matrix.shape[0], employees_interaction_matrix.shape[0]))
    employees_interaction_matrix.to_csv('employees_interaction_matrix.csv', index=False)


# local project variables
# project_distribution
# OPS
# OSS

# if len(glob('project_*')) == 0:
#     project_number = 0
#     os.mkdir('project_{}'.format(project_number))
#     print('project folder was created')
# else:
#     max_project_number = max([int(x.split('_')[-1]) for x in glob('project_*')])
#     if len(glob('project_{}/*'.format(max_project_number))):
#         project_number = max_project_number
#         print('project folder already created')
#     else:
#         project_number = max([int(x.split('_')[-1]) for x in glob('project_*')])+1
#         os.mkdir('project_{}'.format(project_number))
#         print('project folder was created')
# project_path = 'project_{}'.format(project_number)



st.title('Оценка эффективности проектной команды')

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

st.header('Конфигуратор проекта')
required_time = st.number_input('Введите ожидаемое количество дней на выполнение проекта', 
                                min_value=1, 
                                format='%d')
project_type = st.selectbox(
     'Выберите тип проекта (от этого зависит важность времени выполнения)',
     ('Срочный', 'Стандартный', 'Исследовательский'))


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
st.header('Конфигуратор команды')
required_specialists_number = st.number_input('Выберите нужно количество специалистов', 
                                              min_value=0, max_value=specialists.shape[0], 
                                              format='%d')

project_distribution = pd.DataFrame()
project_distribution['skill'] = ['' for i in range(required_specialists_number)]
project_distribution['level'] = ['' for i in range(project_distribution.shape[0])]
# project_distribution = project_distribution.sort_values(['skill'])
# if st.checkbox('Выберите специальность и требуемый уровень'):

gb = GridOptionsBuilder.from_dataframe(project_distribution)
gb.configure_default_column(editable=True)
gb.configure_column('skill', cellEditor='agRichSelectCellEditor',
                    cellEditorParams={'values':['ML','DS','DA','DE','BACKEND','FRONTEND','DESIGN','PM','PO']})
gb.configure_column('level',cellEditor='agRichSelectCellEditor',
                    cellEditorParams={'values':['Junior', 'Middle', 'Senior']})
gb.configure_grid_options(enableRangeSelection=True)
response = AgGrid(project_distribution, gridOptions=gb.build(),
                  fit_columns_on_grid_load=True, allow_unsafe_jscode=True,
                  enable_enterprise_modules=True)
project_distribution = response['data'].sort_values(['skill', 'level'], ascending=False)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Рекомендуемые специалисты
# Подбор исходя из is_busy, skills и employees_interaction_matrix

def get_best_communication(ids_, ids):
    best = 0
    best_id = 0
    for id in ids_:
        if employees_interaction_matrix.loc[ids, id].mean() > best:
            best = employees_interaction_matrix.loc[ids, id].mean()
            best_id = id
    return best_id


project_specialists_ids = []

if project_distribution[project_distribution.skill != ''].shape[0] > 0:

    st.header('Рекомендуемая команда')

    specialists_notexist = []
    for i in range(project_distribution.shape[0]):
        skill, level = project_distribution.skill.iloc[i], project_distribution.level.iloc[i]
        print(skill, level)
        free_levels = [x for x in range(1, 4) if x != level]
        suitable = specialists[(specialists['skill_{}'.format(skill)] == level)&(specialists.is_busy == 0)]
        if suitable.shape[0] < 1:
            suitable = specialists[(specialists['skill_{}'.format(skill)] == max(free_levels))&(specialists.is_busy == 0)]
        if suitable.shape[0] < 1:
            suitable = specialists[(specialists['skill_{}'.format(skill)] == min(free_levels))&(specialists.is_busy == 0)]
        if suitable.shape[0] < 1:
            project_distribution.loc[i, 'id'] = 0
            specialists_notexist.append([skill, level])
        if suitable.shape[0] == 1 and project_distribution.shape[0] > 0:
            project_distribution.loc[i, 'id'] = suitable.id.iloc[0]
            specialists.loc[specialists.id == suitable.id.iloc[0], 'is_busy'] = 1
            project_specialists_ids.append(suitable.id.iloc[0])
        if suitable.shape[0] > 1:
            for j in range(suitable.shape[0]):
                best_id = get_best_communication(suitable.id.values, project_distribution.id.values)
            project_distribution.loc[i, 'id'] = best_id
            specialists.loc[specialists.id == best_id, 'is_busy'] = 1
            project_specialists_ids.append(best_id)

    if len(specialists_notexist) == 0:
        project_distribution.id = project_distribution.id.astype(int)
        project_distribution['psycho'] = specialists.loc[specialists.id.isin(project_distribution.id), 'psycho'].values
        st.text('id-идентификатор специалиста в БД, psycho-психологическое состояние, навык-специальность, level-уровень подготовки')
        st.table(project_distribution[['id', 'psycho', 'skill', 'level']].set_index('id'))
    else:
        st.text('Этих специалистов нет в базе:'+str(specialists_notexist))

    st.header('Матрица взаимодействия команды')
    st.text('100 - норма, чем больше, тем лучше взаимодействие и наоборот')
    fig, ax = plt.subplots()
    sns.heatmap(employees_interaction_matrix.loc[project_specialists_ids, [str(x) for x in project_specialists_ids]], ax=ax)
    st.write(fig)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 
    st.header('Оценка проекта специалистами до начала')
    st.text('Оцените вашу уверенность в сроках. Если 100% - уверенность в выполнении проекта дата в дату, сколько процентов составит реальный срок?' \
            + '\n' + 'Пример: 120% означает, что вы предполагаете задержку срока выполнения на 20%')
    st.text('Уверенность в успешности. 100% - полная уверенность в том, что проект реально выполнить в плане задач и качества, на сколько процентов успешности рассчитываете вы?' \
            + '\n' + 'Пример: 90% означает, что вы ожидаете какие-то сложности и недовыполнение проекта на 10%')
    
    OPS = pd.DataFrame()
    OPS['id'] = project_distribution.id
    OPS[['Сроки (Старт)',
        'Успешность (Старт)']] = np.random.randint(80, 120, size=(project_distribution.shape[0], 2))
    st.table(OPS)

    st.header('Оценка проекта руководителем до начала')
    OPS_head = pd.DataFrame()
    OPS_head['Сроки (Старт)'] = [st.number_input('Оцените вашу уверенность в выполнении сроков проекта', 
                                                min_value=0, max_value=200, 
                                                format='%d')]
    OPS_head['Успешность (Старт)'] = [st.number_input('Оцените вашу уверенность в успешности выполнения проекта', 
                                                     min_value=0, max_value=200, 
                                                     format='%d')]

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

start_project = st.checkbox('Запустить проект')

if start_project:

    st.header('Оценка проекта специалистами по завершении')
    st.text('Реальные сроки по ощущениям. \nПример: 100% - дата в дату, 120% - не хватило 20% времени, 80% - времени было в избыток на 20%')
    st.text('Успешность по ощущениям. Пример: 100% - проект выполнен от начала до конца, 90% - недовыполнение проекта на 10%, 115% - удалось улучшить проект на 15% по сравнению с ТЗ')
    
    # OPS = pd.DataFrame()
    # OPS['id'] = project_distribution.id
    OPS[['Сроки (Конец)',
        'Успешность (Конец)']] = np.random.randint(80, 120, size=(project_distribution.shape[0], 2))
    st.table(OPS[['id', 'Сроки (Конец)', 'Успешность (Конец)']])


    st.header('Оценка проекта руководителем по завершении')
    OPS_head['Сроки (Конец)'] = [st.number_input('Оцените выполнение сроков проекта', 
                                                min_value=0, max_value=200, 
                                                format='%d')]
    OPS_head['Успешность (Конец)'] = [st.number_input('Оцените успешность выполнения проекта', 
                                                     min_value=0, max_value=200, 
                                                     format='%d')]

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    st.header('Оценка специалистов специалистами')

    OSS_quality_matrix = pd.DataFrame()
    OSS_quality_matrix.index = project_distribution.id
    OSS_quality_matrix[project_distribution.id] = np.random.randint(80, 140, 
                                                                    size=(OSS_quality_matrix.shape[0], 
                                                                        OSS_quality_matrix.shape[0]))
    OSS_communication_matrix = pd.DataFrame()
    OSS_communication_matrix.index = project_distribution.id
    OSS_communication_matrix[project_distribution.id] = np.random.randint(60, 100, 
                                                                        size=(OSS_communication_matrix.shape[0], 
                                                                                OSS_communication_matrix.shape[0]))


    st.subheader('Оценка успешности выполнения работы')
    st.table(OSS_quality_matrix)
    st.subheader('Оценка качества взаимодействия')
    st.table(OSS_communication_matrix)

    OSS = pd.DataFrame()
    OSS['id'] = project_distribution.id
    OSS['Средняя оценка успешности выполнения работы'] = np.round(OSS_quality_matrix.apply(lambda x: x.sum() / OSS.shape[0], axis=1).values).astype(int)
    OSS['Средняя оценка качества взаимодействия'] = np.round(OSS_communication_matrix.apply(lambda x: x.sum() / OSS.shape[0], axis=1).values).astype(int)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    st.header('Психологическое состояние сотрудников')
    st.text('Специалисты проходят тест, после чего их ответы аггрегируются в коэффициент от 0 до 100')

    st.text('Оцените по пятибалльной шкале степень важности для него каждого из 12 указанных ниже факторов, \nвлияющих на его отношение к работе и мотивацию к деятельности.' \
        + '\nГде: 5 – согласен, 4 – мне близко это утверждение (скорее да, чем нет), 3 – затрудняюсь ответить' \
            + '\n2 – скорее нет, чем да, 1 – я не согласен')
    psycho_test = pd.read_csv('psycho_test.csv')
    st.table(psycho_test)
    psycho_state = pd.DataFrame()
    psycho_state['id'] = project_distribution.id
    psycho_state['Состояние'] = np.random.randint(60, 100, size=psycho_state.shape[0])
    st.table(psycho_state)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Оценка проекта заказчиком
    OPS_head.loc[1, OPS_head.columns] = [100, 100, np.random.randint(80, 120), np.random.randint(80, 120)]
    OPS_head = OPS_head.astype(int)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    st.header('Итоговые показатели')

    specialists_time_coeff = 1
    if OPS['Сроки (Конец)'].mean() < OPS['Сроки (Старт)'].mean() and OPS['Сроки (Старт)'].mean() <= 100:
        specialists_time_coeff = OPS['Сроки (Старт)'].mean() / OPS['Сроки (Конец)'].mean()
    elif OPS['Сроки (Конец)'].mean() > OPS['Сроки (Старт)'].mean() and OPS['Сроки (Конец)'].mean() > 100:
        specialists_time_coeff =  OPS['Сроки (Старт)'].mean() / OPS['Сроки (Конец)'].mean()

    head_time_coeff = 1
    if OPS_head['Сроки (Конец)'].iloc[0] < OPS_head['Сроки (Старт)'].iloc[0] and OPS_head['Сроки (Старт)'].iloc[0] <= 100:
        head_time_coeff = OPS_head['Сроки (Старт)'].iloc[0] / OPS_head['Сроки (Конец)'].iloc[0]
    elif OPS_head['Сроки (Конец)'].iloc[0] > OPS_head['Сроки (Старт)'].iloc[0] and OPS_head['Сроки (Конец)'].iloc[0] > 100:
        head_time_coeff = OPS_head['Сроки (Старт)'].iloc[0] / OPS_head['Сроки (Конец)'].iloc[0]

    empl_time_coeff = 1
    if OPS_head['Сроки (Конец)'].iloc[1] < OPS_head['Сроки (Старт)'].iloc[1] and OPS_head['Сроки (Старт)'].iloc[1] <= 100:
        empl_time_coeff = OPS_head['Сроки (Старт)'].iloc[1] / OPS_head['Сроки (Конец)'].iloc[1]
    elif OPS_head['Сроки (Конец)'].iloc[1] > OPS_head['Сроки (Старт)'].iloc[1] and OPS_head['Сроки (Конец)'].iloc[1] > 100:
        empl_time_coeff = OPS_head['Сроки (Старт)'].iloc[1] / OPS_head['Сроки (Конец)'].iloc[1]

    
    specialists_succ_coeff = 1
    if OPS['Успешность (Конец)'].mean() < OPS['Успешность (Старт)'].mean() and OPS['Успешность (Конец)'].mean() < 100:
        specialists_succ_coeff = OPS['Успешность (Конец)'].mean() / OPS['Успешность (Старт)'].mean()
    elif OPS['Успешность (Конец)'].mean() > OPS['Успешность (Старт)'].mean() and OPS['Успешность (Конец)'].mean() >= 100:
        specialists_succ_coeff =  OPS['Успешность (Конец)'].mean() / OPS['Успешность (Старт)'].mean()

    head_succ_coeff = 1
    if OPS_head['Успешность (Конец)'].iloc[0] < OPS_head['Успешность (Старт)'].iloc[0] and OPS_head['Успешность (Конец)'].iloc[0] < 100:
        head_succ_coeff = OPS_head['Успешность (Конец)'].iloc[0] / OPS_head['Успешность (Старт)'].iloc[0]
    elif OPS_head['Успешность (Конец)'].iloc[0] > OPS_head['Успешность (Старт)'].iloc[0] and OPS_head['Успешность (Конец)'].iloc[0] >= 100:
        head_succ_coeff = OPS_head['Успешность (Конец)'].iloc[0] / OPS_head['Успешность (Старт)'].iloc[0]

    empl_succ_coeff = 1
    if OPS_head['Успешность (Конец)'].iloc[1] < OPS_head['Успешность (Старт)'].iloc[1] and OPS_head['Успешность (Конец)'].iloc[1] < 100:
        empl_succ_coeff = OPS_head['Успешность (Конец)'].iloc[1] / OPS_head['Успешность (Старт)'].iloc[1]
    elif OPS_head['Успешность (Конец)'].iloc[1] > OPS_head['Успешность (Старт)'].iloc[1] and OPS_head['Успешность (Конец)'].iloc[1] >= 100:
        empl_succ_coeff = OPS_head['Успешность (Конец)'].iloc[1] / OPS_head['Успешность (Старт)'].iloc[1]

    

    # OSS = pd.DataFrame()
    # OSS['id'] = project_distribution.id
    # OSS['Средняя оценка успешности выполнения работы'] = np.round(OSS_quality_matrix.apply(lambda x: x.sum() / OSS.shape[0], axis=1).values).astype(int)
    # OSS['Средняя оценка качества взаимодействия'] = np.round(OSS_communication_matrix.apply(lambda x: x.sum() / OSS.shape[0], axis=1).values).astype(int)

    Results = pd.DataFrame()
    Results['Кто'] = ['Специалисты', 'Руководитель', 'Заказчик']
    Results.loc[0, 'Оценка сроков'] = [100*specialists_time_coeff]
    Results.loc[1, 'Оценка сроков'] = [100*head_time_coeff]
    Results.loc[2, 'Оценка сроков'] = [100*empl_time_coeff]

    Results.loc[0, 'Оценка успешности'] = [0.5*OSS['Средняя оценка успешности выполнения работы'].mean() + 0.5*100*specialists_succ_coeff]
    Results.loc[1, 'Оценка успешности'] = [100*head_succ_coeff]
    Results.loc[2, 'Оценка успешности'] = [100*empl_succ_coeff]

    Results.loc[0, 'Оценка взаимодействия'] = [0.5*OSS['Средняя оценка качества взаимодействия'].mean() + 0.5*psycho_state['Состояние'].mean()]
    Results.loc[1, 'Оценка взаимодействия'] = [-1]
    Results.loc[2, 'Оценка взаимодействия'] = [-1]

    Results = Results.set_index('Кто').astype(int)

    st.subheader('Общие')
    st.table(Results)

    # Результаты по сотрудникам

    Results_s = pd.DataFrame()
    Results_s['id'] = project_specialists_ids
    Results_s['Оценка сроков'] = 100*specialists_time_coeff
    Results_s['Оценка успешности'] = OSS['Средняя оценка успешности выполнения работы'].values
    Results_s['Оценка взаимодействия'] = OSS['Средняя оценка качества взаимодействия']
    Results_s = Results_s.set_index('id').astype(int)
    
    st.subheader('Отдельно по специалистам')
    st.table(Results_s)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


