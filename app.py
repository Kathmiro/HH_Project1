import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
import os
from pathlib import Path

# Настройка страницы
st.set_page_config(
    page_title="Анализ вакансий Habr Career",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Загрузка данных с корректным определением пути
@st.cache_data
def load_data():
    # Определение абсолютного пути к файлу относительно расположения скрипта
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "vacancies.csv"

    # Проверка существования файла
    if not data_path.exists():
        st.error(f"Файл данных не найден: {data_path}")
        # Альтернативный путь (если запуск из корня проекта)
        alt_path = Path("data/processed/vacancies.csv")
        if alt_path.exists():
            data_path = alt_path
            st.success(f"Найден альтернативный путь к данным: {alt_path}")
        else:
            return pd.DataFrame()  # Возвращаем пустой датафрейм при отсутствии файла

    # Загрузка и предобработка данных
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # Конвертация дат
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

    return df


# Функция для создания облака слов
def generate_wordcloud(text_series, stopwords=None):
    if stopwords is None:
        stopwords = set()

    # Проверка на пустую серию
    if text_series.empty:
        return plt.figure(figsize=(10, 5))

    # Объединяем все тексты с проверкой типа
    text = ' '.join([str(text) for text in text_series if isinstance(text, str)])

    if not text.strip():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Недостаточно данных для создания облака слов",
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig

    # Создаем облако слов
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        colormap='viridis',
        collocations=False
    ).generate(text)

    # Отображаем облако слов
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


# Основное приложение
def main():
    st.title("📊 Анализ вакансий Habr Career")

    # Загрузка данных
    df = load_data()

    # Проверка успешности загрузки данных
    if df.empty:
        st.error("Не удалось загрузить данные. Пожалуйста, проверьте наличие файла данных.")
        return

    # Боковая панель с фильтрами
    st.sidebar.header("Фильтры")

    # Фильтр по городам
    cities = df['city'].dropna().unique().tolist()
    all_cities = ['Все города'] + sorted(cities)
    selected_city = st.sidebar.selectbox("Выберите город", all_cities)

    # Фильтр по компаниям
    companies = df['company'].value_counts().head(20).index.tolist()
    top_companies = ['Все компании'] + companies
    selected_company = st.sidebar.selectbox("Выберите компанию", top_companies)

    # Фильтр по формату работы
    remote_col = 'remote_work' if 'remote_work' in df.columns else ('remote' if 'remote' in df.columns else None)
    if remote_col:
        work_format = st.sidebar.radio(
            "Формат работы",
            ["Все", "Удаленно", "Офис"]
        )

    # Применение фильтров
    filtered_df = df.copy()

    if selected_city != 'Все города':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]

    if selected_company != 'Все компании':
        filtered_df = filtered_df[filtered_df['company'] == selected_company]

    if remote_col and work_format != 'Все':
        if work_format == 'Удаленно':
            filtered_df = filtered_df[filtered_df[remote_col] == True]
        else:
            filtered_df = filtered_df[filtered_df[remote_col] == False]

    # Отображение базовой информации
    st.header("Общая информация")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего вакансий", filtered_df.shape[0])
    with col2:
        st.metric("Количество компаний", filtered_df['company'].nunique())
    with col3:
        if 'salary_rub' in filtered_df.columns:
            salary_data = filtered_df[filtered_df['salary_rub'].notnull()]
            if not salary_data.empty:
                avg_salary = salary_data['salary_rub'].mean()
                st.metric("Средняя зарплата (₽)", f"{avg_salary:,.0f}")
            else:
                st.metric("Средняя зарплата (₽)", "Нет данных")

    # Вкладки для разных секций дашборда
    tab1, tab2, tab3, tab4 = st.tabs(["Данные", "Анализ", "Навыки", "Отчет"])

    # Вкладка "Данные"
    with tab1:
        st.subheader("Просмотр данных")

        # Выбор колонок для отображения
        all_columns = df.columns.tolist()
        default_columns = ['title', 'company', 'city', 'remote' if 'remote' in all_columns else 'remote_work',
                           'salary_rub', 'seniority_level']
        default_columns = [col for col in default_columns if col in all_columns]

        selected_columns = st.multiselect(
            "Выберите колонки для отображения",
            all_columns,
            default=default_columns
        )

        if selected_columns:
            st.dataframe(filtered_df[selected_columns], height=400)
        else:
            st.dataframe(filtered_df, height=400)

        # Статистика по пропущенным значениям
        st.subheader("Пропущенные значения")
        missing_values = filtered_df.isnull().sum()
        missing_df = pd.DataFrame({
            'Колонка': missing_values.index,
            'Количество пропущенных': missing_values.values,
            'Процент пропущенных': (missing_values / len(filtered_df) * 100).values
        })
        missing_df = missing_df[missing_df['Количество пропущенных'] > 0].sort_values('Количество пропущенных',
                                                                                      ascending=False)

        if not missing_df.empty:
            st.dataframe(missing_df)

            # Визуализация пропущенных значений
            fig = px.bar(
                missing_df.head(10),  # Показываем только топ-10 колонок с пропусками
                x='Колонка',
                y='Процент пропущенных',
                title='Процент пропущенных значений по колонкам',
                color='Процент пропущенных',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("В выбранных данных нет пропущенных значений")

    # Вкладка "Анализ"
    with tab2:
        st.subheader("Распределение вакансий")

        col1, col2 = st.columns(2)

        # Распределение по городам
        with col1:
            if selected_city == 'Все города' and filtered_df['city'].nunique() > 1:
                city_counts = filtered_df['city'].value_counts().head(10)
                fig = px.bar(
                    x=city_counts.index,
                    y=city_counts.values,
                    labels={'x': 'Город', 'y': 'Количество вакансий'},
                    title='Географическое распределение вакансий',
                    color=city_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Выбран конкретный город или недостаточно данных для визуализации")

        # Распределение по компаниям
        with col2:
            if selected_company == 'Все компании' and filtered_df['company'].nunique() > 1:
                company_counts = filtered_df['company'].value_counts().head(10)
                fig = px.bar(
                    x=company_counts.index,
                    y=company_counts.values,
                    labels={'x': 'Компания', 'y': 'Количество вакансий'},
                    title='Топ компаний-работодателей',
                    color=company_counts.values,
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Выбрана конкретная компания или недостаточно данных для визуализации")

        # Анализ зарплат
        if 'salary_rub' in filtered_df.columns:
            st.subheader("Анализ зарплат")

            salary_data = filtered_df[filtered_df['salary_rub'].notnull()]

            if not salary_data.empty and len(salary_data) >= 3:  # Минимальный порог для визуализации
                col1, col2 = st.columns(2)

                # Гистограмма зарплат
                with col1:
                    fig = px.histogram(
                        salary_data,
                        x='salary_rub',
                        nbins=20,
                        labels={'salary_rub': 'Зарплата (₽)'},
                        title='Распределение зарплат',
                        color_discrete_sequence=['#2d8a8a']
                    )

                    fig.add_vline(
                        x=salary_data['salary_rub'].median(),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Медиана: {salary_data['salary_rub'].median():,.0f}₽"
                    )

                    fig.add_vline(
                        x=salary_data['salary_rub'].mean(),
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Среднее: {salary_data['salary_rub'].mean():,.0f}₽"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Ящик с усами по городам
                with col2:
                    if selected_city == 'Все города' and filtered_df['city'].nunique() > 1:
                        top_cities = filtered_df['city'].value_counts().head(5).index.tolist()
                        city_salary_data = filtered_df[
                            filtered_df['city'].isin(top_cities) & filtered_df['salary_rub'].notnull()]

                        if not city_salary_data.empty and len(city_salary_data) >= 3:
                            fig = px.box(
                                city_salary_data,
                                x='city',
                                y='salary_rub',
                                labels={'city': 'Город', 'salary_rub': 'Зарплата (₽)'},
                                title='Распределение зарплат по городам',
                                color='city',
                                color_discrete_sequence=px.colors.qualitative.Dark2
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Недостаточно данных для построения диаграммы распределения зарплат по городам")
                    else:
                        # Если выбран конкретный город, показываем распределение по уровням должности
                        if 'seniority_level' in filtered_df.columns:
                            seniority_salary_data = filtered_df[
                                filtered_df['seniority_level'].notnull() & filtered_df['salary_rub'].notnull()]

                            if not seniority_salary_data.empty and len(seniority_salary_data) >= 3:
                                fig = px.box(
                                    seniority_salary_data,
                                    x='seniority_level',
                                    y='salary_rub',
                                    labels={'seniority_level': 'Уровень', 'salary_rub': 'Зарплата (₽)'},
                                    title='Распределение зарплат по уровням должности',
                                    color='seniority_level',
                                    color_discrete_sequence=px.colors.qualitative.Set2
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Недостаточно данных для визуализации зарплат по уровням должности")

                # Добавляем анализ зарплат по категориям, если доступен
                if 'salary_category' in filtered_df.columns:
                    salary_cat_data = filtered_df[filtered_df['salary_category'].notnull()]
                    if not salary_cat_data.empty:
                        # Распределение по категориям зарплат
                        st.subheader("Распределение по категориям зарплат")
                        cat_counts = salary_cat_data['salary_category'].value_counts().reindex([
                            'до 100K', '100K-200K', '200K-300K', '300K-500K', 'от 500K'
                        ])

                        # Убираем NaN значения
                        cat_counts = cat_counts.dropna()

                        if not cat_counts.empty:
                            fig = px.bar(
                                x=cat_counts.index,
                                y=cat_counts.values,
                                labels={'x': 'Категория зарплаты', 'y': 'Количество вакансий'},
                                title='Распределение вакансий по категориям зарплат',
                                color=cat_counts.values,
                                color_continuous_scale='Viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Недостаточно данных о зарплатах для построения визуализаций")

        # Анализ удаленной работы
        if remote_col:
            st.subheader("Формат работы")

            remote_counts = filtered_df[remote_col].value_counts().reset_index()
            if not remote_counts.empty:
                remote_counts.columns = ['Удаленная работа', 'Количество']
                remote_counts['Удаленная работа'] = remote_counts['Удаленная работа'].map(
                    {True: 'Удаленно', False: 'Офис'})

                fig = px.pie(
                    remote_counts,
                    values='Количество',
                    names='Удаленная работа',
                    title='Распределение вакансий по формату работы',
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.3
                )
                # Добавляем проценты
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

    # Вкладка "Навыки"
    with tab3:
        st.subheader("Анализ требуемых навыков")

        # Собираем все навыки из отфильтрованных данных
        all_skills = []
        for skills in filtered_df['skills'].dropna():
            if isinstance(skills, str) and skills:
                all_skills.extend([skill.strip() for skill in skills.split(',') if skill.strip()])

        if all_skills:
            skills_freq = pd.Series(all_skills).value_counts()

            col1, col2 = st.columns(2)

            # Топ навыков
            with col1:
                top_skills = skills_freq.head(15)
                fig = px.bar(
                    x=top_skills.values,
                    y=top_skills.index,
                    orientation='h',
                    labels={'x': 'Количество упоминаний', 'y': 'Навык'},
                    title='Топ-15 наиболее востребованных навыков',
                    color=top_skills.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            # Облако навыков
            with col2:
                # Создаем стоп-слова для более информативного облака
                stopwords = set(['и', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'за', 'из', 'у'])
                wordcloud_fig = generate_wordcloud(pd.Series(all_skills), stopwords=stopwords)
                st.pyplot(wordcloud_fig)

            # Анализ связи между количеством навыков и зарплатой
            if 'skills_count' in filtered_df.columns and 'salary_rub' in filtered_df.columns:
                st.subheader("Навыки и зарплаты")

                # Фильтруем данные, убирая NaN значения
                scatter_data = filtered_df[
                    filtered_df['salary_rub'].notnull() &
                    filtered_df['skills_count'].notnull()
                    ]

                if not scatter_data.empty and len(scatter_data) >= 5:  # Минимальный порог для анализа
                    # Определяем опциональные параметры для scatter plot
                    scatter_kwargs = {
                        'x': 'skills_count',
                        'y': 'salary_rub',
                        'trendline': 'ols',
                        'labels': {
                            'skills_count': 'Количество навыков',
                            'salary_rub': 'Зарплата (₽)'
                        },
                        'title': 'Зависимость зарплаты от количества требуемых навыков',
                        'hover_data': ['company', 'title']
                    }

                    # Добавляем цвет по уровню должности, если доступен
                    if 'seniority_level' in scatter_data.columns and scatter_data['seniority_level'].notnull().any():
                        scatter_kwargs['color'] = 'seniority_level'

                    # Добавляем размер точек по опыту, только если нет NaN значений
                    if 'experience_years' in scatter_data.columns:
                        # Создаем копию данных без NaN в experience_years
                        exp_data = scatter_data.dropna(subset=['experience_years'])
                        if not exp_data.empty and len(exp_data) >= 5:
                            # Используем только данные без NaN в experience_years
                            scatter_data = exp_data
                            scatter_kwargs['size'] = 'experience_years'
                            scatter_kwargs['size_max'] = 15

                    # Создаем график
                    fig = px.scatter(scatter_data, **scatter_kwargs)
                    st.plotly_chart(fig, use_container_width=True)

                    # Вычисляем корреляцию
                    corr = scatter_data[['skills_count', 'salary_rub']].corr().iloc[0, 1]
                    corr_text = "сильная положительная" if corr > 0.7 else \
                        "умеренная положительная" if corr > 0.3 else \
                            "слабая положительная" if corr > 0 else \
                                "слабая отрицательная" if corr > -0.3 else \
                                    "умеренная отрицательная" if corr > -0.7 else "сильная отрицательная"

                    st.info(f"Корреляция между количеством навыков и зарплатой: {corr:.2f} ({corr_text})")
                else:
                    st.info("Недостаточно данных для анализа связи между навыками и зарплатой")

            # Добавляем анализ навыков по уровням должности, если доступен
            if 'seniority_level' in filtered_df.columns:
                st.subheader("Навыки по уровням должности")

                # Подготавливаем данные
                skill_by_level = {}
                for level in filtered_df['seniority_level'].dropna().unique():
                    level_skills = []
                    for skills in filtered_df[filtered_df['seniority_level'] == level]['skills'].dropna():
                        if isinstance(skills, str) and skills:
                            level_skills.extend([skill.strip() for skill in skills.split(',') if skill.strip()])

                    if level_skills:
                        skill_by_level[level] = pd.Series(level_skills).value_counts().head(10)

                if skill_by_level:
                    # Создаем столбцы для каждого уровня
                    level_cols = st.columns(min(len(skill_by_level), 3))

                    for i, (level, skills) in enumerate(skill_by_level.items()):
                        col_idx = i % len(level_cols)
                        with level_cols[col_idx]:
                            fig = px.bar(
                                x=skills.values,
                                y=skills.index,
                                orientation='h',
                                labels={'x': 'Количество', 'y': 'Навык'},
                                title=f'Топ навыки для уровня {level}',
                                color=skills.values,
                                color_continuous_scale='Plasma'
                            )
                            fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("В выбранных данных недостаточно информации о навыках для анализа")

    # Вкладка "Отчет"
    with tab4:
        st.subheader("Ключевые выводы")

        # Основные факты
        st.markdown(f"""
        ### Основные факты:

        1. **Всего проанализировано вакансий:** {filtered_df.shape[0]}
        2. **Количество уникальных компаний:** {filtered_df['company'].nunique()}
        3. **Количество локаций:** {filtered_df['city'].nunique()}
        """)

        # Зарплаты
        if 'salary_rub' in filtered_df.columns:
            salary_data = filtered_df[filtered_df['salary_rub'].notnull()]
            if not salary_data.empty:
                st.markdown(f"""
                ### Статистика зарплат:

                - **Средняя зарплата:** {salary_data['salary_rub'].mean():,.0f} ₽
                - **Медианная зарплата:** {salary_data['salary_rub'].median():,.0f} ₽
                - **Минимальная зарплата:** {salary_data['salary_rub'].min():,.0f} ₽
                - **Максимальная зарплата:** {salary_data['salary_rub'].max():,.0f} ₽
                """)

        # Формат работы
        if remote_col:
            remote_pct = filtered_df[remote_col].mean() * 100
            st.markdown(f"""
            ### Формат работы:

            - **Удаленная работа:** {remote_pct:.1f}% вакансий
            - **Офисная работа:** {100 - remote_pct:.1f}% вакансий
            """)

        # Топ навыков
        if all_skills:
            st.markdown("### Топ-10 востребованных навыков:")

            top_skills = pd.Series(all_skills).value_counts().head(10)
            skills_md = ""
            for i, (skill, count) in enumerate(top_skills.items(), 1):
                skills_md += f"{i}. **{skill}:** {count} упоминаний\n"

            st.markdown(skills_md)

        # Корреляции
        if 'skills_count' in filtered_df.columns and 'salary_rub' in filtered_df.columns:
            corr_data = filtered_df[filtered_df['skills_count'].notnull() & filtered_df['salary_rub'].notnull()]
            if not corr_data.empty and len(corr_data) >= 5:
                corr = corr_data[['skills_count', 'salary_rub']].corr().iloc[0, 1]
                st.markdown(f"""
                ### Корреляции:

                - **Корреляция между количеством навыков и зарплатой:** {corr:.2f}
                """)

        # Основные выводы
        st.subheader("Выводы и рекомендации")

        st.markdown("""
        ### Выводы:

        1. **Географическая концентрация:** Большинство вакансий сконцентрировано в крупных городах, с преобладанием в Москве и Санкт-Петербурге, что отражает централизованную структуру технологического рынка труда в России.

        2. **Трансформация рабочих форматов:** Наблюдается значительная доля удаленных вакансий, что свидетельствует о структурной перестройке организационных моделей в технологическом секторе.

        3. **Компетентностная дифференциация:** Данные демонстрируют иерархическую структуру навыков с доминированием Python и SQL, формирующих базовый инструментальный фундамент специалиста в сфере данных.

        4. **Экономическая стратификация:** Анализ зарплатного распределения выявляет выраженную асимметрию с концентрацией предложений в среднем сегменте и наличием премиальных позиций.

        ### Рекомендации:

        1. **Для соискателей:** Сфокусироваться на развитии ключевых навыков (Python, SQL) с дополнительной специализацией в инфраструктурных технологиях (PostgreSQL, Docker, Git), повышающих интегральную ценность специалиста.

        2. **Для работодателей:** Диверсифицировать форматы работы, предлагая гибкие и удаленные модели для расширения пула квалифицированных кандидатов и снижения географических ограничений найма.

        3. **Для образовательных учреждений:** Адаптировать учебные программы с акцентом на интеграцию технических навыков с системным мышлением и проектным опытом, отражая комплексный характер современных требований индустрии.

        4. **Для исследователей рынка:** Расширить аналитическую базу через включение данных с других платформ рекрутинга для формирования более полной картины технологического рынка труда.
        """)


# Запуск приложения
if __name__ == "__main__":
    main()