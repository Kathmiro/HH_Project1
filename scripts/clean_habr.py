#!/usr/bin/env python3
"""clean_habr.py — интеллектуальная очистка и нормализация данных о вакансиях.

Осуществляет многоуровневую обработку JSONL-файлов с улучшенной сегментацией
категориальных данных, нормализацией типов и извлечением скрытых паттернов.

Запуск:
    python scripts/clean_habr.py
    python scripts/clean_habr.py --src "data/raw/*.jsonl" --out data/processed/vacancies.csv --eda
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

# Константы
SRC_MASK = "data/raw/*.jsonl"
OUT_FILE = "data/processed/vacancies.csv"

# Списки для нормализации и валидации данных
KNOWN_CITIES = {
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Нижний Новгород",
    "Казань", "Челябинск", "Омск", "Самара", "Ростов-на-Дону", "Уфа", "Красноярск",
    "Воронеж", "Пермь", "Волгоград", "Краснодар", "Саратов", "Тюмень", "Тольятти",
    "Ижевск", "Барнаул", "Иркутск", "Ульяновск", "Хабаровск", "Ярославль", "Владивосток",
    "Махачкала", "Томск", "Оренбург", "Кемерово", "Новокузнецк", "Рязань", "Астрахань",
    "Набережные Челны", "Пенза", "Киров", "Липецк", "Чебоксары", "Калининград", "Тула",
    "Ставрополь", "Дубна", "Иннополис", "Сочи", "Минск", "Алматы", "Киев", "Одесса"
}

EMPLOYMENT_TYPES = {
    "Полный рабочий день", "Неполный рабочий день", "Гибкий график",
    "Проектная работа", "Стажировка", "Контрактная работа", "Практика"
}

REMOTE_FORMATS = {
    "Можно удаленно", "Удаленная работа", "Remote", "Дистанционно",
    "Удаленно", "Удалённо", "Удаленка"
}

# Технологии для извлечения
TECH_KEYWORDS = [
    "python", "java", "javascript", "js", "typescript", "ts", "c\\+\\+", "c#", "go",
    "php", "ruby", "swift", "kotlin", "sql", "nosql", "postgres", "postgresql", "mongodb",
    "react", "angular", "vue", "node", "django", "flask", "spring", "fastapi",
    "docker", "kubernetes", "git", "ci/cd", "aws", "azure", "gcp", "spark", "hadoop",
    "machine learning", "ml", "искусственный интеллект", "ai", "data science",
    "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "matplotlib",
    "r", "scala", "dbt", "airflow", "kafka", "elasticsearch", "redis", "clickhouse",
    "backend", "frontend", "fullstack", "devops", "qa", "тестирование",
    "linux", "unix", "bash", "powershell", "excel", "tableau", "power bi", "looker",
    "rest api", "graphql", "microservices", "rabbitmq", "etl", "eda"
]


def read_jsonl(path: Path) -> List[dict]:
    """Читает данные из JSONL файла с обработкой ошибок."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp if line.strip()]
    except Exception as e:
        print(f"Ошибка чтения файла {path}: {str(e)}")
        return []


def normalize_city(value: Any) -> str | None:
    """Нормализует значение города, отделяя его от формата работы."""
    if pd.isna(value) or value is None:
        return None

    value_str = str(value).strip()

    # Проверка на известный город
    if value_str in KNOWN_CITIES:
        return value_str

    # Отфильтровываем форматы работы
    if value_str in EMPLOYMENT_TYPES or value_str in REMOTE_FORMATS:
        return None

    # Проверка на смешанные форматы
    for format_text in list(EMPLOYMENT_TYPES) + list(REMOTE_FORMATS):
        if format_text in value_str:
            # Извлекаем город из смешанного формата
            city_part = value_str.replace(format_text, "").strip()
            city_part = re.sub(r'[,•.]', '', city_part).strip()

            if city_part in KNOWN_CITIES:
                return city_part

    # Проверка на частичное совпадение с известными городами
    for city in KNOWN_CITIES:
        if city in value_str or value_str in city:
            return city

    # Если явно указано "удаленка" - нет города
    for remote_keyword in ["удален", "remote", "дистанц"]:
        if remote_keyword in value_str.lower():
            return None

    # Возвращаем как есть, если не удалось нормализовать
    return value_str


def detect_remote(row: pd.Series) -> bool:
    """Определяет, является ли вакансия удаленной на основе нескольких полей."""
    # Прямое указание удаленной работы
    if pd.notnull(row.get('remote')) and row['remote'] is True:
        return True

    # Проверка города на указание удаленной работы
    city_str = str(row.get('city', '')).lower() if pd.notnull(row.get('city')) else ''
    if any(keyword in city_str for keyword in ['удален', 'remote', 'дистанц']):
        return True

    # Проверка employment на указание удаленной работы
    employment_str = str(row.get('employment', '')).lower() if pd.notnull(row.get('employment')) else ''
    if any(keyword in employment_str for keyword in ['удален', 'remote', 'дистанц']):
        return True

    # Проверка описания и условий
    for field in ['description', 'requirements', 'conditions']:
        if pd.notnull(row.get(field)):
            text = str(row[field]).lower()
            remote_phrases = ['удаленная работа', 'удаленно', 'remote work', 'работа из дома']
            if any(phrase in text for phrase in remote_phrases):
                return True

    return False


def normalize_employment(value: Any) -> str | None:
    """Нормализует тип занятости, отделяя его от других данных."""
    if pd.isna(value) or value is None:
        return None

    value_str = str(value).strip()

    # Проверка на известный тип занятости
    if value_str in EMPLOYMENT_TYPES:
        return value_str

    # Проверка на формат удаленной работы
    if value_str in REMOTE_FORMATS:
        return "Удаленная работа"

    # Извлечение известных типов занятости из текста
    for emp_type in EMPLOYMENT_TYPES:
        if emp_type in value_str:
            return emp_type

    # Проверка на ключевые слова
    if re.search(r'\bполн\w+\s+рабоч\w+\s+день\b', value_str, re.I):
        return "Полный рабочий день"
    elif re.search(r'\bнеполн\w+\s+рабоч\w+\s+день\b', value_str, re.I):
        return "Неполный рабочий день"
    elif re.search(r'\bстажир\w+\b', value_str, re.I):
        return "Стажировка"
    elif re.search(r'\bгибк\w+\s+график\b', value_str, re.I):
        return "Гибкий график"
    elif re.search(r'\bудаленн\w+\b|\bremote\b', value_str, re.I):
        return "Удаленная работа"

    return value_str


def extract_tech_skills(text: str) -> List[str]:
    """Извлекает технические навыки из текста."""
    if not isinstance(text, str) or not text:
        return []

    text = text.lower()
    found_skills = []

    # Поиск по ключевым словам
    for keyword in TECH_KEYWORDS:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text):
            found_skills.append(keyword)

    # Поиск по шаблонам
    skill_patterns = [
        r'знание ([\w\s,+#\.]+?)(;|$|\.|\))',
        r'опыт работы с ([\w\s,+#\.]+?)(;|$|\.|\))',
        r'навыки ([\w\s,+#\.]+?)(;|$|\.|\))',
        r'технологии: ([\w\s,+#\.]+?)(;|$|\.|\))',
        r'стек: ([\w\s,+#\.]+?)(;|$|\.|\))',
    ]

    for pattern in skill_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            skill_text = match.group(1).strip()
            # Разбиваем по запятым и другим разделителям
            skills = re.split(r'[,;/]', skill_text)
            for skill in skills:
                skill = skill.strip().lower()
                if skill and len(skill) > 2:
                    found_skills.append(skill)

    return list(set(found_skills))


def extract_experience(text: str) -> int | None:
    """Извлекает опыт работы из текста требований с расширенным распознаванием."""
    if not isinstance(text, str) or not text:
        return None

    text = text.lower()

    # Расширенные шаблоны для поиска упоминаний опыта
    patterns = [
        r'опыт\s+работы\s+от\s+(\d+)\s*(?:года|лет|год)',
        r'опыт\s+работы\s+(\d+)(?:\+)?\s*(?:года|лет|год)',
        r'опыт\s+от\s+(\d+)\s*(?:года|лет|год)',
        r'опыт\s+(\d+)(?:\+)?\s*(?:года|лет|год)',
        r'стаж\s+от\s+(\d+)\s*(?:года|лет|год)',
        r'стаж\s+работы\s+(\d+)\s*(?:года|лет|год)',
        r'не\s+менее\s+(\d+)\s*(?:года|лет|год)',
        r'минимум\s+(\d+)\s*(?:года|лет|год)',
        r'(\d+)\s*(?:года|лет|год)\s+опыта',
        r'experience\s+of\s+(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s+experience',
        r'at\s+least\s+(\d+)\s*years?',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                years = int(match.group(1))
                # Валидация: опыт обычно от 0 до 20 лет
                if 0 <= years <= 20:
                    return years
            except (ValueError, IndexError):
                pass

    # Поиск словесных указаний на опыт
    if re.search(r'\bначинающ\w+\b|\bджуниор\b|\bjunior\b|\bentry[- ]level\b|\bстажер\b|\bстажёр\b', text):
        return 0
    elif re.search(r'\bмладш\w+\b|\bджун\w+\b', text):
        return 1
    elif re.search(r'\bсредн\w+\b|\bмидл\w+\b|\bmiddle\b', text):
        return 2
    elif re.search(r'\bстарш\w+\b|\bсеньор\w+\b|\bсиньор\w+\b|\bsenior\b', text):
        return 3
    elif re.search(r'\bлид\w*\b|\bведущ\w+\b|\bteam[- ]?lead\b|\bруководител\w+\b', text):
        return 5

    return None


def calc_salary(row: pd.Series) -> float | None:
    """Рассчитывает оценку зарплаты с учетом различных форматов данных."""
    # Проверка на существование полей
    if 'salary_from' not in row and 'salary_to' not in row:
        return None

    from_val = row.get('salary_from')
    to_val = row.get('salary_to')

    # Проверка валидности
    if pd.isna(from_val):
        from_val = None
    if pd.isna(to_val):
        to_val = None

    # Расчет в зависимости от доступных данных
    if from_val is not None and to_val is not None:
        # Средняя между нижней и верхней границей
        return (float(from_val) + float(to_val)) / 2
    elif from_val is not None:
        # Только нижняя граница (+15% как оценка)
        return float(from_val) * 1.15
    elif to_val is not None:
        # Только верхняя граница (-15% как оценка)
        return float(to_val) * 0.85

    return None


def to_rub(row: pd.Series) -> float | None:
    """Конвертирует зарплату в рубли с актуальными курсами валют."""
    if 'salary_estimated' not in row or pd.isna(row['salary_estimated']):
        return None

    if 'currency' not in row or pd.isna(row['currency']):
        # Если валюта не указана, предполагаем рубли
        return float(row['salary_estimated'])

    # Актуальные курсы валют (май 2025)
    rates = {
        "RUB": 1.0,
        "USD": 92.5,
        "EUR": 100.2,
        "GBP": 118.3,
        "JPY": 0.61,
        "CHF": 102.8,
        "CNY": 12.7,
        "KZT": 0.21,
    }

    currency = str(row['currency']).upper()

    # Проверяем наличие курса валюты
    if currency in rates:
        return float(row['salary_estimated']) * rates[currency]

    # Если валюта неизвестна, возвращаем как есть
    return float(row['salary_estimated'])


def categorize_salary(salary: float | None) -> str | None:
    """Категоризирует зарплату по диапазонам."""
    if salary is None or pd.isna(salary):
        return None

    if salary < 100000:
        return "до 100K"
    elif salary < 200000:
        return "100K-200K"
    elif salary < 300000:
        return "200K-300K"
    elif salary < 500000:
        return "300K-500K"
    else:
        return "от 500K"


def categorize_experience(years: float | None) -> str | None:
    """Категоризирует опыт работы по уровням."""
    if years is None or pd.isna(years):
        return None

    if years < 1:
        return "Без опыта"
    elif years < 3:
        return "Junior (1-2 года)"
    elif years < 5:
        return "Middle (3-4 года)"
    elif years < 7:
        return "Senior (5-6 лет)"
    else:
        return "Lead (7+ лет)"


def detect_seniority(row: pd.Series) -> str | None:
    """Определяет уровень позиции на основе заголовка и требований."""
    # Если уже есть опыт в годах, используем его для определения уровня
    if pd.notnull(row.get('experience_years')):
        exp_years = row['experience_years']
        if exp_years < 1:
            return "Junior"
        elif exp_years < 3:
            return "Junior"
        elif exp_years < 5:
            return "Middle"
        elif exp_years < 7:
            return "Senior"
        else:
            return "Lead"

    # Поиск в заголовке
    title = str(row.get('title', '')).lower() if pd.notnull(row.get('title')) else ''

    if re.search(r'\bjunior\b|\bджуниор\b|\bмладший\b|\bстажер\b|\bстажёр\b|\binternet\b', title):
        return "Junior"
    elif re.search(r'\bmiddle\b|\bмидл\b|\bсредний\b', title):
        return "Middle"
    elif re.search(r'\bsenior\b|\bсеньор\b|\bстарший\b|\bведущий\b|\bведущ\b|\blead\b', title):
        return "Senior"
    elif re.search(r'\bteam\s*lead\b|\bруководитель\b|\bhead\b|\bдиректор\b|\barchitect\b|\barchitecture\b', title):
        return "Lead"

    # Если не нашли в заголовке, проверяем требования
    requirements = str(row.get('requirements', '')).lower() if pd.notnull(row.get('requirements')) else ''

    if re.search(r'\bопыт работы от 5\b|\bопыт от 5\b|\bmore than 5 years\b|\bот 5 лет\b', requirements):
        return "Senior"
    elif re.search(r'\bопыт работы от 3\b|\bопыт от 3\b|\bmore than 3 years\b|\bот 3 лет\b', requirements):
        return "Middle"
    elif re.search(r'\bопыт работы от 1\b|\bопыт от 1\b|\bmore than 1 year\b|\bот 1 года\b', requirements):
        return "Junior"
    elif re.search(r'\bбез опыта\b|\bno experience\b|\binternet\b|\bстажировка\b', requirements):
        return "Junior"

    # По умолчанию
    return None


def extract_job_function(title: str) -> str | None:
    """Извлекает основную функцию из заголовка вакансии."""
    if not isinstance(title, str) or not title:
        return None

    title = title.lower()

    # Словарь категорий и соответствующих им ключевых слов
    categories = {
        "Data Engineer": ["data engineer", "инженер данных", "специалист по данным", "дата инженер"],
        "Data Scientist": ["data scientist", "data science", "дата сайентист", "специалист по data science"],
        "ML Engineer": ["ml engineer", "machine learning engineer", "инженер машинного обучения"],
        "Data Analyst": ["data analyst", "аналитик данных", "дата аналитик"],
        "BI Developer": ["bi developer", "bi-developer", "bi аналитик", "business intelligence"],
        "Backend Developer": ["backend", "back-end", "бэкенд", "серверный разработчик"],
        "Frontend Developer": ["frontend", "front-end", "фронтенд", "клиентский разработчик"],
        "Fullstack Developer": ["fullstack", "full-stack", "фулстак", "фуллстак"],
        "DevOps Engineer": ["devops", "девопс", "infrastructure engineer"],
        "QA Engineer": ["qa engineer", "тестировщик", "quality assurance", "тестирование"],
        "Product Manager": ["product manager", "продуктовый менеджер", "менеджер продукта"],
        "Project Manager": ["project manager", "проектный менеджер", "менеджер проекта", "руководитель проекта"],
        "Team Lead": ["team lead", "tech lead", "тимлид", "лид команды", "руководитель команды"],
        "UI/UX Designer": ["ui/ux", "ux/ui", "ui designer", "ux designer", "дизайнер интерфейсов"],
        "Database Administrator": ["dba", "database administrator", "администратор баз данных"],
        "System Administrator": ["system administrator", "системный администратор"],
        "IT Security": ["security", "безопасность", "защита информации", "информационная безопасность"]
    }

    # Проверка на соответствие категориям
    for category, keywords in categories.items():
        if any(keyword in title for keyword in keywords):
            return category

    # Если не нашли конкретную категорию, попробуем определить более общую
    if any(tech in title for tech in ["разработчик", "developer", "программист", "coder"]):
        return "Software Developer"
    elif any(tech in title for tech in ["аналитик", "analyst"]):
        return "Analyst"
    elif any(tech in title for tech in ["инженер", "engineer"]):
        return "Engineer"
    elif any(tech in title for tech in ["менеджер", "manager", "руководитель", "head"]):
        return "Manager"

    return None


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает дополнительные признаки из данных."""
    df = df.copy()

    # Преобразуем категориальные поля и извлекаем признаки
    print("Нормализация полей и извлечение признаков...")

    # Нормализация города
    if 'city' in df.columns:
        df['city_normalized'] = df['city'].apply(normalize_city)

    # Нормализация занятости
    if 'employment' in df.columns:
        df['employment_normalized'] = df['employment'].apply(normalize_employment)

    # Определение удаленной работы
    df['remote_work'] = df.apply(detect_remote, axis=1)

    # Извлечение опыта работы
    if 'requirements' in df.columns:
        df['experience_years'] = df['requirements'].apply(extract_experience)

    # Извлечение технологий из текстовых полей
    for field in ['description', 'requirements', 'conditions']:
        if field in df.columns:
            df[f'{field}_techs'] = df[field].apply(extract_tech_skills)

    # Объединение всех найденных технологий
    tech_cols = [col for col in df.columns if col.endswith('_techs')]
    if tech_cols:
        df['all_techs'] = df.apply(
            lambda row: list(set(
                sum([row.get(col, []) for col in tech_cols if isinstance(row.get(col), list)], [])
            )),
            axis=1
        )

    # Расчет зарплаты
    if all(col in df.columns for col in ['salary_from', 'salary_to']):
        df['salary_estimated'] = df.apply(calc_salary, axis=1)

        # Конвертация в рубли
        if 'currency' in df.columns:
            df['salary_rub'] = df.apply(to_rub, axis=1)

            # Категоризация зарплаты
            df['salary_category'] = df['salary_rub'].apply(categorize_salary)

    # Категоризация опыта
    if 'experience_years' in df.columns:
        df['experience_category'] = df['experience_years'].apply(categorize_experience)

    # Определение уровня позиции
    df['seniority_level'] = df.apply(detect_seniority, axis=1)

    # Извлечение функции из заголовка
    if 'title' in df.columns:
        df['job_function'] = df['title'].apply(extract_job_function)

    # Подсчет количества скиллов
    if 'skills' in df.columns:
        df['skills_count'] = df['skills'].apply(
            lambda x: len(x.split(',')) if isinstance(x, str) else 0
        )

    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует и очищает данные."""
    print("Приведение типов данных...")

    # Приводим типы данных
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

    for col in ('salary_from', 'salary_to'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'remote' in df.columns:
        df['remote'] = df['remote'].astype('boolean')

    # Заполняем пропуски в базовых полях
    df['company'] = df.get('company', pd.Series()).fillna('N/A')

    # Обработка списков скиллов
    if 'skills' in df.columns:
        # Преобразование списка в строку через запятую
        df['skills'] = df['skills'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else (x or '')
        )

    # Очистка текстовых полей
    for field in ['description', 'requirements', 'conditions']:
        if field in df.columns:
            # Очистка от HTML тегов и лишних пробелов
            df[field] = df[field].str.replace(r'<[^>]+>', ' ', regex=True)
            df[field] = df[field].str.replace(r'\s+', ' ', regex=True).str.strip()
            df[field] = df[field].fillna('')

    # Извлекаем дополнительные признаки
    df = extract_features(df)

    # Замена основных полей нормализованными версиями
    if 'city_normalized' in df.columns:
        df['city_original'] = df['city']  # Сохраняем оригинал
        df['city'] = df['city_normalized']
        df.drop('city_normalized', axis=1, inplace=True)

    if 'employment_normalized' in df.columns:
        df['employment_original'] = df['employment']  # Сохраняем оригинал
        df['employment'] = df['employment_normalized']
        df.drop('employment_normalized', axis=1, inplace=True)

    # Преобразуем списки в строки для CSV
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )

    return df


def show_eda(df: pd.DataFrame) -> None:
    """Выводит расширенную аналитическую статистику по данным."""
    print("\n" + "=" * 50)
    print("БАЗОВАЯ СТАТИСТИКА")
    print("=" * 50)

    print(f"\nЗаписей: {len(df)}, Столбцов: {len(df.columns)}")

    print("\nТИПЫ ДАННЫХ:")
    for col, dtype in df.dtypes.items():
        print(f"{col}: {dtype}")

    print("\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Количество': missing,
        'Процент': missing_pct
    }).sort_values('Количество', ascending=False)
    print(missing_df[missing_df['Количество'] > 0])

    print("\n" + "=" * 50)
    print("РАСПРЕДЕЛЕНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ")
    print("=" * 50)

    # Распределение по городам
    if 'city' in df.columns:
        print("\nТОП-10 ГОРОДОВ:")
        city_counts = df['city'].value_counts().head(10)
        for city, count in city_counts.items():
            if pd.notnull(city) and city:
                print(f"{city}: {count} ({count / len(df):.1%})")

    # Распределение по компаниям
    if 'company' in df.columns:
        print("\nТОП-10 КОМПАНИЙ:")
        company_counts = df['company'].value_counts().head(10)
        for company, count in company_counts.items():
            print(f"{company}: {count} ({count / len(df):.1%})")

    # Распределение по формату работы
    if 'remote_work' in df.columns:
        print("\nФОРМАТ РАБОТЫ:")
        remote_counts = df['remote_work'].value_counts()
        for remote, count in remote_counts.items():
            print(f"{'Удаленно' if remote else 'Офис'}: {count} ({count / len(df):.1%})")

    # Распределение по типу занятости
    if 'employment' in df.columns:
        print("\nТИП ЗАНЯТОСТИ:")
        emp_counts = df['employment'].value_counts().head(10)
        for emp, count in emp_counts.items():
            if pd.notnull(emp) and emp:
                print(f"{emp}: {count} ({count / len(df):.1%})")

    print("\n" + "=" * 50)
    print("АНАЛИЗ ЧИСЛОВЫХ ДАННЫХ")
    print("=" * 50)

    # Статистика зарплат
    if 'salary_rub' in df.columns:
        salary_data = df[df['salary_rub'].notnull()]
        if not salary_data.empty:
            print("\nСТАТИСТИКА ЗАРПЛАТ (руб):")
            salary_stats = salary_data['salary_rub'].describe()
            for stat, value in salary_stats.items():
                print(f"{stat}: {value:,.0f}")

            if 'salary_category' in df.columns:
                print("\nРАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ ЗАРПЛАТ:")
                category_counts = df['salary_category'].value_counts().sort_index()
                for category, count in category_counts.items():
                    if pd.notnull(category):
                        print(f"{category}: {count} ({count / len(df):.1%})")

    # Статистика опыта работы
    if 'experience_years' in df.columns:
        exp_data = df[df['experience_years'].notnull()]
        if not exp_data.empty:
            print("\nСТАТИСТИКА ТРЕБУЕМОГО ОПЫТА РАБОТЫ (лет):")
            exp_stats = exp_data['experience_years'].describe()
            for stat, value in exp_stats.items():
                print(f"{stat}: {value:.1f}")

            print("\nРАСПРЕДЕЛЕНИЕ ПО ОПЫТУ РАБОТЫ:")
            exp_counts = df['experience_years'].value_counts().sort_index()
            for exp, count in exp_counts.items():
                if pd.notnull(exp):
                    print(f"{exp} лет: {count} ({count / len(df):.1%})")

    # Распределение по уровню должности
    if 'seniority_level' in df.columns:
        print("\nРАСПРЕДЕЛЕНИЕ ПО УРОВНЮ ДОЛЖНОСТИ:")
        level_counts = df['seniority_level'].value_counts()
        for level, count in level_counts.items():
            if pd.notnull(level):
                print(f"{level}: {count} ({count / len(df):.1%})")

    # Распределение по функциям
    if 'job_function' in df.columns:
        print("\nРАСПРЕДЕЛЕНИЕ ПО ФУНКЦИЯМ:")
        function_counts = df['job_function'].value_counts().head(10)
        for function, count in function_counts.items():
            if pd.notnull(function):
                print(f"{function}: {count} ({count / len(df):.1%})")

    print("\n" + "=" * 50)
    print("АНАЛИЗ ТЕХНИЧЕСКИХ НАВЫКОВ")
    print("=" * 50)

    # Популярные технологии
    if 'all_techs' in df.columns:
        print("\nТОП-20 ТЕХНОЛОГИЙ:")
        # Собираем все технологии из строк с разделителями
        all_tech_counts = Counter()
        for techs_str in df['all_techs'].dropna():
            if isinstance(techs_str, str) and techs_str.strip():
                techs = [tech.strip() for tech in techs_str.split(',')]
                all_tech_counts.update(techs)

        # Выводим топ технологий
        for tech, count in all_tech_counts.most_common(20):
            print(f"{tech}: {count} ({count / sum(all_tech_counts.values()):.1%})")

    # Корреляции с зарплатой
    if 'salary_rub' in df.columns and 'experience_years' in df.columns:
        salary_exp_data = df[df['salary_rub'].notnull() & df['experience_years'].notnull()]
        if len(salary_exp_data) > 5:
            print("\nКОРРЕЛЯЦИЯ МЕЖДУ ОПЫТОМ И ЗАРПЛАТОЙ:")
            corr = salary_exp_data[['experience_years', 'salary_rub']].corr().iloc[0, 1]
            print(f"Коэффициент корреляции: {corr:.3f}")

            if corr > 0.7:
                print("Сильная положительная корреляция: зарплата значительно растет с опытом")
            elif corr > 0.3:
                print("Умеренная положительная корреляция: зарплата растет с опытом")
            elif corr > 0:
                print("Слабая положительная корреляция: зарплата немного растет с опытом")
            elif corr > -0.3:
                print("Слабая отрицательная корреляция: зарплата немного снижается с опытом")
            else:
                print("Умеренная или сильная отрицательная корреляция: необычная тенденция")

    if 'skills_count' in df.columns and 'salary_rub' in df.columns:
        skills_salary_data = df[df['salary_rub'].notnull() & df['skills_count'].notnull()]
        if len(skills_salary_data) > 5:
            print("\nКОРРЕЛЯЦИЯ МЕЖДУ КОЛИЧЕСТВОМ НАВЫКОВ И ЗАРПЛАТОЙ:")
            corr = skills_salary_data[['skills_count', 'salary_rub']].corr().iloc[0, 1]
            print(f"Коэффициент корреляции: {corr:.3f}")

    print("\n" + "=" * 50)
    print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА")
    print("=" * 50)

    # Распределение удаленки по городам
    if 'city' in df.columns and 'remote_work' in df.columns:
        print("\nУДАЛЕННАЯ РАБОТА ПО ГОРОДАМ:")
        for city in df['city'].value_counts().head(5).index:
            if pd.notnull(city) and city:
                city_data = df[df['city'] == city]
                remote_count = city_data['remote_work'].sum()
                total_count = len(city_data)
                if total_count > 0:
                    remote_percent = (remote_count / total_count) * 100
                    print(f"{city}: {remote_count}/{total_count} ({remote_percent:.1f}% удаленно)")

    # Распределение зарплат по уровням должности
    if 'seniority_level' in df.columns and 'salary_rub' in df.columns:
        print("\nСРЕДНИЕ ЗАРПЛАТЫ ПО УРОВНЯМ ДОЛЖНОСТИ:")
        for level in df['seniority_level'].dropna().unique():
            level_data = df[(df['seniority_level'] == level) & df['salary_rub'].notnull()]
            if not level_data.empty:
                avg_salary = level_data['salary_rub'].mean()
                print(f"{level}: {avg_salary:,.0f} руб. ({len(level_data)} вакансий)")


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(description="Очистка данных Habr Career")
    p.add_argument("--src", type=str, default=SRC_MASK, help="маска JSONL-файлов")
    p.add_argument("--out", type=str, default=OUT_FILE, help="выходной файл")
    p.add_argument("--eda", action="store_true", help="вывести расширенную статистику")
    p.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="кодировка для CSV (по умолчанию utf-8-sig для корректного отображения кириллицы)"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="ограничение количества обрабатываемых записей (0 = все)"
    )
    return p.parse_args()


def main() -> None:
    """Основная функция скрипта."""
    args = parse_args()
    files = [Path(p) for p in glob.glob(args.src)]
    if not files:
        print("Ошибка: Нет входных файлов.")
        return

    print(f"Найдено {len(files)} файлов для обработки.")

    # Чтение всех JSONL файлов
    records = []
    for fp in tqdm(files, desc="Чтение файлов"):
        batch = read_jsonl(fp)
        if batch:
            records.extend(batch)
            print(f"  * {fp.name}: {len(batch)} записей")
        else:
            print(f"  * {fp.name}: ошибка чтения или пустой файл")

    if not records:
        print("Ошибка: Не удалось прочитать данные из файлов.")
        return

    print(f"Прочитано {len(records):,} записей из {len(files)} файлов")

    # Применяем лимит если задан
    if args.limit > 0 and args.limit < len(records):
        print(f"Применяется ограничение: обработка {args.limit} из {len(records)} записей")
        records = records[:args.limit]

    # Создаем DataFrame и обрабатываем данные
    df = pd.DataFrame(records)

    # Сохраняем исходное количество колонок для отчетности
    original_columns = len(df.columns)

    # Обработка данных
    df = normalize(df)

    # Вывод статистики
    if args.eda:
        show_eda(df)

    # Сохранение результата
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        # Используем правильную кодировку для сохранения CSV
        df.to_csv(out_path, index=False, encoding=args.encoding)

    added_columns = len(df.columns) - original_columns
    print(f"\nСохранено {len(df):,} строк и {len(df.columns)} колонок (+{added_columns} новых признаков) → {out_path}")
    print(f"Кодировка файла: {args.encoding}")


if __name__ == "__main__":
    main()
