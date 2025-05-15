#!/usr/bin/env python3
"""collect_habr.py — скрейпер вакансий Habr Career.

Интеллектуальный сбор и структурирование данных с платформы Habr Career
с корректным разделением локаций и форматов работы.

Примеры запуска:
    python scripts/collect_habr.py --pages 50 -q "data science"
    python scripts/collect_habr.py --pages 100 -q python --detailed
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import requests
from bs4 import BeautifulSoup, Tag
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential
from tqdm import tqdm

# Константы
BASE_URL = "https://career.habr.com/vacancies"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}
DATA_DIR = Path("data/raw")
PAUSE = 0.5  # сек между запросами
DETAIL_PAUSE = 1.5  # сек между запросами детальных страниц

# Список известных городов России для валидации
KNOWN_CITIES = {
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Нижний Новгород",
    "Казань", "Челябинск", "Омск", "Самара", "Ростов-на-Дону", "Уфа", "Красноярск",
    "Воронеж", "Пермь", "Волгоград", "Краснодар", "Саратов", "Тюмень", "Тольятти",
    "Ижевск", "Барнаул", "Иркутск", "Ульяновск", "Хабаровск", "Ярославль", "Владивосток",
    "Махачкала", "Томск", "Оренбург", "Кемерово", "Новокузнецк", "Рязань", "Астрахань",
    "Набережные Челны", "Пенза", "Киров", "Липецк", "Чебоксары", "Калининград", "Тула",
    "Ставрополь", "Дубна", "Иннополис", "Сочи", "Минск", "Алматы", "Киев", "Одесса"
}

# Форматы работы и занятости для фильтрации
EMPLOYMENT_FORMATS = {
    "Полный рабочий день", "Неполный рабочий день", "Можно удаленно",
    "Удаленная работа", "Гибридный график", "Контрактная работа",
    "Стажировка", "Проектная работа", "Практика",
}


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
def fetch(url: str) -> requests.Response:
    """Выполняет HTTP запрос с адаптивными повторными попытками."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp
    except Exception as e:
        tqdm.write(f"Ошибка запроса {url}: {str(e)}")
        raise


def parse_salary(text: str) -> Tuple[int | None, int | None, str | None]:
    """Разбирает строку зарплаты в числа и валюту с расширенной поддержкой форматов."""
    if not isinstance(text, str) or not text.strip() or "договор" in text.lower():
        return None, None, None

    # Нормализация текста
    text = text.replace("\u202f", " ").replace("\xa0", " ").strip()

    # Определение валюты
    cur_map = {"₽": "RUB", "$": "USD", "€": "EUR", "руб": "RUB", "dollar": "USD", "евро": "EUR"}
    currency = next((code for sign, code in cur_map.items() if sign in text), None)

    # Очистка текста от символов валют
    for sign in cur_map:
        text = text.replace(sign, "")

    # Извлечение числовых значений
    nums = [int(n.replace(" ", "")) for n in re.findall(r"\d[\d ]*", text)]

    if not nums:
        return None, None, currency

    # Определение "от" и "до" в зарплате
    if len(nums) == 1:
        if re.search(r"от|\bfrom\b", text.lower()):
            return nums[0], None, currency
        elif re.search(r"до|\bto\b|не более", text.lower()):
            return None, nums[0], currency
        else:
            return nums[0], nums[0], currency  # Точное значение

    # Если два числа - диапазон
    if len(nums) >= 2:
        return nums[0], nums[1], currency

    return None, None, currency


def build_url(page: int, query: str | None) -> str:
    """Строит URL для запроса списка вакансий."""
    params = {"page": str(page), "type": "all", "sort": "date"}
    if query:
        params["q"] = query
    return f"{BASE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"


def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Удаляет пустые значения из словаря."""
    return {k: v for k, v in d.items() if v not in (None, "", [], set())}


def normalize_city(location: str) -> Tuple[str | None, str | None]:
    """
    Нормализует информацию о локации, разделяя город и формат работы.

    Returns:
        Tuple[str | None, str | None]: (город, формат_работы)
    """
    if not location:
        return None, None

    location = location.strip()

    # Проверка на известный город
    if location in KNOWN_CITIES:
        return location, None

    # Проверка на формат работы
    if location in EMPLOYMENT_FORMATS:
        return None, location

    # Проверка на смешанные форматы
    for emp_format in EMPLOYMENT_FORMATS:
        if emp_format in location:
            # Извлечение города из смешанного формата
            city_part = location.replace(emp_format, "").strip()
            city_part = re.sub(r'[,•]', '', city_part).strip()

            if city_part and city_part in KNOWN_CITIES:
                return city_part, emp_format

            return None, emp_format

    # Эвристическое определение города
    for city in KNOWN_CITIES:
        if city in location or location in city:
            return city, None

    # Если не удалось определить однозначно
    return location, None


def get_section_text(soup: BeautifulSoup, title: str) -> str:
    """Извлекает текст секции из страницы вакансии."""
    try:
        # Поиск по нескольким возможным заголовкам
        possible_titles = [title]
        if title == "Описание вакансии":
            possible_titles.extend(["О вакансии", "О проекте", "О компании и вакансии"])
        elif title == "Ожидания от кандидата":
            possible_titles.extend(["Требования", "Требуемый опыт", "Мы ожидаем", "Требуемые навыки"])
        elif title == "Условия работы":
            possible_titles.extend(["Мы предлагаем", "Что мы предлагаем", "Бонусы и преимущества"])

        # Поиск по всем возможным заголовкам
        for possible_title in possible_titles:
            section = soup.find("h3", string=lambda s: s and possible_title.lower() in s.lower())
            if section:
                break

        if not section:
            return ""

        # Поиск содержимого секции
        content_div = section.find_next("div", class_="style-ugc")
        if not content_div:
            # Альтернативный поиск для других форматов
            content_div = section.find_next(["div", "ul"])
            if not content_div or content_div.name == "h3":
                return ""

        # Извлечение текста с сохранением структуры
        text = content_div.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        tqdm.write(f"Ошибка при извлечении секции {title}: {str(e)}")
        return ""


def extract_skills_from_text(text: str) -> List[str]:
    """Извлекает навыки из текста описания или требований."""
    if not text:
        return []

    # Шаблоны для поиска навыков
    skill_patterns = [
        r"знание ([\w\s,+#]+)",
        r"навыки ([\w\s,+#]+)",
        r"опыт работы с ([\w\s,+#]+)",
        r"владение ([\w\s,+#]+)",
        r"технологии: ([\w\s,+#]+)",
        r"стек: ([\w\s,+#]+)",
        r"технический стек: ([\w\s,+#]+)",
    ]

    extracted_skills = []

    for pattern in skill_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            skill_text = match.group(1).strip()
            skills = [s.strip() for s in re.split(r'[,;]', skill_text)]
            extracted_skills.extend([s for s in skills if len(s) > 2])

    return list(set(extracted_skills))


def parse_card(card: Tag) -> Dict[str, Any]:
    """Парсит карточку вакансии из списка с улучшенной обработкой метаданных."""
    try:
        # Извлечение заголовка и URL
        title_link = card.select_one("a.vacancy-card__title-link")
        if not title_link:
            return {}

        href = title_link.get("href", "")
        vac_id_match = re.search(r"/vacancies/(\d+)", href)
        vac_id = vac_id_match.group(1) if vac_id_match else href
        url = f"https://career.habr.com{href}"

        # Компания
        company_tag = card.select_one("div.vacancy-card__company-title a")
        company = company_tag.get_text(strip=True) if company_tag else None

        # Метаданные (город, тип работы)
        meta_tag = card.select_one("div.vacancy-card__meta")
        city = None
        employment = None
        remote = False

        if meta_tag:
            meta_text = meta_tag.get_text(" ", strip=True)

            # Признак удаленной работы
            remote = any(keyword in meta_text.lower() for keyword in ["удал", "remote", "дистанц"])

            # Разбор метаданных по разделителю
            parts = [p.strip() for p in meta_text.split("•") if p.strip()]

            if parts:
                # Первый элемент обычно город
                city_raw, emp_format = normalize_city(parts[0])
                city = city_raw

                # Если из первого элемента извлечен формат работы, используем его
                if emp_format:
                    employment = emp_format

                # Второй элемент обычно тип занятости
                if len(parts) > 1 and not employment:
                    employment = parts[1]

        # Зарплата
        salary_block = card.select_one("div.vacancy-card__salary")
        salary_raw = salary_block.get_text(" ", strip=True) if salary_block else ""
        salary_from, salary_to, currency = parse_salary(salary_raw)

        # Навыки
        skills_tags = card.select("div.vacancy-card__skills a")
        skills = [a.get_text(strip=True) for a in skills_tags]

        # Если навыков нет в блоке, пытаемся извлечь из описания
        if not skills:
            description_tag = card.select_one("div.vacancy-card__description")
            if description_tag:
                desc_text = description_tag.get_text(strip=True)
                extra_skills = extract_skills_from_text(desc_text)
                if extra_skills:
                    skills.extend(extra_skills)

        # Дата публикации
        time_tag = card.select_one("time")
        pub_date = time_tag.get("datetime") if time_tag and time_tag.has_attr("datetime") else None

        return clean_dict({
            "id": vac_id,
            "title": title_link.get_text(strip=True),
            "company": company,
            "city": city,
            "employment": employment,
            "remote": remote,
            "published_at": pub_date,
            "salary_from": salary_from,
            "salary_to": salary_to,
            "currency": currency,
            "skills": skills,
            "url": url,
        })
    except Exception as e:
        tqdm.write(f"Ошибка при парсинге карточки: {str(e)}")
        return {}


def get_details(url: str) -> Dict[str, Any]:
    """Парсит детальную информацию с страницы вакансии с расширенным извлечением данных."""
    try:
        soup = BeautifulSoup(fetch(url).text, "lxml")

        # Извлечение основных секций
        description = get_section_text(soup, "Описание вакансии")
        requirements = get_section_text(soup, "Ожидания от кандидата")
        conditions = get_section_text(soup, "Условия работы")

        # Дополнительное извлечение адреса, если доступно
        address_block = soup.select_one("div.content-section__title:-soup-contains('Местоположение')")
        address = None
        if address_block:
            address_section = address_block.find_parent("div", class_="content-section")
            if address_section:
                address_content = address_section.select_one("span.inline-list")
                if address_content:
                    address = address_content.get_text(strip=True)

        # Дополнительные навыки из текста описания и требований
        extra_skills = []
        if requirements:
            req_skills = extract_skills_from_text(requirements)
            if req_skills:
                extra_skills.extend(req_skills)

        if description:
            desc_skills = extract_skills_from_text(description)
            if desc_skills:
                extra_skills.extend(desc_skills)

        result = clean_dict({
            "description": description,
            "requirements": requirements,
            "conditions": conditions,
            "address": address,
            "extra_skills": list(set(extra_skills)) if extra_skills else None,
        })

        return result
    except Exception as e:
        tqdm.write(f"Ошибка при парсинге деталей {url}: {str(e)}")
        return {}


def scrape(pages: int, query: str | None, detailed: bool = False) -> List[Dict[str, Any]]:
    """Собирает данные о вакансиях с улучшенной обработкой и валидацией."""
    rows = []

    for page in tqdm(range(1, pages + 1), desc="Страницы", unit="стр"):
        url = build_url(page, query)

        try:
            response = fetch(url)
            soup = BeautifulSoup(response.text, "lxml")

            # Проверка на captcha или блокировку
            if "captcha" in response.text.lower() or "заблокирован" in response.text.lower():
                tqdm.write("Возможно сработала защита от ботов. Увеличиваем паузу.")
                time.sleep(PAUSE * 5)
                continue

            cards = soup.select("div.vacancy-card")

            if not cards:
                # Проверяем, действительно ли вакансий нет
                no_results = soup.select_one("div.no-content")
                if no_results:
                    tqdm.write(f"Стр. {page}: вакансий нет (конец)")
                    break
                else:
                    tqdm.write(
                        f"Стр. {page}: не удалось найти карточки вакансий, но нет сообщения об отсутствии результатов")
                    # Пробуем еще раз с другой паузой
                    time.sleep(PAUSE * 3)
                    continue

            # Парсинг карточек с валидацией
            page_rows = []
            for card in cards:
                card_data = parse_card(card)
                if card_data and "title" in card_data and "url" in card_data:
                    page_rows.append(card_data)

            if not page_rows:
                tqdm.write(f"Стр. {page}: не удалось получить данные из карточек")
                continue

            # Сбор детальной информации
            if detailed:
                with tqdm(total=len(page_rows), desc="Детали", leave=False) as pbar:
                    for row in page_rows:
                        if "url" in row:
                            details = get_details(row["url"])

                            # Обновление основных полей
                            row.update(details)

                            # Объединение основных и дополнительных навыков
                            if "extra_skills" in details and details["extra_skills"]:
                                if "skills" not in row:
                                    row["skills"] = []
                                row["skills"] = list(set(row["skills"] + details["extra_skills"]))
                                row.pop("extra_skills", None)

                            pbar.update(1)
                            time.sleep(DETAIL_PAUSE)

                tqdm.write(f"Стр. {page}: собрано {len(page_rows)} вакансий с деталями")
            else:
                tqdm.write(f"Стр. {page}: собрано {len(page_rows)} вакансий")

            rows.extend(page_rows)

            # Адаптивная пауза
            time.sleep(PAUSE + (page % 5) * 0.2)  # Небольшая вариация для избежания блокировки

        except Exception as e:
            tqdm.write(f"Ошибка при обработке страницы {page}: {str(e)}")
            time.sleep(PAUSE * 3)  # Увеличенная пауза при ошибке
            continue

    return rows


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    p = argparse.ArgumentParser(description="Сбор вакансий Habr Career")
    p.add_argument("--pages", type=int, default=50, help="Максимальное количество страниц (по умолчанию: 50)")
    p.add_argument("-q", "--query", type=str, default="python", help="Поисковый запрос (по умолчанию: 'python')")
    p.add_argument("--out", type=str, help="JSONL‑файл вывода")
    p.add_argument("--detailed", action="store_true", help="Собирать детальную информацию по каждой вакансии")
    p.add_argument("--pause", type=float, default=0.5, help="Пауза между запросами (сек)")
    p.add_argument("--detail-pause", type=float, default=1.5, help="Пауза между запросами детальных страниц (сек)")
    return p.parse_args()


def main() -> None:
    """Основная функция скрипта."""
    args = parse_args()

    # Установка пауз из параметров
    global PAUSE, DETAIL_PAUSE
    PAUSE = args.pause
    DETAIL_PAUSE = args.detail_pause

    fname = f"vacancies_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out = Path(args.out) if args.out else DATA_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"Собираем до {args.pages} страниц, запрос='{args.query}', detailed={args.detailed} → {out}"
    )
    data = scrape(args.pages, args.query, args.detailed)

    if not data:
        tqdm.write("Предупреждение: не удалось собрать данные!")
        return

    with out.open("w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    tqdm.write(f"Готово! Собрано {len(data):,} вакансий.")


if __name__ == "__main__":
    main()
