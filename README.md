# Делали:

* Моисеенко Дмитрий (11-104)
* Харин Ильдар (11-104)

---

## Задание 1:


1. Скачать минимум 100 текстовых страниц с помощью краулера из предварительно подготовленного списка
- список страниц, сайтов можно найти в интернете
- каждая страница должна содержать текст (ссылки на js, css файлы недопустимы)
- язык текста  должен быть одинаков для всех страниц

2. Записать каждую страницу в текстовый файл ("выкачка")
- очищать выкачку от html разметки  НЕ надо(выкачиваем вместе с разметкой)

3. Создать файл index.txt в котором хранится номер файла и ссылка на страницу

---

### Инструкция:

0. Создать виртуальное окружение - `python -m venv venv`
1. Активировать виртуальное окружение - `source venv/bin/activate`
2. Установить зависимости - `pip install -r requirements.txt`
3. Запустить краулер - `python crawler.py`

---

## Задание 2


1. Из сохраненных документов выделить отдельные слова (токенизация) и получить список токенов

- список не должен содержать дубликатов, союзов, предлогов, чисел
- список не должен содержать "мусора" (слов содержащих одновременно буквы и цифры, обрывки разметки и тд.)
- язык текста должен быть одинаков для всех страниц

2. Сгруппировать токены по леммам

---

### Инструкция:

0. Создать виртуальное окружение - `python -m venv venv`
1. Активировать виртуальное окружение - `source venv/bin/activate`
2. Установить зависимости - `pip install -r requirements.txt`
3. Запустить краулер - `python crawler.py`
4. Запустить организатор файлов - `python organizer.py`
5. Запустить токенизатор - `python tokenizer.py`

## Задание 3
1. Создать инвертированный список терминов (индекс)

2. Реализовать булев поиск по построенному индексу

---

### Инструкция:

0. Создать виртуальное окружение - `python -m venv venv`
1. Активировать виртуальное окружение - `source venv/bin/activate`
2. Установить зависимости - `pip install -r requirements.txt`
3. Запустить краулер - `python crawler.py`
4. Запустить организатор файлов - `python organizer.py`
5. Запустить токенизатор - `python tokenizer.py`
6. Запустить создание инвертированного индекса - `python inverted_index.py`
7. Запустить поиск по индексу (для выхода из поиска отправьте пустую строку) - `python search.py`

---

## Задание 4

1. Для каждого скачанного документа из Задания 1:
   - Подсчитать TF каждого термина (см. список терминов из Задания 2).
   - Подсчитать IDF для термина.
   - Подсчитать TF для каждой лемматизированной формы (см. список форм из Задания 2) как отношение суммы вхождений числа терминов к общему количеству терминов в документе.
   - Подсчитать IDF.

2. Для оценки выполнения задания необходимо предоставить:
   - Ссылку на рабочую версию кода в репозитории.
   - Txt файлы по списку терминов и подсчитанными TF-IDF. Каждый файл соответствует одному файлу выкачки и содержит данные в формате:  
     `<термин><пробел><idf><пробел><tf-idf><\n>`  
   - Txt файлы по списку лемматизированных форм и подсчитанными TF-IDF. Каждый файл соответствует одному файлу выкачки и содержит данные в формате:  
     `<лемма><пробел><idf><пробел><tf-idf><\n>`

---

### Инструкция:

0. Создать виртуальное окружение - `python -m venv venv`
1. Активировать виртуальное окружение - `source venv/bin/activate`
2. Установить зависимости - `pip install -r requirements.txt`
3. Запустить краулер - `python crawler.py`
4. Запустить организатор файлов - `python organizer.py`
5. Запустить токенизатор - `python tokenizer.py`
6. Запустить создание инвертированного индекса - `python inverted_index.py`
7. Запустить расчет TF-IDF - `python calculate_tf_idf.py`


---

## Задание 5

1. Разработать поисковую систему на основе векторного поиска по построенному индексу

2. Для оценки выполнения задания необходимо предоставить:
   - Ссылку на рабочую версию кода в репозитории.

---

### Инструкция:

0. Создать виртуальное окружение - `python -m venv venv`
1. Активировать виртуальное окружение - `source venv/bin/activate`
2. Установить зависимости - `pip install -r requirements.txt`
3. Запустить краулер - `python crawler.py`
4. Запустить организатор файлов - `python organizer.py`
5. Запустить токенизатор - `python tokenizer.py`
6. Запустить создание инвертированного индекса - `python inverted_index.py`
7. Запустить расчет TF-IDF - `python calculate_tf_idf.py`
8. Запустить векторный поиск TF-IDF - `python vector_index_search.py`

## Задание 6

1. Добавить в разработанную  в Задании 5 поисковую систему на основе векторного поиска по построенному индексу

2. 1)WEB  интерфейс, 2)ранжирование, 3)вывод топ 10 результатов по введенному запросу
---

### Инструкция:

0. повторить шаги из предыдущих заданий
1. `python flask_vector.py`
