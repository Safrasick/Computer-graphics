# Программа для расчета стоимости печати изображения
## Описание
Программа позволяет рассчитать расход тонера для полиграфического отпечатка по выбранному изображению. Для цветов CMYK вычисляется процент использования тонера, пользователь выбирает стоимость каждого картриджа и рассчитывается общая стоимость печати.

## Процесс работы
- Пользователь выбирает файл для печати.
- На основе названия, выбранное изображение преобразуется в цветовую модель из RGB в CMYK. 
- Рассчитывается процент использоваания тонера для каждого цвета.
- Пользователь вводит стоимость для каждого цвета картриджей и самой работы.
- Программа рассчитывает и выводит стоимость печати изображения.

## Уставнока
1. Клонируйте репозиторий.
2. Создайте и активируйте виртуальное окружение.

## Виртуальное окружение

### Создание:
```python 
python -m venv .venv
```
### Активация на Windows:
```python 
.venv/scripts/activate
```
### Активация на Unix или MacOS:
```python 
source .venv/bin/activate
```
# Интерфейс с помощью Flask

## Запуск программы с помощью Docker
``` 
docker-compose build
``` 
``` 
docker compose up
```
Далее переход по ссылке http://localhost:5000, если не доступен по данной то использовать эту http://127.0.0.1:5000/

# Запуск консольного приложения
## Зависимости

Для корректной работы программы необходимо установить зависимости:
```python 
pip install -r req.txt
```
Для корректной работы на следующих ОС потребуется дополнительная команда:
Ubuntu
```
sudo apt-get install python3-tk 
```
Fedora
```
sudo dnf install python3-tkinter
```
MacOS
```
brew install python-tk
```
## Запуск программы
```python 
python main.py
```

### Источники
Для редактирования README.md-файла, создания Dockerfile и Docker-compose использовались свои прошлогодние работы с курса "Введение в проектную деятельность". Для файлов requirements.txt и req.txt использовалась команда pip freeze. Для html вёрстки использовались сайты https://timeweb.com/ru/community/articles/verstka-sayta-instrukciya-dlya-nachinayushchih и https://docs.aspose.com/html/ru/net/tutorial/working-with-html-color/. При поднятии сервера на flask использовалась прошлогодняя программа. При создании консольного приложения использовались наработки из прошлогоднего проекта, знания библиотеки tkinter основано на данном проекте и сайте https://metanit.com/python/tkinter/.
