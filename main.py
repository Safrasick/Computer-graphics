import cv2
import numpy as np
import fitz  # PyMuPDF
from tkinter import Tk, Label, Entry, Button, StringVar, Toplevel, filedialog
from tkinter import messagebox
import ttkbootstrap as ttk  # Для красивого интерфейса в стиле Bootstrap

def rgb_to_cmyk(r, g, b, rgb_scale=255, cmyk_scale=100):
    """Преобразует значения RGB в цветовую схему CMYK."""
    # Если RGB равен (0, 0, 0), то это чисто черный цвет
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, cmyk_scale
    
    # Преобразуем RGB [0, 255] в диапазон [0, 1] для каждого компонента
    r = r / rgb_scale
    g = g / rgb_scale
    b = b / rgb_scale
    
    # Вычисляем компоненты CMY
    c = 1 - r
    m = 1 - g
    y = 1 - b
    
    # Находим минимальное значение среди CMY
    k = min(c, m, y)
    
    # Если K равен 1, то цвет полностью черный
    if k == 1:
        return 0, 0, 0, cmyk_scale
    
    # Преобразуем C, M, Y в значения [0, 1] с учётом K
    c = (c - k) / (1 - k)
    m = (m - k) / (1 - k)
    y = (y - k) / (1 - k)
    
    # Преобразуем значения в диапазон [0, cmyk_scale]
    return (c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale)

def bgr_to_cmyk(image, cmyk_scale=100):
    """Преобразует BGR изображение в цветовую модель CMYK с использованием векторизованных операций."""
    # Преобразуем изображение из BGR в RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    
    # Вычисляем компоненты CMY
    c = 1 - rgb_image[:, :, 0]
    m = 1 - rgb_image[:, :, 1]
    y = 1 - rgb_image[:, :, 2]
    
    # Находим компонент K
    k = np.minimum(np.minimum(c, m), y)
    
    # Избегаем деления на ноль
    with np.errstate(divide='ignore', invalid='ignore'):
        c = (c - k) / (1 - k)
        m = (m - k) / (1 - k)
        y = (y - k) / (1 - k)
        
        # Заменяем NaN значениями 0, где K=1
        c[np.isnan(c)] = 0
        m[np.isnan(m)] = 0
        y[np.isnan(y)] = 0
    
    # Масштабируем до [0, cmyk_scale]
    c = (c * cmyk_scale).astype(np.float32)
    m = (m * cmyk_scale).astype(np.float32)
    y = (y * cmyk_scale).astype(np.float32)
    k = (k * cmyk_scale).astype(np.float32)
    
    # Объединяем каналы CMYK
    cmyk_image = np.stack((c, m, y, k), axis=-1)
    
    return cmyk_image

def calculate_toner_usage(cmyk_image):
    """Рассчитывает процент использования тонера для каждого цвета CMYK."""
    C, M, Y, K = cmyk_image[:, :, 0], cmyk_image[:, :, 1], cmyk_image[:, :, 2], cmyk_image[:, :, 3]
    total_pixels = C.size  # Общее количество пикселей в изображении
    
    # Рассчитываем процентное заполнение для каждого канала (0-100)
    c_percent = np.sum(C) / (100 * total_pixels) * 100  # Процент Cyan
    m_percent = np.sum(M) / (100 * total_pixels) * 100  # Процент Magenta
    y_percent = np.sum(Y) / (100 * total_pixels) * 100  # Процент Yellow
    k_percent = np.sum(K) / (100 * total_pixels) * 100  # Процент Black
    
    return c_percent, m_percent, y_percent, k_percent  # Возвращаем проценты заполнения

def calculate_print_cost(cmyk_usage, cartridge_costs, paper_cost, labor_cost):
    """Рассчитывает общую стоимость печати на основе процента использования тонера и стоимости картриджей."""
    c_percent, m_percent, y_percent, k_percent = cmyk_usage
    c_cost, m_cost, y_cost, k_cost = cartridge_costs
    
    # Рассчитываем стоимость тонера для каждого цвета
    toner_cost = (c_percent / 100 * c_cost +
                  m_percent / 100 * m_cost +
                  y_percent / 100 * y_cost +
                  k_percent / 100 * k_cost)
    
    # Общая стоимость включает стоимость тонера, бумаги и работы
    total_cost = toner_cost + paper_cost + labor_cost
    
    return total_cost

def extract_images_from_pdf(pdf_path):
    """Извлекает изображения из PDF-файла."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        mode = "RGBA" if pix.alpha else "RGB"
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Конвертируем изображение в формат BGR
        if pix.n == 4: 
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3: 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)  # Добавляем изображение в список
    return images

def load_file():
    """Загружает файл изображения или PDF и извлекает изображения из него."""
    global images
    file_path = filedialog.askopenfilename(filetypes=[("Image and PDF files", "*.jpg *.png *.jpeg *.pdf")])
    
    if file_path:  # Проверяем, что файл был выбран
        print(f"Выбран файл: {file_path}")  # Отладочная информация
    
        if file_path.lower().endswith(".pdf"):
            try:
                images = extract_images_from_pdf(file_path)  # Извлекаем изображения из PDF
                if not images:
                    messagebox.showerror("Ошибка", "Не удалось извлечь изображения из PDF.")
                    return
            except Exception as e:
                messagebox.showerror("Ошибка", f"Произошла ошибка при извлечении изображений из PDF: {e}")
                return
        else:
            # Попробуем загрузить изображение с помощью cv2.imread()
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение. Убедитесь, что файл поддерживаемого формата. Путь к файлу: {file_path}")
                print(f"Ошибка загрузки изображения по пути: {file_path}")  # Отладочная информация
                return
            images = [image]  # Сохраняем загруженное изображение в список
            print(f"Изображение загружено успешно. Размер: {image.shape}")  # Отладочная информация

        messagebox.showinfo("Успех", f"Файл {file_path} загружен.")
        file_path_var.set(file_path)  # Отображаем путь к файлу в интерфейсе
    else:
        messagebox.showerror("Ошибка", "Файл не был выбран.")

def calculate():
    """Выполняет расчет стоимости печати на основе введенных данных и загруженных изображений."""
    if not images:
        messagebox.showerror("Ошибка", "Нет загруженных изображений для расчёта.")
        return
    
    try:
        # Получаем стоимость картриджей и другие затраты из полей ввода
        c_cost = float(c_cost_var.get())
        m_cost = float(m_cost_var.get())
        y_cost = float(y_cost_var.get())
        k_cost = float(k_cost_var.get())
        paper_cost = float(paper_cost_var.get())
        labor_cost = float(labor_cost_var.get())
        copies = int(copies_var.get())  # Получаем количество копий
    except ValueError:
        messagebox.showerror("Ошибка", "Пожалуйста, введите корректные числовые значения.")
        return
    
    total_cost = 0  # Инициализируем общую стоимость
    for idx, image in enumerate(images, start=1):
        cmyk_image = bgr_to_cmyk(image)  # Преобразуем изображение в CMYK
        cmyk_usage = calculate_toner_usage(cmyk_image)  # Рассчитываем использование тонера
        # Рассчитываем стоимость печати для каждого изображения
        cost = calculate_print_cost(cmyk_usage, (c_cost, m_cost, y_cost, k_cost), paper_cost, labor_cost)
        total_cost += cost  # Суммируем общую стоимость
        result_var.set(f"C: {cmyk_usage[0]:.2f}%, M: {cmyk_usage[1]:.2f}%, Y: {cmyk_usage[2]:.2f}%, K: {cmyk_usage[3]:.2f}%")
    
    total_print_cost = total_cost * copies  # Рассчитываем общую стоимость для тиража
    total_cost_var.set(f"Общая стоимость на {copies} копий: {total_print_cost:.2f}")

# Создание красивого интерфейса с помощью ttkbootstrap
window = ttk.Window(themename="minty")
window.title("Расчет стоимости печати")
window.geometry('800x700')  # Изменил размер для удобства

images = []  # Инициализируем переменную images

file_path_var = StringVar()
c_cost_var = StringVar()
m_cost_var = StringVar()
y_cost_var = StringVar()
k_cost_var = StringVar()
paper_cost_var = StringVar()
labor_cost_var = StringVar()
copies_var = StringVar()
result_var = StringVar()
total_cost_var = StringVar()

# Разметка интерфейса
ttk.Label(window, text="Загрузить файл:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
ttk.Button(window, text="Загрузить", command=load_file).grid(row=0, column=1, padx=10, pady=10)
ttk.Label(window, textvariable=file_path_var, wraplength=400).grid(row=0, column=2, padx=10, pady=10, sticky='w')

ttk.Label(window, text="Стоимость картриджа Cyan:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=c_cost_var).grid(row=1, column=1, padx=10, pady=5)

ttk.Label(window, text="Стоимость картриджа Magenta:").grid(row=2, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=m_cost_var).grid(row=2, column=1, padx=10, pady=5)

ttk.Label(window, text="Стоимость картриджа Yellow:").grid(row=3, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=y_cost_var).grid(row=3, column=1, padx=10, pady=5)

ttk.Label(window, text="Стоимость картриджа Black:").grid(row=4, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=k_cost_var).grid(row=4, column=1, padx=10, pady=5)

ttk.Label(window, text="Стоимость бумаги:").grid(row=5, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=paper_cost_var).grid(row=5, column=1, padx=10, pady=5)

ttk.Label(window, text="Стоимость работы:").grid(row=6, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=labor_cost_var).grid(row=6, column=1, padx=10, pady=5)

ttk.Label(window, text="Количество копий:").grid(row=7, column=0, padx=10, pady=5, sticky='w')
ttk.Entry(window, textvariable=copies_var).grid(row=7, column=1, padx=10, pady=5)

ttk.Button(window, text="Рассчитать", command=calculate).grid(row=8, column=0, columnspan=3, padx=10, pady=20)

ttk.Label(window, text="Процентное заполнение:").grid(row=9, column=0, padx=10, pady=5, sticky='w')
ttk.Label(window, textvariable=result_var).grid(row=9, column=1, padx=10, pady=5, sticky='w')

ttk.Label(window, text="Общая стоимость:").grid(row=10, column=0, padx=10, pady=5, sticky='w')
ttk.Label(window, textvariable=total_cost_var).grid(row=10, column=1, padx=10, pady=5, sticky='w')

window.mainloop()
