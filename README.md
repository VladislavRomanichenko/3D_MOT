# Tracker Prediction

ROS2 пакет для 3D трекинга объектов с предсказанием траекторий движения. Проект предназначен для отслеживания и прогнозирования движения динамических объектов в 3D пространстве с использованием фильтра Калмана и алгоритмов ассоциации данных.

Проект состоит из следующих основных компонентов:

- **Tracker3D**: Основной класс для трекинга объектов
- **Trajectory**: Класс для управления траекториями объектов
- **Inference**: ROS2 узел для обработки сообщений и координации
- **Evaluation**: Система оценки качества трекинга

## Установка

```bash
git clone <repository-url>
cd <repository>

# В корневой директории workspace
colcon build 
source install/setup.bash
```

## Использование

### Запуск трекера

```bash
# Запуск с параметрами по умолчанию
ros2 launch tracker_prediction tracker.launch.py

# Запуск в режиме сохранения данных для подсчёта метрик
ros2 launch tracker_prediction evaluation.launch.py
```

### Основные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `tracker_flag` | Включить/выключить трекинг | `true` |
| `target_frame` | Целевая система координат | `local_map` |
| `frame_odom` | Система координат одометрии | `odom` |
| `timeout` | Таймаут трансформаций (сек) | `0.01` |
| `evaluation_mode` | Режим метрик | `false` |
| `predictor_logging` | Логирование предсказаний | `false` |

### Параметры фильтра Калмана

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `state_func_covariance` | Ковариация функции состояния | `0.2` |
| `measure_func_covariance` | Ковариация функции измерения | `0.1` |
| `prediction_score_decay` | Затухание оценки предсказания | `0.01` |
| `LiDAR_scanning_frequency` | Частота сканирования LiDAR (Гц) | `10.0` |

### Параметры трекинга

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `max_prediction_num` | Максимальное количество предсказаний | `20` |
| `max_prediction_num_for_new_object` | Максимальное количество предсказаний для новых объектов | `8` |
| `association_threshold` | Порог ассоциации | `1.0` |
| `input_score` | Входная оценка | `0.5` |
| `init_score` | Начальная оценка | `0.15` |
| `update_score` | Оценка обновления | `-0.3` |
| `post_score` | Пост-обработка оценки | `0.55` |

## Топики

### Входные топики
- `/centerpoint/objects3d` (`objects_msgs/msg/ObjectArray`) - Детекции объектов от детектора

### Выходные топики
- `/tracking/tracking_objects` (`objects_msgs/msg/DynamicObjectArray`) - Отслеживаемые динамические объекты с предсказаниями

## Оценка качества

Проект включает систему оценки качества трекинга по стандарту KITTI 3D MOT:

### Метрики трекера
- **MOTA** (Multiple Object Tracking Accuracy) - Общая точность трекинга
- **MOTP** (Multiple Object Tracking Precision) - Точность трекинга
- **F1-Score** - F1-мера
- **Precision/Recall** - Точность и полнота
- **sMOTA** - Стабилизированная MOTA

### Метрики предиктора

- **FDE/ADE** Final Displacement Error и Average Displacement Error по ссылке https://git.integrant.ru/sdbcs-nio3/alg/obstacle_detection/fde_ade_metrics


## Структура проекта

```
tracker_prediction/
├── src/                    # Исходный код C++
│   ├── inference.cpp      # ROS2 узел
│   ├── tracker.cpp        # Основной класс трекера
│   ├── trajectory.cpp     # Управление траекториями
│   └── utils.cpp          # Вспомогательные функции
├── include/               # Заголовочные файлы
│   ├── tracker.hpp        # Интерфейс трекера
│   ├── trajectory.hpp     # Интерфейс траекторий
│   ├── inference.hpp      # Интерфейс ROS2 узла
│   ├── config.hpp         # Конфигурация
│   └── utils.hpp          # Утилиты
├── evaluation/            # Система оценки
│   ├── evaluate_kitti3dmot.py  # Оценка по KITTI метрик MOT
│   ├── evaluate_node.py        # Публикует GT данные, для FDE/ADE метрик
│   └── label/                  # Датасеты для метрик
├── launch/                # Launch файлы
│   ├── tracker.launch.py       # Запуск трекера
│   └── evaluation.launch.py    # Сохраняет данные для подсчёта метрик
├── results/               # Данные для метрик после обработки их трекером
└── CMakeLists.txt         
```

