## Телеграмм бот переноса стиля

Предполагалось, что бот будет работать в двух режимах:
1. Переносить стиль с одного загруженного изображения на другое
2. Переносить на загруженное изображение стиль картин Фриды Кало.

Первый вариант работает в "медленном" режиме и переносит стиль в режиме реального времени. За основу взят алгоритм, описанный здесь https://pytorch.org/tutorials/advanced/neural_style_tutorial.html. Для корректной работы алгоритма в боте функция трансформации была переписана в класс, а из используемой для получения фич модели VGG19 были предварительно извлечены только те слои, которые используются в алгоритме. В бота загружается уже мини-модель, содержащая несколько первых слоев VGG19.

Второй вариант основан на генеративно-состязательной модели CycleGAN. Она обучалась имитировать стиль картин Фриды Кало. В процессе обучения использовался датасет с картинами разных художников и, соответственно, картины Фриды Кало. Хорошего результата при обучении достичь не удалось. Скорее всего, недостаточное время обучения сыграло здесь основную роль.
