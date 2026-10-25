from collections import Counter

RU = {
    "person": "человек", "bicycle": "велосипед", "car": "машина",
    "motorcycle": "мотоцикл", "bus": "автобус", "train": "поезд",
    "truck": "грузовик", "boat": "лодка", "bird": "птица", "cat": "кошка",
    "dog": "собака", "horse": "лошадь", "sheep": "овца", "cow": "корова",
    "elephant": "слон", "bear": "медведь", "zebra": "зебра",
    "giraffe": "жираф", "backpack": "рюкзак", "umbrella": "зонт",
    "handbag": "сумка", "bottle": "бутылка", "cup": "чашка",
    "fork": "вилка", "knife": "нож", "spoon": "ложка", "bowl": "миска",
    "banana": "банан", "apple": "яблоко", "pizza": "пицца",
    "chair": "стул", "couch": "диван", "potted plant": "комнатное растение",
    "bed": "кровать", "dining table": "обеденный стол", "tv": "телевизор",
    "laptop": "ноутбук", "mouse": "мышь", "keyboard": "клавиатура",
    "cell phone": "телефон", "book": "книга", "clock": "часы",
    "vase": "ваза", "scissors": "ножницы", "teddy bear": "плюшевый мишка",
    "traffic light": "светофор", "stop sign": "знак стоп", "bench": "скамейка",
}

_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def describe(path, min_conf=0.4):
    result = get_model()(path, verbose=False)[0]
    counts = Counter()
    for box in result.boxes:
        if float(box.conf[0]) >= min_conf:
            name = result.names[int(box.cls[0])]
            counts[RU.get(name, name)] += 1
    return ["%s (%d)" % (name, n) for name, n in counts.most_common()]
