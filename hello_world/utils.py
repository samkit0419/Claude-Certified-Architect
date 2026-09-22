def average(numbers):
    if not numbers:
        raise ValueError("average() arg is an empty sequence")
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def get_last_item(items):
    if not items:
        raise ValueError("get_last_item() arg is an empty sequence")
    return items[len(items) - 1]
