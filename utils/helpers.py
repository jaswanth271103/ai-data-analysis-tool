def human_readable(num):
    for unit in ["", "K", "M", "B"]:
        if abs(num) < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
