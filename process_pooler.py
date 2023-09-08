from concurrent.futures import ProcessPoolExecutor


def task(n):
    
    print(n)
    return n * n

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(task, range(10)):
            print(result)
