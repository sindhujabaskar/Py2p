import matplotlib.pyplot as plt

def plot_something(data):
    plt.figure(figsize=(10,5))
    plt.plot(data)
    plt.title("Sample Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    return plt