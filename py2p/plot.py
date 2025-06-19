import matplotlib.pyplot as plt

def plot_onething(data):
    plt.figure(figsize=(10,5))
    plt.plot(data)
    plt.title("Sample Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    return plt

def plot_twothings(data, data2):
    plt.subplots(1, 2, figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(data)
    plt.title("Sample Plot 1")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.subplot(1, 2, 2)
    plt.plot(data2)
    plt.title("Sample Plot 2")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
    return plt

# %%
# plt.plot(database.toolkit.timestamps['sub-SB03','ses-01'],database.calculate.interp_deltaf_f['sub-SB03','ses-01'][77])
