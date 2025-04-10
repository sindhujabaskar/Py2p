from py2p.io import load_suite2p_outputs
from py2p.config import DATA_DIR

def main():
    data = load_suite2p_outputs(DATA_DIR)
    
    print('Hello', data)


if __name__ == "__main__":
    main()

